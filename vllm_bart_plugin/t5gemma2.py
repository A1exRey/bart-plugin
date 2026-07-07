# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Derived from the T5Gemma2 implementation in HuggingFace transformers;
# licensed under the Apache License, Version 2.0.
"""PyTorch T5Gemma2 model for vLLM (text-to-text).

T5Gemma2 (e.g. ``google/t5gemma-2-270m-270m``) is an encoder-decoder model
built from Gemma-style blocks.  Its decoder does NOT use separate self- and
cross-attention: every decoder layer runs ONE merged attention whose keys and
values are ``concat(self KV, cross KV)`` under a single softmax, where the
cross K/V are projected from the (constant) encoder output with the layer's
own k/v projections, receive k-norm but no rotary embedding, and are always
fully visible (the sliding window only restricts the self half).

vLLM has no attention primitive for merged attention, so this plugin runs the
model as a *decoder-only* multimodal model:

- The encoder text is smuggled in as a ``"text"`` multimodal item (the same
  trick bart.py uses).  The processor prepends N placeholder tokens to the
  decoder prompt, one per encoder token, so the sequence is
  ``[P]*N ++ [BOS, decoder tokens...]`` and the scheduler allocates paged KV
  slots for the encoder prefix.
- ``embed_multimodal`` runs the (bidirectional) text encoder; the runner
  scatters its output into the placeholder rows of ``inputs_embeds``.
- In every decoder layer, the post-layernorm hidden states of the prefix rows
  are re-substituted with the raw encoder output before the fused qkv
  projection, so those rows produce exactly HF's cross K/V; they are written
  once into the ordinary paged KV cache at prefill.
- Prefix keys are rotated at the constant position N; by RoPE's relative
  property this reproduces HF's "rotated q at t, un-rotated cross k" scores
  exactly (see t5gemma2_prefix.py).  Decode steps are then 100% standard
  paged causal decoding.

The math of this scheme is validated against the HF reference in
tests/test_t5gemma2_math.py.

Constraints enforced at load time (all with actionable errors):

- ``hf_overrides=t5gemma2_hf_overrides`` (exported here) must be passed so
  vLLM treats the model as decoder-only.
- ``max_model_len`` must not exceed the decoder's sliding window (512 for
  google/t5gemma-2-270m-270m — read ``config.decoder.sliding_window``):
  below it, causal sliding and causal full attention coincide, so the
  decoder is exact without configuring a sliding window.  Longer contexts
  are a follow-up.  The encoder applies HF's real bidirectional sliding
  masks and is exact at any length.
- Prefix caching must be off and multimodal items must not be chunked, so a
  placeholder run always starts at position 0 within its scheduled chunk.

Image input is supported: pass the encoder input as
``multi_modal_data={"text": {"text": "caption: <start_of_image>",
"images": [pil_image]}}``.  Each ``<start_of_image>`` marker in the text is
expanded by the HF processor into boi + mm_tokens_per_image (256)
image-soft-tokens + eoi; the SigLIP tower + projector features replace those
rows before the text encoder runs.  Under the 512-token window of the 270m
checkpoint exactly one image fits (~262 encoder tokens), leaving ~240 tokens
of generation budget.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions

IS_LEGACY = False
try:
    from vllm.v1.attention.backend import AttentionType
    from vllm.model_executor.layers.attention import Attention
    from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
except ImportError:  # pragma: no cover - legacy path not supported here
    from vllm.attention.backends.abstract import AttentionType
    from vllm.attention.layer import Attention
    from vllm.multimodal.profiling import BaseDummyInputsBuilder

    IS_LEGACY = True

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsQuant,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

try:
    from vllm.inputs import MultiModalDataDict
except ImportError:
    from vllm.multimodal.inputs import MultiModalDataDict  # type: ignore[no-redef]
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from .t5gemma2_prefix import (
    PrefixContext,
    compute_prefix_rope_positions,
    substitute_prefix_rows,
)

logger = __import__("logging").getLogger(__name__)


def _check_transformers_version() -> None:
    import transformers

    version = getattr(transformers, "__version__", "0")
    major = int(version.split(".", 1)[0])
    if major < 5:
        raise RuntimeError(
            "T5Gemma2 requires transformers >= 5.0 (found "
            f"{version}).  Install it with: pip install 'transformers>=5.0' "
            "or: pip install 'vllm-bart-plugin[t5gemma2]'"
        )


_check_transformers_version()


def t5gemma2_hf_overrides(hf_config):
    """``hf_overrides`` callable that lets vLLM run T5Gemma2 decoder-only.

    Usage::

        from vllm_bart_plugin.t5gemma2 import t5gemma2_hf_overrides
        llm = LLM(model="google/t5gemma-2-270m-270m",
                  hf_overrides=t5gemma2_hf_overrides, ...)

    The dict form ``hf_overrides={"is_encoder_decoder": False}`` works too
    (vLLM applies it via ``config.update`` after construction, which the
    strict ``T5Gemma2Config`` allows; only init-time kwargs are rejected)
    and is the form to use with ``vllm serve --hf-overrides``.

    This only affects how vLLM routes the model (decoder-only multimodal
    path, which is what the prefix-KV scheme needs); the plugin still
    computes the exact encoder-decoder function.
    """
    hf_config.is_encoder_decoder = False
    return hf_config


def _text_config_of(config, *, is_encoder: bool):
    return config.encoder.text_config if is_encoder else config.decoder


class T5Gemma2ScaledWordEmbedding(VocabParallelEmbedding):
    """Embedding scaled by sqrt(hidden_size) with a separate EOI row.

    Mirrors HF's ``T5Gemma2TextScaledWordEmbedding``: the checkpoint stores
    an extra ``eoi_embedding`` parameter that replaces the embedding of the
    end-of-image token.  Irrelevant for pure text, but kept so weights load
    exactly and image support can build on it.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float,
        eoi_token_index: int,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale
        self.eoi_token_index = eoi_token_index
        self.eoi_embedding = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = super().forward(input_ids)
        # HF multiplies by an embed_scale buffer cast to the weight dtype.
        scale = torch.tensor(self.embed_scale, dtype=embeds.dtype,
                             device=embeds.device)
        embeds = embeds * scale
        is_eoi = (input_ids == self.eoi_token_index).unsqueeze(-1)
        return torch.where(is_eoi, self.eoi_embedding.to(embeds.dtype), embeds)


class T5Gemma2MLP(nn.Module):
    def __init__(
        self,
        text_config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if text_config.hidden_activation not in (
            "gelu_pytorch_tanh",
            "gelu_tanh",
        ):
            raise NotImplementedError(
                "T5Gemma2 plugin only supports gelu_pytorch_tanh, got "
                f"{text_config.hidden_activation!r}"
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            text_config.hidden_size,
            [text_config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            text_config.intermediate_size,
            text_config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class _T5Gemma2AttentionBase(nn.Module):
    """Shared qkv/norm/rope plumbing for encoder and decoder attention."""

    def __init__(
        self,
        text_config,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = text_config.hidden_size
        self.total_num_heads = text_config.num_attention_heads
        self.total_num_kv_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim
        self.layer_type = text_config.layer_types[layer_idx]

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = text_config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=text_config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=text_config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=text_config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=text_config.rms_norm_eps)

        # transformers v5 keys rope_parameters by layer type; fall back to a
        # flat dict for configs that don't.
        rope_parameters = text_config.rope_parameters
        if isinstance(rope_parameters, dict) and self.layer_type in rope_parameters:
            rope_parameters = rope_parameters[self.layer_type]
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=text_config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rope_parameters,
        )

    def _project_qkv(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q).flatten(-2, -1)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k).flatten(-2, -1)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v


def bidirectional_sliding_window_mask(
    num_tokens: int, sliding_window: int, device=None
) -> torch.Tensor:
    """HF's bidirectional sliding-window mask (True = may attend).

    Mirrors ``sliding_window_mask_function(w, is_causal=False)`` from the HF
    reference: token i attends to token j iff
    ``0 <= i-j < (w+1)//2  or  0 < j-i < w//2 + 1``.
    Note the window is split into a look-behind and a look-ahead half, so
    this differs from full attention as soon as the input exceeds ~w/2
    tokens.
    """
    left = (sliding_window + 1) // 2
    right = sliding_window // 2 + 1
    idx = torch.arange(num_tokens, device=device)
    dist = idx[:, None] - idx[None, :]  # q_idx - kv_idx
    return ((dist >= 0) & (dist < left)) | ((dist < 0) & (-dist < right))


class T5Gemma2EncoderAttention(_T5Gemma2AttentionBase):
    """Bidirectional encoder self-attention.

    Runs densely outside the KV cache (the encoder executes once per request
    inside ``embed_multimodal``), so the exact HF masks can be applied:
    no mask on full-attention layers, the bidirectional sliding-window mask
    on sliding layers.  This is exact for any encoder length.
    """

    def __init__(self, text_config, layer_idx, quant_config=None, prefix=""):
        super().__init__(text_config, layer_idx, quant_config, prefix)
        if getattr(text_config, "attn_logit_softcapping", None) is not None:
            raise NotImplementedError(
                "attn_logit_softcapping is not supported in the T5Gemma2 "
                "encoder."
            )
        self.is_sliding = self.layer_type == "sliding_attention"

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        sliding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv(hidden_states, positions)
        num_tokens = hidden_states.shape[0]
        q = q.view(1, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(1, num_tokens, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(1, num_tokens, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sliding_mask if self.is_sliding else None,
            scale=self.scaling,
        )
        out = out.transpose(1, 2).reshape(num_tokens, -1)
        out, _ = self.o_proj(out)
        return out


class T5Gemma2MergedDecoderAttention(_T5Gemma2AttentionBase):
    """HF's merged self+cross attention, expressed as prefix-KV causal
    attention over the paged cache.

    When ``prefix_ctx`` is set (prefill chunks containing encoder placeholder
    rows), the post-layernorm hidden states of those rows are replaced with
    the raw encoder output before the fused qkv projection, so the rows'
    K/V match HF's per-layer ``k_proj(enc)`` / ``v_proj(enc)`` (k-norm is
    then applied uniformly, RoPE rotates them at the constant position N —
    both per the HF convention; see module docstring).  Their query outputs
    are garbage but are never sampled and never read by decoder rows.
    """

    def __init__(
        self,
        text_config,
        layer_idx,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        super().__init__(text_config, layer_idx, quant_config, prefix)
        # No per_layer_sliding_window: under the enforced
        # max_model_len <= sliding_window cap, causal-sliding == causal-full
        # for the self half, and the cross half (the prefix rows) must never
        # be masked out — which a real sliding window would do.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=getattr(
                text_config, "attn_logit_softcapping", None
            ),
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def forward(
        self,
        rope_positions: torch.Tensor,
        hidden_states: torch.Tensor,
        prefix_ctx: PrefixContext | None,
    ) -> torch.Tensor:
        if prefix_ctx is not None:
            hidden_states = substitute_prefix_rows(
                hidden_states, prefix_ctx.is_prefix, prefix_ctx.enc_rows
            )
        q, k, v = self._project_qkv(hidden_states, rope_positions)
        out = self.attn(q, k, v)
        out, _ = self.o_proj(out)
        return out


class T5Gemma2EncoderLayer(nn.Module):
    def __init__(self, text_config, layer_idx, quant_config=None, prefix=""):
        super().__init__()
        self.self_attn = T5Gemma2EncoderAttention(
            text_config, layer_idx, quant_config, prefix=f"{prefix}.self_attn"
        )
        hidden = text_config.hidden_size
        eps = text_config.rms_norm_eps
        self.pre_self_attn_layernorm = GemmaRMSNorm(hidden, eps=eps)
        self.post_self_attn_layernorm = GemmaRMSNorm(hidden, eps=eps)
        self.mlp = T5Gemma2MLP(text_config, quant_config, prefix=f"{prefix}.mlp")
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        sliding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, sliding_mask)
        hidden_states = residual + self.post_self_attn_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(
            hidden_states
        )
        return hidden_states


class T5Gemma2DecoderLayer(nn.Module):
    def __init__(
        self, text_config, layer_idx, cache_config=None, quant_config=None,
        prefix="",
    ):
        super().__init__()
        self.self_attn = T5Gemma2MergedDecoderAttention(
            text_config,
            layer_idx,
            cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )
        hidden = text_config.hidden_size
        eps = text_config.rms_norm_eps
        self.pre_self_attn_layernorm = GemmaRMSNorm(hidden, eps=eps)
        self.post_self_attn_layernorm = GemmaRMSNorm(hidden, eps=eps)
        self.mlp = T5Gemma2MLP(text_config, quant_config, prefix=f"{prefix}.mlp")
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden, eps=eps)

    def forward(
        self,
        rope_positions: torch.Tensor,
        hidden_states: torch.Tensor,
        prefix_ctx: PrefixContext | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(
            rope_positions, hidden_states, prefix_ctx
        )
        hidden_states = residual + self.post_self_attn_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(
            hidden_states
        )
        return hidden_states


class T5Gemma2MultiModalProjector(nn.Module):
    """SigLIP patch features -> mm_tokens_per_image text-space embeddings.

    Mirrors HF's T5Gemma2MultiModalProjector (identical to Gemma3's):
    average-pool the patch grid down to tokens_per_side^2 tokens, RMSNorm
    over the vision hidden dim, then project into the text hidden dim.
    """

    def __init__(self, encoder_config):
        super().__init__()
        vision_config = encoder_config.vision_config
        text_hidden_size = encoder_config.text_config.hidden_size
        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(vision_config.hidden_size, text_hidden_size)
        )
        self.mm_soft_emb_norm = GemmaRMSNorm(
            vision_config.hidden_size, eps=vision_config.layer_norm_eps
        )
        self.patches_per_image = int(
            vision_config.image_size // vision_config.patch_size
        )
        self.tokens_per_side = int(encoder_config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = vision_outputs.shape
        x = vision_outputs.transpose(1, 2).reshape(
            batch_size,
            hidden_size,
            self.patches_per_image,
            self.patches_per_image,
        ).contiguous()
        x = self.avg_pool(x).flatten(2).transpose(1, 2)
        x = self.mm_soft_emb_norm(x)
        x = torch.matmul(x, self.mm_input_projection_weight)
        return x.type_as(vision_outputs)


class T5Gemma2TextEncoder(nn.Module):
    """The bidirectional text encoder; weight path encoder.text_model.*"""

    def __init__(
        self, text_config, eoi_token_index: int, quant_config=None, prefix=""
    ):
        super().__init__()
        self.config = text_config
        self.embed_tokens = T5Gemma2ScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            embed_scale=math.sqrt(text_config.hidden_size),
            eoi_token_index=eoi_token_index,
        )
        self.layers = nn.ModuleList(
            [
                T5Gemma2EncoderLayer(
                    text_config,
                    layer_idx,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        sliding_mask = None
        window = self.config.sliding_window
        if window is not None and any(
            layer_type == "sliding_attention"
            for layer_type in self.config.layer_types
        ):
            sliding_mask = bidirectional_sliding_window_mask(
                input_ids.shape[0], window, device=hidden_states.device
            )
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, sliding_mask)
        return self.norm(hidden_states)


class T5Gemma2Encoder(nn.Module):
    """The multimodal encoder, matching HF's module tree: a bidirectional
    text encoder plus a SigLIP vision tower whose (pooled, projected)
    features replace the image-soft-token rows of the text embeddings."""

    def __init__(
        self, encoder_config, eoi_token_index, quant_config=None, prefix=""
    ):
        super().__init__()
        from vllm.model_executor.models.siglip import SiglipVisionModel

        self.text_model = T5Gemma2TextEncoder(
            encoder_config.text_config,
            eoi_token_index,
            quant_config,
            prefix=f"{prefix}.text_model",
        )
        self.vision_tower = SiglipVisionModel(
            encoder_config.vision_config,
            quant_config,
            prefix=f"{prefix}.vision_tower",
        )
        self.multi_modal_projector = T5Gemma2MultiModalProjector(
            encoder_config
        )

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(num_images, 3, H, W) -> (num_images, mm_tokens_per_image, T)"""
        vision_outputs = self.vision_tower(pixel_values)
        return self.multi_modal_projector(vision_outputs)


class T5Gemma2Decoder(nn.Module):
    def __init__(
        self,
        decoder_config,
        eoi_token_index,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        super().__init__()
        self.config = decoder_config
        self.embed_tokens = T5Gemma2ScaledWordEmbedding(
            decoder_config.vocab_size,
            decoder_config.hidden_size,
            embed_scale=math.sqrt(decoder_config.hidden_size),
            eoi_token_index=eoi_token_index,
        )
        self.layers = nn.ModuleList(
            [
                T5Gemma2DecoderLayer(
                    decoder_config,
                    layer_idx,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(decoder_config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            decoder_config.hidden_size, eps=decoder_config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        rope_positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        prefix_ctx: PrefixContext | None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(rope_positions, hidden_states, prefix_ctx)
        return self.norm(hidden_states)


class T5Gemma2Model(nn.Module):
    def __init__(
        self, config, cache_config=None, quant_config=None, prefix=""
    ):
        super().__init__()
        eoi = config.eoi_token_index
        if eoi is None:
            eoi = config.encoder.eoi_token_index
        self.encoder = T5Gemma2Encoder(
            config.encoder, eoi, quant_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = T5Gemma2Decoder(
            config.decoder,
            eoi,
            cache_config,
            quant_config,
            prefix=f"{prefix}.decoder",
        )


# --------------------------------------------------------------------------
# Multimodal processing: the whole encoder input (text, optionally with
# images) is ONE "text"-modality item, expanded into leading placeholder
# tokens of the (single, decoder) prompt.  An item is either a plain string
# or {"text": str, "images": [PIL.Image, ...]} — the images are part of the
# item because the encoder consumes text and image soft tokens as a single
# atomic sequence.
# --------------------------------------------------------------------------


def _split_encoder_item(item: object) -> tuple[str, list]:
    """Normalize a "text" mm item to (text, images)."""
    if isinstance(item, str):
        return item, []
    if isinstance(item, Mapping):
        text = item.get("text", "")
        images = item.get("images") or []
        if not isinstance(text, str):
            raise TypeError(f"'text' must be a string, got {type(text)}")
        if not isinstance(images, (list, tuple)):
            images = [images]
        return text, list(images)
    raise TypeError(
        "T5Gemma2 'text' items must be a string or a "
        "{'text': str, 'images': [...]} mapping, got " + str(type(item))
    )


class T5Gemma2EncoderItems(ProcessorBatchItems):
    """Items of the "text" modality: str or {"text":..., "images": [...]}"""

    def __init__(self, data) -> None:
        if data is None:
            data = [""]
        elif isinstance(data, (str, Mapping)):
            data = [data]
        super().__init__(data, "text")


class T5Gemma2DataParser(MultiModalDataParser):
    def _parse_text_data(self, data):
        if data is None or (hasattr(data, "__len__") and not len(data)):
            return T5Gemma2EncoderItems(None)
        if isinstance(data, (str, Mapping)):
            return T5Gemma2EncoderItems(data)
        if isinstance(data, (list, tuple)) and all(
            isinstance(item, (str, Mapping)) for item in data
        ):
            return T5Gemma2EncoderItems(list(data))
        raise TypeError(
            "Text data must be a string, a {'text', 'images'} mapping, or "
            f"a list of those; got {type(data)}"
        )

    def _get_subparsers(self):
        return {"text": self._parse_text_data}


class T5Gemma2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs):
        # Resolves to Gemma3Processor (the auto-mapped processor for
        # model_type "t5gemma2"); only needed for image items.
        return self.ctx.get_hf_processor(**kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"text": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        return {"text": seq_len}

    def get_data_parser(self) -> MultiModalDataParser:
        return T5Gemma2DataParser()

    def get_placeholder_token_id(self) -> int:
        config = self.get_hf_config()
        placeholder = getattr(config, "image_token_index", None)
        if placeholder is None:
            placeholder = config.encoder.image_token_index
        vocab_size = config.decoder.vocab_size
        if not 0 <= placeholder < vocab_size:
            raise ValueError(
                f"Placeholder token id {placeholder} is outside the "
                f"vocabulary (size {vocab_size})."
            )
        return placeholder

    def tokenize_encoder_text(self, text: str) -> list[int]:
        tokenizer = self.get_tokenizer()
        # Gemma tokenizers prepend BOS by default; this matches how HF users
        # feed the T5Gemma2 encoder (tokenizer defaults).
        return tokenizer.encode(text, add_special_tokens=True)

    def process_encoder_item(
        self, item: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One encoder item -> (encoder_input_ids (1, N), pixel_values).

        Image items go through the real HF processor (Gemma3Processor),
        which expands each <start_of_image> marker in the text into the
        boi + mm_tokens_per_image image-soft-tokens + eoi block and
        preprocesses the images; pure-text items use the tokenizer alone.
        Text-only items get an empty pixel tensor so that every item
        carries a uniform field set.
        """
        text, images = _split_encoder_item(item)
        if images:
            hf_processor = self.get_hf_processor()
            processed = hf_processor(
                text=text, images=images, return_tensors="pt"
            )
            return processed["input_ids"], processed["pixel_values"]
        encoder_input_ids = torch.tensor([self.tokenize_encoder_text(text)])
        return encoder_input_ids, torch.zeros((0, 3, 1, 1))


class T5Gemma2DummyInputsBuilder(
    BaseDummyInputsBuilder[T5Gemma2ProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_texts = mm_counts.get("text", 0)
        if num_texts == 0:
            return {}
        # Leave headroom for BOS/specials added by the tokenizer.
        num_words = max(1, seq_len - 8)
        return {"text": " ".join(["word"] * num_words)}


class T5Gemma2MultiModalProcessor(
    BaseMultiModalProcessor[T5Gemma2ProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        """Process the encoder item and tokenize the decoder prompt.

        Produces ``encoder_input_ids`` + ``pixel_values`` (from the "text"
        mm item, which may carry images) and ``input_ids`` (the decoder
        prompt, BOS included via tokenizer defaults).
        """
        from transformers.feature_extraction_utils import BatchFeature

        tokenizer = self.info.get_tokenizer()
        result: dict[str, Any] = {}

        if mm_data and "texts" in mm_data:
            encoder_items = mm_data["texts"]
            encoder_item = encoder_items[0] if encoder_items else ""
            encoder_input_ids, pixel_values = (
                self.info.process_encoder_item(encoder_item)
            )
            max_model_len = self.info.ctx.model_config.max_model_len
            if encoder_input_ids.shape[-1] >= max_model_len:
                raise ValueError(
                    f"The encoder input needs {encoder_input_ids.shape[-1]} "
                    f"tokens but max_model_len is {max_model_len} (each "
                    "image costs ~mm_tokens_per_image+6 tokens and the "
                    "generated tokens share the same budget)."
                )
            result["encoder_input_ids"] = encoder_input_ids
            # Keep the batch dim so the batched("text") field sees one
            # entry per item: (1, num_images, 3, H, W).
            result["pixel_values"] = pixel_values.unsqueeze(0)

        if (
            isinstance(prompt, (list, tuple))
            and len(prompt) > 0
            and isinstance(prompt[0], int)
        ):
            result["input_ids"] = torch.tensor([list(prompt)])
        else:
            # The decoder sequence must start with the decoder BOS; Gemma
            # tokenizers add it by default (an empty prompt becomes [BOS]).
            result["input_ids"] = tokenizer(
                prompt if prompt else "",
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"]

        return BatchFeature(result)

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            encoder_input_ids=MultiModalFieldConfig.batched("text"),
            pixel_values=MultiModalFieldConfig.batched("text"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Prepend one placeholder token per encoder token.

        The placeholders reserve paged-KV slots for the encoder prefix and
        are replaced by encoder-output embeddings at runtime.  The count
        must match ``embed_multimodal``'s output length exactly, so it is
        read from the already-processed ``encoder_input_ids`` when
        available, else recomputed with the identical processing call.
        """
        if mm_items.get_count("text", strict=False) == 0:
            return []

        text_items = mm_items.get_items("text", T5Gemma2EncoderItems)
        placeholder = self.info.get_placeholder_token_id()

        def get_insertion(item_idx: int) -> list[int]:
            num_tokens = None
            try:
                item_kwargs = out_mm_kwargs["text"][item_idx]
                num_tokens = int(
                    item_kwargs["encoder_input_ids"].data.shape[-1]
                )
            except (KeyError, IndexError, TypeError, AttributeError):
                pass
            if num_tokens is None:
                encoder_input_ids, _ = self.info.process_encoder_item(
                    text_items.get(item_idx)
                )
                num_tokens = encoder_input_ids.shape[-1]
            return [placeholder] * num_tokens

        return [
            PromptInsertion(
                modality="text",
                target=PromptIndexTargets.start(),
                insertion=get_insertion,
            )
        ]


# Vision-tower weights are not handled by the name mapping: load_weights
# routes them (prefix-stripped) into SiglipVisionModel.load_weights, which
# does its own q/k/v fusion.
_VISION_TOWER_PREFIX = "model.encoder.vision_tower."
_SKIP_WEIGHT_PREFIXES = (_VISION_TOWER_PREFIX,)

_STACKED_WEIGHT_MAPPING = [
    # (vllm_fused_param, hf_shard_name, shard_id)
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]


_ENCODER_PREFIX = "model.encoder."
_ENCODER_TEXT_PREFIX = "model.encoder.text_model."


def map_t5gemma2_weight_name(
    name: str,
) -> tuple[str, str | int | None] | None:
    """Map an HF checkpoint weight name to (vllm param name, shard_id).

    Returns None for weights that are intentionally skipped (vision tower).
    ``shard_id`` is None for plain 1:1 weights and a shard identifier for
    weights that load into a fused vLLM parameter (qkv_proj, gate_up_proj).

    Both known checkpoint layouts are accepted: the transformers-5.13 module
    tree nests the text encoder under ``model.encoder.text_model.*``, while
    the Hub checkpoints (e.g. google/t5gemma-2-270m-270m) store it flat as
    ``model.encoder.embed_tokens/layers/norm``.  Flat names are normalized
    to the nested layout this plugin's module tree uses.
    """
    if name.startswith(_SKIP_WEIGHT_PREFIXES):
        return None
    # HF stores the LM head as lm_head.out_proj; vLLM's ParallelLMHead is
    # the module itself.
    if name.startswith("lm_head.out_proj."):
        name = name.replace("lm_head.out_proj.", "lm_head.")
    if name.startswith(_ENCODER_PREFIX) and not name.startswith(
        (_ENCODER_TEXT_PREFIX, "model.encoder.multi_modal_projector.")
    ):
        name = _ENCODER_TEXT_PREFIX + name[len(_ENCODER_PREFIX):]
    for param_name, shard_name, shard_id in _STACKED_WEIGHT_MAPPING:
        if f".{shard_name}." in name:
            return name.replace(shard_name, param_name), shard_id
    return name, None


@MULTIMODAL_REGISTRY.register_processor(
    T5Gemma2MultiModalProcessor,
    info=T5Gemma2ProcessingInfo,
    dummy_inputs=T5Gemma2DummyInputsBuilder,
)
class T5Gemma2ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsQuant
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("text"):
            return None
        raise ValueError("Only the 'text' modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self._validate_vllm_config(vllm_config)

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.model = T5Gemma2Model(
            config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        decoder_config = config.decoder
        self.placeholder_token_id = getattr(
            config, "image_token_index", None
        )
        if self.placeholder_token_id is None:
            self.placeholder_token_id = config.encoder.image_token_index

        self.lm_head = ParallelLMHead(
            decoder_config.vocab_size,
            decoder_config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )
        self.logits_processor = LogitsProcessor(
            decoder_config.vocab_size,
            soft_cap=decoder_config.final_logit_softcapping,
        )

        # Mask stashed by embed_input_ids for the immediately following
        # forward of the same flattened batch (the runner calls them
        # back-to-back).  Consumed (popped) by forward.
        self._pending_prefix_mask: torch.Tensor | None = None

    @staticmethod
    def _validate_vllm_config(vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config
        config = model_config.hf_config

        if model_config.is_encoder_decoder:
            raise ValueError(
                "T5Gemma2 must run on vLLM's decoder-only path.  Pass "
                'hf_overrides={"is_encoder_decoder": False} when '
                "constructing the LLM (or the equivalent callable "
                "vllm_bart_plugin.t5gemma2.t5gemma2_hf_overrides)."
            )

        decoder_config = config.decoder
        has_sliding = any(
            layer_type == "sliding_attention"
            for layer_type in decoder_config.layer_types
        )
        window = decoder_config.sliding_window
        if has_sliding and window is not None:
            if model_config.max_model_len > window:
                raise ValueError(
                    "The T5Gemma2 plugin is exact only when the total "
                    "context fits in the sliding window "
                    f"({window} tokens); got max_model_len="
                    f"{model_config.max_model_len}.  Pass "
                    f"max_model_len<={window}."
                )

        scheduler_config = vllm_config.scheduler_config
        if getattr(
            scheduler_config, "enable_chunked_prefill", False
        ) and not getattr(scheduler_config, "disable_chunked_mm_input", False):
            raise ValueError(
                "T5Gemma2 requires the encoder placeholder run to be "
                "scheduled in one piece.  Pass disable_chunked_mm_input="
                "True (or enable_chunked_prefill=False)."
            )

        cache_config = vllm_config.cache_config
        if getattr(cache_config, "enable_prefix_caching", False):
            raise ValueError(
                "T5Gemma2 does not support prefix caching yet (a cache hit "
                "inside the encoder placeholder run would corrupt the "
                "prefix-KV bookkeeping).  Pass enable_prefix_caching=False."
            )

    # ------------------------------------------------------------------
    # Multimodal interface
    # ------------------------------------------------------------------

    def get_language_model(self) -> nn.Module:
        return self.model.decoder

    def _parse_and_validate_encoder_input(
        self, **kwargs: object
    ) -> list[torch.Tensor]:
        encoder_input_ids = kwargs.get("encoder_input_ids")
        if encoder_input_ids is None:
            return []
        if isinstance(encoder_input_ids, torch.Tensor):
            # Batched to (num_items, 1, N) or (num_items, N).
            items = encoder_input_ids.flatten(0, -2).unbind(0)
            return [item for item in items]
        if isinstance(encoder_input_ids, (list, tuple)):
            return [
                item.flatten() if isinstance(item, torch.Tensor) else item
                for item in encoder_input_ids
            ]
        raise ValueError(
            "Incorrect type of encoder_input_ids: "
            f"{type(encoder_input_ids)}"
        )

    def _parse_pixel_values(
        self, pixel_values: object, num_items: int
    ) -> list[torch.Tensor | None]:
        """Per-item image tensors aligned with the encoder_input_ids items.

        Text-only items carry an empty (0, 3, 1, 1) tensor (emitted by the
        processor so every item has a uniform field set); those become None.
        """
        if pixel_values is None:
            return [None] * num_items
        if isinstance(pixel_values, torch.Tensor):
            items = list(pixel_values.unbind(0))
        elif isinstance(pixel_values, (list, tuple)):
            items = list(pixel_values)
        else:
            raise ValueError(
                f"Incorrect type of pixel_values: {type(pixel_values)}"
            )
        parsed: list[torch.Tensor | None] = []
        for item in items:
            # Squeeze a leading batch dim of 1: (1, n, 3, H, W) -> (n, ...)
            if isinstance(item, torch.Tensor) and item.dim() == 5:
                item = item.squeeze(0)
            if isinstance(item, torch.Tensor) and item.numel() == 0:
                item = None
            parsed.append(item)
        if len(parsed) != num_items:
            raise ValueError(
                f"Got {len(parsed)} pixel_values items for "
                f"{num_items} encoder text items."
            )
        return parsed

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        encoder_input_ids_list = self._parse_and_validate_encoder_input(
            **kwargs
        )
        if not encoder_input_ids_list:
            return []
        pixel_values_list = self._parse_pixel_values(
            kwargs.get("pixel_values"), len(encoder_input_ids_list)
        )

        encoder_outputs: list[torch.Tensor] = []
        for encoder_input_ids, pixel_values in zip(
            encoder_input_ids_list, pixel_values_list
        ):
            encoder_input_ids = encoder_input_ids.flatten()
            positions = torch.arange(
                encoder_input_ids.numel(),
                dtype=torch.long,
                device=encoder_input_ids.device,
            )

            inputs_embeds = None
            if pixel_values is not None:
                # HF flow: embed the text (incl. eoi handling), then
                # overwrite the image-soft-token rows with the projected
                # SigLIP features, then run the text encoder on embeds.
                image_features = self.model.encoder.get_image_features(
                    pixel_values.to(device=encoder_input_ids.device)
                )
                inputs_embeds = self.model.encoder.text_model.embed_tokens(
                    encoder_input_ids
                ).clone()
                image_mask = encoder_input_ids == self.placeholder_token_id
                num_image_rows = int(image_mask.sum())
                if num_image_rows != image_features.shape[0] * (
                    image_features.shape[1]
                ):
                    raise ValueError(
                        f"Encoder text has {num_image_rows} image-soft-token "
                        f"rows (id {self.placeholder_token_id}) but the "
                        "vision tower produced "
                        f"{image_features.shape[0] * image_features.shape[1]}"
                        " feature rows.  Each image needs exactly one "
                        "<start_of_image> marker in the encoder text."
                    )
                inputs_embeds[image_mask] = image_features.flatten(0, 1).to(
                    inputs_embeds.dtype
                )

            encoder_outputs.append(
                self.model.encoder.text_model(
                    encoder_input_ids, positions, inputs_embeds=inputs_embeds
                )
            )
        return encoder_outputs

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or is_multimodal is None:
            self._pending_prefix_mask = None
            return self.model.decoder.embed_tokens(input_ids)

        # The multimodal rows of the merged embeddings ARE the encoder
        # outputs; remember which rows they are for the upcoming forward.
        self._pending_prefix_mask = is_multimodal
        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _resolve_prefix_ctx(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> PrefixContext | None:
        mask = self._pending_prefix_mask
        self._pending_prefix_mask = None
        if mask is None and input_ids is not None:
            mask = input_ids == self.placeholder_token_id
        if mask is None:
            return None

        mask = mask.to(device=positions.device)
        num_tokens = positions.shape[0]
        if mask.shape[0] > num_tokens:
            mask = mask[:num_tokens]
        elif mask.shape[0] < num_tokens:
            # The runner may pad the batch (e.g. for CUDA graphs); padded
            # rows are never multimodal.
            mask = torch.nn.functional.pad(
                mask, (0, num_tokens - mask.shape[0]), value=False
            )

        if not bool(mask.any()):
            return None

        rope_positions = compute_prefix_rope_positions(mask, positions)
        enc_rows = inputs_embeds[mask]
        return PrefixContext(
            is_prefix=mask, enc_rows=enc_rows, rope_positions=rope_positions
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.decoder.embed_tokens(input_ids)

        prefix_ctx = self._resolve_prefix_ctx(
            input_ids, positions, inputs_embeds
        )
        rope_positions = (
            prefix_ctx.rope_positions if prefix_ctx is not None else positions
        )
        return self.model.decoder(rope_positions, inputs_embeds, prefix_ctx)

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Raw checkpoint tensors kept for embedding tying.
        encoder_embed_weight: torch.Tensor | None = None
        encoder_eoi_embedding: torch.Tensor | None = None
        vision_weights: list[tuple[str, torch.Tensor]] = []

        for name, loaded_weight in weights:
            if name.startswith(_VISION_TOWER_PREFIX):
                vision_name = name[len(_VISION_TOWER_PREFIX):]
                # vLLM's SiglipVisionModel expects vision_model.*; some
                # transformers versions nest the SigLIP body directly
                # under vision_tower (no vision_model segment).
                if not vision_name.startswith("vision_model."):
                    vision_name = "vision_model." + vision_name
                vision_weights.append((vision_name, loaded_weight))
                continue
            mapped = map_t5gemma2_weight_name(name)
            if mapped is None:
                continue
            name, shard_id = mapped

            if name == "model.encoder.text_model.embed_tokens.weight":
                encoder_embed_weight = loaded_weight
            elif name == "model.encoder.text_model.embed_tokens.eoi_embedding":
                encoder_eoi_embedding = loaded_weight

            if name not in params_dict:
                logger.warning("Skipping unexpected T5Gemma2 weight %r", name)
                continue
            param = params_dict[name]
            if shard_id is not None:
                param.weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # SiglipVisionModel does its own stacked-shard fusion.
        if vision_weights:
            loaded_vision = self.model.encoder.vision_tower.load_weights(
                vision_weights
            )
            loaded_params.update(
                _VISION_TOWER_PREFIX + name for name in loaded_vision
            )

        # decoder.embed_tokens and lm_head are tied to the encoder embedding
        # and may be absent from the checkpoint.
        tied = [
            ("model.decoder.embed_tokens.weight", encoder_embed_weight),
            ("model.decoder.embed_tokens.eoi_embedding",
             encoder_eoi_embedding),
            ("lm_head.weight", encoder_embed_weight),
        ]
        for name, source in tied:
            if name in loaded_params:
                continue
            if source is None:
                raise ValueError(
                    f"Cannot tie {name}: "
                    "model.encoder.text_model.embed_tokens was not found "
                    "in the checkpoint."
                )
            param = params_dict[name]
            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )
            weight_loader(param, source)
            loaded_params.add(name)

        return loaded_params
