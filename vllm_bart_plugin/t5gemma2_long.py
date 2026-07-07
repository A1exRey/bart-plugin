# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context (beyond the decoder sliding window) support for T5Gemma2.

Within the sliding window the plugin's prefix-KV scheme is exact with plain
causal paged attention (see t5gemma2.py).  Beyond it, HF's semantics for the
merged self+cross attention are:

- every decoder query attends ALL N encoder-prefix rows (cross keys are never
  windowed), and
- decoder-self rows within a causal look-behind window ``w`` on
  ``sliding_attention`` layers (everything, on ``full_attention`` layers).

vLLM's stock machinery cannot express that (a ``SlidingWindowSpec`` manager
physically frees the out-of-window blocks holding the prefix rows), so long
mode replaces the decoder attention with an exact two-pass decomposition over
the SAME paged KV cache:

- retention: every decoder layer reports ``FullAttentionSpec`` (the layer is
  constructed with no sliding window), so blocks are never freed;
- the placeholder run is padded to ``N_pad = round_up(N_real, PREFIX_PAD)``
  rows (PREFIX_PAD == KV block size), so the decoder-self region starts on a
  block boundary; only the first N_real rows carry encoder embeddings
  (``is_embed``), and rows [N_real, N_pad) are attended by NOTHING;
- pass A: every query over KV rows [0, N_real), non-causal, unwindowed;
- pass B: every query over the block table shifted by N_pad/block_size
  blocks (the self region re-addressed from block 0), causal, with window
  ``[w-1, 0]`` on sliding layers;
- the passes' outputs are combined with vLLM's exact LSE-merge kernel
  (``merge_attn_states``, the cascade-attention primitive).

Key sets are disjoint for every query, so the merge equals one softmax over
HF's merged mask — verified against the HF reference on CPU in
tests/test_t5gemma2_math.py (``_two_pass_attention`` and the beyond-window
tests).

Per-request N at decode time comes from ``hf_config.is_mm_prefix_lm``: the
v1 runner then republishes each running request's multimodal embed ranges as
``CommonAttentionMetadata.mm_req_doc_ranges`` on EVERY step, and the builder
below turns them into per-request prefix lengths.  The runner skips ranges
longer than ``hf_text_config.sliding_window`` (which resolves to
``config.decoder`` for T5Gemma2), which is why
``t5gemma2_long_context_hf_overrides`` nulls ``decoder.sliding_window`` and
stashes the real value in ``decoder.plugin_prefix_lm_sliding_window`` for
the model to read.

v1 limitations (validated with actionable errors in t5gemma2.py):
FLASH_ATTN backend only (the two passes need FlashAttention's LSE return),
``block_size == 16``, no speculative decoding, unquantized KV cache, and no
CUDA-graph capture of the attention (piecewise graphs still work;
capture-time metadata carries no multimodal ranges, so captured prefix
lengths would be wrong).
"""

import functools
import logging
from dataclasses import dataclass, fields

import torch

from .t5gemma2_prefix import (
    PREFIX_PAD,
    prefix_lens_from_doc_ranges,
    round_up,
    shifted_block_table,
)

logger = logging.getLogger(__name__)

# Custom attribute on config.decoder holding the real sliding window while
# decoder.sliding_window itself is nulled (see module docstring).
SLIDING_WINDOW_STASH_ATTR = "plugin_prefix_lm_sliding_window"

__all__ = [
    "HAS_LONG_CONTEXT_VLLM",
    "SLIDING_WINDOW_STASH_ATTR",
    "T5Gemma2PrefixAttention",
    "is_long_context_mode",
    "long_context_sliding_window",
    "t5gemma2_long_context_hf_overrides",
]


def t5gemma2_long_context_hf_overrides(hf_config):
    """``hf_overrides`` callable enabling contexts beyond the sliding window.

    Usage::

        from vllm_bart_plugin.t5gemma2 import (
            t5gemma2_long_context_hf_overrides,
        )
        llm = LLM(model="google/t5gemma-2-270m-270m",
                  hf_overrides=t5gemma2_long_context_hf_overrides,
                  max_model_len=8192, ...)

    Does everything ``t5gemma2_hf_overrides`` does, plus:

    - sets ``is_mm_prefix_lm`` so the vLLM runner republishes each request's
      multimodal ranges every step (the decode-time N channel).  Side
      effects: vLLM force-disables chunked mm input (already a plugin
      requirement) and raises ``max_num_batched_tokens`` to fit the largest
      multimodal item;
    - stashes ``decoder.sliding_window`` in
      ``decoder.plugin_prefix_lm_sliding_window`` and nulls the original so
      (a) the runner's range-length skip cannot drop our (long) prefix range
      and (b) vLLM applies no sliding-window handling of its own — the
      window is enforced exactly by the two-pass attention instead.
    """
    hf_config.is_encoder_decoder = False
    decoder_config = getattr(hf_config, "decoder", None)
    if decoder_config is None:
        # vLLM probes the hf_overrides callable with a dummy bare
        # PreTrainedConfig just to read model_type (transformers_utils/
        # config.py) BEFORE loading the real config; probe mutations are
        # discarded.  The real T5Gemma2 config (which has .decoder) gets
        # the full treatment on the second call.
        return hf_config
    if not hasattr(decoder_config, SLIDING_WINDOW_STASH_ATTR):
        setattr(
            decoder_config,
            SLIDING_WINDOW_STASH_ATTR,
            decoder_config.sliding_window,
        )
        decoder_config.sliding_window = None
    hf_config.is_mm_prefix_lm = True
    return hf_config


def is_long_context_mode(hf_config) -> bool:
    return hasattr(hf_config.decoder, SLIDING_WINDOW_STASH_ATTR)


def long_context_sliding_window(hf_config) -> "int | None":
    """The decoder's real sliding window, wherever it currently lives."""
    if is_long_context_mode(hf_config):
        return getattr(hf_config.decoder, SLIDING_WINDOW_STASH_ATTR)
    return hf_config.decoder.sliding_window


# The attention machinery needs vLLM's v1 extension points (vLLM >= 0.24 as
# pinned).  The config helpers above must stay importable without them.
try:
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import (
        AttentionCGSupport,
        AttentionType,
        subclass_attention_backend_with_overrides,
    )
    from vllm.v1.attention.selector import get_attn_backend

    HAS_LONG_CONTEXT_VLLM = True
except ImportError:  # pragma: no cover - legacy vLLM
    HAS_LONG_CONTEXT_VLLM = False


if HAS_LONG_CONTEXT_VLLM:

    @functools.lru_cache
    def create_t5gemma2_prefix_backend(underlying_attn_backend):
        # Deferred: pulls in vllm_flash_attn, which may be absent on
        # CPU-only installs where the rest of the plugin still works.
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadata,
            flash_attn_varlen_func,
        )
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

        underlying_builder = underlying_attn_backend.get_builder_cls()
        underlying_impl = underlying_attn_backend.get_impl_cls()

        @dataclass
        class T5Gemma2PrefixAttentionMetadata(FlashAttentionMetadata):
            # Per-request number of real encoder rows (N_real); 0 for
            # requests without multimodal input (dummy/capture rows).
            prefix_seqused: "torch.Tensor | None" = None
            # Per-request decoder-self KV length:
            # seq_len - round_up(N_real, PREFIX_PAD), clamped at 0.
            self_seqused: "torch.Tensor | None" = None
            # Stock block table with each row shifted left by
            # N_pad/block_size blocks, so the self region starts at block 0.
            self_block_table: "torch.Tensor | None" = None
            max_prefix_len: int = 1
            max_self_len: int = 1

        class T5Gemma2PrefixAttentionBuilder(underlying_builder):
            # Capture-time CommonAttentionMetadata has no mm ranges (the
            # dummy batch has no requests), so captured prefix lengths
            # would be baked as 0.  Piecewise CUDA graphs are unaffected.
            _cudagraph_support = AttentionCGSupport.NEVER
            supports_update_block_table = False

            @classmethod
            def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
                return AttentionCGSupport.NEVER

            def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
                super().__init__(
                    kv_cache_spec, layer_names, vllm_config, device
                )
                if kv_cache_spec.block_size != PREFIX_PAD:
                    raise ValueError(
                        "T5Gemma2 long-context mode requires block_size="
                        f"{PREFIX_PAD} (the prefix pad granularity); got "
                        f"{kv_cache_spec.block_size}.  Pass block_size="
                        f"{PREFIX_PAD}."
                    )
                # The stock FA3 ahead-of-time scheduler plans a SINGLE call
                # over seq_lens; that plan is wrong for both passes.
                # scheduler_metadata is optional, so never produce it.
                self.aot_schedule = False

            def use_cascade_attention(self, *args, **kwargs) -> bool:
                return False

            def build(
                self,
                common_prefix_len,
                common_attn_metadata,
                fast_build=False,
            ):
                base = super().build(
                    common_prefix_len, common_attn_metadata, fast_build
                )
                num_reqs = common_attn_metadata.num_reqs
                prefix_lens = prefix_lens_from_doc_ranges(
                    common_attn_metadata.mm_req_doc_ranges, num_reqs
                )
                pad_lens = [round_up(n, PREFIX_PAD) for n in prefix_lens]

                prefix_seqused = torch.tensor(
                    prefix_lens, dtype=torch.int32, device=self.device
                )
                pad_lens_t = torch.tensor(
                    pad_lens, dtype=torch.int32, device=self.device
                )
                self_seqused = (base.seq_lens - pad_lens_t).clamp_(min=0)
                shift_blocks = pad_lens_t // self.block_size
                self_block_table = shifted_block_table(
                    base.block_table, shift_blocks
                )

                metadata = T5Gemma2PrefixAttentionMetadata(
                    **{f.name: getattr(base, f.name) for f in fields(base)},
                    prefix_seqused=prefix_seqused,
                    self_seqused=self_seqused,
                    self_block_table=self_block_table,
                    # max_seqlen_k for pass A / pass B.  Rows with
                    # seqused_k == 0 are fine (CrossAttention zeroes cached
                    # decode rows the same way); keep the max >= 1.
                    max_prefix_len=max(max(prefix_lens, default=0), 1),
                    max_self_len=max(base.max_seq_len, 1),
                )
                # Our forward never takes the stock single-call path, so
                # the FA4 PrefixLM mask-mod branch must not either.
                metadata.mm_prefix_range_tensor = None
                return metadata

        class T5Gemma2PrefixAttentionImpl(underlying_impl):
            def __init__(self, *args, self_window=None, **kwargs):
                super().__init__(*args, **kwargs)
                if self.sliding_window != (-1, -1):
                    raise ValueError(
                        "T5Gemma2PrefixAttention must be constructed "
                        "without a sliding window (the window is applied "
                        f"by pass B); got {self.sliding_window}."
                    )
                if getattr(self, "sinks", None) is not None:
                    raise ValueError(
                        "Attention sinks are incompatible with the "
                        "two-pass T5Gemma2 long-context attention."
                    )
                self._pass_b_window = (
                    [self_window - 1, 0] if self_window is not None else None
                )

            def forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=None,
                output_block_scale=None,
            ):
                if attn_metadata is None:
                    # Profiling run.
                    return output.fill_(0)
                if not isinstance(
                    attn_metadata, T5Gemma2PrefixAttentionMetadata
                ):
                    return super().forward(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        output_scale,
                        output_block_scale,
                    )
                if output_scale is not None or output_block_scale is not None:
                    raise NotImplementedError(
                        "fused output quantization is not supported for "
                        "T5Gemma2PrefixAttention"
                    )
                if getattr(self, "dcp_world_size", 1) > 1:
                    raise NotImplementedError(
                        "decode context parallelism is not supported in "
                        "T5Gemma2 long-context mode"
                    )
                assert attn_metadata.causal is True

                num_actual_tokens = attn_metadata.num_actual_tokens
                key_cache, value_cache = kv_cache.unbind(1)

                q = query[:num_actual_tokens]
                cu_seqlens_q = attn_metadata.query_start_loc
                max_seqlen_q = attn_metadata.max_query_len

                descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
                q_descale = (
                    layer._q_scale.expand(descale_shape)
                    if self.supports_quant_query_input
                    else None
                )
                k_descale = layer._k_scale.expand(descale_shape)
                v_descale = layer._v_scale.expand(descale_shape)

                # Pass A: every query over the N_real encoder-prefix rows,
                # fully visible (HF: cross keys are never windowed).
                prefix_out, prefix_lse = flash_attn_varlen_func(
                    q=q,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=attn_metadata.prefix_seqused,
                    max_seqlen_k=attn_metadata.max_prefix_len,
                    softmax_scale=self.scale,
                    causal=False,
                    window_size=None,
                    block_table=attn_metadata.block_table,
                    softcap=self.logits_soft_cap,
                    return_softmax_lse=True,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=attn_metadata.max_num_splits,
                )

                # Pass B: every query over the decoder-self region (block
                # table shifted to start at row N_pad), causal, with the
                # layer's own look-behind window on sliding layers.
                self_out, self_lse = flash_attn_varlen_func(
                    q=q,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=attn_metadata.self_seqused,
                    max_seqlen_k=attn_metadata.max_self_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self._pass_b_window,
                    block_table=attn_metadata.self_block_table,
                    softcap=self.logits_soft_cap,
                    return_softmax_lse=True,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=attn_metadata.max_num_splits,
                )

                # Exact single-softmax combination; queries with an empty
                # side (prefix-row queries in pass B, no-mm requests in
                # pass A) have +-inf LSE there, which the merge kernels
                # neutralize.
                merge_attn_states(
                    output[:num_actual_tokens],
                    prefix_out,
                    prefix_lse,
                    self_out,
                    self_lse,
                )
                return output

        return subclass_attention_backend_with_overrides(
            name_prefix="T5Gemma2Prefix_",
            attention_backend_cls=underlying_attn_backend,
            overrides={
                "get_builder_cls": lambda: T5Gemma2PrefixAttentionBuilder,
                "get_impl_cls": lambda: T5Gemma2PrefixAttentionImpl,
            },
        )

    class T5Gemma2PrefixAttention(Attention):
        """Decoder attention for T5Gemma2 long-context mode.

        Used for EVERY decoder layer (full-attention layers simply run
        pass B unwindowed): plain causal attention would let full-attention
        queries see the [N_real, N_pad) padding rows, so all layers must
        use the two-pass exclusion.  KV cache writes stay on the stock
        path; the KV-cache spec is the base class's ``FullAttentionSpec``
        (the layer has no sliding window), which retains all blocks for
        the request lifetime.
        """

        def __init__(
            self,
            *,
            self_window: "int | None",
            cache_config=None,
            **kwargs,
        ):
            kv_cache_dtype = (
                cache_config.cache_dtype
                if cache_config is not None
                else "auto"
            )
            underlying = get_attn_backend(
                kwargs["head_size"],
                torch.get_default_dtype(),
                kv_cache_dtype,
                attn_type=AttentionType.DECODER,
            )
            if underlying.get_name() != "FLASH_ATTN":
                raise ValueError(
                    "T5Gemma2 long-context mode requires the FLASH_ATTN "
                    "backend (the two-pass attention needs FlashAttention's "
                    "LSE return); resolved backend: "
                    f"{underlying.get_name()}.  Note FlashAttention kernels "
                    "are fp16/bf16-only, so dtype=float32 always resolves "
                    "elsewhere -- run long mode in bfloat16 (fp16 overflows "
                    "Gemma-family activations).  Within the sliding window, "
                    "other backends (and fp32) work without long mode."
                )
            super().__init__(
                cache_config=cache_config,
                attn_backend=create_t5gemma2_prefix_backend(underlying),
                attn_type=AttentionType.DECODER,
                self_window=self_window,
                **kwargs,
            )
            # Long-context overrides null decoder.sliding_window; nothing
            # may reintroduce one here (it would change the KV spec AND
            # double-apply the window in pass B's underlying call).
            assert self.sliding_window is None, (
                "T5Gemma2PrefixAttention constructed with a sliding window; "
                "was t5gemma2_long_context_hf_overrides applied?"
            )

else:  # pragma: no cover - legacy vLLM

    class T5Gemma2PrefixAttention:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "T5Gemma2 long-context mode requires vLLM's v1 attention "
                "extension points (vLLM 0.24); this vLLM is too old.  "
                "Within the sliding window the plugin works without long "
                "mode."
            )
