# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU integration test: the real vLLM T5Gemma2 modules vs HF transformers.

This goes one level deeper than test_t5gemma2_math.py: it builds the actual
vLLM module tree from vllm_bart_plugin/t5gemma2.py (fused QKV, GemmaRMSNorm,
vLLM rotary embedding, scaled vocab-parallel embeddings), loads a real tiny
HF checkpoint through ``load_weights`` (exercising the stacked-shard mapping
and embedding tying), and compares:

1. encoder outputs against the HF encoder (real MMEncoderAttention on CPU),
2. full prefix-KV forward logits against HF teacher-forced logits, with
   vLLM's paged ``Attention`` swapped for a causal-SDPA stand-in (the paged
   backend needs a GPU; everything around it is the real plugin code).

If this passes, the remaining GPU risk is confined to the paged attention
backend itself and the processor/runner integration.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
transformers = pytest.importorskip("transformers", minversion="5.0")

import torch.nn.functional as F  # noqa: E402


@pytest.fixture(scope="module")
def vllm_cpu_env():
    """Single-process distributed env + CPU vLLM config context."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(vllm_config):
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method="tcp://127.0.0.1:29547",
                backend="gloo",
            )
        if not model_parallel_is_initialized():
            initialize_model_parallel(1, 1)
        yield vllm_config


class FakeCausalAttention(torch.nn.Module):
    """Stand-in for vLLM's paged ``Attention`` on CPU.

    Accepts the same constructor arguments and computes plain causal
    attention over one flattened prefill sequence — which is exactly what
    the paged backend computes for a single-request prefill.
    """

    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads=None,
        cache_config=None,
        quant_config=None,
        logits_soft_cap=None,
        prefix="",
        attn_type=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        assert logits_soft_cap is None, "tiny config has no attn softcap"

    def forward(self, q, k, v):
        total = q.shape[0]
        q = q.view(1, total, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(1, total, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = v.view(1, total, self.num_kv_heads, self.head_size).transpose(1, 2)
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scale
        )
        return out.transpose(1, 2).reshape(total, -1)


def _tiny_hf_model():
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2Config,
        T5Gemma2DecoderConfig,
        T5Gemma2EncoderConfig,
        T5Gemma2TextConfig,
    )
    from transformers.models.t5gemma2.modeling_t5gemma2 import (
        T5Gemma2ForConditionalGeneration as HFModel,
    )

    text_kwargs = dict(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        sliding_window=16,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        query_pre_attn_scalar=8,
    )
    vision_kwargs = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=14,
        patch_size=7,
    )
    torch.manual_seed(7)
    config = T5Gemma2Config(
        encoder=T5Gemma2EncoderConfig(
            text_config=T5Gemma2TextConfig(**text_kwargs),
            vision_config=vision_kwargs,
            mm_tokens_per_image=4,
        ),
        decoder=T5Gemma2DecoderConfig(**text_kwargs),
    )
    hf_model = HFModel(config)
    hf_model.config._attn_implementation = "eager"

    # Randomize RMSNorm weights (zero-initialized by HF) so norm placement
    # bugs cannot cancel out.
    gen = torch.Generator().manual_seed(8)
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if "norm" in name.lower():
                param.uniform_(-0.3, 0.3, generator=gen)
    hf_model.eval()
    return hf_model.float(), config


def _build_vllm_shell(config, vllm_cpu_env):
    import vllm_bart_plugin.t5gemma2 as t5
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
    )

    with set_current_vllm_config(vllm_cpu_env):
        vllm_model = t5.T5Gemma2ForConditionalGeneration.__new__(
            t5.T5Gemma2ForConditionalGeneration
        )
        torch.nn.Module.__init__(vllm_model)
        vllm_model.config = config
        vllm_model.model = t5.T5Gemma2Model(
            config, cache_config=None, quant_config=None, prefix="model"
        )
        vllm_model.placeholder_token_id = 98
        vllm_model.lm_head = ParallelLMHead(
            config.decoder.vocab_size,
            config.decoder.hidden_size,
            prefix="lm_head",
        )
        vllm_model.logits_processor = LogitsProcessor(
            config.decoder.vocab_size,
            soft_cap=config.decoder.final_logit_softcapping,
        )
        vllm_model._pending_prefix_mask = None
    return vllm_model


def _load_and_check(vllm_model, weights):
    loaded = vllm_model.load_weights(weights)
    # Everything in the vLLM tree must have been covered (incl. tied).
    param_names = {name for name, _ in vllm_model.named_parameters()}
    missing = param_names - loaded
    assert not missing, f"params never loaded: {missing}"
    vllm_model.eval()
    return vllm_model


@pytest.fixture(scope="module")
def loaded_models(vllm_cpu_env, monkeypatch_module):
    """Tiny HF model + the vLLM plugin model loaded from its state dict."""
    import vllm_bart_plugin.t5gemma2 as t5

    hf_model, config = _tiny_hf_model()

    monkeypatch_module.setattr(t5, "Attention", FakeCausalAttention)

    vllm_model = _build_vllm_shell(config, vllm_cpu_env)
    _load_and_check(vllm_model, hf_model.state_dict().items())
    return hf_model, vllm_model


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@torch.no_grad()
def test_encoder_matches_hf(loaded_models):
    hf_model, vllm_model = loaded_models
    torch.manual_seed(11)
    enc_ids = torch.randint(3, 90, (1, 6))

    ref = hf_model.model.encoder(input_ids=enc_ids).last_hidden_state[0]
    got = vllm_model.model.encoder.text_model(
        enc_ids[0], torch.arange(6)
    )
    torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-5)


@torch.no_grad()
def test_encoder_sliding_window_matches_hf(loaded_models):
    """Inputs LONGER than half the sliding window: the bidirectional
    sliding mask differs from full attention here (window=16 -> tokens see
    only ~8 neighbours each way), so this catches a wrong/missing mask."""
    hf_model, vllm_model = loaded_models
    torch.manual_seed(13)
    n = 14  # > window//2 = 8
    enc_ids = torch.randint(3, 90, (1, n))

    ref = hf_model.model.encoder(input_ids=enc_ids).last_hidden_state[0]
    got = vllm_model.model.encoder.text_model(enc_ids[0], torch.arange(n))
    torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-5)

    # Sanity: the mask really is non-trivial at this length.
    from vllm_bart_plugin.t5gemma2 import bidirectional_sliding_window_mask

    mask = bidirectional_sliding_window_mask(n, 16)
    assert not bool(mask.all())


@torch.no_grad()
def test_flat_checkpoint_layout_loads_identically(
    loaded_models, vllm_cpu_env
):
    """The Hub checkpoints store the text encoder flat under model.encoder.*
    (no text_model segment) — loading that layout must give the same model.
    This reproduces the layout from the user's google/t5gemma-2-270m-270m
    load failure."""
    hf_model, vllm_model = loaded_models

    flat_weights = []
    for name, tensor in hf_model.state_dict().items():
        # Keep vision keys under model.encoder.vision_tower as-is; flatten
        # only the text_model segment, like the real checkpoint.
        flat_weights.append(
            (name.replace("model.encoder.text_model.", "model.encoder."),
             tensor)
        )
    # The real checkpoint also omits the tied decoder/lm_head weights.
    flat_weights = [
        (name, tensor)
        for name, tensor in flat_weights
        if not name.startswith(("model.decoder.embed_tokens.", "lm_head."))
    ]

    other = _build_vllm_shell(vllm_model.config, vllm_cpu_env)
    _load_and_check(other, flat_weights)

    torch.manual_seed(14)
    enc_ids = torch.randint(3, 90, (1, 7))
    a = vllm_model.model.encoder.text_model(enc_ids[0], torch.arange(7))
    b = other.model.encoder.text_model(enc_ids[0], torch.arange(7))
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(
        other.lm_head.weight, vllm_model.lm_head.weight
    )


@torch.no_grad()
def test_full_forward_matches_hf_teacher_forced(loaded_models):
    """End-to-end: embed_multimodal -> prefix merge -> forward -> logits."""
    hf_model, vllm_model = loaded_models
    torch.manual_seed(12)
    n, t = 6, 4
    enc_ids = torch.randint(3, 90, (1, n))
    dec_ids = torch.randint(3, 90, (1, t))

    ref = hf_model(input_ids=enc_ids, decoder_input_ids=dec_ids).logits[0]

    # Emulate the v1 runner's multimodal flow on one request:
    (enc_out,) = vllm_model.embed_multimodal(encoder_input_ids=enc_ids)
    input_ids = torch.cat(
        [
            torch.full((n,), vllm_model.placeholder_token_id),
            dec_ids[0],
        ]
    )
    is_mm = torch.zeros(n + t, dtype=torch.bool)
    is_mm[:n] = True
    inputs_embeds = vllm_model.embed_input_ids(
        input_ids, multimodal_embeddings=[enc_out], is_multimodal=is_mm
    )
    positions = torch.arange(n + t)
    hidden = vllm_model.forward(
        input_ids=None, positions=positions, inputs_embeds=inputs_embeds
    )
    logits = vllm_model.compute_logits(hidden[n:])[:, : ref.shape[-1]]

    torch.testing.assert_close(logits, ref, rtol=2e-4, atol=2e-4)
