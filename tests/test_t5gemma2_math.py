# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only math tests for the T5Gemma2 prefix-KV decode strategy.

These tests do NOT need vLLM or a GPU.  They instantiate a tiny random
HuggingFace ``T5Gemma2ForConditionalGeneration`` and check that the plugin's
prefix-KV formulation of the decoder (encoder outputs injected as leading
placeholder rows of a single causal sequence, re-substituted before every
layer's qkv projection, prefix keys rotated at the constant position N)
reproduces the reference merged self+cross attention exactly.

If these tests pass, the decode *math* is right; what remains for GPU
verification is only the vLLM plumbing (KV cache, processor, weights).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers", minversion="5.0")

import torch.nn.functional as F  # noqa: E402

from vllm_bart_plugin.t5gemma2_prefix import (  # noqa: E402
    compute_prefix_rope_positions,
    substitute_prefix_rows,
)

from transformers.models.t5gemma2.configuration_t5gemma2 import (  # noqa: E402
    T5Gemma2Config,
    T5Gemma2DecoderConfig,
    T5Gemma2EncoderConfig,
    T5Gemma2TextConfig,
)
from transformers.models.t5gemma2.modeling_t5gemma2 import (  # noqa: E402
    T5Gemma2ForConditionalGeneration,
    apply_rotary_pos_emb,
    repeat_kv,
)

VOCAB = 99
HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
SLIDING_WINDOW = 16


def _text_kwargs():
    return dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=512,
        sliding_window=SLIDING_WINDOW,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        query_pre_attn_scalar=8,
    )


def _tiny_vision_kwargs():
    # Small SigLIP so instantiation is cheap; unused by the text-only tests.
    return dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=14,
        patch_size=7,
    )


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = T5Gemma2Config(
        encoder=T5Gemma2EncoderConfig(
            text_config=T5Gemma2TextConfig(**_text_kwargs()),
            vision_config=_tiny_vision_kwargs(),
            mm_tokens_per_image=4,
        ),
        decoder=T5Gemma2DecoderConfig(**_text_kwargs()),
    )
    model = T5Gemma2ForConditionalGeneration(config)
    model.config._attn_implementation = "eager"

    # _init_weights zeroes every RMSNorm weight (scale == 1); randomize them
    # so the test cannot pass with a wrong norm placement by accident.
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight") or (
                "q_norm" in name or "k_norm" in name
            ):
                param.uniform_(-0.3, 0.3, generator=gen)

    model.eval()
    return model.float()


def _prefix_decoder_forward(
    model,
    encoder_hidden_states: torch.Tensor,
    decoder_input_ids: torch.Tensor,
) -> torch.Tensor:
    """The plugin's decode strategy, in pure torch on HF weights.

    Runs the decoder as ONE causal sequence [prefix rows = encoder outputs]
    ++ [decoder token rows] and returns logits for the decoder rows.
    Mirrors what vllm_bart_plugin/t5gemma2.py does with vLLM layers.
    """
    dec = model.model.decoder
    cfg = dec.config
    n = encoder_hidden_states.shape[1]
    t = decoder_input_ids.shape[1]
    total = n + t

    enc_rows = encoder_hidden_states[0]  # (N, H)
    dec_embeds = dec.embed_tokens(decoder_input_ids)[0]  # (T, H), scaled

    # Flattened single-request batch, the shape vLLM presents at prefill.
    hidden = torch.cat([enc_rows, dec_embeds], dim=0)  # (N+T, H)
    is_prefix = torch.zeros(total, dtype=torch.bool)
    is_prefix[:n] = True
    positions = torch.arange(total)  # prefix at 0..N-1, decoder at N..N+T-1
    rope_positions = compute_prefix_rope_positions(is_prefix, positions)

    for i, layer in enumerate(dec.layers):
        layer_type = cfg.layer_types[i]
        attn = layer.self_attn

        residual = hidden
        hs = layer.pre_self_attn_layernorm(hidden)
        # The crux: K/V for prefix rows must come from the RAW encoder
        # output at every layer (HF recomputes cross K/V per layer from the
        # constant encoder_hidden_states with no decoder-side layernorm).
        hs = substitute_prefix_rows(hs, is_prefix, enc_rows)

        q = attn.q_proj(hs).view(1, total, -1, HEAD_DIM).transpose(1, 2)
        k = attn.k_proj(hs).view(1, total, -1, HEAD_DIM).transpose(1, 2)
        v = attn.v_proj(hs).view(1, total, -1, HEAD_DIM).transpose(1, 2)

        q = attn.q_norm(q)
        k = attn.k_norm(k)

        cos, sin = dec.rotary_emb(hidden, rope_positions[None], layer_type)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, attn.num_key_value_groups)
        v = repeat_kv(v, attn.num_key_value_groups)
        # Plain causal attention: under total <= sliding_window this equals
        # both HF mask variants ("sliding_attention" and "full_attention").
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=attn.scaling
        )
        out = out.transpose(1, 2).reshape(1, total, -1)[0]
        out = attn.o_proj(out)

        hidden = residual + layer.post_self_attn_layernorm(out)

        residual = hidden
        hs = layer.pre_feedforward_layernorm(hidden)
        hs = layer.mlp(hs)
        hidden = residual + layer.post_feedforward_layernorm(hs)

    hidden = dec.norm(hidden)
    logits = model.lm_head(hidden[n:])
    softcap = cfg.final_logit_softcapping
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return logits  # (T, VOCAB)


@torch.no_grad()
def test_prefix_formulation_matches_hf_teacher_forced(tiny_model):
    torch.manual_seed(2)
    n, t = 5, 4
    assert n + t <= SLIDING_WINDOW
    enc_ids = torch.randint(3, VOCAB, (1, n))
    dec_ids = torch.randint(3, VOCAB, (1, t))

    ref = tiny_model(input_ids=enc_ids, decoder_input_ids=dec_ids).logits[0]

    enc_out = tiny_model.model.encoder(input_ids=enc_ids).last_hidden_state
    got = _prefix_decoder_forward(tiny_model, enc_out, dec_ids)

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_prefix_formulation_matches_hf_incremental_decode(tiny_model):
    """Step-by-step HF decoding (EncoderDecoderCache) vs the prefix run."""
    torch.manual_seed(3)
    n, t = 6, 5
    enc_ids = torch.randint(3, VOCAB, (1, n))
    dec_ids = torch.randint(3, VOCAB, (1, t))

    enc_out = tiny_model.model.encoder(input_ids=enc_ids).last_hidden_state

    # HF stepwise with cache: exercises their cross-KV cache path.
    from transformers.modeling_outputs import BaseModelOutput

    past = None
    step_logits = []
    for step in range(t):
        out = tiny_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_out),
            decoder_input_ids=dec_ids[:, step : step + 1],
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        step_logits.append(out.logits[0, -1])
    ref = torch.stack(step_logits)

    got = _prefix_decoder_forward(tiny_model, enc_out, dec_ids)

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_prefix_rows_do_not_depend_on_garbage_propagation(tiny_model):
    """Decoder rows must be invariant to what the prefix rows' own attention
    outputs are: corrupting the post-attention prefix rows of the residual
    stream must not change decoder logits (they are re-substituted each
    layer, and no decoder query reads a prefix row's hidden state)."""
    torch.manual_seed(4)
    n, t = 4, 3
    enc_ids = torch.randint(3, VOCAB, (1, n))
    dec_ids = torch.randint(3, VOCAB, (1, t))
    enc_out = tiny_model.model.encoder(input_ids=enc_ids).last_hidden_state

    ref = tiny_model(input_ids=enc_ids, decoder_input_ids=dec_ids).logits[0]

    # Same as _prefix_decoder_forward but with the prefix rows of the
    # residual stream zeroed after every layer.
    dec = tiny_model.model.decoder
    cfg = dec.config
    total = n + t
    enc_rows = enc_out[0]
    hidden = torch.cat([enc_rows, dec.embed_tokens(dec_ids)[0]], dim=0)
    is_prefix = torch.zeros(total, dtype=torch.bool)
    is_prefix[:n] = True
    rope_positions = compute_prefix_rope_positions(
        is_prefix, torch.arange(total)
    )

    for i, layer in enumerate(dec.layers):
        attn = layer.self_attn
        residual = hidden
        hs = layer.pre_self_attn_layernorm(hidden)
        hs = substitute_prefix_rows(hs, is_prefix, enc_rows)
        q = attn.q_proj(hs).view(1, total, -1, HEAD_DIM).transpose(1, 2)
        k = attn.k_proj(hs).view(1, total, -1, HEAD_DIM).transpose(1, 2)
        v = attn.v_proj(hs).view(1, total, -1, HEAD_DIM).transpose(1, 2)
        q, k = attn.q_norm(q), attn.k_norm(k)
        cos, sin = dec.rotary_emb(
            hidden, rope_positions[None], cfg.layer_types[i]
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, attn.num_key_value_groups)
        v = repeat_kv(v, attn.num_key_value_groups)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=attn.scaling
        )
        out = attn.o_proj(out.transpose(1, 2).reshape(1, total, -1)[0])
        hidden = residual + layer.post_self_attn_layernorm(out)
        residual = hidden
        hidden = residual + layer.post_feedforward_layernorm(
            layer.mlp(layer.pre_feedforward_layernorm(hidden))
        )
        # Corrupt the prefix rows: decoder rows must not notice.
        hidden[is_prefix] = 12345.0

    hidden = dec.norm(hidden)
    got = tiny_model.lm_head(hidden[n:])
    softcap = cfg.final_logit_softcapping
    if softcap is not None:
        got = torch.tanh(got / softcap) * softcap

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_compute_prefix_rope_positions_single_request():
    is_prefix = torch.tensor([True] * 3 + [False] * 4)
    positions = torch.arange(7)
    got = compute_prefix_rope_positions(is_prefix, positions)
    assert got.tolist() == [3, 3, 3, 3, 4, 5, 6]


def test_compute_prefix_rope_positions_multi_request_batch():
    # Two prefill requests flattened back-to-back: N=2 (+3 dec) then N=4
    # (+1 dec), followed by two decode-only rows of other requests.
    is_prefix = torch.tensor(
        [True, True, False, False, False]
        + [True, True, True, True, False]
        + [False, False]
    )
    positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 17, 9])
    got = compute_prefix_rope_positions(is_prefix, positions)
    assert got.tolist() == [2, 2, 2, 3, 4, 4, 4, 4, 4, 4, 17, 9]


def test_compute_prefix_rope_positions_adjacent_runs_not_merged():
    # Request A is ALL prefix rows in this chunk (its decode comes later);
    # request B's prefix follows immediately.  The position reset separates
    # the runs.
    is_prefix = torch.tensor([True] * 3 + [True] * 2 + [False])
    positions = torch.tensor([0, 1, 2, 0, 1, 2])
    got = compute_prefix_rope_positions(is_prefix, positions)
    assert got.tolist() == [3, 3, 3, 2, 2, 2]


def test_compute_prefix_rope_positions_rejects_split_run():
    # A chunked-prefill split placeholder run does not start at position 0.
    is_prefix = torch.tensor([True, True, False])
    positions = torch.tensor([5, 6, 7])
    with pytest.raises(ValueError, match="chunked prefill"):
        compute_prefix_rope_positions(is_prefix, positions)


def test_compute_prefix_rope_positions_decode_only_noop():
    is_prefix = torch.zeros(4, dtype=torch.bool)
    positions = torch.tensor([7, 12, 3, 9])
    got = compute_prefix_rope_positions(is_prefix, positions)
    assert got.tolist() == positions.tolist()
