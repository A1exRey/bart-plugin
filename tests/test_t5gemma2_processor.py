# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the T5Gemma2 processor and glue code.

Needs vLLM importable (CPU install is fine) but no GPU and no engine;
processor objects are built with ``__new__`` + fakes, mirroring
test_vllm_018_compat.py.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("transformers", minversion="5.0")


class FakeTokenizer:
    """Deterministic fake: 1 token per whitespace-separated word + BOS."""

    bos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        tokens = [100 + i for i in range(len(text.split()))]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens
        return tokens

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        return {
            "input_ids": torch.tensor(
                [self.encode(text, add_special_tokens)]
            )
        }


class FakeInfo:
    def __init__(self, placeholder=262144):
        self._tokenizer = FakeTokenizer()
        self._placeholder = placeholder

    def get_tokenizer(self):
        return self._tokenizer

    def get_placeholder_token_id(self):
        return self._placeholder

    def tokenize_encoder_text(self, text):
        return self._tokenizer.encode(text, add_special_tokens=True)


def _make_processor():
    from vllm_bart_plugin.t5gemma2 import T5Gemma2MultiModalProcessor

    processor = T5Gemma2MultiModalProcessor.__new__(
        T5Gemma2MultiModalProcessor
    )
    processor.info = FakeInfo()
    return processor


def test_call_hf_processor_tokenizes_encoder_text_and_decoder_prompt():
    processor = _make_processor()
    out = processor._call_hf_processor(
        "hello there", {"texts": ["one two three"]}, {}, {}
    )
    # encoder text: BOS + 3 words
    assert out["encoder_input_ids"].shape == (1, 4)
    assert out["encoder_input_ids"][0, 0].item() == 2
    # decoder prompt: BOS + 2 words
    assert out["input_ids"].shape == (1, 3)
    assert out["input_ids"][0, 0].item() == 2


def test_call_hf_processor_accepts_pretokenized_decoder_prompt():
    processor = _make_processor()
    out = processor._call_hf_processor([7, 8, 9], {}, {}, {})
    assert torch.equal(out["input_ids"], torch.tensor([[7, 8, 9]]))
    assert "encoder_input_ids" not in out


def test_call_hf_processor_empty_decoder_prompt_gets_bos():
    processor = _make_processor()
    out = processor._call_hf_processor("", {"texts": ["a b"]}, {}, {})
    assert out["input_ids"].tolist() == [[2]]


def test_prompt_updates_insert_one_placeholder_per_encoder_token():
    from vllm.multimodal.processing import PromptInsertion

    from vllm_bart_plugin.bart import TextProcessorItems

    processor = _make_processor()

    class FakeItems:
        def get_count(self, modality, strict=True):
            assert modality == "text"
            return 1

        def get_items(self, modality, typ):
            return TextProcessorItems("one two three four")

    (update,) = processor._get_prompt_updates(FakeItems(), {}, {})
    assert isinstance(update, PromptInsertion)
    assert update.modality == "text"
    insertion = update.content(0)
    # BOS + 4 words = 5 encoder tokens -> 5 placeholders
    assert insertion == [262144] * 5


def test_prompt_updates_empty_without_text_items():
    processor = _make_processor()

    class FakeItems:
        def get_count(self, modality, strict=True):
            return 0

    assert processor._get_prompt_updates(FakeItems(), {}, {}) == []


def _make_model_shell():
    from vllm_bart_plugin.t5gemma2 import T5Gemma2ForConditionalGeneration

    model = T5Gemma2ForConditionalGeneration.__new__(
        T5Gemma2ForConditionalGeneration
    )
    model.placeholder_token_id = 262144
    model._pending_prefix_mask = None
    return model


def test_parse_encoder_input_tensor_and_list():
    model = _make_model_shell()
    # Batched tensor (num_items, 1, N)
    batched = torch.arange(8).reshape(2, 1, 4)
    items = model._parse_and_validate_encoder_input(
        encoder_input_ids=batched
    )
    assert len(items) == 2 and items[0].tolist() == [0, 1, 2, 3]
    # List of (1, N) tensors with different lengths
    items = model._parse_and_validate_encoder_input(
        encoder_input_ids=[torch.tensor([[1, 2]]), torch.tensor([[3]])]
    )
    assert [item.tolist() for item in items] == [[1, 2], [3]]
    # Absent
    assert model._parse_and_validate_encoder_input() == []


def test_resolve_prefix_ctx_from_stashed_mask():
    model = _make_model_shell()
    n, t = 3, 2
    positions = torch.arange(n + t)
    inputs_embeds = torch.randn(n + t, 8)
    mask = torch.tensor([True] * n + [False] * t)

    model._pending_prefix_mask = mask
    ctx = model._resolve_prefix_ctx(None, positions, inputs_embeds)
    assert ctx is not None
    assert ctx.rope_positions.tolist() == [3, 3, 3, 3, 4]
    assert torch.equal(ctx.enc_rows, inputs_embeds[:n])
    # Stash is consumed.
    assert model._pending_prefix_mask is None


def test_resolve_prefix_ctx_pads_mask_for_padded_batches():
    model = _make_model_shell()
    mask = torch.tensor([True, True, False])
    model._pending_prefix_mask = mask
    positions = torch.tensor([0, 1, 2, 0, 0])  # padded to 5 rows
    inputs_embeds = torch.randn(5, 8)
    ctx = model._resolve_prefix_ctx(None, positions, inputs_embeds)
    assert ctx.is_prefix.tolist() == [True, True, False, False, False]
    assert ctx.rope_positions.tolist() == [2, 2, 2, 0, 0]


def test_resolve_prefix_ctx_decode_only_returns_none():
    model = _make_model_shell()
    model._pending_prefix_mask = torch.zeros(4, dtype=torch.bool)
    ctx = model._resolve_prefix_ctx(
        None, torch.tensor([5, 9, 2, 7]), torch.randn(4, 8)
    )
    assert ctx is None


def test_resolve_prefix_ctx_fallback_from_input_ids():
    model = _make_model_shell()
    input_ids = torch.tensor([262144, 262144, 2, 17])
    positions = torch.arange(4)
    inputs_embeds = torch.randn(4, 8)
    ctx = model._resolve_prefix_ctx(input_ids, positions, inputs_embeds)
    assert ctx.is_prefix.tolist() == [True, True, False, False]
    assert ctx.rope_positions.tolist() == [2, 2, 2, 3]


def test_hf_overrides_helper_flips_flag():
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2Config,
    )

    from vllm_bart_plugin.t5gemma2 import t5gemma2_hf_overrides

    config = T5Gemma2Config()
    assert config.is_encoder_decoder is True
    config = t5gemma2_hf_overrides(config)
    assert config.is_encoder_decoder is False


def test_weight_name_mapping_covers_real_hf_checkpoint_keys():
    """Every key of a real (tiny) HF T5Gemma2 state dict must map to the
    vLLM module tree this plugin builds, with fused shards routed."""
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2Config,
        T5Gemma2DecoderConfig,
        T5Gemma2EncoderConfig,
        T5Gemma2TextConfig,
    )
    from transformers.models.t5gemma2.modeling_t5gemma2 import (
        T5Gemma2ForConditionalGeneration as HFModel,
    )

    from vllm_bart_plugin.t5gemma2 import map_t5gemma2_weight_name

    text_kwargs = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    vision_kwargs = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=14,
        patch_size=7,
    )
    config = T5Gemma2Config(
        encoder=T5Gemma2EncoderConfig(
            text_config=T5Gemma2TextConfig(**text_kwargs),
            vision_config=vision_kwargs,
            mm_tokens_per_image=4,
        ),
        decoder=T5Gemma2DecoderConfig(**text_kwargs),
    )
    hf_model = HFModel(config)

    # The exact parameter tree the vLLM model exposes (per layer).
    def expected_names(side):
        names = {
            f"model.{side}.embed_tokens.weight",
            f"model.{side}.embed_tokens.eoi_embedding",
            f"model.{side}.norm.weight",
        }
        for i in range(2):
            layer = f"model.{side}.layers.{i}"
            names |= {
                f"{layer}.self_attn.qkv_proj.weight",
                f"{layer}.self_attn.o_proj.weight",
                f"{layer}.self_attn.q_norm.weight",
                f"{layer}.self_attn.k_norm.weight",
                f"{layer}.mlp.gate_up_proj.weight",
                f"{layer}.mlp.down_proj.weight",
                f"{layer}.pre_self_attn_layernorm.weight",
                f"{layer}.post_self_attn_layernorm.weight",
                f"{layer}.pre_feedforward_layernorm.weight",
                f"{layer}.post_feedforward_layernorm.weight",
            }
        return names

    valid_targets = (
        expected_names("encoder.text_model")
        | expected_names("decoder")
        | {"lm_head.weight"}
    )

    seen_targets = set()
    for key in hf_model.state_dict():
        mapped = map_t5gemma2_weight_name(key)
        if mapped is None:
            assert key.startswith(
                ("model.encoder.vision_tower.",
                 "model.encoder.multi_modal_projector.")
            ), f"unexpectedly skipped {key}"
            continue
        target, shard_id = mapped
        assert target in valid_targets, (
            f"HF key {key!r} mapped to unknown vLLM param {target!r}"
        )
        if target.endswith(("qkv_proj.weight", "gate_up_proj.weight")):
            assert shard_id is not None
        else:
            assert shard_id is None
        seen_targets.add(target)

    # Everything except possibly-tied embeddings must be covered by the
    # checkpoint (tied weights are backfilled by load_weights).
    tied_ok = {
        "model.decoder.embed_tokens.weight",
        "model.decoder.embed_tokens.eoi_embedding",
        "lm_head.weight",
    }
    missing = valid_targets - seen_targets - tied_ok
    assert not missing, f"vLLM params never produced by mapping: {missing}"
