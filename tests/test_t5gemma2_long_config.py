# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the long-context config helpers (no vLLM needed)."""

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers", minversion="5.0")


def test_long_context_hf_overrides_tolerates_dummy_probe_config():
    """vLLM probes the hf_overrides callable with a dummy bare
    PreTrainedConfig (no .decoder) just to read model_type, BEFORE loading
    the real config; the callable must pass it through unchanged instead
    of raising AttributeError."""
    from transformers import PretrainedConfig

    from vllm_bart_plugin.t5gemma2_long import (
        t5gemma2_long_context_hf_overrides,
    )

    dummy = PretrainedConfig()
    out = t5gemma2_long_context_hf_overrides(dummy)
    assert out is dummy
    assert out.is_encoder_decoder is False


def test_long_context_hf_overrides_stash_and_idempotency():
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2Config,
    )

    from vllm_bart_plugin.t5gemma2_long import (
        SLIDING_WINDOW_STASH_ATTR,
        is_long_context_mode,
        long_context_sliding_window,
        t5gemma2_long_context_hf_overrides,
    )

    config = T5Gemma2Config()
    original_window = config.decoder.sliding_window
    assert not is_long_context_mode(config)
    assert long_context_sliding_window(config) == original_window

    config = t5gemma2_long_context_hf_overrides(config)
    assert config.is_encoder_decoder is False
    assert config.is_mm_prefix_lm is True
    assert config.decoder.sliding_window is None
    assert is_long_context_mode(config)
    assert long_context_sliding_window(config) == original_window
    assert (
        getattr(config.decoder, SLIDING_WINDOW_STASH_ATTR) == original_window
    )

    # Applying twice must not stash the nulled value over the real one.
    config = t5gemma2_long_context_hf_overrides(config)
    assert long_context_sliding_window(config) == original_window
