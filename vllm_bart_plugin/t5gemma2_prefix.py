# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-KV helpers for the T5Gemma2 vLLM plugin.

T5Gemma2's decoder uses *merged* self+cross attention: a single softmax over
``concat(self KV, cross KV)`` where the cross K/V are projected from the
(constant) encoder output with the layer's own k/v projections, get k-norm but
NO rotary embedding, and are always fully visible to every decoder query.

vLLM has no attention primitive for that, so the plugin runs the decoder as a
*decoder-only* sequence ``[P]*N ++ [decoder tokens]``: the N placeholder rows
carry the encoder output as their embeddings and their K/V are written into
the ordinary paged KV cache during prefill.  Standard causal attention over
``[prefix][decoder]`` then equals HF's merged mask, and rotating the prefix
keys at the constant position N makes the scores exactly match HF's
"rotated q at t, un-rotated cross k" convention thanks to RoPE's relative
property::

    (R_{N+t} q) . (R_N k) = q^T R_{-t} k = (R_t q) . k

This module contains the small, vLLM-free pieces of that scheme so they can
be unit-tested on CPU without vLLM (or a GPU) installed.
"""

from dataclasses import dataclass

import torch

__all__ = [
    "PrefixContext",
    "compute_prefix_rope_positions",
    "substitute_prefix_rows",
]


@dataclass
class PrefixContext:
    """Per-forward context describing encoder-prefix rows in a flattened batch.

    Attributes:
        is_prefix: Bool tensor of shape ``(num_tokens,)``; True for rows that
            are encoder-output placeholder rows.
        enc_rows: Tensor of shape ``(num_prefix_rows, hidden)`` holding the raw
            encoder outputs for the True rows of ``is_prefix`` (in order).
        rope_positions: Long tensor of shape ``(num_tokens,)``: the positions
            to use for rotary embedding.  Equal to the scheduler positions for
            decoder rows and to N (the request's encoder length) for prefix
            rows.
    """

    is_prefix: torch.Tensor
    enc_rows: torch.Tensor
    rope_positions: torch.Tensor


def compute_prefix_rope_positions(
    is_prefix: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Compute rotary positions with each prefix run pinned to its length N.

    ``positions`` are vLLM's absolute per-request token positions for a
    flattened (possibly multi-request) batch.  A "prefix run" is a maximal
    stretch of consecutive rows where ``is_prefix`` is True and ``positions``
    increase by exactly 1; because placeholder rows are always the first N
    tokens of their request (positions ``0..N-1``), the run's N is
    ``positions[last] + 1``.  Runs from different requests are separated by a
    position discontinuity, so they are never merged even when adjacent.

    Returns a new tensor equal to ``positions`` except that every row of a
    prefix run is set to that run's N.
    """
    if is_prefix.dtype != torch.bool:
        raise TypeError("is_prefix must be a bool tensor")
    if is_prefix.shape != positions.shape or is_prefix.dim() != 1:
        raise ValueError(
            "is_prefix and positions must be 1-D tensors of the same length, "
            f"got {tuple(is_prefix.shape)} and {tuple(positions.shape)}"
        )

    rope_positions = positions.clone()
    idx = is_prefix.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return rope_positions

    # A new run starts where the row index or the position value is not the
    # predecessor's + 1.
    row_break = idx[1:] != idx[:-1] + 1
    pos_break = positions[idx][1:] != positions[idx][:-1] + 1
    breaks = (row_break | pos_break).nonzero(as_tuple=False).flatten() + 1
    starts = torch.cat([idx.new_zeros(1), breaks])
    ends = torch.cat([breaks, idx.new_tensor([idx.numel()])])

    for s, e in zip(starts.tolist(), ends.tolist()):
        run = idx[s:e]
        first_pos = int(positions[run[0]])
        if first_pos != 0:
            raise ValueError(
                "Encoder placeholder run does not start at position 0 "
                f"(got {first_pos}).  This usually means chunked prefill "
                "split a multimodal item across chunks; run with "
                "disable_chunked_mm_input=True or enable_chunked_prefill="
                "False."
            )
        n_prefix = int(positions[run[-1]]) + 1
        rope_positions[run] = n_prefix
    return rope_positions


def substitute_prefix_rows(
    hidden_states: torch.Tensor,
    is_prefix: torch.Tensor,
    enc_rows: torch.Tensor,
) -> torch.Tensor:
    """Return ``hidden_states`` with prefix rows replaced by encoder outputs.

    Called on the *post-layernorm* hidden states right before each decoder
    layer's fused qkv projection: HF projects the cross K/V from the raw
    encoder output (no decoder-side layernorm), so the substituted rows yield
    exactly HF's ``k_proj(enc)`` / ``v_proj(enc)``.  A copy is returned; the
    input is left untouched (it is the residual stream).
    """
    out = hidden_states.clone()
    out[is_prefix] = enc_rows.to(out.dtype)
    return out
