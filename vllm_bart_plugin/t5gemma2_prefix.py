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
    "PREFIX_PAD",
    "PrefixContext",
    "compute_prefix_rope_positions",
    "prefix_lens_from_doc_ranges",
    "shifted_block_table",
    "substitute_prefix_rows",
]

# Long-context mode pads every placeholder run to a multiple of this, so the
# decoder-self region always starts on a KV-block boundary and the two-pass
# attention can slice the block table by whole blocks.  Must equal the KV
# cache block size (FlashAttention requires block sizes that are multiples of
# 16, and the plugin validates ``cache_config.block_size == PREFIX_PAD``).
PREFIX_PAD = 16


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


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
    pad_multiple: int = 1,
) -> torch.Tensor:
    """Compute rotary positions with each prefix run pinned to its length N.

    ``positions`` are vLLM's absolute per-request token positions for a
    flattened (possibly multi-request) batch.  A "prefix run" is a maximal
    stretch of consecutive rows where ``is_prefix`` is True and ``positions``
    increase by exactly 1; because placeholder rows are always the first N
    tokens of their request (positions ``0..N-1``), the run's N is
    ``positions[last] + 1``.  Runs from different requests are separated by a
    position discontinuity, so they are never merged even when adjacent.

    ``pad_multiple``: in long-context mode the placeholder run is padded to
    ``N_pad = round_up(N_real, PREFIX_PAD)`` and the first decoder token sits
    at absolute position ``N_pad``, so the pin must be ``N_pad`` for the RoPE
    relative identity to hold.  The mask may cover either only the N_real
    embedded rows (the runner's ``is_multimodal`` mask) or all N_pad
    placeholder rows (the ``input_ids == placeholder`` fallback); rounding the
    run length up to ``pad_multiple`` yields the same pin either way.  The
    default of 1 keeps short mode byte-identical.

    Returns a new tensor equal to ``positions`` except that every row of a
    prefix run is set to that run's (rounded-up) N.
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
        n_prefix = round_up(int(positions[run[-1]]) + 1, pad_multiple)
        rope_positions[run] = n_prefix
    return rope_positions


def prefix_lens_from_doc_ranges(
    ranges: "dict[int, list[tuple[int, int]]] | None",
    num_reqs: int,
) -> list[int]:
    """Per-request encoder-prefix lengths (N_real) from mm placeholder ranges.

    ``ranges`` is ``CommonAttentionMetadata.mm_req_doc_ranges``: for each
    request index, the inclusive ``(start, end)`` ranges of embedded
    multimodal rows, produced by the runner from
    ``mm_position.extract_embeds_range()`` when ``is_mm_prefix_lm`` is set.
    The plugin emits exactly one mm item whose embedded rows are the leading
    encoder rows ``0..N_real-1``, so a request's N_real is ``end + 1`` of its
    single range.  Requests without ranges (dummy/profiling rows, or padding
    slots past the real batch) get 0.
    """
    lens = [0] * num_reqs
    if not ranges:
        return lens
    for req_idx, req_ranges in ranges.items():
        if req_idx >= num_reqs or not req_ranges:
            continue
        if len(req_ranges) != 1 or req_ranges[0][0] != 0:
            raise ValueError(
                "T5Gemma2 expects a single leading encoder placeholder run "
                f"per request; got embed ranges {req_ranges} for request "
                f"index {req_idx}."
            )
        lens[req_idx] = int(req_ranges[0][1]) + 1
    return lens


def shifted_block_table(
    block_table: torch.Tensor,
    shift_blocks: torch.Tensor,
) -> torch.Tensor:
    """Shift each request's block-table row left by ``shift_blocks[i]`` blocks.

    Used by the two-pass attention to make the decoder-self region (which
    starts at KV row ``N_pad``, a whole number of blocks) addressable as a
    key sequence starting at block 0.  Positions shifted in from beyond the
    row's end are clamped to the last column; they are never dereferenced
    because the accompanying ``seqused_k`` excludes them.

    ``block_table``: ``(num_reqs, max_blocks)`` int tensor.
    ``shift_blocks``: ``(num_reqs,)`` int tensor of whole-block shifts.
    """
    num_reqs, max_blocks = block_table.shape
    cols = torch.arange(max_blocks, device=block_table.device)
    idx = cols.unsqueeze(0) + shift_blocks.to(block_table.device).unsqueeze(1)
    idx = idx.clamp_(max=max_blocks - 1)
    return torch.gather(block_table, 1, idx)


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
