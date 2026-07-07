# Design: T5Gemma2 contexts beyond the decoder sliding window

Status: IMPLEMENTED (vllm_bart_plugin/t5gemma2_long.py), verified against
vLLM 0.24 internals and the HF reference on CPU; GPU parity via
`scripts/parity_t5gemma2.py --long-context [--image --num-images 2]`.
Enables: multi-image requests, long-document inputs (anything past
`config.decoder.sliding_window`, 512 for the 270m checkpoint).

## Problem

The plugin runs T5Gemma2 as decoder-only with the encoder output injected as
N prefix rows (positions `0..N-1`) of each request's paged KV sequence.  For
lengths within the window this is exact with plain causal attention.  Beyond
it, the HF semantics for *sliding-window decoder layers* are:

- a decoder query at merged position `N+t` attends to **all N prefix rows**
  (encoder keys are never windowed — confirmed upstream by transformers
  PR #45540, the fix for issue #45521), and
- decoder self rows within a causal window `w`.

Full-attention decoder layers attend everything causally.

## What does NOT work in vLLM 0.24 (verified in source)

- **`per_layer_sliding_window=w`** — produces a `SlidingWindowSpec`, whose
  manager (`v1/core/single_type_kv_cache_manager.py`, `SlidingWindowManager`)
  **physically frees out-of-window blocks** (`remove_skipped_blocks` →
  `block_pool.free_blocks`).  The prefix rows would be recycled, not masked.
  Also, the `Attention` layer accepts only a single int window — no
  asymmetric `(left, right)`, no per-request left edge.
- **`StaticSinkAttention`** — StreamingLLM-style, but the sink is a single
  global, block-aligned, constant-length KV shared by every request, under
  FULL (unwindowed) attention.  Cannot represent per-request encoder rows.
- **gpt-oss "sinks" (`s_aux`)** — a per-head learned scalar added to the
  softmax denominator.  Not a token; grants no visibility.
- **Built-in cascade attention** — asserts no sliding window and requires a
  batch-common prefix; ours is per-request.

## The implemented design: retention + exact two-pass LSE merge

All in `vllm_bart_plugin/t5gemma2_long.py`; enabled by passing
`hf_overrides=t5gemma2_long_context_hf_overrides` (exported from
`vllm_bart_plugin.t5gemma2` too).

1. **Retention.**  In long mode EVERY decoder layer uses
   `T5Gemma2PrefixAttention`, constructed without a sliding window, so the
   stock `Attention.get_kv_cache_spec` reports `FullAttentionSpec` — one KV
   cache group, `FullAttentionManager`, no block ever freed.  (Full layers
   must use the custom attention too: plain causal would attend the
   padding rows introduced below.)
2. **Block-aligned prefix padding.**  The processor pads the placeholder
   run from `N_real` to `N_pad = round_up(N_real, PREFIX_PAD=16)` using ONE
   mm item whose `PromptUpdateDetails.is_embed` marks only the first
   `N_real` rows.  The runner scatters encoder embeddings only into
   `is_embed` rows; the padding rows keep the placeholder-token embedding
   and are attended by NOTHING.  RoPE pins prefix rows to `N_pad`
   (`compute_prefix_rope_positions(..., pad_multiple=16)`): by the relative
   identity `(R_{N_pad+t} q)·(R_{N_pad} k) = (R_t q)·k` the scores match HF
   exactly, and decoder self-self distances are shift-invariant.
3. **Visibility, exactly.**  The custom impl (a `FlashAttentionImpl`
   subclass installed via `subclass_attention_backend_with_overrides`) runs
   two FlashAttention calls per layer and merges them in LSE space:
   - pass A: all queries over KV rows `[0, N_real)` of the stock block
     table, `causal=False`, unwindowed → `(out_p, lse_p)`;
   - pass B: all queries over the block table gathered left by
     `N_pad/16` blocks (the self region re-addressed from block 0),
     `seqused_k = seq_len − N_pad`, `causal=True`,
     `window_size=[w−1, 0]` on sliding layers / unwindowed on full layers
     → `(out_s, lse_s)`;
   - `merge_attn_states(out, out_p, lse_p, out_s, lse_s)` (the exact merge
     kernel vLLM ships for cascade attention; FA's LSE layout
     `(nheads, tokens)` is already what it wants).
   Key sets are disjoint for every query (rows `[N_real, N_pad)` appear in
   neither pass), so the merge equals ONE softmax over HF's merged mask —
   proven against the HF reference in tests/test_t5gemma2_math.py
   (beyond-window teacher-forced + incremental, w=8, N_real=5→N_pad=16).
   Right-aligned FA windows keep pass B correct under chunked prefill of
   the decoder tail; empty sides (prefix-row queries in pass B, no-mm rows
   in pass A) produce ±inf LSE, which both merge kernels neutralize.
4. **Per-request N at decode time** (the old open problem).  The overrides
   set `hf_config.is_mm_prefix_lm = True`; the v1 runner then rebuilds
   `mm_req_doc_ranges` from each RUNNING request's persistent
   `mm_features` on EVERY step (`gpu_model_runner.py:2318-2336`) and the
   builder subclass converts them to per-request `N_real` / `N_pad` /
   shifted block tables (`prefix_lens_from_doc_ranges`,
   `shifted_block_table`).  Two traps and their defusals:
   - `use_mm_prefix` normally forces FA4 in backend auto-selection; our
     layers pass an explicit `attn_backend=`, which bypasses
     `get_attn_backend` entirely (`attention.py:318-330`).
   - the runner SKIPS ranges longer than `hf_text_config.sliding_window`
     (`gpu_model_runner.py:2331-2334`), and `hf_text_config` resolves to
     `config.decoder` for T5Gemma2.  The overrides therefore stash the
     real window in `decoder.plugin_prefix_lm_sliding_window` and null
     `decoder.sliding_window` (the model reads the stash; vLLM applies no
     SWA handling of its own — which is exactly what retention wants).

## v1 constraints (all validated with actionable errors)

- FLASH_ATTN backend only (LSE return is exposed there); other backends
  keep working within the window without long mode.
- `block_size == 16` (`== PREFIX_PAD`; the pass-B block-table slice must be
  whole blocks).
- No speculative decoding, no quantized KV cache, prefix caching off
  (pre-existing), chunked mm input disabled (pre-existing; also force-set
  by `is_mm_prefix_lm`).
- **CUDA graphs:** the builder reports `AttentionCGSupport.NEVER`, so
  attention is excluded from full-graph capture (piecewise graphs still
  work; short mode is unaffected).  Reason: capture-time
  `CommonAttentionMetadata.mm_req_doc_ranges` is empty (the dummy batch has
  no requests), so captured prefix lengths would be baked as 0.  Follow-up
  recipe: override `build_for_cudagraph_capture` to fill the persistent
  buffers with realistic block-aligned lengths (precedent:
  `_get_encoder_seq_lens(for_cudagraph_capture=True)`,
  `gpu_model_runner.py:1873-1881`), then relax to the underlying builder's
  support level.
- KV memory: retention means sliding layers no longer save memory vs. full
  layers — acceptable for a 270m model.
- The encoder side needs nothing: it already applies HF's exact
  bidirectional sliding masks at any length.
- The HF reference for beyond-window comparisons needs transformers >= 5.7
  (PR #45540: cross-attention cache was truncated to a stale
  `sliding_window` class default for long encoder inputs); the `t5gemma2`
  extra pins this.

## Validation

- `tests/test_t5gemma2_math.py` (CPU, no vLLM): two-pass LSE-merge identity
  vs a single merged-mask softmax; beyond-window teacher-forced AND
  incremental HF parity with the padded formulation (w=8, N+T=24 and 27);
  unit tests for `pad_multiple` pinning, `prefix_lens_from_doc_ranges`,
  `shifted_block_table`.
- `tests/test_t5gemma2_processor.py`: padded `is_embed` insertion, budget
  check including padding, override stash/idempotency, long-mode RoPE pin.
- GPU (user-run): `scripts/parity_t5gemma2.py --long-context` in float32
  (token-exact greedy gate) and bfloat16; `--long-context --image
  --num-images 2` for the multi-image case.
