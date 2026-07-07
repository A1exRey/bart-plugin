# Design: T5Gemma2 contexts beyond the decoder sliding window

Status: researched against vLLM 0.24 internals, not yet implemented.
Prerequisite for: multi-image requests, long-document inputs
(anything past `config.decoder.sliding_window`, 512 for the 270m checkpoint).

## Problem

The plugin runs T5Gemma2 as decoder-only with the encoder output injected as
N prefix rows (positions `0..N-1`) of each request's paged KV sequence.  For
lengths within the window this is exact with plain causal attention.  Beyond
it, the HF semantics for *sliding-window decoder layers* are:

- a decoder query at merged position `N+t` attends to **all N prefix rows**
  (encoder keys are never windowed), and
- decoder self rows within a causal window `w`.

Full-attention decoder layers already work at any length.

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

## The viable design: retention + exact two-pass LSE merge

1. **Retention.**  Sliding-window decoder layers report
   `FullAttentionSpec` (optionally with `sliding_window=w` for bookkeeping)
   instead of `SlidingWindowSpec`, so `FullAttentionManager` never frees
   blocks and the prefix KV survives for the request lifetime.  Retention
   alone does not restore visibility — the plain causal window mask would
   still hide the prefix once the query is more than `w` past it.
2. **Visibility, exactly.**  Implement the layer forward as two
   FlashAttention calls over the SAME retained block table, merged in
   LSE space:
   - pass A: rows `0..N-1`, `causal=False`, `window_size=[-1,-1]`
     → `(out_p, lse_p)`;
   - pass B: the decoder-self region, `causal=True`,
     `window_size=[w-1, 0]` → `(out_s, lse_s)`;
   - `merge_attn_states(out, out_p, lse_p, out_s, lse_s)`
     (`vllm/v1/attention/ops/merge_attn_states.py` — the exact merge kernel
     vLLM ships for cascade attention).
   `flash_attn_varlen_func(..., return_softmax_lse=True)` provides the LSEs
   on FA2/FA3/FA4.  This reproduces "all prefix visible + causal window
   over self" with no custom CUDA and no approximation; N varies freely per
   request (it lives in the pass metadata, not in any spec constant).

Implementation surface (mirroring how `StaticSinkAttention` /
`PrefillPrefixLMAttention` extend vLLM): a custom `Attention` subclass that
wraps the FlashAttention backend via `subclass_attention_backend`, overrides
`get_kv_cache_spec`, and builds two-pass metadata (per-request `seqused_k` /
block-table slices for the prefix and self regions).

## Open problems

- **Per-request N at decode time.**  During prefill the model derives N from
  placeholder positions; the backend's metadata builder must know N for
  every RUNNING request at every decode step.  The builder sees block tables
  and sequence lengths, not model state — N needs a persistent side channel
  (e.g. recovered from the placeholder run in the request's prompt token
  ids, which the builder can access via the runner's request metadata).
- **Backend scope.**  FlashAttention only at first (LSE return is exposed
  there); other backends would fall back to the capped-length mode.
- **Chunked prefill of the self pass** interacts with the window boundary;
  simplest v1 keeps `disable_chunked_mm_input` and caps
  `max_num_batched_tokens` conservatively.
- **KV memory**: retention means sliding layers no longer save memory
  vs. full layers — acceptable for a 270m model; document it.
- The encoder side needs nothing: it already applies HF's exact
  bidirectional sliding masks at any length.

## Validation plan

- Extend `tests/test_t5gemma2_math.py` with a beyond-window case: HF
  reference (merged mask with sliding self half) vs. the two-pass LSE-merge
  formulation, on a tiny config with, say, w=8 and N+T=24.
- GPU: `scripts/parity_t5gemma2.py` with `--max-model-len` above the window
  and long sources; multi-image case once the cap is lifted.
