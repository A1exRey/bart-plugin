# Theoretical results: T5Gemma2 on FlashAttention

This note states and proves the correctness results behind the two T5Gemma2
patches, relates them to the open upstream problem (transformers issue
[#45522](https://github.com/huggingface/transformers/issues/45522):
*"T5Gemma2ForConditionalGeneration does not support Flash Attention 2 yet"*),
and derives the complexity / expected-speed-up model that
`scripts/bench_t5gemma2.py` measures empirically.

The two patches:

1. **Patch 1 — prefix-KV reduction** (PR #1, `vllm_bart_plugin/t5gemma2.py`):
   T5Gemma2's *merged* self+cross decoder attention is re-expressed as plain
   causal decoder-only attention over `[encoder prefix][decoder tokens]`.
   Exact for total lengths within the decoder sliding window `w`.
2. **Patch 2 — two-pass LSE merge** (PR #2, `vllm_bart_plugin/t5gemma2_long.py`):
   beyond `w`, the merged attention is decomposed into two FlashAttention
   calls over disjoint key sets, recombined exactly in log-sum-exp space.
   Exact at any length.

Everything below is proven mechanically in `tests/test_t5gemma2_math.py`
(CPU, vs the HF reference, ~1e-4 in float32) and gated on GPU by
`scripts/parity_t5gemma2.py`; this document gives the closed-form arguments.

## 1. Setting and notation

A T5Gemma2 decoder layer computes ONE attention per layer ("merged
attention"). For a decoder query at local position `t` (`0 <= t < T`), with
`N` encoder output rows `e_0..e_{N-1}`:

- **keys/values** are `concat(cross KV, self KV)`:
  - cross `k_j = knorm(k_proj(e_j))`, `v_j = v_proj(e_j)` — projected with the
    layer's own weights from the *raw* encoder output, **no rotary embedding**;
  - self `k_s, v_s` from decoder hidden states, keys rotated at position `s`;
- **queries** `q_t` are q-normed and rotated at the decoder-local position `t`;
- **mask**: all `N` cross keys are visible to every query (never windowed —
  confirmed upstream by transformers PR #45540); self keys are causal, and on
  `sliding_attention` layers additionally restricted to `t - s < w`;
- one softmax over the concatenated, masked scores.

HF's cross score convention is therefore `(R_t q) · k` with `k` un-rotated,
where `R_p` is the (block-diagonal) rotation by position `p`, satisfying the
relative property `R_a^T R_b = R_{b-a}`.

vLLM has no primitive for merged attention — the same architectural fact that
keeps #45522 open in transformers, where FlashAttention-2 is rejected at load
for this model. Both patches work by *reducing* merged attention to
primitives FlashAttention-class kernels already implement, instead of writing
a new kernel.

## 2. Patch 1: the prefix-KV reduction (exact for N + T <= w)

The plugin runs the model decoder-only on the sequence
`[P]*N ++ [BOS, decoder tokens]`: the `N` placeholder rows carry the encoder
output as embeddings, their K/V land in the ordinary paged KV cache at
prefill, and RoPE positions of the prefix rows are pinned to the constant `N`.

**Lemma 1 (RoPE pinning).** For any pin `P̂` and decoder query at absolute
position `P̂ + t`:

```
cross:  (R_{P̂+t} q) · (R_{P̂} k)   = (R_t q) · k            (HF's convention)
self:   (R_{P̂+t} q) · (R_{P̂+s} k) = (R_t q) · (R_s k)      (shift-invariance)
```

*Proof.* `(R_a x)·(R_b y) = x^T R_{b-a} y`, which depends only on `b - a`:
for the cross pair `b - a = -t`, identical to rotating only the query by `t`;
for the self pair `b - a = s - t`, identical to HF's un-shifted layout. ∎

So rotating prefix *keys* at the constant pin reproduces HF's "rotated q,
un-rotated cross k" scores exactly, even though vLLM applies RoPE uniformly.
The pin value is arbitrary; the plugin uses `N` (short mode) / `N_pad` (long
mode) so decoder rows keep their scheduler positions.

**Lemma 2 (substitution invariant).** In every decoder layer, the
post-layernorm hidden states of the prefix rows are replaced by the raw
encoder output *before* the fused qkv projection
(`substitute_prefix_rows`). Consequently, at every layer `ℓ` the prefix
rows' K/V equal HF's cross K/V for layer `ℓ` exactly, and the (meaningless)
residual stream of the prefix rows is unobservable: decoder rows interact
with prefix rows **only** through those pinned K/V, and prefix-row logits are
never sampled. ∎

**Theorem 1 (short-mode exactness).** If `max_model_len <= w` (so
`N + T <= w`), plain causal **full** attention over `[prefix][decoder]` with
Lemma-1 positions and Lemma-2 K/V computes, at every decoder row of every
layer, exactly HF's merged attention.

*Proof.* (i) *Visibility:* causal attention lets query `N + t` see all `N`
prefix rows (they precede it) — HF's "cross always visible" — and self rows
`s <= t`. On sliding layers HF's extra constraint `t - s < w` is implied by
`t < N + T <= w`, so causal-sliding = causal-full; on full layers the masks
coincide trivially. (ii) *Scores:* equal by Lemmas 1–2 (q/k-norms and
projections are the same modules). (iii) Same key set + same scores + same
values under one softmax ⇒ same output. ∎

The encoder side needs no theorem: the plugin applies HF's exact
bidirectional sliding-window masks, so it is exact at any length in both
modes. Corollary: greedy decoding is token-identical in exact arithmetic
(measured token-exact in float32 on GPU; ~1e-4 on CPU).

**Why this matters for #45522:** Theorem 1 turns merged attention into the
*stock* decoder-only attention op. The plugin instantiates vLLM's standard
`Attention` layer, so short mode is **backend-agnostic** — FlashAttention,
FlashInfer, Triton, whichever backend vLLM selects — with paged KV, CUDA
graphs, and continuous batching for free.

## 3. Patch 2: the two-pass LSE merge (exact at any length)

Beyond the window, HF's mask is genuinely hybrid: unwindowed over cross keys,
windowed-causal over self keys. No single vLLM attention call can express it
(a `SlidingWindowSpec` would physically free the prefix blocks — see
`docs/t5gemma2_long_context_design.md`). Patch 2 splits the key set instead.

**Lemma 3 (exact softmax merge over disjoint key sets).** For a query `q`
with visible key set `S = A ⊔ B` (disjoint union), let
`(o_A, l_A)` be the attention output and log-sum-exp of scores restricted to
`A`, likewise `(o_B, l_B)`. Then

```
o_S = ( e^{l_A} · o_A + e^{l_B} · o_B ) / ( e^{l_A} + e^{l_B} )
```

*Proof.* Both numerator and denominator of softmax-attention are sums over
keys; disjointness lets each sum split, and `e^{l_A}` restores the
un-normalized numerator `Σ_{j∈A} e^{s_j} v_j = e^{l_A} o_A`. An empty set
contributes `l = -inf`, i.e. weight `e^{-inf} = 0`. ∎

This is precisely what vLLM's cascade-attention kernel
(`merge_attn_states`) computes, and FlashAttention's `varlen` entry point
returns the needed `l` values.

**Proposition (padding neutrality).** The prefix run is padded from
`N_real` to `N_pad = round_up(N_real, 16)` so the self region starts on a
KV-block boundary. Rows `[N_real, N_pad)` belong to **neither** pass's key
set, so they contribute nothing; pinning RoPE at `N_pad` and starting decoder
tokens at absolute position `N_pad` keeps both identities of Lemma 1 intact
(the cross identity uses the pin, the self identity is shift-invariant). ∎

**Theorem 2 (long-mode exactness).** For any `N_real`, `T`, and layer type,
the two-pass computation

- **pass A**: all queries over KV rows `[0, N_real)`, non-causal, unwindowed
  → `(o_A, l_A)`;
- **pass B**: all queries over the block table shifted left by
  `N_pad/16` blocks (the self region re-addressed from block 0), causal,
  FlashAttention native window `[w-1, 0]` on sliding layers →
  `(o_B, l_B)`;
- merge via Lemma 3

equals HF's merged attention exactly.

*Proof.* Pass A's key set is exactly HF's cross set with exactly HF's scores
(Lemma 1 cross identity at pin `N_pad`, Lemma 2 K/V). Pass B's key set is
exactly HF's visible self set — causality and the `t - s < w` window are
enforced natively by the kernel — with exactly HF's scores (shift
invariance). The sets are disjoint and their union is HF's full visible set
(Proposition), so Lemma 3 gives one softmax over HF's merged mask. ∎

Mechanically verified: `tests/test_t5gemma2_math.py` checks the merge
identity against a single merged-mask softmax and checks beyond-window
teacher-forced *and* incremental decoding against the HF reference
(`w=8`, `N_real=5 → N_pad=16`).

**Theorem 3 (work optimality of the decomposition).** The total number of
scored (query, key) pairs across both passes equals exactly the number of
unmasked entries of HF's merged mask — the padding rows are excluded from
both passes and no pair is scored twice. Hence the two-pass scheme performs
the same attention FLOPs as a hypothetical native merged-attention kernel;
its only overhead is one extra kernel launch and one `merge_attn_states`
pass, i.e. `O(T · h · d)` extra memory traffic per layer — independent of
context length `N + T`. ∎

## 4. Relation to transformers #45522 (FlashAttention support)

- **Upstream status:** open. HF rejects
  `attn_implementation="flash_attention_2"` for
  `T5Gemma2ForConditionalGeneration`; the blocker named in the issue is the
  merged self+cross attention, for which no FA-compatible formulation existed.
  The companion issue #45521 documents eager attention failing above ~4K
  tokens, so in practice transformers has no long-context-capable attention
  path for this model's decoder.
- **What these patches establish:** merged attention *does* have an exact
  FA-compatible formulation. Theorem 1 reduces it to a stock causal kernel
  (within the window); Theorems 2–3 reduce it, at any length, to two
  standard FA calls plus an exact O(T) merge — using only capabilities
  FlashAttention already exposes (varlen batching, native right-aligned
  windows, LSE return). Nothing here is vLLM-specific mathematics: the same
  decomposition would resolve #45522 inside transformers, with the practical
  caveat that FA kernels are fp16/bf16-only (Gemma activations overflow
  fp16, so **bfloat16 is the only usable FA dtype** — measured, see
  `scripts/parity_t5gemma2.py`).

## 5. Is the method FlashAttention-only?

No — the *reduction* is kernel-agnostic; the two modes have different
requirements:

| | requirement | backends today |
|---|---|---|
| short mode (Patch 1) | any paged causal attention | **all** vLLM backends (FLASH_ATTN, FLASHINFER, TRITON_ATTN, …) — the plugin uses the stock `Attention` layer and never touches the backend |
| long mode (Patch 2) | paged attention that **returns per-query LSE**, non-causal + windowed-causal variants | **FLASH_ATTN only**, enforced with an actionable error |

The FLASH_ATTN restriction in long mode is an *implementation* boundary, not
a mathematical one: Lemma 3 needs the log-sum-exp of each pass, and vLLM
currently exposes an LSE return only through
`flash_attn_varlen_func(..., return_softmax_lse=True)`. FlashInfer computes
LSEs internally as well (its own cascade/merge machinery is built on the
same identity as Lemma 3), so a FLASHINFER port of pass A/B is feasible
future work; the merge step (`merge_attn_states`) is already
backend-neutral. Anything strictly analogous holds for other
tiled-softmax kernels — the method needs *a* FlashAttention-class kernel,
not FlashAttention specifically.

## 6. Complexity and expected speed-up

Let `L = N + T` (total merged length), `h` heads, `d` head dim, `w` the
sliding window. Per decoder layer:

**Memory.**

| path | attention memory |
|---|---|
| HF eager / mask-materializing SDPA | `O(h · L²)` — explicit merged mask (and, eager, the score matrix); this is why #45521-era eager fails at a few K tokens |
| plugin (both modes) | `O(L · h · d)` — tiled kernels never materialize scores; KV is paged |

The asymptotic memory drop from `L²` to `L` is the enabling result for the
128K-context claim in the issue: at `L = 128K`, `bf16` merged-mask
materialization alone is `~L²·heads` ≈ tens of TB, while paged KV is linear.

**Prefill attention work** (unmasked score entries, by Theorem 3):

| layer type | HF semantics | plugin work |
|---|---|---|
| sliding | cross `T·N` + self `Σ_t min(t, w)` | identical: pass A `T·N` + pass B `≈ T·w` |
| full | cross `T·N` + self `T²/2` | identical |

So on sliding layers the plugin does `Θ(T·(N + w))` attention work where an
unwindowed kernel would do `Θ(T·(N + T))` — the native FA window skips
out-of-window tiles entirely; the theoretical prefill speed-up factor on
sliding layers grows as `(N + T)/(N + w)` with context length. Versus HF
*eager* the additional gain is a bandwidth constant (no `L²` score/mask
round-trips through HBM).

**Decode step** (one query): `O(N + min(t, w))` per sliding layer,
`O(N + t)` per full layer — same asymptotics as a cached HF decode, so the
per-step win is constant-factor: fused kernels, no mask construction, plus
vLLM machinery (paged KV, continuous batching across requests, CUDA graphs —
full graphs in short mode, piecewise in long mode).

**Overhead of long mode over short mode** (Theorem 3): two launches + one
`O(T·h·d)` merge per layer, length-independent. Retention is the real cost:
sliding layers keep all `T` KV rows instead of `w`, so long-mode KV memory
matches a full-attention model — acceptable for the 270m checkpoint, and the
price of exactness given vLLM's block manager (see the design doc).

**What to expect in measurements** (`scripts/bench_t5gemma2.py`):

1. *Short contexts (≤ w):* modest per-request gains vs HF (the model is small
   and `L` short, so launch overhead and the encoder dominate); large gains
   in batched throughput from continuous batching.
2. *Beyond the window:* the gap widens with `N` — HF has only
   mask-materializing paths (its FA2 path is the very thing #45522 requests),
   the plugin stays linear-memory and window-optimal. Expect HF to hit
   OOM/latency cliffs where the plugin degrades linearly.
3. *bfloat16 only* for long mode (FA dtype constraint, §4).

The benchmark reports prefill latency, end-to-end latency, and generated
tokens/s for each stack (vLLM short/long, HF eager/SDPA) over configurable
encoder lengths and batch sizes.
