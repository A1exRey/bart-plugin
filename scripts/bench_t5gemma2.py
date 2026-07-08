# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU speed benchmark: vLLM T5Gemma2 plugin vs HuggingFace transformers.

Companion to docs/t5gemma2_flash_attention_theory.md (§6): measures the
speed-up the prefix-KV reduction + two-pass FlashAttention decomposition
deliver over transformers, which has no FlashAttention path for T5Gemma2
(issue #45522) and therefore only mask-materializing attention.

Run on a machine with a GPU and access to the (gated) checkpoint:

    huggingface-cli login
    # within the sliding window (short mode, any vLLM backend):
    python scripts/bench_t5gemma2.py
    # beyond the window (long mode, FLASH_ATTN backend, bfloat16 only):
    python scripts/bench_t5gemma2.py --long-context --enc-lens 768 1536 3072
    # pick stacks / batch sizes:
    python scripts/bench_t5gemma2.py --stacks vllm hf-eager --batch-sizes 1 8

For each (stack, encoder length, batch size) cell the benchmark reports:

- prefill  - latency of a single forward over the batch (max_tokens=1 for
             vLLM; a 1-token generate for HF), ms per request;
- e2e      - end-to-end greedy generation latency for --max-new-tokens;
- decode   - generated tokens/s across the batch, with the prefill time
             subtracted (isolates the per-step decode cost).

Methodology notes:

- greedy (temperature 0), fixed --max-new-tokens with min_tokens/
  min_new_tokens forced, so every stack generates the same token count and
  tokens/s are comparable even where outputs differ numerically;
- one warmup iteration per cell (kernel autotuning, CUDA graph capture,
  vLLM profiling run), then --iters timed iterations, median reported;
- HF runs use torch.cuda.synchronize around timers; vLLM's generate() is
  synchronous already;
- encoder inputs are synthesized to an exact token length by repeating
  sentences and truncating at the token level, so lengths are identical
  across stacks;
- HF eager materializes O(L^2) masks/scores (see the theory doc): expect
  latency cliffs or OOM at long lengths -- cells that raise OOM are
  reported as such, not fatal.
"""

import argparse
import gc
import statistics
import sys
import time

SEED_SENTENCES = [
    "The tower is 324 metres tall, about the same height as an 81-storey "
    "building, and the tallest structure in Paris.",
    "Photosynthesis is the process by which green plants use sunlight to "
    "synthesize food from carbon dioxide and water.",
    "In 1969, Apollo 11 landed the first humans on the Moon.",
    "The Great Barrier Reef is the world's largest coral reef system.",
]


def make_source(tokenizer, target_tokens: int, variant: int) -> str:
    """Text whose encoder tokenization is exactly ``target_tokens`` long."""
    parts = []
    i = variant
    while len(tokenizer(" ".join(parts)).input_ids) < target_tokens:
        parts.append(SEED_SENTENCES[i % len(SEED_SENTENCES)])
        i += 1
    ids = tokenizer(" ".join(parts)).input_ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


class VllmStack:
    name = "vllm"

    def __init__(self, args, max_model_len: int):
        from vllm import LLM

        from vllm_bart_plugin.t5gemma2 import (
            t5gemma2_hf_overrides,
            t5gemma2_long_context_hf_overrides,
        )

        kwargs = {"block_size": 16} if args.long_context else {}
        self.llm = LLM(
            model=args.model,
            hf_overrides=(
                t5gemma2_long_context_hf_overrides
                if args.long_context
                else t5gemma2_hf_overrides
            ),
            max_model_len=max_model_len,
            enable_prefix_caching=False,
            disable_chunked_mm_input=True,
            dtype=args.dtype,
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
            **kwargs,
        )
        self.name = "vllm-long" if args.long_context else "vllm"

    def _prompts(self, sources):
        return [
            {"prompt": "", "multi_modal_data": {"text": source}}
            for source in sources
        ]

    def prefill(self, sources):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=1)
        start = time.perf_counter()
        self.llm.generate(self._prompts(sources), params, use_tqdm=False)
        return time.perf_counter() - start

    def generate(self, sources, max_new_tokens):
        from vllm import SamplingParams

        # min_tokens pins the generation length so tokens/s are comparable
        # across stacks even where greedy outputs differ numerically.
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            min_tokens=max_new_tokens,
            ignore_eos=True,
        )
        start = time.perf_counter()
        outs = self.llm.generate(self._prompts(sources), params,
                                 use_tqdm=False)
        elapsed = time.perf_counter() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        return elapsed, tokens


class HfStack:
    def __init__(self, args, attn_implementation: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = f"hf-{attn_implementation}"
        self.torch = torch
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model, attn_implementation=attn_implementation
        )
        self.model = self.model.to(
            device=args.device, dtype=getattr(torch, args.dtype)
        )
        self.model.eval()

    def _inputs(self, sources):
        return self.tokenizer(
            sources, return_tensors="pt", padding=True
        ).to(self.device)

    def _generate(self, sources, max_new_tokens):
        inputs = self._inputs(sources)
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
            )
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        # generate() prepends the decoder BOS; everything after it was
        # produced during this call.
        tokens = out.shape[0] * (out.shape[1] - 1)
        return elapsed, tokens

    def prefill(self, sources):
        return self._generate(sources, 1)[0]

    def generate(self, sources, max_new_tokens):
        return self._generate(sources, max_new_tokens)


def bench_cell(stack, sources, args):
    """One (stack, enc_len, batch) cell -> dict of metrics (or the error)."""
    try:
        stack.prefill(sources)  # warmup
        prefill = [stack.prefill(sources) for _ in range(args.iters)]
        stack.generate(sources, args.max_new_tokens)  # warmup
        e2e, tokens = [], 0
        for _ in range(args.iters):
            elapsed, tokens = stack.generate(sources, args.max_new_tokens)
            e2e.append(elapsed)
    except Exception as exc:  # OOM etc.: report the cell, keep the sweep
        return {"error": f"{type(exc).__name__}: {exc}"[:120]}
    prefill_ms = median_ms(prefill)
    e2e_ms = median_ms(e2e)
    decode_s = max(statistics.median(e2e) - statistics.median(prefill), 1e-9)
    return {
        "prefill_ms_per_req": prefill_ms / len(sources),
        "e2e_ms_per_req": e2e_ms / len(sources),
        "decode_tok_per_s": tokens / decode_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5gemma-2-270m-270m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float32"],
                        help="bfloat16 is the production dtype and the only "
                             "one long mode supports (FlashAttention is "
                             "fp16/bf16-only; fp16 overflows Gemma "
                             "activations)")
    parser.add_argument("--stacks", nargs="+",
                        default=["vllm", "hf-eager", "hf-sdpa"],
                        choices=["vllm", "hf-eager", "hf-sdpa"],
                        help="stacks to benchmark; 'vllm' means long mode "
                             "when --long-context is set")
    parser.add_argument("--enc-lens", nargs="+", type=int, default=None,
                        help="encoder lengths in tokens; defaults to "
                             "[w/4, w/2, w-64] short / [1.5w, 3w] long")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 4, 16])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--long-context", action="store_true",
                        help="beyond-sliding-window mode (two-pass "
                             "FlashAttention; requires FLASH_ATTN backend "
                             "and bfloat16)")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="disable CUDA graphs in vLLM (graphs are ON by "
                             "default here: this is a speed benchmark)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                        help="leave headroom for the HF stacks, which are "
                             "loaded alongside vLLM")
    args = parser.parse_args()

    if args.long_context and args.dtype == "float32":
        parser.error("--long-context requires bfloat16 (FlashAttention "
                     "kernels are fp16/bf16-only)")

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model)
    window = getattr(config.decoder, "sliding_window", None) or 4096
    if args.enc_lens is None:
        args.enc_lens = (
            [int(window * 1.5), 3 * window]
            if args.long_context
            else [window // 4, window // 2, window - 64]
        )
    budget = max(args.enc_lens) + args.max_new_tokens + 8
    if not args.long_context and budget > window:
        parser.error(
            f"enc_len {max(args.enc_lens)} + {args.max_new_tokens} new "
            f"tokens exceeds the sliding window ({window}); add "
            "--long-context")
    max_model_len = args.max_model_len or budget

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"model={args.model} dtype={args.dtype} window={window} "
          f"max_model_len={max_model_len} long_context={args.long_context}")
    print(f"enc_lens={args.enc_lens} batch_sizes={args.batch_sizes} "
          f"max_new_tokens={args.max_new_tokens} iters={args.iters} "
          f"(median reported)")

    # vLLM first (it profiles GPU memory at startup), then HF stacks.
    stacks = []
    for name in args.stacks:
        if name == "vllm":
            stacks.append(VllmStack(args, max_model_len))
        else:
            stacks.append(HfStack(args, name.removeprefix("hf-")))
        gc.collect()

    header = (f"{'stack':<10} {'enc':>6} {'batch':>5} "
              f"{'prefill ms/req':>15} {'e2e ms/req':>12} "
              f"{'decode tok/s':>13}")
    print("\n" + header)
    print("-" * len(header))
    rows = []
    for enc_len in args.enc_lens:
        for batch in args.batch_sizes:
            sources = [
                make_source(tokenizer, enc_len, variant)
                for variant in range(batch)
            ]
            for stack in stacks:
                cell = bench_cell(stack, sources, args)
                rows.append((stack.name, enc_len, batch, cell))
                if "error" in cell:
                    print(f"{stack.name:<10} {enc_len:>6} {batch:>5} "
                          f"   FAILED: {cell['error']}")
                else:
                    print(f"{stack.name:<10} {enc_len:>6} {batch:>5} "
                          f"{cell['prefill_ms_per_req']:>15.1f} "
                          f"{cell['e2e_ms_per_req']:>12.1f} "
                          f"{cell['decode_tok_per_s']:>13.1f}")

    # Speed-up summary: vLLM vs the fastest HF stack per cell.
    vllm_rows = {(e, b): c for n, e, b, c in rows if n.startswith("vllm")}
    hf_rows: dict[tuple[int, int], dict] = {}
    for n, e, b, c in rows:
        if n.startswith("hf-") and "error" not in c:
            best = hf_rows.get((e, b))
            if best is None or c["e2e_ms_per_req"] < best["e2e_ms_per_req"]:
                hf_rows[(e, b)] = c
    if vllm_rows and hf_rows:
        print("\nspeed-up (vLLM vs best HF stack per cell):")
        for key in sorted(set(vllm_rows) & set(hf_rows)):
            v, h = vllm_rows[key], hf_rows[key]
            if "error" in v:
                continue
            enc_len, batch = key
            print(f"  enc={enc_len:>6} batch={batch:>3}: "
                  f"e2e x{h['e2e_ms_per_req'] / v['e2e_ms_per_req']:.2f}, "
                  f"prefill x{h['prefill_ms_per_req'] / v['prefill_ms_per_req']:.2f}, "
                  f"decode x{v['decode_tok_per_s'] / h['decode_tok_per_s']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
