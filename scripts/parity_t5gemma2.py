# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU parity harness: vLLM T5Gemma2 plugin vs HuggingFace transformers.

Run on a machine with a GPU and access to the (gated) checkpoint:

    huggingface-cli login
    # decisive exactness gate (implementation correctness):
    python scripts/parity_t5gemma2.py --dtype float32
    # production-dtype view (kernel-level numeric drift is expected):
    python scripts/parity_t5gemma2.py

Checks, in order of increasing integration depth:

1. prefill  - teacher-forced next-token logprobs on the decoder prompt
              (vLLM prompt_logprobs vs HF forward).  Catches wiring bugs in
              the prefix-KV scheme, RoPE offsets, and tokenization drift.
2. greedy   - 64-token greedy continuations vs HF generate().  Gates
              (token-exact) only in float32: in bfloat16, HF-eager and
              FlashAttention logits differ by ~0.1 logprob, so greedy
              near-ties legitimately flip; the check reports divergence
              steps as information.
3. batching - teacher-forced logprobs computed solo vs inside a mixed
              batch (catches slot/mask misalignment under continuous
              batching, which produces HUGE diffs; small drift from
              batch-size-dependent kernel reductions is tolerated).

Why two dtypes: float32 makes vLLM and HF numerically comparable, so any
diff above ~1e-2 is a real bug.  bfloat16 diffs up to a few tenths of a
logprob are cross-implementation kernel noise, not plugin bugs — the
float32 run is what distinguishes the two.

If (1) fails in float32, nothing else can work - fix that first.
"""

import argparse
import sys

SOURCES = [
    "The tower is 324 metres (1,063 ft) tall, about the same height as an "
    "81-storey building, and the tallest structure in Paris.",
    "Photosynthesis is the process by which green plants use sunlight to "
    "synthesize food from carbon dioxide and water.",
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, Apollo 11 landed the first humans on the Moon. Neil "
    "Armstrong became the first person to step onto the lunar surface.",
    "Machine learning is a field of study in artificial intelligence "
    "concerned with the development of statistical algorithms that can "
    "learn from data and generalize to unseen data.",
    "Der schnelle braune Fuchs springt ueber den faulen Hund.",
    "Water boils at 100 degrees Celsius at sea level atmospheric pressure.",
    "The Great Barrier Reef is the world's largest coral reef system, "
    "composed of over 2,900 individual reefs.",
]

DECODER_PREFIXES = ["", "The", "Summary:"]


def load_hf(model_name: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, dtype=getattr(torch, dtype)
    ).to(device)
    model.eval()
    return tokenizer, model


def derive_max_model_len(model_name: str) -> int:
    """The plugin is exact up to the decoder's sliding window."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    window = getattr(config.decoder, "sliding_window", None)
    return min(window or 4096, 4096)


def load_vllm(
    model_name: str, enforce_eager: bool, max_model_len: int, dtype: str
):
    from vllm import LLM

    from vllm_bart_plugin.t5gemma2 import t5gemma2_hf_overrides

    return LLM(
        model=model_name,
        hf_overrides=t5gemma2_hf_overrides,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
        disable_chunked_mm_input=True,
        dtype=dtype,
        enforce_eager=enforce_eager,
    )


def hf_teacher_forced_logprobs(tokenizer, model, source, decoder_prefix):
    """Logprob of each decoder-prompt token given the previous ones."""
    import torch

    device = model.device
    enc_ids = tokenizer(source, return_tensors="pt").input_ids.to(device)
    dec_ids = tokenizer(
        decoder_prefix if decoder_prefix else "",
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids=enc_ids, decoder_input_ids=dec_ids).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)[0]
    # logprob of token t given tokens < t (skip BOS at t=0)
    out = []
    for t in range(1, dec_ids.shape[1]):
        out.append(float(logprobs[t - 1, dec_ids[0, t]]))
    return dec_ids[0].tolist(), out


def check_prefill(tokenizer, hf_model, llm, args) -> bool:
    from vllm import SamplingParams

    print("\n=== 1. prefill parity (teacher-forced logprobs) ===")
    worst = 0.0
    ok = True
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    for source in SOURCES[: args.num_cases]:
        for prefix in DECODER_PREFIXES:
            if not prefix:
                continue  # need >=1 non-BOS decoder token to compare
            dec_ids, hf_lp = hf_teacher_forced_logprobs(
                tokenizer, hf_model, source, prefix
            )
            out = llm.generate(
                [{
                    "prompt": prefix,
                    "multi_modal_data": {"text": source},
                }],
                params,
            )[0]
            # prompt_logprobs covers [P]*N + decoder tokens; the last
            # len(dec_ids) entries are the decoder rows.
            plp = out.prompt_logprobs
            n_dec = len(dec_ids)
            dec_rows = plp[-(n_dec - 1):]  # rows for dec_ids[1:]
            for (token_id, hf_val), row in zip(
                zip(dec_ids[1:], hf_lp), dec_rows
            ):
                if row is None or token_id not in row:
                    print(f"  MISSING logprob for token {token_id}")
                    ok = False
                    continue
                diff = abs(row[token_id].logprob - hf_val)
                worst = max(worst, diff)
                if diff > args.prefill_tol:
                    ok = False
                    print(
                        f"  DIFF {diff:.4f} on token {token_id} "
                        f"(src={source[:40]!r}, prefix={prefix!r})"
                    )
    print(f"  worst |logprob diff| = {worst:.5f} "
          f"({'OK' if ok else 'FAIL'}, tol {args.prefill_tol})")
    return ok


def check_greedy(tokenizer, hf_model, llm, args) -> bool:
    import torch
    from vllm import SamplingParams

    gating = args.dtype == "float32"
    print(f"\n=== 2. greedy decode parity "
          f"({'gating' if gating else 'informational in ' + args.dtype}) ===")
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    ok = True
    for source in SOURCES[: args.num_cases]:
        enc_ids = tokenizer(source, return_tensors="pt").input_ids.to(
            hf_model.device
        )
        with torch.no_grad():
            hf_out = hf_model.generate(
                input_ids=enc_ids,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_tokens,
            )[0].tolist()
        # HF output starts with the decoder BOS; drop it.
        hf_tokens = hf_out[1:]

        out = llm.generate(
            [{"prompt": "", "multi_modal_data": {"text": source}}], params
        )[0]
        vllm_tokens = list(out.outputs[0].token_ids)

        n = min(len(hf_tokens), len(vllm_tokens))
        first = next(
            (i for i in range(n) if hf_tokens[i] != vllm_tokens[i]), None
        )
        if first is None:
            print(f"  exact ({n} tokens): {source[:40]!r}")
        else:
            # In low precision a near-tie can flip; only float32 gates.
            if gating:
                ok = False
            print(f"  diverges at step {first}/{n} (src={source[:40]!r})")
            print(f"    hf  : ...{hf_tokens[max(0, first - 2):first + 2]}")
            print(f"    vllm: ...{vllm_tokens[max(0, first - 2):first + 2]}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def _last_token_logprobs(llm, prompts, params):
    """Teacher-forced logprob of the final decoder-prompt token, per
    request.  One forward pass - no autoregressive compounding."""
    values = []
    for out in llm.generate(prompts, params):
        token = out.prompt_token_ids[-1]
        row = out.prompt_logprobs[-1]
        values.append(
            row[token].logprob if row and token in row else float("nan")
        )
    return values


def check_batching(llm, args) -> bool:
    """Slot/mask misalignment under continuous batching corrupts logits
    catastrophically; batch-size-dependent kernel reduction order only
    perturbs them slightly.  So compare teacher-forced logprobs solo vs
    batched, with the same tolerance as the prefill check."""
    from vllm import SamplingParams

    print("\n=== 3. batched == solo (teacher-forced logprobs) ===")
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    prompts = [
        {
            "prompt": "The answer is",
            "multi_modal_data": {"text": source},
        }
        for source in SOURCES
    ]
    solo = [
        _last_token_logprobs(llm, [prompt], params)[0] for prompt in prompts
    ]
    batched = _last_token_logprobs(llm, prompts, params)

    ok = True
    worst = 0.0
    for i, (a, b) in enumerate(zip(solo, batched)):
        diff = abs(a - b)
        worst = max(worst, diff)
        if not diff <= args.prefill_tol:  # catches NaN too
            ok = False
            print(f"  DIFF {diff:.4f} for request {i} "
                  f"(solo={a:.4f}, batched={b:.4f})")
    print(f"  worst |logprob diff| = {worst:.5f} "
          f"({'OK' if ok else 'FAIL'}, tol {args.prefill_tol}, "
          f"{len(prompts)} requests)")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5gemma-2-270m-270m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float32"],
                        help="float32 = decisive exactness gate; bfloat16 = "
                             "production dtype (kernel noise expected)")
    parser.add_argument("--prefill-tol", type=float, default=None,
                        help="default: 1e-2 for float32, 0.5 for bfloat16")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="defaults to the decoder sliding window")
    parser.add_argument("--no-eager", action="store_true",
                        help="run vLLM with CUDA graphs enabled")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["prefill", "greedy", "batching"])
    args = parser.parse_args()

    if args.prefill_tol is None:
        args.prefill_tol = 1e-2 if args.dtype == "float32" else 0.5

    max_model_len = args.max_model_len or derive_max_model_len(args.model)
    print(f"dtype = {args.dtype}, max_model_len = {max_model_len}, "
          f"prefill tol = {args.prefill_tol}")
    tokenizer, hf_model = load_hf(args.model, args.device, args.dtype)
    llm = load_vllm(
        args.model,
        enforce_eager=not args.no_eager,
        max_model_len=max_model_len,
        dtype=args.dtype,
    )

    results = {}
    if "prefill" not in args.skip:
        results["prefill"] = check_prefill(tokenizer, hf_model, llm, args)
    if "greedy" not in args.skip:
        results["greedy"] = check_greedy(tokenizer, hf_model, llm, args)
    if "batching" not in args.skip:
        results["batching"] = check_batching(llm, args)

    print("\n=== summary ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
