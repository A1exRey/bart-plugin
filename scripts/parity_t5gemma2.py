# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU parity harness: vLLM T5Gemma2 plugin vs HuggingFace transformers.

Run on a machine with a GPU and access to the (gated) checkpoint:

    huggingface-cli login
    python scripts/parity_t5gemma2.py --model google/t5gemma-2-270m-270m

Checks, in order of increasing integration depth:

1. prefill  - teacher-forced next-token logprobs on the decoder prompt
              (vLLM prompt_logprobs vs HF forward).  Catches wiring bugs in
              the prefix-KV scheme, RoPE offsets, and tokenization drift.
              Expect max |diff| ~1e-2 in bfloat16.
2. greedy   - 64-token greedy continuations, token-exact vs HF generate().
3. batching - a mixed-length batch must reproduce the solo-run outputs
              (catches slot/mask misalignment under continuous batching).

If (1) fails, nothing else can work - fix that first.
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


def load_hf(model_name: str, device: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    return tokenizer, model


def derive_max_model_len(model_name: str) -> int:
    """The plugin is exact up to the decoder's sliding window."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    window = getattr(config.decoder, "sliding_window", None)
    return min(window or 4096, 4096)


def load_vllm(model_name: str, enforce_eager: bool, max_model_len: int):
    from vllm import LLM

    from vllm_bart_plugin.t5gemma2 import t5gemma2_hf_overrides

    return LLM(
        model=model_name,
        hf_overrides=t5gemma2_hf_overrides,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
        disable_chunked_mm_input=True,
        dtype="bfloat16",
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

    print("\n=== 2. greedy decode parity ===")
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
        if hf_tokens[:n] != vllm_tokens[:n]:
            ok = False
            first = next(
                i for i in range(n) if hf_tokens[i] != vllm_tokens[i]
            )
            print(f"  MISMATCH at step {first} (src={source[:40]!r})")
            print(f"    hf  : {hf_tokens[:first + 3]}")
            print(f"    vllm: {vllm_tokens[:first + 3]}")
        else:
            print(f"  exact ({n} tokens): {source[:40]!r}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def check_batching(llm, args) -> bool:
    from vllm import SamplingParams

    print("\n=== 3. batched == solo ===")
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [
        {"prompt": "", "multi_modal_data": {"text": source}}
        for source in SOURCES
    ]
    solo = [
        llm.generate([prompt], params)[0].outputs[0].token_ids
        for prompt in prompts
    ]
    batched = [
        out.outputs[0].token_ids for out in llm.generate(prompts, params)
    ]
    ok = True
    for i, (a, b) in enumerate(zip(solo, batched)):
        if list(a) != list(b):
            ok = False
            print(f"  MISMATCH for request {i}: solo={a[:8]} batch={b[:8]}")
    print(f"  {'OK' if ok else 'FAIL'} ({len(prompts)} requests)")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5gemma-2-270m-270m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--prefill-tol", type=float, default=2e-2)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="defaults to the decoder sliding window")
    parser.add_argument("--no-eager", action="store_true",
                        help="run vLLM with CUDA graphs enabled")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["prefill", "greedy", "batching"])
    args = parser.parse_args()

    max_model_len = args.max_model_len or derive_max_model_len(args.model)
    print(f"max_model_len = {max_model_len}")
    tokenizer, hf_model = load_hf(args.model, args.device)
    llm = load_vllm(
        args.model,
        enforce_eager=not args.no_eager,
        max_model_len=max_model_len,
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
