# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example: text-to-text generation with T5Gemma2 on vLLM.

T5Gemma2 is an encoder-decoder model; this plugin runs it through vLLM's
decoder-only path with the encoder input passed as a "text" multimodal item
(see vllm_bart_plugin/t5gemma2.py for how and why).  Three settings are
mandatory and enforced by the model with clear errors:

- ``hf_overrides=t5gemma2_hf_overrides`` (routes vLLM to the decoder-only
  path; the dict form ``{"is_encoder_decoder": False}`` works as well),
- ``max_model_len <= 4096`` (the decoder sliding window; below it the
  implementation is exact vs HuggingFace),
- ``enable_prefix_caching=False`` and un-chunked multimodal input.

Note: google/t5gemma-2-270m-270m is a gated model - run
``huggingface-cli login`` first.
"""

from vllm import LLM, SamplingParams

from vllm_bart_plugin.t5gemma2 import t5gemma2_hf_overrides


def main():
    llm = LLM(
        model="google/t5gemma-2-270m-270m",
        hf_overrides=t5gemma2_hf_overrides,
        max_model_len=4096,
        enable_prefix_caching=False,
        disable_chunked_mm_input=True,
        dtype="bfloat16",
        enforce_eager=True,  # relax once CUDA-graph runs are validated
    )

    source_texts = [
        "The tower is 324 metres (1,063 ft) tall, about the same height as "
        "an 81-storey building, and the tallest structure in Paris. Its "
        "base is square, measuring 125 metres (410 ft) on each side.",
        "Photosynthesis is the process by which green plants use sunlight "
        "to synthesize food from carbon dioxide and water, generating "
        "oxygen as a byproduct.",
    ]

    # The encoder input goes in via multi_modal_data; the (decoder) prompt
    # is empty - the tokenizer supplies the decoder BOS automatically.
    prompts = [
        {"prompt": "", "multi_modal_data": {"text": text}}
        for text in source_texts
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params)

    for source, output in zip(source_texts, outputs):
        print("=" * 70)
        print(f"Encoder input: {source[:80]}...")
        print(f"Generated:     {output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
