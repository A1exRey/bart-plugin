# vLLM BART Model Plugin

This plugin adds support for BART (Bidirectional and Auto-Regressive Transformers) and Florence-2 models to vLLM's inference engine.

## Overview

BART is an encoder-decoder model that is particularly effective for sequence-to-sequence tasks like summarization, translation, and text generation. Florence-2 is a vision-language model from Microsoft that uses a BART-based language backbone with a DaViT vision encoder. This plugin integrates these models with vLLM's plugin system, allowing you to use them with vLLM's optimized inference engine.

## Installation

### Prerequisites

This plugin requires [uv](https://docs.astral.sh/uv/) for package management. If you don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:
```bash
git clone <repository-url>
cd bart-plugin
```

2. Install the plugin in development mode:
```bash
uv pip install -e .
```

Or install directly:
```bash
uv pip install .
```

### Verify Installation

After installation, vLLM should automatically discover and load the BART plugin. You can verify by checking the vLLM logs when initializing a model.

```bash
python verify_plugin.py
```

## Usage

### Basic Usage

### BART

Run `python example_usage.py` or

```python
import vllm_bart_plugin
from vllm import LLM, SamplingParams
model_name = "facebook/bart-large-cnn"

llm = LLM(
    model=model_name,
    max_model_len=1024,
    gpu_memory_utilization=0.5,
    dtype="float16",
)
params = SamplingParams(temperature=0.0, max_tokens=20)
outputs = llm.generate(
    [
        {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "The president of the United States is",
                },
            },
            "decoder_prompt": "<s>Donald",
        },
        {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "<s>",
                },
            },
            "decoder_prompt": "<s>The capital of France is",
        },
    ],
    sampling_params=params,
)
for o in outputs:
    generated_text = o.outputs[0].text
    print("output:", generated_text)
```

### Florence-2

For Florence-2 vision-language models, see `example_florence2_usage.py`.

### T5Gemma 2

T5Gemma 2 (e.g. `google/t5gemma-2-270m-270m`, a gated model) is supported
text-to-text.  It requires `transformers>=5.0`
(`pip install vllm-bart-plugin[t5gemma2]`) and three mandatory settings:

```python
from vllm import LLM, SamplingParams
from vllm_bart_plugin.t5gemma2 import t5gemma2_hf_overrides

llm = LLM(
    model="google/t5gemma-2-270m-270m",
    hf_overrides=t5gemma2_hf_overrides,  # route via the decoder-only path
    max_model_len=512,   # <= decoder sliding window (exactness bound; 512 for the 270m)
    enable_prefix_caching=False,
    disable_chunked_mm_input=True,
)

# The encoder input is passed as a "text" multimodal item; the prompt is
# the (optional) decoder prefix.
outputs = llm.generate(
    [{"prompt": "", "multi_modal_data": {"text": "Text to transform..."}}],
    SamplingParams(temperature=0.0, max_tokens=64),
)
```

T5Gemma 2's decoder fuses self- and cross-attention into a single softmax
(merged attention), which has no direct vLLM primitive.  The plugin runs the
model on vLLM's decoder-only path with the encoder output injected as prefix
rows of the paged KV cache — mathematically exact below the sliding-window
bound (see `vllm_bart_plugin/t5gemma2.py` and `tests/test_t5gemma2_math.py`
for the derivation and its proof against the HF reference).  The encoder
applies HF's exact bidirectional sliding-window masks and is exact at any
length.  Current limitations: text-only (the SigLIP vision tower is
skipped), total context capped at the decoder sliding window
(`config.decoder.sliding_window`, 512 for the 270m checkpoint), no prefix
caching.  See `example_t5gemma2_usage.py` and `scripts/parity_t5gemma2.py`.

## Plugin Architecture

This plugin follows vLLM's plugin system architecture:

1. **Entry Point**: Registered via setuptools entry_points in `setup.py`
2. **Registration Function**: `register_bart_model()` in `__init__.py` registers the model with vLLM's ModelRegistry
3. **Model Implementation**: The BART model class in `bart.py` implements vLLM's model interfaces

### Plugin Discovery

vLLM automatically discovers plugins using Python's entry point mechanism. The plugin is registered under the `vllm.plugins` group and is loaded when vLLM initializes.

## Model Features

The BART implementation includes:

- **Encoder-Decoder Architecture**: Full support for BART's encoder-decoder structure
- **Cross-Attention**: Proper implementation of cross-attention between encoder and decoder
- **Multi-Modal Support**: Integration with vLLM's multimodal processing pipeline
- **Quantization Support**: Compatible with vLLM's quantization features
- **Tensor Parallelism**: Support for distributed inference

## Supported Models

This plugin should work with any BART-based model from HuggingFace, including:

- `facebook/bart-large-cnn`
- `facebook/bart-large`
- Other BART variants and fine-tuned models

### Florence-2 Models

- `microsoft/Florence-2-base`
- `microsoft/Florence-2-large`

Note: Florence-2 requires `trust_remote_code=True` and uses a separate tokenizer (`Isotr0py/Florence-2-tokenizer`).

### T5Gemma 2 Models

- `google/t5gemma-2-270m-270m` (gated; requires HF authentication)

Note: requires `transformers>=5.0` and the settings shown in the usage
section above (`hf_overrides=t5gemma2_hf_overrides`, `max_model_len` at
most the decoder sliding window — 512 for the 270m checkpoint,
`enable_prefix_caching=False`, `disable_chunked_mm_input=True`).

## Evaluation

To evaluate the model on CNN/DailyMail summarization:

```bash
# Install evaluation dependencies
pip install datasets rouge-score

# Quick test
python scripts/eval_cnn_dailymail.py --num-samples 10

# Full evaluation on test set
python scripts/eval_cnn_dailymail.py --split test
```

See `scripts/eval_cnn_dailymail.py` for more options and reference ROUGE scores.

## TODO
 - [ ] Support `MBartForConditionalGeneration`
 - [x] Support `Florence2ForConditionalGeneration`
 - [x] Support `T5Gemma2ForConditionalGeneration` (text-to-text)
 - [ ] T5Gemma2: image inputs (SigLIP vision tower)
 - [ ] T5Gemma2: contexts beyond the sliding window, prefix caching

## Environment Variables

You can control plugin loading with the `VLLM_PLUGINS` environment variable:

```bash
# Load all plugins (default)
export VLLM_PLUGINS=all

# Load only specific plugins
export VLLM_PLUGINS=bart

# Disable all plugins
export VLLM_PLUGINS=none
```

### `VLLM_BART_ENCODER_MAX_SEQ_PADDING`

Enable a small optimization that **batches encoder forwards** by padding all encoder inputs in a batch to the maximum encoder sequence length, running the encoder once, then slicing outputs back to per-item lengths. As performance is highly dependent on input sequence lenght distribution, this feature is disabled by default.

```bash
export VLLM_BART_ENCODER_MAX_SEQ_PADDING=1
```

Notes:
- Requires `pad_token_id` to be set in the HF config. If it is missing, the plugin will log a warning and keep the optimization disabled.


## Development

### Project Structure

```
bart-plugin/
├── vllm_bart_plugin/
│   ├── __init__.py          # Plugin registration
│   └── bart.py              # BART model implementation
├── setup.py                 # Package configuration and entry points
├── README.md                # This file
└── LICENSE                  # License file
```

### Running Tests

```bash
# Install development dependencies
uv pip install -e .

# Install pre-commit hooks
pre-commit install

# Run fast tests
pytest -m tests/
```

## Troubleshooting

### Plugin Not Loading

If the plugin isn't being discovered:

1. Verify installation: `uv pip list | grep vllm-bart-plugin`
2. Check entry points: `python -c "from importlib.metadata import entry_points; print(list(entry_points(group='vllm.plugins')))"`
3. Enable verbose logging: Set `VLLM_LOGGING_LEVEL=DEBUG`
4. Run the verification script: `python verify_plugin.py`

### Model Not Found

If vLLM doesn't recognize the BART model:

1. Ensure the plugin loaded successfully (check logs) you should see
```bash
[2025-12-19 14:32:11] INFO __init__.py:33: Successfully registered BART model with vLLM
```
2. Verify the model architecture name matches: `BartForConditionalGeneration`
3. Try explicitly setting `trust_remote_code=False`

### Import Errors

Make sure all dependencies are installed:

```bash
uv pip install vllm torch transformers
```
