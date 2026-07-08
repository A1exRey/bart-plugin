# Evaluation Scripts

## T5Gemma2 Speed Benchmark

Measures the plugin's speed-up over HuggingFace transformers (which has no
FlashAttention path for T5Gemma2 — transformers issue #45522). Reports
prefill latency, end-to-end latency, and decode tokens/s per stack, plus a
vLLM-vs-best-HF speed-up summary. Theory and expected results:
`docs/t5gemma2_flash_attention_theory.md`.

```bash
# within the sliding window (short mode):
python bench_t5gemma2.py
# beyond the window (long mode, two-pass FlashAttention, bfloat16):
python bench_t5gemma2.py --long-context --enc-lens 768 1536 3072
```

## CNN/DailyMail Evaluation

### Install Dependencies

```bash
uv pip install datasets rouge-score
```

### Run Evaluation

Quick test (10 samples):

```bash
python eval_cnn_dailymail.py
```

### Full Evaluation

For complete evaluation, specify the split:

```bash
# Test set (11,490 samples)
python eval_cnn_dailymail.py --split test

# Validation set (11,332 samples)
python eval_cnn_dailymail.py --split validation
```

