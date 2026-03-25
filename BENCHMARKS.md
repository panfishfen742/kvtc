# KVTC Benchmarks

All benchmarks run on Mistral-7B-Instruct-v0.3 (4-bit quantized via BitsAndBytes) on NVIDIA RTX 5080.

## Mistral-7B Results

**Model config:** 32 layers, 8 KV heads, head_dim=128

### Compression vs Quality Tradeoff

| Mode | Bit Budget | Key Cosine | Value Cosine | Compression | Notes |
|------|:----------:|:----------:|:------------:|:-----------:|-------|
| High Quality | 25% | 0.9996 | 0.9982 | 7.9× | Near-lossless |
| Balanced | 12% | 0.9950 | 0.9818 | 16.2× | Paper's target range |
| Aggressive | 6% | 0.9895 | 0.9654 | 31.7× | Maximum compression |

### Per-Prompt Breakdown (Balanced 12% mode)

| Prompt | Tokens | Middle | Compression | Key Cosine | Value Cosine |
|--------|:------:|:------:|:-----------:|:----------:|:------------:|
| Attention mechanisms | 140 | 8 | 16.0× | 0.9991 | 0.9962 |
| Web app guide | 145 | 13 | 16.2× | 0.9976 | 0.9872 |
| AI history | 161 | 29 | 16.5× | 0.9883 | 0.9619 |

Longer middle regions (more tokens to compress) show the expected quality/compression tradeoff.

### TinyLlama-1.1B Results

**Model config:** 22 layers, 4 KV heads, head_dim=64

| Prompt | Tokens | Middle | Compression | Key Cosine | Value Cosine |
|--------|:------:|:------:|:-----------:|:----------:|:------------:|
| Attention mechanisms | 200 | 68 | 8.0× | 0.9964 | 0.9965 |
| Web app guide | 228 | 96 | 8.0× | 0.9939 | 0.9950 |
| AI history | 272 | 140 | 8.1× | 0.9891 | 0.9927 |

## Memory Savings

For Mistral-7B with 8K context (8,192 tokens):

| Mode | FP16 KV Cache | Compressed | Savings |
|------|:-------------:|:----------:|:-------:|
| No compression | 2,048 MB | — | — |
| High Quality (8×) | — | 256 MB | 1,792 MB saved |
| Balanced (16×) | — | 128 MB | 1,920 MB saved |
| Aggressive (32×) | — | 64 MB | 1,984 MB saved |

*Calculation: 32 layers × 8 heads × 8192 tokens × 128 dim × 2 (K+V) × 2 bytes (FP16) = 2,048 MB*

## Timing

All times on CPU (RTX 5080, no GPU kernels used):

| Operation | Time | Notes |
|-----------|:----:|-------|
| Calibration (10 texts) | ~2s | One-time per model |
| Compress (per prompt) | ~30-60s | Dominated by DP algorithm |
| Decompress (per prompt) | ~0.5-1s | Fast inverse transforms |

**Note:** Compression time is high because this is a pure Python/PyTorch reference implementation. The paper uses nvCOMP GPU-accelerated DEFLATE and optimized kernels. Production implementations would be orders of magnitude faster.

## Reproducing

```bash
# Install
pip install -e ".[dev]"

# Unit tests (no model download needed)
pytest src/test_kvtc.py -v

# Mistral-7B benchmark (requires ~5GB VRAM with 4-bit)
python bench_mistral.py

# TinyLlama quick test (requires ~2GB VRAM or CPU)
RUN_REAL_MODEL_TEST=1 pytest src/test_real_model.py -v
```
