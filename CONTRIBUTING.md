# Contributing to KVTC

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/OnlyTerp/kvtc.git
cd kvtc
pip install -e ".[dev]"
pytest src/test_kvtc.py  # all 38 tests should pass
```

## What We Need Help With

- **GPU-accelerated entropy coding** — Replace zlib with nvCOMP or custom CUDA kernels
- **Triton kernels** — Fused PCA transform + quantize kernel for inference speed
- **More model benchmarks** — Test on Llama-3, Qwen, Gemma, etc.
- **vLLM integration** — KV cache manager plugin for production serving
- **Pipelined decompression** — Layer-by-layer decompress overlapped with attention

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make sure all existing tests pass: `pytest src/test_kvtc.py`
3. Add tests for any new functionality
4. Update docs if you change the public API
5. Open a PR with a clear description

## Code Style

- Type annotations on all public functions
- Docstrings on all public APIs
- Pure PyTorch — avoid unnecessary dependencies
- Tests go in `src/test_kvtc.py`

## Reporting Issues

Open a GitHub issue with:
- What you expected vs what happened
- Minimal reproduction code
- Your environment (Python version, PyTorch version, OS)
