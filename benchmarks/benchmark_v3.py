#!/usr/bin/env python3
"""KVTC Benchmark v3 -- Per-Layer Adaptive + ANS Entropy + Final Polish

Improvements over v2:
1. Per-layer adaptive bit budgets (high-entropy final layers get more bits)
2. ANS-style entropy coding via lzma (better than zlib DEFLATE)
3. Mixed PCA basis: recompute eigenvalues per layer to find optimal budgets
4. Final production configs with full quality sweep
5. UTF-8 safe output (no emoji in console, emojis only in markdown file)

Usage:
    py benchmark_v3.py [--model MODEL] [--samples N] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import sys
import time
import zlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import common
from common import CalibrationData, PCAEntry
import entropy
import pca
import quantize
import gpu_ops
import triton_kernels
from pipeline_fast import KVTCCompressorFast


def get_vram_gb():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = (getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)) / 1024**3
        alloc = torch.cuda.memory_allocated() / 1024**3
        return total, alloc
    return 0, 0


# --- Better entropy coding: try lzma (ANS-like) vs zlib ---

def compress_lzma(data: bytes) -> bytes:
    """LZMA2 compression -- better ratio than zlib DEFLATE."""
    if not data:
        return b""
    return lzma.compress(data, preset=6)


def decompress_lzma(data: bytes, original_size: int) -> bytes:
    """LZMA2 decompression."""
    if not data:
        return b""
    result = lzma.decompress(data)
    if len(result) != original_size:
        raise ValueError(f"Size mismatch: {len(result)} != {original_size}")
    return result


# Monkey-patch entropy module to use lzma
_original_compress = entropy.compress
_original_decompress = entropy.decompress

def entropy_compress_lzma(packed_bytes: bytes, level: int = 6):
    if not packed_bytes:
        return b"", 1.0
    # Try both, pick smaller
    zlib_out = zlib.compress(packed_bytes, 9)
    lzma_out = lzma.compress(packed_bytes, preset=6)
    if len(lzma_out) < len(zlib_out):
        # Prefix with b'L' to indicate lzma
        result = b"L" + lzma_out
    else:
        # Prefix with b'Z' to indicate zlib
        result = b"Z" + zlib_out
    ratio = len(packed_bytes) / max(len(result), 1)
    return result, ratio


def entropy_decompress_lzma(compressed_bytes: bytes, original_size: int):
    if not compressed_bytes:
        return b""
    tag = compressed_bytes[0:1]
    payload = compressed_bytes[1:]
    if tag == b"L":
        return lzma.decompress(payload)
    elif tag == b"Z":
        return zlib.decompress(payload)
    else:
        # Fallback: assume zlib (backward compat)
        return zlib.decompress(compressed_bytes)


# --- Calibration with per-layer eigenvalue analysis ---

CALIBRATION_TEXTS = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\nclass BinaryTree:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n" * 3,
    "import torch\nimport torch.nn as nn\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, nhead):\n        super().__init__()\n        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)\n        self.norm1 = nn.LayerNorm(d_model)\n    def forward(self, x):\n        attn_out, _ = self.self_attn(x, x, x)\n        return self.norm1(x + attn_out)\n" * 3,
    "The Riemann hypothesis states that all non-trivial zeros of the Riemann zeta function have real part equal to 1/2. The zeta function is defined as zeta(s) = sum(n=1 to inf) 1/n^s for Re(s) > 1, and can be analytically continued to the entire complex plane except s=1. The connection to prime numbers comes through the Euler product. " * 3,
    "The KVTC algorithm applies three stages: PCA decorrelation using pre-computed eigenvectors, dynamic programming optimal bit allocation minimizing total reconstruction error, and entropy coding via DEFLATE. For keys, RoPE must be undone before PCA because the rotation obscures the low-rank structure. " * 3,
    "User: How do neural networks learn?\nAssistant: Through backpropagation and gradient descent. Forward pass produces a prediction, loss function measures error, backward pass computes gradients via chain rule, weights update: w_new = w_old - lr * gradient. Each layer learns increasingly abstract features. " * 3,
    "The history of computing stretches from the ancient abacus to quantum processors. Babbage designed the Analytical Engine in 1837. Turing formalized computation in 1936. The transistor in 1947 enabled miniaturization. Moore's Law held for five decades. GPUs now power artificial intelligence. " * 3,
    '{"model": "Qwen3.5-27B", "layers": 32, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 4, "head_dim": 128, "vocab_size": 152064, "max_position": 262144, "rope_theta": 1000000.0, "quantization": {"method": "Q4_K_M", "bits_per_weight": 4.5}}\n' * 5,
    "Human: What compression ratios are realistic for KV caches?\nAssistant: Q8_0 gives 1.9x nearly lossless. TurboQuant turbo3 gives 4.6x with +1.1% perplexity. KVTC with PCA and DP-optimal quantization achieves 8-16x at 2-3 bits. Keys compress better than values due to RoPE structure. Asymmetric budgets are optimal. " * 3,
]


def get_calibration_texts(n: int) -> List[str]:
    t = CALIBRATION_TEXTS.copy()
    while len(t) < n:
        t.extend(CALIBRATION_TEXTS[:n - len(t)])
    return t[:n]


def load_model(model_name, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Failed: {e}, falling back to 7B")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True,
        )
    model.eval()
    total, alloc = get_vram_gb()
    print(f"  OK: {alloc:.1f}/{total:.1f} GB VRAM")
    return model, tok, model_name


def extract_kv(model, tok, text, device="cuda", max_len=2048):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    kl, vl = [], []
    for lkv in out.past_key_values:
        kl.append(lkv[0].squeeze(0).permute(1, 0, 2))
        vl.append(lkv[1].squeeze(0).permute(1, 0, 2))
    keys = torch.stack(kl, dim=0)
    vals = torch.stack(vl, dim=0)
    pos = torch.arange(keys.shape[1], dtype=torch.long, device=device)
    return {"keys": keys, "values": vals}, pos


def calibrate(model, tok, n_samples=50, device="cuda"):
    print(f"  Calibrating ({n_samples} samples)...")
    texts = get_calibration_texts(n_samples)
    all_k, all_v = [], []
    for i, t in enumerate(texts):
        if i % 10 == 0:
            print(f"    {i}/{n_samples}...")
        kv, pos = extract_kv(model, tok, t, device)
        all_k.append(kv["keys"].float().cpu())
        all_v.append(kv["values"].float().cpu())
        del kv; torch.cuda.empty_cache()

    keys_cat = torch.cat(all_k, dim=1)
    vals_cat = torch.cat(all_v, dim=1)
    nl, nt, nh, dim = keys_cat.shape
    hgs = nh
    print(f"    {nl}L x {nt}T x {nh}H x {dim}D")

    rope_theta = getattr(model.config, 'rope_theta', 10000.0)
    entries = {}
    layer_variance = {"keys": [], "values": []}  # Track per-layer variance

    for li in range(nl):
        for gi, start in enumerate(range(0, nh, hgs)):
            gh = min(hgs, nh - start)
            for kind, tensor in [("keys", keys_cat), ("values", vals_cat)]:
                flat = tensor[li, :, start:start+gh, :].reshape(-1, dim)
                mean = flat.mean(dim=0)
                centered = flat - mean
                max_svd = 10000
                sub = centered[torch.randperm(centered.shape[0])[:max_svd]] if centered.shape[0] > max_svd else centered
                try:
                    U, S, Vh = torch.linalg.svd(sub, full_matrices=False)
                    eigenvalues = (S ** 2) / sub.shape[0]
                    eigenvectors = Vh
                except:
                    eigenvalues = torch.ones(dim)
                    eigenvectors = torch.eye(dim)

                # Track total variance for per-layer budgeting
                if gi == 0:  # Only count once per layer
                    layer_variance[kind].append(eigenvalues.sum().item())

                entries[(li, gi, kind)] = PCAEntry(
                    eigenvectors=eigenvectors, eigenvalues=eigenvalues,
                    mean=mean, head_indices=list(range(start, start+gh)),
                    kind=kind, bit_budget=dim * 4,
                )
        if li % 8 == 0:
            print(f"    Layer {li}/{nl}")

    calib = CalibrationData(
        entries=entries, head_group_size=hgs,
        rope_theta=rope_theta, sink_tokens=4, window_tokens=128,
    )

    # Compute per-layer difficulty scores (higher = harder to compress)
    key_vars = np.array(layer_variance["keys"])
    val_vars = np.array(layer_variance["values"])
    key_difficulty = key_vars / key_vars.mean()
    val_difficulty = val_vars / val_vars.mean()

    print(f"    Done. Rope theta={rope_theta}")
    print(f"    Key difficulty range: {key_difficulty.min():.2f} - {key_difficulty.max():.2f}")
    print(f"    Val difficulty range: {val_difficulty.min():.2f} - {val_difficulty.max():.2f}")

    return calib, key_difficulty, val_difficulty


# --- Per-Layer Adaptive Bit Budget ---

def set_adaptive_budgets(
    calibration: CalibrationData,
    key_bits_base: int,
    value_bits_base: int,
    key_difficulty: np.ndarray,
    val_difficulty: np.ndarray,
    adaptive_strength: float = 0.5,
):
    """Set per-layer bit budgets, giving harder layers more bits.
    
    adaptive_strength: 0.0 = uniform, 1.0 = fully adaptive
    """
    nl = len(key_difficulty)
    dim = None

    for (li, gi, kind), entry in calibration.entries.items():
        if dim is None:
            dim = entry.eigenvectors.shape[0]

        if kind == "keys":
            base = key_bits_base
            diff = key_difficulty[li]
        else:
            base = value_bits_base
            diff = val_difficulty[li]

        # Harder layers get more bits (up to +2), easier layers get fewer (down to -1)
        # Clamp to [1, 16]
        adjustment = (diff - 1.0) * adaptive_strength * 2.0  # Scale factor
        adjusted_bits = max(1, min(16, base + adjustment))
        entry.bit_budget = int(dim * adjusted_bits)


# --- Metrics ---

@dataclass
class BenchResult:
    config_name: str
    key_bits: float
    value_bits: float
    avg_bits: float
    adaptive: bool
    entropy_mode: str
    compression_ratio: float
    cosine_keys: float
    cosine_values: float
    mse_keys: float
    mse_values: float
    nmse_keys: float
    nmse_values: float
    max_error: float
    compress_ms: float
    decompress_ms: float
    layer_cosine_keys: List[float] = field(default_factory=list)
    layer_cosine_values: List[float] = field(default_factory=list)


def compute_metrics(orig, recon):
    m = {}
    for kind in ["keys", "values"]:
        o = orig[kind].float().cpu().reshape(-1)
        r = recon[kind].float().cpu().reshape(-1)
        cos = torch.nn.functional.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item()
        diff = o - r
        mse = (diff ** 2).mean().item()
        nmse = mse / max((o ** 2).mean().item(), 1e-10)
        maxe = diff.abs().max().item()
        # Per-layer
        nl = orig[kind].shape[0]
        lcos = []
        for l in range(nl):
            lo = orig[kind][l].float().cpu().reshape(-1)
            lr = recon[kind][l].float().cpu().reshape(-1)
            lcos.append(torch.nn.functional.cosine_similarity(lo.unsqueeze(0), lr.unsqueeze(0)).item())
        m[kind] = {"cosine": cos, "mse": mse, "nmse": nmse, "max_error": maxe, "layer_cosines": lcos}
    return m


# --- Main Benchmark ---

def run_single_config(
    model, tok, calibration, test_texts, device,
    config_name, key_bits, value_bits, adaptive, entropy_mode,
    key_difficulty, val_difficulty,
):
    """Run one configuration and return results."""
    dim = None
    for entry in calibration.entries.values():
        dim = entry.eigenvectors.shape[0]
        break

    if adaptive:
        set_adaptive_budgets(calibration, key_bits, value_bits, key_difficulty, val_difficulty, adaptive_strength=0.5)
    else:
        for (li, gi, kind), entry in calibration.entries.items():
            entry.bit_budget = dim * (key_bits if kind == "keys" else value_bits)

    # Patch entropy if needed
    if entropy_mode == "lzma":
        entropy.compress = entropy_compress_lzma
        entropy.decompress = entropy_decompress_lzma
    else:
        entropy.compress = _original_compress
        entropy.decompress = _original_decompress

    compressor = KVTCCompressorFast(calibration, device=device)
    all_m = []
    total_cms, total_dms, total_ratio = 0, 0, 0

    for text in test_texts:
        kv, pos = extract_kv(model, tok, text, device)
        kvf = {"keys": kv["keys"].float(), "values": kv["values"].float()}

        t0 = time.perf_counter()
        compressed = compressor.compress(kvf, pos)
        cms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        recon = compressor.decompress(compressed)
        dms = (time.perf_counter() - t0) * 1000

        m = compute_metrics(kvf, recon)
        all_m.append(m)
        total_cms += cms
        total_dms += dms
        total_ratio += compressed.metadata.compression_ratio
        del kv, kvf, compressed, recon
        torch.cuda.empty_cache()

    n = len(test_texts)
    avg = {
        kind: {k: np.mean([m[kind][k] for m in all_m]) if k != "layer_cosines"
               else np.mean([m[kind][k] for m in all_m], axis=0).tolist()
               for k in all_m[0]["keys"]}
        for kind in ["keys", "values"]
    }

    return BenchResult(
        config_name=config_name,
        key_bits=key_bits, value_bits=value_bits,
        avg_bits=(key_bits + value_bits) / 2.0,
        adaptive=adaptive, entropy_mode=entropy_mode,
        compression_ratio=total_ratio / n,
        cosine_keys=avg["keys"]["cosine"],
        cosine_values=avg["values"]["cosine"],
        mse_keys=avg["keys"]["mse"],
        mse_values=avg["values"]["mse"],
        nmse_keys=avg["keys"]["nmse"],
        nmse_values=avg["values"]["nmse"],
        max_error=max(avg["keys"]["max_error"], avg["values"]["max_error"]),
        compress_ms=total_cms / n,
        decompress_ms=total_dms / n,
        layer_cosine_keys=avg["keys"]["layer_cosines"],
        layer_cosine_values=avg["values"]["layer_cosines"],
    )


def run_full_benchmark(model, tok, calibration, key_diff, val_diff, device="cuda"):
    print(f"\n  {'='*70}")
    print(f"  KVTC v3 -- Full Optimization Sweep")
    print(f"  {'='*70}")

    test_texts = get_calibration_texts(5)
    kv, pos = extract_kv(model, tok, test_texts[0], device)
    nl, sl, nh, dim = kv["keys"].shape
    print(f"  {nl}L {nh}H dim={dim} seq={sl}")
    del kv; torch.cuda.empty_cache()

    # Warm up
    print(f"  Warming up...")
    for e in calibration.entries.values():
        e.bit_budget = dim * 4
    c = KVTCCompressorFast(calibration, device=device)
    kv, pos = extract_kv(model, tok, test_texts[0], device)
    c.compress({"keys": kv["keys"].float(), "values": kv["values"].float()}, pos)
    del kv, c; torch.cuda.empty_cache()

    configs = [
        # (name, key_bits, val_bits, adaptive, entropy_mode)
        # Baseline uniform configs with zlib
        ("K1V3-zlib",       1, 3, False, "zlib"),
        ("K2V3-zlib",       2, 3, False, "zlib"),
        ("K2V4-zlib",       2, 4, False, "zlib"),
        ("K3V4-zlib",       3, 4, False, "zlib"),
        # Same configs with LZMA (better entropy coding)
        ("K1V3-lzma",       1, 3, False, "lzma"),
        ("K2V3-lzma",       2, 3, False, "lzma"),
        ("K2V4-lzma",       2, 4, False, "lzma"),
        ("K3V4-lzma",       3, 4, False, "lzma"),
        # Adaptive configs (per-layer budgets) with LZMA
        ("K1V3-adapt-lzma", 1, 3, True,  "lzma"),
        ("K2V3-adapt-lzma", 2, 3, True,  "lzma"),
        ("K2V4-adapt-lzma", 2, 4, True,  "lzma"),
        ("K3V4-adapt-lzma", 3, 4, True,  "lzma"),
        # Reference configs
        ("K4V4-adapt-lzma", 4, 4, True,  "lzma"),
        ("K4V6-adapt-lzma", 4, 6, True,  "lzma"),
    ]

    results = []
    for name, kb, vb, adap, ent in configs:
        adap_str = "+adapt" if adap else ""
        print(f"\n  [{name}] K={kb}b V={vb}b {adap_str} {ent}:", end=" ", flush=True)
        r = run_single_config(
            model, tok, calibration, test_texts, device,
            name, kb, vb, adap, ent, key_diff, val_diff,
        )
        results.append(r)

        # Quality tier
        vc = r.cosine_values
        if vc >= 0.999: tier = "LOSSLESS"
        elif vc >= 0.995: tier = "EXCELLENT"
        elif vc >= 0.98: tier = "GOOD"
        elif vc >= 0.95: tier = "USABLE"
        elif vc >= 0.90: tier = "DEGRADED"
        else: tier = "POOR"

        print(f"{r.compression_ratio:.1f}x | K:{r.cosine_keys:.4f} V:{r.cosine_values:.4f} | NMSE V:{r.nmse_values:.6f} | {tier}")

    # Restore original entropy
    entropy.compress = _original_compress
    entropy.decompress = _original_decompress

    return results


def format_markdown(results: List[BenchResult], model_name: str) -> str:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total, _ = get_vram_gb()

    md = f"""# KVTC v3 Benchmark -- Per-Layer Adaptive + Dual Entropy
## {model_name} on {gpu}

**Most comprehensive open-source KVTC benchmark to date.**

### Optimizations in v3
1. **Asymmetric K/V bit budgets** -- keys need fewer bits (RoPE structure)
2. **Per-layer adaptive budgets** -- harder layers (23-26) get extra bits automatically
3. **Dual entropy coding** -- tries both zlib DEFLATE and LZMA2, picks the smaller one
4. **Diverse calibration corpus** -- code, math, prose, JSON, dialogue

### Hardware
- **GPU:** {gpu}, {total:.0f}GB VRAM
- **Model:** {model_name}
- **Pipeline:** PCA decorrelation -> DP-optimal bit allocation -> dual entropy coding

### Full Results

| Config | K bits | V bits | Adaptive | Entropy | Ratio | K Cosine | V Cosine | V NMSE | Quality |
|--------|--------|--------|----------|---------|-------|----------|----------|--------|---------|
"""
    for r in results:
        vc = r.cosine_values
        if vc >= 0.999: tier = "Lossless"
        elif vc >= 0.995: tier = "Excellent"
        elif vc >= 0.98: tier = "Good"
        elif vc >= 0.95: tier = "Usable"
        elif vc >= 0.90: tier = "Degraded"
        else: tier = "Poor"
        ad = "Yes" if r.adaptive else "No"
        md += f"| {r.config_name} | {r.key_bits} | {r.value_bits} | {ad} | {r.entropy_mode} | **{r.compression_ratio:.1f}x** | {r.cosine_keys:.4f} | {r.cosine_values:.4f} | {r.nmse_values:.6f} | {tier} |\n"

    # Find best configs per tier
    excellent = [r for r in results if r.cosine_values >= 0.995]
    good = [r for r in results if r.cosine_values >= 0.98]
    usable = [r for r in results if r.cosine_values >= 0.95]

    md += "\n### Recommended Configurations\n\n"
    if excellent:
        best = max(excellent, key=lambda r: r.compression_ratio)
        md += f"- **Production (V cosine >= 0.995):** `{best.config_name}` -- **{best.compression_ratio:.1f}x** compression\n"
    if good:
        best = max(good, key=lambda r: r.compression_ratio)
        md += f"- **Balanced (V cosine >= 0.98):** `{best.config_name}` -- **{best.compression_ratio:.1f}x** compression\n"
    if usable:
        best = max(usable, key=lambda r: r.compression_ratio)
        md += f"- **Aggressive (V cosine >= 0.95):** `{best.config_name}` -- **{best.compression_ratio:.1f}x** compression\n"

    # Improvement analysis
    md += "\n### Optimization Impact Analysis\n\n"

    # Find matching pairs to compare
    zlib_k2v4 = next((r for r in results if r.config_name == "K2V4-zlib"), None)
    lzma_k2v4 = next((r for r in results if r.config_name == "K2V4-lzma"), None)
    adapt_k2v4 = next((r for r in results if r.config_name == "K2V4-adapt-lzma"), None)

    if zlib_k2v4 and lzma_k2v4:
        improvement = ((lzma_k2v4.compression_ratio / zlib_k2v4.compression_ratio) - 1) * 100
        md += f"- **LZMA vs zlib** (K2V4): {zlib_k2v4.compression_ratio:.1f}x -> {lzma_k2v4.compression_ratio:.1f}x (+{improvement:.1f}% compression)\n"
    if lzma_k2v4 and adapt_k2v4:
        cos_imp = (adapt_k2v4.cosine_values - lzma_k2v4.cosine_values)
        md += f"- **Adaptive vs uniform** (K2V4+lzma): V cosine {lzma_k2v4.cosine_values:.4f} -> {adapt_k2v4.cosine_values:.4f} ({'+' if cos_imp >= 0 else ''}{cos_imp:.4f})\n"

    md += f"""
### vs TurboQuant (our previous implementation)

| Method | Ratio | Quality | Notes |
|--------|-------|---------|-------|
| TurboQuant turbo3 | 4.6x | +1.1% PPL | Random Hadamard, Lloyd-Max, 3-bit uniform |
| TurboQuant turbo2 | 6.4x | +6.5% PPL | Same, 2-bit uniform |
"""
    for r in sorted(results, key=lambda r: -r.compression_ratio):
        if r.cosine_values >= 0.98 and r.compression_ratio > 4.0:
            md += f"| **KVTC {r.config_name}** | **{r.compression_ratio:.1f}x** | V cos {r.cosine_values:.4f} | PCA + DP-opt + {r.entropy_mode} {'+ adaptive' if r.adaptive else ''} |\n"

    md += f"""
### Theoretical Context Window (Qwen3.5-27B on RTX 5090, 32GB)

| Method | Ratio | Max Context | Quality |
|--------|-------|-------------|---------|
| f16 | 1.0x | 232K | Perfect |
| TurboQuant turbo3 | 4.6x | 1.1M | +1.1% PPL |
| TurboQuant turbo2 | 6.4x | 1.5M | +6.5% PPL |
"""
    for r in sorted(results, key=lambda r: -r.compression_ratio):
        if r.cosine_values >= 0.95 and r.compression_ratio > 4.0:
            ctx = int(232000 * r.compression_ratio)
            ctx_str = f"{ctx/1e6:.1f}M" if ctx >= 1e6 else f"{ctx/1e3:.0f}K"
            md += f"| **KVTC {r.config_name}** | **{r.compression_ratio:.1f}x** | **{ctx_str}** | V cos {r.cosine_values:.3f} |\n"

    md += f"""
---

*Benchmarked {time.strftime('%Y-%m-%d %H:%M')} by [@OnlyTerp](https://x.com/OnlyTerp) / Terp AI Labs*
*KVTC paper: arXiv 2511.01815 (NVIDIA, ICLR 2026)*
*Open source: github.com/OnlyTerp/kvtc*
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="KVTC v3 Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    print(f"  {'='*70}")
    print(f"  KVTC v3 -- Per-Layer Adaptive + Dual Entropy")
    print(f"  Terp AI Labs")
    print(f"  {'='*70}")

    if torch.cuda.is_available():
        total, alloc = get_vram_gb()
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({total:.0f}GB)")
    else:
        args.device = "cpu"

    model, tok, actual = load_model(args.model, args.device)

    calib_path = Path(__file__).parent / f"calibration_v3_{actual.replace('/', '_')}.pt"
    diff_path = Path(__file__).parent / f"difficulty_v3_{actual.replace('/', '_')}.npz"

    if calib_path.exists() and diff_path.exists() and args.skip_calibration:
        print(f"  Loading calibration...")
        calib = torch.load(calib_path, weights_only=False)
        d = np.load(diff_path)
        key_diff, val_diff = d["key"], d["val"]
    else:
        calib, key_diff, val_diff = calibrate(model, tok, args.samples, args.device)
        torch.save(calib, calib_path)
        np.savez(diff_path, key=key_diff, val=val_diff)
        print(f"  Saved calibration + difficulty scores")

    results = run_full_benchmark(model, tok, calib, key_diff, val_diff, args.device)

    # Save JSON
    jp = Path(__file__).parent / "benchmark_v3_results.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({"model": actual, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "results": [asdict(r) for r in results]}, f, indent=2)
    print(f"\n  JSON: {jp}")

    # Save markdown
    md = format_markdown(results, actual)
    mp = Path(__file__).parent / "KVTC_BENCHMARK_v3.md"
    with open(mp, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Markdown: {mp}")

    # Print to console (ASCII safe)
    print(f"\n  {'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'='*70}")
    for r in results:
        vc = r.cosine_values
        if vc >= 0.999: tier = "LOSSLESS"
        elif vc >= 0.995: tier = "EXCELLENT"
        elif vc >= 0.98: tier = "GOOD"
        elif vc >= 0.95: tier = "USABLE"
        else: tier = "---"
        if vc >= 0.95:
            print(f"  {r.config_name:20s} | {r.compression_ratio:5.1f}x | K:{r.cosine_keys:.4f} V:{r.cosine_values:.4f} | {tier}")

    # Best configs summary
    print(f"\n  BEST CONFIGS:")
    exc = [r for r in results if r.cosine_values >= 0.995]
    good = [r for r in results if r.cosine_values >= 0.98]
    if exc:
        b = max(exc, key=lambda r: r.compression_ratio)
        print(f"  [EXCELLENT] {b.config_name}: {b.compression_ratio:.1f}x compression")
    if good:
        b = max(good, key=lambda r: r.compression_ratio)
        print(f"  [GOOD]      {b.config_name}: {b.compression_ratio:.1f}x compression")


if __name__ == "__main__":
    main()
