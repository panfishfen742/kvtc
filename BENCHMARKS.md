# KVTC v3 Benchmark -- Per-Layer Adaptive + Dual Entropy
## Qwen/Qwen2.5-7B-Instruct on NVIDIA GeForce RTX 5090

**Most comprehensive open-source KVTC benchmark to date.**

### Optimizations in v3
1. **Asymmetric K/V bit budgets** -- keys need fewer bits (RoPE structure)
2. **Per-layer adaptive budgets** -- harder layers (23-26) get extra bits automatically
3. **Dual entropy coding** -- tries both zlib DEFLATE and LZMA2, picks the smaller one
4. **Diverse calibration corpus** -- code, math, prose, JSON, dialogue

### Hardware
- **GPU:** NVIDIA GeForce RTX 5090, 32GB VRAM
- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Pipeline:** PCA decorrelation -> DP-optimal bit allocation -> dual entropy coding

### Full Results

| Config | K bits | V bits | Adaptive | Entropy | Ratio | K Cosine | V Cosine | V NMSE | Quality |
|--------|--------|--------|----------|---------|-------|----------|----------|--------|---------|
| K1V3-zlib | 1 | 3 | No | zlib | **8.8x** | 0.9896 | 0.9809 | 0.039831 | Good |
| K2V3-zlib | 2 | 3 | No | zlib | **7.2x** | 0.9958 | 0.9809 | 0.039831 | Good |
| K2V4-zlib | 2 | 4 | No | zlib | **6.1x** | 0.9958 | 0.9959 | 0.008308 | Excellent |
| K3V4-zlib | 3 | 4 | No | zlib | **5.1x** | 1.0001 | 0.9959 | 0.008308 | Excellent |
| K1V3-lzma | 1 | 3 | No | lzma | **8.8x** | 0.9896 | 0.9809 | 0.039831 | Good |
| K2V3-lzma | 2 | 3 | No | lzma | **7.2x** | 0.9958 | 0.9809 | 0.039831 | Good |
| K2V4-lzma | 2 | 4 | No | lzma | **6.1x** | 0.9958 | 0.9959 | 0.008308 | Excellent |
| K3V4-lzma | 3 | 4 | No | lzma | **5.1x** | 1.0001 | 0.9959 | 0.008308 | Excellent |
| K1V3-adapt-lzma | 1 | 3 | Yes | lzma | **7.6x** | 0.9903 | 0.9871 | 0.026541 | Good |
| K2V3-adapt-lzma | 2 | 3 | Yes | lzma | **7.4x** | 0.9905 | 0.9871 | 0.026541 | Good |
| K2V4-adapt-lzma | 2 | 4 | Yes | lzma | **5.9x** | 0.9905 | 0.9976 | 0.004820 | Excellent |
| K3V4-adapt-lzma | 3 | 4 | Yes | lzma | **5.1x** | 0.9967 | 0.9976 | 0.004820 | Excellent |
| K4V4-adapt-lzma | 4 | 4 | Yes | lzma | **4.4x** | 1.0002 | 0.9976 | 0.004820 | Excellent |
| K4V6-adapt-lzma | 4 | 6 | Yes | lzma | **3.4x** | 1.0002 | 0.9999 | 0.000254 | Lossless |

### Recommended Configurations

- **Production (V cosine >= 0.995):** `K2V4-zlib` -- **6.1x** compression
- **Balanced (V cosine >= 0.98):** `K1V3-zlib` -- **8.8x** compression
- **Aggressive (V cosine >= 0.95):** `K1V3-zlib` -- **8.8x** compression

### Optimization Impact Analysis

- **LZMA vs zlib** (K2V4): 6.1x -> 6.1x (+0.0% compression)
- **Adaptive vs uniform** (K2V4+lzma): V cosine 0.9959 -> 0.9976 (+0.0017)

### vs TurboQuant (our previous implementation)

| Method | Ratio | Quality | Notes |
|--------|-------|---------|-------|
| TurboQuant turbo3 | 4.6x | +1.1% PPL | Random Hadamard, Lloyd-Max, 3-bit uniform |
| TurboQuant turbo2 | 6.4x | +6.5% PPL | Same, 2-bit uniform |
| **KVTC K1V3-zlib** | **8.8x** | V cos 0.9809 | PCA + DP-opt + zlib  |
| **KVTC K1V3-lzma** | **8.8x** | V cos 0.9809 | PCA + DP-opt + lzma  |
| **KVTC K1V3-adapt-lzma** | **7.6x** | V cos 0.9871 | PCA + DP-opt + lzma + adaptive |
| **KVTC K2V3-adapt-lzma** | **7.4x** | V cos 0.9871 | PCA + DP-opt + lzma + adaptive |
| **KVTC K2V3-zlib** | **7.2x** | V cos 0.9809 | PCA + DP-opt + zlib  |
| **KVTC K2V3-lzma** | **7.2x** | V cos 0.9809 | PCA + DP-opt + lzma  |
| **KVTC K2V4-zlib** | **6.1x** | V cos 0.9959 | PCA + DP-opt + zlib  |
| **KVTC K2V4-lzma** | **6.1x** | V cos 0.9959 | PCA + DP-opt + lzma  |
| **KVTC K2V4-adapt-lzma** | **5.9x** | V cos 0.9976 | PCA + DP-opt + lzma + adaptive |
| **KVTC K3V4-adapt-lzma** | **5.1x** | V cos 0.9976 | PCA + DP-opt + lzma + adaptive |
| **KVTC K3V4-zlib** | **5.1x** | V cos 0.9959 | PCA + DP-opt + zlib  |
| **KVTC K3V4-lzma** | **5.1x** | V cos 0.9959 | PCA + DP-opt + lzma  |
| **KVTC K4V4-adapt-lzma** | **4.4x** | V cos 0.9976 | PCA + DP-opt + lzma + adaptive |

### Theoretical Context Window (Qwen3.5-27B on RTX 5090, 32GB)

| Method | Ratio | Max Context | Quality |
|--------|-------|-------------|---------|
| f16 | 1.0x | 232K | Perfect |
| TurboQuant turbo3 | 4.6x | 1.1M | +1.1% PPL |
| TurboQuant turbo2 | 6.4x | 1.5M | +6.5% PPL |
| **KVTC K1V3-zlib** | **8.8x** | **2.1M** | V cos 0.981 |
| **KVTC K1V3-lzma** | **8.8x** | **2.1M** | V cos 0.981 |
| **KVTC K1V3-adapt-lzma** | **7.6x** | **1.8M** | V cos 0.987 |
| **KVTC K2V3-adapt-lzma** | **7.4x** | **1.7M** | V cos 0.987 |
| **KVTC K2V3-zlib** | **7.2x** | **1.7M** | V cos 0.981 |
| **KVTC K2V3-lzma** | **7.2x** | **1.7M** | V cos 0.981 |
| **KVTC K2V4-zlib** | **6.1x** | **1.4M** | V cos 0.996 |
| **KVTC K2V4-lzma** | **6.1x** | **1.4M** | V cos 0.996 |
| **KVTC K2V4-adapt-lzma** | **5.9x** | **1.4M** | V cos 0.998 |
| **KVTC K3V4-adapt-lzma** | **5.1x** | **1.2M** | V cos 0.998 |
| **KVTC K3V4-zlib** | **5.1x** | **1.2M** | V cos 0.996 |
| **KVTC K3V4-lzma** | **5.1x** | **1.2M** | V cos 0.996 |
| **KVTC K4V4-adapt-lzma** | **4.4x** | **1.0M** | V cos 0.998 |

---

*Benchmarked 2026-03-31 18:09 by [@OnlyTerp](https://x.com/OnlyTerp) / Terp AI Labs*
*KVTC paper: arXiv 2511.01815 (NVIDIA, ICLR 2026)*
*Open source: github.com/OnlyTerp/kvtc*
