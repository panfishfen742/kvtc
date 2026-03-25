# Implementation Notes

## Algorithm: Three-Stage Transform Coding

KVTC (arxiv 2511.01815) compresses KV caches using a pipeline inspired by classical media codecs (JPEG, H.264). The three stages are applied only to the "middle" tokens — attention sinks and the sliding window are preserved exactly.

### Stage 1: PCA Feature Decorrelation

**Why PCA?** Raw KV cache vectors have correlated dimensions — particularly within attention head groups. PCA decorrelates these dimensions and orders them by variance, enabling Stage 2 to allocate bits efficiently.

**RoPE handling (critical for keys):** Rotary Position Embeddings rotate key vectors based on their position in the sequence. This rotation obscures the low-rank structure that PCA needs. We **undo RoPE before PCA** and **reapply it after decompression**. Values don't use RoPE, so no special handling is needed.

The RoPE inverse is exact: for a rotation by angle θ, the inverse is a rotation by -θ. We verified this with roundtrip tests (error < 1e-5).

**Calibration:** PCA eigenvectors are computed once from a small calibration dataset (~10 texts through the model). The eigenvectors and eigenvalues are stored per `(layer, head_group, key_or_value)`. For a 7B model with 32 layers and 8 KV heads, this produces 32 × 8 × 2 = 512 PCA entries. Storage overhead is small relative to model size.

**SVD vs eigendecomposition:** We use `torch.linalg.svd` on the centered data matrix rather than eigendecomposition of the covariance matrix. SVD is numerically more stable and directly gives us the eigenvectors (right singular vectors) and eigenvalues (squared singular values / (n-1)).

### Stage 2: Adaptive Quantization via Dynamic Programming

**The DP objective:** Given eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λ_d (from PCA) and a total bit budget B, find bit widths b₁, ..., b_d that minimize total reconstruction error:

```
minimize  Σᵢ λᵢ / 4^bᵢ     (quantization error per component)
subject to  Σᵢ bᵢ ≤ B       (total bit budget)
            0 ≤ bᵢ ≤ 16     (per-component bit width)
```

The error model `λᵢ / 4^bᵢ` comes from uniform quantization theory: each additional bit halves the quantization step size, reducing MSE by 4× (= 2² in each direction).

**0-bit allocation = pruning:** When the DP assigns 0 bits to a component, that component is dropped entirely. This is the DP discovering that the reconstruction error from keeping the component (at any bit width) exceeds the error from ignoring it. High-variance components get more bits; low-variance trailing components get pruned.

**Complexity:** O(d × B × max_bits) where d = head_dim, B = bit budget, max_bits = 16. For d=128, B=32, max_bits=16 this is ~65K operations per PCA entry. Fast enough for a reference implementation.

### Stage 3: Entropy Coding (DEFLATE)

**Why entropy coding?** After quantization, many components share the same few values (especially 0-bit pruned components and low-bit components). This statistical redundancy is exploitable by lossless compression.

**Implementation:** We use Python's `zlib` (DEFLATE algorithm) rather than NVIDIA's `nvCOMP` GPU library. This keeps the implementation dependency-light and cross-platform. The paper uses nvCOMP for GPU-accelerated DEFLATE in production.

**Compression boost:** Entropy coding typically adds 1.2-1.5× additional compression beyond quantization alone. The exact boost depends on the data distribution of quantized values.

## Token Protection

**Attention sinks (first 4 tokens):** The paper's ablation studies show that compressing the initial tokens can collapse model accuracy at high compression ratios. These tokens receive disproportionate attention weight regardless of content.

**Sliding window (last 128 tokens):** Recent tokens carry the most relevant context for generation. Compressing them adds latency for minimal memory savings.

Both are stored in their original FP16/BF16 precision and concatenated with the decompressed middle region during restoration.

## Design Decisions

### Paper vs Implementation Choices

| Paper | Our Implementation | Reasoning |
|-------|-------------------|-----------|
| nvCOMP GPU DEFLATE | zlib CPU DEFLATE | Cross-platform, no CUDA dependency |
| Offline calibration server | `CalibrationData` with save/load | Self-contained, serializable |
| Layer-by-layer chunked decompression | Full batch decompression | Simpler for reference impl |
| Production inference integration | HuggingFace DynamicCache wrapper | Correctness over performance |

### What we DON'T implement (and why)
- **GPU-accelerated entropy coding:** Would require nvCOMP or custom CUDA kernels. The CPU zlib path is correct and sufficient for validation.
- **Pipelined decompression:** The paper describes decompressing layer-by-layer to overlap with attention computation. Our implementation decompresses all layers before returning.
- **Grouped head PCA across heads:** We compute PCA per individual head (head_group_size=1 default). The paper groups heads for efficiency; our approach maximizes per-head decorrelation quality.

## Testing

- `src/test_kvtc.py`: 38 unit tests covering PCA roundtrip, RoPE correctness, DP bit allocation, bit packing, entropy coding, full pipeline, edge cases
- `src/test_real_model.py`: Optional TinyLlama integration test (gated behind `RUN_REAL_MODEL_TEST=1`)
- `bench_mistral.py`: Mistral-7B-Instruct benchmark with multiple compression levels
