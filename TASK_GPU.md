# TASK: GPU-Accelerate KVTC Pipeline

## Goal
Make KVTC compression/decompression run entirely on GPU using PyTorch tensor ops and Triton kernels. Target: <500ms total for 1024 tokens on RTX 5090 (currently 8-10 seconds).

## Current Bottlenecks (in order of impact)

### 1. DP Bit Allocation (quantize.py `dp_bit_allocation`) — ~40% of time
- Pure Python nested loops: O(components × budget × max_bits)
- For dim=64, budget=358, max_bits=16: 64 × 358 × 17 = 389K iterations in Python
- **FIX**: Replace with closed-form greedy allocation. The DP is solving: minimize sum(λᵢ/4^bᵢ) subject to sum(bᵢ) ≤ B. This has a well-known greedy solution:
  - Sort components by λᵢ (eigenvalue) descending
  - Greedily assign bits to the component where adding 1 bit gives the biggest MSE reduction: Δ = λᵢ × (1/4^b - 1/4^(b+1)) = λᵢ × 3/(4^(b+1))
  - Use a priority queue or just vectorized torch ops
  - This is O(B × log(d)) instead of O(d × B × 16)

### 2. Per-Component Quantize Loop (pipeline.py compress) — ~25% of time
```python
indices_list = [
    uniform_quantize(pca_values[:, component], int(bit_widths[component].item()), ...)
    for component in range(dim)
]
```
- Iterates over each dimension in Python
- **FIX**: Batch quantize all components at once using vectorized torch ops:
  - scales and zero_points are already tensors
  - `indices = torch.round(pca_values / scales.unsqueeze(0) + zero_points.unsqueeze(0))`
  - Then clamp per-component using bit_widths: `qmax = (1 << bit_widths) - 1`
  - Single tensor op instead of dim=64 separate calls

### 3. compute_quant_params Loop (quantize.py) — ~5% of time
```python
for idx, bits in enumerate(bit_widths.tolist()):
```
- **FIX**: Vectorize entirely:
  - `qmax = (1 << bit_widths.float()) - 1` (handle 0-bit with mask)
  - `scales = (maxs - mins) / qmax.clamp(min=1)`
  - `zero_points = -mins / scales`

### 4. Bit Packing (entropy.py `pack_bits`) — ~15% of time
- Python loop over every value
- **FIX**: For fixed bit-widths per component, use torch bit-shift ops to pack. For variable widths, a Triton kernel.

### 5. PCA Transform Stays on GPU — ~5% of time
- Already a matrix multiply: `pca_values = centered @ eigenvectors`
- Currently forces .cpu() — keep on GPU through the pipeline
- Only move to CPU for bit packing (final step)

### 6. Decompression per-component loop (pipeline.py decompress) — ~10%
Same pattern as compress — vectorize the dequantize.

## Implementation Plan

### File: src/gpu_ops.py (NEW)
Contains all GPU-optimized operations:
- `greedy_bit_allocation(eigenvalues, bit_budget, max_bits=16)` — vectorized greedy
- `batch_quantize(pca_values, bit_widths, scales, zero_points)` — single tensor op
- `batch_dequantize(indices, bit_widths, scales, zero_points)` — single tensor op  
- `vectorized_quant_params(pca_values, bit_widths)` — no loops
- `fast_pack_bits(indices, bit_widths)` — torch-based bit packing (CPU fallback ok)

### File: src/pipeline.py (MODIFY)
- Import gpu_ops, use them instead of loop-based functions
- Keep tensors on GPU through PCA transform → quantize → only CPU for bit packing
- Add `device` parameter to KVTCCompressor

### File: src/quantize.py (KEEP as fallback)
- Don't delete — keep as CPU reference implementation

### File: src/benchmark_gpu.py (MODIFY)  
- Add per-stage timing breakdown
- Print speedup vs old pipeline

## Constraints
- Must pass all 38 existing tests (src/test_kvtc.py)
- Must produce identical compression ratios and cosine similarities (within float tolerance)
- Pure PyTorch + Triton — no custom CUDA C++ kernels
- Triton is optional (graceful fallback to torch ops if triton not available)
- Python 3.11, PyTorch 2.10, CUDA 12.8, RTX 5090

## Success Criteria
- [ ] Compression time < 500ms for 512 tokens on RTX 5090
- [ ] Same quality metrics (cosine sim within 0.001 of current)
- [ ] All 38 unit tests still pass
- [ ] Per-stage timing breakdown in benchmark output
