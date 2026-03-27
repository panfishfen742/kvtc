# TASK: vLLM Integration for KVTC

## Goal
Make KVTC work inside vLLM so that after prefill, the paged KV cache is freed and decode attention uses KVTC's compressed store. This gives 3× more concurrent users or 3× longer context on the same GPU.

## Reference Implementation
Study 0xSero/turboquant's vLLM integration (https://github.com/0xSero/turboquant):
- `turboquant/vllm_attn_backend.py` — monkey-patches vLLM's attention backend
- `turboquant/triton_kernels.py` — 3 fused Triton kernels for decode attention
- `turboquant/kv_cache.py` — manages compressed KV store alongside vLLM's paged cache
- They intercept `do_kv_cache_update` to capture K/V into compressed store
- After prefill, they replace paged cache tensors with 1-byte dummies and call `torch.cuda.empty_cache()`
- Decode uses fused Triton kernels that compute attention directly from compressed data

## KVTC Architecture (different from TurboQuant)
KVTC compresses differently:
1. Undo RoPE on keys (data-dependent, not random rotation)
2. PCA transform: project into decorrelated principal components
3. DP bit allocation: assign 0-16 bits per component based on eigenvalue
4. Uniform quantize each component
5. Bit-pack + DEFLATE entropy coding

For vLLM integration, we CANNOT use DEFLATE during serving (too slow to decompress per attention op).
Instead, we store the PCA-quantized indices directly (no entropy coding) and reconstruct on-the-fly.

## Implementation Plan

### File: src/vllm_backend.py (NEW)
KVTC vLLM attention backend. Key classes:

1. `KVTCLayerState` — per-layer state holding:
   - PCA calibration data (eigenvectors, eigenvalues, means per head group)
   - Quantized KV store: integer indices + quantization params (scales, zero_points, bit_widths)
   - Ring buffer for accumulating tokens before quantizing
   - Sink buffer (first N tokens in FP16)
   - Bit allocation (precomputed per head group from calibration)

2. `PatchedCacheUpdate` — intercepts vLLM's `do_kv_cache_update`:
   - During prefill: capture K/V, apply RoPE inverse on keys, PCA transform, quantize, store indices
   - Sinks (first 4 tokens) stored in FP16, never quantized
   - Normal paged cache write still happens (needed for prefill attention)

3. `PatchedForward` — intercepts vLLM's attention forward:
   - During prefill: use normal flash attention (paged cache exists)
   - During decode: reconstruct K/V from quantized store, compute attention
   - Reconstruction: dequantize indices → PCA inverse → reapply RoPE → standard attention

4. `free_kv_cache(model)` — after prefill completes:
   - Replace paged KV cache tensors with 1-byte dummies per layer
   - Call torch.cuda.empty_cache()
   - Set mode to "active" (use KVTC for decode)

5. `hook_model(model, calibration_data)` — entry point:
   - Walk model layers, find attention modules
   - Create KVTCLayerState per layer
   - Monkey-patch do_kv_cache_update and forward
   - Return handle for free_kv_cache

### File: src/vllm_triton.py (NEW)
Fused Triton decode attention kernel for KVTC:

Unlike TurboQuant which computes scores from packed indices + codebook lookup,
KVTC needs to:
1. Dequantize PCA indices → PCA coordinates (uniform dequant, vectorized)
2. PCA inverse transform: multiply by eigenvectors^T (matrix multiply)
3. Reapply RoPE (element-wise sin/cos)
4. Compute attention score: dot product with query

Fused kernel: for each KV token in a block:
- Load quantized indices (variable width, but precomputed per component)
- Dequantize: (idx - zero_point) * scale (element-wise)
- PCA inverse: multiply by eigenvectors^T row (loaded into shared memory)
- RoPE: apply sin/cos rotation (for keys only)
- Dot with query → attention score
- Online softmax accumulation (flash-attention style)

Value reconstruction is simpler (no RoPE), same dequant + PCA inverse.

### File: src/calibrate_vllm.py (NEW)
Calibration utilities that work with vLLM models:
- Hook into a running vLLM model during warmup
- Collect KV cache samples from initial requests
- Compute PCA bases and save to disk
- Load calibration data at startup

### File: proof.py (NEW)
A/B benchmark (like 0xSero's proof.py):
- Run vLLM baseline in one process, measure VRAM + output
- Run vLLM + KVTC in another process, measure VRAM + output
- Compare: same output? How much VRAM freed? Context capacity?

## Key Differences from TurboQuant's Approach
1. **PCA is data-dependent** — needs calibration pass (TurboQuant uses random rotation, no calibration)
2. **Variable bit widths per component** — DP allocates 0-16 bits optimally (TurboQuant uses fixed 3-bit keys, 2-bit values)
3. **No codebook** — KVTC uses uniform quantization, not Lloyd-Max codebook
4. **RoPE handling** — must undo before PCA on keys, reapply after reconstruction
5. **Higher quality** — 0.998 cosine vs TurboQuant's ~0.95 at similar compression

## Constraints
- Must work with vLLM 0.17.0+ (same version 0xSero used)
- Triton kernels required for decode performance
- Calibration data loaded from disk at startup (computed offline)
- Must support GQA (grouped query attention) — most modern models use this
- Sink tokens (first 4) and recent tokens (sliding window) kept in FP16

## Existing Code to Use
- `src/gpu_ops.py` — greedy_bit_allocation, batch_quantize, batch_dequantize, vectorized_quant_params
- `src/pca.py` — pca_transform, pca_inverse, apply_rope, apply_rope_inverse, PCACalibrator
- `src/pipeline_fast.py` — KVTCCompressorFast (reference for the full pipeline)
- `src/common.py` — CalibrationData, PCAEntry dataclasses

## Success Criteria
- [ ] vLLM serves requests with KVTC enabled (no crashes)
- [ ] VRAM freed after prefill (measurable with nvidia-smi)
- [ ] Output quality matches baseline (cosine > 0.99 on logits)
- [ ] Decode latency within 2× of baseline flash attention
- [ ] Works with at least one model (Qwen 2.5-3B or Mistral-7B)

## Testing Model
Use Qwen/Qwen2.5-3B-Instruct — small enough to fit on any GPU, proven to work with our KVTC pipeline (0.999 cosine).
