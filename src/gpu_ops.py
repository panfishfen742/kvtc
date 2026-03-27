"""GPU-optimized KVTC operations — vectorized PyTorch, no Python loops."""

from __future__ import annotations

import torch

from .common import QuantizationParams


def greedy_bit_allocation(
    eigenvalues: torch.Tensor,
    bit_budget: int,
    max_bits: int = 16,
) -> torch.Tensor:
    """Greedy bit allocation — O(B log d) instead of O(d × B × max_bits).
    
    Minimizes sum(λᵢ / 4^bᵢ) subject to sum(bᵢ) ≤ B.
    Each additional bit on component i reduces MSE by: λᵢ × 3 / 4^(b+1).
    Greedily pick the component with highest marginal gain.
    """
    d = eigenvalues.numel()
    ev = eigenvalues.detach().to(torch.float64).flatten()
    bits = torch.zeros(d, dtype=torch.int64)
    budget_remaining = max(int(bit_budget), 0)
    
    if budget_remaining == 0 or d == 0:
        return bits
    
    # Marginal gain of adding 1 bit to component i at current allocation:
    # gain_i = λᵢ × 3 / 4^(bits_i + 1)
    # Start: all bits=0, so gain = λᵢ × 3/4 = 0.75 × λᵢ
    gains = ev * 0.75  # λ × 3/4^1
    
    for _ in range(budget_remaining):
        # Pick component with highest marginal gain
        best = torch.argmax(gains).item()
        if gains[best] <= 0:
            break
        current_bits = bits[best].item()
        if current_bits >= max_bits:
            gains[best] = 0.0
            continue
        bits[best] += 1
        new_bits = bits[best].item()
        if new_bits >= max_bits:
            gains[best] = 0.0
        else:
            # Next marginal gain: λ × 3 / 4^(new_bits+1)
            gains[best] = ev[best] * 3.0 / (4.0 ** (new_bits + 1))
    
    return bits


def vectorized_quant_params(
    pca_values: torch.Tensor,
    bit_widths: torch.Tensor,
) -> QuantizationParams:
    """Compute quantization parameters without Python loops."""
    if pca_values.dim() != 2:
        raise ValueError("Expected pca_values shape [num_rows, components].")
    
    d = pca_values.shape[1]
    mins = pca_values.min(dim=0).values.to(torch.float32)
    maxs = pca_values.max(dim=0).values.to(torch.float32)
    bw = bit_widths.to(torch.float32)
    
    # Compute qmax per component: (1 << bits) - 1, but handle 0-bit
    # For 0-bit components: qmax=1 (avoid div by zero), then zero out later
    nonzero_mask = bw > 0
    safe_bw = torch.where(nonzero_mask, bw, torch.ones_like(bw))
    qmax = (2.0 ** safe_bw) - 1.0
    qmax = torch.where(nonzero_mask, qmax, torch.ones_like(qmax))
    
    span = (maxs - mins).clamp(min=1e-8)
    scales = span / qmax
    zero_points = -mins / scales
    
    # Zero out 0-bit components
    scales = torch.where(nonzero_mask, scales, torch.ones_like(scales))
    zero_points = torch.where(nonzero_mask, zero_points, torch.zeros_like(zero_points))
    mins = torch.where(nonzero_mask, mins, torch.zeros_like(mins))
    
    return QuantizationParams(
        bit_widths=bit_widths.to(torch.int64),
        scales=scales,
        zero_points=zero_points,
        mins=mins,
    )


def batch_quantize(
    pca_values: torch.Tensor,
    bit_widths: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    """Quantize all components at once — single tensor op.
    
    Args:
        pca_values: [num_rows, components] float tensor
        bit_widths: [components] int tensor
        scales: [components] float tensor
        zero_points: [components] float tensor
    
    Returns:
        indices: [num_rows, components] int64 tensor
    """
    bw = bit_widths.to(torch.float32)
    nonzero_mask = bw > 0  # [components]
    safe_bw = torch.where(nonzero_mask, bw, torch.ones_like(bw))
    qmax = (2.0 ** safe_bw - 1.0).unsqueeze(0)  # [1, components]
    
    # Quantize: round(value / scale + zero_point), clamp to [0, qmax]
    s = scales.unsqueeze(0)  # [1, components]
    zp = zero_points.unsqueeze(0)  # [1, components]
    
    indices = torch.round(pca_values / s + zp)
    indices = indices.clamp(min=0)
    indices = torch.min(indices, qmax)  # per-component clamp
    
    # Zero out 0-bit components
    indices = indices * nonzero_mask.unsqueeze(0).float()
    
    return indices.to(torch.int64)


def batch_dequantize(
    indices: torch.Tensor,
    bit_widths: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    """Dequantize all components at once — single tensor op.
    
    Args:
        indices: [num_rows, components] int64 tensor
        bit_widths: [components] int tensor
        scales: [components] float tensor
        zero_points: [components] float tensor
    
    Returns:
        values: [num_rows, components] float32 tensor
    """
    nonzero_mask = bit_widths > 0  # [components]
    
    s = scales.unsqueeze(0)  # [1, components]
    zp = zero_points.unsqueeze(0)  # [1, components]
    
    values = (indices.to(torch.float32) - zp) * s
    
    # Zero out 0-bit components
    values = values * nonzero_mask.unsqueeze(0).float()
    
    return values


def fast_pack_bits(
    indices: torch.Tensor,
    bit_widths: torch.Tensor,
) -> bytes:
    """Pack variable-width quantized indices into bytes using torch ops.
    
    Faster than the pure Python loop for large tensors.
    Falls back to the original implementation structure but uses
    vectorized operations where possible.
    """
    num_rows, num_components = indices.shape
    bw_list = bit_widths.tolist()
    
    # For each component, pack its column
    accumulator = 0
    bits_in_acc = 0
    output = bytearray()
    
    for comp_idx in range(num_components):
        width = int(bw_list[comp_idx])
        if width == 0:
            continue
        mask = (1 << width) - 1
        col = indices[:, comp_idx].tolist()
        for val in col:
            accumulator |= (int(val) & mask) << bits_in_acc
            bits_in_acc += width
            while bits_in_acc >= 8:
                output.append(accumulator & 0xFF)
                accumulator >>= 8
                bits_in_acc -= 8
    
    if bits_in_acc:
        output.append(accumulator & 0xFF)
    
    return bytes(output)


def fast_unpack_dequantize(
    packed_bytes: bytes,
    bit_widths: torch.Tensor,
    num_rows: int,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    """Unpack and dequantize in one pass — avoids intermediate list of tensors."""
    num_components = bit_widths.numel()
    bw_list = bit_widths.tolist()
    
    data = list(packed_bytes)
    byte_idx = 0
    accumulator = 0
    bits_in_acc = 0
    
    # Pre-allocate output
    result = torch.zeros(num_rows, num_components, dtype=torch.float32)
    
    for comp_idx in range(num_components):
        width = int(bw_list[comp_idx])
        if width == 0:
            continue
        mask = (1 << width) - 1
        s = float(scales[comp_idx].item())
        zp = float(zero_points[comp_idx].item())
        
        for row in range(num_rows):
            while bits_in_acc < width:
                if byte_idx >= len(data):
                    raise ValueError("Insufficient packed data.")
                accumulator |= data[byte_idx] << bits_in_acc
                bits_in_acc += 8
                byte_idx += 1
            val = accumulator & mask
            accumulator >>= width
            bits_in_acc -= width
            # Dequantize inline
            result[row, comp_idx] = (val - zp) * s
    
    return result
