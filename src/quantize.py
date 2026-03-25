"""Quantization utilities for KVTC."""

from __future__ import annotations

from typing import List

import torch

from .common import QuantizationParams


def _component_groups(eigenvalues: torch.Tensor, group_size: int) -> List[torch.Tensor]:
    return [eigenvalues[i : i + group_size] for i in range(0, eigenvalues.numel(), group_size)]


def dp_bit_allocation(
    eigenvalues: torch.Tensor,
    bit_budget: int,
    max_bits: int = 16,
    group_size: int = 1,
) -> torch.Tensor:
    """Allocate per-component bit widths with dynamic programming."""

    values = eigenvalues.detach().to(torch.float64).flatten()
    groups = _component_groups(values, group_size)
    max_budget = max(int(bit_budget), 0)
    num_groups = len(groups)
    inf = float("inf")
    dp = [[inf] * (max_budget + 1) for _ in range(num_groups + 1)]
    back = [[0] * (max_budget + 1) for _ in range(num_groups + 1)]
    dp[0][0] = 0.0
    for idx, group in enumerate(groups, start=1):
        variance = float(group.sum().item())
        width = group.numel()
        for used in range(max_budget + 1):
            if dp[idx - 1][used] == inf:
                continue
            for bits in range(0, max_bits + 1):
                cost = used + bits * width
                if cost > max_budget:
                    continue
                error = variance if bits == 0 else variance / (4 ** bits)
                candidate = dp[idx - 1][used] + error
                if candidate < dp[idx][cost]:
                    dp[idx][cost] = candidate
                    back[idx][cost] = bits
    best_budget = min(range(max_budget + 1), key=lambda budget: dp[num_groups][budget])
    widths = torch.zeros(values.numel(), dtype=torch.int64)
    remaining = best_budget
    for idx in range(num_groups, 0, -1):
        bits = back[idx][remaining]
        group = groups[idx - 1]
        start = (idx - 1) * group_size
        widths[start : start + group.numel()] = bits
        remaining -= bits * group.numel()
    return widths


def uniform_quantize(values: torch.Tensor, n_bits: int, scale: float, zero_point: float) -> torch.Tensor:
    """Quantize values with uniform affine quantization."""

    if n_bits == 0:
        return torch.zeros_like(values, dtype=torch.int64)
    qmax = (1 << n_bits) - 1
    indices = torch.round(values / scale + zero_point)
    return indices.clamp(0, qmax).to(torch.int64)


def uniform_dequantize(indices: torch.Tensor, n_bits: int, scale: float, zero_point: float) -> torch.Tensor:
    """Dequantize integer indices back to floating-point values."""

    if n_bits == 0:
        return torch.zeros_like(indices, dtype=torch.float32)
    return (indices.to(torch.float32) - zero_point) * scale


def compute_quant_params(pca_values: torch.Tensor, bit_widths: torch.Tensor) -> QuantizationParams:
    """Compute affine quantization parameters per PCA component."""

    if pca_values.dim() != 2:
        raise ValueError("Expected pca_values shape [num_rows, components].")
    mins = pca_values.min(dim=0).values.to(torch.float32)
    maxs = pca_values.max(dim=0).values.to(torch.float32)
    scales = torch.ones_like(mins)
    zero_points = torch.zeros_like(mins)
    for idx, bits in enumerate(bit_widths.tolist()):
        if bits == 0:
            mins[idx] = 0.0
            scales[idx] = 1.0
            zero_points[idx] = 0.0
            continue
        qmax = float((1 << bits) - 1)
        span = max(float(maxs[idx] - mins[idx]), 1e-8)
        scales[idx] = span / qmax
        zero_points[idx] = -mins[idx] / scales[idx]
    return QuantizationParams(
        bit_widths=bit_widths.to(torch.int64),
        scales=scales,
        zero_points=zero_points,
        mins=mins,
    )
