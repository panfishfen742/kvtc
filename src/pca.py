"""PCA and RoPE helpers for KVTC."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import torch

from .common import CalibrationData, PCAEntry


def _validate_head_dim(head_dim: int) -> None:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")


def _rope_angles(
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_head_dim(head_dim)
    base = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (base / head_dim))
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(
    keys: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
    head_dim: int | None = None,
) -> torch.Tensor:
    """Apply RoPE to keys along the final dimension."""

    dim = head_dim or keys.shape[-1]
    cos, sin = _rope_angles(positions, dim, rope_theta, keys.device, keys.dtype)
    even = keys[..., ::2]
    odd = keys[..., 1::2]
    while cos.dim() < even.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    output = torch.empty_like(keys)
    output[..., ::2] = rotated_even
    output[..., 1::2] = rotated_odd
    return output


def apply_rope_inverse(
    keys: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
    head_dim: int | None = None,
) -> torch.Tensor:
    """Undo RoPE from keys along the final dimension."""

    dim = head_dim or keys.shape[-1]
    cos, sin = _rope_angles(positions, dim, rope_theta, keys.device, keys.dtype)
    even = keys[..., ::2]
    odd = keys[..., 1::2]
    while cos.dim() < even.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated_even = even * cos + odd * sin
    rotated_odd = -even * sin + odd * cos
    output = torch.empty_like(keys)
    output[..., ::2] = rotated_even
    output[..., 1::2] = rotated_odd
    return output


def pca_transform(kv_tensor: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
    """Project vectors into PCA space."""

    return kv_tensor @ eigenvectors


def pca_inverse(pca_tensor: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
    """Reconstruct vectors from PCA space."""

    return pca_tensor @ eigenvectors.transpose(-2, -1)


class PCACalibrator:
    """Collect KV samples and compute PCA bases."""

    def __init__(self, head_group_size: int = 1, rope_theta: float = 10000.0) -> None:
        self.head_group_size = head_group_size
        self.rope_theta = rope_theta
        self._samples: DefaultDict[Tuple[int, int, str], List[torch.Tensor]] = defaultdict(list)

    def collect(
        self,
        layer_idx: int,
        kind: str,
        tensor: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> None:
        """Collect a `[tokens, heads, dim]` tensor for calibration."""

        if tensor.dim() != 3:
            raise ValueError("Expected tensor shape [tokens, heads, dim].")
        tokens, heads, _ = tensor.shape
        for group_idx, start in enumerate(range(0, heads, self.head_group_size)):
            chunk = tensor[:, start : start + self.head_group_size, :]
            if kind == "keys" and positions is not None:
                chunk = apply_rope_inverse(chunk, positions, rope_theta=self.rope_theta, head_dim=chunk.shape[-1])
            flattened = chunk.reshape(tokens * chunk.shape[1], chunk.shape[-1]).detach().cpu()
            self._samples[(layer_idx, group_idx, kind)].append(flattened)

    def compute(self, bit_budget_ratio: float = 0.25) -> CalibrationData:
        """Compute PCA bases and bit budgets for all collected groups."""

        entries: Dict[Tuple[int, int, str], PCAEntry] = {}
        for key, sample_list in self._samples.items():
            matrix = torch.cat(sample_list, dim=0).to(torch.float32)
            mean = matrix.mean(dim=0)
            centered = matrix - mean
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            eigenvalues = (singular_values.square() / max(centered.shape[0] - 1, 1)).to(torch.float32)
            bit_budget = max(1, int(matrix.shape[-1] * 16 * bit_budget_ratio))
            _, group_idx, kind = key
            head_start = group_idx * self.head_group_size
            entries[key] = PCAEntry(
                eigenvectors=vh.transpose(0, 1).contiguous(),
                eigenvalues=eigenvalues.contiguous(),
                mean=mean.contiguous(),
                head_indices=list(range(head_start, head_start + self.head_group_size)),
                kind=kind,
                bit_budget=bit_budget,
            )
        return CalibrationData(entries=entries, head_group_size=self.head_group_size, rope_theta=self.rope_theta)
