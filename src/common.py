"""Shared data structures for KVTC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


CalibrationKey = Tuple[int, int, str]


@dataclass
class PCAEntry:
    """PCA statistics for one layer, head-group, and tensor kind."""

    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    mean: torch.Tensor
    head_indices: List[int]
    kind: str
    bit_budget: int
    pca_mins: torch.Tensor | None = None
    pca_maxs: torch.Tensor | None = None
    bit_widths: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    zero_points: torch.Tensor | None = None


@dataclass
class CalibrationData:
    """Serialized calibration artifact consumed by the compressor."""

    entries: Dict[CalibrationKey, PCAEntry]
    head_group_size: int
    rope_theta: float = 10000.0
    sink_tokens: int = 4
    window_tokens: int = 128


@dataclass
class QuantizationParams:
    """Uniform quantization parameters per PCA component."""

    bit_widths: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    mins: torch.Tensor


@dataclass
class CompressedSection:
    """Compressed payload for one layer/group/kind middle region."""

    layer_idx: int
    group_idx: int
    kind: str
    compressed_bytes: bytes
    packed_size: int
    lengths: List[int]
    num_rows: int
    group_heads: int
    bit_widths: List[int]
    scales: List[float]
    zero_points: List[float]
    mins: List[float]


@dataclass
class CompressionMetadata:
    """Metadata required for reconstruction."""

    positions_middle: List[int]
    original_shape: Tuple[int, int, int, int]
    sink_len: int
    middle_len: int
    window_len: int
    compression_ratio: float


@dataclass
class CompressedKVCache:
    """Container returned by the full KVTC pipeline."""

    sinks: Dict[str, torch.Tensor]
    window: Dict[str, torch.Tensor]
    compressed_sections: List[CompressedSection] = field(default_factory=list)
    metadata: CompressionMetadata | None = None
