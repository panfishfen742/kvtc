"""Full KVTC compression pipeline."""

from __future__ import annotations

from typing import Dict, List

import torch

from .common import CalibrationData, CompressedKVCache, CompressedSection, CompressionMetadata
from .entropy import compress as entropy_compress
from .entropy import decompress as entropy_decompress
from .entropy import pack_bits, unpack_bits
from .pca import apply_rope, apply_rope_inverse, pca_inverse, pca_transform
from .quantize import compute_quant_params, dp_bit_allocation, uniform_dequantize, uniform_quantize


class KVTCCompressor:
    """Compress and decompress KV caches with KVTC."""

    def __init__(self, calibration_data: CalibrationData) -> None:
        self.calibration_data = calibration_data

    def compress(self, kv_cache: Dict[str, torch.Tensor], positions: torch.Tensor) -> CompressedKVCache:
        """Run the full KVTC pipeline."""

        keys = kv_cache["keys"]
        values = kv_cache["values"]
        if keys.shape != values.shape:
            raise ValueError("Keys and values must have the same shape.")
        if keys.dim() != 4:
            raise ValueError("Expected KV tensors with shape [layers, tokens, heads, dim].")
        layers, tokens, heads, dim = keys.shape
        sink_len = min(self.calibration_data.sink_tokens, tokens)
        residual = max(tokens - sink_len, 0)
        window_len = min(self.calibration_data.window_tokens, residual)
        middle_len = max(tokens - sink_len - window_len, 0)
        middle_positions = positions[sink_len : sink_len + middle_len]
        sinks = {
            "keys": keys[:, :sink_len].clone(),
            "values": values[:, :sink_len].clone(),
        }
        window = {
            "keys": keys[:, tokens - window_len :].clone() if window_len else keys[:, 0:0].clone(),
            "values": values[:, tokens - window_len :].clone() if window_len else values[:, 0:0].clone(),
        }
        compressed_sections: List[CompressedSection] = []
        total_original_bytes = 0
        total_compressed_bytes = 0
        if middle_len > 0:
            for layer_idx in range(layers):
                for kind, tensor in (("keys", keys), ("values", values)):
                    middle = tensor[layer_idx, sink_len : sink_len + middle_len]
                    for group_idx, start in enumerate(range(0, heads, self.calibration_data.head_group_size)):
                        entry = self.calibration_data.entries[(layer_idx, group_idx, kind)]
                        group = middle[:, start : start + self.calibration_data.head_group_size, :]
                        group_heads = group.shape[1]
                        work = group
                        if kind == "keys":
                            work = apply_rope_inverse(
                                work,
                                middle_positions,
                                rope_theta=self.calibration_data.rope_theta,
                                head_dim=dim,
                            )
                        rows = work.reshape(middle_len * group_heads, dim).to(torch.float32)
                        centered = rows - entry.mean.to(rows.device)
                        pca_values = pca_transform(centered, entry.eigenvectors.to(rows.device))
                        bit_widths = dp_bit_allocation(entry.eigenvalues.to(rows.device), entry.bit_budget, group_size=1)
                        params = compute_quant_params(pca_values, bit_widths)
                        indices_list = [
                            uniform_quantize(
                                pca_values[:, component],
                                int(bit_widths[component].item()),
                                float(params.scales[component].item()),
                                float(params.zero_points[component].item()),
                            )
                            for component in range(dim)
                        ]
                        packed = pack_bits(indices_list, bit_widths.tolist())
                        compressed_bytes, _ = entropy_compress(packed)
                        total_original_bytes += rows.numel() * rows.element_size()
                        total_compressed_bytes += len(compressed_bytes)
                        compressed_sections.append(
                            CompressedSection(
                                layer_idx=layer_idx,
                                group_idx=group_idx,
                                kind=kind,
                                compressed_bytes=compressed_bytes,
                                packed_size=len(packed),
                                lengths=[rows.shape[0]] * dim,
                                num_rows=rows.shape[0],
                                group_heads=group_heads,
                                bit_widths=bit_widths.tolist(),
                                scales=params.scales.tolist(),
                                zero_points=params.zero_points.tolist(),
                                mins=params.mins.tolist(),
                            )
                        )
        ratio = total_original_bytes / max(total_compressed_bytes, 1) if total_original_bytes else 1.0
        metadata = CompressionMetadata(
            positions_middle=middle_positions.tolist(),
            original_shape=(layers, tokens, heads, dim),
            sink_len=sink_len,
            middle_len=middle_len,
            window_len=window_len,
            compression_ratio=ratio,
        )
        return CompressedKVCache(sinks=sinks, window=window, compressed_sections=compressed_sections, metadata=metadata)

    def decompress(self, compressed_cache: CompressedKVCache) -> Dict[str, torch.Tensor]:
        """Reverse the full KVTC pipeline."""

        if compressed_cache.metadata is None:
            raise ValueError("Missing metadata.")
        layers, tokens, heads, dim = compressed_cache.metadata.original_shape
        sink_len = compressed_cache.metadata.sink_len
        middle_len = compressed_cache.metadata.middle_len
        window_len = compressed_cache.metadata.window_len
        result = {
            "keys": torch.zeros((layers, tokens, heads, dim), dtype=compressed_cache.sinks["keys"].dtype),
            "values": torch.zeros((layers, tokens, heads, dim), dtype=compressed_cache.sinks["values"].dtype),
        }
        if sink_len:
            result["keys"][:, :sink_len] = compressed_cache.sinks["keys"]
            result["values"][:, :sink_len] = compressed_cache.sinks["values"]
        if window_len:
            result["keys"][:, tokens - window_len :] = compressed_cache.window["keys"]
            result["values"][:, tokens - window_len :] = compressed_cache.window["values"]
        positions_middle = torch.tensor(compressed_cache.metadata.positions_middle, dtype=torch.long)
        for section in compressed_cache.compressed_sections:
            entry = self.calibration_data.entries[(section.layer_idx, section.group_idx, section.kind)]
            packed = entropy_decompress(section.compressed_bytes, section.packed_size)
            unpacked = unpack_bits(packed, section.bit_widths, section.lengths)
            dequantized = torch.stack(
                [
                    uniform_dequantize(
                        component,
                        int(section.bit_widths[idx]),
                        float(section.scales[idx]),
                        float(section.zero_points[idx]),
                    )
                    for idx, component in enumerate(unpacked)
                ],
                dim=-1,
            )
            restored = pca_inverse(dequantized, entry.eigenvectors.to(dequantized.device)) + entry.mean.to(dequantized.device)
            group = restored.reshape(middle_len, section.group_heads, dim)
            if section.kind == "keys":
                group = apply_rope(
                    group,
                    positions_middle,
                    rope_theta=self.calibration_data.rope_theta,
                    head_dim=dim,
                )
            start = section.group_idx * self.calibration_data.head_group_size
            result[section.kind][section.layer_idx, sink_len : sink_len + middle_len, start : start + section.group_heads] = group.to(
                result[section.kind].dtype
            )
        return result

