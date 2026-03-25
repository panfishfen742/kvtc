"""Cache integration for KVTC."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .common import CompressedKVCache
from .pipeline import KVTCCompressor

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover
    class DynamicCache:  # type: ignore[override]
        """Minimal fallback cache base class."""

        pass


class KVTCCache(DynamicCache):
    """Dynamic cache wrapper with transparent KVTC compression."""

    def __init__(self, compressor: KVTCCompressor, compression_ratio_target: float = 5.0) -> None:
        super().__init__()
        self.compressor = compressor
        self.compression_ratio_target = compression_ratio_target
        self.live_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.compressed_cache: Dict[int, CompressedKVCache] = {}

    def update(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Store a live layer cache."""

        self.live_cache[layer_idx] = {"keys": keys, "values": values}
        self.compressed_cache.pop(layer_idx, None)

    def evict_to_compressed(self, layer_idx: int, positions: torch.Tensor) -> CompressedKVCache:
        """Compress a live layer cache."""

        layer = self.live_cache.pop(layer_idx)
        kv_cache = {"keys": layer["keys"].unsqueeze(0), "values": layer["values"].unsqueeze(0)}
        compressed = self.compressor.compress(kv_cache, positions)
        self.compressed_cache[layer_idx] = compressed
        return compressed

    def restore_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Restore a compressed layer back into live form."""

        if layer_idx in self.live_cache:
            return self.live_cache[layer_idx]
        compressed = self.compressed_cache.pop(layer_idx)
        restored = self.compressor.decompress(compressed)
        live = {"keys": restored["keys"][0], "values": restored["values"][0]}
        self.live_cache[layer_idx] = live
        return live

    def is_compressed(self, layer_idx: int) -> bool:
        """Return whether a layer currently resides in compressed form."""

        return layer_idx in self.compressed_cache

    def get_layer(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Return the live layer if present."""

        return self.live_cache.get(layer_idx)
