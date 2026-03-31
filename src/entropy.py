"""Entropy coding for KVTC — DEFLATE (zlib) based compression/decompression."""

from __future__ import annotations

from typing import List, Tuple

import zlib

import torch


def compress(packed_bytes: bytes, level: int = 6) -> Tuple[bytes, float]:
    """Compress packed bytes using zlib DEFLATE.
    
    Returns:
        (compressed_bytes, compression_ratio)
    """
    if not packed_bytes:
        return b"", 1.0
    
    compressed = zlib.compress(packed_bytes, level)
    ratio = len(packed_bytes) / max(len(compressed), 1)
    return compressed, ratio


def decompress(compressed_bytes: bytes, original_size: int) -> bytes:
    """Decompress zlib-compressed bytes.
    
    Args:
        compressed_bytes: The zlib-compressed data
        original_size: Expected size of decompressed data (for validation)
    
    Returns:
        Decompressed bytes
    """
    if not compressed_bytes:
        return b""
    
    decompressed = zlib.decompress(compressed_bytes)
    if len(decompressed) != original_size:
        raise ValueError(
            f"Decompressed size {len(decompressed)} != expected {original_size}"
        )
    return decompressed


def pack_bits(
    indices_list: List[torch.Tensor],
    bit_widths: List[int],
) -> bytes:
    """Pack a list of quantized index tensors into a byte stream.
    
    Each tensor in indices_list corresponds to one PCA component.
    Components are packed sequentially: all rows of comp 0, then comp 1, etc.
    
    Args:
        indices_list: List of [num_rows] int64 tensors, one per component
        bit_widths: Bit width for each component
    
    Returns:
        Packed byte stream
    """
    accumulator = 0
    bits_in_acc = 0
    output = bytearray()
    
    for comp_idx, (indices, width) in enumerate(zip(indices_list, bit_widths)):
        width = int(width)
        if width == 0:
            continue
        mask = (1 << width) - 1
        for val in indices.tolist():
            accumulator |= (int(val) & mask) << bits_in_acc
            bits_in_acc += width
            while bits_in_acc >= 8:
                output.append(accumulator & 0xFF)
                accumulator >>= 8
                bits_in_acc -= 8
    
    if bits_in_acc:
        output.append(accumulator & 0xFF)
    
    return bytes(output)


def unpack_bits(
    packed: bytes,
    bit_widths: List[int],
    lengths: List[int],
) -> List[torch.Tensor]:
    """Unpack a byte stream into a list of quantized index tensors.
    
    Args:
        packed: Packed byte stream
        bit_widths: Bit width for each component
        lengths: Number of values per component
    
    Returns:
        List of [num_rows] int64 tensors, one per component
    """
    data = list(packed)
    byte_idx = 0
    accumulator = 0
    bits_in_acc = 0
    
    result = []
    for comp_idx, (width, length) in enumerate(zip(bit_widths, lengths)):
        width = int(width)
        length = int(length)
        
        if width == 0:
            result.append(torch.zeros(length, dtype=torch.int64))
            continue
        
        mask = (1 << width) - 1
        vals = torch.empty(length, dtype=torch.int64)
        
        for i in range(length):
            while bits_in_acc < width:
                if byte_idx >= len(data):
                    raise ValueError("Insufficient packed data.")
                accumulator |= data[byte_idx] << bits_in_acc
                bits_in_acc += 8
                byte_idx += 1
            vals[i] = accumulator & mask
            accumulator >>= width
            bits_in_acc -= width
        
        result.append(vals)
    
    return result
