"""Bit packing and entropy coding for KVTC."""

from __future__ import annotations

import zlib
from typing import Iterable, List

import torch


def pack_bits(indices_list: Iterable[torch.Tensor], bit_widths: Iterable[int]) -> bytes:
    """Pack variable-width integer arrays into a byte stream."""

    accumulator = 0
    bits_in_accumulator = 0
    output = bytearray()
    for indices, width in zip(indices_list, bit_widths):
        if width == 0:
            continue
        mask = (1 << width) - 1
        for value in indices.reshape(-1).tolist():
            accumulator |= (int(value) & mask) << bits_in_accumulator
            bits_in_accumulator += width
            while bits_in_accumulator >= 8:
                output.append(accumulator & 0xFF)
                accumulator >>= 8
                bits_in_accumulator -= 8
    if bits_in_accumulator:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_bits(packed_bytes: bytes, bit_widths: Iterable[int], lengths: Iterable[int]) -> List[torch.Tensor]:
    """Unpack variable-width integers from a byte stream."""

    data = list(packed_bytes)
    byte_idx = 0
    accumulator = 0
    bits_in_accumulator = 0
    output: List[torch.Tensor] = []
    for width, length in zip(bit_widths, lengths):
        if width == 0:
            output.append(torch.zeros(length, dtype=torch.int64))
            continue
        values = []
        mask = (1 << width) - 1
        for _ in range(length):
            while bits_in_accumulator < width:
                if byte_idx >= len(data):
                    raise ValueError("Insufficient packed data.")
                accumulator |= data[byte_idx] << bits_in_accumulator
                bits_in_accumulator += 8
                byte_idx += 1
            values.append(accumulator & mask)
            accumulator >>= width
            bits_in_accumulator -= width
        output.append(torch.tensor(values, dtype=torch.int64))
    return output


def compress(packed_bytes: bytes) -> tuple[bytes, float]:
    """Compress packed bytes with DEFLATE and report the compression ratio."""

    if not packed_bytes:
        return b"", 1.0
    compressed = zlib.compress(packed_bytes)
    return compressed, len(packed_bytes) / max(len(compressed), 1)


def decompress(compressed_bytes: bytes, original_size: int) -> bytes:
    """Restore DEFLATE-compressed bytes."""

    if not compressed_bytes:
        return b""
    restored = zlib.decompress(compressed_bytes)
    if len(restored) != original_size:
        raise ValueError("Decompressed size mismatch.")
    return restored
