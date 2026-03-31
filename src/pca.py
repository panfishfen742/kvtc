"""PCA and RoPE utilities for KVTC."""

from __future__ import annotations

import torch
import math


def pca_transform(centered: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
    """Project centered data into PCA space.
    
    Args:
        centered: [num_rows, dim] centered data (mean already subtracted)
        eigenvectors: [dim, dim] or [k, dim] PCA basis (rows are eigenvectors)
    
    Returns:
        [num_rows, k] PCA coefficients
    """
    # eigenvectors from SVD: Vh has shape [k, dim], each row is an eigenvector
    # projection = centered @ Vh.T
    return centered @ eigenvectors.T


def pca_inverse(pca_values: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
    """Reconstruct from PCA space back to original space.
    
    Args:
        pca_values: [num_rows, k] PCA coefficients
        eigenvectors: [k, dim] PCA basis
    
    Returns:
        [num_rows, dim] reconstructed data (without mean)
    """
    return pca_values @ eigenvectors


def _rotary_embedding(positions: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Compute rotary position embeddings (cos, sin).
    
    Args:
        positions: [seq_len] position indices
        dim: head dimension
        theta: RoPE base frequency
    
    Returns:
        (cos, sin) each of shape [seq_len, dim//2]
    """
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=positions.device) / half_dim))
    # [seq_len, half_dim]
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.
    
    Args:
        x: [seq_len, heads, dim] or [seq_len, dim]
        cos: [seq_len, dim//2]
        sin: [seq_len, dim//2]
    
    Returns:
        Tensor with same shape as x, with RoPE applied
    """
    if x.dim() == 3:
        # [seq_len, heads, dim]
        seq_len, heads, dim = x.shape
        half_dim = dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        # Broadcast cos/sin: [seq_len, 1, half_dim]
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)
        out1 = x1 * cos_b - x2 * sin_b
        out2 = x2 * cos_b + x1 * sin_b
        return torch.cat([out1, out2], dim=-1)
    elif x.dim() == 2:
        # [seq_len, dim]
        dim = x.shape[-1]
        half_dim = dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2], dim=-1)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
    head_dim: int = 128,
) -> torch.Tensor:
    """Apply RoPE rotation to key/value tensor.
    
    Args:
        x: [seq_len, heads, dim] tensor
        positions: [seq_len] position indices
        rope_theta: RoPE base frequency
        head_dim: dimension of each head
    
    Returns:
        Tensor with RoPE applied
    """
    cos, sin = _rotary_embedding(positions, head_dim, rope_theta)
    cos = cos.to(x.dtype).to(x.device)
    sin = sin.to(x.dtype).to(x.device)
    return _apply_rotary_emb(x, cos, sin)


def apply_rope_inverse(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
    head_dim: int = 128,
) -> torch.Tensor:
    """Undo RoPE rotation (apply inverse rotation).
    
    RoPE inverse is just applying with negated sin (rotation in opposite direction).
    
    Args:
        x: [seq_len, heads, dim] tensor with RoPE already applied
        positions: [seq_len] position indices
        rope_theta: RoPE base frequency
        head_dim: dimension of each head
    
    Returns:
        Tensor with RoPE undone
    """
    cos, sin = _rotary_embedding(positions, head_dim, rope_theta)
    cos = cos.to(x.dtype).to(x.device)
    sin = sin.to(x.dtype).to(x.device)
    # Inverse rotation: negate sin
    return _apply_rotary_emb(x, cos, -sin)
