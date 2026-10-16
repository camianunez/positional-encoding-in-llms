"""Adjacent-pair RoPE with explicit positions and a fixed spectrum."""
import math
import torch

def frequencies(dim, base=10000.0, *, single=False, device=None, dtype=torch.float32):
    if dim <= 0 or dim % 2:
        raise ValueError("rotary dimension must be positive and even")
    if not math.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than one")
    if single:
        return torch.ones(dim // 2, device=device, dtype=dtype)
    return base ** (-torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)

def rotate(x, positions, base=10000.0, *, scale=1.0, single=False):
    """Rotate x[..., sequence, head_dim]; positions has shape [sequence].

    Phases use at least float32 before casting the result to the input dtype.
    A fixed scale > 1 divides positions by scale (position interpolation).
    """
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be positive and finite")
    if positions.ndim != 1 or positions.shape[0] != x.shape[-2]:
        raise ValueError("positions must have one entry for each sequence token")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    theta = frequencies(x.shape[-1], base, single=single, device=x.device, dtype=dtype)
    angle = positions.to(device=x.device, dtype=dtype)[:, None] * theta / scale
    c, s = angle.cos(), angle.sin()
    a, b = x.to(dtype)[..., 0::2], x.to(dtype)[..., 1::2]
    return torch.stack((a*c-b*s, a*s+b*c), dim=-1).flatten(-2).to(x.dtype)

def sinusoidal(positions, dim):
    angle = positions.float()[:, None] * frequencies(dim, device=positions.device)
    return torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)

def alibi_slopes(heads, *, device=None):
    """Original power-of-two slope schedule, sufficient for this lab."""
    if heads <= 0 or heads & (heads - 1):
        raise ValueError("reference schedule requires a power-of-two head count")
    return 2.0 ** (-8.0 * torch.arange(1, heads+1, device=device) / heads)

def causal_bias(qpos, kpos, heads, *, alibi=False, device=None):
    lag = qpos[:, None] - kpos[None, :]
    bias = torch.zeros(heads, len(qpos), len(kpos), device=device)
    if alibi:
        bias = bias - alibi_slopes(heads, device=device)[:, None, None] * lag
    return bias.masked_fill(lag[None] < 0, float("-inf"))
