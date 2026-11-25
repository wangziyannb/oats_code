"""AWQ-style fake quantization of S + L2(L1(x)), in original coordinates.

PyTorch stores weights transposed relative to the row-vector notation in the
experiment: L1.weight is B.T and L2.weight is A.T. All returned weights already
include inverse channel scales. Never apply the diagnostic scales at inference.

The asymmetric group quantizer and activation-only scale grid follow llm-awq
d6e797a42b9ef7778de8ee2352116e0f48a78d61, quantizer.py / auto_scale.py (MIT).
Pair-output scoring, feedback, and sparse-mask preservation are custom changes.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AWQConfig:
    bits: int = 4
    group_size: int = 128
    grid: int = 20
    row_chunk: int = 512
    sparse_refresh: int = 10

    def __post_init__(self):
        if not 2 <= self.bits <= 8:
            raise ValueError("bits must be between 2 and 8")
        if min(self.group_size, self.grid, self.row_chunk, self.sparse_refresh) < 1:
            raise ValueError("AWQ sizes and refresh intervals must be positive")


def effective_group_size(width, requested):
    if width <= 0 or requested <= 0:
        raise ValueError("Input width and group size must be positive")
    group = min(width, requested)
    if width % group:
        raise ValueError(f"Input width {width} is not divisible by group size {group}")
    return group


@torch.no_grad()
def fake_quant(weight, config=AWQConfig(), preserve_mask=False):
    """Official asymmetric AWQ formula in FP32, with optional mask projection.

For sparse weights only, nonzero values rounded to zero are projected to the
nearest nonzero quantization level. This preserves exactly the selected mask.
Low-rank factors use the unmodified asymmetric formula.
"""
    validate_tensor(weight, "weight", ndim=2)
    group = effective_group_size(weight.shape[1], config.group_size)
    qmax = 2 ** config.bits - 1
    result = torch.empty_like(weight)
    # A rank-32/128 second factor is narrow enough to quantize in one batch.
    # Chunking it by 512 rows would mostly pay Python/CUDA launch overhead.
    # Group reductions and arithmetic are identical; wide weights stay chunked.
    row_chunk = weight.shape[0] if weight.shape[1] <= config.group_size else config.row_chunk
    for start in range(0, weight.shape[0], row_chunk):
        source = weight[start:start + row_chunk]
        w = source.float().reshape(-1, group)
        lo, hi = w.amin(1, keepdim=True), w.amax(1, keepdim=True)
        step = (hi - lo).clamp_min(1e-5) / qmax
        zp = (-torch.round(lo / step)).clamp(0, qmax)
        codes = (torch.round(w / step) + zp).clamp(0, qmax)
        q = (codes - zp) * step
        if preserve_mask:
            q = torch.where((w != 0) & (q == 0), torch.where(w > 0, step, -step), q)
            q = torch.where(w == 0, 0.0, q)
        result[start:start + row_chunk] = q.reshape_as(source).to(weight.dtype)
    return result


def scale_candidates(activation, grid):
    validate_tensor(activation, "activation")
    if grid < 1:
        raise ValueError("Scale grid must be positive")
    # Use FP32 means even when calibration activations are FP16.
    stat = activation.reshape(-1, activation.shape[-1]).float().abs().mean(0)
    for index in range(grid):
        ratio = index / grid
        scale = stat.pow(ratio).clamp_min(1e-4)
        yield ratio, scale / (scale.max() * scale.min()).sqrt()


@torch.no_grad()
def quantize_linear(weight, scale, config, preserve_mask=False):
    scaled = weight.float() * scale.float().view(1, -1)
    return (fake_quant(scaled, config, preserve_mask) / scale.view(1, -1)).to(weight.dtype)


def pair_output(x, first, second):
    return F.linear(F.linear(x, first), second)


def output_mse(output, target):
    return (output.float() - target.float()).square().mean()


@torch.no_grad()
def quantize_pair_at_scales(first, second, input_scale, rank_scale, config):
    """Apply D and H equivalences and return effective original-coordinate weights.

    Scaled coordinates: x/D -> Q(H^-1 L1 D) -> Q(L2 H).
    Fold D and H back into each weight for the existing two-linear module:
      L1_eff = H Q(H^-1 L1 D) D^-1
      L2_eff = Q(L2 H) H^-1.
    Thus both branches continue to receive the original input x. There is no
    shared upstream activation mutation, and no scale is applied twice.
    """
    first_scaled = first.float() * input_scale[None, :] / rank_scale[:, None]
    second_scaled = second.float() * rank_scale[None, :]
    first_q = fake_quant(first_scaled, config) * rank_scale[:, None] / input_scale[None, :]
    second_q = fake_quant(second_scaled, config) / rank_scale[None, :]
    return first_q.to(first.dtype), second_q.to(second.dtype)


@torch.no_grad()
def search_pair(first, second, x, target, config=AWQConfig()):
    """Coordinate scale search scored ONLY on ||target - L2q(L1q(x))||².

    Start fresh after every refit. First-factor candidates see a quantized
    second factor. Then recompute Zq from the chosen first factor, and use its
    statistics (never X or the FP factor output) for rank-channel candidates.
    Every candidate is scored with the actual two quantized factors.
    """
    h = torch.ones(first.shape[0], device=first.device, dtype=torch.float32)
    candidates = list(scale_candidates(x, config.grid))
    losses = []
    for ratio, d in candidates:
        q1, q2 = quantize_pair_at_scales(first, second, d, h, config)
        losses.append(output_mse(pair_output(x, q1, q2), target))
    # Synchronize once per grid, rather than once for every candidate. Rebuild
    # only the winner; retaining all candidate weights would waste GPU memory.
    winner = int(torch.stack(losses).argmin())
    input_ratio, d = candidates[winner]
    best_loss = float(losses[winner])
    q1, q2 = quantize_pair_at_scales(first, second, d, h, config)
    zq = F.linear(x, q1)
    z_stat = zq.float().abs().mean(0)
    # The identity candidate is included; the rank search cannot worsen the
    # selected pair's objective (subject only to identical floating operations).
    rank_ratio, best_h = 0.0, h
    first_search_loss = best_loss
    candidates = list(scale_candidates(zq, config.grid))
    losses = []
    for ratio, candidate_h in candidates:
        candidate1, candidate2 = quantize_pair_at_scales(first, second, d, candidate_h, config)
        losses.append(output_mse(pair_output(x, candidate1, candidate2), target))
    winner = int(torch.stack(losses).argmin())
    loss = float(losses[winner])
    if loss < best_loss:
        best_loss = loss
        rank_ratio, best_h = candidates[winner]
        q1, q2 = quantize_pair_at_scales(first, second, d, best_h, config)
    info = {
        "input_ratio": input_ratio, "rank_ratio": rank_ratio,
        "first_search_mse": first_search_loss, "pair_mse": best_loss,
        "zq_abs_mean_min": float(z_stat.min()),
        "zq_abs_mean_max": float(z_stat.max()),
        "zq_abs_mean_avg": float(z_stat.mean()),
        "first_group_size": effective_group_size(first.shape[1], config.group_size),
        "second_group_size": effective_group_size(second.shape[1], config.group_size),
    }
    scales = {"first_input": d.detach().clone(), "rank": best_h.detach().clone()}
    return q1, q2, scales, info


@torch.no_grad()
def search_sparse(weight, x, target, config=AWQConfig()):
    candidates = list(scale_candidates(x, config.grid))
    losses = []
    for ratio, scale in candidates:
        q = quantize_linear(weight, scale, config, preserve_mask=True)
        losses.append(output_mse(F.linear(x, q), target))
    winner = int(torch.stack(losses).argmin())
    ratio, scale = candidates[winner]
    q = quantize_linear(weight, scale, config, preserve_mask=True)
    return q, scale, {"sparse_ratio": ratio, "sparse_search_mse": float(losses[winner])}


def validate_tensor(value, name, ndim=None):
    if not value.is_floating_point() or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty floating-point tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")


class InLoopQuantizer:
    """Quantization only. OATS owns SVD, masks, iteration order and termination."""

    def __init__(self, x, config=AWQConfig(), dtype=torch.float16):
        validate_tensor(x, "calibration inputs", ndim=2)
        self.x = x.to(dtype=dtype)
        self.config = config
        self.dtype = dtype
        self.sparse_scale = None
        self.info = {}

    @torch.no_grad()
    def pair(self, first, second, residual, iteration):
        target = F.linear(self.x.float(), residual.float())
        first_q, second_q, _, info = search_pair(
            first.to(self.dtype), second.to(self.dtype), self.x, target, self.config)
        validate_tensor(first_q, "quantized V", ndim=2)
        validate_tensor(second_q, "quantized U", ndim=2)
        self.info = {**info, "pair_scale_refreshed": True}
        return first_q.float(), second_q.float()

    @torch.no_grad()
    def sparse(self, sparse, residual, iteration):
        source_mask = sparse.eq(0)
        source = sparse.to(self.dtype)
        # Avoid silently losing an active element during the storage dtype cast.
        if not torch.equal(source_mask, source.eq(0)):
            raise ValueError("Sparse dtype conversion changes the mask")
        refresh = iteration % self.config.sparse_refresh == 0
        if refresh:
            target = F.linear(self.x.float(), residual.float())
            q, self.sparse_scale, info = search_sparse(source, self.x, target, self.config)
            self.info.update(info)
        else:
            q = quantize_linear(source, self.sparse_scale, self.config, preserve_mask=True)
        validate_tensor(q, "quantized sparse", ndim=2)
        if not torch.equal(source_mask, q.eq(0)):
            raise ValueError("Sparse quantization changes the mask")
        self.info["sparse_scale_refreshed"] = refresh
        return q.float()
