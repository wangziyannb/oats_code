"""Fixed-mask, diagonal-objective refinement of explicit quantization codes.

Independent research adapter: OATS supplies the factors and the 2:4 mask.
This module never computes an SVD, selects a new mask, or mutates its inputs.
All channel transforms are fixed; integer codes and positive group steps are
the only optimization variables. Zero points remain fixed in this version.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import time

import torch
import torch.nn.functional as F

from .quantize import AWQConfig, effective_group_size, search_pair, search_sparse, validate_tensor


@dataclass(frozen=True)
class RefineConfig:
    sweeps: int = 3
    update_scales: bool = True
    pair_candidates: int = 16
    min_step: float = 1e-12
    storage_dtype: str = 'float16'

    def __post_init__(self):
        if self.sweeps < 0 or self.pair_candidates < 1:
            raise ValueError('Invalid sweep/candidate count')
        if not math.isfinite(self.min_step) or self.min_step <= 0:
            raise ValueError('min_step must be finite and positive')
        if self.storage_dtype not in ('float16', 'bfloat16', 'float32'):
            raise ValueError('Unsupported storage dtype')


def nearest_code(target, spacing, zero, bits=4, active=None, old=None):
    """Nearest LEGAL affine code; sparse active positions cannot use zero.

    Broadcasts over coordinates and candidates. Inactive sparse positions are
    exactly zero. Ties retain the old code when supplied.
    """
    limit = 2 ** bits - 1
    z = zero.to(target.dtype)
    code = (target / spacing + z).round().clamp(0, limit)
    if active is not None:
        below, above = (z - 1).clamp(0, limit), (z + 1).clamp(0, limit)
        dl = (target - (below - z) * spacing).square().masked_fill(below == z, float('inf'))
        dr = (target - (above - z) * spacing).square().masked_fill(above == z, float('inf'))
        replacement = torch.where(dl <= dr, below, above)
        code = torch.where(active & (code == z), replacement, code)
        code = torch.where(active, code, z)
    if old is not None:
        old = old.to(target.dtype)
        better = (target - (code - z) * spacing).square() < (target - (old - z) * spacing).square()
        code = torch.where(better, code, old)
    return code.to(torch.uint8)


@dataclass
class CodeMatrix:
    codes: torch.Tensor
    steps: torch.Tensor
    zeros: torch.Tensor
    row_gain: torch.Tensor
    col_gain: torch.Tensor
    group_size: int
    bits: int = 4
    mask: torch.Tensor | None = None

    @classmethod
    def from_weight(cls, weight, group_size=128, *, row_gain=None, col_gain=None, preserve_mask=False):
        validate_tensor(weight, 'grid source', ndim=2)
        if weight.dtype not in (torch.float32, torch.float64):
            weight = weight.float()
        rows, cols = weight.shape
        group = effective_group_size(cols, group_size)
        rg = torch.ones(rows, device=weight.device, dtype=weight.dtype) if row_gain is None else row_gain.to(weight)
        cg = torch.ones(cols, device=weight.device, dtype=weight.dtype) if col_gain is None else col_gain.to(weight)
        if rg.shape != (rows,) or cg.shape != (cols,):
            raise ValueError('Channel gain shape mismatch')
        for gain in (rg, cg):
            if not torch.isfinite(gain).all() or not (gain > 0).all():
                raise ValueError('Channel gains must be finite and positive')
        scaled = weight / rg[:, None] / cg[None, :]
        groups = scaled.reshape(rows, -1, group)
        lo, hi = groups.amin(-1), groups.amax(-1)
        steps = (hi - lo).clamp_min(1e-5) / 15
        zeros = (-torch.round(lo / steps)).clamp(0, 15).to(torch.uint8)
        mask = weight.ne(0) if preserve_mask else None
        state = cls(torch.empty_like(weight, dtype=torch.uint8), steps, zeros,
                    rg.clone(), cg.clone(), group, mask=mask)
        spacing, z = state.parameters()
        state.codes = nearest_code(weight, spacing, z, active=mask)
        state.validate()
        return state

    def parameters(self, rows=slice(None), cols=slice(None)):
        # Index arrays use paired advanced indexing, useful for sparse-factor moves.
        if torch.is_tensor(rows) or torch.is_tensor(cols):
            r = torch.arange(self.codes.shape[0], device=self.codes.device)[rows] if isinstance(rows, slice) else rows
            c = torch.arange(self.codes.shape[1], device=self.codes.device)[cols] if isinstance(cols, slice) else cols
            return (self.steps[r, c // self.group_size] * self.row_gain[r] * self.col_gain[c],
                    self.zeros[r, c // self.group_size])
        spacing = self.steps.repeat_interleave(self.group_size, 1)
        spacing = spacing * self.row_gain[:, None] * self.col_gain[None, :]
        return spacing[rows, cols], self.zeros.repeat_interleave(self.group_size, 1)[rows, cols]

    def values(self):
        spacing, zero = self.parameters()
        return (self.codes.to(spacing.dtype) - zero.to(spacing.dtype)) * spacing

    def clone(self):
        return copy.deepcopy(self)

    def validate(self):
        if self.bits != 4 or self.codes.dtype != torch.uint8 or self.codes.ndim != 2:
            raise ValueError('This adapter supports explicit W4 uint8 codes only')
        rows, cols = self.codes.shape
        if self.group_size < 1 or cols % self.group_size:
            raise ValueError('Invalid quantization group')
        if self.steps.shape != (rows, cols // self.group_size) or self.zeros.shape != self.steps.shape or self.zeros.dtype != torch.uint8:
            raise ValueError('Invalid grid parameter shapes')
        if self.row_gain.shape != (rows,) or self.col_gain.shape != (cols,):
            raise ValueError('Invalid gain shapes')
        for value in (self.steps, self.row_gain, self.col_gain):
            if not value.is_floating_point() or not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError('Invalid grid steps/gains')
        if (self.codes > 15).any() or (self.zeros > 15).any():
            raise ValueError('Code outside W4 alphabet')
        values = self.values()
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite dequantized weights')
        if self.mask is not None:
            if self.mask.dtype != torch.bool or self.mask.shape != self.codes.shape:
                raise ValueError('Invalid fixed mask')
            if not torch.equal(values.ne(0), self.mask):
                raise ValueError('Quantization changed the fixed mask')

    def state_dict(self):
        return {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in vars(self).items()}

    @classmethod
    def from_state_dict(cls, state, device='cpu'):
        result = cls(**{k: v.to(device).clone() if torch.is_tensor(v) else v for k, v in state.items()})
        result.validate()
        return result


@torch.no_grad()
def initialize_awq(weight, sparse, u, v, samples, group_size=128, grid=20):
    """AWQ-style initialization shared by every refinement arm.

    Same residual-scored scale search as the quantization-only adapter, FP32,
    no clipping or neighboring-layer fusion. This is not the old posthoc arm.
    """
    tensors = [value.detach().float() for value in (weight, sparse, u, v, samples)]
    weight, sparse, u, v, samples = tensors
    cfg = AWQConfig(group_size=group_size, grid=grid)
    _, _, scales, pair_info = search_pair(v, u, samples, F.linear(samples, weight - sparse), cfg)
    v_grid = CodeMatrix.from_weight(v, group_size, row_gain=scales['rank'], col_gain=1 / scales['first_input'])
    u_grid = CodeMatrix.from_weight(u, group_size, col_gain=1 / scales['rank'])
    target = F.linear(samples, weight - u_grid.values() @ v_grid.values())
    _, scale, sparse_info = search_sparse(sparse, samples, target, cfg)
    s_grid = CodeMatrix.from_weight(sparse, group_size, col_gain=1 / scale, preserve_mask=True)
    return {'S': s_grid, 'U': u_grid, 'V': v_grid}, {'pair': pair_info, 'sparse': sparse_info,
        'initializer': 'awq_residual_scale_search_fp32', 'clipping': False,
        'neighbor_fusion': False, 'scale_grid': grid}


class DiagonalRefiner:
    """Gauss-Seidel integer optimization for 1/2 ||(W-S-UV)D||_F^2.

    Output rows of U and input columns of V can be updated in parallel for
    fixed latent index; different latent indices must remain sequential.
    Each complete phase is checked against a freshly recomputed objective.
    """
    def __init__(self, weight, diagonal, grids, config=RefineConfig()):
        validate_tensor(weight, 'weight', ndim=2)
        validate_tensor(diagonal, 'diagonal', ndim=1)
        if weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('Refinement arithmetic must use FP32 or FP64')
        if diagonal.shape != (weight.shape[1],) or not (diagonal > 0).all():
            raise ValueError('Expected a positive diagonal of input width')
        self.w = weight.detach().clone()
        self.h = diagonal.detach().to(weight).clone()
        self.grids = {key: grids[key].clone() for key in ('S', 'U', 'V')}
        self.config = config
        for value in self.grids.values():
            value.validate()
            if value.steps.dtype != weight.dtype or value.codes.device != weight.device:
                raise ValueError('Grid dtype/device must match the objective')
        mask = self.grids['S'].mask
        if mask is None or mask.shape != weight.shape or weight.shape[1] % 4:
            raise ValueError('A fixed 2:4 mask is required')
        if not mask.reshape(weight.shape[0], -1, 4).sum(-1).eq(2).all():
            raise ValueError('Invalid 2:4 mask')
        self.refresh()
        if not self.storage_valid():
            raise ValueError('Initial weights cannot preserve the mask/finiteness in the storage dtype')
        self.records = []

    def refresh(self):
        self.s, self.u, self.v = (self.grids[k].values() for k in ('S', 'U', 'V'))
        if self.u.ndim != 2 or self.v.ndim != 2 or self.u.shape[1] != self.v.shape[0]:
            raise ValueError('Incompatible factor shapes')
        if (self.u.shape[0], self.v.shape[1]) != self.w.shape:
            raise ValueError('Factor shape does not match weight')
        self.e = self.w - self.s - self.u @ self.v

    def objective(self):
        return float((self.e.square() * self.h).sum(dtype=torch.float64) * .5)

    def storage_valid(self):
        dtype = getattr(torch, self.config.storage_dtype)
        sparse = self.s.to(dtype)
        return (bool(torch.equal(sparse.ne(0), self.grids['S'].mask)) and
                bool(torch.isfinite(sparse).all()) and
                bool(torch.isfinite(self.u.to(dtype)).all()) and
                bool(torch.isfinite(self.v.to(dtype)).all()))

    @torch.no_grad()
    def phase(self, name, operation, sweep=0):
        before = self.objective()
        saved = {key: (g.codes.clone(), g.steps.clone()) for key, g in self.grids.items()}
        start = time.perf_counter()
        operation()
        self.refresh()
        proposed = self.objective()
        storage_valid = self.storage_valid()
        accepted = math.isfinite(proposed) and proposed <= before and storage_valid
        if not accepted:
            for key, (codes, steps) in saved.items():
                self.grids[key].codes, self.grids[key].steps = codes, steps
            self.refresh()
        after = self.objective()
        record = {'sweep': sweep, 'phase': name, 'before': before, 'proposed': proposed if math.isfinite(proposed) else None,
                  'after': after, 'accepted': accepted, 'seconds': time.perf_counter() - start,
                  'proposed_storage_valid': storage_valid,
                  'changed_codes': {k: int((self.grids[k].codes != saved[k][0]).sum()) for k in saved}}
        self.records.append(record)
        return record

    def codes_u(self):
        grid = self.grids['U']
        gram = (self.v * self.h) @ self.v.T
        gradient = (self.e * self.h) @ self.v.T
        rows = torch.arange(self.u.shape[0], device=self.w.device)
        for k in range(self.u.shape[1]):
            curvature = gram[k, k]
            spacing, zero = grid.parameters(rows, k)
            target = self.u[:, k] + gradient[:, k] / curvature.clamp_min(torch.finfo(self.w.dtype).tiny)
            code = nearest_code(target, spacing, zero, old=grid.codes[:, k])
            code = torch.where(curvature > 0, code, grid.codes[:, k])
            value = (code.to(self.w.dtype) - zero) * spacing
            delta = value - self.u[:, k]
            grid.codes[:, k] = code
            self.u[:, k] = value
            gradient -= delta[:, None] * gram[k, :]

    def codes_v(self):
        grid = self.grids['V']
        gram = self.u.T @ self.u
        gradient = self.u.T @ (self.e * self.h)
        cols = torch.arange(self.v.shape[1], device=self.w.device)
        for k in range(self.v.shape[0]):
            curvature = gram[k, k] * self.h
            spacing, zero = grid.parameters(k, cols)
            target = self.v[k] + gradient[k] / curvature.clamp_min(torch.finfo(self.w.dtype).tiny)
            code = nearest_code(target, spacing, zero, old=grid.codes[k])
            code = torch.where(curvature > 0, code, grid.codes[k])
            value = (code.to(self.w.dtype) - zero) * spacing
            delta = value - self.v[k]
            grid.codes[k] = code
            self.v[k] = value
            gradient -= gram[:, k, None] * (delta * self.h)[None, :]

    def codes_s(self):
        grid = self.grids['S']
        spacing, zero = grid.parameters()
        grid.codes = nearest_code(self.w - self.u @ self.v, spacing, zero,
                                  active=grid.mask, old=grid.codes)

    def scales_s(self):
        grid = self.grids['S']
        shape = (*grid.steps.shape, grid.group_size)
        c = self.s / grid.steps.repeat_interleave(grid.group_size, 1)
        numerator = (self.e * c * self.h).reshape(shape).sum(-1)
        denominator = (c.square() * self.h).reshape(shape).sum(-1)
        value = grid.steps + numerator / denominator.clamp_min(torch.finfo(self.w.dtype).tiny)
        # A legal nonzero integer code can be only one step from zero. Bound the
        # group step for EVERY active location, independently of current codes,
        # so future code moves remain nonzero after storage-dtype conversion.
        storage = torch.finfo(getattr(torch, self.config.storage_dtype))
        smallest_subnormal = storage.tiny * storage.eps
        gain = grid.row_gain[:, None] * grid.col_gain[None, :]
        gain = gain.masked_fill(~grid.mask, float('inf')).reshape(shape).amin(-1)
        lower = (smallest_subnormal / gain).clamp_min(self.config.min_step)
        grid.steps = torch.where(denominator > 0, torch.maximum(value, lower), grid.steps)

    def scales_u(self):
        grid = self.grids['U']
        gram = (self.v * self.h) @ self.v.T
        gradient = (self.e * self.h) @ self.v.T
        for group in range(grid.steps.shape[1]):
            sl = slice(group * grid.group_size, (group + 1) * grid.group_size)
            c = self.u[:, sl] / grid.steps[:, group, None]
            numerator = (gradient[:, sl] * c).sum(1)
            denominator = ((c @ gram[sl, sl]) * c).sum(1)
            value = grid.steps[:, group] + numerator / denominator.clamp_min(torch.finfo(self.w.dtype).tiny)
            value = torch.where(denominator > 0, value.clamp_min(self.config.min_step), grid.steps[:, group])
            delta = c * (value - grid.steps[:, group])[:, None]
            grid.steps[:, group] = value
            self.u[:, sl] += delta
            gradient -= delta @ gram[sl, :]

    def scales_v(self):
        grid = self.grids['V']
        gram = self.u.T @ self.u
        gradient = self.u.T @ (self.e * self.h)
        for k in range(self.v.shape[0]):
            c = self.v[k] / grid.steps[k].repeat_interleave(grid.group_size)
            numerator = (gradient[k] * c).reshape(-1, grid.group_size).sum(-1)
            denominator = gram[k, k] * (c.square() * self.h).reshape(-1, grid.group_size).sum(-1)
            value = grid.steps[k] + numerator / denominator.clamp_min(torch.finfo(self.w.dtype).tiny)
            value = torch.where(denominator > 0, value.clamp_min(self.config.min_step), grid.steps[k])
            delta = c * (value - grid.steps[k]).repeat_interleave(grid.group_size)
            grid.steps[k] = value
            self.v[k] += delta
            gradient -= gram[:, k, None] * (delta * self.h)[None, :]

    @staticmethod
    def pair_choice(s, factor, sp, sz, fp, fz, gs, gf, a, b, cross, valid):
        """Exact two-coordinate search: enumerate S codes, round conditional factor."""
        candidates = torch.arange(16, device=s.device, dtype=s.dtype)[None, :]
        ds = (candidates - sz[:, None]) * sp[:, None] - s[:, None]
        tiny = torch.finfo(s.dtype).tiny
        target = factor[:, None] + (gf[:, None] - cross[:, None] * ds) / b.clamp_min(tiny)[:, None]
        fc = nearest_code(target, fp[:, None], fz[:, None])
        df = (fc.to(s.dtype) - fz[:, None]) * fp[:, None] - factor[:, None]
        change = -gs[:, None] * ds - gf[:, None] * df + .5 * (
            a[:, None] * ds.square() + 2 * cross[:, None] * ds * df + b[:, None] * df.square())
        change.masked_fill_(candidates == sz[:, None], float('inf'))
        best, index = change.min(1)
        take = valid & (b > 0) & torch.isfinite(best) & (best < 0)
        row = torch.arange(s.numel(), device=s.device)
        return index.to(torch.uint8), fc[row, index], take

    def pairs_su(self):
        sg, ug = self.grids['S'], self.grids['U']
        rows = torch.arange(self.w.shape[0], device=self.w.device)
        gram = (self.v * self.h) @ self.v.T
        gradient = (self.e * self.h) @ self.v.T
        for k in range(self.u.shape[1]):
            candidates = (self.v[k].square() * self.h).topk(min(self.config.pair_candidates, self.w.shape[1])).indices
            eligible = sg.mask[:, candidates]
            j = candidates[eligible.to(torch.int32).argmax(1)]
            valid = eligible.any(1)
            sp, sz = sg.parameters(rows, j)
            fp, fz = ug.parameters(rows, k)
            s, factor = self.s[rows, j], self.u[:, k]
            a = self.h[j]
            b = gram[k, k].expand_as(a)
            cross = self.h[j] * self.v[k, j]
            sc, fc, take = self.pair_choice(s, factor, sp, sz, fp, fz,
                self.e[rows, j] * self.h[j], gradient[:, k], a, b, cross, valid)
            sc = torch.where(take, sc, sg.codes[rows, j])
            fc = torch.where(take, fc, ug.codes[:, k])
            ns = (sc.to(self.w.dtype) - sz) * sp
            nf = (fc.to(self.w.dtype) - fz) * fp
            ds, df = ns - s, nf - factor
            sg.codes[rows, j], ug.codes[:, k] = sc, fc
            self.s[rows, j], self.u[:, k] = ns, nf
            self.e -= df[:, None] * self.v[k]
            self.e[rows, j] -= ds
            gradient -= df[:, None] * gram[k]
            gradient -= (ds * self.h[j])[:, None] * self.v[:, j].T

    def pairs_sv(self):
        sg, vg = self.grids['S'], self.grids['V']
        cols = torch.arange(self.w.shape[1], device=self.w.device)
        gram = self.u.T @ self.u
        gradient = self.u.T @ (self.e * self.h)
        for k in range(self.v.shape[0]):
            candidates = self.u[:, k].square().topk(min(self.config.pair_candidates, self.w.shape[0])).indices
            eligible = sg.mask[candidates, :]
            i = candidates[eligible.to(torch.int32).argmax(0)]
            valid = eligible.any(0)
            sp, sz = sg.parameters(i, cols)
            fp, fz = vg.parameters(k, cols)
            s, factor = self.s[i, cols], self.v[k]
            a = self.h
            b = gram[k, k] * self.h
            cross = self.u[i, k] * self.h
            sc, fc, take = self.pair_choice(s, factor, sp, sz, fp, fz,
                self.e[i, cols] * self.h, gradient[k], a, b, cross, valid)
            sc = torch.where(take, sc, sg.codes[i, cols])
            fc = torch.where(take, fc, vg.codes[k])
            ns = (sc.to(self.w.dtype) - sz) * sp
            nf = (fc.to(self.w.dtype) - fz) * fp
            ds, df = ns - s, nf - factor
            sg.codes[i, cols], vg.codes[k] = sc, fc
            self.s[i, cols], self.v[k] = ns, nf
            self.e -= self.u[:, k, None] * df[None, :]
            self.e[i, cols] -= ds
            gradient -= gram[:, k, None] * (df * self.h)[None, :]
            gradient -= self.u[i, :].T * (ds * self.h)[None, :]

    @torch.no_grad()
    def run(self, paired=False, callback=None):
        operations = [('codes_u', self.codes_u), ('codes_v', self.codes_v), ('codes_s', self.codes_s)]
        if paired:
            operations += [('pairs_su', self.pairs_su), ('pairs_sv', self.pairs_sv)]
        if self.config.update_scales:
            operations += [('scales_u', self.scales_u), ('scales_v', self.scales_v), ('scales_s', self.scales_s)]
        for sweep in range(1, self.config.sweeps + 1):
            for name, operation in operations:
                record = self.phase(name, operation, sweep)
                if callback is not None:
                    callback(record)
        for grid in self.grids.values():
            grid.validate()
        return self.grids

    def state_dict(self):
        return {'format': 'oats-code-refine-v1', 'objective': 'fixed_diagonal_weighted_reconstruction',
                'fixed_mask': True, 'scales_already_applied': False,
                'config': asdict(self.config), 'packed_int4': False,
                'grids': {key: value.state_dict() for key, value in self.grids.items()}}
