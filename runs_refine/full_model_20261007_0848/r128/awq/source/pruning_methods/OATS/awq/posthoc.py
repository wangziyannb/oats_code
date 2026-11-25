"""Post-hoc quantization adapted from awq_methods.zip (see SOURCES.json).

Only quantization routines are ported. The native OATS CompressedLinear owns
S, V (first), U (second). No pruning or SVD implementation is imported here.
"""
from __future__ import annotations
import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from pruning_methods.OATS.compressed_linear import CompressedLinear
from .reference.quantizer import pseudo_quantize_tensor

PROJECTION_NAMES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                    'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')

def require_residue(module, name):
    if not isinstance(module, CompressedLinear):
        raise TypeError(f"{name}: expected native OATS CompressedLinear")
    return module

def tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def move_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tree(x, device) for x in value)
    if isinstance(value, list):
        return [move_tree(x, device) for x in value]
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    return value


def count_2_4(weight: torch.Tensor) -> Tuple[int, int]:
    if weight.ndim != 2 or weight.shape[1] % 4:
        return 0, 0
    zeros = weight.eq(0).reshape(weight.shape[0], -1, 4).sum(dim=-1)
    return zeros.numel(), int((zeros != 2).sum().item())


@torch.no_grad()
def mask_preserving_awq_quant(
    weight: torch.Tensor,
    official_pseudo_quantize_tensor,
    *,
    w_bit: int,
    group_size: int,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Official asymmetric fake quantization plus fixed-mask projection."""

    original_mask = weight.eq(0)
    quantized, scales, zeros = official_pseudo_quantize_tensor(
        weight,
        n_bit=w_bit,
        zero_point=True,
        q_group_size=group_size,
        get_scale_zp=True,
    )
    quantized = quantized.reshape(weight.shape[0], -1, group_size)
    grouped_weight = weight.reshape(weight.shape[0], -1, group_size)
    grouped_mask = original_mask.reshape(weight.shape[0], -1, group_size)
    grouped_scales = scales.reshape(weight.shape[0], -1, 1).to(quantized)

    rounded_active = (~grouped_mask) & quantized.eq(0)
    rounded_count = int(rounded_active.sum().item())
    if rounded_count:
        nearest_nonzero = torch.where(
            grouped_weight >= 0, grouped_scales, -grouped_scales
        )
        quantized[rounded_active] = nearest_nonzero.expand_as(quantized)[rounded_active]
    quantized[grouped_mask] = 0
    return quantized.reshape_as(weight), rounded_count, scales, zeros


@torch.no_grad()
def module_equivalence(
    modules: Sequence[CompressedLinear],
    before_input: torch.Tensor,
    after_input: torch.Tensor,
    before_outputs: Sequence[torch.Tensor],
) -> Dict[str, float]:
    max_abs = 0.0
    numerator = 0.0
    denominator = 0.0
    for module, reference in zip(modules, before_outputs):
        current = module(after_input).float()
        reference = reference.to(current.device).float()
        difference = current - reference
        max_abs = max(max_abs, float(difference.abs().max().item()))
        numerator += float(difference.pow(2).sum().item())
        denominator += float(reference.float().pow(2).sum().item())
    return {
        "max_abs": max_abs,
        "relative_mse": numerator / max(denominator, 1e-30),
    }


@torch.no_grad()
def apply_norm_to_residue_scale(
    norm: nn.Module,
    modules: Sequence[CompressedLinear],
    scales: torch.Tensor,
) -> None:
    norm_scale = scales.to(norm.weight)
    norm.weight.div_(norm_scale)
    if getattr(norm, "bias", None) is not None:
        norm.bias.div_(norm_scale)
    for module in modules:
        scale = scales.to(module.S).reshape(1, -1)
        module.S.mul_(scale)
        module.V.mul_(scale)


@torch.no_grad()
def apply_residue_to_residue_scale(
    previous: CompressedLinear,
    target: CompressedLinear,
    scales: torch.Tensor,
) -> None:
    previous_scale = scales.to(previous.S).reshape(-1, 1)
    previous.S.div_(previous_scale)
    previous.U.div_(previous_scale)
    if previous.bias is not None:
        previous.bias.div_(scales.to(previous.bias))

    target_scale = scales.to(target.S).reshape(1, -1)
    target.S.mul_(target_scale)
    target.V.mul_(target_scale)


@torch.no_grad()
def auto_clip_sparse_layer(
    weight: torch.Tensor,
    input_feat: torch.Tensor,
    official_pseudo_quantize_tensor,
    *,
    w_bit: int,
    group_size: int,
    n_grid: int,
    max_shrink: float,
    n_sample_token: int,
) -> torch.Tensor:
    """Official AWQ clipping search with fixed-mask fake quantization."""

    original_shape = weight.shape
    features = input_feat.reshape(-1, input_feat.shape[-1])
    features = features.reshape(1, features.shape[0], -1, group_size)
    stride = max(1, features.shape[1] // n_sample_token)
    features = features[:, 0::stride]
    grouped_weight = weight.reshape(weight.shape[0], 1, -1, group_size)

    output_batch = 256 if weight.shape[0] % 256 == 0 else 64
    if weight.shape[0] % output_batch:
        raise ValueError(
            f"Output dimension {weight.shape[0]} is not divisible by {output_batch}"
        )
    all_best: List[torch.Tensor] = []
    for batch_idx in range(weight.shape[0] // output_batch):
        current_weight = grouped_weight[
            batch_idx * output_batch : (batch_idx + 1) * output_batch
        ]
        original_max = current_weight.abs().amax(dim=-1, keepdim=True)
        best_max = original_max.clone()
        # The official code uses ``ones_like * 1e9`` as an infinity sentinel.
        # ``full_like(..., 1e9)`` raises on newer Torch because FP16 cannot
        # represent 1e9, so spell the same sentinel as infinity explicitly.
        min_errors = torch.full_like(original_max, float("inf"))
        device_features = features.to(current_weight.device)
        original_output = (device_features * current_weight).sum(dim=-1)

        for shrink_idx in range(int(max_shrink * n_grid)):
            max_value = original_max * (1 - shrink_idx / n_grid)
            clipped = torch.clamp(current_weight, -max_value, max_value)
            flat = clipped.reshape(-1, original_shape[1])
            quantized, _, _, _ = mask_preserving_awq_quant(
                flat,
                official_pseudo_quantize_tensor,
                w_bit=w_bit,
                group_size=group_size,
            )
            quantized = quantized.reshape_as(current_weight)
            candidate_output = (device_features * quantized).sum(dim=-1)
            errors = (
                (candidate_output - original_output)
                .pow(2)
                .mean(dim=1)
                .reshape_as(min_errors)
            )
            better = errors < min_errors
            min_errors[better] = errors[better]
            best_max[better] = max_value[better]
            del clipped, flat, quantized, candidate_output, errors
        all_best.append(best_max)
        del device_features, original_output, min_errors
    return torch.cat(all_best, dim=0).squeeze(1)


@torch.no_grad()
def apply_clip(weight: torch.Tensor, max_values: torch.Tensor) -> None:
    original_shape = weight.shape
    grouped = weight.reshape(*max_values.shape[:2], -1)
    grouped.clamp_(-max_values.to(grouped), max_values.to(grouped))
    weight.copy_(grouped.reshape(original_shape))


def capture_first_layer_input(
    model: nn.Module,
    samples: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    layers = model.model.layers
    captured: List[torch.Tensor] = []
    layer_kwargs: Dict[str, Any] = {}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            captured.append(hidden_states.detach())
            layer_kwargs.update(kwargs)
            raise CatcherExit

    layers[0] = layers[0].to(device)
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(device))
    except CatcherExit:
        pass
    finally:
        layers[0] = layers[0].module
    if not captured:
        raise RuntimeError("Failed to capture input to decoder layer 0")
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    return captured[0], layer_kwargs


def capture_projection_inputs(
    layer: nn.Module,
    layer_input: torch.Tensor,
    layer_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    features: Dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def save_input(_module, args):
            features[name] = args[0].detach().cpu()

        return save_input

    for name in (
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ):
        handles.append(layer.get_submodule(name).register_forward_pre_hook(hook(name)))
    try:
        output = tensor_output(layer(layer_input, **layer_kwargs)).detach()
    finally:
        for handle in handles:
            handle.remove()
    missing = set((
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    )) - set(features)
    if missing:
        raise RuntimeError(f"Failed to capture projection inputs: {sorted(missing)}")
    return output, features

@dataclass
class BranchQuantStats:
    layer: int
    module: str
    branch: str
    shape: Tuple[int, int]
    group_size: int
    numel: int
    mse: float
    max_abs_error: float
    rounded_active_to_zero_before_projection: int = 0
    mask_equal: Any = None
    groups_2_4: int = 0
    violations_2_4: int = 0


@torch.no_grad()
def dense_awq_quant(
    weight: torch.Tensor,
    official_pseudo_quantize_tensor,
    *,
    w_bit: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got {tuple(weight.shape)}")
    if group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError(
            f"Invalid group size {group_size} for weight {tuple(weight.shape)}"
        )
    quantized, scales, zeros = official_pseudo_quantize_tensor(
        weight,
        n_bit=w_bit,
        zero_point=True,
        q_group_size=group_size,
        get_scale_zp=True,
    )
    return quantized, scales, zeros


@torch.no_grad()
def prebalance_factors(module, probe: torch.Tensor) -> Dict[str, float]:
    """Exact gauge initialization that equalizes A-row and B-column maxima."""

    reference = module(probe).detach().float()
    a = module.V
    b = module.U
    amax = a.float().abs().amax(dim=1).clamp_min(1e-8)
    bmax = b.float().abs().amax(dim=0).clamp_min(1e-8)
    scale = torch.sqrt(amax / bmax).clamp(1e-4, 1e4).to(a)
    a.div_(scale.reshape(-1, 1))
    b.mul_(scale.reshape(1, -1).to(b))
    current = module(probe).float()
    difference = current - reference
    return {
        "scale_min": float(scale.min().item()),
        "scale_max": float(scale.max().item()),
        "max_abs": float(difference.abs().max().item()),
        "relative_mse": float(
            difference.pow(2).sum().item()
            / max(reference.pow(2).sum().item(), 1e-30)
        ),
    }


@torch.no_grad()
def search_external_allbranch_scale(
    inspect_module: nn.Module,
    target_modules: Sequence,
    input_feat: torch.Tensor,
    kwargs: Dict[str, Any],
    official_pseudo_quantize_tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """AWQ input-channel search with S, A, and B all fake-quantized."""

    device = next(inspect_module.parameters()).device
    x = input_feat.to(device)
    kwargs = move_tree(kwargs, device)
    original_output = tensor_output(inspect_module(x, **kwargs)).detach()
    x_max = x.abs().reshape(-1, x.shape[-1]).mean(dim=0)
    originals = [
        (
            module.S.detach().clone(),
            module.V.detach().clone(),
            module.U.detach().clone(),
        )
        for module in target_modules
    ]
    quantized_b = [
        dense_awq_quant(
            b,
            official_pseudo_quantize_tensor,
            w_bit=args.w_bit,
            group_size=args.b_group_size,
        )[0]
        for _, _, b in originals
    ]

    best_error = math.inf
    best_ratio = -1.0
    best_scales = None
    history: List[float] = []
    sparse_projection_history: List[int] = []
    try:
        for grid_idx in range(args.external_scale_grid):
            ratio = grid_idx / args.external_scale_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).reshape(-1)
            scales = scales / torch.sqrt(scales.max() * scales.min())
            projected = 0
            for module, (s, a, _b), qb in zip(target_modules, originals, quantized_b):
                input_scale = scales.reshape(1, -1).to(s)
                qs, corrected, _, _ = mask_preserving_awq_quant(
                    s * input_scale,
                    official_pseudo_quantize_tensor,
                    w_bit=args.w_bit,
                    group_size=args.sparse_group_size,
                )
                qa, _, _ = dense_awq_quant(
                    a * input_scale.to(a),
                    official_pseudo_quantize_tensor,
                    w_bit=args.w_bit,
                    group_size=args.a_group_size,
                )
                projected += corrected
                module.S.data.copy_(qs / input_scale)
                module.V.data.copy_(qa / input_scale.to(qa))
                module.U.data.copy_(qb)

            candidate = tensor_output(inspect_module(x, **kwargs))
            error = float(
                (candidate.float() - original_output.float()).pow(2).mean().item()
            )
            history.append(error)
            sparse_projection_history.append(projected)
            if error < best_error:
                best_error = error
                best_ratio = ratio
                best_scales = scales.detach().clone()
            for module, (s, a, b) in zip(target_modules, originals):
                module.S.data.copy_(s)
                module.V.data.copy_(a)
                module.U.data.copy_(b)
            del candidate
    finally:
        for module, (s, a, b) in zip(target_modules, originals):
            module.S.data.copy_(s)
            module.V.data.copy_(a)
            module.U.data.copy_(b)

    if best_scales is None:
        raise RuntimeError(f"External AWQ search failed: {history}")
    return best_scales, {
        "best_ratio": best_ratio,
        "best_error": best_error,
        "history": history,
        "sparse_projection_history": sparse_projection_history,
    }


@torch.no_grad()
def search_rank_gauge_scale(
    module,
    input_feat: torch.Tensor,
    official_pseudo_quantize_tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Search A<-D^-1 A, B<-BD using the complete quantized Residue output."""

    device = module.S.device
    x = input_feat.to(device)
    s = module.S.detach()
    a = module.V.detach()
    b = module.U.detach()
    original_output = module(x).detach()
    qs, sparse_projected, _, _ = mask_preserving_awq_quant(
        s,
        official_pseudo_quantize_tensor,
        w_bit=args.w_bit,
        group_size=args.sparse_group_size,
    )
    sparse_output = F.linear(x, qs)
    hidden = F.linear(x, a)
    hidden_scale = hidden.abs().reshape(-1, hidden.shape[-1]).mean(dim=0)

    best_error = math.inf
    best_ratio = -1.0
    best_scales = None
    history: List[float] = []
    for grid_idx in range(args.rank_scale_grid):
        ratio = grid_idx / args.rank_scale_grid
        scales = hidden_scale.pow(ratio).clamp(min=1e-4).reshape(-1)
        scales = scales / torch.sqrt(scales.max() * scales.min())
        qa, _, _ = dense_awq_quant(
            a / scales.reshape(-1, 1).to(a),
            official_pseudo_quantize_tensor,
            w_bit=args.w_bit,
            group_size=args.a_group_size,
        )
        qb, _, _ = dense_awq_quant(
            b * scales.reshape(1, -1).to(b),
            official_pseudo_quantize_tensor,
            w_bit=args.w_bit,
            group_size=args.b_group_size,
        )
        candidate = sparse_output + F.linear(F.linear(x, qa), qb)
        if module.bias is not None:
            candidate = candidate + module.bias
        error = float(
            (candidate.float() - original_output.float()).pow(2).mean().item()
        )
        history.append(error)
        if error < best_error:
            best_error = error
            best_ratio = ratio
            best_scales = scales.detach().clone()
        del qa, qb, candidate
    if best_scales is None:
        raise RuntimeError(f"Rank gauge AWQ search failed: {history}")
    return best_scales, {
        "best_ratio": best_ratio,
        "best_error": best_error,
        "history": history,
        "sparse_projected": sparse_projected,
    }


@torch.no_grad()
def apply_rank_gauge_scale(module, scales: torch.Tensor) -> None:
    module.V.div_(scales.to(module.V).reshape(-1, 1))
    module.U.mul_(scales.to(module.U).reshape(1, -1))


@torch.no_grad()
def rank_gauge_equivalence(module, feature: torch.Tensor, scales: torch.Tensor) -> Dict[str, float]:
    probe = feature.reshape(-1, feature.shape[-1])[:32].to(module.S.device)
    reference = module(probe).detach().float()
    apply_rank_gauge_scale(module, scales)
    current = module(probe).float()
    difference = current - reference
    return {
        "max_abs": float(difference.abs().max().item()),
        "relative_mse": float(
            difference.pow(2).sum().item()
            / max(reference.pow(2).sum().item(), 1e-30)
        ),
    }


@torch.no_grad()
def auto_clip_dense_layer(
    weight: torch.Tensor,
    input_feat: torch.Tensor,
    official_pseudo_quantize_tensor,
    *,
    w_bit: int,
    group_size: int,
    n_grid: int,
    max_shrink: float,
    n_sample_token: int,
) -> torch.Tensor:
    """Official AWQ clipping objective with support for rank-32 A matrices."""

    features = input_feat.reshape(-1, input_feat.shape[-1])
    features = features.reshape(1, features.shape[0], -1, group_size)
    stride = max(1, features.shape[1] // n_sample_token)
    features = features[:, 0::stride]
    grouped_weight = weight.reshape(weight.shape[0], 1, -1, group_size)
    if weight.shape[0] % 256 == 0:
        output_batch = 256
    elif weight.shape[0] % 64 == 0:
        output_batch = 64
    else:
        output_batch = weight.shape[0]

    best_values: List[torch.Tensor] = []
    for batch_idx in range(weight.shape[0] // output_batch):
        current = grouped_weight[
            batch_idx * output_batch : (batch_idx + 1) * output_batch
        ]
        original_max = current.abs().amax(dim=-1, keepdim=True)
        best_max = original_max.clone()
        min_errors = torch.full_like(original_max, float("inf"))
        device_features = features.to(current.device)
        original_output = (device_features * current).sum(dim=-1)
        for shrink_idx in range(int(max_shrink * n_grid)):
            max_value = original_max * (1 - shrink_idx / n_grid)
            clipped = torch.clamp(current, -max_value, max_value)
            flat = clipped.reshape(-1, weight.shape[1])
            quantized, _, _ = dense_awq_quant(
                flat,
                official_pseudo_quantize_tensor,
                w_bit=w_bit,
                group_size=group_size,
            )
            quantized = quantized.reshape_as(current)
            candidate_output = (device_features * quantized).sum(dim=-1)
            errors = (
                (candidate_output - original_output)
                .pow(2)
                .mean(dim=1)
                .reshape_as(min_errors)
            )
            better = errors < min_errors
            min_errors[better] = errors[better]
            best_max[better] = max_value[better]
            del clipped, flat, quantized, candidate_output, errors
        best_values.append(best_max)
        del device_features, original_output, min_errors
    return torch.cat(best_values, dim=0).squeeze(1)


def process_layer(
    layer_idx: int,
    layer: nn.Module,
    features: Dict[str, torch.Tensor],
    layer_kwargs: Dict[str, Any],
    official_pseudo_quantize_tensor,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[BranchQuantStats]]:
    started = time.time()
    modules = {
        name: require_residue(layer.get_submodule(name), name)
        for name in PROJECTION_NAMES
    }
    result: Dict[str, Any] = {
        "layer": layer_idx,
        "prebalance": {},
        "external_scales": {},
        "rank_scales": {},
        "clips": {},
    }

    for name, module in modules.items():
        feature_name = (
            "self_attn.q_proj"
            if name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
            else "mlp.gate_proj"
            if name in ("mlp.gate_proj", "mlp.up_proj")
            else name
        )
        probe = features[feature_name].reshape(-1, features[feature_name].shape[-1])[:32]
        probe = probe.to(module.S.device)
        result["prebalance"][name] = prebalance_factors(module, probe)

    groups = [
        (
            "attention_qkv",
            layer.self_attn,
            [modules["self_attn.q_proj"], modules["self_attn.k_proj"], modules["self_attn.v_proj"]],
            features["self_attn.q_proj"],
            layer_kwargs,
            layer.input_layernorm,
            None,
        ),
        (
            "attention_output",
            modules["self_attn.o_proj"],
            [modules["self_attn.o_proj"]],
            features["self_attn.o_proj"],
            {},
            None,
            modules["self_attn.v_proj"],
        ),
        (
            "mlp_gate_up",
            layer.mlp,
            [modules["mlp.gate_proj"], modules["mlp.up_proj"]],
            features["mlp.gate_proj"],
            {},
            layer.post_attention_layernorm,
            None,
        ),
        (
            "mlp_down",
            modules["mlp.down_proj"],
            [modules["mlp.down_proj"]],
            features["mlp.down_proj"],
            {},
            None,
            modules["mlp.up_proj"],
        ),
    ]

    scaled_features: Dict[str, torch.Tensor] = {}
    for group_name, inspect_module, targets, feature, kwargs, norm, previous in groups:
        scales, search = search_external_allbranch_scale(
            inspect_module,
            targets,
            feature,
            kwargs,
            official_pseudo_quantize_tensor,
            args,
        )
        probe = feature.reshape(-1, feature.shape[-1])[:32].to(targets[0].S.device)
        before = [target(probe).detach().cpu() for target in targets]
        if norm is not None:
            apply_norm_to_residue_scale(norm, targets, scales)
        else:
            apply_residue_to_residue_scale(previous, targets[0], scales)
        after_probe = probe / scales.to(probe).reshape(1, -1)
        equivalence = module_equivalence(targets, probe, after_probe, before)
        scale_shape = (1,) * (feature.ndim - 1) + (-1,)
        scaled = feature / scales.cpu().to(feature).reshape(scale_shape)
        if group_name == "attention_qkv":
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                scaled_features[name] = scaled
        elif group_name == "attention_output":
            scaled_features["self_attn.o_proj"] = scaled
        elif group_name == "mlp_gate_up":
            for name in ("mlp.gate_proj", "mlp.up_proj"):
                scaled_features[name] = scaled
        else:
            scaled_features["mlp.down_proj"] = scaled
        result["external_scales"][group_name] = {
            **search,
            "min": float(scales.min().item()),
            "max": float(scales.max().item()),
            "equivalence": equivalence,
        }
        print(
            f"[layer {layer_idx:02d}] external {group_name}: "
            f"ratio={search['best_ratio']:.2f} mse={search['best_error']:.6e} "
            f"equiv={equivalence['relative_mse']:.3e}",
            flush=True,
        )

    for name, module in modules.items():
        scales, search = search_rank_gauge_scale(
            module,
            scaled_features[name],
            official_pseudo_quantize_tensor,
            args,
        )
        equivalence = rank_gauge_equivalence(module, scaled_features[name], scales)
        result["rank_scales"][name] = {
            **search,
            "min": float(scales.min().item()),
            "max": float(scales.max().item()),
            "equivalence": equivalence,
        }
        print(
            f"[layer {layer_idx:02d}] rank {name}: "
            f"ratio={search['best_ratio']:.2f} mse={search['best_error']:.6e} "
            f"equiv={equivalence['relative_mse']:.3e}",
            flush=True,
        )

    if not args.skip_clip:
        for name, module in modules.items():
            if any(token in name for token in ("q_", "k_", "query", "key", "Wqkv")):
                result["clips"][name] = {"skipped": True, "reason": "official_qk_skip"}
                continue
            clip_started = time.time()
            x = scaled_features[name].to(module.S.device)
            sparse_max = auto_clip_sparse_layer(
                module.S,
                x,
                official_pseudo_quantize_tensor,
                w_bit=args.w_bit,
                group_size=args.sparse_group_size,
                n_grid=args.clip_grid,
                max_shrink=args.clip_max_shrink,
                n_sample_token=args.clip_sample_tokens,
            )
            apply_clip(module.S, sparse_max)
            a_max = auto_clip_dense_layer(
                module.V,
                x,
                official_pseudo_quantize_tensor,
                w_bit=args.w_bit,
                group_size=args.a_group_size,
                n_grid=args.clip_grid,
                max_shrink=args.clip_max_shrink,
                n_sample_token=args.clip_sample_tokens,
            )
            apply_clip(module.V, a_max)
            qa, _, _ = dense_awq_quant(
                module.V,
                official_pseudo_quantize_tensor,
                w_bit=args.w_bit,
                group_size=args.a_group_size,
            )
            quantized_hidden = F.linear(x, qa)
            b_max = auto_clip_dense_layer(
                module.U,
                quantized_hidden,
                official_pseudo_quantize_tensor,
                w_bit=args.w_bit,
                group_size=args.b_group_size,
                n_grid=args.clip_grid,
                max_shrink=args.clip_max_shrink,
                n_sample_token=args.clip_sample_tokens,
            )
            apply_clip(module.U, b_max)
            result["clips"][name] = {
                "skipped": False,
                "sparse_max_min": float(sparse_max.min().item()),
                "sparse_max_max": float(sparse_max.max().item()),
                "a_max_min": float(a_max.min().item()),
                "a_max_max": float(a_max.max().item()),
                "b_max_min": float(b_max.min().item()),
                "b_max_max": float(b_max.max().item()),
                "b_activation": "Q(A)X",
                "seconds": time.time() - clip_started,
            }
            print(
                f"[layer {layer_idx:02d}] clip all {name}: "
                f"seconds={time.time() - clip_started:.1f}",
                flush=True,
            )
            del x, sparse_max, a_max, qa, quantized_hidden, b_max

    stats: List[BranchQuantStats] = []
    for name, module in modules.items():
        sparse_original = module.S.detach().clone()
        sparse_mask = sparse_original.eq(0)
        qs, corrected, _, _ = mask_preserving_awq_quant(
            sparse_original,
            official_pseudo_quantize_tensor,
            w_bit=args.w_bit,
            group_size=args.sparse_group_size,
        )
        module.S.data.copy_(qs)
        groups_2_4, violations = count_2_4(module.S)
        stats.append(
            BranchQuantStats(
                layer=layer_idx,
                module=name,
                branch="S",
                shape=tuple(sparse_original.shape),
                group_size=args.sparse_group_size,
                numel=sparse_original.numel(),
                mse=float((sparse_original.float() - qs.float()).pow(2).mean().item()),
                max_abs_error=float((sparse_original.float() - qs.float()).abs().max().item()),
                rounded_active_to_zero_before_projection=corrected,
                mask_equal=bool(torch.equal(sparse_mask, qs.eq(0))),
                groups_2_4=groups_2_4,
                violations_2_4=violations,
            )
        )

        for branch, weight, group_size in (
            ("A", module.V, args.a_group_size),
            ("B", module.U, args.b_group_size),
        ):
            original = weight.detach().clone()
            quantized, _, _ = dense_awq_quant(
                original,
                official_pseudo_quantize_tensor,
                w_bit=args.w_bit,
                group_size=group_size,
            )
            weight.data.copy_(quantized)
            stats.append(
                BranchQuantStats(
                    layer=layer_idx,
                    module=name,
                    branch=branch,
                    shape=tuple(original.shape),
                    group_size=group_size,
                    numel=original.numel(),
                    mse=float((original.float() - quantized.float()).pow(2).mean().item()),
                    max_abs_error=float((original.float() - quantized.float()).abs().max().item()),
                )
            )
            del original, quantized
        print(
            f"[layer {layer_idx:02d}] quant {name}: "
            f"S-mask={stats[-3].mask_equal} violations={violations} "
            f"projected={corrected} A=W4G{args.a_group_size} B=W4G{args.b_group_size}",
            flush=True,
        )
        del sparse_original, sparse_mask, qs

    result["seconds"] = time.time() - started
    return result, stats
