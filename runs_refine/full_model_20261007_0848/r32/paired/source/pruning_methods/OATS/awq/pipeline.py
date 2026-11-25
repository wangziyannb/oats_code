"""Model IO and orchestration for native OATS + quantization."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from pruning_methods.OATS.compressed_linear import CompressedLinear
from .posthoc import (PROJECTION_NAMES, capture_first_layer_input,
                      capture_projection_inputs, count_2_4, move_tree, process_layer)
from .reference.quantizer import pseudo_quantize_tensor
from .quantize import validate_tensor

FORMAT = 'oats-awq-v1'
DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def atomic_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(destination)


def write_json(destination, payload):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)


def claim_run(directory, manifest, resume=False):
    """A checkpoint can only be resumed under its original experiment settings."""
    directory = Path(directory)
    path = directory / 'manifest.json'
    if path.exists():
        if not resume:
            raise ValueError(f'Output already exists: {directory}; use a new directory or --resume')
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Resume configuration does not match the saved manifest')
        if (directory / 'checkpoint.pt').exists():
            raise ValueError('This run is already complete')
    else:
        if resume:
            raise ValueError('No manifest exists to resume')
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f'Output directory is not empty: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)


def validate_architecture(model):
    cfg = model.config
    if cfg.model_type != 'llama' or cfg.num_key_value_heads != cfg.num_attention_heads:
        raise ValueError('Version 1 supports Llama2-style separate projections without GQA')
    for layer in model.model.layers:
        for name in PROJECTION_NAMES:
            module = layer.get_submodule(name)
            if not isinstance(module, (torch.nn.Linear, CompressedLinear)):
                raise TypeError(f'Unsupported projection: {name}')


def audit_model(model, rank, groups=False):
    validate_architecture(model)
    if rank not in (32, 128):
        raise ValueError('Supported ranks are 32 and 128')
    report = {'modules': 0, 'groups_2_4': 0, 'violations_2_4': 0}
    for index, layer in enumerate(model.model.layers):
        for name in PROJECTION_NAMES:
            module = layer.get_submodule(name)
            if not isinstance(module, CompressedLinear) or module.rank != rank:
                raise ValueError(f'Layer {index} {name}: expected OATS rank {rank}')
            if module.V.shape != (rank, module.in_features) or module.U.shape != (module.out_features, rank):
                raise ValueError(f'Layer {index} {name}: invalid factor shapes')
            for key in ('S', 'V', 'U'):
                validate_tensor(getattr(module, key), f'{index}.{name}.{key}', ndim=2)
            if module.in_features % 4:
                raise ValueError(f'{name}: input width is not divisible by 4')
            if groups and module.in_features % 128:
                raise ValueError(f'{name}: input width is not divisible by group size 128')
            n, violations = count_2_4(module.S)
            report['modules'] += 1
            report['groups_2_4'] += n
            report['violations_2_4'] += violations
    if report['violations_2_4']:
        raise ValueError(f'Invalid 2:4 sparse backbone: {report}')
    return report


def replace_from_state(layer, state, rank):
    """Reconstruct shapes against the base architecture, not a directory name."""
    for name in PROJECTION_NAMES:
        original = layer.get_submodule(name)
        expected = {'S': (original.out_features, original.in_features),
                    'V': (rank, original.in_features), 'U': (original.out_features, rank)}
        for key, shape in expected.items():
            value = state.get(f'{name}.{key}')
            if value is None or tuple(value.shape) != shape:
                raise ValueError(f'Missing or incompatible {name}.{key}; expected {shape}')
            validate_tensor(value, f'{name}.{key}', ndim=2)
        new = CompressedLinear(original.in_features, rank, original.out_features,
                               original.bias is not None, device='cpu', dtype=state[f'{name}.S'].dtype)
        parent_name, _, leaf = name.rpartition('.')
        setattr(layer.get_submodule(parent_name), leaf, new)
    layer.load_state_dict(state, strict=True)


def load_legacy_layers(model, directory, rank, partial=False):
    directory = Path(directory)
    state = torch.load(directory / 'prune_chkpt.pt', map_location='cpu', weights_only=True)
    completed = int(state['prune_start_idx'])
    total = len(model.model.layers)
    if completed < 0 or completed >= total or (not partial and completed != total - 1):
        raise ValueError('OATS source checkpoint is incomplete or has an invalid layer index')
    for index in range(completed + 1):
        values = torch.load(directory / f'oats_chkpt_{index}.pt', map_location='cpu', weights_only=True)
        if next(iter(values.values())).dtype != next(model.parameters()).dtype:
            raise ValueError('Source checkpoint dtype differs from the requested model dtype')
        replace_from_state(model.model.layers[index], values, rank)
    if not partial:
        audit_model(model, rank)
    return model


def save_bundle(model, path, rank, method, metadata):
    audit = audit_model(model, rank)
    payload = {
        'format': FORMAT, 'method': method, 'rank': rank,
        'model_config': model.config.to_dict(),
        'dtype': next(model.parameters()).dtype.__str__().removeprefix('torch.'),
        'module_ranks': {f'model.layers.{i}.{name}': rank
                         for i in range(len(model.model.layers)) for name in PROJECTION_NAMES},
        'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        # HF rotary frequencies are nonpersistent and may have been dtype-cast.
        # Preserve the actual inference state instead of silently regenerating it.
        'nonpersistent_buffers': {
            f'{name}.{key}' if name else key: module._buffers[key].detach().cpu()
            for name, module in model.named_modules()
            for key in module._non_persistent_buffers_set
            if module._buffers.get(key) is not None
        },
        'metadata': metadata, 'audit': audit,
        'fake_quantization': method in ('posthoc', 'inloop', 'refine_awq', 'refine_coordinate', 'refine_paired'), 'packed_int4': False,
        'scales_already_applied': True,
    }
    atomic_save(payload, path)


def load_bundle(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Not a complete OATS AWQ checkpoint')
    cfg = LlamaConfig.from_dict(payload['model_config'])
    cfg._attn_implementation = 'sdpa'
    dtype = getattr(torch, payload['dtype'])
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = LlamaForCausalLM(cfg)
    finally:
        torch.set_default_dtype(previous_dtype)
    expected_ranks = {f'model.layers.{i}.{name}': payload['rank']
                      for i in range(len(model.model.layers)) for name in PROJECTION_NAMES}
    if payload['module_ranks'] != expected_ranks:
        raise ValueError('Checkpoint module-rank metadata does not match the architecture')
    for index, layer in enumerate(model.model.layers):
        prefix = f'model.layers.{index}.'
        state = {key[len(prefix):]: value for key, value in payload['state_dict'].items() if key.startswith(prefix)}
        replace_from_state(layer, state, payload['rank'])
    model.load_state_dict(payload['state_dict'], strict=True)
    for name, value in payload['nonpersistent_buffers'].items():
        parent, _, key = name.rpartition('.')
        module = model.get_submodule(parent)
        if key not in module._non_persistent_buffers_set:
            raise ValueError(f'Unexpected nonpersistent buffer in checkpoint: {name}')
        module._buffers[key] = value
    model.eval()
    audit_model(model, payload['rank'])
    return model, {key: value for key, value in payload.items() if key != 'state_dict'}


@torch.no_grad()
def quantize_posthoc(model, samples, args, output=None, resume=False):
    """Quantize completed OATS layers, retaining the archive's teacher propagation."""
    audit_model(model, args.rank, groups=True)
    original_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = False
    device = torch.device(args.device)
    layers_report, statistics = [], []
    try:
        layer_input, layer_kwargs = capture_first_layer_input(model, samples, device)
        for index, layer in enumerate(model.model.layers):
            layer.to(device)
            # Capture BEFORE quantization, including when replaying a completed layer.
            next_input, features = capture_projection_inputs(
                layer, layer_input.to(device), move_tree(layer_kwargs, device))
            layer_file = Path(output) / f'posthoc_layer_{index}.pt' if output else None
            if resume and layer_file and layer_file.exists():
                saved = torch.load(layer_file, map_location='cpu', weights_only=True)
                layer.load_state_dict(saved['state_dict'], strict=True)
                result, stats = saved['report'], saved['statistics']
            else:
                result, raw_stats = process_layer(index, layer, features,
                    move_tree(layer_kwargs, device), pseudo_quantize_tensor, args)
                stats = [asdict(stat) for stat in raw_stats]
                if any(s['branch'] == 'S' and (not s['mask_equal'] or s['violations_2_4']) for s in stats):
                    raise ValueError('Post-hoc quantization changed a sparse mask')
                if layer_file:
                    atomic_save({'state_dict': {k: v.detach().cpu() for k, v in layer.state_dict().items()},
                                 'report': result, 'statistics': stats}, layer_file)
            layers_report.append(result)
            statistics.extend(stats)
            layer.cpu()
            layer_input = next_input.cpu()
            del next_input, features
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    finally:
        model.config.use_cache = original_cache
    return {'method': 'posthoc', 'layers': layers_report, 'statistics': statistics,
            'audit': audit_model(model, args.rank), 'clipping': not args.skip_clip,
            'calibration_propagation': 'pre_quantization_block_output',
            'coordinate_system': 'scales_fused_into_full_model', 'packed_int4': False}


@torch.no_grad()
def evaluate_tokens(model, tokens, seqlen=128):
    """Nonoverlapping WikiText-style PPL with shifted targets, batch size one."""
    if seqlen < 2:
        raise ValueError('PPL sequence length must be at least two')
    tokens = tokens.reshape(-1)
    blocks = tokens.numel() // seqlen
    if blocks == 0:
        raise ValueError('Not enough evaluation tokens for one complete block')
    model.eval()
    device = next(model.parameters()).device
    nll, count = 0.0, 0
    for index in range(blocks):
        batch = tokens[index * seqlen:(index + 1) * seqlen].unsqueeze(0).to(device)
        logits = model(batch, use_cache=False).logits.float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               batch[:, 1:].reshape(-1), reduction='sum')
        nll += float(loss)
        count += seqlen - 1
    ppl = float(torch.exp(torch.tensor(nll / count, dtype=torch.float64)))
    if not torch.isfinite(torch.tensor(ppl)):
        raise ValueError('Non-finite perplexity')
    return {'ppl': ppl, 'predicted_tokens': count, 'seqlen': seqlen, 'batch_size': 1}
