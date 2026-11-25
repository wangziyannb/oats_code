"""Load self-contained archives consolidated from historical OATS layer exports.

This archive utility supports native CompressedLinear models with Llama/GQA and
Qwen3 configurations. It does not change the quantization experiment loader.
"""
from __future__ import annotations

from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.initialization import no_init_weights

from pruning_methods.OATS.compressed_linear import CompressedLinear

FORMAT = 'oats-full-archive-v1'
PROJECTIONS = (
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
    'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
)


def empty_model(config_dict, dtype, module_ranks):
    """Construct the architecture without expensive random weight initialization."""
    values = dict(config_dict)
    model_type = values.pop('model_type')
    if model_type not in ('llama', 'qwen3'):
        raise ValueError(f'Unsupported archived architecture: {model_type}')
    config = AutoConfig.for_model(model_type, **values)
    config._attn_implementation = 'sdpa'
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    expected_names = {
        f'model.layers.{i}.{name}'
        for i in range(config.num_hidden_layers) for name in PROJECTIONS
    }
    if set(module_ranks) != expected_names:
        raise ValueError('Archive module-rank metadata does not cover the architecture')
    for name, rank in module_ranks.items():
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f'Invalid rank for {name}: {rank}')
        original = model.get_submodule(name)
        replacement = CompressedLinear(
            original.in_features, rank, original.out_features,
            original.bias is not None, device='cpu', dtype=dtype,
        )
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent), leaf, replacement)
    return model.eval()


def nonpersistent_buffers(model):
    return {
        f'{name}.{key}' if name else key: value.detach().cpu()
        for name, module in model.named_modules()
        for key in module._non_persistent_buffers_set
        if (value := module._buffers.get(key)) is not None
    }


def load_archive(path, *, mmap=True):
    """Load on CPU with no base-model download, then optionally move to a device."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True, mmap=mmap)
    if payload.get('format') != FORMAT:
        raise ValueError('Not an oats-full-archive-v1 checkpoint')
    model = empty_model(payload['model_config'], getattr(torch, payload['dtype']),
                        payload['module_ranks'])
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    expected = nonpersistent_buffers(model)
    if expected.keys() != payload['nonpersistent_buffers'].keys():
        raise ValueError('Archive nonpersistent buffers do not match the architecture')
    for name, value in payload['nonpersistent_buffers'].items():
        if value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid archived buffer: {name}')
        parent, _, key = name.rpartition('.')
        model.get_submodule(parent)._buffers[key] = value
    if model.config.tie_word_embeddings:
        if not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
            raise ValueError('Tied input/output embedding weights differ')
        model.tie_weights()
    metadata = {k: v for k, v in payload.items()
                if k not in ('state_dict', 'nonpersistent_buffers')}
    return model.eval(), metadata
