"""Consolidate legacy final models, verify exact preservation, then remove duplicates."""
from pathlib import Path
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open
from maintenance.full_model_archive import (
    FORMAT, PROJECTIONS, empty_model, load_archive, nonpersistent_buffers,
)

REPORT = Path(__file__).with_name('legacy_cleanup.json')
HUB = Path('/home/zwang53/.cache/huggingface/hub')
BASES = {
    'llama2': ('meta-llama--Llama-2-7b-hf', '01c7f73d771dfac7d292323805ebc428287df4f9'),
    'llama3.1': ('meta-llama--Llama-3.1-8B', 'd04e592bb4f6aa9cfee91e2e20afa771667e1d4b'),
    'qwen3_1.7b': ('Qwen--Qwen3-1.7B-Base', 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'),
    'qwen3_8b': ('Qwen--Qwen3-8B-Base', '49e3418fbbbca6ecbdf9608b4d22e5a407081db4'),
}
DIRECTORIES = [
    'old/llama2_oats_24_r32', 'old/llama2_oats_24_r64', 'old/llama2_oats_24_r128',
    'llama3.1_oats_24_r32', 'old/llama3.1_oats_24_r128',
    'qwen3_1.7b_oats_24_r32', 'old/qwen3_1.7b_oats_24_r128',
    'qwen3_8b_oats_24_r32', 'old/qwen3_8b_oats_24_r64', 'old/qwen3_8b_oats_24_r128',
]
torch.set_num_threads(16)


def save_report(report):
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(REPORT)


@torch.inference_mode()
def consolidate(directory, report):
    root = ROOT / directory
    destination = root / 'checkpoint.pt'
    assert not destination.exists(), destination
    family = root.name.split('_oats_')[0]
    rank = int(root.name.rsplit('_r', 1)[1])
    repository, revision = BASES[family]
    snapshot = HUB / ('models--' + repository) / 'snapshots' / revision
    config = json.loads((snapshot / 'config.json').read_text())
    layers_count = config['num_hidden_layers']
    resume = root / 'prune_chkpt.pt'
    progress = torch.load(resume, map_location='cpu', weights_only=True, mmap=True)
    assert int(progress['prune_start_idx']) == layers_count - 1, root
    del progress
    layer_paths = [root / f'oats_chkpt_{i}.pt' for i in range(layers_count)]
    assert len(list(root.glob('oats_chkpt_*.pt'))) == layers_count
    exports = list(root.glob('*param_dict*.pt'))
    assert len(exports) == 1, (root, exports)
    exported = torch.load(exports[0], map_location='cpu', weights_only=True, mmap=True)
    ranks = {f'model.layers.{i}.{p}': rank for i in range(layers_count) for p in PROJECTIONS}
    model = empty_model(config, torch.float16, ranks)
    handles = [safe_open(str(p), framework='pt', device='cpu')
               for p in sorted(snapshot.glob('*.safetensors'))]
    base = {key: handle for handle in handles for key in handle.keys()}
    state = {}
    matched_norms = 0
    matched_exports = 0
    sparse_violations = 0
    for i, path in enumerate(layer_paths):
        local = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        expected = model.model.layers[i].state_dict()
        assert local.keys() == expected.keys(), path
        for key, value in local.items():
            assert value.shape == expected[key].shape and value.dtype == torch.float16, (path, key)
            assert torch.isfinite(value).all(), (path, key)
            full_key = f'model.layers.{i}.{key}'
            if key.endswith(('.S', '.U', '.V')):
                suffix = {'S': 'sp.weight', 'V': 'lr.l1.weight', 'U': 'lr.l2.weight'}[key[-1]]
                old_key = full_key[:-1] + suffix
                assert torch.equal(value, exported[old_key]), (path, old_key)
                matched_exports += 1
                if key.endswith('.S'):
                    sparse_violations += int((value.reshape(-1, 4).ne(0).sum(-1) > 2).sum())
            else:
                assert torch.equal(value, base[full_key].get_tensor(full_key).to(value.dtype)), full_key
                matched_norms += 1
            state[full_key] = value
    assert matched_exports == len(exported) == layers_count * 7 * 3
    assert sparse_violations == 0, sparse_violations
    for key, value in model.state_dict().items():
        if key in state:
            continue
        assert not key.startswith('model.layers.'), key
        base_key = key
        if key == 'lm_head.weight' and key not in base and config.get('tie_word_embeddings'):
            base_key = 'model.embed_tokens.weight'
        state[key] = base[base_key].get_tensor(base_key).to(torch.float16)
        assert state[key].shape == value.shape and torch.isfinite(state[key]).all(), key
    if config.get('tie_word_embeddings'):
        assert torch.equal(state['lm_head.weight'], state['model.embed_tokens.weight'])
        state['lm_head.weight'] = state['model.embed_tokens.weight']
    model.load_state_dict(state, strict=True, assign=True)
    probe_tokens = torch.tensor([[1, 17, 42, 9]])
    probe_logits = model(probe_tokens, use_cache=False).logits.detach().clone()
    assert torch.isfinite(probe_logits).all(), root
    buffers = nonpersistent_buffers(model)
    metadata = {
        'source_directory': directory, 'source_export': str(exports[0].relative_to(ROOT)),
        'base_model': repository.replace('--', '/'), 'base_revision': revision,
        'base_config_source': str(snapshot / 'config.json'),
        'preservation': 'All saved OATS layer tensors and exported S/U/V are bitwise unchanged.',
        'base_reconstruction': 'Embedding, final norm, and output head restored from the locally cached base model.',
        'base_identity_check': 'Every saved uncompressed layer norm matches this base snapshot exactly after FP16 casting.',
        'historical_buffer_limit': 'Historical files did not save nonpersistent rotary buffers; regenerated from the base configuration under current Transformers, then archived explicitly.',
        'historical_ppl': None, 'reoptimized': False, 'requantized': False,
        'probe_tokens': probe_tokens.tolist(), 'probe_logits': probe_logits,
        'torch_version': str(torch.__version__),
    }
    payload = {
        'format': FORMAT, 'method': 'legacy_oats', 'rank': rank,
        'model_config': model.config.to_dict(), 'dtype': 'float16',
        'module_ranks': ranks, 'state_dict': state, 'nonpersistent_buffers': buffers,
        'metadata': metadata, 'fake_quantization': False, 'packed_int4': False,
    }
    temporary = destination.with_suffix('.pt.tmp')
    assert not temporary.exists(), temporary
    print('Saving:', directory, flush=True)
    torch.save(payload, temporary)
    restored, _ = load_archive(temporary)
    restored_state = restored.state_dict()
    for key, value in state.items():
        assert torch.equal(value, restored_state[key]), (root, key)
    for key, value in buffers.items():
        assert torch.equal(value, nonpersistent_buffers(restored)[key]), (root, key)
    restored_logits = restored(probe_tokens, use_cache=False).logits
    assert torch.equal(restored_logits, probe_logits), root
    temporary.replace(destination)
    entry = {
        'path': str(destination.relative_to(ROOT)), 'bytes': destination.stat().st_size,
        'base_model': metadata['base_model'], 'base_revision': revision, 'rank': rank,
        'layers': layers_count, 'export_tensors_exact': matched_exports,
        'base_norm_tensors_exact': matched_norms, 'reloaded_tensors_exact': len(state),
        'sparse_violations': sparse_violations, 'reload_logits_exact': True,
        'deleted_files': [],
    }
    removal = layer_paths + exports + [resume]
    for path in removal:
        entry['deleted_files'].append({
            'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'inode': path.stat().st_ino, 'deleted': False,
        })
    report['models'].append(entry)
    save_report(report)
    # Drop mapped source tensors before unlinking their files.
    del restored, restored_state, model, state, payload, local, expected, exported, value
    gc.collect()
    for item in entry['deleted_files']:
        path = ROOT / item['path']
        st = path.stat()
        assert st.st_ino == item['inode'] and st.st_size == item['bytes']
        path.unlink()
        item['deleted'] = True
    entry['bytes_removed'] = sum(x['bytes'] for x in entry['deleted_files'])
    save_report(report)
    print('Consolidated and verified:', directory, 'net bytes freed:',
          entry['bytes_removed'] - entry['bytes'], flush=True)


if __name__ == '__main__':
    if REPORT.exists():
        raise RuntimeError('This cleanup has a report already; inspect it before rerunning')
    report = {'status': 'running', 'models': []}
    save_report(report)
    for directory in DIRECTORIES:
        consolidate(directory, report)
    report['status'] = 'complete'
    report['bytes_removed'] = sum(x['bytes_removed'] for x in report['models'])
    report['bytes_written'] = sum(x['bytes'] for x in report['models'])
    report['net_bytes_freed'] = report['bytes_removed'] - report['bytes_written']
    save_report(report)
    print('Finished:', report['net_bytes_freed'], 'net bytes freed', flush=True)
