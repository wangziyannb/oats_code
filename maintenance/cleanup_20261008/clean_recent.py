"""One-time, audited removal of intermediates already represented in full models."""
from pathlib import Path
import json
import os
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).with_name('recent_cleanup.json')
torch.set_num_threads(16)


def save(report):
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(REPORT)


def main():
    if REPORT.exists():
        raise RuntimeError('This one-time cleanup already has a report; inspect it before rerunning')
    bundles = sorted(ROOT.glob('runs_awq/**/checkpoint.pt')) + sorted(
        ROOT.glob('runs_refine/full_model*/**/checkpoint.pt'))
    assert len(bundles) == 12, len(bundles)
    report = {'status': 'validating', 'verified_models': [], 'delete_files': []}
    candidates = set()
    for path in bundles:
        model = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert model['format'] == 'oats-awq-v1'
        state = model['state_dict']
        for key in ('model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'):
            assert key in state, (path, key)
        assert len(model['module_ranks']) == 224
        assert model['nonpersistent_buffers']
        layers = list(path.parent.glob('layers/oats_chkpt_*.pt'))
        layers += list(path.parent.glob('layers/layer_*.pt'))
        layers += list(path.parent.glob('posthoc_layer_*.pt'))
        assert len(layers) == 32, (path, len(layers))
        checked = 0
        for layer_path in sorted(layers):
            index = int(layer_path.stem.rsplit('_', 1)[1])
            saved = torch.load(layer_path, map_location='cpu', weights_only=True, mmap=True)
            local = saved.get('state_dict', saved)
            prefix = f'model.layers.{index}.'
            actual = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            assert local.keys() == actual.keys(), layer_path
            for key, value in local.items():
                assert value.dtype == actual[key].dtype
                assert torch.equal(value, actual[key]), (layer_path, key)
                checked += 1
            candidates.add(layer_path)
        candidates.update(path.parent.glob('layers/prune_chkpt.pt'))
        candidates.update(path.parent.glob('codes/**/*.pt'))
        report['verified_models'].append({
            'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'layers_matched_exactly': len(layers), 'tensors_matched_exactly': checked,
        })
        print('Verified:', path.relative_to(ROOT), flush=True)
        save(report)
        del model, state, saved, local, actual
    candidates.update(ROOT.glob('runs_refine/first_block_20261007/**/*.pt'))
    protected = set(bundles)
    for path in sorted(candidates):
        assert path.is_file() and not path.is_symlink() and path not in protected
        report['delete_files'].append({
            'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'inode': path.stat().st_ino, 'deleted': False,
        })
    report['status'] = 'deleting'
    report['bytes_to_remove'] = sum(x['bytes'] for x in report['delete_files'])
    save(report)
    for i, item in enumerate(report['delete_files']):
        path = ROOT / item['path']
        st = path.stat()
        assert st.st_ino == item['inode'] and st.st_size == item['bytes']
        path.unlink()
        item['deleted'] = True
        if i % 100 == 0:
            save(report)
    report['status'] = 'complete'
    report['bytes_removed'] = report['bytes_to_remove']
    save(report)
    for model in report['verified_models']:
        assert (ROOT / model['path']).stat().st_size == model['bytes']
    print('Removed bytes:', report['bytes_removed'], 'files:', len(candidates), flush=True)


if __name__ == '__main__':
    main()
