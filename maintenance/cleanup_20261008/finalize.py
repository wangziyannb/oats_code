"""Read-only validation of retained models, followed by a compact cleanup report."""
from pathlib import Path
import json
import os
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).parent


def main():
    recent = json.loads((REPORT_DIR / 'recent_cleanup.json').read_text())
    legacy = json.loads((REPORT_DIR / 'legacy_cleanup.json').read_text())
    assert recent['status'] == legacy['status'] == 'complete'
    assert len(recent['verified_models']) == 12 and len(legacy['models']) == 10
    deleted = recent['delete_files'] + [
        entry for model in legacy['models'] for entry in model['deleted_files']]
    for entry in deleted:
        assert entry['deleted'] and not (ROOT / entry['path']).exists(), entry['path']
    models = []
    for info in recent['verified_models'] + legacy['models']:
        path = ROOT / info['path']
        assert path.stat().st_size == info['bytes'], path
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        state = data['state_dict']
        assert all(k in state for k in ('model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'))
        layer_count = data['model_config']['num_hidden_layers']
        assert len(data['module_ranks']) == layer_count * 7
        for name, rank in data['module_ranks'].items():
            s, u, v = (state[name + '.' + part] for part in ('S', 'U', 'V'))
            assert s.shape == (u.shape[0], v.shape[1])
            assert u.shape[1] == v.shape[0] == rank == data['rank']
        models.append({
            'path': info['path'], 'bytes': info['bytes'], 'format': data['format'],
            'method': data['method'], 'rank': data['rank'], 'dtype': data['dtype'],
            'layers': layer_count, 'model_type': data['model_config']['model_type'],
            'load_function': ('maintenance.full_model_archive.load_archive'
                              if data['format'] == 'oats-full-archive-v1'
                              else 'pruning_methods.OATS.awq.pipeline.load_bundle'),
        })
        del data, state, s, u, v
    leftovers = []
    for pattern in ('**/oats_chkpt_*.pt', '**/posthoc_layer_*.pt', '**/prune_chkpt.pt',
                    'runs_refine/**/codes/**/*.pt', 'runs_refine/**/layers/layer_*.pt',
                    'runs_refine/first_block_20261007/**/*.pt', '**/*param_dict*.pt', '**/*.pt.tmp'):
        leftovers.extend(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern))
    assert not leftovers, leftovers
    allocated = int(subprocess.check_output(['du', '-s', '-B1', str(ROOT)], text=True).split()[0])
    apparent = int(subprocess.check_output(['du', '-s', '-b', str(ROOT)], text=True).split()[0])
    summary = {
        'status': 'complete', 'date': '2026-10-08',
        'complete_model_count': len(models),
        'complete_model_bytes': sum(x['bytes'] for x in models),
        'original_complete_models_unchanged': 12,
        'legacy_exports_consolidated_and_verified': 10,
        'deleted_files_count': len(deleted),
        'deleted_file_bytes': sum(x['bytes'] for x in deleted),
        'new_legacy_archive_bytes': legacy['bytes_written'],
        'net_artifact_bytes_freed': recent['bytes_removed'] + legacy['net_bytes_freed'],
        'repository_before_allocated_bytes': 682337751040,
        'repository_before_apparent_bytes': 682313742364,
        'repository_after_allocated_bytes': allocated,
        'repository_after_apparent_bytes': apparent,
        'remaining_targeted_intermediates': leftovers,
        'preserved': ['complete model weights/configs', 'source and environment',
                      'experiment JSON/CSV/logs', 'input calibration data', 'small reload probes'],
        'models': models,
    }
    (REPORT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'models'}, indent=2))


if __name__ == '__main__':
    main()
