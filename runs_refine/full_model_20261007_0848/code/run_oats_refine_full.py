#!/usr/bin/env python3
"""Full-model fixed-mask quantization refinement with quantized block propagation.

Starts from completed native OATS factors; does not re-run SVD or change masks.
Each arm has its own calibration trajectory. Evaluation strictly reloads the
complete FP checkpoint, and uses the original batch-one WikiText PPL routine.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F

from general_utils.config import config
from general_utils.utils import map_tensors
from models.adapters.llama_adapter import LlamaModelAdapter
from models.model_utils import get_layer0_inputs
from pruning_methods.OATS.awq.pipeline import (
    DTYPES, atomic_save, audit_model, claim_run, evaluate_tokens, load_bundle,
    replace_from_state, save_bundle, validate_architecture, write_json,
)
from pruning_methods.OATS.awq.posthoc import PROJECTION_NAMES
from pruning_methods.OATS.awq.refine import (
    CodeMatrix, DiagonalRefiner, RefineConfig, initialize_awq,
)


WORKER_DEVICE = None


def init_worker(queue):
    global WORKER_DEVICE
    WORKER_DEVICE = queue.get()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.device(WORKER_DEVICE).type == 'cuda':
        torch.cuda.set_device(torch.device(WORKER_DEVICE))


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def source_files():
    root = Path(__file__).resolve().parent
    names = ['run_oats_refine_full.py', 'run_oats_awq.py']
    names += [str(p.relative_to(root)) for folder in ('pruning_methods/OATS', 'models', 'general_utils')
              for p in sorted((root / folder).rglob('*.py'))]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


@torch.no_grad()
def projection_metrics(ref, samples, dtype):
    target = F.linear(samples.float(), ref.w)
    x = samples.to(dtype)
    actual = F.linear(x, ref.s.to(dtype)) + F.linear(F.linear(x, ref.v.to(dtype)), ref.u.to(dtype))
    if not torch.isfinite(actual).all():
        raise ValueError('Nonfinite projection output in storage dtype')
    return {'objective': ref.objective(), 'sample_output_mse': float((actual.float() - target).square().mean()),
            'mask_preserved': ref.storage_valid()}


@torch.no_grad()
def solve_projection(task, device=None):
    device = torch.device(device or WORKER_DEVICE)
    dtype = getattr(torch, task['settings']['storage_dtype'])
    w, s, u, v, h, x = [task[k].to(device=device, dtype=torch.float32)
                         for k in ('weight', 'S', 'U', 'V', 'diagonal', 'samples')]
    started = time.perf_counter()
    grids, initialization = initialize_awq(w, s, u, v, x, 128, task['scale_grid'])
    ref = DiagonalRefiner(w, h, grids, RefineConfig(**task['settings']))
    before = projection_metrics(ref, x, dtype)
    init_seconds = time.perf_counter() - started
    records = []
    started = time.perf_counter()
    if task['arm'] != 'awq':
        ref.run(paired=task['arm'] == 'paired', callback=records.append)
    after = projection_metrics(ref, x, dtype)
    if not torch.equal(ref.s.to(dtype).ne(0), s.ne(0)):
        raise ValueError('Original OATS sparse mask was changed')
    if after['objective'] > before['objective']:
        raise ValueError('Refinement objective increased')
    path = Path(task['artifact'])
    atomic_save(ref.state_dict(), path)
    restored = torch.load(path, map_location='cpu', weights_only=True)
    factors = {}
    for name, expected in zip(('S', 'U', 'V'), (ref.s, ref.u, ref.v)):
        actual = CodeMatrix.from_state_dict(restored['grids'][name], device).values()
        if not torch.equal(expected, actual):
            raise ValueError('Code artifact did not round-trip exactly')
        factors[name] = actual.to(dtype).cpu()
    path.with_suffix('.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    result = {'name': task['name'], 'device': str(device), 'initialization': initialization,
              'initialization_seconds': init_seconds, 'refinement_and_io_seconds': time.perf_counter() - started,
              'before': before, 'after': after, 'phases': len(records),
              'rejected_phases': sum(not r['accepted'] for r in records), 'reload_exact': True}
    write_json(path.with_suffix('.json'), result)
    return {'report': result, 'factors': factors}


@torch.no_grad()
def capture_statistics(layer, positional, keyword, device, tokens, sequences):
    """All tokens determine H; only uniformly sampled tokens enter AWQ search."""
    layer.to(device)
    quota = max(1, (tokens + sequences - 1) // sequences)
    states = {name: {'sum': torch.zeros(layer.get_submodule(name).in_features, device=device, dtype=torch.float64),
                     'count': 0, 'sample_count': 0, 'samples': []} for name in PROJECTION_NAMES}

    def hook(name):
        def collect(module, inputs, output):
            x = inputs[0].detach()
            if not torch.isfinite(x).all():
                raise ValueError(f'Nonfinite calibration activation: {name}')
            state = states[name]
            state['sum'] += x.float().square().reshape(-1, x.shape[-1]).sum(0, dtype=torch.float64)
            state['count'] += x.numel() // x.shape[-1]
            for sequence in x:
                take = min(quota, tokens - state['sample_count'], sequence.shape[0])
                if take > 0:
                    indices = torch.linspace(0, sequence.shape[0] - 1, take, device=device).round().long()
                    state['samples'].append(sequence.index_select(0, indices).cpu())
                    state['sample_count'] += take
        return collect

    handles = [layer.get_submodule(name).register_forward_hook(hook(name)) for name in PROJECTION_NAMES]
    try:
        for args, kwargs in zip(positional, keyword):
            args, kwargs = map_tensors([args, kwargs], device=device)
            layer(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        layer.cpu()
    return {name: {'diagonal': (state['sum'] / state['count']).float().clamp_min(1e-8).cpu(),
                   'samples': torch.cat(state['samples']), 'input_tokens': state['count']}
            for name, state in states.items()}


@torch.no_grad()
def propagate(layer, positional, keyword, device):
    layer.to(device)
    result = []
    try:
        for args, kwargs in zip(positional, keyword):
            moved_args, moved_kwargs = map_tensors([args, kwargs], device=device)
            out = layer(*moved_args, **moved_kwargs)
            out = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(out).all():
                raise ValueError('Nonfinite quantized block output')
            result.append((out.cpu(), *args[1:]))
    finally:
        layer.cpu()
    return result


@torch.no_grad()
def refine_model(model, baseline, loader, output, *, rank, arm, settings, device,
                 tokens=512, scale_grid=20, pool=None, resume=False, on_layer=None):
    """Replay committed blocks on resume; never apply stored scales twice."""
    validate_architecture(model)
    output, baseline = Path(output), Path(baseline)
    (output / 'layers').mkdir(parents=True, exist_ok=True)
    model.eval()
    original_cache = model.config.use_cache
    model.config.use_cache = False
    config.device, config.dtype = torch.device(device), next(model.parameters()).dtype
    adapter = LlamaModelAdapter(model)
    positional, keyword, sequences = [], [], 0
    for batch in loader:
        _, args, kwargs = get_layer0_inputs(adapter, batch)
        positional.append(args)
        keyword.append(kwargs)
        sequences += batch['input_ids'].shape[0]
    if not sequences:
        raise ValueError('Empty calibration data')
    reports = []
    started = time.perf_counter()
    try:
        for index, layer in enumerate(model.model.layers):
            checkpoint = output / 'layers' / f'layer_{index}.pt'
            if resume and checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if saved['rank'] != rank or saved['arm'] != arm or saved['settings'] != asdict(settings):
                    raise ValueError('Layer checkpoint configuration mismatch')
                replace_from_state(layer, saved['state_dict'], rank)
                report = saved['report']
                print(f'[replay] {arm} rank={rank} layer={index}', flush=True)
            else:
                layer_started = time.perf_counter()
                stats = capture_statistics(layer, positional, keyword, device, tokens, sequences)
                state = torch.load(baseline / f'oats_chkpt_{index}.pt', map_location='cpu', weights_only=True)
                tasks = []
                for name in PROJECTION_NAMES:
                    module = layer.get_submodule(name)
                    s, u, v = (state[f'{name}.{key}'] for key in ('S', 'U', 'V'))
                    if s.shape != module.weight.shape or u.shape != (module.out_features, rank) or v.shape != (rank, module.in_features):
                        raise ValueError(f'Invalid baseline rank/shape at layer {index} {name}')
                    if any(t.dtype != config.dtype for t in (s, u, v)):
                        raise ValueError('Baseline dtype differs from requested model dtype')
                    tasks.append({'name': name, 'weight': module.weight.detach().cpu(), 'S': s, 'U': u, 'V': v,
                                  **stats[name], 'arm': arm, 'settings': asdict(settings), 'scale_grid': scale_grid,
                                  'artifact': str(output / 'codes' / f'layer_{index}' / (name + '.pt'))})
                if pool is None:
                    solved = [solve_projection(task, device) for task in tasks]
                else:
                    futures = [pool.submit(solve_projection, task) for task in sorted(tasks, key=lambda t: t['weight'].numel(), reverse=True)]
                    solved = [future.result() for future in as_completed(futures)]
                by_name = {item['report']['name']: item for item in solved}
                for name in PROJECTION_NAMES:
                    for key, value in by_name[name]['factors'].items():
                        state[f'{name}.{key}'] = value
                replace_from_state(layer, state, rank)
                report = {'layer': index, 'arm': arm, 'rank': rank,
                          'input_tokens_per_projection': {k: v['input_tokens'] for k, v in stats.items()},
                          'projections': [by_name[name]['report'] for name in PROJECTION_NAMES],
                          'capture_and_refinement_seconds': time.perf_counter() - layer_started}
                del tasks, solved, by_name, stats, state
                # Propagation must succeed before this block becomes resumable.
            positional = propagate(layer, positional, keyword, device)
            if not (resume and checkpoint.exists()):
                atomic_save({'rank': rank, 'arm': arm, 'settings': asdict(settings), 'report': report,
                             'state_dict': {k: v.detach().cpu() for k, v in layer.state_dict().items()}}, checkpoint)
            reports.append(report)
            write_json(output / 'status.json', {'status': 'refining', 'rank': rank, 'arm': arm,
                       'completed_layers': index + 1, 'total_layers': len(model.model.layers),
                       'elapsed_seconds': time.perf_counter() - started})
            print(f'[layer done] {arm} rank={rank} {index + 1}/{len(model.model.layers)} elapsed={time.perf_counter() - started:.1f}s', flush=True)
            if on_layer is not None:
                on_layer(index, positional)
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
    finally:
        model.config.use_cache = original_cache
    return {'method': 'refine_' + arm, 'layers': reports, 'audit': audit_model(model, rank),
            'calibration_sequences': sequences, 'calibration_propagation': 'quantized_block_output',
            'capture': 'dense_current_block_on_quantized_prefix_inputs', 'packed_int4': False,
            'baseline_masks_and_ranks_fixed': True, 'refinement_seconds': time.perf_counter() - started}


def manifest_for(args):
    baseline = args.baseline_dir.resolve()
    source_manifest = json.loads((baseline / 'manifest.json').read_text())
    if source_manifest.get('mode') != 'prepare' or source_manifest['rank'] != args.rank or source_manifest['dtype'] != args.dtype:
        raise ValueError('Expected an unquantized OATS prepare source with matching rank/dtype')
    if Path(source_manifest['model']).resolve() != Path(args.model).resolve():
        raise ValueError('Base model differs from the OATS source')
    if source_manifest['calibration_path_identity'] != identity(args.calibration_path):
        raise ValueError('Calibration source differs from the OATS source')
    model_hash = hashlib.sha256((Path(args.model) / 'config.json').read_bytes()).hexdigest()
    if model_hash != source_manifest['model_config_sha256']:
        raise ValueError('Base model configuration differs from the OATS source')
    model_weights = [identity(p) for p in sorted(Path(args.model).glob('*.safetensors')) + sorted(Path(args.model).glob('pytorch_model*.bin'))]
    if model_weights != source_manifest['model_weights']:
        raise ValueError('Base model weights differ from the OATS source')
    layers = baseline / 'layers'
    done = torch.load(layers / 'prune_chkpt.pt', map_location='cpu', weights_only=True)
    total = json.loads((Path(args.model) / 'config.json').read_text())['num_hidden_layers']
    if done['prune_start_idx'] != total - 1 or not (baseline / 'checkpoint.pt').exists():
        raise ValueError('OATS source is incomplete')
    values = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('resume', 'output_dir', 'dry_run')}
    values.update(source_sha256=source_files(), baseline_manifest=source_manifest,
                  source_layers=[identity(layers / f'oats_chkpt_{i}.pt') for i in range(total)],
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  objective='fixed_diagonal_weighted_reconstruction',
                  quantization={'bits': 4, 'groups': {'S': 128, 'V': 128, 'U': min(args.rank, 128)},
                                'clipping': False, 'neighbor_fusion': False, 'fixed_masks': True},
                  calibration='all supplied C4 sequences; all tokens for H; uniform samples for AWQ',
                  calibration_propagation='quantized_block_output', torch_version=str(torch.__version__))
    return values


def run(args):
    from general_utils.data_utils import prepare_presampled_dataloader
    from transformers import LlamaForCausalLM
    manifest = manifest_for(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    claim_run(args.output_dir, manifest, args.resume)
    for name in manifest['source_sha256']:
        target = args.output_dir / 'source' / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(__file__).parent / name, target)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.devices[0]
    if torch.device(device).type == 'cuda':
        torch.cuda.set_device(torch.device(device))
    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=DTYPES[args.dtype], attn_implementation='sdpa').eval()
    loader = prepare_presampled_dataloader(str(args.calibration_path), batch_size=args.batch_size)
    settings = RefineConfig(args.sweeps, True, args.pair_candidates, storage_dtype=str(DTYPES[args.dtype]).removeprefix('torch.'))
    context = mp.get_context('spawn')
    queue = context.Queue()
    for worker_device in args.devices:
        queue.put(worker_device)
    try:
        with ProcessPoolExecutor(len(args.devices), mp_context=context, initializer=init_worker, initargs=(queue,)) as pool:
            report = refine_model(model, args.baseline_dir / 'layers', loader, args.output_dir,
                                  rank=args.rank, arm=args.arm, settings=settings, device=device,
                                  tokens=args.tokens, scale_grid=args.scale_grid, pool=pool, resume=args.resume)
    finally:
        queue.close()
        queue.join_thread()
    probe = next(iter(loader))['input_ids'][:1, :128]
    model.to(device)
    with torch.no_grad():
        logits = model(probe.to(device), use_cache=False).logits.cpu()
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite full-model probe logits')
    model.cpu()
    atomic_save({'input_ids': probe, 'logits': logits}, args.output_dir / 'reload_probe.pt')
    save_bundle(model, args.output_dir / 'checkpoint.pt', args.rank, 'refine_' + args.arm,
                {'base_model': args.model, 'manifest': manifest, 'report': report})
    write_json(args.output_dir / 'quant_report.json', report)
    write_json(args.output_dir / 'status.json', {'status': 'complete', 'rank': args.rank, 'arm': args.arm,
               'completed_layers': len(model.model.layers), 'total_layers': len(model.model.layers)})
    print(f'Saved complete refined model: {args.output_dir / "checkpoint.pt"}', flush=True)


def evaluate(args):
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model, metadata = load_bundle(args.checkpoint)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model.to(device)
    probe = torch.load(args.checkpoint.parent / 'reload_probe.pt', map_location='cpu', weights_only=True)
    with torch.no_grad():
        actual = model(probe['input_ids'].to(device), use_cache=False).logits.cpu()
    if not torch.equal(actual, probe['logits']):
        raise ValueError(f'Full model reload logits differ: max={float((actual - probe["logits"]).abs().max())}')
    tokens = torch.load(args.eval_tokens, map_location='cpu', weights_only=True)
    started = time.perf_counter()
    metrics = evaluate_tokens(model, tokens, args.ppl_seqlen)
    metrics.update(checkpoint_method=metadata['method'], stored_dtype=metadata['dtype'], reload_logits_exact=True,
                   evaluation_seconds=time.perf_counter() - started, eval_tokens_identity=identity(args.eval_tokens))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'metrics.json').exists():
        raise ValueError('Evaluation result already exists')
    write_json(args.output_dir / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    full = sub.add_parser('quantize')
    full.add_argument('--model', required=True)
    full.add_argument('--baseline-dir', type=Path, required=True, help='Completed r*/prepare directory')
    full.add_argument('--calibration-path', type=Path, required=True)
    full.add_argument('--output-dir', type=Path, required=True)
    full.add_argument('--rank', type=int, choices=(32, 128), required=True)
    full.add_argument('--arm', choices=('awq', 'coordinate', 'paired'), required=True)
    full.add_argument('--devices', nargs='+', default=['cuda:0'])
    full.add_argument('--dtype', choices=tuple(DTYPES), default='fp16')
    full.add_argument('--batch-size', type=int, default=8)
    full.add_argument('--tokens', type=int, default=512)
    full.add_argument('--scale-grid', type=int, default=20)
    full.add_argument('--sweeps', type=int, default=3)
    full.add_argument('--pair-candidates', type=int, default=16)
    full.add_argument('--cpu-threads', type=int, default=4)
    full.add_argument('--resume', action='store_true')
    full.add_argument('--dry-run', action='store_true')
    ev = sub.add_parser('eval')
    ev.add_argument('--checkpoint', type=Path, required=True)
    ev.add_argument('--eval-tokens', type=Path, required=True)
    ev.add_argument('--output-dir', type=Path, required=True)
    ev.add_argument('--device', default='cuda:0')
    ev.add_argument('--ppl-seqlen', type=int, default=128)
    ev.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    for key in ('batch_size', 'tokens', 'scale_grid', 'sweeps', 'pair_candidates', 'cpu_threads'):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.mode == 'quantize':
        try:
            args.devices = [str(torch.device(d)) for d in args.devices]
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))
        if not 1 <= len(args.devices) <= 4 or len(set(args.devices)) != len(args.devices):
            parser.error('Use one to four distinct devices')
        if any(d != 'cpu' and (not d.startswith('cuda:') or int(d.split(':')[1]) >= torch.cuda.device_count()) for d in args.devices):
            parser.error('Devices must be CPU or explicit visible CUDA indices')
    elif args.ppl_seqlen < 2:
        parser.error('PPL length must be at least two')
    return args


if __name__ == '__main__':
    args = parse_args()
    (run if args.mode == 'quantize' else evaluate)(args)
