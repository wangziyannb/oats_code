#!/usr/bin/env python3
"""Capture and compare fixed-mask code refinement on the first Llama block.

Research probe, not a full-model PPL run. Dense first-block inputs are identical
for both ranks. Half of the original OATS calibration sequences are reserved
from the NEW quantizer; they were not held out from the original OATS fit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F

from pruning_methods.OATS.awq.refine import CodeMatrix, DiagonalRefiner, RefineConfig, initialize_awq

NAMES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def source_hashes():
    base = Path(__file__).parent
    names = ('run_oats_refine.py', 'pruning_methods/OATS/awq/refine.py', 'pruning_methods/OATS/awq/quantize.py')
    return {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names}


def claim(args):
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output must be new/empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(source_sha256=source_hashes(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    created_at=datetime.now(timezone.utc).isoformat(), torch_version=str(torch.__version__))
    write_json(output / 'manifest.json', manifest)
    for name in manifest['source_sha256']:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent / name, destination)
    return output


@torch.no_grad()
def capture(args):
    from transformers import LlamaForCausalLM
    from general_utils.config import config
    from general_utils.data_utils import prepare_presampled_dataloader
    from general_utils.utils import map_tensors
    from models.adapters.llama_adapter import LlamaModelAdapter
    from models.model_utils import get_layer0_inputs

    output = claim(args)
    device = torch.device(args.device)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
    config.device, config.dtype = device, dtype
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation='sdpa').eval()
    if model.config.num_key_value_heads != model.config.num_attention_heads:
        raise ValueError('This probe currently covers Llama2 without GQA')
    model.config.use_cache = False
    loader = prepare_presampled_dataloader(str(args.calibration_path), batch_size=args.batch_size)
    count = len(loader.dataset)
    if not 0 < args.train_sequences < count:
        raise ValueError('Need nonempty train and validation sequence subsets')
    per_train = max(1, (args.tokens + args.train_sequences - 1) // args.train_sequences)
    per_validation = max(1, (args.tokens + count - args.train_sequences - 1) // (count - args.train_sequences))
    layer = model.model.layers[0].to(device)
    buffers = {name: {'train': [], 'validation': [], 'train_count': 0, 'validation_count': 0,
                      'sum_square': torch.zeros(layer.get_submodule(name).in_features, device=device, dtype=torch.float64),
                      'input_tokens': 0} for name in NAMES}
    sequence_offset = 0

    def hook(name):
        def collect(module, inputs, result):
            state = buffers[name]
            x = inputs[0].detach()
            if not torch.isfinite(x).all():
                raise ValueError(f'Nonfinite calibration input: {name}')
            for local, sequence in enumerate(x):
                train = sequence_offset + local < args.train_sequences
                split, quota = ('train', per_train) if train else ('validation', per_validation)
                if train:
                    state['sum_square'] += sequence.float().square().sum(0, dtype=torch.float64)
                    state['input_tokens'] += sequence.shape[0]
                remaining = args.tokens - state[split + '_count']
                take = min(quota, remaining, sequence.shape[0])
                if take > 0:
                    indices = torch.linspace(0, sequence.shape[0] - 1, take, device=device).round().long()
                    state[split].append(sequence.index_select(0, indices).cpu())
                    state[split + '_count'] += take
        return collect

    handles = [layer.get_submodule(name).register_forward_hook(hook(name)) for name in NAMES]
    adapter = LlamaModelAdapter(model)
    try:
        for batch in loader:
            _, positional, keyword = get_layer0_inputs(adapter, batch)
            positional, keyword = map_tensors([positional, keyword], device=device)
            layer(*positional, **keyword)
            sequence_offset += batch['input_ids'].shape[0]
            print(f'Captured {sequence_offset}/{count} sequences', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    manifest = {'model': str(Path(args.model).resolve()), 'model_config_sha256': hashlib.sha256((Path(args.model) / 'config.json').read_bytes()).hexdigest(),
                'calibration_identity': identity(args.calibration_path), 'layer': 0, 'dtype': args.dtype,
                'train_sequences': args.train_sequences, 'validation_sequences': count - args.train_sequences,
                'validation_scope': 'held out from new quantizer only; original OATS used all sequences',
                'input_propagation': 'dense first block, shared by both ranks',
                'diagonal': 'mean squared activation over ALL tokens in train sequences, clamped at 1e-8',
                'sampling': 'uniform positions per sequence; disjoint sequence splits', 'projections': {}}
    for name in NAMES:
        state = buffers[name]
        payload = {'name': name, 'weight': layer.get_submodule(name).weight.detach().cpu(),
                   'diagonal': (state['sum_square'] / state['input_tokens']).float().clamp_min(1e-8).cpu(),
                   'train_samples': torch.cat(state['train']), 'validation_samples': torch.cat(state['validation'])}
        save(output / (name + '.pt'), payload)
        manifest['projections'][name] = {'weight_shape': list(payload['weight'].shape),
                                       'train_sample_shape': list(payload['train_samples'].shape),
                                       'validation_sample_shape': list(payload['validation_samples'].shape)}
    write_json(output / 'capture.json', manifest)
    print(f'Capture complete: {output}', flush=True)


WORKER_DEVICE = None


def init_worker(device_queue):
    global WORKER_DEVICE
    WORKER_DEVICE = device_queue.get()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    if WORKER_DEVICE.startswith('cuda'):
        torch.cuda.set_device(torch.device(WORKER_DEVICE))
        torch.backends.cuda.matmul.allow_tf32 = False


@torch.no_grad()
def measure(ref, train, validation):
    result = {'objective': ref.objective(),
              'weighted_relative_error': (2 * ref.objective() / max(float((ref.w.square() * ref.h).sum(dtype=torch.float64)), 1e-30)).__pow__(.5),
              'mask_preserved': bool(torch.equal(ref.s.ne(0), ref.grids['S'].mask)),
              'fp16_mask_preserved': bool(torch.equal(ref.s.half().ne(0), ref.grids['S'].mask))}
    for name, samples in [('train', train), ('validation', validation)]:
        x = samples.float()
        target = F.linear(x, ref.w)
        actual = F.linear(x, ref.s) + F.linear(F.linear(x, ref.v), ref.u)
        mse = float((actual - target).square().mean())
        result[name + '_mse'] = mse
        result[name + '_relative_mse'] = mse / max(float(target.square().mean()), 1e-30)
        half = samples.half()
        half_actual = F.linear(half, ref.s.half()) + F.linear(F.linear(half, ref.v.half()), ref.u.half())
        result[name + '_fp16_mse'] = float((half_actual.float() - target).square().mean())
    if not result['mask_preserved'] or not result['fp16_mask_preserved']:
        raise ValueError('Sparse mask changed in optimization or FP16 storage')
    return result


@torch.no_grad()
def solve_projection(task):
    device = torch.device(WORKER_DEVICE)
    rank, name = task['rank'], task['name']
    folder = Path(task['output']) / f'r{rank}' / name
    folder.mkdir(parents=True, exist_ok=False)
    data = torch.load(Path(task['capture']) / (name + '.pt'), map_location='cpu', weights_only=True)
    state = torch.load(task['layer_checkpoint'], map_location='cpu', weights_only=True)
    w = data['weight'].to(device=device, dtype=torch.float32)
    h = data['diagonal'].to(device)
    train = data['train_samples'].to(device)
    validation = data['validation_samples'].to(device)
    s, u, v = (state[f'{name}.{key}'].to(device=device, dtype=torch.float32) for key in ('S', 'U', 'V'))
    if u.shape != (w.shape[0], rank) or v.shape != (rank, w.shape[1]) or s.shape != w.shape:
        raise ValueError('Baseline projection shape/rank mismatch')
    del state, data
    started = time.perf_counter()
    grids, init_info = initialize_awq(w, s, u, v, train, task['group_size'], task['scale_grid'])
    init_seconds = time.perf_counter() - started
    settings = RefineConfig(task['sweeps'], task['update_scales'], task['pair_candidates'])
    report = {'rank': rank, 'name': name, 'device': str(device), 'config': asdict(settings),
              'initialization': init_info, 'initialization_seconds': init_seconds, 'arms': {}}
    initial = DiagonalRefiner(w, h, grids, settings)
    report['arms']['awq'] = measure(initial, train, validation)
    save(folder / 'awq.pt', initial.state_dict())
    baseline_error = w - s - u @ v
    report['unquantized_oats_weighted_relative_error'] = float(
        ((baseline_error.square() * h).sum(dtype=torch.float64) / (w.square() * h).sum(dtype=torch.float64)).sqrt())
    with (folder / 'phases.jsonl').open('w') as log:
        def callback(arm):
            def emit(record):
                log.write(json.dumps({'arm': arm, **record}, allow_nan=False) + '\n')
                log.flush()
            return emit
        coordinate_grids = None
        for arm in ('coordinate', 'coordinate_then_codes', 'coordinate_then_pair', 'paired'):
            ref = DiagonalRefiner(w, h, coordinate_grids if arm.startswith('coordinate_then_') else grids, settings)
            start = time.perf_counter()
            if arm.startswith('coordinate_then_'):
                phases = ('pairs_su', 'pairs_sv') if arm == 'coordinate_then_pair' else ('codes_u', 'codes_v', 'codes_s')
                for phase in phases:
                    callback(arm)(ref.phase(phase, getattr(ref, phase), 1))
            else:
                ref.run(paired=arm == 'paired', callback=callback(arm))
            elapsed = time.perf_counter() - start
            if arm == 'coordinate':
                coordinate_grids = {k: g.clone() for k, g in ref.grids.items()}
            report['arms'][arm] = {**measure(ref, train, validation), 'refinement_seconds': elapsed,
                'rejected_phases': sum(not r['accepted'] for r in ref.records),
                'pair_objective_gain': sum(r['before'] - r['after'] for r in ref.records if r['phase'].startswith('pairs_'))}
            save(folder / (arm + '.pt'), ref.state_dict())
            # Verify real-size serialized grids, including the folded channel
            # transforms, reconstruct exactly before declaring an arm complete.
            restored = torch.load(folder / (arm + '.pt'), map_location='cpu', weights_only=True)
            for key, value in restored['grids'].items():
                loaded = CodeMatrix.from_state_dict(value, device)
                if not torch.equal(loaded.values(), ref.grids[key].values()):
                    raise ValueError('Quantization state changed on reload')
            report['arms'][arm]['reload_exact'] = True
            print(f'r{rank} {name} {arm}: {report["arms"][arm]["weighted_relative_error"]:.6f} ({elapsed:.2f}s)', flush=True)
    write_json(folder / 'metrics.json', report)
    del initial, grids, coordinate_grids, ref, w, s, u, v, train, validation, restored, loaded
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report


def benchmark(args):
    capture_info = json.loads((args.capture_dir / 'capture.json').read_text())
    for rank in args.ranks:
        baseline = args.baseline_dir / f'r{rank}' / 'prepare'
        info = json.loads((baseline / 'manifest.json').read_text())
        if info['mode'] != 'prepare' or info['rank'] != rank or info['dtype'] != capture_info['dtype']:
            raise ValueError('Baseline rank/method/dtype mismatch')
        if info['model_config_sha256'] != capture_info['model_config_sha256'] or info['model'] != capture_info['model']:
            raise ValueError('Capture and OATS baseline use different base models')
        if info['calibration_path_identity'] != capture_info['calibration_identity']:
            raise ValueError('Calibration file differs from the original baseline')
        if not (baseline / 'checkpoint.pt').is_file():
            raise ValueError('OATS baseline is incomplete')
    output = claim(args)
    devices = args.devices
    if len(devices) > 4 or len(devices) != len(set(devices)):
        raise ValueError('Use at most four distinct devices')
    if any(str(torch.device(d)) != d or (d != 'cpu' and (not d.startswith('cuda:') or torch.device(d).index >= torch.cuda.device_count())) for d in devices):
        raise ValueError('Invalid device list')
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    for device in devices:
        queue.put(device)
    results = []
    status = {'status': 'running', 'completed_projections': 0, 'total_projections': len(args.ranks) * len(args.projections),
              'capture': capture_info, 'started_at': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'status.json', status)
    try:
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx, initializer=init_worker, initargs=(queue,)) as pool:
            # Rank barrier: finish all selected r=32 projections before r=128.
            for rank in args.ranks:
                tasks = [{'rank': rank, 'name': name, 'capture': str(args.capture_dir.resolve()), 'output': str(output),
                          'layer_checkpoint': str((args.baseline_dir / f'r{rank}/prepare/layers/oats_chkpt_0.pt').resolve()),
                          'sweeps': args.sweeps, 'update_scales': not args.no_scale_update,
                          'pair_candidates': args.pair_candidates, 'scale_grid': args.scale_grid, 'group_size': args.group_size}
                         for name in args.projections]
                status['current_rank'] = rank
                write_json(output / 'status.json', status)
                futures = [pool.submit(solve_projection, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
                    status['completed_projections'] = len(results)
                    status['updated_at'] = datetime.now(timezone.utc).isoformat()
                    write_json(output / 'status.json', status)
                    write_json(output / 'results.json', sorted(results, key=lambda x: (x['rank'], x['name'])))
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        write_json(output / 'status.json', status)
        raise
    status.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
    write_json(output / 'status.json', status)
    queue.close()
    print(f'Benchmark complete: {output}', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--model', required=True)
    cap.add_argument('--calibration-path', type=Path, required=True)
    cap.add_argument('--output-dir', type=Path, required=True)
    cap.add_argument('--device', default='cuda:0')
    cap.add_argument('--dtype', choices=('fp16','bf16','fp32'), default='fp16')
    cap.add_argument('--batch-size', type=int, default=8)
    cap.add_argument('--train-sequences', type=int, default=64)
    cap.add_argument('--tokens', type=int, default=512)
    bench = sub.add_parser('benchmark')
    bench.add_argument('--capture-dir', type=Path, required=True)
    bench.add_argument('--baseline-dir', type=Path, required=True)
    bench.add_argument('--output-dir', type=Path, required=True)
    bench.add_argument('--ranks', type=int, nargs='+', choices=(32,128), default=[32,128])
    bench.add_argument('--devices', nargs='+', default=['cuda:0'])
    bench.add_argument('--projections', nargs='+', choices=NAMES, default=list(NAMES))
    bench.add_argument('--sweeps', type=int, default=3)
    bench.add_argument('--scale-grid', type=int, default=20)
    bench.add_argument('--group-size', type=int, default=128)
    bench.add_argument('--pair-candidates', type=int, default=16)
    bench.add_argument('--no-scale-update', action='store_true')
    args = parser.parse_args(argv)
    for key in ('batch_size','train_sequences','tokens','sweeps','scale_grid','group_size','pair_candidates'):
        if hasattr(args,key) and getattr(args,key) < 1:
            parser.error(f'{key} must be positive')
    if args.mode == 'benchmark' and (len(set(args.ranks)) != len(args.ranks) or len(set(args.projections)) != len(args.projections)):
        parser.error('Ranks and projections must not contain duplicates')
    return args


if __name__ == '__main__':
    args = parse_args()
    (capture if args.mode == 'capture' else benchmark)(args)
