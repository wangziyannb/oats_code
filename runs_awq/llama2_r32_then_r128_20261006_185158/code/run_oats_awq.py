#!/usr/bin/env python3
"""Native OATS baselines and two quantization-only AWQ adapters."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from general_utils.config import config
from models.adapters.llama_adapter import LlamaModelAdapter
from pruning_methods.OATS.prune_oats_compress import prune_oats_compress
from pruning_methods.OATS.awq.pipeline import (
    DTYPES, audit_model, claim_run, evaluate_tokens, load_bundle, load_legacy_layers,
    quantize_posthoc, save_bundle, validate_architecture, write_json,
)
from pruning_methods.OATS.awq.quantize import AWQConfig
from pruning_methods.OATS.awq.posthoc import PROJECTION_NAMES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'posthoc', 'inloop', 'eval'))
    parser.add_argument('--model', required=True, help='Llama2 HF identifier or local base-model directory')
    parser.add_argument('--rank', type=int, choices=(32, 128), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Complete bundle, or legacy OATS directory for posthoc')
    parser.add_argument('--device', default=None, help='Calibration/evaluation device; defaults to first --devices entry or cuda:0')
    parser.add_argument('--devices', nargs='+', help='One to four distinct visible CUDA devices for prepare/inloop projection workers')
    parser.add_argument('--worker-cpu-threads', type=int, default=1)
    parser.add_argument('--dtype', choices=tuple(DTYPES), default='fp16')
    parser.add_argument('--num-iters', type=int, default=80)
    parser.add_argument('--calibration-path', type=Path, help='OATS presampled .pth, or posthoc 2-D input_ids tensor')
    parser.add_argument('--cal-nsamples', type=int, default=128)
    parser.add_argument('--cal-seqlen', type=int, default=2048)
    parser.add_argument('--cal-batch-size', type=int, default=32)
    parser.add_argument('--tuple-keys', default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--awq-tokens', type=int, default=512)
    parser.add_argument('--scale-grid', type=int, default=20)
    parser.add_argument('--sparse-refresh', type=int, default=10)
    parser.add_argument('--ppl-seqlen', type=int, default=128)
    parser.add_argument('--eval-tokens', type=Path, help='Optional local input_ids tensor for offline evaluation')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.devices:
        if args.mode not in ('prepare', 'inloop'):
            parser.error('--devices only applies to prepare/inloop')
        if not 1 <= len(args.devices) <= 4:
            parser.error('--devices supports at most four GPUs')
        try:
            devices = [torch.device(device) for device in args.devices]
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))
        if any(device.type != 'cuda' or device.index is None for device in devices):
            parser.error('--devices requires explicit CUDA indices, e.g. cuda:0 cuda:1')
        args.devices = [str(device) for device in devices]
        if len(set(args.devices)) != len(args.devices):
            parser.error('--devices entries must be distinct')
        args.device = args.device or args.devices[0]
        if args.device not in args.devices:
            parser.error('--device must belong to --devices (no extra calibration GPU)')
    else:
        args.device = args.device or 'cuda:0'
    if args.mode in ('posthoc', 'eval') and args.checkpoint is None:
        parser.error('--checkpoint is required for posthoc/eval')
    if args.mode in ('prepare', 'inloop') and args.checkpoint is not None:
        parser.error('Use --resume for interrupted prepare/inloop runs, not --checkpoint')
    for key in ('num_iters', 'cal_nsamples', 'cal_seqlen', 'cal_batch_size', 'awq_tokens', 'scale_grid', 'sparse_refresh', 'worker_cpu_threads'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.ppl_seqlen < 2:
        parser.error('--ppl-seqlen must be >= 2')
    if args.mode == 'eval' and args.resume:
        parser.error('Evaluation does not support --resume')
    return args


def posthoc_settings(args):
    return argparse.Namespace(
        rank=args.rank, device=args.device, w_bit=4, sparse_group_size=128,
        a_group_size=128, b_group_size=32, external_scale_grid=args.scale_grid,
        rank_scale_grid=args.scale_grid, skip_clip=False, clip_grid=20,
        clip_max_shrink=0.5, clip_sample_tokens=512,
    )


def file_identity(path):
    path = Path(path).resolve()
    if path.is_dir():
        files = sorted(path.glob('oats_chkpt_*.pt')) + [path / 'prune_chkpt.pt']
        return [file_identity(item) for item in files]
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def manifest_for(args):
    values = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('resume', 'dry_run', 'output_dir')}
    if args.devices:
        values['parallel_execution'] = {
            'strategy': 'block_projection_spawn', 'devices': args.devices,
            'calibration_device': args.device,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'worker_cpu_threads': args.worker_cpu_threads,
        }
    else:
        # Preserve compatibility with manifests created by the single-device CLI.
        values.pop('devices', None)
        values.pop('worker_cpu_threads', None)
    for key in ('checkpoint', 'calibration_path', 'eval_tokens'):
        if getattr(args, key):
            values[key + '_identity'] = file_identity(getattr(args, key))
    model = Path(args.model)
    if model.exists():
        values['model'] = str(model.resolve())
        values['model_config_sha256'] = hashlib.sha256((model / 'config.json').read_bytes()).hexdigest()
        values['model_weights'] = [file_identity(p) for p in sorted(model.glob('*.safetensors')) + sorted(model.glob('pytorch_model*.bin'))]
    source = Path(__file__).parent / 'pruning_methods/OATS/awq/SOURCES.json'
    values['quantization_source'] = json.loads(source.read_text())
    values.update({'iterate_selection': 'last', 'early_stopping': False,
                   'pruning_algorithm': 'native_oats_weighted_full_svd',
                   'bits': 4, 'groups': {'S': 128, 'V': 128,
                   'U': 32 if args.mode == 'posthoc' else min(args.rank, 128)},
                   'clipping': args.mode == 'posthoc'})
    if args.mode == 'posthoc':
        values['quant_calibration'] = {'dataset': 'override' if args.calibration_path else 'pileval',
                                      'text_samples': 128, 'block_size': 512, 'seed': 42}
    return values


def load_base(args):
    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=DTYPES[args.dtype],
                                            attn_implementation='sdpa')
    model.eval()
    validate_architecture(model)
    for layer in model.model.layers:
        for name in PROJECTION_NAMES:
            module = layer.get_submodule(name)
            if args.rank > min(module.weight.shape) or module.in_features % 128:
                raise ValueError(f'{name}: incompatible rank or W4G128 dimensions')
    return model


def pruning_calibration(args):
    from general_utils.data_utils import (prepare_presampled_dataloader,
                                         get_dataset, prepare_dataloader)
    if args.calibration_path:
        loader = prepare_presampled_dataloader(str(args.calibration_path),
                    batch_size=args.cal_batch_size, tuple_keys=args.tuple_keys)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        loader = prepare_dataloader(get_dataset('c4', train=True), tokenizer,
                    max_seqlen=args.cal_seqlen, batch_size=args.cal_batch_size,
                    nsamples=args.cal_nsamples, seed=args.seed)
    # Count actual sequences; do not alter or truncate a supplied calibration file.
    sequences = len(loader.dataset)
    return loader, sequences


def posthoc_calibration(args):
    if args.calibration_path:
        samples = torch.load(args.calibration_path, map_location='cpu', weights_only=True)
        if not torch.is_tensor(samples) or samples.ndim != 2 or samples.numel() == 0:
            raise ValueError('Posthoc calibration override must be a nonempty [sequences, tokens] input_ids tensor')
        return samples
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    data = load_dataset('mit-han-lab/pile-val-backup', split='validation').shuffle(seed=42)
    samples = []
    for row in data:
        ids = tokenizer.encode(row['text'].strip())
        if not ids or len(ids) > 512:
            continue
        samples.append(torch.tensor([ids], dtype=torch.long))
        if len(samples) == 128:
            break
    if len(samples) != 128:
        raise ValueError('PileVal did not provide 128 eligible calibration texts')
    joined = torch.cat(samples, dim=1)
    blocks = joined.numel() // 512
    if not blocks:
        raise ValueError('Insufficient PileVal calibration tokens')
    return joined[:, :blocks * 512].reshape(blocks, 512)


def run(args):
    manifest = manifest_for(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for small-model checks')
    for device in args.devices or [args.device]:
        parsed_device = torch.device(device)
        if parsed_device.type == 'cuda' and parsed_device.index is not None and parsed_device.index >= torch.cuda.device_count():
            raise ValueError(f'{device} is outside the visible CUDA device list')
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
    claim_run(args.output_dir, manifest, args.resume)
    torch.manual_seed(42 if args.mode == 'posthoc' else args.seed)
    config.device, config.dtype = torch.device(args.device), DTYPES[args.dtype]
    if args.mode in ('prepare', 'inloop'):
        model = load_base(args)
        work = args.output_dir / 'layers'
        work.mkdir(exist_ok=True)
        if args.resume and (work / 'prune_chkpt.pt').exists():
            load_legacy_layers(model, work, args.rank, partial=True)
            completed = torch.load(work / 'prune_chkpt.pt', map_location='cpu', weights_only=True)['prune_start_idx']
            # Discard diagnostics from an interrupted, uncommitted layer before replay.
            log_path = args.output_dir / 'iterations.jsonl'
            if log_path.exists():
                kept = []
                for line in log_path.read_text().splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record['layer'] <= completed:
                        kept.append(line)
                log_path.write_text('\n'.join(kept) + ('\n' if kept else ''))
        loader, sequences = pruning_calibration(args)
        hyper = dict(unscaled=False, rank_ratio=0.1, fixed_rank=args.rank,
                     prune_level='row', num_iters=args.num_iters, compress=True, sparsity_type='2:4')
        quant = None
        if args.mode == 'inloop':
            quant = {'config': asdict(AWQConfig(grid=args.scale_grid, sparse_refresh=args.sparse_refresh)),
                     'tokens': args.awq_tokens, 'sequences': sequences}
        pool_context = nullcontext(None)
        if args.devices:
            from pruning_methods.OATS.awq.parallel import ProjectionPool
            pool_context = ProjectionPool(args.devices, args.worker_cpu_threads)
        with pool_context as projection_pool, (args.output_dir / 'iterations.jsonl').open('a' if args.resume else 'w') as log:
            def report(record):
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
            prune_oats_compress(LlamaModelAdapter(model), 0.5, None, loader, hyper,
                                str(work), prune_n=2, prune_m=4, quantization=quant, report=report,
                                projection_pool=projection_pool)
        report_data = {'method': args.mode, 'iterate_selection': 'last', 'num_iters': args.num_iters,
                       'actual_calibration_sequences': sequences, 'quantization': quant,
                       'audit': audit_model(model, args.rank), 'packed_int4': False}
    elif args.mode == 'posthoc':
        if args.checkpoint.is_dir():
            source_manifest = args.checkpoint.parent / 'manifest.json'
            if source_manifest.exists() and json.loads(source_manifest.read_text()).get('mode') != 'prepare':
                raise ValueError('Posthoc requires an unquantized OATS source, not an inloop run')
            model = load_legacy_layers(load_base(args), args.checkpoint, args.rank)
        else:
            model, metadata = load_bundle(args.checkpoint)
            if metadata['method'] != 'prepare' or metadata['rank'] != args.rank:
                raise ValueError('Posthoc requires an unquantized OATS baseline of the requested rank')
            if next(model.parameters()).dtype != DTYPES[args.dtype]:
                raise ValueError('Source checkpoint dtype differs; specify its original --dtype')
        samples = posthoc_calibration(args)
        report_data = quantize_posthoc(model, samples, posthoc_settings(args), args.output_dir, args.resume)
        report_data['calibration_shape'] = list(samples.shape)
    else:
        model, metadata = load_bundle(args.checkpoint)
        if metadata['rank'] != args.rank:
            raise ValueError('Requested rank differs from checkpoint')
        if args.eval_tokens:
            tokens = torch.load(args.eval_tokens, map_location='cpu', weights_only=True)
        else:
            from datasets import load_dataset
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            tokens = tokenizer('\n\n'.join(data['text']), return_tensors='pt').input_ids
        # Restore exactly the stored dtype; --dtype never requantizes/casts an eval artifact.
        report_data = evaluate_tokens(model.to(args.device), tokens, args.ppl_seqlen)
        report_data.update({'checkpoint_method': metadata['method'], 'stored_dtype': metadata['dtype']})
        write_json(args.output_dir / 'metrics.json', report_data)
        print(json.dumps(report_data, indent=2))
        return
    save_bundle(model, args.output_dir / 'checkpoint.pt', args.rank, args.mode,
                {'base_model': args.model, 'manifest': manifest, 'report': report_data})
    write_json(args.output_dir / 'quant_report.json', report_data)
    print(f'Saved complete model: {args.output_dir / "checkpoint.pt"}')


if __name__ == '__main__':
    run(parse_args())
