#!/usr/bin/env python3
"""Run rank 32 then 128, with three refinement arms and strict-reload PPL.

Quantization uses up to four projection workers. Once a rank is quantized,
independent batch-one evaluations run on separate GPUs before the next rank.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--calibration-path', type=Path, required=True)
    parser.add_argument('--eval-tokens', type=Path, required=True)
    parser.add_argument('--code-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Frozen runtime source directory')
    parser.add_argument('--resume', action='store_true', help='Recover completed stages and continue the queue')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'status.json').exists() and not args.resume:
        raise ValueError('This queue has already run; inspect and resume individual stages')
    if args.resume and not (root / 'status.json').exists():
        raise ValueError('No queue exists to resume')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    physical_gpus = [int(d) for d in visible.split(',') if d.isdigit()]
    if len(physical_gpus) != 4 or len(set(physical_gpus)) != 4 or any(d < 0 for d in physical_gpus):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to four distinct authorized physical indices')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONUNBUFFERED='1')
    script = str(args.code_dir.resolve() / 'run_oats_refine_full.py')
    state = {'status': 'running', 'started_at': now(), 'runner_pid': os.getpid(),
             'physical_gpus': physical_gpus, 'ranks': [32, 128], 'arms': ['awq', 'coordinate', 'paired'], 'stages': []}
    results = []
    if args.resume:
        state = json.loads((root / 'status.json').read_text())
        if state['status'] == 'complete':
            raise ValueError('Queue is already complete')
        for stage in state['stages']:
            stage.setdefault('physical_gpus', state['physical_gpus'])
        state.setdefault('device_transitions', []).append({'at': now(), 'from': state['physical_gpus'], 'to': physical_gpus})
        state.update(status='running', resumed_at=now(), runner_pid=os.getpid(), physical_gpus=physical_gpus)
        state.pop('error', None)
        state.pop('control_note', None)
        state.pop('finished_at', None)
        if (root / 'results.json').exists():
            results = json.loads((root / 'results.json').read_text())
    running = []
    (root / 'logs').mkdir(exist_ok=True)

    def start(rank, arm, mode, device=None):
        output = root / f'r{rank}' / (arm + ('_eval' if mode == 'eval' else ''))
        argv = [sys.executable, '-u', script, mode, '--output-dir', str(output)]
        if mode == 'quantize':
            argv += ['--model', args.model, '--baseline-dir', str(args.baseline_dir.resolve() / f'r{rank}' / 'prepare'),
                     '--calibration-path', str(args.calibration_path.resolve()), '--rank', str(rank), '--arm', arm,
                     '--devices', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', '--dtype', 'fp16',
                     '--batch-size', '8', '--tokens', '512', '--scale-grid', '20', '--sweeps', '3', '--pair-candidates', '16']
        else:
            argv += ['--checkpoint', str(root / f'r{rank}' / arm / 'checkpoint.pt'),
                     '--eval-tokens', str(args.eval_tokens.resolve()), '--device', device, '--ppl-seqlen', '128']
        previous = [s for s in state['stages'] if (s['rank'], s['arm'], s['mode']) == (rank, arm, mode)]
        artifact = output / ('metrics.json' if mode == 'eval' else 'checkpoint.pt')
        if args.resume and artifact.exists():
            if not previous or previous[-1]['argv'] not in (argv, argv + ['--resume']):
                raise ValueError('Completed stage command differs from the requested experiment')
            if mode == 'quantize':
                completed = json.loads((output / 'status.json').read_text())
                report = json.loads((output / 'quant_report.json').read_text())
                if completed['status'] != 'complete' or report['audit']['violations_2_4']:
                    raise ValueError('Incomplete or invalid quantization stage')
            else:
                metrics = json.loads(artifact.read_text())
                if not metrics['reload_logits_exact']:
                    raise ValueError('Previous evaluation failed strict reload')
                previous[-1]['metrics'] = metrics
                if not any((r['rank'], r['method']) == (rank, arm) for r in results):
                    results.append({'rank': rank, 'method': arm, **metrics})
                    write(root / 'results.json', results)
            previous[-1].update(status='complete', recovered_from_complete_artifact=True)
            write(root / 'status.json', state)
            print(f'{now()} SKIP completed rank={rank} arm={arm} mode={mode}', flush=True)
            return
        if args.resume and mode == 'quantize' and (output / 'manifest.json').exists():
            # The quantizer itself checks all settings, including the device mask.
            argv.append('--resume')
        path = root / 'logs' / f'r{rank}_{arm}_{mode}.log'
        log = path.open('a' if args.resume else 'x')
        log.write(json.dumps({'argv': argv, 'started_at': now()}) + '\n')
        log.flush()
        process = subprocess.Popen(argv, cwd=args.code_dir, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        stage = {'rank': rank, 'arm': arm, 'mode': mode, 'status': 'running', 'pid': process.pid,
                 'started_at': now(), 'argv': argv, 'output': str(output), 'log': str(path),
                 'physical_gpus': physical_gpus}
        state['stages'].append(stage)
        running.append((process, log, stage))
        write(root / 'status.json', state)
        print(f'{now()} START rank={rank} arm={arm} mode={mode}', flush=True)

    def wait_all():
        while running:
            for process, log, stage in list(running):
                returncode = process.poll()
                if returncode is None:
                    continue
                log.close()
                stage.update(returncode=returncode, finished_at=now(), status='failed' if returncode else 'complete')
                running.remove((process, log, stage))
                if returncode:
                    raise RuntimeError(f'Stage failed: {stage["log"]}')
                artifact = Path(stage['output']) / ('metrics.json' if stage['mode'] == 'eval' else 'checkpoint.pt')
                if not artifact.exists():
                    raise RuntimeError(f'Missing completed artifact: {artifact}')
                if stage['mode'] == 'eval':
                    stage['metrics'] = json.loads(artifact.read_text())
                    results.append({'rank': stage['rank'], 'method': stage['arm'], **stage['metrics']})
                    write(root / 'results.json', results)
                print(f'{now()} DONE rank={stage["rank"]} arm={stage["arm"]} mode={stage["mode"]}', flush=True)
            state['updated_at'] = now()
            write(root / 'status.json', state)
            if running:
                time.sleep(5)

    def stop(signum, frame):
        raise InterruptedError(f'Received signal {signum}')
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for rank in (32, 128):
            for arm in state['arms']:
                start(rank, arm, 'quantize')
                wait_all()
            for index, arm in enumerate(state['arms']):
                start(rank, arm, 'eval', f'cuda:{index}')
            wait_all()
        state.update(status='complete', finished_at=now())
    except BaseException as error:
        for process, log, stage in running:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()
            stage.update(status='interrupted', finished_at=now())
        state.update(status='failed', error=str(error), finished_at=now())
        write(root / 'status.json', state)
        lock.close()
        raise
    write(root / 'status.json', state)
    lock.close()


if __name__ == '__main__':
    main()
