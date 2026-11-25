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
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'status.json').exists():
        raise ValueError('This queue has already run; inspect and resume individual stages')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible != '0,1,2,3':
        raise ValueError('This authorized study requires CUDA_VISIBLE_DEVICES=0,1,2,3')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONUNBUFFERED='1')
    script = str(Path(__file__).parent / 'run_oats_refine_full.py')
    state = {'status': 'running', 'started_at': now(), 'runner_pid': os.getpid(),
             'physical_gpus': [0, 1, 2, 3], 'ranks': [32, 128], 'arms': ['awq', 'coordinate', 'paired'], 'stages': []}
    results = []
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
        path = root / 'logs' / f'r{rank}_{arm}_{mode}.log'
        log = path.open('x')
        log.write(json.dumps({'argv': argv, 'started_at': now()}) + '\n')
        log.flush()
        process = subprocess.Popen(argv, cwd=Path(__file__).parent, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        stage = {'rank': rank, 'arm': arm, 'mode': mode, 'status': 'running', 'pid': process.pid,
                 'started_at': now(), 'argv': argv, 'output': str(output), 'log': str(path)}
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
        raise
    write(root / 'status.json', state)


if __name__ == '__main__':
    main()
