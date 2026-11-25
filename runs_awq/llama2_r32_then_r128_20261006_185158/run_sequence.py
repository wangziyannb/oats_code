#!/usr/bin/env python3
"""Run the authorized rank 32 then rank 128 experiments, stopping on error."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'
PYTHON = '/home/zwang53/oats_code/.venv-awq/bin/python'
BASE = json.loads((ROOT / 'data_preflight.json').read_text())['model']
CALIBRATION = '/home/zwang53/oats_code/calib_llama2-7b.pth'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='4,5,6,7',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
           HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
           HF_HUB_DISABLE_PROGRESS_BARS='1')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def commands():
    result = []
    for rank in (32, 128):
        rank_dir = ROOT / f'r{rank}'
        for mode in ('prepare', 'posthoc', 'inloop', 'prepare_eval', 'posthoc_eval', 'inloop_eval'):
            is_eval = mode.endswith('_eval')
            command = 'eval' if is_eval else mode
            output = rank_dir / mode
            argv = [PYTHON, '-u', str(CODE / 'run_oats_awq.py'), command,
                    '--model', BASE, '--rank', str(rank), '--output-dir', str(output),
                    '--dtype', 'fp16', '--device', 'cuda:0', '--num-iters', '80']
            if mode in ('prepare', 'inloop'):
                argv += ['--devices', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3',
                         '--calibration-path', CALIBRATION, '--seed', '0',
                         '--cal-nsamples', '128', '--cal-seqlen', '2048',
                         '--cal-batch-size', '32', '--awq-tokens', '512',
                         '--scale-grid', '20', '--sparse-refresh', '10',
                         '--worker-cpu-threads', '1']
            elif mode == 'posthoc':
                argv += ['--checkpoint', str(rank_dir / 'prepare' / 'checkpoint.pt'), '--scale-grid', '20']
            else:
                argv += ['--checkpoint', str(rank_dir / mode.removesuffix('_eval') / 'checkpoint.pt'),
                         '--ppl-seqlen', '128']
            result.append({'rank': rank, 'stage': mode, 'argv': argv,
                           'output_dir': str(output), 'log': str(ROOT / 'logs' / f'r{rank}_{mode}.log'),
                           'expected': str(output / ('metrics.json' if is_eval else 'checkpoint.pt')),
                           'status': 'pending'})
    return result


def main():
    lock = (ROOT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'status.json').exists():
        raise RuntimeError('This queue already ran; inspect status and resume individual stages explicitly.')
    for name, expected in json.loads((ROOT / 'source_sha256.json').read_text()).items():
        if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen source changed: {name}')
    (ROOT / 'logs').mkdir(exist_ok=True)
    state = {'status': 'running', 'started_at': now(), 'runner_pid': os.getpid(),
             'physical_gpus': [4, 5, 6, 7], 'ranks': [32, 128], 'stages': commands()}
    write_json(ROOT / 'commands.json', {'environment': {k: ENV[k] for k in (
        'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'TOKENIZERS_PARALLELISM', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE')},
        'stages': state['stages']})
    process = None
    stage = None

    def stop(signum, frame):
        raise InterruptedError(f'Queue received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for index, stage in enumerate(state['stages']):
            stage.update(status='running', started_at=now())
            state.update(current_stage=index, updated_at=now())
            write_json(ROOT / 'status.json', state)
            print(f"{now()} START rank={stage['rank']} stage={stage['stage']}", flush=True)
            with Path(stage['log']).open('x') as logfile:
                logfile.write(json.dumps({'started_at': now(), 'argv': stage['argv']}) + '\n')
                logfile.flush()
                process = subprocess.Popen(stage['argv'], cwd=CODE, env=ENV,
                                           stdout=logfile, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                stage['pid'] = process.pid
                while True:
                    state['updated_at'] = now()
                    stage['log_bytes'] = Path(stage['log']).stat().st_size
                    write_json(ROOT / 'status.json', state)
                    try:
                        returncode = process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            stage.update(returncode=returncode, finished_at=now())
            if returncode:
                raise RuntimeError(f"rank={stage['rank']} stage={stage['stage']} exited {returncode}; see {stage['log']}")
            if not Path(stage['expected']).is_file():
                raise RuntimeError(f"Stage returned success without artifact: {stage['expected']}")
            stage['status'] = 'complete'
            if stage['stage'].endswith('_eval'):
                stage['metrics'] = json.loads(Path(stage['expected']).read_text())
            write_json(ROOT / 'status.json', state)
            print(f"{now()} DONE rank={stage['rank']} stage={stage['stage']}", flush=True)
            process = None
        state.update(status='complete', finished_at=now(), updated_at=now())
        write_json(ROOT / 'results.json', [{'rank': s['rank'], 'method': s['stage'].removesuffix('_eval'),
                                          **s['metrics']} for s in state['stages'] if 'metrics' in s])
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if stage is not None:
            stage.update(status='failed', finished_at=now())
        state.update(status='failed', error=str(error), finished_at=now(), updated_at=now())
        print(f'{now()} STOP {error}', file=sys.stderr, flush=True)
        write_json(ROOT / 'status.json', state)
        raise
    write_json(ROOT / 'status.json', state)


if __name__ == '__main__':
    main()
