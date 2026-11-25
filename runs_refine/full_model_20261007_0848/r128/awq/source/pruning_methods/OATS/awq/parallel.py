"""Bounded projection parallelism; each worker calls the unchanged OATS solver.

One persistent spawned process owns each selected device. Only CPU tensors cross
process boundaries, and only the coordinator installs modules or writes files.
"""
from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
from multiprocessing.connection import wait
import time
import traceback

import torch


def _solve(task, device):
    from general_utils.config import config
    from pruning_methods.OATS.prune_oats_compress import altern_ls
    from .quantize import AWQConfig, InLoopQuantizer

    config.device, config.dtype = torch.device(device), task['dtype']
    weight = task['weight'].to(config.device)
    diagonal = task['diagonal'].to(config.device)
    quantizer = None
    if task['quantization'] is not None:
        quantizer = InLoopQuantizer(task['samples'].to(config.device),
            AWQConfig(**task['quantization'].get('config', {})), config.dtype)
    records = []
    started = time.perf_counter()
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        factors = altern_ls(weight, diagonal, task['num_iters'], task['rank'], task['sparsity'],
            prune_level=task['prune_level'], prune_n=task['prune_n'], prune_m=task['prune_m'],
            quantizer=quantizer, report=records.append if task['report'] else None)
    # CPU copies also synchronize completion before recording elapsed time.
    factors = tuple(value.detach().cpu() for value in factors)
    return {'name': task['name'], 'factors': factors, 'records': records,
            'device': device, 'seconds': time.perf_counter() - started}


def _worker(connection, device, settings):
    try:
        torch.set_num_threads(settings['cpu_threads'])
        torch.set_float32_matmul_precision(settings['matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = settings['allow_tf32']
        torch.use_deterministic_algorithms(settings['deterministic'], warn_only=settings['warn_only'])
        if torch.device(device).type == 'cuda':
            torch.cuda.set_device(torch.device(device))
        connection.send({'ready': True})
        while True:
            task = connection.recv()
            if task is None:
                break
            try:
                result = _solve(task, device)
                connection.send({'result': result})
                # Release references before the next large matrix is received.
                del result, task
                if torch.device(device).type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception:
                connection.send({'error': traceback.format_exc()})
                break
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send({'error': traceback.format_exc()})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ProjectionPool:
    """At most four workers; a layer is committed only after every task succeeds."""

    def __init__(self, devices, cpu_threads=1):
        self.devices = [str(torch.device(device)) for device in devices]
        if not 1 <= len(self.devices) <= 4:
            raise ValueError('Projection parallelism requires one to four devices')
        gpu_devices = [device for device in self.devices if device != 'cpu']
        if len(set(gpu_devices)) != len(gpu_devices):
            raise ValueError('GPU devices must be distinct')
        if any(torch.device(d).type not in ('cpu', 'cuda') for d in self.devices):
            raise ValueError('Unsupported worker device')
        if cpu_threads < 1:
            raise ValueError('Worker CPU thread count must be positive')
        self.settings = {'cpu_threads': cpu_threads,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}
        self.workers = []
        self.closed = False

    def __enter__(self):
        ctx = mp.get_context('spawn')
        try:
            for device in self.devices:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(child, device, self.settings), daemon=True)
                process.start()
                child.close()
                self.workers.append((process, parent))
            for process, connection in self.workers:
                if not connection.poll(120):
                    raise RuntimeError(f'Worker {process.pid} did not initialize within 120 seconds')
                message = connection.recv()
                if not message.get('ready'):
                    raise RuntimeError(f'Projection worker initialization failed: {message}')
        except BaseException:
            self.close(terminate=True)
            raise
        return self

    def solve_layer(self, tasks):
        if self.closed or not self.workers:
            raise RuntimeError('ProjectionPool must be used as an open context manager')
        names = [task['name'] for task in tasks]
        if len(names) != len(set(names)):
            raise ValueError('Projection task names must be unique')
        for task in tasks:
            for key in ('weight', 'diagonal', 'samples'):
                value = task.get(key)
                if value is not None and value.device.type != 'cpu':
                    raise ValueError('Only CPU tensors may cross worker boundaries')
        # The three rectangular MLP matrices generally dominate full-SVD work.
        # Largest-first scheduling reduces the tail without splitting any SVD.
        pending = sorted(tasks, key=lambda task: task['weight'].numel() * min(task['weight'].shape), reverse=True)
        active, results = {}, {}
        try:
            for index, (_, connection) in enumerate(self.workers):
                if pending:
                    task = pending.pop(0)
                    connection.send(task)
                    active[index] = task['name']
            while active:
                handles = [self.workers[i][1] for i in active] + [self.workers[i][0].sentinel for i in active]
                ready = wait(handles, timeout=1)
                for index in list(active):
                    process, connection = self.workers[index]
                    if connection in ready:
                        try:
                            message = connection.recv()
                        except (EOFError, OSError) as error:
                            raise RuntimeError(f'Projection worker lost while solving {active[index]}') from error
                        if 'error' in message:
                            raise RuntimeError(f'Projection {active[index]} failed on {self.devices[index]}:\n{message["error"]}')
                        result = message['result']
                        if result['name'] != active[index]:
                            raise RuntimeError('Projection worker returned an unexpected task')
                        results[result['name']] = result
                        print(f"[projection done] {result['name']} on {result['device']}: {result['seconds']:.2f}s", flush=True)
                        del active[index]
                        if pending:
                            task = pending.pop(0)
                            connection.send(task)
                            active[index] = task['name']
                    elif process.sentinel in ready:
                        raise RuntimeError(f'Projection worker exited while solving {active[index]} (exit {process.exitcode})')
            # Keep module installation and diagnostics in the original projection order.
            return [results[name] for name in names]
        except BaseException:
            self.close(terminate=True)
            raise

    def close(self, terminate=False):
        if self.closed:
            return
        self.closed = True
        for process, connection in self.workers:
            if terminate:
                if process.is_alive():
                    process.terminate()
            else:
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process, connection in self.workers:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            connection.close()

    def __exit__(self, exc_type, exc_value, tb):
        self.close(terminate=exc_type is not None)
