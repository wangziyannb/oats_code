"""Projection scheduling, worker failures, and real four-GPU equivalence."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import torch

from general_utils.config import config
from models.adapters.llama_adapter import LlamaModelAdapter
from pruning_methods.OATS.prune_oats_compress import altern_ls, prune_oats_compress
from pruning_methods.OATS.awq.parallel import ProjectionPool
from pruning_methods.OATS.awq.pipeline import audit_model, load_bundle, load_legacy_layers, save_bundle
from run_oats_awq import parse_args, manifest_for
import test_oats_awq as fixtures


def task(name, shape=(16, 16)):
    return {'name': name, 'weight': torch.randn(*shape),
            'diagonal': torch.linspace(.2, 2, shape[1]).unsqueeze(0),
            'samples': None, 'quantization': None, 'dtype': torch.float32,
            'num_iters': 2, 'rank': 2, 'sparsity': .5,
            'prune_level': 'row', 'prune_n': 2, 'prune_m': 4, 'report': True}


class SchedulingTests(unittest.TestCase):
    def test_cli_cap_membership_and_single_device_manifest_compatibility(self):
        base=['inloop','--model','meta-llama/Llama-2-7b-hf','--rank','32','--output-dir','unused']
        args=parse_args(base+['--devices','cuda:0','cuda:1','cuda:2','cuda:3'])
        self.assertEqual(args.device,'cuda:0')
        self.assertEqual(len(manifest_for(args)['parallel_execution']['devices']),4)
        single=manifest_for(parse_args(base))
        self.assertNotIn('devices',single)
        self.assertNotIn('worker_cpu_threads',single)
        for extras in (['--devices','cuda:0','cuda:1','cuda:2','cuda:3','cuda:4'],
                       ['--devices','cuda:0','cuda:0'],
                       ['--devices','cuda:1','cuda:2','--device','cuda:0'],
                       ['--devices','cpu']):
            with self.subTest(extras=extras),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): parse_args(base+extras)
        with self.assertRaises(ValueError): ProjectionPool(['cpu']*5)

    def test_cpu_process_results_preserve_solver_and_order(self):
        torch.set_num_threads(1)
        torch.manual_seed(321)
        config.device=torch.device('cpu');config.dtype=torch.float32
        jobs=[task('small'),task('large',(32,16)),task('third')]
        with contextlib.redirect_stdout(io.StringIO()):
            expected=[altern_ls(j['weight'],j['diagonal'],2,2,.5,prune_n=2,prune_m=4) for j in jobs]
            with ProjectionPool(['cpu','cpu']) as pool:
                processes=[p for p,_ in pool.workers]
                actual=pool.solve_layer(jobs)
                again=pool.solve_layer(jobs[:1])
                self.assertEqual([p.pid for p,_ in pool.workers],[p.pid for p in processes])
        self.assertEqual([r['name'] for r in actual],['small','large','third'])
        for result,reference in zip(actual,expected):
            for value,other in zip(result['factors'],reference):
                torch.testing.assert_close(value,other,atol=0,rtol=0)
            self.assertEqual([r['iteration'] for r in result['records']],[1,2])
        self.assertEqual(again[0]['name'],'small')
        self.assertTrue(all(not process.is_alive() for process in processes))

    def test_worker_failure_is_propagated_and_all_workers_stop(self):
        good=task('good');bad=task('bad');bad['diagonal'][0,0]=float('nan')
        with contextlib.redirect_stdout(io.StringIO()):
            with ProjectionPool(['cpu','cpu']) as pool:
                processes=[p for p,_ in pool.workers]
                with self.assertRaisesRegex(RuntimeError,'bad.*failed'):
                    pool.solve_layer([good,bad])
                self.assertTrue(pool.closed)
                self.assertTrue(all(not process.is_alive() for process in processes))


@unittest.skipUnless(torch.cuda.device_count() >= 4,'Four visible GPUs required')
class FourGPUIntegrationTests(unittest.TestCase):
    def test_actual_four_gpu_cli(self):
        import json
        import subprocess
        import sys
        torch.set_num_threads(1)
        helper = fixtures.IntegrationTests()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            helper.fixture(torch.float32).save_pretrained(root / 'base')
            torch.save([{'input_ids': torch.randint(0, 64, (16,))} for _ in range(2)], root / 'cal.pt')
            command = [sys.executable, 'run_oats_awq.py', 'inloop', '--model', str(root / 'base'),
                       '--rank', '32', '--output-dir', str(root / 'out'),
                       '--calibration-path', str(root / 'cal.pt'), '--cal-batch-size', '1',
                       '--num-iters', '1', '--scale-grid', '2', '--awq-tokens', '8',
                       '--devices', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', '--device', 'cuda:3']
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            model, metadata = load_bundle(root / 'out' / 'checkpoint.pt')
            self.assertEqual(audit_model(model, 32)['modules'], 14)
            execution = metadata['metadata']['manifest']['parallel_execution']
            self.assertEqual(execution['calibration_device'], 'cuda:3')
            self.assertEqual(execution['devices'], [f'cuda:{i}' for i in range(4)])
            rows = [json.loads(line) for line in (root / 'out' / 'iterations.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 14)
            self.assertEqual({row['worker_device'] for row in rows}, set(execution['devices']))

    @torch.no_grad()
    def test_four_gpu_matches_serial_and_resume(self):
        torch.set_num_threads(1)
        config.device=torch.device('cuda:0');config.dtype=torch.float16
        torch.cuda.set_device(config.device)
        helper=fixtures.IntegrationTests()
        torch.manual_seed(4321)
        samples=torch.randint(0,64,(2,16))
        loader=[{'input_ids':sample.unsqueeze(0)} for sample in samples]
        devices=[f'cuda:{i}' for i in range(4)]
        with contextlib.redirect_stdout(io.StringIO()), ProjectionPool(devices) as pool:
            for rank in (32,128):
                for mode in ('prepare','inloop'):
                    with self.subTest(rank=rank,mode=mode),tempfile.TemporaryDirectory() as folder:
                        root=Path(folder);(root/'serial').mkdir();(root/'parallel').mkdir()
                        serial=helper.fixture(torch.float16);parallel=copy.deepcopy(serial)
                        hyper={'unscaled':False,'rank_ratio':.1,'fixed_rank':rank,'prune_level':'row','num_iters':2}
                        quant=None if mode=='prepare' else {'config':{'grid':2,'sparse_refresh':2},'tokens':8,'sequences':2}
                        records=[]
                        prune_oats_compress(LlamaModelAdapter(serial),.5,None,loader,hyper,str(root/'serial'),
                            prune_n=2,prune_m=4,quantization=quant)
                        prune_oats_compress(LlamaModelAdapter(parallel),.5,None,loader,hyper,str(root/'parallel'),
                            prune_n=2,prune_m=4,quantization=quant,report=records.append,projection_pool=pool)
                        self.assertEqual(len(records),28)
                        self.assertEqual({r['worker_device'] for r in records},set(devices))
                        self.assertEqual(audit_model(parallel,rank)['violations_2_4'],0)
                        for key,value in serial.state_dict().items():
                            torch.testing.assert_close(value,parallel.state_dict()[key],atol=0,rtol=0)
                        serial.cuda(0);parallel.cuda(0)
                        expected=serial(samples.cuda(0),use_cache=False).logits
                        torch.testing.assert_close(parallel(samples.cuda(0),use_cache=False).logits,expected,atol=0,rtol=0)
                        parallel.cpu()
                        save_bundle(parallel,root/'bundle.pt',rank,mode,{'devices':devices})
                        restored,_=load_bundle(root/'bundle.pt')
                        torch.testing.assert_close(restored.cuda(0)(samples.cuda(0),use_cache=False).logits,expected,atol=0,rtol=0)
                        # Replay the second block from the actual saved first-block state.
                        (root/'resume').mkdir()
                        # Recreate the first-block activation checkpoint using an interrupted run.
                        import importlib
                        from unittest.mock import patch
                        module=importlib.import_module('pruning_methods.OATS.prune_oats_compress')
                        interrupted=helper.fixture(torch.float16)
                        original_save=module.save_checkpoint
                        def stop(*a,**kw):
                            original_save(*a,**kw)
                            raise RuntimeError('stop after block')
                        with patch.object(module,'save_checkpoint',side_effect=stop):
                            with self.assertRaisesRegex(RuntimeError,'stop after block'):
                                prune_oats_compress(LlamaModelAdapter(interrupted),.5,None,loader,hyper,str(root/'resume'),
                                    prune_n=2,prune_m=4,quantization=quant,projection_pool=pool)
                        resumed=load_legacy_layers(helper.fixture(torch.float16),root/'resume',rank,partial=True)
                        prune_oats_compress(LlamaModelAdapter(resumed),.5,None,loader,hyper,str(root/'resume'),
                            prune_n=2,prune_m=4,quantization=quant,projection_pool=pool)
                        for key,value in parallel.state_dict().items():
                            torch.testing.assert_close(value,resumed.state_dict()[key],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
