"""Offline full-model refinement, propagation, and interrupted-run regression."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from general_utils.config import config
from models.adapters.llama_adapter import LlamaModelAdapter
from pruning_methods.OATS.prune_oats_compress import prune_oats_compress
from pruning_methods.OATS.awq.pipeline import audit_model, evaluate_tokens, load_bundle, save_bundle
from pruning_methods.OATS.awq.refine import RefineConfig
from run_oats_refine_full import capture_statistics, refine_model


class FullRefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(99)

    def fixture(self, dtype):
        cfg = LlamaConfig(vocab_size=64, hidden_size=256, intermediate_size=512,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=64, attention_dropout=0.)
        cfg._attn_implementation = 'sdpa'
        return LlamaForCausalLM(cfg).to(dtype).eval()

    @torch.no_grad()
    def exercise(self, device, dtype, ranks, arms):
        config.device, config.dtype = torch.device(device), dtype
        dense = self.fixture(dtype)
        samples = torch.randint(0, 64, (2, 16))
        loader = [{'input_ids': sample.unsqueeze(0)} for sample in samples]
        for rank in ranks:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                baseline = root / 'baseline'
                baseline.mkdir()
                prune_oats_compress(LlamaModelAdapter(copy.deepcopy(dense)), .5, None, loader,
                                   {'unscaled': False, 'rank_ratio': .1, 'fixed_rank': rank,
                                    'prune_level': 'row', 'num_iters': 2}, str(baseline), prune_n=2, prune_m=4)
                settings = RefineConfig(1, True, 2, storage_dtype=str(dtype).removeprefix('torch.'))
                for arm in arms:
                    with self.subTest(rank=rank, arm=arm, device=device):
                        model = copy.deepcopy(dense)
                        output = root / arm
                        output.mkdir()
                        incoming, outgoing = [], []
                        def capture(*args, **kwargs):
                            incoming.append(args[1][0][0].clone())
                            return capture_statistics(*args, **kwargs)
                        def finished(index, positional):
                            outgoing.append(positional[0][0].clone())
                        with patch('run_oats_refine_full.capture_statistics', side_effect=capture):
                            report = refine_model(model, baseline, loader, output, rank=rank, arm=arm,
                                                  settings=settings, device=device, tokens=8, scale_grid=2,
                                                  on_layer=finished)
                        torch.testing.assert_close(incoming[1], outgoing[0], atol=0, rtol=0)
                        self.assertEqual(report['audit']['modules'], 14)
                        self.assertEqual(report['audit']['violations_2_4'], 0)
                        for layer in report['layers']:
                            self.assertTrue(all(p['reload_exact'] for p in layer['projections']))
                            self.assertTrue(all(p['after']['objective'] <= p['before']['objective'] for p in layer['projections']))
                        model.to(device)
                        logits = model(samples.to(device), use_cache=False).logits.cpu()
                        model.cpu()
                        save_bundle(model, output / 'checkpoint.pt', rank, 'refine_' + arm, report)
                        restored, metadata = load_bundle(output / 'checkpoint.pt')
                        self.assertTrue(metadata['fake_quantization'])
                        torch.testing.assert_close(restored.to(device)(samples.to(device), use_cache=False).logits.cpu(), logits, atol=0, rtol=0)
                        self.assertGreater(evaluate_tokens(restored, samples, 16)['ppl'], 0)
                        if arm != 'paired':
                            continue
                        interrupted = root / (arm + '_interrupted')
                        interrupted.mkdir()
                        def fail_after_first(index, positional):
                            if index == 0:
                                raise RuntimeError('simulated interruption')
                        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                            refine_model(copy.deepcopy(dense), baseline, loader, interrupted, rank=rank, arm=arm,
                                         settings=settings, device=device, tokens=8, scale_grid=2,
                                         on_layer=fail_after_first)
                        resumed = copy.deepcopy(dense)
                        refine_model(resumed, baseline, loader, interrupted, rank=rank, arm=arm,
                                     settings=settings, device=device, tokens=8, scale_grid=2, resume=True)
                        for name, value in model.state_dict().items():
                            torch.testing.assert_close(resumed.state_dict()[name], value, atol=0, rtol=0)
                        with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                            refine_model(copy.deepcopy(dense), baseline, loader, interrupted, rank=rank, arm='coordinate',
                                         settings=settings, device=device, tokens=8, scale_grid=2, resume=True)
                        self.assertEqual(audit_model(resumed, rank)['violations_2_4'], 0)

    def test_cpu_both_ranks_all_arms_propagation_reload_resume(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.exercise('cpu', torch.float32, (32, 128), ('awq', 'coordinate', 'paired'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_fp16_both_ranks_propagation_reload_resume(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.exercise('cuda:0', torch.float16, (32, 128), ('paired',))


if __name__ == '__main__':
    unittest.main()
