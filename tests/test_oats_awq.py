"""Quantization semantics and frozen-OATS regression tests; no model downloads."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from general_utils.config import config
from models.adapters.llama_adapter import LlamaModelAdapter
from pruning_methods.OATS.oats_wrapper import WrappedOATS
from pruning_methods.OATS.prune_oats_compress import altern_ls, prune_oats_compress
from pruning_methods.OATS.awq.quantize import (AWQConfig, InLoopQuantizer,
    fake_quant, effective_group_size, search_pair, quantize_pair_at_scales)
from pruning_methods.OATS.awq.posthoc import (prebalance_factors, apply_rank_gauge_scale,
    apply_norm_to_residue_scale, apply_residue_to_residue_scale)
from pruning_methods.OATS.compressed_linear import CompressedLinear
from pruning_methods.OATS.awq.reference.quantizer import pseudo_quantize_tensor
from pruning_methods.OATS.awq.pipeline import (audit_model, claim_run, load_bundle,
    load_legacy_layers, save_bundle, quantize_posthoc, evaluate_tokens)
from oats_original_reference import altern_ls as original_altern_ls
from run_oats_awq import parse_args, posthoc_settings


class IdentityQuantizer:
    def pair(self, first, second, residual, iteration):
        return first, second

    def sparse(self, sparse, residual, iteration):
        return sparse


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        config.device = torch.device('cpu')
        config.dtype = torch.float32

    def run_silent(self, fn, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def test_unquantized_matches_original_exactly_each_iteration(self):
        w = torch.randn(8, 12)
        d = torch.linspace(0.2, 2, 12).unsqueeze(0)
        for iterations in (1, 2, 4):
            expected = self.run_silent(original_altern_ls, w, d, iterations, 2, .5, prune_n=2, prune_m=4)
            actual = self.run_silent(altern_ls, w, d, iterations, 2, .5, prune_n=2, prune_m=4)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_identity_quantization_preserves_oats_trajectory(self):
        w = torch.randn(8, 12)
        d = torch.linspace(0.2, 2, 12).unsqueeze(0)
        trajectory, records = [], []
        svd = torch.linalg.svd
        with patch('torch.linalg.svd', wraps=svd) as call:
            result = self.run_silent(altern_ls, w, d, 4, 2, .5, prune_n=2, prune_m=4,
                    quantizer=IdentityQuantizer(), report=records.append,
                    observer=lambda i, v, u, s: trajectory.append((v.clone(), u.clone(), s.clone())))
            self.assertEqual(call.call_count, 4)
            self.assertTrue(all(c.kwargs == {'full_matrices': False} for c in call.call_args_list))
        for index, factors in enumerate(trajectory):
            reference = self.run_silent(original_altern_ls, w, d, index+1, 2, .5, prune_n=2, prune_m=4)
            for a, b in zip(factors, reference):
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
        for a, b in zip(result, trajectory[-1]):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual([r['iteration'] for r in records], [1, 2, 3, 4])
        self.assertTrue(all(r['iterate_selection'] == 'last' for r in records))

    def test_feedback_is_quantized_and_last_iterate_is_returned(self):
        w, d = torch.randn(8, 12), torch.linspace(.1, 3, 12).unsqueeze(0)
        sqrt_d = d.sqrt()
        svd_inputs, sparse_targets, trajectory = [], [], []
        svd = torch.linalg.svd
        class ControlledQuantizer(IdentityQuantizer):
            def pair(self, first, second, residual, iteration):
                return first * .8, second * .7
            def sparse(self, sparse, residual, iteration):
                sparse_targets.append((sparse.clone(), residual.clone()))
                return sparse * (10 if iteration == 2 else .6)
        def traced_svd(value, **kwargs):
            svd_inputs.append(value.clone())
            return svd(value, **kwargs)
        with patch('torch.linalg.svd', side_effect=traced_svd):
            result = self.run_silent(altern_ls, w, d, 3, 2, .5, prune_n=2, prune_m=4,
                quantizer=ControlledQuantizer(),
                observer=lambda i, v, u, s: trajectory.append((v.clone(),u.clone(),s.clone())))
        torch.testing.assert_close(svd_inputs[0], w * sqrt_d)
        for index, (v,u,s) in enumerate(trajectory):
            torch.testing.assert_close(sparse_targets[index][1], w-u@v)
            # Native OATS projects the weighted residual, before sparse quantization.
            weighted = (w-u@v)*sqrt_d
            mask = torch.zeros_like(weighted, dtype=torch.bool)
            for col in range(0, 12, 4):
                idx = weighted[:,col:col+4].abs().topk(2, dim=1, largest=False).indices
                mask.scatter_(1, col+idx, True)
            torch.testing.assert_close(sparse_targets[index][0], weighted.masked_fill(mask,0)/sqrt_d)
            if index < 2:
                torch.testing.assert_close(svd_inputs[index+1], (w-s)*sqrt_d)
        errors = [(w-s-u@v).norm() for v,u,s in trajectory]
        self.assertGreater(errors[-1], min(errors[:-1]))
        for a,b in zip(result,trajectory[-1]):
            torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_quantizer_reference_mask_and_invalid_inputs(self):
        cfg = AWQConfig(group_size=4)
        w = torch.randn(8, 12)
        torch.testing.assert_close(fake_quant(w, cfg),
            pseudo_quantize_tensor(w, n_bit=4, zero_point=True, q_group_size=4), atol=0, rtol=0)
        sparse = torch.tensor([[0., 0., .00001, 1., 0., -.2, 0., -.00001]])
        q = fake_quant(sparse, cfg, preserve_mask=True)
        self.assertTrue(torch.equal(sparse.eq(0), q.eq(0)))
        torch.testing.assert_close(fake_quant(torch.zeros(8,12),cfg),torch.zeros(8,12))
        for rank in (32,128):
            self.assertEqual(effective_group_size(rank,128),rank)
            self.assertEqual(fake_quant(torch.randn(256,rank)).shape,(256,rank))
        with self.assertRaises(ValueError): effective_group_size(130,128)
        with self.assertRaises(ValueError): fake_quant(torch.ones(2,5),cfg)
        for bad in (float('nan'),float('inf')):
            with self.assertRaises(ValueError): fake_quant(torch.full((2,4),bad),cfg)
        with self.assertRaises(ValueError): AWQConfig(grid=0)

    def test_pair_search_handles_zero_activations_and_effective_coordinates(self):
        first, second, x = torch.randn(4,8),torch.randn(8,4),torch.zeros(3,8)
        cfg = AWQConfig(group_size=4,grid=3)
        q1,q2,scales,info=search_pair(first,second,x,torch.zeros(3,8),cfg)
        self.assertEqual(info['pair_mse'],0)
        a,b=quantize_pair_at_scales(first,second,scales['first_input'],scales['rank'],cfg)
        torch.testing.assert_close(q1,a);torch.testing.assert_close(q2,b)
        self.assertLessEqual(info['pair_mse'],info['first_search_mse'])

    def test_sparse_refresh_interval_and_actual_two_factor_output(self):
        quant=InLoopQuantizer(torch.randn(5,8),AWQConfig(group_size=4,grid=2,sparse_refresh=2),torch.float32)
        records=[]
        self.run_silent(altern_ls,torch.randn(8,8),torch.ones(1,8),3,2,.5,prune_n=2,prune_m=4,
                        quantizer=quant,report=records.append)
        self.assertEqual([r['sparse_scale_refreshed'] for r in records],[True,False,True])
        self.assertTrue(all(r['pair_scale_refreshed'] for r in records))
        self.assertTrue(all(r['output_mse']>=0 for r in records))

    @torch.no_grad()
    def test_activation_sampling_does_not_change_oats_statistics(self):
        layer=torch.nn.Linear(8,8)
        original=WrappedOATS(layer)
        sampled=WrappedOATS(layer,activation_tokens=8,calibration_sequences=4)
        for _ in range(2):
            x=torch.randn(2,6,8)
            original.add_batch(x,layer(x));sampled.add_batch(x,layer(x))
        torch.testing.assert_close(original.scaler_row,sampled.scaler_row,atol=0,rtol=0)
        self.assertEqual(sum(t.shape[0] for t in sampled.activation_samples),8)
        self.assertEqual(sampled.nsamples,original.nsamples)

    @torch.no_grad()
    def test_equivalent_scales_for_native_oats_layers(self):
        def module():
            m=CompressedLinear(8,4,8,True)
            for p in m.parameters(): p.normal_()
            return m
        x=torch.randn(2,8);m=module();before=m(x).clone()
        prebalance_factors(m,x)
        torch.testing.assert_close(m(x),before,atol=1e-5,rtol=1e-5)
        apply_rank_gauge_scale(m,torch.linspace(.5,2,4))
        torch.testing.assert_close(m(x),before,atol=1e-5,rtol=1e-5)
        norm=torch.nn.LayerNorm(8);before=m(norm(x)).clone();scale=torch.linspace(.5,2,8)
        apply_norm_to_residue_scale(norm,[m],scale)
        torch.testing.assert_close(m(norm(x)),before,atol=1e-5,rtol=1e-5)
        previous,target=module(),module();before=target(previous(x)).clone()
        apply_residue_to_residue_scale(previous,target,scale)
        torch.testing.assert_close(target(previous(x)),before,atol=2e-5,rtol=1e-5)

    def test_manifest_rejects_mixing_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)/'run';manifest={'method':'inloop','rank':32}
            claim_run(dest,manifest)
            claim_run(dest,manifest,resume=True)
            with self.assertRaises(ValueError): claim_run(dest,{'method':'prepare','rank':32},resume=True)
            with self.assertRaises(ValueError): claim_run(dest,manifest)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self,dtype):
        cfg=LlamaConfig(vocab_size=64,hidden_size=256,intermediate_size=512,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,
                max_position_embeddings=64,attention_dropout=0.)
        cfg._attn_implementation='sdpa'
        torch.manual_seed(123)
        model=LlamaForCausalLM(cfg).to(dtype).eval()
        # HF rotary computes frequencies in FP32 even for FP16 models.
        model.model.rotary_emb.float()
        return model

    @torch.no_grad()
    def exercise(self,device,dtype):
        config.device=torch.device(device);config.dtype=dtype
        samples=torch.randint(0,64,(2,16))
        for rank in (32,128):
            baseline=None
            for mode in ('prepare','inloop','posthoc'):
                with self.subTest(device=device,rank=rank,mode=mode),tempfile.TemporaryDirectory() as folder:
                    root=Path(folder)
                    if mode=='posthoc':
                        model=copy.deepcopy(baseline)
                        args=posthoc_settings(argparse_namespace(rank=rank,device=device,scale_grid=2))
                        args.clip_grid=4
                        result=quantize_posthoc(model,samples,args,root)
                        self.assertEqual(len(result['statistics']),42)
                        self.assertTrue(all(s['mask_equal'] for s in result['statistics'] if s['branch']=='S'))
                        # Replay completed post-hoc layers from the unquantized source.
                        replay=copy.deepcopy(baseline)
                        quantize_posthoc(replay,samples,args,root,resume=True)
                        for k,v in model.state_dict().items():
                            torch.testing.assert_close(v,replay.state_dict()[k],atol=0,rtol=0)
                    else:
                        model=self.fixture(dtype)
                        quant=None if mode=='prepare' else {'config':{'grid':2,'sparse_refresh':2},'tokens':8,'sequences':2}
                        reports=[]
                        hyper={'unscaled':False,'rank_ratio':.1,'fixed_rank':rank,'prune_level':'row','num_iters':2}
                        loader=[{'input_ids':samples[i:i+1]} for i in range(2)]
                        prune_oats_compress(LlamaModelAdapter(model),.5,None,loader,hyper,str(root),
                            prune_n=2,prune_m=4,quantization=quant,report=reports.append)
                        self.assertEqual(len(reports),28)
                        # Old per-layer OATS files remain loadable and shape/rank checked.
                        legacy=load_legacy_layers(self.fixture(dtype),root,rank)
                        for k,v in model.state_dict().items():
                            torch.testing.assert_close(v,legacy.state_dict()[k],atol=0,rtol=0)
                        if mode=='prepare': baseline=copy.deepcopy(model)
                    self.assertEqual(audit_model(model,rank)['modules'],14)
                    model.to(device)
                    expected=model(samples.to(device),use_cache=False).logits
                    checkpoint=root/'bundle.pt'
                    save_bundle(model,checkpoint,rank,mode,{'test':True})
                    restored,metadata=load_bundle(checkpoint)
                    self.assertEqual(metadata['method'],mode)
                    restored.to(device)
                    actual=restored(samples.to(device),use_cache=False).logits
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                    metrics=evaluate_tokens(restored,samples,16)
                    self.assertGreater(metrics['ppl'],0)
                    self.assertEqual(metrics['predicted_tokens'],30)

    def test_cpu_both_ranks_modes_and_reload(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.exercise('cpu',torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_fp16_both_ranks_modes_and_reload(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.exercise('cuda:0',torch.float16)


def argparse_namespace(**kwargs):
    import argparse
    return argparse.Namespace(**kwargs)


if __name__=='__main__':
    unittest.main()
