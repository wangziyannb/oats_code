"""Independent exhaustive checks for the new integer-code optimizer (offline)."""
import copy
import itertools
import tempfile
import unittest

import torch

from pruning_methods.OATS.awq.refine import (CodeMatrix, DiagonalRefiner, RefineConfig,
                                            initialize_awq, nearest_code)
from pruning_methods.OATS.awq.quantize import AWQConfig, fake_quant


class RefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(917)

    def fixture(self, dtype=torch.float64, device='cpu', rank=2, width=8, rows=3):
        w = torch.randn(rows, width, dtype=dtype, device=device)
        h = torch.linspace(.2, 2, width, dtype=dtype, device=device)
        s = torch.randn_like(w)
        s.reshape(rows, -1, 4)[:, :, 2:] = 0
        u = torch.randn(rows, rank, dtype=dtype, device=device)
        v = torch.randn(rank, width, dtype=dtype, device=device)
        grids = {'S': CodeMatrix.from_weight(s, 4, preserve_mask=True),
                 'U': CodeMatrix.from_weight(u, min(rank, 4)),
                 'V': CodeMatrix.from_weight(v, 4)}
        return w, h, grids

    @staticmethod
    def loss(w, h, grids):
        s, u, v = (grids[k].values() for k in ('S', 'U', 'V'))
        return float(((w - s - u @ v).square() * h).sum() / 2)

    def test_grid_reference_mask_and_roundtrip(self):
        for sparse in (False, True):
            x = torch.randn(6, 16)
            if sparse:
                x.reshape(6, -1, 4)[:, :, 2:] = 0
            grid = CodeMatrix.from_weight(x, 4, preserve_mask=sparse)
            torch.testing.assert_close(grid.values(), fake_quant(x, AWQConfig(group_size=4), sparse), atol=1e-6, rtol=1e-6)
            with tempfile.TemporaryFile() as f:
                torch.save(grid.state_dict(), f); f.seek(0)
                loaded = CodeMatrix.from_state_dict(torch.load(f, weights_only=True))
            torch.testing.assert_close(loaded.values(), grid.values(), atol=0, rtol=0)
        # Zero inputs, and endpoints where choosing +/-step naively would be illegal.
        z = CodeMatrix.from_weight(torch.zeros(4, 8), 4)
        self.assertEqual(float(z.values().abs().sum()), 0)
        for zero in (0, 15):
            code = nearest_code(torch.tensor([0.]), torch.ones(1), torch.tensor([zero]), active=torch.tensor([True]))
            self.assertNotEqual(int(code), zero)
            self.assertTrue(0 <= int(code) <= 15)

    def test_coordinate_sweeps_equal_exhaustive_discrete_minimizers(self):
        w, h, initial = self.fixture()
        for branch, operation in [('U', 'codes_u'), ('V', 'codes_v'), ('S', 'codes_s')]:
            expected = copy.deepcopy(initial)
            grid = expected[branch]
            if branch == 'U':
                order = [(i, k) for k in range(grid.codes.shape[1]) for i in range(grid.codes.shape[0])]
            else:
                order = list(itertools.product(range(grid.codes.shape[0]), range(grid.codes.shape[1])))
            for i, j in order:
                if grid.mask is not None and not grid.mask[i, j]:
                    continue
                old = int(grid.codes[i, j]); best, best_code = self.loss(w, h, expected), old
                for code in range(16):
                    if grid.mask is not None and code == int(grid.zeros[i, j // grid.group_size]):
                        continue
                    grid.codes[i, j] = code
                    value = self.loss(w, h, expected)
                    if value < best:
                        best, best_code = value, code
                grid.codes[i, j] = best_code
            actual = DiagonalRefiner(w, h, initial)
            actual.phase(operation, getattr(actual, operation))
            torch.testing.assert_close(actual.grids[branch].codes, grid.codes, atol=0, rtol=0)

    def test_pair_sweeps_equal_full_legal_pair_enumeration(self):
        w, h, initial = self.fixture(rows=2, width=4)
        for branch, operation in [('U', 'pairs_su'), ('V', 'pairs_sv')]:
            expected = copy.deepcopy(initial)
            sg, fg = expected['S'], expected[branch]
            for k in range(2):
                u, v = expected['U'].values(), expected['V'].values()
                if branch == 'U':
                    candidates = (v[k].square() * h).topk(4).indices.tolist()
                    coordinates = [(i, next(j for j in candidates if sg.mask[i,j]), i, k) for i in range(w.shape[0])]
                else:
                    candidates = u[:, k].square().topk(w.shape[0]).indices.tolist()
                    coordinates = [(next(i for i in candidates if sg.mask[i,j]), j, k, j)
                                   for j in range(w.shape[1]) if any(sg.mask[i,j] for i in candidates)]
                for i,j,fi,fj in coordinates:
                    old_s, old_f = int(sg.codes[i,j]), int(fg.codes[fi,fj])
                    best, best_codes = self.loss(w,h,expected), (old_s,old_f)
                    for sc,fc in itertools.product(range(16),repeat=2):
                        if sc == int(sg.zeros[i,j//sg.group_size]):
                            continue
                        sg.codes[i,j],fg.codes[fi,fj]=sc,fc
                        value=self.loss(w,h,expected)
                        if value < best:
                            best,best_codes=value,(sc,fc)
                    sg.codes[i,j],fg.codes[fi,fj]=best_codes
            actual=DiagonalRefiner(w,h,initial,RefineConfig(pair_candidates=16))
            actual.phase(operation,getattr(actual,operation))
            self.assertAlmostEqual(actual.objective(),self.loss(w,h,expected),places=10)
            for key in expected:
                torch.testing.assert_close(actual.grids[key].codes,expected[key].codes,atol=0,rtol=0)

    def test_pair_escapes_a_coordinatewise_fixed_point(self):
        def grid(value,step,mask=None):
            value=torch.tensor(value,dtype=torch.float64)
            rows,cols=value.shape
            return CodeMatrix((value/step+8).round().to(torch.uint8),torch.full((rows,1),step,dtype=torch.float64),
                torch.full((rows,1),8,dtype=torch.uint8),torch.ones(rows,dtype=torch.float64),
                torch.ones(cols,dtype=torch.float64),cols,mask=mask)
        mask=torch.tensor([[True,True,False,False]])
        grids={'S':grid([[1,1,0,0]],1.,mask),'U':grid([[0]],.9),'V':grid([[1,0,0,0]],1.)}
        ref=DiagonalRefiner(torch.tensor([[1.1,1,0,0]],dtype=torch.float64),torch.ones(4,dtype=torch.float64),grids)
        before=ref.objective()
        for name in ('codes_u','codes_v','codes_s'):
            ref.phase(name,getattr(ref,name))
        self.assertAlmostEqual(ref.objective(),before)
        ref.phase('pairs_su',ref.pairs_su)
        self.assertLess(ref.objective(),1e-25)
        self.assertTrue(torch.equal(ref.s.ne(0),mask))

    def test_scale_steps_and_guard(self):
        w,h,grids=self.fixture()
        for name in ('scales_u','scales_v','scales_s'):
            ref=DiagonalRefiner(w,h,grids)
            before=ref.objective()
            record=ref.phase(name,getattr(ref,name))
            self.assertTrue(record['accepted'])
            self.assertLessEqual(ref.objective(),before)
            for key in grids:
                torch.testing.assert_close(ref.grids[key].codes,grids[key].codes,atol=0,rtol=0)
        ref=DiagonalRefiner(w,h,grids)
        old=ref.objective()
        record=ref.phase('deliberately_bad',lambda:ref.grids['U'].steps.mul_(1e10))
        self.assertFalse(record['accepted'])
        self.assertEqual(ref.objective(),old)
        for key in grids:
            torch.testing.assert_close(ref.grids[key].values(),grids[key].values(),atol=0,rtol=0)

    def test_sparse_scale_cannot_underflow_when_exported_to_fp16(self):
        w,h,grids=self.fixture()
        # The target has no residual for S. An unconstrained scale fit collapses
        # the still-active sparse codes to zero when cast to FP16.
        w=grids['U'].values()@grids['V'].values()
        ref=DiagonalRefiner(w,h,grids)
        record=ref.phase('scales_s',ref.scales_s)
        self.assertTrue(record['accepted'])
        self.assertTrue(torch.equal(ref.s.half().ne(0),grids['S'].mask))
        self.assertLessEqual(record['after'],record['before'])
        before=ref.s.clone()
        record=ref.phase('underflow',lambda:ref.grids['S'].steps.mul_(1e-9))
        self.assertFalse(record['accepted'])
        self.assertFalse(record['proposed_storage_valid'])
        torch.testing.assert_close(ref.s,before,atol=0,rtol=0)

    def test_invalid_inputs(self):
        for bad in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):CodeMatrix.from_weight(torch.full((2,4),bad),4)
        with self.assertRaises(ValueError):CodeMatrix.from_weight(torch.randn(2,5),4)
        with self.assertRaises(ValueError):RefineConfig(sweeps=-1)
        w,h,grids=self.fixture()
        with self.assertRaises(ValueError):DiagonalRefiner(w,torch.zeros_like(h),grids)
        broken=copy.deepcopy(grids);broken['S'].mask[0,0]=False
        with self.assertRaises(ValueError):DiagonalRefiner(w,h,broken)
        broken=copy.deepcopy(grids);broken['U'].codes[0,0]=17
        with self.assertRaises(ValueError):DiagonalRefiner(w,h,broken)

    def exercise_ranks(self,device):
        for rank in (32,128):
            w,h,grids=self.fixture(torch.float32,device,rank=rank,width=128,rows=16)
            initial=copy.deepcopy(grids)
            for paired in (False,True):
                ref=DiagonalRefiner(w,h,grids,RefineConfig(sweeps=2,pair_candidates=4))
                before=ref.objective();ref.run(paired=paired)
                self.assertLessEqual(ref.objective(),before)
                self.assertTrue(all(r['after']<=r['before'] for r in ref.records))
                for key in grids:
                    torch.testing.assert_close(grids[key].values(),initial[key].values(),atol=0,rtol=0)
                payload=ref.state_dict()
                loaded={k:CodeMatrix.from_state_dict(v,device) for k,v in payload['grids'].items()}
                replay=DiagonalRefiner(w,h,loaded)
                torch.testing.assert_close(replay.s+replay.u@replay.v,ref.s+ref.u@ref.v,atol=0,rtol=0)
                self.assertTrue(torch.equal(ref.s.ne(0),initial['S'].mask))

    def test_both_ranks_cpu_and_complete_code_state_reload(self):
        self.exercise_ranks('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_both_ranks_cuda_and_fp16_export(self):
        self.exercise_ranks('cuda:0')
        w,h,grids=self.fixture(torch.float32,'cuda:0',rank=32,width=128,rows=16)
        samples=torch.randn(16,128,device='cuda:0',dtype=torch.float16)
        init,_=initialize_awq(w.half(),grids['S'].values().half(),grids['U'].values().half(),grids['V'].values().half(),samples,grid=2)
        ref=DiagonalRefiner(w,h,init,RefineConfig(sweeps=1));ref.run(paired=True)
        s,u,v=(ref.grids[k].values().half() for k in ('S','U','V'))
        self.assertTrue(torch.equal(s.ne(0),init['S'].mask))
        out=torch.nn.functional.linear(samples,s)+torch.nn.functional.linear(torch.nn.functional.linear(samples,v),u)
        self.assertTrue(torch.isfinite(out).all())


if __name__=='__main__':
    unittest.main()
