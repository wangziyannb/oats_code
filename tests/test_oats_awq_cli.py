"""Offline production-entrypoint and actual interrupted-run recovery checks."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from run_oats_awq import parse_args, run
from pruning_methods.OATS.awq.pipeline import load_bundle


class EntryPointTests(unittest.TestCase):
    def test_offline_entrypoints_and_interrupted_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder)
            cfg=LlamaConfig(vocab_size=64,hidden_size=256,intermediate_size=512,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=64)
            torch.manual_seed(321)
            LlamaForCausalLM(cfg).save_pretrained(root/'base')
            samples=torch.randint(0,64,(2,16))
            torch.save([{'input_ids':ids} for ids in samples],root/'cal.pt')
            torch.save(samples,root/'tokens.pt')
            def args(mode,output,extra=()):
                return parse_args([mode,'--model',str(root/'base'),'--rank','32',
                    '--output-dir',str(root/output),'--device','cpu','--dtype','fp32',
                    '--num-iters','2','--scale-grid','2','--awq-tokens','8',
                    '--cal-batch-size','1',*extra])
            dry=args('prepare','dry',['--dry-run','--calibration-path',str(root/'cal.pt')])
            run(dry)
            self.assertFalse((root/'dry').exists())
            common=['--calibration-path',str(root/'cal.pt')]
            for mode in ('prepare','inloop'):
                production=args(mode,mode,common)
                run(production)
                complete,_=load_bundle(root/mode/'checkpoint.pt')
                interrupted=args(mode,mode+'_resume',common)
                prune_module=importlib.import_module('pruning_methods.OATS.prune_oats_compress')
                original_save=prune_module.save_checkpoint
                def stop_after_committed_layer(*a,**kw):
                    original_save(*a,**kw)
                    raise RuntimeError('simulated interruption after layer commit')
                with patch.object(prune_module,'save_checkpoint',side_effect=stop_after_committed_layer):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                        run(interrupted)
                changed=copy.deepcopy(interrupted);changed.resume=True;changed.num_iters=3
                with self.assertRaisesRegex(ValueError,'configuration'):
                    run(changed)
                interrupted.resume=True
                run(interrupted)
                restored,_=load_bundle(root/(mode+'_resume')/'checkpoint.pt')
                for key,value in complete.state_dict().items():
                    torch.testing.assert_close(value,restored.state_dict()[key],atol=0,rtol=0)
                rows=[json.loads(line) for line in (root/(mode+'_resume')/'iterations.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),28)
            source=str(root/'prepare'/'checkpoint.pt')
            run(args('posthoc','posthoc',['--checkpoint',source,'--calibration-path',str(root/'tokens.pt')]))
            for mode in ('prepare','inloop','posthoc'):
                run(args('eval',mode+'_eval',['--checkpoint',str(root/mode/'checkpoint.pt'),
                    '--eval-tokens',str(root/'tokens.pt'),'--ppl-seqlen','16']))
                result=json.loads((root/(mode+'_eval')/'metrics.json').read_text())
                self.assertGreater(result['ppl'],0)
                self.assertEqual(result['checkpoint_method'],mode)
            with self.assertRaisesRegex(ValueError,'unquantized'):
                run(args('posthoc','bad_source',['--checkpoint',str(root/'inloop'/'checkpoint.pt'),
                    '--calibration-path',str(root/'tokens.pt')]))


if __name__=='__main__':
    unittest.main()
