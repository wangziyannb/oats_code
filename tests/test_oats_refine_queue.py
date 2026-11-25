"""Queue recovery/device handoff; no models, GPUs, or subprocesses."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_oats_refine_study as study


class QueueTests(unittest.TestCase):
    def test_resume_skips_artifacts_and_changes_only_future_gpu_assignments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            started = []
            class Process:
                def __init__(self, argv, **kwargs):
                    self.pid = 100 + len(started)
                    output = Path(argv[argv.index('--output-dir') + 1])
                    output.mkdir(parents=True, exist_ok=True)
                    mode = argv[3]
                    started.append((mode, output, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                    if mode == 'quantize':
                        (output / 'checkpoint.pt').touch()
                        (output / 'status.json').write_text(json.dumps({'status': 'complete'}))
                        (output / 'quant_report.json').write_text(json.dumps({'audit': {'violations_2_4': 0}}))
                    else:
                        (output / 'metrics.json').write_text(json.dumps({'reload_logits_exact': True, 'ppl': 25.}))
                def poll(self):
                    return 0
            argv = ['study', '--output-dir', str(root), '--baseline-dir', str(root / 'baseline'),
                    '--model', 'test-model', '--calibration-path', str(root / 'calibration'),
                    '--eval-tokens', str(root / 'tokens')]
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'), \
                 patch.object(study.subprocess, 'Popen', Process), patch.object(study.signal, 'signal'), \
                 contextlib.redirect_stdout(io.StringIO()):
                study.main()
            self.assertEqual(len(started), 12)
            state = json.loads((root / 'status.json').read_text())
            state.update(status='switch_pending')
            state['stages'] = state['stages'][:7]
            state['stages'][-1]['status'] = 'running'  # Child finished while dispatcher was stopped.
            (root / 'status.json').write_text(json.dumps(state))
            results = json.loads((root / 'results.json').read_text())
            (root / 'results.json').write_text(json.dumps([r for r in results if r['rank'] == 32]))
            for name in ('coordinate', 'paired', 'awq_eval', 'coordinate_eval', 'paired_eval'):
                shutil.rmtree(root / 'r128' / name)
            started.clear()
            with patch.object(sys, 'argv', argv + ['--resume']), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='4,5,6,7'), \
                 patch.object(study.subprocess, 'Popen', Process), patch.object(study.signal, 'signal'), \
                 contextlib.redirect_stdout(io.StringIO()):
                study.main()
            self.assertEqual(len(started), 5)
            self.assertTrue(all(mask == '4,5,6,7' for _, _, mask in started))
            state = json.loads((root / 'status.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(len(state['stages']), 12)
            self.assertTrue(all(s['physical_gpus'] == [0,1,2,3] for s in state['stages'][:7]))
            self.assertTrue(all(s['physical_gpus'] == [4,5,6,7] for s in state['stages'][7:]))
            self.assertEqual(len(json.loads((root / 'results.json').read_text())), 6)


if __name__ == '__main__':
    unittest.main()
