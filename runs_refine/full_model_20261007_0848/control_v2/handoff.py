"""Let the active quantizer finish, then resume future jobs on physical 4-7."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path(__file__).resolve().parent.parent
request = json.loads((root / 'gpu_switch_request.json').read_text())
active = request['active_stages']
assert len(active) == 1 and (active[0]['rank'], active[0]['arm'], active[0]['mode']) == (128, 'awq', 'quantize')
child = active[0]['pid']


def process_state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
    except FileNotFoundError:
        return None


print(f'Waiting for active rank128 AWQ process {child}; no new task will start on 0-3.', flush=True)
while process_state(child) not in (None, 'Z'):
    time.sleep(5)
completed = json.loads((root / 'r128/awq/status.json').read_text())
if completed['status'] != 'complete' or not (root / 'r128/awq/checkpoint.pt').exists():
    raise RuntimeError('Active quantization did not finish successfully; inspect its log before continuing')
old = request['old_dispatcher_pid']
if process_state(old) is not None:
    cmd = Path(f'/proc/{old}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert str(root / 'code/run_oats_refine_study.py') in cmd
    assert process_state(old) in ('T', 't')
    (root / 'control_v2/status_before_handoff.json').write_bytes((root / 'status.json').read_bytes())
    # This stopped dispatcher has no running experiment child left.
    os.kill(old, signal.SIGKILL)
    time.sleep(1)
original = json.loads((root / 'launch.json').read_text())['argv']
argv = [sys.executable, '-u', str(root / 'control_v2/run_oats_refine_study.py')]
argv += original[original.index('--output-dir'):]
argv += ['--code-dir', str(root / 'code'), '--resume']
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
(root / 'control_v2/launch.json').write_text(json.dumps({
    'argv': argv, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
    'handoff_at': datetime.now(timezone.utc).isoformat(),
    'reason': 'User requested physical GPUs 4-7 for subsequent experiments'}, indent=2) + '\n')
print('Active task completed; resuming all remaining stages on physical GPUs 4-7.', flush=True)
os.execv(sys.executable, argv)
