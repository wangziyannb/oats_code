"""Wait for the user's browser OAuth, then continue the already authorized transfer.

Authentication output may contain credentials. Keep it in a mode-0600 file
outside the repository; never print raw authentication output.
"""
from pathlib import Path
import os
import subprocess
import sys

WORK = Path(__file__).parent.resolve()
CONFIG = Path.home() / '.config/rclone/oats_upload.conf'
AUTH_LOG = CONFIG.with_name('oats_upload_auth.log')

fd = os.open(AUTH_LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, 'w') as log:
    result = subprocess.run([
        'rclone', 'config', 'reconnect', 'gdrive_oats:',
        '--config', str(CONFIG), '--auto-confirm',
    ], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
if result.returncode:
    print('Google authorization did not complete; all local models remain.', flush=True)
    sys.exit(result.returncode)
print('Google authorization completed. Continuing verified model upload.', flush=True)
with (WORK / 'transfer.log').open('a') as log:
    result = subprocess.run([sys.executable, str(WORK / 'transfer.py')],
                            stdout=log, stderr=subprocess.STDOUT)
print('Upload process exit status:', result.returncode, flush=True)
sys.exit(result.returncode)
