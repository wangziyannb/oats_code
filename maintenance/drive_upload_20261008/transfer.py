"""Upload the explicitly inventoried models, verify Drive hashes, then unlink locally."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).parent
REMOTE = 'gdrive_oats:'
FOLDER_ID = '1O51iAZNdpGColYAD9haZaTcgku0Fyakq'
FOLDER_URL = 'https://drive.google.com/drive/folders/1O51iAZNdpGColYAD9haZaTcgku0Fyakq'
BASE_FLAGS = ['--config', str(Path.home() / '.config/rclone/oats_upload.conf'),
              '--drive-root-folder-id', FOLDER_ID, '--retries', '5',
              '--low-level-retries', '20', '--contimeout', '30s', '--timeout', '5m']
STATE_FILE = WORK / 'state.json'
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def event(message):
    print(now(), message, flush=True)


def run(args, *, log=None):
    command = ['rclone', *args, *BASE_FLAGS]
    if log:
        with log.open('a') as out:
            subprocess.run(command, check=True, stdout=out, stderr=subprocess.STDOUT)
        return None
    return subprocess.check_output(command, text=True)


def snapshot(path):
    assert path.is_file() and not path.is_symlink(), path
    stat = path.stat()
    return {'inode': stat.st_ino, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def fingerprint(entry):
    path = ROOT / entry['path']
    before = snapshot(path)
    assert before['bytes'] == entry['bytes'], path
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(16 * 1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    assert snapshot(path) == before, f'Changed while hashing: {path}'
    return {'source_identity': before, 'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}


def verified_listing(models):
    values = json.loads(run(['lsjson', REMOTE, '--recursive', '--files-only', '--hash-type', 'MD5']))
    atomic_json(WORK / 'drive_listing.json', values)
    by_path = {}
    for value in values:
        by_path.setdefault(value['Path'], []).append(value)
    result = {}
    for entry in models:
        matches = by_path.get(entry['path'], [])
        assert len(matches) == 1, f'Missing or ambiguous remote: {entry["path"]}'
        saved = matches[0]
        assert not saved['IsDir'] and saved['Size'] == entry['bytes'], saved
        hashes = {key.lower(): value.lower() for key, value in saved.get('Hashes', {}).items()}
        assert hashes.get('md5') == entry['md5'], saved
        assert saved.get('ID'), saved
        result[entry['path']] = saved
    return result


def main():
    lock_file = (WORK / 'transfer.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = json.loads((ROOT / 'maintenance/cleanup_20261008/summary.json').read_text())
    models = source['models']
    assert len(models) == 22 and sum(x['bytes'] for x in models) == 298643233198
    for entry in models:
        relative = Path(entry['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.name == 'checkpoint.pt' and snapshot(ROOT / relative)['bytes'] == entry['bytes']
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        assert state['status'] in ('failed', 'awaiting_google_authorization') and state['deleted_count'] == 0
        assert state['destination_folder_id'] == FOLDER_ID
        assert [x['path'] for x in state['models']] == [x['path'] for x in models]
        assert [x['bytes'] for x in state['models']] == [x['bytes'] for x in models]
        models = state['models']
        for entry in models:
            if 'source_identity' in entry:
                assert snapshot(ROOT / entry['path']) == entry['source_identity'], entry['path']
        state['previous_error'] = state.pop('error', None)
        state['resumed_at'] = now()
        state['status'] = 'hashing'
    else:
        state = {'status': 'hashing', 'started_at': now(), 'destination_folder_id': FOLDER_ID,
                 'destination_url': FOLDER_URL, 'models': models, 'deleted_count': 0,
                 'deleted_bytes': 0, 'user_authorized_upload_and_local_deletion': True}
    atomic_json(STATE_FILE, state)
    try:
        event('Hashing all 22 complete models (MD5 and SHA256).')
        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = {executor.submit(fingerprint, entry): entry for entry in models if 'md5' not in entry}
            for job in as_completed(jobs):
                entry = jobs[job]
                entry.update(job.result())
                entry['status'] = 'hashed'
                atomic_json(STATE_FILE, state)
                event('Hashed ' + entry['path'])
        manifest = {
            'destination': 'hopper3/oats_code', 'destination_url': FOLDER_URL,
            'created_at': state['started_at'], 'model_count': 22,
            'total_model_bytes': sum(x['bytes'] for x in models),
            'models': [{k: v for k, v in x.items() if k not in ('status', 'source_identity')} for x in models],
        }
        atomic_json(WORK / 'models_manifest.json', manifest)
        (WORK / 'model_paths.txt').write_text(''.join(x['path'] + '\n' for x in models))
        (WORK / 'MD5SUMS').write_text(''.join(x['md5'] + '  ' + x['path'] + '\n' for x in models))
        (WORK / 'SHA256SUMS').write_text(''.join(x['sha256'] + '  ' + x['path'] + '\n' for x in models))
        state['status'] = 'uploading_manifest'
        atomic_json(STATE_FILE, state)
        for filename in ('models_manifest.json', 'MD5SUMS', 'SHA256SUMS'):
            run(['copyto', str(WORK / filename), REMOTE + filename, '--checksum', '--immutable'])
        state['status'] = 'uploading'
        atomic_json(STATE_FILE, state)
        event('Uploading with 4 concurrent transfers; no local deletion during upload.')
        run(['copy', str(ROOT), REMOTE, '--files-from-raw', str(WORK / 'model_paths.txt'),
             '--checksum', '--immutable', '--transfers', '4', '--checkers', '8',
             '--drive-chunk-size', '64M', '--drive-stop-on-upload-limit',
             '--stats', '30s', '--stats-log-level', 'INFO', '--log-level', 'INFO',
             '--use-json-log'], log=WORK / 'rclone.jsonl')
        state['status'] = 'verifying'
        atomic_json(STATE_FILE, state)
        event('Upload command succeeded. Independently verifying all Drive sizes and MD5 hashes.')
        remote = verified_listing(models)
        for entry in models:
            assert snapshot(ROOT / entry['path']) == entry['source_identity'], entry['path']
            entry['drive_file_id'] = remote[entry['path']]['ID']
            entry['status'] = 'verified'
            entry['verified_at'] = now()
        state['status'] = 'verified'
        atomic_json(STATE_FILE, state)
        # Save the verified file-ID/hash receipt remotely before deleting any local model.
        atomic_json(WORK / 'verified_receipt.json', state)
        run(['copyto', str(WORK / 'verified_receipt.json'), REMOTE + 'verified_receipt.json',
             '--checksum', '--immutable'])
        state['status'] = 'deleting_local_models'
        atomic_json(STATE_FILE, state)
        for entry in models:
            path = ROOT / entry['path']
            assert snapshot(path) == entry['source_identity'], path
            pointer = {
                'original_path': entry['path'], 'bytes': entry['bytes'],
                'md5': entry['md5'], 'sha256': entry['sha256'],
                'drive_file_id': entry['drive_file_id'], 'drive_folder_url': FOLDER_URL,
                'remote_relative_path': entry['path'], 'verified_at': entry['verified_at'],
                'load_function': entry['load_function'],
            }
            atomic_json(path.with_suffix('.drive.json'), pointer)
            path.unlink()
            entry['status'] = 'local_deleted'
            entry['deleted_at'] = now()
            state['deleted_count'] += 1
            state['deleted_bytes'] += entry['bytes']
            atomic_json(STATE_FILE, state)
            event('Verified on Drive and deleted locally: ' + entry['path'])
        # Final remote inventory remains authoritative after the local unlink step.
        verified_listing(models)
        state['status'] = 'complete'
        state['completed_at'] = now()
        atomic_json(STATE_FILE, state)
        run(['copyto', str(STATE_FILE), REMOTE + 'completed_transfer.json', '--checksum', '--immutable'])
        event('Complete: 22 verified models on Drive, 298643233198 bytes removed locally.')
    except BaseException as error:
        state['failed_during'] = state['status']
        state['status'] = 'failed'
        state['error'] = repr(error)
        atomic_json(STATE_FILE, state)
        event('Stopped: ' + repr(error))
        raise


if __name__ == '__main__':
    main()
