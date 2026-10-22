"""Copy an exact locked local source into a new read-only snapshot.

Only copies explicitly locked files, verifies all copied bytes, and leaves
the published source unchanged. No model/library/GPU initialization occurs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_lock.read_text(encoding='utf-8'))
    root = Path(source['path']).resolve()
    target = args.output.resolve()
    if target.exists() or args.receipt.exists() or target == root or target.is_relative_to(root):
        raise ValueError('snapshot and receipt must be new; source cannot be the destination')
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    if actual != set(source['files']):
        raise ValueError('source inventory changed')
    target.mkdir(parents=True, exist_ok=False)
    total = 0
    for name, expected in source['files'].items():
        relative = Path(name)
        origin = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not origin.is_relative_to(root):
            raise ValueError('source lock escapes the selected source')
        copied = target / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, copied)
        if sha(copied) != expected:
            raise ValueError(f'source or copied bytes changed: {name}')
        total += copied.stat().st_size
        copied.chmod(stat.S_IREAD)
    receipt = {'status': 'EXACT_READ_ONLY_SOURCE_SNAPSHOT_ONLY',
               'created_utc': datetime.now(timezone.utc).isoformat(),
               'published_source': str(root), 'copied_file_bytes': total,
               'published_source_unchanged': True, 'model_loaded': False,
               'source': {'path': str(target), 'files': source['files']}}
    with args.receipt.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'path': str(target), 'files': len(actual),
                      'copied_file_bytes': total}))


if __name__ == '__main__':
    main()
