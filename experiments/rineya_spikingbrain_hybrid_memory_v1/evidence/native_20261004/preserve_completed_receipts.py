"""Preserve completed JSON receipts outside ignored runs; never alter artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/titan_transition_v2_20261004_memory_efficient'
DESTINATION = Path(__file__).resolve().parent/'completed_receipts'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    assert (RUN/'comparison.json').is_file()
    assert len(list(RUN.glob('*/receipt.json'))) == 21
    DESTINATION.mkdir(exist_ok=False)
    paths = [RUN/'comparison.json']
    for filename in ('receipt.json', 'request.json', 'delta.json'):
        paths.extend(sorted(RUN.glob('*/'+filename)))
    paths.append(RUN/'teacher/teacher.json')
    manifest = {'status': 'EXACT_COMPLETED_RECEIPTS_PRESERVED',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_run': str(RUN), 'matrix_exit_code_observed': 0, 'files': {},
                'binary_artifacts_remain_in_source_run': True}
    for source in paths:
        relative = source.relative_to(RUN)
        destination = DESTINATION/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        observed = sha(source)
        assert sha(destination) == observed
        manifest['files'][str(relative)] = {'sha256': observed, 'bytes': source.stat().st_size}
    with (DESTINATION/'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': manifest['status'], 'files': len(paths),
                      'bytes': sum(item['bytes'] for item in manifest['files'].values())}))


if __name__ == '__main__':
    main()
