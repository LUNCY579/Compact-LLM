"""Snapshot or verify preserved historical source/evidence without sealed access."""
import argparse
import json
from pathlib import Path
import sys

PROGRAM = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROGRAM))
from shared.hashing import canonical_sha256,sha256_file,write_json
from shared.guards import guard_non_sealed_path
from shared.legacy import REPO,ARMR,GRAFT


def inventory():
    files = {}
    for root in (ARMR,GRAFT):
        for path in root.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts:
                continue
            guard_non_sealed_path(path)
            files[path.relative_to(REPO).as_posix()] = sha256_file(path)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    current = inventory()
    if args.verify:
        expected = json.loads(args.snapshot.read_text(encoding='utf-8'))
        if expected['files'] != current or expected['root_sha256'] != canonical_sha256(current):
            raise RuntimeError('historical source or evidence changed')
        print(json.dumps({'status':'PASS','files':len(current),'root_sha256':expected['root_sha256']}))
    else:
        write_json(args.snapshot,{'files':current,'root_sha256':canonical_sha256(current),
                   'scope':'both historical experiments including local artifacts, excluding Python caches',
                   'sealed_data_opened':False})
        print(json.dumps({'status':'SNAPSHOT','files':len(current),'root_sha256':canonical_sha256(current)}))


if __name__ == '__main__':
    main()
