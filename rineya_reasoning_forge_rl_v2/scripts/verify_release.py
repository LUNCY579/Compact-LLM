"""Verify immutable release files, then optionally rerun semantic dataset checks."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_rl.common import dumps
from forge_rl.validate import verify_manifest, validate
from forge_rl.audits import audit_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--fast', action='store_true', help='Hashes and byte counts only; do not recompute solvers')
    args = parser.parse_args()
    checked = verify_manifest(args.root, 'RELEASE_MANIFEST.json')
    print(dumps({'status': 'PASS', 'release_files_verified': checked, 'scope': 'sha256 and byte lengths'}))
    if not args.fast:
        print(dumps(validate(args.root)))
        print(dumps(audit_release(args.root)))


if __name__ == '__main__':
    main()
