"""Read-only, dependency-free integrity and blocking report. Cannot launch training."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def verify_manifest(root: Path = ROOT) -> str:
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    files = manifest['sha256']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*.py')}
    actual |= {'README.md', 'registration.json'}
    if set(files) != actual:
        raise ValueError('missing or unexpected scoped source file')
    for rel, expected in files.items():
        path = PurePosixPath(rel)
        if path.is_absolute() or '..' in path.parts or '\\' in rel:
            raise ValueError('unsafe manifest path')
        target = root.joinpath(*path.parts)
        if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('nonregular or escaping source')
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError('source hash mismatch: ' + rel)
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    value = hashlib.sha256(encoded).hexdigest()
    if value != manifest['source_sha256']:
        raise ValueError('aggregate digest mismatch')
    return value


def report(root: Path = ROOT) -> dict:
    digest = verify_manifest(root)
    r = json.loads((root / 'registration.json').read_text(encoding='utf-8'))
    blockers = ['native model adapter not implemented', 'asynchronous native trainer not implemented',
                'actual-checkpoint qualification not executed', 'native budget/optimizer approval absent',
                'fresh corpus/evaluator and text-overlap qualification absent',
                'effect margins and confirmation power plan not frozen']
    return {'experiment': r['experiment'], 'status': 'BLOCKED_NATIVE_AND_TRAINING',
            'source_sha256': digest, 'blockers': blockers, 'model_calls': 0,
            'training_updates': 0, 'sealed_evaluation': False, 'release_permitted': False}


def main() -> int:
    try:
        result = report()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {'status': 'BLOCKED_INTEGRITY', 'error': str(exc), 'model_calls': 0}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
