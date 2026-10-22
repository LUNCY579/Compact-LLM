"""Standard-library-only integrity and exclusive external-output helpers."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
RECORDS_ROOT = REPO_ROOT.parent / 'rineya-quality-records' / 'update-order-v1'

class Blocked(RuntimeError):
    pass

def require(condition, reason):
    if not condition:
        raise Blocked(reason)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def now():
    return datetime.now(timezone.utc).isoformat()

def external_path(path):
    result = Path(path).resolve()
    require(result.is_relative_to(RECORDS_ROOT.resolve()) and result != RECORDS_ROOT.resolve(),
            'new outputs must be below BusinessProject/rineya-quality-records/update-order-v1/<run-id>')
    require(not result.is_relative_to(REPO_ROOT.resolve()) and '.codex' not in result.parts,
            'new outputs cannot be in tracked source or .codex')
    return result

def write_json(path, value):
    path = external_path(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

def code_inventory():
    paths = list((PACKAGE_ROOT / 'sbhm').glob('*.py'))
    paths += list((PACKAGE_ROOT / 'quality_followup').rglob('*.py'))
    paths += [PACKAGE_ROOT / 'scripts' / 'build_robust_corpus.py']
    return {str(p.relative_to(PACKAGE_ROOT)).replace('\\', '/'): sha(p) for p in sorted(paths)}
