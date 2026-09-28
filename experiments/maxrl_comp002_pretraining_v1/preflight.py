"""Read-only Q0 source/data binding. Never loads a model or authorizes training."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

BASE = 'a97df24c2c7dd062985942ead8c3375bf20c66b9'
MANIFEST = '86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44'
HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=unique, parse_constant=invalid)


def local_file(root: Path, relative: str) -> Path:
    require(type(relative) is str and relative and '\\' not in relative,
            'invalid relative path')
    fragment = Path(relative)
    require(not fragment.is_absolute() and '..' not in fragment.parts,
            'path escapes root')
    result = (root / fragment).resolve()
    require(result.is_relative_to(root.resolve()) and result.is_file(),
            'missing/escaped file: ' + relative)
    return result


def observed_environment():
    packages = {}
    for name in ('torch', 'transformers', 'tokenizers', 'safetensors', 'pytest'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'role': 'inspection host, NOT selected experiment runtime'}


def inspect(repo: Path, data: Path):
    """Verify immutable files and atomic scopes without opening any response pool."""
    lock = read_json(HERE / 'source_lock.json')
    require(lock['base_commit'] == BASE and lock['manifest_sha256'] == MANIFEST,
            'qualification source lock drift')
    require(sha(data / 'manifest.json') == MANIFEST, 'wrong full corpus manifest')
    manifest = read_json(data / 'manifest.json')
    for package, inventory in lock['source_inventories'].items():
        root = repo / 'experiments' / package
        observed = {p.relative_to(root).as_posix(): sha(p)
                    for p in sorted(root.rglob('*.py'))}
        require(observed == inventory, 'source inventory drift: ' + package)
    for relative, expected in lock['document_sha256'].items():
        require(sha(local_file(repo, relative)) == expected, 'document drift: ' + relative)
    for relative, expected in manifest['files'].items():
        p = local_file(data, relative)
        require(p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'],
                'data hash/size mismatch: ' + relative)
    rows = [json.loads(line) for line in (data / 'canonical/train.jsonl').open()]
    banks = {}
    for skill in ('R', 'S'):
        relative = 'manifests/atomic_' + skill + '.json'
        plan = read_json(data / relative)
        selected = [r for r in rows if r['role'] == 'train' and r['tag'] == 'core'
                    and r['skills'] == [skill]]
        require(len(plan['ids']) == len(set(plan['ids'])) == len(selected) == 3000,
                'atomic count/duplicate mismatch')
        require(set(plan['ids']) == {r['id'] for r in selected}, 'atomic bank contamination')
        banks[skill] = {'manifest_sha256': sha(data / relative), 'rows': len(selected),
                       'ids_sha256': digest(sorted(plan['ids'])),
                       'parents': len({r['parent_id'] for r in selected})}
    calibration = [json.loads(line) for line in (data / 'canonical/calibration.jsonl').open()]
    calibration_ids = {skill: sorted(r['id'] for r in calibration
                       if r['tag'] == 'core' and r['skills'] == [skill]) for skill in ('R', 'S')}
    require(all(len(ids) == 150 for ids in calibration_ids.values()), 'calibration scope drift')
    require(not any(r['skills'] == ['R', 'S'] for r in calibration), 'RS calibration leakage')
    guard = [json.loads(line) for line in (data / 'canonical/guard_dev.jsonl').open()]
    require(len(guard) == 1200 and len({r['id'] for r in guard}) == 1200,
            'guard_dev bank count')
    return {
        'experiment': 'MAXRL-COMP-002', 'boundary': 'Q0_SOURCE_DATA_BINDING_ONLY',
        'status': 'SOURCE_DATA_PASS_NATIVE_BLOCKED', 'base_commit': BASE,
        'manifest_sha256': MANIFEST, 'verified_data_files': len(manifest['files']),
        'source_lock_sha256': sha(HERE / 'source_lock.json'),
        'qualification_candidate': ['R', 'S'],
        'pair_status': 'study-guide candidate; not a claim of prior pair registration',
        'atomic_banks': banks,
        'calibration': {k: {'rows': len(v), 'ids_sha256': digest(v)}
                        for k, v in calibration_ids.items()},
        'guard_dev': {'rows': 1200, 'bank_sha256': sha(data / 'canonical/guard_dev.jsonl'),
                      'source_correct_enrollment': 'NOT_RUN'},
        'observed_environment': observed_environment(),
        'native_model_loads': 0, 'native_tokenizer_loads': 0, 'native_forward_calls': 0,
        'native_backward_calls': 0, 'native_generations': 0, 'native_training_runs': 0,
        'optimizer_steps': 0, 'training_authorized': False, 'native_qualified': False,
        'blocked': ['exact selected model/assets', 'exact tokenizer/template/token IDs',
                    'selected runtime/loader and bounded host permit',
                    'actual trainable/tied-parameter inventory',
                    'native completion/EOS/logprob/gradient receipts',
                    'actual source competence and source-correct enrollment'],
        'interpretation': 'No acquisition, retention, composition, release or orthogonality claim.'}


def write_receipt(result, out: Path, repo: Path, data: Path):
    target = out.resolve()
    forbidden = [data.resolve(),
                 (repo / 'experiments/maxrl_composable_skills_v2').resolve(),
                 (repo / 'experiments/maxrl_verifiable_data_v1').resolve()]
    require(not any(target.is_relative_to(p) for p in forbidden), 'protected output path')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = inspect(args.repo, args.data)
        write_receipt(result, args.out, args.repo, args.data)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, 'BLOCKED: ' + str(error) + '\n')
    print(result['status'])
    # A successful static inspection is deliberately NOT a launch-ready exit 0.
    raise SystemExit(2)


if __name__ == '__main__':
    main()
