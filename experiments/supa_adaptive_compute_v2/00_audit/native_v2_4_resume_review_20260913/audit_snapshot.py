"""Independent standard-library receipt/count review; never imports programme code."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROGRAMME = HERE.parents[1]
REPO = PROGRAMME.parents[1]
REGISTRATION = '081a8f6bf3c7396c97f13e3d4a7d55bf7311b37dcc0b0bc7bfd32478fa44241e'
ARMS = ('A0', 'AM', 'A1', 'A3', 'A16')
INPUTS = {}
CHECKS = []


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def relative(path):
    return path.resolve().relative_to(REPO).as_posix()


def check(value, label):
    CHECKS.append({'check': label, 'pass': bool(value)})


def read(path):
    data = path.read_bytes()
    INPUTS[relative(path)] = hashlib.sha256(data).hexdigest()
    def pairs(items):
        values = {}
        for key, value in items:
            if key in values:
                raise ValueError('duplicate JSON key ' + key)
            values[key] = value
        return values
    return json.loads(data, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write(name, value):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def run():
    cutoff = datetime.now(timezone.utc)
    cutoff_stamp = cutoff.timestamp()
    registration_path = PROGRAMME / 'native_armr/evidence/00_preregistration' / REGISTRATION / 'registration.json'
    registration = read(registration_path)
    config = registration['config']
    check(registration['registration_sha256'] == REGISTRATION and registration['status'] == 'FROZEN', 'registration identity frozen')
    current_config = read(PROGRAMME / 'native_armr/configs/protocol_gpu_v2_4.json')
    check(current_config == config, 'current v2.4 config equals registered config')
    for name, expected in registration['sources'].items():
        path = REPO / name
        actual = sha(path)
        INPUTS[name] = actual
        check(actual == expected, 'source bytes: ' + name)
    manifest_path = PROGRAMME / 'data/generated/v2/native_armr/manifest.json'
    bank = read(manifest_path)
    architecture_path = manifest_path.parent / bank['datasets']['architecture_dev']['path']
    INPUTS[relative(architecture_path)] = sha(architecture_path)
    check(INPUTS[relative(architecture_path)] == bank['datasets']['architecture_dev']['sha256'], 'frozen architecture JSONL bytes')
    cases = [json.loads(line) for line in architecture_path.read_text(encoding='utf-8').splitlines() if line]
    expected = {row['example_id']: row for row in cases}
    check(len(cases) == len(expected) == len({row['scenario_id'] for row in cases}) == 128, '128 independent architecture originals')
    check(all(row['track'] == 'native_armr' and row['split'] == 'architecture_dev' and row['variant'] == 'original' for row in cases), 'Native architecture role only')
    shard_count = config['default_shard_count']
    shards = {i: [row['example_id'] for row in cases if int(canonical(row['scenario_id']), 16) % shard_count == i]
              for i in range(shard_count)}
    empty_shards = [i for i, ids in shards.items() if not ids]
    completed = {arm: [] for arm in ARMS}
    seen = {arm: set() for arm in ARMS}
    invalid = {arm: 0 for arm in ARMS}
    safety = {arm: 0 for arm in ARMS}
    failures = []
    unfinished = []
    identity_hashes = set()
    evidence_root = PROGRAMME / 'native_armr/evidence/01_architecture_dev'
    directories = sorted(path for path in evidence_root.iterdir() if path.is_dir())
    for directory in directories:
        final_path = directory / 'run_manifest.json'
        started_path = directory / 'started_manifest.json'
        is_final = final_path.is_file() and final_path.stat().st_mtime <= cutoff_stamp
        candidate = final_path if is_final else started_path
        if not candidate.is_file() or candidate.stat().st_mtime > cutoff_stamp:
            continue
        # Read unrelated headers only for registration selection; do not claim
        # historical runs, import helpers or consume their scored output rows.
        header = json.loads(candidate.read_text(encoding='utf-8'))
        if header.get('registration_sha256') != REGISTRATION:
            continue
        value = read(candidate)
        arm = value['arm']
        check(arm in ARMS, 'registered arm: ' + directory.name)
        check(value['role'] == 'architecture_dev' and value['track'] == 'native_armr', 'receipt role: ' + directory.name)
        check(value['config_sha256'] == canonical(config), 'config binding: ' + directory.name)
        check(value['dataset_manifest_sha256'] == INPUTS[relative(manifest_path)], 'dataset binding: ' + directory.name)
        check(value['sources'] == registration['sources'], 'source map binding: ' + directory.name)
        check(value['shard_count'] == shard_count and value['example_ids'] == shards[value['shard_index']], 'deterministic shard coverage: ' + directory.name)
        if not is_final:
            unfinished.append({'directory': relative(directory), 'arm': arm, 'shard_index': value['shard_index'],
                'status_at_cutoff': 'NO_FINAL_RECEIPT', 'started_manifest_sha256': INPUTS[relative(candidate)],
                'partial_case_files_at_cutoff': sum(p.stat().st_mtime <= cutoff_stamp for p in (directory / 'cases').glob('*.json')),
                'process_liveness': 'not determined by this filesystem-only review'})
            continue
        if value['status'] == 'FAILED':
            count = sum(p.stat().st_mtime <= cutoff_stamp for p in (directory / 'cases').glob('*.json'))
            results_exists = (directory / 'results.json').exists()
            identity_exists = (directory / 'model_identity.json').exists()
            error = value.get('failure_receipt', {})
            zero = value.get('completed_examples') == 0 and count == 0 and not results_exists
            admission = zero and error.get('type') == 'MemoryError' and error.get('message', '').startswith('insufficient available physical RAM:') and not identity_exists
            failures.append({'directory': relative(directory), 'arm': arm, 'shard_index': value['shard_index'],
                'manifest_sha256': INPUTS[relative(candidate)], 'completed_examples': value.get('completed_examples'),
                'observed_case_json_files': count, 'results_json_exists': results_exists, 'model_identity_json_exists': identity_exists,
                'zero_example_failure_verified': zero, 'preload_ram_admission_failure_verified': admission,
                'failure_type': error.get('type'), 'message': error.get('message'), 'end_utc': value.get('end_utc')})
            continue
        check(value['status'] == 'COMPLETE', 'supported terminal status: ' + directory.name)
        if value['status'] != 'COMPLETE':
            continue
        rows = read(directory / 'results.json')
        check(INPUTS[relative(directory / 'results.json')] == value['results_sha256'], 'results bytes: ' + directory.name)
        ids = [row['example_id'] for row in rows]
        check(ids == value['example_ids'], 'complete ordered result coverage: ' + directory.name)
        check(not seen[arm].intersection(ids) and len(ids) == len(set(ids)), 'no duplicate arm examples: ' + directory.name)
        check(value.get('canonical_unchanged') is True and value.get('state_sha256_before') == value.get('state_sha256_after')
              and bool(value.get('state_sha256_before')), 'recorded frozen model state: ' + directory.name)
        check(value.get('checkpoint_files_before') == value.get('checkpoint_files_after')
              and value['checkpoint_files_before']['model.safetensors'] == config['checkpoint_sha256'], 'recorded canonical checkpoint bytes: ' + directory.name)
        check(value.get('trainable_parameters') == 0 and value.get('backbone_gradients') == 0, 'recorded no backbone training: ' + directory.name)
        check(all(value.get(key) == config[key] for key in ('seed', 'threads', 'dtype', 'device', 'max_prompt_seconds')), 'registered execution runtime: ' + directory.name)
        check(bool(value.get('runtime_qualification_directory')) and bool(value.get('runtime_qualification_receipt_sha256')), 'GPU qualification reference present: ' + directory.name)
        identity_hashes.add(canonical({key: value.get(key) for key in
            ('state_sha256_before', 'checkpoint_files_before', 'model_class', 'qwen_implementation_sha256')}))
        for row in rows:
            origin = expected[row['example_id']]
            check(all(row.get(key) == origin.get(key) for key in ('scenario_id', 'family', 'track', 'split', 'variant', 'template_id')), 'raw row identity: ' + row['example_id'] + ':' + arm)
            raw_path = directory / 'cases' / (canonical(row['example_id']) + '.json')
            raw = read(raw_path)
            check(raw == row, 'raw row equals aggregate: ' + row['example_id'] + ':' + arm)
            invalid[arm] += bool(row.get('invalid'))
            safety[arm] += bool(row.get('safety_stopped'))
        seen[arm].update(ids)
        completed[arm].append({'directory': relative(directory), 'manifest_sha256': INPUTS[relative(candidate)],
            'shard_index': value['shard_index'], 'examples': len(rows), 'results_sha256': value['results_sha256']})
    check(len(identity_hashes) <= 1, 'same canonical model identity across completed paired arms')
    confirmatory_receipts = []
    confirmatory_root = PROGRAMME / 'native_armr/evidence/02_confirmatory_run'
    if confirmatory_root.exists():
        for path in sorted(confirmatory_root.glob('*/run_manifest.json')):
            if path.stat().st_mtime <= cutoff_stamp:
                value = json.loads(path.read_text(encoding='utf-8'))
                if value.get('registration_sha256') == REGISTRATION:
                    read(path)
                    confirmatory_receipts.append({'path': relative(path), 'status': value['status'], 'arm': value['arm'], 'role': value['role']})
        for path in sorted(confirmatory_root.glob('*/started_manifest.json')):
            if path.stat().st_mtime <= cutoff_stamp and not (path.parent / 'run_manifest.json').exists():
                value = json.loads(path.read_text(encoding='utf-8'))
                if value.get('registration_sha256') == REGISTRATION:
                    read(path)
                    confirmatory_receipts.append({'path': relative(path), 'status': 'NO_FINAL_RECEIPT', 'arm': value['arm'], 'role': value['role']})
    summary = {arm: {'completed_examples': len(seen[arm]), 'expected_examples': 128,
        'complete_shards': len(completed[arm]), 'nonempty_expected_shards': shard_count - len(empty_shards),
        'missing_examples': 128 - len(seen[arm]), 'invalid_outputs_in_complete_shards': invalid[arm],
        'safety_stops_in_complete_shards': safety[arm], 'full_coverage': seen[arm] == set(expected)} for arm in ARMS}
    coverage_ready = all(row['full_coverage'] for row in summary.values())
    issues = [row for row in CHECKS if not row['pass']]
    report = {'schema_version': 'independent_native_v2_4_resume_snapshot_v1',
        'status': 'PASS' if not issues else 'ISSUES_FOUND', 'snapshot_cutoff_utc': cutoff.isoformat(),
        'cutoff_rule': 'Only final files whose filesystem mtime is at or before cutoff count; later finals remain unfinished at cutoff.',
        'registration_sha256': REGISTRATION, 'config_sha256': canonical(config),
        'summary': summary, 'empty_shard_indices': empty_shards, 'completed_runs': completed,
        'failed_attempts': failures, 'unfinished_at_cutoff': unfinished,
        'matching_confirmation_receipts': confirmatory_receipts,
        'confirmation_licensed_at_cutoff': False if not coverage_ready or issues else None,
        'confirmation_reason': 'All five arms must cover all 128 frozen cases with matching identity and validity before confirmation; current coverage is incomplete.' if not coverage_ready else 'Coverage is complete; full official gate verification is still required.',
        'checks_passed': len(CHECKS) - len(issues), 'checks_failed': issues,
        'models_loaded': 0, 'torch_imported': 'torch' in sys.modules, 'programme_helpers_imported': False,
        'checkpoint_weight_files_opened': False, 'tensor_deserialization': False, 'git_commands_run': False,
        'limitations': ['This snapshot verifies JSON/hash bindings and counts, not model execution or numerical qualification replay.',
            'Raw responses were reconciled with saved rows but not independently rescored; no efficacy summary or partial-arm accuracy is computed.',
            'Live queue/process status is not inferred from files; the parent owns and polls queue session71035.',
            'Later worker progress is excluded by the stated cutoff and does not invalidate the earlier snapshot.']}
    write('input_hashes.json', INPUTS)
    write('checks.json', CHECKS)
    report['input_hashes_sha256'] = sha(HERE / 'input_hashes.json')
    report['checks_sha256'] = sha(HERE / 'checks.json')
    report['audit_script_sha256'] = sha(Path(__file__))
    write('receipt.json', report)
    lines = ['# Native v2.4 resume snapshot — 2026-09-13', '',
        'Cutoff: ' + cutoff.isoformat() + '.', '',
        'Registration: `' + REGISTRATION + '`.', '',
        '| Arm | Completed cases | Complete nonempty shards | Missing cases |',
        '|---|---:|---:|---:|']
    lines += [f"| {arm} | {summary[arm]['completed_examples']}/128 | {summary[arm]['complete_shards']}/{summary[arm]['nonempty_expected_shards']} | {summary[arm]['missing_examples']} |" for arm in ARMS]
    lines += ['', 'Shards ' + ', '.join(map(str, empty_shards)) + ' are empty under the frozen deterministic allocation; they are not missing work.', '',
        f"Verified {sum(item['preload_ram_admission_failure_verified'] for item in failures)} zero-example RAM admission failures among {len(failures)} failed attempts. "
        'These failures do not constitute operator rejection or completed examples.', '',
        '**Confirmation remains closed.** ' + report['confirmation_reason'], '',
        f"Matching confirmation receipts at cutoff: {len(confirmatory_receipts)}. Checks: {report['checks_passed']} passed, {len(issues)} failed.", '',
        'Scope: standard-library JSON/hash/count audit only. No models, tensors, checkpoint weights, programme helpers or Git commands were used. '
        'No partial-arm accuracy or efficacy claim is made. Parent-owned queue liveness is outside this filesystem snapshot.']
    with (HERE / 'audit.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'cutoff': report['snapshot_cutoff_utc'],
        'summary': summary, 'failures': failures, 'unfinished': unfinished,
        'confirmation_receipts': len(confirmatory_receipts), 'checks_passed': report['checks_passed'],
        'checks_failed': issues, 'audit_directory': str(HERE)}, indent=2))


if __name__ == '__main__':
    run()
