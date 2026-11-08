"""Complete AM mechanics review; reuse snapshot JSON/hash helpers, no tensors."""
import ast
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import unicodedata
from typing import Any

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / 'native_v2_4_resume_review_20260913/audit_snapshot.py'
spec = importlib.util.spec_from_file_location('native_snapshot_helpers', HELPER)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.HERE = HERE
REPO, PROGRAMME = audit.REPO, audit.PROGRAMME
read, sha, canonical, check, relative = audit.read, audit.sha, audit.canonical, audit.check, audit.relative
REG = audit.REGISTRATION


def hashed(path, expected=None):
    path = Path(path)
    actual = sha(path)
    audit.INPUTS[relative(path)] = actual
    if expected is not None:
        check(actual == expected, 'artifact bytes: ' + relative(path))
    return actual


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def run():
    cutoff = datetime.now(timezone.utc)
    reg = read(PROGRAMME / 'native_armr/evidence/00_preregistration' / REG / 'registration.json')
    cfg = reg['config']
    check(reg['registration_sha256'] == REG and reg['status'] == 'FROZEN', 'frozen registration identity')
    check(read(PROGRAMME / 'native_armr/configs/protocol_gpu_v2_4.json') == cfg, 'registered config unchanged')
    for name, expected in reg['sources'].items():
        hashed(REPO / name, expected)
    hashed(HELPER)
    manifest_path = PROGRAMME / 'data/generated/v2/native_armr/manifest.json'
    bank = read(manifest_path)
    data_path = manifest_path.parent / bank['datasets']['architecture_dev']['path']
    hashed(data_path, bank['datasets']['architecture_dev']['sha256'])
    cases = [json.loads(line) for line in data_path.read_text(encoding='utf-8').splitlines() if line]
    expected = {c['example_id']: c for c in cases}
    check(len(cases) == len(expected) == len({c['scenario_id'] for c in cases}) == 128, '128 independent architecture originals')
    check(all(c['track'] == 'native_armr' and c['split'] == 'architecture_dev' and c['variant'] == 'original' for c in cases), 'Native architecture role only')
    shards = {i: [c['example_id'] for c in cases if int(canonical(c['scenario_id']), 16) % cfg['default_shard_count'] == i]
              for i in range(cfg['default_shard_count'])}

    calibration_path = REPO / cfg['control_calibration']['artifact_path']
    cal = read(calibration_path)
    hashed(calibration_path, cfg['control_calibration']['artifact_sha256'])
    prior = read(HERE.parent / 'native_v2_4_calibration_review/receipt.json')
    check(prior['status'] == 'PASS' and prior['checks_passed'] == 1996, 'existing independent calibration audit passed')
    for name, expected_hash in prior['input_sha256'].items():
        hashed(REPO / name, expected_hash)
    check(cal['status'] == 'COMPLETE' and cal['canonical_unchanged'], 'successful frozen calibration')
    check(cal['alpha'] == cfg['control']['alpha'] == math.sqrt(cal['fit']['native_energy'] / cal['fit']['control_energy']), 'fit-only alpha reconstructed')
    check(cal['fit']['scenarios'] == 96 and cal['holdout']['scenarios'] == 32 and cal['answer_labels_used'] is False, 'calibration split and label boundary')
    check(cal['source_hashes'] == cal['copy_hashes'], 'calibration source and copy branch equality')
    check(cal['control_state_receipt']['state_sha256_before'] == cal['control_state_receipt']['state_sha256_after'] == cal['control_state_sha256'], 'calibration control state frozen')
    inventory = {'added_parameter_count': 28314624, 'added_active_parameters': 28314624,
                 'extra_matmul_flops_per_token': 56623104}
    check(cfg['control']['additional_parameters'] == inventory['added_parameter_count'] and cfg['control']['additional_active_parameters'] == inventory['added_active_parameters'] and cfg['control']['additional_matmul_flops_per_token'] == inventory['extra_matmul_flops_per_token'], 'registered complete-FFN inventory')
    check(inventory['extra_matmul_flops_per_token'] / 62914560 == .9, 'AM is 90 percent of A3 dominant extra matrix work')

    # Reuse only the registered pure-Python validity function. Do not import
    # armr.metrics, whose module-level torch import would violate this review.
    flags_path = REPO / 'experiments/anchor_routed_middle_recurrence_v1/src/armr/metrics.py'
    tree = ast.parse(flags_path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'output_flags')
    namespace = {'unicodedata': unicodedata, 'Any': Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(flags_path), 'exec'), namespace)
    output_flags = namespace['output_flags']

    headers, failures, unfinished = {'A0': [], 'AM': []}, [], []
    for directory in sorted((PROGRAMME / 'native_armr/evidence/01_architecture_dev').iterdir()):
        if not directory.is_dir():
            continue
        final = directory / 'run_manifest.json'
        path = final if final.is_file() and final.stat().st_mtime <= cutoff.timestamp() else directory / 'started_manifest.json'
        if not path.is_file() or path.stat().st_mtime > cutoff.timestamp():
            continue
        header = json.loads(path.read_text(encoding='utf-8'))
        if header.get('registration_sha256') != REG or header.get('arm') not in headers:
            continue
        m = read(path)
        if path != final:
            unfinished.append(relative(directory))
        elif m['status'] != 'COMPLETE':
            raw_count = len(list((directory / 'cases').glob('*.json')))
            f = m.get('failure_receipt') or {}
            admission = m.get('completed_examples') == 0 and raw_count == 0 and not (directory / 'results.json').exists() and not (directory / 'model_identity.json').exists() and f.get('type') == 'MemoryError' and f.get('message', '').startswith('insufficient available physical RAM:')
            failures.append({'directory': relative(directory), 'arm': m['arm'], 'failure': f,
                             'raw_case_count': raw_count, 'verified_zero_example_RAM_admission': admission})
            check(admission, 'all prior A0/AM failures are zero-example RAM admission: ' + directory.name)
        else:
            headers[m['arm']].append((directory, m))
    check(not unfinished, 'no unfinished A0/AM attempts at cutoff')
    seen, rows_by_arm = {'A0': set(), 'AM': set()}, {'A0': [], 'AM': []}
    qualification_dirs = set()
    for arm, runs in headers.items():
        for directory, m in runs:
            label = arm + ':' + directory.name
            check(m['role'] == 'architecture_dev' and m['track'] == 'native_armr' and m['config_sha256'] == canonical(cfg) and m['dataset_manifest_sha256'] == audit.INPUTS[relative(manifest_path)], 'receipt config/data/role binding: ' + label)
            check(m['sources'] == reg['sources'], 'receipt source binding: ' + label)
            check(m['shard_count'] == cfg['default_shard_count'] and m['example_ids'] == shards[m['shard_index']], 'deterministic shard allocation: ' + label)
            check(not seen[arm].intersection(m['example_ids']), 'no duplicate completed examples: ' + label)
            seen[arm].update(m['example_ids'])
            check(m['state_sha256_before'] == m['state_sha256_after'] and m['canonical_unchanged'] is True and m['checkpoint_files_before'] == m['checkpoint_files_after'] and m['checkpoint_files_before']['model.safetensors'] == cfg['checkpoint_sha256'], 'recorded backbone/checkpoint identity: ' + label)
            check(m['trainable_parameters'] == m['backbone_gradients'] == 0, 'zero backbone trainable parameters and gradients: ' + label)
            identity = read(directory / 'model_identity.json')
            keys = ('state_sha256_before', 'checkpoint_files_before', 'model_class', 'parameters', 'qwen_implementation_sha256', 'runtime_identity', 'torch', 'transformers')
            check(all(identity[k] == m[k] for k in keys), 'saved identity matches final receipt: ' + label)
            check(all(m[k] == cfg[k] for k in ('seed', 'threads', 'dtype', 'device', 'max_prompt_seconds')) and m['cache_mode'] == 'full_prefix_no_cache', 'frozen execution configuration: ' + label)
            qdir = Path(m['runtime_qualification_directory'])
            qm = read(qdir / 'run_manifest.json')
            hashed(qdir / 'run_manifest.json', m['runtime_qualification_receipt_sha256'])
            check(qm['status'] == 'COMPLETE' and qm['qualification_pass'] and qm['registration_sha256'] == REG and qm['runtime_identity'] == m['runtime_identity'], 'matching qualified GPU runtime: ' + label)
            if qdir not in qualification_dirs:
                qr = read(qdir / 'report.json')
                hashed(qdir / 'report.json', qm['report_sha256'])
                check(qr['qualification_pass'] and len(qr['comparisons']) == 40 and all(c['pass'] and all(c['gates'].values()) for c in qr['comparisons']), 'all 40 CPU/GPU numeric comparisons passed')
                for name, expected_hash in qr['artifacts'].items():
                    hashed(qdir / name, expected_hash)
                qualification_dirs.add(qdir)
            if arm == 'AM':
                check(m['operator'] == inventory and m['operator_state_sha256'] == cal['control_state_sha256'], 'calibrated copied-control identity and cost: ' + label)
                check(all(m[k] == 0 for k in ('operator_trainable_parameters', 'operator_gradients', 'operator_trainable_parameters_after', 'operator_gradients_after')), 'zero control training and gradients before and after: ' + label)
                baseline = Path(m['baseline_run_directory'])
                bm = read(baseline / 'run_manifest.json')
                hashed(baseline / 'run_manifest.json', m['baseline_manifest_sha256'])
                check(bm['status'] == 'COMPLETE' and bm['arm'] == 'A0' and bm['registration_sha256'] == REG and bm['example_ids'] == m['example_ids'], 'complete matching A0 reference: ' + label)
            values = read(directory / 'results.json')
            hashed(directory / 'results.json', m['results_sha256'])
            check([r['example_id'] for r in values] == m['example_ids'], 'complete ordered result coverage: ' + label)
            check(set(m['logits_sha256']) == {canonical(r['example_id']) + '.pt' for r in values}, 'exact logit artifact inventory: ' + label)
            for row in values:
                tag = arm + ':' + row['example_id']
                raw = read(directory / 'cases' / (canonical(row['example_id']) + '.json'))
                check(raw == row and finite(raw), 'finite raw row equals aggregate: ' + tag)
                hashed(directory / 'logits' / (canonical(row['example_id']) + '.pt'), m['logits_sha256'][canonical(row['example_id']) + '.pt'])
                check(all(row.get(k) == expected[row['example_id']].get(k) for k in ('scenario_id', 'family', 'track', 'split', 'variant', 'template_id')), 'raw case identity: ' + tag)
                flags = output_flags(row['response'], row['token_ids'])
                check(all(row[k] == v for k, v in flags.items()), 'raw validity flags reconstructed: ' + tag)
                check(not flags['invalid'] and row['safety_stopped'] is False, 'valid output and no safety stop: ' + tag)
                check(1 <= len(row['token_ids']) <= min(cfg['max_new_tokens'], expected[row['example_id']].get('max_new_tokens', cfg['max_new_tokens'])) and row['truncated'] == (row['termination'] == 'token_cap') and row['eos_emitted'] == (row['termination'] == 'eos'), 'generation cap and termination metadata: ' + tag)
                lower, upper = cfg['hidden_rms_ratio_bounds']
                check(lower <= row['trajectory_min_rms_ratio'] <= row['trajectory_max_rms_ratio'] <= upper and all(lower <= h['rms_ratio_to_normal'] <= upper for h in row['hidden']), 'full recorded RMS trajectory within frozen bounds: ' + tag)
                check(0 < row['wall_seconds'] <= cfg['max_prompt_seconds'] and row['peak_rss_bytes'] <= cfg['max_rss_gib'] * 2**30 and row['peak_cuda_allocated_bytes'] <= cfg['max_cuda_memory_gib'] * 2**30, 'per-case deadline and memory bounds: ' + tag)
                extra = inventory['extra_matmul_flops_per_token'] if arm == 'AM' else 0
                work = row['processed_token_positions'] * (28 * 31457280 + extra) + len(row['token_ids']) * 2 * 1024 * 151936
                check(row['generation_dense_matmul_flops_estimate'] == work, 'dominant matrix FLOPs independently reconstructed: ' + tag)
                rows_by_arm[arm].append(row)
    for arm in headers:
        check(seen[arm] == set(expected), 'full 128-case coverage: ' + arm)
        check(len(headers[arm]) == sum(bool(ids) for ids in shards.values()), 'one complete attempt per nonempty shard: ' + arm)
    check(max(datetime.fromisoformat(m['end_utc']) for _, m in headers['A0']) < min(datetime.fromisoformat(m['start_utc']) for _, m in headers['AM']), 'all A0 examples completed before any completed AM run began')
    identities = {canonical({k: m[k] for k in ('state_sha256_before', 'checkpoint_files_before', 'parameters', 'model_class', 'qwen_implementation_sha256', 'torch', 'transformers')}) for runs in headers.values() for _, m in runs}
    check(len(identities) == 1, 'identical canonical model across A0 and AM')
    check('torch' not in sys.modules, 'no torch import or tensor deserialization')
    issues = [item for item in audit.CHECKS if not item['pass']]
    summary = {arm: {'examples': len(rows), 'completed_nonempty_shards': len(headers[arm]),
        'invalid_outputs': sum(bool(r['invalid']) for r in rows), 'safety_stops': sum(bool(r['safety_stopped']) for r in rows),
        'trajectory_min_rms_ratio': min(r['trajectory_min_rms_ratio'] for r in rows),
        'trajectory_max_rms_ratio': max(r['trajectory_max_rms_ratio'] for r in rows),
        'mean_case_wall_seconds': statistics.mean(r['wall_seconds'] for r in rows),
        'maximum_run_peak_RSS_bytes': max(m['peak_memory_bytes'] for _, m in headers[arm]),
        'maximum_run_peak_CUDA_allocated_bytes': max(m['peak_cuda_allocated_bytes'] for _, m in headers[arm])}
        for arm, rows in rows_by_arm.items()}
    audit.write('input_hashes.json', audit.INPUTS)
    audit.write('checks.json', audit.CHECKS)
    report = {'schema_version': 'independent_native_v2_4_complete_control_mechanics_v1', 'status': 'PASS' if not issues else 'ISSUES_FOUND',
        'cutoff_utc': cutoff.isoformat(), 'registration_sha256': REG, 'checks_passed': len(audit.CHECKS) - len(issues),
        'checks_failed': issues, 'summary': summary, 'control_alpha': cal['alpha'], 'control_state_sha256': cal['control_state_sha256'],
        'operator_inventory': inventory, 'failed_attempts': failures, 'unfinished_at_cutoff': unfinished,
        'completed_runs': {arm: [relative(d) for d, _ in runs] for arm, runs in headers.items()},
        'input_hashes_sha256': sha(HERE / 'input_hashes.json'), 'checks_sha256': sha(HERE / 'checks.json'),
        'script_sha256': sha(Path(__file__)), 'reused_helper_sha256': sha(HELPER), 'models_loaded': 0,
        'torch_imported': False, 'tensor_deserialization': False, 'efficacy_statistics_computed': False,
        'confirmation_licensed_by_this_review': False,
        'limitations': ['Receipt and artifact-byte reconciliation, not inference or numerical tensor replay.',
            'The recorded control state is bound to the separately audited calibration; real model state is not reloaded.',
            'Validity flags reuse the exact frozen output_flags function extracted by AST, with no module or torch import.',
            'Raw response-to-token decoding cannot be independently reconstructed without loading the tokenizer.',
            'Mean latency is observational and affected by machine contention; no causal cost comparison is claimed.',
            'This review covers A0 and AM only. Full five-arm architecture audit remains mandatory before confirmation.',
            'AM has 90 percent of A3 additional dominant matrix FLOPs; it is not an exact compute match.']}
    audit.write('receipt.json', report)
    text = '\n'.join(['# Complete Native v2.4 control mechanics audit', '',
        f"Status: **{report['status']}**. {report['checks_passed']} checks passed; {len(issues)} failed. Cutoff: {cutoff.isoformat()}.", '',
        f"A0 covers {summary['A0']['examples']}/128 and AM covers {summary['AM']['examples']}/128 frozen originals. "
        'No efficacy statistics are calculated. Confirmation requires the complete five-arm matrix.', '',
        f"AM alpha: {cal['alpha']}; copied state SHA256: `{cal['control_state_sha256']}`.", '',
        f"AM recorded RMS range: [{summary['AM']['trajectory_min_rms_ratio']}, {summary['AM']['trajectory_max_rms_ratio']}], "
        f"inside [0.25, 4.0]. Invalid outputs: {summary['AM']['invalid_outputs']}; safety stops: {summary['AM']['safety_stops']}.", '',
        'The audit reconciles raw JSON, source/config/data/calibration/GPU bindings, zero training inventories, baseline completeness, '
        'saved logits as bytes, and reconstructed dominant FLOPs. No models or tensor libraries were loaded.', '',
        'Limitations:', '', *['- ' + s for s in report['limitations']]])
    with (HERE / 'audit.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'checks_passed', 'checks_failed', 'summary')}, indent=2))
    return 0 if not issues else 1


if __name__ == '__main__':
    raise SystemExit(run())
