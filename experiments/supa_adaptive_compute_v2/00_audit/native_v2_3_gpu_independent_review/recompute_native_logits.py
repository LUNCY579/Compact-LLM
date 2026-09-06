"""Independent one-thread saved-logit audit; no torch/model/GPU/checkpoint use.

Replicates the earlier runtime audit's restricted tensor parser and NumPy FP64
metric method. New receipts are exclusive. Existing evidence is read-only.
"""
from __future__ import annotations
import os
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
for name in THREAD_KEYS:
    os.environ[name] = '1'
import collections
import datetime
import hashlib
import io
import json
import pathlib
import pickle
import platform
import sys
import zipfile
import numpy as np
from threadpoolctl import threadpool_info

HERE = pathlib.Path(__file__).resolve().parent
PROGRAMME = HERE.parents[1]
REPO = PROGRAMME.parents[1]
ROOT = PROGRAMME / 'native_armr/evidence/00_runtime_qualification'
EVIDENCE = ROOT / '20260905T235200_982255Z_36998242'
PREVIOUS = ROOT / '20260905T233921_233855Z_41832612'
METHOD = PROGRAMME / '00_audit/runtime_independent_review_20260905/recompute_native_logits.py'
CONFIG = PROGRAMME / 'native_armr/configs/protocol_gpu_v2_3.json'
OUTPUT = HERE / 'native_logit_recomputation.json'
# Unchanged audit-roundoff tolerances from the earlier independent audit.
# Scientific gates below are applied exactly, without this roundoff allowance.
TOLERANCES = {'max_abs': {'atol': 0.0, 'rtol': 0.0},
              'relative_l2': {'atol': 1e-18, 'rtol': 1e-12},
              'cosine': {'atol': 5e-14, 'rtol': 0.0}}
FROZEN_GATES = {'max_abs_logit_error': 0.002,
                'max_relative_logit_l2': 0.0002,
                'minimum_logit_cosine': 0.999999}
FLOAT_STORAGE = object()
INPUT_HASHES = {}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bytes(path):
    path = pathlib.Path(path).resolve()
    require(path.is_relative_to(PROGRAMME), 'input outside programme')
    payload = path.read_bytes()
    digest = sha(payload)
    require(INPUT_HASHES.get(str(path), digest) == digest, 'input changed during audit')
    INPUT_HASHES[str(path)] = digest
    return payload


def read_json(path):
    def constants(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(read_bytes(path), parse_constant=constants, object_pairs_hook=pairs)


def tensor_metadata(storage, offset, shape, stride, requires_grad, hooks):
    require(requires_grad is False and not hooks, 'unexpected tensor grad/hook metadata')
    return {'storage': storage, 'offset': offset, 'shape': shape, 'stride': stride}


class TensorMetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('torch._utils', '_rebuild_tensor_v2'): tensor_metadata,
                   ('torch', 'FloatStorage'): FLOAT_STORAGE,
                   ('collections', 'OrderedDict'): collections.OrderedDict}
        require((module, name) in allowed, 'unexpected pickle global')
        return allowed[module, name]

    def persistent_load(self, value):
        require(isinstance(value, tuple) and len(value) == 5
                and value[0] == 'storage' and value[1] is FLOAT_STORAGE
                and value[2] == '0' and value[3] == 'cpu'
                and type(value[4]) is int and value[4] > 0,
                'unexpected storage metadata')
        return {'key': value[2], 'count': value[4]}


def read_logits(relative, expected_hash):
    path = (EVIDENCE / relative).resolve()
    require(path.is_relative_to(EVIDENCE), 'artifact outside qualification directory')
    payload = read_bytes(path)
    require(sha(payload) == expected_hash, 'artifact hash mismatch: ' + relative)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        expected = {'archive/data.pkl', 'archive/.format_version',
                    'archive/.storage_alignment', 'archive/byteorder',
                    'archive/data/0', 'archive/version', 'archive/.data/serialization_id'}
        require(len(names) == len(set(names)) and set(names) == expected
                and archive.read('archive/byteorder') == b'little',
                'unexpected serialized tensor format')
        meta = TensorMetadataReader(io.BytesIO(archive.read('archive/data.pkl'))).load()
        require(isinstance(meta, dict) and meta['offset'] == 0
                and meta['shape'] == (151936,) and meta['stride'] == (1,)
                and meta['storage'] == {'key': '0', 'count': 151936},
                'unexpected first-token logit vector dimensions')
        storage = archive.read('archive/data/0')
        require(len(storage) == 151936 * 4, 'tensor storage byte count mismatch')
        values = np.frombuffer(storage, dtype='<f4').astype(np.float64)
        require(bool(np.isfinite(values).all()), 'nonfinite logit vector')
    return values, payload, {'path': relative, 'sha256': sha(payload),
                            'hash_matches_report': True, 'bytes': len(payload),
                            'shape': [151936], 'stored_dtype': 'float32'}


def write_exclusive(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def main():
    require(not OUTPUT.exists(), 'completed audit output already exists')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pools = threadpool_info()
    require(bool(pools) and all(pool['num_threads'] == 1 for pool in pools),
            'NumPy thread pools must be verified at one thread')
    report = read_json(EVIDENCE / 'report.json')
    manifest = read_json(EVIDENCE / 'run_manifest.json')
    read_json(EVIDENCE / 'started_manifest.json')
    require(manifest['status'] == 'COMPLETE' and manifest['qualification_pass'] is True
            and report['qualification_pass'] is True, 'current qualification is not COMPLETE/PASS')
    require(manifest['report_sha256'] == INPUT_HASHES[str(EVIDENCE / 'report.json')],
            'report-to-manifest hash mismatch')
    config = read_json(CONFIG)
    registration_path = PROGRAMME / 'native_armr/evidence/00_preregistration' / manifest['registration_sha256'] / 'registration.json'
    registration = read_json(registration_path)
    contract = {key: registration[key] for key in
                ('track', 'config_sha256', 'config_path', 'config', 'sources', 'dataset_manifests')}
    require(registration['status'] == 'FROZEN' and registration['track'] == 'native_armr'
            and canonical(contract) == registration['registration_sha256'] == manifest['registration_sha256'],
            'frozen registration contract binding mismatch')
    require(registration['config'] == config
            and registration['config_path'] == CONFIG.relative_to(REPO).as_posix()
            and registration['config_sha256'] == INPUT_HASHES[str(CONFIG)]
            and canonical(config) == manifest['config_sha256']
            and config['runtime_qualification'] == manifest['runtime_qualification'] == 'native_gpu_fp32_v2_3'
            and registration['sources'] == manifest['sources'], 'qualification/config/source binding mismatch')
    require(report['thresholds'] == config['qualification']
            and all(config['qualification'][key] == value for key, value in FROZEN_GATES.items()),
            'frozen numerical qualification gates differ')
    read_bytes(METHOD)
    previous = read_json(PREVIOUS / 'run_manifest.json')
    read_json(PREVIOUS / 'started_manifest.json')
    require(previous['status'] == 'FAILED' and previous['qualification_pass'] is False,
            'prior attempt must remain explicitly FAILED')
    require(previous['registration_sha256'] == manifest['registration_sha256']
            and previous['config_sha256'] == manifest['config_sha256'],
            'previous CPU artifacts belong to another protocol')
    driver_events = PROGRAMME / '00_audit/gpu_driver_events_20260905T2347.json'
    read_json(driver_events)
    fixture_ids = report['fixture_ids']
    arms = set(config['arms'])
    require(arms == {'A0', 'A1', 'A3', 'A16', 'AM'} and len(fixture_ids) == len(set(fixture_ids)) == 8,
            'expected five arms and eight calibration fixtures')
    expected_cells = {(arm, example) for arm in arms for example in fixture_ids}
    cells = [(row['arm'], row['example_id']) for row in report['comparisons']]
    require(len(cells) == len(set(cells)) == 40 and set(cells) == expected_cells,
            'incomplete or duplicated qualification cells')
    comparisons, cpu_recovery, artifacts_seen = [], [], set()
    for row in report['comparisons']:
        key = row['arm'] + '_' + canonical(row['example_id'])
        cpu_name, gpu_name = 'logits/cpu_' + key + '.pt', 'logits/cuda_0_' + key + '.pt'
        reference, cpu_payload, cpu_receipt = read_logits(cpu_name, report['artifacts'][cpu_name])
        actual, _, gpu_receipt = read_logits(gpu_name, report['artifacts'][gpu_name])
        difference = actual - reference
        reference_norm, actual_norm = float(np.linalg.norm(reference)), float(np.linalg.norm(actual))
        require(min(reference_norm, actual_norm) > 1e-12, 'unexpected near-zero logit norm')
        observed = {'max_abs': float(np.max(np.abs(difference))),
                    'relative_l2': float(np.linalg.norm(difference) / reference_norm),
                    'cosine': float(np.dot(reference, actual) / (reference_norm * actual_norm))}
        recorded = row['logits']
        require(set(recorded) == set(TOLERANCES), 'unexpected recorded metric schema')
        deltas = {name: abs(observed[name] - recorded[name]) for name in TOLERANCES}
        agreement = {name: deltas[name] <= rule['atol'] + rule['rtol'] * abs(recorded[name])
                     for name, rule in TOLERANCES.items()}
        gates = {'absolute': observed['max_abs'] <= FROZEN_GATES['max_abs_logit_error'],
                 'relative': observed['relative_l2'] <= FROZEN_GATES['max_relative_logit_l2'],
                 'cosine': observed['cosine'] >= FROZEN_GATES['minimum_logit_cosine']}
        comparisons.append({'arm': row['arm'], 'example_id': row['example_id'],
                            'artifacts': [cpu_receipt, gpu_receipt], 'recorded': recorded,
                            'recomputed_numpy_float64': observed, 'absolute_metric_differences': deltas,
                            'metric_matches_recorded_within_tolerance': agreement,
                            'registered_logit_gates': gates, 'pass': all(agreement.values()) and all(gates.values())})
        prior_path = PREVIOUS / cpu_name
        recovery = {'arm': row['arm'], 'example_id': row['example_id'], 'path': cpu_name,
                    'current_sha256': cpu_receipt['sha256'], 'prior_present': prior_path.is_file()}
        if recovery['prior_present']:
            prior_payload = read_bytes(prior_path)
            recovery.update(prior_sha256=sha(prior_payload), exact_file_bytes_identical=prior_payload == cpu_payload)
            del prior_payload
        cpu_recovery.append(recovery)
        artifacts_seen.update((cpu_name, gpu_name))
        del reference, actual, difference, cpu_payload
    require(len(artifacts_seen) == 80 and artifacts_seen == set(report['artifacts']), 'incomplete saved artifact coverage')
    require({path.relative_to(EVIDENCE).as_posix() for path in (EVIDENCE / 'logits').glob('*.pt')} == artifacts_seen,
            'qualification tensor directory differs from the report inventory')
    for path, expected in INPUT_HASHES.items():
        require(sha(pathlib.Path(path).read_bytes()) == expected, 'input changed during audit: ' + path)
    require('torch' not in sys.modules and not any(name.startswith(('shared.', 'armr.')) for name in sys.modules),
            'forbidden model/programme import')
    present = sum(row['prior_present'] for row in cpu_recovery)
    identical = sum(row.get('exact_file_bytes_identical', False) for row in cpu_recovery)
    all_pass = all(row['pass'] for row in comparisons)
    result = {'schema': 'native_v2_3_first_token_logit_independent_recomputation_v1',
              'status': 'PASS' if all_pass else 'FAIL', 'start_utc': started,
              'end_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'script_sha256': sha(pathlib.Path(__file__).read_bytes()),
              'methodology_source_sha256': INPUT_HASHES[str(METHOD)],
              'evidence_directory': str(EVIDENCE), 'registration_sha256': manifest['registration_sha256'],
              'config_canonical_sha256': manifest['config_sha256'],
              'comparison_count': len(comparisons), 'artifacts_hashed_and_decoded': len(artifacts_seen),
              'all_recorded_metric_triples_reproduced': all(all(row['metric_matches_recorded_within_tolerance'].values()) for row in comparisons),
              'all_recomputed_logit_gates_pass': all(all(row['registered_logit_gates'].values()) for row in comparisons),
              'maximum_absolute_metric_disagreement': {name: max(row['absolute_metric_differences'][name] for row in comparisons) for name in TOLERANCES},
              'maximum_recomputed_absolute_logit_error': max(row['recomputed_numpy_float64']['max_abs'] for row in comparisons),
              'maximum_recomputed_relative_logit_l2': max(row['recomputed_numpy_float64']['relative_l2'] for row in comparisons),
              'minimum_recomputed_logit_cosine': min(row['recomputed_numpy_float64']['cosine'] for row in comparisons),
              'frozen_logit_gates': FROZEN_GATES, 'audit_roundoff_tolerances_fixed_before_execution': TOLERANCES,
              'scientific_gates_use_roundoff_allowance': False,
              'python': platform.python_version(), 'numpy': np.__version__, 'cpu_thread_limit': 1,
              'thread_environment': {key: os.environ[key] for key in THREAD_KEYS}, 'threadpools': pools,
              'threadpool_one_thread_verified': True, 'models_loaded': 0, 'torch_imported': False,
              'gpu_execution': False, 'checkpoint_access': False, 'generation_performed': False,
              'programme_comparison_helpers_imported': False, 'git_commands_run': False,
              'previous_attempt': {'directory': str(PREVIOUS), 'status': previous['status'],
                                   'qualification_pass': previous['qualification_pass'],
                                   'failure_type': previous['failure_receipt']['type'],
                                   'failure_message': previous['failure_receipt']['message'],
                                   'driver_event_receipt': str(driver_events),
                                   'cpu_artifacts_present': present, 'cpu_file_bytes_identical': identical,
                                   'all_present_cpu_artifacts_byte_identical': present == identical,
                                   'partial_artifact_equality_does_not_qualify_failed_attempt': True,
                                   'comparisons': cpu_recovery},
              'all_inputs_unchanged_at_completion': True, 'input_hashes': INPUT_HASHES,
              'scope': 'Independent saved-logit numeric recomputation and prior CPU artifact byte comparison only.',
              'limitations': ['No model rerun; hidden states, NLL, greedy/repeat identity, gradients and checkpoint integrity are not independently recomputed here.',
                              'Config/registration/report bytes are verified; Git commitment is not checked by this audit.',
                              'Prior TDR attempt remains FAILED; its CPU artifact equality is recovery evidence only.'],
              'comparisons': comparisons}
    write_exclusive(OUTPUT, result)
    print(json.dumps({key: result[key] for key in ('status', 'comparison_count', 'artifacts_hashed_and_decoded',
                                                  'maximum_absolute_metric_disagreement', 'maximum_recomputed_absolute_logit_error',
                                                  'all_recorded_metric_triples_reproduced', 'all_recomputed_logit_gates_pass')}, indent=2))
    print(json.dumps({'prior_cpu_artifacts_present': present, 'prior_cpu_file_bytes_identical': identical,
                      'prior_attempt_status': previous['status'], 'output': str(OUTPUT)}))
    if not all_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        failure = HERE / 'audit_failure.json'
        if not failure.exists():
            write_exclusive(failure, {'status': 'FAILED', 'type': type(error).__name__, 'message': str(error),
                                      'input_hashes': INPUT_HASHES, 'models_loaded': 0, 'gpu_execution': False})
        raise
