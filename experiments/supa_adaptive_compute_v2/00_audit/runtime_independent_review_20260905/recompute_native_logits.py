"""Independent, CPU-only NumPy audit of saved native first-token logits.
No torch import, model loading, programme helper, or arbitrary pickle globals.
Usage: python recompute_native_logits.py
The output is exclusive; rerunning cannot overwrite the completed receipt.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import collections
import datetime
import hashlib
import io
import json
import math
import pathlib
import pickle
import platform
import sys
import zipfile
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PROGRAMME = HERE.parents[1]
EVIDENCE = PROGRAMME / 'native_armr/evidence/00_runtime_qualification/20260905T191958_868458Z_82f1162e'
OUTPUT = HERE / 'native_logit_recomputation.json'
# Fixed before examining any recomputed metric. Exact maximum-absolute agreement;
# tight tolerance for different FP64 norm/reduction arithmetic and cosine rounding.
TOLERANCES = {
    'max_abs': {'atol': 0.0, 'rtol': 0.0},
    'relative_l2': {'atol': 1e-18, 'rtol': 1e-12},
    'cosine': {'atol': 5e-14, 'rtol': 0.0},
}
FLOAT_STORAGE = object()

def sha(payload):
    return hashlib.sha256(payload).hexdigest()

def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode())

def tensor_metadata(storage, offset, shape, stride, requires_grad, hooks):
    if requires_grad is not False or hooks:
        raise ValueError('unexpected tensor grad or hook metadata')
    return {'storage': storage, 'offset': offset, 'shape': shape, 'stride': stride}

class TensorMetadataReader(pickle.Unpickler):
    """Allow only metadata constructors; never import or execute torch code."""
    def find_class(self, module, name):
        allowed = {
            ('torch._utils', '_rebuild_tensor_v2'): tensor_metadata,
            ('torch', 'FloatStorage'): FLOAT_STORAGE,
            ('collections', 'OrderedDict'): collections.OrderedDict,
        }
        if (module, name) not in allowed:
            raise ValueError(f'unexpected pickle global: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, value):
        if (not isinstance(value, tuple) or len(value) != 5 or
                value[0] != 'storage' or value[1] is not FLOAT_STORAGE or
                value[2] != '0' or value[3] != 'cpu' or
                type(value[4]) is not int or value[4] <= 0):
            raise ValueError('unexpected storage metadata')
        return {'key': value[2], 'count': value[4]}

def read_logits(relative, expected_hash):
    path = EVIDENCE / relative
    if not path.resolve().is_relative_to(EVIDENCE.resolve()):
        raise ValueError('artifact outside evidence directory')
    payload = path.read_bytes()
    observed_hash = sha(payload)
    if observed_hash != expected_hash:
        raise ValueError(f'artifact hash mismatch: {relative}')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive entries')
        expected = {'archive/data.pkl', 'archive/.format_version',
                    'archive/.storage_alignment', 'archive/byteorder',
                    'archive/data/0', 'archive/version',
                    'archive/.data/serialization_id'}
        if set(names) != expected or archive.read('archive/byteorder') != b'little':
            raise ValueError('unexpected serialized tensor format')
        meta = TensorMetadataReader(io.BytesIO(archive.read('archive/data.pkl'))).load()
        if (not isinstance(meta, dict) or meta['offset'] != 0 or
                meta['shape'] != (151936,) or meta['stride'] != (1,) or
                meta['storage'] != {'key': '0', 'count': 151936}):
            raise ValueError('unexpected first-token logit vector dimensions')
        storage = archive.read('archive/data/0')
        if len(storage) != 151936 * 4:
            raise ValueError('tensor storage byte count mismatch')
        values = np.frombuffer(storage, dtype='<f4').astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite logit vector')
    return values, {'path': relative, 'sha256': observed_hash,
                    'hash_matches_report': True, 'bytes': len(payload),
                    'shape': list(values.shape), 'stored_dtype': 'float32'}

def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prior_hashes = {name: sha((HERE / name).read_bytes())
                    for name in ('audit.md', 'receipt.json')}
    report_payload = (EVIDENCE / 'report.json').read_bytes()
    report = json.loads(report_payload)
    manifest_payload = (EVIDENCE / 'run_manifest.json').read_bytes()
    manifest = json.loads(manifest_payload)
    if sha(report_payload) != manifest['report_sha256']:
        raise ValueError('report hash mismatch')
    pools = []
    pool_inspection = 'environment limits only'
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        if any(pool.get('num_threads') != 1 for pool in pools):
            raise ValueError('NumPy thread pool exceeds one thread')
        pool_inspection = 'threadpoolctl verified all loaded pools use one thread'
    except ImportError:
        pass
    comparisons = []
    artifacts_seen = set()
    for row in report['comparisons']:
        key = row['arm'] + '_' + canonical_sha(row['example_id'])
        cpu_name = 'logits/cpu_' + key + '.pt'
        gpu_name = 'logits/cuda_0_' + key + '.pt'
        reference, cpu_receipt = read_logits(cpu_name, report['artifacts'][cpu_name])
        actual, gpu_receipt = read_logits(gpu_name, report['artifacts'][gpu_name])
        difference = actual - reference
        reference_norm = float(np.linalg.norm(reference))
        actual_norm = float(np.linalg.norm(actual))
        if min(reference_norm, actual_norm) <= 1e-12:
            raise ValueError('unexpected near-zero norm')
        observed = {
            'max_abs': float(np.max(np.abs(difference))),
            'relative_l2': float(np.linalg.norm(difference) / reference_norm),
            'cosine': float(np.dot(reference, actual) / (reference_norm * actual_norm)),
        }
        recorded = row['logits']
        deltas = {name: abs(observed[name] - recorded[name]) for name in TOLERANCES}
        agreements = {name: deltas[name] <= rule['atol'] + rule['rtol'] * abs(recorded[name])
                      for name, rule in TOLERANCES.items()}
        q = report['thresholds']
        gates = {
            'absolute': observed['max_abs'] <= q['max_abs_logit_error'],
            'relative': observed['relative_l2'] <= q['max_relative_logit_l2'],
            'cosine': observed['cosine'] >= q['minimum_logit_cosine'],
        }
        comparisons.append({
            'arm': row['arm'], 'example_id': row['example_id'],
            'artifacts': [cpu_receipt, gpu_receipt],
            'recorded': recorded, 'recomputed_numpy_float64': observed,
            'absolute_metric_differences': deltas,
            'metric_matches_recorded_within_tolerance': agreements,
            'registered_logit_gates': gates,
            'pass': all(agreements.values()) and all(gates.values()),
        })
        artifacts_seen.update((cpu_name, gpu_name))
        del reference, actual, difference
    if len(comparisons) != 40 or len(artifacts_seen) != 80 or artifacts_seen != set(report['artifacts']):
        raise ValueError('incomplete artifact coverage')
    unchanged = {name: sha((HERE / name).read_bytes()) == digest
                 for name, digest in prior_hashes.items()}
    if not all(unchanged.values()):
        raise ValueError('earlier audit changed during inspection')
    results = {
        'schema': 'native_first_token_logit_independent_recomputation_v1',
        'start_utc': started,
        'end_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script_sha256': sha(pathlib.Path(__file__).read_bytes()),
        'evidence_directory': str(EVIDENCE),
        'report_sha256': sha(report_payload),
        'run_manifest_sha256': sha(manifest_payload),
        'python': platform.python_version(), 'numpy': np.__version__,
        'cpu_thread_limit': 1,
        'thread_environment': {key: os.environ[key] for key in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        'threadpool_verification': pool_inspection, 'threadpools': pools,
        'method': 'Read each saved FP32 tensor storage with zipfile and restricted metadata parser; convert to NumPy float64; compute maximum absolute difference, error L2/reference L2, dot/(reference L2 * actual L2); retain one CPU/GPU pair at a time.',
        'comparison_rule': 'abs(recomputed-recorded) <= atol + rtol*abs(recorded)',
        'comparison_tolerances_fixed_before_execution': TOLERANCES,
        'tolerance_rationale': 'Maximum absolute difference must be exact; relative norms permit FP64 reduction rounding, and cosine allows a 5e-14 absolute roundoff margin. Frozen scientific gates are applied without this matching tolerance.',
        'frozen_logit_gates': {key: report['thresholds'][key] for key in (
            'max_abs_logit_error', 'max_relative_logit_l2', 'minimum_logit_cosine')},
        'models_loaded': 0, 'torch_imported': 'torch' in sys.modules,
        'programme_comparison_helpers_imported': False, 'gpu_execution': False,
        'checkpoint_access': False, 'architecture_accuracy_inspected': False,
        'comparison_count': len(comparisons), 'artifacts_hashed_and_decoded': len(artifacts_seen),
        'earlier_audit_sha256': prior_hashes, 'earlier_audit_preserved': unchanged,
        'maximum_absolute_metric_disagreement': {
            name: max(row['absolute_metric_differences'][name] for row in comparisons)
            for name in TOLERANCES},
        'all_recorded_metric_triples_reproduced': all(
            all(row['metric_matches_recorded_within_tolerance'].values()) for row in comparisons),
        'all_recomputed_logit_gates_pass': all(
            all(row['registered_logit_gates'].values()) for row in comparisons),
        'status': 'PASS' if all(row['pass'] for row in comparisons) else 'FAIL',
        'comparisons': comparisons,
    }
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(results, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: value for key, value in results.items()
                      if key not in ('comparisons', 'threadpools')}, indent=2))
    if results['status'] != 'PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
