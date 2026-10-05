"""Independent CPU mutation review. No native driver, decoder, model, or binary reads."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib, importlib.util, json, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STAGE = PACKAGE / 'artifacts/short_phase_audit_work'
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(PACKAGE))
import phase_replay as phase
import audit_adapter as adapter

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def source_hashes():
    return {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [STAGE / 'phase_replay.py', STAGE / 'audit_adapter.py',
                      STAGE / 'test_phase_audit.py',
                      STAGE / 'pinned_dependencies/RAM2_auditor_d69810.py']}

def resign(values):
    markers, summary, worker, journal = values
    # Rebuild all self-authenticated counts, ordinals, clocks and byte claims,
    # so negatives exercise substantive gates rather than stale checksums.
    stack, next_id = [], 0
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    for index, marker in enumerate(markers):
        p = marker['phase']
        if p['event'] == 'attempt':
            next_id += 1
            stack.append(next_id)
            p['ordinal'] = next_id
        else:
            p['ordinal'] = stack.pop()
        marker['name'] = 'phase_%04d_%s' % (p['ordinal'], 'start' if p['event'] == 'attempt' else 'end')
        p['monotonic'] = index / 100000.
        p['utc'] = (start + timedelta(microseconds=index)).isoformat()
    nodes, roots, counts = phase.tree(markers)
    summary['phase_accounting'].update(operations=counts, last_ordinal=len(nodes), last_phase=deepcopy(markers[-1]['phase']))
    size = sum(len((phase.canonical(v) + '\n').encode('ascii')) for v in markers)
    summary['phase_IO'].update(markers_returned=len(markers), marker_bytes_attempted=size, marker_bytes_returned=size)
    summary['phase_IO']['operations']['phase_marker_write'] = {'attempted': len(markers), 'returned': len(markers), 'failed': 0, 'pending': 0}
    summary['sha256'] = phase.digest({k: v for k, v in summary.items() if k != 'sha256'})
    return values

def balanced(name, result=None):
    begin = dict(ordinal=1, operation=name, event='attempt', monotonic=0.,
                 utc='2026-10-05T00:00:00+00:00', historical_cause_established=False)
    end = {**begin, 'event': 'returned', 'result': result, 'error': None}
    return [{'name': 'phase_0001_start', 'phase': begin}, {'name': 'phase_0001_end', 'phase': end}]

def remove_operation(markers, name):
    first = next(i for i, m in enumerate(markers) if m['phase']['operation'] == name and m['phase']['event'] == 'attempt')
    ordinal = markers[first]['phase']['ordinal']
    last = next(i for i in range(first + 1, len(markers)) if markers[i]['phase']['ordinal'] == ordinal)
    del markers[first:last + 1]

def returned(markers, name):
    return next(m['phase'] for m in markers if m['phase']['operation'] == name and m['phase']['event'] == 'returned')

def main():
    before = source_hashes()
    supplier = load('independent_cpu_phase_supplier', PACKAGE / 'tests/test_short_phase_collection.py')
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        unused, cuda, markers, final, worker, journal, identity = supplier.complete_collection(directory)
    good = [markers, final, worker, journal]
    controls = []
    adapted_source, adaptations = adapter.prepared_source()
    if len(adaptations) != 14:
        raise ValueError('independent exact14 source adaptation review')
    loader_profile = phase.loader_source((PACKAGE / 'native_loader.py').read_bytes())
    if loader_profile['asset_order'] == sorted(loader_profile['asset_order']):
        raise ValueError('canonical receipt order negative control unexpectedly disappeared')
    def negative(name, mutate, resign_evidence=True):
        values = deepcopy(good)
        mutate(values)
        if resign_evidence:
            resign(values)
        try:
            phase.replay(*values, fixture=True)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            controls.append({'id': name, 'rejected': True, 'error': str(exc)})
        else:
            controls.append({'id': name, 'rejected': False, 'error': None})
    positive = phase.replay(*deepcopy(good), fixture=True)
    negative('unknown_balanced_root', lambda v: v[0].extend(balanced('UNREGISTERED_BALANCED_PHASE')))
    negative('missing_completed_guard', lambda v: remove_operation(v[0], 'guard_completed'))
    def extra_child(v):
        index = next(i for i, m in enumerate(v[0]) if m['phase']['operation'] == 'main_disk_admission' and m['phase']['event'] == 'attempt')
        v[0][index + 1:index + 1] = balanced('UNREGISTERED_CHILD')
    negative('unknown_balanced_main_child', extra_child)
    negative('below_disk_admission_floor', lambda v: returned(v[0], 'main_disk_admission').update(result=1))
    negative('wrong_framework_versions', lambda v: returned(v[0], 'main_framework_version_read').update(result={'torch': 'FORGED', 'transformers': 'FORGED'}))
    negative('post_import_growth_reserve_missing', lambda v: returned(v[0], 'main_post_import_growth_observation').update(result={'available_bytes': 1, 'worker_RSS_bytes': 0}))
    negative('false_success_terminal_query', lambda v: returned(v[0], 'terminal_cuda_mem_get_info').update(error={'type': 'OSError', 'message': 'swallowed actual query failure'}), False)
    negative('unknown_Journal_event', lambda v: v[3].append({'event': 'UNREGISTERED_EVENT'}))
    def reorder_returns(v):
        indices = [i for i, e in enumerate(v[3]) if e['event'] == 'returned']
        left, right = indices[-2:]
        v[3][left], v[3][right] = v[3][right], v[3][left]
    negative('reordered_Journal_terminal_events', reorder_returns)
    negative('wrong_primitive_result', lambda v: returned(v[0], 'primitive_inference_forward').update(result='UNEXPECTED_RETAINED_VALUE'))
    negative('summary_self_authorizes_scoring', lambda v: v[1].update(scoring_authorized=True))
    negative('summary_self_authorizes_fit', lambda v: v[1].update(native_fit_established=True))
    negative('changed_worker_scope', lambda v: v[2].update(scope='FORGED_NATIVE_AUTHORITY'))
    negative('changed_worker_config', lambda v: v[2].update(config_sha256='0' * 64))
    negative('changed_worker_commit', lambda v: v[2].update(source_commit='FORGED_COMMIT'))
    for name, raw in [('duplicate_json_key', b'{"name":"x","name":"y"}\n'),
                      ('nonfinite_literal', b'{"x":NaN}\n'),
                      ('missing_newline', b'{}'),
                      ('oversized_line', b'x' * 65537 + b'\n')]:
        try:
            phase.parse(raw)
        except (ValueError, json.JSONDecodeError):
            controls.append({'id': name, 'rejected': True})
        else:
            controls.append({'id': name, 'rejected': False})
    forbidden = [n for n in ('torch', 'transformers', 'tokenizers') if n in sys.modules]
    after = source_hashes()
    receipt = {'schema_version': 1, 'kind': 'INDEPENDENT_CPU_PHASE_REPLAY_MUTATION_REVIEW',
               'positive_fixture': positive, 'controls': controls,
               'exact_dependency14_adaptations': adaptations,
               'independent_source_loader_asset_order': loader_profile['asset_order'],
               'source_shape_count': len(loader_profile['shapes']),
               'all_negatives_rejected': all(c['rejected'] for c in controls),
               'source_before': before, 'source_after': after, 'source_bytes_unchanged': before == after,
               'forbidden_runtime_imports': forbidden, 'native_driver_called': False,
               'model_binary_or_GPU_execution': False, 'native_PASS_or_study_result': False}
    receipt['sha256'] = phase.digest(receipt)
    target = HERE / ('independent_phase_review_%d.json' % time.time_ns())
    target.write_text(phase.canonical(receipt) + '\n', encoding='ascii')
    print(json.dumps({'receipt': str(target), 'negative_controls': len(controls),
                      'failed_controls': [c['id'] for c in controls if not c['rejected']],
                      'source_bytes_unchanged': before == after, 'forbidden_runtime_imports': forbidden}))
    return 0 if receipt['all_negatives_rejected'] and before == after and not forbidden else 1

if __name__ == '__main__':
    raise SystemExit(main())
