"""Resumable per-cell CPU/GPU qualification, separate from efficacy generation.

Each CPU and GPU phase is a separate process/run with committed completion
evidence. A lost CUDA context cannot erase an already completed CPU reference.
Hard cell rejection is distinct from resource/integrity failures and pending
work. No partial fixture denominator or implicit numerical tolerance is used.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import time
import traceback

from . import upstream
from .provenance import STAGE, REPO, stage_sources, verify_registration, require_committed
from shared.hashing import canonical_sha256, sha256_file, write_json
from shared.manifests import git_identity, utc_now

EVIDENCE = STAGE / 'evidence/01_runtime_qualification'
SCHEMA = 'cross_scale_generation_cell_qualification_phase_v1'


class QualificationPendingError(RuntimeError):
    pass


def _check(ok, message):
    if not ok:
        raise RuntimeError(message)


def _fixtures(config):
    from shared.datasets import load_frozen_dataset
    rows = load_frozen_dataset(REPO / config['dataset_manifest'], track='cross_scale', role='calibration')
    families = sorted({r['family'] for r in rows})
    _check(len(families) == 8, 'eight calibration families required')
    chosen = [min((r for r in rows if r['family'] == f), key=lambda r: r['example_id']) for f in families]
    _check(len({r['scenario_id'] for r in chosen}) == 8, 'qualification scenarios are not unique')
    return chosen


def _root(config, registration, cell_id):
    _check(cell_id in config['cells'], 'unregistered qualification cell')
    return EVIDENCE / registration['registration_sha256'] / cell_id


def _binding(config, registration, cell_id, phase):
    _check(phase in ('cpu', 'gpu'), 'unknown qualification phase')
    _check(phase != 'gpu' or config['device'] == 'cuda:0', 'GPU phase requires GPU screen config')
    return {'schema_version': SCHEMA, 'track': 'cross_scale', 'purpose': 'runtime_qualification_only',
            'registration_sha256': registration['registration_sha256'],
            'config_sha256': canonical_sha256(config), 'source_sha256': canonical_sha256(registration['sources']),
            'sources': registration['sources'], 'cell_id': cell_id, 'cell': config['cells'][cell_id],
            'phase': phase, 'device': 'cpu' if phase == 'cpu' else 'cuda:0',
            'dataset_manifest_sha256': config['dataset_manifest_sha256'],
            'thresholds': config['qualification'], 'generation_role': 'calibration'}


def _hash_artifacts(directory):
    return {p.relative_to(directory).as_posix(): sha256_file(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name not in ('run_manifest.json', 'abandonment.json')}


def _read_phase(directory, config, registration, cell_id, phase, *, committed=True):
    start = upstream._read(directory / 'started.json')
    final = directory / 'run_manifest.json'
    if not final.exists():
        final = directory / 'abandonment.json'
    if not final.exists():
        raise QualificationPendingError('unfinished qualification phase: ' + str(directory))
    receipt = upstream._read(final)
    binding = _binding(config, registration, cell_id, phase)
    _check(all(start.get(k) == v and receipt.get(k) == v for k, v in binding.items()), 'phase binding changed')
    _check(all(receipt.get(k) == v for k, v in start.items() if k != 'status'), 'phase rewrote started contract')
    _check(receipt.get('status') in ('COMPLETE', 'FAILED', 'ABANDONED'), 'invalid phase terminal status')
    _check(receipt.get('artifacts_sha256') == _hash_artifacts(directory), 'phase artifact inventory changed')
    if receipt['status'] == 'ABANDONED':
        _check(receipt.get('original_process_absent') is True, 'phase abandonment lacks process verification')
    if receipt['status'] == 'FAILED':
        _check((directory / 'failure.json').is_file(), 'phase failure receipt missing')
    if receipt['status'] == 'COMPLETE':
        report = upstream._read(directory / 'report.json')
        _check(report.get('cell_id') == cell_id, 'phase report contains another cell')
        validate_phase_report(report, config, _fixtures(config), phase)
        validate_identity_artifacts(directory, report, phase)
        _check(sha256_file(directory / 'report.json') == receipt['report_sha256'], 'phase report changed')
    if committed:
        require_committed(final)
        require_committed(directory / 'started.json')
        if receipt['status'] == 'COMPLETE':
            require_committed(directory / 'report.json')
    return receipt, final


def _phase_inventory(config, registration, cell_id, phase, *, committed=True):
    root = upstream._within(_root(config, registration, cell_id) / phase, EVIDENCE)
    entries = []
    for directory in sorted(root.iterdir()) if root.exists() else []:
        directory = upstream._within(directory, root)
        _check(directory.is_dir() and directory.name.startswith('attempt_') and directory.name[8:].isdigit(),
               'unknown qualification attempt')
        receipt, path = _read_phase(directory, config, registration, cell_id, phase, committed=committed)
        entries.append((directory, receipt, path))
    _check(len([x for x in entries if x[1]['status'] == 'COMPLETE']) <= 1, 'duplicate complete qualification phases')
    return entries


def _result(directory, receipt, path, status, **extra):
    return {'status': status, 'phase': 'qualification', 'directory': str(directory),
            'receipt_sha256': sha256_file(path), 'receipt': receipt, **extra}


def _find_phase(config, registration, cell_id, phase, *, committed=True):
    entries = _phase_inventory(config, registration, cell_id, phase, committed=committed)
    hard = [x for x in entries if x[1]['status'] == 'FAILED' and x[1].get('failure_class') == 'cell_rejection']
    if hard:
        _check(len(hard) == 1 and not any(x[1]['status'] == 'COMPLETE' for x in entries),
               'hard qualification rejection cannot be silently retried')
        d, r, p = hard[0]
        return _result(d, r, p, 'FAILED', reason=upstream._read(d / 'failure.json')['message'])
    completed = [x for x in entries if x[1]['status'] == 'COMPLETE']
    if not completed:
        raise QualificationPendingError('no complete qualification phase: ' + str(_root(config, registration, cell_id) / phase))
    d, r, p = completed[0]
    return _result(d, r, p, 'PASS')


def validate_phase_report(report, config, fixtures, phase):
    cell_id = report.get('cell_id')
    _check(cell_id in config['cells'], 'phase report cell is unknown')
    arm = config['cells'][cell_id]['arm']
    ids = [r['example_id'] for r in fixtures]
    _check(report.get('fixture_ids') == ids and report.get('fixture_prompt_sha256') ==
           {r['example_id']: r['prompt_sha256'] for r in fixtures}, 'qualification fixture coverage changed')
    rows = report.get('cases', {})
    _check(set(rows) == set(ids), 'qualification cell requires all eight fixtures')
    identity = report.get('model_identity', {})
    _check(identity.get('canonical_unchanged') is True and identity.get('model_class') == config['model_class'],
           'canonical qualification identity failed')
    _check(identity.get('state_sha256_before') == identity.get('state_sha256_after')
           and identity.get('state_sha256_before'), 'canonical phase state changed')
    _check(identity.get('checkpoint_files_before') == identity.get('checkpoint_files_after')
           and identity['checkpoint_files_before']['model.safetensors'] == config['checkpoint_sha256'],
           'canonical checkpoint changed')
    _check(identity.get('trainable_parameters') == 0 and identity.get('backbone_gradients') == 0,
           'qualification backbone trained or has gradients')
    runtime = report.get('phase_runtime_identity', {})
    _check(runtime.get('device') == ('cpu' if phase == 'cpu' else 'cuda:0') and
           runtime.get('dtype') == 'float32' and runtime.get('threads') == 4 and
           runtime.get('seed') == config['seed'] and runtime.get('torch') == '2.12.1+cu126' and
           runtime.get('transformers') == '5.14.1' and runtime.get('attention') == 'eager' and
           runtime.get('use_cache') is False, 'qualification runtime differs from frozen profile')
    _check(all(report.get(k) is True for k in ('disabled_identity', 'zero_gate_identity', 'restored_identity')),
           'disabled/zero/restoration identity failed')
    _check(report['operator_before'] == report['operator_after'], 'operator changed during qualification')
    if arm in ('XG', 'XS', 'XR'):
        _check(isinstance(report.get('donor_identity_before'), dict), 'graft phase lacks donor checkpoint identity')
        _check(report['donor_identity_before'] == report.get('donor_identity_after'),
               'donor checkpoint identity changed during qualification')
        _check(report['donor_identity_before']['revision'] == config['donor_revision'], 'donor revision differs')
    else:
        _check(report.get('donor_identity_before') is None and report.get('donor_identity_after') is None,
               'nongraft phase unexpectedly reports donor use')
    operator = report['operator_before']
    expected_parameters = 0 if arm == 'X0' else 106168320 if arm == 'XM' else 100930816
    _check(operator.get('parameters') == expected_parameters, 'qualified operator parameter count differs')
    if operator['parameters']:
        _check(operator.get('trainable_parameters') == 0 and operator.get('backbone_gradients') == 0
               and isinstance(operator.get('state_sha256'), str) and len(operator['state_sha256']) == 64,
               'qualification operator is not frozen')
    for r in rows.values():
        _check(r['row']['token_ids'] and not r['row'].get('safety_stopped'), 'invalid fixture completion')
        _check(isinstance(r['row']['gold_nll'], (float, int)) and math.isfinite(r['row']['gold_nll']), 'nonfinite NLL')
        _check(phase != 'gpu' or r.get('repeat_verified') is True, 'GPU repeat not verified')
        _check(set(r['hidden_shapes']) == {str(i) for i in config['qualification']['capture_layers']},
               'qualification hidden site coverage changed')
    return True


def _load_case(directory, record):
    import torch
    path = upstream._within(directory / record['tensor_path'], directory)
    _check(sha256_file(path) == record['tensor_sha256'], 'qualification tensors changed')
    value = torch.load(path, map_location='cpu', weights_only=True)
    _check(set(value) == {'logits', 'hidden'}, 'unknown qualification tensor payload')
    tensors = [value['logits'], *value['hidden'].values()]
    _check(all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.device.type == 'cpu'
               and not t.requires_grad and torch.isfinite(t).all() for t in tensors), 'invalid saved qualification tensor')
    _check(value['logits'].ndim == 1 and value['logits'].numel() > 0, 'invalid first-token logits')
    _check({k: list(v.shape) for k, v in value['hidden'].items()} == record['hidden_shapes'], 'hidden tensor shape changed')
    return value


def compare_phase_reports(cpu_dir, gpu_dir, config):
    from scripts.qualify_native_gpu import numeric_agreement
    cpu = upstream._read(cpu_dir / 'report.json')
    gpu = upstream._read(gpu_dir / 'report.json')
    for key in ('state_sha256_before', 'checkpoint_files_before', 'qwen_implementation_sha256'):
        _check(cpu['model_identity'][key] == gpu['model_identity'][key], 'CPU/GPU canonical identity differs')
    _check(cpu['operator_before'] == gpu['operator_before'], 'CPU/GPU operator tensors differ')
    _check(cpu.get('donor_identity_before') == gpu.get('donor_identity_before'), 'CPU/GPU donor checkpoint identities differ')
    q, rows = config['qualification'], []
    for identity in cpu['fixture_ids']:
        left, right = cpu['cases'][identity], gpu['cases'][identity]
        a, b = _load_case(cpu_dir, left), _load_case(gpu_dir, right)
        logits = numeric_agreement(a['logits'], b['logits'])
        hidden = {key: numeric_agreement(a['hidden'][key], b['hidden'][key], tokenwise=True) for key in a['hidden']}
        nll = abs(left['row']['gold_nll'] - right['row']['gold_nll'])
        gates = {'logit_abs': logits['max_abs'] <= q['max_abs_logit_error'],
                 'logit_relative': logits['relative_l2'] <= q['max_relative_logit_l2'],
                 'logit_cosine': logits['cosine'] >= q['minimum_logit_cosine'],
                 'nll': nll <= q['max_abs_nll_error'],
                 'greedy_identity': left['row']['token_ids'] == right['row']['token_ids'],
                 'gpu_repeat': right['repeat_verified'] is True,
                 'hidden': all(v['relative_l2'] <= q['max_relative_hidden_l2'] and
                               v['cosine'] >= q['minimum_hidden_cosine'] for v in hidden.values())}
        rows.append({'example_id': identity, 'logits': logits, 'hidden': hidden,
                     'absolute_nll_error': nll, 'gates': gates, 'pass': all(gates.values())})
    return {'comparisons': rows, 'pass': all(row['pass'] for row in rows)}


def verify_cell_qualification(config, registration, cell_id):
    _check(registration['sources'] == stage_sources(), 'qualification source changed')
    cpu = _find_phase(config, registration, cell_id, 'cpu')
    if cpu['status'] == 'PASS' and config['device'] == 'cpu':
        from shared.runtime import configure_runtime, runtime_identity
        configure_runtime(config)
        _check(upstream._read(Path(cpu['directory']) / 'report.json')['phase_runtime_identity'] == runtime_identity(config),
               'CPU qualification runtime changed')
    if cpu['status'] == 'FAILED' or config['device'] == 'cpu':
        return cpu
    gpu = _find_phase(config, registration, cell_id, 'gpu')
    if gpu['status'] == 'FAILED':
        return gpu
    from shared.runtime import configure_runtime, runtime_identity
    configure_runtime(config)
    _check(upstream._read(Path(gpu['directory']) / 'report.json')['phase_runtime_identity'] == runtime_identity(config),
           'GPU qualification hardware/backend changed')
    path = _root(config, registration, cell_id) / 'comparison.json'
    if not path.exists():
        raise QualificationPendingError('complete CPU/GPU comparison is required')
    report = upstream._read(path)
    _check(report['registration_sha256'] == registration['registration_sha256']
           and report['config_sha256'] == canonical_sha256(config)
           and report['cpu_receipt_sha256'] == cpu['receipt_sha256']
           and report['gpu_receipt_sha256'] == gpu['receipt_sha256'], 'comparison binding changed')
    recomputed = compare_phase_reports(Path(cpu['directory']), Path(gpu['directory']), config)
    _check(report['result'] == recomputed, 'qualification comparison gates changed')
    require_committed(path)
    return _result(path.parent, report, path, 'PASS' if recomputed['pass'] else 'FAILED',
                   reason=None if recomputed['pass'] else 'CPU/GPU agreement gates failed')


def get_qualification_inventory(config, registration):
    result = {}
    for cell_id in config['cells']:
        try:
            result[cell_id] = verify_cell_qualification(config, registration, cell_id)
        except QualificationPendingError:
            pass
    return result


def _capture(model, tokenizer, case, config, wrapper):
    import torch
    from shared.generation import evaluate_case
    hidden = {}
    def capture(index, output):
        if str(index) not in hidden:
            _check(isinstance(output, torch.Tensor), 'unknown decoder output type')
            hidden[str(index)] = output.detach().cpu().clone()
    handles = [model.model.layers[i].register_forward_hook(
        lambda module, args, output, i=i: capture(i, output)) for i in config['qualification']['capture_layers']]
    try:
        row, logits = evaluate_case(model, tokenizer, case, config, wrapper)
    finally:
        for handle in handles:
            handle.remove()
    _check(set(hidden) == {str(i) for i in config['qualification']['capture_layers']}, 'hidden capture incomplete')
    return json.loads(json.dumps(row, allow_nan=False)), {'logits': logits.cpu(), 'hidden': hidden}


def _same(a, b):
    import torch
    ra, ta = a
    rb, tb = b
    return (ra['token_ids'] == rb['token_ids'] and ra['gold_nll'] == rb['gold_nll']
            and torch.equal(ta['logits'], tb['logits']) and set(ta['hidden']) == set(tb['hidden'])
            and all(torch.equal(ta['hidden'][k], tb['hidden'][k]) for k in ta['hidden']))


def _save_capture(directory, stem, capture):
    import torch
    row, tensors = capture
    target = upstream._within(directory / 'tensors' / (stem + '.pt'), directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as handle:
        torch.save(tensors, handle)
    return {'row': row, 'tensor_path': target.relative_to(directory).as_posix(),
            'tensor_sha256': sha256_file(target),
            'hidden_shapes': {k: list(v.shape) for k, v in tensors['hidden'].items()}}


def validate_identity_artifacts(directory, report, phase):
    """Recompute exact repeats and all bypass/restoration checks from raw saves."""
    probes = report.get('identity_probes', {})
    _check(set(probes) == {'baseline', 'disabled', 'zero_gate', 'restored'}, 'identity raw-probe coverage incomplete')
    baseline = (probes['baseline']['row'], _load_case(directory, probes['baseline']))
    if report['cell_id'] == 'X0':
        first = report['cases'][report['fixture_ids'][0]]
        _check(_same(baseline, (first['row'], _load_case(directory, first))), 'intact baseline changed within phase')
    for mode in ('disabled', 'zero_gate', 'restored'):
        probe = probes[mode]
        actual = _same(baseline, (probe['row'], _load_case(directory, probe)))
        _check(actual and report[mode + '_identity'] is actual, 'identity probe failed raw recomputation: ' + mode)
    for item in report['cases'].values():
        first = _load_case(directory, item)
        if phase == 'gpu':
            repeated = item.get('repeat_capture')
            _check(isinstance(repeated, dict), 'GPU repeated raw capture missing')
            actual = _same((item['row'], first), (repeated['row'], _load_case(directory, repeated)))
            _check(actual and item['repeat_verified'] is actual, 'GPU repeated raw capture differs')
        else:
            _check('repeat_capture' not in item and item['repeat_verified'] is False, 'CPU report falsely claims GPU repetition')
    return True


def _operator_snapshot(wrapper):
    from shared.integrity import frozen_inventory, state_hash
    from .operators import buffer_hashes
    module = getattr(wrapper, 'expert', getattr(wrapper, 'transform', None))
    if module is None:
        return {'parameters': 0, 'state_sha256': None, 'buffers': {}}
    return {**frozen_inventory(module), 'state_sha256': state_hash(module), 'buffers': buffer_hashes(module)}


def _failure_class(error, model_loaded):
    message = str(error).lower()
    if not model_loaded and isinstance(error, MemoryError) and message.startswith('insufficient '):
        return 'resource_admission'
    if isinstance(error, (MemoryError, TimeoutError)) or 'cuda' in message or 'machine lease' in message:
        return 'runtime_resource'
    if ('malformed generation' in message or 'hidden norm emergency' in message
            or 'hidden rms' in message or 'nonfinite' in message or 'repeat identity' in message):
        return 'cell_rejection'
    return 'fatal_integrity'


def run_phase(config_path, cell_id, phase, *, attempt=0):
    from contextlib import ExitStack
    import psutil
    import torch
    from .execution import operator_factory
    from shared.model_loading import load_supa
    from shared.integrity import assert_unchanged
    from shared.runtime import runtime_identity, configure_runtime, cuda_memory
    from shared.timing import machine_lease
    from armr.hooks import frozen_inference
    config_path = upstream._within(config_path, STAGE / 'configs')
    registration = verify_registration(config_path)
    config = registration['config']
    _check(type(attempt) is int and attempt >= 0, 'invalid phase attempt')
    entries = _phase_inventory(config, registration, cell_id, phase, committed=False)
    _check(not any(x[1]['status'] == 'COMPLETE' or x[1].get('failure_class') == 'cell_rejection' for x in entries),
           'complete or hard-rejected phase cannot rerun')
    _check(attempt == max((x[1]['attempt'] for x in entries), default=-1) + 1, 'phase attempts must be consecutive')
    if cell_id != 'X0':
        base = (verify_cell_qualification(config, registration, 'X0') if phase == 'gpu'
                else _find_phase(config, registration, 'X0', phase))
        _check(base['status'] == 'PASS', 'baseline phase must pass before candidate qualification')
    if phase == 'gpu':
        reference = _find_phase(config, registration, cell_id, 'cpu')
        _check(reference['status'] == 'PASS', 'CPU reference must pass before GPU qualification')
    phase_config = {**config, 'device': 'cpu' if phase == 'cpu' else 'cuda:0',
                    'min_available_gib': config['qualification']['cpu_min_available_gib'] if phase == 'cpu' else config['min_available_gib']}
    fixtures = _fixtures(config)
    directory = _root(config, registration, cell_id) / phase / f'attempt_{attempt:03d}'
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {**_binding(config, registration, cell_id, phase), **git_identity(),
               'status': 'STARTED', 'attempt': attempt, 'pid': os.getpid(),
               'process_create_time': psutil.Process().create_time(), 'start_utc': utc_now()}
    write_json(directory / 'started.json', receipt)
    print(f'QUALIFICATION_DIR={directory}', flush=True)
    model = tokenizer = wrapper = None
    identity = None
    started = time.perf_counter()
    failure = None
    lease_acquired = False
    with ExitStack() as lease_stack:
        try:
            lease_stack.enter_context(machine_lease())
            lease_acquired = True
            configure_runtime(phase_config)
            model, tokenizer, identity = load_supa(phase_config)
            report = {'cell_id': cell_id, 'fixture_ids': [r['example_id'] for r in fixtures],
                      'fixture_prompt_sha256': {r['example_id']: r['prompt_sha256'] for r in fixtures},
                      'cases': {}, 'phase_runtime_identity': runtime_identity(phase_config),
                      'identity_probes': {}, 'donor_identity_before': None, 'donor_identity_after': None}
            with frozen_inference(model), operator_factory(model, 'X0', phase_config) as base_wrapper:
                baseline = _capture(model, tokenizer, fixtures[0], phase_config, base_wrapper)
            report['identity_probes']['baseline'] = _save_capture(directory, 'identity_baseline', baseline)
            wrapper = operator_factory(model, cell_id, phase_config)
            if hasattr(wrapper, 'donor_identity'):
                from .provenance import verify_export
                export = config['upstream_export']
                expected_donor = verify_export(REPO / export['manifest_path'], export['manifest_sha256'])['context']['donor_identity']
                _check(wrapper.donor_identity == expected_donor, 'qualification donor differs from selected export')
                report['donor_identity_before'] = wrapper.donor_identity
            report['operator_before'] = _operator_snapshot(wrapper)
            with wrapper, frozen_inference(model):
                for case in fixtures:
                    value = _capture(model, tokenizer, case, phase_config, wrapper)
                    repeated = False
                    again = None
                    if phase == 'gpu':
                        again = _capture(model, tokenizer, case, phase_config, wrapper)
                        repeated = _same(value, again)
                    key = canonical_sha256(case['example_id'])
                    item = {**_save_capture(directory, key, value), 'repeat_verified': repeated}
                    if again is not None:
                        item['repeat_capture'] = _save_capture(directory, key + '_repeat', again)
                    write_json(directory / 'cases' / (key + '.json'), item)
                    report['cases'][case['example_id']] = item
                    if phase == 'gpu':
                        _check(repeated, 'GPU repeat identity failed')
                    print(f'QUAL {phase} {cell_id} {case["example_id"]}', flush=True)
            report['operator_after'] = _operator_snapshot(wrapper)
            _check(report['operator_before'] == report['operator_after'], 'operator state changed')
            del wrapper
            wrapper = None
            gc.collect()
            for mode, result_key in (('disabled', 'disabled_identity'), ('zero_gate', 'zero_gate_identity'), (None, 'restored_identity')):
                probe_cell = 'X0' if mode is None else cell_id
                wrapper = operator_factory(model, probe_cell, phase_config, diagnostic_mode=mode)
                with wrapper, frozen_inference(model):
                    probe_value = _capture(model, tokenizer, fixtures[0], phase_config, wrapper)
                    report[result_key] = _same(baseline, probe_value)
                    label = mode or 'restored'
                    report['identity_probes'][label] = _save_capture(directory, 'identity_' + label, probe_value)
                _check(report[result_key], result_key + ' failed')
                del wrapper
                wrapper = None
                gc.collect()
            if report['donor_identity_before'] is not None:
                from cross_scale.src.donor_layers import donor_identity
                _, report['donor_identity_after'] = donor_identity(phase_config)
                _check(report['donor_identity_before'] == report['donor_identity_after'], 'donor checkpoint changed')
            identity.update(assert_unchanged(model, identity['state_sha256_before'], Path(identity['model_path']), identity['checkpoint_files_before']))
            report['model_identity'] = json.loads(json.dumps(identity, allow_nan=False))
            validate_phase_report(report, config, fixtures, phase)
            validate_identity_artifacts(directory, report, phase)
            _check(stage_sources() == registration['sources'], 'stage source changed during qualification')
            write_json(directory / 'report.json', report)
            receipt.update(status='COMPLETE', report_sha256=sha256_file(directory / 'report.json'), completed_fixtures=8)
        except BaseException as error:
            failure = error
            kind = _failure_class(error, model is not None) if lease_acquired else 'resource_admission'
            write_json(directory / 'failure.json', {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
            receipt.update(status='FAILED', failure_class=kind, completed_fixtures=len(list((directory / 'cases').glob('*.json'))))
            if model is not None and identity is not None:
                try:
                    receipt['failure_integrity'] = assert_unchanged(model, identity['state_sha256_before'], Path(identity['model_path']), identity['checkpoint_files_before'])
                except BaseException as integrity_error:
                    receipt['failure_integrity_error'] = str(integrity_error)
                    receipt['failure_class'] = 'fatal_integrity'
            if wrapper is not None and 'report' in locals() and 'operator_before' in report:
                try:
                    after = _operator_snapshot(wrapper)
                    receipt['failure_operator_integrity'] = {'before': report['operator_before'], 'after': after,
                                                             'unchanged': after == report['operator_before']}
                    if after != report['operator_before']:
                        receipt['failure_class'] = 'fatal_integrity'
                except BaseException as integrity_error:
                    receipt['failure_operator_integrity_error'] = str(integrity_error)
                    receipt['failure_class'] = 'fatal_integrity'
            if 'report' in locals() and report.get('donor_identity_before') is not None:
                try:
                    from cross_scale.src.donor_layers import donor_identity
                    _, after = donor_identity(phase_config)
                    receipt['failure_donor_integrity'] = {'before': report['donor_identity_before'], 'after': after,
                                                          'unchanged': after == report['donor_identity_before']}
                    if after != report['donor_identity_before']:
                        receipt['failure_class'] = 'fatal_integrity'
                except BaseException as integrity_error:
                    receipt['failure_donor_integrity_error'] = str(integrity_error)
                    receipt['failure_class'] = 'fatal_integrity'
        finally:
            receipt.update(end_utc=utc_now(), wall_seconds=time.perf_counter() - started,
                           peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss))
            try:
                receipt['cuda_memory'] = (cuda_memory(phase_config) if model is not None or
                    (phase == 'gpu' and torch.cuda.is_initialized()) else {'not_initialized': True})
            except BaseException as resource_error:
                receipt['cuda_memory_error'] = str(resource_error)
                if receipt['status'] == 'COMPLETE':
                    receipt.update(status='FAILED', failure_class='runtime_resource')
                    write_json(directory / 'failure.json', {'type': type(resource_error).__name__, 'message': str(resource_error)})
            receipt['artifacts_sha256'] = _hash_artifacts(directory)
            write_json(directory / 'run_manifest.json', receipt)
            del model, tokenizer, wrapper
            gc.collect()
            if phase == 'gpu' and torch.cuda.is_initialized():
                try:
                    torch.cuda.empty_cache()
                except BaseException:
                    # The already recorded CUDA failure must retain its receipt.
                    pass
    if receipt['status'] == 'FAILED' and receipt['failure_class'] != 'cell_rejection':
        raise QualificationPendingError('qualification stopped for review: ' + str(directory)) from failure
    return directory


def finalize_cell(config_path, cell_id):
    registration = verify_registration(config_path)
    config = registration['config']
    cpu = _find_phase(config, registration, cell_id, 'cpu')
    if cpu['status'] == 'FAILED' or config['device'] == 'cpu':
        return cpu
    gpu = _find_phase(config, registration, cell_id, 'gpu')
    if gpu['status'] == 'FAILED':
        return gpu
    result = compare_phase_reports(Path(cpu['directory']), Path(gpu['directory']), config)
    path = _root(config, registration, cell_id) / 'comparison.json'
    write_json(path, {'status': 'PASS' if result['pass'] else 'FAILED', 'phase': 'qualification',
                      'cell_id': cell_id, 'registration_sha256': registration['registration_sha256'],
                      'config_sha256': canonical_sha256(config), 'cpu_receipt_sha256': cpu['receipt_sha256'],
                      'gpu_receipt_sha256': gpu['receipt_sha256'], 'result': result, 'created_utc': utc_now()})
    return path


def abandon_phase(config_path, cell_id, phase, attempt):
    import psutil
    registration = verify_registration(config_path)
    directory = _root(registration['config'], registration, cell_id) / phase / f'attempt_{attempt:03d}'
    _check(not (directory / 'run_manifest.json').exists() and not (directory / 'abandonment.json').exists(), 'phase already terminal')
    start = upstream._read(directory / 'started.json')
    _check(type(start.get('pid')) is int and not psutil.pid_exists(start['pid']), 'original phase process is live or unknown')
    write_json(directory / 'abandonment.json', {**start, 'status': 'ABANDONED',
               'original_process_absent': True, 'abandoned_utc': utc_now(), 'artifacts_sha256': _hash_artifacts(directory)})
    return directory / 'abandonment.json'
