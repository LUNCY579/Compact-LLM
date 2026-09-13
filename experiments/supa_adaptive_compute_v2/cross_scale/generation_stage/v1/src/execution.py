"""Immutable, architecture-only Stage 2 execution and raw-evidence audit.

The shared scorer/generator remain unchanged. This caller owns registration,
cell lifecycle and evidence because Stage 1 never licensed generation. A failed
cell has no partial accuracy denominator. Interrupted attempts require explicit
PID-verified abandonment; resumption cannot quietly replace evidence.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

from shared.hashing import canonical_sha256, sha256_file, write_json
from shared.guards import require_within
from shared.legacy import ROOT as PROGRAMME, REPO
from shared.evaluation import shard_cases
from shared.manifests import git_identity, utc_now

STAGE = Path(__file__).resolve().parents[1]
EVIDENCE = STAGE / 'evidence/04_small_generation'
AUDITS = STAGE / 'evidence/05_screen_audit'
ROLE = 'architecture_dev'
SCHEMA = 'cross_scale_stage2_generation_shard_v1'
IDENTITY_KEYS = ('state_sha256_before', 'checkpoint_files_before', 'model_class',
                 'torch', 'transformers', 'qwen_implementation_sha256', 'parameters', 'runtime_identity')
ROW_METADATA = ('example_id', 'case_id', 'scenario_id', 'family', 'split', 'track',
                'variant', 'template_id', 'template_unseen_in_engineering')


class EvidenceError(RuntimeError):
    """Evidence is changed, ambiguous, incomplete or outside the frozen stage."""


class CellRejected(RuntimeError):
    """The cell has a retained terminal mechanical failure."""


def _check(condition, message):
    if not condition:
        raise EvidenceError(message)


def _read(path):
    def bad_constant(value):
        raise EvidenceError('nonfinite JSON constant: ' + value)
    def pairs(values):
        result = {}
        for key, value in values:
            _check(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=bad_constant, object_pairs_hook=pairs)


def _finite(value):
    if isinstance(value, float):
        _check(math.isfinite(value), 'nonfinite evidence value')
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def _context(config_path):
    from .provenance import verify_registration, stage_sources
    from shared.datasets import load_frozen_dataset
    config_path = require_within(config_path, STAGE)
    registration = verify_registration(config_path)
    config = _read(config_path)
    _check(registration['config'] == config, 'configuration differs from stage registration')
    _check(config['data']['allowed_roles'] == [ROLE] and config['data']['track'] == 'cross_scale',
           'only Track B architecture development is executable')
    _check(registration['sources'] == stage_sources(), 'stage source identity changed')
    dataset = require_within(REPO / config['dataset_manifest'], PROGRAMME / 'data')
    _check(sha256_file(dataset) == config['dataset_manifest_sha256'], 'dataset hash mismatch')
    cases = load_frozen_dataset(dataset, track='cross_scale', role=ROLE)
    validate_screen_cases(cases)
    return config_path, config, registration, cases


def validate_screen_cases(cases):
    _check(len(cases) == 128 and len({x['example_id'] for x in cases}) == 128
           and len({x['scenario_id'] for x in cases}) == 128,
           'exactly 128 distinct frozen architecture scenarios required')
    _check(all(x['track'] == 'cross_scale' and x['split'] == ROLE and x['variant'] == 'original'
               for x in cases), 'foreign track, role or variant in generation inputs')


def _binding(config_path, config, registration):
    return {'schema_version': SCHEMA, 'track': 'cross_scale', 'role': ROLE,
            'registration_sha256': registration['registration_sha256'],
            'config_path': str(config_path.resolve()), 'config_sha256': canonical_sha256(config),
            'config_file_sha256': sha256_file(config_path), 'sources': registration['sources'],
            'source_sha256': canonical_sha256(registration['sources']),
            'dataset_manifest': config['dataset_manifest'],
            'dataset_manifest_sha256': config['dataset_manifest_sha256'],
            'upstream_export': config['upstream_export'], 'matched_control': config['matched_control'],
            'seed': config['seed'], 'threads': config['threads'], 'dtype': config['dtype'],
            'device': config['device'], 'cache_mode': 'full_prefix_no_cache',
            'max_prompt_seconds': config['max_prompt_seconds'], 'shard_count': config['shard_count']}


def _root(registration):
    digest = registration['registration_sha256']
    _check(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'invalid registration ID')
    return require_within(EVIDENCE / digest, EVIDENCE)


def _artifacts(directory):
    """All raw files, including partial/orphan artifacts, are bound at termination."""
    result = {}
    for path in sorted(directory.rglob('*')):
        resolved = require_within(path, directory)
        _check(resolved == path.absolute(), 'aliased raw evidence path')
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            if relative not in ('run_manifest.json', 'abandonment.json'):
                result[relative] = sha256_file(path)
    return result


def _expected_contract(binding, cell_id, config, cases, shard_index, attempt):
    _check(cell_id in config['cells'], 'unregistered cell')
    _check(type(attempt) is int and attempt >= 0, 'attempt must be a nonnegative integer')
    selected = shard_cases(cases, shard_index, config['shard_count'])
    _check(bool(selected), 'empty deterministic shard')
    return {**binding, 'cell_id': cell_id, 'cell': config['cells'][cell_id],
            'arm': config['cells'][cell_id]['arm'], 'shard_index': shard_index, 'attempt': attempt,
            'example_ids': [c['example_id'] for c in selected]}, selected


def _validate_terminal(directory, terminal, started, contract):
    _check(started.get('status') == 'STARTED', 'missing started status')
    _check(all(started.get(k) == v for k, v in contract.items()), 'started contract identity mismatch')
    _check(terminal.get('status') in ('COMPLETE', 'FAILED', 'ABANDONED'), 'invalid terminal status')
    _check(all(terminal.get(k) == v for k, v in started.items() if k != 'status'),
           'terminal receipt rewrote started contract')
    _check(terminal.get('artifacts_sha256') == _artifacts(directory), 'raw artifact inventory/hash changed')
    _check(terminal.get('started_sha256') == sha256_file(directory / 'started.json'), 'started receipt changed')
    if terminal['status'] == 'FAILED':
        _check((directory / 'failure.json').is_file(), 'failed attempt lacks failure receipt')
        _check(terminal.get('failure_class') in ('hard_failure', 'resource_admission'), 'unknown failure class')
        if terminal['failure_class'] == 'resource_admission':
            _check(terminal.get('completed_examples') == 0 and terminal.get('model_loaded') is False,
                   'only zero-example pre-load admission failures may retry')
    if terminal['status'] == 'ABANDONED':
        _check(terminal.get('original_process_absent') is True, 'unverified abandonment')
    return terminal


def inventory(config_path, config, registration, cases):
    """Validate every attempt, including failed ones; never choose a convenient retry."""
    root, binding = _root(registration), _binding(config_path, config, registration)
    state = {cell: {'complete': {}, 'failed': None, 'attempts': {}} for cell in config['cells']}
    if not root.exists():
        return state
    for cell_dir in sorted(root.iterdir()):
        cell = cell_dir.name
        _check(cell_dir.is_dir() and cell in state and cell_dir.resolve().parent == root,
               'unknown or aliased cell directory')
        target = state[cell]
        for shard_dir in sorted(cell_dir.iterdir()):
            _check(shard_dir.is_dir() and re.fullmatch('shard_[0-9]{3}', shard_dir.name), 'unknown shard directory')
            index = int(shard_dir.name[6:])
            require_within(shard_dir, cell_dir)
            _check(shard_dir.resolve() == cell_dir.resolve() / shard_dir.name, 'aliased shard directory')
            attempts = []
            for attempt_dir in sorted(shard_dir.iterdir()):
                _check(attempt_dir.is_dir() and re.fullmatch('attempt_[0-9]{3,}', attempt_dir.name), 'unknown attempt directory')
                _check(attempt_dir.resolve().parent == shard_dir.resolve(), 'aliased attempt directory')
                attempt = int(attempt_dir.name[8:])
                _check(attempt_dir.name == f'attempt_{attempt:03d}' and
                       attempt_dir.resolve() == shard_dir.resolve() / attempt_dir.name, 'noncanonical attempt directory')
                contract, _ = _expected_contract(binding, cell, config, cases, index, attempt)
                started = _read(attempt_dir / 'started.json')
                final, abandoned = attempt_dir / 'run_manifest.json', attempt_dir / 'abandonment.json'
                _check(not (final.exists() and abandoned.exists()), 'ambiguous terminal receipts')
                _check(final.exists() or abandoned.exists(),
                       f'unfinished worker requires explicit PID-verified abandonment: {attempt_dir}')
                receipt_path = final if final.exists() else abandoned
                terminal = _validate_terminal(attempt_dir, _read(receipt_path), started, contract)
                entry = {'directory': str(attempt_dir.resolve()), 'manifest_path': str(receipt_path.resolve()),
                         'manifest_sha256': sha256_file(receipt_path), 'manifest': terminal}
                attempts.append(entry)
                if terminal['status'] == 'COMPLETE':
                    _check(index not in target['complete'], 'duplicate complete shard')
                    target['complete'][index] = entry
                if terminal['status'] == 'FAILED' and terminal['failure_class'] == 'hard_failure':
                    _check(target['failed'] is None, 'multiple hard failures for a terminal cell')
                    target['failed'] = {'status': 'FAILED', 'phase': 'generation', 'cell_id': cell,
                                        **{k: entry[k] for k in ('directory', 'manifest_path', 'manifest_sha256')},
                                        'failure_receipt': terminal['failure_receipt']}
            target['attempts'][index] = attempts
        # A terminal failure closes every later shard, not just its own index.
        if target['failed']:
            failure = _read(target['failed']['manifest_path'])
            _check(all(e['manifest']['start_utc'] <= failure['end_utc']
                       for es in target['attempts'].values() for e in es),
                   'a new attempt was started after this cell terminally failed')
    return state


def _read_complete(entry, config, cases, cell_id):
    from shared.integrity import validate_inference_receipt
    from shared.scoring import score_response
    from armr.metrics import output_flags
    directory, manifest = Path(entry['directory']), entry['manifest']
    _check(sha256_file(Path(entry['manifest_path'])) == entry['manifest_sha256'], 'terminal receipt changed')
    _check(_artifacts(directory) == manifest['artifacts_sha256'], 'raw artifact changed after inventory')
    validate_inference_receipt(manifest, config)
    _check(manifest.get('operator_integrity_passed') is True, 'operator integrity not proven')
    _check(manifest['cell_id'] == cell_id and manifest['cell'] == config['cells'][cell_id], 'cell changed')
    _validate_operator_receipt(manifest, config)
    expected = shard_cases(cases, manifest['shard_index'], config['shard_count'])
    names = [canonical_sha256(case['example_id']) for case in expected]
    _check(set(p.name for p in (directory / 'cases').iterdir()) == {n + '.json' for n in names},
           'partial, extra or missing case files')
    _check(set(p.name for p in (directory / 'logits').iterdir()) == {n + '.pt' for n in names},
           'partial, extra or missing logits')
    rows = []
    for case, name in zip(expected, names):
        row = _read(directory / 'cases' / (name + '.json'))
        _finite(row)
        _check(all(row.get(k) == case.get(k) for k in ROW_METADATA), 'raw case metadata differs from frozen bank')
        _check(row.get('cell_id') == cell_id and row.get('arm') == config['cells'][cell_id]['arm'], 'raw cell identity mismatch')
        _check(row.get('run_id') == manifest['run_id'], 'raw run identity mismatch')
        _check(row.get('gate') == config['cells'][cell_id]['gate'], 'raw gate mismatch')
        _check(all(row.get(k) == v for k, v in manifest['operator'].items()), 'row cost differs from actual wrapper inventory')
        _check(row.get('safety_stopped') is False, 'safety-stopped row cannot complete')
        score = score_response(row['response'], case['scorer'])
        _check(row['score'] == float(score['score']) and row['correct'] == bool(score['score'] == 1), 'independent rescoring mismatch')
        _check(row.get('score_ambiguous') == score.get('ambiguous', False), 'score ambiguity mismatch')
        flags = output_flags(row['response'], row['token_ids'])
        _check(all(row.get(k) == v for k, v in flags.items()), 'independent validity flags mismatch')
        _check(not any(flags[k] for k in ('repetition_loop', 'script_corruption', 'control_count', 'replacement_count')),
               'malformed output cannot complete')
        bounds = config['hidden_rms_ratio_bounds']
        _check(bounds[0] <= row['trajectory_min_rms_ratio'] <= row['trajectory_max_rms_ratio'] <= bounds[1], 'hidden guard violation')
        _check(row['termination'] in ('eos', 'token_cap') and bool(row['eos_emitted']) == (row['termination'] == 'eos')
               and bool(row['truncated']) == (row['termination'] == 'token_cap'), 'termination mismatch')
        rows.append(row)
    _check(_read(directory / 'results.json') == rows, 'aggregate rows differ from raw case evidence')
    _check(manifest.get('completed_examples') == len(expected), 'partial denominator')
    return rows


def _validate_operator_receipt(manifest, config):
    cost = manifest.get('operator', {})
    keys = ('added_parameter_count', 'added_active_parameters', 'extra_matmul_flops_per_token')
    _check(set(cost) == set(keys) and all(type(cost[k]) is int and cost[k] >= 0 for k in keys),
           'complete integer actual operator inventory required')
    if manifest['arm'] == 'X0':
        _check(all(cost[k] == 0 for k in keys), 'X0 has unexpected additional work')
        return
    _check(bool(manifest.get('operator_state_sha256_before'))
           and manifest['operator_state_sha256_before'] == manifest.get('operator_state_sha256_after'), 'operator state changed or missing')
    for side in ('before', 'after'):
        inv = manifest.get('operator_inventory_' + side, {})
        _check(inv.get('trainable_parameters') == 0 and inv.get('backbone_gradients') == 0,
               'operator trainable/gradient inventory incomplete')
    receipt = manifest.get('operator_integrity_receipt') or {}
    _check(receipt.get('trainable_parameters') == 0 and receipt.get('backbone_gradients') == 0,
           'wrapper freeze receipt incomplete')
    _check(receipt.get('state_sha256_before') == manifest['operator_state_sha256_before']
           and receipt.get('state_sha256_after') == manifest['operator_state_sha256_after'],
           'wrapper and caller operator identities differ')
    for before, after in (('buffers_before', 'buffers_after'), ('treatment_before', 'treatment_after')):
        _check(before in receipt and after in receipt and receipt[before] == receipt[after],
               'operator buffer/treatment identity changed or missing')
    if manifest['arm'] == 'XM':
        control = config['matched_control']
        _check(manifest['operator_state_sha256_before'] == control['state_sha256'], 'XM differs from calibrated state')
        _check(cost == {'added_parameter_count': control['additional_parameters'],
                       'added_active_parameters': control['additional_active_parameters'],
                       'extra_matmul_flops_per_token': control['additional_matmul_flops_per_token']},
               'XM does not have registered matched compute')
    else:
        _check(receipt.get('gate_before') == manifest['cell']['gate'] == receipt.get('gate_after'), 'donor gate changed')
        donor = manifest.get('donor_identity_before')
        _check(bool(donor) and donor == manifest.get('donor_identity_after'), 'donor checkpoint integrity incomplete')


def _baseline(state, config, cases):
    base = state['X0']
    _check(base['failed'] is None and set(base['complete']) == set(range(config['shard_count'])),
           'complete matched X0 baseline required before any candidate shard')
    identities = set()
    for entry in base['complete'].values():
        _read_complete(entry, config, cases, 'X0')
        identities.add(canonical_sha256({k: entry['manifest'].get(k) for k in IDENTITY_KEYS}))
    _check(len(identities) == 1, 'baseline model/runtime identities differ')
    return base['complete']


def operator_factory(model, cell_id, config, *, diagnostic_mode=None):
    """Only immutable exported maps and the registered label-free XM scalar."""
    from .operators import make_graft_wrapper, make_intact_wrapper
    from .provenance import verify_export
    _check(diagnostic_mode in (None, 'disabled', 'zero_gate'), 'unknown qualification diagnostic mode')
    cell = config['cells'][cell_id]
    gate = 0. if diagnostic_mode == 'zero_gate' else cell['gate']
    enabled = diagnostic_mode != 'disabled'
    if cell['arm'] == 'X0':
        return make_intact_wrapper(model)
    if cell['arm'] == 'XM':
        from .compute_control import CrossScaleComputeControl
        from .control_calibration import verify_calibration_artifact
        control = config['matched_control']
        artifact = verify_calibration_artifact(REPO / control['artifact_path'], control['artifact_sha256'], require_committed=True)
        _check(artifact['alpha'] == control['alpha'] and artifact['control_state_sha256'] == control['state_sha256'], 'XM scalar/state differs from calibration')
        return CrossScaleComputeControl(model, alpha=control['alpha'], gate=gate, enabled=enabled,
                    expected_state_sha256=control['state_sha256'], layer=config['selected_pair']['supa_layer'],
                    emergency_bounds=tuple(config['hidden_rms_ratio_bounds']))
    import torch
    from cross_scale.src.donor_layers import donor_identity, load_donor_layer
    export = config['upstream_export']
    verify_export(REPO / export['manifest_path'], export['manifest_sha256'], require_committed=True)
    tensor_path = require_within(REPO / export['tensor_path'], STAGE)
    _check(sha256_file(tensor_path) == export['tensor_sha256'], 'exported bridges changed')
    bridges = torch.load(tensor_path, map_location='cpu', weights_only=True)
    _check(set(bridges) == {'XG', 'XS', 'XR'}, 'exported arm set mismatch')
    donor_config = {**config, 'donor_layers': [config['selected_pair']['donor_layer']]}
    path, identity = donor_identity(donor_config)
    donor, rope, architecture = load_donor_layer(donor_config, config['selected_pair']['donor_layer'], path)
    del rope
    wrapper = make_graft_wrapper(model, donor, architecture, bridges[cell['arm']], arm=cell['arm'],
                insertion_layer=config['selected_pair']['supa_layer'], donor_index=config['selected_pair']['donor_layer'],
                gate=gate, enabled=enabled, emergency_bounds=tuple(config['hidden_rms_ratio_bounds']))
    wrapper.donor_identity = identity
    return wrapper


def run_shard(config_path, cell_id, shard_index, *, shard_count=None, attempt=0, model_path=None):
    from .qualification import verify_cell_qualification
    from .provenance import stage_sources
    from shared.model_loading import load_supa
    from shared.integrity import assert_unchanged, state_hash, frozen_inventory
    from shared.generation import evaluate_case
    from shared.runtime import synchronize, cuda_memory
    from shared.timing import machine_lease
    from armr.hooks import frozen_inference
    import psutil
    import torch
    config_path, config, registration, cases = _context(config_path)
    _check(shard_count in (None, config['shard_count']), 'unregistered shard count')
    contract, selected = _expected_contract(_binding(config_path, config, registration), cell_id, config, cases, shard_index, attempt)
    state = inventory(config_path, config, registration, cases)
    if state[cell_id]['failed']:
        raise CellRejected('cell has terminal generation failure: ' + cell_id)
    _check(shard_index not in state[cell_id]['complete'], 'duplicate complete shard refused')
    old = state[cell_id]['attempts'].get(shard_index, [])
    _check(attempt == (max((e['manifest']['attempt'] for e in old), default=-1) + 1), 'attempts must resume consecutively')
    baseline = None if cell_id == 'X0' else _baseline(state, config, cases)[shard_index]
    if baseline is not None:
        base_qualification = verify_cell_qualification(config, registration, 'X0')
        _check(base_qualification.get('status') == 'PASS', 'X0 qualification is not PASS')
        _check(all(entry['manifest'].get('runtime_qualification') == base_qualification
                   for entry in state['X0']['complete'].values()), 'X0 qualification binding differs')
    qualification = verify_cell_qualification(config, registration, cell_id)
    if qualification.get('status') == 'FAILED':
        raise CellRejected('cell has terminal qualification failure: ' + cell_id)
    _check(qualification.get('status') == 'PASS', 'cell qualification is not PASS')
    directory = _root(registration) / cell_id / f'shard_{shard_index:03d}' / f'attempt_{attempt:03d}'
    contract.update(runtime_qualification_receipt_sha256=qualification['receipt_sha256'],
                    runtime_qualification_directory=qualification['directory'],
                    runtime_qualification=qualification,
                    baseline_manifest_path=None if baseline is None else baseline['manifest_path'],
                    baseline_manifest_sha256=None if baseline is None else baseline['manifest_sha256'])
    manifest = {**contract, **git_identity(), 'status': 'STARTED', 'start_utc': utc_now(),
                'pid': os.getpid(), 'process_create_time': psutil.Process().create_time(),
                'run_id': f"{registration['registration_sha256']}:{cell_id}:{shard_index}:{attempt}"}
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'started.json', manifest)
    print(f'RUN_DIR={directory}', flush=True)
    start = time.perf_counter()
    model = tokenizer = wrapper = extra = None
    identity = None
    results = []
    failure = None
    lease, lease_acquired = machine_lease(), False
    try:
        lease.__enter__()
        lease_acquired = True
        model, tokenizer, identity = load_supa(config, model_path)
        manifest.update(identity)
        write_json(directory / 'model_identity.json', identity)
        if baseline:
            _check(sha256_file(Path(baseline['manifest_path'])) == baseline['manifest_sha256'], 'baseline changed before inference')
            _check(all(identity.get(k) == baseline['manifest'].get(k) for k in IDENTITY_KEYS), 'candidate model/runtime differs from X0')
        wrapper = operator_factory(model, cell_id, config)
        extra = getattr(wrapper, 'transform', getattr(wrapper, 'expert', None))
        before = None if extra is None else state_hash(extra)
        if extra is not None:
            manifest['operator_inventory_before'] = frozen_inventory(extra)
        manifest['operator'] = {k: getattr(wrapper, k, 0) for k in
                               ('added_parameter_count', 'added_active_parameters', 'extra_matmul_flops_per_token')}
        manifest['donor_identity_before'] = getattr(wrapper, 'donor_identity', None)
        (directory / 'cases').mkdir()
        (directory / 'logits').mkdir()
        with wrapper, frozen_inference(model):
            for case in selected:
                name = canonical_sha256(case['example_id'])
                base = None
                if baseline:
                    path = Path(baseline['directory']) / 'logits' / (name + '.pt')
                    _check(sha256_file(path) == baseline['manifest']['artifacts_sha256']['logits/' + name + '.pt'], 'baseline logits changed')
                    base = torch.load(path, map_location='cpu', weights_only=True)
                row, first = evaluate_case(model, tokenizer, case, config, wrapper, base)
                row.update(cell_id=cell_id, arm=config['cells'][cell_id]['arm'], gate=config['cells'][cell_id]['gate'],
                           run_id=manifest['run_id'], **manifest['operator'])
                write_json(directory / 'cases' / (name + '.json'), row)
                with (directory / 'logits' / (name + '.pt')).open('xb') as handle:
                    torch.save(first.detach().cpu(), handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                results.append(row)
                del first, base
                print(f"{cell_id} {case['example_id']} completed={len(results)}/{len(selected)}", flush=True)
        if extra is not None:
            manifest['operator_inventory_after'] = frozen_inventory(extra)
            manifest['operator_state_sha256_before'] = before
            manifest['operator_state_sha256_after'] = state_hash(extra)
            _check(before == manifest['operator_state_sha256_after'], 'frozen operator changed')
        manifest['operator_integrity_receipt'] = getattr(wrapper, 'integrity_receipt', None)
        if manifest['donor_identity_before'] is not None:
            from cross_scale.src.donor_layers import donor_identity
            _, manifest['donor_identity_after'] = donor_identity(config)
            _check(manifest['donor_identity_after'] == manifest['donor_identity_before'], 'donor checkpoint changed')
        manifest.update(assert_unchanged(model, identity['state_sha256_before'], Path(identity['model_path']), identity['checkpoint_files_before']))
        _check(stage_sources() == registration['sources'], 'source changed during execution')
        _context(config_path)  # Recheck data/export/control/registration after inference.
        manifest['operator_integrity_passed'] = True
        write_json(directory / 'results.json', results)
        manifest.update(status='COMPLETE', completed_examples=len(results))
    except BaseException as exc:
        failure = exc
        manifest.update(status='FAILED', completed_examples=len(results), model_loaded=model is not None,
                        failure_class='resource_admission' if model is None and not results and (
                            isinstance(exc, MemoryError) or isinstance(exc, FileExistsError) and not lease_acquired) else 'hard_failure',
                        failure_receipt={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        write_json(directory / 'failure.json', manifest['failure_receipt'])
        if model is not None and identity is not None:
            try:
                manifest['failure_integrity'] = assert_unchanged(model, identity['state_sha256_before'], Path(identity['model_path']), identity['checkpoint_files_before'])
            except BaseException as check_error:
                manifest['failure_integrity'] = {'canonical_unchanged': False, 'error': str(check_error)}
        manifest['operator_integrity_receipt'] = getattr(wrapper, 'integrity_receipt', None)
    finally:
        try:
            synchronize(config['device'])
            manifest.update(cuda_memory(config))
        except BaseException as error:
            manifest['final_runtime_error'] = str(error)
            if manifest['status'] == 'COMPLETE':
                failure = error
                manifest.update(status='FAILED', failure_class='hard_failure', model_loaded=model is not None,
                                failure_receipt={'type': type(error).__name__, 'message': str(error)})
                write_json(directory / 'failure.json', manifest['failure_receipt'])
        memory = psutil.Process().memory_info()
        manifest.update(end_utc=utc_now(), wall_seconds=time.perf_counter()-start,
                        peak_memory_bytes=getattr(memory, 'peak_wset', memory.rss),
                        started_sha256=sha256_file(directory / 'started.json'), artifacts_sha256=_artifacts(directory))
        try:
            write_json(directory / 'run_manifest.json', manifest)
        finally:
            model = tokenizer = wrapper = extra = None
            gc.collect()
            try:
                if config['device'].startswith('cuda'):
                    torch.cuda.empty_cache()
            finally:
                if lease_acquired:
                    lease.__exit__(None, None, None)
    if failure is not None:
        raise failure
    return directory


def abandon_shard(config_path, run_directory, reason):
    """An absent original process permits a new receipt, never a replacement."""
    import psutil
    config_path, config, registration, cases = _context(config_path)
    directory = require_within(run_directory, _root(registration))
    _check(isinstance(reason, str) and bool(reason.strip()), 'explicit abandonment reason required')
    _check(not (directory / 'run_manifest.json').exists() and not (directory / 'abandonment.json').exists(), 'attempt is already terminal')
    started = _read(directory / 'started.json')
    contract, _ = _expected_contract(_binding(config_path, config, registration), started['cell_id'], config, cases, started['shard_index'], started['attempt'])
    _check(all(started.get(k) == v for k, v in contract.items()), 'incompatible interrupted attempt')
    expected = _root(registration) / started['cell_id'] / f"shard_{started['shard_index']:03d}" / f"attempt_{started['attempt']:03d}"
    _check(directory == expected.resolve(), 'attempt path identity mismatch')
    pid = started.get('pid')
    _check(type(pid) is int and pid > 0, 'missing original PID')
    # PID reuse is deliberately conservative: any extant PID prevents abandonment.
    _check(not psutil.pid_exists(pid), 'original PID is still live or reused; abandonment refused')
    result = {**started, 'status': 'ABANDONED', 'original_process_absent': True,
              'absence_checked_utc': utc_now(), 'end_utc': utc_now(), 'reason': reason.strip(),
              'post_state_integrity': 'unavailable; interrupted process did not issue final receipt',
              'started_sha256': sha256_file(directory / 'started.json'), 'artifacts_sha256': _artifacts(directory)}
    write_json(directory / 'abandonment.json', result)
    return directory / 'abandonment.json'


def run_campaign(config_path):
    """Sequential fresh-process shards; resource failures pause, hard cells close."""
    import subprocess
    import sys
    from .qualification import verify_cell_qualification
    config_path, config, registration, cases = _context(config_path)
    for cell_id in config['cells']:
        state = inventory(config_path, config, registration, cases)
        if state[cell_id]['failed']:
            if cell_id == 'X0':
                raise CellRejected('X0 failed; screen is closed')
            continue
        qualification = verify_cell_qualification(config, registration, cell_id)
        if qualification.get('status') == 'FAILED':
            if cell_id == 'X0':
                raise CellRejected('X0 qualification failed; screen is closed')
            continue
        _check(qualification.get('status') == 'PASS', 'cell qualification is not PASS')
        for index in range(config['shard_count']):
            state = inventory(config_path, config, registration, cases)
            if state[cell_id]['failed']:
                break
            if index in state[cell_id]['complete']:
                continue
            attempt = max((e['manifest']['attempt'] for e in state[cell_id]['attempts'].get(index, [])), default=-1) + 1
            command = [sys.executable, str(STAGE / 'scripts/run_shard.py'), '--config', str(config_path),
                       '--cell', cell_id, '--shard-index', str(index), '--attempt', str(attempt)]
            completed = subprocess.run(command, check=False)
            state = inventory(config_path, config, registration, cases)
            if completed.returncode and not state[cell_id]['failed']:
                raise RuntimeError('shard stopped for resource admission or infrastructure; explicit resume required')
            if state[cell_id]['failed']:
                if cell_id == 'X0':
                    raise CellRejected('X0 failed; screen is closed')
                break
    return audit_screen(config_path)


def _audit_logits(entry, row, baseline, config):
    from armr.metrics import logit_metrics
    import torch
    name = canonical_sha256(row['example_id']) + '.pt'
    value = torch.load(Path(entry['directory']) / 'logits' / name, map_location='cpu', weights_only=True)
    base = torch.load(Path(baseline['directory']) / 'logits' / name, map_location='cpu', weights_only=True)
    _check(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and value.ndim == 1 and value.shape == base.shape,
           'invalid saved first-logit tensor')
    metrics = logit_metrics(base, value, config['top_k'], row.get('gold_token'))
    _check(all(row.get(k) == v for k, v in metrics.items()), 'saved first-logit metrics do not reproduce')


def _audit_fragments(entry, config, cases, baseline):
    """Rescore retained incomplete artifacts, never create a completed denominator."""
    from shared.scoring import score_response
    from armr.metrics import output_flags
    import torch
    directory, manifest = Path(entry['directory']), entry['manifest']
    expected = {canonical_sha256(case['example_id']): case for case in
                shard_cases(cases, manifest['shard_index'], config['shard_count'])}
    rows = []
    for relative in manifest['artifacts_sha256']:
        if not relative.startswith('cases/'):
            continue
        path = Path(relative)
        _check(path.parent.as_posix() == 'cases' and path.suffix == '.json' and path.stem in expected,
               'unknown partial case artifact')
        row, case = _read(directory / relative), expected[path.stem]
        _finite(row)
        _check(all(row.get(k) == case.get(k) for k in ROW_METADATA), 'partial case identity differs from frozen bank')
        _check(row.get('cell_id') == manifest['cell_id'] and row.get('arm') == manifest['arm']
               and row.get('gate') == manifest['cell']['gate'] and row.get('run_id') == manifest['run_id'],
               'partial treatment/run identity differs')
        score = score_response(row['response'], case['scorer'])
        _check(row.get('score') == float(score['score']) and row.get('correct') == bool(score['score'] == 1),
               'partial case independent rescoring mismatch')
        flags = output_flags(row['response'], row['token_ids'])
        _check(all(row.get(k) == value for k, value in flags.items()), 'partial case validity flags mismatch')
        if 'logits/' + path.stem + '.pt' in manifest['artifacts_sha256']:
            _audit_logits(entry, row, baseline, config)
        rows.append(row)
    # A process can die after a tensor write and before subsequent bookkeeping.
    # Such orphan tensors are hash-/shape-/finiteness-audited, never given scores.
    for relative in manifest['artifacts_sha256']:
        if not relative.startswith('logits/'):
            continue
        path = Path(relative)
        _check(path.parent.as_posix() == 'logits' and path.suffix == '.pt' and path.stem in expected,
               'unknown partial logit artifact')
        tensor = torch.load(directory / relative, map_location='cpu', weights_only=True)
        _check(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
               and tensor.ndim == 1 and bool(torch.isfinite(tensor).all()), 'invalid partial logit tensor')
    return rows


def audit_screen(config_path):
    from .qualification import verify_cell_qualification
    from .provenance import verify_export
    from .analysis import analyze_screen
    import torch
    config_path, config, registration, cases = _context(config_path)
    export = config['upstream_export']
    verify_export(REPO / export['manifest_path'], export['manifest_sha256'],
                  require_committed=True, verify_upstream_now=True)
    torch.set_num_threads(config['threads'])
    state = inventory(config_path, config, registration, cases)
    baseline = _baseline(state, config, cases)
    by_cell, failed_cells, raw_hashes = {}, {}, {}
    identity = canonical_sha256({k: baseline[0]['manifest'].get(k) for k in IDENTITY_KEYS})
    for cell_id, cell_state in state.items():
        qualification = verify_cell_qualification(config, registration, cell_id)
        if qualification.get('status') == 'FAILED':
            _check(not cell_state['attempts'], 'generation followed terminal qualification failure')
            failed_cells[cell_id] = qualification
            continue
        _check(qualification.get('status') == 'PASS', 'cell qualification is not PASS')
        for attempts in cell_state['attempts'].values():
            for entry in attempts:
                raw_hashes[entry['manifest_path']] = entry['manifest_sha256']
        if not cell_state['failed']:
            _check(set(cell_state['complete']) == set(range(config['shard_count'])), 'no partial cell aggregate allowed')
        rows = []
        for index, entry in sorted(cell_state['complete'].items()):
            manifest = entry['manifest']
            _check(canonical_sha256({k: manifest.get(k) for k in IDENTITY_KEYS}) == identity, 'paired model/runtime identity mismatch')
            _check(manifest.get('runtime_qualification') == qualification, 'qualification binding mismatch')
            if cell_id != 'X0':
                _check(manifest.get('baseline_manifest_sha256') == baseline[index]['manifest_sha256']
                       and manifest.get('baseline_manifest_path') == baseline[index]['manifest_path'], 'paired baseline binding mismatch')
            selected = _read_complete(entry, config, cases, cell_id)
            for row in selected:
                _audit_logits(entry, row, baseline[index], config)
                rows.append(row)
        fragments = []
        for index, attempts in cell_state['attempts'].items():
            for entry in attempts:
                if entry['manifest']['status'] != 'COMPLETE':
                    _check(entry['manifest'].get('runtime_qualification') == qualification, 'partial qualification binding mismatch')
                    fragments.extend(_audit_fragments(entry, config, cases, baseline[index]))
        if cell_state['failed']:
            failed_cells[cell_id] = {**cell_state['failed'], 'retained_case_artifacts': len(rows) + len(fragments),
                                    'fragment_count_is_not_an_accuracy_denominator': True}
            by_cell[cell_id] = rows + fragments  # Pure analysis explicitly excludes failed cells.
            continue
        _check(len(rows) == len({r['example_id'] for r in rows}) == 128, 'duplicate or incomplete case denominator')
        by_cell[cell_id] = rows
    _check('X0' in by_cell and 'X0' not in failed_cells, 'complete X0 baseline required')
    report = analyze_screen(cases=cases, by_cell=by_cell, failed_cells=failed_cells, config=config)
    # Re-read complete inventories immediately before publishing a decision.
    _check(inventory(config_path, config, registration, cases) == state, 'evidence inventory changed during audit')
    _context(config_path)
    output_root = AUDITS / registration['registration_sha256']
    existing = list(output_root.glob('attempt_*')) if output_root.exists() else []
    attempt = max((int(p.name[8:]) for p in existing if re.fullmatch('attempt_[0-9]{3,}', p.name)), default=-1) + 1
    output = output_root / f'attempt_{attempt:03d}'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'raw_manifest_hashes.json', raw_hashes)
    write_json(output / 'report.json', report)
    selection = {'status': report['status'], 'selected': report['selected'],
                 'registration_sha256': registration['registration_sha256'],
                 'config_sha256': canonical_sha256(config),
                 'confirmatory_generation_licensed': False}
    write_json(output / 'selection.json', selection)
    write_json(output / 'audit_receipt.json', {**_binding(config_path, config, registration), **git_identity(),
                'status': 'COMPLETE', 'created_utc': utc_now(), 'analysis_role': 'candidate_selection',
                'complete_cells': sorted(set(by_cell) - set(failed_cells)), 'failed_cells': failed_cells, 'n_per_complete_cell': 128,
                'independent_rescoring': True, 'first_logit_metrics_recomputed': True,
                'confirmation_accessed': False, 'artifacts_sha256': _artifacts(output)})
    return output
