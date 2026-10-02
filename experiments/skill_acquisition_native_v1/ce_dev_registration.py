"""Concrete finite five-read DEV drafts; all grants and adoption remain false.

Profiles are source-derived byte floors and explicit abort ceilings, never
measured timings. Existing BootstrapLedger/NativeResourceGate accounting is
used by the future entry. No new accounting authority or model work is issued.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

from . import ce_dev_runtime as runtime, ce_dev_scope as scope
from . import ce_native_services as services, ce_registration as common
from . import ce_qualification_artifacts as raw
from . import ce_completed_training as completed, ce_training_supervisor as training_supervisor
from . import ce_service_bootstrap_verifier as service_replay, ce_qualified_receipt as finalizer_replay

SCHEMA = 'CE8-DEV-REGISTRATION-DRAFT-1'
BUDGET = 'CE8-DEV-EVALUATION-BUDGET-PROPOSAL-1'
ENTRY = 'CE8-DEV-ENTRY-SPEC-1'
CONFIG = 'CE8-DEV-SOURCE-CONFIGURATION-1'
ENTRY_KINDS = ('dev_entry_source', 'dev_entry_metadata', 'dev_entry_lease', 'dev_entry_lease_release', 'dev_entry_lease_assert', 'dev_entry_verify')
WORKER_KINDS = ('worker_os_binding', 'worker_source_verification', 'worker_runtime_binding', 'worker_handle_close', 'worker_handle_duplicate')
require, canonical, digest = raw.require, raw.canonical, raw.digest


def completed_training_reader_bounds():
    """Four bounded journals, sixteen references and the saved donor body."""
    journals = [finalizer_replay.MAX_JOURNAL, training_supervisor.MAX_JOURNAL,
                service_replay.MAX_JOURNAL, service_replay.MAX_JOURNAL]
    donor = 8 * 1024 ** 3
    metadata = (len(completed.REFS) + 8) * raw.MAX_JSON
    floors = {'operation_attempts_upper': 256,
        'io_read_bytes_upper': donor + sum(journals) + metadata + 64 * raw.CHUNK,
        'identity_hash_attempts_upper': 2 * finalizer_replay.MAX_EVENTS + 2 * training_supervisor.MAX_EVENTS
            + 4 * service_replay.MAX_EVENTS + 20000,
        'identity_hash_bytes_upper': 16 * sum(journals) + 2 * donor + 128 * metadata,
        'reducer_work_bytes_upper': 64 * sum(journals) + 32 * donor + 128 * metadata}
    bounds = {key: 1 << max(0, (value - 1).bit_length()) for key, value in floors.items()}
    return bounds, {'reference_count': len(completed.REFS), 'journal_byte_ceilings': journals,
        'donor_byte_ceiling': donor, 'derived_floor': floors,
        'consumer_preflights': ['completed_training_pair', 'completed_training_CE_journal', 'completed_training_service'],
        'source_callback_ceiling': {'qualification_reducer_overhead': 128, 'qualification_artifact_readback': 32},
        'unit_prices_measured': False}


def operation_census(model_name):
    require(model_name in ('Supa', 'warm_anchor', 'CE375_donor'), 'fixed DEV model role')
    donor = model_name == 'CE375_donor'
    exact = dict.fromkeys(runtime.KINDS, 1)
    exact.update(dev_decode=1950, dev_journal=1953, dev_fingerprint=2, dev_delta_rebuild=int(donor))
    # All spans recheck cached source identity, without recursive OS queries.
    # Paid live worker/lock refresh belongs to each explicit gate wrapper.
    gate_count = 1958 + int(donor)
    sample_cap = 4096
    return {'runtime_starts': exact, 'runtime_gates': gate_count, 'resource_samples_upper': sample_cap,
        'spare_resource_samples': sample_cap - gate_count, 'cached_permit_rechecks_upper': sum(exact.values()) + 2,
        'source_verifier_calls_upper': sum(exact.values()) + 2 + gate_count + 32,
        'source_verifier_startup_constructor_spare': 32,
        'read_rows': 1950, 'generated_tokens_upper': 1950 * 128,
        'live_refresh_contract': {'worker_os_binding_per_gate': 0, 'dev_entry_lease_assert_per_permit_recheck': 1,
            'cached_worker_identity_os_queries': 0},
        'unit_prices_measured': False, 'model_or_os_execution': False}


def _compose(profiles, calls, setup, suffix, physical):
    counters = dict.fromkeys(services.COUNTERS, 0)
    for kind, count in calls.items():
        require(type(count) is int and count >= 0 and kind in profiles, 'explicit finite DEV callback count')
        price = services._profile(profiles[kind])
        for key, field in {'read_bytes': 'read_bytes_upper', 'write_bytes': 'write_bytes_upper',
            'hash_attempts': 'identity_hash_attempts_upper', 'hash_bytes': 'identity_hash_bytes_upper',
            'wall_seconds': 'wall_s_upper'}.items(): counters[key] += count * price[field]
        counters['operation_attempts'] += count
    for price in (setup, suffix):
        services._profile(price)
        for key, field in {'read_bytes': 'read_bytes_upper', 'write_bytes': 'write_bytes_upper',
            'hash_attempts': 'identity_hash_attempts_upper', 'hash_bytes': 'identity_hash_bytes_upper',
            'wall_seconds': 'wall_s_upper'}.items(): counters[key] += price[field]
    counters['wall_seconds'] += physical
    # Fixed simple metadata operations have no tensor arrays in event fields.
    # Reserve32KiB for each phase+wall event, plus source header/seal material.
    metadata = (4 * sum(calls.values()) + 32) * 32768 + 64 * raw.CHUNK
    margins = {'metadata_bytes': metadata, 'write_bytes': metadata, 'hash_bytes': metadata,
               'hash_attempts': 4 * sum(calls.values()) + 32, 'read_bytes': 64 * raw.CHUNK,
               'operation_attempts': 32, 'wall_seconds': 0}
    for key, amount in margins.items(): counters[key] += amount
    caps = {key: 1 << max(0, int(value - 1).bit_length()) for key, value in counters.items()}
    return caps, {'permanent_profile_totals_with_actual_wall_and_metadata': counters, 'self_record_margins': margins}


def propose_budget(proposal, documents, current_q_documents, *, output_root):
    require(documents['DEV_RUN_PLAN.json']['total_greedy_completions'] == 9750
            and documents['DEV_RUN_PLAN.json']['evaluation_authorized'] is False,
            'fixed full9750 unapproved DEV plan')
    root = Path(output_root); require(root.is_absolute(), 'absolute future DEV owner')
    qscope = current_q_documents['SCOPE.json']; service = current_q_documents['SERVICE_SPEC.json']
    native = qscope['native_backend']; qprofiles = qscope['qualifier_accounting']['operation_profiles']
    source = current_q_documents['SOURCE_SNAPSHOT.json']
    source_bytes = sum(ref['bytes'] for ref in source['source_files'].values())
    bank_bytes = sum(documents['DEV_EVALUATION_SCOPE.json']['metadata_refs'][name]['bytes'] for name in
                     ('canonical_dev', 'canonical_guard_dev'))
    asset = raw._json(native['asset_receipt_utf8']); full_fp32 = 2 * asset['model_header']['data_bytes']
    warm_bytes = native['warm_delta_ref']['bytes']; selected_fp32 = 4 * proposal['inventory']['numel']
    physical = 19200
    profiles = {
        'dev_module_import': common.profile(source_bytes + 64 * raw.CHUNK, 0, len(source['source_files']) + 64, source_bytes + 64 * raw.CHUNK, 60),
        'dev_worker': common.profile(wall=30), 'dev_source_read': common.profile(bank_bytes + 2, 0, 64, bank_bytes + 64 * raw.CHUNK, 60),
        'dev_tokenizer': dict(qprofiles['tokenizer_load']), 'dev_configure': common.profile(wall=60),
        'dev_model_load': dict(qprofiles['model_load']),
        'dev_delta_rebuild': common.profile(8 * selected_fp32 + 64 * raw.CHUNK, 0, 512, 8 * selected_fp32 + 64 * raw.CHUNK, 600),
        'dev_fingerprint': common.profile(0, 0, asset['model_header']['tensors'] + 64, full_fp32 + 64 * raw.CHUNK, 60),
        'dev_decode': common.profile(0, 0, 4, 4 * raw.CHUNK, 30),
        'dev_journal': common.profile(64 * raw.CHUNK + 1, 128 * 1024, 4, 128 * raw.CHUNK, 5),
        'dev_receipt': common.profile(raw.MAX_JSON + 1, raw.MAX_JSON, 8, 4 * raw.MAX_JSON, 60),
        'dev_cleanup': common.profile(wall=physical)}
    # The existing cold loader verifies the model/anchor repeatedly. Supa uses
    # the same conservative bound as warm, without inventing lower timings.
    require(profiles['dev_model_load']['read_bytes_upper'] >= native['asset_receipt_ref']['bytes']
            and profiles['dev_model_load']['identity_hash_bytes_upper'] >= full_fp32 + warm_bytes,
            'known cold loader/model/warm byte floors')
    bootstrap_profiles = {kind: copy.deepcopy(service['bootstrap']['operation_profiles'][kind]) for kind in WORKER_KINDS}
    bootstrap_profiles.update(profiles)
    training_bounds, training_contract = completed_training_reader_bounds()
    bootstrap_profiles.update({
        'dev_entry_source': common.profile(128 * raw.CHUNK, 0, 20000, 256 * raw.CHUNK, 120),
        'dev_entry_metadata': common.profile(64 * raw.CHUNK, 16 * raw.CHUNK, 512, 256 * raw.CHUNK, 120),
        'dev_entry_lease': common.profile(64 * raw.CHUNK, 4096, 64, 64 * raw.CHUNK, 180),
        'dev_entry_lease_assert': common.profile(4097, 0, 1, 4097, 5),
        'dev_entry_verify': common.profile(0, 0, 64, 64 * raw.MAX_JSON, 5),
        'qualification_reducer_overhead': common.profile(0, 0, 2 * finalizer_replay.MAX_EVENTS, 32 * 1024 ** 3, 120),
        'qualification_artifact_readback': common.profile(8 * 1024 ** 3 + 1, 0, 1, 8 * 1024 ** 3 + 1, 120),
        'dev_entry_lease_release': common.profile(4 * 4097, 4096, 4, 4 * 4096, physical),
        'resource_probe': copy.deepcopy(service['bootstrap']['operation_profiles']['resource_probe']),
        'resource_cuda_probe': common.profile(wall=5)})
    setup = common.profile(64 * raw.CHUNK, 64 * raw.CHUNK, 10000, 256 * raw.CHUNK, 120)
    suffix = common.profile(128 * raw.CHUNK, 16 * raw.CHUNK, 64, 256 * raw.CHUNK, 120)
    constructor = {'read_bytes': 128 * raw.CHUNK, 'write_bytes': 128 * raw.CHUNK,
        'hash_attempts': 20000, 'hash_bytes': 512 * raw.CHUNK, 'metadata_bytes': 128 * raw.CHUNK,
        'operation_attempts': 32, 'wall_seconds': 120}
    result = {'schema': BUDGET, 'approved': False, 'evaluation_authorized': False,
        'production_training_authorized': False, 'dispatch_authorized': False, 'unit_prices_measured': False,
        'basis': 'EXPLICIT_UNMEASURED_DEV_ABORT_CEILINGS', 'reads': {},
        'source_byte_facts': {'source_bytes': source_bytes, 'DEV_and_guard_canonical_bytes': bank_bytes,
            'full_model_fp32_bytes': full_fp32, 'warm_saved_bytes': warm_bytes, 'selected_donor_fp32_bytes': selected_fp32},
        'missing_evidence': ['actual completed375 deployed donor and runtime/model bindings',
            'source-frozen DEV entry live-refresh ABI review', 'fresh-process resource fit and complete OS IO unit measurements',
            'explicit DEV evaluation authorization'],
        'total_completions': 9750, 'total_generated_tokens_upper': 1248000,
        'completed_training_consumer': training_contract}
    for run_name, model_name, _ in runtime.RUNS:
        census = operation_census(model_name); calls = copy.deepcopy(census['runtime_starts']); cap = census['resource_samples_upper']
        calls.update(resource_probe=cap, resource_cuda_probe=cap, worker_os_binding=64,
            dev_entry_lease=1, dev_entry_lease_assert=census['source_verifier_calls_upper'],
            dev_entry_verify=census['source_verifier_calls_upper'],
            dev_entry_lease_release=1, dev_entry_source=8, dev_entry_metadata=16,
            worker_source_verification=2 * len(source['source_files']) + 32, worker_handle_close=16, worker_handle_duplicate=4,
            worker_runtime_binding=8,
            **training_contract['source_callback_ceiling'])
        caps, composition = _compose(bootstrap_profiles, calls, setup, suffix, physical)
        resource = copy.deepcopy(service['resource']); resource['sample_attempts_upper'] = cap
        resource['resource_probe']['output_root'] = str(root)
        for kind in runtime.KINDS: resource['host_transient_bytes'].setdefault(kind, 64 * raw.CHUNK)
        bootstrap = {'owned_directory': str(root / run_name / 'bootstrap'), 'operation_profiles': copy.deepcopy(bootstrap_profiles),
            'setup_bounds': setup, 'finalization_bounds': suffix, 'caps': caps,
            'operation_profile_rule': services.INPUT_RESERVATION_RULE, 'explicit_reservation_kinds': ['input_io'],
            'physical_wall_seconds_upper': physical}
        result['reads'][run_name] = {'model_name': model_name, 'census': census, 'declared_callback_counts': calls,
            'composition': composition, 'policy': {'schema': runtime.POLICY, 'output_directory': str(root),
                'physical_wall_seconds_upper': physical, 'journal_bytes_upper': 256 * raw.CHUNK,
                'receipt_bytes_upper': raw.MAX_JSON, 'operation_profiles': copy.deepcopy(profiles)},
            'entry_spec': {'schema': ENTRY, 'bootstrap': bootstrap, 'constructor_bounds': copy.deepcopy(constructor),
                'training_reader_bounds': copy.deepcopy(training_bounds),
                'resource': resource, 'worker': None,
                'lease': {'helper_ref': copy.deepcopy(qscope['checkpoint_lease']['helper_ref']),
                    'lock_path': qscope['checkpoint_lease']['lock_path'], 'label': 'comp-rs-001-dev-' + run_name.lower().replace('_', '-'),
                    'timeout_seconds': 180, 'lock_bytes_upper': 4096,
                    'operation_bounds': {'acquire': copy.deepcopy(bootstrap_profiles['dev_entry_lease']),
                        'release': copy.deepcopy(bootstrap_profiles['dev_entry_lease_release']),
                        'assert': copy.deepcopy(bootstrap_profiles['dev_entry_lease_assert'])}}, 'rpc_spec': None}}
    result['aggregate_separate_owner_caps'] = {key: sum(read['entry_spec']['bootstrap']['caps'][key] for read in result['reads'].values())
                                               for key in services.COUNTERS}
    return result


def build_documents(scope_documents, budget, *, source):
    require(type(budget) is dict and budget.get('schema') == BUDGET and budget.get('approved') is False
            and budget.get('unit_prices_measured') is False and budget.get('total_completions') == 9750
            and set(budget.get('reads', {})) == {name for name, _, _ in runtime.RUNS}, 'unapproved exact five-read DEV budget')
    scope_value = scope_documents['DEV_EVALUATION_SCOPE.json']; plan = scope_documents['DEV_RUN_PLAN.json']
    require(scope_value['evaluation_authorized'] is False and plan['evaluation_authorized'] is False
            and scope_value['greedy'] == runtime.GREEDY and plan['total_greedy_completions'] == 9750,
            'fixed DEV scope/greedy plan without grant')
    contexts = {}
    for read_id, model_name, _ in runtime.RUNS:
        proposal = budget['reads'][read_id]
        require(proposal['model_name'] == model_name and proposal['census'] == operation_census(model_name)
                and set(proposal['policy']['operation_profiles']) == runtime.KINDS
                and proposal['policy']['operation_profiles']['dev_cleanup']['wall_s_upper'] >= proposal['policy']['physical_wall_seconds_upper'],
                'complete fixed read operation census and prepaid lifetime cleanup')
        calls = proposal['declared_callback_counts']; census = proposal['census']; entry = proposal['entry_spec']
        require(all(calls.get(kind) == count for kind, count in census['runtime_starts'].items())
                and calls.get('dev_entry_lease_assert') >= census['source_verifier_calls_upper']
                and calls.get('dev_entry_verify') >= census['source_verifier_calls_upper']
                and calls.get('resource_probe') == calls.get('resource_cuda_probe') == entry['resource']['sample_attempts_upper']
                and type(entry['resource']['sample_attempts_upper']) is int and 0 < entry['resource']['sample_attempts_upper'] <= 100000,
                'all exact runtime/lease/resource callbacks permanently reserved')
        require(all(proposal['policy']['operation_profiles'][kind] == entry['bootstrap']['operation_profiles'][kind]
                for kind in runtime.KINDS), 'runtime prices equal original bootstrap source profiles')
        ctor = entry['constructor_bounds']
        require(type(ctor) is dict and set(ctor) == set(services.COUNTERS)
                and all(type(number) is int and number > 0 for number in ctor.values()), 'complete finite seven-counter constructor prepayment')
        reader_bounds, contract = completed_training_reader_bounds()
        require(entry.get('training_reader_bounds') == reader_bounds
                and all(calls.get(kind, 0) >= count for kind, count in contract['source_callback_ceiling'].items()),
                'full donor/four-journal consumer independently prepaid')
        require(entry['bootstrap']['operation_profiles']['dev_entry_lease'] == entry['lease']['operation_bounds']['acquire']
                and entry['bootstrap']['operation_profiles']['dev_entry_lease_release'] == entry['lease']['operation_bounds']['release']
                and entry['bootstrap']['operation_profiles']['dev_entry_lease_assert'] == entry['lease']['operation_bounds']['assert'],
                'actual lease source operations match prepaid profiles')
        caps, _ = _compose(proposal['entry_spec']['bootstrap']['operation_profiles'], proposal['declared_callback_counts'],
            proposal['entry_spec']['bootstrap']['setup_bounds'], proposal['entry_spec']['bootstrap']['finalization_bounds'],
            proposal['policy']['physical_wall_seconds_upper'])
        require(proposal['entry_spec']['bootstrap']['caps'] == caps, 'no discounted DEV composed owner caps')
        contexts[read_id] = {'schema': runtime.CONTEXT, 'evaluation_authorized': False, 'production_training_authorized': False,
            'read_id': read_id, 'request_sha256': None, 'scope_sha256': digest(scope_value), 'scope': copy.deepcopy(scope_value),
            'plan': copy.deepcopy(plan), 'policy': copy.deepcopy(proposal['policy']), 'runtime': None, 'model': None,
            'source': copy.deepcopy(source), 'native_qualified': False}
    documents = copy.deepcopy(scope_documents)
    documents.update({'DEV_BUDGET_PROPOSAL.json': copy.deepcopy(budget), 'DEV_CONTEXT_DRAFTS.json': contexts,
        'DEV_ENTRY_CONFIGURATION_DRAFT.json': {'schema': CONFIG, 'context_refs': {name: None for name, _, _ in runtime.RUNS},
            'entry_specs': {name: copy.deepcopy(budget['reads'][name]['entry_spec']) for name, _, _ in runtime.RUNS},
            'training_selection': None, 'training_selection_sha256': None, 'configuration_sha256': None},
        'DEV_REGISTRATION_DRAFT.json': {'schema': SCHEMA, 'approved': False, 'evaluation_authorized': False,
            'dispatch_authorized': False, 'production_training_authorized': False, 'unit_prices_measured': False,
            'source_head_basis': source.get('source_head_basis'), 'source_freeze': False,
            'scope_sha256': digest(scope_value), 'run_plan_sha256': digest(plan), 'total_completions': 9750}})
    return documents


def write_draft(documents, directory):
    require(documents['DEV_REGISTRATION_DRAFT.json'].get('schema') == SCHEMA
            and documents['DEV_REGISTRATION_DRAFT.json'].get('evaluation_authorized') is False,
            'unapproved DEV draft writer')
    directory = Path(directory)
    require(directory.is_absolute() and not directory.is_symlink() and not any(os.path.lexists(directory / name) for name in documents),
            'create-only DEV registration files')
    directory.mkdir(parents=True, exist_ok=True); refs = {}
    for name, value in documents.items():
        require(Path(name).name == name and name.endswith('.json'), 'fixed DEV metadata filename')
        blob = canonical(value) + b'\n'; require(len(blob) <= 8 * raw.CHUNK, 'bounded DEV registration JSON')
        with (directory / name).open('xb') as stream:
            require(stream.write(blob) == len(blob), 'short DEV registration write'); stream.flush(); os.fsync(stream.fileno())
        refs[name] = {'path': str(directory / name), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    return refs
