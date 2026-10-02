"""Concrete create-only canonical 375-update production packet drafts.

All prices are explicit caller-proposed finite reservations. No qualification
result, scope authorization, source adoption or production dispatch is issued.
The unchanged full TRAIN plan and prepared receipt are retained in the actual
native ledger specification rather than represented by placeholder hashes.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from types import MappingProxyType

from . import ce_training_admission as admission, ce_training_runtime as runtime
from . import ce_registration as qualification_draft, ce_controller as controller
from . import ce_journal as journal, ce_plan, ce_native_backend as backend
from . import ce_qualified_receipt as completed, ce_qualification as qualification
from . import ce_qualification_artifacts as raw, ce_native_services as services
from . import ce_native_authority as authority
from . import ce_scope_adapters as scope_adapters, ce_service_prepayment as prepayment

SCHEMA = 'CE8-CANONICAL-PRODUCTION-DRAFT-1'
BUDGET = 'CE8-PRODUCTION-BUDGET-PROPOSAL-1'
STATUS = 'DRAFT_UNAPPROVED_PRODUCTION'
MAX_FILE = 8 * raw.CHUNK
require, canonical, digest = raw.require, raw.canonical, raw.digest


def _roles(code):
    functions = {'bootstrap_outer': runtime.bootstrap_outer, 'outer_issuer': runtime.outer_issuer,
        'outer_verifier': runtime.outer_verifier, 'outer_reservation': runtime.outer_reservation,
        'production_admission': runtime.admit_context, 'worker_observation': scope_adapters.worker_observation,
        'bootstrap_constructor_prepayment': prepayment.bootstrap_constructor_prepayment,
        'native_journal': runtime.native_journal, 'publication': runtime.publication_role,
        'interruption': runtime.interruption_role, 'completion_seal': runtime.completion_seal_role}
    result = {}
    for role, function in functions.items():
        path = function.__module__.replace('.', '/') + '.py'
        require(raw._hex(code.get(path)), 'concrete production source role missing: ' + role)
        result[role] = {'module': function.__module__, 'symbol': function.__name__, 'source_sha256': code[path]}
    return result


def _budget(value, receipt):
    required = {'schema', 'approved', 'production_training_authorized', 'measured_prices', 'reservation_basis',
        'physical_deadline_seconds', 'journal_operation_bounds', 'backend_operation_bounds', 'journal_setup_bounds',
        'native_caps', 'bootstrap_bounds', 'service_bootstrap', 'training_reader_bounds', 'resource', 'publication'}
    require(type(value) is dict and set(value) == required and value['schema'] == BUDGET
            and value['approved'] is False and value['production_training_authorized'] is False
            and value['measured_prices'] is False and type(value['reservation_basis']) is str and value['reservation_basis'],
            'explicit unapproved unmeasured production budget proposal')
    deadline = value['physical_deadline_seconds']
    require(type(deadline) in (int, float) and 0 < deadline < float('inf'), 'finite full production physical deadline')
    profiles, caps = value['journal_operation_bounds'], value['native_caps']
    require(type(profiles) is dict and set(profiles) == journal.NATIVE_KINDS, 'all native CE journal operation profiles')
    for profile in profiles.values(): journal._profile(profile)
    setup = journal._profile(value['journal_setup_bounds'])
    require(type(caps) is dict and set(caps) == set(journal.COUNTERS) | {'schema', 'synthetic_cpu', 'operation_starts'}
            and caps['schema'] == journal.NATIVE_CAP_SCHEMA and caps['synthetic_cpu'] is False,
            'exact complete native journal caps')
    starts = caps['operation_starts']
    require(type(starts) is dict and set(starts) == journal.NATIVE_KINDS
            and all(type(count) is int and count > 0 for count in starts.values())
            and all(type(caps[key]) is int and caps[key] >= (0 if key == 'resumes' else 1)
                    for key in journal.COUNTERS if key != 'wall_seconds')
            and type(caps['wall_seconds']) in (int, float) and 0 < caps['wall_seconds'] < float('inf')
            and caps['metadata_reserved_bytes'] >= 64 * raw.CHUNK and caps['invocations'] >= caps['resumes'] + 1,
            'finite cumulative starts/invocations/resumes/token/IO/hash/metadata/wall caps')
    minimum = {'loss_forward': 3000, 'backward': 3000, 'tokenizer_render': 3000, 'fingerprint': 753,
        'checkpoint_configuration': 375, 'gradient_reset': 375, 'gradient_validation': 375,
        'gradient_clipping': 375, 'optimizer': 375, 'gradient_cleanup': 375,
        'checkpoint_save': 7, 'checkpoint_adapter_io': 7}
    require(all(starts[kind] >= count for kind, count in minimum.items()), 'start caps below full375 CE8/checkpoint schedule')
    census = receipt['census']
    for direction, kind in (('forward', 'loss_forward'), ('backward', 'backward')):
        require(caps[direction + '_selected_tokens'] >= starts[kind] * census['max_labeled_tokens']
                and caps[direction + '_sequence_tokens'] >= starts[kind] * census['max_sequence_positions'],
                'token caps below retained worst-row attempts')
    # These are reservation arithmetic lower bounds, not measurements or a
    # claim that the allocation/resource guards will admit an actual run.
    mapping = {'io_read_bytes': 'read_bytes_upper', 'io_write_bytes': 'write_bytes_upper',
               'identity_hash_attempts': 'identity_hash_attempts_upper', 'identity_hash_bytes': 'identity_hash_bytes_upper'}
    for counter, field in mapping.items():
        floor = setup[field] * caps['invocations'] + sum(starts[kind] * profiles[kind][field] for kind in profiles)
        require(caps[counter] >= floor, 'native total below permanently retained operation reservations: ' + counter)
    require(profiles['cleanup']['wall_s_upper'] >= deadline and caps['wall_seconds'] >= deadline
            + setup['wall_s_upper'] * caps['invocations'] + sum(starts[kind] * profiles[kind]['wall_s_upper'] for kind in profiles),
            'full model lifetime cleanup and physical+reserved wall costs must fit')
    backend_profiles = value['backend_operation_bounds']
    require(type(backend_profiles) is dict and set(backend_profiles) == backend.OPERATION_KINDS,
            'exact native backend profiles including pre-journal construction')
    for kind, profile in backend_profiles.items():
        require(type(profile) is dict and set(profile) == set(backend.PROFILE_FIELDS)
                and all(type(count) is int and count >= 0 for count in profile.values()), 'strict backend reservation counts')
        if kind in profiles:
            require(profile == {key: profiles[kind][key] for key in backend.PROFILE_FIELDS}, 'backend/journal price overlap differs')
    controller._bounds(value['bootstrap_bounds'])
    service = value['service_bootstrap']
    require(type(service) is dict and type(service.get('operation_profiles')) is dict
            and {'qualification_reducer_overhead', 'qualification_artifact_readback', 'outer_reservation', 'publication'}
                <= set(service['operation_profiles']) and set(service.get('caps', {})) == set(services.COUNTERS),
            'separate finite service/bootstrap/completed-pair reader prices required')
    for profile in service['operation_profiles'].values(): services._profile(profile)
    services._profile(service['setup_bounds']); services._profile(service['finalization_bounds'])
    for key, number in service['caps'].items():
        require(type(number) in (int, float) and not isinstance(number, bool) and 0 < number < float('inf')
                and (key == 'wall_seconds' or type(number) is int), 'strict finite service caps')
    # Reserve up to31 overhead and9 file callbacks for the fixed consumer
    # path. Service defaults permanently charge their whole profile.
    pair_callbacks = {'qualification_reducer_overhead': 31, 'qualification_artifact_readback': 9}
    for counter, field in {'read_bytes': 'read_bytes_upper', 'hash_attempts': 'identity_hash_attempts_upper',
                           'hash_bytes': 'identity_hash_bytes_upper', 'wall_seconds': 'wall_s_upper'}.items():
        require(service['caps'][counter] >= sum(count * service['operation_profiles'][kind][field]
                for kind, count in pair_callbacks.items()), 'service caps below completed-pair permanent profile reservations')
    overhead = service['operation_profiles']['qualification_reducer_overhead']
    require(overhead['identity_hash_attempts_upper'] >= 2 * completed.MAX_EVENTS
            and overhead['identity_hash_bytes_upper'] >= 16 * completed.MAX_JOURNAL,
            'completed-pair chain replay hash upper must be priced before production')
    require(type(value['training_reader_bounds']) is dict and set(value['training_reader_bounds']) == set(raw.BUDGET_FIELDS)
            and all(type(number) is int and number > 0 for number in value['training_reader_bounds'].values()),
            'explicit independent completed-pair read/hash/reducer bounds')
    publication = value['publication']
    require(type(publication) is dict and set(publication) == {'max_json_bytes', 'artifact_read_bytes_upper'}
            and type(publication['max_json_bytes']) is int and 0 < publication['max_json_bytes'] <= raw.MAX_JSON
            and type(publication['artifact_read_bytes_upper']) is int and publication['artifact_read_bytes_upper'] > 0,
            'finite donor/journal/publication read ceiling')
    profile = service['operation_profiles']['publication']
    require(profile['read_bytes_upper'] >= publication['artifact_read_bytes_upper'] + 2 * publication['max_json_bytes'] + 2
            and profile['identity_hash_bytes_upper'] >= publication['artifact_read_bytes_upper'] + 4 * publication['max_json_bytes']
            and profile['identity_hash_attempts_upper'] >= 8 and profile['write_bytes_upper'] >= publication['max_json_bytes'],
            'whole donor/journal/publication envelope before375 updates')
    require(type(value['resource']) is dict and set(value['resource']) == {'limits', 'resource_probe', 'host_transient_bytes'},
            'explicit full production resource guard policy')
    stages = set(journal.NATIVE_KINDS) | backend.OPERATION_KINDS | {'ce_boundary', 'runtime_anchor', 'backend_construct',
        'controller_cleanup', 'donor_publish_verify', 'checkpoint_resume'}
    transients = value['resource']['host_transient_bytes']
    require(type(transients) is dict and stages <= set(transients)
            and all(type(number) is int and number >= 0 for number in transients.values()),
            'all actual production resource barrier transients explicitly declared')
    return copy.deepcopy(value)


def build_documents(qualification_documents, plan, preparation, current_source, budget, *, output_root,
                    qualification_selection=None):
    """Pure metadata draft assembly; absent completed source pin stays unresolved."""
    original = copy.deepcopy(qualification_documents)
    record, qualified_scope = original['REGISTRATION_DRAFT.json'], original['SCOPE.json']
    require(record['selected_decision_ref']['sha256'] == qualification_draft.DECISION_SHA
            and record['choice'] == 'canonical_only' and qualified_scope['arm'] == 'canonical'
            and qualified_scope['allowed_data_roles'] == ['train'], 'existing selected canonical qualification basis')
    batches = ce_plan.validate_plan(plan)
    require(len(batches) == 375 and sum(map(len, batches)) == 3000 and plan['mode'] == 'unapproved_proposal'
            and plan['label_binding']['arm'] == 'canonical', 'same full3000 TRAIN canonical target plan')
    receipt = original['EXPECTED_PREPARED_RECEIPT.json']['expected_receipt']
    recipe, runtime_doc, anchors = original['RECIPE.json']['recipe'], original['RUNTIME.json'], original['ANCHORS.json']
    recipe_sha = digest(recipe)
    context = {'arm': 'canonical', 'refs': {'preparation_ref': receipt['preparation_ref'], 'plan_ref': receipt['plan_ref']},
        'bindings': {'anchors': anchors, 'plan': plan, 'scope': {'static_recipe_sha256': recipe_sha},
                     'ingestion': {'expected_census': qualification.declared_census(preparation, 'canonical')}}}
    qualification.validate_actual_ingestion(context, receipt)
    require(recipe_sha == qualified_scope['static_recipe_sha256'] and recipe['synthetic_cpu'] is False
            and recipe['accumulation_rows'] == 8 and recipe['microbatch_rows'] == 1
            and anchors['own_start_sha256'] == anchors['common_anchor_sha256'], 'unchanged CE8/common warm recipe')
    root = Path(output_root)
    require(root.is_absolute() and not root.is_relative_to(Path(current_source['source_checkout']))
            and root != Path(qualified_scope['native_backend']['output_directory']), 'separate production output owner')
    code = current_source['code_sha256']; old_code = original['SOURCE_SNAPSHOT.json']['code_sha256']
    require(all(raw._hex(code.get(path)) and code[path] == old_code.get(path) for path in admission.CORE_SOURCE_PATHS[1:]),
            'qualified numerical/ingestion/CE/checkpoint/runtime sources changed')
    bound = _budget(budget, receipt)
    dependency = {'schema': 'CE8-COMPLETED-QUALIFICATION-DEPENDENCY-1', 'source_selected': False,
                  'completed_pair_verified': False, 'selection_sha256': None, 'selection': None,
                  'condition': 'Exact source-selected joint receipt, finalizer seal and original preclose must pass the consumer before authorization.'}
    if qualification_selection is not None:
        require(type(qualification_selection) is MappingProxyType and qualification_selection is completed._SOURCE_SELECTION
                and raw._hex(completed._SOURCE_SELECTION_SHA256), 'exact future source-selected qualification dependency')
        selected = completed._shape(copy.deepcopy(dict(qualification_selection)))
        require(digest(selected) == completed._SOURCE_SELECTION_SHA256 and selected['binding']['static_recipe_sha256'] == recipe_sha
                and selected['prepared_receipt_sha256'] == receipt['receipt_sha256']
                and selected['runtime_sha256'] == digest(runtime_doc) and selected['anchors_sha256'] == digest(anchors),
                'source-selected qualified recipe/runtime/anchor/full-census changed')
        dependency.update(source_selected=True, selection_sha256=completed._SOURCE_SELECTION_SHA256, selection=selected)
    native = copy.deepcopy(qualified_scope['native_backend'])
    native.update(output_directory=str(root), operation_bounds=bound['backend_operation_bounds'],
                  operation_bounds_status='PROPOSED_BOUNDED_PRODUCTION_UPPER_NOT_AUTHORIZED',
                  source_freeze={'head': current_source.get('source_head_basis', record['source_head_basis']), 'frozen': False})
    price = {'schema': admission.PRICE_POLICY, 'approved': False, 'basis': 'AUTHORIZED_BOUNDED_PRODUCTION_UPPER',
             'measured_prices': False, 'operation_bounds_sha256': digest(native['operation_bounds']), 'justification_ref': None}
    scope = {'schema': admission.SCOPE, 'role': 'production_training', 'approved': False,
        'production_training_authorized': False, 'qualification_authorized': False, 'evaluation_authorized': False,
        'arm': 'canonical', 'choice_sha256': record['selected_decision_ref']['sha256'], 'allowed_data_roles': ['train'],
        'static_recipe_sha256': recipe_sha, 'native_backend': native, 'production_price_policy': price,
        'native_execution_stages': sorted(authority.STAGES - {'longest_masks', 'likelihood_gradient_parity', 'adam_clipping', 'resource_io_measurement'}),
        'execution': {'rows': 3000, 'epochs': 1, 'boundaries': 375, 'run_seed': 20261001,
                      'checkpoint_boundaries': copy.deepcopy(admission.SCHEDULE)},
        'checkpoint_lease': copy.deepcopy(qualified_scope['checkpoint_lease']),
        'training_runtime': {'schema': runtime.SPEC, 'owned_directory': str(root / 'publication'), **bound['publication']}}
    scope['checkpoint_lease']['label'] = 'comp-rs-001-ce8-canonical-production'
    ledger = {'schema': 'CE8-NATIVE-LEDGER-SPEC-1', 'owned_directory': str(root / 'ledger'), 'plan': copy.deepcopy(plan),
        'ingestion_receipt_sha256': receipt['receipt_sha256'], 'caps': bound['native_caps'],
        'bindings': {'scope': journal.NATIVE_SCOPE, 'recipe': copy.deepcopy(recipe), 'source': copy.deepcopy(code),
            'runtime': copy.deepcopy(runtime_doc['decoding_preimage']), 'anchors': {key: anchors[key] for key in
                ('own_start_sha256', 'common_anchor_sha256')}, 'prepared_rows': copy.deepcopy(receipt['prepared_row_bindings']),
            'operation_bounds': bound['journal_operation_bounds'], 'setup_bounds': bound['journal_setup_bounds'],
            'metadata_roots': [str(root)]}}
    justification = {'schema': admission.PRICE_JUSTIFICATION, 'approved': False, 'production_training_authorized': False,
        'basis': price['basis'], 'measured_prices': False, 'prices_are_measurements': False,
        'reservation_basis': bound['reservation_basis'], 'static_recipe_sha256': recipe_sha,
        'source_code_sha256': copy.deepcopy(code), 'operation_bounds_sha256': price['operation_bounds_sha256'],
        'qualified_receipt_sha256': None, 'production_budget': None, 'budget_proposal': bound}
    service = copy.deepcopy(original['SERVICE_SPEC.json']); service.pop('supervision', None); service.pop('scope_adapters', None)
    service.update(approved_code_sha256=copy.deepcopy(code), role_descriptors=_roles(code), physical_deadline_seconds=bound['physical_deadline_seconds'],
                   bootstrap=copy.deepcopy(bound['service_bootstrap']), resource=copy.deepcopy(bound['resource']))
    service['bootstrap']['owned_directory'] = str(root / 'service-bootstrap')
    service['bootstrap']['physical_wall_seconds_upper'] = bound['physical_deadline_seconds']
    service['resource']['resource_probe']['output_root'] = str(root)
    service['constructor_prepayment']['owned_directory'] = str(root / 'service-bootstrap')
    service['training_admission'] = {'qualification_root': None if dependency['selection'] is None else dependency['selection']['owned_root'],
                                   'reader_bounds': bound['training_reader_bounds']}
    documents = {'RECIPE.json': copy.deepcopy(original['RECIPE.json']), 'RUNTIME.json': copy.deepcopy(runtime_doc),
        'ANCHORS.json': copy.deepcopy(anchors), 'SOURCE_SNAPSHOT.json': copy.deepcopy(current_source),
        'EXPECTED_PREPARED_RECEIPT.json': copy.deepcopy(original['EXPECTED_PREPARED_RECEIPT.json']),
        'INGESTION_SPEC.json': copy.deepcopy(original['INGESTION_SPEC.json']), 'NATIVE_LEDGER_SPEC.json': ledger,
        'ATTEMPT_CAPS.json': {'schema': journal.NATIVE_CAP_SCHEMA, 'caps': bound['native_caps'], 'production_training_authorized': False},
        'RESOURCES.json': {'schema': 'CE8-PRODUCTION-RESOURCE-POLICY-1', **copy.deepcopy(bound['resource']),
                           'physical_deadline_seconds': bound['physical_deadline_seconds'], 'production_fit_measured': False},
        'QUALIFICATION_DEPENDENCY.json': dependency, 'PRODUCTION_PRICE_JUSTIFICATION.json': justification,
        'SERVICE_SPEC.json': service, 'SCOPE.json': scope, 'EVALUATION_SCOPE.json': copy.deepcopy(original['EVALUATION_SCOPE.json'])}
    backend_path = admission.CORE_SOURCE_PATHS[0]
    if code[backend_path] != old_code.get(backend_path):
        documents['BACKEND_GATE_COMPATIBILITY.json'] = {'schema': admission.BACKEND_COMPATIBILITY, 'approved': False,
            'module': backend_path, 'changed_symbols': ['validate_backend_spec'],
            'qualified_backend_ref': copy.deepcopy(original['SOURCE_SNAPSHOT.json']['source_files'][backend_path]),
            'production_backend_ref': copy.deepcopy(current_source['source_files'][backend_path])}
    documents['REGISTRATION_DRAFT.json'] = {'schema': SCHEMA, 'status': STATUS, 'approved': False,
        'training_authorized': False, 'production_dispatch_authorized': False, 'dispatch_authorized': False,
        'qualification_authorized': False, 'evaluation_authorized': False, 'native_qualified': False,
        'selected_decision_ref': copy.deepcopy(record['selected_decision_ref']), 'source_freeze': False,
        'source_head_basis': native['source_freeze']['head'], 'physical_deadline_seconds': bound['physical_deadline_seconds'],
        'dose': {'rows': 3000, 'epochs': 1, 'updates': 375, 'states': 376, 'microbatch_rows': 1, 'equal_row_denominator': 8},
        'checkpoint_boundaries': copy.deepcopy(admission.SCHEDULE), 'bootstrap_bounds': bound['bootstrap_bounds'],
        'qualification_pair_source_selected': dependency['source_selected'], 'unit_prices_measured': False}
    return documents


def write_draft(documents, directory, external_refs):
    """Publish actual raw references in dependency order, without authorizing."""
    value = copy.deepcopy(documents); record = value['REGISTRATION_DRAFT.json']
    require(record.get('schema') == SCHEMA and record.get('status') == STATUS
            and all(record.get(key) is False for key in ('approved', 'training_authorized', 'dispatch_authorized', 'production_dispatch_authorized')),
            'only unapproved production draft materialization')
    require(value['SCOPE.json'].get('approved') is False and value['SCOPE.json'].get('production_training_authorized') is False
            and value['PRODUCTION_PRICE_JUSTIFICATION.json'].get('approved') is False
            and value['PRODUCTION_PRICE_JUSTIFICATION.json'].get('production_training_authorized') is False,
            'draft scope and price justification cannot authorize production')
    expected = value['EXPECTED_PREPARED_RECEIPT.json']['expected_receipt']
    require(type(external_refs) is dict and set(external_refs) == {'decision', 'preparation', 'plan'}
            and external_refs['decision'] == record['selected_decision_ref']
            and {key: external_refs['preparation'][key] for key in ('path', 'sha256')} == expected['preparation_ref']
            and {key: external_refs['plan'][key] for key in ('path', 'sha256')} == expected['plan_ref'],
            'unchanged external selected decision/full canonical TRAIN plan and preparation')
    directory = Path(directory)
    require(directory.is_absolute() and not directory.is_symlink(), 'absolute regular production registration owner')
    names = set(value) | {'REQUEST.json', 'SCOPE_AUTHORIZATION.json', 'PACKET_INDEX.json'}
    require(not any(os.path.lexists(directory / name) for name in names), 'create-only production draft files exist')
    directory.mkdir(parents=True, exist_ok=True); refs = {}
    def emit(name, body):
        blob = canonical(body) + b'\n'; require(len(blob) <= MAX_FILE, 'production metadata file ceiling')
        with (directory / name).open('xb') as stream:
            require(stream.write(blob) == len(blob), 'short production draft write'); stream.flush(); os.fsync(stream.fileno())
        refs[name] = {'path': str(directory / name), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    deferred = {'SCOPE.json', 'SERVICE_SPEC.json', 'PRODUCTION_PRICE_JUSTIFICATION.json'}
    for name, body in value.items():
        if name not in deferred: emit(name, body)
    dependency = value['QUALIFICATION_DEPENDENCY.json']; selection = dependency['selection']
    qualified_ref = None if selection is None else selection['references']['receipt']
    justification = value['PRODUCTION_PRICE_JUSTIFICATION.json']
    justification['qualified_receipt_sha256'] = None if qualified_ref is None else qualified_ref['sha256']
    justification['production_budget'] = {'rows': 3000, 'epochs': 1, 'updates': 375,
        'checkpoint_boundaries': copy.deepcopy(admission.SCHEDULE), 'resources_sha256': refs['RESOURCES.json']['sha256'],
        'attempt_caps_sha256': refs['ATTEMPT_CAPS.json']['sha256']}
    emit('PRODUCTION_PRICE_JUSTIFICATION.json', justification)
    scope = value['SCOPE.json']; scope.update(resources_sha256=refs['RESOURCES.json']['sha256'],
        attempt_caps_sha256=refs['ATTEMPT_CAPS.json']['sha256'], native_ledger_ref=refs['NATIVE_LEDGER_SPEC.json'],
        expected_prepared_receipt_ref=refs['EXPECTED_PREPARED_RECEIPT.json'])
    scope['production_price_policy']['justification_ref'] = refs['PRODUCTION_PRICE_JUSTIFICATION.json']
    if 'BACKEND_GATE_COMPATIBILITY.json' in refs: scope['backend_gate_compatibility_ref'] = refs['BACKEND_GATE_COMPATIBILITY.json']
    emit('SCOPE.json', scope)
    reference_map = {'selectedoperatorchoice_ref': external_refs['decision'], 'recipe_ref': refs['RECIPE.json'],
        'scope_ref': refs['SCOPE.json'], 'source_ref': refs['SOURCE_SNAPSHOT.json'], 'runtime_ref': refs['RUNTIME.json'],
        'anchors_ref': refs['ANCHORS.json'], 'evaluation_ref': refs['EVALUATION_SCOPE.json'],
        'preparation_ref': {key: external_refs['preparation'][key] for key in ('path', 'sha256')},
        'plan_ref': {key: external_refs['plan'][key] for key in ('path', 'sha256')}, 'ingestion_ref': refs['INGESTION_SPEC.json'],
        'resources_ref': refs['RESOURCES.json'], 'attempt_caps_ref': refs['ATTEMPT_CAPS.json'], 'qualifiedreceipt_ref': qualified_ref}
    require(reference_map['selectedoperatorchoice_ref'] == record['selected_decision_ref']
            and reference_map['preparation_ref'] == value['INGESTION_SPEC.json']['preparation_ref'], 'unchanged external choice/TRAIN source references')
    request = controller.build_request('training', 'canonical', reference_map, record['bootstrap_bounds'])
    request['scope_authorization_ref'] = None
    emit('SCOPE_AUTHORIZATION.json', {'schema': admission.AUTHORIZATION, 'approved': False, 'training_authorized': False,
         'qualification_authorized': False, 'request_preimage_sha256': digest(controller.request_preimage(request))})
    request['scope_authorization_ref'] = refs['SCOPE_AUTHORIZATION.json']; emit('REQUEST.json', request)
    emit('SERVICE_SPEC.json', value['SERVICE_SPEC.json'])
    emit('PACKET_INDEX.json', {'schema': SCHEMA, 'status': STATUS, 'references': copy.deepcopy(refs),
        'training_authorized': False, 'production_dispatch_authorized': False, 'evaluation_authorized': False,
        'qualification_pair_source_selected': dependency['source_selected'], 'native_qualified': False})
    return refs
