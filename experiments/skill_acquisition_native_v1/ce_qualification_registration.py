"""Create-only, canonical discarded-qualification configuration materialization.

The delegated selected operator decision permits this new experiment. This
writer authorizes only its finite qualification scope, never production or
evaluation. Request/scope files precede service configuration, which prevents
their raw hash from depending on a configuration containing that same hash.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

from . import ce_registration as draft, ce_controller as controller
from . import ce_qualification_entry as entry, ce_native_authority as authority
from . import ce_scope_adapters as scope_adapters, ce_service_prepayment as prepayment
from . import ce_parent_publication as publication

require, canonical, digest = authority.require, authority.canonical, authority.digest


def authorize_documents(documents, *, source_head):
    value = copy.deepcopy(documents)
    record = value['REGISTRATION_DRAFT.json']
    require(record['selected_decision_ref']['sha256'] == draft.DECISION_SHA
            and record['choice'] == 'canonical_only' and record['qualification_weights'].startswith('discarded'),
            'exact selected delegated canonical discarded experiment')
    source = value['SOURCE_SNAPSHOT.json']
    require(type(source_head) is str and len(source_head) == 40
            and record['source_head_basis'] == source_head and source['code_sha256'], 'exact reviewed code commit basis')
    scope = value['SCOPE.json']
    require(scope['role'] == 'qualification_only' and scope['arm'] == 'canonical'
            and scope['choice_sha256'] == draft.DECISION_SHA and scope['allowed_data_roles'] == ['train']
            and not scope['production_training_authorized'] and not scope['evaluation_authorized'], 'TRAIN-only qualification scope')
    scope['approved'] = scope['qualification_authorized'] = True
    scope['native_backend']['operation_bounds_status'] = 'AUTHORIZED_QUALIFICATION_UPPER'
    scope['native_backend']['source_freeze'] = {'head': source_head, 'frozen': True}
    source['source_freeze'] = True; source['source_head_basis'] = source_head
    record.update(status='AUTHORIZED_DISCARDED_QUALIFICATION_ONLY', approved=True, qualification_authorized=True,
                  source_freeze=True, dispatch_authorized=False, native_qualified=False,
                  operator_actor='assistant_under_delegated_experiment_authority', human_answered_pending_question=False)
    require(not record['training_authorized'] and not record['evaluation_authorized'], 'separate training/evaluation grants remain required')
    return value


def role_descriptors(code):
    roles = {'bootstrap_outer': scope_adapters.bootstrap_outer, 'outer_issuer': scope_adapters.outer_issuer,
        'outer_verifier': scope_adapters.outer_verifier, 'outer_reservation': scope_adapters.outer_reservation,
        'worker_observation': scope_adapters.worker_observation,
        'bootstrap_constructor_prepayment': prepayment.bootstrap_constructor_prepayment,
        'publication': publication.publication_role, 'interruption': publication.interruption_role,
        'completion_seal': publication.completion_seal_role}
    result = {}
    for role, function in roles.items():
        hashed = code.get(function.__module__.replace('.', '/') + '.py')
        require(authority._hex(hashed), 'fixed role missing from reviewed full source: ' + role)
        result[role] = {'module': function.__module__, 'symbol': function.__name__, 'source_sha256': hashed}
    return result


def write_qualification_configuration(documents, directory, external_refs, *, launch_window, source_head):
    """Finite metadata writes; caller first reviews the concrete source/caps."""
    documents = authorize_documents(documents, source_head=source_head)
    directory = Path(directory)
    require(directory.is_absolute() and not directory.is_symlink(), 'absolute regular configuration artifact owner')
    require(type(launch_window) is dict and set(launch_window) == {'not_before', 'not_after'}
            and all(type(v) is int and v > 0 for v in launch_window.values())
            and launch_window['not_before'] < launch_window['not_after'], 'explicit future worker creation window')
    names = set(documents) | {'REQUEST.json', 'SCOPE_AUTHORIZATION.json', 'PARENT_CONFIGURATION.json', 'PACKET_INDEX.json'}
    require(not any(os.path.lexists(directory / name) for name in names), 'create-only qualification configuration exists')
    directory.mkdir(parents=True, exist_ok=True); refs = {}
    def emit(name, value):
        blob = canonical(value) + b'\n'
        require(len(blob) <= 8 * draft.MIB, 'bounded qualification metadata file')
        with (directory / name).open('xb') as stream:
            require(stream.write(blob) == len(blob), 'short qualification metadata write')
            stream.flush(); os.fsync(stream.fileno())
        refs[name] = {'path': str(directory / name), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    for name, value in documents.items():
        if name not in ('SCOPE.json', 'SERVICE_SPEC.json'): emit(name, value)
    scope = documents['SCOPE.json']
    scope['attempt_caps_sha256'] = refs['ATTEMPT_CAPS.json']['sha256']; scope['resources_sha256'] = refs['RESOURCES.json']['sha256']
    scope['expected_prepared_receipt_ref'] = refs['EXPECTED_PREPARED_RECEIPT.json']; emit('SCOPE.json', scope)
    references = {'selectedoperatorchoice_ref': external_refs['decision'], 'recipe_ref': refs['RECIPE.json'],
        'scope_ref': refs['SCOPE.json'], 'source_ref': refs['SOURCE_SNAPSHOT.json'], 'runtime_ref': refs['RUNTIME.json'],
        'anchors_ref': refs['ANCHORS.json'], 'evaluation_ref': refs['EVALUATION_SCOPE.json'],
        'preparation_ref': {k: external_refs['preparation'][k] for k in ('path', 'sha256')},
        'plan_ref': {k: external_refs['plan'][k] for k in ('path', 'sha256')}, 'ingestion_ref': refs['INGESTION_SPEC.json'],
        'resources_ref': refs['RESOURCES.json'], 'attempt_caps_ref': refs['ATTEMPT_CAPS.json']}
    request = controller.build_request('qualification', 'canonical', references, scope['qualifier_bootstrap']['self_bounds'])
    request.update(approved=True, mode='registered_scope', qualification_authorized=True, training_authorized=False, status='AUTHORIZED_QUALIFICATION_ONLY',
                   scope_authorization_ref=None)
    emit('SCOPE_AUTHORIZATION.json', {'schema': 'CE8-QUALIFICATION-AUTHORIZATION-1', 'approved': True,
        'qualification_authorized': True, 'training_authorized': False,
        'operator_actor': 'assistant_under_delegated_experiment_authority', 'human_answered_pending_question': False,
        'request_preimage_sha256': digest(controller.request_preimage(request))})
    request['scope_authorization_ref'] = refs['SCOPE_AUTHORIZATION.json']; emit('REQUEST.json', request)
    service = documents['SERVICE_SPEC.json']; source = documents['SOURCE_SNAPSHOT.json']; code = source['code_sha256']
    service['role_descriptors'] = role_descriptors(code)
    template = scope['checkpoint_processes']['launch_templates']['uninterrupted']
    observation_bounds = dict(template['observation_bounds'], wall_seconds_upper=19200)
    service['scope_adapters'] = {'schema': scope_adapters.SPEC, 'static_recipe_sha256': scope['static_recipe_sha256'],
        'owned_directory': str(Path(service['bootstrap']['owned_directory']) / 'capsules'),
        'worker': {'python_ref': template['python_ref'], 'process_image_ref': template['process_image_ref'],
            'request_ref': refs['REQUEST.json'], 'source_checkout': source['source_checkout'], 'source_files': source['source_files'],
            'approved_code_sha256': code, 'entry_module': entry.ENTRY_MODULE,
            'argv': [template['process_image_ref']['path'], '-m', entry.ENTRY_MODULE, '--request', refs['REQUEST.json']['path']],
            'launch_not_before_filetime_100ns': launch_window['not_before'],
            'launch_not_after_filetime_100ns': launch_window['not_after'], 'bounds': observation_bounds}}
    emit('SERVICE_SPEC.json', service)
    rpc = copy.deepcopy(scope['checkpoint_qualification']['local_issuer']); rpc.update(deadline_seconds=19200, rpc_seconds=19200)
    # Whole-worker deadline includes model work; individual ready/control/finish
    # IO is separately bounded and uses the fixed service input_io ceiling.
    rpc['operation_bounds']['wall_s_upper'] = 60
    configuration = {'schema': entry.SCHEMA, 'request': request, 'service_spec': service,
        'request_ref': refs['REQUEST.json'], 'service_spec_ref': refs['SERVICE_SPEC.json'], 'rpc_spec': rpc}
    configuration['configuration_sha256'] = digest(configuration)
    entry.validate_configuration(configuration); emit('PARENT_CONFIGURATION.json', configuration)
    emit('PACKET_INDEX.json', {'schema': 'CE8-AUTHORIZED-QUALIFICATION-CONFIGURATION-1', 'references': copy.deepcopy(refs),
        'qualification_authorized': True, 'training_authorized': False, 'evaluation_authorized': False,
        'dispatch_authorized': False, 'native_qualified': False, 'source_head_basis': source_head})
    return refs, configuration
