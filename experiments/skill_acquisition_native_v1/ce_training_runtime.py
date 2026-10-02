"""Concrete production CE8 adapters; source registration/dispatch stay closed.

The qualified pair and a separate production authorization precede the native
permit. This module reuses the existing backend, durable journal and one GPU
lease. Completion artifacts describe executed training, not a learning claim.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import uuid

from . import ce_native_services as services, ce_native_authority as authority
from . import ce_scope_adapters as scope, ce_controller as controller
from . import ce_qualification_artifacts as raw, ce_native_backend as native
from . import ce_training_admission as admission, ce_training_journal as journal_role
from . import ce_journal, update

SPEC = 'CE8-TRAINING-RUNTIME-POLICY-1'
PUBLICATION = 'CE8-TRAINING-BODY-COMPLETION-1'
require, digest = authority.require, raw.digest
OUTER_STAGES = frozenset(('qualification_metadata', 'controller_sources', 'ingestion_sources',
                         'ingestion_tokenizer', 'ingestion_boundary'))


def _owner(operation):
    return scope._owner(operation)


def _training(context):
    require(type(context) is dict and context.get('phase') == 'training'
            and context.get('qualifiedreceipt_verified') is True
            and context.get('production_training_authorized') is True
            and context.get('admission_evidence_verified') is True,
            'completed qualification pair and separate source production grant required')
    require(context['bindings']['scope'].get('schema') == admission.SCOPE
            and context['bindings']['scope'].get('role') == 'production_training', 'exact production scope')
    return context


def outer_verifier(issuer, request_sha256, scope_sha256, phase):
    require(type(issuer) is services.NativeControllerServices, 'original production metadata service issuer')
    issuer._admit()
    require(phase == issuer.request['phase'] == 'training'
            and request_sha256 == digest(controller.request_preimage(issuer.request))
            and scope_sha256 == issuer.request['scope_ref']['sha256'], 'exact production request/scope/phase')
    return True


def outer_reservation(issuer, stage, refmap):
    require(type(issuer) is services.NativeControllerServices, 'original production reservation issuer')
    outer_verifier(issuer, digest(controller.request_preimage(issuer.request)), issuer.request['scope_ref']['sha256'], 'training')
    require(stage in OUTER_STAGES and (issuer.context is not None or stage == 'qualification_metadata'),
            'production outer permits only bounded metadata and ingestion')
    declared = [issuer.request[key] for key in controller.REFS]
    require(type(refmap) is dict and all(value in declared or key == 'arm' and value == issuer.request['arm']
            for key, value in refmap.items()), 'production reference outside original request')
    with update._span(issuer.bootstrap.operation, 'outer_reservation',
                      dict(issuer.bootstrap.spec['operation_profiles']['outer_reservation'], stage=stage)):
        return {'stage': stage, 'native_qualified': False}


def bootstrap_outer(request, operation, spec):
    owner = _owner(operation)
    require(owner.context is None and request == owner.request and spec == owner.spec and request['phase'] == 'training',
            'source production metadata bootstrap before admission')
    policy = scope._policy(owner)
    outer = controller.OuterAdmission(request, {'mode': 'registered_scope', 'static_recipe_sha256': policy['static_recipe_sha256']},
                                      owner, outer_reservation, outer_verifier, bootstrap=True)
    scope.read_request_metadata(request, policy['worker']['request_ref'], operation)
    return outer


def admit_context(request, code_sha256, outer, operation, service):
    require(type(service) is services.NativeControllerServices and service is _owner(operation)
            and request == service.request and service.context is None, 'original source training admission owner')
    policy = service.spec.get('training_admission')
    require(type(policy) is dict and set(policy) == {'qualification_root', 'reader_bounds'}
            and type(policy['qualification_root']) is str and Path(policy['qualification_root']).is_absolute(),
            'explicit independently priced completed-pair reader policy')
    from . import ce_qualified_receipt as completed
    selected = completed.selected_qualification()  # source pin remains None by default
    require(selected['owned_root'] == policy['qualification_root'], 'qualified-pair reader cannot select another artifact root')
    reader = raw.RawEvidenceReader(operation, policy['reader_bounds'], policy['qualification_root'])
    def preflight(name, contract):
        require(name == 'completed_qualification_pair' and contract['reader_bounds'] == reader.remaining,
                'same original source-prepaid qualification-pair reader')
        service.bootstrap.tick_wall()
    return admission.admit_training(request, code_sha256=code_sha256, outer_admission=outer, io_observer=operation,
                                    qualified_selection=selected, qualified_reader=reader, qualifier_preflight=preflight)


def native_authority_verifier(issuer, context_sha256, request_sha256, scope_sha256, phase, arm):
    require(type(issuer) is services.NativeControllerServices, 'original source production execution issuer')
    issuer._admit(); context = _training(issuer.context)
    require(phase == context['phase'] == 'training' and arm == context['arm'] == 'canonical'
            and context_sha256 == digest(authority.context_preimage(context))
            and request_sha256 == context['request_sha256'] and scope_sha256 == context['scope_sha256'],
            'complete source production context before native permit')
    scope._full_worker(issuer, context, refresh=False)  # no recursive priced OS callback
    return True


def outer_issuer(request, context, operation):
    owner = _owner(operation); _training(context)
    require(request == owner.request and context is owner.context, 'same fully observed production service context')
    scope._full_worker(owner, context)
    if context.get('native_execution_permit') is None:
        context['native_execution_permit'] = authority.NativeExecutionPermit(context, owner, native_authority_verifier,
            operation=operation, gate=owner.gate)
    authority.require_execution_permit(context['native_execution_permit'], context, 'ingestion_tokenizer',
                                       operation=operation, gate=owner.gate)
    return owner


def create_training_backend(context):
    permit = context.get('native_execution_permit') if type(context) is dict else None
    authority.require_execution_permit(permit, context, 'backend_construct')
    _training(context)
    owner = _owner(context['native_services']['operation'])
    require(owner.context is context and owner.supervisor_lease is not None
            and owner.operation.ledger is None and context['native_services'].get('training_backend') is None,
            'one metadata backend under original owned lease before journal/model')
    backend = native.NativeCEBackend(context, permit)
    context['native_services']['training_backend'] = backend
    return backend


native_journal = journal_role.native_journal


@contextmanager
def training_lease(service, context):
    require(type(service) is services.NativeControllerServices, 'original source production lease service')
    service._admit(); _training(context)
    require(service.context is context, 'same admitted production worker context')
    permit = context.get('native_execution_permit')
    authority.require_execution_permit(permit, context, 'checkpoint_resume', operation=service.operation, gate=service.gate)
    require(service.supervisor_lease is None and service.lease_token is None and service.operation.ledger is None,
            'one original lease before production journal; no nested acquisition')
    from .ce_checkpoint_lease import SupervisorGpuLease
    owner = SupervisorGpuLease(context, permit, service.worker_handle, None, service.operation, service.gate)
    primary = None
    try:
        owner.acquire(); service.supervisor_lease, service.lease_token = owner, owner.token
        service.gate.supervisor = owner
        yield owner
    except BaseException as error:
        primary = error; raise
    finally:
        if owner.release_seal is not None and not owner.closed:
            try: owner.close()
            except BaseException as failure:
                if primary is not None:
                    raw._accounting(primary, failure); raise primary from failure
                raise
            finally:
                if owner.closed: service.lease_token = None; service.gate.supervisor = None


def validate_completion(snapshot, donor, context):
    """Pure completion coherence; never grants training or execution."""
    progress = snapshot.get('progress', {})
    require(type(progress.get('C')) is type(progress.get('U')) is int and progress['C'] == progress['U'] == 375
            and snapshot.get('synthetic_cpu') is False and snapshot.get('fatal') is False
            and snapshot.get('unknown_wall') is False and snapshot.get('requires_restore') is False
            and snapshot.get('active_attempt') is None and type(snapshot.get('pending_operations')) is int
            and snapshot['pending_operations'] == 0,
            'all 375 proved updates and terminal cleared production work')
    anchors = context['bindings']['anchors']
    require(type(donor) is dict and donor.get('schema') == 'SKILL-ACQUISITION-DONOR-DELTA-1'
            and donor.get('common_anchor_sha256') == anchors['common_anchor_sha256']
            and donor.get('own_start_sha256') == anchors['own_start_sha256']
            and donor.get('reference') == 'EXACT_REBUILT_COMMON_WARM_ANCHOR'
            and all(raw._hex(donor.get(key)) for key in ('file_sha256', 'data_section_sha256', 'state_sha256', 'final_master_sha256')),
            'source production donor and independent common warm reference')
    return snapshot


def _stream_reference(path, maximum, expected_sha=None):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum, 'bounded retained production file')
    size = path.stat().st_size; hashed = hashlib.sha256(); count = 0
    with path.open('rb') as stream:
        while block := stream.read(min(raw.CHUNK, size + 1 - count)):
            count += len(block); require(count <= size, 'production artifact grew'); hashed.update(block)
    require(count == size == path.stat().st_size and (expected_sha is None or hashed.hexdigest() == expected_sha),
            'production artifact bytes/hash changed')
    return {'path': str(path), 'bytes': size, 'sha256': hashed.hexdigest()}


def publication_role(context, donor, journal, service_snapshot, operation):
    permit = context.get('native_execution_permit') if type(context) is dict else None
    authority.require_execution_permit(permit, context, 'completion_publication')
    owner = _owner(operation); _training(context)
    require(owner.context is context and type(journal) is ce_journal.CeJournal and journal.synthetic_cpu is False
            and operation.ledger is journal and owner.lease_token is None and owner.supervisor_lease.closed,
            'released actual production lease and original durable journal before publication')
    require(owner.bound_backend.model is owner.bound_backend.optimizer is owner.bound_backend.tokenizer is None,
            'actual model/optimizer/tokenizer cleanup precedes production completion')
    spec = context['bindings']['scope'].get('training_runtime')
    require(type(spec) is dict and set(spec) == {'schema', 'owned_directory', 'max_json_bytes', 'artifact_read_bytes_upper'}
            and spec['schema'] == SPEC and raw._int(spec['max_json_bytes'], 1) and spec['max_json_bytes'] <= raw.MAX_JSON
            and raw._int(spec['artifact_read_bytes_upper'], 1), 'finite registered production publication scope')
    profile = owner.bootstrap.spec['operation_profiles']['publication']
    maximum = spec['artifact_read_bytes_upper']
    require(profile['read_bytes_upper'] >= maximum + 2 * spec['max_json_bytes'] + 2
            and profile['identity_hash_bytes_upper'] >= maximum + 4 * spec['max_json_bytes']
            and profile['identity_hash_attempts_upper'] >= 8 and profile['write_bytes_upper'] >= spec['max_json_bytes'],
            'whole donor/journal/publication IO/hash envelope before paths')
    with update._span(owner.bootstrap.operation, 'publication', dict(profile)) as returned:
        snapshot = validate_completion(journal.snapshot(), donor, context)
        invocation_id = journal.current_invocation
        require(invocation_id is not None and not journal.invocations[invocation_id]['ended'],
                'one active original invocation before production clock end')
        root = Path(context['bindings']['scope']['native_backend']['output_directory']).resolve()
        directory = Path(spec['owned_directory'])
        require(directory.resolve().is_relative_to(root) and not directory.is_symlink(), 'owned production completion directory')
        donor_path = Path(donor['path']); require(donor_path.resolve().is_relative_to(root), 'donor outside admitted production owner')
        references = {}; remaining = maximum
        for name, path, expected in (('donor', donor_path, donor['file_sha256']), ('header', journal.header_path, None), ('journal', journal.path, None)):
            reference = _stream_reference(path, remaining, expected); references[name] = reference; remaining -= reference['bytes']
        journal.end_invocation('COMPLETE')
        # End changes the journal: retain the actual ended bytes, not the live
        # prefix SHA read above. The second pass also requires prepaid capacity.
        end_size = journal.path.stat().st_size
        require(remaining >= end_size, 'ended journal second read exceeds original publication reservation')
        references['journal'] = _stream_reference(journal.path, remaining); remaining -= references['journal']['bytes']
        payload = {'schema': PUBLICATION, 'status': 'TRAINING_BODY_COMPLETE', 'body_completed': True,
            'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
            'static_recipe_sha256': context['static_recipe_sha256'], 'qualifiedreceipt_ref': context['refs']['qualifiedreceipt_ref'],
            'invocation_id': invocation_id, 'journal_snapshot': journal.snapshot(), 'donor': copy.deepcopy(donor), 'raw_refs': references,
            'source_service_bootstrap_snapshot': copy.deepcopy(service_snapshot), 'native_qualified': False,
            'os_terminal_verified': False, 'classification': 'actual source training body; separate terminal/cost consumer required; no learning claim'}
        blob = raw.canonical(payload) + b'\n'; require(len(blob) <= spec['max_json_bytes'], 'production completion JSON cap')
        directory.mkdir(parents=True, exist_ok=True); path = directory / 'TRAINING_BODY_COMPLETE.json'
        with path.open('xb') as stream: require(stream.write(blob) == len(blob), 'short production completion'); stream.flush(); os.fsync(stream.fileno())
        with path.open('rb') as stream: require(stream.read(len(blob) + 1) == blob, 'production completion exact readback')
        returned.update(bytes_read=maximum - remaining + len(blob), bytes_written=len(blob), identity_hash_attempts_actual=5,
                        identity_hash_bytes_actual=maximum - remaining + len(blob))
        return {'status': 'TRAINING_BODY_COMPLETE', 'native_qualified': False, 'os_terminal_verified': False,
                'invocation_id': invocation_id,
                'receipt_ref': {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}, 'raw_refs': references}


def completion_seal_role(context, result, journal, bootstrap, operation):
    permit = context.get('native_execution_permit') if type(context) is dict else None
    authority.require_execution_permit(permit, context, 'completion_publication')
    owner = _owner(operation)
    require(owner.context is context and bootstrap is owner.bootstrap and type(journal) is ce_journal.CeJournal
            and result.get('status') == 'TRAINING_BODY_COMPLETE' and journal.current_invocation is None
            and result.get('invocation_id') in journal.invocations
            and journal.invocations[result['invocation_id']]['ended'],
            'original source production body and ended invocation before seal')
    return {'status': 'UNQUALIFIED_TRAINING_BODY_SEAL', 'body_ref': copy.deepcopy(result['receipt_ref']),
        'journal_snapshot': journal.snapshot(), 'bootstrap_prefix': bootstrap.snapshot(),
        'native_qualified': False, 'os_terminal_verification_required': True,
        'claim': 'body bytes retained; external original worker terminal and final service-cost seal remain required'}


def interruption_role(context, error, journal, service_snapshot, operation):
    owner = _owner(operation)
    require(owner.context is context, 'original source production interruption owner')
    with update._span(owner.bootstrap.operation, 'interruption', dict(owner.bootstrap.spec['operation_profiles']['interruption'])):
        clock_failure = None
        if type(journal) is ce_journal.CeJournal and journal.current_invocation is not None:
            invocation = journal.invocations[journal.current_invocation]
            if not invocation['ended']:
                try: journal.end_invocation('INTERRUPTED')
                except BaseException as failure:
                    clock_failure = {'type': type(failure).__name__, 'message': str(failure)[:1000]}
        payload = {'status': 'TRAINING_INTERRUPTED', 'error': copy.deepcopy(error),
            'journal_snapshot': journal.snapshot() if type(journal) is ce_journal.CeJournal else None,
            'journal_clock_failure': clock_failure, 'source_service_bootstrap_snapshot': copy.deepcopy(service_snapshot),
            'native_qualified': False, 'costs_refunded': False}
        profile = owner.bootstrap.spec['operation_profiles']['interruption']
        blob = raw.canonical(payload) + b'\n'
        require(len(blob) <= raw.MAX_JSON and profile['write_bytes_upper'] >= len(blob)
                and profile['read_bytes_upper'] >= len(blob) + 1
                and profile['identity_hash_attempts_upper'] >= 1 and profile['identity_hash_bytes_upper'] >= len(blob),
                'whole interruption metadata envelope before publication')
        path = owner.bootstrap.directory / ('TRAINING_INTERRUPTION.' + uuid.uuid4().hex + '.json')
        with path.open('xb') as stream: require(stream.write(blob) == len(blob), 'short training interruption'); stream.flush(); os.fsync(stream.fileno())
        with path.open('rb') as stream: require(stream.read(len(blob) + 1) == blob, 'training interruption exact readback')
        payload['receipt_ref'] = {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
        return payload
