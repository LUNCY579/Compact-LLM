"""Concrete source scope/worker adapters; no native registrations or execution.

The existing NativeControllerServices instance is the opaque issuer. Its
ControllerServicePermit admits bounded metadata before raw source admission;
NativeExecutionPermit is considered only after exact live worker context.
Raw request SHA, logical request preimage and derived setup refs are distinct.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import time
import uuid

from . import ce_native_services as services, ce_controller as controller
from . import ce_native_authority as authority, ce_worker_evidence as worker
from . import ce_qualification_artifacts as raw, ce_qualification_accounting as accounting
from . import ce_bootstrap_accounting as boot, ce_publication as publication

SPEC = 'CE8-SCOPE-ADAPTER-SPEC-1'
CAPSULE = 'CE8-SOURCE-BOOTSTRAP-CAPSULE-1'
SETUP = 'CE8-SOURCE-SETUP-PREPAYMENT-1'
METADATA_STAGE = 'qualification_metadata'
FULL_STAGES = frozenset(('controller_sources', 'qualification_metadata', 'ingestion_sources',
    'ingestion_tokenizer', 'ingestion_boundary', 'ingestion_qualification_rows'))
require, digest = authority.require, raw.digest


def _owner(operation):
    require(type(operation) is services.StableDispatcher, 'actual stable source service dispatcher required')
    owner = getattr(operation, 'service_owner', None)
    require(type(owner) is services.NativeControllerServices and owner.operation is operation,
            'exact original source service owner required')
    owner._admit()  # Existing closed service registry denies before IO/OS.
    return owner


def _policy(owner):
    policy = owner.spec.get('scope_adapters')
    require(type(policy) is dict and policy.get('schema') == SPEC and raw._hex(policy.get('static_recipe_sha256'))
            and type(policy.get('worker')) is dict and type(policy.get('owned_directory')) is str
            and Path(policy['owned_directory']).is_absolute(), 'explicit bounded source adapter policy')
    return policy


def read_request_metadata(request, reference, operation):
    """Paid exact raw request readback, arithmetic only; no authority is issued."""
    require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'}
            and type(reference['path']) is str and Path(reference['path']).is_absolute()
            and raw._hex(reference['sha256']) and raw._int(reference['bytes'], 1)
            and reference['bytes'] <= raw.MAX_JSON, 'bounded exact raw controller request reference')
    from . import update
    with update._span(operation, 'scope_metadata', {'read_bytes_upper': reference['bytes'] + 1,
            'identity_hash_attempts_upper': 2, 'identity_hash_bytes_upper': 2 * raw.MAX_JSON}) as event:
        path = Path(reference['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == reference['bytes'], 'regular exact raw controller request')
        with path.open('rb') as stream: blob = stream.read(reference['bytes'] + 1)
        require(len(blob) == reference['bytes'] and hashlib.sha256(blob).hexdigest() == reference['sha256'], 'raw controller request byte/hash drift')
        observed = raw._json(blob)
        require(observed == request, 'raw controller request differs from actual service request')
        logical = digest(controller.request_preimage(observed))
        event.update(bytes_read=len(blob), identity_hash_attempts_actual=2,
            identity_hash_bytes_actual=len(blob) + len(raw.canonical(controller.request_preimage(observed))))
        return {'request_ref': copy.deepcopy(reference), 'request_preimage_sha256': logical,
                'native_qualified': False, 'execution_verified': False}


def worker_specification(context, policy, facts, boot_identity):
    """Pure interpretation of actual OS facts supplied by the paid source body."""
    declared = copy.deepcopy(policy['worker'])
    require(not {'schema', 'mode', 'expected_pid', 'expected_creation_filetime_100ns', 'expected_boot_identity',
                 'request_sha256', 'scope_sha256'} & set(declared), 'worker policy excludes derived OS/logical identity')
    require(type(facts) is dict and type(facts.get('pid')) is int and facts['pid'] > 0
            and type(facts.get('creation_filetime_100ns')) is int and facts['creation_filetime_100ns'] > 0
            and facts.get('signaled') is False and facts.get('exit_code') == 259
            and facts.get('argv') == declared.get('argv')
            and Path(facts.get('image_path', '')) == Path(declared['process_image_ref']['path']), 'actual current worker image/whole argv/liveness')
    require(declared['approved_code_sha256'] == context['bindings']['source']['code_sha256']
            and declared['source_checkout'] == context['bindings']['scope']['native_backend']['source_checkout'],
            'worker exact source origins/admitted code map')
    result = dict(declared, schema=worker.SCOPE, mode='registered_metadata', expected_pid=facts['pid'],
        expected_creation_filetime_100ns=facts['creation_filetime_100ns'], expected_boot_identity=boot_identity,
        request_sha256=context['request_sha256'], scope_sha256=context['scope_sha256'])
    worker.validate_spec(result)
    return result


def outer_verifier(issuer, request_sha256, scope_sha256, phase):
    require(type(issuer) is services.NativeControllerServices, 'actual original service outer issuer required')
    issuer._admit()
    require(request_sha256 == digest(controller.request_preimage(issuer.request))
            and scope_sha256 == issuer.request['scope_ref']['sha256'] and phase == issuer.request['phase'] == 'qualification',
            'outer exact logical request/scope/qualification phase')
    return True


def outer_reservation(issuer, stage, refmap):
    require(type(issuer) is services.NativeControllerServices, 'actual original service reservation issuer required')
    outer_verifier(issuer, digest(controller.request_preimage(issuer.request)), issuer.request['scope_ref']['sha256'], issuer.request['phase'])
    bootstrap_only = issuer.context is None
    require(stage == METADATA_STAGE if bootstrap_only else stage in FULL_STAGES, 'outer stage outside source metadata/ingestion authority')
    declared = {key: issuer.request[key] for key in controller.REFS}
    require(type(refmap) is dict and all(value in declared.values() or key == 'arm' and value == issuer.request['arm']
            for key, value in refmap.items()), 'outer exact admitted reference map')
    from . import update
    profile = issuer.bootstrap.spec['operation_profiles']['outer_reservation']
    with update._span(issuer.operation, 'outer_reservation', dict(profile, stage=stage, refmap_sha256=digest(refmap))):
        return {'stage': stage, 'metadata_only': True, 'native_qualified': False}


def bootstrap_outer(request, operation, spec):
    owner = _owner(operation)
    require(request == owner.request and spec == owner.spec and owner.context is None,
            'original service metadata bootstrap before source admission')
    policy = _policy(owner)
    # This existing adapter restricts bootstrap to qualification metadata only.
    context = {'mode': 'registered_scope', 'static_recipe_sha256': policy['static_recipe_sha256']}
    adapter = controller.OuterAdmission(request, context, owner, outer_reservation, outer_verifier, bootstrap=True)
    read_request_metadata(request, policy['worker']['request_ref'], operation)
    return adapter


def worker_metadata_verifier(issuer, spec_sha256, request_sha256, scope_sha256):
    require(type(issuer) is services.NativeControllerServices, 'source service worker metadata issuer required')
    issuer._admit()
    bound = getattr(issuer, '_scope_worker_spec', None)
    require(type(bound) is dict and digest(bound) == spec_sha256 and bound['request_sha256'] == request_sha256
            and bound['scope_sha256'] == scope_sha256 and request_sha256 == digest(controller.request_preimage(issuer.request))
            and scope_sha256 == issuer.request['scope_ref']['sha256'], 'source-owned exact current worker metadata scope')
    return True


def worker_observation(context, operation):
    owner = _owner(operation); policy = _policy(owner)
    require(context.get('admission_evidence_verified') is True and context['request_sha256'] == digest(controller.request_preimage(owner.request))
            and context['scope_sha256'] == owner.request['scope_ref']['sha256']
            and context['static_recipe_sha256'] == policy['static_recipe_sha256'], 'raw source scope admitted before worker observation')
    # No arbitrary callback or proof dictionary substitutes for actual OS reads.
    budget = worker._Budget(policy['worker'], operation)
    api = budget.run('worker_os_binding', worker._WindowsOS); handle = None; primary = None
    try:
        def open_current():
            nonlocal handle
            handle = api.open(os.getpid()); return handle
        budget.run('worker_os_binding', open_current)
        facts = budget.run('worker_os_binding', lambda: api.snapshot(handle))
        boot_identity = budget.run('worker_os_binding', api.boot)
        require(facts['pid'] == os.getpid(), 'OS handle must describe this source service process')
        bound = budget.run('worker_source_verification', lambda: worker_specification(context, policy, facts, boot_identity))
        owner._scope_worker_spec = copy.deepcopy(bound)
        permit = worker.WorkerObservationPermit(bound, owner, worker_metadata_verifier, operation)
        return bound, permit
    except BaseException as error: primary = error; raise
    finally:
        if handle is not None:
            try: budget.run('worker_handle_close', lambda: api.close(handle))
            except BaseException as error:
                # Close remains source-authorized if observer admission rejects.
                try: api.close(handle)
                except BaseException as closing:
                    if primary is not None: primary.worker_close_failure = str(closing)[:1000]
                if primary is not None: raw._accounting(primary, error); raise primary from error
                raise


def _full_worker(owner, context, *, refresh=True):
    require(owner.context is context and type(owner.worker_handle) is worker.WinWorkerHandle,
            'actual original service complete context/retained worker handle')
    handle = owner.worker_handle
    require(handle._seal is worker._HANDLE_SEAL and handle.native is True and not handle.closed
            and handle.observer is owner.operation and handle.spec == getattr(owner, '_scope_worker_spec', None),
            'same source-owned retained native metadata handle/spec/observer')
    if refresh:
        proof = worker.verify_worker_handle(handle, {'request_sha256': context['request_sha256'],
            'scope_sha256': context['scope_sha256'], 'worker_identity_sha256': handle.identity_sha256}, require_native=True, terminal=False)
    else:
        # A permit recheck cannot create priced OS work through Q, which would
        # recursively recheck the same execution permit. Explicit service/OS
        # barriers refresh this handle; here inspect its already frozen origin.
        worker._admit(handle.spec, handle._permit, handle.observer)
        proof = owner.worker_proof
        require(digest(worker.worker_identity_preimage(handle.identity, handle.sources,
            context['request_sha256'], context['scope_sha256'])) == handle.identity_sha256 == proof['worker_identity_sha256'],
            'retained worker/source identity was mutated after observation')
    require(context.get('worker_proof_sha256') == digest(proof) and proof == owner.worker_proof
            and {name: ref['sha256'] for name, ref in proof['sources']['source_files'].items()} == context['bindings']['source']['code_sha256'],
            'complete actual worker/source context must be frozen before model authority')
    return proof


def native_authority_verifier(issuer, context_sha256, request_sha256, scope_sha256, phase, arm):
    require(type(issuer) is services.NativeControllerServices, 'actual original controller execution issuer required')
    issuer._admit(); context = issuer.context
    require(type(context) is dict and digest(authority.context_preimage(context)) == context_sha256
            and context['request_sha256'] == request_sha256 and context['scope_sha256'] == scope_sha256
            and context['phase'] == phase == 'qualification' and context['arm'] == arm,
            'full admitted source/worker context identity before native execution')
    _full_worker(issuer, context, refresh=False)
    return True


def _write_capsule(path, value, operation):
    """A fixed metadata artifact; all work stays under original service prices."""
    from . import update
    with update._span(operation, 'scope_capsule', {'read_bytes_upper': 2 * raw.MAX_JSON + 2,
            'write_bytes_upper': raw.MAX_JSON, 'identity_hash_attempts_upper': 1,
            'identity_hash_bytes_upper': raw.MAX_JSON}) as result:
        blob = raw.canonical(value) + b'\n'; require(len(blob) <= raw.MAX_JSON, 'bounded source capsule JSON')
        path = Path(path); require(path.is_absolute() and not path.is_symlink() and not os.path.lexists(path), 'create-only source capsule')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / (path.name + '.' + uuid.uuid4().hex + '.partial')
        with temporary.open('xb') as stream:
            require(stream.write(blob) == len(blob), 'short source capsule write'); stream.flush(); os.fsync(stream.fileno())
        with temporary.open('rb') as stream: observed = stream.read(len(blob) + 1)
        require(observed == blob, 'source capsule temporary readback')
        publication._atomic_no_replace(temporary, path, path.parent)
        with path.open('r+b') as stream: os.fsync(stream.fileno()); observed = stream.read(len(blob) + 1)
        require(observed == blob, 'source capsule final readback')
        result.update(bytes_read=2 * len(blob), bytes_written=len(blob), identity_hash_attempts_actual=1, identity_hash_bytes_actual=len(blob))
        return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def read_setup_capsules(context, operation):
    """Read exact derived refs and linkage; this is not execution authority."""
    refs = context.get('qualifier_setup_refs')
    require(type(refs) is dict and set(refs) == {'ingestion_bootstrap_ref', 'outer_setup_reservation_ref'}, 'exact derived source setup refs')
    values = {}
    from . import update
    for key, ref in refs.items():
        require(type(ref) is dict and set(ref) == {'path', 'sha256', 'bytes'} and raw._hex(ref['sha256'])
                and type(ref['path']) is str and Path(ref['path']).is_absolute()
                and raw._int(ref['bytes'], 1) and ref['bytes'] <= raw.MAX_JSON, 'bounded actual source setup capsule reference')
        with update._span(operation, 'scope_capsule', {'read_bytes_upper': ref['bytes'] + 1,
                'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': ref['bytes']}) as record:
            path = Path(ref['path']); require(path.is_file() and not path.is_symlink() and path.stat().st_size == ref['bytes'], 'regular retained source setup capsule')
            with path.open('rb') as stream: blob = stream.read(ref['bytes'] + 1)
            require(len(blob) == ref['bytes'] and hashlib.sha256(blob).hexdigest() == ref['sha256'], 'source setup capsule byte/hash drift')
            value = raw._json(blob)
            require(type(value) is dict and raw.canonical(value) + b'\n' == blob, 'canonical source setup capsule')
            require(value.get('request_sha256') == context['request_sha256'] and value.get('scope_sha256') == context['scope_sha256']
                    and value.get('static_recipe_sha256') == context['static_recipe_sha256']
                    and value.get('code_sha256') == context['bindings']['source']['code_sha256']
                    and value.get('worker_proof_sha256') == context['worker_proof_sha256'], 'source setup capsule exact raw scope/source/worker linkage')
            values[key] = value
            record.update(bytes_read=len(blob), identity_hash_attempts_actual=1, identity_hash_bytes_actual=len(blob))
    source, setup = values['ingestion_bootstrap_ref'], values['outer_setup_reservation_ref']
    payment = setup.get('constructor_prepayment')
    require(source.get('schema') == CAPSULE and setup.get('schema') == SETUP
            and source.get('service_bootstrap_schema') == services.BOOTSTRAP
            and type(source.get('service_bootstrap_snapshot')) is dict
            and type(payment) is dict
            and set(payment) == {'schema', 'reservation_ref', 'bounds', 'request_sha256', 'service_spec_sha256',
                'bootstrap_spec_sha256', 'consumed', 'metadata_only', 'native_qualified', 'execution_verified'}
            and payment['schema'] == 'CE8-SERVICE-CONSTRUCTOR-PREPAYMENT-1'
            and payment['request_sha256'] == context['request_sha256'] and payment['consumed'] is True
            and payment['metadata_only'] is True and payment['native_qualified'] is False and payment['execution_verified'] is False
            and source.get('constructor_prepayment') == payment,
            'actual matching original service bootstrap/prepayment capsule schemas')
    reference = payment['reservation_ref']
    require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'} and raw._hex(reference['sha256'])
            and type(reference['path']) is str and Path(reference['path']).is_absolute()
            and raw._int(reference['bytes'], 1) and reference['bytes'] <= raw.MAX_JSON, 'actual original constructor reservation ref')
    with update._span(operation, 'scope_capsule', {'read_bytes_upper': reference['bytes'] + 1,
            'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': reference['bytes']}):
        path = Path(reference['path']); require(path.is_file() and not path.is_symlink() and path.stat().st_size == reference['bytes'], 'regular original constructor reservation')
        with path.open('rb') as stream: blob = stream.read(reference['bytes'] + 1)
        require(len(blob) == reference['bytes'] and hashlib.sha256(blob).hexdigest() == reference['sha256'], 'original constructor reservation bytes/hash drift')
        reserved = raw._json(blob)
        require(raw.canonical(reserved) + b'\n' == blob and reserved.get('schema') == payment['schema']
                and reserved.get('status') == 'PERMANENTLY_RESERVED'
                and all(reserved.get(key) == payment[key] for key in ('request_sha256', 'service_spec_sha256', 'bootstrap_spec_sha256', 'bounds'))
                and reserved.get('metadata_only') is True and reserved.get('native_qualified') is False,
                'actual original permanent constructor reservation linkage')
        snapshot = source['service_bootstrap_snapshot']
        require(snapshot.get('schema') == services.BOOTSTRAP and snapshot.get('request_sha256') == context['request_sha256']
                and snapshot.get('fatal') is False and raw._hex(snapshot.get('head'))
                and snapshot.get('native_qualified') is False and type(snapshot.get('cumulative')) is dict
                and set(snapshot['cumulative']) == set(services.COUNTERS), 'actual original service bootstrap snapshot schema/costs')
    return values


def materialize_setup_refs(owner, context, proof):
    """Produce real runtime refs without changing any raw scope/request bytes."""
    require(type(owner) is services.NativeControllerServices and owner.context is context, 'original full service capsule owner')
    owner._admit(); policy = _policy(owner)
    require(context.get('qualifier_setup_refs') is None, 'actual setup refs are frozen once before native permit')
    from . import ce_service_prepayment as prepaid, update
    payment = owner.constructor_prepayment
    require(type(payment) is prepaid.ServiceConstructorPrepayment
            and any(type(payment) is item for item in services._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES), 'actual consumed service constructor prepayment type')
    profile = owner.bootstrap.spec['operation_profiles']['scope_capsule']
    with update._span(owner.operation, 'scope_capsule', dict(profile, reason='source_setup_ref_materialization')):
        evidence = copy.deepcopy(payment.evidence)
        require(payment.consumed is True and evidence.get('consumed') is True
                and payment.request_sha256 == context['request_sha256']
                and payment.service_spec_sha256 == digest(owner.spec)
                and payment.bootstrap_spec_sha256 == digest(owner.bootstrap.spec), 'original actually consumed constructor reservation/spec/request')
        common = {'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
            'static_recipe_sha256': context['static_recipe_sha256'], 'code_sha256': context['bindings']['source']['code_sha256'],
            'worker_proof_sha256': digest(proof), 'constructor_prepayment': evidence,
            'native_qualified': False, 'execution_verified': False}
        directory = Path(policy['owned_directory'])
        require(directory.is_relative_to(Path(owner.spec['bootstrap']['owned_directory'])), 'source capsules under original service owner')
        source = dict(common, schema=CAPSULE, service_bootstrap_schema=services.BOOTSTRAP,
            service_bootstrap_snapshot=owner.bootstrap.snapshot(),
            classification='actual retained metadata reservations/source admission; no tokenizer/model execution claim')
        setup = dict(common, schema=SETUP, classification='original consumed metadata constructor prepayment; separate Q setup ticket is issued later')
        refs = {'ingestion_bootstrap_ref': _write_capsule(directory / 'INGESTION_BOOTSTRAP.json', source, owner.operation),
            'outer_setup_reservation_ref': _write_capsule(directory / 'OUTER_SETUP_RESERVATION.json', setup, owner.operation)}
        context['qualifier_setup_refs'] = refs
        read_setup_capsules(context, owner.operation)
        derived = accounting.accounting_specification(context)
        context['qualifier_accounting_owner'] = {'accounting_spec_sha256': digest(derived),
            'prepared_receipt_sha256': derived['ingestion_receipt_sha256']}
    return refs


def outer_issuer(request, context, operation):
    owner = _owner(operation)
    require(request == owner.request and owner.context is context, 'exact fully admitted service outer context')
    _full_worker(owner, context)
    if context.get('qualifier_setup_refs') is None:
        materialize_setup_refs(owner, context, owner.worker_proof)
    if context.get('native_execution_permit') is None:
        context['native_execution_permit'] = authority.NativeExecutionPermit(context, owner, native_authority_verifier,
            operation=operation, gate=owner.gate)
    authority.require_execution_permit(context['native_execution_permit'], context, 'ingestion_tokenizer',
        operation=operation, gate=owner.gate)
    return owner


def bootstrap_native_issuer(context, permit, proof):
    """Reviewed source identity for existing BootstrapAccounting construction."""
    authority.require_execution_permit(permit, context, 'journal_construct')
    owner = _owner(context['native_services']['operation'])
    require(_full_worker(owner, context) == proof, 'bootstrap exact complete worker execution context')
    return owner


def bootstrap_prepayment_verifier(handle, context, proof, expected_owner):
    require(type(context) is dict, 'complete existing bootstrap execution context required')
    authority.require_execution_permit(context.get('native_execution_permit'), context, 'journal_construct')
    service = _owner(context['native_services']['operation']); actual = _full_worker(service, context)
    require(type(handle) is boot.PrepaymentHandle and handle._native is True
            and type(handle._ledger) is boot.BootstrapAccounting and handle._ledger.context is context
            and handle._ledger.proof == proof == actual and handle.evidence['owner'] == expected_owner
            and handle.evidence['status'] == 'RESERVED' and not handle._ledger.closed
            and not handle._ledger.unknown_wall and not handle._ledger.fatal,
            'existing actual source prepaid bootstrap ticket/current worker owner')
    return True
