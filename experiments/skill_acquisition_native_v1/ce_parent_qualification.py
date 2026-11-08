"""Concrete parent qualification composition; native source registries stay closed.

This module creates existing source-owned classes and genuine setup tickets.
It supplies no callback selected by JSON, no fake artifact reference and no
experiment/production grant. Model/framework/OS work belongs to later existing
guarded bodies, never this metadata backend constructor.
"""
from __future__ import annotations

import copy
from pathlib import Path

from . import ce_native_authority as authority, ce_native_backend as backend
from . import ce_native_qualification as executor, ce_native_services as services
from . import ce_qualification_accounting as accounting, ce_bootstrap_accounting as boot
from . import ce_checkpoint_processes as processes, ce_checkpoint_qualification as checkpoint
from . import ce_checkpoint_lease as lease, ce_resource_evidence as resource
from . import ce_resource_qualification as measurement, ce_process_launch as launch
from . import ce_worker_evidence as worker, ce_qualification as qualification

# Exact source registration values only, never request fields or function args.
_BOOTSTRAP_ISSUER = None
_BOOTSTRAP_VERIFIER = None
_LAUNCH_ISSUER = None
_LAUNCH_VERIFIER = None
require, digest = authority.require, authority.digest


def _registered_services():
    boot.require_registered_verifier(_BOOTSTRAP_VERIFIER)
    require(_BOOTSTRAP_ISSUER is not None and any(_BOOTSTRAP_ISSUER is item for item in boot._REGISTERED_NATIVE_ISSUERS),
            'parent qualification bootstrap issuer remains unregistered')
    require(_LAUNCH_ISSUER is not None and any(_LAUNCH_VERIFIER is item for item in launch._REGISTERED_LAUNCH_VERIFIERS),
            'parent qualification child launch issuer/verifier remains unregistered')


def _service_owner(context):
    fields=context.get('native_services')
    require(type(fields) is dict and type(fields.get('operation')) is services.StableDispatcher
            and type(fields.get('gate')) is services.NativeResourceGate, 'actual parent stable dispatcher/resource gate required')
    service=getattr(fields['operation'],'service_owner',None)
    require(type(service) is services.NativeControllerServices and service.context is context
            and service.operation is fields['operation'] and service.gate is fields['gate'],
            'exact original NativeControllerServices owner binding required')
    service._admit()
    require(type(service.worker_handle) is worker.WinWorkerHandle and type(service.supervisor_lease) is lease.SupervisorGpuLease
            and service.supervisor_lease.context is context and service.supervisor_lease.operation is service.operation,
            'actual retained parent worker and already-owned supervisor lease required')
    return service


def _bootstrap(context,permit,proof):
    service=_service_owner(context); fields=context['native_services']
    require(context.get('worker_proof_sha256')==digest(proof), 'parent accounting receives the frozen actual worker proof')
    existing=fields.get('qualification_bootstrap')
    if existing is None:
        existing=boot.BootstrapAccounting(context,permit,proof,native_issuer=_BOOTSTRAP_ISSUER)
        fields['qualification_bootstrap']=existing
    require(type(existing) is boot.BootstrapAccounting and existing.context is context and existing.permit is permit
            and existing.proof==proof and not existing.fixture and not existing.closed,
            'exact live separately admitted parent bootstrap accounting required')
    require(service.operation.ledger is None or service.operation.ledger is fields.get('parent_qualification_accounting'),
            'parent bootstrap may not borrow another qualification owner')
    return existing


def reserve_parent_accounting(context,permit,proof,prepared_receipt):
    """Reserve the actual frozen constructor before any Q header/directory work."""
    authority.require_execution_permit(permit,context,'journal_construct')
    _registered_services()
    service=_service_owner(context)
    authority.require_execution_permit(permit,context,'journal_construct',operation=service.operation,gate=service.gate)
    require(service.operation.ledger is None and context['native_services'].get('parent_qualification_accounting') is None,
            'parent Q construction once before dispatcher attachment')
    qualification.validate_actual_ingestion(context,prepared_receipt)
    spec=accounting.accounting_specification(context)
    owner={'accounting_spec_sha256':digest(spec),'prepared_receipt_sha256':prepared_receipt['receipt_sha256']}
    require(context.get('qualifier_accounting_owner')==owner,
            'actual parent Q owner must equal frozen spec and actual prepared receipt')
    # These are genuine pre-model predecessor artifacts frozen before issuance.
    # They cannot be invented or inserted into immutable JSON after this point.
    for key in ('ingestion_bootstrap_ref','outer_setup_reservation_ref'):
        reference=spec.get(key)
        require(type(reference) is dict and authority._hex(reference.get('sha256'))
                and type(reference.get('path')) is str and Path(reference['path']).is_absolute()
                and type(reference.get('bytes')) is int and reference['bytes']>0,
                'actual frozen predecessor reference missing: '+key)
    outer=_bootstrap(context,permit,proof)
    bounds=accounting._price(spec['setup_bounds']); bounds['metadata_bytes_upper']=2*accounting.MAX_LINE
    handle=outer.reserve_setup(owner,bounds)
    try:
        result=accounting.create_accounting(context,permit,proof,prepared_receipt,
            bootstrap_handle=handle,bootstrap_verifier=_BOOTSTRAP_VERIFIER)
    except BaseException as primary:
        if handle.evidence['status'] in ('RESERVED','STARTED'):
            try: handle.fail(primary)
            except BaseException as failure:
                primary.parent_reservation_failure={'type':type(failure).__name__,'message':str(failure)[:1000]}
                raise primary from failure
        raise
    require(type(result) is accounting.QualificationAccounting and result.context is context and result.bootstrap_handle is handle,
            'actual native Q constructor must consume its source-owned reservation')
    context['native_services']['parent_qualification_accounting']=result
    context['native_services']['parent_prepared_receipt']=copy.deepcopy(prepared_receipt)
    return result


def _bind_processes(context,permit,prepared,core,owner,proof):
    service=_service_owner(context)
    require(type(core) is backend.NativeCEBackend and core.context is context and core.permit is permit
            and type(owner) is accounting.QualificationAccounting and owner.context is context
            and service.operation.ledger is owner and context['native_services'].get('parent_qualification_accounting') is owner,
            'actual attached Q owner and native backend before parent process construction')
    fields=context['native_services'].get('checkpoint_qualification')
    require(type(fields) is dict and fields.get('handoff') is service.supervisor_lease,
            'checkpoint processes must use the one existing supervisor lease')
    existing=fields.get('processes')
    if existing is not None:
        require(type(existing) is processes.CheckpointProcesses and existing.context is context and existing.accounting is owner,
                'parent checkpoint processes cannot change scope/account owner')
        return existing
    outer=_bootstrap(context,permit,proof); spec=context['bindings']['scope']['checkpoint_processes']
    expected={'accounting_spec_sha256':digest(spec),'prepared_receipt_sha256':prepared.receipt['receipt_sha256']}
    require(context['bindings']['scope'].get('checkpoint_processes_owner')==expected, 'exact parent process constructor owner')
    handle=outer.reserve_setup(expected,spec['bounds'])
    try:
        result=processes.CheckpointProcesses(context,permit,prepared.receipt,proof,backend=core,owner=owner,
            handoff=service.supervisor_lease,parent_handle=service.worker_handle,bootstrap_handle=handle,
            bootstrap_verifier=_BOOTSTRAP_VERIFIER,launch_issuer=_LAUNCH_ISSUER,launch_verifier=_LAUNCH_VERIFIER)
    except BaseException as primary:
        if handle.evidence['status'] in ('RESERVED','STARTED'):
            try: handle.fail(primary)
            except BaseException as failure:
                primary.parent_process_reservation_failure={'type':type(failure).__name__,'message':str(failure)[:1000]}
                raise primary from failure
        raise
    fields['processes']=result
    return result


def parent_checkpoint_adapter(context,permit,prepared,core,owner,proof,binding):
    authority.require_execution_permit(permit,context,'checkpoint_resume')
    _registered_services()
    _bind_processes(context,permit,prepared,core,owner,proof)
    return checkpoint.checkpoint_adapter(context,permit,prepared,core,owner,proof,binding)


def parent_resource_adapter(context,permit,prepared,core,owner,proof,binding):
    authority.require_execution_permit(permit,context,'resource_io_measurement')
    service=_service_owner(context)
    recorder=context['native_services'].get('resource_recorder')
    require(type(recorder) is resource.NativeResourceRecorder and recorder.context is context and recorder.permit is permit
            and recorder.operation is service.operation, 'actual shared parent resource recorder required')
    return measurement.resource_adapter(context,permit,prepared,core,owner,proof,binding)


def create_parent_qualification(context):
    """Fixed metadata factory; no imported framework, model load or OS query."""
    permit=context.get('native_execution_permit') if type(context) is dict else None
    authority.require_execution_permit(permit,context,'backend_construct')
    _registered_services()
    require(context['phase']=='qualification', 'parent composition cannot run production')
    service=_service_owner(context)
    authority.require_execution_permit(permit,context,'backend_construct',operation=service.operation,gate=service.gate)
    require(service.operation.ledger is None and service.supervisor_lease.backend is None,
            'pure parent backend factory after lease entry and before model/Q work')
    # Ingestion supplies actual barrier sharing; creating the recorder itself
    # imports no framework or samples resources in this metadata factory.
    recorder=service.bind_resource_recorder(context,permit)
    require(type(recorder) is resource.NativeResourceRecorder and recorder.context is context
            and recorder.permit is permit and recorder.operation is service.operation,
            'same source-owned recorder must precede parent model work')
    context['native_services']['resource_recorder']=recorder
    core=backend.NativeCEBackend(context,permit)
    try:
        result=executor.NativeQualificationExecutor(context,permit,core,
            {'accounting_factory':reserve_parent_accounting,'checkpoint_adapter':parent_checkpoint_adapter,
             'resource_adapter':parent_resource_adapter})
    except BaseException as primary:
        try: core.cleanup()
        except BaseException as cleanup:
            primary.parent_metadata_cleanup_failure={'type':type(cleanup).__name__,'message':str(cleanup)[:1000]}
            raise primary from cleanup
        raise
    return result
