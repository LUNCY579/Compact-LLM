"""Concrete source/teacher/tokenizer entry preparation; native issuers closed.

This factory revalidates full canonical/policy TRAIN and the chosen 3000-row
teacher, then renders the pinned tokenizer census through ce_ingestion. It
creates no model capability, C/U progress or native qualification. The existing
outer metadata verifier and service factory registries are deliberately empty.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

from . import ce_controller as controller, ce_native_services as services
from . import ce_worker_evidence as worker, ce_ingestion as ingestion

FIXTURE = 'CE8-ENTRY-PREPARATION-FIXTURE-1'
_AUTHORITY_SEAL = object()
STAGES = frozenset(('ingestion_sources', 'ingestion_tokenizer', 'ingestion_boundary', 'ingestion_qualification_rows'))
require, digest = worker.require, worker.digest


class _RowsMetadataAuthority:
    """Fixed wrapper of existing service/OS authorities, never a model permit."""
    __slots__ = ('_seal', 'context', 'request', 'spec', 'permit', 'handle', 'observation', 'operation', 'gate', 'profile')
    def __init__(self, common, permit, operation, gate, proof, request, service_spec, handle, observation):
        services._admit(permit, request, service_spec)
        require(type(operation) is services.StableDispatcher and type(gate) is services.NativeResourceGate
                and type(handle) is worker.WinWorkerHandle and type(observation) is worker.WorkerObservationPermit,
                'actual stable entry dispatcher/resource gate/OS observation authority required')
        observation.require(handle.spec, operation)
        actual = worker.verify_worker_handle(handle, {key: proof[key] for key in
            ('worker_identity_sha256', 'request_sha256', 'scope_sha256')}, require_native=True)
        current_pid = handle.budget.run('worker_os_binding', os.getpid)
        require(actual == proof and actual['os_identity']['pid'] == current_pid
                and common.get('mode') == 'registered_scope' and request['phase'] == common['phase'] == 'qualification'
                and request['arm'] == common['arm'] and proof['request_sha256'] == common['request_sha256']
                and proof['scope_sha256'] == common['scope_sha256'] == request['scope_ref']['sha256']
                and common['request_sha256'] == digest(controller.request_preimage(request))
                and common['bindings']['source']['code_sha256'] == service_spec['approved_code_sha256']
                and gate.spec == service_spec['resource'], 'entry metadata scope/request/source/worker/callback binding')
        profile = services._profile(service_spec['bootstrap']['operation_profiles']['input_io'])
        for key, value in (('_seal', _AUTHORITY_SEAL), ('context', copy.deepcopy(common)), ('request', copy.deepcopy(request)),
                ('spec', copy.deepcopy(service_spec)), ('permit', permit), ('handle', handle), ('observation', observation),
                ('operation', operation), ('gate', gate), ('profile', copy.deepcopy(profile))):
            object.__setattr__(self, key, value)
    def __setattr__(self, key, value): raise worker.WorkerEvidenceError('entry metadata authority is immutable')


def _verify_entry_outer_authority(value, request_sha256, scope_sha256, phase):
    """Existing outer-registry symbol; returns only after real metadata checks."""
    require(type(value) is _RowsMetadataAuthority and getattr(value, '_seal', None) is _AUTHORITY_SEAL,
            'source-owned exact entry metadata authority required')
    services._admit(value.permit, value.request, value.spec)
    require(request_sha256 == digest(controller.request_preimage(value.request)) == value.context['request_sha256']
            and scope_sha256 == value.context['scope_sha256'] and phase == 'qualification', 'entry outer metadata scope changed')
    value.observation.require(value.handle.spec, value.operation)
    actual = worker.verify_worker_handle(value.handle, {key: value.context[key] for key in ('request_sha256', 'scope_sha256')}
        | {'worker_identity_sha256': value.handle.identity_sha256}, require_native=True)
    require(actual['sources']['source_files'] == value.handle.sources['source_files'], 'entry observed source bundle changed')
    return True


def _reserve_entry_outer(value, stage, refs):
    require(type(value) is _RowsMetadataAuthority and stage in STAGES, 'entry outer may admit only source/tokenizer/row metadata')
    ingestion._observe(value.operation, value.gate, stage, dict(value.profile), lambda: None)


def _check_factory_origin(value):
    name = __name__.replace('.', '/') + '.py'
    reference = value.handle.sources['source_files'].get(name)
    require(type(reference) is dict and reference['sha256'] == value.context['bindings']['source']['code_sha256'].get(name),
            'concrete entry factory missing from complete observed approved source map')
    def verify():
        require(Path(__file__).resolve() == Path(reference['path']).resolve()
                == (Path(value.handle.spec['source_checkout']) / name).resolve(), 'entry factory running source origin drift')
    ingestion._observe(value.operation, value.gate, 'ingestion_sources', dict(value.profile), verify)
    worker._file(reference, value.handle.budget)


def _prepare(common, request, outer, operation, gate, *, tokenizer=None, renderer=None):
    controller.require_preparation_cuda_uninitialized()
    verified = ingestion.verify_sources(request['preparation_ref'], request['plan_ref'], request['arm'], outer,
                                        operation=operation, gate=gate)
    prepared = ingestion.PreparedRows(verified, tokenizer, outer_admission=outer, operation=operation, gate=gate, renderer=renderer)
    controller.require_preparation_cuda_uninitialized()
    require(type(prepared) is ingestion.PreparedRows and prepared.receipt == common['prepared_receipt'],
            'fresh full TRAIN/teacher/tokenizer/row-hash/census receipt differs from retained entry receipt')
    return prepared


def prepare_checkpoint_entry_rows(common, service_permit, operation, gate, proof, *, request, service_spec,
                                  worker_handle, observation_permit):
    """Registered source role ABI; no caller-selected tokenizer or renderer."""
    # Reject before context/permit/proof/path traversal, source work or imports.
    require(any(_verify_entry_outer_authority is value for value in controller._REGISTERED_OUTER_AUTHORITY_VERIFIERS),
            'entry outer metadata verifier remains unregistered')
    authority = _RowsMetadataAuthority(common, service_permit, operation, gate, proof, request, service_spec,
                                       worker_handle, observation_permit)
    _check_factory_origin(authority)
    outer = controller.OuterAdmission(request, common, authority, _reserve_entry_outer, _verify_entry_outer_authority)
    return _prepare(common, request, outer, operation, gate)


def fixture_prepare_entry_rows(context, common, request, outer, operation, gate, *, tokenizer, renderer):
    require(context == {'schema': FIXTURE, 'native_qualified': False, 'execution_verified': False}
            and common.get('mode') == 'synthetic_fixture', 'explicit non-native entry preparation fixture')
    return _prepare(common, request, outer, operation, gate, tokenizer=tokenizer, renderer=renderer)
