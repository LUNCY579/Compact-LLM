"""Shared dormant CE8 process/lifecycle coordinator; no native registration/grant.

Qualification and training are distinct authorized scopes. The current proposal
route refuses before IO/imports/writes. A future registered backend must supply
the listed capabilities and actually qualify the delayed native implementation.
Service/backend injection tests sequencing, not native execution or efficacy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
import uuid

SCHEMA = 'CE8-CONTROLLER-REQUEST-1'
CONTEXT = 'CE8-OUTER-ADMISSION-CONTEXT-1'
OUTER = 'CE8-OUTER-ADMISSION-1'
REFS = ('selectedoperatorchoice_ref', 'recipe_ref', 'scope_ref', 'source_ref', 'runtime_ref',
        'anchors_ref', 'evaluation_ref', 'preparation_ref', 'plan_ref', 'ingestion_ref',
        'resources_ref', 'attempt_caps_ref', 'scope_authorization_ref')
BOUND_FIELDS = ('operation_attempts_upper', 'io_read_bytes_upper', 'io_write_bytes_upper',
                'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'metadata_bytes_upper',
                'wall_seconds_upper')
QUALIFICATION_STAGES = ('ingestion_tokenizer', 'runtime_anchor', 'longest_masks',
    'likelihood_gradient_parity', 'adam_clipping', 'checkpoint_resume', 'delta_roundtrip',
    'resource_io_measurement')
ACTIVATION_GAPS = {
    'native_ce8_update': 'native core requires an opaque permit; issuer registry remains empty and parity is unqualified',
    'native_ce8_checkpoint': 'delayed NativeCECheckpoints requires a permit and fresh-process CUDA restore qualification',
    'native_ce8_journal_bootstrap': 'native ledger artifact, stable dispatcher and admitted prepared census require registered verification',
    'outer_setup_hash_pricing': 'native setup/source/tokenizer/hash work needs bound admission and measured production prices',
    'fresh_process_boot_resources': 'verified OS boot identity, fresh worker, lease, host/GPU/disk guards and watchdog must be bound',
    'full_source_ingestion': 'canonical/policy TRAIN, label plans and actual token census must be revalidated',
    'owned_seal_rollback': 'native checkpoint ancestry/fresh fences, interruption artifacts and completion cost seals need registered adapters',
}
# Populated only by a future reviewed source change registering actual native
# implementations and their scope/source verifier. Request JSON cannot do it.
_REGISTERED_NATIVE_FACTORIES = frozenset()
# Opaque capabilities are meaningful only under a reviewed, source-owned
# issuer/verifier. A caller's object plus an always-true callback is a fixture,
# and cannot register a native authority or open tokenizer/framework work.
_REGISTERED_OUTER_AUTHORITY_VERIFIERS = frozenset()


class AdmissionError(ValueError): pass
class MissingCapability(AdmissionError): pass


def require(condition, message):
    if not condition: raise AdmissionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def require_preparation_cuda_uninitialized():
    """Safe initialization query only; never import Torch or start CUDA here."""
    framework = sys.modules.get('torch')
    if framework is not None:
        require(framework.cuda.is_initialized() is False,
                'preparation initialized CUDA before the sole backend configuration owner')


def checkpoint_boundaries(execution):
    require(type(execution) is dict and type(execution.get('boundaries')) is int
            and execution['boundaries'] == 375, 'frozen 3000-row/375-boundary CE8 execution recipe')
    schedule = execution.get('checkpoint_boundaries')
    require(type(schedule) is list and schedule and all(type(value) is int and 0 <= value <= 375 for value in schedule)
            and schedule == sorted(set(schedule)) and schedule[0] == 0 and schedule[-1] == 375,
            'explicit sorted checkpoint boundaries including C0 and final C375')
    return tuple(schedule)
def _hex(value): return type(value) is str and len(value) == 64 and not set(value) - set('0123456789abcdef')
def request_preimage(request): return {k: v for k, v in request.items() if k != 'scope_authorization_ref'}


def _ref(value):
    require(type(value) is dict and type(value.get('path')) is str and value['path']
            and _hex(value.get('sha256')), 'explicit path/SHA256 reference required')


def _bounds(value):
    require(type(value) is dict and set(value) == set(BOUND_FIELDS), 'complete inline bootstrap bounds required')
    for key in BOUND_FIELDS[:-1]: require(type(value[key]) is int and value[key] >= 0, 'strict finite count: ' + key)
    require(value['operation_attempts_upper'] > 0 and type(value['wall_seconds_upper']) in (int, float)
            and math.isfinite(value['wall_seconds_upper']) and value['wall_seconds_upper'] > 0, 'finite positive bootstrap wall/attempt bound')


def build_request(phase, arm, refs, bootstrap_bounds):
    """Build reviewable unapproved request data, never choose a seed or price."""
    require(phase in ('qualification', 'training') and arm in ('canonical', 'permuted'), 'phase/arm')
    require(type(refs) is dict and set(refs) <= set(REFS) | {'qualifiedreceipt_ref'}, 'unknown request reference')
    _bounds(bootstrap_bounds)
    for value in refs.values():
        if value is not None: _ref(value)
    return {'schema': SCHEMA, 'mode': 'unapproved_proposal', 'phase': phase, 'arm': arm,
            'status': 'UNAPPROVED_PROPOSAL', 'approved': False, 'qualification_authorized': False,
            'training_authorized': False, 'bootstrap_bounds': copy.deepcopy(bootstrap_bounds),
            **{key: copy.deepcopy(refs.get(key)) for key in REFS},
            'qualifiedreceipt_ref': copy.deepcopy(refs.get('qualifiedreceipt_ref'))}


def validate_request(request, *, synthetic_fixture=False):
    require(type(request) is dict and request.get('schema') == SCHEMA, 'controller request schema')
    require(request.get('phase') in ('qualification', 'training') and request.get('arm') in ('canonical', 'permuted'), 'controller phase/arm')
    if synthetic_fixture:
        require(request.get('mode') == 'synthetic_controller_fixture'
                and request.get('approved') is False and request.get('training_authorized') is False
                and request.get('qualification_authorized') is False, 'explicit fixture is not native authority')
    else:
        # This refusal precedes path handling, service callbacks and imports.
        require(request.get('approved') is True and request.get('mode') == 'registered_scope', 'unapproved CE controller route denied before IO')
        if request['phase'] == 'qualification':
            require(request.get('qualification_authorized') is True and request.get('training_authorized') is False
                    and request.get('status') == 'AUTHORIZED_QUALIFICATION_ONLY', 'separate qualification-only authorization')
        else:
            require(request.get('training_authorized') is True and request.get('status') == 'AUTHORIZED_TRAINING'
                    and request.get('qualification_authorized') is False, 'separate production training authorization')
            _ref(request.get('qualifiedreceipt_ref'))
    _bounds(request.get('bootstrap_bounds'))
    for key in REFS: _ref(request.get(key))
    canonical(request)
    return request


def validate_outer(context, request):
    require(type(context) is dict and context.get('schema') == OUTER and context.get('admitted') is True
            and context.get('mode') in ('synthetic_fixture', 'registered_scope')
            and context.get('phase') == request['phase'] and context.get('arm') == request['arm']
            and context.get('request_sha256') == digest(request_preimage(request))
            and context.get('scope_sha256') == request['scope_ref']['sha256']
            and context.get('source_sha256') == request['source_ref']['sha256']
            and context.get('runtime_sha256') == request['runtime_ref']['sha256']
            and all(_hex(context.get(key)) for key in ('scope_sha256', 'static_recipe_sha256', 'source_sha256', 'runtime_sha256')),
            'outer admission binding drift')
    return context


_PERMIT_SEAL = object()


class NativeStagePermit:
    """Opaque permit for a scoped metadata/source/tokenizer stage, not model work."""
    __slots__ = ('_stage', '_refs', '_request', '_scope', '_qualification_rows')
    def __init__(self, stage, refs, request_hash, scope_hash, *, seal, qualification_rows=None):
        require(seal is _PERMIT_SEAL, 'permits are issued only by the bound outer adapter')
        self._stage, self._refs, self._request, self._scope = stage, digest(refs), request_hash, scope_hash
        self._qualification_rows = copy.deepcopy(qualification_rows)
    def matches(self, stage, refs, request_hash, scope_hash):
        return (stage, digest(refs), request_hash, scope_hash) == (self._stage, self._refs, self._request, self._scope)
    @property
    def qualification_rows(self):
        require(self._stage == 'ingestion_qualification_rows' and type(self._qualification_rows) is dict,
                'permit has no qualification row declaration')
        return copy.deepcopy(self._qualification_rows)


def qualification_row_declaration(context):
    """Pure interpretation of frozen scope, never an execution grant."""
    require(type(context) is dict and context.get('phase') == 'qualification', 'qualification-only row scope')
    execution = context.get('bindings', {}).get('scope', {}).get('qualifier_execution')
    require(type(execution) is dict, 'explicit frozen qualification execution rows required')
    witnesses, gradient, batches = (execution.get(key) for key in ('all_witness_ids', 'gradient_batch_ids', 'adam_batches'))
    valid_ids = lambda value: type(value) is list and all(type(rid) is str and rid for rid in value) and len(set(value)) == len(value)
    require(valid_ids(witnesses) and len(witnesses) == 90 and valid_ids(gradient) and len(gradient) == 8
            and set(gradient) <= set(witnesses) and type(batches) is list and 2 <= len(batches) <= 8
            and all(valid_ids(batch) and len(batch) == 8 and set(batch) <= set(witnesses) for batch in batches),
            'exact all-witness/gradient/Adam row declarations')
    declaration = {'schema': 'CE8-QUALIFICATION-ROW-DECLARATION-1', 'scope_sha256': context.get('scope_sha256'),
                   'request_sha256': context.get('request_sha256'), 'witness_row_ids': copy.deepcopy(witnesses),
                   'gradient_row_ids': copy.deepcopy(gradient), 'adam_batches': copy.deepcopy(batches)}
    return dict(declaration, declaration_sha256=digest(declaration))


class OuterAdmission:
    """Concrete stage/refmap adapter shared by qualification and ingestion.

    ``authority`` is a non-JSON service-issued capability. Its verifier is a
    trusted registered service, never an input approval boolean. The returned
    dictionary is an accounting receipt only and supplies no model permission.
    Bootstrap instances admit only bounded qualification metadata reads.
    """
    def __init__(self, request, context, authority, reserve, verify_authority, *, bootstrap=False):
        require(authority is not None and not isinstance(authority, (dict, list, str, int, float, bool)),
                'opaque scope capability required, not JSON flags')
        require(callable(reserve) and callable(verify_authority), 'bound reservation/authority services required')
        require(type(context) is dict and context.get('mode') in ('synthetic_fixture', 'registered_scope'), 'outer mode')
        from .ce_native_authority import context_preimage
        self.request, self.context, self.authority = copy.deepcopy(request), copy.deepcopy(context_preimage(context)), authority
        self.reserve, self.verify_authority, self.bootstrap = reserve, verify_authority, bootstrap
        self._verify_issuer()

    def _verify_issuer(self):
        if self.context.get('mode') == 'registered_scope':
            require(self.request.get('mode') == 'registered_scope'
                    and any(self.verify_authority is value for value in _REGISTERED_OUTER_AUTHORITY_VERIFIERS),
                    'no registered native outer authority verifier; fixture callbacks cannot issue native permits')

    def __call__(self, stage, refmap):
        self._verify_scope(stage, refmap)
        self.reserve(self.authority, stage, copy.deepcopy(refmap))  # completed permanent preflight before caller work
        receipt = {'schema': OUTER, 'admitted': True, 'mode': self.context['mode'],
                   'phase': self.request['phase'], 'arm': self.request['arm'],
                   'request_sha256': digest(request_preimage(self.request)),
                   'scope_sha256': self.request['scope_ref']['sha256'],
                   'source_sha256': self.request['source_ref']['sha256'],
                   'runtime_sha256': self.request['runtime_ref']['sha256'],
                   'static_recipe_sha256': self.context['static_recipe_sha256']}
        return validate_outer(receipt, self.request)

    def _verify_scope(self, stage, refmap):
        self._verify_issuer()  # also rejects promotion of a constructed fixture
        require(type(stage) is str and stage and type(refmap) is dict, 'one explicit stage/refmap callback')
        require(not self.bootstrap or stage == 'qualification_metadata', 'bootstrap has no framework/model authority')
        require(self.verify_authority(self.authority, digest(request_preimage(self.request)),
                self.request['scope_ref']['sha256'], self.request['phase']) is True, 'scope capability verification failed')
        declared = {key: self.request[key] for key in REFS}
        require(all(value in declared.values() or key == 'arm' and value == self.request['arm']
                    for key, value in refmap.items()), 'reference outside admitted request')

    def native_permit(self, stage, refmap):
        self._verify_scope(stage, refmap)
        require(self.context['mode'] == 'registered_scope' and stage in
                ('qualification_metadata', 'ingestion_sources', 'ingestion_tokenizer', 'ingestion_boundary', 'ingestion_qualification_rows'),
                'outer permit cannot authorize model work')
        rows = qualification_row_declaration(self.context) if stage == 'ingestion_qualification_rows' else None
        return NativeStagePermit(stage, refmap, digest(request_preimage(self.request)),
                                 self.request['scope_ref']['sha256'], seal=_PERMIT_SEAL, qualification_rows=rows)


def build_backend_command(request, context, request_ref, *, resume=False, reason=None, synthetic_fixture=False):
    """Pure command data only. No subprocess, shell, environment or module default."""
    validate_request(request, synthetic_fixture=synthetic_fixture)
    require(synthetic_fixture or any(context.get('backend_factory') is value for value in _REGISTERED_NATIVE_FACTORIES),
            'no registered native worker exists; command is not qualified')
    _ref(request_ref)
    require(context.get('request_file_sha256') == request_ref['sha256'], 'verified raw command request file identity')
    worker = context.get('worker') if type(context) is dict else None
    require(type(worker) is dict and worker.get('registered_source_verified') is True
            and type(worker.get('python_executable')) is str and worker['python_executable']
            and _hex(worker.get('python_sha256')) and type(worker.get('checkout')) is str and worker['checkout']
            and type(worker.get('module')) is str and re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', worker['module'])
            and _hex(worker.get('module_source_sha256')) and type(worker.get('environment')) is dict
            and all(type(k) is str and type(v) is str for k, v in worker['environment'].items()), 'explicit verified fresh-worker source/runtime required')
    require(context.get('request_sha256') == digest(request_preimage(request)), 'worker request binding drift')
    require(type(resume) is bool and (not resume or type(reason) is str and reason.strip()), 'documented resume reason')
    args = [worker['python_executable'], '-m', worker['module'], '--request', request_ref['path'],
            '--phase', request['phase'], '--arm', request['arm']]
    if resume: args.extend(['--resume', '--resume-reason', reason])
    return {'argv': args, 'cwd': worker['checkout'], 'environment': copy.deepcopy(worker['environment']),
            'fresh_process_required': True, 'request_sha256': request_ref['sha256'], 'dispatched': False,
            'synthetic_fixture': synthetic_fixture, 'qualified_native': False}


def _stage(observer, name, callback, request_hash, **fields):
    identifier = uuid.uuid4().hex
    observer(name, 'started', operation_id=identifier, request_sha256=request_hash, **fields)
    try: value = callback()
    except BaseException as error:
        try: observer(name, 'failed', operation_id=identifier, request_sha256=request_hash,
                      error={'type': type(error).__name__, 'message': str(error)[:1000]}, **fields)
        except BaseException as record_error:
            error.accounting_failure = {'type': type(record_error).__name__, 'message': str(record_error)[:1000]}
        raise
    observer(name, 'returned', operation_id=identifier, request_sha256=request_hash, **fields)
    return value


class _OwnedContext:
    """Close a service context without allowing it to hide failed work."""
    def __init__(self, manager): self.manager = manager
    def __enter__(self): return self.manager.__enter__()
    def __exit__(self, error_type, error, traceback):
        try: self.manager.__exit__(error_type, error, traceback)
        except BaseException as exit_error:
            if error is not None:
                error.context_exit_failure = {'type': type(exit_error).__name__, 'message': str(exit_error)[:1000]}
                raise error from exit_error
            raise
        return False


def run_controller(request, services, backend_factory=None, *, synthetic_fixture=False, resume=False, reason=None):
    """Own lifecycle wiring for a separately admitted registered worker.

    Services must implement preflight_bootstrap/admit_request/outer_admission,
    observer/verify_sources/verify_worker/lease/guard/watchdog/configure_runtime/
    prepare_rows/open_journal/checkpoints/seal/interruption. Backend factories
    are metadata-only; model loading belongs to admitted runtime_anchor/load.
    No implementation of those native capabilities is supplied here.
    """
    validate_request(request, synthetic_fixture=synthetic_fixture)
    require(type(resume) is bool and (not resume or type(reason) is str and reason.strip()), 'documented resume')
    if not synthetic_fixture:
        if not any(backend_factory is value for value in _REGISTERED_NATIVE_FACTORIES):
            raise MissingCapability('registered CE8 native backend missing: ' + ', '.join(ACTIVATION_GAPS))
    else:
        require(backend_factory is not None and getattr(backend_factory, 'synthetic_only', False) is True, 'explicit fixture backend only')
    request_hash = digest(request_preimage(request))
    observer = None; context = None; journal = None; backend = None; primary = None
    try:
        # Bootstrap bounds are admitted before even referenced scope/source IO.
        services.preflight_bootstrap(request['bootstrap_bounds'], request_hash)
        context = services.admit_request(request)
        require(type(context) is dict and context.get('schema') == CONTEXT
                and context.get('request_sha256') == request_hash and context.get('admission_evidence_verified') is True
                and context.get('phase') == request['phase'] and context.get('arm') == request['arm'], 'phase-specific verified admission required')
        if request['phase'] == 'training':
            require(context.get('qualifiedreceipt_verified') is True and context.get('production_training_authorized') is True,
                    'training requires PASS qualification and separate scope grant')
        observer = services.observer(context)
        outer = services.make_outer_admission(request, context)
        require(isinstance(outer, OuterAdmission), 'concrete bound outer admission adapter required')
        validate_outer(outer('controller_sources', {key: request[key] for key in REFS}), request)
        verified = _stage(observer, 'ingestion_sources', lambda: services.verify_sources(request, context, outer), request_hash)
        proof = _stage(observer, 'fresh_worker_boot', lambda: services.verify_worker(context), request_hash)
        require(type(proof) is dict and proof.get('fresh_process') is True and proof.get('boot_verified') is True
                and _hex(proof.get('boot_identity_sha256')) and _hex(proof.get('worker_identity_sha256')),
                'fresh worker and verified boot/process evidence required')
        def admitted_work():
            nonlocal backend, journal
            work_error = None
            try:
                require_preparation_cuda_uninitialized()
                _stage(observer, 'framework_setup', lambda: services.configure_preparation(context), request_hash)
                require_preparation_cuda_uninitialized()
                prepared = _stage(observer, 'ingestion_tokenizer', lambda: services.prepare_rows(verified, context, outer), request_hash)
                require_preparation_cuda_uninitialized()
                services.guard(context, 'before_model')
                backend = backend_factory(context)  # pure factory; qualified model work is a later stage
                if request['phase'] == 'qualification':
                    stages = context.get('stage_plan')
                    require(type(stages) is list and len(stages) == len(QUALIFICATION_STAGES)
                            and all(type(stage) is dict and stage.get('stage') == name
                                    and type(stage.get('ordinal')) is int and stage['ordinal'] == index
                                    and type(stage.get('caps')) is dict
                                    for index, (stage, name) in enumerate(zip(stages, QUALIFICATION_STAGES))),
                            'bounded qualification stage plan')
                    journal = _stage(observer, 'qualification_accounting',
                        lambda: backend.begin_qualification(context, prepared, proof, observer, outer), request_hash)
                    require(journal is not None, 'qualification requires retained accounting before model work')
                    artifacts = {'ingestion_tokenizer': prepared.receipt}
                    for name in QUALIFICATION_STAGES[1:]:
                        services.guard(context, name)
                        artifacts[name] = _stage(observer, name,
                            lambda name=name: backend.qualify_stage(name, prepared, artifacts, context), request_hash)
                    return artifacts
                execution = context.get('execution')
                schedule = checkpoint_boundaries(execution)
                journal = services.open_journal(context, prepared, proof, resume, reason)
                _stage(observer, 'runtime_anchor', lambda: backend.load(context, prepared, observer), request_hash)
                checkpoints = services.checkpoints(context, backend, journal)
                if resume:
                    restored = _stage(observer, 'checkpoint_restore', lambda: backend.restore(checkpoints, journal), request_hash)
                    journal.install_restore(restored)
                    backend.refresh_policy(journal)  # actual tensor fingerprint, never symbolic snapshot
                else:
                    _stage(observer, 'checkpoint_save', lambda: backend.save(checkpoints, journal), request_hash)
                last_saved = None if resume else digest(journal.progress_record())
                while journal.progress_record()['C'] < 375:
                    C = journal.progress_record()['C']; services.guard(context, 'ce_boundary')
                    rows = prepared.iter_boundary(C)
                    result = backend.update(rows, journal)
                    journal.commit_boundary(result)
                    progress = journal.progress_record()
                    require(progress['C'] == C + 1 and progress['U'] == C + 1, 'one proved CE8 boundary/update')
                    if progress['C'] in schedule:
                        backend.save(checkpoints, journal); last_saved = digest(progress)
                if digest(journal.progress_record()) != last_saved: backend.save(checkpoints, journal)
                return _stage(observer, 'donor_publish_verify', lambda: backend.donor(context, journal), request_hash)
            except BaseException as error:
                work_error = error
                raise
            finally:
                # Physical cleanup must still own the lease and watchdog. In
                # particular, an interrupted update cannot release ownership
                # before its model resources have been closed.
                if backend is not None:
                    try:
                        _stage(observer, 'controller_cleanup', lambda: services.cleanup(context, backend), request_hash)
                    except BaseException as cleanup_error:
                        if work_error is not None:
                            work_error.cleanup_failure = {'type': type(cleanup_error).__name__, 'message': str(cleanup_error)[:1000]}
                            raise work_error from cleanup_error
                        raise
                    finally:
                        backend = None

        completed = False
        with _OwnedContext(services.lease(context)):
            services.guard(context, 'before_framework')
            with _OwnedContext(services.watchdog(context)):
                artifacts = admitted_work()
                completed = True
        require(completed, 'worker context suppressed a failed lifecycle')
        publication = _stage(observer, 'completion_publication',
            lambda: services.publish_completion(context, artifacts, journal), request_hash)
        # Context exits and publication have completed before this terminal row.
        # A failed watchdog or lease release cannot follow a success seal.
        seal = services.seal(context, publication, journal)
        if request['phase'] == 'qualification':
            return {'status': 'QUALIFICATION_ARTIFACTS_SEALED', 'artifacts': artifacts, 'seal': seal,
                    'native_qualified': False, 'claim_source': 'separate qualification producer validates artifacts and costs'}
        return {'status': 'TRAINING_ARTIFACTS_SEALED', 'donor': artifacts, 'seal': seal,
                'native_qualified': False, 'claim_source': 'verified external artifact/cost seal; no learning claim here'}
    except BaseException as error:
        primary = error
        raise
    finally:
        if primary is not None and context is not None:
            try: services.interruption(context, primary, journal)
            except BaseException as receipt_error:
                primary.interruption_receipt_failure = {'type': type(receipt_error).__name__, 'message': str(receipt_error)[:1000]}
