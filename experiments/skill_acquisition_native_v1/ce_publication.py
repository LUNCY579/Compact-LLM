"""Create-only completion publication; native issuer/verifier routes are closed.

File durability and replay consistency do not establish native execution.
Failed temporary/final files are retained, never replaced or removed.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import time
import uuid

from . import ce_bootstrap_accounting as boot, ce_native_authority as authority
from . import ce_qualification as q, ce_qualification_artifacts as raw, ce_execution_verifier as execution

SCHEMA = 'CE8-COMPLETION-PUBLICATION-1'
SPEC = 'CE8-PUBLICATION-SPEC-1'
MAX_JSON, CHUNK = 4 * 1024 ** 2, 1024 ** 2
_REGISTERED_PUBLICATION_ISSUERS = frozenset()
_REGISTERED_EXECUTION_VERIFIERS = frozenset()
canonical, digest, require = q.canonical, q.digest, q.require
WINDOWS_REFERENCE = 'https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw'


def _clone(value): return raw._json(canonical(value))
def _ref(value):
    require(type(value) is dict and {'path', 'sha256', 'bytes'} <= set(value)
            and type(value['path']) is str and Path(value['path']).is_absolute()
            and q.hex64(value['sha256']) and q.integer(value['bytes'], 1), 'retained publication input reference')
    return value


def _atomic_no_replace(temporary, final, owned_directory):
    root = Path(owned_directory).resolve(strict=True)
    source, target = Path(temporary).resolve(strict=True), Path(final).resolve()
    require(source.parent == target.parent == root and not Path(temporary).is_symlink()
            and not Path(final).is_symlink(), 'resolved publication paths must share exact owned directory')
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        library = ctypes.WinDLL('kernel32', use_last_error=True)
        move = library.MoveFileExW
        move.argtypes, move.restype = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD], wintypes.BOOL
        # WRITE_THROUGH only; no replacement or cross-volume copy flag.
        if not move(str(source), str(target), 0x8): raise ctypes.WinError(ctypes.get_last_error())
        return {'method': 'MoveFileExW_WRITE_THROUGH', 'flags': 8, 'replacement_allowed': False,
                'same_directory': True, 'reference': WINDOWS_REFERENCE}
    os.link(source, target)
    directory = os.open(str(root), os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return {'method': 'exclusive_hardlink_directory_fsync', 'replacement_allowed': False, 'same_directory': True}


class CompletionPublisher:
    def __init__(self, context, permit, proof, prepared_receipt, *, bootstrap_handle=None,
                 bootstrap_verifier=None, native_issuer=None):
        fixture = type(context) is dict and context.get('schema') == boot.FIXTURE
        if not fixture:
            require(any(native_issuer is item for item in _REGISTERED_PUBLICATION_ISSUERS),
                    'native completion publisher issuer is unregistered')
            boot.require_registered_verifier(bootstrap_verifier)
            owner = context['bindings']['scope']['publication_owner']
            boot.verify_prepayment(bootstrap_handle, context, proof, owner, verifier=bootstrap_verifier)
            authority.require_execution_permit(permit, context, 'completion_publication')
        else:
            require(context.get('mode') == 'synthetic_fixture' and context.get('approved') is False
                    and context.get('training_authorized') is False and permit is None, 'explicit publication fixture only')
        scope = context['bindings']['scope']; spec = scope.get('publication')
        require(type(spec) is dict and spec.get('schema') == SPEC, 'source-bound publication specification required')
        require(type(spec.get('owned_directory')) is str and Path(spec['owned_directory']).is_absolute()
                and q.integer(spec.get('max_receipt_bytes'), 1) and spec['max_receipt_bytes'] <= MAX_JSON,
                'bounded publication directory/receipt')
        q._costs(spec.get('bounds'))
        require(type(spec.get('reader_bounds')) is dict and set(spec['reader_bounds']) == set(raw.BUDGET_FIELDS)
                and all(q.integer(value) for value in spec['reader_bounds'].values()), 'complete publication replay bounds')
        require(type(proof) is dict and q.hex64(proof.get('worker_identity_sha256'))
                and q.hex64(proof.get('boot_identity_sha256')), 'exact publication worker/boot identity')
        owner = {'accounting_spec_sha256': digest(spec), 'prepared_receipt_sha256': prepared_receipt['receipt_sha256']}
        if fixture and bootstrap_handle is None:
            outer = boot.BootstrapAccounting(context, None, proof)
            bootstrap_handle = outer.reserve_setup(owner, spec['bounds'])
        boot.verify_prepayment(bootstrap_handle, context, proof, owner, verifier=bootstrap_verifier)
        self.context, self.permit, self.proof, self.prepared = context, permit, _clone(proof), _clone(prepared_receipt)
        self.fixture, self.spec, self.handle = fixture, _clone(spec), bootstrap_handle
        self.counts = dict.fromkeys(q.COST_FIELDS, 0)
        self.remaining_reader = dict(spec['reader_bounds'])
        self.started = self.last_clock = time.monotonic()
        self.finished = False
        self.handle.begin()
        try:
            self.root = Path(scope['native_backend']['output_directory']).resolve()
            self.directory = Path(spec['owned_directory']).resolve()
            require(not Path(spec['owned_directory']).is_symlink() and self.directory.is_relative_to(self.root), 'owned publication path')
            self.external_inputs = _clone(spec.get('external_input_refs', []))
            self.reader = raw.RawEvidenceReader(self.operation, self.remaining_reader, self.root)
        except BaseException as error:
            try: self.handle.fail(error)
            except BaseException as accounting_error:
                error.accounting_failure = {'type': type(accounting_error).__name__, 'message': str(accounting_error)[:1000]}
                raise error from accounting_error
            raise

    def _charge(self, values):
        for key, value in values.items():
            require(key in self.counts and q.finite(value) and (key == 'wall_seconds_upper' or type(value) is int), 'finite publication charge')
            require(self.counts[key] + value <= self.spec['bounds'][key], 'publication reservation ceiling: ' + key)
        for key, value in values.items(): self.counts[key] += value
    def tick(self):
        now = time.monotonic(); require(now >= self.last_clock, 'publication clock regressed')
        self._charge({'wall_seconds_upper': now - self.last_clock}); self.last_clock = now
    def operation(self, kind, phase, operation_id, **fields):
        require(type(operation_id) is str and operation_id and phase in ('started', 'returned', 'failed'), 'priced publication operation')
        self.tick()
        if phase == 'started':
            self._charge({'operation_attempts_upper': fields.get('operation_attempts_upper', 1),
                'io_read_bytes_upper': fields.get('io_read_bytes_upper', fields.get('read_bytes_upper', 0)),
                'io_write_bytes_upper': fields.get('io_write_bytes_upper', fields.get('write_bytes_upper', 0)),
                'identity_hash_attempts_upper': fields.get('identity_hash_attempts_upper', 0),
                'identity_hash_bytes_upper': fields.get('identity_hash_bytes_upper', 0),
                'metadata_bytes_upper': fields.get('metadata_bytes_upper', 0)})
    def _owned_ref(self, reference):
        reference = _ref(reference); path = Path(reference['path'])
        external = any({k: reference[k] for k in ('path', 'sha256', 'bytes')} == {k: value[k] for k in ('path', 'sha256', 'bytes')}
                       for value in self.external_inputs)
        require(path.resolve().is_relative_to(self.root) or external, 'foreign publication artifact outside source-bound owner')
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == reference['bytes'], 'regular exact publication artifact')
        return reference
    def _binding(self):
        return {'stage': 'resource_io_measurement', 'request_sha256': self.context['request_sha256'],
            'scope_sha256': self.context['scope_sha256'], 'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': self.proof['worker_identity_sha256'], 'attempt_id': 'completion-publication'}
    def _read_ref(self, reference, role):
        reference = self._owned_ref(reference)
        require(reference['bytes'] <= MAX_JSON, 'only bounded JSON may be materialized')
        wrapped = execution._decorated(self.reader, reference, self._binding(), role)
        return b''.join(self.reader._blocks(wrapped, self._binding(), role))
    def _stream_ref(self, reference, role):
        reference = self._owned_ref(reference)
        wrapped = execution._decorated(self.reader, reference, self._binding(), role)
        count = 0
        for block in self.reader._blocks(wrapped, self._binding(), role):
            require(len(block) <= CHUNK, 'bounded publication verification chunk')
            count += len(block)
            require(count <= reference['bytes'], 'publication artifact stream grew')
        require(count == reference['bytes'], 'publication artifact stream byte count')
        return count
    def _validate(self, snapshot, stage_artifacts, ledger_refs, terminal):
        require(type(ledger_refs) is dict and set(ledger_refs) == {'header', 'journal'}, 'exact completion ledger pair')
        require(type(stage_artifacts) is dict and set(stage_artifacts) == set(q.STAGES), 'all raw stage artifacts required')
        require(type(terminal) is dict and terminal.get('worker_identity_sha256') == self.proof['worker_identity_sha256']
                and terminal.get('request_sha256') == self.context['request_sha256']
                and terminal.get('scope_sha256') == self.context['scope_sha256']
                and type(terminal.get('exit_code')) is int and terminal['exit_code'] == 0,
                'same terminal worker/request/scope/exit required')
        context = dict(self.context, prepared_receipt=self.prepared)
        header = execution._read_json(self.reader, self._owned_ref(ledger_refs['header']), self._binding(), 'qualification_header')
        execution._header(context, header)
        state, workers, _ = execution._replay(self.reader, self._owned_ref(ledger_refs['journal']), self._binding(), context, header,
                                             {self.proof['worker_identity_sha256']})
        complete, pending = execution._seal(snapshot, state)
        require(complete and not pending, 'incomplete/unknown/pending cleanup cannot publish completion')
        for stage, refs in stage_artifacts.items():
            require(type(refs) is list and refs, 'nonempty bound raw stage artifacts')
            expected = [a['evidence']['raw_refs'] for a in state.attempts if a['stage'] == stage and a['status'] == 'RETURNED']
            if expected: require(refs == expected[-1], 'stage artifact set differs from replayed evidence')
            else: require(stage == 'ingestion_tokenizer' and state.spec.get('ingestion_bootstrap_ref'), 'undeclared external stage evidence')
            for reference in refs:
                self._stream_ref(reference, 'published_raw_' + stage)
        return state

    def publish(self, snapshot, stage_artifacts, ledger_refs, terminal_proof, *, final_name='QUALIFICATION_COMPLETION.json', terminal_handle=None):
        require(not self.finished and type(final_name) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}\.json', final_name),
                'one create-only completion file name')
        temporary = self.directory / (final_name + '.' + uuid.uuid4().hex + '.partial')
        final = self.directory / final_name
        try:
            self.tick()
            require(not final.exists() and not final.is_symlink(), 'completion never overwrites existing final')
            self._validate(snapshot, stage_artifacts, ledger_refs, terminal_proof)
            if not self.fixture:
                from .ce_worker_evidence import verify_worker_handle
                actual = verify_worker_handle(terminal_handle, {'request_sha256': self.context['request_sha256'],
                    'scope_sha256': self.context['scope_sha256'], 'worker_identity_sha256': self.proof['worker_identity_sha256']}, terminal=True)
                require(actual == terminal_proof, 'terminal OS handle differs from retained terminal proof')
            receipt = dict(_clone(snapshot), publication_schema=SCHEMA, status='UNQUALIFIED_COMPLETION', completed=True,
                native_qualified=False, execution_verified=False, request_sha256=self.context['request_sha256'],
                scope_sha256=self.context['scope_sha256'], static_recipe_sha256=self.context['static_recipe_sha256'],
                source_code_sha256=_clone(self.context['bindings']['source']['code_sha256']),
                stage_artifacts=_clone(stage_artifacts), ledger_refs=_clone(ledger_refs), terminal_worker=_clone(terminal_proof),
                publication_prepayment=self.handle.evidence,
                qualification_verifier_registered=False)
            blob = canonical(receipt) + b'\n'; require(len(blob) <= self.spec['max_receipt_bytes'], 'bounded completion receipt')
            self.operation('publication', 'started', uuid.uuid4().hex, write_bytes_upper=len(blob),
                read_bytes_upper=2 * len(blob), identity_hash_attempts_upper=3, identity_hash_bytes_upper=3 * len(blob),
                metadata_bytes_upper=len(blob))
            self.directory.mkdir(parents=True, exist_ok=True)
            with temporary.open('xb') as stream:
                for start in range(0, len(blob), CHUNK):
                    block = blob[start:start + CHUNK]
                    require(stream.write(block) == len(block), 'short publication write')
                stream.flush(); os.fsync(stream.fileno())
            with temporary.open('rb') as stream: observed = stream.read(len(blob) + 1)
            require(observed == blob, 'temporary receipt exact readback')
            contract = _atomic_no_replace(temporary, final, self.directory)
            with final.open('r+b') as stream:
                os.fsync(stream.fileno()); observed = stream.read(len(blob) + 1)
            require(observed == blob, 'final completion exact readback')
            final_sha256 = hashlib.sha256(blob).hexdigest()
            self.tick(); self.handle.complete(); self.finished = True
            return {'publication_schema': SCHEMA, 'status': 'UNQUALIFIED_COMPLETION', 'native_qualified': False,
                'receipt_ref': {'path': str(final), 'bytes': len(blob), 'sha256': final_sha256},
                'publication_contract': contract, 'charged_bounds': _clone(self.counts), 'prepayment': self.handle.evidence,
                'partial_retention': 'failed temporary/final files are preserved'}
        except BaseException as error:
            try: self.handle.fail(error)
            except BaseException as accounting_error:
                error.accounting_failure = {'type': type(accounting_error).__name__, 'message': str(accounting_error)[:1000]}
                raise error from accounting_error
            raise
