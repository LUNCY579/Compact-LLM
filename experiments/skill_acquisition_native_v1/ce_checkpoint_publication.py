"""Retain checkpoint child evidence under an independent outer prepayment.

The child is still OS-live when it returns its branch envelope. Its supervisor
must independently verify termination. This source issues no native authority.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
import uuid

from . import ce_publication as publication, ce_bootstrap_accounting as boot
from . import ce_native_authority as authority, ce_qualification_artifacts as raw
from . import ce_qualification_accounting as accounting, ce_execution_verifier as execution

SPEC = 'CE8-CHECKPOINT-BRANCH-PUBLICATION-SPEC-1'
RETAINED = 'CE8-CHECKPOINT-RETAINED-BRANCH-1'
RESULT = 'CE8-CHECKPOINT-WORKER-RESULT-1'
require, digest, canonical = raw.require, raw.digest, raw.canonical


def publication_specification(context):
    scope = context['bindings']['scope']
    common = scope.get('checkpoint_branch_publication')
    declaration = context.get('checkpoint_child_declaration')
    require(type(common) is dict and common.get('schema') == SPEC
            and type(declaration) is dict and declaration.get('branch') in ('uninterrupted', 'prefix', 'resumed'),
            'frozen child publication policy and branch declaration required')
    root = Path(scope['checkpoint_qualification']['owned_directory'])
    require(root.is_absolute(), 'absolute checkpoint publication owner')
    expected = dict(common, owned_directory=str(root / 'children' / declaration['branch'] / 'published'))
    require(context.get('checkpoint_child_publication', expected) == expected,
            'child publication may derive only its branch directory')
    return publication._clone(expected)


class CheckpointBranchPublisher(publication.CompletionPublisher):
    """Two phases: retain live state refs, then publish after model/lease release."""
    def __init__(self, context, permit, proof, prepared_receipt, *, bootstrap_handle=None,
                 bootstrap_verifier=None, native_issuer=None):
        fixture = type(context) is dict and context.get('schema') == boot.FIXTURE
        if not fixture:
            require(any(native_issuer is item for item in publication._REGISTERED_PUBLICATION_ISSUERS),
                    'native checkpoint publication issuer is unregistered')
            boot.require_registered_verifier(bootstrap_verifier)
            authority.require_execution_permit(permit, context, 'completion_publication')
        else:
            require(context.get('mode') == 'synthetic_fixture' and context.get('approved') is False
                    and context.get('training_authorized') is False and permit is None,
                    'explicit non-native checkpoint publication fixture')
        spec = publication_specification(context)
        require(type(spec.get('max_receipt_bytes')) is int and 0 < spec['max_receipt_bytes'] <= raw.MAX_JSON,
                'bounded child publication JSON')
        publication.q._costs(spec.get('bounds'))
        require(type(spec.get('reader_bounds')) is dict and set(spec['reader_bounds']) == set(raw.BUDGET_FIELDS)
                and all(raw._int(value, 1) for value in spec['reader_bounds'].values()), 'finite child publication reader bounds')
        owner = {'accounting_spec_sha256': digest(spec), 'prepared_receipt_sha256': prepared_receipt['receipt_sha256']}
        if fixture and bootstrap_handle is None:
            outer = boot.BootstrapAccounting(context, None, proof)
            bootstrap_handle = outer.reserve_setup(owner, spec['bounds'])
        boot.verify_prepayment(bootstrap_handle, context, proof, owner, verifier=bootstrap_verifier)
        self.context, self.permit, self.proof, self.prepared = context, permit, publication._clone(proof), publication._clone(prepared_receipt)
        self.fixture, self.spec, self.handle = fixture, spec, bootstrap_handle
        self.counts = dict.fromkeys(publication.q.COST_FIELDS, 0)
        self.remaining_reader = dict(spec['reader_bounds'])
        self.started = self.last_clock = time.monotonic()
        self.finished, self.retained, self.current_binding, self.retained_ledger = False, None, None, None
        self.handle.begin()
        try:
            self.root = Path(context['bindings']['scope']['native_backend']['output_directory']).resolve()
            self.directory = Path(spec['owned_directory']).resolve()
            require(not Path(spec['owned_directory']).is_symlink() and self.directory.is_relative_to(self.root)
                    and self.directory != self.root, 'source-owned child publication directory')
            self.external_inputs = []
            self.reader = raw.RawEvidenceReader(self.operation, self.remaining_reader, self.root)
        except BaseException as error:
            self._fail(error)
            raise

    def _fail(self, error):
        self.finished = True
        try: self.handle.fail(error)
        except BaseException as accounting_error:
            raw._accounting(error, accounting_error)
            raise error from accounting_error

    def _admit(self, context, permit):
        require(context is self.context and permit is self.permit and not self.finished,
                'exact live child publication context/permit')
        if not self.fixture:
            authority.require_execution_permit(permit, context, 'completion_publication')
        require(publication_specification(context) == self.spec, 'frozen child publication policy drift')

    def _binding(self):
        require(self.current_binding is not None, 'child publication binding precedes reads')
        return self.current_binding

    def _reference_file(self, path, role):
        maximum = self.spec['max_receipt_bytes']
        identifier = uuid.uuid4().hex
        self.operation('child_reference', 'started', identifier, read_bytes_upper=maximum + 1,
                       identity_hash_attempts_upper=2, identity_hash_bytes_upper=maximum + 1 + raw.MAX_BINDING_BYTES)
        try:
            path = Path(path)
            require(path.resolve().is_relative_to(self.root) and path.is_file() and not path.is_symlink()
                    and 0 < path.stat().st_size <= maximum, 'bounded owned child JSON reference')
            with path.open('rb') as stream: content = stream.read(maximum + 1)
            require(len(content) == path.stat().st_size and len(content) <= maximum, 'child JSON changed during reference read')
            return {'path': str(path.resolve()), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
                    'role': role, 'binding_sha256': digest(self._binding())}
        finally: self.tick()

    def _write(self, name, value, role, *, ledger=None):
        maximum = self.spec['max_receipt_bytes']; identifier = uuid.uuid4().hex
        self.operation('child_publication', 'started', identifier, write_bytes_upper=maximum,
            read_bytes_upper=2 * maximum + 2, identity_hash_attempts_upper=4,
            identity_hash_bytes_upper=3 * maximum + raw.MAX_BINDING_BYTES, metadata_bytes_upper=maximum)
        content = canonical(value) + b'\n'
        require(len(content) <= maximum, 'child publication JSON ceiling')
        final = self.directory / name
        temporary = self.directory / (name + '.' + uuid.uuid4().hex + '.partial')
        require(final.parent == temporary.parent == self.directory and not final.is_symlink()
                and not os.path.lexists(final), 'create-only exact child publication path')
        if ledger is not None:
            ledger.reserve_metadata('qualification_json', len(content), write_id=identifier, path=str(final))
        self.directory.mkdir(parents=True, exist_ok=True)
        with temporary.open('xb') as stream:
            require(stream.write(content) == len(content), 'short child publication write')
            stream.flush(); os.fsync(stream.fileno())
        with temporary.open('rb') as stream: observed = stream.read(len(content) + 1)
        require(observed == content, 'child temporary readback mismatch')
        publication._atomic_no_replace(temporary, final, self.directory)
        with final.open('r+b') as stream:
            os.fsync(stream.fileno()); observed = stream.read(len(content) + 1)
        require(observed == content, 'child final readback mismatch')
        self.tick()
        return {'path': str(final), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
                'role': role, 'binding_sha256': digest(self._binding())}

    def retain_branch(self, context, permit, request, binding, checkpoint, ledger, backend, *, negative_control_evidence):
        self._admit(context, permit)
        require(self.retained is None and type(ledger) is accounting.QualificationAccounting
                and ledger.context is context and ledger.active is not None,
                'one actual current-child qualification ledger before retention')
        raw.binding(binding)
        expected_binding = {'stage': 'checkpoint_resume', 'worker_identity_sha256': self.proof['worker_identity_sha256'],
            'attempt_id': ledger.active['attempt_id'], 'request_sha256': context['request_sha256'],
            'scope_sha256': context['scope_sha256'], 'static_recipe_sha256': context['static_recipe_sha256'],
            'code_sha256': context['bindings']['source']['code_sha256']}
        require(binding == expected_binding and ledger.active['stage'] == 'checkpoint_resume'
                and request['branch'] == context['checkpoint_child_declaration']['branch'],
                'exact retained child stage/source/worker binding')
        self.current_binding = publication._clone(binding)
        self.retained_ledger = ledger
        try:
            current = ledger.progress_record()
            saved = checkpoint.get('progress') if type(checkpoint) is dict else None
            require(type(checkpoint) is dict and checkpoint.get('schema') == 'SKILL-ACQUISITION-NATIVE-CHECKPOINT-1'
                    and checkpoint['bindings'].get('qualification_evidence_binding') == binding
                    and type(saved) is dict and all(saved.get(key) == current.get(key) for key in ('C', 'U', 'history'))
                    and type(saved.get('C')) is type(saved.get('U')) is int
                    and type(saved.get('generation')) is int and 0 <= saved['generation'] <= current['generation']
                    and saved['C'] == saved['U'] == (1 if request['branch'] == 'prefix' else 2),
                    'actual current-child checkpoint manifest')
            if not self.fixture:
                from .ce_native_backend import NativeCEBackend
                from .ce_checkpoint_qualification import RESTORE_SCHEMA
                require(type(backend) is NativeCEBackend and backend.context is context and backend.permit is permit
                        and backend.model is not None and backend.optimizer is not None,
                        'actual live owned backend precedes child state retention')
                recovered = negative_control_evidence.get('recovery_restore_record')
                require(type(recovered) is dict and recovered.get('schema') == RESTORE_SCHEMA
                        and recovered.get('manifest_sha256') == checkpoint.get('manifest_sha256')
                        and recovered.get('source_worker_binding') == binding and recovered.get('progress') == current
                        and not ledger.requires_restore and not ledger.restore_pending,
                        'saved endpoint ancestor must have its installed current-child recovery fence')
            manifest_path = Path(checkpoint['payload']['path']).with_suffix('.json')
            manifest_ref = self._reference_file(manifest_path, 'checkpoint_manifest')
            content = self._read_ref(manifest_ref, 'checkpoint_manifest')
            require(raw._json(content) == checkpoint and canonical(checkpoint) + b'\n' == content,
                    'saved manifest bytes differ from actual returned checkpoint')
            controls_ref = self._write('NEGATIVE_CONTROLS.json', {'schema': raw.SCHEMA, **binding,
                'evidence_kind': 'checkpoint_negative_controls', 'payload': negative_control_evidence},
                'checkpoint_negative_controls', ledger=ledger)
            child_context = authority.context_preimage(context)
            child_context = dict(child_context, prepared_receipt=self.prepared)
            self.retained = {'schema': RETAINED, 'branch': request['branch'], 'binding': publication._clone(binding),
                'observed_lineage': publication._clone(context['checkpoint_child_observed_lineage']),
                'child_context': publication._clone(child_context), 'final_manifest_ref': manifest_ref,
                'negative_controls_ref': controls_ref, 'raw_refs': [manifest_ref, controls_ref],
                'native_qualified': False, 'execution_verified': False}
            return publication._clone(self.retained)
        except BaseException as error:
            self._fail(error)
            raise

    def publish_after_release(self, context, permit, request, binding, retained, ledger, *, result_schema, native_qualified):
        self._admit(context, permit)
        require(result_schema == RESULT and native_qualified is False and retained == self.retained
                and retained is not None and binding == self.current_binding, 'exact retained child branch before publication')
        require(ledger is self.retained_ledger and type(ledger) is accounting.QualificationAccounting
                and ledger.context is context and ledger._current_proof == self.proof,
                'exact original child accounting owner before publication reads')
        try:
            snapshot = ledger.snapshot()
            require(not snapshot['worker_open'] and not snapshot['fatal'] and not snapshot['unknown_wall']
                    and not snapshot['pending_operations'] and ledger.active is None
                    and not ledger.requires_restore and not ledger.restore_pending,
                    'child ledger/model cleanup must end before outer publication')
            header_ref = self._reference_file(ledger.header_path, 'qualification_header')
            journal_path = ledger.path
            oid = uuid.uuid4().hex
            self.operation('child_journal_reference', 'started', oid, read_bytes_upper=execution.MAX_LEDGER + 1,
                identity_hash_attempts_upper=1, identity_hash_bytes_upper=execution.MAX_LEDGER + 1)
            require(journal_path.resolve().is_relative_to(self.root) and journal_path.is_file()
                    and not journal_path.is_symlink() and 0 < journal_path.stat().st_size <= execution.MAX_LEDGER,
                    'bounded owned retained child journal')
            # The potentially large original journal is streamed rather than JSON-materialized.
            size = journal_path.stat().st_size
            hashed, read = hashlib.sha256(), 0
            with journal_path.open('rb') as stream:
                while block := stream.read(min(raw.CHUNK, size + 1 - read)):
                    read += len(block); require(read <= size, 'child journal grew during publication'); hashed.update(block)
            require(read == size == journal_path.stat().st_size, 'child journal changed during publication')
            journal_ref = {'path': str(journal_path), 'sha256': hashed.hexdigest(), 'bytes': size}
            seal_ref = self._write('CHILD_ACCOUNTING_SNAPSHOT.json', snapshot, 'qualification_seal')
            payload = {key: value for key, value in retained.items() if key not in ('schema', 'raw_refs')}
            payload.update(schema=RESULT, ledger_refs={'header': header_ref, 'journal': journal_ref, 'seal': seal_ref},
                body_completed=True, os_terminal_verified=False, native_qualified=False, execution_verified=False,
                outer_publication_prepayment=self.handle.evidence,
                classification='child body evidence; separate retained supervisor handle must prove OS termination')
            reference = self._write('CHECKPOINT_BRANCH.json', {'schema': raw.SCHEMA, **binding,
                'evidence_kind': 'checkpoint_branch', 'payload': payload}, 'checkpoint_branch')
            self.tick(); self.handle.complete(); self.finished = True
            return {'raw_artifact': reference, 'body_completed': True, 'os_terminal_verified': False,
                'native_qualified': False, 'execution_verified': False, 'charged_bounds': publication._clone(self.counts),
                'publication_prepayment': self.handle.evidence}
        except BaseException as error:
            self._fail(error)
            raise
