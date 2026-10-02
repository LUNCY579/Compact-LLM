"""Concrete child accounting/prefix import bodies beneath existing closed authority.

The raw readers and copies prove retained metadata, bytes and accounting only.
Native construction also requires existing execution, OS-observation, prefix
admission and bootstrap prepayment services; their registries remain empty.
No framework, model or child process is imported or launched here.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import uuid

from . import ce_native_authority as authority, ce_qualification_artifacts as raw
from . import ce_qualification_accounting as accounting, ce_execution_verifier as replay
from . import ce_checkpoint_qualification as cp, ce_checkpoint_worker as worker
from . import ce_bootstrap_accounting as bootstrap
from .ce_checkpoint_local_issuer import validate_parent_view

SPEC = 'CE8-CHECKPOINT-CHILD-INITIALIZER-SPEC-1'
require = authority.require


def child_directory(context, branch):
    require(branch in cp.BRANCHES, 'exact qualification child branch')
    root = Path(context['bindings']['scope']['checkpoint_qualification']['owned_directory'])
    require(root.is_absolute(), 'absolute common checkpoint qualification owner')
    return root / 'children' / branch


def read_parent_evidence(reader, context, refs, binding, lineage):
    """Replay the original live parent snapshot; retain all parent costs externally."""
    require(type(reader) is raw.RawEvidenceReader and type(refs) is dict
            and set(refs) == {'header', 'journal', 'seal'}, 'paid original parent evidence reader/refs')
    with reader._overhead('checkpoint_parent_header_and_owner', work_bytes=32 * raw.MAX_JSON,
                          hash_attempts=16, hash_bytes=32 * raw.MAX_JSON):
        header = replay._header(context, replay._read_json(reader, refs['header'], binding, 'qualification_header'))
    identity = binding['worker_identity_sha256']
    state, workers, _ = replay._replay(reader, refs['journal'], binding, context, header, {identity})
    snapshot = replay._read_json(reader, refs['seal'], binding, 'qualification_seal')
    with reader._overhead('checkpoint_parent_live_stage', work_bytes=32 * refs['seal']['bytes']):
        replay._seal(snapshot, state)
        require(not state.fatal and not state.unknown_wall and state.worker_open
                and state.active is not None and state.active['stage'] == 'checkpoint_resume'
                and state.active['attempt_id'] == binding['attempt_id']
                and state.progress_record()['C'] == state.progress_record()['U'] == 2
                and not state._ce.active_attempt and not state.requires_restore and not state.restore_pending,
                'actual live parent checkpoint stage/U2 with resolved CE endpoint')
        require(workers.get(identity) == lineage['parent_proof'], 'raw parent worker proof differs from observed lineage')
        expected = {'schema': 'CE8-CHECKPOINT-PARENT-STAGE-SEAL-1', 'stage': 'checkpoint_resume',
            'completed': False, 'worker_open': True, 'C': 2, 'U': 2,
            'worker_identity_sha256': identity, 'request_sha256': context['request_sha256'],
            'scope_sha256': context['scope_sha256'], 'ledger_head': state._head(), 'binding_sha256': state.binding_sha256}
        require(lineage.get('parent_stage_seal') == expected, 'declared parent stage seal differs from raw replay')
        require(all(op.get('lifetime_cleanup') for op in state.operations.values() if op['phase'] == 'started'),
                'only retained parent model lifetime cleanup may remain pending')
    return {'ledger_head': state._head(), 'binding_sha256': state.binding_sha256,
        'derived_totals': copy.deepcopy(state.cumulative), 'snapshot': snapshot,
        'native_qualified': False, 'execution_verified': False}


def _json(reader, reference, binding, role, maximum=raw.MAX_JSON):
    require(raw._int(reference.get('bytes'), 1) and reference['bytes'] <= maximum, 'bounded original checkpoint metadata')
    decorated = replay._decorated(reader, reference, binding, role)
    blob = b''.join(reader._blocks(decorated, binding, role))
    with reader._overhead('checkpoint_import_JSON', work_bytes=32 * len(blob),
                          hash_attempts=4, hash_bytes=16 * len(blob)):
        value = raw._json(blob)
        require(type(value) is dict and raw.canonical(value) + b'\n' == blob, 'canonical original checkpoint JSON')
    return value


def read_checkpoint_store(reader, references, binding, *, expected_manifest=None, maximum_bytes):
    """Verify the exact four original files and publication chain without tensors."""
    from . import checkpoint as base
    require(type(references) is dict and set(references) == {'bindings', 'journal', 'manifest', 'payload'}
            and raw._int(maximum_bytes, 1), 'complete bounded original checkpoint store')
    require(all(type(ref) is dict and raw._int(ref.get('bytes'), 1) for ref in references.values())
            and sum(ref['bytes'] for ref in references.values()) <= maximum_bytes, 'checkpoint store aggregate byte ceiling')
    bindings = _json(reader, references['bindings'], binding, 'checkpoint_bindings')
    manifest = _json(reader, references['manifest'], binding, 'checkpoint_manifest')
    with reader._overhead('checkpoint_import_manifest', work_bytes=32 * raw.MAX_JSON,
                          hash_attempts=8, hash_bytes=32 * raw.MAX_JSON):
        require(manifest.get('schema') == cp.CHECKPOINT_SCHEMA
                and raw.digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) == manifest.get('manifest_sha256')
                and manifest.get('bindings') == bindings and manifest.get('bindings_sha256') == raw.digest(bindings)
                and bindings.get('qualification_evidence_binding') == binding, 'original checkpoint source/binding/body hashes')
        if expected_manifest is not None:
            require(manifest == expected_manifest, 'raw original prefix manifest differs from declared observed lineage')
        base.validate_progress(manifest.get('progress'), bindings['qualification_batches'])
        require(manifest['progress']['C'] == manifest['progress']['U'] in (1, 2), 'retained qualification checkpoint endpoint')
        inventory = bindings.get('inventory'); specifications = manifest.get('specifications')
        require(type(inventory) is list and inventory and type(specifications) is dict,
                'complete original selected checkpoint inventory/specifications')
        names = {'rng::cpu', 'rng::cuda::0'}
        for item in inventory:
            require(type(item) is dict and set(item) == {'name', 'shape', 'dtype'} and item['dtype'] == 'F32'
                    and type(item['name']) is str and item['name'] and type(item['shape']) is list
                    and all(type(size) is int and size > 0 for size in item['shape']), 'strict original FP32 inventory')
            for prefix in ('param::', 'exp_avg::', 'exp_avg_sq::'):
                name = prefix + item['name']; names.add(name)
                require(specifications.get(name) == ['F32', item['shape']], 'complete original parameter/Adam shapes')
            name = 'step::' + item['name']; names.add(name)
            require(specifications.get(name) == ['F32', []], 'original scalar Adam counter')
        require(len({item['name'] for item in inventory}) == len(inventory) and set(specifications) == names
                and manifest.get('inventory') == inventory and manifest.get('inventory_sha256') == raw.digest(inventory)
                and type(manifest.get('Adam_scalar_steps')) is int
                and manifest['Adam_scalar_steps'] == manifest['progress']['U']
                and manifest.get('Adam_initialized') == {item['name']: True for item in inventory}
                and all(type(value) is bool for value in manifest['Adam_initialized'].values())
                and type(manifest.get('rng', {}).get('cuda_count')) is int and manifest['rng']['cuda_count'] == 1
                and all(specifications[name][0] == 'U8' and len(specifications[name][1]) == 1
                        and raw._int(specifications[name][1][0], 1) and specifications[name][1][0] <= 65536
                        for name in ('rng::cpu', 'rng::cuda::0')), 'complete original Adam/native RNG inventory')
        saved = manifest['payload']; reference = references['payload']
        require({k: saved[k] for k in ('path', 'bytes')} == {k: reference[k] for k in ('path', 'bytes')}
                and saved['file_sha256'] == reference['sha256'], 'exact original manifest payload reference')
        sequence = manifest.get('sequence')
        require(type(sequence) is int and sequence > 0
                and Path(references['manifest']['path']).name == 'checkpoint-%09d.json' % sequence
                and Path(reference['path']).name == 'checkpoint-%09d.safetensors' % sequence,
                'exact original checkpoint sequence/file names')
        require(Path(references['bindings']['path']).name == base.BINDING_FILE
                and Path(references['journal']['path']).name == base.JOURNAL_FILE
                and len({str(Path(ref['path']).parent) for ref in references.values()}) == 1,
                'one original checkpoint store owner')
    journal = references['journal']; require(journal['bytes'] <= base.MAX_JOURNAL, 'original checkpoint cost journal ceiling')
    decorated = replay._decorated(reader, journal, binding, 'checkpoint_journal')
    blob = b''.join(reader._blocks(decorated, binding, 'checkpoint_journal'))
    with reader._overhead('checkpoint_import_publication_chain', work_bytes=32 * journal['bytes'],
                          hash_attempts=200000, hash_bytes=4 * journal['bytes']):
        require(blob.endswith(b'\n'), 'complete original checkpoint journal tail')
        previous, intents, saves = None, [], []
        for seq, line in enumerate(blob.splitlines(keepends=True)):
            require(len(line) <= raw.MAX_JSON and seq < 200000, 'bounded original checkpoint journal row')
            row = raw._json(line); body = {k: v for k, v in row.items() if k != 'sha256'}
            require(type(row.get('seq')) is int and row['seq'] == seq and row['previous'] == previous
                    and row['binding_sha256'] == manifest['bindings_sha256'] and raw.digest(body) == row.get('sha256')
                    and raw.canonical(row) + b'\n' == line, 'original checkpoint journal hash/order/binding drift')
            previous = row['sha256']
            if row['kind'] == 'manifest_intent' and row.get('sequence') == sequence: intents.append(row)
            if (row['kind'] == 'operation' and row.get('operation') == 'checkpoint_save'
                    and row.get('operation_id') == manifest['save_operation_id'] and row.get('phase') == 'returned'):
                saves.append(row)
        require(len(intents) == len(saves) == 1 and intents[0]['manifest_sha256'] == manifest['manifest_sha256']
                and intents[0]['operation_id'] == manifest['save_operation_id']
                and saves[0]['manifest_sha256'] == manifest['manifest_sha256'], 'original durable manifest publication/save evidence')
    # Stream and validate the actual payload header/layout and full file SHA.
    decorated = replay._decorated(reader, references['payload'], binding, 'checkpoint_payload')
    with reader._overhead('checkpoint_import_payload_layout', work_bytes=16 * references['payload']['bytes']):
        for _ in cp._payload_blocks(reader, decorated, binding, manifest['specifications']): pass
    return manifest


def copy_checkpoint_store(reader, references, binding, directory, *, owned_root, operation, gate,
                          profile, metadata_reserve, corrupt_last_byte=False):
    """Create-only streamed import; failed/partial work keeps its full reservation."""
    from . import update
    require(type(reader) is raw.RawEvidenceReader and callable(operation) and callable(gate)
            and callable(metadata_reserve), 'explicit paid checkpoint copy owner')
    total = sum(ref['bytes'] for ref in references.values())
    payload_bytes = references['payload']['bytes']
    require(profile['read_bytes_upper'] >= total + (payload_bytes + 2 if corrupt_last_byte else 0)
            and profile['write_bytes_upper'] >= total + int(corrupt_last_byte)
            and profile['identity_hash_attempts_upper'] >= int(corrupt_last_byte)
            and profile['identity_hash_bytes_upper'] >= (payload_bytes + 1 if corrupt_last_byte else 0),
            'checkpoint copy/mutation analytical IO/hash lower bounds')
    with update._span(operation, 'checkpoint_adapter_io', dict(profile, reason='isolated_corruption_copy' if corrupt_last_byte else 'original_prefix_import')) as result:
        gate('checkpoint_adapter_io')
        owner, target = Path(owned_root), Path(directory)
        require(owner.is_absolute() and target.is_absolute() and target != owner
                and target.is_relative_to(owner) and not owner.is_symlink()
                and not target.exists() and target.resolve().is_relative_to(owner.resolve()), 'fresh isolated checkpoint copy owner')
        target.mkdir(parents=True, exist_ok=False)
        copied, written, read = {}, 0, 0
        for key in ('bindings', 'journal', 'manifest', 'payload'):
            ref = references[key]; path = target / Path(ref['path']).name
            require(not path.is_symlink(), 'regular create-only checkpoint copy')
            if key != 'payload': metadata_reserve('checkpoint_json' if key != 'journal' else 'checkpoint_journal', ref['bytes'], write_id=uuid.uuid4().hex, path=str(path))
            role = 'checkpoint_' + key
            decorated = replay._decorated(reader, ref, binding, role)
            stream = reader._blocks(decorated, binding, role); primary = None
            try:
                with path.open('xb') as destination:
                    for block in stream:
                        read += len(block); count = destination.write(block); written += count
                        require(count == len(block), 'short streamed checkpoint copy')
                    destination.flush(); os.fsync(destination.fileno())
            except BaseException as error: primary = error; raise
            finally:
                try: stream.close()
                except BaseException as error:
                    if primary is not None: raw._accounting(primary, error); raise primary from error
                    raise
            copied[key] = dict(ref, path=str(path))
        mutation = None
        if corrupt_last_byte:
            path = Path(copied['payload']['path']); offset = payload_bytes - 1
            with path.open('r+b') as stream:
                stream.seek(offset); before = stream.read(1); read += len(before)
                require(len(before) == 1, 'actual corruption target byte')
                after = bytes([before[0] ^ 1]); stream.seek(offset); count = stream.write(after); written += count
                require(count == 1, 'short actual payload mutation write')
                stream.flush(); os.fsync(stream.fileno())
            hashed, hash_count = hashlib.sha256(), 0
            with path.open('rb') as stream:
                while block := stream.read(min(raw.CHUNK, payload_bytes + 1 - hash_count)):
                    read += len(block); hash_count += len(block)
                    require(hash_count <= payload_bytes, 'isolated payload grew during mutation readback')
                    hashed.update(block)
            require(hash_count == payload_bytes, 'isolated payload truncated during mutation readback')
            mutation = {'offset': offset, 'before_hex': before.hex(), 'after_hex': after.hex(),
                        'original_sha256': copied['payload']['sha256'], 'mutated_sha256': hashed.hexdigest()}
            require(mutation['original_sha256'] != mutation['mutated_sha256'], 'actual isolated payload mutation')
        result.update(bytes_read=read, bytes_written=written, copied_refs=copied, mutation=mutation,
            identity_hash_attempts_actual=int(corrupt_last_byte), identity_hash_bytes_actual=payload_bytes if corrupt_last_byte else 0)
    return {'directory': str(target), 'references': copied, 'mutation': mutation, 'native_qualified': False}


def bind_returned_restore(accounting_owner, restored, binding):
    """Bind the actual current-child returned operation; never edit its raw row."""
    matches = [op for op in accounting_owner.operations.values() if op['kind'] == 'checkpoint_restore'
        and op['phase'] == 'returned' and op['worker_id'] == binding['worker_identity_sha256']
        and op['attempt_id'] == binding['attempt_id'] and op['stage'] == binding['stage']
        and all(op['result'].get(k) == restored.get(k) for k in ('manifest_sha256', 'C', 'U', 'generation', 'symbolic_snapshot'))]
    require(len(matches) == 1, 'one exact actual current-child returned restore operation')
    record = dict(copy.deepcopy(restored), schema=cp.RESTORE_SCHEMA, source_worker_binding=copy.deepcopy(binding))
    operation = dict(copy.deepcopy(matches[0]), source_worker_binding=copy.deepcopy(binding))
    cp.validate_restore_fence(record, accounting_owner.progress_record(), operation, current_binding=binding)
    return record, operation


class CheckpointChildInitializer:
    """Existing typed authorities plus real replay/import/accounting implementation."""
    def __init__(self, context, permit, proof, *, operation, gate, reader, bootstrap_handle,
                 bootstrap_verifier=None, admission_handle=None, predecessor_verifier=None,
                 parent_handle=None, child_handle=None, prefix_handle=None):
        fixture = type(context) is dict and context.get('schema') == accounting.FIXTURE
        if not fixture:
            authority.require_execution_permit(permit, context, 'checkpoint_resume')
            bootstrap.require_registered_verifier(bootstrap_verifier)
            require(any(predecessor_verifier is item for item in accounting._REGISTERED_CHILD_PREFIX_VERIFIERS),
                    'source-owned checkpoint predecessor verifier is unregistered')
        else:
            require(context.get('mode') == 'synthetic_fixture' and context.get('approved') is False
                    and context.get('training_authorized') is False and permit is None, 'explicit initializer metadata fixture only')
        require(callable(operation) and callable(gate) and type(reader) is raw.RawEvidenceReader
                and reader.operation is operation, 'stable prepaid child observer/gate/raw reader identity')
        initial_reader_bounds = dict(reader.remaining)
        if not fixture:
            authority.require_execution_permit(permit, context, 'checkpoint_resume', operation=operation, gate=gate)
        owner = context['checkpoint_child_accounting']['owner']
        bootstrap.verify_prepayment(bootstrap_handle, context, proof, owner, verifier=bootstrap_verifier)
        with reader._overhead('child_initializer_context_and_owner', work_bytes=64 * raw.MAX_JSON,
                              hash_attempts=16, hash_bytes=64 * raw.MAX_JSON):
            spec = accounting.accounting_specification(context)
            require(owner == {'accounting_spec_sha256': raw.digest(spec),
                              'prepared_receipt_sha256': spec['ingestion_receipt_sha256']}, 'exact prepaid derived child owner')
        payload = context['checkpoint_child_observed_lineage']
        require(type(admission_handle) is accounting.CheckpointChildAdmission
                and admission_handle._seal is accounting._CHILD_PREFIX_SEAL
                and admission_handle._native is (not fixture) and admission_handle._request == context['request_sha256']
                and admission_handle._worker == proof['worker_identity_sha256'] and admission_handle._payload == payload,
                'existing exact source-issued child prefix admission required')
        if not fixture:
            from . import ce_worker_evidence as evidence
            for role, handle, expected in (('parent', parent_handle, payload['parent_proof']), ('child', child_handle, proof)):
                require(type(handle) is evidence.WinWorkerHandle, 'actual retained parent/child OS handle required')
                actual = evidence.verify_worker_handle(handle, {k: expected[k] for k in
                    ('request_sha256', 'scope_sha256', 'worker_identity_sha256')}, require_native=True, terminal=False)
                if role == 'parent': validate_parent_view(actual, expected)
                else: require(actual == expected, 'actual current-child proof differs from initializer lineage')
        policy = context['bindings']['scope']['checkpoint_qualification'].get('child_initializer')
        require(type(policy) is dict and policy.get('schema') == SPEC
                and type(policy.get('reader_bounds')) is dict and set(policy['reader_bounds']) == set(raw.BUDGET_FIELDS)
                and all(raw._int(v) for v in policy['reader_bounds'].values())
                and policy['reader_bounds'] == initial_reader_bounds
                and raw._int(policy.get('max_store_bytes'), 1)
                and type(policy.get('branch_attempt_ids')) is dict and set(policy['branch_attempt_ids']) == set(cp.BRANCHES)
                and all(type(v) is str and 0 < len(v) <= 128 for v in policy['branch_attempt_ids'].values()),
                'frozen finite child initializer read/copy/attempt policy')
        self.context, self.permit, self.proof, self.operation, self.gate, self.reader = context, permit, copy.deepcopy(proof), operation, gate, reader
        self.bootstrap_handle, self.bootstrap_verifier = bootstrap_handle, bootstrap_verifier
        self.admission, self.predecessor_verifier, self.policy = admission_handle, predecessor_verifier, copy.deepcopy(policy)
        self.fixture, self.prefix_handle, self.accounting, self.parent_evidence = fixture, prefix_handle, None, None
        self.latest_manifest = None

    def _admit(self, context, permit):
        require(context is self.context and permit is self.permit, 'same admitted child initializer context/capability')
        if not self.fixture:
            authority.require_execution_permit(permit, context, 'checkpoint_resume', operation=self.operation, gate=self.gate)

    def construct_accounting(self, context, permit, proof, prepared_receipt, request):
        self._admit(context, permit); worker.validate_request(request, context)
        require(proof == self.proof and self.accounting is None, 'one exact child accounting constructor')
        lineage = worker.validate_observed_lineage(request, context, proof)
        inputs = context['checkpoint_child_inputs']; parent = inputs['parent_context']; binding = inputs['parent_binding']
        require(parent['bindings']['scope'] == context['bindings']['scope']
                and parent['bindings']['source'] == context['bindings']['source']
                and parent['bindings']['anchors'] == context['bindings']['anchors']
                and parent['static_recipe_sha256'] == context['static_recipe_sha256']
                and parent['scope_sha256'] == context['scope_sha256']
                and parent['prepared_receipt'] == prepared_receipt
                and parent['request_sha256'] == lineage['parent_request_sha256']
                and binding['worker_identity_sha256'] == lineage['parent_worker_identity_sha256'], 'exact common parent study/owner/census/source')
        self.parent_evidence = read_parent_evidence(self.reader, parent, lineage['parent_ledger_refs'], binding, lineage)
        self.accounting = accounting.create_accounting(context, permit, proof, prepared_receipt,
            bootstrap_handle=self.bootstrap_handle, bootstrap_verifier=self.bootstrap_verifier)
        return self.accounting

    def predecessor_capability(self, context, permit, proof, payload):
        self._admit(context, permit)
        require(self.parent_evidence is not None and proof == self.proof and payload == self.admission._payload,
                'raw replay and exact admitted child lineage precede prefix installation')
        return self.admission

    def begin_child_stage(self, owner, request):
        self._admit(self.context, self.permit); require(owner is self.accounting, 'own child stage accounting')
        attempt = self.policy['branch_attempt_ids'][request['branch']]
        owner.begin_stage('checkpoint_resume', attempt); return attempt

    def checkpoint_directory(self, context, request):
        self._admit(context, self.permit); return str(child_directory(context, request['branch']) / 'checkpoints')

    def admit_prefix_storage(self, context, permit, request, bindings):
        self._admit(context, permit); require(request['branch'] == 'resumed' and self.accounting is not None, 'resumed prefix owner')
        lineage = context['checkpoint_child_observed_lineage']; refs = context['checkpoint_child_inputs']['prefix_store']
        original = lineage['prefix_manifest']; expected = original['bindings']['qualification_evidence_binding']
        require(refs['manifest'] == lineage['prefix_manifest_ref'], 'exact inherited prefix raw manifest reference')
        if not self.fixture:
            from . import ce_worker_evidence as evidence
            require(type(self.prefix_handle) is evidence.WinWorkerHandle, 'actual terminal independent prefix worker handle')
            evidence.verify_worker_handle(self.prefix_handle, {k: expected[k] for k in
                ('request_sha256', 'scope_sha256', 'worker_identity_sha256')}, require_native=True, terminal=True)
        manifest = read_checkpoint_store(self.reader, refs, expected, expected_manifest=original,
                                         maximum_bytes=self.policy['max_store_bytes'])
        require(manifest['progress']['U'] == 1 and manifest['progress']['history'] == lineage['inherited_history']
                and manifest['bindings_sha256'] == lineage['prefix_checkpoint_bindings_sha256']
                and all(manifest['bindings'][key] == bindings[key] for key in
                    ('code', 'runtime', 'recipe', 'inventory', 'common_anchor_sha256', 'own_start_sha256', 'qualification_batches', 'data')),
                'original prefix exact history/source/runtime/recipe/inventory/anchors/census')
        result = self._copy(refs, expected, child_directory(context, 'resumed') / 'imported_prefix')
        return {'directory': result['directory'], 'bindings': copy.deepcopy(manifest['bindings'])}

    def _copy(self, refs, binding, directory, *, corrupt=False):
        require(self.accounting is not None, 'own child ledger before checkpoint copy')
        return copy_checkpoint_store(self.reader, refs, binding, directory,
            owned_root=self.context['bindings']['scope']['checkpoint_qualification']['owned_directory'],
            operation=self.operation, gate=self.gate, profile=self.accounting.profiles['checkpoint_adapter_io'],
            metadata_reserve=self.accounting.reserve_metadata, corrupt_last_byte=corrupt)

    def bound_restore_result(self, owner, restored, binding):
        self._admit(self.context, self.permit); require(owner is self.accounting, 'own actual child restore accounting')
        return bind_returned_restore(owner, restored, binding)

    def open_owned_corruption_store(self, context, permit, owner, request, binding):
        self._admit(context, permit); require(owner is self.accounting, 'own child corruption accounting')
        from . import checkpoint as base, update
        profile = owner.profiles['checkpoint_adapter_io']
        # Reserve path/stat/metadata/hash work before inspecting the saved store.
        with update._span(self.operation, 'checkpoint_adapter_io', dict(profile, reason='current_saved_checkpoint_refs')) as result:
            self.gate('checkpoint_adapter_io')
            directory = Path(self.checkpoint_directory(context, request)); manifests = sorted(directory.glob('checkpoint-*.json'))
            require(manifests and len(manifests) <= 2, 'bounded own saved checkpoint manifest inventory')
            paths = {'bindings': directory / base.BINDING_FILE, 'journal': directory / base.JOURNAL_FILE,
                'manifest': manifests[-1], 'payload': manifests[-1].with_suffix('.safetensors')}
            refs, byte_count = {}, 0
            for key, path in paths.items():
                require(path.is_file() and not path.is_symlink() and path.resolve().parent == directory.resolve(), 'own saved checkpoint regular path')
                size = path.stat().st_size; byte_count += size
                require(byte_count <= self.policy['max_store_bytes'] and byte_count <= profile['read_bytes_upper']
                        and byte_count <= profile['identity_hash_bytes_upper'] and profile['identity_hash_attempts_upper'] >= 4,
                        'prepaid current saved checkpoint reference hash bounds')
                hashed, observed = hashlib.sha256(), 0
                with path.open('rb') as stream:
                    while block := stream.read(min(raw.CHUNK, size + 1 - observed)):
                        observed += len(block); require(observed <= size, 'own saved checkpoint grew during reference hash')
                        hashed.update(block)
                require(observed == size, 'own saved checkpoint truncated during reference hash')
                refs[key] = {'path': str(path), 'sha256': hashed.hexdigest(), 'bytes': size}
            result.update(bytes_read=byte_count, identity_hash_attempts_actual=4,
                          identity_hash_bytes_actual=byte_count, retained_refs=refs)
        manifest = read_checkpoint_store(self.reader, refs, binding, maximum_bytes=self.policy['max_store_bytes'])
        self.latest_manifest = manifest
        copied = self._copy(refs, binding, child_directory(context, request['branch']) / 'corruption_control', corrupt=True)
        self.corruption_mutation = copied['mutation']
        with update._span(self.operation, 'qualification_checkpoint_configuration',
                          dict(owner.profiles['qualification_checkpoint_configuration'], reason='corrupt_store_open')):
            result = cp.QualificationCheckpoints(context, permit, copied['directory'], manifest['bindings'],
                sum(math.prod(item['shape']) for item in manifest['bindings']['inventory']),
                operation=self.operation, gate=self.gate, metadata_reserve=owner.reserve_metadata)
        return result.attach_ledger(self.operation)

    def stale_fence_record(self, context, owner, request, binding):
        self._admit(context, self.permit); require(owner is self.accounting and self.latest_manifest is not None, 'verified own saved ancestor before stale attempt')
        progress = owner.progress_record(); progress['symbolic_snapshot'] = progress['snapshot']
        record = dict(copy.deepcopy(progress), schema=cp.RESTORE_SCHEMA, progress=progress,
            manifest_sha256=self.latest_manifest['manifest_sha256'], install_fence_required=True,
            source_worker_binding=copy.deepcopy(binding))
        return worker.validate_stale_attempt(record, owner.progress_record())

    def finish_child_accounting(self, owner, request, primary, *, retained=None):
        self._admit(self.context, self.permit); require(owner is self.accounting, 'own child accounting finish')
        if primary is not None:
            if owner.active is not None: owner.fail_stage(primary)
            return
        require(type(retained) is dict and type(retained.get('raw_refs')) is list and retained['raw_refs']
                and retained.get('branch') == request['branch'] and retained.get('native_qualified') is False,
                'actual retained branch references after backend cleanup')
        owner.finish_stage(retained['raw_refs'])
        # Handler owns end() after this method, then supervisor lease return.
