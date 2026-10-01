"""Shared real qualification stage bodies; no issuer, dispatch or native PASS.

The source registries remain empty. Construction and every stage require an
exact native permit before IO/framework/model access. Qualification accounting
is separate from production CE history. Missing process/resource services refuse
instead of manufacturing observations. Raw artifacts support later recomputation;
only an independent execution verifier can establish actual native provenance.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import uuid

from . import ce_native_authority as authority, ce_qualification as qualification, update

SCHEMA = 'CE8-NATIVE-QUALIFICATION-EXECUTOR-SPEC-1'
RAW_SCHEMA = 'CE8-QUALIFICATION-RAW-EVIDENCE-1'
ROW_FIELDS = ('id', 'prompt_tokens', 'labeled_tokens', 'full_sequence_tokens',
              'prompt_ids_sha256', 'completion_ids_sha256', 'prepared_row_sha256')
CHUNK = 1024 ** 2
_REGISTERED_ACCOUNTING_FACTORIES = frozenset()
_REGISTERED_CHECKPOINT_ADAPTERS = frozenset()
_REGISTERED_RESOURCE_ADAPTERS = frozenset()
require = authority.require


class MissingQualificationService(authority.NativeAdmissionError): pass


def validate_execution_spec(context, receipt):
    """Pure exact row/limit declaration validation, never an execution grant."""
    spec = context['bindings']['scope'].get('qualifier_execution')
    require(type(spec) is dict and spec.get('schema') == SCHEMA, 'explicit qualification execution specification required')
    rows = receipt.get('prepared_row_bindings')
    require(type(rows) is list and rows and all(type(row) is dict and type(row.get('id')) is str for row in rows),
            'actual prepared qualification bindings required')
    ordered = [row['id'] for row in rows]
    require(len(set(ordered)) == len(ordered), 'unique actual qualification row IDs')
    union = {rid for item in receipt['witnesses'].values() for rid in item['row_ids']}
    witnesses = [rid for rid in ordered if rid in union]
    require(len(witnesses) == 90 and spec.get('all_witness_ids') == witnesses, 'all 90 exact ordered max witnesses required')
    gradient = spec.get('gradient_batch_ids')
    require(type(gradient) is list and len(gradient) == 8 and all(type(rid) is str and rid in union for rid in gradient)
            and len(set(gradient)) == 8, 'separate exact eight-row gradient batch')
    batches = spec.get('adam_batches')
    allowed = set(witnesses)
    require(type(batches) is list and 2 <= len(batches) <= 8
            and all(type(batch) is list and len(batch) == 8 and all(type(rid) is str and rid in allowed for rid in batch)
                    and len(set(batch)) == 8 for batch in batches), 'explicit bounded qualification Adam batches required')
    require(type(spec.get('owned_artifact_directory')) is str and Path(spec['owned_artifact_directory']).is_absolute(),
            'explicit qualification artifact owner required')
    # Normalize lexical traversal without touching the filesystem. Resolved
    # containment and symlink checks are repeated inside priced write work.
    artifact_root = Path(os.path.abspath(spec['owned_artifact_directory']))
    output_root = Path(os.path.abspath(context['bindings']['scope']['native_backend']['output_directory']))
    require(artifact_root != output_root and artifact_root.is_relative_to(output_root), 'qualification artifact owner outside admitted native output')
    for key in ('json_artifact_bytes_upper', 'tensor_artifact_bytes_upper', 'artifact_chunk_bytes'):
        require(type(spec.get(key)) is int and spec[key] > 0, 'strict finite qualification artifact limit: ' + key)
    require(spec['artifact_chunk_bytes'] <= CHUNK and spec['artifact_chunk_bytes'] % 4 == 0,
            'qualification gradient chunks must be F32-aligned and at most 1MiB')
    bounds = spec.get('reader_bounds')
    require(type(bounds) is dict and set(bounds) == {'operation_attempts_upper', 'io_read_bytes_upper',
            'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'reducer_work_bytes_upper'}
            and all(type(value) is int and value > 0 for value in bounds.values()), 'explicit priced raw readback limits')
    return copy.deepcopy(spec)


def _registered(candidate, registry, label):
    if not any(candidate is actual for actual in registry):
        raise MissingQualificationService('source-registered ' + label + ' is not implemented/admitted')
    return candidate


def _inventory_offsets(inventory):
    result, offset = [], 0
    for entry in inventory:
        require(type(entry) is dict and entry.get('dtype') == 'F32' and type(entry.get('name')) is str
                and type(entry.get('shape')) is list and entry['shape']
                and all(type(size) is int and size > 0 for size in entry['shape']), 'exact F32 gradient inventory')
        count = math.prod(entry['shape'])
        result.append({'name': entry['name'], 'shape': list(entry['shape']), 'offset_elements': offset, 'numel': count})
        offset += count
    require(result and len({item['name'] for item in result}) == len(result), 'distinct gradient tensor names')
    return result, offset


class NativeQualificationExecutor:
    """Common executor, independent of the selected canonical/permuted teacher."""
    synthetic_only = False

    def __init__(self, context, native_permit, backend, services):
        authority.require_execution_permit(native_permit, context, 'backend_construct')
        require(context['phase'] == 'qualification', 'qualification executor cannot run production')
        from .ce_native_backend import NativeCEBackend
        require(type(backend) is NativeCEBackend and type(services) is dict, 'exact native backend and explicit services required')
        require(backend.permit is native_permit, 'qualification backend must own this exact native permit')
        require(authority.digest(authority.context_preimage(backend.context))
                == authority.digest(authority.context_preimage(context)), 'qualification backend/admission context drift')
        self.context, self.permit, self.backend, self.services = context, native_permit, backend, services
        self.operation, self.gate = backend.operation, backend.gate
        authority.require_execution_permit(native_permit, context, 'backend_construct', operation=self.operation, gate=self.gate)
        self.accounting = self.spec = self.prepared = self.proof = self.observer = self.outer = None
        self.completed, self.outputs = [], {}
        self.stage = self.attempt_id = None
        self._evidence_reader = None
        self._cuda_ready = False

    def _admit(self, stage, context=None):
        authority.require_execution_permit(self.permit, self.context if context is None else context,
            stage, operation=self.operation, gate=self.gate)
        require(self.context['phase'] == 'qualification', 'qualification-only scope required')

    def begin_qualification(self, context, prepared, proof, observer, outer):
        self._admit('journal_construct', context)
        require(self.accounting is None and callable(observer) and callable(outer), 'begin qualification once with bound observers')
        require(type(proof) is dict and proof.get('schema') == 'CE8-QUALIFICATION-WORKER-PROOF-1'
                and proof.get('fresh_process') is True and proof.get('boot_verified') is True and proof.get('source_verified') is True
                and type(proof.get('worker_id')) is str and proof['worker_id']
                and authority._hex(proof.get('boot_identity_sha256')) and authority._hex(proof.get('worker_identity_sha256')),
                'verified fresh worker and exact retained worker identity required')
        require(context.get('worker_proof_sha256') == authority.digest(proof), 'qualification worker proof changed after authority issuance')
        qualification.validate_actual_ingestion(context, prepared.receipt)
        spec = validate_execution_spec(context, prepared.receipt)
        factory = _registered(self.services.get('accounting_factory'), _REGISTERED_ACCOUNTING_FACTORIES, 'qualification accounting factory')
        require(callable(getattr(self.operation, 'attach_qualification_accounting', None)), 'stable dispatcher qualification linkage required')
        accounting = factory(context, self.permit, proof, prepared.receipt)
        require(getattr(accounting, 'scope_role', None) == 'qualification_only'
                and getattr(accounting, 'request_sha256', None) == context['request_sha256']
                and getattr(accounting, 'scope_sha256', None) == context['scope_sha256'], 'separate qualification accounting provenance')
        self.operation.attach_qualification_accounting(accounting)
        require(getattr(self.operation, 'ledger', None) is accounting, 'qualification dispatcher must point to exact stage ledger')
        self.spec, self.accounting, self.prepared = spec, accounting, prepared
        self.proof, self.observer, self.outer = copy.deepcopy(proof), observer, outer
        self.ingestion_receipt = prepared.receipt
        self.bindings = {row['id']: {key: row[key] for key in ROW_FIELDS} for row in self.ingestion_receipt['prepared_row_bindings']}
        return accounting

    def _span(self, kind, **fields):
        metadata = {'qualification_stage': self.stage, 'attempt_id': self.attempt_id, **fields}
        # Device construction is inert; synchronization inside update._span
        # occurs only after the backend has completed its sole CUDA configure.
        # Metadata/setup spans cannot initialize CUDA ahead of that owner.
        device = self.backend.torch.device('cuda:0') if self._cuda_ready else None
        return update._span(self.operation, kind, metadata, device)

    def _binding(self):
        return {'stage': self.stage, 'request_sha256': self.context['request_sha256'],
            'scope_sha256': self.context['scope_sha256'], 'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': self.proof['worker_identity_sha256'], 'attempt_id': self.attempt_id}

    def _rows(self, ids):
        require(callable(getattr(self.prepared, 'iter_ids', None)), 'bound qualification iter_ids renderer required')
        for start in range(0, len(ids), 8):
            for row in self.prepared.iter_ids(ids[start:start + 8]):
                from .ce_update import prepared_row_binding
                require(prepared_row_binding(row) == self.bindings[row['id']], 'qualification prepared row content changed')
                yield row

    def _target(self, role, extension):
        root = Path(self.spec['owned_artifact_directory'])
        path = root / (self.stage + '.' + self.attempt_id + '.' + role + extension)
        require(path.parent == root, 'qualification artifact lexical path containment')
        return path

    def _owned_path(self, path):
        root = Path(self.spec['owned_artifact_directory'])
        owner = Path(self.context['bindings']['scope']['native_backend']['output_directory']).resolve()
        resolved = root.resolve()
        require(not root.is_symlink() and not path.is_symlink() and resolved.is_relative_to(owner)
                and resolved != owner and path.resolve().parent == resolved, 'qualification artifact resolved path containment')

    def _write_json(self, kind, payload, role):
        path = self._target(role, '.json')
        maximum = self.spec['json_artifact_bytes_upper']
        with self._span('qualification_artifact', path=str(path), write_bytes_upper=maximum,
                        read_bytes_upper=maximum, identity_hash_attempts_upper=3, identity_hash_bytes_upper=3 * maximum,
                        reducer_work_bytes_upper=maximum):
            self.gate('qualification_artifact')
            self._owned_path(path)
            binding = self._binding()
            envelope = {'schema': RAW_SCHEMA, **binding, 'evidence_kind': kind, 'payload': payload}
            raw = authority.canonical(envelope) + b'\n'
            require(len(raw) <= maximum, 'qualification JSON artifact ceiling')
            binding_sha256 = authority.digest(binding)
            self.accounting.reserve_metadata('qualification_json', len(raw), write_id=uuid.uuid4().hex, path=str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            with path.open('rb') as stream: saved = stream.read(len(raw) + 1)
            require(saved == raw, 'qualification artifact exact raw readback')
            hashed = hashlib.sha256(raw).hexdigest()
        return {'path': str(path), 'sha256': hashed, 'bytes': len(raw), 'role': role,
                'binding_sha256': binding_sha256}

    def _write_gradient(self, role):
        require(sys.byteorder == 'little', 'qualification raw F32 requires little endian')
        inventory, count = _inventory_offsets(self.backend.inventory)
        size = 4 * count; require(size <= self.spec['tensor_artifact_bytes_upper'], 'qualification gradient artifact ceiling')
        path = self._target(role, '.f32'); named = dict(self.backend.model.named_parameters())
        with self._span('qualification_tensor_write', path=str(path), write_bytes_upper=size,
                        read_bytes_upper=size, identity_hash_attempts_upper=3,
                        identity_hash_bytes_upper=2 * size + self.spec['json_artifact_bytes_upper']):
            self.gate('qualification_tensor_write')
            self._owned_path(path)
            binding_sha256 = authority.digest(self._binding())
            versions = {entry['name']: (id(named[entry['name']].grad), named[entry['name']].grad._version) for entry in inventory
                        if named[entry['name']].grad is not None}
            require(len(versions) == len(inventory), 'complete live gradient inventory required before write')
            path.parent.mkdir(parents=True, exist_ok=True)
            hashed = hashlib.sha256(); written = 0
            with path.open('xb') as stream:
                for entry in inventory:
                    value = named[entry['name']].grad
                    require(value is not None and value.dtype == self.backend.torch.float32
                            and value.is_contiguous() and list(value.shape) == entry['shape'], 'actual F32 selected gradient required')
                    flat = value.detach().reshape(-1)
                    for offset in range(0, flat.numel(), self.spec['artifact_chunk_bytes'] // 4):
                        piece = flat[offset:offset + self.spec['artifact_chunk_bytes'] // 4].to('cpu').contiguous()
                        raw = memoryview(piece.numpy()).cast('B')
                        stream.write(raw); hashed.update(raw); written += len(raw)
                stream.flush(); os.fsync(stream.fileno())
            require(written == size, 'gradient raw byte count changed')
            readback, observed = hashlib.sha256(), 0
            with path.open('rb') as stream:
                while raw := stream.read(self.spec['artifact_chunk_bytes']):
                    readback.update(raw); observed += len(raw)
            require(observed == size and readback.hexdigest() == hashed.hexdigest(), 'gradient raw file readback mismatch')
            require(all(named[name].grad is not None and (id(named[name].grad), named[name].grad._version) == version
                        for name, version in versions.items()),
                    'live gradients changed during raw publication')
        return {'path': str(path), 'sha256': hashed.hexdigest(), 'bytes': size, 'role': role,
                'binding_sha256': binding_sha256}, inventory

    def _reader(self):
        from .ce_qualification_artifacts import RawEvidenceReader
        if self._evidence_reader is None:
            self._evidence_reader = RawEvidenceReader(self.operation, self.spec['reader_bounds'], self.spec['owned_artifact_directory'],
                                                     chunk_bytes=self.spec['artifact_chunk_bytes'])
        return self._evidence_reader

    def _runtime(self):
        observed = self.backend.load(self.context, self.prepared, self.observer)
        self._cuda_ready = True
        reference = self._write_json('runtime_anchor', {'observations': observed,
            'identity': self.backend.identity, 'inventory': self.backend.inventory, 'worker_proof': self.proof}, 'runtime_anchor')
        return observed, [reference]

    def _masks(self):
        torch = self.backend.torch
        from experiments.maxrl_comp002_pretraining_v1.mechanics import completion_labels
        masks = []
        for row in self._rows(self.spec['all_witness_ids']):
            P, C = len(row['prompt_ids']), len(row['completion_ids'])
            with self._span('qualification_mask', row_id=row['id'], selected_tokens=C, full_sequence_tokens=P + C):
                self.gate('qualification_mask')
                sequence = torch.tensor([row['prompt_ids'] + row['completion_ids']], dtype=torch.long, device='cuda:0')
                attention = torch.ones_like(sequence, dtype=torch.bool)
                labels = completion_labels(sequence, attention, [P], [C], self.backend.recipe['eos_ids'], ['eos'],
                                           self.backend.recipe['max_target_tokens'])
                masks.append({'id': row['id'], 'prompt_ids': row['prompt_ids'], 'completion_ids': row['completion_ids'],
                    'input_ids': sequence[0].tolist(), 'attention_mask': attention[0].long().tolist(),
                    'labels': labels[0].tolist(), 'padding_tokens': 0, 'padding_side': 'none',
                    'predictor_positions': list(range(P - 1, P + C - 1)), 'target_positions': list(range(P, P + C))})
        reference = self._write_json('masks', {'rows': masks}, 'masks')
        reduction = self._reader().read_masks(reference, self._binding(), [self.bindings[rid] for rid in self.spec['all_witness_ids']])
        return reduction['observations'], [reference]

    def _likelihood(self):
        torch = self.backend.torch
        from experiments.maxrl_comp002_pretraining_v1 import s0_generate as generate
        self.backend.model.eval(); rows = []
        for row in self._rows(self.spec['all_witness_ids']):
            with self._span('qualification_likelihood', row_id=row['id'], selected_tokens=len(row['completion_ids']),
                            full_sequence_tokens=len(row['prompt_ids']) + len(row['completion_ids'])):
                self.gate('qualification_likelihood')
                with torch.no_grad():
                    cached = generate.cached_logprobs(self.backend.model, row['prompt_ids'], row['completion_ids'], prefill=True)
                    bulk = generate.teacher_forced_logprobs(self.backend.model, [row['prompt_ids']], [row['completion_ids']],
                                                           self.backend.recipe['pad_id'], padding_side='none')[0]
                rows.append({'id': row['id'], 'completion_ids': row['completion_ids'],
                    'prepared_row_sha256': self.bindings[row['id']]['prepared_row_sha256'],
                    'cached_logprobs': cached.detach().tolist(), 'bulk_logprobs': bulk.detach().tolist()})
                del cached, bulk
        reference = self._write_json('likelihood', {'rows': rows}, 'likelihood')
        reduction = self._reader().read_likelihood(reference, self._binding(), [self.bindings[rid] for rid in self.spec['all_witness_ids']])
        return reduction['observations'], [reference]

    def _gradients(self):
        torch = self.backend.torch
        from experiments.maxrl_comp002_pretraining_v1 import s0_generate as generate, warmstart_checkpoint as states
        rows = list(self._rows(self.spec['gradient_batch_ids']))
        require(len(rows) == 8 and self.backend.U == 0 and not self.backend.optimizer.state, 'gradient parity requires pristine U0 Adam')
        start = self.backend.fingerprint('qualification_gradient_reference_start')
        initial_modes = {name: module.training for name, module in self.backend.model.named_modules()}
        initial_checkpointing = bool(self.backend.model.is_gradient_checkpointing)
        with self._span('qualification_rng'):
            self.gate('qualification_rng'); rng_metadata, rng_tensors = states.rng_state()
        rng_evidence = {'metadata': rng_metadata, 'tensors': {name: tensor.tolist() for name, tensor in rng_tensors.items()}}
        references, configurations = [], []
        primary = None
        try:
            for independent, role in ((True, 'gradient_reference'), (False, 'gradient_candidate')):
                with self._span('qualification_gradient_reset'):
                    self.gate('qualification_gradient_reset'); self.backend.optimizer.zero_grad(set_to_none=True)
                    self.backend.model.train()
                with self._span('qualification_checkpoint_configuration'):
                    self.gate('qualification_checkpoint_configuration')
                    if independent:
                        self.backend.model.gradient_checkpointing_disable()
                        require(not self.backend.model.is_gradient_checkpointing, 'independent gradient reference must disable decoder checkpointing')
                        checkpointing = {'decoder': False, 'head': False}
                    else:
                        checkpointing = {'decoder': self.backend.pilot.enable_checkpointing(self.backend.model), 'head': True}
                    modes = {name: module.training for name, module in self.backend.model.named_modules()}
                    if configurations: require(modes == configurations[0]['module_modes'], 'gradient passes need identical module modes')
                    configurations.append({'role': role, 'checkpointing': checkpointing, 'module_modes': modes})
                with self._span('qualification_rng'):
                    self.gate('qualification_rng'); states.restore_rng(rng_metadata, rng_tensors)
                for row in rows:
                    P, C = len(row['prompt_ids']), len(row['completion_ids'])
                    kind = 'qualification_reference_forward' if independent else 'qualification_candidate_forward'
                    with self._span(kind, row_id=row['id'], selected_tokens=C, full_sequence_tokens=P + C,
                                    prepared_row_sha256=self.bindings[row['id']]['prepared_row_sha256']):
                        self.gate(kind)
                        if independent:
                            sequence = torch.tensor([row['prompt_ids'] + row['completion_ids']], dtype=torch.long, device='cuda:0')
                            hidden = self.backend.model.model(input_ids=sequence, use_cache=False).last_hidden_state
                            logits = self.backend.model.get_output_embeddings()(hidden[0, P - 1:P + C - 1]).float()
                            targets = torch.tensor(row['completion_ids'], dtype=torch.long, device='cuda:0')
                            loss = torch.nn.functional.cross_entropy(logits, targets, reduction='sum') / (8 * C)
                        else:
                            values = generate.teacher_forced_logprobs(self.backend.model, [row['prompt_ids']], [row['completion_ids']],
                                self.backend.recipe['pad_id'], padding_side='none', checkpoint_head=True)[0]
                            loss = -values.mean() / 8
                        require(bool(torch.isfinite(loss)), 'finite actual qualification loss required')
                    with self._span('qualification_backward', row_id=row['id'], selected_tokens=C, full_sequence_tokens=P + C):
                        self.gate('qualification_backward'); loss.backward()
                    if independent: del sequence, hidden, logits, targets
                    else: del values
                    del loss
                with self._span('qualification_gradient_validation'):
                    self.gate('qualification_gradient_validation')
                    update._gradient_proof(dict(self.backend.model.named_parameters()), torch)
                raw, inventory = self._write_gradient(role); references.append(raw)
            payload = {'dtype': 'F32', 'fixed_loss_denominator': 8, 'loss_reduction': self.backend.recipe['loss_reduction'],
                'start_anchor_sha256': start, 'rows': [self.bindings[rid] for rid in self.spec['gradient_batch_ids']],
                'tensor_inventory': inventory, 'reference_ref': references[0], 'candidate_ref': references[1],
                'rng_start': rng_evidence, 'gradient_configurations': configurations, 'initial_module_modes': initial_modes}
            descriptor = self._write_json('gradient', payload, 'gradient_manifest')
            reduction = self._reader().read_gradient(descriptor, self._binding(), payload['rows'], self.backend.inventory,
                                                     start_anchor_sha256=start)
            references.append(descriptor)
            return reduction['observations'], references
        except BaseException as error:
            primary = error; raise
        finally:
            try:
                with self._span('qualification_gradient_reset'):
                    self.backend.optimizer.zero_grad(set_to_none=True); self.backend.model.eval()
                with self._span('qualification_checkpoint_configuration'):
                    if initial_checkpointing: self.backend.pilot.enable_checkpointing(self.backend.model)
                    else: self.backend.model.gradient_checkpointing_disable()
                    for name, module in self.backend.model.named_modules(): module.training = initial_modes[name]
                with self._span('qualification_rng'): states.restore_rng(rng_metadata, rng_tensors)
                require(self.backend.fingerprint('qualification_gradient_unchanged_anchor') == start, 'gradient qualification changed model state')
            except BaseException as cleanup_error:
                if primary is not None:
                    primary.qualification_cleanup_failure = {'type': type(cleanup_error).__name__, 'message': str(cleanup_error)[:1000]}
                    raise primary from cleanup_error
                raise

    def _adam(self):
        require(callable(getattr(self.accounting, 'operation_evidence', None)), 'durable operation-order readback required before Adam work')
        results, ordering = [], []
        for ids in self.spec['adam_batches']:
            rows = list(self._rows(ids))
            result = self.backend.update(rows, self.accounting)
            self.accounting.commit_boundary(result)
            results.append(result)
            clipping = self.accounting.operation_evidence(result['clipping_operation']['operation_id'])
            optimizer = self.accounting.operation_evidence(result['optimizer_operation']['operation_id'])
            require(clipping['kind'] == 'gradient_clipping' and optimizer['kind'] == 'optimizer'
                    and type(clipping['returned_seq']) is type(optimizer['started_seq']) is int
                    and clipping['returned_seq'] < optimizer['started_seq'], 'returned clipping must precede actual Adam start')
            ordering.append({'clipping': clipping, 'optimizer': optimizer})
        reference = self._write_json('adam_clipping', {'boundaries': results, 'operation_order': ordering,
            'progress': self.accounting.progress_record()}, 'adam_clipping')
        observed = {'actual_optimizer_steps': len(results), 'applied_U': [row['U_after'] for row in results],
            'finite_confined_gradients': all(row['gradient_validation_operation']['finite_confined'] for row in results),
            'frozen_state_unchanged': all(row['frozen_state_sha256'] == self.backend.frozen_sha256 for row in results),
            'clip_before_adam': all(row['clipping']['returned_seq'] < row['optimizer']['started_seq'] for row in ordering),
            'clip_max_norm': self.backend.recipe['gradient_clipping']['max_norm']}
        return observed, [reference]

    def _delta(self):
        donor = self.backend.qualification_delta(self.context, self.accounting)
        roundtrip = donor['roundtrip']
        observed = {'own_start_sha256': self.context['bindings']['anchors']['own_start_sha256'],
            'common_reference_sha256': self.context['bindings']['anchors']['common_anchor_sha256'],
            'delta_dtype': 'float32', 'delta_rebuilt_verified': roundtrip['proved_from_saved_delta'],
            'delta_file_sha256': donor['file_sha256'], 'delta_data_sha256': donor['data_section_sha256'],
            'final_master_sha256': donor['final_master_sha256'], 'rebuilt_endpoint_sha256': donor['deployed_arm_sha256'],
            'rounding_residual_max_abs': roundtrip['residual_max_abs'], 'rounding_residual_l2': roundtrip['residual_norm']}
        reference = self._write_json('delta_roundtrip', {'donor': donor, 'observations': observed}, 'delta_roundtrip')
        return observed, [reference, {'path': donor['path'], 'sha256': donor['file_sha256'], 'bytes': donor['bytes'],
            'role': 'qualification_delta', 'binding_sha256': reference['binding_sha256']}]

    def _external(self, stage):
        key, registry = ('checkpoint_adapter', _REGISTERED_CHECKPOINT_ADAPTERS) if stage == 'checkpoint_resume' else (
                         'resource_adapter', _REGISTERED_RESOURCE_ADAPTERS)
        adapter = _registered(self.services.get(key), registry, stage + ' service')
        result = adapter(self.context, self.permit, self.prepared, self.backend, self.accounting,
                         self.proof, self._binding())
        require(type(result) is dict and type(result.get('raw_artifacts')) is list and result['raw_artifacts']
                and type(result.get('observations')) is dict, 'retained actual service artifacts required')
        # These callbacks need source-owned process/resource implementations and
        # independent artifact verification. The current empty registries make
        # flags/fixture callbacks unusable as substitutes for measured evidence.
        return result['observations'], result['raw_artifacts']

    def qualify_stage(self, name, prepared, artifacts, context):
        self._admit(name, context)
        require(self.accounting is not None and prepared is self.prepared
                and prepared.receipt == self.ingestion_receipt, 'begin verified qualification accounting before stages')
        stages = list(qualification.STAGES[1:])
        require(name in stages and self.completed == stages[:stages.index(name)], 'ordered single qualification stage attempt')
        require(artifacts.get('ingestion_tokenizer') == self.ingestion_receipt
                and all(artifacts.get(key) == value for key, value in self.outputs.items()), 'prior raw stage artifacts changed')
        require(self.spec == context['bindings']['scope']['qualifier_execution'], 'qualification execution specification changed')
        self.stage, self.attempt_id = name, uuid.uuid4().hex
        self.accounting.begin_stage(name, self.attempt_id)
        try:
            if name == 'runtime_anchor': observed, references = self._runtime()
            elif name == 'longest_masks': observed, references = self._masks()
            elif name == 'likelihood_gradient_parity':
                observed, references = self._likelihood()
                gradients, raw = self._gradients(); observed.update(gradients); references += raw
            elif name == 'adam_clipping': observed, references = self._adam()
            elif name == 'delta_roundtrip': observed, references = self._delta()
            else: observed, references = self._external(name)
            qualification.validate_stage_observations(context, name, observed, self.ingestion_receipt)
            self.accounting.finish_stage(references)
            result = {'stage': name, 'attempt_id': self.attempt_id, 'observations': observed,
                'raw_artifacts': references, 'native_qualified': False, 'execution_verified': False,
                'classification': 'recorded numerical evidence; independent execution/artifact/cost verification still required'}
            self.completed.append(name); self.outputs[name] = result
            return result
        except BaseException as error:
            try: self.accounting.fail_stage(error)
            except BaseException as record_error:
                error.qualification_accounting_failure = {'type': type(record_error).__name__, 'message': str(record_error)[:1000]}
            raise

    def cleanup(self):
        # Backend owns the permanent cleanup reservation and deallocation even
        # after its authority/resource/accounting checks have become failures.
        try: self._admit('cleanup')
        except BaseException as primary:
            from .ce_native_backend import NativeCEBackend, _PREPAID_CLEANUP_SEAL
            backend = getattr(self, 'backend', None)
            if type(backend) is not NativeCEBackend or getattr(backend, '_cleanup_seal', None) is not _PREPAID_CLEANUP_SEAL:
                raise
            try: backend.cleanup()
            except BaseException as cleanup_error:
                primary.qualification_cleanup_failure = {'type': type(cleanup_error).__name__, 'message': str(cleanup_error)[:1000]}
                raise primary from cleanup_error
            raise
        return self.backend.cleanup()
