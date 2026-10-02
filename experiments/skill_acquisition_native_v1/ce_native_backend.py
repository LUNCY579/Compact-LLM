"""Delayed real FP32 CE8 lifecycle bodies; no registered authority or dispatch.

Every native entry requires the source-issued permit before deferred imports,
model inspection or IO. The empty authority registry currently makes every
entry unreachable. These implementations do not issue qualification or claim
learning. The controller owns the lease/watchdog, measured prices and terminal
publication; this backend owns actual model/Adam/checkpoint/donor mechanics.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
import time
import uuid
from pathlib import Path

from . import ce_native_authority as authority, update

SCHEMA = 'CE8-NATIVE-BACKEND-SPEC-1'
CHECKPOINT_MODE = 'CE8_NATIVE_FP32_1'
PROFILE_FIELDS = ('read_bytes_upper', 'write_bytes_upper',
                  'identity_hash_attempts_upper', 'identity_hash_bytes_upper')
OPERATION_KINDS = frozenset(('module_import', 'configured_runtime', 'process_rng_seed',
    'input_io', 'tokenizer_load', 'model_load', 'inventory_configuration', 'fingerprint',
    'journal_construct', 'checkpoint_construct', 'checkpoint_adapter_io', 'donor_reference', 'donor_write',
    'donor_roundtrip', 'donor_chunk', 'cleanup'))
B_NUMEL = 220233216
PINNED_ASSET_RECEIPT_SHA256 = '9d9908cbcbb83a7feb75ad45ac38bd1d768f2841def8a6fb7ad9f4b6444798a3'
TOKENIZER_ASSETS = ('config.json', 'generation_config.json', 'tokenizer.json',
                    'tokenizer_config.json', 'vocab.json', 'merges.txt')
_PREPAID_CLEANUP_SEAL = object()

require = authority.require


def _path(value):
    require(type(value) is str and value and Path(value).is_absolute(), 'explicit absolute native path required')
    return value


def _ref(value):
    require(type(value) is dict and type(value.get('path')) is str
            and authority._hex(value.get('sha256')), 'explicit native raw-hash reference required')
    _path(value['path'])
    return value


def loader_work_lower_bounds(context):
    """Pure unavoidable payload/read/hash work of the pinned loader.

    Exact raw receipt bytes prove these metadata facts; a dictionary claiming
    another blob's SHA does not. These conservative lower bounds omit framing,
    import/config/AutoTokenizer work and extra verification passes. Passing them
    cannot establish an upper price, resource fit, qualification or permission.
    """
    spec = context['bindings']['scope']['native_backend']
    references = {key: spec.get(key) for key in ('asset_receipt_ref', 'tokenizer_receipt_ref', 'warm_delta_ref')}
    for key, ref in references.items():
        _ref(ref)
        require(type(ref.get('bytes')) is int and ref['bytes'] > 0, 'explicit bound file bytes required: ' + key)
    raw_text = spec.get('asset_receipt_utf8')
    require(type(raw_text) is str and raw_text, 'exact pinned asset receipt bytes required for analytical prices')
    raw = raw_text.encode('utf8')
    asset_ref = references['asset_receipt_ref']
    require(len(raw) == asset_ref['bytes'] and hashlib.sha256(raw).hexdigest()
            == asset_ref['sha256'] == PINNED_ASSET_RECEIPT_SHA256, 'analytical asset facts are not the pinned raw receipt')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate asset receipt key'); out[key] = value
        return out
    assets = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(authority.NativeAdmissionError('nonfinite asset metadata')))
    require(type(assets) is dict and assets.get('stage') == 'Q1_ASSET_CANDIDATE'
            and Path(assets.get('model_path', '')) == Path(spec['model_dir']), 'analytical pinned model path/stage drift')
    files, header = assets.get('assets'), assets.get('model_header')
    require(type(files) is dict and type(header) is dict and header.get('dtypes') == ['BF16']
            and type(header.get('data_bytes')) is int and header['data_bytes'] > 0
            and header['data_bytes'] % 2 == 0 and type(header.get('tensors')) is int and header['tensors'] > 0,
            'pinned BF16 header facts required')
    for name in ('model.safetensors', *TOKENIZER_ASSETS):
        require(type(files.get(name)) is dict and type(files[name].get('bytes')) is int
                and files[name]['bytes'] > 0 and authority._hex(files[name].get('sha256')), 'pinned asset byte/hash facts: ' + name)
    anchors = context['bindings']['anchors']
    require(files['model.safetensors']['sha256'] == anchors['base_weights_sha256']
            and files['model.safetensors']['bytes'] >= header['data_bytes'] + 8
            and header['data_bytes'] >= 2 * B_NUMEL, 'pinned base payload/B inventory facts drift')
    A, T = asset_ref['bytes'], references['tokenizer_receipt_ref']['bytes']
    warm_file, warm_data = references['warm_delta_ref']['bytes'], 4 * B_NUMEL
    require(warm_file >= warm_data + 8, 'registered B warm delta file below its FP32 payload')
    S, weight_file, config = header['data_bytes'], files['model.safetensors']['bytes'], files['config.json']['bytes']
    tokenizer_files = sum(files[name]['bytes'] for name in TOKENIZER_ASSETS)
    def bound(read, attempts, hashed):
        return {'read_bytes_upper': read, 'write_bytes_upper': 0,
                'identity_hash_attempts_upper': attempts, 'identity_hash_bytes_upper': hashed}
    return {
        # _read_ref's child observed_file reserves size+1 before a bounded read.
        'input_io': bound(max(A, T, warm_file) + 1, 1, max(A, T, warm_file) + 1),
        # _receipt reads each receipt for SHA and JSON; six asset hashes are
        # guaranteed before AutoTokenizer's additional uncounted reads.
        'tokenizer_load': bound(2 * A + 2 * T + tokenizer_files, 2 + len(TOKENIZER_ASSETS), A + T + tokenizer_files),
        # Weight file SHA, cold/warm FP32 identity, stored BF16 verification and
        # warm-data SHA. Stored tensor hashes alone require header.tensors calls;
        # six additional file/state passes give a conservative attempt floor.
        'model_load': bound(2 * A + weight_file + 2 * S + warm_data + config,
                            header['tensors'] + 6, A + weight_file + 5 * S + warm_data + config),
        'fingerprint': bound(0, 1, 2 * S),
    }


def validate_backend_spec(context):
    """Pure shape/binding validation; a valid dictionary grants no execution."""
    require(type(context) is dict and type(context.get('bindings')) is dict, 'complete backend context required')
    bindings = context['bindings']; scope = bindings.get('scope')
    require(type(scope) is dict, 'separately admitted backend scope required')
    spec = scope.get('native_backend')
    require(type(spec) is dict and spec.get('schema') == SCHEMA, 'explicit native backend specification required')
    require(type(spec.get('run_seed')) is int and 0 <= spec['run_seed'] < 2 ** 63, 'explicit bounded run seed required')
    require(spec.get('inventory') == {'blocks': [14, 27], 'numel': B_NUMEL}, 'exact common B14-27 inventory required')
    for key in ('source_checkout', 'output_directory', 'model_dir'): _path(spec.get(key))
    for key in ('asset_receipt_ref', 'tokenizer_receipt_ref', 'warm_delta_ref'): _ref(spec.get(key))
    warm, anchors = spec['warm_delta_ref'], bindings.get('anchors')
    require(type(anchors) is dict and all(authority._hex(warm.get(key)) for key in ('data_sha256', 'state_sha256'))
            and warm['data_sha256'] == anchors.get('warm_delta_sha256')
            and spec['tokenizer_receipt_ref']['sha256'] == anchors.get('tokenizer_receipt_sha256')
            and anchors.get('own_start_sha256') == anchors.get('common_anchor_sha256'), 'exact independent common warm reference required')
    recipe = bindings.get('recipe', {}).get('recipe')
    require(type(recipe) is dict and recipe.get('synthetic_cpu') is False
            and authority.digest(recipe) == context.get('static_recipe_sha256'), 'native static recipe identity required')
    names = recipe.get('trainable_names')
    require(type(names) is list and names and all(type(name) is str and name for name in names)
            and names == sorted(set(names))
            and all(any(name.startswith('model.layers.%d.' % layer) for layer in range(14, 28)) for name in names),
            'frozen exact B inventory names required')
    inventory = spec.get('parameter_inventory')
    require(type(inventory) is list and [entry.get('name') if type(entry) is dict else None for entry in inventory] == names,
            'exact native parameter shape inventory required')
    require(all(set(entry) == {'name', 'shape', 'dtype'} and entry['dtype'] == 'F32'
                and type(entry['shape']) is list and entry['shape']
                and all(type(size) is int and size > 0 for size in entry['shape']) for entry in inventory)
            and sum(math.prod(entry['shape']) for entry in inventory) == B_NUMEL, 'native B shape/dtype/numel binding')
    require(recipe.get('optimizer', {}).get('capturable') is True
            and type(recipe.get('accumulation_rows')) is int and recipe['accumulation_rows'] == 8
            and type(recipe.get('microbatch_rows')) is int and recipe['microbatch_rows'] == 1,
            'explicit CUDA Adam/eight-row recipe required')
    runtime = bindings.get('runtime', {}).get('decoding_preimage')
    require(type(runtime) is dict and runtime.get('device') == 'cuda:0' and runtime.get('dtype') == 'float32',
            'explicit FP32 cuda:0 runtime required')
    profiles = spec.get('operation_bounds')
    require(type(profiles) is dict and set(profiles) >= OPERATION_KINDS, 'complete explicitly priced native operation profiles required')
    for name, profile in profiles.items():
        require(type(name) is str and type(profile) is dict and set(profile) == set(PROFILE_FIELDS)
                and all(type(profile[key]) is int and profile[key] >= 0 for key in PROFILE_FIELDS),
                'strict native operation profile: ' + str(name))
        require((profile['identity_hash_attempts_upper'] == 0) == (profile['identity_hash_bytes_upper'] == 0),
                'native operation hash count/bytes coherence')
    for kind, minimum in loader_work_lower_bounds(context).items():
        require(all(profiles[kind][key] >= value for key, value in minimum.items()),
                'native profile below known pinned loader work: ' + kind)
    require(type(spec.get('operation_bounds_status')) is str
            and spec['operation_bounds_status'] in ('MEASURED_PRODUCTION', 'AUTHORIZED_QUALIFICATION_UPPER'),
            'unmeasured proposal profiles cannot execute')
    require(context.get('phase') != 'training' or spec['operation_bounds_status'] == 'MEASURED_PRODUCTION',
            'production requires measured operation prices')
    # Lexical checks do not touch the filesystem; the bound runtime performs the
    # resolved outside-checkout/regular-file proof inside observed work.
    require(Path(spec['source_checkout']) not in (Path(spec['output_directory']), *Path(spec['output_directory']).parents),
            'native output must be outside source checkout')
    return copy.deepcopy(spec)


def _ce_plan_rows(plan):
    """Strict CE8 plan hook shared with the native storage implementation."""
    require(type(plan) is list and len(plan) == 375, 'exact 375 native CE8 boundaries required')
    rows, seen = [], set()
    for index, entry in enumerate(plan):
        require(type(entry) is dict and type(entry.get('boundary')) is int and entry['boundary'] == index
                and type(entry.get('logical_exposure')) is int and entry['logical_exposure'] == 0, 'native CE8 boundary/exposure order')
        ids = entry.get('row_ids')
        require(type(ids) is list and len(ids) == 8 and all(type(rid) is str and rid for rid in ids)
                and len(set(ids)) == 8 and not seen.intersection(ids), 'eight globally distinct native CE rows')
        rows.append(list(ids)); seen.update(ids)
    return rows


class NativeCECheckpoints:
    """Permit-gated CE8 plan/progress adapter over actual FP32/CUDA storage."""
    def __init__(self, directory, context, bindings, numel, gate, *, metadata_reserve, operation, native_permit):
        authority.require_execution_permit(native_permit, context, 'checkpoint_construct', operation=operation, gate=gate)
        require(type(bindings) is dict and bindings.get('checkpoint_mode') == CHECKPOINT_MODE
                and bindings.get('synthetic_cpu') is False, 'native CE8 checkpoint binding mode')
        admitted = context['bindings']; recipe = admitted['recipe']['recipe']
        require(Path(directory) == Path(admitted['scope']['native_backend']['output_directory']) / 'checkpoints',
                'native checkpoint directory outside admitted output owner')
        require(bindings.get('code') == admitted['source']['code_sha256']
                and bindings.get('runtime') == admitted['runtime']['decoding_preimage']
                and bindings.get('recipe') == recipe
                and bindings.get('inventory') == admitted['scope']['native_backend']['parameter_inventory']
                and numel == B_NUMEL and type(numel) is int, 'native checkpoint source/runtime/recipe/inventory drift')
        require(all(bindings.get(key) == admitted['anchors'][key] for key in
                ('own_start_sha256', 'common_anchor_sha256')), 'native checkpoint anchor drift')
        require(bindings.get('ordered_plan') == admitted['plan']['ordered_plan']
                and bindings.get('ordered_plan_sha256') == authority.digest(bindings['ordered_plan']), 'native checkpoint exact plan drift')
        require(bindings.get('data', {}).get('plan_sha256') == admitted['plan']['plan_sha256'], 'native checkpoint data/plan binding drift')
        require(callable(metadata_reserve), 'permanent metadata reservation required before native storage')
        bound_ledger = getattr(operation, 'ledger', operation)
        require(metadata_reserve == getattr(bound_ledger, 'reserve_metadata', None), 'native checkpoint metadata callback changed')
        self.context, self.permit, self.operation, self.gate = context, native_permit, operation, gate
        from . import checkpoint as base
        class Storage(base.NativeCheckpoints):
            def _extract_plan_rows(self, value): return _ce_plan_rows(value['ordered_plan'])
            def _scope(self, model, optimizer):
                result = super()._scope(model, optimizer)
                require(result[-1].type == 'cuda' and result[-1].index == 0, 'native CE checkpoint requires actual cuda:0 tensors')
                return result
            def _validate_progress(self, progress):
                base.validate_progress(progress, self.plan_rows)
                require(progress['C'] == progress['U'] and all(row['updated'] is True for row in progress['history']),
                        'native CE8 requires every C boundary to apply exactly one U')
                return progress
        self.storage = Storage(directory, bindings, numel, gate, metadata_reserve=metadata_reserve)

    def _admit(self, stage):
        return authority.require_execution_permit(self.permit, self.context, stage,
                                                 operation=self.operation, gate=self.gate)
    def attach_ledger(self, ledger):
        self._admit('checkpoint_construct')
        require(ledger is self.operation, 'native checkpoint accounting facade changed')
        self.storage.attach_ledger(ledger); return self
    def save(self, model, optimizer, progress, ledger):
        self._admit('checkpoint_save')
        require(ledger is self.operation, 'native checkpoint save accounting facade changed')
        return self.storage.save(model, optimizer, progress, ledger)
    def restore(self, model, optimizer):
        self._admit('checkpoint_restore')
        return self.storage.restore(model, optimizer)


class NativeCEBackend:
    """Actual delayed model lifecycle, shared by canonical and permuted plans."""
    synthetic_only = False

    def __init__(self, context, native_permit):
        authority.require_execution_permit(native_permit, context, 'backend_construct')
        services = context.get('native_services')
        require(type(services) is dict and callable(services.get('operation')) and callable(services.get('gate')),
                'bound native accounting/resource services required')
        self.operation, self.gate = services['operation'], services['gate']
        authority.require_execution_permit(native_permit, context, 'backend_construct', operation=self.operation, gate=self.gate)
        self.spec, self.context, self.permit = validate_backend_spec(context), context, native_permit
        self.recipe = copy.deepcopy(context['bindings']['recipe']['recipe'])
        # Validate the complete numerical/mask contract before deferred model
        # work, rather than discovering an invalid recipe at the first update.
        from . import ce_update
        ce_update.validate_recipe(dict(self.recipe, boundary_id=0, generation=0, U_before=0,
            symbolic_snapshot='0' * 64, start_policy_id='0' * 64), native_permit=native_permit)
        self.model = self.optimizer = self.tokenizer = self.torch = None
        self.trainable_names = list(self.recipe['trainable_names'])
        self.identity = self.ingestion_receipt = self.frozen_sha256 = self.policy = None
        self.last_checkpoint_progress = None
        self.closed = False
        self._cleanup_operation_id = self._cleanup_reserved_at = None
        self._cleanup_reserved = self._cleanup_finished = False
        self._cleanup_seal = None

    def _admit(self, stage, context=None):
        evidence = authority.require_execution_permit(self.permit, self.context if context is None else context,
            stage, recipe_sha256=authority.digest(self.recipe), operation=self.operation, gate=self.gate)
        require(self.spec == self.context['bindings']['scope']['native_backend'], 'native backend specification changed after admission')
        require(self.trainable_names == self.recipe['trainable_names'], 'native backend trainable names changed after admission')
        return evidence

    def _span(self, kind, **metadata):
        require(kind in self.spec['operation_bounds'], 'unpriced native backend operation')
        fields = dict(self.spec['operation_bounds'][kind], **metadata)
        if kind == 'fingerprint':
            # The durable ledger prices its whole profile and treats these two
            # fields as EXTRA passes beyond the builtin state_bytes lower.
            # Re-emitting a total profile would require N >= N+1 forever.
            fields.pop('identity_hash_attempts_upper')
            fields.pop('identity_hash_bytes_upper')
        return update._span(self.operation, kind, fields)

    def _ready(self):
        require(not self.closed and self.model is not None and self.optimizer is not None, 'native model/Adam not live')

    def _read_ref(self, reference):
        with self._span('input_io', path=reference['path']):
            self.gate('input_io')
            from .runner import observed_file
            actual = observed_file(reference['path'], self.operation)
            require(actual['sha256'] == reference['sha256'], 'native raw reference changed')
            return actual

    def load(self, context, prepared, observer):
        self._admit('runtime_anchor', context)
        require(not self.closed and self.model is None and self.optimizer is None, 'native load is once per fresh backend')
        require(callable(observer), 'controller stage observer required')
        from . import ce_qualification as qualification
        qualification.validate_actual_ingestion(context, prepared.receipt)
        self.ingestion_receipt = prepared.receipt
        self._reserve_cleanup()
        with self._span('module_import'):
            self.gate('module_import')
            from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as runtime, s0_pilot as pilot
            from .runner import verify_module_origins
            runtime.assert_runtime()
            import torch
            self.torch, self.runtime, self.pilot = torch, runtime, pilot
            verify_module_origins(context['bindings']['source']['code_sha256'])
            # The scope-bound resource service must check actual CUDA free and
            # reserved memory on every later guard, including after allocation.
            require(hasattr(self.gate, 'framework'), 'native resource gate must expose its framework binding')
            self.gate.framework = torch
        with self._span('configured_runtime'):
            self.gate('configured_runtime'); self.runtime.configure('cuda:0')
            actual = self.runtime.decoding_preimage('cuda:0')
            require(actual == context['bindings']['runtime']['decoding_preimage'], 'actual native runtime differs from frozen preimage')
        with self._span('process_rng_seed'):
            self.gate('process_rng_seed')
            import numpy as np
            seed = self.spec['run_seed']
            random.seed(seed); np.random.seed(seed % 2 ** 32)
            self.torch.manual_seed(seed); self.torch.cuda.manual_seed_all(seed)
            self.process_rng_seed = {'run_seed': seed, 'python': seed, 'numpy': seed % 2 ** 32,
                'torch_cpu': seed, 'torch_cuda_all': seed, 'restore_replaces_all_process_rng': True}
        with self._span('input_io', reason='native_bound_path_proof'):
            self.gate('input_io')
            require(Path(self.runtime.MODEL_DIR).resolve() == Path(self.spec['model_dir']).resolve(), 'native model path changed')
            require(Path(self.runtime.ASSET_RECEIPT).resolve() == Path(self.spec['asset_receipt_ref']['path']).resolve()
                    and self.spec['asset_receipt_ref']['sha256'] == self.runtime.ASSET_RECEIPT_SHA, 'native asset receipt pin changed')
            require(Path(self.runtime.TOKENIZER_RECEIPT).resolve() == Path(self.spec['tokenizer_receipt_ref']['path']).resolve()
                    and self.spec['tokenizer_receipt_ref']['sha256'] == self.runtime.TOKENIZER_RECEIPT_SHA, 'native tokenizer receipt pin changed')
            self.runtime.assert_outside_checkout(self.spec['output_directory'], self.spec['source_checkout'])
        self._read_ref(self.spec['asset_receipt_ref']); self._read_ref(self.spec['tokenizer_receipt_ref'])
        self._read_ref(self.spec['warm_delta_ref'])
        with self._span('tokenizer_load'):
            self.gate('tokenizer_load')
            self.tokenizer = self.runtime.load_tokenizer(self.spec['model_dir'], self.spec['tokenizer_receipt_ref']['path'],
                                                       asset_receipt=self.spec['asset_receipt_ref']['path'])
        with self._span('model_load'):
            self.gate('model_load')
            require(callable(getattr(self.gate, 'launch', None)), 'fresh adopted launch resource check required')
            self.gate.launch()
            anchors = context['bindings']['anchors']
            self.model, self.identity = self.runtime.load_anchor(self.spec['model_dir'], 'cuda:0', self.spec['asset_receipt_ref']['path'],
                delta_path=self.spec['warm_delta_ref']['path'], delta_sha256=self.spec['warm_delta_ref']['data_sha256'],
                arm_sha256=anchors['common_anchor_sha256'])
            require(self.identity['state_fingerprint'] == anchors['own_start_sha256']
                    and self.identity['weights_sha256'] == anchors['base_weights_sha256'], 'actual common warm/base identity drift')
        with self._span('inventory_configuration'):
            self.gate('inventory_configuration')
            chosen = self.pilot.set_inventory(self.model, *self.spec['inventory']['blocks'])
            require(chosen['trainable_names'] == self.trainable_names and chosen['trainable_numel'] == B_NUMEL,
                    'actual native B parameter/name inventory changed')
            named = dict(self.model.named_parameters())
            self.inventory = [{'name': name, 'shape': list(named[name].shape), 'dtype': 'F32'} for name in self.trainable_names]
            require(self.inventory == self.spec['parameter_inventory'], 'actual native parameter shapes differ from frozen source inventory')
            self.checkpointing = self.pilot.enable_checkpointing(self.model)
            settings = self.recipe['optimizer']; selected = set(self.trainable_names)
            self.optimizer = self.torch.optim.AdamW([p for name, p in self.model.named_parameters() if name in selected],
                lr=settings['lr'], betas=tuple(settings['betas']), eps=settings['eps'], weight_decay=settings['weight_decay'],
                foreach=settings['foreach'], capturable=settings['capturable'])
            require(not self.optimizer.state, 'native CE Adam must start genuinely fresh and lazy')
            self.model.eval()
        self.policy = self.fingerprint('initial_common_warm', initial=True)
        require(self.policy == context['bindings']['anchors']['common_anchor_sha256'], 'inventory/Adam setup changed common warm anchor')
        return {'actual_anchor_sha256': self.policy, 'tokenizer_receipt_sha256': self.spec['tokenizer_receipt_ref']['sha256'],
            'actual_runtime_preimage': actual, 'decoder_checkpointing': {'use_reentrant': False},
            'head_checkpointing': True, 'inventory': chosen, 'fresh_lazy_adam': True,
            'process_rng_seed': self.process_rng_seed, 'native_qualified': False}

    def fingerprint(self, reason, *, initial=False):
        self._admit('refresh_policy'); self._ready()
        state = dict(self.model.named_parameters(remove_duplicate=False))
        state.update(dict(self.model.named_buffers(remove_duplicate=False)))
        state_bytes = sum(value.numel() * value.element_size() for value in {id(v): v for v in state.values()}.values())
        with self._span('fingerprint', reason=reason, state_bytes=state_bytes):
            self.gate('fingerprint')
            values = self.runtime.state_digests(self.model)
            policy = self.runtime.combine(values)
            _, _, U = update._actual_optimizer(self.model, self.optimizer, self.recipe, self.torch)
            frozen = self.runtime.combine({name: value for name, value in values.items() if name not in self.trainable_names})
            if initial:
                require(U == 0 and not self.optimizer.state, 'initial native fingerprint requires U0/fresh Adam')
                self.frozen_sha256 = frozen
            else: require(frozen == self.frozen_sha256, 'native frozen parameters/buffers changed')
            self.policy, self.U = policy, U
            return policy

    def _journal(self, journal):
        require(self.operation is journal or getattr(self.operation, 'ledger', None) is journal,
                'CE native operation facade must be linked to this durable journal')
        require(not getattr(journal, 'fatal', True) and not getattr(journal, 'unknown_wall', True),
                'native journal cannot admit fatal/unknown-wall work')

    def open_journal(self, context, prepared, proof, resume, reason):
        """Open the verified durable artifact before framework/model loading.

        Pure metadata/ingestion bootstrap remains in its separately retained
        outer ledger. This journal charges every subsequent model/update/storage
        operation; no already-executed model costs are silently migrated here.
        """
        self._admit('journal_construct', context)
        require(self.model is None and self.optimizer is None, 'link native durable journal before model allocation')
        require(type(resume) is bool and (not resume or type(reason) is str and reason.strip()), 'documented native journal resume')
        require(type(proof) is dict and proof.get('fresh_process') is True and proof.get('boot_verified') is True
                and authority._hex(proof.get('boot_identity_sha256')), 'verified fresh worker boot evidence required')
        from . import ce_qualification, ce_journal
        ce_qualification.validate_actual_ingestion(context, prepared.receipt)
        admitted = context['bindings']; artifact = admitted.get('native_ledger')
        reference = context.get('refs', {}).get('native_ledger_ref')
        _ref(reference)
        require(reference == admitted['scope'].get('native_ledger_ref') and type(artifact) is dict
                and artifact.get('schema') == 'CE8-NATIVE-LEDGER-SPEC-1', 'verified explicit native ledger artifact required')
        require(artifact.get('owned_directory') == str(Path(self.spec['output_directory']) / 'ledger')
                and artifact.get('plan') == admitted['plan']
                and artifact.get('ingestion_receipt_sha256') == prepared.receipt['receipt_sha256'], 'native ledger owner/plan/actual ingestion binding')
        bindings = artifact.get('bindings')
        require(type(bindings) is dict and bindings.get('source') == admitted['source']['code_sha256']
                and bindings.get('runtime') == admitted['runtime']['decoding_preimage']
                and bindings.get('recipe') == self.recipe
                and bindings.get('anchors') == {key: admitted['anchors'][key] for key in
                    ('own_start_sha256', 'common_anchor_sha256')}
                and bindings.get('prepared_rows') == prepared.receipt['prepared_row_bindings'], 'native ledger source/runtime/recipe/census changed')
        require(callable(getattr(self.operation, 'attach_journal', None)), 'stable source-bound dispatcher must own journal linkage')
        with self._span('journal_construct'):
            self.gate('journal_construct')
            journal = ce_journal.CeJournal(artifact['owned_directory'], artifact['plan'], bindings, artifact['caps'],
                                         synthetic_cpu=False, native_permit=self.permit, resume=resume)
        self.operation.attach_journal(journal)
        self._journal(journal)
        journal.start_invocation(resume=resume, boot_id=proof['boot_identity_sha256'], reason=reason)
        return journal

    def checkpoints(self, context, journal):
        self._admit('checkpoint_construct', context); self._ready(); self._journal(journal)
        require(self.ingestion_receipt is not None, 'native source/token census precedes checkpoint bindings')
        admitted = context['bindings']; plan = admitted['plan']
        bindings = {'checkpoint_mode': CHECKPOINT_MODE, 'synthetic_cpu': False,
            'code': admitted['source']['code_sha256'], 'runtime': admitted['runtime']['decoding_preimage'],
            'recipe': self.recipe, 'inventory': self.inventory, 'ordered_plan': plan['ordered_plan'],
            'ordered_plan_sha256': authority.digest(plan['ordered_plan']),
            'own_start_sha256': admitted['anchors']['own_start_sha256'],
            'common_anchor_sha256': admitted['anchors']['common_anchor_sha256'],
            'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
            'process_rng_seed': self.process_rng_seed,
            'data': {'plan_sha256': plan['plan_sha256'], 'ingestion_receipt_sha256': self.ingestion_receipt['receipt_sha256'],
                'source_receipt_sha256': self.ingestion_receipt['source_receipt_sha256'],
                'prepared_row_bindings_sha256': authority.digest(self.ingestion_receipt['prepared_row_bindings']),
                'census': self.ingestion_receipt['census']}}
        from .checkpoint import constructor_hash_lower_bound
        with self._span('checkpoint_construct', **constructor_hash_lower_bound(bindings)):
            self.gate('checkpoint_construct')
            checkpoints = NativeCECheckpoints(Path(self.spec['output_directory']) / 'checkpoints', context,
                bindings, B_NUMEL, self.gate, metadata_reserve=journal.reserve_metadata,
                operation=self.operation, native_permit=self.permit)
            return checkpoints.attach_ledger(self.operation)

    def update(self, rows, journal):
        self._admit('ce_boundary'); self._ready(); self._journal(journal)
        progress = journal.progress_record()
        require(progress['C'] == progress['U'] == self.U, 'native CE Adam/progress divergence')
        from . import ce_update
        recipe = dict(self.recipe, boundary_id=progress['C'], U_before=progress['U'],
            generation=progress['generation'], symbolic_snapshot=progress['snapshot'], start_policy_id=self.policy)
        result = ce_update.boundary_update(self.model, self.optimizer, rows, recipe, self.operation, native_permit=self.permit)
        require(result['frozen_state_sha256'] == self.frozen_sha256 and result['U_after'] == self.U + 1,
                'actual native CE result/frozen/Adam proof changed')
        self.policy, self.U = result['final_policy_id'], result['U_after']
        return result

    def save(self, checkpoints, journal):
        self._admit('checkpoint_save'); self._ready(); self._journal(journal)
        progress = journal.progress_record()
        require(progress['C'] == progress['U'] == self.U, 'native save only at committed CE boundary')
        with self._span('checkpoint_adapter_io'):
            self.gate('checkpoint_adapter_io')
            manifest = checkpoints.save(self.model, self.optimizer, progress, self.operation)
            require(manifest['progress'] == progress, 'native saved progress readback changed')
        self.last_checkpoint_progress = copy.deepcopy(progress)
        return manifest

    def restore(self, checkpoints, journal):
        self._admit('checkpoint_restore'); self._ready(); self._journal(journal)
        checkpoints.attach_ledger(self.operation)
        with self._span('checkpoint_adapter_io'):
            self.gate('checkpoint_adapter_io')
            restored = checkpoints.restore(self.model, self.optimizer)
        self.last_checkpoint_progress = None
        # Controller installs this one-use fence before refresh_policy. Costs
        # remain in the live journal and are never restored from the checkpoint.
        return restored

    def refresh_policy(self, journal):
        self._admit('refresh_policy'); self._ready(); self._journal(journal)
        require(not journal.requires_restore and not journal.restore_pending, 'install fresh restore fence before actual policy proof')
        policy = self.fingerprint('restored_actual_state')
        require(journal.progress_record()['U'] == self.U, 'restored actual Adam U differs from installed fence')
        return policy

    def _donor(self, context, journal, stage, prefix):
        self._admit(stage, context); self._ready(); self._journal(journal)
        progress = journal.progress_record()
        require(progress['C'] == progress['U'] == self.U and self.U > 0, 'donor requires completed cleared CE updates')
        if context['phase'] == 'training': require(self.U == 375, 'production CE donor requires all 375 updates')
        final = self.fingerprint('donor_final_master')
        anchors, warm = context['bindings']['anchors'], self.spec['warm_delta_ref']
        bindings = {'schema': update.DONOR_SCHEMA, 'synthetic_cpu': False,
            'approved': True, 'training_authorized': context['phase'] == 'training',
            'base_weights_sha256': anchors['base_weights_sha256'], 'warm_delta_file_sha256': warm['sha256'],
            'warm_delta_data_sha256': warm['data_sha256'], 'warm_delta_state_sha256': warm['state_sha256'],
            'common_anchor_sha256': anchors['common_anchor_sha256'], 'own_start_sha256': anchors['own_start_sha256'],
            'own_start_kind': 'independent', 'own_start_components': [], 'final_master_sha256': final,
            'frozen_state_sha256': self.frozen_sha256, 'trainable_names_sha256': update.digest(sorted(self.trainable_names)),
            'operation': self.operation, 'gate': self.gate}
        target = Path(self.spec['output_directory']) / (prefix + '.' + progress['snapshot'] + '.safetensors')
        donor = update.extract_donor(self.model, Path(self.spec['model_dir']) / 'model.safetensors', warm['path'],
                                    self.trainable_names, target, bindings, native_permit=self.permit)
        require(self.fingerprint('post_donor_unchanged_master') == final, 'native donor extraction mutated trained master')
        return donor

    def donor(self, context, journal):
        self._admit('donor_publish_verify', context)
        require(context['phase'] == 'training', 'qualification cannot publish a production donor')
        return self._donor(context, journal, 'donor_publish_verify', 'DONOR')

    def qualification_delta(self, context, journal):
        self._admit('delta_roundtrip', context)
        require(context['phase'] == 'qualification', 'qualification delta is separately scoped discarded work')
        return self._donor(context, journal, 'delta_roundtrip', 'QUALIFICATION_DELTA')

    def _reserve_cleanup(self):
        self._admit('cleanup')
        require(self._cleanup_operation_id is None, 'one native cleanup reservation per backend lifetime')
        self._cleanup_operation_id, self._cleanup_reserved_at = uuid.uuid4().hex, time.perf_counter()
        # This permanent deferred reservation precedes every framework/model
        # allocation. Its registered wall ceiling covers the whole lifetime;
        # physical release still happens when later guards or writes refuse.
        self.operation('cleanup', 'started', operation_id=self._cleanup_operation_id,
            deferred_model_cleanup=True, **self.spec['operation_bounds']['cleanup'])
        self._cleanup_reserved = True
        self._cleanup_seal = _PREPAID_CLEANUP_SEAL

    def cleanup(self):
        admission_failure = None
        try: self._admit('cleanup')
        except BaseException as error:
            # A source-created permanent lifetime reservation permits only
            # deallocation of already-held resources when authority expires.
            # Unconstructed/counterfeit objects cannot turn flags into that
            # reservation, and the original denial is still raised below.
            if getattr(self, '_cleanup_seal', None) is not _PREPAID_CLEANUP_SEAL or getattr(self, '_cleanup_reserved', False) is not True:
                raise
            admission_failure = error
        if not self._cleanup_reserved:
            require(self.model is self.optimizer is self.tokenizer is self.torch is None,
                    'no native allocations without prepaid cleanup')
            self.closed = True
            return
        if self._cleanup_finished: return
        primary, extras = admission_failure, []
        try:
            try: self.gate('cleanup')
            except BaseException as error:
                if primary is None: primary = error
                else: extras.append({'type': type(error).__name__, 'message': str(error)[:1000]})
            try:
                if self.optimizer is not None: self.optimizer.zero_grad(set_to_none=True)
            except BaseException as error:
                if primary is None: primary = error
                else: extras.append({'type': type(error).__name__, 'message': str(error)[:1000]})
        finally:
            self.optimizer = self.model = self.tokenizer = None
            self.closed = True
            try:
                gc.collect()
                if self.torch is not None and self.torch.cuda.is_initialized():
                    self.torch.cuda.synchronize(0); self.torch.cuda.empty_cache()
            except BaseException as error:
                if primary is None: primary = error
                else: extras.append({'type': type(error).__name__, 'message': str(error)[:1000]})
        phase = 'failed' if primary is not None else 'returned'
        fields = {'wall_s': time.perf_counter() - self._cleanup_reserved_at,
                  'deferred_model_cleanup': True, 'references_cleared': True}
        if primary is not None: fields['error'] = {'type': type(primary).__name__, 'message': str(primary)[:1000]}
        try: self.operation('cleanup', phase, operation_id=self._cleanup_operation_id, **fields)
        except BaseException as error:
            if primary is None: primary = error
            else: extras.append({'type': type(error).__name__, 'message': str(error)[:1000]})
        self._cleanup_finished = True
        if primary is not None:
            primary.cleanup_secondary_failures = extras
            raise primary


def create_backend(context):
    """Future registered factory. This function is not in any dispatch registry."""
    permit = context.get('native_execution_permit') if type(context) is dict else None
    authority.require_execution_permit(permit, context, 'backend_construct')
    return NativeCEBackend(context, permit)
