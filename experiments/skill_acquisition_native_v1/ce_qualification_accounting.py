"""Separate durable qualification reservations; no issuer, backend or native grant.

Native construction requires an existing source-issued permit and frozen actual
worker/ingestion evidence before paths. Pure fixtures are explicitly labeled.
The protected CE state validates declared qualification Adam batches, never the
production exposure order. This ledger does not establish numerical efficacy,
checkpoint qualification or native PASS.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from . import ce_journal as ce, ce_qualification as q, ce_native_authority as authority

SCHEMA = 'CE8-QUALIFICATION-ACCOUNTING-1'
SPEC = 'CE8-QUALIFICATION-ACCOUNTING-SPEC-1'
FIXTURE = 'CE8-QUALIFICATION-ACCOUNTING-FIXTURE-1'
PROOF = 'CE8-QUALIFICATION-WORKER-PROOF-1'
HEADER, JOURNAL = 'QUALIFICATION_BINDINGS.json', 'QUALIFICATION_RESERVATIONS.jsonl'
MAX_LINE, CHUNK = 4 * 1024 ** 2, 1024 ** 2
KINDS = ce.NATIVE_KINDS | frozenset(('journal_construct', 'qualification_mask',
    'qualification_likelihood', 'qualification_gradient_reset',
    'qualification_reference_forward', 'qualification_candidate_forward',
    'qualification_backward', 'qualification_gradient_validation', 'qualification_rng',
    'qualification_tensor_write', 'qualification_artifact', 'qualification_checkpoint_configuration',
    'qualification_artifact_readback', 'qualification_reducer_overhead', 'resource_probe', 'resource_cuda_probe'))
COST_FIELDS = q.COST_FIELDS
canonical, digest, require = q.canonical, q.digest, q.require


def _clone(value): return json.loads(canonical(value))
def _profile(value):
    require(type(value) is dict and set(value) in (set(ce.PROFILE), set(ce.PROFILE) | {'reducer_work_bytes_upper'}), 'complete qualification profile')
    ce._profile({k: value[k] for k in ce.PROFILE})
    require(ce._int(value.get('reducer_work_bytes_upper', 0)), 'finite reducer work profile')
    return value

def _normalize(fields):
    fields = dict(fields)
    for alias, key in (('io_read_bytes_upper', 'read_bytes_upper'), ('io_write_bytes_upper', 'write_bytes_upper')):
        if alias in fields:
            require(key not in fields or fields[key] == fields[alias], 'conflicting physical IO aliases')
            fields[key] = fields.pop(alias)
    return fields
def _price(value):
    return {'operation_attempts_upper': 1, 'io_read_bytes_upper': value['read_bytes_upper'],
        'io_write_bytes_upper': value['write_bytes_upper'],
        'identity_hash_attempts_upper': value['identity_hash_attempts_upper'],
        'identity_hash_bytes_upper': value['identity_hash_bytes_upper'],
        'metadata_bytes_upper': 0, 'wall_seconds_upper': value['wall_s_upper']}
def _add(left, right): return {key: left.get(key, 0) + right.get(key, 0) for key in COST_FIELDS}


class QualificationAccounting:
    def __init__(self, context, permit, proof, prepared_receipt=None, *,
                 bootstrap_handle=None, bootstrap_verifier=None, clock=None):
        from . import ce_bootstrap_accounting as boot
        fixture = type(context) is dict and context.get('schema') == FIXTURE
        if not fixture:
            # Registry refusal precedes context/header/replay hashes and paths.
            services = context.get('native_services', {}) if type(context) is dict else {}
            if type(services) is not dict: services = {}
            if bootstrap_handle is None: bootstrap_handle = services.get('bootstrap_handle')
            if bootstrap_verifier is None: bootstrap_verifier = services.get('bootstrap_verifier')
            boot.require_registered_verifier(bootstrap_verifier)
            owner = context.get('bindings', {}).get('scope', {}).get('qualifier_accounting_owner') if type(context) is dict else None
            boot.verify_prepayment(bootstrap_handle, context, proof, owner, verifier=bootstrap_verifier)
            require(clock is None, 'native qualification monotonic clock cannot be injected')
        self.clock = time.monotonic if clock is None else clock
        require(callable(self.clock), 'explicit monotonic clock source')
        self._wall_origin = self.clock()
        require(q.finite(self._wall_origin), 'finite qualification monotonic origin')
        self._wall_ready = False
        self.actual_wall_seconds, self.last_clock, self.worker_open = 0.0, self._wall_origin, False
        if fixture and bootstrap_handle is None:
            outer = boot.BootstrapAccounting(context, None, proof, clock=self.clock)
            spec = context['bindings']['scope']['qualifier_accounting']
            bounds = _price(spec['setup_bounds'])
            bounds['metadata_bytes_upper'] = MAX_LINE * 2
            owner = {'accounting_spec_sha256': digest(spec), 'prepared_receipt_sha256': prepared_receipt['receipt_sha256']}
            bootstrap_handle = outer.reserve_setup(owner, bounds)
            boot.verify_prepayment(bootstrap_handle, context, proof, owner)
        require(type(bootstrap_handle) is boot.PrepaymentHandle, 'actual outer prepayment handle required')
        self.bootstrap_handle = bootstrap_handle
        self.bootstrap_handle.begin()
        try:
            self._initialize(context, permit, proof, prepared_receipt)
            expected_owner = {'accounting_spec_sha256': digest(self.spec), 'prepared_receipt_sha256': prepared_receipt['receipt_sha256']}
            require(self.bootstrap_handle.evidence['owner'] == expected_owner, 'prepaid owner differs from actual constructor specification')
            self._wall_ready = True
            if not self.unknown_wall: self.tick_wall()
            self.bootstrap_handle.complete()
        except BaseException as error:
            try: self.bootstrap_handle.fail(error)
            except BaseException as record_error:
                error.bootstrap_accounting_failure = {'type': type(record_error).__name__, 'message': str(record_error)[:1000]}
                raise error from record_error
            raise

    def _initialize(self, context, permit, proof, prepared_receipt=None):
        # No path/proof traversal or IO before the native capability check.
        fixture = type(context) is dict and context.get('schema') == FIXTURE
        if fixture:
            require(context.get('mode') == 'synthetic_fixture' and context.get('phase') == 'qualification'
                    and context.get('approved') is False and context.get('training_authorized') is False
                    and permit is None, 'explicit non-native qualification accounting fixture only')
        else:
            authority.require_execution_permit(permit, context, 'journal_construct')
            require(context['phase'] == 'qualification', 'qualification accounting never authorizes production')
        require(type(proof) is dict and proof.get('schema', PROOF) == PROOF
                and proof.get('fresh_process') is True and proof.get('boot_verified') is True
                and q.hex64(proof.get('worker_identity_sha256'))
                and q.hex64(proof.get('boot_identity_sha256')), 'verified explicit qualification worker proof required')
        if not fixture:
            require(context.get('worker_proof_sha256') == digest(proof), 'worker proof differs from frozen native context')
        require(context.get('arm') in ('canonical', 'permuted') and q.hex64(context.get('request_sha256'))
                and q.hex64(context.get('scope_sha256')), 'one qualification request/scope/arm required')
        scope = context.get('bindings', {}).get('scope', {})
        spec = scope.get('qualifier_accounting')
        require(type(spec) is dict and spec.get('schema') == SPEC, 'explicit qualification accounting specification required')
        require(type(spec.get('owned_directory')) is str and Path(spec['owned_directory']).is_absolute(), 'absolute accounting owner required')
        require(type(prepared_receipt) is dict and prepared_receipt.get('receipt_sha256') == spec.get('ingestion_receipt_sha256')
                and digest({k: v for k, v in prepared_receipt.items() if k != 'receipt_sha256'}) == prepared_receipt['receipt_sha256'],
                'exact frozen actual prepared row receipt required')
        if not fixture: q.validate_actual_ingestion(context, prepared_receipt)
        profiles = spec.get('operation_profiles')
        require(type(profiles) is dict and set(profiles) == KINDS, 'explicit profile for every qualification operation required')
        for value in profiles.values(): _profile(value)
        require(q.integer(profiles['qualification_reducer_overhead'].get('reducer_work_bytes_upper'), 1), 'explicit reducer work upper required')
        q._costs(spec.get('global_caps'))
        require(type(spec.get('stage_caps')) is dict and set(spec['stage_caps']) == set(q.STAGES), 'all exact qualification stage caps required')
        for value in spec['stage_caps'].values(): q._costs(value, stage=True)
        _profile(spec.get('setup_bounds'))
        require(q.finite(spec.get('physical_wall_seconds_upper'), True), 'explicit physical qualification deadline')
        execution = scope.get('qualifier_execution')
        require(type(execution) is dict, 'qualification batch plan required')
        rows = prepared_receipt.get('prepared_row_bindings')
        require(type(rows) is list and rows and all(type(row) is dict and type(row.get('id')) is str for row in rows)
                and len({row['id'] for row in rows}) == len(rows), 'distinct compact prepared row census required')
        census = {row['id']: row for row in rows}
        witnesses, gradient, batches = (execution.get(k) for k in ('all_witness_ids', 'gradient_batch_ids', 'adam_batches'))
        valid = lambda ids: type(ids) is list and ids and len(set(ids)) == len(ids) and all(type(rid) is str and rid in census for rid in ids)
        require(valid(witnesses) and valid(gradient) and len(gradient) == 8 and set(gradient) <= set(witnesses)
                and type(batches) is list and 2 <= len(batches) <= 8
                and all(valid(batch) and len(batch) == 8 and set(batch) <= set(witnesses) for batch in batches),
                'exact bounded witness/gradient/Adam qualification rows required')
        for row in rows:
            require(all(q.integer(row.get(k), 1) for k in ('prompt_tokens', 'labeled_tokens', 'full_sequence_tokens', 'prepared_hash_bytes_upper'))
                    and row['full_sequence_tokens'] == row['prompt_tokens'] + row['labeled_tokens']
                    and all(q.hex64(row.get(k)) for k in ('prompt_ids_sha256', 'completion_ids_sha256', 'prepared_row_sha256', 'target_sha256')),
                    'exact compact qualification row hash/token schema')
        recipe = context.get('bindings', {}).get('recipe', {}).get('recipe')
        from . import ce_update
        require(type(recipe) is dict and recipe.get('synthetic_cpu') is fixture
                and digest(recipe) == context.get('static_recipe_sha256'), 'qualification recipe identity/device binding')
        ce_update.validate_recipe(dict(recipe, boundary_id=0, generation=0, U_before=0,
            symbolic_snapshot='0' * 64, start_policy_id='0' * 64), native_permit=permit)
        self.fixture, self.permit, self.context, self.proof = fixture, permit, context, _clone(proof)
        self.spec, self.profiles, self.census, self.recipe = _clone(spec), _clone(profiles), _clone(census), _clone(recipe)
        self.allowed_rows, self.adam_batches = set(witnesses), _clone(batches)
        self.scope_role = 'qualification_only'
        self.request_sha256, self.scope_sha256 = context['request_sha256'], context['scope_sha256']
        self._current_proof = self.proof
        self._spec_sha256, self._profiles_sha256 = digest(self.spec), digest(self.profiles)
        self.native_activation_gaps = {
            'outer_setup_reservation': 'A registered outer bootstrap ledger must permanently reserve constructor/header/replay work before these files exist.',
            'directory_publication': 'File fsync is implemented; platform-specific parent-directory publication durability still needs verified native service evidence.'}
        predecessor = spec.get('ingestion_bootstrap_ref')
        if not fixture:
            setup_ref = spec.get('outer_setup_reservation_ref')
            require(type(setup_ref) is dict and q.hex64(setup_ref.get('sha256'))
                    and type(setup_ref.get('path')) is str and Path(setup_ref['path']).is_absolute()
                    and q.integer(setup_ref.get('bytes'), 1), 'outer-owned durable setup reservation reference required before paths')
            require(type(predecessor) is dict and q.hex64(predecessor.get('sha256'))
                    and type(predecessor.get('path')) is str and Path(predecessor['path']).is_absolute()
                    and q.integer(predecessor.get('bytes'), 1), 'scope-bound external ingestion/bootstrap evidence required')
        self.external_predecessors = {'ingestion_tokenizer'} if predecessor is not None else set()
        self.reducer_reserved_bytes = 0
        self.reducer_by_stage = dict.fromkeys(q.STAGES, 0)
        self.fatal = self.unknown_wall = False
        self.active = None
        self.operations, self.attempts, self.events, self.metadata_ids = {}, [], [], set()
        self.cumulative = {k: 0 for k in COST_FIELDS}
        self.by_stage = {stage: {k: 0 for k in COST_FIELDS} for stage in q.STAGES}
        self.directory = Path(spec['owned_directory']).resolve()
        require(not Path(spec['owned_directory']).is_symlink(), 'regular owned qualification directory')
        owner = scope.get('native_backend', {}).get('output_directory', spec['owned_directory'])
        self.metadata_root = Path(owner).resolve()
        require(self.directory.is_relative_to(self.metadata_root) and not Path(owner).is_symlink(), 'accounting directory outside admitted output owner')
        if not fixture:
            checkout = Path(scope['native_backend']['source_checkout']).resolve()
            require(not self.metadata_root.is_relative_to(checkout), 'qualification output must be outside source checkout')
        self.header_path, self.path = self.directory / HEADER, self.directory / JOURNAL
        self.header = {'schema': SCHEMA, 'fixture': fixture, 'native_qualified': False,
            'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'], 'arm': context['arm'],
            'static_recipe_sha256': context['static_recipe_sha256'], 'specification': self.spec,
            'prepared_receipt_sha256': prepared_receipt['receipt_sha256'], 'recipe': self.recipe,
            'anchors': _clone(context['bindings']['anchors']), 'source': _clone(context['bindings'].get('source', {})),
            'runtime': _clone(context['bindings'].get('runtime', {})),
            'bootstrap_owner': self.bootstrap_handle.evidence['owner'],
            'wall_rule': 'source-owned time.monotonic intervals with same-boot worker/end evidence'}
        self.binding_sha256 = digest(self.header)
        self._make_ce_state(context, rows)
        setup = self.spec['setup_bounds']
        raw_header = canonical(self.header) + b'\n'
        require(len(raw_header) <= MAX_LINE, 'qualification header ceiling')
        self._known_size = 0
        self._header_cost_prepaid = False
        header_costs = {'metadata_bytes_upper': len(raw_header), 'io_write_bytes_upper': len(raw_header),
                       'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': len(canonical(self.header))}
        if self.header_path.exists() or self.path.exists():
            require(self.header_path.is_file() and self.path.is_file() and not self.header_path.is_symlink()
                    and not self.path.is_symlink(), 'both regular qualification history files required')
            size = self.header_path.stat().st_size + self.path.stat().st_size
            require(size <= setup['read_bytes_upper'] and size <= self.spec['global_caps']['metadata_bytes_upper']
                    and setup['identity_hash_bytes_upper'] >= size, 'bounded paid replay before reads')
            require(setup['identity_hash_attempts_upper'] >= size // 200 + 1,
                    'conservative replay hash-attempt admission before reads')
            # The explicit outer worker proof admits this complete bounded
            # bootstrap; its retained setup row charges the full reservation.
            with self.header_path.open('rb') as stream: observed = stream.read(MAX_LINE + 1)
            require(observed == raw_header, 'qualification frozen header drift')
            read_count = len(observed)
            with self.path.open('rb') as stream:
                while raw := stream.readline(MAX_LINE + 1):
                    read_count += len(raw); require(read_count <= size, 'qualification history grew during replay')
                    require(len(raw) <= MAX_LINE and raw.endswith(b'\n'), 'partial qualification tail; no repair/truncation')
                    entry = ce._json(raw); body = {k: v for k, v in entry.items() if k != 'sha256'}
                    require(canonical(entry) + b'\n' == raw and body['seq'] == len(self.events)
                            and body['previous'] == self._head() and body['binding_sha256'] == self.binding_sha256
                            and digest(body) == entry['sha256'], 'qualification history hash/order/binding drift')
                    self._apply(body['kind'], body['payload'], replay=True)
                    self.events.append(entry); self._known_size += len(raw)
            require(read_count == size and setup['identity_hash_attempts_upper'] >= len(self.events) + 1, 'replay bytes/hash attempt envelope')
        else:
            require(setup['write_bytes_upper'] >= len(raw_header), 'setup header write envelope')
            self._check_charge(None, _add(_price(setup), header_costs))
            self._charge(None, header_costs); self._header_cost_prepaid = True
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.header_path.open('xb') as stream:
                for start in range(0, len(raw_header), CHUNK):
                    block = raw_header[start:start + CHUNK]
                    require(stream.write(block) == len(block), 'short qualification header')
                stream.flush(); os.fsync(stream.fileno())
            with self.path.open('xb') as stream: stream.flush(); os.fsync(stream.fileno())
            self._known_size = 0
        self._event('setup', {'profile': setup, 'header_costs': header_costs if self._header_cost_prepaid else {}})
        self._event('worker', {'proof': self.proof, 'monotonic': self._wall_origin})

    def _make_ce_state(self, context, rows):
        # Protected pure helper composition AFTER the outer admission. No base
        # journal constructor, path/IO, framework or synthetic native permit.
        state = object.__new__(ce.CeJournal)
        state.synthetic_cpu = self.fixture
        state.native_permit = self.permit
        state.allowed_kinds, state.profiles = ce.NATIVE_KINDS, self.profiles
        state.census = {row['id']: {k: row[k] for k in ('id', 'prompt_tokens', 'labeled_tokens', 'full_sequence_tokens',
            'prompt_ids_sha256', 'completion_ids_sha256', 'prepared_row_sha256')} for row in rows}
        state.bindings = {'recipe': self.recipe, 'prepared_rows': [self.census[rid] for batch in self.adam_batches for rid in batch],
            'anchors': {k: context['bindings']['anchors'][k] for k in ('own_start_sha256', 'common_anchor_sha256')}}
        state.plan_rows = self.adam_batches
        state.plan = {'plan_sha256': digest({'qualification_adam_batches': self.adam_batches})}
        state.static_recipe_sha256 = context['static_recipe_sha256']
        state.progress = {'C': 0, 'U': 0, 'history': [], 'generation': 0}
        state.progress['snapshot'] = digest(dict(state.progress, plan_sha256=state.plan['plan_sha256']))
        state.operations, state.events, state.used_restores = {}, self.events, set()
        state.active_attempt = state.current_invocation = None
        state.requires_restore = state.restore_pending = state.unknown_wall = state.fatal = False
        self._ce = state

    @property
    def generation(self): return self._ce.generation
    @property
    def symbolic_snapshot(self): return self._ce.symbolic_snapshot
    @property
    def requires_restore(self): return self._ce.requires_restore
    @property
    def restore_pending(self): return self._ce.restore_pending
    def _head(self): return self.events[-1]['sha256'] if self.events else None
    def _admit(self):
        require(not self.fatal and not self.unknown_wall, 'qualification fatal or unknown-wall history')
        if not self.fixture:
            authority.require_execution_permit(self.permit, self.context, 'journal_operation')
            require(digest(self.spec) == self._spec_sha256 and digest(self.profiles) == self._profiles_sha256,
                    'native qualification accounting specification/prices mutated')
    def _check_charge(self, stage, amounts):
        for key, value in amounts.items():
            require(key in COST_FIELDS and ce._finite(value) and (key == 'wall_seconds_upper' or type(value) is int), 'finite qualification reservation')
            require(self.cumulative[key] + value <= self.spec['global_caps'][key], 'global qualification ceiling: ' + key)
            if stage is not None: require(self.by_stage[stage][key] + value <= self.spec['stage_caps'][stage][key], 'stage qualification ceiling: ' + key)
    def _charge(self, stage, amounts):
        self._check_charge(stage, amounts)
        self.cumulative = _add(self.cumulative, amounts)
        if stage is not None: self.by_stage[stage] = _add(self.by_stage[stage], amounts)
    def _event(self, kind, payload):
        self._admit()
        if self._wall_ready and kind not in ('wall', 'worker', 'end'): self.tick_wall()
        body = {'seq': len(self.events), 'previous': self._head(), 'binding_sha256': self.binding_sha256,
                'kind': kind, 'payload': _clone(payload)}
        raw = canonical(dict(body, sha256=digest(body))) + b'\n'
        require(len(raw) <= MAX_LINE, 'qualification event ceiling')
        stage, amounts, action = self._prepare(kind, body['payload'])
        overhead = {'metadata_bytes_upper': len(raw), 'io_write_bytes_upper': len(raw),
                    'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': len(canonical(body))}
        self._charge(stage, _add(amounts, overhead))  # permanent admission precedes metadata IO
        try:
            require(not self.path.is_symlink() and self.path.stat().st_size == self._known_size, 'qualification single writer history')
            with self.path.open('ab') as stream:
                require(stream.write(raw) == len(raw), 'short qualification event'); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            self.fatal = True; self._ce.requires_restore = True
            raise
        action()
        self.events.append(ce._json(raw)); self._known_size += len(raw)
    def _apply(self, kind, payload, *, replay):
        stage, amounts, action = self._prepare(kind, payload)
        body = {'seq': len(self.events), 'previous': self._head(), 'binding_sha256': self.binding_sha256,
                'kind': kind, 'payload': payload}
        raw = canonical(dict(body, sha256=digest(body))) + b'\n'
        self._charge(stage, _add(amounts, {'metadata_bytes_upper': len(raw), 'io_write_bytes_upper': len(raw),
            'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': len(canonical(body))})); action()
    def _prepare(self, kind, payload):
        if kind == 'setup':
            amount = _price(payload['profile'])
            if not self._header_cost_prepaid: amount = _add(amount, payload.get('header_costs', {}))
            return None, amount, lambda: setattr(self, '_header_cost_prepaid', False)
        if kind == 'wall':
            require(payload['boot_identity_sha256'] == self._current_proof['boot_identity_sha256']
                    and payload['worker_identity_sha256'] == self._current_proof['worker_identity_sha256']
                    and q.finite(payload['monotonic']) and payload['monotonic'] >= self.last_clock
                    and payload['elapsed_delta'] == payload['monotonic'] - self.last_clock, 'same-boot exact qualification wall interval')
            def apply():
                self.last_clock = payload['monotonic']; self.actual_wall_seconds += payload['elapsed_delta']
            return payload['stage'], {'wall_seconds_upper': payload['elapsed_delta']}, apply
        if kind == 'end':
            require(self.active is None and not any(op['phase'] == 'started' and op['worker_id'] == self._current_proof['worker_identity_sha256']
                    for op in self.operations.values()), 'all current worker operations must return before wall end')
            require(payload['worker_identity_sha256'] == self._current_proof['worker_identity_sha256']
                    and q.finite(payload.get('monotonic')) and payload['monotonic'] == self.last_clock,
                    'exact wall end worker/monotonic identity')
            return None, {}, lambda: setattr(self, 'worker_open', False)
        if kind == 'worker':
            def apply():
                if self.worker_open or self.active is not None or any(op['phase'] == 'started' for op in self.operations.values()):
                    if self.active is not None: self.active['status'] = 'INTERRUPTED'
                    self.active = None
                    self._ce.requires_restore = True; self.unknown_wall = True
                self._current_proof = payload['proof']
                self.last_clock = payload['monotonic']; self.worker_open = True
                self._ce.current_invocation = payload['proof']['worker_identity_sha256']
            return None, {}, apply
        if kind == 'begin':
            stage = payload['stage']
            require(self.worker_open and stage in q.STAGES and self.active is None and type(payload['attempt_id']) is str and payload['attempt_id'], 'one explicit stage/attempt')
            require(not any(a['attempt_id'] == payload['attempt_id'] for a in self.attempts), 'attempt identity reused')
            count = sum(a['stage'] == stage for a in self.attempts)
            require(count < self.spec['stage_caps'][stage]['max_attempts'], 'stage attempt ceiling')
            completed = self.external_predecessors | {a['stage'] for a in self.attempts if a['status'] == 'RETURNED'}
            require(set(q.STAGES[:q.STAGES.index(stage)]) <= completed, 'qualification predecessor stages must return')
            def apply(): self.active = dict(payload, status='STARTED'); self.attempts.append(self.active)
            return stage, {}, apply
        if kind == 'operation': return self._operation(payload)
        if kind == 'commit':
            require(self.active and self.active['stage'] in ('adam_clipping', 'checkpoint_resume'), 'CE commit is qualification Adam/checkpoint stage only')
            _, action = self._ce._prepare_commit(payload['result'])
            return self.active['stage'], {}, action
        if kind == 'restore':
            require(self.active and self.active['stage'] == 'checkpoint_resume', 'qualification restore stage only')
            _, action = self._ce._prepare_restore(payload['record'])
            return self.active['stage'], {}, action
        if kind == 'metadata':
            require(self.active and payload['class'] in ('checkpoint_json', 'checkpoint_journal', 'runner_receipt', 'qualification_json')
                    and type(payload['write_id']) is str and payload['write_id'] not in self.metadata_ids
                    and q.integer(payload['bytes_upper'], 1), 'permanent metadata identity/class')
            path = Path(payload['path']).resolve()
            require(path.is_relative_to(self.metadata_root) and path != self.metadata_root, 'qualification owned metadata root')
            return self.active['stage'], {'metadata_bytes_upper': payload['bytes_upper'], 'io_write_bytes_upper': payload['bytes_upper']}, lambda: self.metadata_ids.add(payload['write_id'])
        if kind in ('finish', 'fail'):
            require(self.active and payload['stage'] == self.active['stage'] and payload['attempt_id'] == self.active['attempt_id'], 'current stage return identity')
            if kind == 'finish':
                require(self._ce.active_attempt is None and not self._ce.requires_restore and not self._ce.restore_pending,
                        'qualification CE must finish or install proved ancestor restore before stage return')
                require(payload['raw_refs'] and not any(op['phase'] == 'started' and op['attempt_id'] == self.active['attempt_id']
                    and not op.get('lifetime_cleanup') for op in self.operations.values()), 'stage raw evidence and completed work required')
            stage = self.active['stage']
            def apply():
                self.active.update(status='RETURNED' if kind == 'finish' else 'FAILED', evidence=payload)
                self.active = None
            return stage, {}, apply
        raise ValueError('unknown qualification accounting event')
    def _operation(self, payload):
        kind, phase, oid, fields = payload['kind'], payload['phase'], payload['id'], _normalize(payload['fields'])
        require(kind in self.profiles and phase in ('started', 'returned', 'failed') and type(oid) is str and oid, 'declared qualification operation')
        if phase == 'started':
            require(self.worker_open and self.active is not None and oid not in self.operations, 'active unique physical qualification attempt')
            stage, aid = self.active['stage'], self.active['attempt_id']
            for key, expected in (('qualification_stage', stage), ('attempt_id', aid), ('worker_id', self._current_proof['worker_identity_sha256'])):
                require(key not in fields or fields[key] == expected, 'qualification callback identity changed')
            profile = self.profiles[kind]
            for key in ce.PROFILE[:-1]: require(ce._int(fields.get(key, 0)) and fields.get(key, 0) <= profile[key], 'profile below declared physical work')
            for stem in ('identity_hash_attempts', 'identity_hash_bytes'):
                if stem + '_lower' in fields: require(ce._int(fields[stem + '_lower']) and fields[stem + '_lower'] <= profile[stem + '_upper'], 'opaque hash lower exceeds qualification profile')
            rid = fields.get('prompt_id', fields.get('row_id'))
            if rid is not None:
                require(rid in self.allowed_rows, 'row outside declared qualification witness scope')
                row = self.census[rid]
                for key, expected in (('prepared_row_sha256', row['prepared_row_sha256']), ('selected_tokens', row['labeled_tokens']), ('full_sequence_tokens', row['full_sequence_tokens'])):
                    require(key not in fields or fields[key] == expected, 'qualification row content/token drift')
            is_ce = fields.get('objective') == 'SUPERVISED_EQUAL_ROW_TOKEN_MEAN_CE'
            if stage == 'checkpoint_resume' and kind in ('checkpoint_save', 'checkpoint_restore', 'fingerprint', 'checkpoint_adapter_io'):
                is_ce = True
            if is_ce:
                require(stage in ('adam_clipping', 'checkpoint_resume'), 'actual CE work outside registered qualification Adam stage')
                _, ce_action = self._ce._prepare_operation({'operation': kind, 'phase': phase, 'id': oid, 'fields': fields})
            else: ce_action = lambda: None
            require(ce._int(fields.get('reducer_work_bytes_upper', 0))
                    and fields.get('reducer_work_bytes_upper', 0) <= profile.get('reducer_work_bytes_upper', 0), 'explicit reducer profile below work')
            attempts_upper = fields.get('operation_attempts_upper', 1)
            require(ce._int(attempts_upper, 1), 'explicit callback operation reservation')
            amounts = _price(profile); amounts['operation_attempts_upper'] = attempts_upper
            if is_ce and kind == 'fingerprint' and fields.get('reason') == 'ce_pre_boundary_identity':
                amounts = _add(amounts, _price(self.profiles['gradient_cleanup']))
            elif is_ce and kind == 'gradient_cleanup': amounts = {k: 0 for k in COST_FIELDS}
            def apply():
                ce_action()
                self.operations[oid] = {'kind': kind, 'phase': phase, 'stage': stage, 'attempt_id': aid,
                    'worker_id': self._current_proof['worker_identity_sha256'], 'profile': profile, 'fields': fields,
                    'is_ce': is_ce, 'lifetime_cleanup': kind == 'cleanup' and fields.get('deferred_model_cleanup') is True,
                    'started_seq': len(self.events)}
                self.reducer_reserved_bytes += fields.get('reducer_work_bytes_upper', 0)
                self.reducer_by_stage[stage] += fields.get('reducer_work_bytes_upper', 0)
            return stage, amounts, apply
        previous = self.operations.get(oid)
        require(previous and previous['kind'] == kind and previous['phase'] == 'started'
                and previous['worker_id'] == self._current_proof['worker_identity_sha256'], 'matching current-worker physical return required')
        require(previous.get('lifetime_cleanup') or self.active and previous['attempt_id'] == self.active['attempt_id'], 'physical return outside active stage')
        for key, upper in (('bytes_read', 'read_bytes_upper'), ('known_bytes_read', 'read_bytes_upper'), ('bytes_written', 'write_bytes_upper'), ('known_bytes_written', 'write_bytes_upper'),
                           ('identity_hash_attempts_actual', 'identity_hash_attempts_upper'), ('identity_hash_bytes_actual', 'identity_hash_bytes_upper'), ('wall_s', 'wall_s_upper')):
            if key in fields: require(ce._finite(fields[key]) and fields[key] <= previous['profile'][upper], 'actual qualification work exceeds reservation')
        if previous['is_ce']:
            _, ce_action = self._ce._prepare_operation({'operation': kind, 'phase': phase, 'id': oid, 'fields': fields})
        else: ce_action = lambda: None
        def apply(): ce_action(); previous.update(phase=phase, result=fields, returned_seq=len(self.events))
        return previous['stage'], {}, apply
    def tick_wall(self):
        self._admit()
        require(self.worker_open, 'qualification worker wall interval already ended')
        now = self.clock()
        require(q.finite(now) and now >= self.last_clock, 'qualification monotonic clock regressed')
        deadline = self.spec.get('physical_wall_seconds_upper')
        require(q.finite(deadline, True) and now - self._wall_origin <= deadline, 'qualification physical deadline')
        if now > self.last_clock:
            stage = self.active['stage'] if self.active is not None else None
            self._event('wall', {'stage': stage, 'boot_identity_sha256': self._current_proof['boot_identity_sha256'],
                'worker_identity_sha256': self._current_proof['worker_identity_sha256'], 'monotonic': now,
                'elapsed_delta': now - self.last_clock})
        return self.actual_wall_seconds
    def end(self):
        self.tick_wall()
        self._event('end', {'worker_identity_sha256': self._current_proof['worker_identity_sha256'],
                           'monotonic': self.last_clock})
        self._wall_ready = False
        return self.snapshot()

    def begin_stage(self, stage, attempt_id):
        return self._event('begin', {'stage': stage, 'attempt_id': attempt_id})
    def operation(self, kind, phase, operation_id, **fields):
        return self._event('operation', {'kind': kind, 'phase': phase, 'id': operation_id, 'fields': fields})
    __call__ = operation
    def commit_boundary(self, result): return self._event('commit', {'result': result})
    def install_restore(self, record): return self._event('restore', {'record': record})
    def progress_record(self): return _clone(self._ce.progress)
    def operation_evidence(self, operation_id):
        require(operation_id in self.operations, 'known retained qualification operation required')
        op = self.operations[operation_id]
        require(op['phase'] == 'returned', 'fully returned qualification operation required')
        return _clone({key: op[key] for key in ('started_seq', 'returned_seq', 'kind', 'fields')})
    def reserve_metadata(self, kind, bytes_upper, *, write_id, path):
        return self._event('metadata', {'class': kind, 'bytes_upper': bytes_upper, 'write_id': write_id, 'path': str(path)})
    def finish_stage(self, raw_refs):
        require(self.active and type(raw_refs) is list and raw_refs, 'retained raw artifacts required')
        checked = []
        for ref in raw_refs:
            require(type(ref) is dict and set(ref) in ({'path', 'sha256', 'bytes'}, {'path', 'sha256', 'bytes', 'role', 'binding_sha256'}) and q.hex64(ref['sha256']) and q.integer(ref['bytes'], 1), 'raw artifact reference')
            path = Path(ref['path']).resolve()
            require(path.is_relative_to(self.metadata_root) and path != self.metadata_root and path.is_file()
                    and not path.is_symlink() and path.stat().st_size == ref['bytes'], 'regular owned raw artifact')
            if 'binding_sha256' in ref:
                expected = {'stage': self.active['stage'], 'request_sha256': self.request_sha256,
                    'scope_sha256': self.scope_sha256, 'static_recipe_sha256': self.context['static_recipe_sha256'],
                    'code_sha256': self.context['bindings']['source']['code_sha256'],
                    'worker_identity_sha256': self._current_proof['worker_identity_sha256'], 'attempt_id': self.active['attempt_id']}
                require(ref['binding_sha256'] == digest(expected) and type(ref['role']) is str and ref['role'], 'exact raw artifact stage/source/worker binding')
            oid = uuid.uuid4().hex
            self.operation('qualification_artifact', 'started', oid, read_bytes_upper=ref['bytes'] + 1,
                identity_hash_attempts_upper=1, identity_hash_bytes_upper=ref['bytes'] + 1)
            observed, size = hashlib.sha256(), 0
            try:
                with path.open('rb') as stream:
                    while block := stream.read(min(CHUNK, ref['bytes'] + 1 - size)):
                        size += len(block); require(size <= ref['bytes'], 'raw artifact grew'); observed.update(block)
                require(size == ref['bytes'] and observed.hexdigest() == ref['sha256'], 'raw artifact changed')
            except BaseException as error:
                try: self.operation('qualification_artifact', 'failed', oid, error_type=type(error).__name__, error=str(error)[:1000])
                except BaseException as record_error:
                    error.accounting_failure = {'type': type(record_error).__name__, 'message': str(record_error)[:1000]}
                    raise error from record_error
                raise
            self.operation('qualification_artifact', 'returned', oid, bytes_read=size,
                identity_hash_attempts_actual=1, identity_hash_bytes_actual=size)
            checked.append(_clone(ref))
        return self._event('finish', {'stage': self.active['stage'], 'attempt_id': self.active['attempt_id'], 'raw_refs': checked})
    def fail_stage(self, error):
        require(self.active is not None, 'current failed stage required')
        return self._event('fail', {'stage': self.active['stage'], 'attempt_id': self.active['attempt_id'],
            'error': {'type': type(error).__name__, 'message': str(error)[:1000]}})
    def seal(self):
        """Fixture consistency seal only; native cost completeness is unproved."""
        self._admit()
        completed = self.external_predecessors | {a['stage'] for a in self.attempts if a['status'] == 'RETURNED'}
        require(self.active is None and completed == set(q.STAGES)
                and not self._ce.active_attempt and not self.requires_restore and not self.restore_pending
                and not any(op['phase'] == 'started' and op['worker_id'] == self.proof['worker_identity_sha256']
                            for op in self.operations.values()), 'all stages and lifetime cleanup must return before seal')
        require(not self.worker_open, 'qualification wall end must be retained before seal')
        require(self.fixture, 'UNSUPPORTED_UNMEASURED: native publication/execution verifier evidence is not implemented')
        return dict(self.snapshot(), status='FIXTURE_SEALED_UNQUALIFIED', native_qualified=False)

    def snapshot(self):
        return {'schema': SCHEMA, 'fixture': self.fixture, 'native_qualified': False,
            'training_authorized': False, 'binding_sha256': self.binding_sha256,
            'ledger_head': self._head(), 'events': len(self.events), 'worker_proof': _clone(self.proof),
            'cumulative': _clone(self.cumulative), 'stages': _clone(self.by_stage),
            'reducer_reserved_bytes': self.reducer_reserved_bytes, 'reducer_by_stage': _clone(self.reducer_by_stage), 'attempts': _clone(self.attempts),
            'pending_operations': [_clone(op) for op in self.operations.values() if op['phase'] == 'started'],
            'fatal': self.fatal, 'unknown_wall': self.unknown_wall, 'progress': self.progress_record(),
            'reservation_rule': 'permanent full upper charges; failed/unknown/restarted work never refunded',
            'cost_scope': 'declared callback upper reservations and serialized journal costs; not complete native measured totals',
            'complete_wall_accounting': not self.worker_open and not self.unknown_wall,
            'actual_wall_seconds': self.actual_wall_seconds, 'last_monotonic': self.last_clock, 'worker_open': self.worker_open,
            'bootstrap_prepayment': self.bootstrap_handle.evidence,
            'native_accounting_status': 'FIXTURE_UNQUALIFIED' if self.fixture else 'UNSUPPORTED_UNMEASURED',
            'native_activation_gaps': _clone(self.native_activation_gaps)}


def create_accounting(context, permit, proof, prepared_receipt, *, bootstrap_handle=None, bootstrap_verifier=None):
    return QualificationAccounting(context, permit, proof, prepared_receipt,
        bootstrap_handle=bootstrap_handle, bootstrap_verifier=bootstrap_verifier)


def replay_state(context, header, prepared_receipt):
    """Framework-free, no-IO event shadow for an independently prepaid verifier.

    It issues no permit, performs no native construction and cannot seal native
    execution. Caller must first bind/read header and prepared bytes under its
    own source-owned verification prepayment.
    """
    state = object.__new__(QualificationAccounting)
    state.fixture, state.permit, state.context = header['fixture'], None, context
    state.spec, state.recipe = _clone(header['specification']), _clone(header['recipe'])
    state.profiles = state.spec['operation_profiles']
    rows = prepared_receipt['prepared_row_bindings']
    state.census = {row['id']: _clone(row) for row in rows}
    execution = context['bindings']['scope']['qualifier_execution']
    state.allowed_rows, state.adam_batches = set(execution['all_witness_ids']), _clone(execution['adam_batches'])
    state.request_sha256, state.scope_sha256 = header['request_sha256'], header['scope_sha256']
    state.scope_role = 'qualification_only'
    state.header, state.binding_sha256 = _clone(header), digest(header)
    state.directory = Path(state.spec['owned_directory'])
    state.metadata_root = Path(context['bindings']['scope']['native_backend']['output_directory'])
    state.external_predecessors = {'ingestion_tokenizer'} if state.spec.get('ingestion_bootstrap_ref') else set()
    state.fatal = state.unknown_wall = False
    state.active, state.operations, state.attempts, state.events, state.metadata_ids = None, {}, [], [], set()
    state.cumulative = dict.fromkeys(COST_FIELDS, 0)
    state.by_stage = {stage: dict.fromkeys(COST_FIELDS, 0) for stage in q.STAGES}
    state.reducer_reserved_bytes, state.reducer_by_stage = 0, dict.fromkeys(q.STAGES, 0)
    state.worker_open, state.last_clock, state.actual_wall_seconds = False, 0.0, 0.0
    state._current_proof, state.proof = {}, {}
    state._header_cost_prepaid, state._wall_ready = False, False
    state._make_ce_state(context, rows)
    return state
