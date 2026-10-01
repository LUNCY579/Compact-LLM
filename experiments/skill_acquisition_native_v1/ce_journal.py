"""Durable CE8 accounting for synthetic CPU fixtures only.

This is a mechanics prototype, never native admission or measured qualification.
Every admitted upper reservation is permanent, including failed/unknown work.
Any unfinished CE attempt needs an ancestor checkpoint restore and fresh fence.
The caller re-fingerprints the actual restored model; a snapshot is not a policy.
Profiles are explicit synthetic upper declarations, not native measurements.
The kernel's cheap recipe hash precedes its first observer callback; constructor
validation and resume verification are bounded setup scope. A future native
controller still needs outer admission/pricing before either setup operation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import os
import time
import uuid

from . import ce_plan

SCHEMA = 'CE8-SYNTHETIC-DURABLE-JOURNAL-1'
CAP_SCHEMA = 'CE8-SYNTHETIC-CAPS-1'
SCOPE = 'CE8_SYNTHETIC_CPU_DURABLE_1'
HEADER = 'CE_LEDGER_BINDINGS.json'
JOURNAL = 'CE_PHYSICAL_COSTS.jsonl'
MAX_LINE = 4 * 1024 ** 2
KINDS = frozenset(('fingerprint', 'checkpoint_configuration', 'gradient_reset',
    'loss_forward', 'backward', 'gradient_validation', 'gradient_clipping',
    'optimizer', 'gradient_cleanup', 'checkpoint_save', 'checkpoint_restore',
    'checkpoint_construct', 'checkpoint_adapter_io', 'input_io', 'receipt_write'))
PROFILE = ('read_bytes_upper', 'write_bytes_upper', 'identity_hash_attempts_upper',
           'identity_hash_bytes_upper', 'wall_s_upper')
COUNTERS = ('forward_selected_tokens', 'forward_sequence_tokens',
    'backward_selected_tokens', 'backward_sequence_tokens', 'identity_hash_attempts',
    'identity_hash_bytes', 'io_read_bytes', 'io_write_bytes', 'metadata_reserved_bytes',
    'invocations', 'resumes', 'wall_seconds')


def require(condition, message):
    if not condition: raise ValueError(message)


canonical = ce_plan.canonical
digest = ce_plan.digest


def _clone(value): return json.loads(canonical(value))
def _int(value, minimum=0): return type(value) is int and value >= minimum
def _finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0 and (not positive or value > 0)


def _profile(value):
    require(type(value) is dict and set(value) == set(PROFILE)
            and all(_int(value[k]) for k in PROFILE[:-1]) and _finite(value['wall_s_upper'], True),
            'explicit finite synthetic upper profile required')
    return value


def _json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON key'); out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


class CeJournal:
    def __init__(self, directory, plan, bindings, caps, *, synthetic_cpu,
                 resume=False, clock=time.perf_counter):
        # These checks precede path conversion, file access and helper imports.
        require(synthetic_cpu is True and type(plan) is dict
                and plan.get('mode') == 'synthetic_fixture', 'native/proposal CE journal scope denied')
        self.plan_rows = [list(rows) for rows in ce_plan.validate_plan(plan)]
        require(type(bindings) is dict and bindings.get('scope') == SCOPE,
                'explicit synthetic CE journal bindings')
        from . import ce_update
        recipe = bindings.get('recipe')
        dynamic = {'boundary_id', 'generation', 'symbolic_snapshot', 'start_policy_id', 'U_before'}
        require(type(recipe) is dict and not dynamic.intersection(recipe), 'static CE recipe only')
        ce_update.validate_recipe(dict(recipe, boundary_id=0, generation=0, U_before=0,
            symbolic_snapshot='0' * 64, start_policy_id='0' * 64))
        anchors = bindings.get('anchors')
        require(type(anchors) is dict and set(anchors) == {'own_start_sha256', 'common_anchor_sha256'}
                and all(ce_plan._hex(v) for v in anchors.values()), 'actual fixture start/reference hashes')
        source, runtime = bindings.get('source'), bindings.get('runtime')
        require(type(source) is dict and source and all(type(k) is str and k and ce_plan._hex(v)
                for k, v in source.items()), 'source digest mapping')
        require(type(runtime) is dict and runtime.get('device') == 'cpu'
                and runtime.get('dtype') == 'float32' and runtime.get('cuda_initialized') is False,
                'synthetic CPU runtime binding')
        prepared = bindings.get('prepared_rows')
        ids = [rid for boundary in self.plan_rows for rid in boundary]
        require(type(prepared) is list and [row.get('id') if type(row) is dict else None for row in prepared] == ids,
                'exact ordered prepared fixture rows')
        targets = {rid: target for entry in plan['label_binding']['boundaries'] for rid, target in entry['row_target_pairs']}
        census = {}
        for i in range(0, len(prepared), 8): ce_update.validate_rows(prepared[i:i + 8], recipe)
        for row in prepared:
            require(row.get('target_sha256') == targets[row['id']], 'prepared target differs from label plan')
            census[row['id']] = ce_update.prepared_row_binding(row)
        require(type(caps) is dict and caps.get('schema') == CAP_SCHEMA and caps.get('synthetic_cpu') is True,
                'explicit synthetic cap schema')
        starts = caps.get('operation_starts')
        require(type(starts) is dict and set(starts) == KINDS and all(_int(v, 1) for v in starts.values()),
                'finite start ceilings for every operation kind')
        require(all(_int(caps.get(k), 0 if k == 'resumes' else 1) for k in COUNTERS if k != 'wall_seconds')
                and _finite(caps.get('wall_seconds'), True) and caps['metadata_reserved_bytes'] >= 64 * 1024 ** 2,
                'explicit token/hash/IO/metadata/invocation/wall ceilings')
        profiles = bindings.get('operation_bounds')
        require(type(profiles) is dict and set(profiles) == KINDS, 'all synthetic operation profiles required')
        for value in profiles.values(): _profile(value)
        setup = _profile(bindings.get('setup_bounds'))
        self.bindings, self.plan, self.caps = _clone(bindings), _clone(plan), _clone(caps)
        self.static_recipe_sha256 = digest(self.bindings['recipe'])
        self.census, self.profiles = census, self.bindings['operation_bounds']
        self.clock = clock
        self.directory = Path(directory).resolve()
        require(not self.directory.is_symlink(), 'owned journal directory')
        self.path, self.header_path = self.directory / JOURNAL, self.directory / HEADER
        self.counters = {key: 0 for key in COUNTERS}
        self.starts = {kind: 0 for kind in KINDS}
        self.operations, self.events, self.invocations = {}, [], {}
        self.used_restores, self.metadata_ids = set(), set()
        self.active_attempt, self.current_invocation = None, None
        self.requires_restore, self.restore_pending, self.unknown_wall, self.fatal = False, False, False, False
        self._last_wall = None
        self._known_size = 0
        self.progress = {'C': 0, 'U': 0, 'history': [], 'generation': 0}
        self.progress['snapshot'] = digest(dict(self.progress, plan_sha256=plan['plan_sha256']))
        self.header = {'schema': SCHEMA, 'classification': 'synthetic CPU mechanics only; no native grant or measurement',
            'plan': self.plan, 'bindings': self.bindings, 'token_census': census, 'caps': self.caps}
        self.binding_sha256 = digest(self.header)
        raw_header = canonical(self.header) + b'\n'
        require(len(raw_header) <= MAX_LINE, 'header byte ceiling')
        read_bytes, replay_hash_bytes, replay_hashes = 0, 0, 0
        if resume:
            require(self.header_path.is_file() and not self.header_path.is_symlink()
                    and self.path.is_file() and not self.path.is_symlink(), 'existing regular ledger required')
            header_size, journal_size = self.header_path.stat().st_size, self.path.stat().st_size
            require(header_size <= MAX_LINE and journal_size <= self.caps['metadata_reserved_bytes']
                    and setup['read_bytes_upper'] >= header_size + journal_size
                    and setup['read_bytes_upper'] <= self.caps['io_read_bytes'], 'bounded resume bootstrap before reads')
            minimum_row = len(canonical({'seq': 0, 'previous': None, 'binding_sha256': '0' * 64,
                                         'kind': '', 'payload': {}, 'sha256': '0' * 64})) + 1
            require(setup['identity_hash_attempts_upper'] >= journal_size // minimum_row + 1
                    and setup['identity_hash_bytes_upper'] >= journal_size,
                    'setup hash envelope below conservative bounded replay before reads')
            with self.header_path.open('rb') as stream: observed = stream.read(MAX_LINE + 1)
            require(len(observed) == header_size <= MAX_LINE, 'header changed/oversized during bounded read')
            read_bytes += len(observed)
            require(observed == raw_header, 'resume plan/data/recipe/runtime/source/caps binding drift')
            self._charge({'io_write_bytes': len(observed), 'metadata_reserved_bytes': len(observed),
                          'identity_hash_attempts': 1, 'identity_hash_bytes': len(canonical(self.header))})
            with self.path.open('rb') as stream:
                while raw := stream.readline(MAX_LINE + 1):
                    require(read_bytes + len(raw) <= header_size + journal_size, 'journal changed beyond bounded bootstrap')
                    require(len(raw) <= MAX_LINE and raw.endswith(b'\n'), 'partial/corrupt ledger tail; no truncation')
                    entry = _json(raw)
                    require(canonical(entry) + b'\n' == raw, 'noncanonical ledger row')
                    body = {k: v for k, v in entry.items() if k != 'sha256'}
                    require(body.get('seq') == len(self.events) and body.get('previous') == self._head()
                            and body.get('binding_sha256') == self.binding_sha256 and digest(body) == entry.get('sha256'),
                            'ledger hash/order/source drift')
                    changes, action = self._prepare(body['kind'], body['payload'])
                    self._charge(self._combined(changes, len(raw), len(canonical(body))))
                    action(); self.events.append(entry)
                    self._known_size += len(raw)
                    read_bytes += len(raw); replay_hash_bytes += len(canonical(body)); replay_hashes += 1
            self.requires_restore = True
        else:
            require(not self.header_path.exists() and not self.path.exists(), 'create-only CE journal')
            require(setup['write_bytes_upper'] >= len(raw_header), 'setup write reservation below header bytes')
            setup_body, setup_raw = self._encoded('setup', {'resume': False})
            startup = self._combined(self._price(setup), len(setup_raw), len(canonical(setup_body)))
            for key, value in {'io_write_bytes': len(raw_header), 'metadata_reserved_bytes': len(raw_header),
                          'identity_hash_attempts': 1, 'identity_hash_bytes': len(canonical(self.header))}.items(): startup[key] = startup.get(key, 0) + value
            self._charge(startup)  # combined admission precedes directory/header/journal creation
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.header_path.open('xb') as stream:
                require(stream.write(raw_header) == len(raw_header), 'short header write'); stream.flush(); os.fsync(stream.fileno())
            with self.path.open('xb') as stream:
                require(stream.write(setup_raw) == len(setup_raw), 'short setup write'); stream.flush(); os.fsync(stream.fileno())
            self.events.append(_json(setup_raw))
            self._known_size = len(setup_raw)
        require(setup['read_bytes_upper'] >= read_bytes and setup['identity_hash_attempts_upper'] >= replay_hashes
                and setup['identity_hash_bytes_upper'] >= replay_hash_bytes, 'setup profile below observed replay work')
        if resume: self._event('setup', {'resume': True})

    @property
    def generation(self): return self.progress['generation']
    @property
    def symbolic_snapshot(self): return self.progress['snapshot']
    def _head(self): return self.events[-1]['sha256'] if self.events else None
    def _price(self, profile):
        return dict(io_read_bytes=profile['read_bytes_upper'], io_write_bytes=profile['write_bytes_upper'],
            identity_hash_attempts=profile['identity_hash_attempts_upper'], identity_hash_bytes=profile['identity_hash_bytes_upper'],
            wall_seconds=profile['wall_s_upper'])
    def _combined(self, changes, raw_bytes, hash_bytes):
        changes = dict(changes)
        for key, amount in (('io_write_bytes', raw_bytes), ('metadata_reserved_bytes', raw_bytes),
                            ('identity_hash_attempts', 1), ('identity_hash_bytes', hash_bytes)):
            changes[key] = changes.get(key, 0) + amount
        return changes
    def _charge(self, changes):
        require(all(k in self.counters or k.startswith('start:') for k in changes), 'unknown cost counter')
        updated, starts = dict(self.counters), dict(self.starts)
        for key, amount in changes.items():
            require(_finite(amount) and (key == 'wall_seconds' or type(amount) is int), 'nonnegative finite cost')
            if key.startswith('start:'):
                kind = key[6:]; require(kind in KINDS, 'unknown operation kind'); starts[kind] += amount
                require(starts[kind] <= self.caps['operation_starts'][kind], 'operation start ceiling: ' + kind)
            else:
                updated[key] += amount; require(updated[key] <= self.caps[key], 'cumulative ceiling: ' + key)
        self.counters, self.starts = updated, starts
    def _event(self, kind, payload):
        require(not self.fatal, 'fatal ledger; no further work')
        body, raw = self._encoded(kind, payload)
        require(len(raw) <= MAX_LINE, 'ledger event byte ceiling')
        changes, action = self._prepare(kind, body['payload'])
        self._charge(self._combined(changes, len(raw), len(canonical(body))))
        try:
            require(self.path.stat().st_size == self._known_size, 'journal changed behind single writer')
            with self.path.open('ab') as stream:
                require(stream.write(raw) == len(raw), 'short journal write'); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            self.fatal = True; self.requires_restore = True
            raise
        action(); self.events.append(_json(raw)); self._known_size += len(raw)
    def _encoded(self, kind, payload):
        body = {'seq': len(self.events), 'previous': self._head(), 'binding_sha256': self.binding_sha256,
                'kind': kind, 'payload': _clone(payload)}
        return body, canonical(dict(body, sha256=digest(body))) + b'\n'
    def _work(self):
        require(self.current_invocation is not None and not self.unknown_wall and not self.fatal,
                'active resolved invocation required')
    def _expected_policy(self, C=None):
        C = self.progress['C'] if C is None else C
        return self.progress['history'][C - 1]['final_policy_id'] if C else self.bindings['anchors']['own_start_sha256']
    def _prepare(self, kind, payload):
        if kind == 'setup':
            return self._price(self.bindings['setup_bounds']), lambda: None
        if kind == 'invocation_start':
            require(self.current_invocation is None and type(payload['id']) is str and payload['id'] not in self.invocations,
                    'one unique active invocation')
            require(type(payload['resume']) is bool and payload['resume'] == bool(self.invocations), 'initial/resume invocation mode')
            require(_finite(payload['perf_started']) and (payload['boot_id'] is None or type(payload['boot_id']) is str), 'clock/boot identity')
            def apply():
                self.invocations[payload['id']] = dict(payload, ended=False, last_perf=payload['perf_started'])
                self.current_invocation = payload['id']; self._last_wall = payload['perf_started']
                if payload['resume']: self.requires_restore = True
            return {'invocations': 1, 'resumes': int(payload['resume'])}, apply
        if kind == 'wall':
            invocation = self.invocations.get(payload['invocation'])
            require(invocation and not invocation['ended'] and _finite(payload['perf'])
                    and payload['perf'] >= invocation['last_perf'] and payload['seconds'] == payload['perf'] - invocation['last_perf'], 'exact monotonic wall delta')
            def apply(): invocation['last_perf'] = payload['perf']; self._last_wall = payload['perf']
            return {'wall_seconds': payload['seconds']}, apply
        if kind == 'unknown_wall':
            return {}, lambda: setattr(self, 'unknown_wall', True)
        if kind == 'invocation_end':
            invocation = self.invocations.get(payload['id'])
            require(invocation and not invocation['ended'], 'invocation end identity')
            if payload['outcome'] == 'COMPLETE':
                require(self.progress['C'] == len(self.plan_rows) and self.active_attempt is None
                        and not any((self.requires_restore, self.restore_pending, self.unknown_wall, self.fatal))
                        and not any(op['phase'] == 'started' and op['invocation'] == self.current_invocation
                                    and op['generation'] == self.generation and op['snapshot'] == self.symbolic_snapshot
                                    and op['attempt'] is not None for op in self.operations.values()), 'complete consistent CE plan/fence required')
            def apply():
                invocation['ended'] = True; invocation['outcome'] = payload['outcome']
                if self.current_invocation == payload['id']: self.current_invocation = None; self._last_wall = None
                if self.active_attempt is not None: self.requires_restore = True
            return {}, apply
        if kind == 'metadata':
            self._work()
            require(type(payload['write_id']) is str and payload['write_id'] and payload['write_id'] not in self.metadata_ids
                    and payload['class'] in {'checkpoint_json', 'checkpoint_journal', 'runner_receipt'}
                    and _int(payload['bytes_upper'], 1), 'permanent unique metadata reservation')
            target = Path(payload['path']).resolve()
            require(target.is_relative_to(self.directory) and target != self.directory, 'owned metadata path')
            return {'metadata_reserved_bytes': payload['bytes_upper'], 'io_write_bytes': payload['bytes_upper']}, lambda: self.metadata_ids.add(payload['write_id'])
        if kind == 'operation': return self._prepare_operation(payload)
        if kind == 'commit': return self._prepare_commit(payload['result'])
        if kind == 'restore': return self._prepare_restore(payload['record'])
        raise ValueError('unknown journal event')

    def _prepare_operation(self, payload):
        kind, phase, oid, fields = payload['operation'], payload['phase'], payload['id'], payload['fields']
        require(kind in KINDS and phase in ('started', 'returned', 'failed') and type(oid) is str and oid,
                'known physical operation/phase/identity')
        if phase != 'started':
            previous = self.operations.get(oid)
            require(previous and previous['kind'] == kind and previous['phase'] == 'started'
                    and previous['invocation'] == self.current_invocation, 'matching current-worker returned operation')
            if kind != 'checkpoint_restore': require(previous['generation'] == self.generation and previous['snapshot'] == self.symbolic_snapshot, 'stale operation return after restore')
            for actual, upper in (('bytes_read', 'read_bytes_upper'), ('known_bytes_read', 'read_bytes_upper'),
                ('bytes_written', 'write_bytes_upper'), ('known_bytes_written', 'write_bytes_upper'),
                ('identity_hash_attempts_actual', 'identity_hash_attempts_upper'), ('identity_hash_bytes_actual', 'identity_hash_bytes_upper')):
                if actual in fields: require(_int(fields[actual]) and fields[actual] <= previous['profile'][upper], 'actual work exceeds reservation')
            for key in ('wall_s', 'wall_seconds'):
                if key in fields: require(_finite(fields[key]) and fields[key] <= previous['profile']['wall_s_upper'], 'actual wall exceeds reservation')
            if kind == 'checkpoint_restore' and phase == 'returned':
                require(_int(fields.get('generation')) and fields['generation'] > self.generation
                        and ce_plan._hex(fields.get('symbolic_snapshot')) and ce_plan._hex(fields.get('manifest_sha256')), 'fresh returned restore fence')
            def apply():
                previous.update(phase=phase, result=fields)
                previous['return_index'] = len(self.events)
                if phase == 'failed' and previous['attempt'] is not None: self.requires_restore = True; self.active_attempt['failed'] = True
            return {}, apply
        self._work()
        require(not self.restore_pending or kind in ('checkpoint_construct', 'checkpoint_restore', 'fingerprint', 'checkpoint_adapter_io', 'input_io'), 'restore fence pending')
        require(oid not in self.operations, 'duplicate operation identity')
        pre = kind == 'fingerprint' and fields.get('reason') == 'ce_pre_boundary_identity'
        scoped = 'boundary_id' in fields and kind not in ('checkpoint_save', 'checkpoint_restore')
        if pre:
            require(not self.requires_restore and self.active_attempt is None and self.progress['C'] < len(self.plan_rows), 'previous CE attempt requires restore or is active')
        elif scoped:
            require(self.active_attempt is not None and (not self.active_attempt['failed'] or kind == 'gradient_cleanup'), 'active CE attempt required')
        elif kind == 'checkpoint_save': require(self.active_attempt is None and not self.requires_restore, 'cannot save unfinished CE live state')
        elif self.requires_restore: require(kind in ('checkpoint_restore', 'fingerprint', 'checkpoint_adapter_io', 'checkpoint_construct', 'input_io'), 'restore required before work')
        if scoped:
            require(type(fields.get('boundary_id')) is int and type(fields.get('generation')) is int
                    and fields.get('boundary_id') == self.progress['C'] and fields.get('generation') == self.generation
                    and fields.get('symbolic_snapshot') == self.symbolic_snapshot
                    and fields.get('start_policy_id') == self._expected_policy()
                    and fields.get('static_recipe_sha256') == self.static_recipe_sha256, 'current CE boundary/policy/fence/recipe')
        profile = self.profiles[kind]
        for key in PROFILE[:-1]: require(_int(fields.get(key, 0)) and fields.get(key, 0) <= profile[key], 'profile below declared work: ' + key)
        if kind == 'fingerprint':
            require(_int(fields.get('state_bytes'), 1)
                    and profile['identity_hash_attempts_upper'] >= 1 + fields.get('identity_hash_attempts_upper', 0)
                    and profile['identity_hash_bytes_upper'] >= fields['state_bytes'] + fields.get('identity_hash_bytes_upper', 0), 'opaque fingerprint profile below raw/framing lower')
            if pre:
                batch = self.bindings['prepared_rows'][self.progress['C'] * 8:(self.progress['C'] + 1) * 8]
                binding_bytes = sum(sum(len(canonical(v)) for v in (r['prompt_ids'], r['completion_ids'],
                    {k: r[k] for k in ('id', 'prompt_ids', 'completion_ids')})) for r in batch)
                require(profile['identity_hash_attempts_upper'] >= 26
                        and profile['identity_hash_bytes_upper'] >= fields['state_bytes'] + binding_bytes + len(canonical(self.bindings['recipe'])),
                        'pre-CE profile must reserve recipe and eight row-binding hash triples before computation')
        for stem in ('identity_hash_attempts', 'identity_hash_bytes'):
            if stem + '_lower' in fields: require(_int(fields[stem + '_lower']) and fields[stem + '_lower'] <= profile[stem + '_upper'], 'opaque hash lower exceeds profile')
        changes = dict(self._price(profile), **{'start:' + kind: 1})
        attempt = oid if pre else self.active_attempt['id'] if scoped else None
        if kind in ('loss_forward', 'backward'):
            require(scoped and fields.get('prompt_id') in self.plan_rows[self.progress['C']], 'planned CE row required')
            row = self.census[fields['prompt_id']]
            expected = {'selected_tokens': row['labeled_tokens'], 'full_sequence_tokens': row['full_sequence_tokens'],
                'prompt_ids_sha256': row['prompt_ids_sha256'], 'completion_ids_sha256': row['completion_ids_sha256'], 'prepared_row_sha256': row['prepared_row_sha256']}
            require(type(fields.get('selected_tokens')) is int and type(fields.get('full_sequence_tokens')) is int
                    and all(fields.get(k) == v for k, v in expected.items()), 'frozen row content/token census drift')
            cursor = self.active_attempt['cursor']; expected_kind = 'loss_forward' if cursor % 2 == 0 else 'backward'
            require(kind == expected_kind and cursor < 16 and fields['prompt_id'] == self.plan_rows[self.progress['C']][cursor // 2], 'ordered eight forward/backward rows')
            if cursor: require(self.operations[self.active_attempt['row_ops'][-1]]['phase'] == 'returned', 'prior row work must return before next start')
            if kind == 'loss_forward':
                raw_row = self.bindings['prepared_rows'][self.progress['C'] * 8 + cursor // 2]
                lower = sum(len(canonical(v)) for v in (raw_row['prompt_ids'], raw_row['completion_ids'],
                    {k: raw_row[k] for k in ('id', 'prompt_ids', 'completion_ids')}))
                require(profile['identity_hash_attempts_upper'] >= 3 and profile['identity_hash_bytes_upper'] >= lower, 'prepared-row hash profile lower')
            prefix = 'forward' if kind == 'loss_forward' else 'backward'
            changes[prefix + '_selected_tokens'] = row['labeled_tokens']; changes[prefix + '_sequence_tokens'] = row['full_sequence_tokens']
        if kind == 'optimizer':
            require(scoped and self.active_attempt['cursor'] == 16 and not self.active_attempt['optimizer']
                    and self.operations[self.active_attempt['row_ops'][-1]]['phase'] == 'returned'
                    and any(op['attempt'] == attempt and op['kind'] == 'gradient_validation' and op['phase'] == 'returned' for op in self.operations.values())
                    and any(op['attempt'] == attempt and op['kind'] == 'gradient_clipping' and op['phase'] == 'returned' for op in self.operations.values())
                    and type(fields.get('U_before')) is int and type(fields.get('expected_U_after')) is int
                    and fields.get('U_before') == self.progress['U'] and fields.get('expected_U_after') == self.progress['U'] + 1,
                    'one Adam attempt after complete eight rows')
        if kind in ('gradient_validation', 'gradient_clipping'):
            stage_ops = [op for op in self.operations.values() if op['attempt'] == attempt]
            require(scoped and self.active_attempt['cursor'] == 16
                    and self.operations[self.active_attempt['row_ops'][-1]]['phase'] == 'returned'
                    and not any(op['kind'] == kind for op in stage_ops), 'one gradient stage after all eight backwards')
            if kind == 'gradient_clipping':
                require(any(op['kind'] == 'gradient_validation' and op['phase'] == 'returned' for op in stage_ops),
                        'returned raw validation required before clipping')
        if kind == 'gradient_cleanup':
            require(scoped and not self.active_attempt['cleanup_started'], 'one prepaid cleanup per CE attempt')
            changes = {}
        if pre:
            for key, amount in self._price(self.profiles['gradient_cleanup']).items(): changes[key] = changes.get(key, 0) + amount
            changes['start:gradient_cleanup'] = 1
        def apply():
            if pre: self.active_attempt = {'id': oid, 'cursor': 0, 'row_ops': [], 'optimizer': None, 'cleanup_started': False, 'failed': False}
            self.operations[oid] = {'kind': kind, 'phase': phase, 'fields': fields, 'profile': profile,
                'attempt': attempt, 'invocation': self.current_invocation, 'generation': self.generation, 'snapshot': self.symbolic_snapshot}
            self.operations[oid]['start_index'] = len(self.events)
            if kind in ('loss_forward', 'backward'): self.active_attempt['cursor'] += 1; self.active_attempt['row_ops'].append(oid)
            if kind == 'optimizer': self.active_attempt['optimizer'] = oid; self.requires_restore = True
            if kind == 'gradient_cleanup': self.active_attempt['cleanup_started'] = True
            if kind == 'checkpoint_restore': self.restore_pending = True; self.requires_restore = True
        return changes, apply

    def _paired(self, record, kind, attempt):
        op = self.operations.get(record.get('operation_id')) or {}
        require(op.get('kind') == kind and op.get('phase') == 'returned' and op.get('attempt') == attempt
                and op.get('generation') == self.generation and op.get('snapshot') == self.symbolic_snapshot,
                'fully returned current CE proof: ' + kind)
        require(all(op['result'].get(k) == v for k, v in record.items() if k not in ('kind', 'operation_id')), 'returned CE proof fields differ')
        return op
    def _prepare_commit(self, result):
        active = self.active_attempt
        require(active and not active['failed'] and active['cursor'] == 16 and not self.restore_pending, 'one complete nonfailed CE attempt')
        C, U, aid = self.progress['C'], self.progress['U'], active['id']
        require(result.get('schema') == 'SKILL-ACQUISITION-CE-BOUNDARY-RESULT-1' and result.get('synthetic_cpu') is True
                and result.get('objective') == 'SUPERVISED_EQUAL_ROW_TOKEN_MEAN_CE'
                and result.get('loss_reduction') == self.bindings['recipe']['loss_reduction'] and result.get('accumulation_rows') == 8
                and all(type(result.get(k)) is int for k in ('boundary_id', 'U_before', 'U_after', 'generation'))
                and result.get('boundary_id') == C and result.get('U_before') == U and result.get('U_after') == U + 1
                and result.get('generation') == self.generation and result.get('symbolic_snapshot') == self.symbolic_snapshot
                and result.get('static_recipe_sha256') == self.static_recipe_sha256
                and result.get('start_policy_id') == self._expected_policy() and result.get('updated') is True
                and result.get('requires_restore') is False and [r.get('id') for r in result.get('rows', [])] == self.plan_rows[C]
                and result.get('optimizer_recipe') == self.bindings['recipe']['optimizer']
                and result.get('gradient_clipping_recipe') == self.bindings['recipe']['gradient_clipping'], 'exact returned CE result scope/counts/recipe')
        scoped_ops = [op for op in self.operations.values() if op['attempt'] == aid]
        require(all(op['phase'] == 'returned' for op in scoped_ops), 'all current CE spans must return before commit')
        expected_counts = {'fingerprint': 2, 'checkpoint_configuration': 1, 'gradient_reset': 1,
            'loss_forward': 8, 'backward': 8, 'gradient_validation': 1, 'gradient_clipping': 1, 'optimizer': 1, 'gradient_cleanup': 1}
        require({k: sum(op['kind'] == k for op in scoped_ops) for k in {op['kind'] for op in scoped_ops}} == expected_counts,
                'exact CE stage population')
        pre = self._paired(result['identity_operation'], 'fingerprint', aid); post = self._paired(result['final_identity_operation'], 'fingerprint', aid)
        require(type(pre['result']['U']) is int and type(post['result']['U']) is int
                and pre['result']['U'] == U and post['result']['U'] == U + 1 and pre['result']['policy_sha256'] == result['start_policy_id']
                and post['result']['policy_sha256'] == result['final_policy_id'] and ce_plan._hex(result['final_policy_id'])
                and pre['result']['frozen_state_sha256'] == post['result']['frozen_state_sha256'] == result['frozen_state_sha256'], 'actual pre/post policy/U/frozen proof')
        order = []
        for row in result['rows']:
            census = self.census[row['id']]
            require(all(type(row.get(k)) is int for k in ('prompt_tokens', 'labeled_tokens', 'full_sequence_tokens',
                                                         'eos_positions_selected', 'prompt_positions_selected', 'pad_positions_selected'))
                    and all(row.get(k) == v for k, v in census.items()) and row.get('eos_positions_selected') == 1
                    and row.get('prompt_positions_selected') == row.get('pad_positions_selected') == 0
                    and row.get('selected_predictor_positions') == list(range(census['prompt_tokens'] - 1, census['full_sequence_tokens'] - 1)), 'returned row content/mask proof')
            require(row.get('target_positions') == list(range(census['prompt_tokens'], census['full_sequence_tokens'])), 'returned target/EOS span')
            for field, kind in (('forward_operation', 'loss_forward'), ('backward_operation', 'backward')):
                op = self._paired(row[field], kind, aid); require(op['fields']['prompt_id'] == row['id'], 'row proof identity'); order.append(row[field]['operation_id'])
        require(order == active['row_ops'], 'returned forward/backward order')
        for field, kind in (('gradient_validation_operation', 'gradient_validation'), ('clipping_operation', 'gradient_clipping'), ('optimizer_operation', 'optimizer')):
            self._paired(result[field], kind, aid)
        require(result['optimizer_operation']['operation_id'] == active['optimizer'], 'actual Adam attempt binding')
        cleanup = [op for op in self.operations.values() if op['attempt'] == aid and op['kind'] == 'gradient_cleanup']
        require(len(cleanup) == 1 and cleanup[0]['phase'] == 'returned' and cleanup[0]['result'].get('optimizer_attempted') is True,
                'successful observed cleanup before commit')
        causal = [self.operations[order[-1]], self.operations[result['gradient_validation_operation']['operation_id']],
                  self.operations[result['clipping_operation']['operation_id']], self.operations[result['optimizer_operation']['operation_id']], post, cleanup[0]]
        require(all(a['return_index'] < b['start_index'] for a, b in zip(causal, causal[1:])), 'CE backward/validation/clip/Adam/post/cleanup causal order')
        validation = self.operations[result['gradient_validation_operation']['operation_id']]['result']
        clipping = self.operations[result['clipping_operation']['operation_id']]
        require(validation.get('finite_confined') is True and _finite(result.get('raw_gradient_norm_float64'))
                and result['raw_gradient_norm_float64'] == validation.get('raw_gradient_norm_float64')
                and _finite(result.get('clipped_gradient_norm_float64'))
                and result['clipped_gradient_norm_float64'] == clipping['result'].get('clipped_gradient_norm_float64')
                and result['clipped_gradient_norm_float64'] <= 1 + 32 * 2 ** -23
                and all(clipping['fields'].get(k) == v for k, v in self.bindings['recipe']['gradient_clipping'].items()), 'actual finite/confined/clipping proof')
        entry = {'boundary': C, 'row_ids': list(self.plan_rows[C]), 'updated': True, 'start_policy_id': result['start_policy_id'],
                 'final_policy_id': result['final_policy_id'], 'result_sha256': digest(result)}
        def apply():
            self.progress['history'].append(entry); self.progress['C'] += 1; self.progress['U'] += 1
            self.progress['snapshot'] = digest(dict(self.progress, plan_sha256=self.plan['plan_sha256']))
            self.active_attempt = None; self.requires_restore = False
        return {}, apply
    def _prepare_restore(self, record):
        progress = record['progress']; C = progress['C']
        from .checkpoint import validate_progress
        validate_progress(progress, self.plan_rows)
        require(C == progress['U'] and all(entry['updated'] is True for entry in progress['history']), 'strict restored CE8 progress')
        require(_int(C) and C == progress['U'] and C <= self.progress['C'] and progress['history'] == self.progress['history'][:C]
                and _int(progress['generation']) and progress['generation'] > self.generation
                and all(type(record[k]) is int and record[k] == progress[k] for k in ('C', 'U', 'generation'))
                and progress['snapshot'] == progress.get('symbolic_snapshot') == record['snapshot'] == record['symbolic_snapshot']
                and record['history'] == progress['history'] and record['install_fence_required'] is True, 'exact ancestor nested restore fence')
        matches = [(oid, op) for oid, op in self.operations.items() if op['kind'] == 'checkpoint_restore' and op['phase'] == 'returned'
            and op['invocation'] == self.current_invocation
            and op['result'].get('manifest_sha256') == record['manifest_sha256'] and oid not in self.used_restores
            and all(op['result'].get(k) == record[k] for k in ('C', 'U', 'generation', 'symbolic_snapshot'))]
        require(len(matches) == 1 and self.restore_pending, 'one unused actual returned checkpoint restore')
        oid, _ = matches[0]
        def apply():
            self.progress = _clone(progress); self.used_restores.add(oid); self.active_attempt = None
            self.requires_restore = False; self.restore_pending = False
        return {}, apply

    def tick_wall(self):
        if self.current_invocation is None: return
        now = self.clock(); previous = self.invocations[self.current_invocation]['last_perf']
        require(_finite(now) and now >= previous, 'monotonic same-boot clock')
        self._event('wall', {'invocation': self.current_invocation, 'perf': now, 'seconds': now - previous})
    def start_invocation(self, *, resume=False, boot_id=None, reason='synthetic fixture'):
        now = self.clock()
        for oid, invocation in list(self.invocations.items()):
            if not invocation['ended']:
                if boot_id is None or invocation['boot_id'] != boot_id or now < invocation['last_perf']:
                    self._event('unknown_wall', {})
                else:
                    self._event('wall', {'invocation': oid, 'perf': now, 'seconds': now - invocation['last_perf']})
                self._event('invocation_end', {'id': oid, 'outcome': 'INTERRUPTED'})
        self._event('invocation_start', {'id': uuid.uuid4().hex, 'resume': resume, 'boot_id': boot_id, 'perf_started': now, 'reason': reason})
        return self.current_invocation
    def end_invocation(self, outcome):
        self.tick_wall(); self._event('invocation_end', {'id': self.current_invocation, 'outcome': outcome})
    def operation(self, kind, phase, operation_id, **fields):
        try:
            self.tick_wall()
            self._event('operation', {'operation': kind, 'phase': phase, 'id': operation_id, 'fields': fields})
        except BaseException:
            if self.active_attempt is not None: self.active_attempt['failed'] = True; self.requires_restore = True
            raise
    __call__ = operation
    def reserve_metadata(self, kind, bytes_upper, *, write_id, path):
        self.tick_wall(); self._event('metadata', {'class': kind, 'bytes_upper': bytes_upper, 'write_id': write_id, 'path': str(Path(path).resolve())})
    def commit_boundary(self, result):
        self.tick_wall(); self._event('commit', {'result': result})
    def install_restore(self, restored):
        self.tick_wall(); self._event('restore', {'record': restored})
    def progress_record(self): return _clone(self.progress)
    def snapshot(self):
        return {'schema': SCHEMA, 'synthetic_cpu': True, 'native_qualified': False, 'binding_sha256': self.binding_sha256,
            'events': len(self.events), 'head': self._head(), 'progress': self.progress_record(),
            'cumulative': dict(self.counters), 'operation_starts': dict(self.starts), 'requires_restore': self.requires_restore,
            'active_attempt': self.active_attempt['id'] if self.active_attempt else None,
            'pending_operations': sum(op['phase'] == 'started' for op in self.operations.values()),
            'unknown_wall': self.unknown_wall, 'fatal': self.fatal,
            'cost_scope': 'declared synthetic upper reservations plus exact serialized journal costs; no native measurement'}
