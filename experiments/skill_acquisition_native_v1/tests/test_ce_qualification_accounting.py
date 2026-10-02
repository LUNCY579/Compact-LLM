"""Pure accounting/causal fixtures; no frameworks, authority issuance or models."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as a, ce_update
from . import test_ce_journal as cpu


def fixture(root):
    _, bindings, _ = cpu.fixture()
    rows = []
    for raw in bindings['prepared_rows']:
        row = dict(raw, **ce_update.prepared_row_binding(raw))
        row['prepared_hash_bytes_upper'] = sum(len(a.canonical(v)) for v in (raw['prompt_ids'], raw['completion_ids'],
             {key: raw[key] for key in ('id', 'prompt_ids', 'completion_ids')}))
        rows.append(row)
    receipt = {'schema': 'QUALIFICATION-INGESTION-FIXTURE-1', 'prepared_row_bindings': rows,
               'model_loads': 0, 'optimizer_steps': 0, 'native_qualified': False}
    receipt['receipt_sha256'] = a.digest(receipt)
    cap = {key: 10 ** 13 for key in a.COST_FIELDS}; cap['wall_seconds_upper'] = 10 ** 9
    profiles = {kind: {'read_bytes_upper': 10 ** 9, 'write_bytes_upper': 10 ** 9,
        'identity_hash_attempts_upper': 100000, 'identity_hash_bytes_upper': 10 ** 9, 'wall_s_upper': 1000}
        for kind in a.KINDS}
    profiles['qualification_reducer_overhead']['reducer_work_bytes_upper'] = 10 ** 7
    spec = {'schema': a.SPEC, 'owned_directory': str(root / 'ledger'),
        'ingestion_receipt_sha256': receipt['receipt_sha256'], 'operation_profiles': profiles,
        'setup_bounds': dict(profiles['input_io']), 'global_caps': cap, 'physical_wall_seconds_upper': 10000,
        'stage_caps': {stage: dict(cap, max_attempts=2) for stage in a.q.STAGES},
        'ingestion_bootstrap_ref': {'path': str(root / 'fixture-bootstrap.json'), 'sha256': 'b' * 64, 'bytes': 10}}
    recipe = bindings['recipe']
    context = {'schema': a.FIXTURE, 'mode': 'synthetic_fixture', 'phase': 'qualification',
        'approved': False, 'training_authorized': False, 'arm': 'canonical',
        'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64, 'static_recipe_sha256': a.digest(recipe),
        'bindings': {'recipe': {'recipe': recipe}, 'anchors': bindings['anchors'],
            'source': {'code_sha256': {'synthetic-fixture': 'c' * 64}}, 'runtime': bindings['runtime'],
            'scope': {'native_backend': {'output_directory': str(root)}, 'qualifier_accounting': spec,
                'qualifier_bootstrap': {'schema': 'CE8-BOOTSTRAP-SPEC-1', 'owned_directory': str(root / 'bootstrap'),
                    'caps': dict(cap), 'self_bounds': dict({key: 10 ** 9 for key in a.COST_FIELDS}, wall_seconds_upper=1000), 'physical_wall_seconds_upper': 10000},
                'qualifier_execution': {'all_witness_ids': [r['id'] for r in rows],
                    'gradient_batch_ids': [r['id'] for r in rows[:8]],
                    'adam_batches': [[r['id'] for r in rows[:8]], [r['id'] for r in rows[8:]]]}}}}
    proof = {'schema': a.PROOF, 'fresh_process': True, 'boot_verified': True,
             'boot_identity_sha256': 'd' * 64, 'worker_identity_sha256': 'e' * 64}
    return context, proof, receipt


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='qualification-accounting-pure-')
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.context, self.proof, self.receipt = fixture(self.root)
    def make(self): return a.QualificationAccounting(self.context, None, self.proof, self.receipt)
    def test_child_accounting_derives_only_exact_branch_directory(self):
        context = copy.deepcopy(self.context)
        context['bindings']['scope']['checkpoint_qualification'] = {'owned_directory': str(self.root / 'checkpoint')}
        context['checkpoint_child_declaration'] = {'branch': 'resumed'}
        common = context['bindings']['scope']['qualifier_accounting']
        frozen = copy.deepcopy(common)
        spec = dict(common, owned_directory=str(self.root / 'checkpoint' / 'children' / 'resumed' / 'ledger'))
        context['checkpoint_child_accounting'] = {'specification': spec, 'owner': {
            'accounting_spec_sha256': a.digest(spec), 'prepared_receipt_sha256': self.receipt['receipt_sha256']}}
        derived = a.accounting_specification(context)
        self.assertEqual(derived, spec)
        derived['global_caps']['io_read_bytes_upper'] = 0
        self.assertEqual(common, frozen)
        for change in ('path', 'caps', 'profile', 'census', 'deadline', 'owner', 'branch'):
            altered = copy.deepcopy(context)
            child = altered['checkpoint_child_accounting']
            if change == 'path': child['specification']['owned_directory'] = str(self.root / 'foreign')
            elif change == 'caps': child['specification']['global_caps']['io_read_bytes_upper'] = 0
            elif change == 'profile': child['specification']['operation_profiles']['qualification_lease']['wall_s_upper'] = 0
            elif change == 'census': child['specification']['ingestion_receipt_sha256'] = '0' * 64
            elif change == 'deadline': child['specification']['physical_wall_seconds_upper'] = 1
            elif change == 'owner': child['owner']['prepared_receipt_sha256'] = '0' * 64
            else: altered['checkpoint_child_declaration']['branch'] = 'foreign'
            with self.subTest(change=change), self.assertRaises(ValueError):
                a.accounting_specification(altered)
        self.assertIn('qualification_lease', a.KINDS)
    def raw(self, name='raw'):
        path = self.root / (name + '.json'); raw = b'{"fixture":true}\n'; path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    def child_payload(self, branch='resumed'):
        bindings = self.context['bindings']; batches = bindings['scope']['qualifier_execution']['adam_batches']
        rawref = {'path': str(self.root / 'parent-proof.json'), 'sha256': 'a' * 64, 'bytes': 10}
        parent = {'worker_identity_sha256': 'f' * 64, 'boot_identity_sha256': self.proof['boot_identity_sha256'],
            'pid': 11, 'creation_filetime_100ns': 100, 'request_sha256': '1' * 64,
            'scope_sha256': self.context['scope_sha256'], 'code_sha256': bindings['source']['code_sha256']}
        history = [{'boundary': 0, 'row_ids': batches[0], 'updated': True,
                    'start_policy_id': bindings['anchors']['own_start_sha256'],
                    'final_policy_id': '2' * 64, 'result_sha256': '3' * 64}]
        original = {'recipe': bindings['recipe']['recipe'], 'code': bindings['source']['code_sha256'],
            'own_start_sha256': bindings['anchors']['own_start_sha256'],
            'common_anchor_sha256': bindings['anchors']['common_anchor_sha256'],
            'qualification_batches': batches}
        manifest = {'schema': 'SKILL-ACQUISITION-NATIVE-CHECKPOINT-1', 'bindings': original,
            'bindings_sha256': a.digest(original),
            'progress': {'C': 1, 'U': 1, 'history': history, 'generation': 0, 'snapshot': '9' * 64}}
        manifest['manifest_sha256'] = a.digest(manifest)
        payload = {'schema': a.CHILD_PREFIX, 'branch': branch, 'parent_request_sha256': '1' * 64,
            'child_request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256'],
            'study_request_sha256': self.context['request_sha256'],
            'static_recipe_sha256': self.context['static_recipe_sha256'], 'code_sha256': bindings['source']['code_sha256'],
            'common_anchor_sha256': bindings['anchors']['common_anchor_sha256'],
            'own_start_sha256': bindings['anchors']['own_start_sha256'], 'qualification_batches': batches,
            'worker_identity_sha256': self.proof['worker_identity_sha256'], 'parent_worker_identity_sha256': 'f' * 64,
            'parent_proof': parent, 'parent_ledger_refs': {key: dict(rawref) for key in ('header', 'journal', 'seal')},
            'parent_stage_seal': {'schema': 'CE8-CHECKPOINT-PARENT-STAGE-SEAL-1', 'stage': 'checkpoint_resume',
                'completed': False, 'worker_open': True, 'C': 2, 'U': 2,
                'worker_identity_sha256': 'f' * 64, 'request_sha256': '1' * 64,
                'scope_sha256': self.context['scope_sha256'], 'ledger_head': '5' * 64, 'binding_sha256': '6' * 64},
            'ingestion_receipt_sha256': self.receipt['receipt_sha256'], 'inherited_U': 1 if branch == 'resumed' else 0,
            'inherited_history': history if branch == 'resumed' else [], 'prefix_manifest_ref': dict(rawref),
            'prefix_checkpoint_bindings_sha256': manifest['bindings_sha256'], 'prefix_manifest': manifest,
            'endpoint_raw_refs': {key: dict(rawref) for key in ('parameters', 'adam', 'rng')}}
        child_ref = dict(rawref, path=str(self.root / 'immutable-child-request.json'), sha256='7' * 64)
        payload['child_request_ref'], payload['child_request_sha256'] = child_ref, child_ref['sha256']
        payload['child_proof'] = dict(self.proof, request_sha256=self.context['request_sha256'],
            scope_sha256=self.context['scope_sha256'], sources={'request_ref': child_ref})
        payload['declaration_sha256'] = a.digest(a.checkpoint_child_declaration(payload))
        return payload
    def child_admission(self, payload):
        return a.CheckpointChildAdmission(self.context, None, self.proof, payload)
    def finish_prior(self, ledger, target):
        for stage in a.q.STAGES[1:a.q.STAGES.index(target)]:
            ledger.begin_stage(stage, 'fixture/' + stage)
            ledger.finish_stage([self.raw(stage)])

    def test_native_services_and_handle_truthiness_are_untouched_before_registry_refusal(self):
        class BoolTrap:
            def __bool__(self): raise AssertionError('handle truthiness before admission')
        class ServiceTrap:
            def __getattribute__(self, key): raise AssertionError('service dereferenced before admission')
        for services in (ServiceTrap(), {'bootstrap_handle': BoolTrap(), 'bootstrap_verifier': BoolTrap()}):
            context = {'schema': 'CE8-OUTER-ADMISSION-CONTEXT-1', 'native_services': services, 'bindings': ServiceTrap()}
            with self.subTest(service_type=type(services).__name__), patch.object(Path, 'open', side_effect=AssertionError('unapproved FS')):
                with self.assertRaisesRegex(ValueError, 'verifier is unregistered'):
                    a.QualificationAccounting(context, None, {}, {}, bootstrap_handle=BoolTrap())

    def test_native_child_prefix_registry_refuses_before_proof_or_model_access(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('unapproved child lineage access')
        with patch.object(Path, 'open', side_effect=AssertionError('unapproved IO')):
            with self.assertRaisesRegex(ValueError, 'verifier is unregistered'):
                a.CheckpointChildAdmission(Trap(), Trap(), Trap(), Trap(), verifier=lambda *args: True)
        self.assertFalse(a._REGISTERED_CHILD_PREFIX_VERIFIERS)

    def test_inherited_prefix_records_external_costs_and_blocks_ce_until_actual_restore(self):
        ledger = self.make(); payload = self.child_payload(); cap = self.child_admission(payload)
        before = copy.deepcopy(ledger.cumulative); wall = ledger.actual_wall_seconds
        ledger.admit_checkpoint_child_prefix(payload, admission_handle=cap)
        self.assertEqual(ledger.progress_record()['C'], 1); self.assertEqual(ledger.progress_record()['U'], 1)
        self.assertEqual(ledger.progress_record()['history'], payload['inherited_history'])
        self.assertTrue(ledger.requires_restore)
        self.assertEqual(ledger.events[-1]['kind'], 'checkpoint_child_prefix')
        self.assertTrue(all(ledger.cumulative[key] >= before[key] for key in a.COST_FIELDS))
        self.assertGreaterEqual(ledger.actual_wall_seconds, wall)
        ledger.begin_stage('checkpoint_resume', 'fixture/resumed')
        with self.assertRaisesRegex(ValueError, 'requires restore'):
            ledger.operation('fingerprint', 'started', 'premature', reason='ce_pre_boundary_identity', state_bytes=100,
                objective='SUPERVISED_EQUAL_ROW_TOKEN_MEAN_CE', boundary_id=1, generation=ledger.generation,
                symbolic_snapshot=ledger.symbolic_snapshot, start_policy_id='2' * 64,
                static_recipe_sha256=self.context['static_recipe_sha256'])
        with self.assertRaises(ValueError): ledger.commit_boundary({'updated': True, 'U_after': 2})

    def test_foreign_prefix_fields_boolean_U_and_missing_raw_refs_are_denied(self):
        good = self.child_payload()
        mutations = (lambda p: p.update(child_request_sha256='0' * 64),
            lambda p: p.update(worker_identity_sha256='0' * 64),
            lambda p: p.update(code_sha256={'foreign': '0' * 64}),
            lambda p: p.update(inherited_U=True), lambda p: p.pop('parent_ledger_refs'),
            lambda p: p.pop('prefix_manifest_ref'),
            lambda p: p['qualification_batches'][0].__setitem__(0, 'foreign-row'),
            lambda p: p['parent_proof'].update(request_sha256='0' * 64),
            lambda p: p['parent_stage_seal'].update(completed=True),
            lambda p: p['prefix_manifest']['progress'].update(U=True))
        for index, mutate in enumerate(mutations):
            payload = copy.deepcopy(good); mutate(payload)
            with self.subTest(mutation=index), self.assertRaises(ValueError): self.child_admission(payload)

    def test_actual_os_parent_proof_shape_is_normalized_without_flattened_native_claims(self):
        payload = self.child_payload()
        parent = payload['parent_proof']
        parent['os_identity'] = {key: parent.pop(key) for key in ('pid', 'creation_filetime_100ns')}
        code = parent.pop('code_sha256')
        parent['sources'] = {'source_files': {key: {'sha256': value} for key, value in code.items()}}
        payload['declaration_sha256'] = a.digest(a.checkpoint_child_declaration(payload))
        cap = self.child_admission(payload)
        self.assertFalse(cap._native)

    def test_static_declaration_excludes_future_worker_and_self_request_hash(self):
        payload = self.child_payload()
        declaration = a.checkpoint_child_declaration(payload)
        for key in ('worker_identity_sha256', 'child_request_sha256', 'child_request_ref', 'child_proof', 'declaration_sha256'):
            self.assertNotIn(key, declaration)
        self.assertEqual(declaration['study_request_sha256'], self.context['request_sha256'])
        self.assertNotEqual(payload['child_request_sha256'], payload['study_request_sha256'])
        cap = self.child_admission(payload)
        self.assertFalse(cap._native)
        changed = copy.deepcopy(payload); changed['child_proof']['sources']['request_ref']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'derived actual child'):
            self.child_admission(changed)

    def test_actual_state_fingerprint_kind_is_priced_and_unknown_kinds_remain_denied(self):
        ledger = self.make(); payload = self.child_payload('prefix')
        ledger.admit_checkpoint_child_prefix(payload, admission_handle=self.child_admission(payload))
        ledger.begin_stage('checkpoint_resume', 'fixture/controls')
        ledger.operation('qualification_state_fingerprint', 'started', 'fingerprint',
                         identity_hash_attempts_upper=2, identity_hash_bytes_upper=100)
        ledger.operation('qualification_state_fingerprint', 'returned', 'fingerprint', wall_s=0)
        self.assertEqual(ledger.operation_evidence('fingerprint')['kind'], 'qualification_state_fingerprint')
        with self.assertRaisesRegex(ValueError, 'declared qualification operation'):
            ledger.operation('unregistered_fingerprint', 'started', 'unknown')

    def test_prefix_single_use_U0_branch_and_stale_restore_are_explicit(self):
        ledger = self.make(); payload = self.child_payload('prefix'); cap = self.child_admission(payload)
        ledger.admit_checkpoint_child_prefix(payload, admission_handle=cap)
        self.assertEqual(ledger.progress_record()['C'], 0); self.assertFalse(ledger.requires_restore)
        with self.assertRaisesRegex(ValueError, 'once'):
            ledger.admit_checkpoint_child_prefix(payload, admission_handle=cap)
        ledger.begin_stage('checkpoint_resume', 'fixture/prefix')
        with self.assertRaises(ValueError): ledger.install_restore({'progress': ledger.progress_record()})

    def test_advancing_clock_end_uses_one_retained_tick_without_stale_timestamp(self):
        class AdvancingClock:
            value = 100.0
            def __call__(self):
                self.value += .001
                return self.value
        clock = AdvancingClock()
        ledger = a.QualificationAccounting(self.context, None, self.proof, self.receipt, clock=clock)
        ledger.end()
        ending = ledger.events[-1]
        self.assertEqual(ending['kind'], 'end')
        self.assertEqual(ending['payload']['monotonic'], ledger.last_clock)
        self.assertFalse(ledger.worker_open)

    def test_actual_monotonic_constructor_idle_stage_and_end_wall_are_retained(self):
        class Clock:
            value = 100.0
            def __call__(self): return self.value
        clock = Clock()
        ledger = a.QualificationAccounting(self.context, None, self.proof, self.receipt, clock=clock)
        self.assertEqual(ledger.actual_wall_seconds, 0)
        clock.value += 7.5; ledger.tick_wall()
        self.assertEqual(ledger.actual_wall_seconds, 7.5)
        ledger.begin_stage('runtime_anchor', 'fixture/anchor')
        clock.value += 2.25; ledger.operation('input_io', 'started', 'read')
        ledger.operation('input_io', 'returned', 'read', wall_s=0)
        ledger.finish_stage([self.raw()])
        clock.value += 1.0; ledger.end()
        self.assertEqual(ledger.actual_wall_seconds, 10.75)
        self.assertFalse(ledger.worker_open)
        self.assertTrue(ledger.snapshot()['complete_wall_accounting'])
        with self.assertRaisesRegex(ValueError, 'worker wall interval already ended'):
            ledger.tick_wall()
        wall = [event['payload'] for event in ledger.events if event['kind'] == 'wall']
        self.assertEqual(sum(event['elapsed_delta'] for event in wall), 10.75)
        with self.assertRaisesRegex(ValueError, 'monotonic identity'):
            ledger._prepare('end', {'worker_identity_sha256': self.proof['worker_identity_sha256'], 'monotonic': 999.0})

    def test_unissued_native_scope_refuses_before_path_or_proof_access(self):
        class Trap:
            def __getattribute__(self, name): raise AssertionError('unadmitted access')
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        with patch.object(Path, 'open', side_effect=AssertionError('unapproved IO')):
            with self.assertRaises(ValueError): a.QualificationAccounting(Trap(), None, Trap(), Trap())
        self.assertEqual(set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}, before)
        self.assertFalse(a.authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)

    def test_fixture_retains_failed_callback_reservations_and_raw_owned_readback(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'fixture/anchor')
        ledger.operation('model_load', 'started', 'load')
        reserved = dict(ledger.cumulative)
        ledger.operation('model_load', 'failed', 'load', error_type='fixture')
        self.assertTrue(all(ledger.cumulative[k] >= v for k, v in reserved.items()))
        ledger.fail_stage(RuntimeError('fixture failed')); ledger.end()
        reopened = self.make()
        self.assertTrue(all(reopened.cumulative[k] >= ledger.cumulative[k] for k in a.COST_FIELDS))
        self.assertEqual(reopened.attempts[0]['status'], 'FAILED')
        self.assertFalse(reopened.snapshot()['native_qualified'])

    def test_raw_alias_conflict_and_caps_deny_before_callback_body(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'fixture/anchor')
        size = ledger.path.stat().st_size
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            ledger.operation('qualification_artifact_readback', 'started', 'conflict', io_read_bytes_upper=2, read_bytes_upper=3)
        self.assertEqual(ledger.path.stat().st_size, size)
        ledger.profiles['input_io']['read_bytes_upper'] = 1
        with self.assertRaisesRegex(ValueError, 'profile below'):
            ledger.operation('input_io', 'started', 'too-much', read_bytes_upper=2)
        self.assertNotIn('too-much', ledger.operations)

    def test_reducer_and_io_alias_reservations_are_permanent_and_stage_bound(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'fixture/anchor')
        ledger.operation('qualification_artifact_readback', 'started', 'read', io_read_bytes_upper=10,
                         identity_hash_attempts_upper=1, identity_hash_bytes_upper=10, operation_attempts_upper=1)
        ledger.operation('qualification_artifact_readback', 'returned', 'read', bytes_read=10)
        evidence = ledger.operation_evidence('read')
        self.assertLess(evidence['started_seq'], evidence['returned_seq'])
        self.assertEqual(evidence['fields']['read_bytes_upper'], 10)
        ledger.operation('qualification_reducer_overhead', 'started', 'reduce', reducer_work_bytes_upper=512)
        ledger.operation('qualification_reducer_overhead', 'failed', 'reduce', error_type='fixture')
        self.assertEqual(ledger.snapshot()['reducer_reserved_bytes'], 512)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            ledger.operation('input_io', 'started', 'wrong-stage', qualification_stage='adam_clipping')

    def test_owned_metadata_and_bound_five_field_raw_artifact(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'fixture/anchor')
        ledger.reserve_metadata('qualification_json', 100, write_id='meta', path=str(self.root / 'artifacts' / 'row.json'))
        with self.assertRaisesRegex(ValueError, 'owned metadata'):
            ledger.reserve_metadata('qualification_json', 100, write_id='outside', path=str(self.root.parent / 'outside.json'))
        binding = {'stage': 'runtime_anchor', 'request_sha256': self.context['request_sha256'],
            'scope_sha256': self.context['scope_sha256'], 'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': self.proof['worker_identity_sha256'], 'attempt_id': 'fixture/anchor'}
        ref = dict(self.raw(), role='fixture', binding_sha256=a.digest(binding))
        ledger.finish_stage([ref]); self.assertEqual(ledger.attempts[-1]['status'], 'RETURNED')
        self.assertIsNone(ledger.active)

    def test_compact_batch_order_and_full_causal_ce_validation(self):
        ledger = self.make(); self.finish_prior(ledger, 'adam_clipping')
        ledger.begin_stage('adam_clipping', 'fixture/adam')
        self.assertEqual([r['id'] for r in ledger._ce.bindings['prepared_rows']],
                         [rid for batch in ledger.adam_batches for rid in batch])
        helper = cpu.Controls(); helper.bindings = {'recipe': ledger.recipe}
        proxy = SimpleNamespace(progress=ledger._ce.progress, generation=ledger.generation,
            symbolic_snapshot=ledger.symbolic_snapshot, static_recipe_sha256=self.context['static_recipe_sha256'],
            plan_rows=ledger.adam_batches, census=ledger._ce.census, operation=ledger.operation)
        result = helper.successful_result(proxy)
        bad = copy.deepcopy(result); bad['rows'][0]['prepared_row_sha256'] = '0' * 64
        with self.assertRaises(ValueError): ledger.commit_boundary(bad)
        ledger.commit_boundary(result)
        self.assertEqual(ledger.progress_record()['C'], 1); self.assertEqual(ledger.progress_record()['U'], 1)
        clip = ledger.operation_evidence(result['clipping_operation']['operation_id'])
        adam = ledger.operation_evidence(result['optimizer_operation']['operation_id'])
        self.assertLess(clip['returned_seq'], adam['started_seq'])
        with self.assertRaises(ValueError): ledger.commit_boundary(result)

    def test_pending_and_interrupted_work_cannot_silently_complete_or_restart(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'fixture/anchor')
        ledger.operation('model_load', 'started', 'unknown')
        with self.assertRaisesRegex(ValueError, 'completed work'):
            ledger.finish_stage([self.raw()])
        previous = dict(ledger.cumulative)
        reopened = self.make()
        self.assertTrue(reopened.unknown_wall)
        self.assertTrue(all(reopened.cumulative[k] >= previous[k] for k in a.COST_FIELDS))
        with self.assertRaisesRegex(ValueError, 'unknown-wall'):
            reopened.begin_stage('runtime_anchor', 'fixture/retry')

    def test_setup_caps_and_replay_read_envelope_refuse_before_writes_or_reads(self):
        spec = self.context['bindings']['scope']['qualifier_accounting']
        saved = copy.deepcopy(spec['global_caps'])
        spec['global_caps']['metadata_bytes_upper'] = 1
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if str(self.root / 'ledger') in str(path): raise AssertionError('inner IO before setup cap')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            with self.assertRaisesRegex(ValueError, 'global qualification ceiling'):
                self.make()
        self.assertFalse((self.root / 'ledger').exists())
        spec['global_caps'] = saved
        ledger = self.make()
        spec['setup_bounds']['read_bytes_upper'] = 1
        with patch.object(Path, 'open', guarded):
            with self.assertRaisesRegex(ValueError, 'bounded paid replay'):
                self.make()
        self.assertFalse(ledger.fatal)

    def test_replay_preserves_header_and_old_charges_and_corrupt_tail_fails_closed(self):
        ledger = self.make()
        ledger.end()
        old = copy.deepcopy(ledger.cumulative)
        header_cost = len(ledger.header_path.read_bytes())
        self.assertGreaterEqual(old['metadata_bytes_upper'], header_cost)
        reopened = self.make()
        self.assertGreater(reopened.cumulative['io_read_bytes_upper'], old['io_read_bytes_upper'])
        self.assertTrue(all(reopened.cumulative[k] >= old[k] for k in a.COST_FIELDS))
        reopened.end()
        with reopened.path.open('ab') as stream: stream.write(b'{')
        corrupt = reopened.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'partial qualification tail'):
            self.make()
        self.assertEqual(reopened.path.read_bytes(), corrupt)

    def test_final_seal_requires_lifetime_cleanup_and_remains_fixture_unqualified(self):
        ledger = self.make()
        ledger.begin_stage('runtime_anchor', 'fixture/anchor')
        ledger.operation('cleanup', 'started', 'cleanup', deferred_model_cleanup=True)
        ledger.finish_stage([self.raw('anchor')])
        for stage in a.q.STAGES[2:]:
            ledger.begin_stage(stage, 'fixture/' + stage)
            ledger.finish_stage([self.raw(stage)])
        with self.assertRaisesRegex(ValueError, 'lifetime cleanup'):
            ledger.seal()
        ledger.operation('cleanup', 'returned', 'cleanup', wall_s=.1)
        ledger.end()
        sealed = ledger.seal()
        self.assertEqual(sealed['status'], 'FIXTURE_SEALED_UNQUALIFIED')
        self.assertFalse(sealed['native_qualified'])
        self.assertTrue(sealed['complete_wall_accounting'])

    def test_lifetime_cleanup_returns_across_completed_stage_without_refund(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'fixture/anchor')
        ledger.operation('cleanup', 'started', 'cleanup', deferred_model_cleanup=True)
        ledger.finish_stage([self.raw()]); ledger.begin_stage('longest_masks', 'fixture/masks')
        ledger.operation('cleanup', 'returned', 'cleanup', wall_s=.1)
        evidence = ledger.operation_evidence('cleanup')
        self.assertEqual(evidence['kind'], 'cleanup')
        self.assertEqual(ledger.operations['cleanup']['stage'], 'runtime_anchor')


if __name__ == '__main__': unittest.main()
