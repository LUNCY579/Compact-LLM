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
    def raw(self, name='raw'):
        path = self.root / (name + '.json'); raw = b'{"fixture":true}\n'; path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
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
