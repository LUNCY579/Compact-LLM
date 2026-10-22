"""Prospective known-size Q accounting; real file IO, no models or grants."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as a
from experiments.skill_acquisition_native_v1 import ce_qualification_artifacts as raw
from .test_ce_qualification_accounting import fixture
from . import test_ce_qualification_artifacts as raw_cpu


MIB, GIB = 1024 ** 2, 1024 ** 3


class ExplicitReservations(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='qualification-explicit-pure-')
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.context, self.proof, self.receipt = fixture(self.root)
        self.spec = self.context['bindings']['scope']['qualifier_accounting']
        self.spec.update(operation_profile_rule=a.EXPLICIT_RESERVATION_RULE,
                         explicit_reservation_kinds=list(a.EXPLICIT_RESERVATION_KINDS))
        profiles = self.spec['operation_profiles']
        profiles['input_io'] = dict(read_bytes_upper=GIB, write_bytes_upper=0,
            identity_hash_attempts_upper=64, identity_hash_bytes_upper=GIB, wall_s_upper=5)
        profiles['qualification_artifact_readback'] = dict(read_bytes_upper=3 * GIB, write_bytes_upper=0,
            identity_hash_attempts_upper=8, identity_hash_bytes_upper=3 * GIB, wall_s_upper=60)
        profiles['qualification_reducer_overhead'] = dict(read_bytes_upper=0, write_bytes_upper=0,
            identity_hash_attempts_upper=10000, identity_hash_bytes_upper=128 * GIB, wall_s_upper=60,
            reducer_work_bytes_upper=512 * GIB)
        profiles['qualification_artifact'] = dict(read_bytes_upper=8 * MIB + 2, write_bytes_upper=4 * MIB,
            identity_hash_attempts_upper=16, identity_hash_bytes_upper=64 * MIB, wall_s_upper=5,
            reducer_work_bytes_upper=4 * MIB)

    def make(self):
        return a.QualificationAccounting(self.context, None, self.proof, self.receipt, clock=lambda: 100.0)

    def small_ref(self, name='raw'):
        path = self.root / (name + '.json'); blob = b'{"fixture":true}\n'; path.write_bytes(blob)
        return dict(path=str(path), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))

    def start(self, ledger): ledger.begin_stage('runtime_anchor', 'fixture/anchor')

    def fields(self, size=17):
        return dict(read_bytes_upper=size, write_bytes_upper=0,
            identity_hash_attempts_upper=1, identity_hash_bytes_upper=size, operation_attempts_upper=1)

    def test_small_row_full_declared_upper_is_permanent_for_return_failure_and_pending(self):
        ledger = self.make(); self.start(ledger)
        before = copy.deepcopy(ledger.by_stage['runtime_anchor'])
        rid = self.receipt['prepared_row_bindings'][0]['id']
        for oid in ('returned', 'failed', 'pending'):
            ledger.operation('input_io', 'started', oid, row_id=rid, **self.fields())
        self.assertEqual(ledger.by_stage['runtime_anchor']['io_read_bytes_upper'] - before['io_read_bytes_upper'], 51)
        self.assertEqual(ledger.by_stage['runtime_anchor']['wall_seconds_upper'] - before['wall_seconds_upper'], 15)
        charged = copy.deepcopy(ledger.by_stage['runtime_anchor'])
        ledger.operation('input_io', 'returned', 'returned', bytes_read=3,
                         identity_hash_attempts_actual=1, identity_hash_bytes_actual=3, wall_s=1)
        ledger.operation('input_io', 'failed', 'failed', known_bytes_read=2, wall_s=2)
        self.assertEqual(ledger.by_stage['runtime_anchor']['io_read_bytes_upper'], charged['io_read_bytes_upper'])
        self.assertEqual(ledger.operations['pending']['profile']['read_bytes_upper'], 17)
        self.assertEqual(ledger.operations['returned']['profile']['wall_s_upper'], 5)
        started = ledger.events[ledger.operations['returned']['started_seq']]['payload']
        self.assertEqual(started['reservation_profile'], ledger.operations['returned']['profile'])
        with self.assertRaisesRegex(ValueError, 'completed work'): ledger.finish_stage([self.small_ref()])
        shadow = a.replay_state(self.context, ledger.header, self.receipt)
        for row in ledger.events:
            shadow._apply(row['kind'], row['payload'], replay=True); shadow.events.append(copy.deepcopy(row))
        self.assertEqual(shadow.cumulative, ledger.cumulative)
        self.assertEqual(shadow.by_stage, ledger.by_stage)

    def test_actual_small_masks_reduction_never_reserves_128_gib_per_callback(self):
        ledger = self.make(); self.start(ledger); ledger.finish_stage([self.small_ref('anchor')])
        ledger.begin_stage('longest_masks', 'fixture/masks')
        controls = raw_cpu.RawArtifactControls(); controls.setUp(); self.addCleanup(controls.doCleanups)
        controls.root = self.root
        controls.expected = dict(stage='longest_masks', request_sha256=self.context['request_sha256'],
            scope_sha256=self.context['scope_sha256'], static_recipe_sha256=self.context['static_recipe_sha256'],
            code_sha256=self.context['bindings']['source']['code_sha256'],
            worker_identity_sha256=self.proof['worker_identity_sha256'], attempt_id='fixture/masks')
        reference = controls.masks()
        bounds = dict(operation_attempts_upper=1000, io_read_bytes_upper=GIB,
            identity_hash_attempts_upper=10000, identity_hash_bytes_upper=GIB, reducer_work_bytes_upper=GIB)
        reader = raw.RawEvidenceReader(ledger.operation, bounds, self.root)
        result = reader.read_masks(reference, controls.expected, controls.bindings)
        self.assertEqual(len(result['observations']['rows']), 90)
        totals = ledger.by_stage['longest_masks']
        self.assertEqual(totals['io_read_bytes_upper'], reference['bytes'] + 1)
        self.assertLess(totals['identity_hash_bytes_upper'], GIB)
        self.assertLess(ledger.reducer_by_stage['longest_masks'], GIB)
        reads = [op for op in ledger.operations.values() if op['kind'] == 'qualification_artifact_readback'
                 and op['stage'] == 'longest_masks']
        self.assertEqual(len(reads), 1); self.assertEqual(reads[0]['profile']['wall_s_upper'], 60)
        reducers = [op for op in ledger.operations.values() if op['kind'] == 'qualification_reducer_overhead']
        self.assertTrue(any(op['profile']['identity_hash_bytes_upper'] == 0 for op in reducers))
        self.assertTrue(all(op['profile']['wall_s_upper'] == 60 for op in reducers))

    def test_large_real_tensor_hash_finish_uses_readback_kind_and_exact_extent(self):
        ledger = self.make(); self.start(ledger)
        size, path = 880 * MIB, self.root / 'large-discarded-fixture.bin'
        with path.open('wb') as stream: stream.truncate(size)
        expected = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(MIB): expected.update(block)
        reference = dict(path=str(path), sha256=expected.hexdigest(), bytes=size)
        before = ledger.by_stage['runtime_anchor']['io_read_bytes_upper']
        ledger.finish_stage([reference])
        self.assertEqual(ledger.attempts[-1]['status'], 'RETURNED')
        self.assertEqual(ledger.by_stage['runtime_anchor']['io_read_bytes_upper'] - before, size + 1)
        operation = next(op for op in ledger.operations.values() if op['kind'] == 'qualification_artifact_readback')
        self.assertEqual(operation['profile']['identity_hash_bytes_upper'], size + 1)
        self.assertEqual(operation['result']['bytes_read'], size)
        self.assertFalse(ledger.snapshot()['native_qualified'])
        with self.assertRaisesRegex(ValueError, 'profile below'):
            a.reservation_profile(self.spec, 'qualification_artifact', self.fields(size + 1))

    def test_invalid_declarations_refuse_before_journal_append_or_body(self):
        ledger = self.make(); self.start(ledger)
        before, journal_size = copy.deepcopy(ledger.cumulative), ledger.path.stat().st_size
        invalid = (dict(read_bytes_upper=True), dict(read_bytes_upper=-1), dict(read_bytes_upper=GIB + 1),
            dict(io_read_bytes_upper=18), dict(identity_hash_attempts_upper=0),
            dict(identity_hash_bytes_upper=0), dict(identity_hash_attempts_upper=1.0),
            dict(wall_s_upper=float('nan')), dict(wall_s_upper=6))
        for index, change in enumerate(invalid):
            fields = dict(self.fields(), **change)
            with self.subTest(index=index), patch.object(Path, 'open', side_effect=AssertionError('unadmitted IO')):
                with self.assertRaises(ValueError): ledger.operation('input_io', 'started', 'bad-%d' % index, **fields)
            self.assertEqual(ledger.cumulative, before); self.assertEqual(ledger.path.stat().st_size, journal_size)
        for fields in (dict(self.fields(), identity_hash_bytes_upper=16), self.fields(0)):
            with self.assertRaises(ValueError): a.reservation_profile(self.spec, 'qualification_artifact_readback', fields)
        with self.assertRaisesRegex(ValueError, 'positive explicit reducer'):
            a.reservation_profile(self.spec, 'qualification_reducer_overhead', dict(read_bytes_upper=0,
                write_bytes_upper=0, identity_hash_attempts_upper=0, identity_hash_bytes_upper=0, reducer_work_bytes_upper=0))

    def test_opaque_incomplete_and_nonallowlisted_work_keep_frozen_maxima(self):
        maximum = self.spec['operation_profiles']['input_io']
        variants = ({}, {'identity_hash_attempts_upper': 1}, {'opaque_source_verification': True,
            'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': 17},
            {'identity_hash_attempts_lower': 1, 'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': 17})
        for fields in variants:
            reserved = a.reservation_profile(self.spec, 'input_io', dict(read_bytes_upper=17, write_bytes_upper=0, **fields))
            self.assertEqual(reserved['read_bytes_upper'], 17)
            self.assertEqual(reserved['identity_hash_bytes_upper'], maximum['identity_hash_bytes_upper'])
            self.assertEqual(reserved['identity_hash_attempts_upper'], maximum['identity_hash_attempts_upper'])
        for kind in ('model_load', 'qualification_artifact', 'fingerprint'):
            self.assertEqual(a.reservation_profile(self.spec, kind, {}), self.spec['operation_profiles'][kind])
        legacy = copy.deepcopy(self.spec); legacy.pop('operation_profile_rule'); legacy.pop('explicit_reservation_kinds')
        self.assertEqual(a.reservation_profile(legacy, 'input_io', self.fields()), legacy['operation_profiles']['input_io'])
        incomplete = dict(read_bytes_upper=0, write_bytes_upper=0, reducer_work_bytes_upper=17)
        self.assertEqual(a.reservation_profile(self.spec, 'qualification_reducer_overhead', incomplete),
                         self.spec['operation_profiles']['qualification_reducer_overhead'])

    def test_return_above_selected_bound_is_fatal_and_keeps_permanent_charge(self):
        ledger = self.make(); self.start(ledger)
        ledger.operation('input_io', 'started', 'small', **self.fields())
        before = copy.deepcopy(ledger.cumulative)
        with self.assertRaisesRegex(ValueError, 'actual qualification work exceeds'):
            ledger.operation('input_io', 'returned', 'small', bytes_read=18)
        self.assertTrue(ledger.fatal); self.assertEqual(ledger.cumulative, before)
        self.assertEqual(ledger.operations['small']['phase'], 'started')
        with self.assertRaisesRegex(ValueError, 'fatal'): ledger.finish_stage([self.small_ref()])

    def test_exact_rule_and_allowlist_and_native_maximum_mutation_refuse(self):
        for change in ({'operation_profile_rule': 'unknown'}, {'explicit_reservation_kinds': ['input_io']},
                       {'explicit_reservation_kinds': list(reversed(a.EXPLICIT_RESERVATION_KINDS))}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                a.validate_reservation_rule(dict(self.spec, **change))
        ledger = self.make(); self.start(ledger); ledger.fixture = False
        ledger.profiles['input_io']['read_bytes_upper'] += 1
        with patch.object(a.authority, 'require_execution_permit', return_value=True), \
             patch.object(Path, 'open', side_effect=AssertionError('mutated native profile IO')):
            with self.assertRaisesRegex(ValueError, 'specification/prices mutated'):
                ledger.operation('input_io', 'started', 'mutated', **self.fields())

    def test_selected_profile_mutation_and_rehashed_profile_discount_refuse(self):
        ledger = self.make(); self.start(ledger)
        ledger.operation('input_io', 'started', 'small', **self.fields())
        ledger.operations['small']['profile']['read_bytes_upper'] = 18
        with self.assertRaisesRegex(ValueError, 'profile mutated'):
            ledger.operation('input_io', 'returned', 'small', bytes_read=18)
        ledger.operations['small']['profile']['read_bytes_upper'] = 17
        ledger.operation('input_io', 'failed', 'small', known_bytes_read=1)
        ledger.fail_stage(RuntimeError('discarded fixture')); ledger.end()
        events = copy.deepcopy(ledger.events)
        target = next(row for row in events if row['kind'] == 'operation' and row['payload']['phase'] == 'started')
        target['payload']['reservation_profile']['read_bytes_upper'] = 1
        previous = None
        for row in events:
            row['previous'] = previous; row.pop('sha256'); row['sha256'] = a.digest(row); previous = row['sha256']
        ledger.path.write_bytes(b''.join(a.canonical(row) + b'\n' for row in events))
        with self.assertRaisesRegex(ValueError, 'reservation profile drift'): self.make()


if __name__ == '__main__': unittest.main()
