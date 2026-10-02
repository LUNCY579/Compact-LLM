"""Paid real-file fixture replay; no OS/process/framework/native execution."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_native_services as s, ce_service_bootstrap_verifier as v


class Clock:
    def __init__(self): self.value = 100.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


def specification(root):
    profile = dict(read_bytes_upper=1000, write_bytes_upper=1000, identity_hash_attempts_upper=10,
                   identity_hash_bytes_upper=1000, wall_s_upper=2)
    return {'owned_directory': str(root), 'caps': dict(read_bytes=10**8, write_bytes=10**8,
        hash_attempts=10**6, hash_bytes=10**8, metadata_bytes=10**8, operation_attempts=1000, wall_seconds=1000),
        'setup_bounds': dict(profile), 'operation_profiles': {'fixture': dict(profile)},
        'physical_wall_seconds_upper': 100, 'finalization_bounds': dict(read_bytes_upper=8*1024**2,
            write_bytes_upper=2*1024**2, identity_hash_attempts_upper=8,
            identity_hash_bytes_upper=16*1024**2, wall_s_upper=10)}


class Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='service-replay-pure-'); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.clock = Clock(); self.spec = specification(self.root)
        self.ledger = s.BootstrapLedger(self.spec, 'a'*64, synthetic_fixture=True, clock=self.clock)
        self.context = {'request_sha256': 'a'*64, 'mode': 'synthetic_fixture', 'verification_binding': {
            'stage': 'service_bootstrap_verification', 'request_sha256': 'a'*64, 'scope_sha256': 'b'*64,
            'static_recipe_sha256': 'c'*64, 'code_sha256': {'fixture.py': 'd'*64},
            'worker_identity_sha256': 'e'*64, 'attempt_id': 'independent-service-replay'}}
        self.calls = []
    def ref(self, path):
        blob = path.read_bytes(); return {'path': str(path), 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
    def reader(self):
        return v.raw.RawEvidenceReader(lambda *args, **fields: self.calls.append((args, fields)),
            {key: 10**12 for key in v.raw.BUDGET_FIELDS}, self.root)
    def verify(self, seal=False):
        return v.verify_service_bootstrap(self.context, self.spec, self.ref(self.ledger.header_path), self.ref(self.ledger.path),
            self.ref(self.ledger.seal_path) if seal else None, reader=self.reader(),
            outer_preflight=lambda *args: self.calls.append(('preflight', args)))
    def rewrite(self, mutate):
        rows = [v.raw._json(line) for line in self.ledger.path.read_bytes().splitlines()]; mutate(rows); previous = None
        for seq, row in enumerate(rows):
            row.update(seq=seq, previous=previous)
            row['sha256'] = v.digest({key: value for key, value in row.items() if key != 'sha256'}); previous = row['sha256']
        self.ledger.path.write_bytes(b''.join(v.canonical(row)+b'\n' for row in rows))

    def test_whole_origin_end_seal_and_failed_costs_replay_without_execution_claim(self):
        self.ledger.operation('fixture', 'started', 'failed')
        self.clock.advance(2); self.ledger.operation('fixture', 'failed', 'failed', error='retained failure', wall_s=2)
        self.clock.advance(3); self.ledger.end(); evidence = self.ledger.inspect_evidence()
        result = self.verify(seal=True)
        self.assertEqual(result['derived_snapshot'], self.ledger.snapshot())
        self.assertEqual(result['derived_totals'], self.ledger.counters)
        self.assertEqual(result['recorded_wall_seconds'], 5); self.assertEqual(result['original_origin'], 100)
        self.assertEqual(result['derived_snapshot']['failed_operations'], ['failed'])
        self.assertEqual(evidence['seal_ref'], self.ref(self.ledger.seal_path))
        self.assertTrue(result['recorded_clock_ended']); self.assertTrue(result['final_suffix_reserved'])
        self.assertFalse(result['suffix_actual_wall_verified']); self.assertFalse(result['execution_verified'])
        self.assertFalse(result['native_qualified']); self.assertEqual(result['status'], 'UNQUALIFIED')
        self.assertEqual(sum(row['bytes'] for row in result['original_offsets']), self.ref(self.ledger.path)['bytes'])

    def test_pending_end_denial_keeps_all_charges_and_original_clock(self):
        self.ledger.operation('fixture', 'started', 'pending'); charged = dict(self.ledger.counters)
        self.clock.advance(2)
        with self.assertRaisesRegex(ValueError, 'cleanup terminal'): self.ledger.end()
        self.assertEqual(charged, self.ledger.counters)
        result = self.verify(); self.assertEqual(result['derived_snapshot']['pending'], ['pending'])
        self.assertFalse(result['recorded_clock_ended']); self.assertFalse(self.ledger.seal_path.exists())

    def test_end_is_create_only_one_inspection_and_no_subsequent_operations(self):
        self.clock.advance(1); self.ledger.end(); self.ledger.inspect_evidence(); before = self.ledger.path.read_bytes()
        for action in (self.ledger.end, self.ledger.inspect_evidence,
                lambda: self.ledger.operation('fixture', 'started', 'late'),
                lambda: s.BootstrapLedger(self.spec, 'a'*64, synthetic_fixture=True, clock=self.clock)):
            with self.assertRaises(ValueError): action()
        self.assertEqual(before, self.ledger.path.read_bytes())

    def test_final_bounds_denial_precedes_final_reservation_and_seal_io(self):
        self.ledger.spec['finalization_bounds']['read_bytes_upper'] = 0
        before = dict(self.ledger.counters)
        with self.assertRaisesRegex(ValueError, 'lower bounds'): self.ledger.end()
        self.assertEqual(before, self.ledger.counters); self.assertIsNone(self.ledger._end_reserved)
        self.assertFalse(self.ledger.seal_path.exists())

    def test_short_final_seal_retains_full_reservation_and_is_fatal(self):
        real_open = Path.open
        class Short:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def write(self, blob): return len(blob)-1
        def opened(path, *args, **kwargs):
            return Short() if path == self.ledger.seal_path else real_open(path, *args, **kwargs)
        with patch.object(Path, 'open', opened):
            with self.assertRaisesRegex(ValueError, 'short final'): self.ledger.end()
        self.assertTrue(self.ledger.fatal); self.assertTrue(self.ledger.sealed)
        self.assertGreaterEqual(self.ledger.counters['read_bytes'], self.spec['finalization_bounds']['read_bytes_upper'])
        self.assertIsNotNone(self.ledger._end_reserved)
        with self.assertRaises(ValueError): self.ledger.inspect_evidence()

    def test_slow_final_readback_rejects_and_clock_cannot_reset(self):
        self.ledger.end(); self.clock.advance(11)
        with self.assertRaisesRegex(ValueError, 'suffix exceeds'): self.ledger.inspect_evidence()
        self.assertTrue(self.ledger.fatal)
        with self.assertRaises(ValueError): self.ledger.tick_wall()

    def test_external_preflight_denial_precedes_fs_and_hash(self):
        refs = self.ref(self.ledger.header_path), self.ref(self.ledger.path)
        def denied(*args): raise RuntimeError('external reservation denied')
        with patch.object(Path, 'open', side_effect=AssertionError('unpaid IO')), patch.object(v, 'digest', side_effect=AssertionError('unpaid hash')):
            with self.assertRaisesRegex(RuntimeError, 'reservation denied'):
                v.verify_service_bootstrap(self.context, self.spec, *refs, reader=self.reader(), outer_preflight=denied)

    def test_rehashed_clock_origin_refund_profile_and_end_order_reject(self):
        self.ledger.operation('fixture', 'started', 'one'); self.clock.advance(1)
        self.ledger.operation('fixture', 'returned', 'one'); self.ledger.end()
        original = self.ledger.path.read_bytes()
        changes = (
            lambda rows: rows[1]['payload'].update(original_origin=99),
            lambda rows: next(row for row in rows if row['kind']=='wall')['payload'].update(elapsed_delta=0),
            lambda rows: next(row for row in rows if row['kind']=='end_reservation')['payload']['bounds'].update(read_bytes_upper=0),
            lambda rows: rows.append(copy.deepcopy(rows[2])),
            lambda rows: next(row for row in rows if row['kind']=='operation' and row['payload']['phase']=='returned')['payload']['fields'].update(bytes_read=1001),
        )
        for change in changes:
            self.ledger.path.write_bytes(original); self.rewrite(change)
            with self.assertRaises(ValueError): self.verify()
        self.ledger.path.write_bytes(original)
        seal = v.raw._json(self.ledger.seal_path.read_bytes()); seal['cumulative']['read_bytes'] -= 1
        self.ledger.seal_path.write_bytes(v.canonical(seal)+b'\n')
        with self.assertRaisesRegex(ValueError, 'refund/status/clock'): self.verify(seal=True)

    def test_partial_chain_and_changed_frozen_spec_reject(self):
        self.ledger.path.write_bytes(self.ledger.path.read_bytes()+b'{partial')
        with self.assertRaisesRegex(ValueError, 'partial service tail'): self.verify()
        self.spec['operation_profiles']['fixture']['read_bytes_upper'] += 1
        with self.assertRaisesRegex(ValueError, 'header/scope/request'): self.verify()

    def test_monotonic_regression_and_physical_deadline_reject(self):
        for time in (99, 201):
            self.clock.value = time
            with self.assertRaises(ValueError): self.ledger.tick_wall()
        self.assertEqual(self.ledger.last_clock, 100)
        self.assertTrue(self.ledger.fatal)

    def test_final_cap_denial_keeps_original_pending_work_and_no_seal(self):
        self.ledger.spec['caps']['read_bytes'] = self.ledger.counters['read_bytes'] + 1
        before = dict(self.ledger.counters)
        with self.assertRaisesRegex(ValueError, 'cap exhausted'): self.ledger.end()
        self.assertEqual(before, self.ledger.counters)
        self.assertFalse(self.ledger.seal_path.exists()); self.assertIsNone(self.ledger._end_reserved)

    def test_native_spec_drift_refuses_before_clock_or_files(self):
        # No native construction or registry toggle: exercise the pure guard on
        # this existing metadata fixture, without granting execution authority.
        self.ledger._native = True
        self.ledger.spec['caps']['read_bytes'] += 1
        with patch.object(Path, 'open', side_effect=AssertionError('drift before IO')):
            with self.assertRaisesRegex(ValueError, 'specification changed'): self.ledger.tick_wall()

    def input_ledger(self):
        spec=specification(self.root/'input-ledger')
        spec['operation_profile_rule']=s.INPUT_RESERVATION_RULE
        spec['operation_profiles']['input_io']=dict(spec['operation_profiles']['fixture'])
        self.spec=spec;self.ledger=s.BootstrapLedger(spec,'a'*64,synthetic_fixture=True,clock=self.clock)
        return self.ledger

    def test_explicit_small_input_upper_reserved_before_work_no_return_refund(self):
        ledger=self.input_ledger();before=dict(ledger.counters)
        ledger.operation('input_io','started','row',read_bytes_upper=17,io_read_bytes_upper=17,
                         write_bytes_upper=0,identity_hash_attempts_upper=1,identity_hash_bytes_upper=17)
        self.assertEqual(ledger.counters['read_bytes']-before['read_bytes'],17)
        self.assertEqual(ledger.operations['row']['profile']['write_bytes_upper'],0)
        self.assertEqual(ledger.operations['row']['profile']['identity_hash_attempts_upper'],1)
        self.assertEqual(ledger.events[-1]['payload']['reservation_profile'],ledger.operations['row']['profile'])
        charged=ledger.counters['read_bytes'];ledger.operation('input_io','returned','row',bytes_read=3,bytes_written=0,
            identity_hash_attempts_actual=1,identity_hash_bytes_actual=3)
        self.assertEqual(ledger.counters['read_bytes'],charged)
        self.assertEqual(self.verify()['derived_totals'],ledger.counters)

    def test_failed_and_pending_explicit_input_remain_fully_reserved(self):
        ledger=self.input_ledger()
        for name in ('failed','pending'):
            ledger.operation('input_io','started',name,io_read_bytes_upper=19,io_write_bytes_upper=0,
                identity_hash_attempts_upper=1,identity_hash_bytes_upper=19)
        ledger.operation('input_io','failed','failed',error='actual failure')
        result=self.verify();self.assertEqual(result['derived_snapshot']['pending'],['pending'])
        self.assertEqual(result['derived_snapshot']['failed_operations'],['failed'])
        self.assertGreaterEqual(result['derived_totals']['read_bytes'],38)

    def test_opaque_or_missing_hash_upper_retains_full_hash_ceiling(self):
        ledger=self.input_ledger();maximum=ledger.spec['operation_profiles']['input_io']
        for index,extra in enumerate(({}, {'identity_hash_attempts_upper':1},
                {'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':17,'identity_hash_attempts_lower':1},
                {'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':17,'opaque_source_verification':True})):
            ledger.operation('input_io','started',str(index),read_bytes_upper=17,write_bytes_upper=0,**extra)
            reserved=ledger.operations[str(index)]['profile']
            self.assertEqual(reserved['read_bytes_upper'],17)
            self.assertEqual(reserved['identity_hash_attempts_upper'],maximum['identity_hash_attempts_upper'])
            self.assertEqual(reserved['identity_hash_bytes_upper'],maximum['identity_hash_bytes_upper'])
            ledger.operation('input_io','returned',str(index))
        self.assertEqual(self.verify()['derived_totals'],ledger.counters)

    def test_rule_does_not_change_other_kinds_or_unmarked_legacy_fixture(self):
        ledger=self.input_ledger();ledger.operation('fixture','started','unchanged',read_bytes_upper=1)
        self.assertEqual(ledger.operations['unchanged']['profile'],ledger.spec['operation_profiles']['fixture'])
        unmarked=copy.deepcopy(self.spec);unmarked.pop('operation_profile_rule')
        self.assertEqual(s.reservation_profile(unmarked,'input_io',{'read_bytes_upper':1}),unmarked['operation_profiles']['input_io'])

    def test_explicit_profile_abuse_denied_before_charges_or_io(self):
        ledger=self.input_ledger();before=dict(ledger.counters)
        for fields in ({'read_bytes_upper':1001}, {'read_bytes_upper':True,'io_read_bytes_upper':1},
                {'read_bytes_upper':1,'io_read_bytes_upper':2},
                {'identity_hash_attempts_upper':0,'identity_hash_bytes_upper':1},
                {'identity_hash_bytes_lower':1001}, {'identity_hash_attempts_upper':True}):
            with patch.object(Path,'open',side_effect=AssertionError('invalid reservation before IO')):
                with self.assertRaises(ValueError):ledger.operation('input_io','started','denied',**fields)
            self.assertEqual(before,ledger.counters)

    def test_return_over_explicit_reservation_is_fatal_full_charge_stays(self):
        ledger=self.input_ledger();ledger.operation('input_io','started','small',read_bytes_upper=1,
            write_bytes_upper=0,identity_hash_attempts_upper=1,identity_hash_bytes_upper=1)
        before=dict(ledger.counters)
        with self.assertRaisesRegex(ValueError,'actual work exceeds'):
            ledger.operation('input_io','returned','small',bytes_read=2)
        self.assertTrue(ledger.fatal);self.assertEqual(before,ledger.counters);self.assertEqual(ledger.snapshot()['pending'],['small'])

    def test_rehashed_declared_profile_discount_is_rejected_by_independent_replay(self):
        ledger=self.input_ledger();ledger.operation('input_io','started','small',read_bytes_upper=17,
            write_bytes_upper=0,identity_hash_attempts_upper=1,identity_hash_bytes_upper=17)
        self.rewrite(lambda rows:next(row for row in rows if row['kind']=='operation')['payload']['reservation_profile'].update(read_bytes_upper=1))
        with self.assertRaisesRegex(ValueError,'permanent input reservation'):self.verify()

    def row_spec(self):
        ledger=self.input_ledger();self.spec['input_io_row_wall_s_upper']=5
        self.spec['operation_profiles']['input_io'].update(read_bytes_upper=1024**2,write_bytes_upper=1024**2,
            identity_hash_attempts_upper=100,identity_hash_bytes_upper=1024**2,wall_s_upper=60)
        # Fresh pure fixture stores the actual frozen rule in its raw header.
        self.spec['owned_directory']=str(self.root/'row-ledger')
        self.ledger=s.BootstrapLedger(self.spec,'a'*64,synthetic_fixture=True,clock=self.clock)
        return self.ledger

    def test_small_row_wall_reserved_at_five_seconds_and_rederived(self):
        ledger=self.row_spec();before=ledger.counters['wall_seconds']
        ledger.operation('input_io','started','row',row_id='actual-bound-row',io_read_bytes_upper=17,
            io_write_bytes_upper=0,identity_hash_attempts_upper=1,identity_hash_bytes_upper=17)
        self.assertEqual(ledger.operations['row']['profile']['wall_s_upper'],5)
        self.assertEqual(ledger.counters['wall_seconds']-before,5)
        self.clock.advance(1);ledger.operation('input_io','returned','row',wall_s=1)
        self.assertEqual(self.verify()['derived_totals'],ledger.counters)
        self.rewrite(lambda rows:next(row for row in rows if row['kind']=='operation')['payload']['reservation_profile'].update(wall_s_upper=1))
        with self.assertRaisesRegex(ValueError,'permanent input reservation'):self.verify()

    def test_non_row_large_opaque_or_incomplete_input_keeps_full_wall_ceiling(self):
        ledger=self.row_spec();base={'row_id':'row','read_bytes_upper':17,'write_bytes_upper':0,
            'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':17}
        changes=({'row_id':None},{'read_bytes_upper':s.INPUT_ROW_BYTES+1},
            {'identity_hash_bytes_upper':s.INPUT_ROW_BYTES+1},{'identity_hash_attempts_upper':5},
            {'opaque_source_verification':True},{'identity_hash_bytes_lower':1},{'write_bytes_upper':1})
        for index,change in enumerate(changes):
            ledger.operation('input_io','started',str(index),**dict(base,**change))
            self.assertEqual(ledger.operations[str(index)]['profile']['wall_s_upper'],60)
            ledger.operation('input_io','returned',str(index))
        incomplete=dict(base);incomplete.pop('write_bytes_upper')
        ledger.operation('input_io','started','incomplete',**incomplete)
        self.assertEqual(ledger.operations['incomplete']['profile']['wall_s_upper'],60)
        self.assertEqual(self.verify()['derived_totals'],ledger.counters)

    def test_row_wall_rule_abuse_or_declared_overrun_refuses_before_work(self):
        ledger=self.row_spec();fields={'row_id':'row','read_bytes_upper':17,'write_bytes_upper':0,
            'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':17,'wall_s_upper':6}
        before=dict(ledger.counters)
        with self.assertRaisesRegex(ValueError,'source fixed row wall'):ledger.operation('input_io','started','denied',**fields)
        self.assertEqual(before,ledger.counters)
        for invalid in (True,0,4,6,float('inf')):
            spec=copy.deepcopy(self.spec);spec['input_io_row_wall_s_upper']=invalid
            with self.assertRaisesRegex(ValueError,'source fixed five'):s.reservation_profile(spec,'input_io',{})
        fields.pop('wall_s_upper');ledger.operation('input_io','started','overrun',**fields);before=dict(ledger.counters)
        with self.assertRaisesRegex(ValueError,'actual work exceeds'):ledger.operation('input_io','returned','overrun',wall_s=6)
        self.assertTrue(ledger.fatal);self.assertEqual(before,ledger.counters)


if __name__ == '__main__': unittest.main()
