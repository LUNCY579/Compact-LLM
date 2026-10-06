"""Actual shared guard raw producer; all libraries/observations are inert."""
from copy import deepcopy
import hashlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import test_library_policy as legacy
import test_library_managed_scope as managed


class RawObservationTests(unittest.TestCase):
    def setUp(self):
        self.f = legacy.LibraryBoundaryTests('runTest'); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.B = self.f.B; self.f.policy['kind'] = self.B.MANAGED_KIND
        self.f.policy['framework_transition'] = [self.f.extra]
        self.f.policy['extension_transitions'][self.f.ext] = {'create': [self.f.ext], 'exec': [self.f.dep]}
        self.f.supplier = managed.ManagedSupplier(self.f.policy)
        self.ticks = 0
        self.identity = {'source_commit': 'a'*40, 'original_scope': 'ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY',
            'owned_birth': {'pid': 101, 'creation_filetime_100ns': 1234567}, 'owned_birth_status': 'VALIDATED',
            'owner_receipt_sha256': 'c'*64, 'context_sha256': 'b'*64,
            'producer_source': deepcopy(self.f.policy['provider_source']),
            'fixture_only': True, 'native_PASS': False, 'native_authority_issued': False}
    def clock(self):self.ticks += 1; return self.ticks*10**6
    def guard(self, clock=None, budget=None):
        return self.B._CPU_make_guard(self.f.policy,self.f.C,self.f.supplier,
            _capability=self.B._CPU_library_capability(),clock_ns=self.clock if clock is None else clock,budget=budget)
    def complete(self):
        guard = self.guard()
        with guard.framework_import():
            self.f.supplier.add(self.f.extra)  # Actual direct load BEFORE child entry.
            with guard.extension(self.f.ext,'create'):self.f.supplier.add(self.f.ext)
            with guard.extension(self.f.ext,'exec'):self.f.supplier.add(self.f.dep)
        guard.close(); return guard
    def replay(self, packet):
        evidence = legacy.source(legacy.HERE/'prefix_library_evidence.py')
        raw = (self.B._raw_canonical(packet)+'\n').encode('ascii')
        plan = [{'id': 1,'parent_id': None,'name': 'framework_import'},
                {'id': 2,'parent_id': 1,'name': 'extension:'+self.f.ext+':create'},
                {'id': 3,'parent_id': 1,'name': 'extension:'+self.f.ext+':exec'}]
        source = self.f.policy['provider_source']
        return evidence.validate_observed_packet(raw,policy=self.f.policy,expected_identity=self.identity,
            selected_file_map=self.f.files,selected_source_map={source['path']:source['descriptor']},
            expected_boundaries=plan,expected_file_sha256=hashlib.sha256(raw).hexdigest())
    def test_source_produced_nested_packet_replays_against_separate_external_plan(self):
        packet = self.complete().export_raw_packet(self.identity)
        result = self.replay(packet)
        self.assertTrue(result['conservation_complete']); self.assertFalse(result['native_PASS'])
        self.assertFalse(result['current_runtime_bytes_qualified'])
        points = packet['observations']
        self.assertEqual([p['event'] for p in points],['initial','enter','enter','exit','enter','exit','exit','precleanup','cleanup'])
        self.assertEqual([(p['boundary']['id'],p['boundary']['parent_id']) for p in points if p['event']=='enter'],[(1,None),(2,1),(3,1)])
        self.assertIn(self.f.extra,[p['path'] for p in points[2]['snapshot']['libraries']])
        self.assertEqual(points[2]['event_trace']['events'][0]['path'],self.f.extra)
    def test_recording_adds_no_provider_reads_and_export_observes_nothing(self):
        guard = self.complete(); calls = list(self.f.supplier.calls)
        packet = guard.export_raw_packet(self.identity); guard.export_raw_packet(self.identity)
        self.assertEqual(calls,self.f.supplier.calls)
        self.assertEqual(sum(p[0]=='scope' for p in calls),9)
        self.assertEqual(sum(p[0]=='libraries' for p in calls),9)
        self.assertEqual(sum(p[0]=='events' for p in calls),9)
        self.assertEqual(packet['producer_claims']['observation_count'],9)
    def test_raw_returned_values_are_copied_before_supplier_mutation(self):
        guard = self.complete(); packet = guard.export_raw_packet(self.identity)
        before = deepcopy(packet['observations'])
        self.f.supplier.event_rows[0]['path'] = 'CPU_changed_after_return'
        self.f.supplier.rows.clear(); packet['observations'][0]['scope']['directories'].clear()
        self.assertEqual(guard.export_raw_packet(self.identity)['observations'],before)
    def test_source_failure_clears_current_values_and_retains_startup_packet(self):
        def fail(path,expected,*,budget=None):raise ValueError('source precheck failure')
        with patch.object(self.f.C,'file_descriptor',side_effect=fail):
            with self.assertRaisesRegex(ValueError,'source precheck failure') as caught:self.guard()
        packet = caught.exception.__library_raw_packet__
        row = packet['observations'][0]
        self.assertEqual(row['event'],'initial');self.assertIsNone(row['scope']);self.assertIsNone(row['snapshot'])
        self.assertIsNone(row['event_trace']);self.assertIsNone(row['directory_identities'])
        self.assertEqual(packet['failure_records'][0]['stage'],'source')
        self.assertEqual(packet['failure_records'][0]['error']['message'],'source precheck failure')
        self.assertEqual(self.f.supplier.calls,[]);self.assertEqual(packet['terminal']['status'],'failed')
    def test_failed_scope_does_not_reuse_last_checkpoint_snapshot_or_events(self):
        guard = self.guard(); self.f.supplier.scope_changes['managed_setup_complete'] = False
        with self.assertRaises(ValueError):
            with guard.framework_import():self.fail('delegate must not run')
        enter = guard.export_raw_packet(self.identity)['observations'][-1]
        self.assertEqual(enter['event'],'enter');self.assertIsNone(enter['snapshot']);self.assertIsNone(enter['event_trace'])
        self.assertFalse(enter['scope']['managed_setup_complete'])
        self.assertIn('snapshot',enter['observation_errors'])
        self.f.supplier.scope_changes.clear()
        with self.assertRaises(ValueError):guard.close()
    def test_original_delegate_once_plus_actual_failed_post_observations(self):
        guard = self.guard(); primary = RuntimeError('original ordinary delegated failure')
        with self.assertRaises(RuntimeError) as caught:
            with guard.framework_import():self.f.supplier.add(self.f.extra);raise primary
        self.assertIs(caught.exception,primary)
        with self.assertRaises(ValueError):guard.close()
        packet = guard.export_raw_packet(self.identity)
        originals = [f for f in packet['failure_records'] if f['error']['message']=='original ordinary delegated failure']
        self.assertEqual(len(originals),1)
        error = next(row for row in packet['observations'] if row['event']=='boundary_error')
        self.assertIn(self.f.extra,[row['path'] for row in error['snapshot']['libraries']])
        self.assertEqual(packet['terminal']['status'],'failed');self.assertFalse(packet['producer_claims']['completion_claimed'])
    def test_primary_begin_and_cleanup_failures_are_both_preserved(self):
        original = self.f.supplier.begin
        def begin(*args):original(*args);raise RuntimeError('original begin failure')
        self.f.supplier.begin = begin; self.f.supplier.close_error = True
        with self.assertRaises(ValueError) as caught:self.guard()
        packet = caught.exception.__library_raw_packet__
        messages = [row['error']['message'] for row in packet['failure_records']]
        self.assertIn('original begin failure',messages)
        self.assertIn('injected managed cookie removal refusal',messages)
        self.assertIsNone(packet['observations'][-1]['snapshot'])
    def test_precleanup_and_final_cleanup_use_reached_current_observations(self):
        guard = self.complete(); packet = guard.export_raw_packet(self.identity)
        pre, final = packet['observations'][-2:]
        self.assertEqual(pre['event'],'precleanup');self.assertEqual(final['event'],'cleanup')
        self.assertIsNone(final['scope']);self.assertIsNotNone(final['cleanup'])
        self.assertEqual(final['event_trace'],pre['event_trace'])
        self.assertEqual(final['directory_identities'],pre['directory_identities'])
        self.assertEqual([row['time_ns'] for row in packet['observations']],sorted(row['time_ns'] for row in packet['observations']))
    def test_final_drain_failure_is_raw_and_cannot_complete(self):
        guard = self.guard(); original = self.f.supplier.events
        def events():
            result = original()
            if self.f.supplier.closed:result['complete'] = False
            return result
        self.f.supplier.events = events
        with self.assertRaises(ValueError):guard.close()
        packet = guard.export_raw_packet(self.identity);final = packet['observations'][-1]
        self.assertFalse(final['event_trace']['complete']);self.assertIsNone(final['snapshot'])
        self.assertFalse(packet['producer_claims']['completion_claimed'])
    def test_hostile_exception_stringification_is_never_called(self):
        calls = []
        class Hostile(RuntimeError):
            def __str__(self):calls.append('str');raise AssertionError('must not stringify')
        guard = self.guard()
        with self.assertRaises(Hostile):
            with guard.framework_import():raise Hostile('ordinary args text')
        with self.assertRaises(ValueError):guard.close()
        packet = guard.export_raw_packet(self.identity)
        row = next(row for row in packet['failure_records'] if row['error']['type']=='Hostile')
        self.assertFalse(row['error']['message_exact']);self.assertEqual(calls,[])
    def test_nonordinary_observation_is_not_repr_or_inspected(self):
        calls = []
        class ModelLike:
            def __repr__(self):calls.append('repr');raise AssertionError('no model inspection')
        self.f.supplier.snapshot_changes['libraries'] = [ModelLike()]
        with self.assertRaises(ValueError) as caught:self.guard()
        self.assertEqual(calls,[])
        packet = caught.exception.__library_raw_packet__
        self.assertIsNone(packet['observations'][0]['snapshot'])
        self.assertIn('snapshot',packet['observations'][0]['observation_errors'])
    def test_missing_birth_remains_unavailable_and_no_observation_is_added(self):
        guard = self.complete(); identity = deepcopy(self.identity)
        identity['owned_birth'] = None;identity['owned_birth_status'] = 'UNAVAILABLE'
        calls = list(self.f.supplier.calls);packet = guard.export_raw_packet(identity)
        self.assertEqual(packet['terminal']['status'],'incomplete')
        self.assertFalse(packet['producer_claims']['completion_claimed']);self.assertEqual(calls,self.f.supplier.calls)
    def test_invalid_clock_retains_initial_partial_and_never_calls_provider(self):
        with self.assertRaisesRegex(ValueError,'raw observation clock') as caught:self.guard(clock=lambda:True)
        packet = caught.exception.__library_raw_packet__
        self.assertIsNone(packet['observations'][0]['time_ns'])
        self.assertEqual(packet['failure_records'][0]['stage'],'clock')
        self.assertEqual(self.f.supplier.calls,[])
    def test_journal_uses_budget_without_changing_authority(self):
        calls = []
        budget = types.SimpleNamespace(check=lambda amount=0:calls.append(amount))
        guard = self.guard(budget=budget);guard.close()
        self.assertTrue(any(amount>0 for amount in calls))
        self.assertFalse(guard.export_raw_packet(self.identity)['native_PASS'])
    def test_expired_budget_cannot_skip_actual_cleanup_or_complete_packet(self):
        state = {'expired': False}
        def check(amount=0):
            if state['expired']:raise RuntimeError('recording budget expired')
        guard = self.guard(budget=types.SimpleNamespace(check=check));state['expired'] = True
        with self.assertRaises((ValueError,RuntimeError)):guard.close()
        self.assertTrue(self.f.supplier.closed);self.assertTrue(guard.closed)
        packet = guard.export_raw_packet(self.identity)
        self.assertFalse(packet['producer_claims']['completion_claimed'])
        self.assertFalse(packet['terminal']['recording_complete'])
    def test_checkpoint_limit_cannot_skip_actual_cleanup(self):
        with patch.object(self.B,'MAX_RAW_OBSERVATIONS',1):
            guard = self.guard()
            with self.assertRaises((ValueError,RuntimeError)):guard.close()
        self.assertTrue(self.f.supplier.closed);self.assertTrue(guard.closed)
        self.assertFalse(guard.export_raw_packet(self.identity)['producer_claims']['completion_claimed'])


if __name__=='__main__':unittest.main()
