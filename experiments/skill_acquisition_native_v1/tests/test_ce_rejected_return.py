"""Pure prepaid failure record/replay fixtures; no ML or native worker work."""
import copy
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_rejected_return as r
from experiments.skill_acquisition_native_v1 import ce_operation_contracts as c
from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as q
from experiments.skill_acquisition_native_v1 import ce_native_services as s
from .test_ce_operation_contracts import q_context, manifest
from .test_ce_service_bootstrap_verifier import specification, Clock
from . import test_ce_service_bootstrap_verifier as service_controls


class PrepaidRejectedReturn(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='prepaid-rejection-pure-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
    def qualification(self,name='q'):
        context,proof,receipt=q_context(self.root/name)
        spec=context['bindings']['scope']['qualifier_accounting'];spec['rejected_return_policy']=r.default_policy()
        ledger=q.QualificationAccounting(context,None,proof,receipt,clock=lambda:100.0)
        ledger.begin_stage('runtime_anchor','fixture/runtime')
        ledger.operation('input_io','started','leaf',wall_contract=c.FILE_HASH_LEAF,
            read_bytes_upper=17,write_bytes_upper=0,identity_hash_attempts_upper=1,identity_hash_bytes_upper=17)
        return context,receipt,ledger
    def bootstrap(self,name='b'):
        spec=specification(self.root/name);spec['operation_contracts']=manifest()
        spec['rejected_return_policy']=r.default_policy();spec['operation_profile_rule']=s.INPUT_RESERVATION_RULE
        spec['explicit_reservation_kinds']=['input_io','worker_source_verification'];spec['caps'].update(wall_seconds=10**9,operation_attempts=100000)
        for row in manifest()['contracts'].values():
            if 'bootstrap' in row['owners'] and row['kind'] in c.MAPPED_KINDS:
                p=spec['operation_profiles'].setdefault(row['kind'],dict(read_bytes_upper=10**6,write_bytes_upper=10**6,
                    identity_hash_attempts_upper=10000,identity_hash_bytes_upper=10**6,wall_s_upper=1))
                p['wall_s_upper']=max(p['wall_s_upper'],row['wall_s_upper'])
        ledger=s.BootstrapLedger(spec,'a'*64,synthetic_fixture=True,clock=Clock())
        ledger.operation('input_io','started','leaf',wall_contract=c.FILE_HASH_LEAF,
            read_bytes_upper=17,write_bytes_upper=0,identity_hash_attempts_upper=1,identity_hash_bytes_upper=17)
        return spec,ledger
    def reject(self,ledger,**fields):
        with self.assertRaises(ValueError):
            ledger.operation('input_io','returned','leaf',wall_contract=c.FILE_HASH_LEAF,**fields)
    def shadow(self,context,receipt,ledger):
        state=q.replay_state(context,ledger.header,receipt)
        for row in ledger.events:
            state._apply(row['kind'],row['payload'],replay=True);state.events.append(copy.deepcopy(row))
        return state
    def test_ticket_fixed_price_before_work_unused_not_refunded(self):
        costs=r.reservation_costs(r.default_policy())
        self.assertEqual(costs,dict(operation_attempts_upper=1,io_read_bytes_upper=0,io_write_bytes_upper=65536,
            metadata_bytes_upper=65536,identity_hash_attempts_upper=8,identity_hash_bytes_upper=33554432,wall_seconds_upper=10))
        _,_,ledger=self.qualification()
        ticket=next(row for row in ledger.events if row['kind']=='rejection_reservation')
        start=ledger.events[ledger.operations['leaf']['started_seq']]
        self.assertLess(ticket['seq'],start['seq']);self.assertFalse(ledger.rejection_used)
        paid=copy.deepcopy(ledger.cumulative)
        ledger.operation('input_io','returned','leaf',wall_contract=c.FILE_HASH_LEAF,wall_s=1,bytes_read=1)
        self.assertGreaterEqual(ledger.cumulative['metadata_bytes_upper'],paid['metadata_bytes_upper'])
        self.assertEqual(ledger.rejection_ticket['bounds'],costs)
    def test_q_finite_rejection_is_durable_prepaid_and_raw_replayed_fatal(self):
        context,receipt,ledger=self.qualification();paid=copy.deepcopy(ledger.cumulative)
        self.reject(ledger,wall_s=61,bytes_read=18)
        self.assertEqual(ledger.cumulative,paid);self.assertTrue(ledger.fatal);self.assertEqual(ledger.operations['leaf']['phase'],'started')
        row=ledger.events[-1];self.assertEqual(row['kind'],'terminal_rejected')
        decoded=r.decode_fields(row['payload']['observed_fields'],r.default_policy())
        self.assertEqual(decoded['wall_s'],61);self.assertEqual(decoded['bytes_read'],18)
        self.assertEqual(json.loads(ledger.path.read_text().splitlines()[-1]),row)
        state=self.shadow(context,receipt,ledger)
        self.assertTrue(state.fatal);self.assertEqual(state.cumulative,ledger.cumulative)
        self.assertEqual(state.operations['leaf']['phase'],'started');self.assertTrue(state.worker_open)
        with self.assertRaises(ValueError):ledger.end()
        with self.assertRaises(ValueError):ledger.seal()
        with self.assertRaises(ValueError):ledger.operation('input_io','started','later',wall_contract=c.FILE_HASH_LEAF)
        with self.assertRaises(ValueError):q.QualificationAccounting(context,None,ledger.proof,receipt,clock=lambda:100.0)
    def test_bootstrap_durable_rejection_no_refund_end_or_success_seal(self):
        _,ledger=self.bootstrap();paid=copy.deepcopy(ledger.counters)
        self.reject(ledger,wall_seconds=61,identity_hash_attempts_actual=2)
        self.assertEqual(ledger.counters,paid);self.assertTrue(ledger.fatal);self.assertFalse(ledger.sealed)
        self.assertEqual(ledger.events[-1]['kind'],'terminal_rejected');self.assertEqual(ledger.operations['leaf']['phase'],'started')
        self.assertEqual(json.loads(ledger.path.read_text().splitlines()[-1]),ledger.events[-1])
        with self.assertRaises(ValueError):ledger.end()
        with self.assertRaises(ValueError):ledger.operation('input_io','started','later',wall_contract=c.FILE_HASH_LEAF)
    def test_independent_service_raw_replay_matches_prepaid_fatal_snapshot(self):
        spec,ledger=self.bootstrap('raw-replay');self.reject(ledger,wall_s=61)
        control=service_controls.Tests();control.setUp();self.addCleanup(control.doCleanups)
        control.root=self.root;control.spec=spec;control.ledger=ledger
        replay=control.verify()
        self.assertEqual(replay['derived_snapshot'],ledger.snapshot())
        self.assertEqual(replay['derived_totals'],ledger.counters)
        self.assertTrue(replay['rejected_return_retained']);self.assertFalse(replay['recorded_clock_ended'])
        self.assertFalse(replay['native_qualified']);self.assertFalse(replay['execution_verified'])
        self.assertFalse(replay['rejection_publication_wall_verified'])
    def test_independent_service_rejection_tamper_and_postfatal_events_deny(self):
        for index,mutate in enumerate((lambda rows:rows[-1]['payload'].update(selected_profile_sha256='0'*64),
            lambda rows:rows.append(copy.deepcopy(rows[1])))):
            spec,ledger=self.bootstrap('raw-tamper-%d'%index);self.reject(ledger,wall_s=61)
            control=service_controls.Tests();control.setUp();self.addCleanup(control.doCleanups)
            control.root=self.root;control.spec=spec;control.ledger=ledger;control.rewrite(mutate)
            with self.subTest(index=index),self.assertRaises(ValueError):control.verify()
    def test_actual_terminal_hash_calls_and_bytes_fit_repriced_ticket(self):
        _,ledger=self.bootstrap('hash-count')
        calls=[]; original_digest=r.digest
        def count_digest(value):
            calls.append(len(r.canonical(value)));return original_digest(value)
        with patch.object(r,'digest',side_effect=count_digest):self.reject(ledger,wall_s=61)
        self.assertEqual(len(calls),7)
        bounds=ledger.rejection_ticket['bounds']
        self.assertLessEqual(len(calls),bounds['identity_hash_attempts_upper'])
        self.assertLessEqual(sum(calls),bounds['identity_hash_bytes_upper'])
        policy=r.default_policy()
        analytic=2*policy['header_bytes_upper']+2*policy['profile_bytes_upper']+3*policy['record_bytes_upper']
        self.assertLessEqual(analytic,policy['hash_bytes_upper'])
    def test_prepaid_wall_covers_encoder_and_hash_pipeline_before_append(self):
        original=r.make_terminal_payload
        for index,owner in enumerate(('qualification','bootstrap')):
            ledger=self.qualification('slow-q')[2] if owner=='qualification' else self.bootstrap('slow-b')[1]
            clock=Clock();ledger.clock=clock
            charged=copy.deepcopy(ledger.cumulative if owner=='qualification' else ledger.counters)
            def slow_encode(*args,**kwargs):
                clock.advance(11);return original(*args,**kwargs)
            with patch.object(r,'make_terminal_payload',side_effect=slow_encode):self.reject(ledger,wall_s=61)
            self.assertTrue(ledger.fatal);self.assertIsNotNone(ledger.rejection_unknown)
            self.assertIn('publication wall exceeded',ledger.rejection_unknown['message'])
            self.assertEqual(ledger.cumulative if owner=='qualification' else ledger.counters,charged)
            self.assertEqual(ledger.operations['leaf']['phase'],'started')
    def test_nan_ieee_payload_bool_and_alias_conflict_exact_typed_diagnostics(self):
        nan=struct.unpack('>d',bytes.fromhex('7ff8000000000042'))[0]
        for index,fields in enumerate((dict(wall_s=nan),dict(wall_s=True),dict(wall_s=1,wall_seconds=2))):
            _,_,ledger=self.qualification('invalid-%d'%index);self.reject(ledger,**fields)
            payload=ledger.events[-1]['payload'];self.assertEqual(payload['observation_status'],'EXACT_TYPED')
            observed=r.decode_fields(payload['observed_fields'],r.default_policy())
            if index==0:
                self.assertTrue(math.isnan(observed['wall_s']))
                self.assertEqual(struct.pack('>d',observed['wall_s']).hex(),'7ff8000000000042')
            else:self.assertEqual(observed['wall_s'],fields['wall_s'])
    def test_role_mismatch_retained_before_normal_clone_and_forced_unknown_bound(self):
        _,_,ledger=self.qualification('role')
        with self.assertRaises(ValueError):ledger.operation('input_io','returned','leaf',wall_s=1)
        self.assertIn('mismatched:wall_contract',ledger.rejection_record['rejection']['codes'])
        _,_,ledger=self.qualification('large')
        self.reject(ledger,wall_s=61,extra='x'*32769)
        self.assertEqual(ledger.rejection_record['observation_status'],'UNREPRESENTABLE_UNKNOWN')
        self.assertIsNone(ledger.rejection_record['observed_fields']);self.assertTrue(ledger.fatal)
        self.assertLess(ledger.path.stat().st_size,65536+30000)
    def test_aggregate_encoder_budget_and_policy_changes_denied(self):
        for fields in ({'x':['a'*1000]*100},{'x':[0]*4097},{'x':2**129}):
            with self.subTest(fields_type=type(fields['x']).__name__),self.assertRaises(ValueError):r.encode_fields(fields,r.default_policy())
        policy=r.default_policy();policy['record_bytes_upper']+=1
        with self.assertRaises(ValueError):r.validate_policy(policy)
    def test_rejection_single_use_short_write_unknown_and_precharge_denial(self):
        _,_,ledger=self.qualification();self.reject(ledger,wall_s=61)
        size=ledger.path.stat().st_size
        with self.assertRaises(ValueError):ledger.operation('input_io','returned','leaf',wall_contract=c.FILE_HASH_LEAF,wall_s=61)
        self.assertEqual(ledger.path.stat().st_size,size)
        _,_,ledger=self.qualification('writefail');paid=copy.deepcopy(ledger.cumulative)
        class ShortWrite(io.BytesIO):
            def write(self,data):return super().write(data[:-1])
        with patch.object(Path,'open',return_value=ShortWrite()):self.reject(ledger,wall_s=61)
        self.assertTrue(ledger.fatal);self.assertTrue(ledger.rejection_used)
        self.assertIsNotNone(ledger.rejection_unknown);self.assertIsNone(ledger.rejection_record)
        self.assertEqual(ledger.cumulative,paid)
        context,proof,receipt=q_context(self.root/'capdeny')
        spec=context['bindings']['scope']['qualifier_accounting'];spec['rejected_return_policy']=r.default_policy()
        spec['global_caps']['metadata_bytes_upper']=65535
        with self.assertRaises(ValueError):q.QualificationAccounting(context,None,proof,receipt,clock=lambda:100)
    def test_rehashed_actual_or_start_link_tampering_denies_independent_replay(self):
        context,receipt,ledger=self.qualification();self.reject(ledger,wall_s=61)
        for change in (lambda p:p.update(started_sha256='0'*64),lambda p:p.update(selected_profile_sha256='0'*64),
            lambda p:p.update(ticket_sha256='0'*64),lambda p:p['rejection'].update(codes=['exceeded:bytes_read'])):
            events=copy.deepcopy(ledger.events);change(events[-1]['payload'])
            state=q.replay_state(context,ledger.header,receipt)
            for row in events[:-1]:state._apply(row['kind'],row['payload'],replay=True);state.events.append(row)
            with self.subTest(change=change),self.assertRaises(ValueError):state._apply(events[-1]['kind'],events[-1]['payload'],replay=True)
    def test_legacy_without_policy_has_no_ticket_or_failure_event(self):
        context,proof,receipt=q_context(self.root/'legacy')
        ledger=q.QualificationAccounting(context,None,proof,receipt,clock=lambda:100)
        ledger.begin_stage('runtime_anchor','runtime');ledger.operation('input_io','started','leaf',wall_contract=c.FILE_HASH_LEAF)
        size=ledger.path.stat().st_size;self.reject(ledger,wall_s=61)
        self.assertEqual(ledger.path.stat().st_size,size);self.assertNotIn('rejection_ticket',ledger.snapshot())


if __name__=='__main__':unittest.main()
