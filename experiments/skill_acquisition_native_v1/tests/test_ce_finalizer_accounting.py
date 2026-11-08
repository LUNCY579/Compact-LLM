"""Pure prepaid finalizer IO/clock controls; no OS/framework/native grants."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_finalizer_accounting as f


class Clock:
    def __init__(self):self.value=100.
    def __call__(self):return self.value
    def advance(self,n):self.value+=n


def spec(root):
    gib=1024**3;mib=1024**2
    full=dict(wall_seconds_upper=7200,operation_attempts_upper=100000,io_read_bytes_upper=256*gib,
        io_write_bytes_upper=gib,identity_hash_attempts_upper=10000000,identity_hash_bytes_upper=2048*gib,
        metadata_bytes_upper=gib,reducer_work_bytes_upper=4096*gib)
    suffix=dict(wall_seconds_upper=120,operation_attempts_upper=1000,io_read_bytes_upper=128*mib,
        io_write_bytes_upper=4*mib,identity_hash_attempts_upper=32,identity_hash_bytes_upper=256*mib,
        metadata_bytes_upper=4*mib,reducer_work_bytes_upper=gib)
    return {'schema':f.SPEC,'owned_directory':str(root/'finalizer'),'max_receipt_bytes':4*mib,
        'physical_wall_seconds_upper':7200,'self_bounds':full,'finalization_bounds':suffix,
        'reader_bounds':{key:full[key] for key in f.raw.BUDGET_FIELDS}}


class Tests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.clock=Clock();self.spec=spec(self.root);self.owner=f.fixture_owner(self.spec,self.root,clock=self.clock)
    def test_native_constructor_and_fixture_os_observation_refuse_before_work(self):
        class Trap:
            def __getattribute__(self,name):raise AssertionError('before opaque source metadata permit')
        with patch.object(Path,'open',side_effect=AssertionError('before opaque source metadata permit')):
            with self.assertRaises(ValueError):f.FinalizerCostOwner(Trap())
            with self.assertRaises(ValueError):self.owner.observe_parent_terminal(Trap(),Trap())
        self.assertFalse(self.owner.native);self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
    def test_full_permanent_reservation_precedes_reads_and_suffix_creates_separate_receipt_cost_refs(self):
        first=f.raw._json(self.owner.path.read_bytes().splitlines()[0]);self.assertEqual(first['kind'],'permanent_self_reservation')
        self.assertEqual(first['payload']['bounds'],self.spec['self_bounds'])
        reservation=copy.deepcopy(self.spec['self_bounds']);stable=self.owner.reader.operation
        self.clock.advance(2)
        ref=self.owner.write_json('QUALIFICATION_RECEIPT.json',{'passed':False,'native_qualified':False,'reservation_ref':self.owner.header_ref})
        evidence=self.owner.close_and_seal();self.assertTrue(evidence['owner_closed'])
        self.assertEqual(evidence['permanent_reserved_bounds'],reservation)
        self.assertIs(self.owner.reader.operation,stable)
        for reference in (ref,evidence['header_ref'],evidence['journal_ref'],evidence['seal_ref']):
            blob=Path(reference['path']).read_bytes();self.assertEqual(hashlib.sha256(blob).hexdigest(),reference['sha256'])
        receipt=f.raw._json(Path(ref['path']).read_bytes());self.assertNotIn('finalizer_seal_ref',receipt)
        rows=[f.raw._json(line) for line in self.owner.path.read_bytes().splitlines()]
        previous=None
        for index,row in enumerate(rows):
            self.assertEqual(row['seq'],index);self.assertEqual(row['previous'],previous)
            self.assertEqual(row['sha256'],f.digest({key:value for key,value in row.items() if key!='sha256'}));previous=row['sha256']
        self.assertEqual(rows[-1]['kind'],'end')
        self.assertIs(self.owner.require_closed_evidence(evidence),evidence)
        with self.assertRaises(ValueError):self.owner.require_closed_evidence(copy.deepcopy(evidence))
        with self.assertRaises(ValueError):self.owner.write_json('LATE.json',{})
    def test_pending_or_failed_attempt_keeps_bounds_and_cannot_seal_success(self):
        self.owner.operation('finalizer_metadata','started','pending',reducer_work_bytes_upper=19)
        before=copy.deepcopy(self.owner.counts)
        with self.assertRaisesRegex(ValueError,'operations terminal'):self.owner.close_and_seal()
        self.assertEqual(before,self.owner.counts)
        failure=self.owner.fail(RuntimeError('source read failed'))
        self.assertEqual(failure['pending'],['pending']);self.assertEqual(failure['permanent_reserved_bounds'],self.spec['self_bounds'])
        self.assertTrue(failure['fatal'])
    def test_declared_upper_overrun_is_fatal_without_refund(self):
        self.owner.operation('qualification_artifact_readback','started','read',io_read_bytes_upper=17,
            identity_hash_attempts_upper=1,identity_hash_bytes_upper=17)
        before=dict(self.owner.counts)
        with self.assertRaisesRegex(ValueError,'exceeded permanent'):
            self.owner.operation('qualification_artifact_readback','returned','read',bytes_read=18)
        self.assertTrue(self.owner.fatal);self.assertEqual(before,self.owner.counts)
        self.assertEqual(self.owner.snapshot()['pending'],['read'])
    def test_short_receipt_write_remains_failed_and_full_bounds_survive(self):
        original=Path.open
        class Short:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def write(self,blob):return len(blob)-1
        def opened(path,mode='r',*args,**kwargs):
            return Short() if path.name=='QUALIFICATION_RECEIPT.json' and mode=='xb' else original(path,mode,*args,**kwargs)
        with patch.object(Path,'open',opened):
            with self.assertRaisesRegex(ValueError,'short finalizer JSON'):self.owner.write_json('QUALIFICATION_RECEIPT.json',{})
        self.assertTrue(self.owner.snapshot()['failed']);self.assertEqual(self.owner.snapshot()['permanent_reserved_bounds'],self.spec['self_bounds'])
    def test_receipt_and_owner_files_are_create_only_and_spec_cannot_mutate(self):
        self.owner.write_json('EVIDENCE.json',{})
        with self.assertRaisesRegex(ValueError,'create-only'):self.owner.write_json('EVIDENCE.json',{})
        with self.assertRaisesRegex(ValueError,'create-only'):f.fixture_owner(self.spec,self.root,clock=self.clock)
        self.owner.spec['self_bounds']['io_read_bytes_upper']+=1
        with self.assertRaisesRegex(ValueError,'immutable'):self.owner.tick()
    def test_source_suffix_requires_opaque_same_owner_and_actual_observation_order(self):
        proof={'request_sha256':'a'*64,'scope_sha256':'b'*64,'exit_code':0}
        obs=f.TerminalObservation(self.owner,None,proof,109.,110.,seal=f._OBSERVATION_SEAL)
        self.owner._terminal_observation=obs
        replay={'derived_snapshot':{'request_sha256':'a'*64,'clock_ended':True,'fatal':False,'pending':[],
            'last_monotonic':100.,'end_reservation':{'bounds':{'wall_s_upper':120}}}}
        result=self.owner.validate_service_suffix(replay,obs)
        self.assertEqual(result['observed_suffix_seconds_upper'],10);self.assertFalse(result['native_observation'])
        with self.assertRaises(ValueError):self.owner.validate_service_suffix(replay,{'monotonic':110,'native':True})
        changed=copy.deepcopy(replay);changed['derived_snapshot']['last_monotonic']=111.
        with self.assertRaises(ValueError):self.owner.validate_service_suffix(changed,obs)
        changed=copy.deepcopy(replay);changed['derived_snapshot']['end_reservation']['bounds']['wall_s_upper']=9
        with self.assertRaises(ValueError):self.owner.validate_service_suffix(changed,obs)
        with self.assertRaises(ValueError):obs._after=100
    def test_original_and_suffix_deadline_cannot_restart(self):
        self.owner.begin_finalization();self.clock.advance(121)
        with self.assertRaisesRegex(ValueError,'suffix deadline'):self.owner.tick()
        self.assertEqual(self.owner._suffix_origin,100.)

    def test_mutated_closed_result_or_caller_closed_flag_cannot_promote_evidence(self):
        with self.assertRaises(ValueError):self.owner.require_closed_evidence({'owner_closed':True})
        result=self.owner.close_and_seal();result['permanent_reserved_bounds']['io_read_bytes_upper']-=1
        with self.assertRaisesRegex(ValueError,'exact successful'):self.owner.require_closed_evidence(result)

    def test_registered_delta_clock_refuses_late_reader_callback_and_retains_pending_full_bounds(self):
        self.owner.preflight('saved_delta_source_rebuild',{'wall_seconds_upper':3600})
        self.owner.operation('qualification_reducer_overhead','started','delta',reason='delta_readback_source_report_linkage')
        self.clock.advance(3601)
        before=dict(self.owner.counts)
        with self.assertRaisesRegex(ValueError,'delta rebuild wall ceiling'):
            self.owner.operation('qualification_artifact_readback','started','late',io_read_bytes_upper=17)
        self.assertEqual(self.owner.counts,before);self.assertNotIn('late',self.owner.operations)
        with self.assertRaisesRegex(ValueError,'delta rebuild wall ceiling'):
            self.owner.operation('qualification_reducer_overhead','failed','delta',reason='delta_readback_source_report_linkage')
        self.assertIsNone(self.owner._delta_rebuild_started)
        self.assertEqual(self.owner.snapshot()['pending'],['delta'])
        self.assertEqual(self.owner.snapshot()['permanent_reserved_bounds'],self.spec['self_bounds'])

    def test_successful_delta_scope_clears_before_later_owner_work(self):
        with self.assertRaisesRegex(ValueError,'explicit registered'):
            self.owner.preflight('saved_delta_source_rebuild',{'wall_seconds_upper':7200})
        self.owner.preflight('saved_delta_source_rebuild',{'wall_seconds_upper':3600})
        self.owner.operation('qualification_reducer_overhead','started','delta',reason='delta_readback_source_report_linkage')
        self.clock.advance(3599)
        self.owner.operation('qualification_reducer_overhead','returned','delta',reason='delta_readback_source_report_linkage')
        self.assertIsNone(self.owner._delta_rebuild_started)
        self.clock.advance(2)
        self.owner.operation('finalizer_metadata','started','later')
        self.owner.operation('finalizer_metadata','returned','later')
        self.assertEqual(self.owner.last_clock,3701.)


if __name__=='__main__':unittest.main()
