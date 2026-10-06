"""Inert native-ABI transport tests; no DLL loading or Windows registration."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,unittest
import prefix_library_observer as O

class Transport:
    def __init__(self):
        self.started=False;self.closed=False;self.rows=[];self.faults=0;self.calls=[];self.change=None
        self.binding={'path':'C:\\CPU_ONLY\\native_library_observer.dll','descriptor':{'bytes':1,'sha256':'a'*64,'role':'observer_binary'},'abi':dict(O.ABI),'dll_handle':101}
    def abi_binding(self):return deepcopy(self.binding)
    def start(self):self.calls.append('start');self.started=True;return 0
    def snapshot(self):
        self.calls.append('snapshot');n=len(self.rows)
        row={'abi_version':1,'row_bytes':2600,'snapshot_bytes':72,'row_capacity':4096,'full_capacity':1024,'base_capacity':256,'phase':5 if self.closed else 2,'faults':self.faults,'attempted':n,'reserved':n,'published_prefix':n,'inflight':0,'cookie_live':0 if self.closed else 1,'register_status':0,'unregister_status':0 if self.closed else O.STATUS_UNSET,'drained':int(self.closed),'complete':int(self.closed),'dropped':0}
        if self.change:row.update(self.change)
        return row
    def copy_rows(self,count):self.calls.append('copy');return deepcopy(self.rows)
    def stop(self):self.calls.append('stop');self.closed=True;return 0,self.snapshot()
    def self_identity(self):return {'pid':101,'creation_filetime_100ns':12345678}
    def verify_preloaded_bootstrap(self,value):self.calls.append('bootstrap_sources')
    def verify_owned_birth(self,value,owner,birth):self.calls.append('owned_birth')
    def add(self,path='C:\\CPU_ONLY\\dep.dll',reason=1):
        def units(text):raw=text.encode('utf-16-le');return [int.from_bytes(raw[i:i+2],'little') for i in range(0,len(raw),2)]
        full=units(path);base=units(Path(path).name)
        self.rows.append({'sequence':len(self.rows)+1,'base_address':1000+len(self.rows),'reason':reason,'image_size':4096,'full_units':len(full),'base_units':len(base),'flags':0,'ready':1,'full_name':full,'base_name':base})

class NativeObserverAdapterControls(unittest.TestCase):
    def setUp(self):
        self.t=Transport();self.value={'fixture_only':True,'native_PASS':False,'native_authority_issued':False,
            'runtime':{'library_event_observer':{'abi':dict(O.ABI),'binary':{'path':self.t.binding['path'],'descriptor':self.t.binding['descriptor']}},
            'files':{'C:\\CPU_ONLY\\dep.dll':{'bytes':1,'sha256':'b'*64,'role':'extension_binary'}},
            'stdlib_layout':{'logical_base_prefix':'C:\\CPU_ALIAS','physical_base_prefix':'C:\\CPU_ONLY'}}}
        self.C=SimpleNamespace(context_value=lambda _:deepcopy(self.value),parse=json.loads,canonical=lambda value:json.dumps(value,sort_keys=True),stage_native_gate=lambda:(_ for _ in ()).throw(ValueError('native stage closed')))
        self.context=object();self.o=O._CPU_observer_fixture(self.context,self.C,self.t,_capability=O._CPU_capability())
    def test_actual_shared_prefix_decode_and_final_drain(self):
        self.o.begin();self.t.add();events=self.o.events();self.assertEqual(events['events'][0]['path'],'C:\\CPU_ONLY\\dep.dll')
        self.assertEqual(self.o.close(),{'kind':'OBSERVED_PROCESS_LIBRARY_EVENT_CLEANUP','complete':True,'closed':True})
        self.assertEqual(self.o.events(),events);self.assertTrue(self.o.raw_rows)
        self.assertFalse(self.o.native_PASS)
    def test_overflow_dropped_or_inflight_cannot_claim_complete(self):
        for fault in ({'faults':1},{'dropped':1},{'inflight':1},{'attempted':1}):
            with self.subTest(fault=fault):
                self.t.change=fault
                with self.assertRaisesRegex(ValueError,'lossless'):self.o.begin()
                self.t.change=None;self.o.started=False
    def test_utf16_truncation_nul_invalid_surrogate_and_base_mismatch_refused(self):
        self.o.begin();self.t.add();original=deepcopy(self.t.rows)
        for change in ({'full_units':1024},{'full_name':[0]*original[0]['full_units']},{'full_name':[0xD800]*original[0]['full_units']},{'base_name':[ord('x')]*original[0]['base_units']}):
            self.t.rows=deepcopy(original);self.t.rows[0].update(change)
            with self.assertRaises((ValueError,UnicodeError)):self.o.events()
    def test_prefix_rewrite_and_unregistered_path_refused(self):
        self.o.begin();self.t.add();self.o.events();self.t.rows[0]['base_address']+=1
        with self.assertRaisesRegex(ValueError,'append-only'):self.o.events()
        self.t.rows=[];self.t.add('C:\\NOT_SELECTED\\dep.dll')
        with self.assertRaisesRegex(ValueError,'qualified observed'):self.o.events()
    def test_abi_handle_and_context_substitution_refused_before_start(self):
        self.t.binding['dll_handle']=102
        with self.assertRaisesRegex(ValueError,'handle/path'):self.o.begin()
        self.t.binding['dll_handle']=101;self.value['native_PASS']=True
        with self.assertRaisesRegex(ValueError,'cannot be rebased'):self.o.begin()
        self.assertNotIn('start',self.t.calls)
    def test_copied_rows_must_equal_stable_snapshot_population(self):
        self.o.begin();self.t.add()
        with patch.object(self.t,'copy_rows',return_value=[]):
            with self.assertRaisesRegex(ValueError,'copied event population'):self.o.events()
    def test_actual_raw_names_case_and_one_known_alias_are_preserved(self):
        self.o.begin();self.t.add('c:\\cpu_alias\\dep.dll')
        events=self.o.events();self.assertEqual(events['events'][0]['path'],'C:\\CPU_ONLY\\dep.dll')
        self.assertEqual(self.o.raw_rows[0]['full_name'],self.t.rows[0]['full_name'])
    def test_bootstrap_process_birth_drift_refused(self):
        self.o.bootstrap('CPU_stage')
        with patch.object(self.t,'self_identity',return_value={'pid':102,'creation_filetime_100ns':12345678}):
            with self.assertRaisesRegex(ValueError,'same current observer process'):self.o.bootstrap('CPU_stage')
    def test_public_native_factory_stage_refuses_before_transport(self):
        with patch.object(O,'_Win32Transport',side_effect=AssertionError('must not load DLL')) as native:
            with self.assertRaisesRegex(ValueError,'native stage closed'):O.make_native_observer(self.context,self.C,{})
            native.assert_not_called()
    def test_cpu_capability_and_non_cpu_context_refused(self):
        with self.assertRaisesRegex(ValueError,'private CPU'):O._CPU_observer_fixture(self.context,self.C,self.t,_capability=object())
        self.value['fixture_only']=False
        with self.assertRaisesRegex(ValueError,'explicit CPU'):O._CPU_observer_fixture(self.context,self.C,self.t,_capability=O._CPU_capability())
    def test_cleanup_unregister_failure_prevents_claim(self):
        self.o.begin()
        with patch.object(self.t,'stop',return_value=(-4,self.t.snapshot())):
            with self.assertRaisesRegex(ValueError,'unregister and drain'):self.o.close()
        self.assertTrue(self.o.failed)
    def test_context_failure_still_unregisters_and_preserves_primary(self):
        self.o.begin();self.value['native_PASS']=True
        with self.assertRaisesRegex(ValueError,'cannot be rebased'):self.o.close()
        self.assertIn('stop',self.t.calls);self.assertTrue(self.t.closed);self.assertTrue(self.o.failed)
    def test_validation_and_unregister_failure_keep_original_error_cause(self):
        self.o.begin();self.value['native_PASS']=True
        with patch.object(self.t,'stop',return_value=(-4,self.t.snapshot())):
            with self.assertRaisesRegex(ValueError,'cannot be rebased') as caught:self.o.close()
        self.assertIsInstance(caught.exception.__cause__,ValueError)
        self.assertIn('unregister and drain',str(caught.exception.__cause__))
    def test_snapshot_numeric_boolean_impostor_refused(self):
        self.t.change={'cookie_live':True}
        with self.assertRaisesRegex(ValueError,'fixed width'):self.o.begin()

if __name__=='__main__':unittest.main()
