"""Closed eleven-leaf source promotion graph; public-shaped mirror is CPU only."""
from pathlib import Path
import ast,hashlib,json,shutil,tempfile,types,unittest,sys
HERE=Path(__file__).resolve().parent;LOCATION=HERE/'location_candidate';PLAN=HERE/'PROMOTION_SOURCE_PLAN001.json'
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
REQUIRED={'prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_native_pipeline.py','prefix_registered_loader.py','prefix_input_owned_auditor.py','prefix_input_owned_supervisor.py','registration_git_join.py','typing_alias_checks.py','prefix_library_observer.py','prefix_library_policy.py','prefix_library_evidence.py'}
def validate_plan(plan):
    if type(plan) is not dict or plan.get('sha256')!=hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k!='sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest():raise ValueError('exact plan seal')
    if set(plan['sources'])!=REQUIRED or any(plan[k] is not False for k in ('promotion_performed','native_authority_issued','native_launch_authorized','future_gate_successor_included')):raise ValueError('exact closed eleven-source roster')
    for name,row in plan['sources'].items():
        path=LOCATION/name;raw=path.read_bytes()
        if row['staged_path']!=str(path) or Path(row['promotion_target']).name!=name or row['descriptor']!={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}:raise ValueError('actual staged source descriptor and target join')
    return {'native_authority_issued':False,'source_count':11}
class NativeTrap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker','native_process','pynvml'):raise AssertionError('native import forbidden '+fullname)
class Controls(unittest.TestCase):
    def setUp(self):self.plan=json.loads(PLAN.read_text());self.temp=tempfile.TemporaryDirectory(prefix='CPU_promotion_');self.pkg=Path(self.temp.name)/'experiments/rineya_synth_centralized_reasoning_v1';self.pkg.mkdir(parents=True);self.trap=NativeTrap();sys.meta_path.insert(0,self.trap)
    def tearDown(self):sys.meta_path.remove(self.trap);self.temp.cleanup()
    def test_all_eleven_sources_compile_and_exact_plan_pins(self):
        self.assertEqual(validate_plan(self.plan)['source_count'],11)
        for n,d in self.plan['sources'].items():
            raw=(LOCATION/n).read_bytes();self.assertEqual(len(raw),d['descriptor']['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),d['descriptor']['sha256']);compile(raw,str(LOCATION/n),'exec')
        self.assertFalse(self.plan['promotion_performed']);self.assertFalse(self.plan['native_launch_authorized']);self.assertFalse(self.plan['future_gate_successor_included'])
    def test_public_shaped_C_W_P_SUP_A_link_and_all_gates_stay_closed(self):
        for p in LOCATION.glob('*.py'):shutil.copy2(p,self.pkg/p.name)
        C=load(self.pkg/'prefix_tensor_contract.py','CPU_public_shape_C');W=load(self.pkg/'prefix_input_owned_worker.py','CPU_public_shape_W');P=load(self.pkg/'prefix_native_pipeline.py','CPU_public_shape_P');S=load(self.pkg/'prefix_input_owned_supervisor.py','CPU_public_shape_SUP');A=load(self.pkg/'prefix_input_owned_auditor.py','CPU_public_shape_A')
        self.assertEqual(C.PACKAGE,self.pkg);self.assertEqual(W.C.PACKAGE,self.pkg);self.assertEqual(S.C.PACKAGE,self.pkg);self.assertEqual(A.PACKAGE,self.pkg);self.assertEqual(W.CONTRACT_PIN,hashlib.sha256((self.pkg/'prefix_tensor_contract.py').read_bytes()).hexdigest());self.assertEqual(W.ENTRY_PIPELINE_PIN,hashlib.sha256((self.pkg/'prefix_native_pipeline.py').read_bytes()).hexdigest());self.assertEqual(P.CONTRACT_BOOTSTRAP['sha256'],W.CONTRACT_PIN)
        with self.assertRaises(ValueError):C.stage_native_gate()
        with self.assertRaises(ValueError):W.C.stage_native_gate()
        with self.assertRaises(ValueError):P.C.stage_native_gate()
        with self.assertRaises(ValueError):S.C.stage_native_gate()
    def test_exact_location_delta_and_durable_predecessor_reference(self):
        C=load(LOCATION/'prefix_tensor_contract.py','CPU_location_C');source=(LOCATION/'prefix_tensor_contract.py').read_text();self.assertIn('evidence/site_disabled_tensor_packet_reviewed_cpu_20261007/predecessor_sources',source)
        predecessor=HERE/'predecessors/prefix_tensor_contract.py';before=predecessor.read_text();expected=before.replace("predecessor=read_bytes(HERE/'predecessor_sources'/name)","predecessor_root=HERE/'predecessor_sources' if HERE.parent.name=='artifacts' else PACKAGE/'evidence/site_disabled_tensor_packet_reviewed_cpu_20261007/predecessor_sources';predecessor=read_bytes(predecessor_root/name)")
        self.assertEqual(source,expected)
        for n in ('prefix_registered_loader.py','prefix_input_owned_auditor.py','prefix_input_owned_supervisor.py','registration_git_join.py','typing_alias_checks.py','prefix_library_observer.py','prefix_library_policy.py','prefix_library_evidence.py'):self.assertEqual((LOCATION/n).read_bytes(),(HERE/'predecessors'/n).read_bytes())
        root=C.PACKAGE/'evidence/site_disabled_tensor_packet_reviewed_cpu_20261007/predecessor_sources';self.assertEqual(len(list(root.glob('*.py'))),6)
    def test_pin_rebinding_and_missing_helper_detected_without_public_mutation(self):
        for p in LOCATION.glob('*.py'):shutil.copy2(p,self.pkg/p.name)
        p=self.pkg/'prefix_tensor_contract.py';p.write_bytes(p.read_bytes()+b'\n# drift\n')
        with self.assertRaisesRegex(ValueError,'exact stable source'):load(self.pkg/'prefix_input_owned_worker.py','CPU_bad_pin_W')
        # Direct byte-map validation rejects an omitted required helper before
        # any registration or native permit could be constructed.
        bad=json.loads(json.dumps(self.plan));bad['sources'].pop('typing_alias_checks.py');bad['sha256']=hashlib.sha256(json.dumps({k:v for k,v in bad.items() if k!='sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError,'eleven-source'):validate_plan(bad)
        public=Path(self.plan['sources']['prefix_tensor_contract.py']['promotion_target']);self.assertEqual(hashlib.sha256(public.read_bytes()).hexdigest(),self.plan['unchanged_public_shared_contract_sha256'])
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_CLOSED_PUBLIC_LOCATION_SOURCE_PROMOTION_CONTROLS','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'public_shaped_mirror_only':True,'public_source_promotion_performed':False,'gate_successor_included':False,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),PLAN,*LOCATION.glob('*.py')]}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
