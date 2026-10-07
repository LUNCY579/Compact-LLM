"""Independent saved closed packet/source graph joins, no launch factory."""
from pathlib import Path
from copy import deepcopy
from types import ModuleType
import hashlib,json,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
def load(p,n):m=ModuleType(n);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);M=load(WORK/'closed_registration.py','independent_closed_constructor');I=load(WORK/'program002_inputs.py','independent_closed_inputs');C=I.module(I.PROGRAM/'import_observation_contract.py','independent_actual_program_contract');INPUTS=I.inputs();FILES=[p for p in WORK.rglob('*') if p.is_file() and 'independent_review' not in p.parts]+[Path(p) for p in INPUTS['source_files']]+[Path(r['path']) for r in INPUTS['metadata_references'].values()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();DETAILS={}
class Controls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='independent_constructor002_');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def build(self):return M.build(**deepcopy(INPUTS),implementation_commit=None,dispatch_head=None,packet_paths={n:str(self.root/(n+'.json')) for n in ('config','runtime','registration')})
    def validate(self,b):C.validate_import_metadata(b['config'],b['runtime'],b['registration'],b['external_dispatch_head']);return M.validate(b)
    def test_saved_packet_refs_actual_validator_and_scope_counts(self):
        receipt=json.loads((WORK/'CLOSED_PROGRAM002_PACKET_RECEIPT001.json').read_bytes());self.assertEqual(receipt['sha256'],M.digest({k:v for k,v in receipt.items() if k!='sha256'}));packets={}
        for name,ref in receipt['packet_references'].items():
            p=Path(ref['path']);raw=p.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['file_sha256']);v=json.loads(raw);self.assertEqual(v['sha256'],ref['body_sha256']);self.assertEqual(v['sha256'],M.digest({k:x for k,x in v.items() if k!='sha256'}));self.assertEqual(raw,(M.canonical(v)+'\n').encode());packets[name]=v
        C.validate_import_metadata(packets['config'],packets['runtime'],packets['registration'],None);bundle={'config':packets['config'],'runtime':packets['runtime'],'registration':packets['registration'],'external_dispatch_head':None,'packet_paths':{n:r['path'] for n,r in receipt['packet_references'].items()},'native_launch_authorized':False,'operation_registration':receipt['operation_registration'],'selected_inputs':receipt['external_selection']};M.validate(bundle)
        self.assertEqual((len(packets['config']['source_files']),len(packets['runtime']['files']),len(packets['runtime']['rosters']),len(packets['runtime']['cache_policy']['preloaded_module_cache_pins']),len(packets['runtime']['cache_policy']['preloaded_bootstrap_modules']),len(packets['runtime']['cache_policy']['preloaded_class_aliases'])),(11,8716,18,59,63,2));self.assertEqual(C.OPERATIONS,M.OPERATIONS);self.assertEqual(receipt['operation_registration'],{'operations':list(C.OPERATIONS),'maximum_markers':128,'whole_scope_seconds':600});self.assertTrue(all(receipt[k] is None for k in ('implementation_commit','dispatch_head','nine_external_dispatch_anchors','owner_approval')));self.assertFalse(receipt['native_binding_main_implemented']);DETAILS['saved_packet_actual_validator_calls']=1
    def test_actual_resealed_source_main_and_permission_refuse(self):
        for kind in ('main','permission','runtimequalified'):
            b=self.build()
            if kind=='main':b['config']['main_source']=str(self.root/'other.py')
            elif kind=='permission':b['config']['permission']['tensor_allocations']=2
            else:b['runtime']['native_runtime_qualified']=True;b['runtime']=M.seal({k:v for k,v in b['runtime'].items() if k!='sha256'});b['registration']['runtime_body_sha256']=b['runtime']['sha256']
            b['config']=M.seal({k:v for k,v in b['config'].items() if k!='sha256'});b['registration']['config_body_sha256']=b['config']['sha256'];b['registration']=M.seal({k:v for k,v in b['registration'].items() if k!='sha256'})
            with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):self.validate(b)
    def test_external_selected_digest_blocks_coordinated_cache_reseal(self):
        b=self.build();selected=deepcopy(b['selected_inputs']);b['runtime']['cache_policy']['preloaded_bootstrap_modules'].pop(next(iter(b['runtime']['cache_policy']['preloaded_bootstrap_modules'])));b['runtime']=M.seal({k:v for k,v in b['runtime'].items() if k!='sha256'});b['registration']['runtime_body_sha256']=b['runtime']['sha256'];b['registration']=M.seal({k:v for k,v in b['registration'].items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'external selected'):M.validate(b,expected_selection=selected)
        inputs=deepcopy(INPUTS);path=next(iter(inputs['observation']['maps']['preloaded_module_cache_pins']));inputs['runtime_candidate']['files'].pop(path);inputs['runtime_candidate']=M.seal({k:v for k,v in inputs['runtime_candidate'].items() if k!='sha256'})
        with self.assertRaises(ValueError):M.build(**inputs,implementation_commit=None,dispatch_head=None,packet_paths={n:str(self.root/(n+'.json')) for n in ('config','runtime','registration')})
    def test_pending_anchors_bytes_cycle_and_native_context_refuse(self):
        b=self.build();self.validate(b);raw={n:(M.canonical(b[n])+'\n').encode() for n in ('config','runtime','registration')}
        with self.assertRaisesRegex(ValueError,'pending'):M.anchors(b,raw)
        b['external_dispatch_head']='a'*40
        with self.assertRaises(ValueError):self.validate(b)
        with self.assertRaises(ValueError):M.native_context(b)
        with self.assertRaises(ValueError):C.stage_native_gate()
        self.assertEqual(b['config']['resources'],C.RESOURCES);self.assertIs(b['config']['native_launch_authorized'],False);self.assertIsNone(b['config']['owner_approval'])
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));sys.meta_path.remove(trap);after=pins();receipt={'kind':'INDEPENDENT_CLOSED_CONSTRUCTOR_PROGRAM002_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_packet_dependency_pins_before':BEFORE,'source_packet_dependency_pins_after':after,'pins_unchanged':BEFORE==after,'details':DETAILS,'actual_program002_validate_import_metadata_called':True,'saved_three_packet_refs_and_exact_bytes_verified':True,'source_count':11,'runtime_count':8716,'rosters':18,'observed_cache_pins':59,'ordinary_modules':63,'aliases':2,'operations':11,'maximum_markers':128,'seconds':600,'actual_implementation_dispatch_owner_anchors_absent':True,'native_binder_or_runtime_qualified':False,'native_API_or_framework_called':False,'native_authority_issued':False};receipt['sha256']=M.digest(receipt)
    with (HERE/'INDEPENDENT_CLOSED_CONSTRUCTOR002.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
