"""Exact closed public proposal joins and material resealing refusals."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import ast,hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_closed_public_proposal002';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');FILES=[p for p in FROZEN.rglob('*') if p.is_file()]
def load(p,n):m=ModuleType(n);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
M=load(FROZEN/'construct_proposal.py','independent_closed_public_M');P=json.loads((FROZEN/'PUBLIC_IMPORT_CLOSED_PROPOSAL002.json').read_bytes());PLAN=PACKAGE/'artifacts/torch_import_observation_work/native_candidate/PUBLIC_IMPORT_SOURCE_PLAN002.json';plan=json.loads(PLAN.read_bytes());FILES.append(PLAN);FILES.extend(Path(p) for p in P['config']['source_files']);Q=PACKAGE/'import_runtime_qualification.py';FILES.append(Q)
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'construct_proposal.py')]['sha256']=='9d36bdc8805df34c009aa87c743b5d5403913a0bed30110453818fa963ddeca0';assert BEFORE[str(FROZEN/'test_proposal.py')]['sha256']=='f69a6b01074f160217723c1f607ea631c17c7b0b8aad8613c7d186aae1916429'
SOURCES=M.digest(P['config']['source_files']);RUNTIME=M.digest(P['runtime']['files']);POLICY=P['runtime_policy_sha256'];DETAILS={}
def check(p):return M.validate(p,source_plan=plan,expected_source_map_sha256=SOURCES,expected_runtime_map_sha256=RUNTIME,expected_runtime_policy_sha256=POLICY)
def reseal(p):
    for n in ('config','runtime'):p[n]=M.seal({k:v for k,v in p[n].items() if k!='sha256'})
    p['registration'].update(config_body_sha256=p['config']['sha256'],runtime_body_sha256=p['runtime']['sha256']);p['registration']=M.seal({k:v for k,v in p['registration'].items() if k!='sha256'});return M.seal({k:v for k,v in p.items() if k!='sha256'})
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap)
class Controls(unittest.TestCase):
    def test_saved_exact_nested_bytes_source_maps_and_actual_Q_fieldsets(self):
        result=check(P);self.assertFalse(result['native_context_minted'])
        for n in ('config','runtime','registration'):
            raw=(FROZEN/('IMPORT_'+n.upper()+'_CLOSED002.json')).read_bytes();self.assertEqual(raw,(M.canonical(P[n])+'\n').encode())
        for p,d in P['config']['source_files'].items():self.assertEqual(BEFORE[p],{k:d[k] for k in ('bytes','sha256')})
        fields={}
        for n in ast.parse(Q.read_bytes()).body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('CONFIG_FIELDS','RUNTIME_FIELDS','REGISTRATION_FIELDS'):fields[n.targets[0].id]=ast.literal_eval(n.value)
        for n in ('config','runtime','registration'):self.assertEqual(set(P[n]),fields[n.upper()+'_FIELDS'])
        self.assertEqual((len(P['config']['source_files']),len(P['runtime']['files']),len(P['runtime']['rosters']),len(P['runtime']['measurement_policy']['initial_loaded_libraries'])),(194,8724,18,39));self.assertEqual(len(P['runtime']['cache_policy']['preloaded_bootstrap_modules']),63);self.assertEqual(len(P['runtime']['cache_policy']['preloaded_module_cache_pins']),59);self.assertEqual(len(P['runtime']['cache_policy']['additional_preload_sources']),10);self.assertEqual(len(P['runtime']['cache_policy']['preloaded_class_aliases']),2);DETAILS['saved_source_count']=194
    def test_source_runtime_and_additive_reference_rebind_refuse(self):
        for kind in ('source','runtime','cache_ref'):
            p=deepcopy(P)
            if kind=='source':path=next(iter(p['config']['source_files']));p['config']['source_files'][path]['sha256']='0'*64;p['registration']['source_files']=p['config']['source_files']
            elif kind=='runtime':path=next(iter(p['runtime']['files']));p['runtime']['files'][path]['sha256']='0'*64;p['config']['runtime_files']=p['registration']['runtime_files']=p['runtime']['files']
            else:p['runtime']['metadata_references']['additional_preload']['body_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):check(reseal(p))
    def test_owner_review_and_dispatch_anchor_forgery_refuse(self):
        for key,value in [('owner_grant_reference',{'approved':True}),('all_current_byte_qualification_review',{'complete_current_byte_review':True}),('external_dispatch_head','a'*40)]:
            p=deepcopy(P);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):check(reseal(p))
    def test_scope_permission_and_resource_weakening_refuse(self):
        for kind in ('scope','permission','resources'):
            p=deepcopy(P)
            if kind=='scope':p['config']['scope']=p['registration']['scope']='ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY'
            elif kind=='permission':p['config']['permission']['tensor_allocations']=2
            else:p['config']['resources']['GPU_free_at_admission_GiB']=1.0
            with self.subTest(kind=kind),self.assertRaises(ValueError):check(reseal(p))
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));sys.meta_path.remove(trap);after=pins();receipt={'kind':'INDEPENDENT_CLOSED_PUBLIC_IMPORT_PROPOSAL002_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_packet_pins_before':BEFORE,'source_packet_pins_after':after,'pins_unchanged':BEFORE==after,'details':DETAILS,'actual_closed_wrapper_and_source_byte_joins_checked':True,'approved_native_qualifier_invoked_on_false_grant':False,'native_context_or_authority_issued':False,'native_API_or_framework_called':False,'full_terminal_audit_or_current_process_bootstrap_qualified':False};receipt['sha256']=M.digest(receipt)
    with (HERE/'INDEPENDENT_CLOSED_PUBLIC_PROPOSAL002.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
