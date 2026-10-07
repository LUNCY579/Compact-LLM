"""Actual selected reader + qualifier before CPU receipt boundary, no permit."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import ast,builtins,hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent
def load(path,name):
    raw=path.read_bytes();m=ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m
BASE=load(HERE/'test_import_runtime_qualification.py','CPU_reader_qualifier_fixture')
CONTRACT=HERE/'import_observation_contract.py';CONTRACT_PIN='f19c53af565a1227a3cd7f116fef2350307c390267e6ace74d5ce6d5476fdd6f'
QUALIFIER=HERE/'import_runtime_qualification.py';QUALIFIER_PIN='9c270b3eeffab6acd69d3d1f618645b5fa7f3d57dff64819088879dce83c0607'
class Controls(BASE.Controls):
    def setup_reader(self):
        self.add_passive_overlay();self.assertEqual(hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),CONTRACT_PIN);self.assertEqual(hashlib.sha256(QUALIFIER.read_bytes()).hexdigest(),QUALIFIER_PIN);self.C=load(CONTRACT,'CPU_actual_reader_contract');self.minted=[];self.gate_calls=[];self.exec_calls=[];self.git_calls=[]
        sources=self.value['config']['source_files'];sources[str(QUALIFIER)]={'bytes':QUALIFIER.stat().st_size,'sha256':QUALIFIER_PIN,'role':'python_source'};git=HERE/'predecessors/registration_git_join.py';sources[str(git)]={'bytes':git.stat().st_size,'sha256':hashlib.sha256(git.read_bytes()).hexdigest(),'role':'python_source'};self.value['registration']['source_files']=sources;self.reseal()
        # Extend the explicitly selected transitive recipe for the actual
        # staged qualifier/Git bytes without declaring public promotion.
        self.recipe['source_files'].update({str(QUALIFIER):sources[str(QUALIFIER)],str(git):sources[str(git)]});self.recipe['sha256']=self.B.digest({k:v for k,v in self.recipe.items() if k!='sha256'});ref=self.reference('source-roster-reader.json',self.recipe);self.value['runtime']['metadata_references']['source_roster']=ref;self.reseal();self.refresh_review()
        C=self.C;C.stage_native_gate=lambda:self.gate_calls.append('CPU_fixture_gate_boundary')
        function=next(n for n in ast.parse(CONTRACT.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_native_manifest');calls=[n.func.id for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)];self.assertEqual(calls.count('stage_native_gate'),1);self.assertEqual(calls.count('_mint_import_native'),1)
        C._mint_import_native=lambda value:self.minted.append(value) or {'kind':'CPU_READER_QUALIFIER_RETURN_ONLY','qualification':value['qualification'],'native_authority_issued':False}
        C.file_descriptor=self.B.file_descriptor;C.B.physical_roster=self.B.physical_roster
        realexec=builtins.exec
        def selected_exec(code,namespace):
            realexec(code,namespace)
            if code.co_filename==str(QUALIFIER):
                self.exec_calls.append('actual_qualifier_source_exec');namespace['B'].file_descriptor=self.B.file_descriptor;namespace['B'].physical_roster=self.B.physical_roster
            elif code.co_filename==str(git):
                self.exec_calls.append('actual_Git_helper_source_exec')
                original=namespace['_CPU_join']
                def immutable_join(root,**kwargs):
                    self.git_calls.append(kwargs);self.assertEqual(kwargs['source_files'],sources);self.assertEqual(kwargs['implementation_commit'],'a'*40);self.assertEqual(kwargs['expected_head'],'b'*40);return {'kind':'EXPLICIT_CPU_IMMUTABLE_SOURCE_JOIN_SPY','native_authority_issued':False}
                namespace['_CPU_join']=immutable_join
        C.exec=selected_exec
    def refresh_review(self):
        runtime=self.value['runtime'];review={'kind':'ROOT_REVIEWED_IMPORT_RUNTIME_BYTE_ROSTER_SOURCE_CACHE_CLOSURE','source_commit':'a'*40,'source_inventory_sha256':self.B.digest(self.value['config']['source_files']),'runtime_closure_sha256':self.B.digest(runtime['files']),'rosters_sha256':self.B.digest(runtime['rosters']),'cache_policy_sha256':self.B.digest(runtime['cache_policy']),'library_search_policy_sha256':self.B.digest(runtime['library_search_policy']),'measurement_policy_sha256':self.B.digest(runtime['measurement_policy']),'measurement_grammar_sha256':self.B.digest(runtime['measurement_grammar']),'complete_current_byte_review':True};review['sha256']=self.B.digest(review);ref=self.reference('reader-review.json',review);self.value['registration'].update(qualification_review_path=ref['path'],qualification_review_file_sha256=ref['file_sha256'],qualification_review_body_sha256=ref['body_sha256']);self.reseal()
    def call_reader(self):
        paths={key:self.root/(key+'.json') for key in ('config','runtime','registration')};refs={}
        for key,path in paths.items():
            value=self.value[key];raw=(self.B.canonical(value)+'\n').encode();path.write_bytes(raw);refs[key]={'file':hashlib.sha256(raw).hexdigest(),'body':value['sha256']}
        return self.C.validate_native_manifest(paths['config'],expected_config_file=refs['config']['file'],expected_config_body=refs['config']['body'],registration_path=str(paths['registration']),expected_registration_file=refs['registration']['file'],expected_registration_body=refs['registration']['body'],runtime_path=str(paths['runtime']),expected_runtime_file=refs['runtime']['file'],expected_runtime_body=refs['runtime']['body'],expected_head='b'*40)
    def test_actual_reader_loads_pinned_qualifier_before_CPU_return(self):
        self.setup_reader();result=self.call_reader();self.assertEqual(result['kind'],'CPU_READER_QUALIFIER_RETURN_ONLY');self.assertEqual(len(self.minted),1);self.assertEqual(self.gate_calls,['CPU_fixture_gate_boundary']);self.assertEqual(self.exec_calls,['actual_Git_helper_source_exec','actual_qualifier_source_exec']);self.assertEqual(len(self.git_calls),1);self.assertFalse(result['qualification']['current_module_execution_proven'])
    def test_source_bound_main_and_role_mutations_refuse_before_mint(self):
        self.setup_reader();original=deepcopy(self.value);self.value['runtime']['cache_policy']['separate_main_source']['path']='C:\\forged-main.py';self.reseal()
        with self.assertRaises(ValueError):self.call_reader()
        self.assertEqual(self.minted,[])
        self.value=deepcopy(original);p=next(p for p,d in self.value['runtime']['files'].items() if d['role']=='torch_binary');self.value['runtime']['files'][p]['role']='python_source';self.reseal()
        with self.assertRaisesRegex(ValueError,'role|selected byte identity'):self.call_reader()
        self.assertEqual(self.minted,[])
    def test_cache_association_rebind_refuses_before_mint(self):
        self.setup_reader();self.value['runtime']['cache_policy']['additional_preload_sources']['psutil']['cache_path']=next(iter(self.value['runtime']['cache_policy']['preloaded_module_cache_pins']));self.reseal()
        with self.assertRaisesRegex(ValueError,'passive receipt'):self.call_reader()
        self.assertEqual(self.minted,[])
    def test_wrong_qualifier_pin_refuses_before_execution_or_mint(self):
        self.setup_reader();self.value['config']['source_files'][str(QUALIFIER)]['sha256']='0'*64;self.reseal()
        with self.assertRaises(ValueError):self.call_reader()
        self.assertNotIn('actual_qualifier_source_exec',self.exec_calls);self.assertEqual(self.minted,[])
if __name__=='__main__':
    methods=('test_actual_reader_loads_pinned_qualifier_before_CPU_return','test_source_bound_main_and_role_mutations_refuse_before_mint','test_cache_association_rebind_refuses_before_mint','test_wrong_qualifier_pin_refuses_before_execution_or_mint');result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in methods));v={'kind':'CPU_ACTUAL_IMPORT_READER_SELECTED_QUALIFIER_JOIN_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_reader_and_qualifier_sources_executed':True,'production_sources_edited':False,'gate_and_mint_boundary_CPU_fixture_only':True,'Git_join_spy_not_actual_Git_qualification':True,'filehash_roster_reader_seams_CPU_injected':True,'native_API_framework_called':False,'native_context_or_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (CONTRACT,QUALIFIER,Path(__file__).resolve(),HERE/'test_import_runtime_qualification.py')}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
