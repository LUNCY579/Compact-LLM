"""Actual map/helper validators; explicit inert physical read seams only."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
from copy import deepcopy
import hashlib,json,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
Q=ModuleType('CPU_import_runtime_qualification');Q.__file__=str(HERE/'import_runtime_qualification.py');exec(compile(Path(Q.__file__).read_bytes(),Q.__file__,'exec'),Q.__dict__)
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_import_roster_');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.B=Q.original_helpers();self.calls=[]
        derivative=json.loads((PACKAGE/'artifacts/runtime_refresh_work/TENSOR_RUNTIME_BYTE_CANDIDATE001.json').read_bytes());addition=json.loads((PACKAGE/'artifacts/library_expectation_registration_work/ADDITIVE_WINDOWS_RUNTIME_CANDIDATE001.json').read_bytes());old=json.loads((PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json').read_bytes());observer=json.loads((PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json').read_bytes())['library_event_observer'];proposal=json.loads((PACKAGE/'artifacts/tensor_two_tensor_proposal_work/TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2_001.json').read_bytes());observation=proposal['observed009_selection']['observation_reference'];files=addition['files'];sources={}
        for path in self.B.package_source_roster():
            p=Path(path);raw=p.read_bytes();role='source_json' if p.suffix=='.json' else 'source_markdown' if p.suffix=='.md' else 'python_source';sources[path]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
        for name in ('import_observation_worker.py','import_observation_contract.py','import_observation_auditor.py','import_observation_supervisor.py','import_runtime_qualification.py','import_resource_guard.py','measurement_guard.py','measurement_replay.py'):
            local=HERE/name;raw=local.read_bytes();sources[str(PACKAGE/name)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
        sources.update(observer['native_source_files']);build=Path(observer['build_receipt']['path']);sources[str(build)]={'bytes':build.stat().st_size,'sha256':observer['build_receipt']['file_sha256'],'role':'source_json'}
        provider=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/history/WORKER_LIBRARY_PUBLICATION_SOURCE_FROZEN_017/source/prefix_library_policy.py';sources[str(provider)]={'bytes':provider.stat().st_size,'sha256':hashlib.sha256(provider.read_bytes()).hexdigest(),'role':'python_source'}
        recipe=self.B.seal({'kind':'REGISTERED_IMPORT_TRANSITIVE_SOURCE_ROSTER','source_files':{str(provider):sources[str(provider)]}}) if hasattr(self.B,'seal') else {'kind':'REGISTERED_IMPORT_TRANSITIVE_SOURCE_ROSTER','source_files':{str(provider):sources[str(provider)]}}
        recipe={k:v for k,v in recipe.items() if k!='sha256'};recipe['sha256']=self.B.digest(recipe);self.recipe=recipe;ref=self.reference('source-roster.json',recipe)
        libraries={p:d for p,d in files.items() if Path(p).suffix.lower() in ('.dll','.pyd','.exe')};libraries[observer['binary']['path']]=observer['binary']['descriptor'];dirs=sorted({str(Path(p).parent) for p in libraries});initial=json.loads((PACKAGE/'artifacts/library_expectation_registration_work/LIBRARY_EXPECTATION_INPUTS001.json').read_bytes())['expected_initial_loaded_libraries'];extensions={p:{'create':None,'exec':None} for p in libraries if Path(p).suffix.lower()=='.pyd' and p not in initial};required=[p for p in extensions if Path(p).name.startswith('_C.')];optional=[p for p in extensions if p not in required]
        policy={'kind':'FINITE_WINDOWS_DLL_BOUNDARY_V2','search_directories':dirs,'system_directory':str(self.B.WINDOWS_SYSTEM_DIRECTORY),'library_files':libraries,'initial_loaded_libraries':initial,'extension_transitions':extensions,'framework_transition':None,'provider_source':{'path':str(provider),'descriptor':sources[str(provider)]}}
        cache={'mode':'SOURCE_COMPILE_WITH_OBSERVED009_FINITE_CACHE_ASSUMPTIONS',**deepcopy(proposal['observed009_selection']['maps']),'interpreter_profile':proposal['observed009_selection']['interpreter_profile'],'separate_main_source':{'path':str(PACKAGE/'import_observation_worker.py'),'descriptor':sources[str(PACKAGE/'import_observation_worker.py')]},'source_execution_history_proven':False,'namespace_roots':{},'bootstrap_observation':observation}
        runtime={'kind':'REGISTERED_IMPORT_RUNTIME_BYTE_CLOSURE','source_commit':'a'*40,'files':files,'rosters':derivative['rosters'],'cache_policy':cache,'initial35_expectation':initial,'initial35_provenance':'observer009 expectation','observed_library_allowed_files':libraries,'exact_effects_or_order_prequalified':False,'native_runtime_qualified':False,'metadata_references':{'source_roster':ref},'inherited65_map':derivative['inherited65_map'],'stdlib779_map':derivative['stdlib779_map'],'external_dependency_files':derivative['external_dependency_files'],'library_search_policy':{'mode':'qualified_complete_registered_DLL_search_rosters','roster_names':list(derivative['rosters']),'unregistered_search_allowed':False,'enforced_boundary':policy},'stdlib_layout':self.B.QUALIFIED_LAYOUT,'windows_library_identities':{p:d for p,d in files.items() if d['role']=='windows_system_binary' and p not in derivative['inherited65_map'] and p not in derivative['stdlib779_map']},'library_event_observer':observer,'measurement_policy':policy,'measurement_grammar':{'schema_version':1,'kind':'FINITE_IMPORT_EFFECT_MEASUREMENT_GRAMMAR_V1','required_extensions':required,'optional_extensions':optional,'maximum_boundaries':1+2*len(extensions),'maximum_depth':64,'maximum_observations':4096,'model':False},'typing_alias_helper':{'path':str(HERE/'predecessors/typing_alias_checks.py'),'descriptor':{'bytes':23139,'sha256':'458764f327d861319f3502774f39d48c58d10fe8e7a476a387f65c3dd1e73742','role':'python_source'}}}
        config={'schema_version':1,'kind':'REGISTERED_OWNED_TORCH_IMPORT_OBSERVATION','scope':Q.SCOPE,'source_commit':'a'*40,'permission':Q.PERMISSION,'resources':self.B.RESOURCES,'source_files':sources,'runtime_files':files,'main_source':str(PACKAGE/'import_observation_worker.py'),'native_launch_authorized':True,'native_authority_issued':False,'owner_approval':None,'model':False,'whole_program_entry_observed':False,'helper_pins':old['helper_pins'],'offline_environment':old['offline_environment'],'runtime_versions':old['runtime_versions'],'GPU_physical_ordinal':0}
        config.update(torch_version=old['runtime_versions']['torch'],tolerances=old['tolerances']);runtime['sha256']=self.B.digest(runtime);config['sha256']=self.B.digest(config)
        review={'kind':'ROOT_REVIEWED_IMPORT_RUNTIME_BYTE_ROSTER_SOURCE_CACHE_CLOSURE','source_commit':'a'*40,'source_inventory_sha256':self.B.digest(sources),'runtime_closure_sha256':self.B.digest(files),'rosters_sha256':self.B.digest(runtime['rosters']),'cache_policy_sha256':self.B.digest(cache),'library_search_policy_sha256':self.B.digest(runtime['library_search_policy']),'measurement_policy_sha256':self.B.digest(policy),'measurement_grammar_sha256':self.B.digest(runtime['measurement_grammar']),'complete_current_byte_review':True};review['sha256']=self.B.digest(review);reference=self.reference('review.json',review)
        reg={'kind':'REGISTERED_IMPORT_OBSERVATION_REVIEW','scope':Q.SCOPE,'source_commit':'a'*40,'config_path':str(self.root/'config.json'),'config_body_sha256':config['sha256'],'runtime_body_sha256':runtime['sha256'],'source_files':sources,'runtime_files':files,'profile_sha256':self.B.digest({'scope':Q.SCOPE,'permission':Q.PERMISSION,'resources':self.B.RESOURCES}),'git_output_exceptions':{},'native_launch_authorized':True,'owner_approval':None,'qualification_review_path':reference['path'],'qualification_review_file_sha256':reference['file_sha256'],'qualification_review_body_sha256':reference['body_sha256']};reg['sha256']=self.B.digest(reg);self.value={'config':config,'runtime':runtime,'registration':reg,'expected_head':'b'*40}
        original_descriptor=self.B.file_descriptor
        def fixture_descriptor(path,d,**kwargs):self.calls.append(str(path));Q.require(str(path) in sources or str(path) in files,'explicit fixture selected descriptor');Q.require((sources.get(str(path)) or files[str(path)])==d,'fixture selected byte identity')
        self.B.file_descriptor=fixture_descriptor;self.B.physical_roster=lambda root,suffixes,excluded,recursive=True,budget=None:next(r['files'] for r in runtime['rosters'].values() if r['root']==str(root))
    def reference(self,name,value):
        path=self.root/name;raw=(self.B.canonical(value)+'\n').encode();path.write_bytes(raw);return {'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256']}
    def run_value(self):return Q._qualify(self.value,Q.QualificationBudget(),self.B)
    def reseal(self):
        for key in ('config','runtime'):
            value=self.value[key];value['sha256']=self.B.digest({k:v for k,v in value.items() if k!='sha256'})
        reg=self.value['registration'];reg.update(config_body_sha256=self.value['config']['sha256'],runtime_body_sha256=self.value['runtime']['sha256']);reg['sha256']=self.B.digest({k:v for k,v in reg.items() if k!='sha256'})
    def test_actual_original_map_schema_helpers_and_full_denominator(self):
        self.assertEqual(self.B.PACKAGE,PACKAGE);self.assertEqual(self.B.CORE,PACKAGE/'evidence/prefix_input_core_reviewed_cpu_20261005');self.assertEqual(self.B.STDLIB_CANDIDATE,PACKAGE/'evidence/stdlib_inventory_reviewed_cpu_20261005/STDLIB_CANDIDATE_001.json')
        result=self.run_value();self.assertFalse(result['current_module_execution_proven']);self.assertGreater(len(self.calls),8700)
    def test_namespace_main_and_historical_cache_rebinding_refuse(self):
        original=deepcopy(self.value)
        cache=self.value['runtime']['cache_policy'];cache['namespace_roots']={'forged':['C:\\fake']}
        self.reseal()
        with self.assertRaisesRegex(ValueError,'namespace'):self.run_value()
        self.value=deepcopy(original);self.value['runtime']['cache_policy']['separate_main_source']['path']=str(PACKAGE/'observer_startup_measurement_v1/worker.py');self.reseal()
        with self.assertRaisesRegex(ValueError,'separate current import main'):self.run_value()
        self.value=deepcopy(original);self.value['runtime']['cache_policy']['preloaded_bootstrap_modules']['ctypes']['source_sha256']='0'*64;self.reseal()
        with self.assertRaisesRegex(ValueError,'cannot be rebound'):self.run_value()
    def test_exact_source_or_role_mutation_refuse(self):
        original=deepcopy(self.value)
        p=next(p for p,d in self.value['runtime']['files'].items() if d['role']=='torch_binary');self.value['runtime']['files'][p]['role']='python_source'
        self.reseal()
        with self.assertRaisesRegex(ValueError,'role'):self.run_value()
        self.value=deepcopy(original);self.value['config']['source_files'].pop(str(PACKAGE/'import_resource_guard.py'));self.value['registration']['source_files']=self.value['config']['source_files'];self.reseal()
        with self.assertRaisesRegex(ValueError,'source graph'):self.run_value()
    def test_observer_ABI_or_roster_mutation_refuse(self):
        original=deepcopy(self.value)
        self.value['runtime']['library_event_observer']['abi']['row_capacity']=1
        self.reseal()
        with self.assertRaisesRegex(ValueError,'ABI'):self.run_value()
        self.value=deepcopy(original);self.value['runtime']['rosters']['torch']['suffixes']=['.py'];self.reseal()
        with self.assertRaisesRegex(ValueError,'finite roots/filter'):self.run_value()
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_IMPORT_RUNTIME_QUALIFICATION_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_original_schema_map_cache_observer_helpers_executed':True,'file_descriptor_and_physical_roster_explicit_inert_seams':True,'native_API_framework_called':False,'native_authority_issued':False}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
