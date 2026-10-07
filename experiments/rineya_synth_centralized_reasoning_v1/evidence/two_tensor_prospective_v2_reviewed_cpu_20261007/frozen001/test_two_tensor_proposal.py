"""Version2 supplement reader delegates unchanged closed packet reader; CPU only."""
from pathlib import Path
import copy,hashlib,json,types,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1];W=HERE.parent/'tensor_observed_admission_work';PACKET=HERE/'TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2_001.json'
C=types.ModuleType('CPU_prospective_reader_contract');C.__file__=str(W/'prefix_tensor_contract.py');raw=Path(C.__file__).read_bytes()
if hashlib.sha256(raw).hexdigest()!='0d12fd01bb1f06c29054943a3a114efc6eac9619c33989e5ce030d97a3cf4263':raise ValueError('externally fixed reviewed closed reader source')
exec(compile(raw,C.__file__,'exec'),C.__dict__)
FIELDS={'schema_version','kind','native_launch_authorized','native_runtime_qualified','native_authority_issued','owner_approval','implementation_commit','dispatch_head','closed_source_packet','observed009_selection','selected_current_helpers','separate_tensor_main','whole_program_registration','early_failure_return','current_witness_index','commands_after_promotion','historical_pending_index_retained','sha256'}
HELPERS={str(W/'registration_git_join.py'),str(W/'typing_alias_checks.py'),str(PACKAGE/'prefix_library_observer.py'),str(PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/history/WORKER_LIBRARY_PUBLICATION_SOURCE_FROZEN_017/source/prefix_library_policy.py')}
WITNESSES={'current_full_runtime_source_cache_roster_review','actual_tensor_main_provider_bootstrap','source_promotion_implementation_dispatch','gate_and_separate_owner_permission','owned_resource_admission_terminal_review'}
PHASES=['startup_source_qualification','startup_owned_metadata_profile','startup_ctypes_observer_binding','startup_registered_roots_loader','startup_native_provider_binding','startup_owned_cleanup_publication']
def require(value,message):
    if not value:raise ValueError(message)
def seal(value):value['sha256']=C.digest({k:v for k,v in value.items() if k!='sha256'});return value
def anchored(entry,path):
    require(type(entry) is dict and set(entry)=={'path','file_sha256','body_sha256'} and entry['path']==str(path),'exact selected external observation path');return C.sealed_json(path,entry['file_sha256'],entry['body_sha256'])
def validate_proposal_v2(p):
    require(type(p) is dict and set(p)==FIELDS and p['sha256']==C.digest({k:v for k,v in p.items() if k!='sha256'}),'exact bounded sealed V2 fields')
    require(type(p['schema_version']) is int and p['schema_version']==2 and p['kind']=='CLOSED_TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2','explicit prospective version2')
    require(all(p[k] is False for k in ('native_launch_authorized','native_runtime_qualified','native_authority_issued')) and all(p[k] is None for k in ('owner_approval','implementation_commit','dispatch_head')),'all authority and implementation/dispatch absent')
    original=C.validate_prospective_packet(p['closed_source_packet']);require(p['historical_pending_index_retained'] is True,'historical None slots remain historical')
    selected=p['observed009_selection'];require(type(selected) is dict and set(selected)=={'admission_reference','runtime_reference','observation_reference','maps','builtin_or_frozen_names','interpreter_profile','observer_main_excluded','tensor_main_observed','source_execution_history_proven','native_runtime_qualified'},'exact009 selection fields')
    admission=anchored(selected['admission_reference'],W/'OBSERVED_ADMISSION_001.json');runtime=anchored(selected['runtime_reference'],PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json');observation=anchored(selected['observation_reference'],Path(admission['observation_path']))
    require(selected['observation_reference']['file_sha256']==admission['observation_file_sha256'] and selected['observation_reference']['body_sha256']==admission['observation_body_sha256'],'observed009 original anchor join')
    modules,caches,aliases=C.observed_bootstrap_maps(observation);expected={'preloaded_bootstrap_modules':modules,'preloaded_module_cache_pins':caches,'preloaded_class_aliases':aliases}
    require(selected['maps']==expected=={k:admission[k] for k in expected} and len(modules)==63 and len(caches)==59 and set(aliases)=={'typing.io','typing.re'} and '__main__' not in modules,'exact original00963/59/2 maps exclude observer main')
    require(selected['builtin_or_frozen_names']==admission['builtin_or_frozen_names'] and len(selected['builtin_or_frozen_names'])==51 and selected['interpreter_profile']==runtime['cache_policy']['interpreter_profile']==p['closed_source_packet']['bootstrap_plan']['interpreter_profile'],'actual009 builtin/profile joins')
    require(selected['observer_main_excluded'] is True and all(selected[k] is False for k in ('tensor_main_observed','source_execution_history_proven','native_runtime_qualified')) and admission['tensor_main'] is None and admission['native_launch_authorized'] is admission['native_authority_issued'] is admission['source_execution_history_proven'] is False,'observation not tensor qualification')
    for name,row in modules.items():
        require(row['path'] in runtime['files'] and row['source_sha256']==runtime['files'][row['path']]['sha256'],'original source-map pins join runtime candidate')
    for path,d in caches.items():require(runtime['files'][path]==d,'exact59cache pins join original runtime candidate')
    helpers=p['selected_current_helpers'];require(type(helpers) is dict and set(helpers)==HELPERS,'exact current helper roster')
    for path,d in helpers.items():C.file_descriptor(path,d);require(d['role']=='python_source','selected helper source only')
    worker=p['closed_source_packet']['source_candidates']['prefix_input_owned_worker.py'];expected_main={'promotion_target':str(PACKAGE/'prefix_input_owned_worker.py'),'staged_source':{'path':worker['path'],'descriptor':{'bytes':worker['bytes'],'sha256':worker['sha256'],'role':'python_source'}},'expected_cache_shape':{'cached_attribute_present':True,'cached_value_kind':'none','cache_path':None,'cache_sha256':None},'actual_observation':None,'native_context_qualified':False};require(p['separate_tensor_main']==expected_main,'separate future main exact source/expected cache, actual remains absent')
    r=p['whole_program_registration'];require(r=={'core_operations':35,'startup_operations':6,'nonguard_success_operations':41,'core_maximum_phases':58,'maximum_guard_queries':14,'guard_sites':8,'optional_first_error_frame':1,'combined_maximum_phases':64,'combined_maximum_markers':128,'startup_order':PHASES,'exact_core_once_required':True,'worker_deadline_seconds':600,'dose':{'tensor_count':2,'dtype':'int64','shape':[1,422],'logical_tensor_bytes':6752,'model_forwards':0,'backwards':0,'optimizer_steps':0,'decodes':0,'grades':0}},'exact source-derived whole scope/input dose cap')
    early=p['early_failure_return'];require(early=={'kind':'NATIVE_SHARED_ENTRY_PRECORE_FAILURE','envelope_fields':['kind','report','runtime_execution_modules','owned_membership_verified','IO','native_PASS','report_requires_final_phase_accounting'],'report_fields':['scope','status','entry_errors','entry_bootstrap_sources','selected_boundary_counters','selected_boundary_source_pins','core_membership_status','precore_owned_membership','runtime_execution_status','native_PASS','native_authority_issued'],'provisional_requires_final_join':True,'writer_failure_preserves_partial':True,'native_authority_issued':False},'exact existing early envelope contract')
    witness=p['current_witness_index'];require(type(witness) is dict and set(witness)==WITNESSES and all(type(v) is dict and set(v)=={'status','witness','requirements'} and v['status']=='PENDING' and v['witness'] is None and type(v['requirements']) is list and v['requirements'] and all(type(x) is str for x in v['requirements']) for v in witness.values()),'exact five pending witness groups no claimed evidence')
    command=p['commands_after_promotion'];require(command['enabled'] is command['worker_standalone_launch_allowed'] is command['direct_supervisor_script_is_a_launch_CLI'] is False and command['supervisor_source']==str(PACKAGE/'prefix_input_owned_supervisor.py') and command['registered_supervisor_api']==str(PACKAGE/'prefix_native_pipeline.py')+'::supervise_registered','future actual API path, not a fabricated current CLI')
    argv=command['child_worker_argv_template'];require(argv==[runtime['cache_policy']['interpreter_profile']['executable'],'-I','-S','-B',str(PACKAGE/'prefix_input_owned_worker.py'),'--config','<exact separately registered config path>','--output-dir','<supervisor-created owned output leaf>'],'exact future owned -I-S-B worker shape')
    require(command['child_must_be_created_suspended_and_Job_assigned_before_resume'] is True and set(command['external_anchor_fields'])=={'expected_config_file','expected_config_body','registration_path','expected_registration_file','expected_registration_body','runtime_path','expected_runtime_file','expected_runtime_body','expected_head'},'original assigned-before-resume/external anchors')
    return {'kind':'CPU_VALIDATED_CLOSED_TWO_TENSOR_PROSPECTIVE_V2','maps':[63,59,2],'current_witnesses':5,'native_launch_authorized':False,'native_runtime_qualified':False,'native_authority_issued':False}
class Controls(unittest.TestCase):
    def setUp(self):self.p=C.parse(C.read_bytes(PACKET))
    def test_actual_original_reader_and_exact009_v2_closed_proposal(self):self.assertEqual(validate_proposal_v2(self.p)['maps'],[63,59,2]);self.assertIsNone(self.p['implementation_commit']);self.assertEqual(len(self.p['closed_source_packet']['pending']),11)
    def test_current_source_and_helper_rebinding_refused(self):
        for change in ('source','helper'):
            p=copy.deepcopy(self.p)
            if change=='source':p['closed_source_packet']['source_candidates']['prefix_input_owned_worker.py']['sha256']='0'*64;seal(p['closed_source_packet'])
            else:p['selected_current_helpers'][str(W/'typing_alias_checks.py')]['sha256']='0'*64
            with self.assertRaises(ValueError):validate_proposal_v2(seal(p))
    def test_admission_observer_main_or_cache_rebinding_refused(self):
        for change in ('module','cache','main'):
            p=copy.deepcopy(self.p)
            if change=='module':p['observed009_selection']['maps']['preloaded_bootstrap_modules']['__main__']=next(iter(p['observed009_selection']['maps']['preloaded_bootstrap_modules'].values()))
            elif change=='cache':next(iter(p['observed009_selection']['maps']['preloaded_module_cache_pins'].values()))['sha256']='0'*64
            else:p['separate_tensor_main']['actual_observation']={'renamed_observer':True}
            with self.assertRaises(ValueError):validate_proposal_v2(seal(p))
    def test_dose_authority_and_pending_witness_rebinding_refused(self):
        for change in ('dose','authority','pending'):
            p=copy.deepcopy(self.p)
            if change=='dose':p['whole_program_registration']['combined_maximum_markers']=130
            elif change=='authority':p['native_launch_authorized']=True
            else:next(iter(p['current_witness_index'].values()))['witness']={'qualified':True}
            with self.assertRaises(ValueError):validate_proposal_v2(seal(p))
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_PROSPECTIVE_V2_SCHEMA_REBINDING_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'actual_existing_closed_packet_reader_called':True,'actual_observed009_maps_reader_called':True,'native_API_framework_called':False,'native_authority_issued':False,'packet_file_sha256':hashlib.sha256(PACKET.read_bytes()).hexdigest(),'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),W/'prefix_tensor_contract.py']}}
    if len(__import__('sys').argv)>1:
        with Path(__import__('sys').argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
