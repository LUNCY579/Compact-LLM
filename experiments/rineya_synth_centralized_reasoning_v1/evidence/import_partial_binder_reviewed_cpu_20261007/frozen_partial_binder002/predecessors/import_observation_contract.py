"""Separate import-only purpose/context; no tensor-context or grant reuse."""
from pathlib import Path
import hashlib,importlib.util,json,math,types
HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in (HERE,*HERE.parents) if p.name=='rineya_synth_centralized_reasoning_v1')
SCOPE='OWNED_TORCH_IMPORT_LIBRARY_EFFECT_OBSERVATION_ONLY'
NATIVE_SCOPE=SCOPE
PERMISSION={'torch_import':True,'CUDA_driver_interfaces_may_be_reached':True,'user_CUDA_context':False,'tensor_allocations':0,'model':False,'forward':False,'backward':False,'optimizer':False,'decoder':False,'scoring':False,'M0':False,'protected_bank_access':False,'paid_services':False}
OPERATIONS=('source_qualification','ownership_verify','resource_before_import','measurement_begin','framework_import','torch_version_read','resource_after_import','measurement_observations','measurement_cleanup','raw_measurement_publication','worker_result_publication')
KIND='CLOSED_ONE_OWNED_TORCH_IMPORT_OBSERVATION'
MAXIMUM_MARKERS=128
SHARED_SOURCE=HERE/'predecessors/prefix_tensor_contract.py'
SHARED_PIN='80a4650ab7c4b386f560de7c18f08d5c8d317f948fc0212afdf5cb5be4d9a577'
raw=SHARED_SOURCE.read_bytes()
if len(raw)>16*1024**2 or hashlib.sha256(raw).hexdigest()!=SHARED_PIN:raise ValueError('selected reviewed source utility predecessor')
B=types.ModuleType('import_only_selected_source_utilities');B.__file__=str(SHARED_SOURCE);exec(compile(raw,str(SHARED_SOURCE),'exec'),B.__dict__)
RESOURCES=B.RESOURCES;QualificationBudget=B.QualificationBudget;QUALIFIED_LAYOUT=B.QUALIFIED_LAYOUT;SITE=B.SITE
require=B.require;canonical=B.canonical;parse=B.parse;digest=B.digest;read_bytes=B.read_bytes;file_descriptor=B.file_descriptor;qualified_path=B.qualified_path;sealed_json=B.sealed_json
# Independent context/capability classes, never original TensorContext records.
def _context_factory():
    records={};caps=set()
    class ImportContext:__slots__=()
    class CPU:__slots__=()
    def cap():v=CPU();caps.add(v);return v
    def check(v):require(type(v) is CPU and v in caps,'private import observation CPU capability')
    def mint(value,capability):
        check(capability);require(value.get('fixture_only') is True and value.get('scope')==SCOPE,'separate CPU import context only');v=ImportContext();records[v]=canonical(value);return v
    def value(v):require(type(v) is ImportContext and v in records,'original separate import observation context');return parse(records[v])
    return cap,check,mint,value
_CPU_capability,_check_CPU,_CPU_context,context_value=_context_factory()
def validate_import_metadata(config,runtime,registration,expected_head):
    cf={'schema_version','kind','scope','source_commit','permission','resources','source_files','runtime_files','main_source','native_launch_authorized','native_authority_issued','owner_approval','model','whole_program_entry_observed','sha256'}
    rf={'kind','source_commit','files','rosters','cache_policy','initial35_expectation','initial35_provenance','observed_library_allowed_files','exact_effects_or_order_prequalified','native_runtime_qualified','metadata_references','sha256'}
    gf={'kind','scope','source_commit','config_path','config_body_sha256','runtime_body_sha256','source_files','runtime_files','profile_sha256','git_output_exceptions','native_launch_authorized','owner_approval','sha256'}
    require(type(config) is dict and set(config)==cf and type(runtime) is dict and rf<=set(runtime)<=rf|{'library_event_observer','measurement_policy','measurement_grammar'} and type(registration) is dict and set(registration)==gf,'exact separate import-only packet fieldsets')
    for value in (config,runtime,registration):require(value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'exact immutable import observation packet seals')
    require(type(config['schema_version']) is int and config['schema_version']==1 and config['kind']==KIND and config['scope']==registration['scope']==SCOPE and config['permission']==PERMISSION and config['resources']==RESOURCES,'exact separate import-only scope/resource purpose')
    implementation=config['source_commit'];require(implementation==runtime['source_commit']==registration['source_commit'] and ((implementation is expected_head is None) or type(implementation) is str and type(expected_head) is str and len(implementation)==len(expected_head)==40 and all(c in '0123456789abcdef' for c in implementation+expected_head) and implementation!=expected_head),'pending or distinct actual implementation/dispatch source identities')
    require(type(config['source_files']) is dict and config['source_files'] and config['main_source'] in config['source_files'] and config['source_files']==registration['source_files'] and config['runtime_files']==registration['runtime_files']==runtime['files'],'exact source/runtime maps and separate own main')
    require(registration['config_body_sha256']==config['sha256'] and registration['runtime_body_sha256']==runtime['sha256'] and registration['profile_sha256']==digest({'scope':SCOPE,'permission':PERMISSION,'resources':RESOURCES}),'exact import observation packet/profile joins')
    require(config['native_launch_authorized'] is config['native_authority_issued'] is config['model'] is config['whole_program_entry_observed'] is registration['native_launch_authorized'] is False and config['owner_approval'] is registration['owner_approval'] is None,'closed candidate has no owner permission or actual program observation')
    require(runtime['native_runtime_qualified'] is runtime['exact_effects_or_order_prequalified'] is False,'measurement unknown effects cannot be qualified PASS')
    return {'scope':SCOPE,'permission':PERMISSION,'config':config,'runtime':runtime,'registration':registration,'native_authority_issued':False}

def stage_native_gate():raise ValueError('separate Torch-import-only owner grant/registered runtime not provided')
def native_context(*args,**kwargs):stage_native_gate()
def owner_grant_template():
    v={'kind':'SEPARATE_TORCH_IMPORT_OBSERVATION_OWNER_GRANT','scope':SCOPE,'permission':PERMISSION,'resources':RESOURCES,'approved':False,'trusted_owner_origin':None,'implementation_commit':None,'dispatch_head':None};v['sha256']=digest(v);return v
