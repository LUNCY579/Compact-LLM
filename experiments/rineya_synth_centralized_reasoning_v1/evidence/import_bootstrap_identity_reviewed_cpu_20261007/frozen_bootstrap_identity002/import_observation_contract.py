"""Separate import-only purpose/context; no tensor-context or grant reuse."""
from pathlib import Path
import ast,hashlib,importlib.util,json,math,types
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
B=types.ModuleType('import_only_selected_source_utilities');B.__file__=str(SHARED_SOURCE);B.__dict__['__import_contract_file__']=str(Path(__file__).resolve());tree=ast.parse(raw,str(SHARED_SOURCE));adaptations=[]
class ImportPurpose(ast.NodeTransformer):
    def visit_Assign(self,node):
        self.generic_visit(node)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name=='NATIVE_SCOPE':node.value=ast.Constant(SCOPE);adaptations.append('scope')
            elif name=='PERMISSION':node.value=ast.parse(repr(PERMISSION),mode='eval').body;adaptations.append('permission')
            elif name=='HERE':node.value=ast.parse('Path(__import_contract_file__).resolve().parent',mode='eval').body;adaptations.append('own_contract_location')
        return node
    def visit_Constant(self,node):
        table={'SEPARATE_TWO_TENSOR_OWNER_SCOPE_GRANT':'SEPARATE_TORCH_IMPORT_OBSERVATION_OWNER_GRANT','VERIFIED_OWNER_TWO_TENSOR_APPROVAL_V1':'VERIFIED_OWNER_TORCH_IMPORT_APPROVAL_V1'}
        if type(node.value) is str and node.value in table:adaptations.append(node.value);return ast.copy_location(ast.Constant(table[node.value]),node)
        return node
tree=ImportPurpose().visit(tree);ast.fix_missing_locations(tree)
if sorted(adaptations)!=sorted(['scope','permission','own_contract_location','SEPARATE_TWO_TENSOR_OWNER_SCOPE_GRANT','VERIFIED_OWNER_TWO_TENSOR_APPROVAL_V1']):raise ValueError('exact five import-only source utility adaptations')
exec(compile(tree,str(SHARED_SOURCE),'exec'),B.__dict__);SOURCE_ADAPTATIONS=tuple(adaptations)

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
    def native(value):
        B.stage_native_gate();require(value.get('fixture_only') is False and value.get('scope')==SCOPE,'actual independently qualified import observation context');v=ImportContext();records[v]=canonical(value);return v
    return cap,check,mint,value,native
_CPU_capability,_check_CPU,_CPU_context,context_value,_mint_import_native=_context_factory();B.context_value=context_value
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

def stage_native_gate():return B.stage_native_gate()
def enroll_native_gate(grant_reference,receipt_reference,binding):return B.enroll_native_gate(grant_reference,receipt_reference,binding)
def enroll_native_gate_from_owned_output(output):return B.enroll_native_gate_from_owned_output(output)
def validate_native_manifest(config_path,**anchors):
    stage_native_gate();budget=QualificationBudget();required={'expected_config_file','expected_config_body','registration_path','expected_registration_file','expected_registration_body','runtime_path','expected_runtime_file','expected_runtime_body','expected_head'};require(set(anchors)==required,'exact external import observation anchors')
    config=sealed_json(config_path,anchors['expected_config_file'],anchors['expected_config_body'],budget=budget);runtime=sealed_json(anchors['runtime_path'],anchors['expected_runtime_file'],anchors['expected_runtime_body'],budget=budget);registration=sealed_json(anchors['registration_path'],anchors['expected_registration_file'],anchors['expected_registration_body'],budget=budget)
    require(config['scope']==registration['scope']==SCOPE and config['permission']==PERMISSION and config['resources']==RESOURCES and config['native_launch_authorized'] is registration['native_launch_authorized'] is True,'separately approved exact import-only purpose')
    require(config['source_commit']==runtime['source_commit']==registration['source_commit'] and config['source_commit']!=anchors['expected_head'] and config['source_files']==registration['source_files'] and config['runtime_files']==registration['runtime_files']==runtime['files'],'import source/runtime/implementation packet joins')
    require(registration['config_body_sha256']==config['sha256'] and registration['runtime_body_sha256']==runtime['sha256'] and runtime['exact_effects_or_order_prequalified'] is False,'measurement never claims predetermined exact effects')
    for mapping in (config['source_files'],runtime['files']):
        for path,d in mapping.items():file_descriptor(path,d,layout=runtime['stdlib_layout'],budget=budget)
    for name,row in runtime['rosters'].items():require(B.physical_roster(row['root'],row['suffixes'],row['excluded_directory_names'],recursive=row['recursive'],budget=budget)==row['files'],'current exact import-stage physical roster')
    helper_path=HERE/'predecessors/registration_git_join.py';d=config['source_files'].get(str(helper_path));require(d is not None,'source-bound import Git helper');file_descriptor(helper_path,d,budget=budget);helper=types.ModuleType('import_selected_git');helper.__file__=str(helper_path);exec(compile(read_bytes(helper_path,budget=budget),str(helper_path),'exec'),helper.__dict__)
    packets={'config':{'path':str(Path(config_path).absolute()),'file_sha256':anchors['expected_config_file'],'body_sha256':anchors['expected_config_body']},'runtime':{'path':anchors['runtime_path'],'file_sha256':anchors['expected_runtime_file'],'body_sha256':anchors['expected_runtime_body']},'registration':{'path':anchors['registration_path'],'file_sha256':anchors['expected_registration_file'],'body_sha256':anchors['expected_registration_body']}}
    helper._CPU_join(PACKAGE.parents[1],packets=packets,source_files=config['source_files'],implementation_commit=config['source_commit'],expected_head=anchors['expected_head'],git_output_exceptions=registration['git_output_exceptions'],_capability=helper._CPU_capability());budget.check()
    value={'scope':SCOPE,'fixture_only':False,'native_PASS':False,'native_authority_issued':False,'config':config,'runtime':runtime,'registration':registration,'anchors':anchors,'config_path':str(Path(config_path).absolute()),'expected_head':anchors['expected_head']}
    qualifier_path=HERE/'import_runtime_qualification.py';descriptor=config['source_files'].get(str(qualifier_path));require(type(descriptor) is dict,'selected independent full import runtime qualifier');file_descriptor(qualifier_path,descriptor,budget=budget);qualifier_raw=read_bytes(qualifier_path,budget=budget);require(len(qualifier_raw)==descriptor['bytes'] and hashlib.sha256(qualifier_raw).hexdigest()==descriptor['sha256'],'actual selected generic qualifier bytes');qualifier=types.ModuleType('own_import_runtime_qualifier');qualifier.__file__=str(qualifier_path);exec(compile(qualifier_raw,str(qualifier_path),'exec'),qualifier.__dict__);value['qualification']=qualifier.qualify_import_value(value,budget)
    return _mint_import_native(value)
def native_context(*args,**kwargs):return validate_native_manifest(*args,**kwargs)
def revalidate_context(context):
    value=context_value(context);current=validate_native_manifest(value['config_path'],**value['anchors']);require(context_value(current)==value,'same immutable import-only context');return value
# Recompile exactly the original observer registry factory/wrappers in this
# scope's globals, with this ImportContext getter and this private CPU checker.
library_event_observer_registration=B.library_event_observer_registration
LIBRARY_OBSERVER_ABI=B.LIBRARY_OBSERVER_ABI
sha=B.sha
_check_CPU_native_context_fixture=_check_CPU
nodes=[node for node in ast.parse(raw,str(SHARED_SOURCE)).body if isinstance(node,ast.FunctionDef) and node.name in ('_library_observer_binding_factory','_register_library_event_observer','verify_native_library_event_observer','_CPU_register_library_event_observer_fixture','_CPU_verify_library_event_observer_fixture')]
require(len(nodes)==5,'exact five source-bound separate-context observer registry definitions')
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SHARED_SOURCE),'exec'),globals());_record_library_observer,_verify_library_observer=_library_observer_binding_factory()

def owner_grant_template():
    v={'kind':'SEPARATE_TORCH_IMPORT_OBSERVATION_OWNER_GRANT','scope':SCOPE,'permission':PERMISSION,'resources':RESOURCES,'approved':False,'trusted_owner_origin':None,'implementation_commit':None,'dispatch_head':None};v['sha256']=digest(v);return v

loader_source=HERE/'predecessors/prefix_registered_loader.py';loader_raw=read_bytes(loader_source);require(hashlib.sha256(loader_raw).hexdigest()=='62c25f6cf5f843a08463dbb7c3c573f6a038f036917fc4130da0aab5d4b7e0c5','selected unchanged bounded code metadata helper');fp_nodes=[node for node in ast.parse(loader_raw,str(loader_source)).body if isinstance(node,ast.FunctionDef) and node.name=='code_fingerprint'];require(len(fp_nodes)==1,'one source-bound complete Python code fingerprint helper');exec(compile(ast.Module(body=fp_nodes,type_ignores=[]),str(loader_source),'exec'),globals())

def loader_context_value(context,current_preloads=None):
    value=context_value(context);result=parse(canonical(value));cache=result['runtime']['cache_policy'];own=cache['separate_main_source'];cache.update(bootstrap_admission_version=2,Torch_root=str(SITE/'torch'),namespace_roots={},tensor_main_source={'path':own['path'],'descriptor':own['descriptor'],'cached_attribute_present':True,'cached_value_kind':'none','cache_path':None,'cache_sha256':None},typing_alias_helper=result['runtime']['typing_alias_helper']);
    if current_preloads is not None:
        require(set(current_preloads)==set(cache['additional_preload_sources']) and not(set(current_preloads)&set(cache['preloaded_bootstrap_modules'])),'actual own-process extra preloads match separate overlay');cache['preloaded_bootstrap_modules']={**cache['preloaded_bootstrap_modules'],**{name:{k:row[k] for k in ('path','source_sha256','cache_path','cache_sha256','cached_attribute_present','cached_value_kind')} for name,row in current_preloads.items()}};cache['preloaded_module_cache_pins']={**cache['preloaded_module_cache_pins'],**{row['cache_path']:row['cache_descriptor'] for row in cache['additional_preload_sources'].values() if row['cache_path'] is not None}}
    return result
