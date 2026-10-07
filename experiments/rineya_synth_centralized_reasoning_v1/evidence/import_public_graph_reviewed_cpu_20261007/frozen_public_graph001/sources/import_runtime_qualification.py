"""Strict separate import runtime byte/roster qualifier; never a permit."""
from pathlib import Path
from types import ModuleType
import ast,hashlib,json,stat
HERE=Path(__file__).resolve().parent;PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
SOURCE=HERE/'predecessors/prefix_tensor_contract.py';SOURCE_PIN='80a4650ab7c4b386f560de7c18f08d5c8d317f948fc0212afdf5cb5be4d9a577'
SCOPE='OWNED_TORCH_IMPORT_LIBRARY_EFFECT_OBSERVATION_ONLY'
PERMISSION={'torch_import':True,'CUDA_driver_interfaces_may_be_reached':True,'user_CUDA_context':False,'tensor_allocations':0,'model':False,'forward':False,'backward':False,'optimizer':False,'decoder':False,'scoring':False,'M0':False,'protected_bank_access':False,'paid_services':False}
def require(v,m):
    if not v:raise ValueError(m)
def original_helpers():
    raw=SOURCE.read_bytes();require(hashlib.sha256(raw).hexdigest()==SOURCE_PIN,'exact source-bound original qualification helpers');m=ModuleType('import_qualification_original_helpers');m.__file__=str(SOURCE);m.__dict__['__qualification_contract_file__']=str(PACKAGE/'prefix_tensor_contract.py');tree=ast.parse(raw,str(SOURCE));changes=[]
    class PackageLocation(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='HERE':
                require(ast.dump(node.value)==ast.dump(ast.parse('Path(__file__).resolve().parent',mode='eval').body),'exact original helper HERE initializer');node.value=ast.parse('Path(__qualification_contract_file__).resolve().parent',mode='eval').body;changes.append('HERE')
            return self.generic_visit(node)
    tree=PackageLocation().visit(tree);require(changes==['HERE'],'one source-bound purpose package-location adaptation');ast.fix_missing_locations(tree);exec(compile(tree,str(SOURCE),'exec'),m.__dict__);m.__compiled_source_descriptor__={'bytes':len(raw),'sha256':SOURCE_PIN,'role':'python_source'};require(m.PACKAGE==PACKAGE,'actual public qualification package identity');return m
B=original_helpers();QualificationBudget=B.QualificationBudget
CONFIG_FIELDS={'schema_version','kind','scope','source_commit','permission','resources','source_files','runtime_files','main_source','native_launch_authorized','native_authority_issued','owner_approval','model','whole_program_entry_observed','helper_pins','offline_environment','runtime_versions','torch_version','tolerances','GPU_physical_ordinal','sha256'}
RUNTIME_FIELDS={'kind','source_commit','files','rosters','cache_policy','initial35_expectation','initial35_provenance','observed_library_allowed_files','exact_effects_or_order_prequalified','native_runtime_qualified','metadata_references','inherited65_map','stdlib779_map','external_dependency_files','library_search_policy','stdlib_layout','windows_library_identities','library_event_observer','measurement_policy','measurement_grammar','typing_alias_helper','sha256'}
REGISTRATION_FIELDS={'kind','scope','source_commit','config_path','config_body_sha256','runtime_body_sha256','source_files','runtime_files','profile_sha256','git_output_exceptions','native_launch_authorized','owner_approval','qualification_review_path','qualification_review_file_sha256','qualification_review_body_sha256','sha256'}
def _seal(value):require(type(value) is dict and value.get('sha256')==B.digest({k:v for k,v in value.items() if k!='sha256'}),'exact separate qualification packet seal')
def passive_preload_overlay(receipt):
    require(type(receipt) is dict and receipt.get('original_observed63_59_maps_not_rewritten') is True and
        receipt.get('psutil_source_or_extension_imported') is False,'source-only independently selected passive preload receipt')
    caches=receipt['additional_cache_candidates'];overlay={}
    for name,row in receipt['source_modules'].items():
        matched=[(p,c) for p,c in caches.items() if c['source_path']==row['path']];require(len(matched)==1,'exact one selected Python preload cache candidate');path,cache=matched[0]
        require(cache['code_source_equivalence'] is True and cache['cache_executed'] is False,'passive added cache equivalence, no execution')
        overlay[name]={'path':row['path'],'descriptor':row['descriptor'],'cached_attribute_present':True,'cached_value_kind':'str','cache_path':path,'cache_descriptor':cache['descriptor'],'source_execution_history_proven':False}
    for name,row in receipt['extension_modules'].items():overlay[name]={'path':row['path'],'descriptor':row['descriptor'],'cached_attribute_present':False,'cached_value_kind':'missing','cache_path':None,'cache_descriptor':None,'source_execution_history_proven':False}
    require(len(overlay)==10 and len(caches)==7,'exact passive7 Python/3 extension candidates');return overlay
def _source_roster(value,helpers,budget):
    # Original package discovery excludes only owned artifacts/evidence/cache.
    result=helpers.package_source_roster();selected=value['config']['source_files'];known_required={str(PACKAGE/name) for name in ('import_observation_worker.py','import_observation_contract.py','import_observation_auditor.py','import_observation_supervisor.py','import_runtime_qualification.py','import_resource_guard.py','measurement_guard.py','measurement_replay.py')}
    require(known_required<=set(selected),'complete separate program public source graph')
    result|=known_required
    # Transitive immutable selected sources are externally named by recipe;
    # they must be exact declared source-only leaves and exist on disk.
    transitive=value['runtime']['metadata_references'].get('source_roster')
    require(type(transitive) is dict and set(transitive)=={'path','file_sha256','body_sha256'},'externally selected complete source-roster recipe')
    recipe=helpers.sealed_json(transitive['path'],transitive['file_sha256'],transitive['body_sha256'],budget=budget);require(set(recipe)=={'kind','source_files','sha256'} and recipe['kind']=='REGISTERED_IMPORT_TRANSITIVE_SOURCE_ROSTER','exact transitive source recipe')
    require(type(recipe['source_files']) is dict and all(selected.get(p)==d for p,d in recipe['source_files'].items()),'exact transitive source descriptor joins');result.update(recipe['source_files'])
    observer=value['runtime']['library_event_observer'];result.update(observer['native_source_files']);result.add(observer['build_receipt']['path'])
    require(set(selected)==result,'complete exact package/transitive source roster')
def _cache(value,helpers,budget):
    runtime=value['runtime'];cache=runtime['cache_policy'];fields={'mode','preloaded_module_cache_pins','preloaded_bootstrap_modules','preloaded_class_aliases','interpreter_profile','separate_main_source','source_execution_history_proven','namespace_roots','bootstrap_observation'}
    require(type(cache) is dict and fields<=set(cache)<=fields|{'additional_preload_sources'} and cache['mode']=='SOURCE_COMPILE_WITH_OBSERVED009_FINITE_CACHE_ASSUMPTIONS' and cache['source_execution_history_proven'] is False,'exact observed source/cache assumption schema')
    require(cache['namespace_roots']=={},'no unobserved namespace admission in no-site import scope')
    observation=cache['bootstrap_observation'];require(type(observation) is dict and set(observation)=={'path','file_sha256','body_sha256'},'independent observed009 selection')
    original=helpers.sealed_json(observation['path'],observation['file_sha256'],observation['body_sha256'],budget=budget)
    modules,pins,aliases=helpers.observed_bootstrap_maps(original);require(len(modules)==63 and len(pins)==59 and set(aliases)=={'typing.io','typing.re'},'actual original63/59/two alias populations')
    require(cache['preloaded_bootstrap_modules']==modules and cache['preloaded_module_cache_pins']==pins and cache['preloaded_class_aliases']==aliases,'exact observed bootstrap maps cannot be rebound')
    require(cache['interpreter_profile']==original['bootstrap']['interpreter_profile']==helpers.expected_no_site_profile(runtime['stdlib_layout']),'exact selected alias/profile assumptions')
    require(original['bootstrap']['tensor_prelude']['tensor_main_executed'] is False and '__main__' not in modules,'observer main never becomes new main witness')
    own=cache['separate_main_source'];require(type(own) is dict and set(own)=={'path','descriptor'} and own['path']==value['config']['main_source']==str(PACKAGE/'import_observation_worker.py') and value['config']['source_files'].get(own['path'])==own['descriptor'],'exact separate current import main source')
    files={**runtime['files'],**value['config']['source_files']};used=set()
    for name,row in modules.items():
        p=row['path'];require(p in files and files[p]['sha256']==row['source_sha256'] and Path(p).suffix in ('.py','.pyd'),'bootstrap exact selected source/extension')
        cp=row['cache_path']
        if cp is not None:require(cp in pins and files.get(cp)==pins[cp] and row['cache_sha256']==pins[cp]['sha256'] and Path(cp)==Path(p).parent/'__pycache__'/(Path(p).stem+'.cpython-311.pyc'),'exact selected3.11 cache association');used.add(cp)
        else:require(row['cache_sha256'] is None,'no phantom unobserved cache pin')
    require(used==set(pins),'no extra cache payloads')
    for alias in aliases.values():require(alias['owner_path'] in files and files[alias['owner_path']]['sha256']==alias['owner_sha256'],'finite alias source owner byte join')
    additional=cache.get('additional_preload_sources',{})
    require(type(additional) is dict and len(additional)<=128 and not set(additional).intersection(set(modules)|set(aliases)|{'__main__'}),'finite separate own-program preload population')
    if additional:
        reference=runtime['metadata_references'].get('additional_preload');require(type(reference) is dict and set(reference)=={'path','file_sha256','body_sha256'},'independent passive preload byte receipt selection')
        receipt=helpers.sealed_json(reference['path'],reference['file_sha256'],reference['body_sha256'],budget=budget);require(additional==passive_preload_overlay(receipt),'separate preload map equals source-derived passive receipt')
    for name,row in additional.items():
        require(type(name) is str and name and type(row) is dict and set(row)=={'path','descriptor','cached_attribute_present','cached_value_kind','cache_path','cache_descriptor','source_execution_history_proven'},'exact separate preload descriptor/cache schema')
        p=row['path'];require(files.get(p)==row['descriptor'] and row['source_execution_history_proven'] is False and Path(p).suffix in ('.py','.pyd'),'selected added preload source bytes, no history claim')
        require(type(row['cached_attribute_present']) is bool and row['cached_value_kind'] in ('missing','none','str'),'typed added cache shape')
        cp=row['cache_path']
        if cp is None:require(row['cache_descriptor'] is None and
            ((row['cached_value_kind']=='missing' and row['cached_attribute_present'] is False) or
             (row['cached_value_kind']=='none' and row['cached_attribute_present'] is True)), 'no invented added cache bytes/presence')
        else:require(files.get(cp)==row['cache_descriptor'] and row['cache_descriptor']['role']=='qualified_bytecode_cache' and row['cached_attribute_present'] is True and row['cached_value_kind']=='str' and Path(cp)==Path(p).parent/'__pycache__'/(Path(p).stem+'.cpython-311.pyc'),'exact added cache byte/path association')
def _qualify(value,budget,helpers):
    config,runtime,registration=value['config'],value['runtime'],value['registration']
    require(set(config)==CONFIG_FIELDS and set(runtime)==RUNTIME_FIELDS and set(registration)==REGISTRATION_FIELDS,'strict separate approved-run import schema')
    for v in (config,runtime,registration):_seal(v)
    require(config['kind']=='REGISTERED_OWNED_TORCH_IMPORT_OBSERVATION' and config['scope']==registration['scope']==SCOPE and config['permission']==PERMISSION and config['resources']==helpers.RESOURCES,'exact distinct import purpose and unchanged limits')
    require(config['native_launch_authorized'] is registration['native_launch_authorized'] is True and config['native_authority_issued'] is config['model'] is config['whole_program_entry_observed'] is False,'registered purpose never event/authority proof')
    require(type(config['schema_version']) is int and config['schema_version']==1 and registration['kind']=='REGISTERED_IMPORT_OBSERVATION_REVIEW' and runtime['kind']=='REGISTERED_IMPORT_RUNTIME_BYTE_CLOSURE','exact separate version/kinds')
    require(registration['profile_sha256']==helpers.digest({'scope':SCOPE,'permission':PERMISSION,'resources':helpers.RESOURCES}),'exact independent import purpose/profile digest')
    implementation=config['source_commit'];require(type(implementation) is str and len(implementation)==40 and all(c in '0123456789abcdef' for c in implementation) and implementation==runtime['source_commit']==registration['source_commit'] and implementation!=value['expected_head'],'implementation ancestor distinct external dispatch')
    require(config['source_files']==registration['source_files'] and config['runtime_files']==registration['runtime_files']==runtime['files'] and registration['config_body_sha256']==config['sha256'] and registration['runtime_body_sha256']==runtime['sha256'],'exact same maps/body joins')
    require(runtime['native_runtime_qualified'] is runtime['exact_effects_or_order_prequalified'] is False,'qualification checks bytes, not actualeffects/nativePASS')
    old=helpers.sealed_json(PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json','53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030','944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62',budget=budget);std=helpers.sealed_json(helpers.STDLIB_CANDIDATE,'e5e185f980e2f0feae7bf49384089eedd4d47f5b41c10a828c6ef2a30ae777a5','f6284b28e689b2c5e42a97c12256b3e35a41395a6b9612c0a01ee786eb28b149',budget=budget)
    require(runtime['inherited65_map']==old['runtime_pins'] and runtime['stdlib779_map']==std['inventory']['runtime_pins'] and len(runtime['inherited65_map'])==65 and len(runtime['stdlib779_map'])==779,'exact original65/779 maps')
    require(config['helper_pins']==old['helper_pins'] and config['offline_environment']==old['offline_environment'] and config['runtime_versions']==old['runtime_versions'] and config['torch_version']==old['runtime_versions']['torch'] and config['tolerances']==old['tolerances'] and type(config['GPU_physical_ordinal']) is int and config['GPU_physical_ordinal']==0,'unchanged helper/offline/runtime/numerical/NVML0 semantics')
    files=runtime['files'];sources=config['source_files'];require(0<len(files)<=32768 and 0<len(sources)<=4096,'bounded runtime/source population')
    require(runtime['stdlib_layout']==helpers.QUALIFIED_LAYOUT,'exact independently known interpreter alias layout')
    known={**runtime['inherited65_map'],**runtime['stdlib779_map']};require(all(p in files and files[p]['sha256']==h for p,h in known.items()),'historical known bytepins retained')
    _source_roster(value,helpers,budget);_cache(value,helpers,budget)
    observer=helpers.library_event_observer_registration(config,runtime,budget=budget);require(observer is not None,'exact registered observer/build source/ABI')
    windows=helpers._windows_library_identities(runtime,budget=budget);observer_files={observer['binary']['path']}
    allowed={'python_source':{'.py'},'python_stub':{'.pyi'},'source_json':{'.json'},'source_markdown':{'.md'},'native_observer_source':{'.c','.h'},'qualified_bytecode_cache':{'.pyc'},'extension_binary':{'.pyd','.dll'},'torch_binary':{'.pyd','.dll'},'interpreter_binary':{'.pyd','.dll','.exe','.zip'},'windows_system_binary':{'.dll'},'observer_binary':{'.dll'}}
    for mapping in (sources,files):
        for p,d in mapping.items():
            require(type(p) is str and 0<len(p)<=4096 and Path(p).is_absolute() and '..' not in Path(p).parts and str(Path(p))==p and type(d) is dict and set(d)=={'bytes','sha256','role'} and type(d['bytes']) is int and 0<=d['bytes']<=8*1024**3 and helpers.sha(d['sha256']) and d['role'] in allowed and Path(p).suffix.lower() in allowed[d['role']],'exact selected descriptor/role/suffix')
            require(d['bytes']>0 or d['role'] in ('python_source','python_stub','source_markdown'),'no empty binary/cache payload')
            if mapping is files:require(d['role']==('observer_binary' if p in observer_files else 'windows_system_binary' if p in windows else helpers.runtime_role(p,runtime['inherited65_map'])),'original exact root-role assignment')
            helpers.file_descriptor(p,d,layout=runtime['stdlib_layout'],budget=budget)
    require(not set(files).intersection(sources) or all(files[p]==sources[p] for p in set(files).intersection(sources)),'conflict-free source/runtime overlap')
    helpers._git_exception_policy(registration['git_output_exceptions'],sources,files)
    covered=set();rosters=runtime['rosters'];require(type(rosters) is dict and len(rosters)==18,'exact declared18 physical roots')
    for name,row in rosters.items():
        require(type(row) is dict and set(row)=={'root','suffixes','excluded_directory_names','recursive','files'},'exact runtime roster policy')
        root,suffixes,excluded,recursive=helpers.roster_policy(name);require((row['root'],row['suffixes'],row['excluded_directory_names'],row['recursive'])==(str(root),suffixes,excluded,recursive),'exact source-derived finite roots/filter')
        physical=helpers.physical_roster(root,suffixes,excluded,recursive=recursive,budget=budget);require(physical==row['files'] and all(p in files and files[p]['bytes']==d['bytes'] for p,d in physical.items()),'complete physical roster byte-map join');covered.update(physical)
    external=runtime['external_dependency_files'];single=str(helpers.SITE/'typing_extensions.py');require(type(external) is dict and all(files.get(p)==d for p,d in external.items()) and set(external)<=covered|{single},'exact external dependency roster/singleton')
    extra_cache={row['cache_path'] for row in runtime['cache_policy'].get('additional_preload_sources',{}).values() if row['cache_path'] is not None}
    require(set(files)==covered|set(known)|set(external)|set(windows)|set(runtime['cache_policy']['preloaded_module_cache_pins'])|extra_cache|observer_files,'exact no-extra runtime denominator')
    policy=runtime['measurement_policy'];grammar=runtime['measurement_grammar'];search=runtime['library_search_policy']
    require(type(search) is dict and set(search)=={'mode','roster_names','unregistered_search_allowed','enforced_boundary'} and search['mode']=='qualified_complete_registered_DLL_search_rosters' and search['unregistered_search_allowed'] is False and type(search['roster_names']) is list and search['roster_names'] and set(search['roster_names'])<=set(rosters) and search['enforced_boundary']==policy,'exact finite measurement search registration')
    pf={'kind','search_directories','system_directory','library_files','initial_loaded_libraries','extension_transitions','framework_transition','provider_source'}
    require(type(policy) is dict and set(policy)==pf and policy['kind']=='FINITE_WINDOWS_DLL_BOUNDARY_V2' and policy['framework_transition'] is None and type(policy['extension_transitions']) is dict,'separate measured-effects policy shape')
    require(type(policy['search_directories']) is list and 0<len(policy['search_directories'])<=128 and len(set(policy['search_directories']))==len(policy['search_directories']) and policy['system_directory']==str(helpers.WINDOWS_SYSTEM_DIRECTORY) and policy['system_directory'] in policy['search_directories'],'finite selected measurement directories')
    require(policy['library_files']==runtime['observed_library_allowed_files'] and all(files.get(p)==d and str(Path(p).parent) in policy['search_directories'] and Path(p).suffix.lower() in ('.dll','.pyd','.exe') for p,d in policy['library_files'].items()),'exact finite measurement allowed byte map')
    initial=policy['initial_loaded_libraries'];require(type(initial) is list and 35<=len(initial)<=128 and len(initial)==len(set(initial)) and all(p in policy['library_files'] for p in initial),'exact finite falsifiable startup expectation')
    # Historical observer35 are kept explicit; own-program preload additions
    # are separately byte-selected assumptions, never a renamed observation.
    require(type(runtime['initial35_expectation']) is list and len(runtime['initial35_expectation'])==len(set(runtime['initial35_expectation']))==35 and set(runtime['initial35_expectation'])<=set(initial),'historical35 preserved in own-program baseline expectation')
    extra=set(initial)-set(runtime['initial35_expectation']);additional=runtime['cache_policy'].get('additional_preload_sources',{});declared={row['path'] for row in additional.values() if Path(row['path']).suffix.lower()=='.pyd'}|set(windows)
    require(extra<=declared,'additional baseline binaries only separately selected preloads/Windows identities')
    if additional:
        reference=runtime['metadata_references']['additional_preload'];receipt=helpers.sealed_json(reference['path'],reference['file_sha256'],reference['body_sha256'],budget=budget);require(initial==receipt['source_derived_baseline_expectation'],'exact own-program39 baseline expectation from passive receipt')
    for p,row in policy['extension_transitions'].items():require(p in policy['library_files'] and Path(p).suffix.lower()=='.pyd' and type(row) is dict and set(row)=={'create','exec'} and row['create'] is row['exec'] is None,'unknown extension effects explicit, no guessed zero')
    provider=policy['provider_source'];require(type(provider) is dict and set(provider)=={'path','descriptor'} and sources.get(provider['path'])==provider['descriptor'],'exact measurement provider source join')
    gf={'schema_version','kind','required_extensions','optional_extensions','maximum_boundaries','maximum_depth','maximum_observations','model'};require(type(grammar) is dict and set(grammar)==gf and type(grammar['schema_version']) is int and grammar['schema_version']==1 and grammar['kind']=='FINITE_IMPORT_EFFECT_MEASUREMENT_GRAMMAR_V1' and grammar['model'] is False,'exact distinct measured-boundary grammar')
    req,opt=grammar['required_extensions'],grammar['optional_extensions'];require(type(req) is list and req and type(opt) is list and all(type(p) is str for p in req+opt) and len(req)==len(set(req)) and len(opt)==len(set(opt)) and not set(req).intersection(opt) and set(req)|set(opt)==set(policy['extension_transitions']) and type(grammar['maximum_boundaries']) is int and grammar['maximum_boundaries']==1+2*len(req+opt)<=2048 and type(grammar['maximum_depth']) is int and grammar['maximum_depth']==64 and type(grammar['maximum_observations']) is int and grammar['maximum_observations']==4096,'finite required/optional extension dose and original bounds')
    review=helpers.sealed_json(registration['qualification_review_path'],registration['qualification_review_file_sha256'],registration['qualification_review_body_sha256'],budget=budget)
    wanted={'kind':'ROOT_REVIEWED_IMPORT_RUNTIME_BYTE_ROSTER_SOURCE_CACHE_CLOSURE','source_commit':implementation,'source_inventory_sha256':helpers.digest(sources),'runtime_closure_sha256':helpers.digest(files),'rosters_sha256':helpers.digest(rosters),'cache_policy_sha256':helpers.digest(runtime['cache_policy']),'library_search_policy_sha256':helpers.digest(runtime['library_search_policy']),'measurement_policy_sha256':helpers.digest(runtime['measurement_policy']),'measurement_grammar_sha256':helpers.digest(runtime['measurement_grammar']),'complete_current_byte_review':True};require(review=={**wanted,'sha256':helpers.digest(wanted)},'independent complete byte/map/policy review joins')
    budget.check();return {'kind':'IMPORT_RUNTIME_BYTE_ROSTER_QUALIFICATION_ONLY','source_inventory_sha256':helpers.digest(sources),'runtime_inventory_sha256':helpers.digest(files),'current_module_execution_proven':False,'native_effects_or_authority_issued':False}
def qualify_import_value(value,budget=None):
    return _qualify(value,QualificationBudget() if budget is None else budget,B)
