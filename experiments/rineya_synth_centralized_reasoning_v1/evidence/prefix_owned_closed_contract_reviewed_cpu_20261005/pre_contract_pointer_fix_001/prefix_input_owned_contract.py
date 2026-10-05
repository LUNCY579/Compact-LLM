"""Separate tensor-purpose candidate contract. All native entry points closed.

This validates registration metadata only. No worker/supervisor/terminal auditor
has been implemented; no field supplied by a caller can open this stage gate.
"""
from pathlib import Path
import hashlib, importlib.util
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
HELPER=HERE/'runtime_candidate_002.py'
HELPER_SHA='79ccd72cc60268181e3cf7b5ad57b8295289ee55f807c77a31773511f6d015cf'
CANDIDATE=HERE/'RUNTIME_CANDIDATE_002.json'
CANDIDATE_FILE='93e4fe7996a9b39963f72c3d433f13ec97dfcde655bb9186fe5995c597fbb87d'
CANDIDATE_BODY='049747fa477da33363affdff23f7a52cd32ec0bc5ec1b786a6c4bbe6e30732fe'
INPUT=PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json'
INPUT_FILE='195f1b3a658959ccbe517ce5aefaef1d0d2c298db0095ace08f744bd6488feb5'
INPUT_BODY='a22c5b27225fc75ae697a5f06a96da18c1cb995a7bf7c6d03eb05f0cb6f06ec9'
PREFIX_SHA='e806d781c2c21c15180b6ec1bc714ed5acc7104948c2859ec6b411761a4a19d1'
FRAME=PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/exception_frames.py'
FRAME_SHA='253caa71942c300c225abef9f33d09f2ac28c4e6f169570ecbea09b926817d0b'
CORE=PACKAGE/'evidence/prefix_input_core_reviewed_cpu_20261005'
CORE_PINS={'contract.py':'b83228e3ae797d2a410b459f73662daba75ecb6f203f3edd8edaf206eb757836',
 'pipeline.py':'700d6934e0fb36b158eacfe5f53330e5e498d9fd9a2bf264b40a077c95bbb9c4',
 'fixture_support.py':'a8d42ca5bb1f481c54db5a76e9eaf9605d486e429a7f422211529d3194426eff',
 'test_prefix_input.py':'9fc7797cde34a1c4c48334b38857b7f682f0ec9e8c1584bca1d2b10563e04924'}
PURPOSE='ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY'
CLOSED=('Torch_imported','binary_model_payload_bytes_read','OS_GPU_resource_probes_performed',
 'native_runtime_closure_complete','Root_binary_qualification_complete','cache_execution_policy_qualified',
 'external_dependencies_qualified','worker_consumption_verified','independent_review_complete','native_launch_authorized')

def require(value,message):
    if not value:raise ValueError(message)

def source_tools():
    # Execute exact reviewed UTF8 source, never a cached pyc/module alias.
    before=HELPER.lstat();require(HELPER.is_file() and not HELPER.is_symlink() and before.st_size<=65536,'bounded exact source helper')
    with HELPER.open('rb') as stream:raw=stream.read(before.st_size+1)
    require(len(raw)==before.st_size and hashlib.sha256(raw).hexdigest()==HELPER_SHA,'exact source helper bytes')
    spec=importlib.util.spec_from_file_location('prefix_owned_runtime_candidate_exact_source',HELPER)
    module=importlib.util.module_from_spec(spec);exec(compile(raw.decode('utf-8'),str(HELPER),'exec'),module.__dict__)
    return module

def validate_candidate(value,tools,inherited,stdlib):
    require(type(value) is dict and value.get('schema_version')==2 and value.get('kind')=='SOURCE_ONLY_FINITE_TORCH_RUNTIME_CANDIDATE_INCOMPLETE_NATIVE_CLOSURE','exact candidate identity')
    require(value.get('sha256')==CANDIDATE_BODY==tools.digest({k:v for k,v in value.items() if k!='sha256'}),'externally selected candidate body')
    require(all(value.get(key) is False for key in CLOSED),'all candidate authority remains closed')
    require(value['inherited65_map']==inherited['runtime_pins'] and value['stdlib779_inventory']==tools.validate_stdlib(stdlib),'exact original maps, not counts')
    require(value['inherited65_anchor']['file_sha256']==tools.INHERITED_FILE and value['inherited65_anchor']['body_sha256']==tools.INHERITED_BODY
            and value['stdlib779_anchor']['file_sha256']==tools.STDLIB_FILE and value['stdlib779_anchor']['body_sha256']==tools.STDLIB_BODY,'exact external original anchors')
    metadata=value['torch_file_metadata'];sources=tools.pin_map(value['torch_source_hashes']);binary=value['torch_UNREAD_binary_obligations'];root=Path(value['torch_physical_root'])
    require(type(metadata) is dict and 0<len(metadata)<=tools.MAX_FILES,'finite executable roster')
    wanted_sources=set();wanted_binary=set();total=0
    for path,entry in metadata.items():
        require(type(path) is str and 0<len(path)<=4096 and Path(path).is_relative_to(root) and '__pycache__' not in Path(path).parts,'contained executable roster without caches')
        suffix=Path(path).suffix;require(suffix in tools.EXECUTABLE_SUFFIXES,'exact executable suffixes')
        require(type(entry) is dict and set(entry)=={'bytes','role'} and type(entry['bytes']) is int and entry['bytes']>=0,'typed executable metadata')
        if suffix in tools.SOURCE_SUFFIXES:
            require(entry['bytes']<=tools.MAX_SOURCE_FILE and entry['role']==('torch_python_source' if suffix=='.py' else 'torch_python_stub_source'),'source role/size')
            wanted_sources.add(path);total+=entry['bytes']
        else:
            require(entry['role']=='torch_binary_UNREAD_METADATA_ONLY','unread binary role');wanted_binary.add(path)
    require(set(sources)==wanted_sources and set(binary)==wanted_binary and total==value['source_bytes']<=tools.MAX_SOURCE_TOTAL,'exact source/binary population and bytes')
    for path in wanted_binary:
        entry=binary[path];require(entry['bytes']==metadata[path]['bytes'] and entry['role']==metadata[path]['role'] and entry['current_binary_bytes_hashed'] is False
            and entry['Root_qualified_byte_verification_required'] is True and entry['expected_inherited_sha256']==inherited['runtime_pins'].get(path),'unread binary denominator and recorded provenance')
        record=entry['installation_record']
        require(record is None or record.get('current_binary_bytes_verified') is False,'RECORD observation does not verify current binaries')
    union=tools.merge_exact(inherited['runtime_pins'],stdlib['inventory']['runtime_pins'],sources)
    require(value['expected_known_pin_union']==union and value['expected_known_pin_union_sha256']==tools.digest(union),'exact conflict-aware union')
    require(value['inherited65_map_sha256']==tools.digest(inherited['runtime_pins']) and value['stdlib779_map_sha256']==tools.digest(stdlib['inventory']['runtime_pins']),'exact inherited map digests')
    require(value['finite_roster_policy']=={'suffixes':sorted(tools.EXECUTABLE_SUFFIXES),'excluded_directory_names':['__pycache__'],'pyc_in_payload_roster':False,'other_Torch_package_data_not_in_executable_roster':True},'explicit unresolved executable/cache policy')
    require(set(value['bootstrap_source_import_observations'])==set(tools.BOOTSTRAP),'exact bootstrap source observations')
    for name,entry in value['bootstrap_source_import_observations'].items():
        path=str(root/Path(name));require(entry['path']==path and entry['source_sha256']==sources[path],'bootstrap source pin join')
    require(value['source_before_after_equal'] is value['metadata_before_after_equal'] is value['roles_are_superset_not_actual_execution_proof'] is True,'source observation status only')
    return union

def registration_candidate():
    tools=source_tools();candidate,_=tools.json_source(CANDIDATE,CANDIDATE_FILE,CANDIDATE_BODY)
    inherited,_=tools.json_source(tools.INHERITED,tools.INHERITED_FILE,tools.INHERITED_BODY)
    stdlib,_=tools.json_source(tools.STDLIB,tools.STDLIB_FILE,tools.STDLIB_BODY)
    union=validate_candidate(candidate,tools,inherited,stdlib)
    supplied,_=tools.json_source(INPUT,INPUT_FILE,INPUT_BODY);ids=supplied['prefix_ids']
    require(type(ids) is list and len(ids)==422 and all(type(x) is int and 0<=x<151936 for x in ids)
            and supplied['attention_mask']==[1]*422 and tools.digest(ids)==PREFIX_SHA,'exact exposed prefix and mask')
    require(supplied['native_launch_authorized'] is supplied['M0_population_membership_allowed'] is supplied['training_or_grading_allowed'] is False,'input carries no execution/study authority')
    for path,pin in [(FRAME,FRAME_SHA),*((CORE/name,pin) for name,pin in CORE_PINS.items())]:
        raw=tools.bounded_bytes(path,tools.MAX_SOURCE_FILE);require(hashlib.sha256(raw).hexdigest()==pin,'durable reviewed source dependency')
    value={'schema_version':1,'kind':'STAGED_TENSOR_ONLY_OWNED_REGISTRATION_CANDIDATE_NOT_NATIVE_AUTHORITY','purpose':PURPOSE,
        'authorization':{'tensor_permission':{'allocations':2,'dtype':'int64','shape':[1,422],'logical_bytes':6752},'no_model':True,
            'forward':False,'backward':False,'optimizer':False,'decoder':False,'scoring':False,'M0_population_membership':False,'native_launch_authorized':False},
        'input_anchor':{'file_sha256':INPUT_FILE,'body_sha256':INPUT_BODY,'prefix_ids_sha256':PREFIX_SHA},
        'frame_helper_source_sha256':FRAME_SHA,'CPU_core_source_pins':CORE_PINS,
        'dose':{'success_operations':35,'guard_sites':8,'maximum_nested_guard_queries':14,'optional_first_error_frames':1,'maximum_phases':58,'maximum_markers':116,'hard_marker_limit':128,'logical_tensor_bytes_not_allocator_budget':6752},
        'resources':inherited['resources'],'tolerances':inherited['tolerances'],'offline_environment':inherited['offline_environment'],
        'runtime_candidate_anchor':{'file_sha256':CANDIDATE_FILE,'body_sha256':CANDIDATE_BODY},'expected_known_runtime_pin_union':union,
        'native_runtime_closure_complete':False,'actual_worker_supervisor_auditor_implemented':False,'native_entry_open':False,
        'remaining_gates':['exact current binary/cache/external-dependency/layout qualification','frozen new source/HEAD/config/profile/inventory with independent review','new actual owned worker/supervisor and independent typed-operation/resource/cleanup/byte terminal audit'],
        'status':'CPU_SOURCE_METADATA_VALIDATED_NATIVE_CLOSED','native_fit_or_failure_cause_claimed':False}
    value['sha256']=tools.digest(value);return value

def native_entry(*args,**kwargs):raise ValueError('staged contract: native worker/paths/probes unavailable before separate reviewed integration')
def preflight_native(*args,**kwargs):return native_entry()
def write_owned_registration(*args,**kwargs):return native_entry()
if __name__=='__main__':native_entry()
