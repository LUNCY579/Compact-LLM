"""Reviewed admitted byte observer; does not call or weaken CPU no_execution."""
from pathlib import Path
from copy import deepcopy
import hashlib,importlib.metadata,json,math,sys
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='encoder_runtime_v2_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE
sys.path.insert(0,str(PACKAGE))
import stdlib_inventory as preparation
from rs_contract import digest,require

POLICY='ADMITTED_ENCODER45_FINITE_STDLIB_BYTE_OBSERVATION_V2'
PURPOSE='TRAIN_long_split_fresh_native_encoder_stdlib_V2_qualification_only'


def validate_admission(admission,phase,recipe_sha256):
    require(not STAGED,'staged admitted byte observation forbidden before paths/probes')
    require(phase in ('before_import','after_math'),'explicit admitted byte phase')
    require(type(recipe_sha256) is str and len(recipe_sha256)==64 and all(c in '0123456789abcdef' for c in recipe_sha256),
        'typed admitted recipe binding')
    require(type(admission) is dict and admission.get('owned_supervision_verified') is True
        and admission.get('model_load_and_constructor_in_scope') is True
        and admission.get('binding')=={'purpose':PURPOSE,'recipe_sha256':recipe_sha256,
            'role':admission.get('binding',{}).get('role'),'seed':101}
        and admission['binding']['role'] in ('trace1','trace2','bulk','prefix_stress'),
        'owned exact purpose/recipe/role/seed binding before byte paths')


def stdlib_roster(expected):
    """Pure metadata enumeration; safe with already-loaded model/encoder modules."""
    require(expected['fixture'] is False and expected['policy']==preparation.VERSION
        and expected['python_version']==preparation.PYTHON_VERSION
        and expected['base_prefix']==str(preparation.BASE.absolute())==str(Path(sys.base_prefix).absolute())
        and expected['executable']==str(Path(sys.executable).absolute()) and sys.version.split()[0]==preparation.PYTHON_VERSION,
        'actual pinned nonfixture interpreter and policy')
    alias=preparation.alias_observation();preparation.validate_alias(alias)
    require(alias==expected['approved_alias_binding'],'exact admitted interpreter alias identity')
    base=preparation.pointer(preparation.PHYSICAL_BASE);paths={}
    def add(path,category):
        key=str(path);require(key not in paths,'unique admitted runtime path');paths[key]=category
    for path in preparation.recursive(base/'Lib',{'.py'},('site-packages','__pycache__')):add(path,'stdlib_python_source')
    for path in preparation.recursive(base/'DLLs',{'.pyd','.dll'}):add(path,'stdlib_extension_runtime')
    for path in sorted(base.iterdir()):
        if path.suffix.lower() in ('.exe','.dll','.zip'):add(preparation.pointer(path,file=True,root=base),'interpreter_root_executable_runtime')
    executable=preparation.pointer(sys.executable,file=True)
    if str(executable) not in paths:add(executable,'executing_Hermes_launcher')
    optional={'path':str(base/'python311.zip'),'present':(base/'python311.zip').exists()}
    require(paths==expected['paths'] and optional==expected['optional_stdlib_zip'],'exact admitted finite roster/optionalZIP')
    return {'paths':dict(sorted(paths.items())),'approved_alias_binding':alias,'optional_stdlib_zip':optional}


def encoder_roster(expected):
    paths={}
    for name,entry in expected['packages'].items():
        require(name in ('tokenizers','jinja2','markupsafe'),'exact admitted encoder package')
        dist=importlib.metadata.distribution(name)
        require(dist.version==expected['versions'][name]==entry['version'],'exact admitted encoder package version')
        base=Path(dist.locate_file(name)).absolute();require(str(base)==entry['root'],'admitted encoder package root')
        current=[]
        for record in dist.files:
            text=str(record)
            if text.endswith(('.py','.pyd')) and '..' not in Path(text).parts:
                path=preparation.pointer(dist.locate_file(record),file=True,root=base);current.append(str(path));paths[str(path)]=name
        physical={str(path) for path in preparation.recursive(base,{'.py','.pyd'})}
        require(set(current)==physical==set(entry['files']),'exact admitted package executable superset')
    require(paths==expected['paths'],'all admitted encoder paths unchanged')
    return dict(sorted(paths.items()))


def observe(registration,call,guard,admission,*,phase,recipe_sha256):
    validate_admission(admission,phase,recipe_sha256)
    require(callable(call) and callable(guard),'counted admitted byte operations')
    std=registration['stdlib_inventory'];encoder=registration['encoder_inventory']
    before=call('v2_%s_stdlib_roster'%phase,lambda:stdlib_roster(std))
    call('v2_%s_encoder_roster'%phase,lambda:encoder_roster(encoder))
    actual={}
    require(0<len(registration['additional_runtime_pins'])<=preparation.MAX_FILES,'finite admitted runtime map')
    for path,want in registration['additional_runtime_pins'].items():
        path=Path(path)
        def hashed(path=path):
            preparation.pointer(path,file=True)
            sha=hashlib.sha256();read=0;size=path.stat().st_size
            require(size<=preparation.MAX_FILE and (size>0 or path.suffix=='.py'),'admitted bounded runtime file')
            with path.open('rb') as stream:
                for data in iter(lambda:stream.read(1024**2),b''):
                    guard('v2_runtime_chunk_before');sha.update(data);read+=len(data);guard('v2_runtime_chunk_after')
            require(read==size==path.stat().st_size,'admitted file immutable during hash')
            preparation.pointer(path,file=True)
            return sha.hexdigest()
        value=call('v2_%s_runtime_file_verify'%phase,hashed);require(value==want,'admitted runtime byte drift')
        actual[str(path)]=value
    after=call('v2_%s_stdlib_roster_post'%phase,lambda:stdlib_roster(std))
    call('v2_%s_encoder_roster_post'%phase,lambda:encoder_roster(encoder))
    require(before==after,'admitted alias/roster stable through hashes')
    return {'policy':POLICY,'phase':phase,'actual_additional_runtime_pins':actual,'stdlib_roster':after,
        'encoder_versions':deepcopy(encoder['versions']),'native_qualification_established':False,
        'complete_machine_trust_claimed':False}
