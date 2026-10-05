"""Distinct staged native-long purpose; preparation grants no dispatch authority."""
from copy import deepcopy
from pathlib import Path
import importlib.metadata
import os
import sys

HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='encoder_runtime_v2_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
import native_contract as native
import inference_contract as inference
import backend_qual_contract as prerequisite
from native_loader import ASSET_PINS,MODEL_DIR
from rs_contract import canonical,digest,require

PURPOSE='TRAIN_long_split_fresh_native_encoder_stdlib_V2_qualification_only'
VERSION='synth-long-native-split-encoder-stdlib-qualified-scope-2'
KIND='SYNTH_LONG_NATIVE_QUALIFICATION_CAPSULE_V2'
AUDIT_KIND='SYNTH_LONG_NATIVE_SCOPE_INDEPENDENT_AUDIT_V2'
NATIVE='UNQUALIFIED_NATIVE_LONG_IMPLEMENTATION'
FAKE='CPU_FAKE_NATIVE_LONG_PROTOCOL'
FILES=('long_native_contract.py','long_native_core.py','long_native_worker.py','long_native_supervisor.py')
PACKET=PACKAGE/'evidence/long_train_prompts_cpu_20261005/TRAIN_PROMPTS_CPU.json'
PACKET_SHA='208f3533fdc22e01cd345e8a746be21cad67681da734814629e64426d497edfe'
PACKET_BODY='4a58b2c32357533da723cd1fb46b0d938eff11f014c70104784f250fe20b6fed'
DERIVED_SHA='5816724569a4e5a553428e6a05a0297fb928bbdf3de84c714566b40dba93d327'
AUTHORIZATION={'scope':PURPOSE,'production_training':False,'study_training':False,'study_warmstart':False,
    'DEV_collection':False,'guard_enrollment':False,'scoring':False,'protected_bank_access':False,'paid_services':False}
ROLES=('trace1','trace2','bulk','prefix_stress')


def source_pins():
    names=('native_contract.py','native_loader.py','native_precision.py','native_mechanics.py','native_process.py','native_nvml.py',
        'native_supervisor.py','native_worker.py','inference_contract.py','inference_component.py','inference_qual_worker.py','long_inference_inputs.py',
        'cpu_tokenizer.py','long_inference_component.py','backend_qual_contract.py','study_schedule.py','encoder_inventory.py','stdlib_inventory.py','rs_contract.py')
    paths=[PACKAGE/n for n in names]+[HERE/n for n in FILES]+[HERE/'encoder_registration_v2.py',HERE/'admitted_runtime_observer.py']
    tests=HERE if STAGED else PACKAGE/'tests'
    paths+=list(tests.glob('test_long_native*.py'))
    require(all(p.is_file() and not p.is_symlink() for p in paths),'complete reviewed long-native source files')
    return {str(p.resolve()):native.filehash(p) for p in paths}


def validate_source_pins(value):
    require(value==source_pins(),'complete exact executing package source closure')
    return True


def base_runtime_pins():
    # Retain every inherited Native7 pin; backend_qual_contract imports ast.
    # This additive source hash reads ast.py only, never a runtime binary here.
    prior=native.read_json(PACKAGE/'configs/SYNTH_NQ001_PROC7.json')['runtime_pins']
    return {**deepcopy(prior),**prerequisite.worker_ram_runtime_files()}


def validate_runtime_pins(value,encoder=None):
    expected=base_runtime_pins()
    if encoder is not None:expected.update(encoder['runtime_pins'])
    require(value==expected,'all inherited64 runtime pins plus exact additive AST and encoder closure')
    return True


def selected(cap,prefix_stress=None):
    require(type(cap) is int and cap in (1024,2048,4096),'exact registered long context')
    require(native.filehash(PACKET)==PACKET_SHA,'reviewed TRAIN packet bytes')
    packet=native.read_json(PACKET)
    require(packet['sha256']==PACKET_BODY==digest({k:v for k,v in packet.items() if k!='sha256'}),'TRAIN packet body')
    require(native.filehash(PACKAGE/'long_inference_component.py')==DERIVED_SHA,'unchanged derived long math')
    item=next(value for value in packet['rows'] if value['context_cap']==cap)
    require(item['source_arm']=='M1' and item['source_split']=='train' and item['source_skill']=='R'
        and digest(item['messages'])==item['row']['prompt_sha256'],'prompt-only TRAIN/messages source')
    if prefix_stress is not None:
        require(type(prefix_stress) is list and len(prefix_stress)==31 and all(type(t) is int and 0<=t<151936
            and t not in (151643,151645) for t in prefix_stress),'predeclared nonEOS stress tokens')
        ids=item['row']['prompt_ids'];require(any(ids[i:i+31]==prefix_stress for i in range(len(ids)-30)),'TRAIN-only contiguous stress span')
    return packet,deepcopy(item)


def work(row,role):
    require(role in ROLES,'one fixed long scope role');p=row['prompt_tokens']
    return {'context_cap':p+32,'generation_cap':32,'parity_calls':2,'parity_positions':838,
        'maximum_forward_calls':3 if role=='bulk' else 34,
        'maximum_forward_input_positions':p+31+838 if role=='bulk' else 32*p+496+838,
        'hard_position_ceiling':134144,'whole_scope_seconds':600,'backwards':0,'optimizer_steps':0}


def encoder_runtime_paths():
    """Roster only, no runtime or encoder binary reads during preparation."""
    result=[]
    for name,version in (('tokenizers','0.22.2'),('jinja2','3.1.6'),('markupsafe',None)):
        package=importlib.metadata.distribution(name)
        require(version is None or package.version==version,'encoder dependency version')
        result += [str(Path(package.locate_file(entry)).resolve()) for entry in package.files
                   if str(entry).endswith(('.py','.pyd')) and '..' not in Path(str(entry)).parts]
    return sorted(set(result))


def encoder_registration(value):
    import encoder_registration_v2 as v2
    return v2.registration(value,base_runtime_pins())


def build_scope(cap,role,*,references=None,prefix_stress=None,encoder=None,authorize=False):
    require(type(authorize) is bool and role in ROLES,'explicit fixed scope authorization')
    packet,item=selected(cap,prefix_stress);dose=work(item['row'],role)
    require(dose['maximum_forward_input_positions']<=134144 and (role!='prefix_stress' or prefix_stress is not None),'unchanged hard ceiling/honest stress role')
    require((role=='trace1' and references is None) or (role!='trace1' and type(references) is list),'freshfirst or explicit audited lineage')
    prior=native.read_json(PACKAGE/'configs/SYNTH_NQ001_PROC7.json');sources=source_pins()
    encoding=None if encoder is None else encoder_registration(encoder)
    runtime=base_runtime_pins()
    if encoding is not None:
        from encoder_registration_v2 import merge_pins
        runtime=merge_pins(runtime,encoding['runtime_pins'])
    identity={'purpose':PURPOSE,'context_cap':cap,'packet_sha256':PACKET_SHA,'source_pins':sources,'runtime_pins':runtime,
        'model_pins':dict(ASSET_PINS),'tokenizer_binding':deepcopy(packet['tokenizer_binding']),'seed':101,
        'precision':deepcopy(prior['precision']),'tolerances':deepcopy(native.TOLERANCES),'prefix_stress_ids':deepcopy(prefix_stress)}
    identity['qualified_short_helper_prerequisite']=prerequisite.inference_binding()
    identity['declared_execution_scope_registration_sha256']=None if encoding is None else encoding['sha256']
    identity['sha256']=digest(identity)
    return {'schema_version':1,'contract':VERSION,'purpose':PURPOSE,'authorization':deepcopy(AUTHORIZATION),
        'owned_qualification_authorized':authorize,'staged_native_dispatch_forbidden':STAGED,'recipe':identity,
        'role':role,'seed':101,'row':item['row'],'messages':item['messages'],'short_parity_row':packet['short_parity_row'],
        'prompt_hash_conventions':{'short':'SHA256_RENDERED_UTF8_CHAT_TEMPLATE','long':'SHA256_CANONICAL_MESSAGES_JSON'},
        'source_item':item,'dose':dose,'source_pins':sources,'runtime_pins':runtime,'model_pins':dict(ASSET_PINS),
        'model_dir':str(MODEL_DIR),'encoder_registration':deepcopy(encoder),'encoder_runtime_registration':encoding,
        'tokenizer_binding':deepcopy(packet['tokenizer_binding']),'packet_path':str(PACKET),'packet_sha256':PACKET_SHA,
        'data_pins':deepcopy(prior['data_pins']),'runtime_versions':deepcopy(native.RUNTIME),'offline_environment':deepcopy(native.OFFLINE_ENV),
        'precision':deepcopy(prior['precision']),'resources':deepcopy(native.RESOURCE_BOUNDS),'tolerances':deepcopy(native.TOLERANCES),
        'references':deepcopy(references),'native_launch_authorized':False,'native_long_context_qualified':False,
        'study_training_authorized':False,'study_warmstart_allowed':False,'guard_enrollment_authorized':False,'scoring_authorized':False}


def preflight(path):
    require(not STAGED,'staged native-long dispatch forbidden before APIs')
    filename=Path(path).absolute();require(filename.resolve().parent==(PACKAGE/'configs').resolve() and filename.is_file()
        and not filename.is_symlink(),'owned root long qualification config')
    body=native.read_json(filename);require(body['sha256']==digest({k:v for k,v in body.items() if k!='sha256'})
        and body['owned_qualification_authorized'] is True,'explicit owned checksum')
    expected=build_scope(body['recipe']['context_cap'],body['role'],references=body['references'],
        prefix_stress=body['recipe']['prefix_stress_ids'],encoder=body['encoder_registration'],authorize=True)
    require(body['encoder_registration'] is not None and canonical({k:v for k,v in body.items() if k!='sha256'})==canonical(expected),
        'complete encoder registration/exact static source/input/authority contract')
    return {**body,'config_path':str(filename.resolve()),'config_sha256':native.filehash(filename),
        'source_commit':inference._committed_checkout(True),'worker_path':str(PACKAGE/'long_native_worker.py')}


def write_scope(path,cap,role,**kwargs):
    require(not STAGED,'staged native-long cannot create a dispatch manifest')
    require(kwargs.get('authorize') is True and kwargs.get('encoder') is not None,'explicit owned authorization and reviewed encoder registration')
    target=Path(path).absolute();require(target.resolve().parent==(PACKAGE/'configs').resolve() and not target.exists()
        and not target.is_symlink(),'new write-once root config')
    inference._committed_checkout(True);body=build_scope(cap,role,**kwargs);body['sha256']=digest(body)
    with target.open('xb') as stream:raw=(canonical(body)+'\n').encode();require(stream.write(raw)==len(raw),'complete scope write');stream.flush();os.fsync(stream.fileno())
    return body
