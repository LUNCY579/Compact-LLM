"""CPU byte inventory preparation; never creates a native registration or permit."""
from copy import deepcopy
from pathlib import Path
import hashlib,importlib.metadata,json,math,os,sys
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='long_encoder_registration_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE
sys.path.insert(0,str(PACKAGE))
import long_native_contract as contract
from rs_contract import canonical,digest,require

KIND='STAGED_CPU_LONG_ENCODER_INVENTORY_NOT_NATIVE_REGISTRATION_V1'
EXPECTED_VERSIONS={'tokenizers':'0.22.2','jinja2':'3.1.6'}
PACKET_SHA=contract.PACKET_SHA
MAX_FILE=32*1024**2
SOURCE_NAMES=('long_native_contract.py','long_native_core.py','long_native_worker.py','long_native_supervisor.py',
    'long_inference_inputs.py','cpu_tokenizer.py','long_inference_component.py','rs_contract.py')
FORBIDDEN={'torch','transformers','tensorflow','jax','flax','tokenizers','jinja2','markupsafe'}


def no_execution():
    require(not any(name.split('.',1)[0] in FORBIDDEN for name in sys.modules),'inventory cannot contain framework or encoder packages')


def regular(path,root=None):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        require(not part.is_symlink(),'symlink pointer forbidden')
        if part.exists():require(not getattr(part.lstat(),'st_file_attributes',0)&0x400,'reparse pointer forbidden')
    require(path.is_file() and path.resolve()==path,'exact regular absolute file pointer')
    if root is not None:require(path.is_relative_to(Path(root).absolute()),'file outside declared package root')
    return path


def file_info(path,root=None):
    path=regular(path,root);size=path.stat().st_size
    require(0<size<=MAX_FILE,'bounded nonempty CPU inventory file')
    value=hashlib.sha256();read=0
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(1024**2),b''):value.update(raw);read+=len(raw)
    require(read==size==path.stat().st_size,'file changed while hashing')
    return {'path':str(path),'bytes':size,'sha256':value.hexdigest()}


def runtime_roster(distribution=importlib.metadata.distribution):
    """The integrated contract's exact .py/.pyd superset, with physical closure."""
    no_execution();versions={};packages={};roster={}
    for name in ('tokenizers','jinja2','markupsafe'):
        dist=distribution(name);version=dist.version
        require(type(version) is str and bool(version) and (name not in EXPECTED_VERSIONS or version==EXPECTED_VERSIONS[name]),'exact installed encoder version')
        base=Path(dist.locate_file(name)).absolute();require(base.is_dir() and not base.is_symlink(),'actual declared package directory')
        declared=[]
        for entry in dist.files:
            text=str(entry)
            if not text.endswith(('.py','.pyd')) or '..' in Path(text).parts:continue
            path=regular(dist.locate_file(entry),base);declared.append(path)
            require(str(path) not in roster,'duplicate executable runtime path')
            roster[str(path)]=name
        require(declared,'nonempty executed package source roster')
        # Physical Python/native files may not silently escape RECORD inventory.
        physical={str(regular(path,base)) for path in base.rglob('*') if path.suffix in ('.py','.pyd')}
        require(physical=={str(path) for path in declared},'exact declared executable superset; omitted or extra package file')
        versions[name]=version;packages[name]={'root':str(base),'version':version,'files':sorted(str(path) for path in declared)}
    return {'versions':versions,'packages':packages,'paths':dict(sorted(roster.items()))}


def runtime_inventory(distribution=importlib.metadata.distribution):
    roster=runtime_roster(distribution);files={path:file_info(path,roster['packages'][package]['root']) for path,package in roster['paths'].items()}
    return {**roster,'files':files,'runtime_pins':{path:value['sha256'] for path,value in files.items()}}


def validate_runtime(value,distribution=importlib.metadata.distribution):
    actual=runtime_inventory(distribution)
    require(value==actual,'current exact encoder version/path/roster/byte pins changed; stale candidate')
    return True


def build_candidate():
    no_execution();runtime=runtime_inventory()
    require(set(runtime['runtime_pins'])==set(contract.encoder_runtime_paths()),'integrated required encoder schema unchanged')
    packet_path=regular(contract.PACKET);packet_info=file_info(packet_path)
    require(packet_info['sha256']==PACKET_SHA,'reviewed input packet byte pin')
    packet=json.loads(packet_path.read_bytes(),object_pairs_hook=contract.native._pairs,parse_constant=contract.native._bad_constant)
    require(packet['sha256']==contract.PACKET_BODY==digest({k:v for k,v in packet.items() if k!='sha256'}),'reviewed packet body')
    binding=packet['tokenizer_binding'];require(binding['dependency_versions']==EXPECTED_VERSIONS
        and binding['native_qualification'] is False and binding['training_admitted'] is False
        and binding['template_kwargs']=={'add_generation_prompt':True,'enable_thinking':False},'unchanged closed prompt encoder policy')
    require(set(binding['assets'])=={'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'},'exact token/template asset roster')
    assets={}
    for name,want in binding['assets'].items():
        actual=file_info(Path(binding['model_dir'])/name)
        require({key:actual[key] for key in ('bytes','sha256')}==want,'pinned prompt asset bytes')
        assets[name]=actual
    cfg=json.loads(Path(assets['tokenizer_config.json']['path']).read_bytes(),object_pairs_hook=contract.native._pairs)
    template=cfg['chat_template'];require(type(template) is str and hashlib.sha256(template.encode()).hexdigest()==binding['chat_template_sha256'],'exact pinned template UTF8')
    bindings={path:file_info(path) for path in binding['binding_sources']}
    require(all(value['sha256']==binding['binding_sources'][path] for path,value in bindings.items()),'prompt helper binding receipt bytes')
    source={str((PACKAGE/name).resolve()):file_info(PACKAGE/name) for name in SOURCE_NAMES}
    source[str(Path(__file__).resolve())]=file_info(__file__)
    source[str(packet_path)]=packet_info
    # These source metadata anchors were already retained in the CPU packet.
    origins={path:file_info(path) for path in packet['source_before']}
    require(packet['source_before']==packet['source_after'] and all(origins[path]==want for path,want in packet['source_before'].items()),'original prompt preparation/source pins')
    body={'schema_version':1,'kind':'SYNTH_LONG_ENCODER_RUNTIME_REGISTRATION_V1','runtime_pins':runtime['runtime_pins'],
        'dependency_versions':runtime['versions'],'executing_package_roster':runtime['packages'],'prompt_assets':assets,
        'chat_template_sha256':binding['chat_template_sha256'],'binding_sources':bindings,'source_pins':source,
        'input_packet':packet_info,'prompt_source_origins':origins,'tokenizer_binding_sha256':digest(binding),
        'runtime_evidence_scope':'exact tokenizers/Jinja2/MarkupSafe executable package superset only',
        'complete_runtime_execution_closure_established':False,
        'required_additional_policy':'reviewed additive stdlib/importloader executable byte roster consumed by native contract/worker',
        'native_launch_authorized':False,'native_qualification_established':False,'study_training_authorized':False,'scoring_authorized':False}
    body['sha256']=digest(body)
    result={'schema_version':1,'kind':KIND,'status':'CPU_ENCODER_BYTE_ROSTER_CANDIDATE_PENDING_INDEPENDENT_REVIEW',
        'runtime_inventory':runtime,'native_registration_preview':body,'scope':'byte hashing and installed metadata only; no encoder execution',
        'complete_runtime_execution_closure_established':False,
        'native_registration_written':False,'native_config_written':False,'native_authority_issued':False}
    result['sha256']=digest(result);no_execution();return result


def verify_candidate(candidate,expected_sha256):
    require(type(expected_sha256) is str and len(expected_sha256)==64 and candidate.get('sha256')==expected_sha256
        ==digest({k:v for k,v in candidate.items() if k!='sha256'}),'externally selected candidate body hash required; no self-signed authority')
    require(candidate.get('kind')==KIND and candidate.get('native_registration_written') is False
        and candidate.get('native_config_written') is False and candidate.get('native_authority_issued') is False
        and candidate.get('complete_runtime_execution_closure_established') is False,'staged package-superset candidate only')
    require(candidate==build_candidate(),'entire current installed/source/asset/template/packet candidate changed')
    return True


def save_staged_candidate(path,value):
    target=Path(path).absolute();require(STAGED and target.parent==HERE and target.suffix=='.json'
        and not target.exists() and not target.is_symlink(),'new staging-only candidate file; never final native registration')
    raw=(canonical(value)+'\n').encode()
    with target.open('xb') as stream:require(stream.write(raw)==len(raw),'complete candidate write');stream.flush();os.fsync(stream.fileno())
    return file_info(target)
