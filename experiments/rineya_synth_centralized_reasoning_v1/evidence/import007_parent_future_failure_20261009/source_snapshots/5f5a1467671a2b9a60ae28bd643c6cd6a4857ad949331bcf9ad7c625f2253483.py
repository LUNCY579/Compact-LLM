"""Explicit additive encoder/finite-stdlib V2 binding; no final registration writer."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,sys,math
HERE=Path(__file__).resolve().parent
STAGED=HERE.name in ('long_native_qualification_work','encoder_runtime_v2_work') and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
import encoder_inventory as encoder_prep
import stdlib_inventory as stdlib_prep
from rs_contract import canonical,digest,require
from rs_contract import _pairs,_bad_constant

KIND='SYNTH_LONG_ENCODER_STDLIB_RUNTIME_REGISTRATION_V2'
POLICY='ENCODER45_FINITE_HERMES_STDLIB_EXECUTION_SCOPE_V2'
ANCHOR_FIELDS={'path','file_sha256','body_sha256'}
SOURCE_NAMES=('encoder_inventory.py','stdlib_inventory.py','encoder_registration_v2.py','admitted_runtime_observer.py')


def strict_json(raw):
    require(type(raw) is bytes and len(raw)<=4*1024**2,'bounded strict V2 evidence')
    def finite(value):
        parsed=float(value);require(math.isfinite(parsed),'finite registration JSON float');return parsed
    return json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,parse_constant=_bad_constant,parse_float=finite)


def filehash(path):
    path=stdlib_prep.pointer(path,file=True)
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(1024**2),b''):value.update(raw)
    return value.hexdigest()


def anchor(value):
    require(type(value) is dict and set(value)==ANCHOR_FIELDS,'externally selected evidence file/body anchors')
    path=Path(value['path']);require(path.is_absolute(),'absolute selected evidence path')
    path=stdlib_prep.pointer(path,file=True)
    require(path.stat().st_size<=4*1024**2 and filehash(path)==value['file_sha256'],'bounded actual frozen evidence file hash')
    body=strict_json(path.read_bytes())
    require(body['sha256']==value['body_sha256']==digest({k:v for k,v in body.items() if k!='sha256'}),'actual canonical evidence body hash')
    return body


def merge_pins(*maps):
    result={}
    for value in maps:
        require(type(value) is dict,'runtime pin map')
        for path,sha in value.items():
            require(type(path) is str and Path(path).is_absolute() and type(sha) is str and len(sha)==64
                and all(c in '0123456789abcdef' for c in sha),'typed absolute runtime hash')
            require(path not in result or result[path]==sha,'conflicting inherited/additive runtime hash')
            result[path]=sha
    return dict(sorted(result.items()))


def expected_sources():
    return {str((HERE/name if name in ('encoder_registration_v2.py','admitted_runtime_observer.py') else PACKAGE/name).resolve()):
        filehash(HERE/name if name in ('encoder_registration_v2.py','admitted_runtime_observer.py') else PACKAGE/name) for name in SOURCE_NAMES}


def validate_parts(package,stdlib,package_review,stdlib_review,inherited,anchors):
    """Pure schema/anchors; no re-hash of runtime/model assets before admission."""
    require(package['kind']==encoder_prep.KIND and package['native_authority_issued'] is False
        and package['native_registration_written'] is False and package['native_config_written'] is False
        and package['complete_runtime_execution_closure_established'] is False,'scoped package byte evidence only')
    enc=package['runtime_inventory'];require(enc['versions']=={'tokenizers':'0.22.2','jinja2':'3.1.6','markupsafe':'3.0.3'}
        and set(enc['packages'])=={'tokenizers','jinja2','markupsafe'},'exact observed encoder versions/three-package policy')
    require(set(enc['runtime_pins'])==set(enc['files'])==set(enc['paths'])
        and all(enc['runtime_pins'][path]==entry['sha256'] and entry['path']==path for path,entry in enc['files'].items()),'complete package runtime source/byte map')
    require(set(enc['runtime_pins'])=={path for value in enc['packages'].values() for path in value['files']},'every declared encoder executable')
    require(len(enc['files'])==45 and {name:len(item['files']) for name,item in enc['packages'].items()}
        =={'tokenizers':17,'jinja2':25,'markupsafe':3},'exact independently reviewed package45 superset')
    preview=package['native_registration_preview'];require(preview['sha256']==digest({k:v for k,v in preview.items() if k!='sha256'})
        and preview['runtime_pins']==enc['runtime_pins'] and preview['dependency_versions']==enc['versions']
        and preview['executing_package_roster']==enc['packages'],'selected package preview maps/policy joined')
    helper_key=next((path for path in preview['source_pins'] if Path(path).name=='encoder_inventory.py'),None)
    require(helper_key is not None and preview['source_pins'][helper_key]['sha256']==filehash(PACKAGE/'encoder_inventory.py'),
        'selected candidate exact executed encoder helper bytes')
    require(preview['input_packet']['sha256']==encoder_prep.PACKET_SHA and preview['input_packet']['sha256']==filehash(encoder_prep.contract.PACKET)
        and preview['chat_template_sha256']==encoder_prep.contract.native.read_json(encoder_prep.contract.PACKET)['tokenizer_binding']['chat_template_sha256'],
        'selected candidate pinned input/template identity')
    require(stdlib['kind']==stdlib_prep.KIND and stdlib['fixture'] is False and stdlib['before_after_byte_equal'] is True
        and stdlib['native_authority_issued'] is False and stdlib['complete_machine_trust_claimed'] is False,'reviewed nonfixture finite runtime bytes')
    std=stdlib['inventory'];require(std['fixture'] is False and std['policy']==stdlib_prep.VERSION
        and std['python_version']=='3.11.15' and std['excluded_directory_names']==['site-packages','__pycache__']
        and std['Windows_system_driver_scope']=='inherited separately; no machine closure claim','exact finite interpreter policy/version')
    expected_alias={'logical_base_prefix':str(stdlib_prep.BASE.absolute()),'physical_base_prefix':str(stdlib_prep.PHYSICAL_BASE.absolute()),
        'file_attributes':0x410,'reparse_tag':0xa0000003,'is_symlink':False}
    require(std['approved_alias_binding']==expected_alias and std['base_prefix']==expected_alias['logical_base_prefix']
        and std['physical_base_prefix']==expected_alias['physical_base_prefix'],'exact known logical/physical alias binding')
    require(set(std['runtime_pins'])==set(std['files'])==set(std['paths'])
        and all(std['runtime_pins'][path]==entry['sha256'] and entry['path']==path for path,entry in std['files'].items()),'complete finite runtime file/source map')
    require(len(std['files'])==779 and {category:list(std['paths'].values()).count(category) for category in set(std['paths'].values())}
        =={'stdlib_python_source':738,'stdlib_extension_runtime':34,'interpreter_root_executable_runtime':6,'executing_Hermes_launcher':1},
        'exact independently reviewed finite779 interpreter/stdlib superset')
    require(std['optional_stdlib_zip']=={'path':str(Path(std['physical_base_prefix'])/'python311.zip'),'present':False},'explicit reviewed absent ZIP')
    require(stdlib['helper_source_sha256']==filehash(PACKAGE/'stdlib_inventory.py'),'reviewed finite inventory helper byte pin')
    require(package_review['status']=='PASS_EXACT_ENCODER_PACKAGE_BYTE_REVIEW'
        and package_review['candidate_file_sha256']==anchors['package']['file_sha256']
        and package_review['candidate_body_sha256']==package['sha256']
        and package_review['helper_source_sha256']==filehash(PACKAGE/'encoder_inventory.py')
        and package_review['runtime_pins_sha256']==digest(enc['runtime_pins'])
        and package_review['observed_versions']==enc['versions']
        and package_review['input_packet_sha256']==preview['input_packet']['sha256']
        and package_review['chat_template_sha256']==preview['chat_template_sha256']
        and package_review['consumption_source_pins']=={path:entry['sha256'] for path,entry in preview['source_pins'].items()}
        and package_review['source_unchanged'] is True
        and package_review['encoder_framework_model_or_GPU_execution'] is False
        and package_review['native_registration_or_config_written'] is False,'independently selected exact package byte/source/template review; status is insufficient')
    require(all(filehash(Path(path))==entry['sha256'] for path,entry in preview['source_pins'].items()),'selected package current relevant helper/input/plumbing source bytes')
    require(stdlib_review['status']=='PASS_CURRENT779_RUNTIME_BYTE_SUPERSET' and stdlib_review['candidate_body_sha256']==stdlib['sha256']
        and stdlib_review['candidate_file_sha256']==anchors['stdlib']['file_sha256']
        and stdlib_review['runtime_files']==len(std['files'])
        and stdlib_review['runtime_bytes']==sum(entry['bytes'] for entry in std['files'].values())
        and stdlib_review['alias_binding']==std['approved_alias_binding']
        and stdlib_review['optional_stdlib_zip']==std['optional_stdlib_zip']
        and stdlib_review['helper_source_before']==stdlib_review['helper_source_after']==stdlib['helper_source_sha256']
        and stdlib_review['forbidden_modules_loaded']==[] and stdlib_review['native_authority_issued'] is False
        and stdlib_review['encoder_framework_model_GPU_execution'] is False,'independently selected exact finite runtime byte review')
    require(inherited==encoder_prep.contract.base_runtime_pins(),'every inherited64 plus additive AST pin retained exactly')
    additional=merge_pins(enc['runtime_pins'],std['runtime_pins']);merged=merge_pins(inherited,additional)
    return {'encoder_inventory':deepcopy(enc),'stdlib_inventory':deepcopy(std),'additional_runtime_pins':additional,'runtime_pins':merged,
        'prompt_registration_preview':deepcopy(package['native_registration_preview'])}


def build_preview(package_anchor,stdlib_anchor,package_review_anchor,stdlib_review_anchor,inherited):
    values=[anchor(value) for value in (package_anchor,stdlib_anchor,package_review_anchor,stdlib_review_anchor)]
    selected={'package':package_anchor,'stdlib':stdlib_anchor,'package_review':package_review_anchor,'stdlib_review':stdlib_review_anchor}
    parts=validate_parts(*values,inherited,selected)
    body={'schema_version':2,'kind':KIND,'policy':POLICY,'evidence_anchors':{'package':deepcopy(package_anchor),'stdlib':deepcopy(stdlib_anchor),
        'package_review':deepcopy(package_review_anchor),'stdlib_review':deepcopy(stdlib_review_anchor)},
        **parts,'inherited_runtime_pins':deepcopy(inherited),'consumption_source_pins':expected_sources(),
        'complete_declared_encoder_execution_scope':True,'complete_machine_trust_claimed':False,
        'native_launch_authorized':False,'native_qualification_established':False,'study_training_authorized':False,'scoring_authorized':False}
    body['sha256']=digest(body);return body


def registration(value,inherited):
    require(not STAGED,'staged V2 registration admission forbidden before paths/probes')
    require(type(value) is dict and set(value)=={'path','sha256'},'externally selected exact V2 registration')
    path=Path(value['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink()
        and path.resolve().parent==(PACKAGE/'artifacts/long_native_audit_registrations').resolve()
        and filehash(path)==value['sha256'],'owned frozen V2 registration bytes')
    require(path.stat().st_size<=4*1024**2,'bounded final V2 registration')
    body=strict_json(path.read_bytes());require(body['sha256']==digest({k:v for k,v in body.items() if k!='sha256'})
        and body['kind']==KIND and body['schema_version']==2 and body['policy']==POLICY,'V2 schema cannot be boolean or self-signed shortcut')
    anchors=body['evidence_anchors']
    rebuilt=build_preview(anchors['package'],anchors['stdlib'],anchors['package_review'],anchors['stdlib_review'],inherited)
    require(canonical(body)==canonical(rebuilt),'entire independently anchored V2 maps/policy/provenance/source consumption unchanged')
    return body
