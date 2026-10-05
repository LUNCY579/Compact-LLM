"""CPU/source-only exact Phase3 inventory preparation. Never launches or probes.

Default inspect validates a stage template. --freeze is reserved for Root after
review and the config commit; it writes one exclusive owned inventory JSON only.
No runtime/decoder/data/model path is opened by this script.
"""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1];ROOT=PACKAGE.parents[1]
AUDIT=PACKAGE/'artifacts/short_phase_audit_work'
sys.path.insert(0,str(AUDIT))
import audit_adapter as auditor
import phase_replay as pure
CONFIG=PACKAGE/'configs/SYNTH_SHORT_M0_ATOMIC_BATCH04_PHASE3_A1.json'
CONFIG_SHA='e6d56e4a592b68bc7e288022b98bd73e94a468f6243625579bfb3bcb7d3c0b4d'
CONFIG_BODY_SHA='f514a097645647184191a42acda0311d0f338484e9027011d5cd5598c212dd38'
PROFILE_SHA='2d5547910aa57b0509a532972d899ceb5c913e41274bfa773d25536e6d8c9676'
CORE64_SHA='7cc9daa0ed13d78c2c78583902b515066fca579a3dd47d054d59c1cdc2e88d71'
HEAD_PLACEHOLDER='__FINAL_CLEAN_LAUNCH_HEAD_AFTER_CONFIG_COMMIT__'
TEMPLATE=HERE/'INVENTORY_TEMPLATE.json'
SECTIONS={'source':'source_pins','runtime':'runtime_pins','decoder':'decoder_runtime_pins','data':'data_pins','model':'model_pins'}
COUNTS={'source':131,'runtime':65,'decoder':17,'data':5,'model':7}
DEPENDENCIES={
 'audit_adapter.py':'12a6bb2e1807f281d22b437dc086b6dc7fa9cc53a6f331f698165ab14c8a3533',
 'phase_replay.py':'270ec43ee8168f7032562095f6c07e49b20fa815e70cb7a648d6c2fb8dd23cbd',
 'test_phase_audit.py':'407a853721d6da3a62861dd27bf75cbf793659ed057f053b97a958f347fd1ba0',
 'fixture_support.py':'0645ea43da2febb0ea958e61fad86cfb07838cf1278869497ed3181a5c50988b',
 'pinned_dependencies/RAM2_auditor_d69810.py':'d69810dccd11a32595dace73a788c8868efabb2a09f593fd33316e92dcb47442',
 'pinned_dependencies/PRESERVATION.json':'ad2f4896cd6ceecaa0d8cd99dff055463e88cfd7748abf0d69919d5ae71262a3'}
FIELDS={'schema_version','id','contract','root_path','package_path','config_path','config_file_sha256',
    'profile_body_sha256','source_commit','inventories','counts','qualified_runtime64_sha256','auditor_dependency_pins','sha256'}
def seal(value):
    value=deepcopy(value);value.pop('sha256',None);value['sha256']=pure.digest(value);return value
def scope():
    raw=pure.read_bytes(CONFIG,8*1024**2);pure.require(hashlib.sha256(raw).hexdigest()==CONFIG_SHA,'exact independently selected config bytes')
    value=pure.loads(raw)
    pure.require(value['sha256']==CONFIG_BODY_SHA==pure.digest({k:v for k,v in value.items() if k!='sha256'}),'exact config body anchor')
    pure.require(pure.digest(value['short_phase_observation_profile'])==PROFILE_SHA,'exact current phase profile body anchor')
    pure.require(value['batch_selector']=={'source_kind':'atomic_RS_DEV','ordinal':4} and len(value['rows'])==64
        and value['dose']['maximum_forward_calls']==2050 and value['dose']['maximum_forward_input_positions']==510720,
        'exact whole64 batch04/2050call/510720position scope')
    pure.require(value['contract']==auditor.VERSION and value['authorization']==auditor.AUTHORIZATION,'exact reviewed purpose/authority')
    return value
def dependency_pins(*,verify=True):
    result={str((AUDIT/name).resolve()):pin for name,pin in DEPENDENCIES.items()}
    if verify:
        for path,pin in result.items():pure.require(pure.file_info(path)['sha256']==pin,'exact independently reviewed auditor dependency '+path)
    return result
def make_template(value):
    inventories={section:deepcopy(value[key]) for section,key in SECTIONS.items()}
    extension=value['short_RAM_monitor_profile']['runtime_extension_pins'];core={p:h for p,h in inventories['runtime'].items() if p not in extension}
    pure.require({k:len(v) for k,v in inventories.items()}==COUNTS,'exact current metadata inventory cardinalities')
    pure.require(len(extension)==1 and all(Path(p).name=='ast.py' and inventories['runtime'].get(p)==h for p,h in extension.items())
        and len(core)==64 and pure.digest(core)==CORE64_SHA,'exact inherited AST/qualified64 metadata')
    return seal({'schema_version':1,'id':auditor.REGISTRATION_ID,'contract':auditor.VERSION,
        'root_path':str(ROOT.resolve()),'package_path':str(PACKAGE.resolve()),'config_path':str(CONFIG.resolve()),
        'config_file_sha256':CONFIG_SHA,'profile_body_sha256':PROFILE_SHA,'source_commit':HEAD_PLACEHOLDER,
        'inventories':inventories,'counts':deepcopy(COUNTS),'qualified_runtime64_sha256':CORE64_SHA,'auditor_dependency_pins':dependency_pins()})
def validate_template(value,body,*,verify_sources=True):
    pure.require(type(value) is dict and set(value)==FIELDS,'exact actual-auditor registration field roster')
    pure.require(value==make_template(body),'every template path/hash/count/dependency/anchor exactly reconstructed')
    if verify_sources:auditor._source_admission(value)
    # This is the original pure new-profile admission. The placeholder is joined
    # literally for metadata validation; public native prerequisites reject it.
    sup={'preflight':{'source_commit':HEAD_PLACEHOLDER},'short_RAM_monitor_diagnostic':{
        'path':'SHORT_RAM_ABORT_DIAGNOSTIC.json','attempted':False,'persisted':False,'write_error':None,'original_abort_reason_preserved':None}}
    auditor._phase3_admission(body,sup,value,PROFILE_SHA,body['short_phase_observation_profile'],{},pure.require)
    return {'status':'PASS_CPU_PHASE3_TEMPLATE_METADATA_ONLY','counts':deepcopy(COUNTS),'config_file_sha256':CONFIG_SHA,
        'config_body_sha256':CONFIG_BODY_SHA,'profile_body_sha256':PROFILE_SHA,'qualified_runtime64_sha256':CORE64_SHA,
        'template_body_sha256':value['sha256'],'launch_HEAD_frozen':False,'native_launch_authorized':False,
        'native_runtime_model_readiness_established':False,'source_bytes_checked':verify_sources}
def git_identity(expected):
    pure.require(type(expected) is str and len(expected)==40 and all(c in '0123456789abcdef' for c in expected),'explicit full final launch HEAD')
    status=subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    pure.require(not status.strip(),'clean committed source/config required for freeze')
    actual=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    pure.require(actual==expected,'actual clean HEAD equals explicitly reviewed final launch HEAD')
    relative=CONFIG.relative_to(ROOT).as_posix()
    committed=subprocess.run(['git','show',expected+':'+relative],cwd=ROOT,capture_output=True,check=True).stdout
    pure.require(committed==pure.read_bytes(CONFIG,8*1024**2),'exact selected config already committed at final launch HEAD')
    return actual
def final_body(template,body,head):
    validate_template(template,body);git_identity(head)
    final=deepcopy(template);final['source_commit']=head;final=seal(final)
    sup={'preflight':{'source_commit':head},'short_RAM_monitor_diagnostic':{
        'path':'SHORT_RAM_ABORT_DIAGNOSTIC.json','attempted':False,'persisted':False,'write_error':None,'original_abort_reason_preserved':None}}
    auditor._phase3_admission(body,sup,final,PROFILE_SHA,body['short_phase_observation_profile'],{},pure.require)
    auditor._source_admission(final);pure.require(git_identity(head)==head,'same final source/config HEAD after source-only checks')
    return final
def write_final(value,name):
    directory=PACKAGE/'artifacts/short_m0_audit_registrations';target=directory/name
    pure.require(Path(name).name==name and name.startswith('SYNTH_SHORT_PHASE3_') and name.endswith('.json')
        and directory.is_dir() and not directory.is_symlink() and not target.exists()
        and target.resolve().parent==directory.resolve(),'new exclusive fixed owned Phase3 inventory path')
    raw=(pure.canonical(value)+'\n').encode('ascii');pure.require(len(raw)<=pure.MAX_JSON,'bounded final registration')
    with target.open('xb') as stream:
        pure.require(stream.write(raw)==len(raw),'complete final registration write');stream.flush()
        # Ordinary durable artifact IO; no host/resource/native probe.
        import os
        os.fsync(stream.fileno())
    pure.require(pure.read_json(target)==value,'exact final registration reread')
    return {'path':str(target.resolve()),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256']}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-template',action='store_true')
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--final-head')
    parser.add_argument('--reviewed-template-file-sha256');parser.add_argument('--output-name')
    args=parser.parse_args();pure.require(not(args.prepare_template and args.freeze),'one explicit preparation or freeze action')
    body=scope()
    if args.prepare_template:
        pure.require(not TEMPLATE.exists(),'stage template is write-once')
        value=make_template(body);validate_template(value,body)
        with TEMPLATE.open('xb') as stream:stream.write((pure.canonical(value)+'\n').encode('ascii'))
    else:
        raw=pure.read_bytes(TEMPLATE);value=pure.loads(raw);validate_template(value,body)
    if args.freeze:
        pure.require(auditor.hash_pin(args.reviewed_template_file_sha256) and hashlib.sha256(raw).hexdigest()==args.reviewed_template_file_sha256,
            'external independently reviewed exact template file anchor')
        pure.require(type(args.output_name) is str,'explicit new owned final inventory name')
        value=final_body(value,body,args.final_head);print(json.dumps({'status':'FROZEN_SOURCE_ONLY_PHASE3_INVENTORY_NO_LAUNCH',**write_final(value,args.output_name)}))
    else:print(json.dumps(validate_template(value,body),sort_keys=True))
if __name__=='__main__':main()
