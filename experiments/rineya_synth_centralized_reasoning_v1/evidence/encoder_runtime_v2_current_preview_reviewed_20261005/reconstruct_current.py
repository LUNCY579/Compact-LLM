"""CPU current-byte reconstruction/preview; no final registration/config writer."""
from pathlib import Path
import hashlib,importlib.metadata,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
assert HERE.name=='encoder_runtime_v2_current_registration_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1]
sys.path.insert(0,str(PACKAGE))
import encoder_inventory as prep
import encoder_registration_v2 as v2
import stdlib_inventory as finite
import long_native_contract as contract
from rs_contract import canonical,digest,require


def info(path,root=None):
    """Separate byte verifier; does not use encoder_inventory's byte routine."""
    path=Path(path).absolute()
    for parent in (path,*path.parents):
        require(not parent.is_symlink() and (not parent.exists() or not getattr(parent.lstat(),'st_file_attributes',0)&0x400),'regular independent observation')
    require(path.is_file() and path.resolve()==path and (root is None or path.is_relative_to(Path(root).absolute())),'exact independent path/root')
    before=path.stat();require(0<before.st_size<=32*1024**2,'bounded independent package/input/source observation')
    sha=hashlib.sha256();read=0
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(1024**2),b''):sha.update(raw);read+=len(raw)
    after=path.stat();require(read==before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,'independent immutable observation')
    return {'path':str(path),'bytes':read,'sha256':sha.hexdigest()}


def write_once(name,body):
    require(Path(name).name==name and name.endswith('.json'),'new stage JSON only')
    path=HERE/name;require(not path.exists(),'never overwrite current/failure evidence')
    body={**body};body['sha256']=digest(body);raw=(canonical(body)+'\n').encode()
    with path.open('xb') as stream:require(stream.write(raw)==len(raw),'full preview evidence write');stream.flush();os.fsync(stream.fileno())
    return body,info(path)


def anchor(path):
    observed=info(path);body=v2.strict_json(Path(path).read_bytes())
    return {'path':observed['path'],'file_sha256':observed['sha256'],'body_sha256':body['sha256']}


def independently_reconstruct(candidate):
    prep.no_execution();finite.no_execution();preview=candidate['native_registration_preview']
    versions={};packages={};paths={};files={}
    for name in ('tokenizers','jinja2','markupsafe'):
        dist=importlib.metadata.distribution(name);base=Path(dist.locate_file(name)).absolute()
        declared=[]
        for record in dist.files:
            if str(record).endswith(('.py','.pyd')) and '..' not in Path(str(record)).parts:
                observed=info(dist.locate_file(record),base);key=observed['path'];require(key not in paths,'unique independent package record')
                declared.append(key);paths[key]=name;files[key]=observed
        physical={str(path.absolute()) for path in base.rglob('*') if path.suffix in ('.py','.pyd')}
        require(physical==set(declared),'independent RECORD vs physical exact executable superset')
        versions[name]=dist.version;packages[name]={'root':str(base),'version':dist.version,'files':sorted(declared)}
    reconstructed={'versions':versions,'packages':packages,'paths':dict(sorted(paths.items())),
        'files':files,'runtime_pins':{path:value['sha256'] for path,value in files.items()}}
    require(reconstructed==candidate['runtime_inventory'],'independent exact package bytes/metadata')
    packet_info=info(contract.PACKET);packet=v2.strict_json(contract.PACKET.read_bytes());binding=packet['tokenizer_binding']
    require(packet_info==preview['input_packet'],'independent frozen input file/body')
    require(packet['sha256']==contract.PACKET_BODY and packet_info['sha256']==contract.PACKET_SHA,'exact tracked prompt packet identity')
    sources={str((PACKAGE/name).resolve()):info(PACKAGE/name) for name in prep.SOURCE_NAMES}
    sources[str(Path(prep.__file__).resolve())]=info(prep.__file__);sources[str(contract.PACKET)]=packet_info
    require(sources==preview['source_pins'],'independent exact current helper/plumbing/input source map')
    assets={name:info(Path(binding['model_dir'])/name) for name in binding['assets']}
    require(assets==preview['prompt_assets'],'independent exact four prompt assets')
    config=v2.strict_json(Path(assets['tokenizer_config.json']['path']).read_bytes())
    template_sha=hashlib.sha256(config['chat_template'].encode()).hexdigest()
    require(template_sha==binding['chat_template_sha256']==preview['chat_template_sha256'],'independent actual template UTF8 hash')
    require({path:info(path) for path in binding['binding_sources']}==preview['binding_sources'],'independent binding receipt bytes')
    require({path:info(path) for path in packet['source_before']}==preview['prompt_source_origins'],'independent preparation source origins')
    require(preview['tokenizer_binding_sha256']==digest(binding),'independent tokenizer binding identity')
    prep.no_execution();finite.no_execution()
    return versions,template_sha,sources


def main():
    started=time.monotonic();source_before=contract.source_pins()
    source_before[str((PACKAGE/'encoder_registration_v2.py').resolve())]=v2.filehash(PACKAGE/'encoder_registration_v2.py')
    source_before[str((PACKAGE/'admitted_runtime_observer.py').resolve())]=v2.filehash(PACKAGE/'admitted_runtime_observer.py')
    source_before[str(Path(__file__).resolve())]=v2.filehash(__file__)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip()
    git_state_before=subprocess.check_output(['git','status','--porcelain'],cwd=PACKAGE,text=True)
    candidate=prep.build_candidate();candidate,candidate_info=write_once('CURRENT_PACKAGE_CANDIDATE_001.json',{k:v for k,v in candidate.items() if k!='sha256'})
    versions,template,sources=independently_reconstruct(candidate)
    review,review_info=write_once('CURRENT_PACKAGE_REVIEW_001.json',{'schema_version':1,'kind':'CPU_INDEPENDENT_CURRENT_PACKAGE_BYTE_RECONSTRUCTION_REVIEW',
        'status':'PASS_EXACT_ENCODER_PACKAGE_BYTE_REVIEW','candidate_file_sha256':candidate_info['sha256'],'candidate_body_sha256':candidate['sha256'],
        'helper_source_sha256':v2.filehash(PACKAGE/'encoder_inventory.py'),'runtime_pins_sha256':digest(candidate['runtime_inventory']['runtime_pins']),
        'observed_versions':versions,'input_packet_sha256':candidate['native_registration_preview']['input_packet']['sha256'],
        'chat_template_sha256':template,'consumption_source_pins':{path:value['sha256'] for path,value in sources.items()},
        'source_unchanged':True,'source_commit':commit,'independent_method':'separate1MiB byte verifier + RECORD/physical roster + explicit template/input/source reconstruction',
        'encoder_framework_model_or_GPU_execution':False,'native_registration_or_config_written':False,'native_authority_issued':False,
        'scope':'CPU package45 exact byte evidence only; externally selected reviewer/file anchors are not self-signed launch authority'})
    std=PACKAGE/'artifacts/stdlib_runtime_closure_work/STDLIB_CANDIDATE_001.json'
    std_review=PACKAGE/'artifacts/stdlib_runtime_closure_work/ROOT_CANDIDATE_REVIEW_001.json'
    selected=[anchor(HERE/'CURRENT_PACKAGE_CANDIDATE_001.json'),anchor(std),anchor(HERE/'CURRENT_PACKAGE_REVIEW_001.json'),anchor(std_review)]
    inherited=contract.base_runtime_pins();preview=v2.build_preview(*selected,inherited)
    require(len(inherited)==65 and all(preview['runtime_pins'][p]==h for p,h in inherited.items()) and len(preview['runtime_pins'])==887,'all inherited65 +exact unique887 union')
    duplicate=next(path for path in inherited if path in preview['additional_runtime_pins'])
    try:v2.merge_pins(inherited,{duplicate:'0'*64})
    except ValueError:conflict_denied=True
    else:raise AssertionError('overlap conflict was accepted')
    old=PACKAGE/'artifacts/encoder_runtime_v2_work'
    try:v2.build_preview(anchor(old/'CURRENT_PACKAGE_CANDIDATE_001.json'),anchor(std),anchor(old/'CURRENT_PACKAGE_REVIEW_001.json'),anchor(std_review),inherited)
    except ValueError as exc:stale_denied=str(exc)
    else:raise AssertionError('pre-integration source candidate was accepted')
    # Preview is stage-only and contains no config/path under the owned final-registration leaf.
    preview_info=write_once('V2_PREVIEW_ONLY_001.json',{k:v for k,v in preview.items() if k!='sha256'})[1]
    source_after={path:v2.filehash(path) for path in source_before};require(source_after==source_before,'entire current source bytes unchanged')
    commit_after=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip()
    git_state_after=subprocess.check_output(['git','status','--porcelain'],cwd=PACKAGE,text=True)
    receipt,receipt_info=write_once('CPU_VALIDATION_001.json',{'schema_version':1,'kind':'CPU_CURRENT_V2_PREVIEW_RECONSTRUCTION_NOT_NATIVE_AUTHORITY',
        'status':'PASS_CURRENT_SOURCE_PACKAGE_REVIEW_AND_V2_PREVIEW_ONLY','source_commit':commit,'source_commit_after':commit_after,
        'git_state_before':git_state_before,'git_state_after':git_state_after,
        'git_state_scope':'CPU preview evidence only; unrelated root work does not grant or bypass the unchanged native clean-commit gate',
        'source_before':source_before,'source_after':source_after,
        'source_unchanged':True,'candidate_file_sha256':candidate_info['sha256'],'candidate_body_sha256':candidate['sha256'],
        'review_file_sha256':review_info['sha256'],'review_body_sha256':review['sha256'],'preview_file_sha256':preview_info['sha256'],
        'preview_body_sha256':preview['sha256'],'selected_anchors':selected,'package_files':45,'stdlib_files':779,'inherited_files':65,
        'additive_files':824,'unique_union_files':887,'overlap_conflict_rejected':conflict_denied,'stale_preintegration_rejected':stale_denied,
        'wall_seconds':time.monotonic()-started,'package_runtime_and_prompt_asset_CPU_hashing':True,'stdlib_runtime_binaries_rehashed':False,
        'encoder_framework_model_GPU_execution':False,'native_registration_or_config_written':False,'native_authority_issued':False,
        'external_caller_selection_and_independent_review_required':True,'native_auditor_and_terminal_V2_adaptation_required':True})
    print(canonical({'status':receipt['status'],'candidate':candidate_info,'review':review_info,'preview':preview_info,'receipt':receipt_info,
        'seconds':receipt['wall_seconds'],'stale':stale_denied,'union':887}))

if __name__=='__main__':main()
