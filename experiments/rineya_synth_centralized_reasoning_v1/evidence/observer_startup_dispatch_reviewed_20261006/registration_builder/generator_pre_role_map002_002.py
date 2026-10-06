"""JSON-only observer startup dispatch construction; closed by default.

No native APIs, binary hashing, loaded-baseline or bootstrap fact is produced.
Source promotion and two-commit/approval/current-byte review are external inputs.
"""
from pathlib import Path
import ast,hashlib,json,math,stat

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
STARTUP=PACKAGE/'evidence/observer_startup_measurement_reviewed_cpu_20261006'
PASSIVE=PACKAGE/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/PASSIVE_HOST_INVENTORY_002.json'
PASSIVE_FILE='8dc646ec2c45f3b7071fa96caa06b14a2c6cf94c7fa43a64c0bbef257bb0d934'
PASSIVE_BODY='1fb617bb17d026bc237465dea0a748e97481811e054e4469771c61fd9bd718e7'
BUILD=PACKAGE/'evidence/native_library_observer_candidate_reviewed_cpu_20261006/observer_source_work/native_library_observer_build.json'
BUILD_FILE='4f862aaf2b487c2f119ebe04972b40db846505f0f7b6848ad81afabea05ff93d'
BUILD_BODY='70d0dc391b0ca1c6e2c3c7b14d7ca5e305e6117abd8dd8e22516add127e11692'
PROMOTION_FILE='e8cddce6b7f52dfe6a87252d23a760539ea525123d6354b84c7b86cd02ee7239'
PROMOTION_BODY='42fb5140930645046f88ba1f2d78410a7651007a34c099d347c86b5147ebbe9c'
APPROVAL_FILE='32352893c88559f9f09c0cf73f574e666feeefc9b5a2930121d0128b9b7c0ebe'
APPROVAL_BODY='f3aaa4c3e56f860fadf3ece1b30852d306296ca060e776fdbd4677eb7dd015d2'
PE_PATH=PACKAGE/'artifacts/audit_reviewer_work/PASSIVE_PE_STARTUP_CANDIDATES_1791256942555417100.json'
PE_FILE='538ee2fb2246f8b4958c965a2a29438b205aaca5899bce024a7e059e71f14949'
PE_BODY='02215bb558f18d4f3c73dc8fb628f3a1850ebece48cc1fc4b0e71bfbf82f8438'
WINDOWS_FRESH_FILE='9fca7b708ae886735b4c16bec769d32228a0bed1c0071c7edd4ccf09905cfd87'
WINDOWS_FRESH_BODY='13a6ae4458960bd3294ecf7ec08b42ed69607cef2279ddb83f86e347f85ebac9'

def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def seal(value):
    body={k:v for k,v in value.items() if k!='sha256'};return {**body,'sha256':digest(body)}
def sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def head(value):return type(value) is str and len(value)==40 and all(c in '0123456789abcdef' for c in value)
def pairs(rows):
    result={}
    for key,value in rows:require(key not in result,'duplicate JSON key');result[key]=value
    return result
def floating(text):
    value=float(text);require(math.isfinite(value),'nonfinite/overflow JSON value');return value
def parse(raw):return json.loads(raw,object_pairs_hook=pairs,parse_float=floating,parse_constant=lambda _:require(False,'nonfinite JSON value'))
def read_source(path):
    p=Path(path).absolute();require(p.suffix in ('.py','.c','.h','.json','.md') and p.resolve()==p,'source/JSON only physical input; no binary reads')
    for part in (p,*p.parents):require(not part.is_symlink() and not(getattr(part.lstat(),'st_file_attributes',0)&0x400),'no source reparse input')
    before=p.lstat();require(stat.S_ISREG(before.st_mode) and before.st_size<=16*1024**2,'bounded source/metadata file')
    with p.open('rb') as f:raw=f.read(before.st_size+1)
    after=p.lstat();require(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'source/metadata stable bytes');return raw
def pinned_json(path,file_pin,body_pin):
    raw=read_source(path);require(hashlib.sha256(raw).hexdigest()==file_pin,'external metadata file pin')
    value=parse(raw);require(type(value) is dict and value['sha256']==body_pin==digest({k:v for k,v in value.items() if k!='sha256'}),'external metadata body pin');return value
def constants(path):
    tree=ast.parse(read_source(path),str(path));names={'SCOPE','PERMISSION','RESOURCES','ABI'};out={}
    for row in tree.body:
        if isinstance(row,ast.Assign) and len(row.targets)==1 and isinstance(row.targets[0],ast.Name) and row.targets[0].id in names:out[row.targets[0].id]=ast.literal_eval(row.value)
    require(set(out)==names,'exact literal startup permission/resource/ABI constants');return out

def descriptor(value):
    require(type(value) is dict and set(value)=={'bytes','sha256','role'} and type(value['bytes']) is int and 0<=value['bytes']<=8*1024**3 and sha(value['sha256']),'exact typed prospective descriptor')

def construct(*,passive,build,package,promoted_sources,implementation_commit=None,dispatch_head=None,approval=None,approved_evidence_reference=None,constants_value,stdlib_layout,offline_environment,additional_windows=None):
    """Construct only; no source promotion, Git check, launch or fact filling."""
    package=Path(package);require(package.is_absolute(),'explicit final package path')
    require(implementation_commit is None or head(implementation_commit),'implementation ancestor full SHA or unresolved null')
    require(dispatch_head is None or (head(dispatch_head) and dispatch_head!=implementation_commit),'distinct external later dispatch HEAD or unresolved null')
    require(passive.get('host_inventory_complete') is passive.get('loaded_module_baseline_observed') is passive.get('namespace_execution_proven') is passive.get('resources_observed') is passive.get('native_PASS') is passive.get('native_authority_issued') is False,'passive inventory is not measured startup or authority')
    old=passive['inherited65_map'];std=passive['stdlib779_map'];require(type(old) is dict and len(old)==65 and type(std) is dict and len(std)==779,'exact original65/779 maps retained')
    startup=package/'observer_startup_measurement_v1';require(type(promoted_sources) is dict and 0<len(promoted_sources)<=4096,'finite prospective source promotion map')
    sources={}
    for p,d in promoted_sources.items():
        descriptor(d);require(type(p) is str and Path(p).is_absolute() and Path(p).suffix in ('.py','.c','.h','.json','.md'),'source-only promoted descriptor paths');sources[p]=dict(d)
    require({str(startup/n) for n in ('contract.py','worker.py','supervisor.py','audit.py','preserve.py')}<=set(sources),'five exact startup implementation source leaves required')
    files={}
    for p,d in passive['files'].items():
        descriptor({k:d[k] for k in ('bytes','sha256','role')})
        # Earlier stage source is not runtime. Keep only external runtime paths;
        # Root later promotes the two already measured binary byte identities.
        if Path(p).is_relative_to(package) and Path(p).suffix in ('.py','.c','.h','.json','.md'):continue
        files[p]={k:d[k] for k in ('bytes','sha256','role')}
    require(all(p in files and files[p]['sha256']==h for p,h in old.items()) and all(p in files and files[p]['sha256']==h for p,h in std.items()),'original inherited pins cannot be omitted/substituted')
    folded={p.casefold():p for p in files}
    for p,d in ({} if additional_windows is None else additional_windows).items():
        descriptor(d);require(Path(p).parent==Path(r'C:\Windows\System32') and Path(p).suffix.lower()=='.dll' and d['role']=='windows_system_binary','finite prospective Windows DLL identities only')
        canonical_path=folded.get(p.casefold())
        if canonical_path is not None:require(files[canonical_path]==d,'casefold overlapping Windows descriptor conflict')
        else:files[p]=dict(d);folded[p.casefold()]=p
    binaries={}
    for role,name in (('observer_binary','native_library_observer.dll'),('observer_fixture_binary','observer_only_fixture.dll')):
        matches=[d for d in files.values() if d['role']==role];require(len(matches)==1,'one measured selected observer/fixture binary')
        for p in [p for p,d in files.items() if d['role']==role]:del files[p]
        target=str(package/name);files[target]=dict(matches[0]);binaries[role]={'path':target,'descriptor':dict(matches[0])}
    require(build['binary']==binaries['observer_binary'] and build['native_PASS'] is build['native_authority_issued'] is False,'byte-exact existing build binary; no new build/qualification claim')
    for p,d in build['native_source_files'].items():require(p in sources and sources[p]==d,'existing build C/H source descriptors match promotion map')
    require(build['source_inputs_committed'] is False,'retain original explicit build source provenance')
    c=constants_value;observer={'kind':'ROOT_REGISTERED_NATIVE_LIBRARY_OBSERVER_V1','adapter_source':{'path':str(package/'prefix_library_observer.py'),'descriptor':sources[str(package/'prefix_library_observer.py')]},'native_source_files':build['native_source_files'],'binary':binaries['observer_binary'],'abi':c['ABI'],'build_receipt':{'path':str(package/'native_library_observer_build.json'),'file_sha256':BUILD_FILE,'body_sha256':BUILD_BODY}}
    require(build['abi']==c['ABI'],'unchanged exact native observer ABI')
    allowed=sorted(p for p in files if Path(p).suffix.lower() in ('.exe','.dll','.pyd'))
    require(allowed,'finite allowed library admission population')
    bootstrap={}
    for p,d in files.items():
        path=Path(p);parts=path.parts
        if 'Lib' in parts and 'site-packages' not in parts and path.suffix=='.py':
            relative=parts[parts.index('Lib')+1:];module='.'.join(relative[:-1]) if path.name=='__init__.py' else '.'.join((*relative[:-1],path.stem))
            if module:bootstrap[module]={'path':p,'source_sha256':d['sha256']}
        elif path.suffix=='.pyd' and path.parent.name=='DLLs':bootstrap[path.stem.split('.')[0]]={'path':p,'source_sha256':d['sha256']}
    bootstrap['__main__']={'path':str(startup/'worker.py'),'source_sha256':sources[str(startup/'worker.py')]['sha256']}
    for p,d in sources.items():
        if Path(p).parent==package and Path(p).suffix=='.py':bootstrap[Path(p).stem]={'path':p,'source_sha256':d['sha256']}
    require({'ctypes','ctypes.wintypes','_ctypes'}<=set(bootstrap),'three exact future startup imports admitted by selected source/runtime bytes')
    cache={'mode':'MEASURE_BYTE_VERIFIED_STARTUP_V1','allowed_bootstrap_modules':bootstrap,'imports':['ctypes','ctypes.wintypes','_ctypes']}
    windows={p:d for p,d in files.items() if Path(p).parent==Path(r'C:\Windows\System32') and p not in old and p not in std}
    authorized=approval is not None
    if authorized:
        require(head(implementation_commit) and type(approval) is dict and approval.get('kind')=='VERIFIED_OWNER_OBSERVER_STARTUP_APPROVAL_V1' and approval.get('sha256')==APPROVAL_BODY and approval.get('approved') is True and approval.get('scope')==c['SCOPE'] and approval.get('native_qualification_established') is False and type(approved_evidence_reference) is str and approved_evidence_reference,'authorized mode requires verified human evidence and actual implementation commit')
        require(approval['sha256']==digest({k:v for k,v in approval.items() if k!='sha256'}),'exact verified approval body; no flag-only substitution')
    review=seal({'kind':'ROOT_REVIEWED_OBSERVER_STARTUP_BYTE_ADMISSION_V1','implementation_commit':implementation_commit,'source_inventory_sha256':digest(sources),'runtime_inventory_sha256':digest(files),'observer_registration_sha256':digest(observer),'allowed_initial_loaded_libraries_sha256':digest(allowed),'bootstrap_policy_sha256':digest(cache),'complete_current_byte_review':False})
    review_bytes=(canonical(review)+'\n').encode('ascii');review_anchor={'path':str(package/'evidence/observer_startup_byte_admission_review.json'),'file_sha256':hashlib.sha256(review_bytes).hexdigest(),'body_sha256':review['sha256']}
    require(type(stdlib_layout) is dict and {'logical_base_prefix','physical_base_prefix'}<=set(stdlib_layout) and type(offline_environment) is dict and offline_environment,'known interpreter layout/original offline environment only')
    runtime=seal({'kind':'PROSPECTIVE_OBSERVER_STARTUP_BYTE_ADMISSION_V1','implementation_commit':implementation_commit,'files':files,'inherited65_map':old,'stdlib779_map':std,'windows_library_identities':windows,'library_event_observer':observer,'stdlib_layout':stdlib_layout,'cache_policy':cache,'allowed_initial_loaded_libraries':allowed,'qualification_review':review_anchor})
    config=seal({'kind':'REGISTERED_OBSERVER_STARTUP_MEASUREMENT_V1','scope':c['SCOPE'],'permission':c['PERMISSION'],'resources':c['RESOURCES'],'implementation_commit':implementation_commit,'source_files':sources,'runtime_files':files,'observer_registration':observer,'binary_fixture':binaries['observer_fixture_binary'],'worker_path':str(startup/'worker.py'),'offline_environment':offline_environment,'native_launch_authorized':authorized})
    registration=seal({'kind':'ROOT_REGISTERED_OBSERVER_STARTUP_LAUNCH_V1','implementation_commit':implementation_commit,'config_body_sha256':config['sha256'],'runtime_body_sha256':runtime['sha256'],'source_files':sources,'runtime_files':files,'profile_sha256':digest({'scope':c['SCOPE'],'permission':c['PERMISSION'],'resources':c['RESOURCES'],'ABI':c['ABI']}),'worker_path':config['worker_path'],'owner_approval':{'approved':authorized,'scope':c['SCOPE'],'evidence_reference':approved_evidence_reference if authorized else ''},'native_launch_authorized':authorized})
    return seal({'kind':'CLOSED_PROSPECTIVE_OBSERVER_STARTUP_DISPATCH_PACKET','config':config,'runtime':runtime,'registration':registration,'review':review,'dispatch_head_external':dispatch_head,'implementation_commit_external':implementation_commit,'source_promotion_complete':False,'dispatch_blobs_committed':False,'bootstrap_observed':False,'loaded_baseline_observed':False,'namespace_execution_proven':False,'resource_qualification_complete':False,'native_PASS':False,'unresolved':['promote reviewed source/2DLL/build receipt bytes and commit implementation ancestor','independent exact current source/runtime byte admission review: complete_current_byte_review remainsfalse','later separate config/runtime/registration dispatch commit with externalHEAD/file/body pins','actual owned startup/baseline/resource observation notinvented bygenerator']})

def inputs(promotion_path=HERE/'PROMOTION_MAP_001.json',*,promotion_file_sha256=PROMOTION_FILE,promotion_body_sha256=PROMOTION_BODY):
    passive=pinned_json(PASSIVE,PASSIVE_FILE,PASSIVE_BODY);build=pinned_json(BUILD,BUILD_FILE,BUILD_BODY)
    promotion=pinned_json(promotion_path,promotion_file_sha256,promotion_body_sha256);sources={};origins={}
    for target,row in promotion['entries'].items():
        d={k:row[k] for k in ('bytes','sha256','role')};descriptor(d)
        if d['role'] in ('observer_binary','observer_fixture_binary'):continue
        raw=read_source(row['source_path']);require(len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'],'exact promoted source origin descriptor')
        sources[target]=d;origins[target]=row['source_path']
    std=pinned_json(PACKAGE/'evidence/stdlib_inventory_reviewed_cpu_20261005/STDLIB_CANDIDATE_001.json','e5e185f980e2f0feae7bf49384089eedd4d47f5b41c10a828c6ef2a30ae777a5','f6284b28e689b2c5e42a97c12256b3e35a41395a6b9612c0a01ee786eb28b149')
    old=pinned_json(PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json','53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030','944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62')
    require(passive['inherited65_map']==old['runtime_pins'] and passive['stdlib779_map']==std['inventory']['runtime_pins'],'passive original maps exactly match external metadata')
    pe=pinned_json(PE_PATH,PE_FILE,PE_BODY);require(pe['complete_Windows_closure'] is pe['API_set_resolution_observed'] is pe['actual_loader_resolution_observed'] is pe['native_PASS'] is pe['native_authority_issued'] is False,'finite PE candidates are not loader-resolution/closure facts')
    current=pinned_json(HERE/'PASSIVE_WINDOWS_CANDIDATES_REVALIDATED_001.json',WINDOWS_FRESH_FILE,WINDOWS_FRESH_BODY)
    require(current['host_inventory_complete'] is current['loaded_module_baseline_observed'] is current['namespace_execution_proven'] is current['resources_observed'] is current['native_PASS'] is current['native_authority_issued'] is False and current['source_candidate_body_sha256']==PE_BODY,'fresh finite Windows byte pins retain PE prospective/unobserved scope')
    extra={p:{'bytes':d['bytes'],'sha256':d['sha256'],'role':'windows_system_binary'} for p,d in current['files'].items()}
    require(extra==pe['windows_candidate_files'],'fresh selected116 Windows candidates match exact original metadata identity set')
    return {'passive':passive,'build':build,'package':PACKAGE,'promoted_sources':sources,'constants_value':constants(origins[str(PACKAGE/'observer_startup_measurement_v1/contract.py')]),'stdlib_layout':{**std['inventory']['approved_alias_binding'],'python_version':std['inventory']['python_version']},'offline_environment':old['offline_environment'],'additional_windows':extra}

def main():
    import argparse,os
    p=argparse.ArgumentParser(description='Closed JSON observer startup packet; no native operations.')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--implementation-commit');p.add_argument('--dispatch-head');p.add_argument('--authorized',action='store_true');p.add_argument('--promotion',type=Path,default=HERE/'PROMOTION_MAP_001.json');p.add_argument('--promotion-file-sha256',default=PROMOTION_FILE);p.add_argument('--promotion-body-sha256',default=PROMOTION_BODY)
    args=p.parse_args();require(not args.output.exists() and args.output.parent.is_dir(),'write-once output packet')
    data=inputs(args.promotion,promotion_file_sha256=args.promotion_file_sha256,promotion_body_sha256=args.promotion_body_sha256)
    approval=None;reference=None
    if args.authorized:
        ap=HERE/'OWNER_APPROVAL_VERIFIED_001.json';approval=pinned_json(ap,APPROVAL_FILE,APPROVAL_BODY);reference=str(ap)+'#'+APPROVAL_BODY
    packet=construct(**data,implementation_commit=args.implementation_commit,dispatch_head=args.dispatch_head,approval=approval,approved_evidence_reference=reference)
    raw=(canonical(packet)+'\n').encode('ascii')
    with args.output.open('xb') as f:require(f.write(raw)==len(raw),'complete packet write');f.flush();os.fsync(f.fileno())
    require(args.output.read_bytes()==raw,'exact packet durable readback')
    print(json.dumps({'body_sha256':packet['sha256'],'runtime_files':len(packet['runtime']['files']),'source_files':len(packet['config']['source_files']),'native_launch_authorized':packet['config']['native_launch_authorized'],'native_PASS':False,'review_complete_current_byte_review':False}))

if __name__=='__main__':main()
