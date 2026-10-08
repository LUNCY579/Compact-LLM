"""Independent final registration/window audit; no native module imports."""
from pathlib import Path
from types import ModuleType, CodeType
import hashlib, importlib.util, json, marshal, os, stat, subprocess, time

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1];ROOT=PACKAGE.parents[1]
INPUTS=PACKAGE/'artifacts/import_exact_registration_work'
EXPECTED='8dccdb88870f85f98f6498d432a85c818803387b'
SELECTION=INPUTS/'ACTUAL_SOURCE_RUNTIME_SELECTION005.json'
READER=HERE/'review_current_bytes.py'
OLD_READER=PACKAGE/'artifacts/import_current_byte_review_work/review_current_bytes.py'
RECEIPT=HERE/'FRESH_IMPORT_CURRENT_BYTE_REVIEW005.json'
REVIEWED_PASSIVE=PACKAGE/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py'
START=time.monotonic()

def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sealed(p):
    v=json.loads(Path(p).read_bytes());require(v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'}),'exact packet seal');return v
def git(*args,data=None):
    return subprocess.run(['git','-c','maintenance.auto=false','-c','gc.auto=0',*args],cwd=ROOT,input=data,capture_output=True,check=True,timeout=120).stdout
def load(p,name):
    m=ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m

selection=sealed(SELECTION);recipe=sealed(INPUTS/'EXACT_IMPORT_RUN_RECIPE005.json')
config=sealed(INPUTS/'IMPORT_OBSERVATION_CONFIG_005.json')
runtime=sealed(INPUTS/'IMPORT_OBSERVATION_RUNTIME_005.json')
registration=sealed(INPUTS/'IMPORT_OBSERVATION_REGISTRATION_005.json')
reader_choice=sealed(HERE/'READER_SUCCESSOR_SELECTION.json')
metadata=[SELECTION,INPUTS/'EXACT_IMPORT_RUN_RECIPE005.json',INPUTS/'IMPORT_SOURCE_RECIPE005.json',INPUTS/'IMPORT_OBSERVATION_CONFIG_005.json',INPUTS/'IMPORT_OBSERVATION_RUNTIME_005.json',INPUTS/'IMPORT_OBSERVATION_REGISTRATION_005.json',INPUTS/'refresh_final.py',INPUTS/'test_final_registration.py',READER,OLD_READER,HERE/'READER_SUCCESSOR_SELECTION.json',REVIEWED_PASSIVE,RECEIPT]
pins_before={str(p):sha(p) for p in metadata}
require(pins_before[str(SELECTION)]=='c4a001e6ecd070855719a9119a488fa6b92beb4cbbd1728b1a99885881821a26','selected005 pin')
require(pins_before[str(READER)]=='10c46df8fa225f94e04a1904450f3a7dc47cc4cd8e9c69c088ab1b200b976b2f','reader005 pin')
require(pins_before[str(OLD_READER)]=='7286041069771edb48f5a801155c0c8903648279b3043138bddd697e7ab7f8c0','old reader pin')
require(pins_before[str(REVIEWED_PASSIVE)]=='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892','passive collector pin')
substitutions={
    "SELECTION=INPUTS/'ACTUAL_SOURCE_RUNTIME_SELECTION002.json'":"SELECTION=INPUTS/'ACTUAL_SOURCE_RUNTIME_SELECTION005.json'",
    "SELECTION_PIN='96007671eed07e5ba062ebd6f429b57ca1d06283481ab8ba26f3914389763466'":"SELECTION_PIN='c4a001e6ecd070855719a9119a488fa6b92beb4cbbd1728b1a99885881821a26'",
    "EXPECTED_HEAD='d39d64b1447c8f0baffdebcc4a0a1092f76bbd88'":"EXPECTED_HEAD='"+EXPECTED+"'",
    'IMPORT_OBSERVATION_RUNTIME_003.json':'IMPORT_OBSERVATION_RUNTIME_005.json',
    'IMPORT_OBSERVATION_CONFIG_003.json':'IMPORT_OBSERVATION_CONFIG_005.json',
    'IMPORT_OBSERVATION_REGISTRATION_003.json':'IMPORT_OBSERVATION_REGISTRATION_005.json',
    "selection['runtime_count']==8724":"selection['runtime_count']==8725",
    "len(pairs)==66,'exact59 plus7 unique passive cache pairs'":"len(pairs)==67,'exact59 plus8 unique passive cache pairs'",
    "'additional_passive_cache_pairs':7":"'additional_passive_cache_pairs':8",
    'FRESH_IMPORT_CURRENT_BYTE_REVIEW001.json':'FRESH_IMPORT_CURRENT_BYTE_REVIEW005.json',
    'FRESH_IMPORT_CURRENT_BYTE_REVIEW_FAILURE001.json':'FRESH_IMPORT_CURRENT_BYTE_REVIEW_FAILURE005.json'}
expected_text=OLD_READER.read_bytes().decode('utf-8');counts={}
for a,b in substitutions.items():
    counts[a]=expected_text.count(a);require(counts[a]>0,'substitution exists');expected_text=expected_text.replace(a,b)
require(expected_text.encode()==READER.read_bytes() and counts==reader_choice['substitution_counts'],'only exact listed reader substitutions; algorithm retained')
sources=selection['source_files'];files=selection['runtime_files'];selected={**sources,**files}
require(selection['implementation_commit']==recipe['implementation_commit']==config['source_commit']==runtime['source_commit']==registration['source_commit']==EXPECTED,'implementation joins')
require(sources==config['source_files']==registration['source_files'] and files==config['runtime_files']==registration['runtime_files']==runtime['files'],'whole source/runtime joins')
require((len(sources),len(files),len(selected))==(196,8725,8921) and not set(sources)&set(files),'exact domains')
require(selection['source_inventory_sha256']==digest(sources) and selection['runtime_inventory_sha256']==digest(files),'inventory digests')
for label,reference in recipe['packet_references'].items():
    path=Path(reference['path']);packet=sealed(path);require(sha(path)==reference['file_sha256'] and packet['sha256']==reference['body_sha256'],'recipe packet refs')
require(recipe['source_runtime_selection']==reader_choice['selection_reference'],'reader/recipe same selection')
require(config['native_launch_authorized'] is registration['native_launch_authorized'] is recipe['native_launch_authorized'] is config['native_authority_issued'] is False,'closed launch')
require(config['owner_approval'] is registration['owner_approval'] is recipe['owner_grant_reference'] is recipe['owner_receipt_reference'] is recipe['external_dispatch_head'] is recipe['complete_current_byte_review'] is None,'no fabricated witnesses')
qualifier=load(PACKAGE/'import_runtime_qualification.py','independent_final_qualifier')
require(config['permission']==recipe['permission']==qualifier.PERMISSION and config['resources']==recipe['resources']==qualifier.B.RESOURCES,'original scope/limits')
require(registration['profile_sha256']==digest({'scope':qualifier.SCOPE,'permission':qualifier.PERMISSION,'resources':qualifier.B.RESOURCES}),'profile digest')
qualifier.import_git_exception_policy(registration['git_output_exceptions'],sources,files,runtime['library_event_observer'],qualifier.B)
qualifier._source_roster({'config':config,'runtime':runtime},qualifier.B,qualifier.QualificationBudget())
require(len(registration['git_output_exceptions'])==recipe['exact_preservation_exception_count']==214,'exact future leaves')
command=recipe['future_command'];anchors=command['nine_anchors'];refs=recipe['packet_references']
require(anchors=={'expected_config_file':refs['config']['file_sha256'],'expected_config_body':refs['config']['body_sha256'],'registration_path':refs['registration']['path'],'expected_registration_file':refs['registration']['file_sha256'],'expected_registration_body':refs['registration']['body_sha256'],'runtime_path':refs['runtime']['path'],'expected_runtime_file':refs['runtime']['file_sha256'],'expected_runtime_body':refs['runtime']['body_sha256'],'expected_head':None},'exact nine anchors')
require(command['argv_prefix']==['-I','-S','-B',str(PACKAGE/'import_dispatch_postrun.py'),'--config',refs['config']['path'],'--output-dir',recipe['output_directory'],'--preserved-dir',recipe['preserved_directory']],'exact postrun command')
require(command['required_owner_reference_arguments']==['--'+prefix+'-'+suffix for prefix in ('grant','owner-receipt') for suffix in ('path','file-sha256','body-sha256')],'all6 owner reference args')
require(command['separate_owner_references_required'] is True and not Path(recipe['output_directory']).exists() and not Path(recipe['preserved_directory']).exists(),'future output absent')
head_before=git('rev-parse','HEAD').decode().strip();status_before=git('status','--porcelain=v1','-z');require(head_before==EXPECTED and status_before==b'','clean final implementation')
# Actual immutable Git source blobs, independent from the builder's comparison.
ordered=sorted(sources);requests=[EXPECTED+':'+Path(p).relative_to(ROOT).as_posix() for p in ordered]
blobs=git('cat-file','--batch',data=('\n'.join(requests)+'\n').encode());cursor=0;blob_rows={};source_bytes=0
for path in ordered:
    end=blobs.index(b'\n',cursor);header=blobs[cursor:end].decode().split();require(len(header)==3 and header[1]=='blob','Git blob header');size=int(header[2]);cursor=end+1;blob=blobs[cursor:cursor+size];cursor+=size;require(blobs[cursor:cursor+1]==b'\n','Git blob boundary');cursor+=1
    disk=Path(path).read_bytes();d=sources[path];require(blob==disk and len(disk)==d['bytes'] and hashlib.sha256(disk).hexdigest()==d['sha256'],'actual immutable source match');source_bytes+=len(disk);blob_rows[path]={'git_object_id':header[0],'bytes':len(disk),'sha256':d['sha256'],'exact_disk_blob_match':True}
require(cursor==len(blobs),'Git blob batch exhausted')
v=sealed(RECEIPT);require(v['implementation_commit']==EXPECTED and v['source_files']==sources and v['runtime_files']==files,'window source/runtime joins')
require(v['review_script_sha256']==pins_before[str(READER)] and v['reader_source_sha256']==pins_before[str(REVIEWED_PASSIVE)],'window reader source joins')
total=sum(d['bytes'] for d in selected.values());require(total==v['actual_hashed_bytes']==4068295348,'complete8921 byte total')
require(v['physical_read_counters']=={'bytes':total,'opens':8921,'opened_metadata_checks':17842},'all8921 streams accounted')
require(v['source_count']==196 and v['runtime_count']==8725 and v['source_inventory_sha256']==digest(sources) and v['runtime_inventory_sha256']==digest(files) and v['selected_descriptor_map_sha256']==digest(selected) and v['selected_path_list_sha256']==digest(sorted(selected)),'all receipt inventory joins')
require({p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in v['observed_files'].items()}==selected,'all observed bytes/descriptors')
require(v['git_before']==v['git_after']=={'HEAD':EXPECTED,'dirty_porcelain':''} and v['known_alias_before']==v['known_alias_after'],'window Git/alias stability')
require(v['whole_selected_byte_review_completed'] is True and v['byte_budget']==16*1024**3 and v['deadline_seconds']==600 and 0<v['single_window_seconds']<600,'complete bounded single window')
for label,anchor in v['metadata_anchors'].items():
    path=Path(anchor['path']);raw=path.read_bytes();value=sealed(path);require(len(raw)==anchor['bytes'] and sha(path)==anchor['file_sha256'] and value['sha256']==anchor['body_sha256'],'window metadata anchors stable')
reader=load(REVIEWED_PASSIVE,'independent_final_passive_reader');R=reader.Readers();layout=runtime['stdlib_layout'];alias={k:layout[k] for k in ('logical_base_prefix','physical_base_prefix','file_attributes','reparse_tag','is_symlink')};rosters={}
for name,row in runtime['rosters'].items():
    root=reader._path(row['root'],R,alias);actual={}
    for folder,dirs,leaves in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in row['excluded_directory_names'])
        for d in dirs:
            m=(Path(folder)/d).lstat();require(not getattr(m,'st_file_attributes',0)&0x400,'no roster directory reparse')
        for leaf in leaves:
            p=Path(folder)/leaf
            if p.suffix.lower() not in row['suffixes']:continue
            m=p.lstat();require(stat.S_ISREG(m.st_mode) and not p.is_symlink() and not getattr(m,'st_file_attributes',0)&0x400,'physical regular roster leaf');actual[str(p)]={'bytes':m.st_size}
        if not row['recursive']:dirs[:]=[]
    require(actual==row['files'],'current physical roster exact');rosters[name]={**row,'files':actual}
require(len(rosters)==18 and rosters==v['rosters_before']==v['rosters_after'] and digest(rosters)==v['rosters_sha256'],'all18 rosters independently recaptured')
def graph(code):
    count=[0,0]
    def visit(x,depth):
        count[0]+=1;require(depth<=64 and count[0]<=65536,'bounded code graph')
        if type(x) is CodeType:return visit((x.co_code,x.co_exceptiontable,x.co_linetable,x.co_argcount,x.co_posonlyargcount,x.co_kwonlyargcount,x.co_nlocals,x.co_stacksize,x.co_flags,x.co_names,x.co_varnames,x.co_freevars,x.co_cellvars,x.co_name,x.co_qualname,x.co_firstlineno,x.co_filename,x.co_consts),depth+1)
        if type(x) is tuple:return tuple(visit(y,depth+1) for y in x)
        if type(x) is frozenset:return frozenset(visit(y,depth+1) for y in x)
        require(x is None or x is Ellipsis or type(x) in (bytes,str,int,float,bool,complex),'finite code constant type')
        if type(x) in (bytes,str):count[1]+=len(x);require(count[1]<=32*1024**2,'bounded code constants')
        return x
    return visit(code,0)
cache_rows=v['passive_cache_code_equivalence'];require(len(cache_rows)==67 and v['historical_cache_pairs']==59 and v['additional_passive_cache_pairs']==8,'all67 cache pairs');cache_bytes=source_cache_bytes=0
for text,row in cache_rows.items():
    cp=reader._path(text,R,alias);sp=reader._path(row['source_path'],R,alias);cached=cp.read_bytes();source=sp.read_bytes();cache_bytes+=len(cached);source_cache_bytes+=len(source)
    require(hashlib.sha256(cached).hexdigest()==row['cache_sha256']==files[text]['sha256'] and hashlib.sha256(source).hexdigest()==row['source_sha256']==selected[row['source_path']]['sha256'],'independent cache/source bytes')
    require(cached[:4]==importlib.util.MAGIC_NUMBER and cached[:4].hex()==row['magic_hex'],'selected CPython magic');flags=int.from_bytes(cached[4:8],'little');require(flags==row['flags'] and flags in (0,1,3),'finite cache header')
    code=marshal.loads(cached[16:]);require(type(code) is CodeType and reader._path(code.co_filename,R,alias)==sp and graph(code)==graph(compile(source,code.co_filename,'exec')),'independent complete graph equivalence')
    require(cached[8:16]==importlib.util.source_hash(source) if flags&1 else int.from_bytes(cached[12:16],'little')==len(source)&0xffffffff,'cache header source binding')
    require(row['cache_executed'] is row['source_execution_history_proven'] is False and row['complete_code_source_equivalence'] is True,'no cache execution/history claim')
sample=dict(sources)
for role in {d['role'] for d in files.values()}:
    p=min((p for p,d in files.items() if d['role']==role),key=lambda p:(files[p]['bytes'],p));sample[p]=files[p]
for row in rosters.values():
    p=min(row['files'],key=lambda p:(files[p]['bytes'],p));sample[p]=files[p]
probe=reader._collect(sorted(sample),sample,alias_table=alias);require({p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in probe['files'].items()}==sample,'representative runtime and all source byte recheck')
for key in ('native_runtime_qualified','native_context_minted','native_authority_issued','native_launch_authorized','native_API_called','Torch_psutil_ctypes_imported','current_main_or_bootstrap_execution_observed','resources_or_library_baseline_observed','DLL_cookie_or_provider_observed','source_execution_history_proven','native_PASS'):
    require(v[key] is False,'native/history claims remain false')
pins_after={str(p):sha(p) for p in metadata};require(pins_before==pins_after and git('rev-parse','HEAD').decode().strip()==head_before and git('status','--porcelain=v1','-z')==status_before,'all metadata/Git stable')
result={'kind':'INDEPENDENT_FINAL005_REGISTRATION_READER_CURRENT_BYTE_WINDOW_REVIEW001','implementation_commit':EXPECTED,'window_receipt_file_sha256':pins_before[str(RECEIPT)],'window_receipt_body_sha256':v['sha256'],'source_count':196,'runtime_count':8725,'selected_files':8921,'whole_window_selected_bytes':total,'full_window_seconds':v['single_window_seconds'],'physical_stream_counters_joined':True,'exact_reader_metadata_substitutions_only':True,'closed_packet_profile_nine_anchors_and_six_owner_args_verified':True,'exact_source_disk_blob_matches':196,'source_blob_rows':blob_rows,'source_bytes_independently_rehashed_and_blob_joined':source_bytes,'rosters_independently_recaptured':18,'cache_pairs_independent_complete_graphs':67,'cache_bytes_independently_rechecked':cache_bytes,'cache_source_bytes_independently_rechecked':source_cache_bytes,'representative_files_plus_all_sources':len(sample),'representative_hash_bytes':probe['actual_hashed_bytes'],'pins_before':pins_before,'pins_after':pins_after,'pins_stable':True,'HEAD_clean_stable':True,'full_runtime_hash_repeated':False,'source_cache_code_objects_executed':False,'native_API_or_framework_called':False,'native_bootstrap_library_baseline_or_resources_qualified':False,'native_authority_issued':False,'review_source_sha256':sha(Path(__file__)),'elapsed_seconds':time.monotonic()-START}
result['sha256']=digest(result)
with (HERE/'INDEPENDENT_FINAL005_WINDOW001.json').open('x',encoding='utf-8',newline='\n') as out:json.dump(result,out,sort_keys=True,indent=2);out.write('\n')
print(canonical({'source_files':196,'runtime_files':8725,'cache_pairs':67,'blob_matches':196,'sample_files':len(sample),'sample_bytes':probe['actual_hashed_bytes'],'body_sha256':result['sha256'],'native_authority_issued':False}))
