"""Finite explicit-path passive byte inventory; no imports/load/probes.

This module hashes selected files only. It never discovers a process/module
baseline or claims namespace/resource/runtime completeness or native authority.
"""
from pathlib import Path
import hashlib,json,math,stat,time

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
ANCHORS={
 'inherited65':('configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json','53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030','944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62'),
 'stdlib779':('evidence/stdlib_inventory_reviewed_cpu_20261005/STDLIB_CANDIDATE_001.json','e5e185f980e2f0feae7bf49384089eedd4d47f5b41c10a828c6ef2a30ae777a5','f6284b28e689b2c5e42a97c12256b3e35a41395a6b9612c0a01ee786eb28b149'),
 'runtime_candidate002':('evidence/prefix_runtime_source_candidate_reviewed_20261005/RUNTIME_CANDIDATE_002.json','93e4fe7996a9b39963f72c3d433f13ec97dfcde655bb9186fe5995c597fbb87d','049747fa477da33363affdff23f7a52cd32ec0bc5ec1b786a6c4bbe6e30732fe')}
ROLES={'python_source':{'.py'},'python_stub':{'.pyi'},'source_json':{'.json'},'source_markdown':{'.md'},'native_observer_source':{'.c','.h'},'fixture_source':{'.c'},'observer_binary':{'.dll'},'observer_fixture_binary':{'.dll'},'interpreter_binary':{'.exe','.dll','.pyd','.zip'},'windows_system_binary':{'.dll'},'extension_binary':{'.pyd','.dll'},'torch_binary':{'.pyd','.dll'},'qualified_bytecode_cache':{'.pyc'}}
MAX_FILES=32768
MAX_TOTAL=16*1024**3
MAX_FILE=8*1024**3
MAX_METADATA=16*1024**2

def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def pairs(rows):
    result={}
    for key,value in rows:require(key not in result,'duplicate metadata key');result[key]=value
    return result
def floating(text):
    value=float(text);require(math.isfinite(value),'nonfinite/overflow metadata float');return value
def parse(raw):return json.loads(raw,object_pairs_hook=pairs,parse_float=floating,parse_constant=lambda _:require(False,'nonfinite metadata'))

class Readers:
    """The sole default filesystem interface; _collect accepts inert replacements."""
    def clock(self):return time.monotonic()
    def metadata(self,path):
        p=Path(path);m=p.lstat()
        return {'bytes':m.st_size,'mtime_ns':m.st_mtime_ns,'inode':m.st_ino,'mode':m.st_mode,'file_attributes':getattr(m,'st_file_attributes',0),'reparse_tag':getattr(m,'st_reparse_tag',None),'is_symlink':p.is_symlink()}
    def resolve(self,path):return Path(path).resolve()
    def open(self,path):return Path(path).open('rb')
    def opened_metadata(self,stream):
        import os
        m=os.fstat(stream.fileno())
        return {'bytes':m.st_size,'mtime_ns':m.st_mtime_ns,'inode':m.st_ino,'mode':m.st_mode}

def _path(text,readers,alias_table):
    require(type(text) is str and 0<len(text)<=4096 and not any(c in text for c in '*?'),'bounded explicit path without wildcard')
    p=Path(text);require(p.is_absolute() and str(p)==text and '..' not in p.parts,'canonical absolute selected path')
    permitted=None
    if alias_table is not None:
        require(type(alias_table) is dict and set(alias_table)=={'logical_base_prefix','physical_base_prefix','file_attributes','reparse_tag','is_symlink'},'exact known alias metadata table')
        logical=Path(alias_table['logical_base_prefix']);physical=Path(alias_table['physical_base_prefix'])
        if p.is_relative_to(logical):
            m=readers.metadata(logical)
            require(m['file_attributes']==alias_table['file_attributes'] and m['reparse_tag']==alias_table['reparse_tag'] and m['is_symlink'] is alias_table['is_symlink'] and readers.resolve(logical)==physical,'unchanged exact approved alias binding')
            mapped=physical/p.relative_to(logical);require(readers.resolve(p)==mapped,'known alias child remains contained');p=mapped;permitted=logical
    require(readers.resolve(p)==p,'exact selected physical path')
    for part in (p,*p.parents):
        m=readers.metadata(part);require(m['is_symlink'] is False and not(m['file_attributes']&0x400),'selected path has no other reparse alias')
    return p

def _collect_impl(paths,descriptorroles,readers=None,*,alias_table=None,progress):
    """Hash a finite exact list. No glob, tree walk, implicit module discovery."""
    readers=Readers() if readers is None else readers
    require(type(paths) is list and 0<len(paths)<=MAX_FILES and all(type(p) is str for p in paths) and len(paths)==len(set(paths)) and type(descriptorroles) is dict and set(descriptorroles)==set(paths),'finite exact path/descriptor population')
    started=readers.clock();total=0;result={};fingerprints={}
    def deadline():require(readers.clock()-started<=600,'bounded entire selected inventory deadline')
    for text in paths:
        deadline();d=descriptorroles[text]
        progress['attempted_paths'].append(text)
        require(type(d) is dict and set(d)=={'bytes','sha256','role'} and type(d['bytes']) is int and 0<=d['bytes']<=MAX_FILE and (d['sha256'] is None or sha(d['sha256'])) and d['role'] in ROLES and Path(text).suffix.lower() in ROLES[d['role']],'typed exact selected role/size/pin; no model payloads')
        total+=d['bytes'];require(total<=MAX_TOTAL,'bounded16GiB selected population')
        path=_path(text,readers,alias_table);before=readers.metadata(path)
        require(stat.S_ISREG(before['mode']) and before['bytes']==d['bytes'],'regular selected exact recorded size')
        h=hashlib.sha256();remaining=d['bytes']
        progress['current_read']={'path':text,'expected_descriptor':dict(d),'consumed_bytes':0,'remaining_bytes':remaining,'opened_metadata_verified':False,'hash_complete':False}
        with readers.open(path) as stream:
            selected_handle={k:before[k] for k in ('bytes','mtime_ns','inode','mode')}
            require(readers.opened_metadata(stream)==selected_handle,'opened handle matches selected pathname metadata')
            progress['current_read']['opened_metadata_verified']=True
            while remaining:
                deadline();chunk=stream.read(min(1024**2,remaining));require(type(chunk) is bytes and 0<len(chunk)<=remaining,'complete bounded selected read progress')
                h.update(chunk);remaining-=len(chunk);progress['current_read'].update(consumed_bytes=d['bytes']-remaining,remaining_bytes=remaining);deadline()
            require(stream.read(1)==b'','selected file growth/truncation refused');deadline()
            require(readers.opened_metadata(stream)==selected_handle,'opened handle metadata stable after bounded read')
        after=readers.metadata(path);require(before==after,'selected pointer/metadata stable across read')
        observed=h.hexdigest();progress['current_read']['hash_complete']=True
        progress['observed_files'][text]={'bytes':d['bytes'],'sha256':observed,'role':d['role'],'physical_path':str(path),'expected_pin_matched':d['sha256'] is None or observed==d['sha256']}
        require(d['sha256'] is None or observed==d['sha256'],'selected original byte pin unchanged')
        result[text]={'bytes':d['bytes'],'sha256':observed,'role':d['role'],'physical_path':str(path)}
        fingerprints[text]=after
    for text,d in result.items():
        deadline();path=_path(text,readers,alias_table);require(readers.metadata(path)==fingerprints[text],'all selected pointers stable at final boundary')
    return {'files':result,'selected_path_list':list(paths),'selected_path_list_sha256':digest(paths),'selected_descriptor_map_sha256':digest(descriptorroles),'actual_hashed_bytes':total,'host_inventory_complete':False,'loaded_module_baseline_observed':False,'namespace_execution_proven':False,'resources_observed':False,'native_PASS':False,'native_authority_issued':False}

def _collect(paths,descriptorroles,readers=None,*,alias_table=None):
    progress={'attempted_paths':[],'observed_files':{}}
    try:return _collect_impl(paths,descriptorroles,readers,alias_table=alias_table,progress=progress)
    except BaseException as original:
        # Attach metadata to the original exception without invoking hooks or
        # replacing its type/traceback. The CLI preserves this partial record.
        try:
            attributes=BaseException.__dict__['__dict__'].__get__(original,type(original))
            if type(attributes) is dict:attributes['__inventory_partial__']=progress
        except BaseException:pass
        raise

def metadata():
    """Read only externally pinned committed JSON; never dereference its paths."""
    values={};references={};reader=Readers()
    for name,(relative,file_pin,body_pin) in ANCHORS.items():
        path=PACKAGE/relative;before=reader.metadata(path);require(stat.S_ISREG(before['mode']) and 0<before['bytes']<=MAX_METADATA,'bounded regular metadata anchor')
        physical=_path(str(path),reader,None)
        with reader.open(physical) as f:raw=f.read(before['bytes']+1)
        require(len(raw)==before['bytes'] and reader.metadata(path)==before and hashlib.sha256(raw).hexdigest()==file_pin,'exact stable external metadata file')
        value=parse(raw);require(type(value) is dict and value['sha256']==body_pin==digest({k:v for k,v in value.items() if k!='sha256'}),'exact external metadata body')
        values[name]=value;references[name]={'path':str(path),'bytes':len(raw),'file_sha256':file_pin,'body_sha256':body_pin}
    old=values['inherited65']['runtime_pins'];stdlib=values['stdlib779']['inventory'];candidate=values['runtime_candidate002']
    require(len(old)==65 and len(stdlib['runtime_pins'])==779 and candidate['inherited65_map']==old and candidate['stdlib779_inventory']['runtime_pins']==stdlib['runtime_pins'],'exact original65/779 map populations/content retained')
    require(candidate['native_runtime_closure_complete'] is candidate['Root_binary_qualification_complete'] is candidate['native_launch_authorized'] is False,'prospective candidate never supplies current qualification')
    return {'references':references,'inherited65_map':old,'stdlib779_map':stdlib['runtime_pins'],'known_alias':stdlib['approved_alias_binding'],'stdlib_version':stdlib['python_version'],'prospective_torch_source_hashes':candidate['torch_source_hashes'],'prospective_torch_binary_obligations':candidate['torch_UNREAD_binary_obligations'],'prospective_known_pin_union':candidate['expected_known_pin_union']}

def build_inventory(manifest,*,expected_manifest_sha256,readers=None):
    """Compile Root's explicitly supplied finite map; never select paths itself."""
    require(type(manifest) is dict and set(manifest)=={'kind','source_commit','files','source_paths','observer_binary_path','fixture_binary_path','metadata_map_digests','sha256'} and manifest['kind']=='ROOT_SELECTED_OBSERVER_ONLY_PASSIVE_INVENTORY_V1','exact Root-selected passive input schema')
    require(sha(expected_manifest_sha256) and manifest['sha256']==expected_manifest_sha256==digest({k:v for k,v in manifest.items() if k!='sha256'}),'external exact selected manifest/body')
    head=manifest['source_commit'];require(type(head) is str and len(head)==40 and all(c in '0123456789abcdef' for c in head),'exact supplied full source HEAD identity')
    pinned=metadata();require(manifest['metadata_map_digests']=={'inherited65':digest(pinned['inherited65_map']),'stdlib779':digest(pinned['stdlib779_map'])},'original inherited65/stdlib779 metadata map joins')
    files=manifest['files'];require(type(files) is dict,'explicit Root-selected file map')
    source_paths=manifest['source_paths'];require(type(source_paths) is list and source_paths and len(set(source_paths))==len(source_paths) and all(p in files and files[p]['role'] in ('python_source','native_observer_source','fixture_source','source_json','source_markdown') and files[p]['sha256'] is not None for p in source_paths),'pinned selected source closure')
    for name,role in (('observer_binary_path','observer_binary'),('fixture_binary_path','observer_fixture_binary')):require(manifest[name] in files and files[manifest[name]]['role']==role and files[manifest[name]]['sha256'] is not None,'exact selected observer/fixture descriptor identity')
    for path,d in files.items():
        recorded=pinned['prospective_known_pin_union'].get(path)
        if recorded is not None:require(d['sha256']==recorded,'selected historical/source pin cannot be substituted')
        obligation=pinned['prospective_torch_binary_obligations'].get(path)
        if obligation is not None:
            expected=obligation['installation_record'];require(d['bytes']==expected['expected_RECORD_bytes'] and d['sha256']==expected['expected_RECORD_sha256'],'selected unread Torch binary preserves exact RECORD obligation')
    selection=list(files);collected=_collect(selection,files,readers,alias_table=pinned['known_alias'])
    value={'kind':'PASSIVE_SELECTED_OBSERVER_INVENTORY_UNQUALIFIED','source_commit':head,'input_manifest_sha256':expected_manifest_sha256,'source_closure_sha256':digest({p:files[p] for p in source_paths}),'metadata_references':pinned['references'],'inherited65_map':pinned['inherited65_map'],'stdlib779_map':pinned['stdlib779_map'],'unselected_prospective_paths':[p for p in pinned['prospective_known_pin_union'] if p not in files],**collected}
    value['sha256']=digest(value);return value

def read_selected_manifest(path,*,file_sha256,body_sha256):
    require(sha(file_sha256) and sha(body_sha256),'external selected manifest file/body anchors')
    readers=Readers();physical=_path(str(Path(path).absolute()),readers,None);before=readers.metadata(physical)
    require(stat.S_ISREG(before['mode']) and 0<before['bytes']<=MAX_METADATA,'bounded regular selection manifest')
    with readers.open(physical) as f:raw=f.read(before['bytes']+1)
    require(len(raw)==before['bytes'] and readers.metadata(physical)==before and hashlib.sha256(raw).hexdigest()==file_sha256,'exact original selection manifest bytes')
    value=parse(raw);require(value.get('sha256')==body_sha256==digest({k:v for k,v in value.items() if k!='sha256'}),'exact original selection body');return value

def main():
    import argparse,os
    parser=argparse.ArgumentParser(description='Passive explicit-path hashing only; no current-host completeness or native qualification.')
    parser.add_argument('--manifest',required=True,type=Path);parser.add_argument('--file-sha256',required=True);parser.add_argument('--body-sha256',required=True);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();require(not args.output.exists() and args.output.parent.is_dir(),'write-once output with existing parent')
    before=Readers().metadata(Path(__file__).absolute());source_raw=Path(__file__).read_bytes();source_sha=hashlib.sha256(source_raw).hexdigest()
    selected=read_selected_manifest(args.manifest,file_sha256=args.file_sha256,body_sha256=args.body_sha256)
    try:value=build_inventory(selected,expected_manifest_sha256=args.body_sha256)
    except BaseException as original:
        attrs=BaseException.__dict__['__dict__'].__get__(original,type(original));partial=attrs.get('__inventory_partial__',{}) if type(attrs) is dict else {}
        error_args=BaseException.args.__get__(original,type(original));message=error_args[0][:4096] if type(error_args) is tuple and len(error_args)==1 and type(error_args[0]) is str else None
        value={'kind':'PASSIVE_SELECTED_INVENTORY_FAILURE_PRESERVED','input_manifest_file_sha256':args.file_sha256,'input_manifest_body_sha256':args.body_sha256,'selected_input':selected,'partial_observations':partial,'error':{'type':type.__getattribute__(type(original),'__name__'),'message':message},'builder_source_sha256':source_sha,'native_PASS':False,'native_authority_issued':False,'host_inventory_complete':False}
        value['sha256']=digest(value);raw=(canonical(value)+'\n').encode('ascii')
        with args.output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        require(args.output.read_bytes()==raw,'durable partial inventory failure readback')
        raise
    require(Readers().metadata(Path(__file__).absolute())==before and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==source_sha,'actual builder source bytes stable before/after')
    value.pop('sha256');value['builder_source']={'path':str(Path(__file__).absolute()),'bytes':len(source_raw),'sha256':source_sha,'current_module_execution_attested':False};value['input_manifest_file_sha256']=args.file_sha256;value['sha256']=digest(value)
    raw=(canonical(value)+'\n').encode('ascii')
    with args.output.open('xb') as f:require(f.write(raw)==len(raw),'complete inventory output');f.flush();os.fsync(f.fileno())
    require(args.output.read_bytes()==raw,'durable passive inventory readback')
    print(json.dumps({'files':len(value['files']),'bytes':value['actual_hashed_bytes'],'body_sha256':value['sha256'],'native_PASS':False,'host_inventory_complete':False}))

if __name__=='__main__':main()
