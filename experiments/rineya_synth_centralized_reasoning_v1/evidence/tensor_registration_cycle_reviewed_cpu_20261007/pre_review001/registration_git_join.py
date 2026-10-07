"""CPU-only implementation-ancestor/dispatch packet Git join candidate."""
from pathlib import Path
import hashlib,json,subprocess

def require(value,message):
    if not value:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode('ascii')).hexdigest()
def full_commit(x):return type(x) is str and len(x)==40 and all(c in '0123456789abcdef' for c in x)
def sha(x):return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def physical(path):
    p=Path(path).absolute();require(p.resolve()==p,'physical selected registration path')
    for x in (p,*p.parents):require(not x.is_symlink() and not(getattr(x.lstat(),'st_file_attributes',0)&0x400),'registration aliases/reparse refused')
    return p
def read(path):
    p=physical(path);b=p.stat();require(p.is_file() and b.st_size<=16*1024**2,'bounded registration/source bytes');raw=p.read_bytes();a=p.stat();require(len(raw)==b.st_size and (b.st_size,b.st_mtime_ns,b.st_ino)==(a.st_size,a.st_mtime_ns,a.st_ino),'stable selected registration/source bytes');return raw
def _factory():
    tokens=set()
    class CPU:__slots__=()
    def cap():x=CPU();tokens.add(x);return x
    def check(x):require(type(x) is CPU and x in tokens,'private CPU registration join capability')
    return cap,check
_CPU_capability,_CPU_check=_factory()
def _CPU_join(repository,*,packets,source_files,implementation_commit,expected_head,git_output_exceptions,_capability):
    """Join real Git objects; no native context or runtime qualification minted."""
    _CPU_check(_capability);root=physical(repository);require(root.is_dir() and full_commit(implementation_commit) and full_commit(expected_head) and implementation_commit!=expected_head,'distinct full implementation and external dispatch commits')
    require(type(packets) is dict and set(packets)=={'config','runtime','registration'} and type(source_files) is dict and 0<len(source_files)<=4096 and type(git_output_exceptions) is dict and len(git_output_exceptions)<=4096,'finite exact packet/source/exception scope')
    def git(*args):return subprocess.run(['git','-c','maintenance.auto=false','-c','gc.auto=0',*args],cwd=root,capture_output=True,check=True,timeout=30).stdout
    require(git('rev-parse','HEAD').decode('ascii').strip()==expected_head,'actual clean dispatch HEAD exact')
    git('merge-base','--is-ancestor',implementation_commit,expected_head)
    dirty=git('status','--porcelain=v1','-z','--untracked-files=all')
    for row in dirty.split(b'\0'):
        if not row:continue
        require(row.startswith(b'?? '),'tracked changes never exempted');path=str(root/row[3:].decode('utf-8','strict'));require(git_output_exceptions.get(path)=={'role':'run_output'},'only exact registered output leaf exemption')
    def relative(text):
        p=physical(text);require(p.is_relative_to(root) and p!=root,'selected Git leaf in repository');return p.relative_to(root).as_posix()
    originals={};values={}
    for name,entry in packets.items():
        require(type(entry) is dict and set(entry)=={'path','file_sha256','body_sha256'} and sha(entry['file_sha256']) and sha(entry['body_sha256']),'exact externally selected packet anchors');raw=read(entry['path']);require(hashlib.sha256(raw).hexdigest()==entry['file_sha256'],'external packet byte anchor')
        value=json.loads(raw);require(type(value) is dict and value.get('sha256')==entry['body_sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'external packet body anchor')
        require(value.get('source_commit')==implementation_commit,'all packets join implementation ancestor')
        require(git('show',expected_head+':'+relative(entry['path']))==raw,'config/runtime/registration exact dispatch committed blobs');values[name]=value;originals[entry['path']]=raw
    require(values['registration'].get('config_body_sha256')==values['config']['sha256'] and values['registration'].get('runtime_body_sha256')==values['runtime']['sha256'],'exact registered config/runtime body joins')
    require(values['config'].get('source_files')==values['registration'].get('source_files')==source_files,'same selected implementation source map')
    for text,d in source_files.items():
        require(type(d) is dict and set(d)=={'bytes','sha256','role'} and type(d['bytes']) is int and 0<=d['bytes']<=16*1024**2 and sha(d['sha256']) and d['role'] in {'python_source','source_json','source_markdown','native_observer_source'},'bounded selected implementation source descriptor');raw=read(text)
        require(len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'] and git('show',implementation_commit+':'+relative(text))==raw,'implementation blob/current selected source exact byte joins');originals[text]=raw
    require(git('rev-parse','HEAD').decode('ascii').strip()==expected_head and git('status','--porcelain=v1','-z','--untracked-files=all')==dirty,'same dispatch HEAD/status after consumed blobs')
    for text,raw in originals.items():require(read(text)==raw,'all original packet/source bytes stable after Git joins')
    return {'kind':'CPU_IMPLEMENTATION_DISPATCH_GIT_JOIN','implementation_commit':implementation_commit,'dispatch_head':expected_head,'packet_bodies':{n:v['sha256'] for n,v in values.items()},'source_inventory_sha256':digest(source_files),'fixture_only':True,'native_runtime_qualified':False,'native_authority_issued':False}
def native_join(*args,**kwargs):raise ValueError('CPU registration Git join has no native entry')
