"""CPU-only implementation-ancestor/dispatch packet Git join candidate."""
from pathlib import Path
import hashlib,json,math,os,stat,subprocess,time

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
    p=physical(path);b=p.stat();require(stat.S_ISREG(b.st_mode) and 0<=b.st_size<=16*1024**2,'bounded registration/source bytes')
    with p.open('rb') as f:o=os.fstat(f.fileno());raw=f.read(b.st_size+1);c=os.fstat(f.fileno())
    a=p.stat();identity=lambda x:(x.st_size,x.st_mtime_ns,x.st_ino,stat.S_IFMT(x.st_mode))
    require(len(raw)==b.st_size and identity(b)==identity(o)==identity(c)==identity(a),'stable bounded opened registration/source bytes');return raw
def pairs(rows):
    out={}
    for key,value in rows:require(key not in out,'duplicate registration JSON field');out[key]=value
    return out
def forbidden_constant(value):raise ValueError('nonfinite registration JSON constant')
def finite_float(text):
    value=float(text);require(math.isfinite(value),'nonfinite overflow registration JSON float');return value
def parse(raw):
    require(type(raw) is bytes and len(raw)<=16*1024**2,'bounded exact JSON bytes');return json.loads(raw.decode('utf-8','strict'),object_pairs_hook=pairs,parse_constant=forbidden_constant,parse_float=finite_float)
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
    started=time.monotonic();total=[0]
    def account(raw):
        total[0]+=len(raw);require(total[0]<=16*1024**3 and time.monotonic()-started<=600,'bounded total Git/source byte and time budget');return raw
    def git(*args):
        require(time.monotonic()-started<600,'bounded Git qualification deadline')
        return account(subprocess.run(['git','-c','maintenance.auto=false','-c','gc.auto=0',*args],cwd=root,capture_output=True,check=True,timeout=min(30,max(.001,600-(time.monotonic()-started)))).stdout)
    require(git('rev-parse','HEAD').decode('ascii').strip()==expected_head,'actual clean dispatch HEAD exact')
    git('merge-base','--is-ancestor',implementation_commit,expected_head)
    selected_packet_paths={entry.get('path') for entry in packets.values() if type(entry) is dict}
    for text,entry in git_output_exceptions.items():
        require(type(text) is str and physical(text).is_relative_to(root) and str(Path(text))==text and Path(text).suffix.lower() in {'.json','.jsonl','.txt','.log','.snapshot'} and entry=={'role':'run_output'} and text not in source_files and text not in selected_packet_paths,'exact physical registered nonexecutable output leaf exception')
    dirty=git('status','--porcelain=v1','-z','--untracked-files=all')
    for row in dirty.split(b'\0'):
        if not row:continue
        require(row.startswith(b'?? '),'tracked changes never exempted');path=str(root/row[3:].decode('utf-8','strict'));require(git_output_exceptions.get(path)=={'role':'run_output'},'only exact registered output leaf exemption')
    def relative(text):
        p=physical(text);require(p.is_relative_to(root) and p!=root,'selected Git leaf in repository');return p.relative_to(root).as_posix()
    originals={};values={}
    for name,entry in packets.items():
        require(type(entry) is dict and set(entry)=={'path','file_sha256','body_sha256'} and sha(entry['file_sha256']) and sha(entry['body_sha256']),'exact externally selected packet anchors');raw=account(read(entry['path']));require(hashlib.sha256(raw).hexdigest()==entry['file_sha256'],'external packet byte anchor')
        value=parse(raw);require(type(value) is dict and value.get('sha256')==entry['body_sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'external packet body anchor')
        require(value.get('source_commit')==implementation_commit,'all packets join implementation ancestor')
        blob=expected_head+':'+relative(entry['path']);size=git('cat-file','-s',blob).decode('ascii').strip();require(size.isdecimal() and len(size)<=10 and int(size)==len(raw),'packet immutable Git blob exact bounded size before capture')
        require(git('show',blob)==raw,'config/runtime/registration exact dispatch committed blobs');values[name]=value;originals[entry['path']]=raw
    require(values['registration'].get('config_body_sha256')==values['config']['sha256'] and values['registration'].get('runtime_body_sha256')==values['runtime']['sha256'],'exact registered config/runtime body joins')
    require(values['config'].get('source_files')==values['registration'].get('source_files')==source_files,'same selected implementation source map')
    require(values['config'].get('runtime_files')==values['registration'].get('runtime_files')==values['runtime'].get('files') and type(values['runtime'].get('files')) is dict,'same selected runtime map in all packets')
    for text,d in source_files.items():
        require(type(d) is dict and set(d)=={'bytes','sha256','role'} and type(d['bytes']) is int and 0<=d['bytes']<=16*1024**2 and sha(d['sha256']) and d['role'] in {'python_source','source_json','source_markdown','native_observer_source'},'bounded selected implementation source descriptor');raw=account(read(text))
        blob=implementation_commit+':'+relative(text);size=git('cat-file','-s',blob).decode('ascii').strip();require(size.isdecimal() and len(size)<=10 and int(size)==len(raw),'source immutable Git blob exact bounded size before capture')
        require(len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'] and git('show',blob)==raw,'implementation blob/current selected source exact byte joins');originals[text]=raw
    require(git('rev-parse','HEAD').decode('ascii').strip()==expected_head and git('status','--porcelain=v1','-z','--untracked-files=all')==dirty,'same dispatch HEAD/status after consumed blobs')
    for text,raw in originals.items():require(read(text)==raw,'all original packet/source bytes stable after Git joins')
    return {'kind':'CPU_IMPLEMENTATION_DISPATCH_GIT_JOIN','implementation_commit':implementation_commit,'dispatch_head':expected_head,'packet_bodies':{n:v['sha256'] for n,v in values.items()},'source_inventory_sha256':digest(source_files),'fixture_only':True,'native_runtime_qualified':False,'native_authority_issued':False}
def native_join(*args,**kwargs):raise ValueError('CPU registration Git join has no native entry')
