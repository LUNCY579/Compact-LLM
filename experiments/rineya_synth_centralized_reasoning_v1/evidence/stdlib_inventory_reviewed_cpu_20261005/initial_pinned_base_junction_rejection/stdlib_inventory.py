"""Finite byte-only interpreter/stdlib superset; no complete-machine authority."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,os,sys
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='stdlib_runtime_closure_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE
sys.path.insert(0,str(PACKAGE))
from rs_contract import canonical,digest,require

VERSION='finite-hermes-python-stdlib-byte-superset-1'
KIND='STAGED_CPU_STDLIB_RUNTIME_SUPERSET_NOT_NATIVE_REGISTRATION_V1'
BASE=Path('C:/Users/clclc/AppData/Roaming/uv/python/cpython-3.11-windows-x86_64-none')
PYTHON_VERSION='3.11.15'
FORBIDDEN={'torch','transformers','tensorflow','jax','flax','tokenizers','jinja2','markupsafe'}
MAX_FILE=64*1024**2
MAX_FILES=4096
MAX_TOTAL=128*1024**2


def no_execution():
    require(not any(name.split('.',1)[0] in FORBIDDEN for name in sys.modules),'inventory may not execute model/encoder packages')


def pointer(path,*,file=False,root=None):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        require(not part.is_symlink(),'symlink runtime pointer forbidden')
        if part.exists():require(not getattr(part.lstat(),'st_file_attributes',0)&0x400,'reparse runtime pointer forbidden')
    require(path.exists() and path.resolve()==path,'exact existing absolute runtime pointer')
    require(path.is_file() if file else path.is_dir(),'typed regular runtime pointer')
    if root is not None:require(path.is_relative_to(Path(root).absolute()),'runtime path escaped declared root')
    return path


def info(path,root=None):
    path=pointer(path,file=True,root=root);size=path.stat().st_size
    require(0<=size<=MAX_FILE and (size>0 or path.suffix=='.py'),'bounded executable/source file; empty Python package sources permitted')
    sha=hashlib.sha256();read=0
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(1024**2),b''):read+=len(raw);sha.update(raw)
    require(read==size==path.stat().st_size,'runtime file changed during hash')
    return {'path':str(path),'bytes':size,'sha256':sha.hexdigest()}


def recursive(directory,extensions,exclude=()):
    directory=pointer(directory);result=[]
    for current,dirs,files in os.walk(directory,followlinks=False):
        dirs[:]=sorted(name for name in dirs if name not in exclude)
        for name in dirs:pointer(Path(current)/name,root=directory)
        for name in sorted(files):
            path=Path(current)/name
            if path.suffix.lower() in extensions:result.append(pointer(path,file=True,root=directory))
    return result


def roster(base,executable,version,*,fixture=False):
    no_execution();base=pointer(base);executable=pointer(executable,file=True)
    require(type(version) is str and version==PYTHON_VERSION,'exact pinned Python3.11.15 version')
    if not fixture:
        require(base==BASE.absolute()==Path(sys.base_prefix).absolute() and executable==Path(sys.executable).absolute()
            and sys.version.split()[0]==PYTHON_VERSION,'actual pinned Hermes base/version/executable')
    lib=pointer(base/'Lib');dlls=pointer(base/'DLLs');paths={}
    def add(path,category):
        key=str(path);require(key not in paths,'duplicate runtime roster path');paths[key]=category
    for path in recursive(lib,{'.py'},('site-packages','__pycache__')):add(path,'stdlib_python_source')
    for path in recursive(dlls,{'.pyd','.dll'}):add(path,'stdlib_extension_runtime')
    # Immediate interpreter-root binary/ZIP roster is exact, including pythonw.
    for path in sorted(base.iterdir()):
        if path.suffix.lower() in ('.exe','.dll','.zip'):
            add(pointer(path,file=True,root=base),'interpreter_root_executable_runtime')
    if str(executable) not in paths:add(executable,'executing_Hermes_launcher')
    optional=base/'python311.zip'
    zip_state={'path':str(optional),'present':optional.exists()}
    if optional.exists():require(str(pointer(optional,file=True,root=base)) in paths,'present standardlibZIP belongs to exact roster')
    require(0<len(paths)<=MAX_FILES,'finite exact runtime roster size')
    return {'base_prefix':str(base),'executable':str(executable),'python_version':version,'fixture':fixture,
        'policy':VERSION,'paths':dict(sorted(paths.items())),'optional_stdlib_zip':zip_state,
        'excluded_directory_names':['site-packages','__pycache__'],'Windows_system_driver_scope':'inherited separately; no machine closure claim'}


def inventory(base=None,executable=None,version=None,*,fixture=False):
    base=BASE if base is None else base;executable=sys.executable if executable is None else executable;version=PYTHON_VERSION if version is None else version
    before=roster(base,executable,version,fixture=fixture)
    files={path:info(path,base if Path(path).is_relative_to(Path(base).absolute()) else None) for path in before['paths']}
    require(sum(item['bytes'] for item in files.values())<=MAX_TOTAL,'finite runtime total bytes')
    after=roster(base,executable,version,fixture=fixture);require(before==after,'runtime roster changed during inventory')
    return {**before,'files':files,'runtime_pins':{path:item['sha256'] for path,item in files.items()}}


def validate(value,expected_body_sha256,*,fixture=False):
    require(type(expected_body_sha256) is str and value.get('sha256')==expected_body_sha256
        ==digest({key:item for key,item in value.items() if key!='sha256'}),'externally selected immutable runtime candidate')
    require(value.get('kind')==KIND and value.get('native_authority_issued') is False and value.get('complete_machine_trust_claimed') is False,'CPU runtime superset only')
    current=inventory(value['inventory']['base_prefix'],value['inventory']['executable'],value['inventory']['python_version'],fixture=fixture)
    require(current==value['inventory'],'exact current runtime version/root/roster/byte closure; stale or omitted path')
    require(value['helper_source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'current inventory helper source bytes')
    return True


def candidate(base=None,executable=None,version=None,*,fixture=False):
    first=inventory(base,executable,version,fixture=fixture);second=inventory(base,executable,version,fixture=fixture)
    require(first==second,'immutable full runtime bytes before/after')
    value={'schema_version':1,'kind':KIND,'inventory':first,'before_after_byte_equal':True,
        'helper_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_execution_policy_scope':'finite interpreter/stdlib executable superset; site packages supplied by separate encoder package45',
        'encoder_packages_executed':False,'framework_model_GPU_executed':False,'native_registration_written':False,
        'native_config_written':False,'native_authority_issued':False,'complete_machine_trust_claimed':False,
        'reviewed_additive_encoder_contract_implemented':False,'fixture':fixture}
    value['sha256']=digest(value);return value


def save_staged(path,value):
    target=Path(path).absolute();require(STAGED and target.parent==HERE and target.suffix=='.json'
        and not target.exists() and not target.is_symlink(),'new staging-only runtime candidate; no final registration/config')
    raw=(canonical(value)+'\n').encode()
    with target.open('xb') as stream:require(stream.write(raw)==len(raw),'complete runtime candidate write');stream.flush();os.fsync(stream.fileno())
    return info(target)
