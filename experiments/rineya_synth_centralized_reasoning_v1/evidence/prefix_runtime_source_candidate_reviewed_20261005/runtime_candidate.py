"""Finite Torch package source hashes + UNREAD payload metadata. No imports/probes.

Do not invoke native.runtime_files or stdlib_inventory.inventory: those read
binary bytes. This preparation copies their reviewed JSON descriptors only.
"""
from pathlib import Path
import hashlib,json,os,stat
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
INHERITED=PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json'
STDLIB=PACKAGE/'evidence/stdlib_inventory_reviewed_cpu_20261005/STDLIB_CANDIDATE_001.json'
MAX_FILES=32768;MAX_SOURCE_FILE=16*1024**2;MAX_SOURCE_TOTAL=256*1024**2
SOURCE_SUFFIXES={'.py','.pyi'}
def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def json_source(path):
    path=Path(path);require(path.suffix=='.json','only JSON metadata anchor');raw=path.read_bytes();require(len(raw)<=16*1024**2,'bounded metadata JSON')
    return json.loads(raw),{'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def pointer(path,root):
    path=Path(path).absolute();require(path.is_relative_to(root) and path.resolve()==path,'exact physical package pointer')
    for p in (path,*path.parents):
        meta=p.lstat();require(not p.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400),'no package reparse alias')
    return path
def metadata_roster(root):
    root=Path(root).absolute();pointer(root,root);values={}
    for current,dirs,files in os.walk(root,followlinks=False):
        dirs.sort()
        for name in dirs:pointer(Path(current)/name,root)
        for name in sorted(files):
            path=pointer(Path(current)/name,root);meta=path.lstat();require(stat.S_ISREG(meta.st_mode),'regular package file metadata')
            require(len(values)<MAX_FILES,'finite package file ceiling')
            values[str(path)]={'bytes':meta.st_size,'role':'torch_python_source' if path.suffix=='.py' else 'torch_python_stub_source' if path.suffix=='.pyi' else 'torch_package_payload_UNREAD_METADATA_ONLY'}
    return dict(sorted(values.items()))
def hash_source(path,expected_size):
    path=Path(path);require(path.suffix in SOURCE_SUFFIXES and 0<=expected_size<=MAX_SOURCE_FILE,'Python text source only; no payload byte reads')
    raw=path.read_bytes();require(len(raw)==expected_size==path.stat().st_size,'stable source text bytes');raw.decode('utf-8')
    return hashlib.sha256(raw).hexdigest()
def merge_exact(*maps):
    result={}
    for mapping in maps:
        for path,pin in mapping.items():require(path not in result or result[path]==pin,'conflicting overlap expected pin');result[path]=pin
    return result
def build_candidate():
    inherited,inherit_ref=json_source(INHERITED);stdlib,stdlib_ref=json_source(STDLIB)
    old=inherited['runtime_pins'];require(len(old)==65,'exact inherited map metadata count')
    inv=stdlib['inventory'];require(len(inv['runtime_pins'])==779 and stdlib['native_authority_issued'] is False,'reviewed finite779 metadata only')
    torch=next(Path(p).parent for p in old if p.endswith('torch\\version.py'))
    before=metadata_roster(torch);sources={};total=0
    for path,entry in before.items():
        if Path(path).suffix in SOURCE_SUFFIXES:
            total+=entry['bytes'];require(total<=MAX_SOURCE_TOTAL,'finite source total ceiling');sources[path]=hash_source(path,entry['bytes'])
    after=metadata_roster(torch);require(before==after,'exact finite Torch roster metadata stable')
    require({path:hash_source(path,before[path]['bytes']) for path in sources}==sources,'all Torch source hashes stable before/after')
    expected=merge_exact(old,inv['runtime_pins'],sources)
    obligations={path:{**entry,'expected_recorded_sha256':expected.get(path),'current_bytes_hashed':False,
        'Root_qualified_byte_verification_required':True} for path,entry in before.items() if path not in sources}
    std_obligations={path:{'expected_reviewed_descriptor':inv['files'][path],'role':inv['paths'][path],
        'current_byte_verification_not_performed':True} for path in inv['runtime_pins']}
    value={'schema_version':1,'kind':'SOURCE_ONLY_FINITE_TORCH_RUNTIME_CANDIDATE_INCOMPLETE_NATIVE_CLOSURE',
        'inherited65_anchor':inherit_ref,'inherited65_map':old,'stdlib779_anchor':stdlib_ref,'stdlib779_inventory':inv,
        'torch_physical_root':str(torch),'torch_file_metadata':before,'torch_source_hashes':sources,
        'torch_UNREAD_payload_obligations':obligations,'stdlib779_current_revalidation_obligations':std_obligations,
        'expected_known_pin_union':expected,'roles_are_superset_not_actual_execution_proof':True,
        'source_before_after_equal':True,'metadata_before_after_equal':True,'source_bytes':total,
        'Torch_imported':False,'binary_model_payload_bytes_read':False,'OS_GPU_resource_probes_performed':False,
        'native_runtime_closure_complete':False,'Root_binary_qualification_complete':False,'worker_consumption_verified':False,
        'independent_review_complete':False,'native_launch_authorized':False,
        'unresolved':['Root verify actual payload/interpreter/system binary bytes and complete missingexpectedpins','exact newworker/preflight/auditor consumption of identical reviewed union and immutable rosters','finite actual dependency source surface review including bootstrap external sitepackages and library search/load layout','stdlib779 same version/logical-physical alias/launcher/optionalZIP identity currently requalified byRoot before nativeuse']}
    value['sha256']=digest(value);return value
if __name__=='__main__':
    target=HERE/'RUNTIME_CANDIDATE_001.json';require(not target.exists(),'write-once ignored candidate')
    value=build_candidate()
    with target.open('x',encoding='utf-8') as stream:stream.write(canonical(value)+'\n')
    print(json.dumps({'torch_source_files':len(value['torch_source_hashes']),'torch_unread_payloads':len(value['torch_UNREAD_payload_obligations']),
        'torch_source_bytes':value['source_bytes'],'inherited':len(value['inherited65_map']),'stdlib':len(value['stdlib779_inventory']['runtime_pins']),
        'body_sha256':value['sha256'],'native_runtime_closure_complete':False}))
