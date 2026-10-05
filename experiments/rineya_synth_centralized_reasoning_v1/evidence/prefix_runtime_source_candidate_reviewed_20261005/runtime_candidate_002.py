"""Source-only finite Torch metadata/source candidate. Native closure is closed."""
from pathlib import Path
import ast, base64, csv, hashlib, io, json, math, os, stat
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
INHERITED=PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json'
STDLIB=PACKAGE/'evidence/stdlib_inventory_reviewed_cpu_20261005/STDLIB_CANDIDATE_001.json'
INHERITED_FILE='53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030'
INHERITED_BODY='944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62'
STDLIB_FILE='e5e185f980e2f0feae7bf49384089eedd4d47f5b41c10a828c6ef2a30ae777a5'
STDLIB_BODY='f6284b28e689b2c5e42a97c12256b3e35a41395a6b9612c0a01ee786eb28b149'
MAX_FILES=32768;MAX_SOURCE_FILE=16*1024**2;MAX_SOURCE_TOTAL=256*1024**2;MAX_JSON=16*1024**2
SOURCE_SUFFIXES={'.py','.pyi'};EXECUTABLE_SUFFIXES=SOURCE_SUFFIXES|{'.pyd','.dll'}
BOOTSTRAP=('__init__.py','_tensor.py','cuda/__init__.py','cuda/memory.py','cuda/random.py','random.py','backends/__init__.py','backends/cuda/__init__.py','backends/cudnn/__init__.py')
def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def duplicate_free(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'duplicate JSON key');result[key]=value
    return result
def finite_float(value):
    result=float(value);require(math.isfinite(result),'nonfinite/overflow JSON number');return result
def reject_constant(value):raise ValueError('nonfinite JSON constant')
def pointer(path,root):
    path=Path(path).absolute();root=Path(root).absolute()
    require(path.is_relative_to(root) and path.resolve()==path,'exact physical pointer')
    for entry in (path,*path.parents):
        metadata=entry.lstat();require(not entry.is_symlink() and not(getattr(metadata,'st_file_attributes',0)&0x400),'no reparse alias')
    return path
def fingerprint(metadata):return(metadata.st_size,metadata.st_mtime_ns,metadata.st_ino)
def bounded_bytes(path,maximum,expected_size=None):
    path=Path(path).absolute();pointer(path,path.parent);before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=maximum,'bounded regular file')
    if expected_size is not None:require(type(expected_size) is int and before.st_size==expected_size,'expected file size')
    with path.open('rb') as stream:raw=stream.read(before.st_size+1)
    pointer(path,path.parent);require(len(raw)==before.st_size and fingerprint(path.lstat())==fingerprint(before),'stable bounded file read')
    return raw
def strict_json(raw):return json.loads(raw,object_pairs_hook=duplicate_free,parse_float=finite_float,parse_constant=reject_constant)
def json_source(path,expected_file,expected_body):
    require(Path(path).suffix=='.json','JSON metadata only');raw=bounded_bytes(path,MAX_JSON)
    require(hashlib.sha256(raw).hexdigest()==expected_file,'external original file anchor');value=strict_json(raw)
    require(type(value) is dict and value.get('sha256')==expected_body,'external original body anchor')
    require(digest({k:v for k,v in value.items() if k!='sha256'})==expected_body,'original body checksum')
    return value,{'path':str(Path(path).absolute()),'bytes':len(raw),'file_sha256':expected_file,'body_sha256':expected_body,'external_anchor_verified':True}
def pin_map(value):
    require(type(value) is dict and len(value)<=MAX_FILES,'finite exact pin map')
    for path,pin in value.items():
        require(type(path) is str and 0<len(path)<=4096,'bounded pin path')
        require(type(pin) is str and len(pin)==64 and all(c in '0123456789abcdef' for c in pin),'exact SHA256 pin')
    return value
def validate_stdlib(value):
    require(value['native_authority_issued'] is False,'stdlib receipt issues no authority');inv=value['inventory'];pins=pin_map(inv['runtime_pins'])
    require(len(pins)==779 and set(pins)==set(inv['files'])==set(inv['paths']),'exact reviewed stdlib roster joins')
    require(inv['policy']=='finite-hermes-python-stdlib-byte-superset-known-junction-2' and inv['python_version']=='3.11.15' and inv['fixture'] is False,'reviewed stdlib layout identity')
    for path,pin in pins.items():
        entry=inv['files'][path];require(entry['path']==path and entry['sha256']==pin and type(entry['bytes']) is int and entry['bytes']>=0,'stdlib exact descriptor pin join')
    return inv
def metadata_roster(root):
    root=Path(root).absolute();pointer(root,root);values={}
    for current,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=sorted(name for name in dirs if name!='__pycache__')
        for name in dirs:pointer(Path(current)/name,root)
        for name in sorted(files):
            path=Path(current)/name
            if path.suffix not in EXECUTABLE_SUFFIXES:continue
            path=pointer(path,root);meta=path.lstat();require(stat.S_ISREG(meta.st_mode) and meta.st_size>=0,'regular executable roster metadata');require(len(values)<MAX_FILES,'finite executable file ceiling')
            role='torch_python_source' if path.suffix=='.py' else 'torch_python_stub_source' if path.suffix=='.pyi' else 'torch_binary_UNREAD_METADATA_ONLY'
            values[str(path)]={'bytes':meta.st_size,'role':role}
    return dict(sorted(values.items()))
def hash_source(path,expected_size):
    require(Path(path).suffix in SOURCE_SUFFIXES,'Python source only; no binary bytes');raw=bounded_bytes(path,MAX_SOURCE_FILE,expected_size);raw.decode('utf-8');return hashlib.sha256(raw).hexdigest()
def merge_exact(*maps):
    result={}
    for mapping in maps:
        for path,pin in pin_map(mapping).items():
            require(path not in result or result[path]==pin,'conflicting overlap expected pin');result[path]=pin
    return dict(sorted(result.items()))
def installation_metadata(torch):
    matches=sorted(torch.parent.glob('torch-*.dist-info'));require(len(matches)==1,'exact installed Torch dist-info metadata roster');root=matches[0];descriptors={};texts={}
    for name in ('RECORD','METADATA','WHEEL'):
        path=pointer(root/name,root);raw=bounded_bytes(path,MAX_SOURCE_FILE);texts[name]=raw.decode('utf-8')
        descriptors[name]={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'installation_text_metadata_not_binary_verification'}
    record={}
    for row in csv.reader(io.StringIO(texts['RECORD'])):
        require(len(row)==3 and len(record)<MAX_FILES,'finite three-field RECORD metadata');name,encoded,size=row
        if not name.startswith('torch/') or Path(name).suffix not in EXECUTABLE_SUFFIXES or '__pycache__' in Path(name).parts:continue
        require('..' not in Path(name).parts and str(torch.parent/Path(name)) not in record,'unique contained RECORD path');expected=None
        if encoded:
            require(encoded.startswith('sha256='),'RECORD expected SHA256 convention');value=encoded[7:];decoded=base64.urlsafe_b64decode(value+'='*((-len(value))%4));require(len(decoded)==32,'RECORD SHA256 length');expected=decoded.hex()
        require(size.isascii() and size.isdigit(),'RECORD expected byte size')
        record[str(torch.parent/Path(name))]={'expected_RECORD_sha256':expected,'expected_RECORD_bytes':int(size),'current_binary_bytes_verified':False}
    dependencies=[line[15:] for line in texts['METADATA'].splitlines() if line.startswith('Requires-Dist: ')];require(len(dependencies)<=128 and all(len(x)<=4096 for x in dependencies),'finite dependency declarations')
    return descriptors,record,dependencies
def build_candidate():
    inherited,inherit_ref=json_source(INHERITED,INHERITED_FILE,INHERITED_BODY);stdlib,stdlib_ref=json_source(STDLIB,STDLIB_FILE,STDLIB_BODY)
    old=pin_map(inherited['runtime_pins']);require(len(old)==65,'anchored inherited65 exact map');inv=validate_stdlib(stdlib)
    roots=[Path(p).parent for p in old if p.endswith('torch\\version.py')];require(len(roots)==1,'unique inherited Torch source root');torch=roots[0]
    before=metadata_roster(torch);sources={};total=0
    for path,entry in before.items():
        if Path(path).suffix in SOURCE_SUFFIXES:
            total+=entry['bytes'];require(total<=MAX_SOURCE_TOTAL,'finite source total ceiling');sources[path]=hash_source(path,entry['bytes'])
    descriptors,record,dependencies=installation_metadata(torch);expected=merge_exact(old,inv['runtime_pins'],sources)
    obligations={path:{**entry,'expected_inherited_sha256':old.get(path),'installation_record':record.get(path),'current_binary_bytes_hashed':False,'Root_qualified_byte_verification_required':True} for path,entry in before.items() if path not in sources}
    bootstrap={}
    for relative in BOOTSTRAP:
        path=torch/Path(relative);tree=ast.parse(bounded_bytes(path,MAX_SOURCE_FILE,before[str(path)]['bytes']).decode('utf-8'))
        roots=sorted({n.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for n in node.names}|{node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.level==0 and node.module})
        bootstrap[relative]={'path':str(path),'source_sha256':sources[str(path)],'literal_absolute_import_roots_including_conditional':roots}
    require(before==metadata_roster(torch),'exact executable metadata roster stable');require({p:hash_source(p,before[p]['bytes']) for p in sources}==sources,'all source hashes stable before/after')
    require((descriptors,record,dependencies)==installation_metadata(torch),'installation text metadata stable')
    value={'schema_version':2,'kind':'SOURCE_ONLY_FINITE_TORCH_RUNTIME_CANDIDATE_INCOMPLETE_NATIVE_CLOSURE',
        'inherited65_anchor':inherit_ref,'inherited65_map':old,'inherited65_map_sha256':digest(old),'stdlib779_anchor':stdlib_ref,'stdlib779_inventory':inv,'stdlib779_map_sha256':digest(inv['runtime_pins']),
        'torch_physical_root':str(torch),'torch_file_metadata':before,'torch_source_hashes':sources,'torch_UNREAD_binary_obligations':obligations,'torch_installation_metadata':descriptors,
        'torch_installation_declared_dependencies_UNQUALIFIED':dependencies,'bootstrap_source_import_observations':bootstrap,
        'stdlib779_current_revalidation_obligations':{p:{'expected_reviewed_descriptor':inv['files'][p],'role':inv['paths'][p],'current_byte_verification_not_performed':True} for p in inv['runtime_pins']},
        'expected_known_pin_union':expected,'expected_known_pin_union_sha256':digest(expected),
        'finite_roster_policy':{'suffixes':sorted(EXECUTABLE_SUFFIXES),'excluded_directory_names':['__pycache__'],'pyc_in_payload_roster':False,'other_Torch_package_data_not_in_executable_roster':True},
        'roles_are_superset_not_actual_execution_proof':True,'source_before_after_equal':True,'metadata_before_after_equal':True,'source_bytes':total,
        'Torch_imported':False,'binary_model_payload_bytes_read':False,'OS_GPU_resource_probes_performed':False,'native_runtime_closure_complete':False,'Root_binary_qualification_complete':False,
        'cache_execution_policy_qualified':False,'external_dependencies_qualified':False,'worker_consumption_verified':False,'independent_review_complete':False,'native_launch_authorized':False,
        'unresolved':['Root actual binary/interpreter/system byte qualification and missing expected pins','existing pyc may execute despite roster exclusion or -B; worker/source-loader cache identity policy unqualified','external bootstrap/declared dependencies and any used nonexecutable data require exact separately registered closure','identical immutable map/roster consumption by new preflight/worker/terminal auditor','current stdlib779 approved alias/interpreter/launcher/optional ZIP identity requalification']}
    value['sha256']=digest(value);return value
if __name__=='__main__':
    target=HERE/'RUNTIME_CANDIDATE_002.json';require(not target.exists(),'write-once corrected ignored candidate');value=build_candidate()
    with target.open('x',encoding='utf-8') as stream:stream.write(canonical(value)+'\n')
    print(json.dumps({'torch_source_files':len(value['torch_source_hashes']),'torch_unread_binaries':len(value['torch_UNREAD_binary_obligations']),'torch_source_bytes':value['source_bytes'],'inherited':len(value['inherited65_map']),'stdlib':len(value['stdlib779_inventory']['runtime_pins']),'body_sha256':value['sha256'],'native_runtime_closure_complete':False}))
