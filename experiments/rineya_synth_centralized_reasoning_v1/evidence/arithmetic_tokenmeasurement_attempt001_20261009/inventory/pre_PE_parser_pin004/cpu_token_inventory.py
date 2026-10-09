"""Current passive CPU-tokenizer byte/roster inventory; no package/native loading."""
from pathlib import Path
from types import ModuleType
import ast, hashlib, json, os, stat, struct, sys, time

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
SITE=Path('C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages')
PYTHON=Path('C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe')
BASE=Path('C:/Users/clclc/AppData/Roaming/uv/python/cpython-3.11.15-windows-x86_64-none')
LOGICAL=Path('C:/Users/clclc/AppData/Roaming/uv/python/cpython-3.11-windows-x86_64-none')
SYSTEM=Path('C:/Windows/System32')
READER=PACKAGE/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py'
READER_PIN='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
PE_SOURCE=PACKAGE/'artifacts/audit_reviewer_work/passive_pe_startup_candidates.py'
VERSIONS={'tokenizers':'0.22.2','jinja2':'3.1.6','markupsafe':'3.0.3'}
PACKAGES=tuple(VERSIONS)
SUFFIXES={'.py','.pyi','.pyc','.pyd','.dll'}
MAX_TOTAL=512*1024**2

def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def seal(value):return {**value,'sha256':digest(value)}
def descriptor(path):
    path=Path(path);raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def reader():
    raw=READER.read_bytes();require(hashlib.sha256(raw).hexdigest()==READER_PIN,'reviewed passive collector pin')
    module=ModuleType('selected_cpu_passive_reader');module.__file__=str(READER);exec(compile(raw,str(READER),'exec'),module.__dict__)
    module.ROLES={**module.ROLES,'dependency_metadata':{'','.typed','.toml','.md','.txt','.rst','.csv'},'tokenizer_asset':{'.json','.txt'},'runtime_configuration':{'.cfg'}}
    module.MAX_TOTAL=MAX_TOTAL;module.MAX_FILE=64*1024**2
    return module

def alias():
    m=LOGICAL.lstat();require(LOGICAL.resolve()==BASE and str(Path(sys.base_prefix))==str(LOGICAL),'exact current Python base alias')
    return {'logical_base_prefix':str(LOGICAL),'physical_base_prefix':str(BASE),'file_attributes':getattr(m,'st_file_attributes',0),'reparse_tag':getattr(m,'st_reparse_tag',None),'is_symlink':LOGICAL.is_symlink()}

def roster(root,kind):
    root=Path(root);require(root.is_dir() and root.resolve()==root and not root.is_symlink(),'physical finite roster root')
    values=[]
    for path in root.rglob('*'):
        if kind=='stdlib' and 'site-packages' in path.relative_to(root).parts:continue
        m=path.lstat();require(not path.is_symlink() and not(getattr(m,'st_file_attributes',0)&0x400),'roster refuses unselected aliases')
        if path.is_file() and (kind=='distribution' or path.suffix.lower() in SUFFIXES or path.name=='py.typed'):
            values.append(str(path))
    return sorted(values)

def role(path):
    suffix=Path(path).suffix.lower()
    if suffix=='.py':return 'python_source'
    if suffix=='.pyi':return 'python_stub'
    if suffix=='.pyc':return 'qualified_bytecode_cache'
    if suffix in ('.pyd','.dll'):return 'windows_system_binary' if Path(path).is_relative_to(SYSTEM) else 'extension_binary'
    if suffix=='.exe':return 'interpreter_binary'
    if suffix=='.cfg':return 'runtime_configuration'
    return 'dependency_metadata'

def pe_parser():
    raw=PE_SOURCE.read_bytes();tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pe_imports')
    namespace={'require':require,'struct':struct};exec(compile(ast.Module(body=[node],type_ignores=[]),str(PE_SOURCE),'exec'),namespace)
    return namespace['pe_imports'],descriptor(PE_SOURCE)

def static_pe_population(seeds,deadline):
    parse,source=pe_parser();queue=list(seeds);files=set(seeds);records=[];edges=[];done=set();read_bytes=0
    local={}
    for text in seeds:local.setdefault(Path(text).name.casefold(),[]).append(text)
    while queue:
        deadline();text=queue.pop(0)
        if text in done:continue
        done.add(text);require(len(done)<=256,'finite CPU PE population')
        raw=Path(text).read_bytes();read_bytes+=len(raw);require(read_bytes<=MAX_TOTAL,'bounded passive PE reads')
        parsed=parse(raw);records.append({'path':text,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'machine':parsed['machine']})
        for row in parsed['imports']:
            name=row['name'];candidates=local.get(name.casefold(),[])
            require(len(candidates)<=1,'unambiguous selected CPU local DLL name')
            resolved=candidates[0] if candidates else None
            if resolved is None and not name.lower().startswith(('api-ms-','ext-ms-')):
                candidate=SYSTEM/name
                if candidate.is_file():resolved=str(candidate.resolve())
            edges.append({'source':text,'kind':row['kind'],'name':name,'resolved_path':resolved,'observed_loader_resolution':False})
            if resolved is not None and resolved not in files:files.add(resolved)
    return sorted(files),{'parser_source':source,'records':records,'edges':edges,'auxiliary_read_bytes':read_bytes,'dynamic_library_effects_observed':False,'transitive_System32_expansion_performed':False,'API_set_and_unresolved_edges':[row for row in edges if row['resolved_path'] is None]}

def verify_inventory(ref):
    require(sys.flags.isolated==1 and sys.flags.no_site==1 and sys.dont_write_bytecode is True,'exact isolated/no-site/no-bytecode CPU interpreter flags')
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'},'exact selected CPU inventory descriptor')
    raw=Path(ref['path']).read_bytes();require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'CPU inventory original file bytes')
    value=json.loads(raw);require(value['kind']=='PASSIVE_ARITHMETIC_CPU_TOKENIZER_INVENTORY_V1' and value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'CPU inventory body seal')
    require(sys.version_info[:3]==(3,11,15) and str(Path(sys.executable))==str(PYTHON) and alias()==value['alias'],'current pinned no-site Python identity')
    for entry in value['rosters']:
        require(roster(entry['root'],entry['kind'])==entry['paths'],'CPU dependency physical roster unchanged before reads')
    declared={path:{k:row[k] for k in ('bytes','sha256','role')} for path,row in value['files'].items()}
    result=reader()._collect(list(declared),declared,alias_table=value['alias'])
    for entry in value['rosters']:require(roster(entry['root'],entry['kind'])==entry['paths'],'CPU dependency physical roster unchanged after reads')
    require(alias()==value['alias'],'CPU Python alias unchanged after reads')
    return {'inventory_file_sha256':ref['sha256'],'inventory_body_sha256':value['sha256'],'files':len(result['files']),'bytes':result['actual_hashed_bytes'],'selected_byte_map_sha256':digest(result['files']),'before_after_rosters_equal':True,'verified':True,'identity_sha256':digest(result['files']),'native_PASS':False}

def build(target):
    started=time.monotonic()
    def deadline():require(time.monotonic()-started<=600,'CPU passive inventory600s')
    require(sys.version_info[:3]==(3,11,15) and str(Path(sys.executable))==str(PYTHON) and sys.flags.isolated==1 and sys.flags.no_site==1 and sys.dont_write_bytecode is True,'pinned isolated/no-site/no-bytecode interpreter')
    forbidden_before=[name for name in sys.modules if name.split('.',1)[0] in {'tokenizers','jinja2','markupsafe','torch','transformers','psutil','ctypes'}];require(not forbidden_before,'passive inventory never loads selected native/packages')
    aliases=alias();rosters=[];paths=set();metadata_versions={}
    for root,kind in ((BASE/'Lib','stdlib'),(BASE/'DLLs','native'),*((SITE/name,'package') for name in PACKAGES),*((SITE/(name+'-'+version+'.dist-info'),'distribution') for name,version in VERSIONS.items())):
        entries=roster(root,kind);rosters.append({'root':str(root),'kind':kind,'paths':entries});paths.update(entries)
    for name,version in VERSIONS.items():
        metadata=SITE/(name+'-'+version+'.dist-info')/'METADATA';raw=metadata.read_bytes();versions=[line[9:] for line in raw.decode().splitlines() if line.startswith('Version: ')];require(versions==[version],'actual dependency metadata exactversion')
        metadata_versions[name]={'version':version,'metadata':descriptor(metadata),'interpretation':'selected installed distribution metadata; package not imported'}
    paths.add(str(PYTHON))
    if (PYTHON.parent/'pythonw.exe').is_file():paths.add(str(PYTHON.parent/'pythonw.exe'))
    for root in (BASE,):
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in ('.exe','.dll'):paths.add(str(path.resolve()))
    venv_cfg=SITE.parents[1]/'pyvenv.cfg';require(venv_cfg.is_file(),'explicit venv configuration');paths.add(str(venv_cfg))
    sources=[PACKAGE/name for name in ('cpu_tokenizer.py','rs_contract.py','rs_verifier.py','arithmetic_guard.py','arithmetic_source_ledger.py','SYNTH_ARITHMETIC_GUARD_POLICY001.json','PILOT_DECISIONS.json','prepare_corpus.py')]
    sources.append(PACKAGE/'artifacts/arithmetic_tokenmeasurement_work/arithmetic_tokenmeasurement.py')
    sources.append(PACKAGE/'artifacts/arithmetic_tokenmeasurement_work/measure_arithmetic_cpu.py')
    sources.extend((READER,Path(__file__).resolve(),PE_SOURCE,
        PACKAGE/'evidence/arithmetic_guard_reviewed_cpu_20261009/current/ARITHMETIC_GUARD_POPULATION001.json',
        PACKAGE/'evidence/arithmetic_source_ledger_reviewed_cpu_20261009/current/ARITHMETIC_SOURCE_LEDGER001.json',
        PACKAGE.parent/'post_training_mimo_centralized_v1/receipts/titan_static_v1/TOKENIZER_RECEIPT.json'))
    for path in sources:require(path.is_file(),'exact arithmetic/tokenizer source metadata');paths.add(str(path.resolve()))
    tree=ast.parse((PACKAGE/'cpu_tokenizer.py').read_bytes());constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for ast_target in node.targets:
                if isinstance(ast_target,ast.Name) and ast_target.id in {'ASSET_PINS','CHAT_TEMPLATE_PIN'}:constants[ast_target.id]=ast.literal_eval(node.value)
    historical=json.loads(sources[-1].read_bytes());asset_root=Path(historical['checkpoint_path']).resolve();require(asset_root.is_dir(),'existing explicit tokenizer assets')
    assets={str(asset_root/name):pin for name,pin in constants['ASSET_PINS'].items()};paths.update(assets)
    require(set(historical['tokenizer_files'])==set(constants['ASSET_PINS']) and all(historical['tokenizer_files'][name]['sha256']==pin for name,pin in constants['ASSET_PINS'].items()),'historical asset pins only')
    template=json.loads((asset_root/'tokenizer_config.json').read_bytes())['chat_template'];require(hashlib.sha256(template.encode()).hexdigest()==constants['CHAT_TEMPLATE_PIN']==historical['chat_template_sha256'],'actual exactchattemplate')
    pe_files,pe=static_pe_population([text for text in sorted(paths) if Path(text).suffix.lower() in ('.exe','.dll','.pyd')]+[str((SYSTEM/name).resolve()) for name in ('kernel32.dll','psapi.dll','ntdll.dll')],deadline);paths.update(pe_files)
    declared={}
    for text in sorted(paths):
        size=Path(text).stat().st_size;selected_role='tokenizer_asset' if text in assets else 'source_json' if Path(text).suffix.lower()=='.json' else role(text)
        declared[text]={'bytes':size,'sha256':assets.get(text),'role':selected_role}
    require(sum(row['bytes'] for row in declared.values())<=MAX_TOTAL,'selectedCPUbytes within512MiB')
    result=reader()._collect(list(declared),declared,alias_table=aliases)
    for entry in rosters:require(roster(entry['root'],entry['kind'])==entry['paths'],'physical CPU rosters unchanged after fullread')
    require(alias()==aliases,'exact alias unchanged after fullread')
    for row in pe['records']:require(result['files'][row['path']]['sha256']==row['sha256'] and result['files'][row['path']]['bytes']==row['bytes'],'passive PE bytes same main readwindow')
    require(not any(name.split('.',1)[0] in {'tokenizers','jinja2','markupsafe','torch','transformers','psutil','ctypes'} for name in sys.modules),'no selected package/native import')
    value=seal({'kind':'PASSIVE_ARITHMETIC_CPU_TOKENIZER_INVENTORY_V1','schema_version':1,'python_version':sys.version,'python_executable':str(PYTHON),'physical_base_prefix':str(BASE),'logical_base_prefix':str(LOGICAL),'selected_site_packages':str(SITE),'asset_root':str(asset_root),'alias':aliases,'dependency_versions':metadata_versions,'chat_template_sha256':constants['CHAT_TEMPLATE_PIN'],'tokenizer_asset_pins':assets,'source_paths':list(map(lambda path:str(path.resolve()),sources)),'rosters':rosters,'PE_static_assumptions':pe,'limits':{'seconds':600,'maximum_selected_bytes':MAX_TOTAL},'files':result['files'],'file_count':len(result['files']),'actual_hashed_bytes':result['actual_hashed_bytes'],'seconds':time.monotonic()-started,'physical_rosters_before_after_equal':True,'selected_packages_imported':False,'native_library_effects_observed':False,'native_PASS':False,'model_or_GPU_run':False,'training_authority_issued':False,'limitations':['Finite passive byte/physical roster inventory, not observed native DLL or executed namespace closure.','Static PE first-level resolution is a prospective exactpath lookup; APIset/unresolved edges remain recorded. SelectedSystem32 transitive graph is not expanded or claimed closed.','Full CPython Lib source/cache/DLLs selected conservatively; no cache/source code-equivalence claim.','No checkpoint config/generation config/weights, model, corpus or protected banks read.']})
    raw=(canonical(value)+'\n').encode('ascii')
    with Path(target).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    require(Path(target).read_bytes()==raw,'durable CPU inventory bytes')
    print(json.dumps({'path':str(Path(target).resolve()),'file_count':value['file_count'],'actual_hashed_bytes':value['actual_hashed_bytes'],'seconds':value['seconds'],'body_sha256':value['sha256'],'file_sha256':hashlib.sha256(raw).hexdigest(),'rosters':len(rosters),'PE_unresolved_edges':len(pe['API_set_and_unresolved_edges']),'selected_packages_imported':False}))
if __name__=='__main__':build(Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'PASSIVE_CPU_TOKENIZER_INVENTORY001.json')
