"""Selected-source observer launch, preservation and terminal replay.

The public path validates its original context before native helper imports.
Counted pathname adapters move outputs only; the original lifecycle remains.
CPU suppliers are explicitly inert and never grant native qualification.
"""
from pathlib import Path
from types import ModuleType
import ast,hashlib,importlib.machinery,importlib.util,os,sys,time

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'observer_startup_measurement_v1'
CONTRACT_SHA='eda31c599f9b93402e0d4a0500386a01d38b0701c643a1e8fb1327f388f44965'
ROOT=Path(r'C:\Users\clclc\BusinessProject\rineya-observer-startup-qualification-20261006')
ROOT_RECORD=PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_OUTPUT_ROOT_V1.json'
HELPERS=('native_process','native_nvml','native_supervisor')
PSUTIL_CORE=('psutil','psutil._common','psutil._ntuples','psutil._pswindows','psutil._psutil_windows')

def require(value,message):
    if not value:raise ValueError(message)

def error_record(error):
    cls=type(error);name=type.__getattribute__(cls,'__name__');message=None;exact=False
    try:
        args=BaseException.args.__get__(error,cls)
        if type.__getattribute__(cls,'__str__') is BaseException.__str__:
            if type(args) is tuple and len(args)==1 and type(args[0]) is str:message=args[0][:2048];exact=len(args[0])<=2048
            elif type(args) is tuple and not args:message='';exact=True
    except BaseException:pass
    return {'type':name[:128],'message':message,'message_exact':exact}

def bootstrap_contract():
    require(HERE==EXPECTED,'staged/preserved launch driver has no native authority')
    path=HERE/'contract.py';raw=path.read_bytes()
    require(0<len(raw)<=1024**2 and hashlib.sha256(raw).hexdigest()==CONTRACT_SHA,'exact original launch contract bootstrap')
    m=ModuleType('observer_driver_contract');m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def selected_module(c,value,path):
    path=Path(path);d=value['config']['source_files'].get(str(path))
    require(type(d) is dict and d['role']=='python_source','exact selected launch source member')
    c.file_descriptor(path,d);raw=c.read_bytes(path)
    require(hashlib.sha256(raw).hexdigest()==d['sha256'],'selected compile bytes match original descriptor')
    m=ModuleType('observer_driver_'+path.stem);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def selected_output_root(c,value):
    d=value['config']['source_files'].get(str(ROOT_RECORD))
    require(type(d) is dict and d['role']=='source_json','externally selected output-root source JSON')
    c.file_descriptor(ROOT_RECORD,d);record=c.parse(c.read_bytes(ROOT_RECORD))
    require(type(record) is dict and set(record)=={'schema_version','kind','scope','root','native_PASS','native_authority_issued','sha256'} and record['schema_version']==1 and record['kind']=='OBSERVER_STARTUP_EXTERNAL_OUTPUT_ROOT_V1' and record['scope']==c.SCOPE and record['root']==str(ROOT) and record['native_PASS'] is record['native_authority_issued'] is False and record['sha256']==c.digest({k:v for k,v in record.items() if k!='sha256'}),'same externally selected finite output root')
    require(ROOT.is_dir() and c.physical(ROOT)==ROOT,'existing exact external output root')
    return ROOT

def paths(root,run_name):
    root=Path(root).absolute();require(type(run_name) is str and 0<len(run_name)<=96 and run_name[0].isalnum() and all(ch.isalnum() or ch in '-_' for ch in run_name),'finite fresh run name')
    raw=root/run_name;preserved=root/(run_name+'_preserved');receipt=root/(run_name+'_driver.json')
    require(root.resolve()==root and root.is_dir() and not raw.exists() and not preserved.exists() and not receipt.exists(),'fresh exact external output identities')
    return raw,preserved,receipt

def _function(module,name):
    raw=Path(module.__file__).read_bytes();tree=ast.parse(raw,str(module.__file__))
    return raw,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)

def _external_function(raw,name,namespace,root):
    tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);count=0
    wanted=ast.dump(ast.parse('PACKAGE / "artifacts"'+(' / "_native_gpu_leases"' if name=='_acquire_lease' else ''),mode='eval').body,include_attributes=False)
    class RootPath(ast.NodeTransformer):
        def visit_Assign(self,n):
            nonlocal count
            if ast.dump(n.value,include_attributes=False)==wanted:
                count+=1;n.value=ast.parse('_driver_root'+(' / "_native_gpu_leases"' if name=='_acquire_lease' else ''),mode='eval').body
            return self.generic_visit(n)
    node=RootPath().visit(node);require(count==1,'one exact original '+name+' output pathname adaptation')
    ns={**namespace,'_driver_root':root};ast.fix_missing_locations(node);exec(compile(ast.Module(body=[node],type_ignores=[]),namespace['__file__'],'exec'),ns)
    return ns[name],count

def supervisor_adapter(s,root):
    """Capture selected S; install two path-only helpers in its original namespace."""
    _,node=_function(s,'supervise');count=0
    class Install(ast.NodeTransformer):
        def visit_Assign(self,n):
            nonlocal count
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='namespace' and ast.dump(n.value,include_attributes=False)==ast.dump(ast.parse('dict(original.__dict__)',mode='eval').body,include_attributes=False):
                count+=1;return [n,*ast.parse('namespace.update(_driver_path_helpers(raw,original.__dict__,_driver_root))').body]
            return n
    node=Install().visit(node);require(count==1,'one selected supervisor namespace pathname adapter insertion')
    proof={'supervisor_namespace_insertions':count,'new_output_path_changes':1,'lease_path_changes':1,'original_lifecycle_changes':3,'mutex_identity_changed':False}
    def helper_factory(raw,namespace,external):
        require(hashlib.sha256(raw).hexdigest()==s.ORIGINAL_SHA,'exact original supervisor before pathname adaptations')
        new,_=_external_function(raw,'_new_output',namespace,external);lease,_=_external_function(raw,'_acquire_lease',namespace,external)
        return {'_new_output':new,'_acquire_lease':lease}
    ns={**s.__dict__,'_driver_path_helpers':helper_factory,'_driver_root':Path(root)};ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node],type_ignores=[]),s.__file__,'exec'),ns);return ns['supervise'],proof

def preservation_adapter(p,root):
    _,node=_function(p,'preserve');count=0;birth_count=0
    wanted={ast.dump(ast.parse("contract.PACKAGE/'artifacts'",mode='eval').body,include_attributes=False):'_driver_root',ast.dump(ast.parse("contract.PACKAGE/'artifacts/_native_gpu_leases'",mode='eval').body,include_attributes=False):"_driver_root/'_native_gpu_leases'"}
    class RootPath(ast.NodeTransformer):
        def visit_Subscript(self,n):
            nonlocal birth_count
            if ast.dump(n,include_attributes=False)==ast.dump(ast.parse("sup['owned_process_identities']",mode='eval').body,include_attributes=False):
                birth_count+=1;return ast.copy_location(ast.parse('_driver_observed_births(contract,sup,output)',mode='eval').body,n)
            return self.generic_visit(n)
        def visit_BinOp(self,n):
            nonlocal count
            replacement=wanted.get(ast.dump(n,include_attributes=False))
            if replacement is not None:count+=1;return ast.copy_location(ast.parse(replacement,mode='eval').body,n)
            return self.generic_visit(n)
    node=RootPath().visit(node);require(count==2 and birth_count==1,'exact preservation paths and actual producer birth-roster read adaptation')
    ns={**p.__dict__,'_driver_root':Path(root),'_driver_observed_births':observed_births};ast.fix_missing_locations(node);exec(compile(ast.Module(body=[node],type_ignores=[]),p.__file__,'exec'),ns)
    return ns['preserve'],{'preservation_path_changes':count,'preservation_birth_projection_changes':birth_count}

def observed_births(c,sup,output):
    """Strict actual launcher/sample projection; never add fields to SUP bytes."""
    rows=[];worker=sup.get('worker_identity')
    if worker is not None:rows.append(worker)
    path=Path(output)/'RESOURCE_OBSERVATIONS.json'
    if path.exists():
        samples=c.parse(c.read_bytes(path));require(type(samples) is list and len(samples)<=32768,'bounded actual resource samples for owned birth preservation')
        for sample in samples:
            require(type(sample) is dict and sample.get('kind') in ('CPU','GPU'),'actual typed resource sample kind')
            if sample['kind']=='CPU':rows.extend(sample['owned_process_identities'])
    result={};by_pid={}
    for row in rows:
        birth={k:row[k] for k in ('pid','creation_filetime_100ns')}
        require(all(type(v) is int and 0<v<2**64 for v in birth.values()),'strict observed PID/FILETIME preservation pair')
        require(birth['pid'] not in by_pid or by_pid[birth['pid']]==birth['creation_filetime_100ns'],'conflicting observed birth for the same owned PID')
        by_pid[birth['pid']]=birth['creation_filetime_100ns']
        result[(birth['pid'],birth['creation_filetime_100ns'])]=birth
    require(len(result)<=4096,'bounded unique original owned birth population');return list(result.values())

class SelectedNamespaces:
    """Own only selected helper/psutil/ctypes module families, with no fallback."""
    def __init__(self,c,value):
        self.c=c;self.value=value;self.owned={};self.table={};self.active=False
        sources=value['config']['source_files'];files=value['runtime']['files'];self.descriptors={**sources,**files}
        for name in HELPERS:
            path=str(c.PACKAGE/(name+'.py'));require(path in sources and sources[path]['role']=='python_source','selected original package helper origin');self.table[name]=(path,False)
        policy=value['runtime']['cache_policy']['allowed_bootstrap_modules']
        roots={'ctypes':str(Path(policy['ctypes']['path']).parent)}
        candidates=[p for p in files if Path(p).name=='__init__.py' and Path(p).parent.name=='psutil'];require(len(candidates)==1,'one selected psutil package root');roots['psutil']=str(Path(candidates[0]).parent)
        self.table['_ctypes']=(policy['_ctypes']['path'],False)
        for family,root in roots.items():
            # Psutil is a finite Windows-only selected runtime closure. Its
            # package source cannot introduce another member via sys.path.
            population=files if family=='psutil' else self.descriptors
            for path,d in population.items():
                p=Path(path)
                if not p.is_relative_to(root) or p.suffix.lower() not in ('.py','.pyd'):continue
                rel=p.relative_to(root);parts=list(rel.parts);package=p.name=='__init__.py';parts=parts[:-1] if package else [*parts[:-1],parts[-1].split('.')[0]]
                name='.'.join([family,*parts])
                if family=='psutil' and name not in PSUTIL_CORE:continue
                require(d['role']==('extension_binary' if p.suffix.lower()=='.pyd' else 'python_source'),'exact selected helper source/extension role')
                require(name not in self.table,'unique selected helper module name');self.table[name]=(path,package)
        require(set(PSUTIL_CORE)<=set(self.table),'complete finite selected psutil Windows core before helper import')
        self.families=(*HELPERS,'psutil','ctypes','_ctypes')
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0] not in self.families:return None
        require(fullname in self.table,'unselected helper namespace cannot fall back to generic import paths')
        selected,package=self.table[fullname];d=self.descriptors[selected];layout=self.value['runtime'].get('stdlib_layout')
        self.c.file_descriptor(selected,d,layout=layout);physical=str(self.c.qualified_path(selected,layout))
        if Path(selected).suffix.lower()=='.pyd':
            loader=importlib.machinery.ExtensionFileLoader(fullname,physical);return importlib.util.spec_from_file_location(fullname,physical,loader=loader)
        owner=self
        class Source:
            def create_module(self,spec):return None
            def exec_module(self,module):
                raw=owner.c.read_bytes(physical);require(hashlib.sha256(raw).hexdigest()==d['sha256'],'exact selected helper executable source bytes')
                module.__file__=physical
                if package:module.__path__=[str(Path(physical).parent)]
                owner.owned[fullname]=module;exec(compile(raw,physical,'exec'),module.__dict__)
        return importlib.util.spec_from_loader(fullname,Source(),origin=physical,is_package=package)
    def begin(self):
        for name,module in tuple(sys.modules.items()):
            if name.split('.',1)[0] not in self.families:continue
            require(name not in HELPERS and name in self.table,'fresh original helper namespace; no foreign preloaded helper')
            expected=self.c.qualified_path(self.table[name][0],self.value['runtime'].get('stdlib_layout'))
            require(getattr(module,'__file__',None)==str(expected),'exact admitted preloaded ctypes/psutil origin assumption')
        sys.meta_path.insert(0,self);self.active=True
    def load(self,name):
        spec=self.find_spec(name);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;self.owned[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(name) is module:del sys.modules[name]
            raise
        require(Path(module.__file__)==self.c.qualified_path(self.table[name][0],self.value['runtime'].get('stdlib_layout')),'actual selected helper module origin after compile')
        return module
    def close(self):
        if self in sys.meta_path:sys.meta_path.remove(self)
        self.active=False
        # Preserve object/code lifetimes; never unload any native extension/DLL.

def write_json(c,path,value):
    # Detach every nested container: later error accumulation must not change
    # the already sealed snapshot returned to its parent driver receipt.
    value=c.seal(c.parse(c.canonical(value)));raw=(c.canonical(value)+'\n').encode('ascii');require(len(raw)<=c.MAX_JSON,'bounded complete driver receipt')
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    require(c.read_bytes(path)==raw,'exact durable driver receipt');return value

def failure_snapshot(c,value,output,target,death,errors):
    """Separate incomplete evidence, never a relaxed original preservation PASS."""
    target.mkdir();files={};total=0;originals={};lease_files={};snapshots={};denial=None
    def copy(path,raw):
        nonlocal total
        total+=len(raw);require(total<=64*1024**2,'bounded complete partial evidence bytes')
        with Path(path).open('xb') as stream:
            require(stream.write(raw)==len(raw),'complete exclusive failure evidence copy');stream.flush();os.fsync(stream.fileno())
        require(c.read_bytes(path)==raw,'exact durable failure evidence copy')
        return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if output.exists():
        names=sorted(p.name for p in output.iterdir());require(len(names)<=256,'bounded partial output roster')
        for name in names:
            raw=c.read_bytes(output/name);files[name]=copy(target/name,raw);originals[output/name]=raw
    owned=[];sup=output/'SUPERVISOR_RECEIPT.json'
    if sup.exists():
        sup_raw=c.read_bytes(sup);record=c.parse(sup_raw)
        for birth in observed_births(c,record,output):owned.append({'identity':birth,'currently_alive':death(dict(birth))})
        if record.get('status')=='ADMISSION_DENIED' and all(record.get(k) is False for k in ('worker_attempted','worker_launched','worker_resumed')):
            require(not owned and 'worker_identity' not in record and 'launch_failure' not in record,'denial preservation cannot invent or suppress an owned worker')
            denial={k:record[k] for k in ('status','worker_attempted','worker_launched','worker_resumed')}
            if 'lease_path' in record:
                lease=Path(record['lease_path']).absolute();lease_root=output.parent/'_native_gpu_leases'
                require(lease.parent==lease_root and c.physical(lease)==lease,'exact original denied lease under selected external output root')
                lease_names=sorted(p.name for p in lease.iterdir())
                require(lease_names==['reservation.json','terminal.json'],'exact actual preworker denial lease population')
                require(c.read_bytes(lease/'terminal.json')==sup_raw,'denied lease terminal byte-identical original SUP')
                for name in lease_names:
                    original=lease/name;raw=c.read_bytes(original);leaf='lease_'+name
                    lease_files[name]={'source_path':str(original),'path':leaf,**copy(target/leaf,raw)};originals[original]=raw
            sources=value['config']['source_files'];require(len(sources)<=4096,'bounded selected denial source population')
            (target/'source_snapshots').mkdir()
            for source,descriptor in sources.items():
                c.file_descriptor(source,descriptor);raw=c.read_bytes(source)
                name=hashlib.sha256(source.encode('utf-8')).hexdigest()+Path(source).suffix
                snapshots[source]={'path':'source_snapshots/'+name,'role':descriptor['role'],**copy(target/'source_snapshots'/name,raw)};originals[Path(source)]=raw
            for original,raw in originals.items():require(c.read_bytes(original)==raw,'final original denial evidence bytes unchanged')
            require(sorted(p.name for p in output.iterdir())==names,'final original denied raw roster unchanged')
            if lease_files:require(sorted(p.name for p in lease.iterdir())==lease_names,'final original denied lease roster unchanged')
    body={'kind':'OBSERVER_DRIVER_FAILURE_SNAPSHOT_V1','scope':c.SCOPE,'source_commit':value['expected_head'],'config_sha256':value['anchors']['expected_config_file'],'raw_files':files,'owned_death_observations':owned,'owned_birth_status':'OBSERVED' if owned else 'UNAVAILABLE','errors':errors,'complete_original_preservation':False,'fixture_only':value['fixture_only'],'native_PASS':False,'native_authority_issued':False}
    if denial is not None:
        body.update(admission_denial=denial,lease_files=lease_files,source_files=value['config']['source_files'],source_snapshot_files=snapshots,external_anchors={'config_path':value['config_path'],'anchors':value['anchors']})
    return write_json(c,target/'DRIVER_FAILURE_SNAPSHOT.json',body)

def _entry_body(context,c,output,target,receipt_path,supervise,preserve,audit,death,revalidate,proof):
    value=c.context_value(context);require(revalidate()==value,'same original driver context before lifecycle')
    errors=[];primary=None;secondary=None;supervisor=None;preservation=None;terminal=None;incomplete=None
    try:supervisor=supervise()
    except BaseException as error:primary=error;errors.append({'stage':'supervisor','error':error_record(error)})
    try:
        preservation=preserve(output,target,contract=c,context=context,revalidate=revalidate,fresh_death=death)
    except BaseException as error:
        if primary is None:primary=error
        else:secondary=error
        errors.append({'stage':'preservation','error':error_record(error)})
        if not target.exists():
            try:incomplete=failure_snapshot(c,value,output,target,death,errors)
            except BaseException as failure:errors.append({'stage':'failure_snapshot','error':error_record(failure)})
    try:
        terminal=audit(target,preservation)
    except BaseException as error:
        if primary is None:primary=error
        elif secondary is None:secondary=error
        errors.append({'stage':'terminal_audit','error':error_record(error)})
    try:require(revalidate()==value,'same original driver context after terminal replay')
    except BaseException as error:
        if primary is None:primary=error
        elif secondary is None:secondary=error
        errors.append({'stage':'final_context','error':error_record(error)})
    result={'kind':'OBSERVER_LAUNCH_DRIVER_RESULT_V1','scope':c.SCOPE,'status':'OBSERVER_DRIVER_FAILED' if errors else 'OBSERVER_DRIVER_RETURNED_PENDING_CURRENT_QUALIFICATION','source_commit':value['expected_head'],'config_sha256':value['anchors']['expected_config_file'],'output_dir':str(output),'preserved_dir':str(target),'source_adaptations':proof,'supervisor':supervisor,'preservation':preservation,'terminal_audit':terminal,'incomplete_preservation':incomplete,'errors':errors,'fixture_only':value['fixture_only'],'native_PASS':False,'native_authority_issued':False}
    try:result=write_json(c,receipt_path,result)
    except BaseException as error:
        if primary is None:raise
        raise primary from error
    if primary is not None:
        if secondary is not None:raise primary from secondary
        raise primary
    return result

def run_registered(config_path,external_path,run_name):
    c=bootstrap_contract();c.native_gate();external=c.parse(c.read_bytes(external_path))
    require(type(external) is dict and set(external)=={'config_path','anchors'} and external['config_path']==str(Path(config_path).absolute()),'exact external launch anchors')
    context=c.validate_context(config_path,**external['anchors']);value=c.context_value(context)
    require(value['fixture_only'] is False,'CPU context cannot enter native launch driver')
    root=selected_output_root(c,value);output,target,receipt=paths(root,run_name)
    c.file_descriptor(__file__,value['config']['source_files'][str(Path(__file__).absolute())])
    for name in ('TEMP','TMP','PYTHONPYCACHEPREFIX'):
        directory=root/name.lower();directory.mkdir(exist_ok=True);os.environ[name]=str(directory)
    sys.dont_write_bytecode=True
    namespaces=None
    try:
        s=selected_module(c,value,HERE/'supervisor.py');p=selected_module(c,value,HERE/'preserve.py');a=selected_module(c,value,HERE/'audit.py')
        supervise,proof=supervisor_adapter(s,root);preserve,p_proof=preservation_adapter(p,root);proof.update(p_proof)
        state={'sampler':None}
        def death(birth):
            require(state['sampler'] is not None,'original owned-death API unavailable; never infer death')
            return state['sampler'].is_alive(birth)
        def lifecycle():
            nonlocal namespaces
            namespaces=SelectedNamespaces(c,value);namespaces.begin()
            process=namespaces.load('native_process');namespaces.load('native_nvml');namespaces.load('native_supervisor');namespaces.load('psutil')
            state['sampler']=process.Sampler()
            return supervise(config_path,output,**external['anchors'])
        def audit(target,preservation):
            require(preservation is not None,'incomplete original preservation cannot satisfy terminal audit')
            raw=c.read_bytes(target/'PRESERVATION.json')
            return a.audit_registered(target,original_directory=output,context=context,contract=c,expected_preservation_file_sha256=hashlib.sha256(raw).hexdigest(),expected_preservation_body_sha256=preservation['sha256'],fresh_death=death)
        return _entry_body(context,c,output,target,receipt,lifecycle,preserve,audit,death,lambda:c.revalidate_context(context),proof)
    finally:
        if namespaces is not None:namespaces.close()

def _CPU_run_fixture(context,c,output,target,receipt,*,_capability,supervise,preservation_module,audit_module,death):
    c._check_CPU_capability(_capability);value=c.context_value(context)
    require(value['fixture_only'] is True and value['native_PASS'] is value['native_authority_issued'] is False,'explicit inert driver CPU context only')
    preserve,proof=preservation_adapter(preservation_module,Path(output).parent)
    def audit(path,original):
        require(original is not None,'incomplete original preservation cannot satisfy CPU terminal audit')
        raw=c.read_bytes(path/'PRESERVATION.json')
        return audit_module._CPU_audit_fixture(path,original_directory=output,context=context,contract=c,_capability=_capability,expected_preservation_file_sha256=hashlib.sha256(raw).hexdigest(),expected_preservation_body_sha256=original['sha256'],fresh_death=death)
    return _entry_body(context,c,Path(output),Path(target),Path(receipt),supervise,preserve,audit,death,lambda:c._CPU_revalidate_context(context,_capability=_capability),proof)

def main():
    # The staging gate precedes parser/context/helper/native imports.
    require(HERE==EXPECTED,'staged/preserved launch driver has no native authority')
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--anchors',type=Path,required=True);parser.add_argument('--run-name',required=True);args=parser.parse_args()
    run_registered(args.config,args.anchors,args.run_name);return 0

if __name__=='__main__':raise SystemExit(main())
