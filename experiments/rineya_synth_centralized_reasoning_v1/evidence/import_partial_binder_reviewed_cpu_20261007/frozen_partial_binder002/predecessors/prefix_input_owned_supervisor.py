"""Owned lifecycle boundary; injected CPU scopes retain original CPU results."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib.util,math,os,re,stat,sys,time
HERE=Path(__file__).resolve().parent
def load(name):
    path=HERE/name;spec=importlib.util.spec_from_file_location('prefix_supervisor_'+path.stem,path);module=importlib.util.module_from_spec(spec)
    before=path.lstat()
    for part in (path,*path.parents):
        info=part.lstat()
        if part.is_symlink() or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('source reparse alias')
    if not stat.S_ISREG(before.st_mode) or not 0<=before.st_size<=16*1024**2:raise ValueError('bounded source module')
    with path.open('rb') as stream:raw=stream.read(before.st_size+1)
    after=path.lstat()
    if len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('stable module source')
    exec(compile(raw.decode('utf-8'),str(path),'exec'),module.__dict__);return module
W=load('prefix_input_owned_worker.py');C=W.C
def error_value(error):
    try:message=str(error)
    except BaseException:return {'type':type(error).__name__,'message':'<original-message-stringification-failed>','message_unavailable':True}
    if type(message) is not str:return {'type':type(error).__name__,'message':'<nonplain-original-message>','message_unavailable':True}
    if len(message)>4096:return {'type':type(error).__name__,'message':message[:4096],'message_truncated':True}
    return {'type':type(error).__name__,'message':message}
def write_new(path,value):
    raw=(C.canonical(value)+'\n').encode('ascii');C.require(len(raw)<=C.MAX_JSON,'bounded exclusive JSON')
    with Path(path).open('xb') as stream:C.require(stream.write(raw)==len(raw),'complete owned JSON write');stream.flush();os.fsync(stream.fileno())
    C.require(C.read_bytes(path)==raw,'owned JSON readback')
def resource_snapshot(value,now,*,admission):
    C.require(type(value) is dict and set(value)=={'monotonic','available_bytes','worker_RSS_bytes','GPU_monotonic','free_bytes','total_bytes','GPU_identity','disk_free_bytes'},'typed complete resource observation')
    C.require(type(now) in (int,float) and math.isfinite(now) and all(type(value[k]) in (int,float) and math.isfinite(value[k]) and 0<=value[k]<=now for k in ('monotonic','GPU_monotonic')),'finite nonfuture observation clocks')
    C.require(0<=now-value['monotonic']<=.2 and 0<=now-value['GPU_monotonic']<=2,'unchanged resource freshness')
    C.require(type(value['GPU_identity']) is str and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',value['GPU_identity']) is not None,'exact bounded GPU UUID observation')
    C.require(all(type(value[k]) is int and value[k]>=0 for k in ('available_bytes','worker_RSS_bytes','free_bytes','total_bytes','disk_free_bytes')) and value['disk_free_bytes']>=4*1024**3,'integer resource bytes and disk floor')
    C.require(value['worker_RSS_bytes']<=2*1024**3 and value['available_bytes']>=(3.3125 if admission else 1.25)*1024**3 and (4 if admission else .5)*1024**3<=value['free_bytes']<=value['total_bytes'],'unchanged resource floors')
    return value
def supervise_injected(manifest,api,host,output):
    C.validate_fixture(manifest);C.require(Path(output).absolute()==Path(host.output).absolute() and not Path(output).exists(),'new exact injected output');Path(output).mkdir()
    timeline=[];primary=None;report=None;owner=None;observations=[]
    def event(name,identity=None):
        item={'event':name,'monotonic':host.clock()}
        C.require(type(item['monotonic']) in (int,float) and math.isfinite(item['monotonic']) and item['monotonic']>=host.started and item['monotonic']-host.started<=600 and (not timeline or item['monotonic']>=timeline[-1]['monotonic']),'whole-scope finite ordered600s deadline')
        if identity is not None:item['identity']=C.identity(identity)
        timeline.append(item)
    try:
        for name in ('admission_first','admission_second'):
            value=resource_snapshot(host.observation(),host.clock(),admission=True)
            if observations:C.require(value['monotonic']-observations[-1]['monotonic']>=2 and value['GPU_monotonic']>observations[-1]['GPU_monotonic'] and value['GPU_identity']==observations[-1]['GPU_identity'] and host.clock()-host.started<=30,'two distinct current same-device2s/30s admission observations')
            observations.append(value);event(name)
            if name=='admission_first':host.wait_admission()
        owner=C.identity(host.create_suspended());event('worker_created_suspended',owner);C.require(host.assign(owner) is True,'Job assigned before resume');event('Job_assigned',owner)
        observations.append(resource_snapshot(host.observation(),host.clock(),admission=True));event('before_resume_admission',owner)
        admission=C.mint_cpu_admission(manifest,owner);C.require(host.resume(owner) is True,'owned resume');event('worker_resumed',owner)
        report=W.run_injected(manifest,admission,owner,api,host,output);event('worker_returned',owner)
    except BaseException as error:primary=error_value(error)
    finally:
        flags={};secondary=[]
        for operation,event_name,flag in (('worker_dead','worker_dead','worker_dead'),('join_samplers','samplers_joined','samplers_joined'),('close_NVML','NVML_closed','NVML_closed'),('close_Job','Job_closed','job_closed'),('release_reservation','reservation_released','reservation_released'),('close_lease','lease_terminal','lease_terminal')):
            try:
                result=getattr(host,operation)(owner);C.require(result is True,'verified '+flag);flags[flag]=True;event(event_name,owner)
            except BaseException as error:flags[flag]=False;secondary.append({'operation':operation,'error':error_value(error)})
        try:current=host.current_death(owner) if owner is not None else []
        except BaseException as error:current=[];secondary.append({'operation':'current_death','error':error_value(error)})
        try:samples=host.resource_samples();validate_samples(samples)
        except BaseException as error:samples=None;secondary.append({'operation':'resource_samples','error':error_value(error)})
        complete_death=owner is not None and type(current) is list and len(current)==1 and current[0].get('identity')==owner and current[0].get('currently_alive') is False
        if owner is not None and not complete_death:secondary.append({'operation':'current_death','type':'ValueError','message':'complete current exact birth death not established'})
        success=primary is None and not secondary and report is not None and report['status']=='CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED'
        terminal={'kind':'CPU_OWNED_TERMINAL_SIMULATION','scope':C.CPU_SCOPE,'status':'CPU_OWNED_TERMINAL_COMPLETE' if success else 'CPU_OWNED_TERMINAL_FAILED',
          'manifest_sha256':manifest['sha256'],'worker_identity':owner,'owned_process_identities':[] if owner is None else [owner],'timeline':timeline,'terminal_flags':flags,
          'current_death':current,'resource_samples':samples,'admission_observations':observations,'worker_report':report,'worker_report_sha256':C.digest(report),
          'error':primary,'primary_message_exact':primary is None or not(primary.get('message_truncated') or primary.get('message_unavailable')),'secondary_errors':secondary,'fixture_only':True,'native_authority_issued':False}
        terminal['sha256']=C.digest(terminal);write_new(Path(output)/'OWNED_TERMINAL.json',terminal)
    return terminal

def validate_samples(value):
    C.require(type(value) is dict and set(value)=={'CPU','GPU'} and all(type(rows) is list and 0<len(rows)<=16384 for rows in value.values()),'complete bounded CPU/GPU sample ledgers')
    for kind,rows in value.items():
        previous=None
        for row in rows:
            keys={'monotonic','available_bytes','worker_RSS_bytes'} if kind=='CPU' else {'monotonic','free_bytes','total_bytes'}
            C.require(type(row) is dict and set(row)==keys and type(row['monotonic']) in (int,float) and math.isfinite(row['monotonic']) and row['monotonic']>=0,'typed finite resource sample')
            C.require(previous is None or 0<=row['monotonic']-previous<=(.2 if kind=='CPU' else 2),'ordered unchanged sampler freshness cadence')
            C.require(all(type(row[k]) is int and row[k]>=0 for k in keys-{'monotonic'}),'integer sample bytes')
            if kind=='CPU':C.require(row['available_bytes']>=1.25*1024**3 and row['worker_RSS_bytes']<=2*1024**3,'unchanged CPU work floors')
            else:C.require(.5*1024**3<=row['free_bytes']<=row['total_bytes'],'unchanged GPU work floor')
            previous=row['monotonic']

def preserve_injected(output,target,manifest):
    C.validate_fixture(manifest);source=Path(output).absolute();target=Path(target).absolute();C.require(not target.exists() and source!=target,'new write-once preservation target');target.mkdir()
    names=sorted(path.name for path in source.iterdir());C.require(all((source/name).is_file() for name in names),'flat owned raw files');raw_map={}
    for name in names:
        raw=C.read_bytes(source/name);descriptor={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};raw_map[name]=descriptor
        with (target/name).open('xb') as stream:C.require(stream.write(raw)==len(raw),'complete preserved copy');stream.flush();os.fsync(stream.fileno())
        C.require(C.read_bytes(target/name)==raw and C.read_bytes(source/name)==raw,'stable original/copy raw bytes')
    for name,descriptor in raw_map.items():C.require(len(C.read_bytes(target/name))==descriptor['bytes'] and hashlib.sha256(C.read_bytes(target/name)).hexdigest()==descriptor['sha256'],'final all-target conservation')
    C.require(sorted(path.name for path in source.iterdir())==names and sorted(path.name for path in target.iterdir())==names,'final original/copied flat roster equality')
    for name,descriptor in raw_map.items():C.require(len(C.read_bytes(source/name))==descriptor['bytes'] and hashlib.sha256(C.read_bytes(source/name)).hexdigest()==descriptor['sha256'],'final all-original conservation')
    for path,descriptor in manifest['source_pins'].items():C.file_descriptor(path,descriptor)
    snapshots={};snapshot_root=target/'source_snapshots';snapshot_root.mkdir()
    for path,descriptor in manifest['source_pins'].items():
        C.file_descriptor(path,descriptor);raw=C.read_bytes(path);name=hashlib.sha256(path.encode('utf-8')).hexdigest()+'.py'
        with (snapshot_root/name).open('xb') as stream:C.require(stream.write(raw)==len(raw),'complete source snapshot');stream.flush();os.fsync(stream.fileno())
        snapshots[path]={'path':'source_snapshots/'+name,**descriptor};C.require(C.read_bytes(snapshot_root/name)==raw,'source snapshot readback')
    C.require(set(p.name for p in snapshot_root.iterdir())=={Path(v['path']).name for v in snapshots.values()},'exact source snapshot roster')
    C.require(sorted(p.name for p in source.iterdir())==names,'final source raw roster unchanged')
    for name,descriptor in raw_map.items():C.require(C.read_bytes(source/name)==C.read_bytes(target/name) and hashlib.sha256(C.read_bytes(source/name)).hexdigest()==descriptor['sha256'],'complete final raw pair conservation')
    for path,descriptor in manifest['source_pins'].items():C.file_descriptor(path,descriptor);C.require(C.read_bytes(path)==C.read_bytes(target/snapshots[path]['path']),'complete final source pair conservation')
    value={'kind':'CPU_PRESERVED_OWNED_TENSOR_SIMULATION','scope':C.CPU_SCOPE,'manifest_sha256':manifest['sha256'],'raw_files':raw_map,'source_files':manifest['source_pins'],'source_snapshot_files':snapshots,'fixture_only':True,'native_authority_issued':False}
    value['sha256']=C.digest(value);write_new(target/'PRESERVATION.json',value);return value

def supervise_native(*args,**kwargs):
    C.stage_native_gate()
    from prefix_native_pipeline import supervise_registered
    return supervise_registered(*args,**kwargs)
def selected_death_verifier(value):
    path=C.PACKAGE/'native_supervisor.py';descriptor=value['config']['source_files'][str(path)]
    prior=sys.modules.get('native_supervisor')
    if prior is not None:C.require(type(getattr(prior,'__file__',None)) is str and Path(prior.__file__).absolute()==path,'no preloaded original supervisor alias')
    module=W.exact_module(path,descriptor['sha256']);C.require(module._pid_alive.__code__.co_filename==str(path) and module.__file__==str(path),'actual called death verifier compiled origin')
    return module._pid_alive
def preserve_native(output,target,config_path,**anchors):
    C.enroll_native_gate_from_owned_output(output,expected_binding={'config_path':str(Path(config_path).absolute()),'anchors':anchors});C.stage_native_gate();context=C.validate_native_manifest(config_path,**anchors);value=C.context_value(context)
    output=Path(output).absolute();target=Path(target).absolute();C.require(output.parent==C.PACKAGE/'artifacts' and target.parent==C.PACKAGE/'evidence' and output.is_dir() and not target.exists(),'exact completed owned output and new evidence leaf')
    sup_raw=C.read_bytes(output/'SUPERVISOR_RECEIPT.json');supervisor=C.parse(sup_raw);C.require(supervisor['config_sha256']==anchors['expected_config_file'] and supervisor['sampler_threads_terminal_verified'] is supervisor['gpu_telemetry_cleanup_verified'] is supervisor['partial_outputs_frozen'] is True and supervisor['unjoined_sampler_threads']==[],'terminal actual samplers/NVML/frozen raw required before preservation')
    C.require(supervisor['output_dir']==str(output) and supervisor['job_closed_kill_on_close'] is True and supervisor['owned_worker_identity_currently_alive'] is False and type(supervisor['returncode']) is int,'actual owned output/job/worker terminal before preservation')
    alive=selected_death_verifier(value)
    resources=C.parse(C.read_bytes(output/'RESOURCE_OBSERVATIONS.json'));roster={}
    def births(obj):
        if type(obj) is dict:
            values=obj.get('owned_process_identities',[])
            if type(values) is list:
                for entry in values:
                    if type(entry) is dict and 'pid' in entry and 'creation_filetime_100ns' in entry:
                        identity=C.identity({k:entry[k] for k in ('pid','creation_filetime_100ns')});roster[(identity['pid'],identity['creation_filetime_100ns'])]=identity
            for child in obj.values():births(child)
        elif type(obj) is list:
            for child in obj:births(child)
    births(resources);worker=supervisor['worker_identity'];identity=C.identity({k:worker[k] for k in ('pid','creation_filetime_100ns')});roster[(identity['pid'],identity['creation_filetime_100ns'])]=identity
    observed=[]
    for owner in roster.values():C.require(alive(owner) is False,'all current exact owned births dead before preservation');observed.append({'identity':owner,'currently_alive':False})
    death={'observed_monotonic':time.monotonic(),'observed_utc':datetime.now(timezone.utc).isoformat(),'supervisor_file_sha256':hashlib.sha256(sup_raw).hexdigest(),'config_file_sha256':anchors['expected_config_file'],'observations':observed,'death_verifier_source_sha256':value['config']['helper_pins']['native_supervisor.py']}
    sources=value['config']['source_files'];names=sorted(p.name for p in output.iterdir());C.require(len(names)<=4096 and all((output/n).is_file() for n in names) and sum((output/n).stat().st_size for n in names)<=128*1024**2 and sum(v['bytes'] for v in sources.values())<=64*1024**2,'finite flat frozen original native raw/source budget');lease_root=Path(supervisor['lease_path']);lease_names=('reservation.json','worker.json','resume.json','terminal.json')
    C.require(lease_root.parent==C.PACKAGE/'artifacts'/'_native_gpu_leases' and lease_root.resolve()==lease_root and set(p.name for p in lease_root.iterdir())==set(lease_names),'exact terminal owned lease four-file roster')
    raw_before={n:C.read_bytes(output/n) for n in names};lease_before={n:C.read_bytes(lease_root/n) for n in lease_names};C.require(lease_before['terminal.json']==sup_raw,'exact original lease terminal bytes equal SUP')
    partial={Path(entry['path']).name:entry for entry in supervisor['partial_outputs']};C.require(set(partial)==set(names)-{'SUPERVISOR_RECEIPT.json'} and all(entry['bytes']==len(raw_before[name]) and entry['sha256']==hashlib.sha256(raw_before[name]).hexdigest() for name,entry in partial.items()),'exact original SUP partial raw byte denominator')
    target.mkdir();raw_files={};lease_files={};snapshots={};(target/'source_snapshots').mkdir()
    def copy_bytes(path,raw):
        with path.open('xb') as stream:C.require(stream.write(raw)==len(raw),'complete preserved native bytes');stream.flush();os.fsync(stream.fileno())
        C.require(C.read_bytes(path)==raw,'preserved native readback');return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    for name,raw in raw_before.items():raw_files[name]=copy_bytes(target/name,raw)
    for name,raw in lease_before.items():lease_files[name]={'source_path':str(lease_root/name),'path':'lease_'+name,**copy_bytes(target/('lease_'+name),raw)}
    for source,descriptor in sources.items():
        C.require(Path(source).suffix in ('.py','.json','.md'),'explicit source-only snapshot');C.file_descriptor(source,descriptor);raw=C.read_bytes(source);name=hashlib.sha256(source.encode('utf-8')).hexdigest()+Path(source).suffix
        copied=copy_bytes(target/'source_snapshots'/name,raw);snapshots[source]={'path':'source_snapshots/'+name,**copied,'role':descriptor['role']}
    C.require(sorted(p.name for p in output.iterdir())==names,'original native raw roster unchanged')
    for name,raw in raw_before.items():C.require(C.read_bytes(output/name)==C.read_bytes(target/name)==raw,'final exact native raw pair')
    for name,raw in lease_before.items():C.require(C.read_bytes(lease_root/name)==C.read_bytes(target/('lease_'+name))==raw,'final exact native lease pair')
    for source,descriptor in sources.items():C.file_descriptor(source,descriptor);C.require(C.read_bytes(source)==C.read_bytes(target/snapshots[source]['path']),'final exact native source pair')
    C.require(set(p.name for p in target.iterdir())==set(names)|{'source_snapshots'}|{'lease_'+n for n in lease_names} and set(p.name for p in (target/'source_snapshots').iterdir())=={Path(v['path']).name for v in snapshots.values()},'complete exact preserved native rosters')
    C.revalidate_context(context)
    receipt={'kind':'PRESERVED_REGISTERED_ONE_PREFIX_NATIVE_TENSOR_DIAGNOSTIC','scope':C.NATIVE_SCOPE,'manifest_sha256':value['config']['sha256'],'raw_files':raw_files,'source_files':sources,'source_snapshot_files':snapshots,'lease_files':lease_files,'current_death_observation':death,'fixture_only':False,'native_authority_issued':False};receipt['sha256']=C.digest(receipt);write_new(target/'PRESERVATION.json',receipt);return receipt
def _CPU_preserve_native_fixture(output,target,context,*,contract,_capability,death_verifier,revalidate):
    """Execute actual preservation body with a separate CPU-only capability.

    Only the native admission/death/revalidation boundary is substituted; all
    terminal/raw/lease/source/roster/copy/fsync/readback logic is shared source.
    """
    contract._check_CPU_native_context_fixture(_capability);value=contract.context_value(context)
    contract.require(value.get('fixture_only') is True and value.get('native_PASS') is False and value.get('native_authority_issued') is False,'only sealed CPU outer context')
    path=contract.PACKAGE/'prefix_input_owned_supervisor.py';descriptor=value['config']['source_files'][str(path)];raw=contract.read_bytes(path)
    contract.require(len(raw)==descriptor['bytes'] and hashlib.sha256(raw).hexdigest()==descriptor['sha256'],'actual selected preservation source before AST compile')
    function=next(n for n in ast.parse(raw,str(path)).body if isinstance(n,ast.FunctionDef) and n.name=='preserve_native')
    function.name='_CPU_shared_preservation_body';changes=[]
    class FixtureBoundary(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='C':
                if node.func.attr=='enroll_native_gate_from_owned_output':changes.append('enrollment');return ast.copy_location(ast.Call(func=ast.Name(id='_CPU_check',ctx=ast.Load()),args=[],keywords=[]),node)
                if node.func.attr=='stage_native_gate':changes.append('stage');return ast.copy_location(ast.Call(func=ast.Name(id='_CPU_check',ctx=ast.Load()),args=[],keywords=[]),node)
                if node.func.attr=='validate_native_manifest':changes.append('validation');return ast.copy_location(ast.Name(id='_CPU_context',ctx=ast.Load()),node)
                if node.func.attr=='revalidate_context':changes.append('revalidate');return ast.copy_location(ast.Call(func=ast.Name(id='_CPU_revalidate',ctx=ast.Load()),args=[],keywords=[]),node)
            return node
        def visit_Dict(self,node):
            node=self.generic_visit(node)
            for key,index in zip(node.keys,range(len(node.keys))):
                if isinstance(key,ast.Constant) and key.value=='fixture_only':node.values[index]=ast.Constant(True);changes.append('classification')
            return node
    function=FixtureBoundary().visit(function);ast.fix_missing_locations(function);contract.require(sorted(changes)==['classification','enrollment','revalidate','stage','validation'],'exact five private preservation boundary adaptations including source-only enrollment')
    namespace=dict(globals());namespace.update(C=contract,_CPU_context=context,_CPU_check=lambda:contract._check_CPU_native_context_fixture(_capability),_CPU_revalidate=revalidate,selected_death_verifier=lambda _:death_verifier)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    result=namespace['_CPU_shared_preservation_body'](output,target,value['config_path'],**value['anchors']);contract.require(result['fixture_only'] is True and result['native_authority_issued'] is False,'private preservation cannot issue native authority');return result
if __name__=='__main__':C.stage_native_gate()
