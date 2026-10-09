"""Owned STUDY worker/supervisor adapter. Staging cannot launch native work.

Scientific optimizer/resume logic remains in the selected study_training source.
Only purpose/worker and externally selected output/lease locations are adapted
from the original supervisor. Worker evidence is pending post-Job terminal audit.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from types import ModuleType
import ast,hashlib,importlib,json,math,os,random,re,sys,time,uuid

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'study_owned_runtime_v1'
SCOPE='REGISTERED_STUDY_ONE_FRESH_NINE_UPDATE_CHUNK_V1'
SUP_SHA='edf2de5b0917150c2c79e6a555dc0399995d12859028e62a7f7ece86b767e691'
CONTEXT='OWNED_STUDY_CONTEXT.json'
CALLBACKS={'frameworks','call','guard','preserve','admission','evidence','cleanup','terminal_audit'}

def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def seal(value):return {**value,'sha256':digest(value)}
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _pairs(items):
    out={}
    for k,v in items:require(k not in out,'duplicate JSON field');out[k]=v
    return out
def _bad_constant(value):raise ValueError('nonfinite JSON')
def read_ref(ref,*,source=False):
    require(type(ref) is dict and set(ref)==({'path','bytes','sha256'} if source else {'path','bytes','sha256','value_sha256'}),'exact selected source/JSON descriptor')
    path=Path(ref['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink() and type(ref['bytes']) is int and 0<ref['bytes']<=16*1024**2,'bounded selected regular bytes')
    with path.open('rb') as stream:raw=stream.read(ref['bytes']+1)
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'selected source/JSON byte pin')
    if source:return raw
    value=json.loads(raw,object_pairs_hook=_pairs,parse_constant=_bad_constant);require(digest(value)==ref['value_sha256'],'selected JSON value pin');return value
def source_module(ref):
    raw=read_ref(ref,source=True);m=ModuleType('selected_owned_'+ref['sha256'][:16]);m.__file__=ref['path'];sys.modules[m.__name__]=m;exec(compile(raw,m.__file__,'exec'),m.__dict__);return m
def original_module(name):
    module=importlib.import_module(name);require(Path(module.__file__).resolve()==PACKAGE/(name+'.py'),'original owned helper source origin');return module
def write_new(path,value):
    raw=(canonical(value)+'\n').encode('ascii')
    with Path(path).open('xb') as stream:require(stream.write(raw)==len(raw),'complete exclusive publication');stream.flush();os.fsync(stream.fileno())
    require(Path(path).read_bytes()==raw,'durable publication reread');return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'payload_sha256':digest(value),'durable':True}
def _cap_factory():
    token=object()
    def get():return token
    def check(value):require(value is token,'private authored owned-runtime capability')
    return get,check
_CPU_capability,_check_cap=_cap_factory()
_CPU_selected_supplier=None

def _CPU_install_selected_supplier(value,engine,prepared,cpu,*,capability):
    global _CPU_selected_supplier
    _check_cap(capability);require(prepared.get('fixture_only') is True,'only authored prepared metadata supplier')
    _CPU_selected_supplier=(Context(value,engine,prepared,capability=_CPU_capability()),cpu)

class Context:
    def __init__(self,value,engine,prepared,*,capability,launch_ref=None):
        _check_cap(capability);self.value=deepcopy(value);self.engine=engine;self.prepared=prepared;self.original=digest(value);self.launch_ref=deepcopy(launch_ref)
def _context_value(context):require(type(context) is Context and digest(context.value)==context.original,'original bound owned context');return deepcopy(context.value)

def _output_check(context,*,worker=False):
    value=_context_value(context);output=Path(value['output']);root=Path(value['output_root'])
    require(output.parent==root and root.resolve()==root and output.resolve()==output and output.is_dir() and not output.is_symlink(),'exact already-created selected owned leaf')
    saved=json.loads((output/CONTEXT).read_bytes(),object_pairs_hook=_pairs,parse_constant=_bad_constant)
    require(saved=={'launch_ref':context.launch_ref,'value':value},'original pre-resume launch context bytes/value')
    allowed={CONTEXT,'supervisor_ownership.json','supervisor_heartbeat.jsonl','worker.stdout.log','worker.stderr.log'}
    require(all(p.name in allowed and p.is_file() and not p.is_symlink() for p in output.iterdir()),'no foreign existing output contents before worker')
    if worker:
        owner=json.loads((output/'supervisor_ownership.json').read_bytes(),object_pairs_hook=_pairs,parse_constant=_bad_constant)
        require(owner.get('config_sha256')==value['engine_config']['sha256'] and type(owner.get('worker_pid')) is int and owner['worker_pid']>0 and type(owner.get('supervisor_pid')) is int and owner['supervisor_pid']>0,'original config-bound supervisor ownership before engine data revalidation')
    return output

def engine_prepare_existing(engine,config_path,anchors,context,*,worker=True):
    """One source-derived output-state substitution, confined to this owned leaf."""
    _output_check(context,worker=worker)
    tree=ast.parse(Path(engine.__file__).read_bytes());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_validate');changed=0
    for node in ast.walk(function):
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not) and isinstance(node.operand,ast.Call) and isinstance(node.operand.func,ast.Attribute) and isinstance(node.operand.func.value,ast.Name) and node.operand.func.value.id=='output' and node.operand.func.attr=='exists':
            node.operand.func.attr='is_dir';replacement=node.operand;node.__class__=ast.Call;node.__dict__.clear();node.__dict__.update(replacement.__dict__);changed+=1
    require(changed==1,'exact sole fresh-output engine assertion adaptation')
    namespace=dict(engine.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),engine.__file__,'exec'),namespace)
    return namespace['_validate'](config_path,anchors)

def _native_launch_context(launch_ref,*,existing_output=None):
    require(HERE==EXPECTED,'staged owned STUDY runtime has no native authority')
    value=read_ref(launch_ref);fields={'kind','scope','engine_source','engine_config','engine_anchors','source_files','runtime_files','output_root','output','runtime_versions','offline_environment','sha256'}
    require(type(value) is dict and set(value)==fields and value['kind']=='REGISTERED_OWNED_STUDY_LAUNCH_V1' and value['scope']==SCOPE and value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'exact owned STUDY launch packet')
    config=read_ref(value['engine_config']);approval=read_ref(config['approval'])
    require(config['fixture_only'] is False and approval.get('kind')=='OWNER_STUDY_TRAINING_CHUNK_AUTHORIZATION_V1' and approval.get('scope')==SCOPE and approval.get('approved') is True and approval.get('fixture_only') is False and approval.get('permissions')=={'training':True,'model':True,'optimizer':True,'scoring':False,'protected_bank_access':False,'paid_services':False,'promotion':False} and approval.get('source_pins')==config['source_pins'] and all(approval.get(k)==config[k] for k in ('arm','chunk','baseline_id','owned_boundary_source','owned_boundary_name','model_directory','output_directory')),'separate exact STUDY permission before runtime/data/API')
    review=read_ref(config['review']);require(review.get('kind')=='INDEPENDENT_STUDY_TRAINING_CHUNK_REVIEW_V1' and review.get('accepted') is True and review.get('fixture_only') is False and review.get('scope')==SCOPE and review.get('source_pins')==config['source_pins'] and review.get('approval_file_sha256')==config['approval']['sha256'],'exact independent STUDY review before runtime/source maps')
    require(value['engine_source']['path']==str(PACKAGE/'study_training_v1/study_training.py') and value['engine_anchors']=={'config_file_sha256':value['engine_config']['sha256'],'config_value_sha256':value['engine_config']['value_sha256']},'exact selected engine config path/file/value anchors')
    require(value['output']==config['output_directory'] and config['owned_boundary_source']['path']==str(EXPECTED/'owned_study.py') and config['owned_boundary_name']=='build_owned_boundaries','exact concrete STUDY owned builder/output')
    root=Path(value['output_root']);repository=PACKAGE.parents[1];business=repository.parent
    require(root.is_absolute() and root.resolve()==root and root.is_dir() and root.is_relative_to(business) and not root.is_relative_to(repository) and '.codex' not in root.parts and Path(value['output']).parent==root,'bound BusinessProject output outside source/.codex')
    required={str(PACKAGE/name) for name in ('native_worker.py','inference_qual_worker.py','native_process.py','native_supervisor.py','native_nvml.py','worker_ram_probe.py','native_contract.py','study_native_chunk.py','study_schedule.py','study_torch_backend.py','native_loader.py','native_mechanics.py','native_precision.py','checkpoint_successor_profile.py')}
    require(required|{value['engine_source']['path'],str(EXPECTED/'owned_study.py')}<=set(value['source_files']),'complete concrete original scientific/owned source members')
    require(all(config['source_pins'].get(path)==ref['sha256'] for path,ref in value['source_files'].items()),'source map exactly belongs to approved engine source closure')
    qualification=read_ref(config['training_qualification_check']['expected_result'])
    require(qualification.get('runtime_files')==value['runtime_files'],'runtime byte map belongs to selected reviewed qualification evidence')
    for mapping in (value['source_files'],value['runtime_files']):
        require(type(mapping) is dict and mapping and len(mapping)<=32768,'finite original source/runtime maps')
        for path,ref in mapping.items():
            require(ref['path']==path and Path(path).suffix.lower() in {'.py','.pyi','.pyc','.pyd','.dll','.exe','.zip','.json','.c','.h'},'no model/data assets in runtime byte preflight');read_ref(ref,source=True)
    require(value['source_files'].get(str(EXPECTED/'owned_study.py'))==config['owned_boundary_source'],'owned builder source belongs to selected original byte closure')
    native=source_module(value['source_files'][str(PACKAGE/'native_contract.py')]);require(value['runtime_versions']==native.RUNTIME and value['offline_environment']==native.OFFLINE_ENV,'original frozen runtime/offline settings')
    engine=source_module(value['engine_source'])
    context=Context(value,engine,None,capability=_CPU_capability(),launch_ref=launch_ref)
    if existing_output is None:prepared=engine._validate(value['engine_config']['path'],value['engine_anchors'])
    else:
        require(Path(existing_output).absolute()==Path(value['output']),'worker exact selected output argument')
        prepared=engine_prepare_existing(engine,value['engine_config']['path'],value['engine_anchors'],context)
    require(prepared['fixture_only'] is False and prepared['config']['scope']==SCOPE,'genuine independently qualified M0/allguard/STUDY engine context')
    prepared['owned_launch_ref']=deepcopy(launch_ref);prepared['owned_engine_config_path']=value['engine_config']['path'];prepared['owned_engine_external_anchors']=deepcopy(value['engine_anchors']);context.prepared=prepared
    return context

def worker_helpers(*,cpu=None):
    """Original worker journal/ownership/heartbeat/terminal functions only."""
    worker=ast.parse((PACKAGE/'native_worker.py').read_bytes());guard=ast.parse((PACKAGE/'inference_qual_worker.py').read_bytes())
    names=('diagnostic_json','_job_query_api','_job_query_error','owned_job_pids','owned_job_resources','in_owned_job','verify_worker_ownership','read_heartbeat','Journal','preserve_terminal')
    definitions=[deepcopy(next(n for n in worker.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)) for name in names]
    definitions.append(deepcopy(next(n for n in guard.body if isinstance(n,ast.FunctionDef) and n.name=='make_guard')))
    namespace={'Path':Path,'defaultdict':defaultdict,'datetime':datetime,'timezone':timezone,'json':json,'math':math,'os':os,'re':re,'time':time,'MAX_OWNED_JOB_PIDS':256,'GIB':1024**3,'canonical':canonical,'require':require,'contract':ModuleType('owned_original_contract_subset')}
    namespace['contract'].filehash=filehash;namespace['contract']._pairs=_pairs;namespace['contract']._bad_constant=_bad_constant
    if cpu is None:namespace['ctypes']=importlib.import_module('ctypes')
    else:
        namespace['os']=cpu.os;namespace['time']=cpu.time
        for node in ast.walk(definitions[6]):
            if isinstance(node,ast.Import) and len(node.names)==1 and node.names[0].name=='psutil':
                node.__class__=ast.Assign;node.__dict__.clear();node.__dict__.update(targets=[ast.Name(id='psutil',ctx=ast.Store())],value=ast.Name(id='_CPU_psutil',ctx=ast.Load()))
        namespace['_CPU_psutil']=cpu.psutil
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[])),str(PACKAGE/'native_worker.py'),'exec'),namespace)
    return namespace

def build_owned_boundaries(prepared_value,*,cpu=None):
    if type(prepared_value) is Context:
        require(cpu is not None,'native selected builder takes original prepared engine metadata');context=prepared_value
    else:
        require(type(prepared_value) is dict and 'owned_launch_ref' in prepared_value,'owned context bridge missing from prepared engine metadata')
        if prepared_value.get('fixture_only') is True:
            require(_CPU_selected_supplier is not None,'no authored supplier by default; fixture cannot mint native scope');context,cpu=_CPU_selected_supplier
        else:context=_native_launch_context(prepared_value['owned_launch_ref'],existing_output=prepared_value['config']['output_directory'])
        expected={k:v for k,v in context.prepared.items() if k not in ('modules','documents')}
        require(prepared_value==expected,'selected engine/builder metadata path/anchors/source identity')
    value=_context_value(context);output=_output_check(context,worker=True);original=worker_helpers(cpu=cpu)
    owner=json.loads((output/'supervisor_ownership.json').read_bytes(),object_pairs_hook=_pairs,parse_constant=_bad_constant)
    ownership=original['verify_worker_ownership'](owner,value['engine_config']['path'],**({} if cpu is None else {'membership':cpu.membership}))
    Journal=original['Journal'];journal=Journal(output);started=(time.monotonic() if cpu is None else cpu.time.monotonic());state={'torch':None,'journal':journal,'original':original,'owner':owner,'closed':False}
    available=(original_module('native_process').system_memory_status if cpu is None else cpu.available)
    def available_bytes():return available()['available_bytes']
    guard=original['make_guard'](output,owner,started,lambda:state['torch'],available=available_bytes,**({} if cpu is None else {'owned_resources':cpu.owned_resources,'monotonic':cpu.time.monotonic,'wall':cpu.time.time}))
    journal.guard=guard;journal.evidence('study_worker_ownership_verified',ownership)
    def frameworks():
        require(cpu is None,'authored CPU cannot import frameworks')
        for name,val in value['offline_environment'].items():os.environ[name]=val
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        import torch
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        import importlib.metadata
        require(str(torch.__version__)==value['runtime_versions']['torch'] and importlib.metadata.version('transformers')==value['runtime_versions']['transformers'],'original pinned framework versions')
        state['torch']=torch;guard('study_after_framework_import')
        require(available_bytes()>=1.25*1024**3+max(0,2*1024**3-original['owned_job_resources'](owner['job_name'])['tree_rss_bytes']),'original whole owned RSS growth reserve before CUDA')
        torch.cuda.init();require(torch.cuda.mem_get_info()[0]>=4*1024**3,'original4GiB GPU preload floor')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
        random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101)
        return torch,Qwen3Config,Qwen3ForCausalLM
    sequence=0
    def preserve(event):
        nonlocal sequence
        require(type(event) is dict,'original scientific publication event')
        if event['event']=='retained_update_path':
            ordinal=event['update_ordinal'];require(type(ordinal) is int and 1<=ordinal<=27 and event['arm']==context.prepared['config']['arm'] and event['chunk_ordinal']==context.prepared['config']['chunk'] and event['recipe_sha256']==context.prepared['recipe']['sha256'],'exact owned retained arm/chunk/recipe/ordinal')
            path=output/('study_retained_update_%02d.archive'%ordinal);require(not path.exists() and not path.is_symlink(),'fresh distinct retained archive');return str(path)
        if event['event']=='checkpoint_path':
            path=output/('study_C%d.archive'%(context.prepared['config']['chunk']*9));require(not path.exists() and not path.is_symlink(),'fresh STUDY archive; no qualification warmstart');return str(path)
        sequence+=1;payload=event['payload'];path=output/('%04d_%s.json'%(sequence,event['event']));descriptor=write_new(path,payload);journal.emit({'event':'study_publication_preserved','name':event['event'],'publication':descriptor});return descriptor
    def admission(request):
        guard('study_fresh_owned_admission');now=time.monotonic() if cpu is None else cpu.time.monotonic()
        return {'binding':deepcopy(request),'authorized':True,'owned_supervision_verified':True,'model_load_and_constructor_in_scope':True,'observed_monotonic':now,'expires_monotonic':now+.2}
    def cleanup():state['closed']=True
    cleanup._owned_state=state
    def terminal_audit(result):
        counts=journal.snapshot();require(all(n['attempted']==n['returned']+n['failed']+n['pending'] for n in counts['operations'].values()),'original primitive cost conservation')
        return {'kind':'WORKER_STUDY_PENDING_POST_JOB_TERMINAL_AUDIT','config_sha256':value['engine_config']['sha256'],'operation_accounting':counts,'owned_cleanup_returned':state['closed'],'native_PASS':False}
    boundaries={'frameworks':frameworks,'call':journal.call,'guard':guard,'preserve':preserve,'admission':admission,'evidence':journal.evidence,'cleanup':cleanup,'terminal_audit':terminal_audit};require(set(boundaries)==CALLBACKS,'exact eight original owned boundaries')
    return boundaries

def worker_body(context,*,cpu=None,factory=None):
    value=_context_value(context);engine=context.engine;prepared=context.prepared
    boundaries=build_owned_boundaries(context,cpu=cpu) if cpu is not None else engine._boundaries(prepared)
    state=boundaries['cleanup']._owned_state
    report={'status':'FAILED_BEFORE_SCIENTIFIC_DISPATCH','resume_descriptor':None}
    try:
        producer=engine._NativeFactory(prepared,boundaries) if factory is None else factory(prepared,boundaries)
        report=engine._run(prepared,producer,boundaries)
    except BaseException as error:
        report={'status':'OWNED_STUDY_WORKER_FAILED','error_type':type(error).__name__,'error':str(error),'resume_descriptor':None,'operation_accounting':state['journal'].snapshot(),'native_PASS':False}
    finally:
        report.update(config_sha256=value['engine_config']['sha256'],worker_scope=SCOPE,fixture_only=cpu is not None,scoring_authorized=False,quality_gain_established=False)
        report=seal({k:v for k,v in report.items() if k!='sha256'})
        state['original']['preserve_terminal'](Path(value['output']),report,state['journal'],state['torch'])
    return report

def supervisor_body(namespace,context):
    """Original lifecycle, exactly two worker literals and one scope literal."""
    raw=(PACKAGE/'native_supervisor.py').read_bytes();require(hashlib.sha256(raw).hexdigest()==SUP_SHA,'original supervisor bytes')
    tree=ast.parse(raw);function=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='supervise'));counts={'worker':0,'scope':0}
    for node in ast.walk(function):
        if isinstance(node,ast.Constant) and node.value=='native_worker.py':node.value='study_owned_runtime_v1/owned_study.py';counts['worker']+=1
        elif isinstance(node,ast.Constant) and node.value=='mechanical_native_qualification_only':node.value=SCOPE;counts['scope']+=1
    require(counts=={'worker':2,'scope':1},'exact original supervisor purpose/worker adaptation')
    value=_context_value(context);root=Path(value['output_root']);engine_config=value['engine_config']
    def preflight(request):
        require(str(Path(request).absolute())==engine_config['path'],'exact selected engine config request');read_ref(engine_config)
        for mapping in (value['source_files'],value['runtime_files']):
            for ref in mapping.values():read_ref(ref,source=True)
        if context.engine is not None and context.launch_ref is not None:
            engine_prepare_existing(context.engine,engine_config['path'],value['engine_anchors'],context,worker=False)
        config=context.prepared['config'];return {'config_path':engine_config['path'],'config_sha256':engine_config['sha256'],'worker_path':str(EXPECTED/'owned_study.py'),'resources':config['resources'],'authorization':{'scope':SCOPE,'production_training':False,'protected_bank_access':False,'paid_services':False}}
    original_new=namespace['_new_output'];original_acquire=namespace['_acquire_lease']
    def new_output(request):
        require(Path(request).absolute()==Path(value['output']),'one selected external STUDY output leaf');out=original_new(request);write_new(out/CONTEXT,{'launch_ref':context.launch_ref,'value':value});return out
    adapted={**namespace,'_preflight':preflight,'_new_output':new_output};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(PACKAGE/'native_supervisor.py'),'exec'),adapted)
    return adapted['supervise'],counts

def _external_original_helpers(namespace,root):
    """Move only output/lease roots; keep PACKAGE and mutex identity unchanged."""
    tree=ast.parse((PACKAGE/'native_supervisor.py').read_bytes());definitions=[];changes={}
    for name,target in (('_new_output','artifacts'),('_acquire_lease','root')):
        fn=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name));changed=0
        for node in ast.walk(fn):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==target:
                node.value=ast.Name(id='_external_root',ctx=ast.Load()) if target=='artifacts' else ast.BinOp(left=ast.Name(id='_external_root',ctx=ast.Load()),op=ast.Div(),right=ast.Constant(value='_native_gpu_leases'));changed+=1
        require(changed==1,'one original external location assignment');definitions.append(fn);changes[name]=changed
    namespace['_external_root']=Path(root);exec(compile(ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[])),str(PACKAGE/'native_supervisor.py'),'exec'),namespace);return changes

def run_registered(launch_ref):
    context=_native_launch_context(launch_ref)
    require(not Path(context.value['output']).exists(),'fresh externally selected native STUDY output')
    # All original APIs are imported after exact permission/qualification/data
    # and selected runtime/source preflight; imports alone never launch a worker.
    sys.path.insert(0,str(PACKAGE));original=original_module('native_supervisor');namespace=dict(original.__dict__);_external_original_helpers(namespace,context.value['output_root']);supervise,proof=supervisor_body(namespace,context)
    return supervise(context.value['engine_config']['path'],context.value['output'])

def main():
    require(HERE==EXPECTED,'staged owned STUDY worker has no native authority')
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    saved=json.loads((args.output_dir/CONTEXT).read_bytes(),object_pairs_hook=_pairs,parse_constant=_bad_constant)
    require(type(saved) is dict and set(saved)=={'launch_ref','value'} and str(args.config.absolute())==saved['value']['engine_config']['path'],'worker original selected config argument')
    context=_native_launch_context(saved['launch_ref'],existing_output=args.output_dir);require(context.value==saved['value'],'original selected launch context unchanged')
    report=worker_body(context);protocol=report.get('protocol_result',{})
    return 0 if protocol.get('status')=='COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED' else 2

if __name__=='__main__':raise SystemExit(main())
