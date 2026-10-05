"""CPU component cost benchmark only; no model, framework, GPU or retry permit.

Prepared while native collection is live. The measurement CLI requires an
explicit CPU-measurement switch and is to be invoked only after root's terminal
signal. It starts a separately owned CPU Job and retains current CPU guards and
durability. GPU costs are excluded rather than mocked into native qualification.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

WORK=Path(__file__).resolve().parent
PACKAGE=WORK.parents[1]
sys.path.insert(0,str(PACKAGE))
import checkpoint_successor_profile as profile
from backend_qual_worker import DurableJournal
from inference_qual_worker import make_guard
from native_worker import verify_worker_ownership
from study_native_chunk import _PrimitiveCollector
from study_torch_backend import StudyTorchBackend
import native_supervisor as supervisor

SCOPE='CPU_COMPONENT_JOURNAL_GUARD_COUNTED_FAKE28_NO_NATIVE_AUTHORITY'
SOURCE_NAMES=('native_worker.py','backend_qual_worker.py','inference_qual_worker.py',
              'study_native_chunk.py','study_torch_backend.py','backend_qual_core.py',
              'checkpoint_successor_profile.py','native_supervisor.py','native_process.py')
EXPECTED_NONREGION={'gradient_reset':1,'base_storage_validate':41,'row_tensor_materialize':20,
 'training_forward':20,'forward_array_integrity_validate':20,'completion_CE_selected_NLL_check':20,
 'training_backward':20,'finite_gradient_clip':1,'optimizer_step':1,'optimizer_boundary_validate':1,'consumed_gradient_clear':1}
EXPECTED_CALLS=1266
EXPECTED_GUARDS=7722


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pins():return {str(p):sha(p) for p in [*(PACKAGE/n for n in SOURCE_NAMES),Path(__file__),WORK/'test_journal_guard_benchmark.py']}
def new_json(path,value):supervisor._json_new(path,value)
def no_framework():require(not any(n in sys.modules for n in ('torch','transformers','jax','tensorflow','flax')),'framework import prohibited')


def constructor_ownership(output,worker,supervisor_pid,supervisor_create_time,config_sha256):
    """Consume the constructor's immutable receipt; never create/replace it."""
    owner=json.loads((Path(output)/'supervisor_ownership.json').read_bytes())
    expected={'worker_pid':worker.process.pid,'root_worker_pid':worker.process.pid,
              'job_name':worker.job.name,'supervisor_pid':supervisor_pid,
              'supervisor_create_time':supervisor_create_time,'config_sha256':config_sha256}
    require(owner==expected and all(type(owner[k]) is int and owner[k]>0 for k in ('worker_pid','root_worker_pid','supervisor_pid'))
            and type(owner['supervisor_create_time']) is float,'actual constructor ownership receipt identity/config mismatch')
    return owner


class Parameter:
    shape=(2,2);dtype='CPU_FIXTURE_BF16_METADATA_ONLY';device='cpu';requires_grad=False
    def data_ptr(self):return id(self)
    def stride(self):return (2,1)
    def untyped_storage(self):return self
    def nbytes(self):return 8
class Layer:
    training=False;gradient_checkpointing=False;dropout=0.
    def __init__(self,index):self.index=index
    def forward(self,value,**kwargs):return value+self.index+1
class Model:
    training=False
    def __init__(self):
        self.model=SimpleNamespace(layers=[Layer(i) for i in range(28)])
        self.config=SimpleNamespace(attention_dropout=0.,use_cache=True)
        self.parameters=[Parameter() for _ in range(28)]
    def named_modules(self):return [('',self)]+[(str(i),v) for i,v in enumerate(self.model.layers)]
    def named_parameters(self,remove_duplicate=False):return [(str(i),v) for i,v in enumerate(self.parameters)]
class Checkpoint:
    def __init__(self):self.early=None;self.tape=[]
    @contextmanager
    def early_context(self,value):
        self.early=value
        try:yield
        finally:self.early=None
    def __call__(self,function,*args,**kwargs):
        require(self.early is False and kwargs.pop('use_reentrant') is False and kwargs.pop('preserve_rng_state') is True,'full CPU fake checkpoint mechanism')
        forward,recompute=kwargs.pop('context_fn')()
        with forward:value=function(*args,**kwargs)
        self.tape.append((function,args,kwargs,recompute));return value
    def backward(self):
        tape,self.tape=self.tape,[]
        for function,args,kwargs,context in reversed(tape):
            with context:function(*args,**kwargs)


class GuardMeter:
    def __init__(self,guard):self.guard=guard;self.calls=0;self.returned=0;self.failed=0;self.elapsed=0.;self.stages=Counter()
    def __call__(self,stage):
        self.calls+=1;self.stages[stage]+=1;start=time.perf_counter()
        try:self.guard(stage)
        except BaseException:self.failed+=1;raise
        else:self.returned+=1
        finally:self.elapsed+=time.perf_counter()-start
    def snapshot(self):return {'attempted':self.calls,'returned':self.returned,'failed':self.failed,'elapsed_seconds':self.elapsed,'stages':dict(self.stages)}
class DirectJournal:
    """Guard-only factor: counts in memory, explicitly no durability measurement."""
    def __init__(self,guard):self.guard=guard;self.counts={};self.evidence_count=0
    def call(self,name,function):
        self.guard('before_'+name);c=self.counts.setdefault(name,dict.fromkeys(('attempted','returned','failed','pending'),0));c['attempted']+=1;c['pending']+=1
        try:value=function()
        except BaseException:c['failed']+=1;c['pending']-=1;raise
        c['returned']+=1;c['pending']-=1;self.guard('after_'+name);return value
    def evidence(self,name,value):self.evidence_count+=1
    def snapshot(self):return {'operations':deepcopy(self.counts)}


class MeteredDurableJournal(DurableJournal):
    """Time the unchanged actual write/flush/fsync; do not batch or defer it."""
    def __init__(self,directory):
        super().__init__(directory);self.emit_returned=0;self.emit_failed=0;self.emit_seconds=0.
        self.evidence_returned=0;self.evidence_failed=0;self.evidence_seconds=0.
    def emit(self,value):
        start=time.perf_counter()
        try:super().emit(value)
        except BaseException:self.emit_failed+=1;raise
        else:self.emit_returned+=1
        finally:self.emit_seconds+=time.perf_counter()-start
    def evidence(self,name,value):
        start=time.perf_counter()
        try:super().evidence(name,value)
        except BaseException:self.evidence_failed+=1;raise
        else:self.evidence_returned+=1
        finally:self.evidence_seconds+=time.perf_counter()-start
    def durability_snapshot(self):
        return {key:getattr(self,key) for key in ('emit_returned','emit_failed','emit_seconds','evidence_returned','evidence_failed','evidence_seconds')}


def exercise(journal,guard):
    """Exact recorded reset→clear frame:20 rows,1120 regions,1266 primitives.

    No loader, losses, real parameters, optimizer or base scan is performed.
    Real wrappers execute with injected scalar bodies and retain all guard sites.
    """
    model,checkpoint=Model(),Checkpoint()
    collector=_PrimitiveCollector(journal.call,guard)
    backend=object.__new__(StudyTorchBackend)
    backend._failed=False;backend.guard=guard;backend.call=collector;backend.primitive_counts={}
    installed=profile.install(model,SimpleNamespace(is_grad_enabled=lambda:True),checkpoint,
        early_stop_context=checkpoint.early_context,call=collector,guard=guard,clock=time.perf_counter)
    guards_before=guard.snapshot();calls_before=deepcopy(collector.snapshot())
    durability_before=journal.durability_snapshot() if hasattr(journal,'durability_snapshot') else None
    start=time.perf_counter();protocol=Counter();partial={}
    def invoke(name,function):
        protocol[name]+=1;guard('qualification_before_'+name)
        value=function();guard('qualification_after_'+name);return value
    primitive=backend._invoke
    try:
        invoke('gradient_reset',lambda:primitive('gradient_reset',lambda:None))
        for row in range(20):
            def forward():
                primitive('base_storage_validate',lambda:None)
                primitive('row_tensor_materialize',lambda:None)
                def original():
                    value=0
                    for layer in model.model.layers:value=layer.forward(value,use_cache=False)
                    require(value==406,'all28 scalar decoder bodies')
                    return value
                output=primitive('training_forward',original)
                primitive('base_storage_validate',lambda:None);return output
            invoke('forward',forward)
            def loss():
                primitive('forward_array_integrity_validate',lambda:None)
                return primitive('completion_CE_selected_NLL_check',lambda:None)
            invoke('loss',loss)
            invoke('backward',lambda:primitive('training_backward',checkpoint.backward))
            journal.evidence('ce_u01_r%02d'%(row+1),{'CPU_COMPONENT_FIXTURE':True,'no_loss_computed':True})
        def clip():
            primitive('base_storage_validate',lambda:None);primitive('finite_gradient_clip',lambda:None)
        invoke('clip',clip)
        def step():
            primitive('optimizer_step',lambda:None);primitive('optimizer_boundary_validate',lambda:None)
            primitive('consumed_gradient_clear',lambda:None)
        invoke('step',step)
        partial['status']='COMPLETE_CPU_COMPONENT_COUNTED_UPDATE_FRAME'
    except BaseException as error:
        partial.update(status='FAILED_CPU_COMPONENT_COUNTED_UPDATE_FRAME',error={'type':type(error).__name__,'message':str(error)})
    finally:
        wall=time.perf_counter()-start
        frame_guards=guard.snapshot();frame_guards={k:frame_guards[k]-guards_before[k] for k in ('attempted','returned','failed','elapsed_seconds')}
        primitive_counts=collector.snapshot()
        for name,item in calls_before.items():
            for key,value in item.items():primitive_counts[name][key]-=value
            if not any(primitive_counts[name].values()):del primitive_counts[name]
        snapshot=installed['snapshot']()
        partial.update(wall_seconds=wall,guards=frame_guards,primitive_operations=primitive_counts,
                       protocol_operations=dict(protocol),decoder_counters=snapshot['counters'],
                       decoder_failed_or_pending=snapshot['failed_or_pending'],regions=snapshot['regions'])
        if durability_before is not None:
            partial['durability']={k:v-durability_before[k] for k,v in journal.durability_snapshot().items()}
            partial['durability']['nested_timing_note']='evidence_seconds includes its emit; do not add it to emit_seconds'
        try:installed['remove']()
        except BaseException as error:partial['cleanup_error']={'type':type(error).__name__,'message':str(error)}
    if partial['status'].startswith('COMPLETE'):
        require(sum(x['attempted'] for x in primitive_counts.values())==EXPECTED_CALLS,'exact1266 primitives')
        require(frame_guards['attempted']==EXPECTED_GUARDS and frame_guards['returned']==EXPECTED_GUARDS,'exact7722 guard sites')
        require(snapshot['counters']=={'forward_regions':560,'recompute_regions':560,'no_grad_regions':0},'exact1120 regions')
        require(all(x['returned']==x['attempted'] and x['failed']==x['pending']==0 for x in primitive_counts.values()),'all actual returned costs retained')
        require({k:v['attempted'] for k,v in primitive_counts.items() if not k.startswith('checkpoint_layer')}==EXPECTED_NONREGION,'recorded controller inventory')
    return partial


def child(output):
    output=Path(output).absolute()
    require(output.is_dir() and output.resolve(strict=True)==output and output.parent==WORK
            and not output.is_symlink(),'CPU measurement writes stay in one new staging child')
    scope=json.loads((output/'CPU_SCOPE.json').read_bytes())
    require(scope.get('scope')==SCOPE and scope.get('no_GPU') is True and scope.get('no_training') is True
            and scope.get('no_native_qualification') is True,'disjoint CPU benchmark authority only')
    owner=json.loads((output/'supervisor_ownership.json').read_bytes())
    verify_worker_ownership(owner,output/'CPU_SCOPE.json');before=pins();summaries=[]
    for mode in ('journal_only','guard_only','full_candidate'):
        leaf=output/mode;leaf.mkdir(exist_ok=False)
        actual_guard=make_guard(output,owner,time.monotonic(),lambda:None)
        meter=GuardMeter((lambda stage:None) if mode=='journal_only' else actual_guard)
        journal=DirectJournal(meter) if mode=='guard_only' else MeteredDurableJournal(leaf)
        journal.guard=meter
        try:result=exercise(journal,meter)
        finally:
            if hasattr(journal,'stream'):journal.stream.close()
        require(result['status'].startswith('COMPLETE') and not result.get('cleanup_error'),'bounded component phase must complete')
        result['scope']=SCOPE;result['factor']=mode;result['GPU_guard_measured']=False
        result['current_CPU_guard_measured']=mode!='journal_only';result['per_record_durability_measured']=mode!='guard_only'
        new_json(leaf/'CPU_component_result.json',result);summaries.append({k:v for k,v in result.items() if k!='regions'})
    require(before==pins(),'source drift');no_framework()
    new_json(output/'CPU_worker_result.json',{'status':'PASS_CPU_ONLY_JOURNAL_GUARD_FACTORS_NOT_NATIVE_FIT',
        'scope':SCOPE,'source_before':before,'source_after':pins(),'factors':summaries,
        'native_launch_authorized':False,'training_authorized':False,'fullrun_estimate_performed':False,
        'GPU_NVML_called':False,'framework_imported':False})


def parent():
    """Invoke only after root terminal signal; CPU Job only, no GPU monitor."""
    no_framework();before=pins();output=WORK/('measurement_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    new_json(output/'CPU_SCOPE.json',{'scope':SCOPE,'matched_rows':20,'regions':1120,'primitive_calls':1266,'guard_calls':7722,
        'no_native_qualification':True,'no_training':True,'no_GPU':True,'no_fullrun_prediction':True})
    worker=None;observations=[];started=time.monotonic();failure=None
    try:
        worker=supervisor._WindowsOwnedWorker([sys.executable,'-I','-B',str(Path(__file__)),'--cpu-child',str(output)],output,
            config_sha256=sha(output/'CPU_SCOPE.json'),deadline_monotonic=started+600)
        owner=constructor_ownership(output,worker,os.getpid(),supervisor._pid_identity(os.getpid())['create_time'],sha(output/'CPU_SCOPE.json'))
        supervisor._heartbeat(output,os.getpid());worker.resume()
        while worker.poll() is None:
            sample=supervisor._cpu_snapshot(output,worker);observations.append(sample)
            require(supervisor._fresh_cpu(sample,time.monotonic()),'CPU fixture external freshness200ms')
            require(sample['host_available_bytes']>=1.25*1024**3 and sample['worker_tree_rss_bytes']<=2*1024**3 and sample['disk_free_bytes']>=4*1024**3,'unchanged CPU resource floors')
            require(time.monotonic()-started<600,'unchanged600s maximum')
            supervisor._heartbeat(output,os.getpid());time.sleep(.1)
        require(worker.poll()==0,'owned CPU component worker failed')
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
        if worker is not None:worker.terminate()
    finally:
        if worker is not None:worker.close()
        require(before==pins(),'source drift during CPU component measurement')
        new_json(output/'CPU_SUPERVISOR_RECEIPT.json',{'status':'PASS_CPU_ONLY_COMPONENT_MEASUREMENT' if failure is None else 'FAILED_CPU_ONLY_COMPONENT_MEASUREMENT',
            'scope':SCOPE,'source_before':before,'source_after':pins(),'failure':failure,'wall_seconds':time.monotonic()-started,
            'observations':observations,'returncode':worker.poll() if worker else None,'Job_closed':worker is not None,
            'GPU_NVML_called':False,'framework_imported':False,'native_launch_authorized':False,'training_authorized':False,
            'inventory':supervisor._inventory(output),'limits':{'CPU_freshness':.2,'CPU_cadence':.1,'disk_GiB':4.,'maximum_seconds':600}})
    print(output)
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-child',type=Path);parser.add_argument('--measure-after-root-terminal',action='store_true')
    args=parser.parse_args()
    if args.cpu_child:child(args.cpu_child)
    else:
        require(args.measure_after_root_terminal,'measurement requires explicit post-terminal CPU switch')
        parent()
