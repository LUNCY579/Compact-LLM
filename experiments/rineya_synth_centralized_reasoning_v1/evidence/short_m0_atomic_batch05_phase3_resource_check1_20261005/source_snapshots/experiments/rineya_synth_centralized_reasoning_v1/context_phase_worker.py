"""Lazy owned NO_MODEL context phases; no loader, tensor or forward path."""
from copy import deepcopy
from datetime import datetime,timezone
import argparse
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import random
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import context_phase_contract as contract
GIB=1024**3


def projected(value):
    if type(value) is float and not math.isfinite(value):return {'invalid_nonfinite_float':repr(value)}
    if value is None or type(value) in (str,int,float,bool):return value
    if type(value) in (list,tuple):return [projected(x) for x in value]
    if type(value) is dict:return {str(k):projected(v) for k,v in value.items()}
    return {'opaque_return_type':type(value).__module__+'.'+type(value).__name__}


class Recorder:
    """No Journal and no guard callback: query recording cannot recurse guards."""
    def __init__(self,preserve,*,clock=time.monotonic):
        self.preserve,self.clock=preserve,clock;self.serial=0;self.operations={};self.errors=[];self.last=None
    def _save(self,name,value):
        try:self.preserve(name,deepcopy(value))
        except BaseException as error:self.errors.append({'type':type(error).__name__,'message':str(error)[:512]})
    def begin(self,name):
        self.serial+=1;ordinal=self.serial
        count=self.operations.setdefault(name,dict.fromkeys(('attempted','returned','failed','pending'),0))
        count['attempted']+=1;count['pending']+=1
        start={'ordinal':ordinal,'operation':name,'event':'attempt','monotonic':self.clock(),
               'utc':datetime.now(timezone.utc).isoformat(),'historical_cause_established':False}
        self.last=start;self._save('phase_%04d_start'%ordinal,start)
        return ordinal
    def end(self,ordinal,name,*,value=None,error=None):
        count=self.operations[name];count['pending']-=1;count['failed' if error is not None else 'returned']+=1
        event={'ordinal':ordinal,'operation':name,'event':'failed' if error is not None else 'returned',
               'monotonic':self.clock(),'utc':datetime.now(timezone.utc).isoformat(),
               'result':projected(value),'error':None if error is None else {'type':type(error).__name__,'message':str(error)},
               'historical_cause_established':False}
        self.last=event;self._save('phase_%04d_end'%ordinal,event)
    def call(self,name,function):
        ordinal=self.begin(name)
        try:value=function()
        except BaseException as error:
            self.end(ordinal,name,error=error);raise
        self.end(ordinal,name,value=value);return value
    def healthy(self):
        if self.errors:raise RuntimeError('diagnostic phase publication failed')
    def snapshot(self):return deepcopy({'operations':self.operations,'last_ordinal':self.serial,'last_phase':self.last,'publication_errors':self.errors})


class GuardCUDAProxy:
    """Exact one-call/result/error proxy for the two existing guard reads."""
    def __init__(self,cuda,recorder):self.actual,self.recorder=cuda,recorder
    def is_initialized(self):return self.recorder.call('guard_cuda_is_initialized',self.actual.is_initialized)
    def mem_get_info(self):return self.recorder.call('guard_cuda_mem_get_info',self.actual.mem_get_info)


class GuardTorchProxy:
    def __init__(self,torch,recorder):self.cuda=GuardCUDAProxy(torch.cuda,recorder)


def observed_guard(actual,recorder):
    def guard(stage):
        name='guard_'+stage;ordinal=recorder.begin(name)
        try:result=actual(stage)
        except BaseException as error:
            recorder.end(ordinal,name,error=error);raise
        recorder.end(ordinal,name,value=result);recorder.healthy();return result
    return guard


def context_phases(api,recorder,guard,runtime_versions):
    """Injected CPU functions or separately reviewed lazy native API suppliers."""
    def call(name,fn):
        guard('before_'+name);value=recorder.call(name,fn);guard('after_'+name);recorder.healthy();return value
    torch=call('torch_import',api['torch_import'])
    call('Qwen3Config_symbol_import',api['Qwen3Config_symbol_import'])
    call('Qwen3ForCausalLM_symbol_import',api['Qwen3ForCausalLM_symbol_import'])
    versions=call('framework_version_read',api['framework_version_read'])
    if versions!={'torch':runtime_versions['torch'],'transformers':runtime_versions['transformers']}:
        raise ValueError('actual framework runtime version drift')
    growth=call('post_import_physical_growth_observation',api['growth_observation'])
    if not (type(growth) is dict and set(growth)=={'available_bytes','worker_RSS_bytes'}
            and all(type(v) is int and v>=0 for v in growth.values())):raise ValueError('typed fresh growth observation')
    if growth['worker_RSS_bytes']>2*GIB or growth['available_bytes']<1.25*GIB+max(0,2*GIB-growth['worker_RSS_bytes']):
        raise ValueError('unchanged post-import full growth reserve')
    call('cuda_context_init',torch.cuda.init)
    memory=call('direct_CUDA_preload_mem_get_info',torch.cuda.mem_get_info)
    if not (type(memory) is tuple and len(memory)==2 and all(type(v) is int and v>=0 for v in memory)
            and memory[0]<=memory[1]):raise ValueError('typed direct CUDA memory observation')
    if memory[0]<4*GIB:raise ValueError('unchanged4GiB preload GPU admission')
    for name,fn in (('memory_allocated',torch.cuda.memory_allocated),('memory_reserved',torch.cuda.memory_reserved),
                    ('max_memory_allocated',torch.cuda.max_memory_allocated),('max_memory_reserved',torch.cuda.max_memory_reserved)):
        value=call('direct_CUDA_'+name,fn)
        if type(value) is not int or value<0:raise ValueError('typed nonnegative allocator observation')
    call('precision_and_thread_configuration',api['configure'])
    call('Python_seed_101',lambda:random.seed(101))
    call('Torch_CPU_seed_101',lambda:torch.manual_seed(101))
    call('Torch_CUDA_seed_all_101',lambda:torch.cuda.manual_seed_all(101))
    call('CUDA_reset_peak_memory_stats',torch.cuda.reset_peak_memory_stats)
    final=call('final_direct_CUDA_mem_get_info',torch.cuda.mem_get_info)
    if not (type(final) is tuple and len(final)==2 and all(type(v) is int and v>=0 for v in final) and final[0]<=final[1]):
        raise ValueError('typed final direct CUDA memory observation')
    for name,fn in (('memory_allocated',torch.cuda.memory_allocated),('memory_reserved',torch.cuda.memory_reserved),
                    ('max_memory_allocated',torch.cuda.max_memory_allocated),('max_memory_reserved',torch.cuda.max_memory_reserved)):
        value=call('final_direct_CUDA_'+name,fn)
        if type(value) is not int or value<0:raise ValueError('typed final allocator observation')
    call('optional_Windows_PID_counters',api['Windows_PID_counters'])
    return {'status':'NO_MODEL_CONTEXT_PHASES_COMPLETE_UNQUALIFIED','phase_accounting':recorder.snapshot(),
            'model_loaded':False,'model_constructed':False,'forwards':0,'backwards':0,'optimizer_steps':0,
            'actual_grading':False,'usable_model_fit_established':False,'historical_phase_or_cause_established':False}


def lazy_native_api(torch_holder):
    def torch_import():
        value=importlib.import_module('torch');torch_holder[0]=value;return value
    def symbol(module,name):return lambda:getattr(importlib.import_module(module),name)
    def growth():
        import psutil
        return {'available_bytes':psutil.virtual_memory().available,'worker_RSS_bytes':psutil.Process().memory_info().rss}
    def configure():
        torch=torch_holder[0];torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    return {'torch_import':torch_import,'Qwen3Config_symbol_import':symbol('transformers.models.qwen3.configuration_qwen3','Qwen3Config'),
        'Qwen3ForCausalLM_symbol_import':symbol('transformers.models.qwen3.modeling_qwen3','Qwen3ForCausalLM'),
        'framework_version_read':lambda:{'torch':str(torch_holder[0].__version__),'transformers':importlib.metadata.version('transformers')},
        'growth_observation':growth,'configure':configure,
        'Windows_PID_counters':lambda:{'supported':False,'reason':'Optional counter provider not registered in this scope','authoritative':False}}


def run(config_path,output):
    contract.integrated()  # Refuse stage before imports, paths, probes or artifacts.
    config=contract.preflight(config_path);output=Path(output).absolute()
    if output.resolve().parent!=(contract.PACKAGE/'artifacts').resolve() or not output.is_dir() or output.is_symlink():raise ValueError('owned new diagnostic output')
    from native_worker import verify_worker_ownership
    from inference_qual_worker import make_guard
    import worker_ram_probe
    owner=contract.native.read_json(output/'supervisor_ownership.json');verify_worker_ownership(owner,Path(config_path).absolute())
    def preserve(name,value):
        from rs_contract import canonical
        with (output/(name+'.json')).open('x',encoding='utf-8') as stream:
            stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    recorder=Recorder(preserve);holder=[None];start=time.monotonic()
    def view():return None if holder[0] is None else GuardTorchProxy(holder[0],recorder)
    actual_guard=make_guard(output,owner,start,view,available=worker_ram_probe.available_bytes)
    guard=observed_guard(actual_guard,recorder);report={};code=2
    try:
        for key,value in config['offline_environment'].items():os.environ[key]=value
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        report=context_phases(lazy_native_api(holder),recorder,guard,config['runtime_versions']);guard('complete');code=0
    except BaseException as error:
        report={'status':'NO_MODEL_CONTEXT_PHASES_FAILED','error':{'type':type(error).__name__,'message':str(error)}}
    finally:
        report.update({'source_commit':config['source_commit'],'config_sha256':config['config_sha256'],
            'phase_accounting':recorder.snapshot(),'wall_seconds':time.monotonic()-start,'scoring_authorized':False,
            'model_loaded':False,'forward_authorized':False,'native_fit_established':False,'historical_cause_established':False})
        preserve('worker_result',report)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();raise SystemExit(run(args.config,args.output_dir))
