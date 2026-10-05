"""Injected CPU-only one-prefix core; exact short-circuit tensor predicates."""
from pathlib import Path
from copy import deepcopy
import json,os,time
import contract
C=contract
def counts():return {'attempted':0,'returned':0,'failed':0,'pending':0}
class Ledger:
    def __init__(self,output):
        self.output=Path(output);self.ops={};self.markers=0;self.bytes=0;self.serial=0;self.io={};self.closed=False
        self.stream=self.action('ledger_create',lambda:(self.output/'PHASE_OBSERVATIONS.jsonl').open('xb'))
    def action(self,name,fn):
        item=self.io.setdefault(name,counts());item['attempted']+=1;item['pending']+=1
        try:v=fn()
        except BaseException:item['pending']-=1;item['failed']+=1;raise
        item['pending']-=1;item['returned']+=1;return v
    def marker(self,name,event,ordinal):
        raw=(C.canonical({'operation':name,'event':event,'ordinal':ordinal})+'\n').encode('ascii')
        def write():
            C.require(not self.closed and self.markers<C.LIMITS['maximum_markers'] and len(raw)<=C.LIMITS['maximum_marker_bytes']
                and self.bytes+len(raw)<=C.LIMITS['maximum_ledger_bytes'],'bounded phase writer')
            C.require(self.stream.write(raw)==len(raw),'complete marker write');self.stream.flush();os.fsync(self.stream.fileno());self.markers+=1;self.bytes+=len(raw)
        self.action('marker_write',write)
    def call(self,name,fn):
        item=self.ops.setdefault(name,counts());item['attempted']+=1;item['pending']+=1;self.serial+=1;oid=self.serial
        try:self.marker(name,'attempt',oid);v=fn()
        except BaseException:item['pending']-=1;item['failed']+=1;self.marker(name,'failed',oid);raise
        item['pending']-=1;item['returned']+=1;self.marker(name,'returned',oid);return v
    def publish(self,name,value):
        def write():
            raw=(C.canonical(value)+'\n').encode('ascii');C.require(len(raw)<=C.LIMITS['maximum_worker_json_bytes'],'bounded report JSON')
            with (self.output/(name+'.json')).open('xb') as stream:C.require(stream.write(raw)==len(raw),'complete report write');stream.flush();os.fsync(stream.fileno())
        return self.call('publication_'+name,write)
    def close(self):
        if not self.closed:self.action('ledger_close',self.stream.close);self.closed=True
def run_native(*args,**kwargs):return C.native_entry(*args,**kwargs)
def dispatch(*args,**kwargs):return C.native_entry(*args,**kwargs)
def run_cpu(api,guard,output,*,fixture_identity=C.CPU_SCOPE,source_pins=None,runtime_pins=None):
    C.require(fixture_identity==C.CPU_SCOPE,'explicit CPU injected identity; no native authority conversion')
    prefix=C.load_input()['prefix_ids'];ledger=Ledger(output);torch=ids=mask=None;primary=secondary=None;report={};initial=None
    def call(name,fn):return ledger.call(name,fn)
    def check(name,fn):
        result=call(name,fn);C.require(type(result)is bool and result,'exact source predicate '+name)
    def gated(name):return call('guard_'+name,lambda:guard(name))
    try:
        initial=call('initial_config_preflight',api['config_preflight']);call('worker_ownership_verify',api['ownership_verify']);ledger.publish('ownership_verified',{'scope':C.CPU_SCOPE})
        gated('pre_import');C.require(call('disk_admission',api['disk_free'])>=4*C.GIB,'unchanged4GiB disk admission')
        torch=call('torch_import',api['torch_import']);gated('after_import');C.require(call('torch_version_read',api['torch_version_read'])==api['runtime_version'],'registered Torch version')
        growth=call('post_import_growth_observation',api['growth_observation']);C.require(set(growth)=={'available_bytes','worker_RSS_bytes'}
            and all(type(v)is int and v>=0 for v in growth.values()) and growth['worker_RSS_bytes']<=2*C.GIB
            and growth['available_bytes']>=1.25*C.GIB+max(0,2*C.GIB-growth['worker_RSS_bytes']),'unchanged full post-import growth reserve')
        gated('after_growth');gated('before_context_init');call('cuda_context_init',torch.cuda.init);gated('after_context_init')
        mem=call('pre_input_direct_CUDA_mem_get_info',torch.cuda.mem_get_info);C.require(type(mem)is tuple and len(mem)==2 and all(type(v)is int for v in mem) and 4*C.GIB<=mem[0]<=mem[1],'unchanged4GiB preload GPU admission')
        call('original_precision_thread_configuration',api['configure']);call('Python_seed_101',lambda:api['python_seed'](101));call('Torch_CPU_seed_101',lambda:torch.manual_seed(101));call('Torch_CUDA_seed_all_101',lambda:torch.cuda.manual_seed_all(101));call('CUDA_reset_peak_memory_stats',torch.cuda.reset_peak_memory_stats)
        gated('before_input_construction');ids=call('input_tensor_create',lambda:torch.tensor([list(prefix)],dtype=torch.long,device='cuda:0'));mask=call('mask_ones_like_create',lambda:torch.ones_like(ids))
        check('input_is_tensor',lambda:torch.is_tensor(ids));check('input_shape_matches',lambda:tuple(ids.shape)==(1,len(prefix)));check('input_dtype_long',lambda:ids.dtype==torch.long);check('input_device_cuda0',lambda:str(ids.device)=='cuda:0');check('input_exact_tolist',lambda:ids.tolist()==[prefix])
        check('mask_is_tensor',lambda:torch.is_tensor(mask));check('mask_shape_matches',lambda:tuple(mask.shape)==tuple(ids.shape));check('mask_dtype_long',lambda:mask.dtype==torch.long);check('mask_device_cuda0',lambda:str(mask.device)=='cuda:0');check('mask_exact_tolist',lambda:mask.tolist()==[[1]*len(prefix)])
        gated('after_input_validation');report['input_summary']={'prefix_ids_sha256':C.digest(prefix),'mask_values_sha256':C.digest([[1]*422]),'shape':[1,422],'dtype':'int64','device':'cuda:0','logical_tensor_bytes':6752,'logical_bytes_not_allocator_reservation':True}
    except BaseException as error:
        primary={'type':type(error).__name__,'message':str(error)}
        try:report['exception_frames']=call('bounded_original_exception_frames_capture',lambda:C.frame_helper().capture_exception_frames(error,source_pins or {},runtime_pins or {}))
        except BaseException:report['exception_frame_diagnostic_failed']={'status':'BOUNDED_FRAME_DIAGNOSTIC_FAILED_PRIMARY_ERROR_PRESERVED'}
    finally:
        for name in ('input_reference_release','mask_reference_release'):
            try:
                if name=='input_reference_release' and ids is not None:call(name,lambda:None);ids=None
                elif name=='mask_reference_release' and mask is not None:call(name,lambda:None);mask=None
            except BaseException as error:secondary={'type':type(error).__name__,'message':str(error)}
        try:
            if initial is not None:C.require(call('final_config_preflight',api['config_preflight'])==initial,'exact final source/config identity')
            if torch is not None:
                initialized=call('terminal_cuda_is_initialized',torch.cuda.is_initialized);C.require(type(initialized)is bool,'typed terminal init')
                terminal={'is_initialized':initialized}
                if initialized:
                    terminal['max_memory_allocated']=call('terminal_cuda_max_memory_allocated',torch.cuda.max_memory_allocated);terminal['max_memory_reserved']=call('terminal_cuda_max_memory_reserved',torch.cuda.max_memory_reserved);terminal['mem_get_info']=call('terminal_cuda_mem_get_info',torch.cuda.mem_get_info)
                report['terminal_allocator']=terminal
            if primary is None and secondary is None:gated('complete')
        except BaseException as error:secondary={'type':type(error).__name__,'message':str(error)}
        report.update(scope=C.CPU_SCOPE,status='CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED' if primary is secondary is None else 'CPU_PREFIX_INPUT_CHECKS_FAILED',error=primary,terminal_error=secondary,logical_tensor_bytes=6752,
            model_loaded=False,model_constructed=False,forwards=0,backwards=0,optimizer_steps=0,decodes=0,grades=0,M0_membership=False,native_launch_authorized=False,scoring_authorized=False,native_fit_established=False,cause_established=False)
        try:ledger.publish('input_check_summary',report.get('input_summary',{'partial_failure':True}));ledger.publish('worker_result',report)
        except BaseException as error:report['status']='CPU_PREFIX_INPUT_CHECKS_FAILED';report['publication_error']={'type':type(error).__name__,'message':str(error)}
        try:ledger.close()
        except BaseException as error:report['status']='CPU_PREFIX_INPUT_CHECKS_FAILED';report['close_error']={'type':type(error).__name__,'message':str(error)}
        report['operation_accounting']=deepcopy(ledger.ops);report['IO_accounting']=deepcopy(ledger.io)
        summary={'scope':C.CPU_SCOPE,'status':report['status'],'operations':report['operation_accounting'],'IO':report['IO_accounting'],'markers':ledger.markers,'marker_bytes':ledger.bytes,'closed':ledger.closed,'native_PASS':False}
        try:
            raw=(C.canonical(summary)+'\n').encode('ascii')
            ledger.action('final_summary_write',lambda:(Path(output)/'PHASE_ACCOUNTING_FINAL.json').write_bytes(raw))
        except BaseException:report['status']='CPU_PREFIX_INPUT_CHECKS_FAILED';report['final_summary_error']=True
    return report
