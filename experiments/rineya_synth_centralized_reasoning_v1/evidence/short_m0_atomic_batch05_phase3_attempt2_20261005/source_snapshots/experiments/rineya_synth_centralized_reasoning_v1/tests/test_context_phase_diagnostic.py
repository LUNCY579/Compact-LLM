"""Synthetic phase/guard controls only; never Torch/NVML/Win32/model probes."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='context_phase_diagnostic_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(CODE));sys.path.append(str(PACKAGE))
def load(name,path,register=True):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    if register:sys.modules[name]=module
    spec.loader.exec_module(module);return module
contract=load('context_phase_contract',CODE/'context_phase_contract.py')
worker=load('context_phase_worker',CODE/'context_phase_worker.py')
assert Path(contract.__file__).resolve().parent==CODE.resolve()
assert Path(worker.__file__).resolve().parent==CODE.resolve()
from inference_qual_worker import make_guard
GIB=1024**3


class CUDA:
    def __init__(self):self.calls=[];self.initialized=False;self.free=5*GIB;self.error=None
    def is_initialized(self):self.calls.append('is_initialized');return self.initialized
    def init(self):self.calls.append('init');self.initialized=True
    def mem_get_info(self):
        self.calls.append('mem_get_info')
        if self.error:raise self.error
        return (self.free,8*GIB)
    def memory_allocated(self):self.calls.append('allocated');return 0
    def memory_reserved(self):self.calls.append('reserved');return 0
    def max_memory_allocated(self):self.calls.append('max_allocated');return 0
    def max_memory_reserved(self):self.calls.append('max_reserved');return 0
    def manual_seed_all(self,value):self.calls.append(('CUDA_seed',value));return None
    def reset_peak_memory_stats(self):self.calls.append('reset_peaks')


def fixture():
    cuda=CUDA();torch=SimpleNamespace(cuda=cuda,manual_seed=lambda value:None)
    saved={}
    def preserve(name,value):
        if name in saved:raise ValueError('writeonce phase name')
        saved[name]=deepcopy(value)
    rec=worker.Recorder(preserve,clock=lambda:1.)
    proxy=worker.GuardTorchProxy(torch,rec)
    actual=make_guard(HERE,{'job_name':'CPU_FAKE_JOB','supervisor_pid':7},0.,lambda:proxy,
        owned_resources=lambda name:{'tree_rss_bytes':GIB},available=lambda:4*GIB,
        monotonic=lambda:1.,wall=lambda:1.,heartbeat_reader=lambda path:{'supervisor_pid':7,'unix_seconds':1.})
    guard=worker.observed_guard(actual,rec)
    api={'torch_import':lambda:torch,'Qwen3Config_symbol_import':lambda:'CPU_CLASS_SYMBOL_NOT_INSTANTIATED',
         'Qwen3ForCausalLM_symbol_import':lambda:'CPU_CLASS_SYMBOL_NOT_INSTANTIATED',
         'framework_version_read':lambda:{k:contract.native.RUNTIME[k] for k in ('torch','transformers')},
         'growth_observation':lambda:{'available_bytes':4*GIB,'worker_RSS_bytes':GIB},
         'configure':lambda:None,'Windows_PID_counters':lambda:{'supported':False,'reason':'CPU fixture unsupported','authoritative':False}}
    return cuda,torch,rec,saved,guard,api


def manifest():
    value={'contract':contract.VERSION,'id':'CPU_FIXTURE_SCOPE','authorization':deepcopy(contract.AUTHORIZATION),
        'resources':deepcopy(contract.native.RESOURCE_BOUNDS),'tolerances':deepcopy(contract.native.TOLERANCES),
        'runtime_versions':deepcopy(contract.native.RUNTIME),'offline_environment':deepcopy(contract.native.OFFLINE_ENV),
        'source_pins':{key:'a'*64 for key in contract.source_roster()},'runtime_pins':{'CPU_RUNTIME_%02d'%i:'b'*64 for i in range(65)},
        'helper_pins':{name:'a'*64 for name in contract.HELPERS},'diagnostic_profile':{'id':contract.VERSION,'no_model':True,
        'no_tensor_allocation':True,'guard_query_proxy':'one exact actual call; nonrecursive phase accounting; unchanged result/error',
        'external_NVML_authority_unchanged':True,'Windows_counters':'optional diagnostic only; unsupported explicit',
        'native_fit_or_historical_cause_established':False}}
    value['sha256']=contract.digest(value);return value


class Controls(unittest.TestCase):
    def test_closed_complete_manifest_and_own_transitive_source_omissions(self):
        contract.validate_manifest(manifest())
        for name in ('context_phase_worker.py','native_contract.py','native_loader.py','inference_qual_worker.py','rs_contract.py'):
            body=manifest();body['source_pins'].pop(str((PACKAGE/name).resolve()))
            body['sha256']=contract.digest({k:v for k,v in body.items() if k!='sha256'})
            with self.assertRaisesRegex(ValueError,'source roster'):contract.validate_manifest(body)
    def test_authority_model_field_ENV_runtime_and_floor_expansions_rejected(self):
        for mutate in (lambda b:b.update(model_dir='PRIVATE'),lambda b:b['authorization'].update(forward=True),
                       lambda b:b['resources'].update(GPU_free_during_work_GiB=.1),
                       lambda b:b['offline_environment'].update(HF_HUB_OFFLINE='0'),lambda b:b['runtime_pins'].pop('CPU_RUNTIME_00')):
            body=manifest();mutate(body);body['sha256']=contract.digest({k:v for k,v in body.items() if k!='sha256'})
            with self.assertRaises(ValueError):contract.validate_manifest(body)
    def test_guard_false_bypasses_memory_true_calls_once_no_recursion(self):
        cuda,torch,rec,saved,guard,api=fixture();guard('false')
        self.assertEqual(cuda.calls,['is_initialized']);self.assertNotIn('guard_cuda_mem_get_info',rec.operations)
        cuda.initialized=True;guard('true')
        self.assertEqual(cuda.calls,['is_initialized','is_initialized','mem_get_info'])
        self.assertEqual(rec.operations['guard_cuda_mem_get_info'],{'attempted':1,'returned':1,'failed':0,'pending':0})
    def test_proxy_exact_object_result_and_exception_with_no_extra_query(self):
        result=object();error=OSError('CPU_EXACT_QUERY_ERROR');calls=[]
        actual=SimpleNamespace(is_initialized=lambda:calls.append('init') or result,
            mem_get_info=lambda:(_ for _ in ()).throw(error))
        rec=worker.Recorder(lambda *args:None,clock=lambda:1.);proxy=worker.GuardCUDAProxy(actual,rec)
        self.assertIs(proxy.is_initialized(),result)
        with self.assertRaises(OSError) as got:proxy.mem_get_info()
        self.assertIs(got.exception,error);self.assertEqual(calls,['init'])
        self.assertEqual(rec.operations['guard_cuda_mem_get_info']['failed'],1)
    def test_same_half_GiB_guard_and_four_GiB_preload_gates(self):
        cuda,torch,rec,saved,guard,api=fixture();cuda.initialized=True;cuda.free=int(.5*GIB)-1
        with self.assertRaisesRegex(ValueError,'GPU free floor'):guard('below_half')
        cuda,torch,rec,saved,guard,api=fixture();cuda.free=3*GIB
        with self.assertRaisesRegex(ValueError,'4GiB'):worker.context_phases(api,rec,guard,contract.native.RUNTIME)
        self.assertIn('cuda_context_init',rec.operations);self.assertNotIn('precision_and_thread_configuration',rec.operations)
    def test_original_RSS_RAM_deadline_and_heartbeat_guards_unchanged(self):
        for rss,available,now,heartbeat,pattern in ((2*GIB+1,4*GIB,1.,1.,'resource guard'),
              (GIB,int(1.25*GIB)-1,1.,1.,'resource guard'),(GIB,4*GIB,601.,1.,'deadline'),
              (GIB,4*GIB,1.,-2.,'heartbeat')):
            actual=make_guard(HERE,{'job_name':'CPU_FAKE','supervisor_pid':7},0.,lambda:None,
                owned_resources=lambda name,rss=rss:{'tree_rss_bytes':rss},available=lambda available=available:available,
                monotonic=lambda now=now:now,wall=lambda:1.,heartbeat_reader=lambda path,h=heartbeat:{'supervisor_pid':7,'unix_seconds':h})
            rec=worker.Recorder(lambda *args:None)
            with self.assertRaisesRegex(ValueError,pattern):worker.observed_guard(actual,rec)('fake')
    def test_poisoned_growth_version_and_allocator_observations_fail(self):
        for bad in ('growth','version','allocated'):
            cuda,torch,rec,saved,guard,api=fixture()
            if bad=='growth':api['growth_observation']=lambda:{'available_bytes':True,'worker_RSS_bytes':GIB}
            elif bad=='version':api['framework_version_read']=lambda:{'torch':'WRONG','transformers':'WRONG'}
            else:cuda.memory_allocated=lambda:float('nan')
            with self.assertRaises(ValueError):worker.context_phases(api,rec,guard,contract.native.RUNTIME)
    def test_full_fake_phases_complete_all_calls_recorded_no_model(self):
        cuda,torch,rec,saved,guard,api=fixture();result=worker.context_phases(api,rec,guard,contract.native.RUNTIME)
        self.assertEqual(result['forwards'],0);self.assertFalse(result['model_constructed'])
        self.assertFalse(result['historical_phase_or_cause_established'])
        self.assertTrue(all(v['pending']==v['failed']==0 and v['attempted']==v['returned'] for v in rec.operations.values()))
        query_names={'guard_cuda_mem_get_info','direct_CUDA_preload_mem_get_info','final_direct_CUDA_mem_get_info'}
        self.assertEqual(sum(v['returned'] for k,v in rec.operations.items() if k in query_names),cuda.calls.count('mem_get_info'))
        self.assertEqual(rec.operations['guard_cuda_is_initialized']['returned'],cuda.calls.count('is_initialized'))
        self.assertTrue(all('model' not in str(call).lower() for call in cuda.calls))
    def test_post_init_guard_failure_preserves_return_then_query_failure(self):
        cuda,torch,rec,saved,guard,api=fixture();original=cuda.init
        def init():original();cuda.error=OSError('CPU_POST_INIT_GUARD_QUERY')
        cuda.init=init
        with self.assertRaisesRegex(OSError,'POST_INIT'):worker.context_phases(api,rec,guard,contract.native.RUNTIME)
        self.assertEqual(rec.operations['cuda_context_init']['returned'],1)
        self.assertEqual(rec.operations['guard_cuda_mem_get_info']['failed'],1)
        self.assertNotIn('direct_CUDA_preload_mem_get_info',rec.operations)
    def test_sink_error_retains_exact_proxy_result_then_fail_closed(self):
        result=(5*GIB,8*GIB);calls=[]
        rec=worker.Recorder(lambda *args:(_ for _ in ()).throw(OSError('CPU_SINK_ERROR')))
        proxy=worker.GuardCUDAProxy(SimpleNamespace(mem_get_info=lambda:calls.append(1) or result),rec)
        self.assertIs(proxy.mem_get_info(),result);self.assertEqual(calls,[1]);self.assertEqual(rec.operations['guard_cuda_mem_get_info']['returned'],1)
        with self.assertRaisesRegex(RuntimeError,'publication failed'):rec.healthy()
    def test_failed_preservation_cannot_mask_actual_guard_query_exception(self):
        cuda,torch,rec,saved,guard,api=fixture();cuda.initialized=True
        error=OSError('CPU_ORIGINAL_GUARD_QUERY');cuda.error=error
        rec.preserve=lambda *args:(_ for _ in ()).throw(OSError('CPU_PUBLICATION_FAILURE'))
        with self.assertRaises(OSError) as observed:guard('original_error')
        self.assertIs(observed.exception,error);self.assertEqual(cuda.calls,['is_initialized','mem_get_info'])
        self.assertTrue(rec.errors);self.assertEqual(rec.operations['guard_cuda_mem_get_info']['failed'],1)
    def test_staged_worker_and_supervisor_refuse_before_any_import_probe_artifact(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            stage=Path(directory)/'package/artifacts/context_phase_diagnostic_work';stage.mkdir(parents=True)
            for name in contract.SOURCE_NAMES:(stage/name).write_bytes((CODE/name).read_bytes())
            stage_contract=load('explicit_stage_diagnostic_contract',stage/'context_phase_contract.py',False)
            with patch.dict(sys.modules,{'context_phase_contract':stage_contract}):
                for name,function in (('context_phase_worker','run'),('context_phase_supervisor','dispatch')):
                    module=load('explicit_'+name,stage/(name+'.py'),False)
                    self.assertIs(module.contract,stage_contract)
                    with self.assertRaisesRegex(ValueError,'staged NO_MODEL'):getattr(module,function)('UNUSED','UNUSED')
        # A preceding CPU tokenizer suite may have loaded these packages.
        code="import sys;sys.path.insert(0,sys.argv[1]);import context_phase_worker;assert not any(n in sys.modules for n in ('torch','transformers','tokenizers'))"
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(PACKAGE)],capture_output=True,text=True,timeout=30)
        self.assertEqual(0,result.returncode,result.stderr)


if __name__=='__main__':unittest.main()
