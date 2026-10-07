"""Concrete lazy decoded M0 long-guard worker; staged native entry is closed.

No import loads frameworks, assets, tokenizer, native process APIs or task data.
Source-only CPU composition never grants decoded native qualification.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
import ast, hashlib, importlib, importlib.metadata, json, math, os, random, re, sys, time

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == 'rineya_synth_centralized_reasoning_v1')
EXPECTED = PACKAGE / 'long_guard_owned_runtime_v1'
CORE = PACKAGE / 'long_guard_collection_v1/long_guard_collection.py'
QUALIFIER = PACKAGE / 'long_guard_native_qualification_v1/decoded_long_audit.py'
SCOPE = 'REGISTERED_LONG2304_M0_DECODED_RAW_COLLECTION_V1'
GIB = 1024**3
def require(ok, text):
    if not ok: raise ValueError(text)
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
def digest(v): return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def filehash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _pairs(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'duplicate owned JSON key');result[key]=value
    return result
def strict_json(raw):
    return json.loads(raw,object_pairs_hook=_pairs,parse_constant=lambda value:require(False,'nonfinite owned JSON'))
def source_module(path):
    module = ModuleType('bound_' + Path(path).stem); module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__); return module
def _capability_factory():
    token = object()
    return lambda: token, lambda v: require(v is token, 'private authored CPU owned runtime capability')
_CPU_capability, _CPU_check = _capability_factory()

def worker_helpers(*, cpu=None):
    """Compile original Journal/ownership/guard/terminal functions unchanged."""
    worker = ast.parse((PACKAGE/'native_worker.py').read_bytes())
    guards = ast.parse((PACKAGE/'inference_qual_worker.py').read_bytes())
    names = ('diagnostic_json', '_job_query_api', '_job_query_error', 'owned_job_pids',
        'owned_job_resources', 'in_owned_job', 'verify_worker_ownership', 'read_heartbeat', 'Journal', 'preserve_terminal')
    definitions = [deepcopy(next(n for n in worker.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)) for name in names]
    definitions.append(deepcopy(next(n for n in guards.body if isinstance(n, ast.FunctionDef) and n.name == 'make_guard')))
    contract = SimpleNamespace(filehash=filehash, _pairs=_pairs, _bad_constant=lambda value: require(False, 'nonfinite JSON'))
    ns = dict(Path=Path, defaultdict=defaultdict, datetime=datetime, timezone=timezone,
        json=json, math=math, os=os, re=re, time=time, MAX_OWNED_JOB_PIDS=256,
        GIB=GIB, canonical=canonical, require=require, contract=contract)
    if cpu is None: ns['ctypes'] = importlib.import_module('ctypes')
    else:
        ns.update(os=cpu.os, time=cpu.time, _CPU_psutil=cpu.psutil)
        ownership = definitions[6]; changed = 0
        for node in ast.walk(ownership):
            if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == 'psutil':
                node.__class__ = ast.Assign; node.__dict__.clear(); node.__dict__.update(targets=[ast.Name(id='psutil',ctx=ast.Store())], value=ast.Name(id='_CPU_psutil',ctx=ast.Load())); changed += 1
        require(changed == 1, 'original ownership psutil seam count')
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[])), str(PACKAGE/'native_worker.py'), 'exec'), ns)
    return ns

class ForwardDose:
    """Physical outer attempts include both419-token parity forwards."""
    def __init__(self, torch, batch):
        self.torch, self.batch = torch, deepcopy(batch); self.current = None
        self.calls = self.positions = 0; self.operations = {}
        require(type(batch['context_cap']) is int and batch['context_cap'] in (1024,2048,4096)
            and batch['maximum_attempted_forwards'] == 2048
            and batch['maximum_input_positions'] == 64*(32*(batch['context_cap']-32)+496), 'exact long64 frozen dose')
    def before(self, model, args, kwargs):
        ids = kwargs.get('input_ids'); require(self.torch.is_tensor(ids) and len(ids.shape)==2
            and tuple(ids.shape[:1])==(1,) and ids.dtype==self.torch.long, 'physical long single typed input')
        length = int(ids.shape[1]); require(self.current in ('zero_parity_base_forward','zero_parity_adapter_forward','inference_forward'), 'registered physical forward operation')
        require((length==419 if self.current.startswith('zero_parity_') else 0<length<=self.batch['context_cap']), 'physical same-cap inference or short419 parity')
        require(self.calls<2050 and self.positions+length<=838+self.batch['maximum_input_positions'], 'physical2050/838-plus-capture position cap exhausted')
        self.calls += 1; self.positions += length
        counts = self.operations.setdefault(self.current, {'attempted_calls':0,'input_positions':0})
        counts['attempted_calls'] += 1; counts['input_positions'] += length
    def wrap(self, call):
        def counted(operation, function):
            if operation not in ('zero_parity_base_forward','zero_parity_adapter_forward','inference_forward'):
                return call(operation,function)
            require(self.current is None, 'nested physical forward boundary'); self.current = operation
            try: return call(operation,function)
            finally: self.current = None
        return counted
    def snapshot(self):
        return deepcopy(dict(attempted_calls=self.calls,input_positions=self.positions,operations=self.operations,
            maximum_forward_calls=2050,maximum_forward_input_positions=838+self.batch['maximum_input_positions']))

class LocalGeneratedDecoder:
    """Exact byte-bound original generated-only decoder, with no fallback."""
    def __init__(self, config):
        self.config = config; self.verify_binding(); os.environ['TOKENIZERS_PARALLELISM']='false'
        from tokenizers import Tokenizer
        self.engine = Tokenizer.from_file(str(Path(config['model_dir'])/'tokenizer.json'))
        require(self.engine.token_to_id('<|im_end|>')==151645 and self.engine.token_to_id('<|endoftext|>')==151643, 'exact decoder EOS identities')
    def verify_binding(self):
        require(importlib.metadata.version('tokenizers')==self.config['runtime_versions']['tokenizers']=='0.22.2', 'actual bound generated decoder runtime')
        for name in ('tokenizer.json','tokenizer_config.json'):
            require(filehash(Path(self.config['model_dir'])/name)==self.config['model_pins'][name], 'decoder asset drift')
        for filename,pin in self.config['decoder_runtime_pins'].items(): require(filehash(filename)==pin, 'decoder source/binary drift')
    def decode(self, ids, *, skip_special_tokens):
        require(skip_special_tokens is True and type(ids) is list and len(ids)<=32
            and all(type(x) is int and 0<=x<151936 for x in ids), 'bounded generated-only decoder input')
        return self.engine.decode(list(ids),skip_special_tokens=True)

def loaded_body(config, source, plan, ordinal, core, component, torch, model, decoder,
                source_hashes, journal, guard, *, helpers, device, report=None):
    """Original scientific calls; only authored CPU wrapper can inject helpers."""
    require(device in ('cpu','cuda:0'), 'explicit owned native or fixture device')
    require(type(ordinal) is int and 1<=ordinal<=36, 'typed exact long64 batch ordinal')
    core.check_plan(source,plan,component.validate_row); batch=plan['batches'][ordinal-1]
    require(config['dose']=={'maximum_forward_calls':2050,'maximum_forward_input_positions':838+batch['maximum_input_positions'],
        'context_cap':batch['context_cap'],'maximum_generated_tokens':32,'rows':64,'optimizer_steps':0,'backwards':0}, 'exact prospectively bound decoded long64 dose')
    require(not any(n.endswith(('.lora_A','.lora_B')) for n,_ in model.named_parameters()), 'fresh dense M0 without checkpoint adapters')
    parity=config['short_parity_row'];component.validate_row(parity,context_cap=512,generation_cap=32)
    require(parity['prompt_tokens']==419, 'qualified short419 parity row')
    dose=ForwardDose(torch,batch);call=dose.wrap(journal.call);report={} if report is None else report;handles=[]
    def remove_all():
        errors=[]
        for handle in handles:
            try: handle.remove()
            except BaseException as exc: errors.append({'type':type(exc).__name__,'message':str(exc)})
        report['cleanup_errors']=errors
        require(not errors, 'owned long hooks removal failed')
    try:
        report['compute']=call('install_fp32_compute',lambda:helpers.compute(model,torch))
        def install_active():
            value=helpers.activation_hooks(model,torch);handles.extend(value['handles']);return value
        active=call('install_activation_hooks',install_active)
        def install_meter():handles.append(model.register_forward_pre_hook(dose.before,with_kwargs=True))
        call('install_physical_forward_meter',install_meter)
        ids=torch.tensor([parity['prompt_ids']],dtype=torch.long,device=device)
        def logits(value):
            require(torch.is_tensor(value) and tuple(value.shape)==(1,419,151936) and value.dtype==torch.float32
                and str(value.device)==device and not value.requires_grad and bool(torch.isfinite(value).all()), 'full finite FP32 parity logits')
            return value
        with torch.no_grad(): dense=logits(call('zero_parity_base_forward',lambda:model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).logits))
        report['adapters']=call('install_qv_lora',lambda:helpers.adapters(model,torch,rank=8,alpha=16))
        report['zero_before']=call('zero_lora_query_before',lambda:helpers.zero(model,torch));require(report['zero_before']['all_B_zero'] is True, 'fresh zero B')
        with torch.no_grad(): adapted=logits(call('zero_parity_adapter_forward',lambda:model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).logits))
        require(bool(torch.equal(dense,adapted)), 'exact physical dense/zeroB parity');del dense,adapted,ids
        journal.evidence('long_zero_parity',{'equal':True,'maximum_error':0.,'fixture_only':device=='cpu','native_qualification_established':False})
        class JournaledDecoder:
            def decode(self, ids, *, skip_special_tokens):
                return call('generated_only_decode',lambda:decoder.decode(ids,skip_special_tokens=skip_special_tokens))
        original_call=journal.call;journal.call=call
        try:report['batch']=core.collect_loaded(source,plan,ordinal,torch=torch,model=model,decoder=JournaledDecoder(),journal=journal,guard=guard,device=device,component=component)
        finally:journal.call=original_call
        costs=report['batch']['collection']['costs']
        require(dose.calls==2+costs['attempted_forwards'] and dose.positions==838+costs['input_positions']['attempted']
            and dose.operations.get('zero_parity_base_forward')=={'attempted_calls':1,'input_positions':419}
            and dose.operations.get('zero_parity_adapter_forward')=={'attempted_calls':1,'input_positions':419}, 'physical terminal parity/capture dose exact join')
        report['zero_after']=call('zero_lora_query_after',lambda:helpers.zero(model,torch));require(report['zero_after']['all_B_zero'] is True, 'zero B changed')
        report['base_postcheck']=call('verify_frozen_base',lambda:helpers.base_verify(torch,model,source_hashes,call,guard))
        require(report['base_postcheck']['verified_source_tensors']==310 and report['base_postcheck']['frozen_base_unchanged'] is True, 'all310 frozen native base tensors unchanged')
        require(active['observations'] and all(n>0 for n in active['observations'].values()), 'all activation sites observed')
        report['activation_observations']=deepcopy(active['observations']);call('generated_decoder_postcheck',decoder.verify_binding)
    finally:
        report['forward_dose']=dose.snapshot()
        primary=sys.exc_info()[0]
        try:call('owned_hooks_remove',remove_all)
        except BaseException:
            # Guard refusal cannot leave resident hooks behind. Retain first
            # cause and run unguarded local cleanup only for these owned handles.
            failure=sys.exc_info()[1];report['cleanup_boundary_error']={'type':type(failure).__name__,'message':str(failure)}
            if 'cleanup_errors' not in report:
                try:remove_all()
                except BaseException:pass
            if primary is None:raise
    return report

def _CPU_loaded(*args, capability, **kwargs):
    _CPU_check(capability);require(kwargs.get('device')=='cpu' and getattr(kwargs.get('torch'),'_synth_fake_component',False) is True, 'inert authored CPU actors only')
    return loaded_body(*args,**kwargs)

def preflight(launch_ref):
    require(HERE==EXPECTED, 'staged decoded long owned runtime has no native authority')
    # A metadata flag cannot supply an absent decoded numerical/ownership audit.
    require(QUALIFIER.is_file(), 'independent decoded long native qualification auditor absent')
    raise ValueError('decoded long qualifier dispatch/owner-approved integration has not been reviewed; native entry closed')

def native_helpers():
    sys.path.insert(0,str(PACKAGE))
    from native_loader import load_titan,verify_frozen_base
    from native_precision import install_fp32_compute,install_qv_lora,install_fp32_activation_assertions,zero_lora_status
    return SimpleNamespace(load=load_titan,base_verify=verify_frozen_base,compute=install_fp32_compute,
        adapters=install_qv_lora,activation_hooks=install_fp32_activation_assertions,zero=zero_lora_status)

def worker_body(config_path, output, config, source, plan, core, component, *, cpu=None, capability=None):
    """Concrete production lifecycle; no public injected native supplier."""
    if cpu is None:preflight(config_path)
    else:_CPU_check(capability);require(getattr(cpu.torch,'_synth_fake_component',False) is True,'inert CPU worker actors only')
    helpers=worker_helpers(cpu=cpu);journal=helpers['Journal'](Path(output));torch=model=None;start=time.monotonic();report={};status='LONG_DECODED_OWNED_WORKER_FAILED';error=None
    try:
        if cpu is None:
            owner=strict_json((Path(output)/'supervisor_ownership.json').read_bytes())
            ownership=helpers['verify_worker_ownership'](owner,config_path)
            journal.evidence('worker_ownership_verified',ownership)
            probe=source_module(PACKAGE/'worker_ram_probe.py')
            guard=helpers['make_guard'](output,owner,start,lambda:torch,available=probe.available_bytes)
        else:owner=cpu.owner;guard=cpu.guard
        journal.guard=guard
        for key,value in config['offline_environment'].items():os.environ[key]=value
        os.environ['CUDA_VISIBLE_DEVICES']='0';guard('pre_import')
        require((cpu.disk_free if cpu else __import__('shutil').disk_usage(output).free)>=4*GIB,'original disk floor')
        if cpu is None:
            import torch as imported_torch
            from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
            from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
            torch=imported_torch
            require(str(torch.__version__)==config['runtime_versions']['torch'] and importlib.metadata.version('transformers')==config['runtime_versions']['transformers'],'actual framework versions')
            import psutil
            rss=helpers['owned_job_resources'](owner['job_name'])['tree_rss_bytes']
            require(psutil.virtual_memory().available>=1.25*GIB+max(0,2*GIB-rss),'post-import whole growth reserve')
            journal.call('cuda_context_init',torch.cuda.init);require(torch.cuda.mem_get_info()[0]>=4*GIB,'preload GPU floor')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
            random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101);torch.cuda.reset_peak_memory_stats()
            scientific=native_helpers();model,loader,hashes=scientific.load(torch,Qwen3Config,Qwen3ForCausalLM,config['model_dir'],journal.call,guard)
            journal.evidence('loader',loader);decoder=journal.call('generated_decoder_construct',lambda:LocalGeneratedDecoder(config));device='cuda:0'
        else:torch,model,decoder,scientific,hashes=cpu.torch,cpu.model,cpu.decoder,cpu.scientific,{};device='cpu'
        loaded_body(config,source,plan,config['ordinal'],core,component,torch,model,decoder,hashes,journal,guard,helpers=scientific,device=device,report=report)
        guard('completed');status='COMPLETE_PENDING_INDEPENDENT_LONG_DECODED_AUDIT'
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)};journal.emit({'event':'worker_failed','error':error})
    finally:
        report.update(status=status,error=error,scope=SCOPE,fixture_only=cpu is not None,native_PASS=False,
            scoring_authorized=False,retention_certified=False,native_qualification_established=False,
            optimizer_steps=0,backwards=0,operation_accounting=journal.snapshot(),requires_independent_postJob_audit=True)
        report['sha256']=digest(report)
        helpers['preserve_terminal'](Path(output),report,journal,torch)
    return report

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();preflight(args.config)
    raise ValueError('native long worker integration remains closed')
if __name__=='__main__':raise SystemExit(main())
