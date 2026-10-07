"""Authored sparse actors exercise concrete owned decoded lifecycle only."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import ast, builtins, hashlib, io, json, math, os, re, sys, tempfile, time, unittest
HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
CORE=PACKAGE/'evidence/long_guard_collection_reviewed_cpu_20261007/current001'
def source(path,name):
    m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
class OwnedControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reads=[];cls.denials=[];cls.imports=[];realio,realopen,realimport=io.open,builtins.open,builtins.__import__
        def read(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,bytes,Path)) and not any(c in str(mode) for c in ('w','a','x','+')):
                p=Path(path).resolve();cls.reads.append(str(p))
                if not (p.is_relative_to(HERE) or p.is_relative_to(CORE) and p.suffix=='.py' or p.parent==PACKAGE and p.suffix=='.py' or p.parent==PACKAGE/'tests' and p.suffix=='.py'):
                    cls.denials.append(str(p));raise AssertionError('actual model/data/qualification/API input forbidden')
            return realio(path,mode,*args,**kwargs)
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','transformers','tokenizers','ctypes','psutil','safetensors','numpy'}:
                cls.imports.append(name);raise AssertionError('native/framework import forbidden')
            return realimport(name,*args,**kwargs)
        cls.traps=[patch('io.open',side_effect=read),patch.object(builtins,'open',side_effect=lambda p,*a,**k:read(p,*a,**k) if isinstance(p,(str,bytes,Path)) else realopen(p,*a,**k)),patch.object(builtins,'__import__',side_effect=imports)]
        for trap in cls.traps:trap.start()
        cls.R=source(HERE/'long_guard_owned_runtime.py','CPU_owned_runtime');cls.D=source(HERE/'long_guard_owned_dispatch.py','CPU_owned_dispatch')
        cls.L=source(CORE/'long_guard_collection.py','CPU_long_core');cls.component=cls.L._helpers()
        tree=ast.parse((PACKAGE/'tests/test_inference_component.py').read_bytes());ns={'deepcopy':deepcopy,'math':math,'SimpleNamespace':SimpleNamespace,'VOCAB':151936}
        names={'Device','Tensor','Array','Parameter','Scalar','SparseRow','Logits','FakeTorch','Model','Decoder'}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names],type_ignores=[]),'original_sparse_actors','exec'),ns)
        class LazyLogits(ns['Tensor']):
            def __init__(self,length,token):super().__init__(shape=(1,length,151936),dtype='float32');self.token=token
            def __getitem__(self,key):return ns['SparseRow'](self.token)
            def finite(self):return True
        class Torch(ns['FakeTorch']):
            cuda=SimpleNamespace(is_initialized=lambda:False)
            def equal(self,a,b):return a.shape==b.shape and a.token==b.token
        class Model(ns['Model']):
            def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.hooks=[];self.cleanup_fail=False;self.parity_bad=False
            def register_forward_pre_hook(self,function,*,with_kwargs):
                assert with_kwargs is True;self.hooks.append(function);model=self
                class Handle:
                    def remove(self):
                        model.hooks.remove(function)
                        if model.cleanup_fail:raise RuntimeError('authored hook cleanup failure')
                return Handle()
            def __call__(self,**kwargs):
                assert self.torch.no_grad_depth>0;ids=kwargs['input_ids'].tolist()[0]
                for hook in list(self.hooks):hook(self,(),kwargs)
                assert kwargs['attention_mask'].tolist()[0]==[1]*len(ids)
                self.calls.append(list(ids));token=self.tokens[min(max(0,len(ids)-self.prompt_length),len(self.tokens)-1)]
                if self.parity_bad and len(self.calls)==2:token+=1
                if self.fault:self.fault(self,None)
                return SimpleNamespace(logits=LazyLogits(len(ids),token))
        class Decoder(ns['Decoder']):
            def __init__(self):super().__init__();self.checks=0
            def verify_binding(self):self.checks+=1
        cls.F=ns;cls.Torch=Torch;cls.Model=Model;cls.Decoder=Decoder
        cpu=SimpleNamespace(os=os,time=time,psutil=SimpleNamespace())
        cls.H=cls.R.worker_helpers(cpu=cpu);cls.cap=cls.R._CPU_capability()
    @classmethod
    def tearDownClass(cls):
        for trap in reversed(cls.traps):trap.stop()
        assert not cls.imports,cls.imports
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE,prefix='CPU_OWNED_');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
    def make_source(self):
        rows=[];metadata=[]
        for family in self.L.FAMILIES:
            for placement in self.L.PLACEMENTS:
                for missing in (0,1,2):
                    for w in range(8):
                        world=self.L.digest(['CPU_WORLD',family,placement,missing,w])
                        for length,cap in self.L.ENVELOPES.items():
                            for variant,flags in self.L.VARIANTS.items():
                                rid=self.L.digest([world,length,variant]);rows.append(dict(row_id=rid,prompt_sha256=self.L.digest(['CPU_PROMPT',rid]),prompt_ids=[5]*length,prompt_tokens=length,attention_mask=[1]*length))
                                metadata.append(dict(row_id=rid,component_id=world,family=family,placement=placement,missing_index=missing,total_sequence_envelope=cap,variant=variant,answerable=flags[0],adversarial=flags[1]))
        return dict(rows=rows,metadata=metadata,declared_domains=[dict(domain_id='CPU_AUTHORED_LONG_DOMAIN',**{'class':'long_context'},row_ids=[r['row_id'] for r in rows],component_ids=[m['component_id'] for m in metadata])])
    def prepared(self,ordinal,tokens=(151645,)):
        src=self.make_source();plan=self.L.freeze_plan(src,self.component.validate_row);batch=plan['batches'][ordinal-1]
        torch=self.Torch();model=self.Model(torch,tokens=tokens,prompt_length=batch['context_cap']-32);decoder=self.Decoder();seen=[]
        parity=dict(row_id=self.L.digest('CPU_PARITY'),prompt_sha256=self.L.digest('CPU_SHORT419'),prompt_ids=[5]*419,prompt_tokens=419,attention_mask=[1]*419)
        cfg=dict(ordinal=ordinal,short_parity_row=parity,offline_environment={},dose=dict(maximum_forward_calls=2050,maximum_forward_input_positions=838+batch['maximum_input_positions'],context_cap=batch['context_cap'],maximum_generated_tokens=32,rows=64,optimizer_steps=0,backwards=0))
        class Active:
            def remove(self):seen.append('activation_removed')
        scientific=SimpleNamespace(compute=lambda m,t:dict(fixture_only=True),activation_hooks=lambda m,t:dict(handles=[Active()],observations={'CPU_SITE':1}),adapters=lambda m,t,**kw:dict(rank=kw['rank'],alpha=kw['alpha']),zero=lambda m,t:dict(all_B_zero=True),base_verify=lambda *a:dict(verified_source_tensors=310,frozen_base_unchanged=True,fixture_only=True))
        return src,plan,cfg,torch,model,decoder,scientific,seen
    def run_loaded(self,ordinal,tokens=(151645,),fault=None,cleanup=False):
        src,plan,cfg,torch,model,decoder,scientific,seen=self.prepared(ordinal,tokens);model.fault=fault;model.cleanup_fail=cleanup
        folder=self.root/f'batch{ordinal}';folder.mkdir();journal=self.H['Journal'](folder);report={}
        try:self.R._CPU_loaded(cfg,src,plan,ordinal,self.L,self.component,torch=torch,model=model,decoder=decoder,source_hashes={},journal=journal,guard=seen.append,helpers=scientific,device='cpu',report=report,capability=self.cap)
        finally:journal.stream.close()
        return report,model,decoder,seen,folder
    def test_three_caps_original64_collection_and2050_outer_meter(self):
        for ordinal in (1,13,25):
            report,model,decoder,seen,folder=self.run_loaded(ordinal)
            self.assertEqual(len(report['batch']['collection']['rows']),64);self.assertEqual(report['forward_dose']['attempted_calls'],66)
            self.assertEqual(report['forward_dose']['input_positions'],838+64*(report['batch']['collection']['rows'][0]['context_cap']-32))
            self.assertEqual(model.hooks,[]);self.assertEqual(decoder.checks,1);self.assertEqual(len(decoder.calls),64)
            events=[json.loads(l) for l in (folder/'events.jsonl').read_bytes().splitlines()]
            self.assertEqual(sum(e.get('event')=='returned' and e.get('operation')=='generated_only_decode' for e in events),64)
            self.assertFalse(report['batch']['native_PASS']);self.assertIn('activation_removed',seen)
    def test_cap32_actual_dose_and_original_stage_bound(self):
        report,model,decoder,_,folder=self.run_loaded(25,tokens=(9,))
        self.assertEqual(report['forward_dose']['attempted_calls'],2050);self.assertEqual(report['forward_dose']['input_positions'],report['forward_dose']['maximum_forward_input_positions'])
        self.assertEqual(len(model.calls),2050);self.assertLess((folder/'stage_long_guard_batch.json').stat().st_size,1024**2)
        self.assertTrue(all(r['truncated'] and len(r['trace'])==32 for r in report['batch']['collection']['rows']))
    def test_physical_meter_rejects_unregistered_wrongcap_and2051(self):
        _,plan,_,torch,_,_,_,_=self.prepared(1);dose=self.R.ForwardDose(torch,plan['batches'][0])
        ids=torch.tensor([[5]*419],dtype=torch.long,device='cpu')
        with self.assertRaisesRegex(ValueError,'operation'):dose.before(None,(),dict(input_ids=ids))
        dose.current='zero_parity_base_forward'
        with self.assertRaisesRegex(ValueError,'419'):dose.before(None,(),dict(input_ids=torch.tensor([[5]*420],dtype=torch.long,device='cpu')))
        dose.current='inference_forward'
        with self.assertRaisesRegex(ValueError,'same-cap'):dose.before(None,(),dict(input_ids=torch.tensor([[5]*1025],dtype=torch.long,device='cpu')))
        dose.calls=2050
        with self.assertRaisesRegex(ValueError,'exhausted'):dose.before(None,(),dict(input_ids=ids))
    def test_decoder_failure_stops_and_removes_hooks(self):
        src,plan,cfg,torch,model,decoder,scientific,seen=self.prepared(1);decoder.fault=lambda ids:(_ for _ in ()).throw(RuntimeError('CPU decode failure'))
        folder=self.root/'failed';folder.mkdir();journal=self.H['Journal'](folder);report={}
        try:
            with self.assertRaisesRegex(RuntimeError,'decode failure'):self.R._CPU_loaded(cfg,src,plan,1,self.L,self.component,torch=torch,model=model,decoder=decoder,source_hashes={},journal=journal,guard=seen.append,helpers=scientific,device='cpu',report=report,capability=self.cap)
        finally:journal.stream.close()
        self.assertEqual(len(model.calls),3);self.assertEqual(model.hooks,[]);self.assertFalse((folder/'stage_long_guard_batch.json').exists())
        events=[json.loads(l) for l in (folder/'events.jsonl').read_bytes().splitlines()];self.assertTrue(any(e.get('event')=='failed' and e.get('operation')=='generated_only_decode' for e in events))
    def test_primary_failure_and_cleanup_errors_preserved(self):
        src,plan,cfg,torch,model,decoder,scientific,seen=self.prepared(1);model.cleanup_fail=True;model.parity_bad=True
        folder=self.root/'failed';folder.mkdir();journal=self.H['Journal'](folder);report={}
        try:
            with self.assertRaisesRegex(ValueError,'physical dense'):self.R._CPU_loaded(cfg,src,plan,1,self.L,self.component,torch=torch,model=model,decoder=decoder,source_hashes={},journal=journal,guard=seen.append,helpers=scientific,device='cpu',report=report,capability=self.cap)
        finally:journal.stream.close()
        self.assertEqual(len(report['cleanup_errors']),1);self.assertEqual(model.hooks,[]);self.assertIn('activation_removed',seen)
    def test_actual_original_terminal_worker_with_inert_actors(self):
        src,plan,cfg,torch,model,decoder,scientific,seen=self.prepared(13);folder=self.root/'worker';folder.mkdir()
        cpu=SimpleNamespace(os=os,time=time,psutil=SimpleNamespace(),owner={},guard=seen.append,disk_free=4*1024**3,torch=torch,model=model,decoder=decoder,scientific=scientific)
        result=self.R.worker_body(self.root/'CPU_CONFIG.json',folder,cfg,src,plan,self.L,self.component,cpu=cpu,capability=self.cap)
        saved=json.loads((folder/'worker_result.json').read_bytes());self.assertEqual(result,saved)
        self.assertEqual(saved['sha256'],self.R.digest({k:v for k,v in saved.items() if k!='sha256'}));self.assertFalse(saved['native_PASS']);self.assertEqual(saved['status'],'COMPLETE_PENDING_INDEPENDENT_LONG_DECODED_AUDIT')
        self.assertTrue((folder/'worker_GPU_final.json').exists());self.assertEqual(model.hooks,[])
    def test_native_early_refusal_and_real_input_traps(self):
        for fn in (self.R.preflight,self.D.run_registered):
            with self.assertRaisesRegex(ValueError,'staged'):fn({})
        with self.assertRaisesRegex(ValueError,'staged'):self.R.worker_body('CPU_CONFIG','CPU_OUTPUT',{},None,None,None,None)
        with self.assertRaisesRegex(ValueError,'private authored'):self.R._CPU_loaded(capability=object(),device='cpu',torch=self.Torch())
        for path in (PACKAGE/'LONG_CONTEXT_GUARD_POLICY.json',PACKAGE/'data/crossed_source2_v1_20261003/train.json',PACKAGE/'model.safetensors'):
            with self.assertRaisesRegex(AssertionError,'input forbidden'):path.read_bytes()
    def test_exact_supervisor_source_substitutions_without_execution(self):
        # Source-only AST proof independent of any native_process or lease call.
        original=ast.parse((PACKAGE/'native_supervisor.py').read_bytes());fn=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='supervise')
        constants=[n.value for n in ast.walk(fn) if isinstance(n,ast.Constant)]
        self.assertEqual(constants.count('native_worker.py'),2);self.assertEqual(constants.count('mechanical_native_qualification_only'),1)
        self.assertIn('_bounds_valid(config)',ast.unparse(fn));self.assertIn('_WindowsOwnedWorker',ast.unparse(fn));self.assertIn('_acquire_lease',ast.unparse(fn))
    def test_strict_original_heartbeat_duplicates_and_nested_nonforward_boundaries(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):self.R.strict_json(b'{"worker_pid":1,"worker_pid":2}')
        path=self.root/'heartbeat.jsonl';path.write_bytes(b'{"unix_seconds":1,"unix_seconds":2}\n')
        with self.assertRaisesRegex(ValueError,'duplicate'):self.H['read_heartbeat'](path)
        _,plan,_,torch,_,_,_,_=self.prepared(1);dose=self.R.ForwardDose(torch,plan['batches'][0]);seen=[]
        def call(op,fn):seen.append(op);return fn()
        wrapped=dose.wrap(call);self.assertEqual(wrapped('outer_asset_check',lambda:wrapped('inner_asset_check',lambda:5)),5)
        self.assertEqual(seen,['outer_asset_check','inner_asset_check']);self.assertIsNone(dose.current)
if __name__=='__main__':
    names=('long_guard_owned_runtime.py','long_guard_owned_dispatch.py','test_long_guard_owned.py');before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names};start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(OwnedControls));after={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
    receipt=dict(kind='AUTHORED_OWNED_DECODED_LONG_RUNTIME_CPU_CONTROLS',tests=result.testsRun,success=result.wasSuccessful(),seconds=time.monotonic()-start,source_before=before,source_after=after,source_unchanged=before==after,read_paths_and_counts=dict(Counter(OwnedControls.reads)),forbidden_import_attempts=OwnedControls.imports,intentional_real_path_denials=len(OwnedControls.denials),fixture_only=True,native_PASS=False,scoring_authorized=False,native_qualification_established=False)
    receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();path=HERE/('CPU_CONTROLS_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(path=str(path),tests=result.testsRun,success=result.wasSuccessful(),source_unchanged=before==after)));raise SystemExit(0 if result.wasSuccessful() and before==after else 1)
