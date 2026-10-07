"""Actual authored long mock producer, original Journal and saved-byte audit."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import ast, builtins, hashlib, io, json, math, os, re, sys, tempfile, time, unittest

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == 'rineya_synth_centralized_reasoning_v1')

def source(path, name):
    module = ModuleType(name); module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__); return module

class LongControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reads = []; cls.denials = []; cls.imports = []
        real_io, real_open, real_import = io.open, builtins.open, builtins.__import__
        def reader(path, mode='r', *args, **kwargs):
            if isinstance(path, (str, bytes, Path)) and not any(c in str(mode) for c in ('w','a','x','+')):
                p = Path(path).resolve(); cls.reads.append(str(p))
                allowed = p.is_relative_to(HERE) or (p.parent == PACKAGE and p.suffix == '.py') or (p.parent == PACKAGE/'tests' and p.suffix == '.py')
                if not allowed: cls.denials.append(str(p)); raise AssertionError('real model/data/API/source inputs forbidden')
            return real_io(path, mode, *args, **kwargs)
        def importing(name, *args, **kwargs):
            if name.split('.',1)[0] in {'torch','transformers','ctypes','psutil','tokenizers','safetensors','numpy'}:
                cls.imports.append(name); raise AssertionError('actual native/framework import forbidden')
            return real_import(name, *args, **kwargs)
        cls.traps = [patch('io.open', side_effect=reader), patch.object(builtins,'open',side_effect=lambda p,*a,**k: reader(p,*a,**k) if isinstance(p,(str,bytes,Path)) else real_open(p,*a,**k)), patch.object(builtins,'__import__',side_effect=importing)]
        for t in cls.traps: t.start()
        cls.L = source(HERE/'long_guard_collection.py','CPU_long_guard_collection'); cls.component = cls.L._helpers()
        ns = {'deepcopy':deepcopy,'math':math,'SimpleNamespace':SimpleNamespace,'VOCAB':151936}
        tree = ast.parse((PACKAGE/'tests/test_inference_component.py').read_bytes())
        names = {'Device','Tensor','Array','Parameter','Scalar','SparseRow','Logits','FakeTorch','Model','Decoder'}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names],type_ignores=[]),'original_authored_component_fakes','exec'),ns)
        # Sparse prefix logits preserve full shape without materializing vocabulary.
        class LazyLogits(ns['Tensor']):
            def __init__(self,length,token):
                super().__init__(shape=(1,length,151936),dtype='float32');self.token=token
            def __getitem__(self,key): return ns['SparseRow'](self.token)
            def finite(self): return True
        class Model(ns['Model']):
            def __call__(self,**kw):
                assert self.torch.no_grad_depth > 0 and kw['use_cache'] is False and kw['return_dict'] is True
                ids = kw['input_ids'].tolist()[0];assert kw['attention_mask'].tolist()[0] == [1]*len(ids)
                self.calls.append((len(ids),ids[-1])); step = len(ids)-self.prompt_length
                token = self.tokens[min(step,len(self.tokens)-1)]
                if self.fault: self.fault(self,None)
                return SimpleNamespace(logits=LazyLogits(len(ids),token))
        ns['LongModel'] = Model
        jns = {'Path':Path,'os':os,'json':json,'re':re,'math':math,'datetime':datetime,'timezone':timezone,'defaultdict':defaultdict,'require':cls.L.require,
               'canonical':lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False),'contract':SimpleNamespace(filehash=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())}
        tree = ast.parse((PACKAGE/'native_worker.py').read_bytes())
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'diagnostic_json','Journal'}],type_ignores=[]),'original_Journal','exec'),jns)
        cls.mock = ns; cls.Journal = jns['Journal']; cls.capability = cls.L._CPU_capability()
    @classmethod
    def tearDownClass(cls):
        for t in reversed(cls.traps): t.stop()
        assert cls.imports == [], cls.imports
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE,prefix='CPU_long_');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
    def make_source(self):
        rows=[];metadata=[]
        for family in self.L.FAMILIES:
            for placement in self.L.PLACEMENTS:
                for missing in (0,1,2):
                    for world in range(8):
                        component=self.L.digest(['CPU_world',family,placement,missing,world])
                        for length,cap in self.L.ENVELOPES.items():
                            for variant,flags in self.L.VARIANTS.items():
                                rid=self.L.digest([component,length,variant]);rows.append({'row_id':rid,'prompt_sha256':self.L.digest(['CPU_prompt',rid]),'prompt_ids':[5]*length,'prompt_tokens':length,'attention_mask':[1]*length})
                                metadata.append({'row_id':rid,'component_id':component,'family':family,'placement':placement,'missing_index':missing,'total_sequence_envelope':cap,'variant':variant,'answerable':flags[0],'adversarial':flags[1]})
        domains=[{'domain_id':'CPU_AUTHORED_LONG_DOMAIN','class':'long_context','row_ids':[r['row_id'] for r in rows],'component_ids':[m['component_id'] for m in metadata]}]
        return {'rows':rows,'metadata':metadata,'declared_domains':domains}
    def ref(self,path):
        raw=path.read_bytes();value=json.loads(raw);return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'value_sha256':self.L.digest(value)}
    def descriptor(self,directory,outputs):
        events=directory/'events.jsonl';raw=events.read_bytes()
        return {'batch':self.ref(directory/'stage_long_guard_batch.json'),'rows':[self.ref(directory/('stage_inference_row_%04d.json'%i)) for i in range(1,65)],
            'tokens':[[self.ref(directory/('stage_raw_row%04d_token%02d.json'%(i,j))) for j in range(1,len(r['trace'])+1)] for i,r in enumerate(outputs,1)],
            'events':{'path':str(events),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}
    def collect(self,source,plan,ordinal,tokens=(9,10,151645),fault=None):
        directory=self.root/('batch%02d'%ordinal);directory.mkdir();torch=self.mock['FakeTorch']();batch=plan['batches'][ordinal-1]
        model=self.mock['LongModel'](torch,tokens=tokens,prompt_length=batch['context_cap']-32);model.fault=fault
        decoder=self.mock['Decoder']();journal=self.Journal(directory);guards=[]
        try:
            result=self.L._CPU_collect(source,plan,ordinal,capability=self.capability,torch=torch,model=model,decoder=decoder,journal=journal,guard=guards.append,device='cpu',component=self.component)
        finally:journal.stream.close()
        self.assertEqual(decoder.calls,[r['raw_generated_ids'] for r in result['collection']['rows']]);self.assertGreater(len(guards),0)
        return result,self.descriptor(directory,result['collection']['rows']),model
    def test_actual_world_length_variant_source_permutation_and_budget(self):
        source=self.make_source();plan=self.L.freeze_plan(source,self.component.validate_row)
        self.assertEqual([r['prompt_tokens'] for r in source['rows'][:12]],[992]*4+[2016]*4+[4064]*4)
        self.assertEqual(plan['batches'][0]['source_indices'][:8],[0,1,2,3,12,13,14,15])
        self.assertEqual(Counter(b['context_cap'] for b in plan['batches']),{1024:12,2048:12,4096:12})
        for i,v in enumerate(plan['capture_to_source']):self.assertEqual(plan['source_to_capture'][v],i)
        for batch in plan['batches']:self.assertEqual(batch['maximum_input_positions'],64*(32*(batch['context_cap']-32)+496))
        for field in ('capture_to_source','source_to_capture'):
            bad=deepcopy(plan);bad[field][0]=bad[field][1]
            with self.assertRaisesRegex(ValueError,'exact frozen'):self.L.check_plan(source,bad,self.component.validate_row)
        bad=deepcopy(source);bad['metadata'][0]['missing_index']=1
        with self.assertRaises(ValueError):self.L.freeze_plan(bad,self.component.validate_row)
    def test_full36_actual_producer_saved_audit_canonical_inverse_join(self):
        source=self.make_source();plan=self.L.freeze_plan(source,self.component.validate_row);saved=[]
        for ordinal in range(1,37):
            _,ref,model=self.collect(source,plan,ordinal);saved.append(ref);self.assertEqual(len(model.calls),192)
        result=self.L.audit_saved_population(source,plan,saved,component=self.component,fixture_only=True)
        self.assertEqual([r['row_id'] for r in result['rows']],[r['row_id'] for r in source['rows']]);self.assertEqual(len(result['rows']),2304)
        for row in result['rows']:
            batch=plan['batches'][row['capture_batch_ordinal']-1];self.assertEqual(row['raw']['source_packet_sha256'],batch['source_packet_sha256']);self.assertEqual(batch['source_indices'][row['capture_row_ordinal']-1],row['source_index'])
        self.assertFalse(result['scoring_authorized']);self.assertFalse(result['retention_certified'])
    def test_cap32_costs_sink_trace_tampering_and_parameter_mutation(self):
        source=self.make_source();plan=self.L.freeze_plan(source,self.component.validate_row)
        result,saved,model=self.collect(source,plan,25,tokens=(9,))
        checked=self.L.audit_saved_batch(source,plan,25,saved,component=self.component,fixture_only=True)
        self.assertEqual(checked['attempted_forwards'],2048);self.assertEqual(checked['input_positions'],plan['batches'][24]['maximum_input_positions'])
        self.assertTrue(all(r['raw']['truncated'] and len(r['raw']['trace'])==32 for r in checked['rows']))
        for change in ('trace','cost','packet','partial'):
            altered=deepcopy(saved)
            path=Path(altered['rows'][0]['path']) if change!='partial' else Path(altered['tokens'][0][0]['path'])
            original=path.read_bytes();value=json.loads(original)
            if change=='trace':value['trace'][0]['prefix_length']-=1
            elif change=='cost':value['costs']['input_positions']['attempted']-=1
            elif change=='packet':value['source_packet_sha256']='f'*64
            else:value['trace'][0]['logprob']=-100.
            path.write_text(json.dumps(value),encoding='utf-8')
            target=self.ref(path)
            if change=='partial':altered['tokens'][0][0]=target
            else:altered['rows'][0]=target
            with self.assertRaises(ValueError):self.L.audit_saved_batch(source,plan,25,altered,component=self.component,fixture_only=True)
            path.write_bytes(original)
        def mutate(model,_):model.parameters['base.weight']._version+=1
        with self.assertRaisesRegex(ValueError,'version changed'):self.collect(source,plan,1,fault=mutate)
        self.assertFalse((self.root/'batch01/stage_long_guard_batch.json').exists())
        self.assertTrue((self.root/'batch01/stage_inference_failure_0001.json').exists())
    def test_native_early_refusal_and_real_read_traps(self):
        with self.assertRaisesRegex(ValueError,'staged long'):self.L.run_registered({},loader=lambda:(_ for _ in ()).throw(AssertionError('loader invoked')))
        for path in (PACKAGE/'data/crossed_source2_v1_20261003/train.json',PACKAGE/'LONG_CONTEXT_GUARD_POLICY.json',PACKAGE/'weights.safetensors'):
            with self.assertRaisesRegex(AssertionError,'inputs forbidden'):path.read_bytes()

if __name__=='__main__':
    before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('long_guard_collection.py','test_long_guard_collection.py')};start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(LongControls))
    after={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in before}
    value={'kind':'LONG_GUARD_ACTUAL_MOCK_PRODUCER_SAVED_RAW_CPU_CONTROLS','tests':result.testsRun,'success':result.wasSuccessful(),'seconds':time.monotonic()-start,'source_before':before,'source_after':after,'source_unchanged':before==after,'read_paths_and_counts':dict(Counter(LongControls.reads)),'forbidden_import_attempts':LongControls.imports,'intentional_real_path_denials':len(LongControls.denials),'fixture_only':True,'native_PASS':False,'scoring_authorized':False,'native_long_guard_qualification_established':False,'limits':['Original long greedy component and original Journal used with authored sparse fake logits; no model/API/data reads.','No native loaded-assets/owned-lifecycle qualification or dispatch supplied; staged entry refuses before reads.','Parameter object/version controls do not replace independent native tensor byte checks.']};value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    p=HERE/'CPU_CONTROLS_001.json'
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'path':str(p),'sha256':value['sha256'],'tests':result.testsRun,'success':result.wasSuccessful(),'source_unchanged':before==after}));raise SystemExit(0 if result.wasSuccessful() and before==after else 1)
