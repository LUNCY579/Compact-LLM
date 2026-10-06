"""Authored task populations and actual original mock raw collector only."""
from collections import defaultdict,Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import ast,builtins,hashlib,io,json,math,os,re,sys,tempfile,time,unittest
HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
def source(path,name):
    m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
class TaskControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reads=[];cls.denials=[];cls.imports=[];open_io=io.open;open_builtin=builtins.open;import_builtin=builtins.__import__
        def reader(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,bytes,Path)) and not any(c in str(mode) for c in ('w','a','x','+')):
                p=Path(path).resolve();cls.reads.append(str(p));allowed=p.is_relative_to(HERE) or (p.parent==PACKAGE and p.suffix=='.py') or(p.parent==PACKAGE/'tests' and p.suffix=='.py')
                if not allowed:cls.denials.append(str(p));raise AssertionError('real task/data/model/API inputs forbidden')
            return open_io(path,mode,*args,**kwargs)
        def importing(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','transformers','ctypes','psutil','tokenizers','safetensors','numpy'}:cls.imports.append(name);raise AssertionError('actual framework/API import forbidden')
            return import_builtin(name,*args,**kwargs)
        cls.traps=[patch('io.open',side_effect=reader),patch.object(builtins,'open',side_effect=lambda p,*a,**kw:reader(p,*a,**kw) if isinstance(p,(str,bytes,Path)) else open_builtin(p,*a,**kw)),patch.object(builtins,'__import__',side_effect=importing)]
        for t in cls.traps:t.start()
        cls.T=source(HERE/'task_guard_pipeline.py','CPU_task_guard_pipeline');cls.modules=cls.T._helpers();component=cls.modules['inference_component'];ns={'deepcopy':deepcopy,'math':math,'SimpleNamespace':SimpleNamespace,'component':component,'VOCAB':151936,'unittest':unittest,'patch':patch}
        for filename,names in [('test_inference_component.py',{'Device','Tensor','Array','Parameter','Scalar','SparseRow','Logits','FakeTorch','Model','Decoder','row'}),('test_inference_qualification.py',{'HookedModel'}),('test_short_collection.py',{'WorkerTests'})]:
            tree=ast.parse((PACKAGE/'tests'/filename).read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (names is None or n.name in names)]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),ns)
        native=SimpleNamespace(filehash=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest());worker_ns={**ns,'Path':Path,'os':os,'json':json,'re':re,'datetime':datetime,'timezone':timezone,'defaultdict':defaultdict,'canonical':lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True),'require':cls.T.require,'contract':native}
        tree=ast.parse((PACKAGE/'native_worker.py').read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('diagnostic_json','Journal')];exec(compile(ast.Module(body=nodes,type_ignores=[]),'native_worker.py','exec'),worker_ns)
        ns['Journal']=worker_ns['Journal'];ns['Path']=Path;ns['digest']=cls.T.digest
        collect_ns={'deepcopy':deepcopy,'require':cls.T.require,'digest':cls.T.digest,'InferenceComponent':component.InferenceComponent,'validate_row':component.validate_row}
        for name in ('install_fp32_compute','install_qv_lora','install_fp32_activation_assertions','zero_lora_status','verify_frozen_base'):collect_ns[name]=lambda *args:None
        tree=ast.parse((PACKAGE/'short_collection_worker.py').read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('ForwardDose','DurableRawCollection','_logits','collect_loaded')];exec(compile(ast.Module(body=nodes,type_ignores=[]),'short_collection_worker.py','exec'),collect_ns);ns['worker']=SimpleNamespace(**collect_ns)
        cls.mock=ns;cls.fixture_owner=ns['WorkerTests']('runTest')
    @classmethod
    def tearDownClass(cls):
        for t in reversed(cls.traps):t.stop()
        assert cls.imports==[],cls.imports
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE,prefix='CPU_task_');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
    def ref(self,name,value):
        p=self.root/name;b=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii');p.write_bytes(b);return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'value_sha256':self.T.digest(value)}
    def make_registered(self,collect=True):
        rows=[];metadata=[]
        for kind in self.T.KINDS:
            for world in range(64):
                for variant in ('authored_original','authored_condition'):
                    for rendering in ('json_lines','typed_ledger'):
                        rid=self.T.digest([kind,world,variant,rendering]);rows.append({'row_id':rid,'prompt_sha256':self.T.digest(['authored',rid]),'prompt_ids':[1],'prompt_tokens':1,'attention_mask':[1]});metadata.append({'row_id':rid,'component_id':'CPU_'+kind+'_'+str(world),'mechanism':kind,'variant':variant,'rendering':rendering})
        pop={'rows':rows,'metadata':metadata,'domain':{'domain_id':self.T.DOMAIN,'class':'task','row_ids':[r['row_id'] for r in rows],'component_ids':[m['component_id'] for m in metadata]}};popref=self.ref('population.json',pop);pins={str(HERE/'task_guard_pipeline.py'):hashlib.sha256((HERE/'task_guard_pipeline.py').read_bytes()).hexdigest()};model={'CPU_AUTHORED_NO_MODEL':'a'*64};configs=[];anchors=[];batches=[]
        for ordinal in range(1,13):
            selected=rows[(ordinal-1)*64:ordinal*64];config={'batch_selector':{'source_kind':'task_guard_DEV','ordinal':ordinal},'rows':selected,'source_packet_sha256':self.T.digest(selected),'model_role':'fresh_dense_M0_zero_B_only_no_STATE1_or_study_checkpoint','model_pins':model,'source_pins':pins,'dose':{'rows':64,'context_cap':512,'maximum_generated_tokens':32}};configref=self.ref('config%02d.json'%ordinal,config);configs.append(configref);audit=self.ref('authored_audit%02d.json'%ordinal,{'CPU_authored_not_native_audit':True});anchor={'preserved_directory':str(self.root/('preserved%02d'%ordinal)),'audit_path':audit['path'],'expected_audit_sha256':audit['sha256'],'expected_config_sha256':configref['sha256']};anchors.append(anchor)
            if collect:
                directory=self.root/('collector%02d'%ordinal);directory.mkdir();fixture=list(self.fixture_owner.fixture(directory));fixture[3]['rows']=selected;fixture[3]['source_packet_sha256']=self.T.digest(selected)
                report=self.fixture_owner.execute(tuple(fixture));outputs=report['collection']['rows'];fixture[4].stream.close();self.assertEqual(len(outputs),64);self.assertEqual(report['forward_dose']['attempted_calls'],194)
                batches.append({'scope':self.modules['score_import'].FIXTURE,'arm':'M0','source_kind':'task_guard_DEV','ordinal':ordinal,'inputs':selected,'outputs':outputs,'source_packet_sha256':self.T.digest(selected),'audit_passed':True,'references':{'config':configref,'audit':audit,'model_pins':model,'source_pins':pins}})
        approval={'kind':'OWNER_TASK_GUARD_M0_PIPELINE_AUTHORIZATION_V1','scope':self.T.SCOPE,'fixture_only':True,'approved':True,'permissions':self.T.PERMISSION,'source_pins':pins,'collector_source_pins':pins,'model_pins':model,'population_value_sha256':popref['value_sha256'],'batch_config_sha256s':[r['sha256'] for r in configs],'capture_anchors_sha256':self.T.digest(anchors),'human_evidence_reference':'CPU_AUTHORED_NO_PERMISSION'};aref=self.ref('approval.json',approval)
        review={'kind':'INDEPENDENT_TASK_GUARD_M0_PIPELINE_SOURCE_REVIEW_V1','scope':self.T.SCOPE,'fixture_only':True,'accepted':True,'source_pins':pins,'collector_source_pins':pins,'model_pins':model,'approval_file_sha256':aref['sha256'],'population_value_sha256':popref['value_sha256'],'batch_config_sha256s':approval['batch_config_sha256s'],'capture_anchors_sha256':approval['capture_anchors_sha256']}
        cfg=self.T.seal({'kind':'REGISTERED_TASK_GUARD_M0_PIPELINE_V1','scope':self.T.SCOPE,'fixture_only':True,'source_pins':pins,'collector_source_pins':pins,'model_pins':model,'approval':aref,'review':self.ref('review.json',review),'population':popref,'batch_configs':configs,'capture_anchors':anchors,'row_order_sha256':self.T.digest([r['row_id'] for r in rows])});return self.ref('registered.json',cfg),cfg,batches
    def test_original_twelve64_collector_to_task768_domain_import(self):
        ref,config,batches=self.make_registered();loader=lambda anchor:batches[config['capture_anchors'].index(anchor)];result=self.T._CPU_prepare(ref,loader,capability=self.T._CPU_capability());self.assertEqual(len(result['rows']),768);self.assertEqual(len(set(result['domain']['component_ids'])),192);self.assertEqual(len(result['batch_references']),12);self.assertFalse(result['scoring_authorized']);self.assertFalse(result['language_qualification_established'])
    def test_approval_and_swapped_capture_manifest_reject_before_loader(self):
        ref,config,_=self.make_registered(collect=False);candidate=deepcopy(config);candidate['capture_anchors'].reverse();candidate=self.T.seal({k:v for k,v in candidate.items() if k!='sha256'});bad=self.ref('changed_registered.json',candidate)
        with self.assertRaisesRegex(ValueError,'permission before'):self.T._CPU_prepare(bad,lambda anchor:(_ for _ in ()).throw(AssertionError('loader before approval')),capability=self.T._CPU_capability())
    def test_actual_population_config_rows_dose_and_origin_drift_reject(self):
        ref,config,_=self.make_registered(collect=False);batch_path=Path(config['batch_configs'][0]['path']);batch_path.write_bytes(b'CHANGED_AUTHORED')
        with self.assertRaisesRegex(ValueError,'byte descriptor'):self.T._CPU_prepare(ref,lambda a:(_ for _ in ()).throw(AssertionError('loader before config validation')),capability=self.T._CPU_capability())
    def test_closed_native_long_and_real_data_read_traps(self):
        with self.assertRaisesRegex(ValueError,'staged task'):self.T.prepare_task_import({})
        with self.assertRaisesRegex(ValueError,'staged task'):self.T.dispatch_task_batch({},1,'not_selected')
        with self.assertRaisesRegex(ValueError,'TRAIN qualification cannot substitute'):self.T.prepare_long_import()
        for p in (PACKAGE/'data/crossed_source2_v1_20261003/train.json',PACKAGE/'TASK_RETENTION_GUARD_POLICY.json',PACKAGE/'weights.safetensors'):
            with self.assertRaisesRegex(AssertionError,'inputs forbidden'):p.read_bytes()

if __name__=='__main__':
    before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('task_guard_pipeline.py','test_task_guard_pipeline.py')};start=time.monotonic();r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(TaskControls));after={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in before};v={'kind':'TASK_GUARD_ORIGINAL_COLLECTOR_IMPORT_CPU_CONTROLS','tests':r.testsRun,'success':r.wasSuccessful(),'seconds':time.monotonic()-start,'source_before':before,'source_after':after,'source_unchanged':before==after,'read_paths_and_counts':dict(Counter(TaskControls.reads)),'forbidden_import_attempts':TaskControls.imports,'intentional_real_path_denials':len(TaskControls.denials),'fixture_only':True,'native_PASS':False,'scoring_authorized':False,'language_qualification_established':False,'long_guard_producer_implemented':False,'limits':['Original collect_loaded/DurableRawCollection/InferenceComponent used with source-extracted FakeTorch/HookedModel, no actual model/math/API.','Native original load_m0_batch is wired conditionally but not executed by authored supplier; self-signed fixture audits never native qualification.','Task source population original768/192 metadata retained; language andlong2304 capture/qualification remain absent.']};v['sha256']=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();p=HERE/'CPU_CONTROLS_001.json'
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'path':str(p),'sha256':v['sha256'],'success':r.wasSuccessful(),'tests':r.testsRun,'source_unchanged':before==after}));raise SystemExit(0 if r.wasSuccessful() and before==after else 1)
