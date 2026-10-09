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
        cls.T=source(HERE/'arithmetic_capture_pipeline.py','CPU_task_guard_pipeline');cls.modules=cls.T._helpers();component=cls.modules['inference_component'];ns={'deepcopy':deepcopy,'math':math,'SimpleNamespace':SimpleNamespace,'component':component,'VOCAB':151936,'unittest':unittest,'patch':patch}
        for filename,names in [('test_inference_component.py',{'Device','Tensor','Array','Parameter','Scalar','SparseRow','Logits','FakeTorch','Model','Decoder','row'}),('test_inference_qualification.py',{'Hook','HookedModel'}),('test_short_collection.py',{'WorkerTests'})]:
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
            for world in range(128):
                for variant in (0,1):
                    for rendering in ('expression_json','fully_parenthesized_infix'):
                        rid=self.T.digest([kind,world,variant,rendering]);rows.append({'row_id':rid,'prompt_sha256':self.T.digest(['authored',rid]),'prompt_ids':[1],'prompt_tokens':1,'attention_mask':[1]});metadata.append({'row_id':rid,'component_id':'CPU_'+kind+'_'+str(world),'mechanism':kind,'variant':variant,'rendering':rendering})
        pop={'rows':rows,'metadata':metadata,'domain':{'domain_id':self.T.DOMAIN,'class':'task','row_ids':[r['row_id'] for r in rows],'component_ids':[m['component_id'] for m in metadata]}};popref=self.ref('population.json',pop);pins={str(HERE/'arithmetic_capture_pipeline.py'):hashlib.sha256((HERE/'arithmetic_capture_pipeline.py').read_bytes()).hexdigest()};collector={str(PACKAGE/'short_collection_worker.py'):hashlib.sha256((PACKAGE/'short_collection_worker.py').read_bytes()).hexdigest()};model={'CPU_AUTHORED_NO_MODEL':'a'*64};configs=[];anchors=[];batches=[]
        for ordinal in range(1,9):
            selected=rows[(ordinal-1)*64:ordinal*64];config={'batch_selector':{'source_kind':'arithmetic_guard_DEV','ordinal':ordinal},'rows':selected,'source_packet_sha256':self.T.digest(selected),'model_role':'fresh_dense_M0_zero_B_only_no_STATE1_or_study_checkpoint','model_pins':model,'source_pins':collector,'dose':{'rows':64,'context_cap':512,'maximum_generated_tokens':32}};configref=self.ref('config%02d.json'%ordinal,config);configs.append(configref);audit=self.ref('authored_audit%02d.json'%ordinal,{'CPU_authored_not_native_audit':True});anchor={'preserved_directory':str(self.root/('preserved%02d'%ordinal)),'audit_path':audit['path'],'expected_audit_sha256':audit['sha256'],'expected_config_sha256':configref['sha256']};anchors.append(anchor)
            if collect:
                directory=self.root/('collector%02d'%ordinal);directory.mkdir();fixture=list(self.fixture_owner.fixture(directory));fixture[3]['rows']=selected;fixture[3]['source_packet_sha256']=self.T.digest(selected)
                report=self.fixture_owner.execute(tuple(fixture));outputs=report['collection']['rows'];fixture[4].stream.close();self.assertEqual(len(outputs),64);self.assertEqual(report['forward_dose']['attempted_calls'],2+sum(len(r['raw_generated_ids']) for r in outputs))
                batches.append({'scope':self.modules['score_import'].FIXTURE,'arm':'M0','source_kind':'arithmetic_guard_DEV','ordinal':ordinal,'inputs':selected,'outputs':outputs,'source_packet_sha256':self.T.digest(selected),'audit_passed':True,'references':{'config':configref,'audit':audit,'model_pins':model,'source_pins':collector}})
        approval={'kind':'OWNER_ARITHMETIC_GUARD_M0_PIPELINE_AUTHORIZATION_V1','scope':self.T.SCOPE,'fixture_only':True,'approved':True,'permissions':self.T.PERMISSION,'source_pins':pins,'collector_source_pins':collector,'model_pins':model,'population_value_sha256':popref['value_sha256'],'batch_config_sha256s':[r['sha256'] for r in configs],'capture_anchors_sha256':self.T.digest(anchors),'human_evidence_reference':'CPU_AUTHORED_NO_PERMISSION'};aref=self.ref('approval.json',approval)
        review={'kind':'INDEPENDENT_ARITHMETIC_GUARD_M0_PIPELINE_SOURCE_REVIEW_V1','scope':self.T.SCOPE,'fixture_only':True,'accepted':True,'source_pins':pins,'collector_source_pins':collector,'model_pins':model,'approval_file_sha256':aref['sha256'],'population_value_sha256':popref['value_sha256'],'batch_config_sha256s':approval['batch_config_sha256s'],'capture_anchors_sha256':approval['capture_anchors_sha256']}
        cfg=self.T.seal({'kind':'REGISTERED_ARITHMETIC_GUARD_M0_PIPELINE_V1','scope':self.T.SCOPE,'fixture_only':True,'source_pins':pins,'collector_source_pins':collector,'model_pins':model,'approval':aref,'review':self.ref('review.json',review),'population':popref,'batch_configs':configs,'capture_anchors':anchors,'row_order_sha256':self.T.digest([r['row_id'] for r in rows])});return self.ref('registered.json',cfg),cfg,batches
    def test_original_twelve64_collector_to_task512_domain_import(self):
        ref,config,batches=self.make_registered();loader=lambda anchor:batches[config['capture_anchors'].index(anchor)];result=self.T._CPU_prepare(ref,loader,capability=self.T._CPU_capability());self.assertEqual(len(result['rows']),512);self.assertEqual(len(set(result['domain']['component_ids'])),128);self.assertEqual(len(result['batch_references']),8);self.assertFalse(result['scoring_authorized']);self.assertFalse(result['language_qualification_established'])
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
    def test_arithmetic_plan_origins_dose_and_staged_owned_ABI(self):
        C=source(HERE/'arithmetic_capture_contract.py','authored_closed_arithmetic_contract')
        ref,config,_=self.make_registered(collect=False);p=self.T.read_ref(config['population'])
        promptrows=[]
        for row,meta in zip(p['rows'],p['metadata']):
            promptrows.append({'row_id':row['row_id'],'parent_id':meta['component_id'],'messages':[{'role':'user','content':'AUTHORED_ARITHMETIC'}],
                'input_ids':row['prompt_ids'],'prompt_tokens':row['prompt_tokens'],'attention_mask':row['attention_mask'],
                'prompt_sha256':row['prompt_sha256'],'prompt_token_sha256':C.digest(row['prompt_ids'])})
        selected={'path':str(self.root/'source.json'),'bytes':1,'sha256':'a'*64}
        plan=C.build_plan(promptrows,p['metadata'],measurement_reference=selected,prompt_reference=selected,policy_reference=selected,ledger_reference=selected,fixture_only=True)
        self.assertEqual(8,len(plan['batches']['arithmetic_guard_DEV']))
        self.assertEqual(2050,plan['batches']['arithmetic_guard_DEV'][0]['maximum_total_forwards'])
        self.assertNotEqual(plan['origins'][0]['message_sha256'],plan['origins'][0]['rendered_chat_sha256'])
        self.assertEqual(128,len({m['component_id'] for m in plan['metadata']}))
        with self.assertRaisesRegex(ValueError,'staged arithmetic'):C.preflight('forbidden')
        O=source(HERE/'arithmetic_capture_owned.py','authored_closed_arithmetic_owned')
        with patch.dict(sys.modules,arithmetic_capture_contract=C):
            with self.assertRaisesRegex(ValueError,'staged arithmetic'):O.dispatch({},'forbidden')
        tree=ast.parse((HERE/'arithmetic_capture_owned.py').read_bytes());node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='ArithmeticScope')
        ns={'ast':ast};exec(compile(ast.Module(body=[node],type_ignores=[]),'authored_owned_scope','exec'),ns)
        original=ast.parse((PACKAGE/'short_collection_supervisor.py').read_bytes());adapted=ast.fix_missing_locations(ns['ArithmeticScope']().visit(deepcopy(original)))
        literals=[n.value for n in ast.walk(adapted) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertNotIn('short_M0_raw_collection_one_fixed_batch_only',literals)
        self.assertIn('arithmetic_M0_raw_collection_one_fixed_batch_only',literals)
        self.assertIn('--output-dir',literals)
        self.assertEqual(2,sum(isinstance(n,ast.Name) and n.id=='ARITHMETIC_WORKER_PATH' for n in ast.walk(adapted)))
        def file_ref(name,value):
            f=self.ref(name,value);return {k:f[k] for k in ('path','bytes','sha256')}
        prompt_ref=file_ref('plan_prompts.json',{'rows':promptrows,'native_scoring_authorized':False})
        policy_ref=file_ref('plan_policy.json',{'sha256':'b'*64})
        ledger_ref=file_ref('plan_ledger.json',{'policy_sha256':'b'*64,'selected_parent_order_sha256':'c'*64})
        measured={'projection_kind':'SYNTH_ARITHMETIC_TOKEN_PROJECTION001','projection_schema_version':1,'full_result_sha256':'d'*64,
            'policy_file':prompt_ref,'status':'CPU_CONTEXT_MEASURED_NATIVE_UNADMITTED','measured_rows':512,'unmeasured_rows':0,
            'overflows':[],'policy_sha256':'b'*64,'parent_order_sha256':'c'*64,'row_order_sha256':C.digest([r['row_id'] for r in promptrows])}
        measured['sha256']=C.digest(measured);measurement_ref=file_ref('plan_measurement.json',measured)
        plan=C.build_plan(promptrows,p['metadata'],measurement_reference=measurement_ref,prompt_reference=prompt_ref,
            policy_reference=policy_ref,ledger_reference=ledger_ref)
        plan_ref=file_ref('closed_plan.json',plan);C.PLAN_PATH=Path(plan_ref['path']);C.PLAN_REF=plan_ref
        self.assertEqual(plan,C.plan())
        altered=deepcopy(measured);altered['policy_file']={**prompt_ref,'sha256':'0'*64}
        altered['sha256']=C.digest({k:v for k,v in altered.items() if k!='sha256'})
        changed_ref=file_ref('wrong_measurement.json',altered);changed=deepcopy(plan);changed['measurement_reference']=changed_ref
        changed['sha256']=C.digest({k:v for k,v in changed.items() if k!='sha256'})
        bad_ref=file_ref('wrong_plan.json',changed);C.PLAN_PATH=Path(bad_ref['path']);C.PLAN_REF=bad_ref
        with self.assertRaisesRegex(ValueError,'exact prompt-file join'):C.plan()
    def test_qualified_whole_response_CPU_scoring_cap_policy_and_enrollment_contribution(self):
        ref,config,_=self.make_registered(collect=False);pop=self.T.read_ref(config['population']);rows=[];labels=[]
        for i,meta in enumerate(pop['metadata']):
            rid=meta['row_id'];labels.append({'row_id':rid,'parent_id':meta['component_id'],'answer':0})
            rows.append({'row_id':rid,'guard_metadata':meta,'raw':{'finish_reason':'generation_cap' if i%2 else 'eos','decoded_response':'{"answer":0}'}})
        packet={'fixture_only':True,'rows':rows,'domain':pop['domain']};policy={'sha256':'b'*64}
        q={'kind':'INDEPENDENT_ARITHMETIC_SCORER_QUALIFICATION001','fixture_only':True,'qualified':True,'domain_id':'arithmetic',
            'scorer_source_sha256':hashlib.sha256((PACKAGE/'arithmetic_guard.py').read_bytes()).hexdigest(),'labels_sha256':self.T.digest(labels),'policy_sha256':policy['sha256']}
        def strict(row,text,policy):
            try:
                obj=json.loads(text,object_pairs_hook=self.T.pairs,parse_constant=self.T.bad)
                good=type(obj) is dict and set(obj)=={'answer'} and type(obj['answer']) is int
            except (ValueError,TypeError):return {'correct':False,'malformed':True}
            return {'correct':good and obj['answer']==row['answer'],'malformed':not good}
        scorer=SimpleNamespace(__file__=str(PACKAGE/'arithmetic_guard.py'),score=strict)
        value=self.T._CPU_score(packet,labels,policy,q,scorer=scorer,capability=self.T._CPU_capability())
        self.assertEqual(512,sum(value['scores'].values()));self.assertEqual(256,sum(d['at_generation_cap'] for d in value['details']))
        for response in ('prefix {"answer":0}','{"answer":0} trailing','{"answer":true}','{"answer":null}'):
            changed=deepcopy(packet);changed['rows'][0]['raw']['decoded_response']=response
            scored=self.T._CPU_score(changed,labels,policy,q,scorer=scorer,capability=self.T._CPU_capability());self.assertFalse(scored['scores'][labels[0]['row_id']])
        badq={**q,'qualified':False}
        with self.assertRaises(ValueError):self.T._CPU_score(packet,labels,policy,badq,scorer=scorer,capability=self.T._CPU_capability())
        with self.assertRaisesRegex(ValueError,'staged arithmetic'):self.T.score_arithmetic_import({})
        with self.assertRaisesRegex(ValueError,'staged arithmetic'):self.T.enrollment_contribution({})
        stats=source(PACKAGE/'dev_statistics.py','authored_arithmetic_statistics');stats.policy=lambda:{'retention':{'required_domain_classes':['task','language','long_context']}}
        enrolled=self.T._CPU_enrollment_contribution(value,baseline_id='AUTHORED_BASELINE',statistics=stats,capability=self.T._CPU_capability())
        self.assertEqual('INCONCLUSIVE',enrolled['status']);self.assertEqual(128,len(enrolled['domains'][0]['eligible_components']))

if __name__=='__main__':
    before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('arithmetic_capture_pipeline.py','arithmetic_capture_contract.py','arithmetic_capture_owned.py','test_arithmetic_capture.py')};start=time.monotonic();r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(TaskControls));after={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in before};v={'kind':'TASK_GUARD_ORIGINAL_COLLECTOR_IMPORT_CPU_CONTROLS','tests':r.testsRun,'success':r.wasSuccessful(),'seconds':time.monotonic()-start,'source_before':before,'source_after':after,'source_unchanged':before==after,'read_paths_and_counts':dict(Counter(TaskControls.reads)),'forbidden_import_attempts':TaskControls.imports,'intentional_real_path_denials':len(TaskControls.denials),'fixture_only':True,'native_PASS':False,'scoring_authorized':False,'language_qualification_established':False,'long_guard_producer_implemented':False,'limits':['Original collect_loaded/DurableRawCollection/InferenceComponent used with source-extracted FakeTorch/HookedModel, no actual model/math/API.','Native original load_m0_batch is wired conditionally but not executed by authored supplier; self-signed fixture audits never native qualification.','Task source population original512/128 metadata retained; language andlong2304 capture/qualification remain absent.']};v['sha256']=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();p=HERE/'CPU_CONTROLS003.json'
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'path':str(p),'sha256':v['sha256'],'success':r.wasSuccessful(),'tests':r.testsRun,'source_unchanged':before==after}));raise SystemExit(0 if r.wasSuccessful() and before==after else 1)
