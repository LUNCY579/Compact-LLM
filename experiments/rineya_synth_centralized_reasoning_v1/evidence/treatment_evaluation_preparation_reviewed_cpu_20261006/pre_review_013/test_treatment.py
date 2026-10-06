"""Authored inert chain/archive/capture fixtures; no model/tokenizer/native load."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
from unittest.mock import patch
import builtins,hashlib,sys,tempfile,unittest

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
def source(path,name):
    m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
T=source(HERE/'treatment_evaluation.py','CPU_treatment_preparation')

class TreatmentControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.imports=[];original=builtins.__import__
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','tokenizers','transformers','safetensors','ctypes','psutil','numpy'}:
                cls.imports.append(name);raise AssertionError('model/tokenizer/native import forbidden: '+name)
            return original(name,*args,**kwargs)
        cls.trap=patch.object(builtins,'__import__',side_effect=imports);cls.trap.start()
        modules,_=T._helpers();cls.restore=modules['final_adapter_inference'];cls.adapter=modules['score_import'];cls.schedule=cls.restore.schedule;cls.chunks=cls.restore.chunks
        cls.registration=cls.schedule.build_registration();cls.cap=T._CPU_capability()
    @classmethod
    def tearDownClass(cls):cls.trap.stop();assert cls.imports==[],cls.imports
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_treatment_',dir=HERE);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();self.calls=[];self.saved={}
        self.chain=self.make_chain('M1');self.latest=deepcopy(self.chain[-1]['terminal_costs'])
    def seal(self,v):return {**v,'sha256':self.schedule.digest(v)}
    def publication(self,name,v):
        raw=self.schedule.canonical(v).encode('utf-8');p=self.root/name;p.write_bytes(raw)
        return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'payload_sha256':self.schedule.digest(v),'durable':True}
    def make_chain(self,arm):
        C=self.chunks;recipe=C.recipe(self.registration,arm,C.NATIVE_SCOPE);chain=[]
        for chunk in (1,2,3):
            binding=C.binding(self.registration,arm,chunk*9)
            counts={'gradient_reset':chunk*9,'forward':chunk*180,'completion_loss':chunk*180,'backward':chunk*180,'finite_gradient_clip':chunk*9,'optimizer_step':chunk*9,'checkpoint_save':chunk,'checkpoint_reload':chunk-1,'checkpoint_publish':chunk-1,'terminal_costs_publish':chunk-1,'checkpoint_promote':chunk-1}
            posted={'schema_version':1,'scope':C.KIND,'protocol':{name:{'attempted':n,'returned':n,'failed':0,'pending':0} for name,n in counts.items()},'positions':{'input_attempted':binding['input_tokens'],'input_returned':binding['input_tokens'],'supervised_attempted':binding['supervised_tokens'],'supervised_returned':binding['supervised_tokens']},'primitive_operations':{'CPU_authored_archive_marker':{'attempted':chunk,'returned':chunk,'failed':0,'pending':0}}}
            archive=('CPU_AUTHORED_INERT_NOT_TENSOR_ARCHIVE '+arm+' '+str(chunk)).encode('ascii');path=self.root/(arm+'_chunk'+str(chunk)+'.inert');path.write_bytes(archive)
            metadata={'arm':arm,'seed':101,'scope':C.NATIVE_SCOPE,'recipe_sha256':recipe['sha256'],'cohort_sha256':recipe['cohort_sha256'],'backend_source_sha256':recipe['backend_source_sha256'],'binding':binding,'native_launch_authorized':False,'training_authorized':False,'native_resume_qualified':False,'qualification_STATE1_used':False,'fresh_start_attestation_sha256':'d'*64,'CPU_authored_inert_archive':True}
            descriptor={'schema_version':1,'kind':C.BACKEND_KIND,'path':str(path),'bytes':len(archive),'sha256':hashlib.sha256(archive).hexdigest(),'metadata':metadata,'metadata_sha256':self.schedule.digest(metadata)}
            envelope=self.seal({'schema_version':1,'kind':C.KIND,'arm':arm,'seed':101,'chunk_ordinal':chunk,'recipe':recipe,'binding':binding,'backend_descriptor':descriptor,'posted_save_costs':posted,'native_resume_qualified':False,'training_authorized':False})
            terminal={'envelope_sha256':envelope['sha256'],'costs':C._project(C._project(posted,'checkpoint_publish'),'terminal_costs_publish')};terminal['costs_sha256']=self.schedule.digest(terminal['costs'])
            costs=C._project(terminal['costs'],'checkpoint_promote');promotion={'envelope_sha256':envelope['sha256'],'durable_terminal_sha256':self.schedule.digest(terminal),'terminal_costs':costs}
            record=self.seal({'schema_version':1,'kind':C.KIND,'status':'SUCCESSFUL_STUDY_CHUNK_PROTOCOL_ONLY','envelope':envelope,'publication':self.publication(arm+str(chunk)+'_envelope.json',envelope),'durable_terminal':terminal,'terminal_publication':self.publication(arm+str(chunk)+'_terminal.json',terminal),'promotion':promotion,'promotion_publication':self.publication(arm+str(chunk)+'_promotion.json',promotion),'terminal_costs':costs});chain.append(record)
        return chain
    def prepare(self,arm='M1',chain=None,latest=None):return T._CPU_prepare(self.registration,arm,self.chain if chain is None else chain,self.latest if latest is None else latest,_capability=self.cap)
    def actors(self,prepared):
        value=prepared.value;outer=self
        def factory(**kwargs):self.calls.append('factory');return {'torch':'INERT_NO_FRAMEWORK','model':'INERT_NO_MODEL','reader':lambda d:Path(d['path']).read_bytes(),'decoder':'INERT_NO_TOKENIZER','source_hashes':{'CPU_AUTHORED_NO_MODEL':'e'*64}}
        def restore(torch,model,reader,call,guard,**kwargs):
            self.calls.append('restore');self.assertEqual(kwargs['arm'],value['arm']);self.assertEqual(kwargs['chunk_chain'],value['chunk_chain']);self.assertEqual(kwargs['registration'],value['training_registration']);self.assertEqual(kwargs['latest_terminal_costs'],value['latest_terminal_costs'])
            return {'arm':value['arm'],'checkpoint_sha256':value['checkpoint_descriptor']['sha256'],'final_binding':deepcopy(value['final_binding']),'inference_allowed':True,'optimizer_restored':False,'training_RNG_restored':False,'CPU_fixture':True,'real_final_study_chain_verified':False,'native_launch_authorized':False,'fixture_only':True}
        def validate(model,row,receipt):self.calls.append('validate');return self.restore.validate_row(row)
        class Component:
            def __init__(self,torch,model,decoder,call,guard,preserve,**kwargs):
                outer.calls.append('component');outer.assertEqual(kwargs['max_attempted_forwards'],384*32);outer.assertEqual(kwargs['context_cap'],512);outer.assertEqual(kwargs['generation_cap'],32);outer.assertEqual(kwargs['device'],'cpu')
                self.sink=preserve
            def collect(self,rows,*,source_packet_sha256):
                outer.calls.append('collect');outputs=[]
                for ordinal,row in enumerate(rows):
                    ids=[7,151645] if ordinal%2==0 else [7]*32;finish='eos' if ids[-1]==151645 else 'generation_cap'
                    output={'row_id':row['row_id'],'prompt_sha256':row['prompt_sha256'],'prompt_ids_sha256':outer.adapter.digest(row['prompt_ids']),'input_row_sha256':outer.adapter.digest(row),'source_packet_sha256':source_packet_sha256,'status':'RAW_RESPONSE_COLLECTED_UNSCORED','scope':outer.adapter.FIXTURE,'capability_correctness_scored':False,'oracle_inputs_read':False,'prompt_truncated':False,'raw_generated_ids':ids,'finish_reason':finish,'truncated':finish=='generation_cap','eos_token_id':151645 if finish=='eos' else None,'context_cap':512,'maximum_generated_tokens':32,'decoded_response':'AUTHORED WHOLE RESPONSE; no JSON repair'}
                    output['trace']=[{'token':token,'logprob':-1.0,'prefix_length':row['prompt_tokens']+index} for index,token in enumerate(ids)];output['selected_logprobs']=[-1.0]*len(ids)
                    outputs.append(output);self.sink('CPU_authored_'+row['row_id'],deepcopy(output))
                return {'rows':outputs,'source_packet_sha256':source_packet_sha256}
        def call(name,function):return function()
        def guard(stage):self.calls.append(stage)
        def preserve(name,value):self.saved[name]=deepcopy(value)
        return factory,call,guard,preserve,restore,Component,validate
    def collect(self,prepared):
        factory,call,guard,preserve,restore,component,validate=self.actors(prepared)
        return T._CPU_collect(prepared,factory,call,guard,preserve,restore_function=restore,component_type=component,validate_input=validate,_capability=self.cap)
    def reseal(self,value):return {**{k:v for k,v in value.items() if k!='sha256'},'sha256':T.digest({k:v for k,v in value.items() if k!='sha256'})}
    def test_same_restore_loop_full384_frozen_lineage_fixed_scorer_metadata(self):
        prepared=self.prepare();collection=self.collect(prepared);packet=T._CPU_prepare_import(prepared,collection,_capability=self.cap)
        self.assertEqual(len(packet['rows']),384);self.assertEqual(len({row['component_id'] for row in packet['rows']}),96)
        self.assertEqual(packet['row_order_sha256'],prepared.value['panel']['row_order_sha256']);self.assertEqual(packet['scorer_sources'],prepared.value['panel']['scorer_sources'])
        self.assertEqual(packet['final_binding']['completed_updates'],27);self.assertEqual(packet['final_binding']['consumed_occurrences'],540)
        self.assertEqual(packet['checkpoint_descriptor'],self.chain[-1]['envelope']['backend_descriptor'])
        self.assertLess(self.calls.index('factory'),self.calls.index('restore'));self.assertLess(self.calls.index('restore'),self.calls.index('component'))
        self.assertEqual(self.calls.count('validate'),384);self.assertEqual(self.calls.count('collect'),6);self.assertEqual(len(self.saved),390)
        self.assertEqual({row['raw']['finish_reason'] for row in packet['rows']},{'eos','generation_cap'})
        self.assertTrue(all(row['raw']['decoded_response']=='AUTHORED WHOLE RESPONSE; no JSON repair' for row in packet['rows']))
        self.assertFalse(packet['scoring_authorized']);self.assertFalse(packet['quality_gain_established'])
    def test_M2_has_own_full_dose_adapter_and_same384_population(self):
        chain=self.make_chain('M2');prepared=self.prepare('M2',chain,chain[-1]['terminal_costs'])
        collection=self.collect(prepared);packet=T._CPU_prepare_import(prepared,collection,_capability=self.cap)
        self.assertEqual(packet['arm'],'M2');self.assertEqual(len(packet['rows']),384)
        self.assertEqual(packet['final_binding']['input_tokens'],130574)
        self.assertEqual(packet['final_binding']['supervised_tokens'],5615)
        self.assertEqual(packet['final_binding']['consumed_occurrences'],540)
        self.assertEqual(packet['row_order_sha256'],self.adapter.load_frozen_panel(PACKAGE)['row_order_sha256'])
        self.assertFalse(packet['scoring_authorized'])
    def test_source_recipe_drift_and_terminal_rollback_refuse_before_assets(self):
        registration=deepcopy(self.registration);registration['source_and_decision_pins']={};registration['sha256']=self.schedule.digest({k:v for k,v in registration.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'source/decision'):
            T._CPU_prepare(registration,'M1',self.chain,self.latest,_capability=self.cap)
        with self.assertRaisesRegex(ValueError,'authoritative final terminal costs'):
            self.prepare(latest=self.chain[1]['terminal_costs'])
        self.assertEqual(self.calls,[])
    def test_wrong_arm_and_partial_chain_refuse_before_assets(self):
        for arm,chain,latest in [('M2',self.chain,self.latest),('M1',self.chain[:2],self.chain[1]['terminal_costs']),('M1',[{'kind':'STATE1'}]*3,self.latest)]:
            with self.subTest(arm=arm,count=len(chain)),self.assertRaises(ValueError):self.prepare(arm,chain,latest)
        self.assertEqual(self.calls,[])
    def test_byte_drift_after_plan_refuses_before_factory(self):
        prepared=self.prepare();Path(self.chain[-1]['envelope']['backend_descriptor']['path']).write_bytes(b'CHANGED INERT')
        with self.assertRaises(ValueError):self.collect(prepared)
        self.assertNotIn('factory',self.calls)
    def test_restore_failure_refuses_component_and_all_captures(self):
        prepared=self.prepare();factory,call,guard,preserve,restore,component,validate=self.actors(prepared)
        def failed(*args,**kwargs):self.calls.append('restore_failed');return {'inference_allowed':False}
        with self.assertRaisesRegex(ValueError,'successful exact final-factor restore'):
            T._CPU_collect(prepared,factory,call,guard,preserve,restore_function=failed,component_type=component,validate_input=validate,_capability=self.cap)
        self.assertNotIn('component',self.calls);self.assertEqual(self.saved,{})
    def test_changed_chain_checkpoint_and_row_order_refuse_consumer(self):
        prepared=self.prepare();original=self.collect(prepared)
        for change in (lambda c:c.update(arm='M2'),lambda c:c.update(checkpoint_sha256='0'*64),lambda c:c.update(chain_sha256='0'*64),lambda c:c['batches'].pop(),lambda c:c['batches'][0]['outputs'].reverse()):
            candidate=deepcopy(original);change(candidate);candidate=self.reseal(candidate)
            with self.subTest(),self.assertRaises(ValueError):T._CPU_prepare_import(prepared,candidate,_capability=self.cap)
    def test_finish_semantics_and_full_generated_response_preserved(self):
        prepared=self.prepare();original=self.collect(prepared)
        candidate=deepcopy(original);candidate['batches'][0]['outputs'][1]['raw_generated_ids']=[7]
        with self.assertRaisesRegex(ValueError,'EOS/cap/truncation'):T._CPU_prepare_import(prepared,self.reseal(candidate),_capability=self.cap)
    def test_public_native_gate_and_forged_capability_before_helpers_or_factory(self):
        with patch.object(T,'_helpers',side_effect=AssertionError('must not read helper/assets/outcomes')):
            with self.assertRaisesRegex(ValueError,'no native authority'):T.run_registered(approval={},qualification={},m0_enrollment={})
            with self.assertRaisesRegex(ValueError,'no native authority'):T._wired_native_producer(None,None)
            with self.assertRaisesRegex(ValueError,'private CPU'):T._CPU_prepare({},'M1',[],{},_capability=object())
    def ref(self,name,value):
        path=self.root/name;data=__import__('json').dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii');path.write_bytes(data)
        return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'value_sha256':T.digest(value)}
    def registered_fixture(self):
        native=__import__('native_contract');stats=__import__('dev_statistics');prepared=self.prepare();value=prepared.value
        factory=self.root/'selected_authored_factory.py';code=b'def make_boundaries(value):\n    from CPU_authored_actor_bridge import boundaries\n    return boundaries\n';factory.write_bytes(code)
        factory_ref={'path':str(factory),'bytes':len(code),'sha256':hashlib.sha256(code).hexdigest()};source_pins=T._source_closure();output=str(self.root/'future_output')
        restored=self.actors(prepared)[4](None,None,None,None,None,arm='M1',chunk_chain=self.chain,registration=self.registration,latest_terminal_costs=self.latest)
        restored['native_launch_authorized']=False
        preservation={'kind':'PRESERVED_FINAL_ADAPTER_CAPTURE_QUALIFICATION_V1','fixture_only':True,'arm':'M1','checkpoint_sha256':value['checkpoint_descriptor']['sha256'],'source_pins':source_pins,'raw_files':{'inert_original_observation':self.ref('fixture_observation.json',{'CPU_authored_not_native_fact':True})}}
        audit={'kind':'INDEPENDENT_FINAL_ADAPTER_CAPTURE_AUDIT_V1','fixture_only':True,'arm':'M1','checkpoint_sha256':value['checkpoint_descriptor']['sha256'],'chain_sha256':T.digest(self.chain),'restore_receipt':restored,'raw_collection_mechanics_verified':True,'source_pins':source_pins,'resources':native.RESOURCE_BOUNDS,'tolerances':native.TOLERANCES,'preservation_value_sha256':T.digest(preservation)}
        qualification=T.seal({'kind':'INDEPENDENT_FINAL_ADAPTER_COLLECTION_QUALIFICATION_V1','fixture_only':True,'arm':'M1','checkpoint_sha256':value['checkpoint_descriptor']['sha256'],'chain_sha256':T.digest(self.chain),'training_registration_sha256':self.registration['sha256'],'final_binding':value['final_binding'],'source_pins':source_pins,'model_pins':self.restore.loader.ASSET_PINS,'resources':native.RESOURCE_BOUNDS,'tolerances':native.TOLERANCES,'restore_receipt':restored,'raw_collection_mechanics_verified':True,'scoring_authorized':False,'evidence':{'independent_audit':self.ref('qualification_audit.json',audit),'preservation':self.ref('qualification_preservation.json',preservation)}})
        baseline={'kind':'CPU_AUTHORED_COMPLETE_M0_BASELINE_FIXTURE','fixture_only':True,'baseline_id':'CPU_authored_baseline','rows':384,'components':96,'panel_sha256':value['panel']['sha256'],'candidate_outputs_used':False}
        domains=[];scores={};bindings={}
        for name in stats.policy()['retention']['required_domain_classes']:
            row='CPU_guard_'+name;domains.append({'domain_id':name,'class':name,'row_ids':[row],'component_ids':['CPU_component_'+name]});scores[row]=True;bindings[name]={'scorer_sha256':'a'*64,'labels_sha256':'b'*64,'qualification_receipt_sha256':'c'*64,'qualified':True}
        enrollment=stats.freeze_guard_enrollment(domains,scores,baseline_id=baseline['baseline_id'],scorer_bindings=bindings)
        approval={'kind':'OWNER_APPROVED_FINAL_TREATMENT_COLLECTION_V1','scope':T.NATIVE_SCOPE,'arm':'M1','fixture_only':True,'approved':True,'permissions':T.PERMISSIONS,'checkpoint_sha256':value['checkpoint_descriptor']['sha256'],'chain_sha256':T.digest(self.chain),'source_pins':source_pins,'factory_source':factory_ref,'factory_name':'make_boundaries','output_directory':output,'human_evidence_reference':'CPU_AUTHORED_NO_HUMAN_PERMISSION'}
        config=T.seal({'schema_version':1,'kind':'REGISTERED_FINAL_TREATMENT_COLLECTION_V1','scope':T.NATIVE_SCOPE,'fixture_only':True,'arm':'M1','training_registration':self.ref('training_registration.json',self.registration),'chunk_chain':self.ref('chain.json',self.chain),'latest_terminal_costs':self.ref('latest.json',self.latest),'panel_registration_sha256':value['panel']['sha256'],'finish_policy':self.ref('finish_policy_copy.json',self.adapter._json((PACKAGE/'DEV_FINISH_POLICY_001.json').read_bytes())),'source_pins':source_pins,'model_pins':self.restore.loader.ASSET_PINS,'resources':native.RESOURCE_BOUNDS,'tolerances':native.TOLERANCES,'approval':self.ref('approval.json',approval),'adapter_qualification':self.ref('qualification.json',qualification),'m0_baseline':self.ref('baseline.json',baseline),'enrollment':self.ref('guard_enrollment.json',enrollment),'factory_source':factory_ref,'factory_name':'make_boundaries','output_directory':output})
        # The finish policy descriptor must select its actual preserved original
        # bytes; copying different formatting cannot rebrand that file pin.
        finish=PACKAGE/'DEV_FINISH_POLICY_001.json';config['finish_policy']={'path':str(finish),'bytes':finish.stat().st_size,'sha256':hashlib.sha256(finish.read_bytes()).hexdigest(),'value_sha256':T.digest(self.adapter._json(finish.read_bytes()))};config=self.reseal(config)
        ref=self.ref('registered_fixture.json',config);anchors={'config_file_sha256':ref['sha256'],'config_value_sha256':ref['value_sha256']}
        def m0_fixture(body,panel):
            self.assertEqual(body['rows'],384);self.assertEqual(body['components'],96);self.assertEqual(body['panel_sha256'],panel['sha256']);self.assertFalse(body['candidate_outputs_used']);return body['baseline_id']
        return Path(ref['path']),anchors,m0_fixture
    def test_exact_external_registered_CPU_actor_factory_and_independent_raw_consumer(self):
        path,anchors,m0=self.registered_fixture();prepared=T._CPU_registered_fixture(path,anchors,m0_fixture=m0,_capability=self.cap)
        factory,call,guard,preserve,restore,component,validate=self.actors(prepared)
        bridge=ModuleType('CPU_authored_actor_bridge');bridge.boundaries={'assets_factory':factory,'call':call,'guard':guard,'preserve':preserve};sys.modules[bridge.__name__]=bridge;self.addCleanup(lambda:sys.modules.pop(bridge.__name__,None))
        self.calls.clear();boundaries=T._factory_boundaries(prepared.value['registration_config'],prepared)
        collection=T._CPU_collect(prepared,boundaries['assets_factory'],boundaries['call'],boundaries['guard'],boundaries['preserve'],restore_function=restore,component_type=component,validate_input=validate,_capability=self.cap)
        refs=[self.ref('raw_%03d.json'%i,output) for i,output in enumerate([r for b in collection['batches'] for r in b['outputs']])]
        Path(prepared.value['registration_config']['output_directory']).mkdir()
        audited_prepared=T._CPU_registered_fixture(path,anchors,m0_fixture=m0,_capability=self.cap,for_collection=False)
        self.assertEqual(audited_prepared.original,prepared.original)
        with self.assertRaisesRegex(ValueError,'fresh collection leaf'):T._CPU_registered_fixture(path,anchors,m0_fixture=m0,_capability=self.cap)
        audit=T._CPU_raw_consumer_fixture(audited_prepared,collection,refs,_capability=self.cap)
        self.assertEqual(len(audit['rows']),384);self.assertEqual(audit['baseline_id'],'CPU_authored_baseline');self.assertFalse(audit['native_PASS']);self.assertFalse(audit['scoring_authorized'])
        wrong=deepcopy(refs);wrong[0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'selected treatment JSON bytes'):T._CPU_raw_consumer_fixture(prepared,collection,wrong,_capability=self.cap)
    def test_missing_or_wrong_external_packet_denied_before_factory(self):
        path,anchors,m0=self.registered_fixture();self.calls.clear()
        wrong=dict(anchors,config_file_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'selected treatment JSON bytes'):T._CPU_registered_fixture(path,wrong,m0_fixture=m0,_capability=self.cap)
        config=self.adapter._json(path.read_bytes());qualification=Path(config['adapter_qualification']['path']);qualification.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'selected treatment JSON bytes'):T._CPU_registered_fixture(path,anchors,m0_fixture=m0,_capability=self.cap)
        self.assertNotIn('factory',self.calls)
    def test_fixture_registered_packet_cannot_enter_native_validator(self):
        path,anchors,m0=self.registered_fixture();self.calls.clear()
        with patch.object(T,'HERE',T.EXPECTED):
            with self.assertRaisesRegex(ValueError,'fixture/native separation'):T.run_registered(path,anchors)
        self.assertNotIn('factory',self.calls)
    def test_original_native_M1_M2_import_gate_stays_closed(self):
        native=self.adapter.load_frozen_panel(PACKAGE)
        for arm in ('M1','M2'):
            with self.assertRaisesRegex(ValueError,'native M1/M2 collector/audit schema unregistered'):self.adapter.prepare_import(native,[],arm=arm)

if __name__=='__main__':unittest.main()
