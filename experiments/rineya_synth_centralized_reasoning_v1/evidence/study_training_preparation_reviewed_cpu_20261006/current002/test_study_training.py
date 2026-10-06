"""Authored inert STUDY composition controls; all non-source real reads denied."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import ast,builtins,collections,hashlib,io,json,math,os,random,sys,tempfile,time,unittest

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
FAKE_SOURCE=PACKAGE/'tests/test_study_native_chunk.py'
def load_source(path,name):
    value=ModuleType(name);value.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),value.__dict__);return value
T=load_source(HERE/'study_training.py','authored_STUDY_composition')

def authored_sources(modules):
    schedule=modules['study_schedule'];pools={};records={};arms={}
    for arm,total,supervised in [('M1',130562,5614),('M2',130574,5615)]:
        rows={};ids=[]
        for i in range(540):
            identity='fixture_'+arm+'_%03d'%i;ids.append(identity);length=total//540+int(i<total%540);target=supervised//540+int(i<supervised%540)
            prompt=[1]*(length-target);completion=[7]*(target-1)+[151645]
            rows[identity]={'row_id':identity,'prompt_ids':prompt,'completion_ids':completion,'input_ids':prompt+completion,'labels':[-100]*len(prompt)+completion,'attention_mask':[1]*length,'prompt_tokens':len(prompt),'total_tokens':length,'supervised_tokens':target}
            records[identity]={'parent_id':'fixture_parent_%02d'%(i%20),'skill':'R' if i<192 else 'S'}
        pools[arm]=rows;updates=[];chunks=[]
        for i in range(27):
            selected=ids[i*20:(i+1)*20];updates.append({'update_ordinal':i+1,'learning_rate':schedule.learning_rates()[i],'first_occurrence_ordinal':i*20+1,'last_occurrence_ordinal':(i+1)*20,'occurrence_row_ids':selected,**schedule._stats(selected,rows,records)})
        for i in range(3):
            selected=ids[i*180:(i+1)*180];chunks.append({'chunk_ordinal':i+1,'first_update_ordinal':i*9+1,'last_update_ordinal':(i+1)*9,'first_occurrence_ordinal':i*180+1,'last_occurrence_ordinal':(i+1)*180,'occurrence_row_ids':selected,'occurrence_order_sha256':schedule.digest(selected),**schedule._stats(selected,rows,records)})
        arms[arm]={'occurrence_order_sha256':schedule.digest(ids),'updates':updates,'resume_chunks':chunks,'totals':schedule._stats(ids,rows,records)}
    registration=T.seal({'CPU_authored':True,'arms':arms,'source_and_decision_pins':{'AUTHORED_TRAIN_SOURCE_METADATA_NO_REAL_DATA':'a'*64}})
    policy={'CPU_authored':True,'retention':{'required_domain_classes':['task','language','long_context']}}
    return registration,pools,records,policy

class StudyControls(unittest.TestCase):
    reads=[];denials=[];imports=[]
    @classmethod
    def setUpClass(cls):
        original_import=builtins.__import__;original_open=io.open;original_builtin=builtins.open
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','tokenizers','transformers','safetensors','ctypes','psutil','numpy','rs_solver','rs_verifier','evaluation_panel'}:cls.imports.append(name);raise AssertionError('framework/native/scorer import forbidden: '+name)
            return original_import(name,*args,**kwargs)
        cls.importtrap=patch.object(builtins,'__import__',side_effect=imports);cls.importtrap.start()
        def checked(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,bytes,Path)) and not any(x in str(mode) for x in ('w','a','x','+')):
                selected=Path(path).resolve();cls.reads.append(str(selected))
                try:selected.relative_to(HERE)
                except ValueError:
                    allowed=(selected.parent==PACKAGE and selected.suffix=='.py') or selected==FAKE_SOURCE
                    if not allowed:cls.denials.append(str(selected));raise AssertionError('real data/model/captures/scorer read forbidden: '+str(selected))
            return original_open(path,mode,*args,**kwargs)
        cls.readtrap=patch('io.open',side_effect=checked);cls.readtrap.start()
        def builtin(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,bytes,Path)):return checked(path,mode,*args,**kwargs)
            return original_builtin(path,mode,*args,**kwargs)
        cls.builtintrap=patch.object(builtins,'open',side_effect=builtin);cls.builtintrap.start()
        cls.modules=T._helpers();cls.registration,cls.pools,cls.records,cls.policy=authored_sources(cls.modules);cls.cap=T._CPU_capability()
        cls.fixture=T._CPU_fixture(cls.registration,cls.pools,cls.records,cls.policy,capability=cls.cap)
    @classmethod
    def tearDownClass(cls):cls.builtintrap.stop();cls.readtrap.stop();cls.importtrap.stop();assert cls.imports==[],cls.imports
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='authored_STUDY_',dir=HERE);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();self.calls=[]
    def ref(self,name,value):
        data=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii');path=self.root/name;path.write_bytes(data)
        return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'value_sha256':T.digest(value)}
    def source_ref(self,name,raw):
        path=self.root/name;path.write_bytes(raw);return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    def guard_enrollment(self):
        stats=self.modules['dev_statistics'];domains=[];scores={};bindings={}
        for name in ('task','language','long_context'):
            row='fixture_guard_'+name;domains.append({'domain_id':name,'class':name,'row_ids':[row],'component_ids':['fixture_guard_component_'+name]});scores[row]=True;bindings[name]={'scorer_sha256':'a'*64,'labels_sha256':'b'*64,'qualification_receipt_sha256':'c'*64,'qualified':True}
        with patch.object(stats,'policy',return_value=deepcopy(self.policy)):return stats.freeze_guard_enrollment(domains,scores,baseline_id='fixture_baseline384',scorer_bindings=bindings)
    def packet(self,arm='M1',chunk=1,previous=None):
        # The CPU evidence source is deliberately inert and cannot enter native;
        # it exercises selected original-byte replay and provenance joins only.
        audit_source=self.source_ref('authored_independent_replay.py',b'def replay(**request):\n    return request["authored_evidence"]\n')
        boundary=self.source_ref('authored_owned_boundary.py',b'def make(value):\n    raise AssertionError("CPU does not execute native owned boundary source")\n')
        pins={**T.source_pins(),audit_source['path']:audit_source['sha256'],boundary['path']:boundary['sha256']};enrollment=self.guard_enrollment();baseline='fixture_baseline384'
        proofs={}
        definitions=[('training_qualification_check','STUDY_TRAINING_FULL27_AND_RESUME_QUALIFIED',{'full27_qualified':True,'full_resumed9_qualified':True,'study_checkpoint_used':False}),('m0_scoring_check','COMPLETE_M0_384_96_SCORED_BEFORE_TREATMENT',{'rows':384,'components':96,'candidate_outputs_used':False}),('guard_enrollment_check','ALL_REQUIRED_GUARD_SCORERS_SOURCE_CORRECT_BEFORE_TREATMENT',{'enrollment_sha256':enrollment['sha256'],'scorer_bindings':enrollment['scorer_bindings'],'qualified_domain_ids':['task','language','long_context']})]
        for name,purpose,extra in definitions:
            result={'fixture_only':True,'purpose':purpose,'baseline_id':baseline,'qualified':True,'CPU_authored':True,**extra};proofs[name]={'source':audit_source,'function':'replay','request':self.ref(name+'_request.json',{'authored_evidence':result}),'expected_result':self.ref(name+'_result.json',result),'purpose':purpose}
        previous_ref=None if previous is None else self.ref('previous_resume.json',previous);latest=None if previous is None else self.ref('latest_costs.json',previous['terminal_costs'])
        registration=self.ref('authored_training_registration.json',self.registration);output=str(self.root/('output_'+arm+'_'+str(chunk)))
        approval={'kind':'OWNER_STUDY_TRAINING_CHUNK_AUTHORIZATION_V1','scope':T.SCOPE,'fixture_only':True,'approved':True,'permissions':T.PERMISSIONS,'arm':arm,'chunk':chunk,'source_pins':pins,'training_registration_value_sha256':registration['value_sha256'],'baseline_id':baseline,'owned_boundary_source':boundary,'owned_boundary_name':'make','model_directory':'CPU_AUTHORED_NO_MODEL','output_directory':output,'previous_resume_value_sha256':None if previous_ref is None else previous_ref['value_sha256'],'human_evidence_reference':'CPU_AUTHORED_NO_REAL_OWNER_AUTHORIZATION'}
        approval_ref=self.ref('approval.json',approval)
        review={'kind':'INDEPENDENT_STUDY_TRAINING_CHUNK_REVIEW_V1','scope':T.SCOPE,'fixture_only':True,'accepted':True,'source_pins':pins,'approval_file_sha256':approval_ref['sha256'],'training_registration_value_sha256':registration['value_sha256'],'baseline_id':baseline,**proofs}
        config=T.seal({'kind':'REGISTERED_STUDY_TRAINING_CHUNK_V1','scope':T.SCOPE,'fixture_only':True,'arm':arm,'chunk':chunk,'source_pins':pins,'model_pins':self.modules['native_loader'].ASSET_PINS,'resources':self.modules['native_contract'].RESOURCE_BOUNDS,'tolerances':self.modules['native_contract'].TOLERANCES,'approval':approval_ref,'review':self.ref('review.json',review),'training_registration':registration,**proofs,'guard_enrollment':self.ref('guard_enrollment.json',enrollment),'baseline_id':baseline,'owned_boundary_source':boundary,'owned_boundary_name':'make','model_directory':'CPU_AUTHORED_NO_MODEL','output_directory':output,'previous_resume':previous_ref,'latest_terminal_costs':latest})
        return self.publish_config(config)
    def publish_config(self,config):
        config=T.seal({k:v for k,v in config.items() if k!='sha256'});ref=self.ref('registered_authored_chunk.json',config);return Path(ref['path']),{'config_file_sha256':ref['sha256'],'config_value_sha256':ref['value_sha256']}
    def actors(self):
        outer=self
        def backend_factory(prepared,boundaries):
            outer.calls.append('inert_backend_construct');modules=prepared['modules'];tree=ast.parse(FAKE_SOURCE.read_bytes());classes=[deepcopy(n) for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('Journal','Preserve','QuadraticAdam')]
            require_count=len(classes);outer.assertEqual(require_count,3)
            namespace={'deepcopy':deepcopy,'Path':Path,'hashlib':hashlib,'json':json,'os':os,'math':math,'random':random,'chunk':modules['study_native_chunk'],'schedule':modules['study_schedule']}
            exec(compile(ast.Module(body=classes,type_ignores=[]),str(FAKE_SOURCE),'exec'),namespace)
            output=Path(prepared['config']['output_directory']);output.mkdir();preserve=namespace['Preserve'](output);boundaries['preserve']=preserve
            fake=namespace['QuadraticAdam'](prepared['config']['arm'],prepared['recipe'],boundaries['call'],boundaries['guard']);outer.backends.append(fake);return fake
        def call(name,function):self.calls.append(name);return function()
        def guard(stage):self.calls.append(stage)
        def cleanup():self.calls.append('owned_cleanup')
        def terminal(result):self.calls.append('terminal_audit');return {'CPU_authored':True,'protocol_status':result['status'],'native_PASS':False}
        return backend_factory,{'call':call,'guard':guard,'preserve':lambda e:None,'admission':lambda q:(_ for _ in ()).throw(AssertionError('CPU cannot invoke native admission')),'cleanup':cleanup,'terminal_audit':terminal}
    def run_packet(self,path,anchors):
        self.backends=[];factory,boundaries=self.actors();return T._CPU_run_registered(path,anchors,self.fixture,factory,boundaries,capability=self.cap)
    def test_full27_three_chunks_exact540_parent_vectors_and_original_resume_math(self):
        previous=None;results=[]
        for chunk in (1,2,3):
            path,anchors=self.packet(chunk=chunk,previous=previous);report=self.run_packet(path,anchors);result=report['protocol_result'];self.assertEqual(result['status'],'COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED',result.get('error'));self.assertEqual(result['prevalidated_whole_occurrences'],180);previous=result['resume_descriptor'];results.append(result)
        binding=previous['envelope']['binding'];self.assertEqual((binding['completed_updates'],binding['consumed_occurrences'],binding['input_tokens'],binding['supervised_tokens'],binding['learning_rate']),(27,540,130562,5614,0.0))
        self.assertEqual(previous['terminal_costs']['protocol']['optimizer_step']['returned'],27);self.assertEqual(previous['terminal_costs']['protocol']['forward']['returned'],540)
        self.assertNotIn('qualification_evidence',previous['envelope']);self.assertFalse(report['native_PASS']);self.assertFalse(report['quality_gain_established'])
        self.assertEqual(self.registration['arms']['M1']['totals']['parent_exposures'],self.registration['arms']['M2']['totals']['parent_exposures']);self.assertEqual(set(self.records[x]['skill'] for x in self.pools['M1']),{'R','S'})
    def test_M2_complete_atomic_source_dose_and_three_chunk_continuation(self):
        previous=None
        for chunk in (1,2,3):
            path,anchors=self.packet(arm='M2',chunk=chunk,previous=previous);report=self.run_packet(path,anchors);previous=report['protocol_result']['resume_descriptor']
        binding=previous['envelope']['binding'];self.assertEqual((binding['input_tokens'],binding['supervised_tokens'],binding['consumed_occurrences']),(130574,5615,540));self.assertEqual(binding['learning_rate'],0.0)
    def test_approval_failure_before_training_archive_assets_and_factory(self):
        path,anchors=self.packet();config=T.read_json({'path':str(path),'bytes':path.stat().st_size,'sha256':anchors['config_file_sha256'],'value_sha256':anchors['config_value_sha256']});approval=T.read_json(config['approval'])
        for update in ({'approved':False},{'arm':'M0'},{'permissions':{}},{'scope':'QUALIFICATION_ONLY'},{'source_pins':{}},{'output_directory':'WRONG'}):
            candidate=deepcopy(approval);candidate.update(update);body=deepcopy(config);body['approval']=self.ref('bad_approval.json',candidate);path,selected=self.publish_config(body);self.calls=[]
            with self.subTest(update=update),patch.object(T,'_helpers',side_effect=AssertionError('helper/data/assets before approval')):
                with self.assertRaisesRegex(ValueError,'exact approved STUDY scope'):self.run_packet(path,selected)
            self.assertEqual(self.calls,[])
    def test_missing_review_refuses_before_training_or_assets(self):
        path,anchors=self.packet();config=json.loads(path.read_text(encoding='utf-8'));review=T.read_json(config['review']);review['accepted']=False;config['review']=self.ref('bad_review.json',review);path,anchors=self.publish_config(config)
        with patch.object(T,'_helpers',side_effect=AssertionError('helpers before independent review')):
            with self.assertRaisesRegex(ValueError,'independently reviewed exact STUDY packet'):self.run_packet(path,anchors)
        self.assertEqual(self.calls,[])
    def test_unqualified_training_incomplete_M0_or_domain_evidence_refuses_factory(self):
        for key,update in [('training_qualification_check',{'qualified':False}),('training_qualification_check',{'full27_qualified':False}),('training_qualification_check',{'full_resumed9_qualified':False}),('m0_scoring_check',{'rows':383}),('m0_scoring_check',{'candidate_outputs_used':True}),('guard_enrollment_check',{'qualified_domain_ids':['task']})]:
            path,anchors=self.packet();config=json.loads(path.read_text(encoding='utf-8'));spec=config[key];proof=T.read_json(spec['expected_result']);proof.update(update);spec['request']=self.ref('changed_request.json',{'authored_evidence':proof});spec['expected_result']=self.ref('changed_result.json',proof);review=T.read_json(config['review']);review[key]=deepcopy(spec);config['review']=self.ref('changed_review.json',review);path,anchors=self.publish_config(config);self.calls=[]
            with self.subTest(key=key,update=update),self.assertRaises(ValueError):self.run_packet(path,anchors)
            self.assertNotIn('inert_backend_construct',self.calls)
    def test_empty_guard_bindings_refuse_before_factory(self):
        path,anchors=self.packet();config=json.loads(path.read_text(encoding='utf-8'));enrollment=T.read_json(config['guard_enrollment']);enrollment['scorer_bindings']={};enrollment=T.seal({k:v for k,v in enrollment.items() if k!='sha256'});config['guard_enrollment']=self.ref('empty_enrollment.json',enrollment);path,anchors=self.publish_config(config)
        with self.assertRaisesRegex(ValueError,'complete all-domain source-correct'):self.run_packet(path,anchors)
        self.assertNotIn('inert_backend_construct',self.calls)
    def test_bad_arm_and_qualification_resume_forbidden(self):
        path,anchors=self.packet();config=json.loads(path.read_text(encoding='utf-8'));config['arm']='M0';path,anchors=self.publish_config(config)
        with self.assertRaisesRegex(ValueError,'one adapted arm'):self.run_packet(path,anchors)
        path,anchors=self.packet(chunk=2,previous={'kind':'STATE1_QUALIFICATION_NOT_STUDY','terminal_costs':{}})
        with self.assertRaisesRegex(ValueError,'successful prior chunk descriptor'):self.run_packet(path,anchors)
        self.assertNotIn('inert_backend_construct',self.calls)
    def test_checkpoint_byte_drift_resume_refuses_before_assets(self):
        path,anchors=self.packet();report=self.run_packet(path,anchors);previous=report['protocol_result']['resume_descriptor'];Path(previous['envelope']['backend_descriptor']['path']).write_bytes(b'ALTERED_INERT_CHECKPOINT');path,anchors=self.packet(chunk=2,previous=previous);self.calls=[]
        with self.assertRaisesRegex(ValueError,'published bytes/hash changed'):self.run_packet(path,anchors)
        self.assertNotIn('inert_backend_construct',self.calls)
    def test_matched_parent_vector_and_exact27_update_end_are_required(self):
        for mutate,pattern in [(lambda r:r['arms']['M2']['totals']['parent_exposures'].update(fixture_parent_00=28),'matched source-parent exposure'),(lambda r:r['arms']['M1']['updates'].pop(),'chunk source occurrence order|27 actual updates')]:
            path,anchors=self.packet();config=json.loads(path.read_text(encoding='utf-8'));registration=deepcopy(self.registration);mutate(registration);registration=T.seal({k:v for k,v in registration.items() if k!='sha256'});config['training_registration']=self.ref('bad_authored_registration.json',registration)
            approval=T.read_json(config['approval']);approval['training_registration_value_sha256']=config['training_registration']['value_sha256'];config['approval']=self.ref('bad_source_approval.json',approval)
            review=T.read_json(config['review']);review['approval_file_sha256']=config['approval']['sha256'];review['training_registration_value_sha256']=config['training_registration']['value_sha256'];config['review']=self.ref('bad_source_review.json',review);path,anchors=self.publish_config(config)
            fixture=T._CPU_fixture(registration,self.pools,self.records,self.policy,capability=self.cap);factory,boundaries=self.actors();self.calls=[]
            with self.subTest(pattern=pattern),self.assertRaisesRegex(ValueError,pattern):T._CPU_run_registered(path,anchors,fixture,factory,boundaries,capability=self.cap)
            self.assertNotIn('inert_backend_construct',self.calls)
    def test_authoritative_latest_resume_costs_cannot_roll_back(self):
        path,anchors=self.packet();previous=self.run_packet(path,anchors)['protocol_result']['resume_descriptor'];path,anchors=self.packet(chunk=2,previous=previous);config=json.loads(path.read_text(encoding='utf-8'));latest=T.read_json(config['latest_terminal_costs']);latest['protocol']['optimizer_step']['attempted']-=1;latest['protocol']['optimizer_step']['returned']-=1;config['latest_terminal_costs']=self.ref('rollback_latest.json',latest);path,anchors=self.publish_config(config);self.calls=[]
        with self.assertRaisesRegex(ValueError,'latest terminal costs required; rollback forbidden'):self.run_packet(path,anchors)
        self.assertNotIn('inert_backend_construct',self.calls)
    def test_staging_native_gate_and_fixture_capability_separation(self):
        with patch.object(T,'_helpers',side_effect=AssertionError('no native data/helpers before stage gate')):
            with self.assertRaisesRegex(ValueError,'staged STUDY worker has no native authority'):T.run_registered('not_selected',{})
            with self.assertRaisesRegex(ValueError,'explicit private authored CPU'):T._CPU_run_registered(None,{},self.fixture,None,{},capability=object())
        path,anchors=self.packet()
        with patch.object(T,'HERE',T.EXPECTED):
            with self.assertRaisesRegex(ValueError,'fixture/native separation'):T.run_registered(path,anchors)
    def test_owned_profile_removal_attempted_even_after_evidence_failure(self):
        events=[];primary=RuntimeError('authored evidence boundary failed')
        def evidence(*args):events.append('evidence');raise primary
        class InertHandle:
            def remove(self):events.append('backend_hook_remove')
        backend=object.__new__(self.modules['study_torch_backend'].StudyTorchBackend);backend.activation_hooks={'handles':[InertHandle()]}
        factory=object.__new__(T._NativeFactory);factory.boundaries={'evidence':evidence};factory.backend=backend;factory.profiles=[{'snapshot':lambda:{'CPU_authored':True},'remove':lambda:events.append('profile_remove')}]
        with self.assertRaisesRegex(RuntimeError,'owned STUDY backend/profile cleanup failed') as caught:factory.close()
        self.assertEqual(events,['evidence','profile_remove','backend_hook_remove']);self.assertIs(caught.exception.__cause__,primary);self.assertEqual(caught.exception.cleanup_errors,(primary,));self.assertIsNone(factory.backend);self.assertEqual(factory.profiles,[])
        factory.close();self.assertEqual(events,['evidence','profile_remove','backend_hook_remove'])
    def test_profile_and_backend_cleanup_errors_retained_and_external_cleanup_attempted(self):
        events=[];primary=RuntimeError('authored profile snapshot failed');removal=RuntimeError('authored profile removal failed');backend_error=RuntimeError('authored backend hook removal failed')
        def snapshot():events.append('profile_snapshot');raise primary
        def remove():events.append('profile_remove');raise removal
        class InertHandle:
            def remove(self):events.append('backend_hook_remove');raise backend_error
        backend=object.__new__(self.modules['study_torch_backend'].StudyTorchBackend);backend.activation_hooks={'handles':[InertHandle()]}
        factory=object.__new__(T._NativeFactory);factory.boundaries={'evidence':lambda *args:events.append('evidence')};factory.backend=backend;factory.profiles=[{'snapshot':snapshot,'remove':remove}]
        schedule=ModuleType('CPU_authored_cleanup_schedule');schedule._inputs=lambda:(None,None,{'M1':{}},None)
        chunks=ModuleType('CPU_authored_cleanup_chunk');chunks.run_chunk=lambda *args,**kwargs:{'CPU_authored':True}
        prepared={'modules':{'study_schedule':schedule,'study_native_chunk':chunks},'config':{'arm':'M1','chunk':1},'registration':{},'previous':None,'latest':None}
        boundaries={'call':lambda name,function:function(),'guard':lambda stage:None,'preserve':lambda event:None,'admission':lambda request:None,'cleanup':lambda:events.append('external_owned_cleanup')}
        with self.assertRaisesRegex(RuntimeError,'owned STUDY backend/profile cleanup failed') as caught:T._run(prepared,factory,boundaries)
        self.assertEqual(events,['profile_snapshot','profile_remove','backend_hook_remove','external_owned_cleanup']);self.assertIs(caught.exception.__cause__,primary);self.assertEqual(caught.exception.cleanup_errors,(primary,removal,backend_error));self.assertIsNone(factory.backend);self.assertEqual(factory.profiles,[])
    def test_real_data_model_tokenizer_and_capture_read_traps(self):
        for path in (PACKAGE/'DEV_FINISH_POLICY_001.json',PACKAGE/'data/crossed_source2_v1_20261003/train_M1.jsonl',PACKAGE/'evidence/fresh_dev_compact002_20261003/panel.json',PACKAGE/'captured_native_response.json',PACKAGE/'model.safetensors',PACKAGE/'tokenizer.json'):
            for reader in (lambda p:builtins.open(p,'rb'),lambda p:io.open(p,'rb'),lambda p:p.open('rb'),lambda p:p.read_bytes()):
                with self.subTest(path=path.name),self.assertRaisesRegex(AssertionError,'real data/model/captures/scorer read forbidden'):reader(path)

if __name__=='__main__':
    names=('study_training.py','test_study_training.py');before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names};start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(StudyControls));after={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
    receipt=T.seal({'kind':'AUTHORED_STUDY_COMPOSITION_CPU_CONTROLS_V1','tests':result.testsRun,'success':result.wasSuccessful(),'seconds':time.monotonic()-start,'source_before':before,'source_after':after,'source_unchanged':before==after,'all_actual_read_paths_and_counts':dict(sorted(collections.Counter(StudyControls.reads).items())),'total_monitored_reads':len(StudyControls.reads),'forbidden_import_attempts':StudyControls.imports,'intentional_real_path_denials':len(StudyControls.denials),'unexpected_denials':len(StudyControls.denials)!=24,'fixtures':'Authored atomic R/S token arrays,20 parent vectors,27 updates/540 occurrences perarm; original scalar CPU fixture backend extracted from source. No real data or model math.','native_wiring_exercised':False,'native_PASS':False,'training_performed':False,'scoring_performed':False,'quality_gain_established':False,'limits':['Selected CPU proof auditor deliberately inert; demonstrates source/byte/request/result joins only, never genuine human approval or scorer/native qualification.','CPU fixtures execute unchanged original run_chunk and resume/publication/recipe code with explicit authored schedule supplier; original Torch backend is source imported only.','Native source can run only at promoted path with a genuine separately reviewed owned boundary and independent qualification/M0/guard adapters; no such packet is supplied.']})
    path=HERE/'CPU_CONTROLS_004.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'path':str(path),'sha256':receipt['sha256'],'tests':result.testsRun,'success':result.wasSuccessful(),'source_unchanged':before==after,'unexpected_denials':receipt['unexpected_denials']}))
    raise SystemExit(0 if result.wasSuccessful() and before==after and not receipt['unexpected_denials'] else 1)
