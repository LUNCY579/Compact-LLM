"""Actual authored producer receipts -> independent post-Job CPU consumer."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import ModuleType
import hashlib,json,threading,time,unittest

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
PRODUCER=PACKAGE/'artifacts/study_owned_runtime_preparation_work/test_owned_study.py'
m=ModuleType('authored_actual_producer_controls');m.__file__=str(PRODUCER);exec(compile(PRODUCER.read_bytes(),str(PRODUCER),'exec'),m.__dict__);m.HERE=HERE

class TerminalControls(m.OwnedControls):
    @classmethod
    def setUpClass(cls):
        super().setUpClass();cls.A=m.module(HERE/'study_terminal_audit.py','independent_STUDY_terminal_consumer')
    def descriptor(self,path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def packet(self,arm='M1',chunk=1,previous=None):
        base_root=self.root;self.root=base_root/('scope_'+arm+'_'+str(chunk)+'_'+str(time.monotonic_ns()));self.root.mkdir()
        context=self.context();context.prepared['config'].update(arm=arm,chunk=chunk);context.prepared['recipe']=context.prepared['modules']['study_native_chunk'].recipe(self.registration,arm,context.prepared['modules']['study_native_chunk'].NATIVE_SCOPE)
        context.prepared['previous']=deepcopy(previous);context.prepared['latest']=None if previous is None else deepcopy(previous['terminal_costs'])
        config={'scope':self.A.SCOPE,'fixture_only':True,'arm':arm,'chunk':chunk,'source_pins':{str(PACKAGE/'auditor.py'):hashlib.sha256((PACKAGE/'auditor.py').read_bytes()).hexdigest()},'previous_resume':None,'latest_terminal_costs':None}
        prior_ref=prior_latest_ref=None
        if previous is not None:
            prior_ref=self.ref('configured_prior_resume.json',previous);prior_latest_ref=self.ref('configured_prior_latest.json',previous['terminal_costs']);config.update(previous_resume=prior_ref,latest_terminal_costs=prior_latest_ref)
        ref=self.ref('authored_engine_config.json',config);context.value['engine_config']=ref;context.value['engine_anchors']={'config_file_sha256':ref['sha256'],'config_value_sha256':ref['value_sha256']};context.original=self.T.digest(context.value)
        output=self.owner_files(context)
        heartbeat_source=m.ast.parse((PACKAGE/'native_supervisor.py').read_bytes());fn=next(n for n in heartbeat_source.body if isinstance(n,m.ast.FunctionDef) and n.name=='_heartbeat');ns={'Path':Path,'time':time,'json':json,'os':m.os,'HEARTBEAT_MAX_BYTES':1024**2};exec(compile(m.ast.Module(body=[fn],type_ignores=[]),str(PACKAGE/'native_supervisor.py'),'exec'),ns);stop=threading.Event()
        def beat():
            while not stop.wait(.1):ns['_heartbeat'](output,6001)
        thread=threading.Thread(target=beat);thread.start()
        def same_original_factory(prepared,boundaries):
            tree=m.ast.parse(m.FAKE.read_bytes());definition=next(n for n in tree.body if isinstance(n,m.ast.ClassDef) and n.name=='QuadraticAdam');ns={'deepcopy':deepcopy,'Path':Path,'hashlib':hashlib,'json':json,'os':m.os,'math':m.math,'random':m.random,'chunk':prepared['modules']['study_native_chunk'],'schedule':prepared['modules']['study_schedule']};exec(compile(m.ast.Module(body=[definition],type_ignores=[]),str(m.FAKE),'exec'),ns)
            class Lazy:
                execution_scope=prepared['modules']['study_native_chunk'].NATIVE_SCOPE
                def __call__(inner,**kwargs):return ns['QuadraticAdam'](arm,prepared['recipe'],kwargs['call'],kwargs['guard'],scope=inner.execution_scope)
            return Lazy()
        try:report=self.T.worker_body(context,cpu=self.host(),factory=same_original_factory)
        finally:stop.set();thread.join(2)
        self.assertEqual(report['protocol_result']['status'],'COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED')
        # Authored terminal host observations carry no native authority. The
        # original worker/journal/archive/publications above are real CPU outputs.
        births=[{'pid':7001,'creation_filetime_100ns':116444736010000001},{'pid':7002,'creation_filetime_100ns':116444736010000002}]
        identity={**births[0],'create_time':123.0,'job_name':'Local\\SYNTH_NQ_JOB_'+'a'*32,'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False}
        def row(start,*,gpu=False,owned=False):
            sample={'sample_started_monotonic':start,'sample_completed_monotonic':start+.01,'host_available_bytes':6*1024**3,'disk_free_bytes':6*1024**3,'worker_tree_rss_bytes':512 if owned else 0,'GPU_index':0,'gpu_uuid':'CPU_AUTHORED_GPU_NOT_OBSERVED','gpu_observation_id':'CPU_'+str(start),'gpu_total_bytes':8*1024**3,'gpu_free_bytes':6*1024**3,'gpu_sample_started_monotonic':start,'gpu_sample_completed_monotonic':start+.01}
            if owned:
                sample['owned_process_identities']=[{**b,'create_time':123.0} for b in births];sample['owned_process_samples']=[{**b,'create_time':123.0,'rss_bytes':256,'sample_started_monotonic':start,'sample_completed_monotonic':start+.005} for b in births]
            sample['kind']='GPU' if gpu else 'CPU';return sample
        observations=[{**row(10.0,owned=True),'phase':'worker_suspended_startup'},row(10.1,owned=True),row(10.0,gpu=True),row(10.1,gpu=True)];self.T.write_new(output/'RESOURCE_OBSERVATIONS.json',observations)
        # Replace the authored heartbeat with exact fresh-cadence fixture rows;
        # no original native evidence is overwritten by this fixture operation.
        (output/'supervisor_heartbeat.jsonl').write_text(''.join(json.dumps({'unix_seconds':100+i*.1,'supervisor_pid':6001})+'\n' for i in range(3)),encoding='utf-8')
        self.T.write_new(output/'worker_GPU_final.json',{'CUDA_initialized':True,'peak_allocated_bytes':1,'peak_reserved_bytes':2,'free_bytes_final':6*1024**3,'CPU_authored_no_GPU_observation':True})
        lease_root=self.root/'_native_gpu_leases';lease=lease_root/('c'*32);lease.mkdir(parents=True)
        bounds={k:v for k,v in self.mods['native_contract'].RESOURCE_BOUNDS.items() if k not in ('CPU_sample_max_age_seconds','GPU_sample_max_age_seconds','GPU_probe_interval_seconds','GPU_probe_timeout_seconds','admission_poll_seconds','admission_seconds')}
        sup={'status':'SUPERVISED_WORKER_EXITED_ZERO','returncode':0,'partial_outputs_frozen':True,'admission_only':False,'bounds':bounds,'cadence':{'CPU_watchdog_seconds':.1,'CPU_freshness_seconds':.2,'GPU_query_interval_seconds':.5,'GPU_query_timeout_seconds':2.0,'GPU_freshness_seconds':2.0,'admission_interval_seconds':2.0,'admission_deadline_seconds':30.0,'admission_samples':2},'config_sha256':ref['sha256'],'worker_identity':identity,'worker_resume_monotonic':10.02,'deadline_monotonic':607.0,'lease_path':str(lease),'elapsed_seconds':5.0,'owned_worker_identity_currently_alive':False,'unjoined_sampler_threads':[],'admission_observations':[row(7.0),row(9.0)],'dispatch_observation':row(9.0)}
        for key in ('worker_attempted','worker_launched','worker_resumed','owned_handle_alive_verified_after_launch','preflight_revalidated_before_current_observations','job_closed_kill_on_close','sampler_threads_terminal_verified','gpu_telemetry_cleanup_verified'):sup[key]=True
        sup['partial_outputs']=[{'path':p.name,**self.A.info(p)} for p in sorted(output.iterdir())]
        self.T.write_new(output/'SUPERVISOR_RECEIPT.json',sup);(lease/'terminal.json').write_bytes((output/'SUPERVISOR_RECEIPT.json').read_bytes())
        self.T.write_new(lease/'worker.json',identity);self.T.write_new(lease/'resume.json',{'worker_identity':identity,'external_monitor_ready_before_resume':True,'resumed_monotonic':10.02,'deadline_monotonic':607.0});self.T.write_new(lease/'reservation.json',{'lease_id':lease.name,'supervisor':{'pid':6001,'create_time':123.0},'output_dir':str(output),'config_sha256':ref['sha256'],'GPU_index':0})
        approval=self.ref('audit_authored_approval.json',{'scope':self.A.SCOPE,'fixture_only':True,'approved':True,'source_pins':config['source_pins']});review=self.ref('audit_authored_review.json',{'accepted':True,'source_pins':config['source_pins'],'approval_file_sha256':approval['sha256']})
        registration=self.ref('audited_training_registration.json',self.registration);latest=self.ref('audited_latest_costs.json',report['protocol_result']['costs'])
        packet={'fixture_only':True,'output_directory':str(output),'lease_root':str(lease_root),'config':self.descriptor(Path(ref['path'])),'approval':self.descriptor(Path(approval['path'])),'review':self.descriptor(Path(review['path'])),'sources':{str(PACKAGE/'auditor.py'):self.descriptor(PACKAGE/'auditor.py')},'worker':self.descriptor(output/'worker_result.json'),'supervisor':self.descriptor(output/'SUPERVISOR_RECEIPT.json'),'training_registration':self.descriptor(Path(registration['path'])),'latest_terminal_costs':self.descriptor(Path(latest['path'])),'original_auditor_sha256':config['source_pins'][str(PACKAGE/'auditor.py')],'raw_files':{p.name:self.A.info(p) for p in output.iterdir()}}
        if previous is not None:packet.update(previous_resume=self.descriptor(Path(prior_ref['path'])),previous_latest_terminal_costs=self.descriptor(Path(prior_latest_ref['path'])))
        chunks=context.prepared['modules']['study_native_chunk'];host=type('AuthoredOnlyHost',(),{'is_alive':lambda self,birth:False})();self.root=base_root;return packet,chunks,host
    def test_both_arms_three_fresh_workers_original_chain_and_final27_540(self):
        for arm,total,supervised in [('M1',130562,5614),('M2',130574,5615)]:
            previous=None;source_records=[]
            for chunk in (1,2,3):
                packet,chunks,host=self.packet(arm,chunk,previous);result=self.A._CPU_consume(packet,chunks,host,capability=self.A._CPU_capability());worker=self.A.read_ref(packet['worker']);previous=worker['protocol_result']['resume_descriptor'];source_records.append(previous)
                self.assertEqual((result['arm'],result['chunk']),(arm,chunk));self.assertFalse(result['native_PASS'])
            final=previous['envelope']['binding'];self.assertEqual((final['completed_updates'],final['consumed_occurrences'],final['input_tokens'],final['supervised_tokens'],final['learning_rate']),(27,540,total,supervised,0.0))
            self.assertEqual(previous['terminal_costs']['protocol']['backward']['returned'],540);self.assertEqual(previous['terminal_costs']['protocol']['optimizer_step']['returned'],27);self.assertEqual(len(source_records),3)
    def test_config_fields_kind_arm_chunk_match_selected_engine_literal(self):
        ref=self.descriptor(m.ENGINE);fields=self.A.config_fields_from_engine(ref);config={name:None for name in fields};config.update(kind='REGISTERED_STUDY_TRAINING_CHUNK_V1',scope=self.A.SCOPE,arm='M1',chunk=1);config=self.A.seal({k:v for k,v in config.items() if k!='sha256'});self.A.validate_config_shape(config,ref)
        for change in (lambda c:c.pop('resources'),lambda c:c.update(extra='foreign'),lambda c:c.update(kind='QUALIFICATION_ONLY'),lambda c:c.update(arm='M0'),lambda c:c.update(chunk=True),lambda c:c.update(chunk=4)):
            candidate=deepcopy(config);change(candidate);candidate=self.A.seal({k:v for k,v in candidate.items() if k!='sha256'})
            with self.subTest(),self.assertRaises(ValueError):self.A.validate_config_shape(candidate,ref)
    def test_prior_latest_rollback_altered_record_and_qualification_state_reject(self):
        first,chunks,host=self.packet();previous=self.A.read_ref(first['worker'])['protocol_result']['resume_descriptor'];packet,chunks,host=self.packet(chunk=2,previous=previous)
        targets=[('previous_resume',lambda v:v.update(kind='STATE1_QUALIFICATION_NOT_STUDY')),('previous_latest_terminal_costs',lambda v:v['protocol']['optimizer_step'].update(attempted=8,returned=8)),('latest_terminal_costs',lambda v:v['protocol']['backward'].update(attempted=359,returned=359))]
        for key,change in targets:
            path=Path(packet[key]['path']);original=path.read_bytes();value=json.loads(original);change(value);path.write_text(json.dumps(value),encoding='utf-8');candidate=deepcopy(packet);candidate[key]=self.descriptor(path)
            with self.subTest(key=key),self.assertRaises(ValueError):self.A._CPU_consume(candidate,chunks,host,capability=self.A._CPU_capability())
            path.write_bytes(original)
    def test_actual_CPU_producer_original_publications_resources_and_terminal_consumer(self):
        packet,chunks,host=self.packet();result=self.A._CPU_consume(packet,chunks,host,capability=self.A._CPU_capability());self.assertEqual(result['chunk'],1);self.assertEqual(len(result['observed_birth_deaths']),2);self.assertFalse(result['native_PASS']);self.assertFalse(result['qualified_resume_authority_issued'])
    def test_malformed_worker_seal_and_live_birth_reject(self):
        packet,chunks,host=self.packet();worker=Path(packet['worker']['path']);original=worker.read_bytes();value=json.loads(original);value['quality_gain_established']=True;worker.write_text(json.dumps(value),encoding='utf-8');packet['worker']=self.descriptor(worker)
        with self.assertRaisesRegex(ValueError,'canonical whole worker'):self.A._CPU_consume(packet,chunks,host,capability=self.A._CPU_capability())
        worker.write_bytes(original);packet['worker']=self.descriptor(worker);host.is_alive=lambda birth:True
        with self.assertRaisesRegex(ValueError,'birth still live/unknown'):self.A._CPU_consume(packet,chunks,host,capability=self.A._CPU_capability())
    def test_raw_omission_source_drift_cleanup_and_partial_SUP_reject(self):
        packet,chunks,host=self.packet()
        for mutate in (lambda p:p['raw_files'].pop('events.jsonl'),lambda p:p['sources'][str(PACKAGE/'auditor.py')].update(sha256='0'*64)):
            candidate=deepcopy(packet);mutate(candidate)
            with self.subTest(),self.assertRaises(ValueError):self.A._CPU_consume(candidate,chunks,host,capability=self.A._CPU_capability())
        sup_path=Path(packet['supervisor']['path']);sup=json.loads(sup_path.read_bytes());sup['status']='OWNED_WORKER_ABORTED';sup_path.write_text(json.dumps(sup),encoding='utf-8');packet['supervisor']=self.descriptor(sup_path)
        with self.assertRaisesRegex(ValueError,'successful original complete frozen'):self.A._CPU_consume(packet,chunks,host,capability=self.A._CPU_capability())
    def test_public_stage_gate_before_receipts_or_API(self):
        with self.assertRaisesRegex(ValueError,'staged post-Job STUDY auditor'):self.A.audit_registered({})
    def test_fresh_host_callback_mutation_cannot_accept_rebased_evidence(self):
        packet,chunks,host=self.packet();sup=json.loads(Path(packet['supervisor']['path']).read_bytes());lease=Path(sup['lease_path'])
        for target in (Path(packet['approval']['path']),lease/'reservation.json',Path(packet['output_directory'])/'foreign_callback_file'):
            existed=target.exists();original=target.read_bytes() if existed else None
            def mutate(birth):target.write_bytes(b'CALLBACK_TAMPER_AUTHORED_ONLY');return False
            host.is_alive=mutate
            with self.subTest(target=target.name),self.assertRaises(ValueError):self.A._CPU_consume(packet,chunks,host,capability=self.A._CPU_capability())
            if existed:target.write_bytes(original)
            else:target.unlink()

if __name__=='__main__':
    selected=['test_actual_CPU_producer_original_publications_resources_and_terminal_consumer','test_malformed_worker_seal_and_live_birth_reject','test_raw_omission_source_drift_cleanup_and_partial_SUP_reject','test_public_stage_gate_before_receipts_or_API','test_fresh_host_callback_mutation_cannot_accept_rebased_evidence','test_both_arms_three_fresh_workers_original_chain_and_final27_540','test_config_fields_kind_arm_chunk_match_selected_engine_literal','test_prior_latest_rollback_altered_record_and_qualification_state_reject'];before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('study_terminal_audit.py','test_study_terminal_audit.py')};start=time.monotonic();result=unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(TerminalControls(n) for n in selected));after={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in before};value={'kind':'INDEPENDENT_STUDY_TERMINAL_CPU_PRODUCER_CONSUMER_CONTROLS','tests':result.testsRun,'success':result.wasSuccessful(),'seconds':time.monotonic()-start,'source_before':before,'source_after':after,'source_unchanged':before==after,'read_paths_and_counts':dict(__import__('collections').Counter(TerminalControls.reads)),'total_monitored_reads':len(TerminalControls.reads),'forbidden_import_attempts':TerminalControls.imports,'fixture_only':True,'native_PASS':False,'native_API_run':False,'scoring_authorized':False,'quality_gain_established':False,'limits':['Original corrected CPUworker/journal/scientific archives/publications are produced in this suite; SUP/resource/lease/death observations explicitly authored, not native.','No actual model/data/qualification/scoring or native API read.','Native fresh source-bound Sampler and full genuinely qualified engine packet remain unexecuted; no resume qualification authority issued.']};value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();path=HERE/'CPU_CONTROLS_003.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'path':str(path),'sha256':value['sha256'],'tests':result.testsRun,'success':result.wasSuccessful(),'source_unchanged':before==after}));raise SystemExit(0 if result.wasSuccessful() and before==after else 1)
