"""One actual mock-producer batch into the saved post-Job CPU consumer."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import ast,builtins,hashlib,io,json,os,time,unittest

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
PRODUCER=PACKAGE/'evidence/long_guard_owned_runtime_reviewed_cpu_20261007/current001'
CORE=PACKAGE/'evidence/long_guard_collection_reviewed_cpu_20261007/current001'
def module(path,name):
    value=ModuleType(name);value.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),value.__dict__);return value

class TerminalControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reads=[];cls.denials=[];cls.imports=[];realio,realopen,realimport=io.open,builtins.open,builtins.__import__
        def read(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,bytes,Path)) and not any(c in str(mode) for c in ('w','a','x','+')):
                p=Path(path).resolve();cls.reads.append(str(p))
                if not (p.is_relative_to(HERE) or p.is_relative_to(PRODUCER) and p.suffix=='.py' or p.is_relative_to(CORE) and p.suffix=='.py' or p.parent==PACKAGE and p.suffix=='.py' or p.parent==PACKAGE/'tests' and p.suffix=='.py'):
                    cls.denials.append(str(p));raise AssertionError('actual model/data/native/qualification inputs forbidden')
            return realio(path,mode,*args,**kwargs)
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','transformers','tokenizers','ctypes','psutil','safetensors','numpy'}:
                cls.imports.append(name);raise AssertionError('framework/native import forbidden')
            return realimport(name,*args,**kwargs)
        cls.traps=[patch('io.open',side_effect=read),patch.object(builtins,'open',side_effect=lambda p,*a,**k:read(p,*a,**k) if isinstance(p,(str,bytes,Path)) else realopen(p,*a,**k)),patch.object(builtins,'__import__',side_effect=imports)]
        for trap in cls.traps:trap.start()
        cls.A=module(HERE/'long_guard_terminal_audit.py','terminal_consumer')
        cls.P=module(PRODUCER/'test_long_guard_owned.py','original_owned_controls')
        # Existing actor constructors are compiled with traps already active.
        # Redirect only runtime source locations; producer source bytes remain
        # original and the underlying mock scientific call is executed below.
        original_source=cls.P.source
        cls.P.source=lambda path,name:original_source(PRODUCER/path.name if path.name in ('long_guard_owned_runtime.py','long_guard_owned_dispatch.py') else path,name)
        cls.P.OwnedControls.setUpClass()
        cls.P.OwnedControls.tearDownClass()
        cls.P.HERE=HERE
        cls.L=cls.P.OwnedControls.L;cls.R=cls.P.OwnedControls.R;cls.component=cls.P.OwnedControls.component
        cls.O=cls.P.OwnedControls
    @classmethod
    def tearDownClass(cls):
        for trap in reversed(cls.traps):trap.stop()
        assert not cls.imports,cls.imports
    def setUp(self):
        self.actor=self.O('test_actual_original_terminal_worker_with_inert_actors');self.actor.setUp();self.addCleanup(self.actor.doCleanups);self.root=self.actor.root
    def ref(self,path,*,json_value=True):
        raw=path.read_bytes();value={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        if json_value:value['value_sha256']=self.A.digest(self.L._json(raw))
        return value
    def write(self,path,value):
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n')
        return self.ref(path)
    def packet(self,tokens=(151645,)):
        source,plan,config,torch,model,decoder,scientific,seen=self.actor.prepared(13,tokens)
        config.update(scope=self.A.SCOPE,fixture_only=True,source_pins={str(PRODUCER/'long_guard_owned_runtime.py'):hashlib.sha256((PRODUCER/'long_guard_owned_runtime.py').read_bytes()).hexdigest(),str(CORE/'long_guard_collection.py'):hashlib.sha256((CORE/'long_guard_collection.py').read_bytes()).hexdigest(),str(PACKAGE/'auditor.py'):hashlib.sha256((PACKAGE/'auditor.py').read_bytes()).hexdigest()})
        config_ref=self.write(self.root/'config.json',config);source_ref=self.write(self.root/'source.json',source);plan_ref=self.write(self.root/'plan.json',plan)
        leaf=self.root/'owned';leaf.mkdir()
        cpu=SimpleNamespace(os=os,time=time,psutil=SimpleNamespace(),owner={},guard=seen.append,disk_free=4*1024**3,torch=torch,model=model,decoder=decoder,scientific=scientific)
        report=self.R.worker_body(Path(config_ref['path']),leaf,config,source,plan,self.L,self.component,cpu=cpu,capability=self.O.cap)
        self.assertEqual(report['status'],'COMPLETE_PENDING_INDEPENDENT_LONG_DECODED_AUDIT')
        # These are explicitly authored host/ownership/resource/lease rows. They
        # describe no actual process birth, GPU observation or native launch.
        births=[{'pid':7101,'creation_filetime_100ns':116444736010000001},{'pid':7102,'creation_filetime_100ns':116444736010000002}]
        identity={**births[0],'create_time':123.0,'job_name':'Local\\SYNTH_NQ_JOB_'+'a'*32,'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False}
        self.write(leaf/'supervisor_ownership.json',{'worker_pid':7101,'root_worker_pid':7101,'job_name':identity['job_name'],'supervisor_pid':6101,'supervisor_create_time':123.0,'config_sha256':config_ref['sha256']})
        self.write(leaf/'stage_worker_ownership_verified.json',{'actual_worker_pid':7102,'launcher_root_pid':7101,'supervisor_pid':6101,'job_name':identity['job_name'],'owned_job_membership_verified':True,'query_job_handle_closed_before_framework_import':True,'CPU_authored_no_native_ownership':True})
        def row(start,*,gpu=False,owned=False):
            sample={'sample_started_monotonic':start,'sample_completed_monotonic':start+.01,'host_available_bytes':6*1024**3,'disk_free_bytes':6*1024**3,'worker_tree_rss_bytes':512 if owned else 0,'GPU_index':0,'gpu_uuid':'CPU_AUTHORED_NO_GPU_OBSERVATION','gpu_observation_id':'CPU_'+str(start),'gpu_total_bytes':8*1024**3,'gpu_free_bytes':6*1024**3,'gpu_sample_started_monotonic':start,'gpu_sample_completed_monotonic':start+.01,'kind':'GPU' if gpu else 'CPU'}
            if owned:
                sample['owned_process_identities']=[{**b,'create_time':123.0} for b in births]
                sample['owned_process_samples']=[{**b,'create_time':123.0,'rss_bytes':256,'sample_started_monotonic':start,'sample_completed_monotonic':start+.005} for b in births]
            return sample
        self.write(leaf/'RESOURCE_OBSERVATIONS.json',[{**row(10.0,owned=True),'phase':'worker_suspended_startup'},row(10.1,owned=True),row(10.0,gpu=True),row(10.1,gpu=True)])
        (leaf/'supervisor_heartbeat.jsonl').write_text(''.join(json.dumps({'unix_seconds':100+i*.1,'supervisor_pid':6101})+'\n' for i in range(3)),encoding='utf-8')
        (leaf/'worker_GPU_final.json').write_text(json.dumps({'CUDA_initialized':True,'peak_allocated_bytes':1,'peak_reserved_bytes':2,'free_bytes_final':6*1024**3,'CPU_authored_no_GPU_observation':True})+'\n',encoding='utf-8')
        lease_root=self.root/'_native_gpu_leases';lease=lease_root/('c'*32);lease.mkdir(parents=True)
        sup={'status':'SUPERVISED_WORKER_EXITED_ZERO','returncode':0,'partial_outputs_frozen':True,'admission_only':False,'bounds':{'worker_process_tree_RSS_GiB':2.0,'system_available_reserve_GiB':1.25,'preworker_available_GiB':3.3125,'GPU_free_at_admission_GiB':4.0,'GPU_free_during_work_GiB':.5,'disk_free_GiB':4.0,'whole_scope_seconds':600,'external_watchdog_milliseconds':100},'cadence':{'CPU_watchdog_seconds':.1,'CPU_freshness_seconds':.2,'GPU_query_interval_seconds':.5,'GPU_query_timeout_seconds':2.0,'GPU_freshness_seconds':2.0,'admission_interval_seconds':2.0,'admission_deadline_seconds':30.0,'admission_samples':2},'config_sha256':config_ref['sha256'],'worker_identity':identity,'worker_resume_monotonic':10.02,'deadline_monotonic':607.0,'lease_path':str(lease),'elapsed_seconds':5.0,'owned_worker_identity_currently_alive':False,'unjoined_sampler_threads':[],'admission_observations':[row(7.0),row(9.0)],'dispatch_observation':row(9.0)}
        for key in ('worker_attempted','worker_launched','worker_resumed','owned_handle_alive_verified_after_launch','preflight_revalidated_before_current_observations','job_closed_kill_on_close','sampler_threads_terminal_verified','gpu_telemetry_cleanup_verified'):sup[key]=True
        def info(path):return {k:v for k,v in self.ref(path,json_value=False).items() if k!='path'}
        sup['partial_outputs']=[{'path':p.name,**info(p)} for p in sorted(leaf.iterdir())];sup_ref=self.write(leaf/'SUPERVISOR_RECEIPT.json',sup)
        (lease/'terminal.json').write_bytes((leaf/'SUPERVISOR_RECEIPT.json').read_bytes())
        self.write(lease/'worker.json',identity);self.write(lease/'resume.json',{'worker_identity':identity,'external_monitor_ready_before_resume':True,'resumed_monotonic':10.02,'deadline_monotonic':607.0});self.write(lease/'reservation.json',{'lease_id':lease.name,'supervisor':{'pid':6101,'create_time':123.0},'output_dir':str(leaf),'config_sha256':config_ref['sha256'],'GPU_index':0})
        saved={'batch':self.ref(leaf/'stage_long_guard_batch.json'),'events':self.ref(leaf/'events.jsonl',json_value=False),'rows':[],'tokens':[]}
        for index,raw in enumerate(report['batch']['collection']['rows'],1):
            saved['rows'].append(self.ref(leaf/('stage_inference_row_%04d.json'%index)))
            saved['tokens'].append([self.ref(leaf/('stage_raw_row%04d_token%02d.json'%(index,step))) for step in range(1,len(raw['trace'])+1)])
        sources={p:self.ref(Path(p),json_value=False) for p in config['source_pins']}
        packet={'fixture_only':True,'output_directory':str(leaf),'lease_root':str(lease_root),'config':config_ref,'source':source_ref,'plan':plan_ref,'worker':self.ref(leaf/'worker_result.json'),'supervisor':sup_ref,'sources':sources,'core_source':sources[str(CORE/'long_guard_collection.py')],'auditor_source':sources[str(PACKAGE/'auditor.py')],'saved':saved,'raw_files':{p.name:info(p) for p in leaf.iterdir()}}
        host=type('AuthoredOnlyHost',(),{'is_alive':lambda self,birth:False})()
        return packet,host
    def consume(self,packet,host):return self.A._CPU_consume(packet,self.L,self.component,host,capability=self.A._CPU_capability())
    def test_actual_mock_worker_to_saved_consumer(self):
        packet,host=self.packet();result=self.consume(packet,host)
        self.assertEqual(result['physical_forward_dose']['attempted_calls'],66);self.assertEqual(len(result['observed_birth_deaths']),2)
        self.assertFalse(result['native_PASS']);self.assertEqual(result['source_packet_sha256'],self.L.read_ref(packet['worker'])['batch']['collection']['source_packet_sha256'])
    def test_native_refusal_and_real_input_traps(self):
        with self.assertRaisesRegex(ValueError,'staged'):self.A.audit_registered({'qualified':True})
        with self.assertRaisesRegex(ValueError,'private authored'):self.A._CPU_consume({},None,None,None,capability=object())
        for path in (PACKAGE/'LONG_CONTEXT_GUARD_POLICY.json',PACKAGE/'data/crossed_source2_v1_20261003/train.json',PACKAGE/'model.safetensors'):
            with self.assertRaisesRegex(AssertionError,'inputs forbidden'):path.read_bytes()
    def test_physical_dose_and_cleanup_contradictions_reject(self):
        packet,host=self.packet();path=Path(packet['worker']['path']);original=path.read_bytes()
        for mutate in (lambda v:v['forward_dose'].update(attempted_calls=64),lambda v:v['forward_dose']['operations']['zero_parity_adapter_forward'].update(input_positions=420),lambda v:v.update(cleanup_errors=[{'type':'authored_failure'}]),lambda v:v.update(backwards=True)):
            value=self.L._json(original);mutate(value);value=self.A.seal({k:v for k,v in value.items() if k!='sha256'});path.write_text(json.dumps(value)+'\n',encoding='utf-8');candidate=deepcopy(packet);candidate['worker']=self.ref(path)
            with self.subTest(),self.assertRaises(ValueError):self.consume(candidate,host)
        path.write_bytes(original)
    def test_partial_corruption_and_wrong_selected_path_reject(self):
        packet,host=self.packet();path=Path(packet['saved']['tokens'][0][0]['path']);original=path.read_bytes();value=self.L._json(original);value['costs']['attempted_forwards']=True;path.write_text(json.dumps(value)+'\n',encoding='utf-8');candidate=deepcopy(packet);candidate['saved']['tokens'][0][0]=self.ref(path)
        with self.assertRaises(ValueError):self.consume(candidate,host)
        path.write_bytes(original);candidate=deepcopy(packet);candidate['saved']['events']['path']=str(self.root/'wrong_events.jsonl')
        with self.assertRaisesRegex(ValueError,'Journal selected'):self.consume(candidate,host)
    def test_raw_omission_conflicting_birth_and_live_or_unknown_death(self):
        packet,host=self.packet();candidate=deepcopy(packet);candidate['raw_files'].pop('events.jsonl')
        with self.assertRaises(ValueError):self.consume(candidate,host)
        for alive in (True,None,0):
            host.is_alive=lambda birth,value=alive:value
            with self.subTest(alive=alive),self.assertRaisesRegex(ValueError,'birth live/unknown'):self.consume(packet,host)
        with self.assertRaisesRegex(ValueError,'conflicting'):self.A.births({'worker_identity':{'pid':1,'creation_filetime_100ns':2}},[{'kind':'CPU','owned_process_identities':[{'pid':1,'creation_filetime_100ns':3}]}])
    def test_fresh_death_callback_drift_cannot_accept_rebased_input(self):
        packet,host=self.packet();path=Path(packet['source']['path']);original=path.read_bytes()
        def drift(birth):path.write_bytes(b'CPU_CALLBACK_MUTATION');return False
        host.is_alive=drift
        with self.assertRaises(ValueError):self.consume(packet,host)
        path.write_bytes(original)

if __name__=='__main__':
    names=('long_guard_terminal_audit.py','test_long_guard_terminal_audit.py');before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names};start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(TerminalControls));after={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
    value={'kind':'AUTHORED_DECODED_LONG_POST_JOB_PRODUCER_CONSUMER_CPU_CONTROLS','tests':result.testsRun,'success':result.wasSuccessful(),'seconds':time.monotonic()-start,'source_before':before,'source_after':after,'source_unchanged':before==after,'read_paths_and_counts':dict(Counter(TerminalControls.reads)),'forbidden_import_attempts':TerminalControls.imports,'intentional_real_path_denials':len(TerminalControls.denials),'fixture_only':True,'native_PASS':False,'scoring_authorized':False,'native_qualification_established':False,'limits':['Actual saved mock runtime worker/Journal/token/final/batch producer is consumed; host birth/death/SUP/resources/leases are explicitly authored CPU fixtures.','No actual model/data/native API/scoring inputs are read.','Native post-Job qualification and dispatch remain closed.']}
    value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();path=HERE/('CPU_CONTROLS_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({'path':str(path),'sha256':value['sha256'],'tests':result.testsRun,'success':result.wasSuccessful(),'source_unchanged':before==after}));raise SystemExit(0 if result.wasSuccessful() and before==after else 1)
