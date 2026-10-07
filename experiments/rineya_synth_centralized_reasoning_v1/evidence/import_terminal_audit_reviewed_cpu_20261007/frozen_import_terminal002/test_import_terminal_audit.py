"""Actual import writer/raw replay + preserved files, authored SUP facts only."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import hashlib,json,shutil,sys,unittest
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
def load(path,name):
    m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
B=load(WORK/'test_import_program.py','CPU_terminal_original_import_program');T=load(HERE/'import_terminal_audit.py','CPU_selected_import_terminal_consumer')
class Controls(B.Controls):
    def build_terminal(self,failed=False):
        try:self.report=self.run_program(failed)
        except RuntimeError:pass
        self.preserved=self.root/'preserved';shutil.copytree(self.output,self.preserved);sources=dict(self.sources)
        for name in ('import_observation_auditor.py','measurement_replay.py'):
            path=WORK/name;sources[str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'role':'python_source'}
        config=T.A.seal({'scope':T.SCOPE,'resources':T.A.RESOURCES,'source_commit':'0'*40,'source_files':sources,'runtime_files':self.files,'helper_pins':{'native_supervisor.py':'c'*64}});runtime=T.A.seal({'source_commit':'0'*40,'files':self.files,'measurement_policy':self.policy,'measurement_grammar':self.grammar});reg=T.A.seal({'scope':T.SCOPE,'source_commit':'0'*40,'source_files':sources,'runtime_files':self.files,'config_body_sha256':config['sha256'],'runtime_body_sha256':runtime['sha256']});self.manifest={**config,'runtime_value':runtime}
        for name,value in (('config',config),('runtime',runtime),('reg',reg)):self.write(self.root/(name+'.json'),value)
        self.config_file=self.desc(self.root/'config.json')['sha256'];self.anchor={'config_path':str(self.root/'config.json'),'anchors':{'expected_config_file':self.config_file,'expected_config_body':config['sha256'],'registration_path':str(self.root/'reg.json'),'expected_registration_file':self.desc(self.root/'reg.json')['sha256'],'expected_registration_body':reg['sha256'],'runtime_path':str(self.root/'runtime.json'),'expected_runtime_file':self.desc(self.root/'runtime.json')['sha256'],'expected_runtime_body':runtime['sha256'],'expected_head':'1'*40}};self.refs={}
        for name in ('grant','owner_receipt'):
            doc=T.A.seal({'kind':'CPU_ONLY_DOCUMENT','role':name});path=self.root/(name+'.json');self.write(path,doc);self.refs[name]={'path':str(path),'file_sha256':self.desc(path)['sha256'],'body_sha256':doc['sha256']}
        self.anchor={'schema_version':2,**self.anchor,'grant_reference':self.refs['grant'],'owner_receipt_reference':self.refs['owner_receipt']};self.pair('OWNED_EXTERNAL_ANCHORS.json',self.anchor)
        records=T.A.parse_ledger((self.output/'PHASE_OBSERVATIONS.jsonl').read_bytes());first,last=records[0]['monotonic'],records[-1]['monotonic'];resume=first-.01;owner={'pid':101,'creation_filetime_100ns':1234567};full={**owner,'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False,'job_name':'CPU_TERMINAL_JOB'};self.lease=self.root/'lease';self.lease.mkdir()
        def obs(clock):return {'sample_started_monotonic':clock,'sample_completed_monotonic':clock,'host_available_bytes':8*T.A.GIB,'worker_tree_rss_bytes':0,'disk_free_bytes':64*T.A.GIB,'gpu_sample_started_monotonic':clock,'gpu_sample_completed_monotonic':clock,'gpu_free_bytes':16*T.A.GIB,'gpu_total_bytes':24*T.A.GIB,'GPU_index':0,'gpu_uuid':'CPU_FIXTURE_GPU','gpu_observation_id':str(clock)}
        sup={'config_sha256':self.config_file,'preflight':{'source_commit':'1'*40,'resources':T.A.RESOURCES},'worker_attempted':True,'worker_launched':True,'worker_resumed':True,'preflight_revalidated_before_current_observations':True,'job_closed_kill_on_close':True,'sampler_threads_terminal_verified':True,'gpu_telemetry_cleanup_verified':True,'partial_outputs_frozen':True,'owned_worker_identity_currently_alive':False,'unjoined_sampler_threads':[],'elapsed_seconds':last-resume+2.1,'worker_identity':full,'worker_resume_monotonic':resume,'deadline_monotonic':resume+600,'admission_observations':[obs(resume-2.02),obs(resume-.02)],'dispatch_observation':obs(resume-.01),'lease_path':str(self.lease),'status':'SUPERVISED_WORKER_FAILED' if failed else 'SUPERVISED_WORKER_EXITED_ZERO','returncode':2 if failed else 0};self.pair('SUPERVISOR_RECEIPT.json',sup);self.pair('supervisor_ownership.json',{'worker_pid':101,'supervisor_pid':100,'job_name':'CPU_TERMINAL_JOB','config_sha256':self.config_file})
        self.identity.update(source_commit='1'*40,owner_receipt_sha256=T.A.digest(json.loads((self.output/'supervisor_ownership.json').read_bytes())),owned_birth=owner)
        packet=json.loads((self.output/'IMPORT_LIBRARY_MEASUREMENT.json').read_bytes());packet['identity']=deepcopy(self.identity);packet=T.A.seal({k:v for k,v in packet.items() if k!='sha256'});self.pair('IMPORT_LIBRARY_MEASUREMENT.json',packet);worker=json.loads((self.output/'worker_result.json').read_bytes());worker.update(measurement_file_sha256=self.desc(self.output/'IMPORT_LIBRARY_MEASUREMENT.json')['sha256'],measurement_body_sha256=packet['sha256']);self.pair('worker_result.json',worker)
        samples=[];clock=resume
        while clock<last+.01:
            samples.extend([{'kind':'CPU','sample_started_monotonic':clock,'host_available_bytes':8*T.A.GIB,'worker_tree_rss_bytes':0,'owned_process_identities':[owner]},{'kind':'GPU','gpu_sample_started_monotonic':clock,'gpu_free_bytes':16*T.A.GIB,'gpu_total_bytes':24*T.A.GIB,'gpu_uuid':'CPU_FIXTURE_GPU'}]);clock+=.05
        for clock in (first,last):samples.extend([{'kind':'CPU','sample_started_monotonic':clock,'host_available_bytes':8*T.A.GIB,'worker_tree_rss_bytes':0,'owned_process_identities':[owner]},{'kind':'GPU','gpu_sample_started_monotonic':clock,'gpu_free_bytes':16*T.A.GIB,'gpu_total_bytes':24*T.A.GIB,'gpu_uuid':'CPU_FIXTURE_GPU'}])
        samples.sort(key=lambda r:r.get('sample_started_monotonic',r.get('gpu_sample_started_monotonic')));self.pair('RESOURCE_OBSERVATIONS.json',samples);leases={};snapshots={};(self.preserved/'source_snapshots').mkdir()
        for name,value in {'reservation.json':{'config_sha256':self.config_file},'worker.json':full,'resume.json':{'worker_identity':full,'external_monitor_ready_before_resume':True,'resumed_monotonic':resume},'terminal.json':sup}.items():self.write(self.lease/name,value);raw=(self.lease/name).read_bytes();(self.preserved/('lease_'+name)).write_bytes(raw);leases[name]={'source_path':str(self.lease/name),'path':'lease_'+name,**self.desc(self.lease/name)}
        for path,d in sources.items():name=hashlib.sha256(path.encode()).hexdigest()+Path(path).suffix;(self.preserved/'source_snapshots'/name).write_bytes(Path(path).read_bytes());snapshots[path]={'path':'source_snapshots/'+name,**d}
        self.preservation=T.A.seal({'kind':'PRESERVED_OWNED_TORCH_IMPORT_OBSERVATION','scope':T.SCOPE,'config_body_sha256':config['sha256'],'raw_files':{p.name:self.desc(p) for p in self.output.iterdir()},'source_files':sources,'source_snapshot_files':snapshots,'lease_files':leases,'current_death_observation':{'observed_monotonic':last+.1,'observed_utc':'2026-10-07T12:00:00+00:00','observations':[{'identity':owner,'currently_alive':False}],'supervisor_file_sha256':self.desc(self.output/'SUPERVISOR_RECEIPT.json')['sha256']},'worker_result_present':True,'native_authority_issued':False,'native_PASS':False});self.write(self.preserved/'PRESERVATION.json',self.preservation)
    def desc(self,path):raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    def write(self,path,value):path.write_text(T.A.canonical(value)+'\n')
    def pair(self,name,value):
        for root in (self.output,self.preserved):self.write(root/name,value)
    def audit(self):
        # A's selected compiled origin is the actual reviewed CPU source path;
        # package projection alone points the new consumer to that source.
        prior=T.PACKAGE;T.PACKAGE=WORK
        try:return T.audit_preserved(self.preserved,original_directory=self.output,manifest=self.manifest,anchors=self.anchor,expected_external_binding={'config_path':self.anchor['config_path'],'anchors':self.anchor['anchors']},expected_gate_references=self.refs,expected_preservation_file_sha256=self.desc(self.preserved/'PRESERVATION.json')['sha256'],expected_preservation_body_sha256=self.preservation['sha256'],measurement_identity=self.identity)
        finally:T.PACKAGE=prior
    def test_actual_writer_and_raw_replay_complete_terminal_unqualified(self):
        self.build_terminal();result=self.audit();self.assertEqual(result['status'],'IMPORT_MEASUREMENT_TERMINAL_VALIDATED_UNQUALIFIED');self.assertFalse(result['native_PASS'])
    def test_actual_failed_import_raw_and_terminal_preserved(self):
        self.build_terminal(True);result=self.audit();self.assertEqual(result['status'],'IMPORT_MEASUREMENT_PARTIAL_FAILURE_PRESERVED')
    def test_resealed_resource_or_death_mutation_refused(self):
        self.build_terminal();original=json.loads((self.output/'RESOURCE_OBSERVATIONS.json').read_bytes());rows=deepcopy(original);rows[0]['host_available_bytes']=0;self.pair('RESOURCE_OBSERVATIONS.json',rows);self.preservation['raw_files']['RESOURCE_OBSERVATIONS.json']=self.desc(self.output/'RESOURCE_OBSERVATIONS.json');self.preservation=T.A.seal({k:v for k,v in self.preservation.items() if k!='sha256'});self.write(self.preserved/'PRESERVATION.json',self.preservation)
        with self.assertRaises(ValueError):self.audit()
        self.pair('RESOURCE_OBSERVATIONS.json',original);self.preservation['raw_files']['RESOURCE_OBSERVATIONS.json']=self.desc(self.output/'RESOURCE_OBSERVATIONS.json');self.preservation['current_death_observation']['observations'][0]['currently_alive']=True;self.preservation=T.A.seal({k:v for k,v in self.preservation.items() if k!='sha256'});self.write(self.preserved/'PRESERVATION.json',self.preservation)
        with self.assertRaisesRegex(ValueError,'recordedownedbirthdead'):self.audit()
    def test_caller_runtime_policy_wrapper_cannot_override_selected_packet(self):
        self.build_terminal();self.manifest['runtime_value']['measurement_policy']=deepcopy(self.policy);self.manifest['runtime_value']['measurement_policy']['framework_transition']=[]
        with self.assertRaisesRegex(ValueError,'caller manifest wrapper'):self.audit()
    def test_resealed_missing_semantic_result_never_complete(self):
        self.build_terminal();worker=json.loads((self.output/'worker_result.json').read_bytes());worker['import_only_operations']['framework_import']['attempted']=2;self.pair('worker_result.json',worker);self.preservation['raw_files']['worker_result.json']=self.desc(self.output/'worker_result.json');self.preservation=T.A.seal({k:v for k,v in self.preservation.items() if k!='sha256'});self.write(self.preserved/'PRESERVATION.json',self.preservation)
        with self.assertRaisesRegex(ValueError,'provisional ownoperation'):self.audit()
        (self.output/'worker_result.json').unlink();(self.preserved/'worker_result.json').unlink();self.preservation['raw_files'].pop('worker_result.json');sup=json.loads((self.output/'SUPERVISOR_RECEIPT.json').read_bytes());sup['status']='SUPERVISED_WORKER_FAILED';sup['returncode']=2;self.pair('SUPERVISOR_RECEIPT.json',sup);self.write(self.lease/'terminal.json',sup);(self.preserved/'lease_terminal.json').write_bytes((self.lease/'terminal.json').read_bytes());self.preservation['raw_files']['SUPERVISOR_RECEIPT.json']=self.desc(self.output/'SUPERVISOR_RECEIPT.json');self.preservation['lease_files']['terminal.json'].update(self.desc(self.lease/'terminal.json'));self.preservation['current_death_observation']['supervisor_file_sha256']=self.desc(self.output/'SUPERVISOR_RECEIPT.json')['sha256'];self.preservation=T.A.seal({k:v for k,v in self.preservation.items() if k!='sha256'});self.write(self.preserved/'PRESERVATION.json',self.preservation);result=self.audit();self.assertEqual(result['status'],'IMPORT_MEASUREMENT_PARTIAL_FAILURE_PRESERVED');self.assertIn('worker_result.json',result['missing_semantic_files'])
if __name__=='__main__':
    names=('test_actual_writer_and_raw_replay_complete_terminal_unqualified','test_actual_failed_import_raw_and_terminal_preserved','test_resealed_resource_or_death_mutation_refused','test_resealed_missing_semantic_result_never_complete','test_caller_runtime_policy_wrapper_cannot_override_selected_packet');result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in names));v={'kind':'CPU_IMPORT_TERMINAL_PRESERVED_BYTE_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_import_writer_and_raw_A_replay':True,'SUP_resource_lease_death_records_authored_CPU_only':True,'native_API_or_framework_called':False,'native_authority_issued':False}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
