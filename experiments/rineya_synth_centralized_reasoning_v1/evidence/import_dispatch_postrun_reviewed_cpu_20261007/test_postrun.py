"""Actual pure terminal/preserved fixture with private inert lifecycle seams."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
from copy import deepcopy
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1];NATIVE=PACKAGE/'artifacts/torch_import_observation_work/native_candidate'
def load(p,name):
    m=ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
BASE=load(NATIVE/'test_import_terminal_audit.py','CPU_postrun_terminal_fixture');D=load(HERE/'import_dispatch_postrun.py','CPU_postrun_driver');T=load(HERE/'import_terminal_audit.py','CPU_postrun_identity_terminal');BASE.T=T
class Controls(BASE.Controls):
    def set_postrun(self,failed=False):
        self.build_terminal(failed);self.events=[];self.package=self.root/'rineya_synth_centralized_reasoning_v1';(self.package/'evidence').mkdir(parents=True);self.target=self.package/'evidence/prefix_input_owned_CPU_preserved';self.receipt=json.loads((self.output/'SUPERVISOR_RECEIPT.json').read_bytes());self.receipt['output_dir']=str(self.output);self.runtime=self.manifest['runtime_value'];self.value={'scope':D.SCOPE,'fixture_only':False,'config':{k:v for k,v in self.manifest.items() if k!='runtime_value'},'runtime':self.runtime,'registration':{'git_output_exceptions':{}},'expected_head':self.anchor['anchors']['expected_head']};self.binding={'config_path':self.anchor['config_path'],'anchors':self.anchor['anchors']};self.context=object()
        rel=Path('evidence/raw_library_pipeline_reviewed_cpu_20261006/prefix_library_evidence.py');dest=self.package/rel;dest.parent.mkdir(parents=True);dest.write_bytes((BASE.T.PACKAGE/rel).read_bytes())
        # CPU package layout adapter keeps explicit target source identities;
        # actual source A27/Rce8 bytes are copied, not rewritten.
        for name in ('import_observation_auditor.py','measurement_replay.py','measurement_guard.py'):
            p=self.package/name;p.write_bytes((BASE.WORK/name).read_bytes());d={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'python_source'};self.value['config']['source_files'][str(p)]=d
        self.value['config']['source_files'].pop(str(BASE.WORK/'import_observation_auditor.py'));self.value['config']['source_files'].pop(str(BASE.WORK/'measurement_replay.py'))
        producer={'path':str(self.package/'measurement_guard.py'),'descriptor':self.value['config']['source_files'][str(self.package/'measurement_guard.py')]}
        self.value['fixture_only']=True;self.expected_identity={**self.identity,'context_sha256':T.A.digest(self.value),'producer_source':producer,'fixture_only':True}
        records=[json.loads(line) for line in (self.output/'PHASE_OBSERVATIONS.jsonl').read_bytes().splitlines()];record=next(r for r in records if r['operation']=='ownership_verify' and r['event']=='returned');record['result']={'actual_worker_pid':self.identity['owned_birth']['pid']};raw=b''.join((T.A.canonical(r)+'\n').encode() for r in records)
        for root in (self.output,self.preserved):(root/'PHASE_OBSERVATIONS.jsonl').write_bytes(raw)
        final=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes());final['marker_bytes']=len(raw);self.pair('PHASE_ACCOUNTING_FINAL.json',final)
        packet=json.loads((self.output/'IMPORT_LIBRARY_MEASUREMENT.json').read_bytes());packet['identity']=self.expected_identity;packet=T.A.seal({k:v for k,v in packet.items() if k!='sha256'});self.pair('IMPORT_LIBRARY_MEASUREMENT.json',packet);worker=json.loads((self.output/'worker_result.json').read_bytes());worker.update(measurement_file_sha256=self.desc(self.output/'IMPORT_LIBRARY_MEASUREMENT.json')['sha256'],measurement_body_sha256=packet['sha256']);self.pair('worker_result.json',worker)
        self.packet_sync();exceptions=self.value['registration']['git_output_exceptions'];exceptions[str(self.target/'PRESERVATION.json')]={'role':'evidence_output'}
        for name in self.preservation['raw_files']:exceptions[str(self.target/name)]={'role':'evidence_output'}
        for name in ('reservation.json','worker.json','resume.json','terminal.json'):exceptions[str(self.target/('lease_'+name))]={'role':'evidence_output'}
        for path in self.value['config']['source_files']:exceptions[str(self.target/'source_snapshots'/(hashlib.sha256(path.encode()).hexdigest()+Path(path).suffix))]={'role':'evidence_source_snapshot','source_path':path}
        # Recompute qualified context digest after registered target selection.
        self.expected_identity['context_sha256']=T.A.digest(self.value);self.update_raw_identity(self.expected_identity)
        self.contract=SimpleNamespace(context_value=lambda token:self.value,revalidate_context=lambda token:self.value,parse=json.loads,read_bytes=lambda p:Path(p).read_bytes(),digest=T.A.digest)
        def preserve(output,target,**kwargs):
            self.events.append('preserve');shutil_copy(self.preserved,target);return self.preservation
        self.supervisor=SimpleNamespace(preserve_terminal=preserve)
    def packet_sync(self):
        self.value['config']=T.A.seal({k:v for k,v in self.value['config'].items() if k!='sha256'});self.value['runtime']=T.A.seal({k:v for k,v in self.value['runtime'].items() if k!='sha256'});self.manifest={**self.value['config'],'runtime_value':self.value['runtime']};self.write(Path(self.binding['config_path']),self.value['config']);self.write(Path(self.binding['anchors']['runtime_path']),self.value['runtime']);reg=T.A.seal({'scope':D.SCOPE,'source_files':self.value['config']['source_files'],'runtime_files':self.value['config']['runtime_files'],'config_body_sha256':self.value['config']['sha256'],'runtime_body_sha256':self.value['runtime']['sha256']});self.write(Path(self.binding['anchors']['registration_path']),reg)
        for name,path,filekey,bodykey in (('config',self.binding['config_path'],'expected_config_file','expected_config_body'),('runtime',self.binding['anchors']['runtime_path'],'expected_runtime_file','expected_runtime_body'),('registration',self.binding['anchors']['registration_path'],'expected_registration_file','expected_registration_body')):
            self.binding['anchors'][filekey]=self.desc(Path(path))['sha256'];self.binding['anchors'][bodykey]=json.loads(Path(path).read_bytes())['sha256']
        self.anchor={'schema_version':2,**deepcopy(self.binding),'grant_reference':self.refs['grant'],'owner_receipt_reference':self.refs['owner_receipt']};self.pair('OWNED_EXTERNAL_ANCHORS.json',self.anchor);self.receipt['config_sha256']=self.binding['anchors']['expected_config_file'];self.pair('SUPERVISOR_RECEIPT.json',self.receipt);owner=json.loads((self.output/'supervisor_ownership.json').read_bytes());owner['config_sha256']=self.receipt['config_sha256'];self.pair('supervisor_ownership.json',owner);self.expected_identity['owner_receipt_sha256']=T.A.digest(owner)
        self.write(self.lease/'terminal.json',self.receipt);(self.preserved/'lease_terminal.json').write_bytes((self.lease/'terminal.json').read_bytes());self.preservation['lease_files']['terminal.json'].update(self.desc(self.lease/'terminal.json'));reservation=json.loads((self.lease/'reservation.json').read_bytes());reservation['config_sha256']=self.receipt['config_sha256'];self.write(self.lease/'reservation.json',reservation);(self.preserved/'lease_reservation.json').write_bytes((self.lease/'reservation.json').read_bytes());self.preservation['lease_files']['reservation.json'].update(self.desc(self.lease/'reservation.json'));self.preservation['config_body_sha256']=self.value['config']['sha256'];self.preservation['source_files']=self.value['config']['source_files'];snapdir=self.preserved/'source_snapshots'
        for p in list(snapdir.iterdir()):p.unlink()
        self.preservation['source_snapshot_files']={}
        for path,d in self.value['config']['source_files'].items():name=hashlib.sha256(path.encode()).hexdigest()+Path(path).suffix;(snapdir/name).write_bytes(Path(path).read_bytes());self.preservation['source_snapshot_files'][path]={'path':'source_snapshots/'+name,**d}
        self.resign()
    def resign(self):
        self.preservation['raw_files']={p.name:self.desc(p) for p in self.output.iterdir()};self.preservation['current_death_observation']['supervisor_file_sha256']=self.desc(self.output/'SUPERVISOR_RECEIPT.json')['sha256'];self.preservation=T.A.seal({k:v for k,v in self.preservation.items() if k!='sha256'});self.write(self.preserved/'PRESERVATION.json',self.preservation)
    def update_raw_identity(self,identity):
        packet=json.loads((self.output/'IMPORT_LIBRARY_MEASUREMENT.json').read_bytes());packet['identity']=deepcopy(identity);packet=T.A.seal({k:v for k,v in packet.items() if k!='sha256'});self.pair('IMPORT_LIBRARY_MEASUREMENT.json',packet);worker=json.loads((self.output/'worker_result.json').read_bytes());worker.update(measurement_file_sha256=self.desc(self.output/'IMPORT_LIBRARY_MEASUREMENT.json')['sha256'],measurement_body_sha256=packet['sha256']);self.pair('worker_result.json',worker);self.resign()
    def postrun(self):
        priorD,priorT=D.PACKAGE,T.PACKAGE;D.PACKAGE=T.PACKAGE=self.package
        try:return D._postrun(self.context,self.receipt,self.target,contract=self.contract,supervisor=self.supervisor,terminal=T,expected_binding=self.binding,gate_references=self.refs,_CPU_fixture=D._CPU_capability())
        finally:D.PACKAGE,T.PACKAGE=priorD,priorT
    def test_actual_terminal_success_only_after_preservation(self):
        self.set_postrun();result=self.postrun();self.assertEqual(result['terminal_status'],'IMPORT_MEASUREMENT_TERMINAL_VALIDATED_UNQUALIFIED');self.assertEqual(self.events,['preserve']);self.assertFalse(result['native_PASS'])
    def test_failed_worker_raw_stays_failed(self):
        self.set_postrun(True);self.assertEqual(self.postrun()['terminal_status'],'IMPORT_MEASUREMENT_PARTIAL_FAILURE_PRESERVED')
    def test_raw_context_or_birth_identity_rebind_refuses(self):
        self.set_postrun();bad=deepcopy(self.expected_identity);bad['context_sha256']='0'*64;self.update_raw_identity(bad)
        with self.assertRaises(ValueError):self.postrun()
        self.doCleanups();self.setUp();self.set_postrun();bad=deepcopy(self.expected_identity);bad['owned_birth']['creation_filetime_100ns']+=1;self.update_raw_identity(bad)
        with self.assertRaises(ValueError):self.postrun()
    def test_absent_semantic_result_reports_partial_terminal(self):
        self.set_postrun(True);(self.output/'worker_result.json').unlink();(self.preserved/'worker_result.json').unlink();self.preservation['worker_result_present']=False;self.resign();result=self.postrun();self.assertEqual(result['terminal_status'],'IMPORT_MEASUREMENT_PARTIAL_FAILURE_PRESERVED')
    def test_preservation_failure_and_absent_owner_refs_never_complete(self):
        self.set_postrun();self.supervisor.preserve_terminal=lambda *a,**k:(_ for _ in ()).throw(OSError('preservation failed'))
        with self.assertRaises(OSError):self.postrun()
        with self.assertRaisesRegex(ValueError,'ownergrant'):D.dispatch('notread',preserved_directory='notcreated',anchors={})
    def test_unregistered_preservation_target_refuses_before_preserve(self):
        self.set_postrun();self.value['registration']['git_output_exceptions']={}
        with self.assertRaisesRegex(ValueError,'exception'):self.postrun()
        self.assertEqual(self.events,[])
    def test_original_no_resume_denial_retains_raw_without_fake_audit(self):
        self.set_postrun();self.receipt.update(worker_resumed=False,worker_attempted=False,worker_launched=False,status='ADMISSION_DENIED',returncode=None);self.pair('SUPERVISOR_RECEIPT.json',self.receipt);result=self.postrun();self.assertEqual(result['terminal_status'],'IMPORT_OBSERVATION_NOT_RESUMED_FAILURE_RETAINED');self.assertIsNone(result['terminal_audit']);self.assertEqual(self.events,[])
def shutil_copy(source,target):
    import shutil
    shutil.copytree(source,target)
if __name__=='__main__':
    names=('test_actual_terminal_success_only_after_preservation','test_failed_worker_raw_stays_failed','test_raw_context_or_birth_identity_rebind_refuses','test_preservation_failure_and_absent_owner_refs_never_complete','test_unregistered_preservation_target_refuses_before_preserve','test_absent_semantic_result_reports_partial_terminal','test_original_no_resume_denial_retains_raw_without_fake_audit');result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in names));v={'kind':'CPU_SOURCE_OWNED_POSTRUN_DRIVER_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_pure_terminal_executed':True,'lifecycle_preserver_inert_fixture_copy':True,'SUP_Job_resource_death_CPU_metadata':True,'native_context_minted':False,'native_API_framework_called':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'import_dispatch_postrun.py',HERE/'import_terminal_audit.py',Path(__file__).resolve())}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
