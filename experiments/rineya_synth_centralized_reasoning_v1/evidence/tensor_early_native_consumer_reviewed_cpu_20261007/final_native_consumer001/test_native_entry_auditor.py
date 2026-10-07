"""CPU schema adapter runs the real native preservation consumer, no APIs.

Writer/core bytes are real inert executions. SUP/Job/resource/death records are
explicitly invented CPU layout metadata and confer no execution evidence.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json,sys,types,unittest,shutil
HERE=Path(__file__).resolve().parent
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
BASE=load(HERE/'test_entry_auditor_producer004.py','CPU_native_entry_base')
A=BASE.A
EARLY_SOURCE=HERE/'final_early_failure001'/'prefix_input_owned_worker.py'
assert hashlib.sha256(EARLY_SOURCE.read_bytes()).hexdigest()=='36c5f9d1462721adc76e46dd62dd79e42654790b442cd5cc824fbcda0ee7fd20'
EARLY=types.ModuleType('CPU_actual_early_source');EARLY.__file__=str(HERE/'prefix_input_owned_worker.py')
exec(compile(EARLY_SOURCE.read_bytes(),EARLY.__file__,'exec'),EARLY.__dict__)

class Controls(BASE.Controls):
    def fixture(self,early=False,actual_early=False,write_failure=False):
        if actual_early:
            self.ledger.close()
            for p in self.output.iterdir():p.unlink()
            self.ledger,self.phases=EARLY._entry_journal_bootstrap(self.output)
            self.selected={item['path']:{'bytes':item['bytes'],'sha256':item['sha256']} for item in self.ledger.entry_bootstrap_sources}
            if write_failure:
                durable=self.ledger.durable
                def failed_write(name,value):
                    if name=='OWNED_WORKER_RETURN.json':raise OSError('actual producer envelope write failure')
                    return durable(name,value)
                self.ledger.durable=failed_write
            actions={name:lambda *a:None for name in self.phases};actions[self.phases[0]]=lambda *a:(_ for _ in ()).throw(ValueError('actual producer source failure'))
            with self.assertRaisesRegex(ValueError,'actual producer source failure'):EARLY._entry_phase_body(self.ledger,self.phases,actions,lambda *a:self.fail('core attempted'),_native_mode=True)
            shutil.copytree(self.output,self.preserved)
            envelope=None if write_failure else json.loads((self.output/'OWNED_WORKER_RETURN.json').read_bytes())
            self.report={} if envelope is None else envelope['report']
        else:
            self.run_entry(failed=0 if early else None)
            (self.output/'CPU_ENTRY_RETURN.json').unlink();(self.preserved/'CPU_ENTRY_RETURN.json').unlink()
        owner={'pid':12345,'creation_filetime_100ns':134000000000000001}
        member={'actual_worker_pid':12345,'actual_parent_pid':12344,'launcher_root_pid':12345,'supervisor_pid':12344,'job_name':'CPU_LAYOUT_JOB','owned_job_membership_verified':True,'query_job_handle_closed_before_framework_import':True}
        report=deepcopy(self.report);final=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes())
        status='NATIVE_INPUT_CHECKS_FAILED' if early else 'NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT'
        final.update(scope=A.NATIVE_SCOPE,status=status)
        report.update(status=status,final_phase_summary=final)
        if not early:
            report['scope']=A.NATIVE_SCOPE
            worker=json.loads((self.output/'worker_result.json').read_bytes());worker.update(scope=A.NATIVE_SCOPE,status=status)
            self.rewrite_pair('worker_result.json',worker);self.rewrite_pair('ownership_verified.json',{'scope':A.NATIVE_SCOPE})
            rows=[json.loads(line) for line in (self.output/'PHASE_OBSERVATIONS.jsonl').read_bytes().splitlines()]
            next(row for row in rows if row['operation']=='worker_ownership_verify' and row['event']=='returned')['result']=member
            raw=b''.join((A.canonical(row)+'\n').encode() for row in rows);final['marker_bytes']=len(raw)
            for root in (self.output,self.preserved):(root/'PHASE_OBSERVATIONS.jsonl').write_bytes(raw)
        elif not actual_early:
            report={key:deepcopy(value) for key,value in report.items() if key in ('status','entry_errors','entry_bootstrap_sources','selected_boundary_counters','selected_boundary_source_pins')}
            report.update(scope=A.NATIVE_SCOPE,core_membership_status='NOT_CORE_ENROLLED',precore_owned_membership=None,
                runtime_execution_status='NO_COMPLETED_CORE_RECORD',native_PASS=False,native_authority_issued=False)
        self.rewrite_pair('PHASE_ACCOUNTING_FINAL.json',final)
        sources={path:{**value,'role':'python_source'} for path,value in self.selected.items()}
        helper=str(HERE/'prefix_input_owned_supervisor.py');sources[helper]={**BASE.descriptor(Path(helper)),'role':'python_source'}
        self.manifest=A.seal({'source_files':sources,'runtime_files':{},'helper_pins':{'native_supervisor.py':sources[helper]['sha256']}})
        self.anchors={'config_path':str(self.root/'CPU_LAYOUT_CONFIG.json'),'anchors':{'expected_config_file':'b'*64}}
        execution={'CPU_LAYOUT_MODULE':{'path':next(iter(sources)),'sha256':next(iter(sources.values()))['sha256']}}
        container={'report':report,'runtime_execution_modules':None if early else execution,'owned_membership_verified':None if early else member,'IO':{'owned_return_write':{'attempted':1,'returned':1,'failed':0,'pending':0}},'native_PASS':False}
        if early:container.update(kind='NATIVE_SHARED_ENTRY_PRECORE_FAILURE',report_requires_final_phase_accounting=True)
        if not actual_early:self.rewrite_pair('OWNED_WORKER_RETURN.json',container)
        self.rewrite_pair('OWNED_EXTERNAL_ANCHORS.json',self.anchors)
        records=A.parse_ledger((self.output/'PHASE_OBSERVATIONS.jsonl').read_bytes());first,last=records[0]['monotonic'],records[-1]['monotonic'];resume=first-.01
        def admission(clock):
            return {'sample_started_monotonic':clock,'sample_completed_monotonic':clock,'host_available_bytes':8*A.GIB,'worker_tree_rss_bytes':0,'disk_free_bytes':64*A.GIB,'gpu_sample_started_monotonic':clock,'gpu_sample_completed_monotonic':clock,'gpu_free_bytes':16*A.GIB,'gpu_total_bytes':24*A.GIB,'GPU_index':0,'gpu_uuid':'CPU_LAYOUT_GPU','gpu_observation_id':str(clock)}
        full={**owner,'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False,'job_name':'CPU_LAYOUT_JOB'}
        lease=self.root/'CPU_LAYOUT_LEASE';lease.mkdir()
        sup={'config_sha256':'b'*64,'preflight':{'source_commit':'a'*40,'resources':A.RESOURCES},'worker_attempted':True,'worker_launched':True,'worker_resumed':True,'preflight_revalidated_before_current_observations':True,'job_closed_kill_on_close':True,'sampler_threads_terminal_verified':True,'gpu_telemetry_cleanup_verified':True,'partial_outputs_frozen':True,'owned_worker_identity_currently_alive':False,'unjoined_sampler_threads':[],'elapsed_seconds':last-resume+2.1,'worker_identity':full,'worker_resume_monotonic':resume,'deadline_monotonic':resume+600,'admission_observations':[admission(resume-2.02),admission(resume-.02)],'dispatch_observation':admission(resume-.01),'lease_path':str(lease),'status':'SUPERVISED_WORKER_EXITED_ZERO' if not early else 'SUPERVISED_WORKER_EXITED_NONZERO','returncode':0 if not early else 2}
        self.rewrite_pair('SUPERVISOR_RECEIPT.json',sup);self.rewrite_pair('supervisor_ownership.json',{'worker_pid':owner['pid'],'supervisor_pid':12344,'job_name':'CPU_LAYOUT_JOB','config_sha256':'b'*64})
        samples=[];clock=resume
        while clock<last+.01:
            samples.extend([{'kind':'CPU','sample_started_monotonic':clock,'host_available_bytes':8*A.GIB,'worker_tree_rss_bytes':0,'owned_process_identities':[owner]}, {'kind':'GPU','gpu_sample_started_monotonic':clock,'gpu_free_bytes':16*A.GIB,'gpu_total_bytes':24*A.GIB,'gpu_uuid':'CPU_LAYOUT_GPU'}]);clock+=.05
        for clock in (first,last):
            samples.extend([{'kind':'CPU','sample_started_monotonic':clock,'host_available_bytes':8*A.GIB,'worker_tree_rss_bytes':0,'owned_process_identities':[owner]}, {'kind':'GPU','gpu_sample_started_monotonic':clock,'gpu_free_bytes':16*A.GIB,'gpu_total_bytes':24*A.GIB,'gpu_uuid':'CPU_LAYOUT_GPU'}])
        samples.sort(key=lambda row:row.get('sample_started_monotonic',row.get('gpu_sample_started_monotonic')));self.rewrite_pair('RESOURCE_OBSERVATIONS.json',samples)
        leases={'reservation.json':{'config_sha256':'b'*64},'worker.json':full,'resume.json':{'worker_identity':full,'external_monitor_ready_before_resume':True,'resumed_monotonic':resume},'terminal.json':sup};leasefiles={}
        for name,value in leases.items():
            BASE.write(lease/name,value);raw=(lease/name).read_bytes();(self.preserved/('lease_'+name)).write_bytes(raw);leasefiles[name]={'source_path':str(lease/name),'path':'lease_'+name,**BASE.descriptor(lease/name)}
        snapshots={};directory=self.preserved/'source_snapshots';directory.mkdir()
        for path,value in sources.items():
            name=hashlib.sha256(path.encode()).hexdigest()+Path(path).suffix;(directory/name).write_bytes(Path(path).read_bytes());snapshots[path]={'path':'source_snapshots/'+name,**value}
        self.preservation=A.seal({'kind':'PRESERVED_REGISTERED_ONE_PREFIX_NATIVE_TENSOR_DIAGNOSTIC','scope':A.NATIVE_SCOPE,'manifest_sha256':self.manifest['sha256'],'raw_files':{p.name:BASE.descriptor(p) for p in self.output.iterdir()},'source_files':sources,'source_snapshot_files':snapshots,'lease_files':leasefiles,'current_death_observation':{'observed_monotonic':last+.1,'observed_utc':'2026-10-07T10:00:00+00:00','supervisor_file_sha256':BASE.descriptor(self.output/'SUPERVISOR_RECEIPT.json')['sha256'],'config_file_sha256':'b'*64,'observations':[{'identity':owner,'currently_alive':False}],'death_verifier_source_sha256':sources[helper]['sha256']},'fixture_only':True,'native_authority_issued':False})
        BASE.write(self.preserved/'PRESERVATION.json',self.preservation)
    def native_replay(self,capability=True):
        return A._native_preserved_body(self.preserved,original_directory=self.output,manifest=self.manifest,source_pins=self.manifest['source_files'],expected_preservation_file_sha256=BASE.descriptor(self.preserved/'PRESERVATION.json')['sha256'],expected_preservation_body_sha256=self.preservation['sha256'],expected_head='a'*40,anchors=self.anchors,_CPU_fixture_capability=A._mint_CPU_native_schema_fixture() if capability else None)
    def test_full_actual_native_consumer_with_explicit_CPU_layout_metadata(self):
        self.fixture();v=self.native_replay();self.assertEqual(v['status'],'CPU_NATIVE_SCHEMA_COMPLETE');self.assertTrue(v['shared_entry_composition_source_exercised']);self.assertFalse(v['native_shared_entry_terminal_qualified'])
    def test_early_provisional_envelope_has_absent_core_facts(self):
        self.fixture(early=True);v=self.native_replay();self.assertEqual(v['status'],'CPU_NATIVE_SCHEMA_FAILURE_PRESERVED');self.assertTrue(v['precore_facts_unavailable'])
    def test_nonfixture_gate_and_resealed_missing_membership_refuse(self):
        self.fixture()
        with self.assertRaises(ValueError):self.native_replay(capability=False)
        value=json.loads((self.output/'OWNED_WORKER_RETURN.json').read_bytes());value['owned_membership_verified']=None;self.rewrite_pair('OWNED_WORKER_RETURN.json',value);self.preservation['raw_files']['OWNED_WORKER_RETURN.json']=BASE.descriptor(self.output/'OWNED_WORKER_RETURN.json');self.preservation=A.seal(self.preservation);BASE.write(self.preserved/'PRESERVATION.json',self.preservation)
        with self.assertRaisesRegex(ValueError,'actual worker Job membership'):self.native_replay()
    def test_resealed_startup_source_pin_override_refuses_in_native_body(self):
        self.fixture();value=json.loads((self.output/'OWNED_WORKER_RETURN.json').read_bytes());value['report']['entry_bootstrap_sources'][0]['sha256']='0'*64
        self.rewrite_pair('OWNED_WORKER_RETURN.json',value);self.preservation['raw_files']['OWNED_WORKER_RETURN.json']=BASE.descriptor(self.output/'OWNED_WORKER_RETURN.json');self.preservation=A.seal(self.preservation);BASE.write(self.preserved/'PRESERVATION.json',self.preservation)
        with self.assertRaisesRegex(ValueError,'actual called source pin'):self.native_replay()
    def test_actual_source_produced_early_envelope_and_writer_failure(self):
        self.fixture(early=True,actual_early=True);value=self.native_replay();self.assertTrue(value['precore_facts_unavailable']);self.assertEqual(value['status'],'CPU_NATIVE_SCHEMA_FAILURE_PRESERVED')
        # A separate real producer failure leaves journal/final bytes and no
        # envelope. The same consumer refuses semantic completion honestly.
        self.tearDown();self.setUp();self.fixture(early=True,actual_early=True,write_failure=True)
        self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists());final=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes());self.assertEqual(final['IO']['owned_return_write']['failed'],1)
        with self.assertRaisesRegex(ValueError,'all original native supervisor/core/log raw files'):self.native_replay()

if __name__=='__main__':
    methods=('test_full_actual_native_consumer_with_explicit_CPU_layout_metadata','test_early_provisional_envelope_has_absent_core_facts','test_nonfixture_gate_and_resealed_missing_membership_refuse','test_resealed_startup_source_pin_override_refuses_in_native_body','test_actual_source_produced_early_envelope_and_writer_failure')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(name) for name in methods));v={'kind':'CPU_NATIVE_SHARED_ENTRY_CONSUMER_SCHEMA_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'native_API_or_framework_called':False,'SUP_Job_resource_death_metadata_invented_CPU_only':True,'actual_native_preserved_body_read_replay_exercised':True,'actual_early_producer_writer_and_write_failure_exercised':True,'native_shared_entry_terminal_qualified':False,'native_authority_issued':False,'source_hashes':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('prefix_input_owned_auditor.py','test_native_entry_auditor.py','test_entry_auditor_producer004.py','final_early_failure001/prefix_input_owned_worker.py')}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
