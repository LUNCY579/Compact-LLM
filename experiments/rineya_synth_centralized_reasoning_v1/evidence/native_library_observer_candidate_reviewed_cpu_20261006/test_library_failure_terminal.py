"""Actual worker pre-core sidecar and preservation; injected terminal facts.

SUP/resource/birth/lease/cleanup are explicit CPU metadata. Actual selected
worker and Guard source produce the failure sidecar; full selected S module and
preservation function execute. No runtime or executing-birth qualification.
"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib,time,unittest
import prefix_library_failure_audit as A
import test_owned_library_entry as O
from test_full_native_cpu_workflow import compiled

HERE=Path(__file__).resolve().parent
GIB=1024**3

class PrecoreFailureTerminalControls(unittest.TestCase):
    def setUp(self):
        self.case=O.OwnedLibraryEntryControls();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.c=self.case;self.f=self.c.fixture;self.C=self.c.C
        plan=[{'id':1,'parent_id':None,'name':'framework_import'}]
        self.f.runtime['library_boundary_plan']=plan;self.f.review['library_boundary_plan_sha256']=self.C.digest(plan)
        self.c.fixture_case.reseal()
        self.alive=False;self.frame=None;self.value=None;self.target=None

    def build(self,stage='ownership'):
        c=self.c
        with c.fixture_case.isolated():
            c.prepare();self.value=self.C.context_value(c.context)
            if stage=='install':
                c.library_supplier.scope_changes['complete']=False
                with self.assertRaises(ValueError):c.run_main()
            else:
                with patch.object(c.inert,'verify',side_effect=RuntimeError('actual inert original ownership failure')):
                    with self.assertRaisesRegex(RuntimeError,'actual inert original ownership failure'):c.run_main()
        raw=c.output;self.assertFalse(set(p.name for p in raw.iterdir())&A.CORE)
        envelope=self.C.parse((raw/'LIBRARY_BOUNDARY_EVIDENCE.json').read_bytes())
        self.assertEqual(envelope['main_errors'][0]['stage'],stage)
        known=[p['time_ns']/1e9 for p in envelope['raw_packet']['observations'] if type(p['time_ns']) is int]
        now=time.monotonic();start=min(known) if known else now;end=max(known) if known else now;resume=start-.01
        launcher={'pid':c.owner['worker_pid'],'creation_filetime_100ns':134000000000000001}
        # This is injected recorded launcher metadata, not an executing birth.
        owned={**launcher,'parent_pid':c.owner['supervisor_pid'],'create_time':1000.5,'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False,'job_name':c.owner['job_name'],'process_handle':101,'job_handle':102}
        self.births=[launcher,{'pid':54321,'creation_filetime_100ns':134000000000000002}]
        lease=self.f.package/'artifacts/_native_gpu_leases/CPU_PRECORE_FAILURE';lease.mkdir(parents=True)
        def admission(clock,index):return {'sample_started_monotonic':clock,'sample_completed_monotonic':clock,'host_available_bytes':8*GIB,'worker_tree_rss_bytes':512*1024**2,'disk_free_bytes':64*GIB,'gpu_sample_started_monotonic':clock,'gpu_sample_completed_monotonic':clock,'gpu_free_bytes':16*GIB,'gpu_total_bytes':24*GIB,'GPU_index':0,'gpu_uuid':'CPU_INERT_GPU0','gpu_observation_id':str(index)}
        samples=[];clock=resume
        while clock<end:
            samples.extend([{'kind':'CPU','sample_started_monotonic':clock,'host_available_bytes':8*GIB,'worker_tree_rss_bytes':512*1024**2,'owned_process_identities':deepcopy(self.births)},
                {'kind':'GPU','gpu_sample_started_monotonic':clock,'gpu_free_bytes':16*GIB,'gpu_total_bytes':24*GIB,'gpu_uuid':'CPU_INERT_GPU0'}]);clock+=.1
        samples.extend([{'kind':'CPU','sample_started_monotonic':end,'host_available_bytes':8*GIB,'worker_tree_rss_bytes':512*1024**2,'owned_process_identities':deepcopy(self.births)},
            {'kind':'GPU','gpu_sample_started_monotonic':end,'gpu_free_bytes':16*GIB,'gpu_total_bytes':24*GIB,'gpu_uuid':'CPU_INERT_GPU0'}])
        self.write(raw/'RESOURCE_OBSERVATIONS.json',samples)
        (raw/'worker.stdout.log').write_bytes(b'');(raw/'worker.stderr.log').write_bytes(b'inert pre-core failure\n')
        (raw/'supervisor_heartbeat.jsonl').write_bytes((self.C.canonical({'unix_seconds':1000.5,'supervisor_pid':c.owner['supervisor_pid']})+'\n').encode())
        sup={'config_sha256':self.value['anchors']['expected_config_file'],'preflight':{'source_commit':self.value['expected_head'],'resources':A.RESOURCES},'output_dir':str(raw),
            'worker_attempted':True,'worker_launched':True,'worker_resumed':True,'preflight_revalidated_before_current_observations':True,'job_closed_kill_on_close':True,
            'sampler_threads_terminal_verified':True,'gpu_telemetry_cleanup_verified':True,'partial_outputs_frozen':True,'owned_worker_identity_currently_alive':False,'unjoined_sampler_threads':[],
            'elapsed_seconds':end-resume+2.05,'worker_identity':owned,'worker_resume_monotonic':resume,'deadline_monotonic':resume+600,'admission_observations':[admission(resume-2.03,0),admission(resume-.025,1)],
            'dispatch_observation':admission(resume-.005,2),'lease_path':str(lease),'status':'SUPERVISED_WORKER_FAILED','returncode':2}
        sup['partial_outputs']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(raw.iterdir())]
        self.write(raw/'SUPERVISOR_RECEIPT.json',sup)
        for n,v in {'worker.json':owned,'reservation.json':{'config_sha256':sup['config_sha256'],'output_dir':str(raw)},
            'resume.json':{'worker_identity':owned,'external_monitor_ready_before_resume':True,'resumed_monotonic':resume},'terminal.json':sup}.items():self.write(lease/n,v)
        S=compiled(self.f.package/'prefix_input_owned_supervisor.py');self.target=self.f.package/'evidence/CPU_PRECORE_FAILURE'
        def revalidate():
            actual=self.C._native_manifest_value(self.f.config_path,**self.f.anchors);v=self.C.context_value(c.context)
            for k in ('fixture_only','native_PASS','native_authority_issued','status'):v.pop(k,None)
            self.assertEqual(actual,v);return actual
        with self.f.activate():
            self.receipt=S._CPU_preserve_native_fixture(raw,self.target,c.context,contract=self.C,_capability=self.C._mint_CPU_native_context_fixture(),death_verifier=lambda _:False,revalidate=revalidate)
        self.frame={'raw':raw,'lease':lease,'sup':sup,'metadata_facts_injected':['SUP/containment/births','CPU/GPU/resources','lease/cleanup/death'], 'actual_guard_raw_packet_produced':True,'actual_worker_publication_exercised':True}
        return self.frame

    def write(self,p,v):p.write_bytes((self.C.canonical(v)+'\n').encode('ascii'))
    def audit(self):
        raw=(self.target/'PRESERVATION.json').read_bytes()
        return A.audit_failure(self.target,original_directory=self.frame['raw'],selected_context=self.value,
            expected_preservation_file_sha256=hashlib.sha256(raw).hexdigest(),expected_preservation_body_sha256=self.receipt['sha256'],
            revalidate=lambda:deepcopy(self.value),fresh_death=lambda _:self.alive)

    def reseal(self,name,value):
        """Reseal raw pairs/SUP/lease/PRES so negatives reach semantic joins."""
        raw=self.frame['raw'];target=self.target;self.write(raw/name,value);self.write(target/name,value)
        sup=self.C.parse((raw/'SUPERVISOR_RECEIPT.json').read_bytes())
        if name=='SUPERVISOR_RECEIPT.json':sup=value
        else:
            sup['partial_outputs']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(raw.iterdir()) if p.name!='SUPERVISOR_RECEIPT.json']
        self.write(raw/'SUPERVISOR_RECEIPT.json',sup);self.write(target/'SUPERVISOR_RECEIPT.json',sup)
        self.write(self.frame['lease']/'terminal.json',sup);self.write(target/'lease_terminal.json',sup)
        pres=self.C.parse((target/'PRESERVATION.json').read_bytes())
        for n in pres['raw_files']:
            b=(raw/n).read_bytes();pres['raw_files'][n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        d=pres['lease_files']['terminal.json'];b=(target/'lease_terminal.json').read_bytes();d.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
        pres['current_death_observation']['supervisor_file_sha256']=pres['raw_files']['SUPERVISOR_RECEIPT.json']['sha256']
        pres['sha256']=self.C.digest({k:v for k,v in pres.items() if k!='sha256'});self.write(target/'PRESERVATION.json',pres);self.receipt=pres

    def test_direct_guard_packet_injected_ownership_terminal_is_incomplete_not_certified(self):
        self.build();r=self.audit();self.assertEqual(r['status'],'INCOMPLETE_FAILURE_PRESERVED')
        self.assertFalse(r['native_PASS']);self.assertFalse(r['completion_established']);self.assertIsNone(r['executing_birth'])
        self.assertFalse(r['executing_birth_certified']);self.assertEqual(r['library_evidence']['replay']['reached_counts']['entered'],0)

    def test_direct_constructor_packet_injected_terminal_preserved_without_birth(self):
        self.build('install');r=self.audit();self.assertFalse(r['completion_established']);self.assertIsNone(r['executing_birth'])
        self.assertTrue(r['library_evidence']['replay']['failure_records'])

    def test_original_copy_roster_and_source_byte_drift_rejected(self):
        self.build()
        for p in (self.target/'worker.stderr.log',self.frame['raw']/'worker.stderr.log',self.f.package/'prefix_library_evidence.py'):
            with self.subTest(path=p.name):
                b=p.read_bytes();p.write_bytes(b+b'changed')
                with self.assertRaises(ValueError):self.audit()
                p.write_bytes(b)
        rogue=self.target/'UNREGISTERED.json';rogue.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'whole preserved failure roster'):self.audit()

    def test_resealed_success_supervisor_or_cleanup_false_rejected(self):
        self.build();sup=deepcopy(self.frame['sup']);sup['returncode']=0;sup['status']='SUPERVISED_WORKER_EXITED_ZERO';self.reseal('SUPERVISOR_RECEIPT.json',sup)
        with self.assertRaisesRegex(ValueError,'nonzero supervisor failure'):self.audit()
        sup=deepcopy(self.frame['sup']);sup['sampler_threads_terminal_verified']=False;self.reseal('SUPERVISOR_RECEIPT.json',sup)
        with self.assertRaisesRegex(ValueError,'terminal containment'):self.audit()
        sup=deepcopy(self.frame['sup']);sup['deadline_monotonic']=sup['worker_resume_monotonic']+601;self.reseal('SUPERVISOR_RECEIPT.json',sup)
        with self.assertRaisesRegex(ValueError,'whole scope and owned clock window'):self.audit()

    def test_resealed_resource_floor_and_exact_death_roster_rejected(self):
        self.build();original=self.C.parse((self.frame['raw']/'RESOURCE_OBSERVATIONS.json').read_bytes());rows=deepcopy(original);rows[0]['host_available_bytes']=0;self.reseal('RESOURCE_OBSERVATIONS.json',rows)
        with self.assertRaisesRegex(ValueError,'CPU floors'):self.audit()
        self.reseal('RESOURCE_OBSERVATIONS.json',original)
        pres=deepcopy(self.receipt);pres['current_death_observation']['observations'].pop()
        pres['sha256']=self.C.digest({k:v for k,v in pres.items() if k!='sha256'});self.receipt=pres;self.write(self.target/'PRESERVATION.json',pres)
        with self.assertRaisesRegex(ValueError,'complete all-birth'):self.audit()

    def test_live_birth_and_wrong_failure_stage_rejected(self):
        self.build();self.alive=True
        with self.assertRaisesRegex(ValueError,'fresh externally bound'):self.audit()
        self.alive=False;env=self.C.parse((self.frame['raw']/'LIBRARY_BOUNDARY_EVIDENCE.json').read_bytes());env['main_errors'][0]['stage']='execute';env['sha256']=self.C.digest({k:v for k,v in env.items() if k!='sha256'});self.reseal('LIBRARY_BOUNDARY_EVIDENCE.json',env)
        with self.assertRaisesRegex(ValueError,'only pre-core'):self.audit()

    def test_fabricated_core_or_lease_substitution_rejected(self):
        self.build();p=self.frame['raw']/'worker_result.json';p.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'whole original raw failure roster'):self.audit()
        p.unlink();value=self.C.parse((self.frame['lease']/'resume.json').read_bytes());value['worker_identity']['creation_filetime_100ns']+=1
        self.write(self.frame['lease']/'resume.json',value);self.write(self.target/'lease_resume.json',value)
        pres=deepcopy(self.receipt);b=(self.target/'lease_resume.json').read_bytes();pres['lease_files']['resume.json'].update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
        pres['sha256']=self.C.digest({k:v for k,v in pres.items() if k!='sha256'});self.receipt=pres;self.write(self.target/'PRESERVATION.json',pres)
        with self.assertRaisesRegex(ValueError,'worker/resume/reservation joins'):self.audit()

    def test_external_plan_and_sidecar_identity_substitution_rejected(self):
        self.build();env=self.C.parse((self.frame['raw']/'LIBRARY_BOUNDARY_EVIDENCE.json').read_bytes());env['owner_receipt_sha256']='0'*64;env['sha256']=self.C.digest({k:v for k,v in env.items() if k!='sha256'});self.reseal('LIBRARY_BOUNDARY_EVIDENCE.json',env)
        with self.assertRaisesRegex(ValueError,'recipe joins'):self.audit()

    def test_actual_selected_main_retains_original_ownership_error_without_core(self):
        self.build();env=self.C.parse((self.frame['raw']/'LIBRARY_BOUNDARY_EVIDENCE.json').read_bytes())
        self.assertEqual(env['main_errors'][0]['error']['message'],'actual inert original ownership failure')
        self.assertIsNone(env['raw_packet']['identity']['owned_birth'])
        self.assertEqual(env['raw_packet']['failure_records'],[])
        self.assertFalse(env['raw_packet']['terminal']['guard_failed'])
        self.assertTrue(self.frame['actual_worker_publication_exercised']);self.assertFalse(A.CORE&set(self.receipt['raw_files']))
        result=self.audit();self.assertFalse(result['native_PASS']);self.assertFalse(result['executing_birth_certified'])


if __name__=='__main__':unittest.main()
