"""Inert source/terminal fixtures only. No actual raw leaf or process probes."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import hashlib,json,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import preserve_terminal as h
p=h.pure
def write(path,value):path.write_bytes((p.canonical(value)+'\n').encode('ascii'))
def fixture(directory,mode='success'):
    root=Path(directory);package=root/'experiments/CPU_PACKAGE';leaf=package/'artifacts/CPU_TERMINAL';saved=package/'evidence/CPU_PRESERVED'
    lease=package/'artifacts/_native_gpu_leases/CPU_LEASE';config=package/'configs/CPU_CONFIG.json';inventory=package/'artifacts/short_m0_audit_registrations/CPU_INVENTORY.json'
    for path in (leaf,lease,config.parent,inventory.parent,saved.parent):path.mkdir(parents=True,exist_ok=True)
    sources={}
    for i in range(131):
        path=package/('native_supervisor.py' if i==0 else 'source_%03d.py'%i);path.write_bytes(('CPU_SOURCE_%03d'%i).encode('ascii'));sources[str(path.resolve())]=p.file_info(path)['sha256']
    scope={'contract':h.registration.auditor.VERSION,'batch_selector':{'source_kind':'atomic_RS_DEV','ordinal':4},'rows':[{'row_id':i} for i in range(64)],'source_pins':sources}
    for name,key in h.registration.SECTIONS.items():
        if name!='source':scope[key]={'CPU_'+name+'_%03d'%i:'b'*64 for i in range(h.registration.COUNTS[name])}
    scope=h.seal(scope);write(config,scope);config_sha=p.file_info(config)['sha256'];head='a'*40
    deps={str((package/'native_supervisor.py').resolve()):sources[str((package/'native_supervisor.py').resolve())]}
    reg=h.seal({'source_commit':head,'config_file_sha256':config_sha,'inventories':{k:scope[v] for k,v in h.registration.SECTIONS.items()},
        'counts':h.registration.COUNTS,'auditor_dependency_pins':deps});write(inventory,reg);reg_sha=p.file_info(inventory)['sha256']
    owner={'pid':41,'create_time':1.0,'creation_filetime_100ns':123,'process_handle':7,'job_handle':8,'job_name':'CPU_JOB',
        'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False}
    supervisor={'pid':42,'create_time':2.0}
    write(lease/'reservation.json',{'supervisor':supervisor,'config_sha256':config_sha,'output_dir':str(leaf)})
    no_worker=mode=='denied'
    sup={'status':'ADMISSION_DENIED' if no_worker else 'SUPERVISED_WORKER_EXITED_ZERO' if mode=='success' else 'SUPERVISED_WORKER_FAILED',
        'output_dir':str(leaf),'lease_path':str(lease),'partial_outputs_frozen':True,'sampler_threads_terminal_verified':True,
        'gpu_telemetry_cleanup_verified':True,'unjoined_sampler_threads':[],'worker_attempted':not no_worker,'worker_launched':not no_worker,
        'worker_resumed':not no_worker,'returncode':None if no_worker else 0 if mode=='success' else 137,
        'config_sha256':config_sha,'preflight':{'source_commit':head,'config_sha256':config_sha},'finished_utc':'2026-10-05T00:00:00+00:00'}
    if not no_worker:
        write(lease/'worker.json',owner);write(lease/'resume.json',{'worker_identity':owner,'external_monitor_ready_before_resume':True,'resumed_monotonic':1.})
        sup.update(worker_identity=owner,worker_resume_monotonic=1.,job_closed_kill_on_close=True,owned_worker_identity_currently_alive=False)
        write(leaf/'supervisor_ownership.json',{'worker_pid':41,'root_worker_pid':41,'job_name':'CPU_JOB','supervisor_pid':42,'supervisor_create_time':2.0,'config_sha256':config_sha})
        child={'pid':43,'create_time':3.,'creation_filetime_100ns':124}
        write(leaf/'RESOURCE_OBSERVATIONS.json',[{'kind':'CPU','owned_process_identities':[h.normalized(owner),child]}])
    if mode=='success':
        write(leaf/'worker_result.json',{'status':'COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED','error':None,'config_sha256':config_sha,
            'source_commit':head,'collection_costs':{'attempted_forwards':64}})
        write(leaf/'PHASE_ACCOUNTING_FINAL.json',{'status':'COMPLETE_PHASE_OBSERVATION'})
        for i in range(138):write(leaf/('stage_CPU_%03d.json'%i),{'CPU_FRAGMENT':i})
    else:write(leaf/'worker.stderr.log',{'CPU_ERROR':'preserve negative evidence'})
    sup['partial_outputs']=[{'path':path.name,**p.file_info(path)} for path in leaf.iterdir()]
    write(leaf/'SUPERVISOR_RECEIPT.json',sup);(lease/'terminal.json').write_bytes((leaf/'SUPERVISOR_RECEIPT.json').read_bytes())
    sup_sha=p.file_info(leaf/'SUPERVISOR_RECEIPT.json')['sha256'];proof_path=None;proof_sha=None
    if not no_worker:
        roster=h.identity_roster(leaf,owner)
        proof=h.seal({'schema_version':1,'kind':'ROOT_CURRENT_OWNED_IDENTITY_DEATH_OBSERVATION_V1','identity':owner,
            'owned_identity_sha256':p.digest(owner),'owned_identities':roster,'owned_identities_sha256':p.digest(roster),
            'identity_checks':[{'identity':v,'currently_alive':False} for v in roster],'currently_alive':False,
            'observer':'native_supervisor._pid_alive','observer_source_sha256':sources[str((package/'native_supervisor.py').resolve())],
            'source_commit':head,'config_file_sha256':config_sha,'supervisor_receipt_file_sha256':sup_sha,'observed_utc':'2026-10-05T00:00:01+00:00'})
        proof_path=package/'CPU_CURRENT_DEATH.json';write(proof_path,proof);proof_sha=p.file_info(proof_path)['sha256']
    kwargs={'config_sha':config_sha,'inventory_sha':reg_sha,'sup_sha':sup_sha,'commit':head,'terminal_signal':h.SIGNAL,
        'current_death_path':proof_path,'current_death_sha':proof_sha,'fixture':True}
    return root,package,config,leaf,saved,inventory,lease,kwargs
class Controls(unittest.TestCase):
    def run_fixture(self,values):
        root,package,config,leaf,saved,inventory,lease,kwargs=values
        with patch.object(h,'PACKAGE',package),patch.object(h,'ROOT',root):return h.preserve(config,leaf,saved,inventory,**kwargs)
    def test_source131_flat_success_copy_and_write_once_are_CPU_only(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            values=fixture(directory);result=self.run_fixture(values)
            self.assertEqual(131,len(result['source_snapshots']));self.assertEqual(143,len(result['raw_files']))
            self.assertFalse(result['full_native_audit_eligible']);self.assertFalse(result['native_PASS_obtained'])
            self.assertEqual(2,result['terminal_ownership']['Root_current_death_observation']['all_observed_owned_births'])
            self.assertFalse((values[4]/'raw').exists())
            with self.assertRaisesRegex(ValueError,'write-once'):self.run_fixture(values)
    def test_failed_and_admission_denied_bytes_preserve_without_success_authority(self):
        for mode in ('failed','denied'):
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                result=self.run_fixture(fixture(directory,mode));self.assertFalse(result['full_native_audit_eligible'])
                self.assertFalse(result['native_PASS_obtained']);self.assertEqual(131,len(result['source_snapshots']))
    def test_missing_signal_unfrozen_cleanup_or_wrong_lease_refuse_before_destination(self):
        for change in ('signal','cleanup','lease'):
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                values=list(fixture(directory));root,package,c,l,s,r,lease,kwargs=values
                if change=='signal':kwargs['terminal_signal']=None
                elif change=='lease':(lease/'terminal.json').write_bytes(b'{}\n')
                else:
                    sup=p.read_json(l/'SUPERVISOR_RECEIPT.json');sup['sampler_threads_terminal_verified']=False;write(l/'SUPERVISOR_RECEIPT.json',sup);kwargs['sup_sha']=p.file_info(l/'SUPERVISOR_RECEIPT.json')['sha256']
                with self.assertRaises(ValueError):self.run_fixture(values)
                self.assertFalse(s.exists())
    def test_current_root_death_proof_missing_or_lost_descendant_refuses_before_copy(self):
        for lost in (False,True):
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                values=list(fixture(directory));kwargs=values[-1]
                if not lost:kwargs['current_death_path']=None;kwargs['current_death_sha']=None
                else:
                    proof=p.read_json(kwargs['current_death_path']);proof['owned_identities'].pop();proof=h.seal(proof);write(kwargs['current_death_path'],proof)
                    kwargs['current_death_sha']=p.file_info(kwargs['current_death_path'])['sha256']
                with self.assertRaises(ValueError):self.run_fixture(values)
                self.assertFalse(values[4].exists())
    def test_changed_raw_and_copy_source_never_publish_final_receipt(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            values=fixture(directory);(values[3]/'stage_CPU_000.json').write_bytes(b'CHANGED')
            with self.assertRaises(ValueError):self.run_fixture(values)
            self.assertFalse(values[4].exists())
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            source=Path(directory)/'source.json';source.write_bytes(b'original');expected=p.file_info(source);source.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError,'stable source'):h.copy_exact(source,Path(directory)/'copy.json',expected)
            self.assertFalse((Path(directory)/'copy.json').exists())
    def test_no_framework_or_pid_probe_imports(self):
        self.assertFalse(any(name in sys.modules for name in ('torch','transformers','tokenizers','native_supervisor','native_process')))
if __name__=='__main__':unittest.main()
