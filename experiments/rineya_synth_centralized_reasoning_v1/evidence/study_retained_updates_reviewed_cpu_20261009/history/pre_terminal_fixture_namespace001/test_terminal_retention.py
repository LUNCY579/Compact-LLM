"""Real inert retained producer -> original-byte terminal consumer; no native evidence."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import ast, hashlib, json, math, os, random, sys, tempfile, time, unittest

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
READS=[];IMPORTS=[]
def audit(event,args):
    if event=='import' and args and args[0].split('.',1)[0] in {'torch','transformers','tokenizers','safetensors','numpy','psutil','ctypes','jax','tensorflow'}:
        IMPORTS.append(args[0]);raise AssertionError('framework/native import denied')
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=str(args[0]).replace('\\','/').lower()
        if any(x in path for x in ('/data/','/weights','/model','/check/','/final/')):raise AssertionError('actual data/model/bank read denied')
        READS.append(str(args[0]))
sys.addaudithook(audit)
sys.path.insert(0,str(PACKAGE))
import study_schedule as schedule

def module(path,name):
    value=ModuleType(name);value.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),value.__dict__);return value
chunk=module(HERE/'study_native_chunk.py','terminal_authored_chunk')
A=module(HERE/'study_terminal_audit.py','terminal_authored_audit')
namespace={'deepcopy':deepcopy,'Path':Path,'hashlib':hashlib,'json':json,'math':math,'os':os,'random':random,'chunk':chunk,'schedule':schedule}
def definitions(path,names):
    tree=ast.parse(path.read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
definitions(PACKAGE/'tests/test_study_native_chunk.py',{'QuadraticAdam'})
definitions(HERE/'test_retained_updates.py',{'ScalarBackend','authored','Preserve'})
REGISTRATION,POOLS=namespace['authored']();namespace['REGISTRATION']=REGISTRATION
ScalarBackend=namespace['ScalarBackend'];OriginalPreserve=namespace['Preserve']

def ref(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def write(path,value):path.write_bytes((A.canonical(value)+'\n').encode('ascii'));return ref(path)

class Journal:
    def __init__(self):self.rows=[];self.counts={};self.last=0
    def emit(self,event):self.rows.append({**event,'utc':datetime.now(timezone.utc).isoformat()})
    def __call__(self,name,function):
        self.last+=1;identity=self.last;counts=self.counts.setdefault(name,dict(attempted=0,returned=0,failed=0,pending=0));counts['attempted']+=1;counts['pending']+=1
        self.emit({'event':'attempt','id':identity,'operation':name})
        try:value=function()
        except BaseException:counts['failed']+=1;counts['pending']-=1;self.emit({'event':'failed','id':identity,'operation':name});raise
        counts['returned']+=1;counts['pending']-=1;self.emit({'event':'returned','id':identity,'operation':name});return value

class Preserve(OriginalPreserve):
    def __init__(self,leaf,journal):super().__init__(leaf);self.journal=journal
    def __call__(self,event):
        value=super().__call__(event)
        if event['event'] not in ('checkpoint_path','retained_update_path'):self.journal.emit({'event':'study_publication_preserved','name':event['event'],'publication':value})
        return value

class TerminalRetentionControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_terminal_retention_',dir=HERE);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(schedule,'validate_registration',lambda value:chunk.require(value==REGISTRATION,'inert registration only')).start()
        patch.object(schedule,'_inputs',lambda:({}, {}, deepcopy(POOLS),{})).start()
    def packet(self,arm='M1',ordinal=1,previous=None,mutation=None):
        root=self.root/(arm+str(ordinal)+'_'+str(time.monotonic_ns()));root.mkdir();leaf=root/'owned_output';leaf.mkdir()
        journal=Journal();policy=chunk.producer_policy();recipe=chunk.recipe(REGISTRATION,arm,chunk.NATIVE_SCOPE,retained_update_policy=policy)
        class Lazy:
            execution_scope=chunk.NATIVE_SCOPE
            required_retained_update_policy=policy
            def __call__(self,**kwargs):return ScalarBackend(arm,recipe,kwargs['call'],kwargs['guard'],scope=self.execution_scope)
        def admission(request):
            now=time.monotonic();return {'binding':request,'authorized':True,'owned_supervision_verified':True,'model_load_and_constructor_in_scope':True,'observed_monotonic':now,'expires_monotonic':now+.1}
        result=chunk.run_chunk(REGISTRATION,arm,ordinal,Lazy(),POOLS[arm].__getitem__,journal,lambda stage:None,Preserve(leaf,journal),retained_update_policy=policy,resume_descriptor=previous,latest_terminal_costs=None if previous is None else previous['terminal_costs'],admission=admission)
        self.assertEqual(result['status'],'COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED',result.get('error'))
        config={'scope':A.SCOPE,'fixture_only':True,'arm':arm,'chunk':ordinal,'source_pins':{str(PACKAGE/'auditor.py'):A.info(PACKAGE/'auditor.py')['sha256']},'previous_resume':None,'latest_terminal_costs':None,'retained_update_policy':policy}
        prior=prior_latest=None
        if previous is not None:
            prior=write(root/'previous.json',previous);prior_latest=write(root/'previous_costs.json',previous['terminal_costs']);config.update(previous_resume=prior,latest_terminal_costs=prior_latest)
        config_ref=write(root/'config.json',config)
        worker={'fixture_only':True,'worker_scope':A.SCOPE,'config_sha256':config_ref['sha256'],'native_PASS':False,'scoring_authorized':False,'quality_gain_established':False,'protocol_result':result,'terminal_audit':{'kind':'WORKER_STUDY_PENDING_POST_JOB_TERMINAL_AUDIT','owned_cleanup_returned':True,'native_PASS':False,'operation_accounting':{'operations':journal.counts,'last_call':journal.last}}}
        if mutation:mutation(journal,worker,leaf)
        worker=A.seal(worker);write(leaf/'worker_result.json',worker)
        (leaf/'events.jsonl').write_bytes(''.join(A.canonical(row)+'\n' for row in journal.rows).encode('ascii'))
        births=[{'pid':7001,'creation_filetime_100ns':116444736010000001},{'pid':7002,'creation_filetime_100ns':116444736010000002}]
        identity={**births[0],'create_time':123.,'job_name':'Local\\SYNTH_NQ_JOB_'+'a'*32,'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False}
        owner={'worker_pid':7001,'supervisor_pid':6001,'supervisor_create_time':123.,'job_name':identity['job_name'],'config_sha256':config_ref['sha256']}
        write(leaf/'supervisor_ownership.json',owner);write(leaf/'stage_study_worker_ownership_verified.json',{'actual_worker_pid':7002,'launcher_root_pid':7001,'supervisor_pid':6001,'job_name':identity['job_name'],'owned_job_membership_verified':True,'query_job_handle_closed_before_framework_import':True})
        def row(start,gpu=False,owned=False):
            v={'sample_started_monotonic':start,'sample_completed_monotonic':start+.01,'host_available_bytes':6*1024**3,'disk_free_bytes':6*1024**3,'worker_tree_rss_bytes':512 if owned else 0,'GPU_index':0,'gpu_uuid':'CPU_AUTHORED_NOT_GPU','gpu_observation_id':'CPU_'+str(start),'gpu_total_bytes':8*1024**3,'gpu_free_bytes':6*1024**3,'gpu_sample_started_monotonic':start,'gpu_sample_completed_monotonic':start+.01,'kind':'GPU' if gpu else 'CPU'}
            if owned:v.update(owned_process_identities=[{**b,'create_time':123.} for b in births],owned_process_samples=[{**b,'create_time':123.,'rss_bytes':256,'sample_started_monotonic':start,'sample_completed_monotonic':start+.005} for b in births])
            return v
        write(leaf/'RESOURCE_OBSERVATIONS.json',[{**row(10.,owned=True),'phase':'worker_suspended_startup'},row(10.1,owned=True),row(10.,gpu=True),row(10.1,gpu=True)])
        (leaf/'supervisor_heartbeat.jsonl').write_bytes(''.join(A.canonical({'unix_seconds':100+i*.1,'supervisor_pid':6001})+'\n' for i in range(3)).encode('ascii'))
        write(leaf/'worker_GPU_final.json',{'CUDA_initialized':True,'peak_allocated_bytes':1,'peak_reserved_bytes':2,'free_bytes_final':6*1024**3,'CPU_authored_no_GPU_observation':True})
        lease_root=root/'_native_gpu_leases';lease=lease_root/('c'*32);lease.mkdir(parents=True)
        sup={'status':'SUPERVISED_WORKER_EXITED_ZERO','returncode':0,'partial_outputs_frozen':True,'admission_only':False,'bounds':{'worker_process_tree_RSS_GiB':2.,'system_available_reserve_GiB':1.25,'preworker_available_GiB':3.3125,'GPU_free_at_admission_GiB':4.,'GPU_free_during_work_GiB':.5,'disk_free_GiB':4.,'whole_scope_seconds':600,'external_watchdog_milliseconds':100},'cadence':{'CPU_watchdog_seconds':.1,'CPU_freshness_seconds':.2,'GPU_query_interval_seconds':.5,'GPU_query_timeout_seconds':2.,'GPU_freshness_seconds':2.,'admission_interval_seconds':2.,'admission_deadline_seconds':30.,'admission_samples':2},'config_sha256':config_ref['sha256'],'worker_identity':identity,'worker_resume_monotonic':10.02,'deadline_monotonic':607.,'lease_path':str(lease),'elapsed_seconds':5.,'owned_worker_identity_currently_alive':False,'unjoined_sampler_threads':[],'admission_observations':[row(7.),row(9.)],'dispatch_observation':row(9.)}
        for key in ('worker_attempted','worker_launched','worker_resumed','owned_handle_alive_verified_after_launch','preflight_revalidated_before_current_observations','job_closed_kill_on_close','sampler_threads_terminal_verified','gpu_telemetry_cleanup_verified'):sup[key]=True
        sup['partial_outputs']=[{'path':p.name,**A.info(p)} for p in sorted(leaf.iterdir())]
        write(leaf/'SUPERVISOR_RECEIPT.json',sup);(lease/'terminal.json').write_bytes((leaf/'SUPERVISOR_RECEIPT.json').read_bytes())
        write(lease/'worker.json',identity);write(lease/'resume.json',{'worker_identity':identity,'external_monitor_ready_before_resume':True,'resumed_monotonic':10.02,'deadline_monotonic':607.});write(lease/'reservation.json',{'lease_id':lease.name,'supervisor':{'pid':6001,'create_time':123.},'output_dir':str(leaf),'config_sha256':config_ref['sha256'],'GPU_index':0})
        approval=write(root/'approval.json',{'scope':A.SCOPE,'fixture_only':True,'approved':True,'source_pins':config['source_pins']});review=write(root/'review.json',{'accepted':True,'source_pins':config['source_pins'],'approval_file_sha256':approval['sha256']})
        packet={'fixture_only':True,'output_directory':str(leaf),'lease_root':str(lease_root),'config':config_ref,'approval':approval,'review':review,'sources':{str(PACKAGE/'auditor.py'):ref(PACKAGE/'auditor.py')},'worker':ref(leaf/'worker_result.json'),'supervisor':ref(leaf/'SUPERVISOR_RECEIPT.json'),'training_registration':write(root/'registration.json',REGISTRATION),'latest_terminal_costs':write(root/'latest_costs.json',result['costs']),'original_auditor_sha256':config['source_pins'][str(PACKAGE/'auditor.py')],'raw_files':{p.name:A.info(p) for p in leaf.iterdir()}}
        if previous is not None:packet.update(previous_resume=prior,previous_latest_terminal_costs=prior_latest)
        host=type('InertDeadBirthObserver',(),{'is_alive':lambda self,birth:False})();return packet,host
    def consume(self,packet,host):return A._CPU_consume(packet,chunk,host,capability=A._CPU_capability())
    def test_actual_all27_both_arms_original_journal_resources_lease_and_death(self):
        for arm in ('M1','M2'):
            previous=None
            for ordinal in (1,2,3):
                packet,host=self.packet(arm,ordinal,previous);value=self.consume(packet,host);self.assertEqual((value['arm'],value['chunk']),(arm,ordinal));self.assertFalse(value['native_PASS']);self.assertEqual(value['original_validator_adaptations'],{'event':1,'ownership':1,'lease':1})
                previous=A.read_ref(packet['worker'])['protocol_result']['resume_descriptor']
                self.assertEqual(len(previous['envelope']['retained_updates']),ordinal*9)
            self.assertEqual(previous['envelope']['binding']['learning_rate'],0.);self.assertEqual(previous['terminal_costs']['primitive_operations']['retained_update_save']['returned'],27)
    def test_missing_duplicate_and_wrong_order_publications_reject(self):
        def missing(journal,worker,leaf):journal.rows.pop(next(i for i,row in enumerate(journal.rows) if row.get('name')=='retained_update_publish'))
        def duplicate(journal,worker,leaf):
            i=next(i for i,row in enumerate(journal.rows) if row.get('name')=='retained_update_publish');journal.rows.insert(i,deepcopy(journal.rows[i]))
        def reorder(journal,worker,leaf):
            ids=[i for i,row in enumerate(journal.rows) if row.get('name')=='retained_update_publish'];journal.rows[ids[0]]['publication'],journal.rows[ids[1]]['publication']=journal.rows[ids[1]]['publication'],journal.rows[ids[0]]['publication']
        for mutation in (missing,duplicate,reorder):
            packet,host=self.packet(mutation=mutation)
            with self.subTest(mutation=mutation.__name__),self.assertRaises(ValueError):self.consume(packet,host)
    def test_GD_missing_extra_and_rebound_update_publication_reject(self):
        def missing(journal,worker,leaf):journal.rows.pop(next(i for i,row in enumerate(journal.rows) if row.get('name')=='GD_export_parity_request'))
        def extra(journal,worker,leaf):
            row=next(row for row in journal.rows if row.get('name')=='retained_update_publish');row['name']='GD_export_parity_request'
        def rebound(journal,worker,leaf):
            result=worker['protocol_result'];result['retained_updates'][0]['record']['binding']['completed_updates']=2
        for mutation in (missing,extra,rebound):
            packet,host=self.packet(mutation=mutation)
            with self.subTest(mutation=mutation.__name__),self.assertRaises(ValueError):self.consume(packet,host)
    def test_corrupt_publication_and_archive_drift_reject(self):
        packet,host=self.packet();worker=A.read_ref(packet['worker']);entry=worker['protocol_result']['retained_updates'][0]
        for path in (Path(entry['publication']['path']),Path(entry['record']['backend_descriptor']['path'])):
            before=path.read_bytes();path.write_bytes(before+b'CHANGED_INERT')
            with self.subTest(path=path.name),self.assertRaises(ValueError):self.consume(packet,host)
            path.write_bytes(before)
    def test_unselected_current_archive_and_extra_retention_cost_reject(self):
        packet,host=self.packet();archive=A.read_ref(packet['worker'])['protocol_result']['retained_updates'][0]['record']['backend_descriptor'];candidate=deepcopy(packet);candidate['raw_files'].pop(Path(archive['path']).name)
        with self.assertRaises(ValueError):self.consume(candidate,host)
        def extra(journal,worker,leaf):
            journal('retained_update_save',lambda:None);worker['terminal_audit']['operation_accounting']={'operations':journal.counts,'last_call':journal.last}
        packet,host=self.packet(mutation=extra)
        with self.assertRaisesRegex(ValueError,'only original exact single'):self.consume(packet,host)
    def test_live_birth_and_changed_guards_still_reject(self):
        packet,host=self.packet();host.is_alive=lambda birth:True
        with self.assertRaisesRegex(ValueError,'birth still live'):self.consume(packet,host)
        packet,host=self.packet();path=Path(packet['supervisor']['path']);sup=A.read_ref(packet['supervisor']);sup['bounds']['GPU_free_at_admission_GiB']=3.;write(path,sup);packet['supervisor']=ref(path)
        with self.assertRaisesRegex(ValueError,'unchanged exact supervised'):self.consume(packet,host)
    def test_stage_gate_no_receipts_or_native_API(self):
        with self.assertRaisesRegex(ValueError,'staged post-Job'):A.audit_registered({})

if __name__=='__main__':
    sources=['study_terminal_audit.py','study_native_chunk.py','study_torch_backend.py','test_terminal_retention.py','test_retained_updates.py'];before={name:A.info(HERE/name) for name in sources};start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TerminalRetentionControls));after={name:A.info(HERE/name) for name in sources}
    value=A.seal({'kind':'AUTHORED_RETAINED_UPDATE_TERMINAL_CONTROLS_V1','tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'success':result.wasSuccessful(),'seconds':time.monotonic()-start,'source_before':before,'source_after':after,'source_unchanged':before==after,'read_paths':sorted(set(READS)),'forbidden_import_attempts':IMPORTS,'fixture_only':True,'native_PASS':False,'native_API_run':False,'actual_export_parity_established':False,'real_corpus_or_bank_read':False,'limits':['Exact staged chunk producer writes real inert scalar archives and complete original journal/publications. SUP/Job/resource/lease/death evidence is explicitly authored.','No framework, native sampler, model, real corpus or protected bank reads. No actual export parity or launch authority.']})
    target=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'TERMINAL_CPU_CONTROLS001.json'
    with target.open('xb') as stream:stream.write((A.canonical(value)+'\n').encode('ascii'))
    print(json.dumps({'path':str(target),'tests':result.testsRun,'success':result.wasSuccessful(),'source_unchanged':before==after,'sha256':value['sha256']}));raise SystemExit(0 if result.wasSuccessful() and before==after else 1)
