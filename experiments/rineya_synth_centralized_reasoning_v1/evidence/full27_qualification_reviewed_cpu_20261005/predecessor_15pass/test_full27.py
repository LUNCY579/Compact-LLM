"""Stdlib qualification recipe/chunk lineage controls; no native execution."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=='full27_qualification_work' and HERE.parent.name=='artifacts' else HERE.parent
IMPORTS=HERE if HERE.name=='full27_qualification_work' else PACKAGE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(PACKAGE/'tests'));sys.path.insert(0,str(IMPORTS))
import full27_contract as contract
import full27_chunk as chunks
import full27_worker as worker
import full27_supervisor as supervisor
import study_native_chunk as study
import study_schedule as schedule
from test_study_native_chunk import QuadraticAdam,Journal,Preserve


class Adapter:
    expected_gradient_count=1
    def __init__(self,scope,call,guard,dose,**kwargs):
        self.scope,self.dose=scope,dose
        self.backend=QuadraticAdam(scope['arm'],scope['recipe'],call,guard)
        self.backend.metadata=lambda binding: self.metadata(binding)
        def read(descriptor,binding):
            raw=self.backend.invoke('fake_archive_read_verified_bytes',lambda:chunks._read_bytes(descriptor['path'],descriptor['bytes'],descriptor['sha256']))
            payload=json.loads(raw)
            if payload['metadata']!=self.metadata(binding):raise ValueError('fake archive metadata')
            return {'state':payload['state']}
        self.backend._read=read
        original=self.backend.forward
        def forward(row):
            self.length=row['total_tokens']
            return original(row)
        self.backend.forward=forward
        invoke=self.backend.invoke
        def counted(name,function):
            def actual():
                if name=='training_forward':self.dose.charge(self.length)
                return function()
            return invoke(name,actual)
        self.backend.invoke=counted
    def metadata(self,binding):
        b=self.backend
        return {'schema_version':1,'kind':chunks.BACKEND_KIND,'experiment':'SYNTH-CRS-001','scope':b.execution_scope,
            'arm':b.arm,'seed':101,'recipe_sha256':b.recipe_sha256,'cohort_sha256':b.cohort_sha256,
            'backend_source_sha256':b.backend_source_sha256,'fresh_start_attestation_sha256':b.fresh_start_attestation['sha256'],
            'binding':deepcopy(binding),'native_launch_authorized':False,'training_authorized':False,'native_resume_qualified':False,'qualification_STATE1_used':False}
    def zero_parity(self,row):
        for name in ('zero_parity_base_forward','zero_parity_adapter_forward'):
            self.backend.invoke(name,lambda:self.dose.charge(row['total_tokens']))
        return {'equal':True,'max_error':0.,'all_B_zero':True,'CPU_fake_only':True}
    def witness(self,binding):
        return self.backend.invoke('fake_boundary_witness',lambda:self._state_witness(self.backend.state(),binding))
    def _state_witness(self,state,binding):return {'kind':'QUALIFICATION_WITNESS_CPU_FAKE','binding':deepcopy(binding),'state':deepcopy(state),'CPU_fake_only':True}
    def _invoke(self,name,function):return self.backend.invoke(name,function)
    def prepare_reference(self,rows):
        b=self.backend;rng=random.Random();rng.setstate(b.rng.getstate());gradient=0.
        for row in rows:
            target=sum(row['completion_ids'][:-1])/max(1,len(row['completion_ids'])-1)/100000
            def reference_forward():
                self.dose.charge(row['total_tokens'])
                return (b.weight+rng.random()*.01-target)*.05
            gradient+=b.invoke('reference_forward',reference_forward)
        return gradient
    def compare_reference(self,value):return {'passed':abs(self.backend.gradient-value)<=1e-6+1e-5*abs(value),'atol':1e-6,'rtol':1e-5,'count':1,'CPU_fake_only':True}
    def close(self):self.backend.invoke('qualification_backend_close',lambda:None)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration=schedule.read_json(PACKAGE/'STUDY_DEV_REGISTRATION_DRAFT.json')
        cls.pools={arm:{r['row_id']:r for r in schedule.read_jsonl(schedule.BUNDLE/('token_pool_'+arm+'.jsonl'))} for arm in ('M1','M2')}
    def scope(self,arm,chunk,previous=None):
        rows=[deepcopy(self.pools[arm][identity]) for update in self.registration['arms'][arm]['updates'] for identity in update['occurrence_row_ids']]
        return {'purpose':contract.PURPOSE,'authorization':deepcopy(contract.AUTHORIZATION),'study_warmstart_allowed':False,
            'arm':arm,'chunk':chunk,'registration':self.registration,'rows':rows,
            'recipe':chunks.recipe(self.registration,arm,chunks.FAKE_SCOPE),'dose':contract.expected_work(self.registration,arm,chunk,rows),
            'previous':previous,'latest_costs':previous['costs'] if previous else None}
    def execute(self,scope,directory,*,adapter=Adapter,journal=None,sink=None):
        directory=Path(directory);directory.mkdir(exist_ok=True);journal=journal or Journal();guard=lambda stage:None
        def factory(scope,**kwargs):return adapter(scope,**kwargs)
        factory.execution_scope=chunks.FAKE_SCOPE
        return worker.run_component(scope,factory,self.pools[scope['arm']].__getitem__,journal,guard,sink or Preserve(directory))
    def test_three_fresh_chunks_each_arm_complete27_globalLR0_and_lineage(self):
        for arm,total in (('M1',130562),('M2',130574)):
            with tempfile.TemporaryDirectory(dir=HERE) as root:
                previous=None
                for ordinal in (1,2,3):
                    scope=self.scope(arm,ordinal,previous);journal=Journal();guard=lambda stage:None
                    directory=Path(root)/str(ordinal);directory.mkdir();sink=Preserve(directory)
                    def factory(scope,**kwargs):return Adapter(scope,**kwargs)
                    factory.execution_scope=chunks.FAKE_SCOPE
                    result=worker.run_component(scope,factory,self.pools[arm].__getitem__,journal,guard,sink)
                    self.assertEqual('COMPLETE_QUALIFICATION_CHUNK_STUDY_UNAUTHORIZED',result['status'],result.get('error'))
                    self.assertEqual(ordinal*9,result['completed_updates']);self.assertFalse(result['study_warmstart_allowed'])
                    self.assertEqual(scope['dose']['maximum_forward_calls'],result['physical_dose']['attempted_calls'])
                    envelope=result['resume_descriptor']['envelope']
                    self.assertEqual(result['qualification_witness'],envelope['qualification_evidence']['witness'])
                    self.assertEqual(ordinal,result['costs']['primitive_operations']['qualification_backend_close']['returned'])
                    previous=result
                self.assertEqual(total,result['costs']['positions']['input_returned'])
                self.assertEqual(27,result['costs']['protocol']['optimizer_step']['returned'])
                self.assertEqual(2,result['costs']['protocol']['checkpoint_reload']['returned'])
                self.assertEqual(0.,result['qualification_witness']['binding']['learning_rate'])
    def test_qualification_recipe_and_KINDS_cannot_equal_final_study(self):
        a=chunks.recipe(self.registration,'M1',chunks.NATIVE_SCOPE);b=study.recipe(self.registration,'M1',study.NATIVE_SCOPE)
        self.assertNotEqual(a['sha256'],b['sha256']);self.assertNotEqual(chunks.KIND,study.KIND)
        self.assertEqual(contract.PURPOSE,a['purpose']);self.assertFalse(a['study_warmstart_allowed'])
    def test_exact_per_scope_model_call_and_position_caps(self):
        for arm,expected in (('M1',[42982,48556,50689]),('M2',[42866,48532,50840])):
            for index,total in enumerate(expected,1):
                dose=self.scope(arm,index)['dose'];self.assertEqual(total,dose['expected_forward_input_positions'])
                self.assertEqual(182 if index==1 else 202,dose['maximum_forward_calls'])
                self.assertEqual(180,dose['native_backwards']);self.assertEqual(9,dose['native_optimizer_steps']);self.assertLess(total,102400)
    def test_staging_no_native_configs_or_dispatch(self):
        if contract.HERE==contract.PACKAGE:
            stage=PACKAGE/'artifacts/full27_qualification_work'
            code="""import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import full27_contract as c,full27_worker as w,full27_supervisor as s
assert c.HERE!=c.PACKAGE
for action in (lambda:c.preflight(c.HERE/'none.json'),lambda:c.write_scope(c.HERE/'none.json','M1',1),lambda:w.run(c.HERE/'none.json',c.HERE),lambda:s.supervise_backend_qualification(c.HERE/'none.json',c.HERE)):
    try:action()
    except ValueError as error:assert 'staged' in str(error)
    else:raise AssertionError('stage dispatch permitted')
assert not any(n in sys.modules for n in ('torch','transformers'))
"""
            result=subprocess.run([sys.executable,'-I','-B','-c',code,str(stage)],capture_output=True,text=True,timeout=30)
            self.assertEqual(0,result.returncode,result.stderr)
            return
        with self.assertRaisesRegex(ValueError,'staged'):contract.preflight(HERE/'none.json')
        with self.assertRaisesRegex(ValueError,'staged'):contract.write_scope(HERE/'none.json','M1',1)
        with self.assertRaisesRegex(ValueError,'staged'):worker.run(HERE/'none.json',HERE)
        with patch.object(supervisor,'_new_output',side_effect=AssertionError('must not create output')):
            with self.assertRaisesRegex(ValueError,'staged'):supervisor.supervise_backend_qualification(HERE/'none.json',HERE)
        self.assertFalse(any(n in sys.modules for n in ('torch','transformers')))
    def test_whole_packet_drift_denied_before_factory(self):
        scope=self.scope('M1',1);scope['rows'][-1]['input_ids'][0]+=1
        called=[]
        def factory(*args,**kwargs):called.append(True);raise AssertionError('must not construct')
        factory.execution_scope=chunks.FAKE_SCOPE
        with self.assertRaisesRegex(ValueError,'all540'):
            worker.run_component(scope,factory,self.pools['M1'].__getitem__,Journal(),lambda stage:None,lambda event:None)
        self.assertEqual([],called)
    def test_reference_corruption_stops_before_first_resumed_step(self):
        class BadReference(Adapter):
            def prepare_reference(self,rows):return super().prepare_reference(rows)+.1
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            first=self.execute(self.scope('M1',1),Path(root)/'1')
            result=self.execute(self.scope('M1',2,first),Path(root)/'2',adapter=BadReference)
            self.assertTrue(result['status'].startswith('FAILED'));self.assertIsNone(result['resume_descriptor'])
            self.assertEqual(9,result['costs']['protocol']['optimizer_step']['returned'])
            self.assertEqual(20,result['costs']['primitive_operations']['reference_forward']['returned'])
            self.assertEqual(2,result['costs']['primitive_operations']['qualification_backend_close']['returned'])
    def test_cleanup_failure_cannot_publish_or_erase_costs(self):
        class BadClose(Adapter):
            def close(self):
                def fail():raise RuntimeError('cleanup injected')
                self.backend.invoke('qualification_backend_close',fail)
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            sink=Preserve(root);result=self.execute(self.scope('M1',1),root,adapter=BadClose,sink=sink)
            self.assertTrue(result['status'].startswith('FAILED'));self.assertIsNone(result['resume_descriptor'])
            self.assertEqual({'attempted':1,'returned':0,'failed':1,'pending':0},result['costs']['primitive_operations']['qualification_backend_close'])
            self.assertEqual(9,result['costs']['protocol']['optimizer_step']['returned'])
            self.assertNotIn('checkpoint_publish',[event['event'] for event in sink.events])
    def test_publication_failure_after_save_keeps_cleanup_and_consumed_costs(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            sink=Preserve(root);sink.fail='terminal_costs_publish'
            result=self.execute(self.scope('M1',1),root,sink=sink)
            self.assertTrue(result['status'].startswith('FAILED'));self.assertIsNone(result['resume_descriptor'])
            self.assertEqual(1,result['costs']['primitive_operations']['qualification_backend_close']['returned'])
            self.assertEqual(180,result['costs']['protocol']['backward']['returned'])
            self.assertEqual(1,result['costs']['protocol']['terminal_costs_publish']['failed'])
    def test_previous_witness_rebind_requires_durable_bytes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            first=self.execute(self.scope('M1',1),Path(root)/'1');scope=self.scope('M1',2,first)
            envelope=scope['previous']['resume_descriptor']['envelope'];envelope['qualification_evidence']['witness']['state']['weight']+=.1
            evidence=envelope['qualification_evidence'];evidence['sha256']=schedule.digest({k:v for k,v in evidence.items() if k!='sha256'})
            envelope['sha256']=schedule.digest({k:v for k,v in envelope.items() if k!='sha256'})
            resume=scope['previous']['resume_descriptor'];resume['sha256']=schedule.digest({k:v for k,v in resume.items() if k!='sha256'})
            with self.assertRaisesRegex(ValueError,'publication|published'):
                self.execute(scope,Path(root)/'2')
    def test_fully_republished_wrong_witness_still_rejected_by_restored_state(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            first=self.execute(self.scope('M1',1),Path(root)/'1');scope=self.scope('M1',2,first)
            resume=scope['previous']['resume_descriptor'];envelope=resume['envelope'];evidence=envelope['qualification_evidence']
            evidence['witness']['state']['weight']+=.1;evidence['sha256']=schedule.digest({k:v for k,v in evidence.items() if k!='sha256'})
            envelope['sha256']=schedule.digest({k:v for k,v in envelope.items() if k!='sha256'})
            rebuilt=Path(root)/'rebuilt';rebuilt.mkdir();writer=Preserve(rebuilt)
            resume['publication']=writer({'event':'checkpoint_publish','payload':envelope})
            terminal=resume['durable_terminal'];terminal['envelope_sha256']=envelope['sha256']
            resume['terminal_publication']=writer({'event':'terminal_costs_publish','payload':terminal})
            promotion=resume['promotion'];promotion['envelope_sha256']=envelope['sha256'];promotion['durable_terminal_sha256']=schedule.digest(terminal)
            resume['promotion_publication']=writer({'event':'checkpoint_promote','payload':promotion})
            resume['sha256']=schedule.digest({k:v for k,v in resume.items() if k!='sha256'})
            result=self.execute(scope,Path(root)/'2')
            self.assertTrue(result['status'].startswith('FAILED'));self.assertIsNone(result['resume_descriptor'])
            self.assertIn('restored qualification boundary',result['error'])
            self.assertEqual(9,result['costs']['protocol']['optimizer_step']['returned'])
            self.assertEqual(1,result['costs']['primitive_operations']['checkpoint_state_restore']['returned'])
    def test_valid_hash_wrong_serialized_state_cannot_promote(self):
        class WrongArchive(Adapter):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);save=self.backend.save_checkpoint;state=self.backend.state
                def corrupted():
                    value=state();value['weight']+=.1;return value
                def bad_save(path,binding):
                    self.backend.state=corrupted
                    try:return save(path,binding)
                    finally:self.backend.state=state
                self.backend.save_checkpoint=bad_save
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            sink=Preserve(root);result=self.execute(self.scope('M1',1),root,adapter=WrongArchive,sink=sink)
            self.assertTrue(result['status'].startswith('FAILED'));self.assertIsNone(result['resume_descriptor'])
            self.assertIn('serialized boundary bytes',result['error'])
            self.assertEqual(1,result['costs']['primitive_operations']['checkpoint_write']['returned'])
            self.assertNotIn('checkpoint_publish',[event['event'] for event in sink.events])
    def test_wrong_arm_and_rollback_denied_before_factory(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            first=self.execute(self.scope('M1',1),Path(root)/'1')
            for arm in ('M2','M1'):
                scope=self.scope(arm,2,first)
                if arm=='M1':scope['latest_costs']['positions']['input_returned']-=1
                with self.assertRaises(ValueError):self.execute(scope,Path(root)/arm)
    def test_native_lazy_admission_denied_before_factory(self):
        scope=self.scope('M1',1);scope['recipe']=chunks.recipe(self.registration,'M1',chunks.NATIVE_SCOPE);called=[]
        def factory(*args,**kwargs):called.append(True);raise AssertionError('must not construct')
        factory.execution_scope=chunks.NATIVE_SCOPE
        with self.assertRaisesRegex(ValueError,'admission'):
            worker.run_component(scope,factory,self.pools['M1'].__getitem__,Journal(),lambda stage:None,lambda event:None)
        self.assertEqual([],called)
    def test_old_qualification_audit_cannot_be_rebranded_completed_B(self):
        with self.assertRaisesRegex(ValueError,'trusted'):
            contract.verify_completed_B(PACKAGE/'evidence/infq001_attempt1_independent_audit_20261004.json','0'*64)
    def test_exact_checkpoint_counts_and_unchanged_native_floors(self):
        for ordinal,count in ((1,180),(2,200),(3,200)):
            with patch.object(contract.prerequisite,'checkpoint_binding',return_value={'counts':{}}):
                amendment=contract.checkpoint_binding(ordinal)
            self.assertEqual(28*count,amendment['counts']['recompute'])
            self.assertEqual(56,amendment['counts']['no_grad'])
        self.assertEqual(600,contract.native.RESOURCE_BOUNDS['whole_scope_seconds'])
        self.assertEqual(.2,contract.native.RESOURCE_BOUNDS['CPU_sample_max_age_seconds'])
        self.assertEqual(4.,contract.native.RESOURCE_BOUNDS['GPU_free_at_admission_GiB'])
        self.assertEqual(.5,contract.native.RESOURCE_BOUNDS['GPU_free_during_work_GiB'])


if __name__=='__main__':unittest.main()
