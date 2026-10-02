"""Explicit metadata-only lease FSM fixtures; no card, OS handle or child spawn."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_checkpoint_lease as c
from experiments.skill_acquisition_native_v1 import ce_native_authority as a
from experiments.skill_acquisition_native_v1 import ce_worker_evidence as w


class Trap:
    def __getattribute__(self,name): raise AssertionError('lease IO/native objects before authority: '+name)


def spec(root):
    profile={'read_bytes_upper':50000,'write_bytes_upper':50000,'identity_hash_attempts_upper':10,
             'identity_hash_bytes_upper':50000,'wall_s_upper':60}
    return {'schema':c.SPEC,'lock_path':str(root/'rineya-gpu-0.lock'),'label':'comp-rs-001-lease-fixture',
        'timeout_seconds':1,'terminal_verify_timeout_seconds':1,'lock_bytes_upper':2048,
        'parent_released_cuda_bytes_upper':0,'child_released_cuda_bytes_upper':0,
        'helper_ref':{'path':str(root/'gpu_lease.py'),'sha256':'f'*64,'bytes':100},
        'operation_bounds':{key:dict(profile) for key in ('acquire','transfer','release')}}


def proof(pid,identity):
    return {'schema':w.PROOF,'worker_identity_sha256':identity,'request_sha256':'a'*64,'scope_sha256':'b'*64,
        'fresh_process':False,'source_verified':False,'boot_verified':False,'native_qualified':False,
        'os_identity':{'schema':w.IDENTITY,'pid':pid,'creation_filetime_100ns':10000+pid,
            'boot_identity':'fixture-boot','image_path':'fixture-python.exe','argv':['fixture-python.exe','-m','fixture']}}


class FixtureModule:
    def __init__(self,path): self.LOCK=path; self.acquisitions=0; self.releases=0; self.updates=[]
    def acquire(self,*,timeout_s,label):
        self.acquisitions+=1
        if self.LOCK.exists(): raise RuntimeError('existing owner; no lock stealing')
        record={'pid':111,'token':'owned-fixture-token','created':1.5,'label':label,'worker_pids':[],
                'protocol':'explicit synthetic fixture; no GPU observation'}
        with self.LOCK.open('x') as stream: json.dump(record,stream)
        return record['token']
    def set_workers(self,token,pids):
        record=json.loads(self.LOCK.read_bytes())
        if record['token']!=token: return
        record['worker_pids']=list(pids); self.LOCK.write_text(json.dumps(record)); self.updates.append(list(pids))
    def release(self,token):
        record=json.loads(self.LOCK.read_bytes())
        if record['token']==token: self.LOCK.unlink(); self.releases+=1


class Observer:
    def __init__(self): self.events=[]; self.deny=None
    def __call__(self,kind,phase,**fields):
        self.events.append((kind,phase,fields))
        if self.deny==(phase,fields.get('lease_action')): raise RuntimeError('paid observer refused')


class FixtureChild:
    def __init__(self,parent,branch,pid=222):
        self.current=proof(pid,('%x'% (pid%16))*64)
        self.ref={'path':str(parent.root/(branch+'.json')),'sha256':c.digest(branch),'bytes':100}
        self.current['sources']={'request_ref':copy.deepcopy(self.ref)}
        p=parent.parent['os_identity']
        self.spec={'request_ref':self.ref,'request_sha256':'a'*64,'scope_sha256':'b'*64,
                   'parent_identity':{key:p[key] for key in ('pid','creation_filetime_100ns','boot_identity')}}
        self.signaled=False; self.exit_code=0; self.terminal_reads=0
    def proof(self): return copy.deepcopy(self.current)
    def terminal_proof(self):
        if not self.signaled: raise ValueError('actual fixture handle still live')
        return {'schema':w.TERMINAL,'worker_identity_sha256':self.current['worker_identity_sha256'],
            'request_sha256':'a'*64,'scope_sha256':'b'*64,'os_identity':copy.deepcopy(self.current['os_identity']),
            'exit_code':self.exit_code,'exit_filetime_100ns':20000,'status_file_used':False,'native_qualified':False}
    def terminal(self,timeout):
        self.terminal_reads+=1
        actual=self.terminal_proof(); actual['schema']='CE8-OWNED-PROCESS-TERMINAL-1'
        actual['launcher_identity']={'pid':333,'creation_filetime_100ns':9999,'boot_identity':'fixture-boot'}
        return actual


class LeaseControls(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root=Path(temp.name)
        self.scope=spec(self.root)
        self.context={'schema':c.FIXTURE,'mode':'synthetic_fixture','approved':False,'training_authorized':False,
            'request_sha256':'a'*64,'scope_sha256':'b'*64,'bindings':{'scope':{'checkpoint_lease':self.scope}}}
        self.parent=proof(111,'c'*64)
        self.backend=SimpleNamespace(model=None,optimizer=None)
        self.module=FixtureModule(Path(self.scope['lock_path'])); self.observer=Observer()
        self.lease=c.fixture_supervisor(self.context,self.backend,self.parent,self.module,self.observer)

    def suspend(self):
        self.lease.acquire()
        return self.lease.release_parent_lease(self.context,None,self.parent)

    def test_native_denies_before_objects_io_import_or_grant(self):
        for permit in (None,True,{'approved':True},object.__new__(a.NativeExecutionPermit)):
            with self.assertRaises(a.NativeAdmissionError): c.SupervisorGpuLease(Trap(),permit,Trap(),Trap(),Trap(),Trap())
            with self.assertRaises(a.NativeAdmissionError): c.DelegatedGpuLease(Trap(),permit,Trap(),Trap(),Trap())
        self.assertFalse(a._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)
        self.assertNotIn('torch',sys.modules); self.assertNotIn('transformers',sys.modules)

    def test_spec_lower_bounds_and_boolean_caps_refuse(self):
        c.validate_spec(self.scope)
        for mutate in (lambda s:s.update(timeout_seconds=True),lambda s:s.update(lock_bytes_upper=True),
            lambda s:s.update(parent_released_cuda_bytes_upper=True),
            lambda s:s['operation_bounds']['transfer'].update(read_bytes_upper=8195),
            lambda s:s['operation_bounds']['transfer'].update(identity_hash_attempts_upper=2),
            lambda s:s['operation_bounds']['acquire'].update(read_bytes_upper=100),
            lambda s:s['operation_bounds']['release'].update(wall_s_upper=True)):
            value=copy.deepcopy(self.scope); mutate(value)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError): c.validate_spec(value)

    def test_parent_keeps_one_shared_lock_for_all_serial_branches(self):
        ticket=self.suspend(); before=json.loads(self.module.LOCK.read_bytes())
        self.assertTrue(self.module.LOCK.exists()); self.assertEqual(self.module.acquisitions,1); self.assertEqual(self.module.releases,0)
        for index,branch in enumerate(c.BRANCHES):
            child=FixtureChild(self,branch,pid=222+index)
            delegation=self.lease.bind_child(child,branch,child.ref)
            self.assertEqual(delegation['child_worker_identity_sha256'],child.current['worker_identity_sha256'])
            self.assertEqual(delegation['owner_marker']['token'],before['token']); self.assertFalse(delegation['native_qualified'])
            with self.assertRaisesRegex(ValueError,'suspended'): self.lease.assert_parent_use()
            child.signaled=True; terminal=child.terminal(1)
            self.lease.finish_child(child,terminal)
            self.assertGreaterEqual(child.terminal_reads,2)  # supervisor rechecks owned process itself
            self.assertTrue(self.module.LOCK.exists()); self.assertEqual(self.module.releases,0)
        self.lease.reacquire_parent_lease(self.context,None,ticket); self.lease.assert_parent_use()
        self.lease.close(); self.assertEqual(self.module.releases,1); self.assertFalse(self.module.LOCK.exists())
        self.lease.close(); self.assertEqual(self.module.releases,1)
        self.assertEqual(self.module.updates,[[222],[],[223],[],[224],[]])
        starts=[fields for _,phase,fields in self.observer.events if phase=='started']
        self.assertTrue(starts[0]['deferred_owned_lease_release'])

    def test_parent_model_must_be_released_before_transfer_or_lock_release(self):
        self.lease.acquire(); self.backend.model=object()
        with self.assertRaisesRegex(ValueError,'parent model'): self.lease.release_parent_lease(self.context,None,self.parent)
        with self.assertRaisesRegex(ValueError,'parent model'): self.lease.close()
        self.assertTrue(self.module.LOCK.exists()); self.assertEqual(self.module.releases,0)
        self.backend.model=None; self.lease.close()

    def test_live_or_unverified_child_blocks_next_transfer_parent_reload_and_close(self):
        ticket=self.suspend(); child=FixtureChild(self,'uninterrupted')
        self.lease.bind_child(child,'uninterrupted',child.ref)
        for work in (lambda:self.lease.bind_child(FixtureChild(self,'prefix'),'prefix',child.ref),
                     lambda:self.lease.reacquire_parent_lease(self.context,None,ticket),lambda:self.lease.close()):
            with self.assertRaises(ValueError): work()
        flags={'status_file_used':False,'exit_code':0,'terminal':True}
        with self.assertRaisesRegex(ValueError,'still live'): self.lease.finish_child(child,flags)
        self.assertEqual(self.lease.state,'CHILD'); self.assertTrue(self.module.LOCK.exists())
        child.signaled=True; self.lease.finish_child(child,child.terminal(1))
        self.lease.reacquire_parent_lease(self.context,None,ticket)
        with self.assertRaisesRegex(ValueError,'failed supervisor'): self.lease.assert_parent_use()
        self.lease.close()

    def test_original_owner_marker_cannot_be_replaced_even_with_same_token(self):
        self.lease.acquire(); record=json.loads(self.module.LOCK.read_bytes()); record['pid']=999
        self.module.LOCK.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'original runtime owner'): self.lease.close()
        self.assertTrue(self.module.LOCK.exists()); self.assertEqual(self.module.releases,0)
        self.assertEqual(json.loads(self.module.LOCK.read_bytes())['pid'],999)

    def test_existing_unknown_lock_is_never_stolen_or_deleted(self):
        self.module.LOCK.write_text(json.dumps({'pid':999,'token':'foreign','worker_pids':[]}))
        with self.assertRaisesRegex(RuntimeError,'no lock stealing'): self.lease.acquire()
        self.assertTrue(self.module.LOCK.exists()); self.assertEqual(self.module.releases,0)

    def test_current_request_parent_lineage_and_branch_are_exact(self):
        self.suspend()
        with self.assertRaisesRegex(ValueError,'serial'): self.lease.bind_child(FixtureChild(self,'prefix'),'prefix',{})
        child=FixtureChild(self,'uninterrupted'); child.spec['parent_identity']['pid']=999
        with self.assertRaisesRegex(ValueError,'parent OS lineage'): self.lease.bind_child(child,'uninterrupted',child.ref)
        self.assertEqual(self.lease.state,'SUSPENDED'); self.assertEqual(self.module.updates,[])
        self.lease.close()

    def test_foreign_terminal_or_response_flags_cannot_replace_retained_os_evidence(self):
        self.suspend(); child=FixtureChild(self,'uninterrupted'); self.lease.bind_child(child,'uninterrupted',child.ref)
        child.signaled=True; terminal=child.terminal(1); terminal['worker_identity_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'OS-terminal'): self.lease.finish_child(child,terminal)
        self.assertEqual(self.lease.state,'CHILD'); self.assertEqual(self.module.updates,[[222]])
        self.lease.finish_child(child,child.terminal(1)); self.lease.close()

    def test_unsuccessful_owned_exit_allows_release_but_not_more_gpu_work(self):
        ticket=self.suspend(); child=FixtureChild(self,'uninterrupted'); self.lease.bind_child(child,'uninterrupted',child.ref)
        child.signaled=True; child.exit_code=1
        with self.assertRaisesRegex(ValueError,'unsuccessfully'): self.lease.finish_child(child,child.terminal(1))
        self.assertEqual(self.lease.state,'SUSPENDED'); self.assertEqual(self.module.updates,[[222],[]])
        self.lease.reacquire_parent_lease(self.context,None,ticket)
        with self.assertRaisesRegex(ValueError,'failed supervisor'): self.lease.assert_parent_use()
        self.lease.close()

    def test_reservation_failure_precedes_helper_access_and_acquire_return_failure_releases_own_lock(self):
        self.observer.deny=('started','acquire')
        with self.assertRaisesRegex(RuntimeError,'paid observer'): self.lease.acquire()
        self.assertEqual(self.module.acquisitions,0); self.assertEqual(self.module.releases,0)
        self.observer.deny=None
        # A separate source-created fixture tests failure after physical acquire.
        other=c.fixture_supervisor(self.context,self.backend,self.parent,self.module,self.observer)
        self.observer.deny=('returned','acquire')
        with self.assertRaisesRegex(RuntimeError,'paid observer'): other.acquire()
        self.assertEqual(self.module.acquisitions,1); self.assertEqual(self.module.releases,1)
        self.assertFalse(self.module.LOCK.exists())

    def test_release_of_genuine_prepaid_ownership_survives_later_admission_failure(self):
        self.lease.acquire(); self.context['scope_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'frozen lease'): self.lease.close()
        self.assertEqual(self.module.releases,1); self.assertFalse(self.module.LOCK.exists())
        self.assertEqual(self.observer.events[-1][:2],('supervisor_lease_release','failed'))

    def test_lazy_backend_is_bound_once_after_lease_and_before_any_model_work(self):
        lease=c.fixture_supervisor(self.context,None,self.parent,self.module,self.observer)
        lease.acquire(); self.assertIsNone(lease.backend)
        lease.assert_parent_use()
        backend=SimpleNamespace(model=None,optimizer=None,tokenizer=None,torch=None,closed=False)
        lease.bind_backend(backend); self.assertIs(lease.backend,backend)
        with self.assertRaisesRegex(ValueError,'one metadata backend'): lease.bind_backend(backend)
        lease.close()
        another=c.fixture_supervisor(self.context,None,self.parent,self.module,self.observer)
        another.acquire(); backend.model=object()
        with self.assertRaisesRegex(ValueError,'metadata-only'): another.bind_backend(backend)
        self.assertIsNone(another.backend); another.close()

    def test_bad_child_evidence_stays_fatal_even_after_successful_os_exit(self):
        ticket=self.suspend(); child=FixtureChild(self,'uninterrupted'); self.lease.bind_child(child,'uninterrupted',child.ref)
        error=ValueError('raw branch digest mismatch')
        self.lease.mark_child_evidence_failed(child,error)
        child.signaled=True; self.lease.finish_child(child,child.terminal(1))
        self.lease.reacquire_parent_lease(self.context,None,ticket)
        with self.assertRaisesRegex(ValueError,'failed supervisor'): self.lease.assert_parent_use()
        self.assertEqual(self.lease.evidence[-3]['action'],'child_evidence_failed')
        self.lease.close()

    def test_incomplete_branch_unit_cannot_resume_parent_computation(self):
        ticket=self.suspend(); self.lease.reacquire_parent_lease(self.context,None,ticket)
        with self.assertRaisesRegex(ValueError,'failed supervisor'): self.lease.assert_parent_use()
        self.lease.close(); self.assertEqual(self.module.releases,1)

    def test_child_entry_binds_current_raw_lock_and_lineage_and_never_acquires(self):
        self.suspend(); child=FixtureChild(self,'uninterrupted')
        delegation=self.lease.bind_child(child,'uninterrupted',child.ref)
        context=copy.deepcopy(self.context)
        context['checkpoint_child_lease_delegation']=delegation
        context['checkpoint_child_observed_lineage']={'child_proof':child.proof(),
            'parent_worker_identity_sha256':self.parent['worker_identity_sha256'],'parent_proof':self.parent}
        handle=object.__new__(w.WinWorkerHandle); token=object(); observer=Observer(); gate=lambda stage:None
        framework=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,synchronize=lambda device:None,memory_allocated=lambda device:0))
        # Isolate the source child's metadata control flow. No source registry,
        # OS handle or framework is created; only these local spies are used.
        with (patch.object(a,'require_execution_permit',return_value=None),
              patch.object(w,'verify_worker_handle',return_value=child.proof()),
              patch.object(c.os,'getpid',return_value=222),patch.dict(sys.modules,{'torch':framework})):
            delegated=c.DelegatedGpuLease(context,token,handle,observer,gate)
            before=self.module.LOCK.read_bytes()
            with delegated.enter_child(context,token,child.proof(),{'branch':'uninterrupted'}):
                self.assertTrue(delegated.entered)
            self.assertFalse(delegated.entered); self.assertEqual(self.module.LOCK.read_bytes(),before)
            self.assertEqual(self.module.acquisitions,1); self.assertEqual(self.module.releases,0)
            owner=json.loads(before); owner['refreshed']=99
            self.module.LOCK.write_text(json.dumps(owner))
            with self.assertRaisesRegex(ValueError,'current original supervisor'):
                with delegated.enter_child(context,token,child.proof(),{'branch':'uninterrupted'}): self.fail('changed raw lock was accepted')
            self.module.LOCK.write_bytes(before)
            context['checkpoint_child_observed_lineage']['parent_proof']=proof(999,'e'*64)
            with self.assertRaisesRegex(ValueError,'observed parent/child'):
                with delegated.enter_child(context,token,child.proof(),{'branch':'uninterrupted'}): self.fail('foreign lineage was accepted')
        self.assertFalse(a._REGISTERED_NATIVE_AUTHORITY_VERIFIERS); self.assertNotIn('torch',sys.modules)
        child.signaled=True; self.lease.finish_child(child,child.terminal(1)); self.lease.close()

    def test_fixture_promotion_and_nested_self_wait_refuse(self):
        self.lease.acquire()
        with self.assertRaisesRegex(ValueError,'already-held'): self.lease.acquire()
        self.assertEqual(self.module.acquisitions,1)
        self.context['approved']=True
        with self.assertRaisesRegex(ValueError,'promote native'): self.lease.assert_parent_use()
        with self.assertRaisesRegex(ValueError,'promote native'): self.lease.close()
        self.assertEqual(self.module.releases,1)


if __name__=='__main__': unittest.main()
