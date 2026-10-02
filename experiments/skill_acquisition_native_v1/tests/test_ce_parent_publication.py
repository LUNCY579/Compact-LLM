"""Actual small-file parent body retention and lifecycle controls; no ML/OS."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_parent_publication as p
from .test_ce_qualification_accounting import fixture


class Tests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.context,self.proof,self.prepared=fixture(self.root)
        self.context['native_services']={}
        self.ledger=p.accounting.QualificationAccounting(self.context,None,self.proof,self.prepared)
        self.artifacts={'ingestion_tokenizer':self.prepared}
        for name in p.qualification.STAGES[1:]:
            path=self.root/(name+'.json');content=b'{"fixture":true}\n';path.write_bytes(content)
            ref={'path':str(path),'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
            self.ledger.begin_stage(name,'pure/'+name);self.ledger.finish_stage([ref])
            self.artifacts[name]={'raw_artifacts':[ref]}
        bounds={key:10**12 for key in p.qualification.COST_FIELDS};bounds['wall_seconds_upper']=1000
        self.context['bindings']['scope']['publication']={'schema':p.publication.SPEC,'owned_directory':str(self.root/'published'),
            'max_receipt_bytes':p.raw.MAX_JSON,'bounds':bounds,'reader_bounds':{key:10**12 for key in p.raw.BUDGET_FIELDS}}
    def publisher(self):return p.ParentBodyPublisher(self.context,None,self.proof,self.prepared)
    def link(self,source,target,owner):os.link(source,target);return {'method':'explicit pure exclusive link'}

    def test_native_roles_and_publisher_refuse_before_work(self):
        class Trap:
            def __getattribute__(self,name):raise AssertionError('before native permission')
        for invoke in (lambda:p.publication_role(Trap(),Trap(),Trap(),Trap(),Trap()),
                       lambda:p.interruption_role(Trap(),Trap(),Trap(),Trap(),Trap()),
                       lambda:p.completion_seal_role(Trap(),Trap(),Trap(),Trap(),Trap()),
                       lambda:p.ParentBodyPublisher(Trap(),Trap(),Trap(),Trap())):
            with self.assertRaises(ValueError):invoke()
        self.assertIsNone(p._PUBLICATION_ISSUER)

    def test_early_failure_retained_by_actual_service_bootstrap_without_model_permission(self):
        from .test_ce_native_services import bootstrap_spec
        spec=bootstrap_spec(self.root/'service-bootstrap')
        spec['operation_profiles']['interruption']=copy.deepcopy(spec['operation_profiles']['fixture'])
        spec['operation_profiles']['interruption'].update(read_bytes_upper=4*p.raw.MAX_JSON,write_bytes_upper=p.raw.MAX_JSON,
            identity_hash_attempts_upper=4,identity_hash_bytes_upper=4*p.raw.MAX_JSON)
        spec['caps'].update(read_bytes=10**9,write_bytes=10**9,hash_bytes=10**9,metadata_bytes=10**9)
        bootstrap=p.services.BootstrapLedger(spec,'a'*64,synthetic_fixture=True)
        service=object.__new__(p.services.NativeControllerServices);service.context=self.context;service.bootstrap=bootstrap
        service.operation=p.services.StableDispatcher(bootstrap);service.operation.service_owner=service;service._admit=lambda:None
        error={'type':'RuntimeError','message':'framework/preparation failed before Q'}
        with patch.object(p.authority,'require_execution_permit',side_effect=AssertionError('early interruption needs no model permit')):
            result=p.interruption_role(self.context,error,None,bootstrap.snapshot(),service.operation)
        content=Path(result['interruption_ref']['path']).read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(),result['interruption_ref']['sha256'])
        self.assertFalse(result['body_completed']);self.assertFalse(result['charges_refunded']);self.assertIsNone(result['q_snapshot'])
        self.assertEqual(list(bootstrap.operations.values())[-1]['phase'],'returned')

    def test_ended_body_retains_original_files_and_never_claims_terminal(self):
        self.ledger.end();original=self.ledger.path.read_bytes();publisher=self.publisher()
        with patch.object(p.publication,'_atomic_no_replace',self.link):
            result=publisher.retain(self.context,self.artifacts,self.ledger,{'fixture':True})
        self.assertEqual(self.ledger.path.read_bytes(),original)
        self.assertEqual(result['status'],'UNQUALIFIED_BODY_COMPLETE');self.assertFalse(result['os_terminal_verified'])
        self.assertFalse(result['native_qualified']);self.assertFalse(result['execution_verified'])
        for reference in (*result['ledger_refs'].values(),result['prepared_receipt_ref'],result['context_ref'],result['body_ref']):
            content=Path(reference['path']).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(),reference['sha256']);self.assertEqual(len(content),reference['bytes'])
        self.assertEqual(publisher.handle.evidence['status'],'RETURNED')

    def test_live_clock_or_changed_stage_refs_fail_and_keep_full_ticket(self):
        publisher=self.publisher();charged=publisher.handle.evidence['bounds']
        with self.assertRaisesRegex(ValueError,'ended resolved'):publisher.retain(self.context,self.artifacts,self.ledger,{})
        self.assertEqual(publisher.handle.evidence['status'],'FAILED');self.assertEqual(publisher.handle.evidence['bounds'],charged)
        self.assertFalse((publisher.directory/'PARENT_BODY_COMPLETE.json').exists())

    def test_short_publication_write_is_retained_failure(self):
        self.ledger.end();publisher=self.publisher()
        original=Path.open
        class Short:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def write(self,raw):return len(raw)-1
        def opened(path,mode='r',*args,**kwargs):
            if mode=='xb':return Short()
            return original(path,mode,*args,**kwargs)
        with patch.object(Path,'open',opened):
            with self.assertRaisesRegex(ValueError,'short parent body'):publisher.retain(self.context,self.artifacts,self.ledger,{})
        self.assertEqual(publisher.handle.evidence['status'],'FAILED');self.assertGreater(publisher.counts['io_write_bytes_upper'],0)

    def test_body_role_ends_q_only_after_actual_release_and_then_reserves_publication(self):
        service=SimpleNamespace(lease_token=None,supervisor_lease=SimpleNamespace(closed=True,child=None),bound_backend=object.__new__(p.backend.NativeCEBackend))
        core=service.bound_backend;core.closed=True;core.model=core.optimizer=core.tokenizer=None
        events=[];real_end=self.ledger.end
        def end():events.append('q_end');return real_end()
        self.ledger.end=end
        publisher=self.publisher()
        def construct(*args):events.append('reserve_publication');return publisher
        with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_owner',return_value=service),
              patch.object(p,'_publisher',side_effect=construct),patch.object(p.publication,'_atomic_no_replace',self.link)):
            result=p.publication_role(self.context,self.artifacts,self.ledger,{},object())
        self.assertEqual(events,['q_end','reserve_publication']);self.assertFalse(self.ledger.worker_open)
        self.assertIs(self.context['native_services']['parent_body_publication'],result)

    def test_live_model_lease_or_child_blocks_q_end_before_publication_io(self):
        core=object.__new__(p.backend.NativeCEBackend);core.closed=True;core.model=core.optimizer=core.tokenizer=None
        service=SimpleNamespace(lease_token=None,supervisor_lease=SimpleNamespace(closed=True,child=None),bound_backend=core)
        for change in ('model','lease','child'):
            value=copy.copy(service);value.supervisor_lease=copy.copy(service.supervisor_lease)
            if change=='model':core.model=object()
            elif change=='lease':value.lease_token='still-owned'
            else:value.supervisor_lease.child=object()
            with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_owner',return_value=value),
                  patch.object(p,'_publisher',side_effect=AssertionError('publication before physical release'))):
                with self.assertRaises(ValueError):p.publication_role(self.context,self.artifacts,self.ledger,{},object())
            core.model=None
        self.assertTrue(self.ledger.worker_open)

    def test_last_readback_ticket_completes_before_original_bootstrap_end_and_inspection(self):
        from .test_ce_native_services import bootstrap_spec
        self.ledger.end();first=self.publisher()
        with patch.object(p.publication,'_atomic_no_replace',self.link):
            result=first.retain(self.context,self.artifacts,self.ledger,{})
        outer=first.handle._ledger
        self.context['native_services']['qualification_bootstrap']=outer
        self.context['native_services']['parent_body_publication']=result
        pubspec=self.context['bindings']['scope']['publication']
        owner={'accounting_spec_sha256':p.digest(pubspec),'prepared_receipt_sha256':self.prepared['receipt_sha256']}
        ticket=outer.reserve_setup(owner,pubspec['bounds'])
        last=p.ParentBodyPublisher(self.context,None,self.proof,self.prepared,bootstrap_handle=ticket)
        boot_spec=bootstrap_spec(self.root/'source-service')
        boot_spec['operation_profiles']['publication']=copy.deepcopy(boot_spec['operation_profiles']['fixture'])
        boot_spec['operation_profiles']['publication'].update(read_bytes_upper=4*p.raw.MAX_JSON,write_bytes_upper=p.raw.MAX_JSON,
            identity_hash_attempts_upper=4,identity_hash_bytes_upper=4*p.raw.MAX_JSON)
        boot_spec['caps'].update(read_bytes=10**9,write_bytes=10**9,hash_bytes=10**9,metadata_bytes=10**9)
        original=p.services.BootstrapLedger(boot_spec,'a'*64,synthetic_fixture=True)
        core=object.__new__(p.backend.NativeCEBackend);core.closed=True;core.model=core.optimizer=core.tokenizer=None
        service=SimpleNamespace(lease_token=None,supervisor_lease=SimpleNamespace(closed=True,child=None),bound_backend=core,bootstrap=original)
        with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_owner',return_value=service),
              patch.object(p,'_publisher',return_value=last)):
            seal=p.completion_seal_role(self.context,result,self.ledger,original,object())
        self.assertEqual(ticket.evidence['status'],'RETURNED');self.assertTrue(outer.closed)
        ref=seal['scope_bootstrap_evidence_ref'];raw=Path(ref['path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256'])
        capsule=p.raw._json(raw);self.assertTrue(capsule['snapshot']['clock_ended'])
        for value in (capsule['header_ref'],capsule['journal_ref']):
            content=Path(value['path']).read_bytes();self.assertEqual(hashlib.sha256(content).hexdigest(),value['sha256'])
        self.assertFalse(seal['os_terminal_verified']);self.assertFalse(seal['native_qualified'])
        with self.assertRaises(ValueError):outer.inspect_evidence()


if __name__=='__main__':unittest.main()
