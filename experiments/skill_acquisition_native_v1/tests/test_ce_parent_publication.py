"""Actual small-file parent body retention and lifecycle controls; no ML/OS."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from contextlib import ExitStack
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

    def terminal_lifetime(self):
        from .test_ce_native_services import bootstrap_spec
        self.ledger.end();first=self.publisher()
        with patch.object(p.publication,'_atomic_no_replace',self.link):
            result=first.retain(self.context,self.artifacts,self.ledger,{})
        outer=first.handle._ledger;pubspec=self.context['bindings']['scope']['publication']
        self.context['native_services'].update(qualification_bootstrap=outer,parent_body_publication=result)
        owner={'accounting_spec_sha256':p.digest(pubspec),'prepared_receipt_sha256':self.prepared['receipt_sha256']}
        process_ticket=outer.reserve_setup({'accounting_spec_sha256':'f'*64,'prepared_receipt_sha256':self.prepared['receipt_sha256']},pubspec['bounds'])
        process_ticket.begin()
        process=object.__new__(p.processes.CheckpointProcesses);process.context=self.context;process.accounting=self.ledger
        process.handle=process_ticket
        process.final_handle_lifetime_closed=False;process.terminal_handle_records=dict.fromkeys(p.checkpoint_reduction.BRANCHES,{})
        self.context['native_services']['checkpoint_terminal_owner']=process
        last=p.ParentBodyPublisher(self.context,None,self.proof,self.prepared,bootstrap_handle=outer.reserve_setup(owner,pubspec['bounds']))
        spec=bootstrap_spec(self.root/'service-final');profile=copy.deepcopy(spec['operation_profiles']['fixture'])
        profile.update(read_bytes_upper=16*p.raw.MAX_JSON,write_bytes_upper=4*p.raw.MAX_JSON,
                       identity_hash_attempts_upper=32,identity_hash_bytes_upper=16*p.raw.MAX_JSON)
        spec['operation_profiles']['publication']=profile
        spec['caps'].update(read_bytes=10**10,write_bytes=10**10,hash_bytes=10**10,metadata_bytes=10**10)
        original=p.services.BootstrapLedger(spec,'a'*64,synthetic_fixture=True)
        core=object.__new__(p.backend.NativeCEBackend);core.closed=True;core.model=core.optimizer=core.tokenizer=None
        service=SimpleNamespace(lease_token=None,supervisor_lease=SimpleNamespace(closed=True,child=None),bound_backend=core,bootstrap=original)
        return result,outer,process_ticket,process,last,service

    def test_child_raw_os_replay_and_closure_precede_ticket_completion_and_scope_end(self):
        result,outer,ticket,process,last,service=self.terminal_lifetime();events=[]
        def replay(*args):
            self.assertFalse(process.final_handle_lifetime_closed);self.assertEqual(ticket.evidence['status'],'STARTED')
            self.assertFalse(outer.closed);events.append('independent_child_replay')
            return {'status':'UNQUALIFIED_CHILD_EVIDENCE_REPLAY','native_qualified':False,'parent_os_terminal_verified':False}
        def close(primary=None):
            self.assertIsNone(primary);events.append('close_owned_clones');process.final_handle_lifetime_closed=True
            process.terminal_handle_records.clear();ticket.complete()
        process.close_terminal_handles=close;original_end=outer.end
        def end():
            self.assertTrue(process.final_handle_lifetime_closed);self.assertEqual(ticket.evidence['status'],'RETURNED')
            events.append('scope_end');return original_end()
        outer.end=end
        with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_owner',return_value=service),
              patch.object(p,'_publisher',return_value=last),patch.object(p,'_replay_retained_checkpoint_children',side_effect=replay),
              patch.object(p.publication,'_atomic_no_replace',self.link)):
            seal=p.completion_seal_role(self.context,result,self.ledger,service.bootstrap,object())
        self.assertEqual(events,['independent_child_replay','close_owned_clones','scope_end'])
        ref=seal['checkpoint_child_terminal_replay_ref'];blob=Path(ref['path']).read_bytes()
        self.assertEqual(hashlib.sha256(blob).hexdigest(),ref['sha256'])
        self.assertFalse(seal['native_qualified']);self.assertFalse(seal['os_terminal_verified'])

    def test_child_replay_failure_closes_handles_keeps_primary_and_full_charges_without_scope_end(self):
        result,outer,ticket,process,last,service=self.terminal_lifetime()
        primary=ValueError('independent child raw mismatch');closed=[];charged=copy.deepcopy(ticket.evidence['bounds'])
        def close(primary=None):
            closed.append(primary);process.final_handle_lifetime_closed=True;process.terminal_handle_records.clear()
            ticket.fail(primary);primary.checkpoint_terminal_handle_cleanup=[{'type':'OSError','message':'retained close receipt failed'}]
            raise primary
        process.close_terminal_handles=close
        with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_owner',return_value=service),
              patch.object(p,'_publisher',return_value=last),patch.object(p,'_replay_retained_checkpoint_children',side_effect=primary)):
            with self.assertRaisesRegex(ValueError,'child raw mismatch') as caught:
                p.completion_seal_role(self.context,result,self.ledger,service.bootstrap,object())
        self.assertIs(caught.exception,primary);self.assertEqual(closed,[primary]);self.assertFalse(outer.closed)
        self.assertEqual(ticket.evidence['status'],'FAILED');self.assertEqual(ticket.evidence['bounds'],charged)
        self.assertEqual(last.handle.evidence['status'],'FAILED')
        self.assertFalse((last.directory/'PARENT_CHILD_TERMINAL_REPLAY.json').exists())

    def test_missing_native_terminal_owner_and_dictionary_handles_refuse_before_raw_io(self):
        context={'native_services':{},'mode':'registered_native'}
        with self.assertRaisesRegex(ValueError,'terminal owner'):p._terminal_owner(context,self.ledger)
        owner=object.__new__(p.processes.CheckpointProcesses);owner.context=self.context;owner.accounting=self.ledger
        owner.terminal_handle_records={branch:{'binding':{'worker_identity_sha256':str(index)*64}}
            for index,branch in enumerate(p.checkpoint_reduction.BRANCHES,1)}
        owner.retained_terminal_handles=lambda:{str(index)*64:{} for index in range(1,4)}
        with patch.object(p.execution,'_read_json',side_effect=AssertionError('forged OS handle before raw IO')):
            with self.assertRaisesRegex(ValueError,'actual retained terminal handles'):
                p._replay_retained_checkpoint_children(SimpleNamespace(),self.context,self.ledger,{},owner)

    def test_final_publisher_construction_failure_closes_owned_clones_and_preserves_cause(self):
        result,outer,ticket,process,last,service=self.terminal_lifetime()
        failure=RuntimeError('final publication reservation denied');closed=[]
        def close(primary=None):
            closed.append(primary);process.final_handle_lifetime_closed=True;process.terminal_handle_records.clear();ticket.fail(primary)
        process.close_terminal_handles=close
        with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_owner',return_value=service),
              patch.object(p,'_publisher',side_effect=failure)):
            with self.assertRaisesRegex(RuntimeError,'reservation denied') as caught:
                p.completion_seal_role(self.context,result,self.ledger,service.bootstrap,object())
        self.assertIs(caught.exception,failure);self.assertEqual(closed,[failure]);self.assertEqual(ticket.evidence['status'],'FAILED')
        self.assertFalse(outer.closed)

    def test_delta_publication_clock_checks_each_callback_and_clears_only_exact_scope(self):
        publisher=self.publisher();publisher.spec['delta_rebuild_wall_seconds_upper']=5
        publisher.last_clock=100;publisher._delta_rebuild_started=100;publisher._delta_rebuild_operation=None
        with patch.object(p.time,'monotonic',return_value=100):
            publisher.operation('qualification_reducer_overhead','started','delta-scope',reason='delta_readback_source_report_linkage')
        with patch.object(p.time,'monotonic',return_value=104):
            publisher.operation('qualification_artifact_readback','started','source-tensor',io_read_bytes_upper=17)
        charged=copy.deepcopy(publisher.counts)
        with patch.object(p.time,'monotonic',return_value=106):
            with self.assertRaisesRegex(ValueError,'delta rebuild wall ceiling'):
                publisher.operation('qualification_artifact_readback','started','late-tensor',io_read_bytes_upper=17)
            self.assertEqual(publisher.counts,charged)
            with self.assertRaisesRegex(ValueError,'delta rebuild wall ceiling'):
                publisher.operation('qualification_reducer_overhead','failed','delta-scope',reason='delta_readback_source_report_linkage')
        self.assertIsNone(publisher._delta_rebuild_started)
        self.assertIsNone(publisher._delta_rebuild_operation)

    def test_successful_delta_window_does_not_include_later_resource_or_publication_work(self):
        publisher=self.publisher();publisher.spec['delta_rebuild_wall_seconds_upper']=5
        publisher.last_clock=100;publisher._delta_rebuild_started=100;publisher._delta_rebuild_operation=None
        with patch.object(p.time,'monotonic',return_value=100):
            publisher.operation('qualification_reducer_overhead','started','delta-scope',reason='delta_readback_source_report_linkage')
        with patch.object(p.time,'monotonic',return_value=104):
            publisher.operation('qualification_reducer_overhead','returned','delta-scope',reason='delta_readback_source_report_linkage')
        self.assertIsNone(publisher._delta_rebuild_started)
        with patch.object(p.time,'monotonic',return_value=110):
            publisher.operation('qualification_reducer_overhead','started','later-resource',reason='resource_measurement')
        self.assertEqual(publisher.counts['wall_seconds_upper'],10)

    def test_final_replay_calls_complete_reductions_with_actual_attempt_and_raw_child_maps(self):
        publisher=self.publisher();publisher.spec['delta_rebuild_wall_seconds_upper']=5
        owner=object.__new__(p.processes.CheckpointProcesses);owner.context=self.context;owner.accounting=self.ledger
        handles,records={},{}
        branch_refs,branch_bindings,terminal_refs,terminal_proofs={}, {}, {}, {}
        binding=p.execution._binding_for(self.context,{'stage':'checkpoint_resume','attempt_id':'original-cp'},self.proof['worker_identity_sha256'])
        for index,branch in enumerate(p.checkpoint_reduction.BRANCHES,1):
            child=dict(binding,worker_identity_sha256=str(index+4)*64,attempt_id=branch)
            handle=object.__new__(p.worker.WinWorkerHandle);handles[child['worker_identity_sha256']]=handle
            terminal={'worker_identity_sha256':child['worker_identity_sha256'],'request_sha256':self.context['request_sha256'],
                'scope_sha256':self.context['scope_sha256'],'exit_code':0,'status_file_used':False}
            ref={'path':str(self.root/(branch+'.json')),'sha256':'f'*64,'bytes':10}
            branch_refs[branch]=ref;branch_bindings[branch]=child;terminal_refs[branch]=dict(ref,path=str(self.root/(branch+'-terminal.json')))
            terminal_proofs[branch]=terminal
            records[branch]={'handle':handle,'binding':child,'proof':{'view':branch},'terminal':terminal,'raw_ref':terminal_refs[branch],
                'terminal_observed_monotonic':100.0+index}
        owner.terminal_handle_records=records;owner.retained_terminal_handles=lambda:handles
        attempt={'stage':'checkpoint_resume','attempt_id':'original-cp','status':'RETURNED','evidence':{'raw_refs':[{'path':'bundle','bytes':10}]}}
        state=SimpleNamespace(attempts=[attempt]);attempt_workers={'original-cp':self.proof['worker_identity_sha256']}
        payload={'branch_refs':branch_refs,'branch_bindings':branch_bindings,'child_terminal_refs':terminal_refs,'child_terminal_proofs':terminal_proofs}
        envelope={'schema':p.raw.SCHEMA,**binding,'evidence_kind':'checkpoint_branches','payload':payload}
        branches={branch:{'binding':branch_bindings[branch],'lineage':{'child_proof':records[branch]['proof']},'derived_costs':{}}
            for branch in p.checkpoint_reduction.BRANCHES}
        reduced={'raw_branches':{'branches':branches}}
        all_reductions={stage:{} for stage in p.qualification.STAGES[1:]};all_reductions['checkpoint_resume']=reduced
        result={'ledger_refs':{key:{'path':key,'bytes':10} for key in ('header','journal','seal')},
                'stage_artifacts':{'checkpoint_resume':attempt['evidence']['raw_refs']}}
        def read(reader,ref,expected,role):
            if role=='checkpoint_branches':return envelope
            if role=='checkpoint_retained_terminal':
                branch=next(name for name in records if records[name]['raw_ref']==ref)
                return {'schema':'CE8-CHECKPOINT-RETAINED-TERMINAL-1','branch':branch,'binding':records[branch]['binding'],
                    'worker_proof':records[branch]['proof'],'terminal_proof':records[branch]['terminal'],
                    'terminal_observed_monotonic':records[branch]['terminal_observed_monotonic'],
                    'native_qualified':False,'execution_verified':False}
            return {}
        def raw_child(reader,ref,child):
            branch=next(name for name in branch_refs if branch_refs[name]==ref)
            return {'branch':branch,'observed_lineage':{'child_proof':records[branch]['proof']}}
        def reductions(ctx,actual_state,actual_workers,reader,child_map,lineage_map,outer_preflight=None):
            self.assertIs(actual_state,state);self.assertIs(actual_workers,attempt_workers)
            self.assertEqual(child_map,{branch_refs[name]['path']:branch_bindings[name] for name in branch_refs})
            self.assertEqual(set(lineage_map),set(handles));self.assertEqual(len(owner.terminal_handle_records),3)
            outer_preflight('checkpoint_branch_reduction',{'reader_bounds':reader.remaining,'references':list(branch_refs.values())})
            return all_reductions,['native activation remains separate']
        with ExitStack() as stack:
            stack.enter_context(patch.object(p.execution,'_read_json',side_effect=read))
            stack.enter_context(patch.object(p.execution,'_header',return_value={}))
            stack.enter_context(patch.object(p.execution,'_replay',return_value=(state,{},attempt_workers)))
            stack.enter_context(patch.object(p.execution,'_seal',return_value=(True,[])))
            stack.enter_context(patch.object(p.checkpoint_reduction,'_envelope',side_effect=raw_child))
            complete=stack.enter_context(patch.object(p.execution,'_reductions',side_effect=reductions))
            stack.enter_context(patch.object(p.execution,'_checkpoint_readback',side_effect=AssertionError('checkpoint-only path bypasses saved-delta proof')))
            stack.enter_context(patch.object(p.child_issuer,'validate_parent_view',return_value=None))
            stack.enter_context(patch.object(p.worker,'verify_worker_handle',side_effect=lambda handle,expected,**kwargs:next(
                record['terminal'] for record in records.values() if record['handle'] is handle)))
            report=p._replay_retained_checkpoint_children(publisher,self.context,self.ledger,result,owner)
        self.assertEqual(complete.call_count,1);self.assertEqual(report['raw_numeric_reductions'],all_reductions)
        self.assertFalse(report['native_qualified']);self.assertFalse(report['execution_verified'])


if __name__=='__main__':unittest.main()
