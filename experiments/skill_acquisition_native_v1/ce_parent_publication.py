"""Live parent body publication; OS-terminal completion belongs to supervisor.

Q ends only after the model and held lease have physically closed. The actual
outer bootstrap prepays every publication/verification byte. Failed/pending
costs remain in retained original histories, and files are never replaced.
Nothing here supplies native PASS, production authority or a terminal process.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import time
import uuid

from . import ce_native_authority as authority, ce_parent_qualification as parent
from . import ce_qualification_accounting as accounting, ce_qualification as qualification
from . import ce_bootstrap_accounting as boot, ce_publication as publication
from . import ce_qualification_artifacts as raw, ce_execution_verifier as execution
from . import ce_native_backend as backend, ce_native_services as services
from . import ce_checkpoint_processes as processes, ce_checkpoint_reduction as checkpoint_reduction
from . import ce_checkpoint_local_issuer as child_issuer, ce_worker_evidence as worker

SCHEMA='CE8-PARENT-QUALIFICATION-BODY-1'
_PUBLICATION_ISSUER=None
require,canonical,digest=authority.require,authority.canonical,authority.digest


def _released(service):
    require(service.lease_token is None and service.supervisor_lease.closed
            and service.supervisor_lease.child is None, 'parent body publication requires physical owned lease/child quiescence')
    core=service.bound_backend
    require(type(core) is backend.NativeCEBackend and core.closed and core.model is core.optimizer is core.tokenizer is None,
            'actual parent model/Adam/tokenizer cleanup precedes body publication')


class ParentBodyPublisher(publication.CompletionPublisher):
    """Reuse priced helpers; deliberately never calls live-worker publish()."""
    def tick(self):
        start=getattr(self,'_delta_rebuild_started',None)
        if start is not None:
            require(0<=time.monotonic()-start<=self.spec['delta_rebuild_wall_seconds_upper'],
                    'independently prepaid parent delta rebuild wall ceiling exceeded')
        return super().tick()

    def operation(self,kind,phase,operation_id,**fields):
        delta_scope=(kind=='qualification_reducer_overhead'
                     and fields.get('reason')=='delta_readback_source_report_linkage')
        if delta_scope and phase=='started':
            require(getattr(self,'_delta_rebuild_started',None) is not None
                    and getattr(self,'_delta_rebuild_operation',None) is None,'source-prepaid delta rebuild scope before its actual reader')
            self._delta_rebuild_operation=operation_id
        closing=delta_scope and phase in ('returned','failed')
        if closing:require(getattr(self,'_delta_rebuild_operation',None)==operation_id,'same source delta reader operation closes its wall window')
        try:return super().operation(kind,phase,operation_id,**fields)
        finally:
            if closing:self._delta_rebuild_started=self._delta_rebuild_operation=None
    def _reference(self,path,role,maximum):
        oid=uuid.uuid4().hex
        self.operation('parent_reference','started',oid,read_bytes_upper=maximum+1,
            identity_hash_attempts_upper=2,identity_hash_bytes_upper=maximum+1+raw.MAX_BINDING_BYTES)
        path=Path(path)
        require(path.resolve().is_relative_to(self.root) and path.is_file() and not path.is_symlink()
                and 0<path.stat().st_size<=maximum, 'bounded owned parent evidence reference')
        size=path.stat().st_size; hashed=hashlib.sha256(); read=0
        with path.open('rb') as stream:
            while block:=stream.read(min(raw.CHUNK,size+1-read)):
                read+=len(block);require(read<=size,'parent evidence grew');hashed.update(block)
        require(read==size==path.stat().st_size,'parent evidence changed during hash')
        self.tick()
        return {'path':str(path.resolve()),'sha256':hashed.hexdigest(),'bytes':size,'role':role,'binding_sha256':digest(self._binding())}

    def _write(self,name,value,role):
        oid=uuid.uuid4().hex; maximum=self.spec['max_receipt_bytes']
        self.operation('parent_body_write','started',oid,write_bytes_upper=maximum,read_bytes_upper=2*maximum+2,
            identity_hash_attempts_upper=4,identity_hash_bytes_upper=3*maximum+raw.MAX_BINDING_BYTES,metadata_bytes_upper=maximum)
        content=canonical(value)+b'\n';require(len(content)<=maximum,'parent body JSON ceiling')
        final=self.directory/name; temporary=self.directory/(name+'.'+uuid.uuid4().hex+'.partial')
        require(final.parent==temporary.parent==self.directory and not final.is_symlink() and not os.path.lexists(final),
                'create-only exact parent body path')
        self.directory.mkdir(parents=True,exist_ok=True)
        with temporary.open('xb') as stream:
            require(stream.write(content)==len(content),'short parent body publication');stream.flush();os.fsync(stream.fileno())
        with temporary.open('rb') as stream:observed=stream.read(len(content)+1)
        require(observed==content,'temporary parent body readback')
        publication._atomic_no_replace(temporary,final,self.directory)
        with final.open('r+b') as stream:os.fsync(stream.fileno());observed=stream.read(len(content)+1)
        require(observed==content,'final parent body readback')
        self.tick()
        return {'path':str(final),'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),
                'role':role,'binding_sha256':digest(self._binding())}

    def retain(self,context,artifacts,ledger,service_snapshot):
        try:
            require(context is self.context and type(ledger) is accounting.QualificationAccounting
                    and ledger.context is context and ledger._current_proof==self.proof, 'exact original parent Q/worker before reads')
            snapshot=ledger.snapshot()
            require(not snapshot['worker_open'] and not ledger.active and not snapshot['pending_operations']
                    and not snapshot['fatal'] and not snapshot['unknown_wall']
                    and not ledger.requires_restore and not ledger.restore_pending, 'ended resolved parent body accounting required')
            require(type(artifacts) is dict and set(artifacts)==set(qualification.STAGES), 'all eight parent body stages required')
            stage_refs={}
            for name in qualification.STAGES:
                if name=='ingestion_tokenizer':
                    require(artifacts[name]==self.prepared,'actual prepared receipt differs at parent publication')
                    continue
                returned=[attempt for attempt in ledger.attempts if attempt['stage']==name and attempt['status']=='RETURNED']
                require(len(returned)==1 and artifacts[name].get('raw_artifacts')==returned[0]['evidence']['raw_refs'],
                        'parent stage refs differ from actual original Q evidence')
                stage_refs[name]=copy.deepcopy(returned[0]['evidence']['raw_refs'])
                for ref in stage_refs[name]: self._stream_ref(ref,'parent_raw_'+name)
            header=self._reference(ledger.header_path,'qualification_header',execution.MAX_HEADER)
            journal=self._reference(ledger.path,'qualification_journal',execution.MAX_LEDGER)
            seal=self._write('PARENT_Q_SNAPSHOT.json',snapshot,'qualification_seal')
            prepared=self._write('PARENT_PREPARED_RECEIPT.json',self.prepared,'prepared_receipt')
            stage_refs['ingestion_tokenizer']=[prepared]
            # Scope/prepared references are separate files so body JSON remains
            # bounded even with a full 3000-row census and complete source map.
            raw_context=dict(authority.context_preimage(context)); raw_context.pop('prepared_receipt',None)
            context_ref=self._write('PARENT_CONTEXT.json',raw_context,'parent_context')
            payload={'schema':SCHEMA,'status':'UNQUALIFIED_BODY_COMPLETE','body_completed':True,
                'os_terminal_verified':False,'native_qualified':False,'execution_verified':False,
                'request_sha256':context['request_sha256'],'scope_sha256':context['scope_sha256'],
                'static_recipe_sha256':context['static_recipe_sha256'],'worker_proof':self.proof,
                'context_ref':context_ref,'prepared_receipt_ref':prepared,'stage_artifacts':stage_refs,
                'ledger_refs':{'header':header,'journal':journal,'seal':seal},
                'source_service_bootstrap_snapshot':service_snapshot,'publication_prepayment':self.handle.evidence,
                'claim':'retained ended body evidence; supervisor must separately prove actual OS exit and run raw/source/cost verification'}
            terminal_owner=_terminal_owner(context,ledger)
            if terminal_owner is not None:
                payload['checkpoint_terminal_evidence_refs']={branch:copy.deepcopy(record['raw_ref'])
                    for branch,record in terminal_owner.terminal_handle_records.items()}
            result=self._write('PARENT_BODY_COMPLETE.json',payload,'parent_body_complete')
            self.tick();self.handle.complete();self.finished=True
            return {'schema':SCHEMA,'status':'UNQUALIFIED_BODY_COMPLETE','body_completed':True,'os_terminal_verified':False,
                'native_qualified':False,'execution_verified':False,'body_ref':result,'ledger_refs':payload['ledger_refs'],
                'prepared_receipt_ref':prepared,'context_ref':context_ref,'stage_artifacts':stage_refs,
                'charged_bounds':copy.deepcopy(self.counts),'publication_prepayment':self.handle.evidence}
        except BaseException as primary:
            self.finished=True
            try:self.handle.fail(primary)
            except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
            raise


def _owner(context,operation):
    service=parent._service_owner(context)
    require(operation is service.operation, 'parent body callback identity differs from frozen source service')
    return service


def _terminal_owner(context,ledger):
    owner=context['native_services'].get('checkpoint_terminal_owner')
    if owner is None and context.get('mode')=='synthetic_fixture':return None
    require(type(owner) is processes.CheckpointProcesses and owner.context is context and owner.accounting is ledger
            and not owner.final_handle_lifetime_closed and set(owner.terminal_handle_records)==set(checkpoint_reduction.BRANCHES),
            'actual all-three checkpoint terminal owner required before final publication')
    return owner


def _close_terminals_on_failure(owner,primary):
    if owner is not None and not owner.final_handle_lifetime_closed:
        try:owner.close_terminal_handles(primary=primary)
        except BaseException as failure:
            if failure is not primary:raw._accounting(primary,failure)


def _replay_retained_checkpoint_children(publisher,context,ledger,result,owner):
    """Independent paid raw/state replay while original child OS clones are live."""
    require(type(owner) is processes.CheckpointProcesses and owner.context is context and owner.accounting is ledger,
            'same original checkpoint process owner before terminal replay')
    handles=owner.retained_terminal_handles()
    records=owner.terminal_handle_records
    require(set(records)==set(checkpoint_reduction.BRANCHES)
            and set(handles)=={record['binding']['worker_identity_sha256'] for record in records.values()}
            and all(type(handle) is worker.WinWorkerHandle for handle in handles.values()),
            'all three distinct actual retained terminal handles, never dictionaries')
    with publisher.reader._overhead('parent_child_replay_setup',work_bytes=128*execution.MAX_HEADER,
                                   hash_attempts=128,hash_bytes=128*execution.MAX_HEADER):
        replay_context=dict(context,prepared_receipt=publisher.prepared)
        verification_binding=publisher._binding();refs=result['ledger_refs']
        header=execution._header(replay_context,execution._read_json(publisher.reader,refs['header'],verification_binding,'qualification_header'))
    state,_,attempt_workers=execution._replay(publisher.reader,refs['journal'],verification_binding,replay_context,header,
                              {publisher.proof['worker_identity_sha256']})
    observed_seal=execution._read_json(publisher.reader,refs['seal'],verification_binding,'qualification_seal')
    with publisher.reader._overhead('parent_checkpoint_seal_and_linkage',
            work_bytes=32*refs['seal']['bytes']+32*refs['journal']['bytes'],
            hash_attempts=execution.MAX_EVENTS,hash_bytes=8*refs['journal']['bytes']):
        complete,pending=execution._seal(observed_seal,state)
    require(complete and not pending,'original ended parent checkpoint replay before releasing child OS handles')
    attempts=[attempt for attempt in state.attempts if attempt['stage']=='checkpoint_resume' and attempt['status']=='RETURNED']
    require(len(attempts)==1 and len(attempts[0]['evidence']['raw_refs'])==1,'one exact parent checkpoint bundle')
    attempt=attempts[0];binding=execution._binding_for(replay_context,attempt,publisher.proof['worker_identity_sha256'])
    require(result['stage_artifacts']['checkpoint_resume']==attempt['evidence']['raw_refs'],'published checkpoint refs differ from original parent ledger')
    envelope=execution._read_json(publisher.reader,attempt['evidence']['raw_refs'][0],binding,'checkpoint_branches')
    require(envelope.get('schema')==raw.SCHEMA and envelope.get('evidence_kind')=='checkpoint_branches'
            and all(envelope.get(key)==value for key,value in binding.items()),'original checkpoint bundle source/attempt/worker binding')
    def preflight(name,contract):
        if name=='checkpoint_branch_reduction':
            require(contract['reader_bounds']==publisher.reader.remaining
                    and contract['references']==list(payload['branch_refs'].values()),
                    'exact already-prepaid final child reducer allowance')
        elif name=='saved_delta_source_rebuild':
            scope=replay_context['bindings']['scope'];native=scope['native_backend']
            expected_roots={'base':native['model_dir'],'warm':str(Path(native['warm_delta_ref']['path']).parent),
                            'output':native['output_directory']}
            wall=publisher.spec.get('delta_rebuild_wall_seconds_upper')
            require(raw._finite(wall) and 0<wall<=publisher.spec['bounds']['wall_seconds_upper']
                    and contract['wall_seconds_upper']==wall
                    and contract['roots']==expected_roots
                    and contract['reader_bounds']==scope['qualifier_execution']['delta_rebuild_reader_bounds']
                    and getattr(publisher,'_delta_rebuild_started',None) is None,
                    'exact separately prepaid saved-delta source roots/readers/wall ceiling')
            publisher._delta_rebuild_started=time.monotonic();publisher._delta_rebuild_operation=None
        else:require(False,'unexpected final raw reducer preflight')
        publisher.tick()
    payload=envelope['payload']
    require(set(payload.get('branch_refs',{}))==set(checkpoint_reduction.BRANCHES)
            and set(payload.get('branch_bindings',{}))==set(checkpoint_reduction.BRANCHES),
            'all raw branch references/bindings before complete stage replay')
    child_bindings,child_lineages={},{}
    for branch in checkpoint_reduction.BRANCHES:
        reference=payload['branch_refs'][branch];child_binding=payload['branch_bindings'][branch]
        require(child_binding==records[branch]['binding'],'raw branch binding differs from actual retained child owner')
        child_payload=checkpoint_reduction._envelope(publisher.reader,reference,child_binding)
        require(child_payload['branch']==branch and type(child_payload.get('observed_lineage')) is dict,
                'exact raw child lineage before complete stage reductions')
        child_bindings[reference['path']]=copy.deepcopy(child_binding)
        child_lineages[child_binding['worker_identity_sha256']]=copy.deepcopy(child_payload['observed_lineage'])
    with publisher.reader._overhead('parent_checkpoint_raw_and_terminal_linkage',work_bytes=128*raw.MAX_JSON,
                                   hash_attempts=128,hash_bytes=128*raw.MAX_JSON):
        try:
            numeric_reductions,activation_gaps=execution._reductions(replay_context,state,attempt_workers,publisher.reader,
                child_bindings,child_lineages,outer_preflight=preflight)
            publisher.tick()
            require(set(numeric_reductions)==set(qualification.STAGES[1:]),'all original seven numeric stage reductions required before child closure')
            reduced=numeric_reductions['checkpoint_resume']
        finally:publisher._delta_rebuild_started=publisher._delta_rebuild_operation=None
    require(set(payload.get('child_terminal_refs',{}))==set(checkpoint_reduction.BRANCHES),'all three original raw terminal refs')
    terminal_results={}
    for branch in checkpoint_reduction.BRANCHES:
        record=records[branch];child=reduced['raw_branches']['branches'][branch]
        require(record['binding']==child['binding'] and payload['child_terminal_refs'][branch]==record['raw_ref'],
                'retained OS owner differs from independently replayed child branch')
        with publisher.reader._overhead('checkpoint_terminal_source_view',work_bytes=32*raw.MAX_JSON,
                                       hash_attempts=16,hash_bytes=32*raw.MAX_JSON):
            child_issuer.validate_parent_view(child['lineage']['child_proof'],record['proof'])
        observed=execution._read_json(publisher.reader,record['raw_ref'],binding,'checkpoint_retained_terminal')
        expected={'schema':'CE8-CHECKPOINT-RETAINED-TERMINAL-1','branch':branch,'binding':record['binding'],
            'worker_proof':record['proof'],'terminal_proof':record['terminal'],
            'terminal_observed_monotonic':record['terminal_observed_monotonic'],
            'native_qualified':False,'execution_verified':False}
        require(observed==expected and payload['child_terminal_proofs'][branch]==record['terminal'],
                'raw retained child terminal proof differs from exact original owner/bundle')
        handle=handles[record['binding']['worker_identity_sha256']]
        actual=worker.verify_worker_handle(handle,{key:record['terminal'][key] for key in
            ('worker_identity_sha256','request_sha256','scope_sha256')},require_native=True,terminal=True)
        require(actual==record['terminal'] and type(actual.get('exit_code')) is int and actual['exit_code']==0
                and actual.get('status_file_used') is False,'actual successful same-handle child terminal/source binding')
        terminal_results[branch]={'binding':copy.deepcopy(record['binding']),'terminal_ref':copy.deepcopy(record['raw_ref']),
            'terminal_proof':copy.deepcopy(actual),'child_derived_reservations':copy.deepcopy(child['derived_costs']),
            'terminal_observed_monotonic':record['terminal_observed_monotonic']}
    return {'schema':'CE8-PARENT-CHILD-TERMINAL-REPLAY-1','status':'UNQUALIFIED_CHILD_EVIDENCE_REPLAY',
        'parent_ledger_refs':copy.deepcopy(refs),'checkpoint_bundle_ref':copy.deepcopy(attempt['evidence']['raw_refs'][0]),
        'child_terminal_results':terminal_results,'child_os_terminal_verified':True,'parent_os_terminal_verified':False,
        'checkpoint_raw_reduction':reduced,'raw_numeric_reductions':numeric_reductions,'activation_gaps':activation_gaps,
        'native_qualified':False,'execution_verified':False,
        'claim':'same-handle child OS terminal and raw checkpoint evidence; live parent terminal and complete native qualification remain separate'}


def _publisher(context,service,ledger):
    boot.require_registered_verifier(parent._BOOTSTRAP_VERIFIER)
    require(any(_PUBLICATION_ISSUER is item for item in publication._REGISTERED_PUBLICATION_ISSUERS),
            'parent body publication issuer remains unregistered')
    outer=context['native_services'].get('qualification_bootstrap')
    require(type(outer) is boot.BootstrapAccounting and outer.context is context and not outer.closed,
            'actual original parent publication bootstrap owner required')
    prepared=context['native_services'].get('parent_prepared_receipt')
    require(type(prepared) is dict and prepared['receipt_sha256']==ledger.spec['ingestion_receipt_sha256'],
            'actual original prepared receipt must be retained before parent publication')
    spec=context['bindings']['scope']['publication']
    owner={'accounting_spec_sha256':digest(spec),'prepared_receipt_sha256':prepared['receipt_sha256']}
    require(context['bindings']['scope'].get('publication_owner')==owner,'exact frozen parent publication owner')
    handle=outer.reserve_setup(owner,spec['bounds'])
    return ParentBodyPublisher(context,context['native_execution_permit'],ledger._current_proof,prepared,
        bootstrap_handle=handle,bootstrap_verifier=parent._BOOTSTRAP_VERIFIER,native_issuer=_PUBLICATION_ISSUER)


def publication_role(context,artifacts,ledger,service_snapshot,operation):
    authority.require_execution_permit(context.get('native_execution_permit') if type(context) is dict else None,context,'completion_publication')
    service=_owner(context,operation);_released(service)
    require(type(ledger) is accounting.QualificationAccounting and ledger.context is context,
            'actual parent Q before closing the original worker clock')
    terminal_owner=_terminal_owner(context,ledger)
    try:
        if ledger.worker_open:ledger.end()
        publisher=_publisher(context,service,ledger)
        result=publisher.retain(context,artifacts,ledger,service_snapshot)
        context['native_services']['parent_body_publication']=result
        return result
    except BaseException as primary:
        _close_terminals_on_failure(terminal_owner,primary)
        raise


def interruption_role(context,error,ledger,service_snapshot,operation):
    # Early framework/preparation failure has no Q/model permission or scope
    # bootstrap yet. Its actual service authority already paid metadata work.
    require(type(operation) is services.StableDispatcher, 'actual original interruption dispatcher')
    service=getattr(operation,'service_owner',None)
    require(type(service) is services.NativeControllerServices and service.context is context
            and service.operation is operation, 'actual source metadata service owner before interruption IO')
    service._admit()
    terminal_owner=context['native_services'].get('checkpoint_terminal_owner')
    if terminal_owner is not None:
        require(type(terminal_owner) is processes.CheckpointProcesses and terminal_owner.context is context,
                'only original owned checkpoint terminal clones may close during interruption')
        interruption=RuntimeError('parent qualification interrupted')
        interruption.original_interruption=copy.deepcopy(error)
        _close_terminals_on_failure(terminal_owner,interruption)
        cleanup={key:copy.deepcopy(getattr(interruption,key)) for key in
            ('checkpoint_terminal_handle_cleanup','accounting_failure') if hasattr(interruption,key)}
        if cleanup:
            error=copy.deepcopy(error)
            error['checkpoint_terminal_cleanup']=cleanup
    outer=context['native_services'].get('qualification_bootstrap')
    if outer is None or ledger is None or getattr(outer,'closed',False) or getattr(outer,'fatal',False) or getattr(outer,'unknown_wall',False):
        return _early_interruption(service,context,error,ledger,service_snapshot)
    require(type(outer) is boot.BootstrapAccounting,'actual parent failure bootstrap required')
    # Existing explicit interruption records retain pending tickets and their
    # full charges. Do not turn incomplete work into an ended success clock.
    pending=sorted(key for key,ticket in outer.tickets.items() if ticket['status'] in ('RESERVED','STARTED'))
    payload={'schema':SCHEMA,'status':'UNQUALIFIED_INTERRUPTED','body_completed':False,'os_terminal_verified':False,
        'native_qualified':False,'execution_verified':False,'error':copy.deepcopy(error),
        'q_snapshot':ledger.snapshot() if ledger is not None else None,'bootstrap_snapshot':outer.snapshot(),
        'service_bootstrap_snapshot':service_snapshot,'charges_refunded':False}
    # Snapshot publication is still separately prepaid; a failed/unknown Q is
    # retained as failure evidence and is never forced through success end().
    publisher=_publisher(context,service,ledger)
    try:
        reference=publisher._write('PARENT_INTERRUPTION.json',payload,'parent_interruption')
        publisher.tick();publisher.handle.complete();publisher.finished=True
        if pending or outer._end_reserved is not None:
            now=outer.clock()
            outer._event('interrupted',{'worker_identity_sha256':outer.proof['worker_identity_sha256'],
                'boot_identity_sha256':outer.proof['boot_identity_sha256'],'last_recorded_monotonic':outer.last_clock,
                'observed_monotonic':now,'pending_ticket_ids':sorted(key for key,ticket in outer.tickets.items()
                    if ticket['status'] in ('RESERVED','STARTED')),
                'pending_end_reservation_sha256':digest(outer._end_reserved) if outer._end_reserved is not None else None},recovery=True)
        return dict(payload,interruption_ref=reference,publication_prepayment=publisher.handle.evidence)
    except BaseException as primary:
        publisher.finished=True
        try:publisher.handle.fail(primary)
        except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
        raise


def _early_interruption(service,context,error,ledger,service_snapshot):
    bootstrap=service.bootstrap
    require(type(bootstrap) is services.BootstrapLedger, 'existing durable early metadata owner')
    oid=uuid.uuid4().hex;profile=bootstrap.spec['operation_profiles']['interruption']
    bootstrap.operation('interruption','started',operation_id=oid,early_metadata_failure=True,**profile)
    try:
        payload={'schema':SCHEMA,'status':'UNQUALIFIED_INTERRUPTED','body_completed':False,
            'os_terminal_verified':False,'native_qualified':False,'execution_verified':False,
            'error':copy.deepcopy(error),'q_snapshot':ledger.snapshot() if ledger is not None else None,
            'source_service_bootstrap_snapshot':service_snapshot,'charges_refunded':False}
        original_scope=context['native_services'].get('qualification_bootstrap')
        if type(original_scope) is boot.BootstrapAccounting:payload['original_scope_bootstrap_snapshot']=original_scope.snapshot()
        blob=canonical(payload)+b'\n'
        require(len(blob)<=raw.MAX_JSON and profile['write_bytes_upper']>=len(blob)
                and profile['read_bytes_upper']>=len(blob)+1 and profile['identity_hash_attempts_upper']>=1
                and profile['identity_hash_bytes_upper']>=len(blob), 'actual early interruption prepayment below known bytes')
        path=bootstrap.directory/('PARENT_INTERRUPTION.'+oid+'.json')
        require(path.parent.resolve()==bootstrap.directory.resolve() and not path.is_symlink(), 'owned create-only early interruption')
        with path.open('xb') as stream:require(stream.write(blob)==len(blob),'short early interruption');stream.flush();os.fsync(stream.fileno())
        with path.open('rb') as stream:observed=stream.read(len(blob)+1)
        require(observed==blob,'early interruption raw readback')
        ref={'path':str(path),'sha256':hashlib.sha256(observed).hexdigest(),'bytes':len(observed)}
        bootstrap.operation('interruption','returned',operation_id=oid,bytes_read=len(observed),bytes_written=len(blob),
            identity_hash_attempts_actual=1,identity_hash_bytes_actual=len(observed))
        return dict(payload,interruption_ref=ref)
    except BaseException as primary:
        try:bootstrap.operation('interruption','failed',operation_id=oid,error={'type':type(primary).__name__,'message':str(primary)[:1000]})
        except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
        raise


def _retain_final_bootstrap(service,evidence):
    original=service.bootstrap
    require(type(original) is services.BootstrapLedger, 'original paid service metadata owner')
    profile=original.spec['operation_profiles']['publication'];oid=uuid.uuid4().hex
    original.operation('publication','started',operation_id=oid,final_scope_bootstrap_capsule=True,**profile)
    try:
        blob=canonical(evidence)+b'\n'
        require(len(blob)<=raw.MAX_JSON and profile['write_bytes_upper']>=len(blob)
                and profile['read_bytes_upper']>=len(blob)+1 and profile['identity_hash_attempts_upper']>=1
                and profile['identity_hash_bytes_upper']>=len(blob), 'final scope bootstrap capsule exceeds metadata prepayment')
        path=original.directory/('FINAL_SCOPE_BOOTSTRAP.'+oid+'.json')
        require(path.parent.resolve()==original.directory.resolve() and not path.is_symlink(), 'owned final bootstrap capsule')
        with path.open('xb') as stream:require(stream.write(blob)==len(blob),'short final bootstrap capsule');stream.flush();os.fsync(stream.fileno())
        with path.open('rb') as stream:observed=stream.read(len(blob)+1)
        require(observed==blob,'final bootstrap capsule raw readback')
        reference={'path':str(path),'sha256':hashlib.sha256(observed).hexdigest(),'bytes':len(observed)}
        original.operation('publication','returned',operation_id=oid,bytes_read=len(observed),bytes_written=len(blob),
            identity_hash_attempts_actual=1,identity_hash_bytes_actual=len(observed))
        return reference
    except BaseException as primary:
        try:original.operation('publication','failed',operation_id=oid,error={'type':type(primary).__name__,'message':str(primary)[:1000]})
        except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
        raise


def completion_seal_role(context,result,ledger,service_bootstrap,operation):
    authority.require_execution_permit(context.get('native_execution_permit') if type(context) is dict else None,context,'completion_publication')
    service=_owner(context,operation);_released(service)
    require(result is context['native_services'].get('parent_body_publication')
            and result.get('status')=='UNQUALIFIED_BODY_COMPLETE' and result.get('os_terminal_verified') is False,
            'exact retained parent body result; no fabricated terminal claim')
    snapshot=ledger.snapshot(); require(not snapshot['worker_open'] and not snapshot['pending_operations'], 'parent body Q cannot change after publication')
    terminal_owner=_terminal_owner(context,ledger)
    publisher=None
    child_replay_ref=None
    try:
        publisher=_publisher(context,service,ledger)
        reference=result['body_ref']
        actual=publisher._reference(reference['path'],'parent_body_complete',publisher.spec['max_receipt_bytes'])
        require(all(actual[key]==reference[key] for key in ('path','sha256','bytes')), 'parent body changed before unqualified seal')
        if terminal_owner is not None:
            replay=_replay_retained_checkpoint_children(publisher,context,ledger,result,terminal_owner)
            child_replay_ref=publisher._write('PARENT_CHILD_TERMINAL_REPLAY.json',replay,'checkpoint_terminal_replay')
            terminal_owner.close_terminal_handles()
            require(terminal_owner.final_handle_lifetime_closed and not terminal_owner.terminal_handle_records
                    and terminal_owner.handle.evidence['status']=='RETURNED',
                    'actual last child clone closure and process ticket completion before scope end')
        publisher.tick();publisher.handle.complete();publisher.finished=True
    except BaseException as primary:
        _close_terminals_on_failure(terminal_owner,primary)
        if publisher is None:raise
        publisher.finished=True
        try:publisher.handle.fail(primary)
        except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
        raise
    outer=context['native_services'].get('qualification_bootstrap')
    require(type(outer) is boot.BootstrapAccounting and outer.context is context and not outer.closed,
            'actual original scope bootstrap before final clock record')
    outer.end()
    final_evidence=outer.inspect_evidence()
    require(final_evidence['snapshot']['clock_ended'] and not final_evidence['snapshot']['unknown_wall'],
            'source-ended final scope bootstrap clock required')
    bootstrap_ref=_retain_final_bootstrap(service,final_evidence)
    return {'schema':SCHEMA,'status':'UNQUALIFIED_BODY_SEAL','body_ref':result['body_ref'],
        'native_qualified':False,'execution_verified':False,'os_terminal_verified':False,
        'supervisor_terminal_verification_required':True,'publication_prepayment':result['publication_prepayment'],
        'seal_readback_prepayment':publisher.handle.evidence,'scope_bootstrap_evidence_ref':bootstrap_ref,
        'checkpoint_child_terminal_replay_ref':child_replay_ref,
        'service_bootstrap_clock_scope':'original metadata service whole-clock provenance still requires supervisor verification'}
