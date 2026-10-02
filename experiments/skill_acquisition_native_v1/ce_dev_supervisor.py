"""Fixed DEV worker supervision with native metadata costs; default CLOSED.

The parent issues only existing launch/OS observation capabilities. A completed
read retains original worker bytes and two ended cost ledgers. Scoring is a
separate consumer, and historical terminal files never become a fresh OS proof.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import os
from pathlib import Path
import sys
import time
from contextlib import redirect_stdout
from types import MappingProxyType

from . import ce_dev_entry as entry, ce_dev_runtime as runtime
from . import ce_native_services as services, ce_process_launch as launch, ce_worker_evidence as worker
from . import ce_qualification_artifacts as raw, ce_execution_verifier as files
from . import ce_qualification_configuration as intake, ce_service_bootstrap_verifier as replay, update

CONFIG='CE8-DEV-SUPERVISOR-CONFIGURATION-1'
SPEC='CE8-DEV-SUPERVISION-SPEC-1'
COMPLETE='CE8-DEV-SUPERVISED-COMPLETION-1'
MODULE=__package__+'.ce_dev_supervisor'
_SOURCE_DECLARATION=_SOURCE_DECLARATION_SHA256=_CONFIGURATION=None
_SOURCE_COMPLETIONS=_SOURCE_COMPLETIONS_SHA256=None
_PREPAY_SEAL=object()
require,canonical,digest=raw.require,raw.canonical,raw.digest


def _pin():
    require(type(_SOURCE_DECLARATION) is MappingProxyType and raw._hex(_SOURCE_DECLARATION_SHA256),
            'DEV supervisor source declaration absent; closed before IO')
    declaration=copy.deepcopy(dict(_SOURCE_DECLARATION));child,_=entry._source_pin()
    require(set(declaration)==set(child) and digest(declaration)==_SOURCE_DECLARATION_SHA256
        and all(declaration[key]==child[key] for key in ('output_root','source_checkout','training_selection_sha256'))
        and declaration['configuration_path']!=child['configuration_path'], 'separate fixed supervisor metadata intake')
    bounds=declaration['intake_bounds']
    require(set(bounds)==set(services.COUNTERS) and all(type(v) is int and v>0 for k,v in bounds.items() if k!='wall_seconds')
        and type(bounds['wall_seconds']) in (int,float) and 0<bounds['wall_seconds']<=120
        and bounds['read_bytes']>=2*intake.MAX_CONFIGURATION+1 and bounds['hash_bytes']>=8*intake.MAX_CONFIGURATION
        and bounds['hash_attempts']>=32 and bounds['write_bytes']>=65536 and bounds['metadata_bytes']>=65536
        and bounds['operation_attempts']>=8,'finite whole supervisor intake before files')
    return declaration


def _blob(ref,maximum):
    runtime._reference(ref);require(ref['bytes']<=maximum,'bounded source read')
    path=Path(ref['path']);require(path.is_file() and not path.is_symlink() and path.stat().st_size==ref['bytes'],'regular exact source file')
    with path.open('rb') as stream:blob=stream.read(ref['bytes']+1)
    require(len(blob)==ref['bytes'] and hashlib.sha256(blob).hexdigest()==ref['sha256'],'source file bytes/hash drift')
    return blob


def validate_configuration(value,child):
    require(type(value) is dict and set(value)=={'schema','worker_configuration_ref','scope_receipt_ref','supervisor_specs','configuration_sha256'}
        and value['schema']==CONFIG and digest({k:v for k,v in value.items() if k!='configuration_sha256'})==value['configuration_sha256'],
        'exact immutable DEV supervisor configuration')
    entry.validate_configuration(child);runtime._reference(value['scope_receipt_ref'])
    require(set(value['supervisor_specs'])==set(entry.RUN_IDS),'five fixed supervisor roles')
    for role,spec in value['supervisor_specs'].items():
        require(type(spec) is dict and set(spec)=={'schema','bootstrap','constructor_bounds','resource','current_worker','launch','reader_bounds'}
            and spec['schema']==SPEC,'finite exact DEV metadata supervisor composition')
        require(set(spec['constructor_bounds'])==set(services.COUNTERS)
            and all(type(v) is int and v>0 for k,v in spec['constructor_bounds'].items() if k!='wall_seconds')
            and type(spec['constructor_bounds']['wall_seconds']) in (int,float) and 0<spec['constructor_bounds']['wall_seconds']<=120,
            'source supervisor permanent constructor bounds')
        require(set(spec['reader_bounds'])==set(raw.BUDGET_FIELDS) and all(type(v) is int and v>0 for v in spec['reader_bounds'].values()),
            'source supervisor finite independent reader')
        current=spec['current_worker'];require(current['entry_module']==MODULE and current['request_ref']==child['context_refs'][role]
            and current['argv'][1:]==['-m',MODULE,'--read-id',role], 'fixed current supervisor OS/source declaration')
        plan=spec['launch'];expected=child['entry_specs'][role]['worker']
        require(plan['worker_module']==entry.ENTRY_MODULE and plan['request_ref']==child['context_refs'][role]
            and plan['argv']==[expected['python_ref']['path'],'-m',entry.ENTRY_MODULE,'--request',plan['request_ref']['path'],'--read-id',role]
            and plan['source_files']==expected['source_files'] and plan['approved_code_sha256']==expected['approved_code_sha256']
            and plan['python_ref']==expected['python_ref'] and plan['process_image_ref']==expected['process_image_ref'],
            'source-fixed DEV child launcher/image/module/request/source argv')
        for kind in ('dev_supervisor_lifetime','dev_supervisor_metadata','dev_supervisor_publication',
                     'dev_supervisor_verify','qualification_artifact_readback','qualification_reducer_overhead'):
            services._profile(spec['bootstrap']['operation_profiles'][kind])
        require(spec['bootstrap']['operation_profiles']['dev_supervisor_lifetime']['wall_s_upper']>=plan['deadline_seconds'],
                'source hashes/admissions prepaid through owned process lifetime')
    return value


def _configuration():
    global _CONFIGURATION
    declaration=_pin()
    if _CONFIGURATION is not None:return copy.deepcopy(_CONFIGURATION)
    role=entry._fixed_read_id(sys.argv[1:]);bounds=declaration['intake_bounds'];origin=time.monotonic();counts=dict.fromkeys(bounds,0)
    directory=Path(declaration['output_root'])/'entry-intake';require(not directory.is_symlink(),'regular source supervisor intake')
    directory.mkdir(parents=True,exist_ok=True);path=directory/('supervisor-'+role+'.jsonl')
    with path.open('xb') as stream:stream.flush();os.fsync(stream.fileno())
    head=intake._append(path,_SOURCE_DECLARATION_SHA256,None,'PERMANENT_RESERVATION',{'bounds':bounds,'read_id':role,
        'metadata_only':True,'model_authority':False},bounds,counts,origin)
    try:
        config_path=Path(declaration['configuration_path']);require(config_path.is_file() and not config_path.is_symlink()
            and 0<config_path.stat().st_size<=intake.MAX_CONFIGURATION,'bounded compiled supervisor configuration')
        with config_path.open('rb') as stream:blob=stream.read(intake.MAX_CONFIGURATION+1)
        value=raw._json(blob);child_blob=_blob(value['worker_configuration_ref'],intake.MAX_CONFIGURATION);child=raw._json(child_blob)
        counts.update(read_bytes=len(blob)+len(child_blob),hash_attempts=16,hash_bytes=8*intake.MAX_CONFIGURATION)
        require(all(counts[k]<=bounds[k] for k in counts),'supervisor intake cap exhausted')
        validate_configuration(value,child)
        worker_declaration,_=entry._source_pin()
        require(value['worker_configuration_ref']['path']==worker_declaration['configuration_path'],'exact compiled child configuration path')
        intake._append(path,_SOURCE_DECLARATION_SHA256,head,'RETURNED',{'configuration_sha256':value['configuration_sha256'],
            'read_id':role,'child_configuration_sha256':child['configuration_sha256']},bounds,counts,origin)
        _CONFIGURATION={'supervisor':copy.deepcopy(value),'worker':copy.deepcopy(child)};return copy.deepcopy(_CONFIGURATION)
    except BaseException as primary:
        try:intake._append(path,_SOURCE_DECLARATION_SHA256,head,'FAILED',{'type':type(primary).__name__,'message':str(primary)[:1000],
            'charges_refunded':False},bounds,counts,origin)
        except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
        raise


class DevSupervisorPrepayment:
    def __init__(self,owner):
        require(type(owner) is DevSupervisor and any(DevSupervisorPrepayment is item for item in services._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES),
                'original source DEV supervisor prepayment before files')
        owner._admit();self.owner,self._seal,self.consumed=owner,_PREPAY_SEAL,False
        spec=owner.spec['bootstrap'];self.spec_sha=digest(spec);self.request_sha=None
        self.started=time.monotonic();self.bounds=copy.deepcopy(owner.spec['constructor_bounds'])
        ref=owner.child['context_refs'][owner.role];setup=services._profile(spec['setup_bounds'])
        record={'schema':'CE8-DEV-SUPERVISOR-CONSTRUCTOR-1','context_ref':ref,'bootstrap_spec_sha256':self.spec_sha,
            'bounds':self.bounds,'status':'PERMANENTLY_RESERVED','model_authority':False}
        blob=canonical(record)+b'\n'
        require(self.bounds['read_bytes']>=ref['bytes']+1+setup['read_bytes_upper']
            and self.bounds['write_bytes']>=len(blob)+setup['write_bytes_upper']+2*raw.MAX_JSON
            and self.bounds['metadata_bytes']>=len(blob)+2*raw.MAX_JSON
            and self.bounds['hash_attempts']>=setup['identity_hash_attempts_upper']+32
            and self.bounds['hash_bytes']>=setup['identity_hash_bytes_upper']+32*raw.MAX_JSON
            and self.bounds['wall_seconds']>=setup['wall_s_upper'],'complete constructor self/dependent upper envelope')
        directory=Path(spec['owned_directory']);require(directory.is_absolute() and not directory.is_symlink()
            and directory.resolve().is_relative_to(Path(_pin()['output_root']).resolve()),'source supervisor constructor owner')
        directory.mkdir(parents=True,exist_ok=True);path=directory/'DEV_SUPERVISOR_CONSTRUCTOR.json'
        with path.open('xb') as stream:require(stream.write(blob)==len(blob),'short permanent constructor write');stream.flush();os.fsync(stream.fileno())
    def read_context(self):
        require(self.request_sha is None and not self.consumed,'one prepaid supervisor context')
        value=raw._json(_blob(self.owner.child['context_refs'][self.owner.role],self.bounds['read_bytes']))
        runtime.validate_context(value);require(value['read_id']==self.owner.role,'fixed supervisor read role')
        require(time.monotonic()-self.started<=self.bounds['wall_seconds'],'supervisor constructor elapsed cap')
        self.request_sha=value['request_sha256'];return value
    def require_constructor(self,spec_sha256,request_sha256):
        require(self._seal is _PREPAY_SEAL and not self.consumed and spec_sha256==self.spec_sha and request_sha256==self.request_sha,
                'exact one-use DEV supervisor constructor reservation')
        self.owner._admit();self.consumed=True


def metadata_verifier(owner,spec_sha,request_sha,scope_sha):
    require(type(owner) is DevSupervisor,'original DEV metadata supervisor')
    with owner.verify_span():
        owner._admit();spec=owner._observation_specs.get(spec_sha)
        require(type(spec) is dict and digest(spec)==spec_sha and spec['request_sha256']==request_sha==owner.context['request_sha256']
            and spec['scope_sha256']==scope_sha==owner.context['scope_sha256'],'exact paid current/child observation scope')
    return True


def launch_verifier(owner,spec_sha,request_sha,scope_sha):
    require(type(owner) is DevSupervisor,'original DEV launch-only supervisor')
    with owner.verify_span():
        owner._admit();require(owner.launch_spec is not None and digest(owner.launch_spec)==spec_sha
            and request_sha==owner.context['request_sha256'] and scope_sha==owner.context['scope_sha256'],'source-fixed DEV launch scope')
    return True


def launch_observation(spec,permit,identity,parent_identity,operation):
    require(type(permit) is launch.NativeLaunchPermit,'existing launch capability before child observation')
    permit.require(spec,operation);owner=permit._issuer
    require(type(owner) is DevSupervisor and operation is owner.operation,'same original metadata cost owner')
    from .ce_child_scope_adapters import observed_child_spec
    bound=observed_child_spec(spec,identity,parent_identity);owner._observation_specs[digest(bound)]=bound
    return bound,worker.WorkerObservationPermit(bound,owner,metadata_verifier,operation)


def verification_binding(context,proof):
    return {'stage':'dev_scoring','request_sha256':context['request_sha256'],'scope_sha256':context['scope_sha256'],
        'static_recipe_sha256':digest(runtime.GREEDY),'code_sha256':context['source']['code_sha256'],
        'worker_identity_sha256':proof['worker_identity_sha256'],'attempt_id':context['read_id']}


def same_worker_proof(observed,reported,spec):
    """Require one OS/core-source identity despite current/supervisor view text."""
    for value in (observed,reported):
        require(type(value) is dict and value.get('schema')==worker.PROOF and value.get('fresh_process') is True
            and value.get('source_verified') is True and value.get('boot_verified') is True
            and value.get('request_sha256')==spec['request_sha256'] and value.get('scope_sha256')==spec['scope_sha256'],
            'exact observed/reported live worker source scope')
        preimage=worker.worker_identity_preimage(value['os_identity'],value['sources'],spec['request_sha256'],spec['scope_sha256'])
        require(digest(preimage)==value['worker_identity_sha256'],'worker core source/OS identity digest')
    require(observed['worker_identity_sha256']==reported['worker_identity_sha256'],
            'ready has different actual worker OS/source identity')


class DevSupervisor:
    def __init__(self,configuration,role):
        require(configuration==_configuration() and role in entry.RUN_IDS,'source DEV supervisor before object access')
        self.configuration,self.child,self.role=configuration,configuration['worker'],role
        self.spec=copy.deepcopy(configuration['supervisor']['supervisor_specs'][role]);self._spec_sha=digest(self.spec)
        self._observation_specs={};self.launch_spec=None;self.current_handle=None;self.completion=None
        payment=DevSupervisorPrepayment(self);self.context=payment.read_context()
        require(Path(self.context['policy']['output_directory']).resolve().is_relative_to(Path(_pin()['output_root']).resolve()),
                'source DEV read output remains inside admitted volume owner')
        self.bootstrap=services.BootstrapLedger(self.spec['bootstrap'],self.context['request_sha256'],prepayment=payment)
        self.operation=self.bootstrap.operation;self.resource=services.NativeResourceGate(self.spec['resource'],self.operation)
        self.reader=raw.RawEvidenceReader(self.operation,self.spec['reader_bounds'],_pin()['output_root'])
    def _admit(self):
        require(self.configuration==_configuration() and digest(self.spec)==self._spec_sha,'immutable source DEV metadata owner')
    def preflight(self,name,contract):
        require(name=='service_bootstrap_raw_verifier' and contract['reader_bounds']==self.reader.remaining,'bounded raw service verifier only')
        self.bootstrap.tick_wall()
    def metadata_span(self):return update._span(self.operation,'dev_supervisor_metadata',dict(self.spec['bootstrap']['operation_profiles']['dev_supervisor_metadata']))
    def verify_span(self):return update._span(self.operation,'dev_supervisor_verify',dict(self.spec['bootstrap']['operation_profiles']['dev_supervisor_verify']))
    def observe_current(self):
        with self.metadata_span():
            self._admit();policy=self.spec['current_worker'];budget=worker._Budget(policy,self.operation)
            api=budget.run('worker_os_binding',worker._WindowsOS);handle=None;primary=None
            try:
                def acquire():
                    nonlocal handle
                    handle=api.open(os.getpid());return handle
                budget.run('worker_os_binding',acquire);facts=budget.run('worker_os_binding',lambda:api.snapshot(handle));boot=budget.run('worker_os_binding',api.boot)
                require(facts['pid']==os.getpid() and facts['argv']==policy['argv'] and facts['image_path']==policy['process_image_ref']['path'],
                        'actual current supervisor module/image/whole argv')
                spec=dict(copy.deepcopy(policy),schema=worker.SCOPE,mode='registered_metadata',expected_pid=facts['pid'],
                    expected_creation_filetime_100ns=facts['creation_filetime_100ns'],expected_boot_identity=boot,
                    request_sha256=self.context['request_sha256'],scope_sha256=self.context['scope_sha256'])
                self._observation_specs[digest(spec)]=spec
                permit=worker.WorkerObservationPermit(spec,self,metadata_verifier,self.operation)
                self.current_handle=worker.open_worker(spec,permit,self.operation);proof=self.current_handle.proof()
                return proof['os_identity']
            except BaseException as error:primary=error;raise
            finally:
                if handle is not None:
                    entered=False
                    try:
                        def close():
                            nonlocal entered
                            entered=True;api.close(handle)
                        budget.run('worker_handle_close',close)
                    except BaseException as failure:
                        if not entered:
                            try:api.close(handle)
                            except BaseException as closing:raw._accounting(failure,closing)
                        if primary is not None:raw._accounting(primary,failure);raise primary from failure
                        raise
    def build_launch(self,parent_identity):
        self._admit();spec=copy.deepcopy(self.spec['launch'])
        spec.update(schema=launch.SPEC,mode='registered_native_worker',request_sha256=self.context['request_sha256'],
            scope_sha256=self.context['scope_sha256'],parent_identity=copy.deepcopy(parent_identity),
            observation_factory_descriptor={'module':MODULE,'symbol':'launch_observation',
                'source_sha256':spec['approved_code_sha256'][MODULE.replace('.','/')+'.py']})
        launch.validate_spec(spec);self.launch_spec=spec;return spec
    def reference(self,path,upper):
        with update._span(self.operation,'dev_supervisor_metadata',dict(self.spec['bootstrap']['operation_profiles']['dev_supervisor_metadata'])) as returned:
            path=Path(path);require(path.resolve().is_relative_to(Path(_pin()['output_root']).resolve()) and path.is_file() and not path.is_symlink(),
                    'original bounded owned DEV artifact')
            hashed=hashlib.sha256();count=0
            with path.open('rb') as stream:
                while block:=stream.read(min(raw.CHUNK,upper+1-count)):
                    count+=len(block);require(count<=upper,'DEV artifact byte ceiling');hashed.update(block)
            result={'path':str(path),'sha256':hashed.hexdigest(),'bytes':count};require(count>0,'nonempty original DEV artifact')
            returned.update(reference=result,bytes_read=count,bytes_written=0,identity_hash_attempts_actual=1,identity_hash_bytes_actual=count)
            return result
    def publish(self,value):
        profile=self.spec['bootstrap']['operation_profiles']['dev_supervisor_publication']
        with update._span(self.operation,'dev_supervisor_publication',dict(profile)) as returned:
            blob=canonical(value)+b'\n';require(len(blob)<=raw.MAX_JSON and len(blob)<=profile['write_bytes_upper']
                and 2*len(blob)<=profile['identity_hash_bytes_upper'] and profile['identity_hash_attempts_upper']>=2,
                'bounded create-only DEV completion self write/hash/readback')
            path=Path(self.spec['bootstrap']['owned_directory'])/'DEV_COMPLETION.json'
            with path.open('xb') as stream:require(stream.write(blob)==len(blob),'short DEV completion write');stream.flush();os.fsync(stream.fileno())
            ref={'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
            require(profile['read_bytes_upper']>=len(blob)+1 and _blob(ref,raw.MAX_JSON)==blob,'original completion readback')
            returned.update(reference=ref,bytes_read=len(blob),bytes_written=len(blob),identity_hash_attempts_actual=2,identity_hash_bytes_actual=2*len(blob))
            return ref
    def close_current(self):
        if self.current_handle is None:return
        handle=self.current_handle;entered=False
        try:
            with self.metadata_span():entered=True;handle.close()
        except BaseException as primary:
            if not entered:
                try:handle.close()
                except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
            raise
        finally:
            if handle.closed:self.current_handle=None


def read_worker(owner,body_ref,proof,terminal,observed_at):
    context=owner.context;binding=verification_binding(context,proof)
    body=files._read_json(owner.reader,body_ref,binding,'DEV_body')
    model_name,index=next((name,index) for role,name,index in runtime.RUNS if role==owner.role)
    expected={'schema':runtime.READ,'status':'DEV_BODY_COMPLETE','read_id':owner.role,'model_name':model_name,'read_index':index,
        'scope_sha256':context['scope_sha256'],'model_sha256':context['model'][model_name+'_sha256'],
        'tokenizer_sha256':context['model']['tokenizer_receipt_ref']['sha256'],'decoding_sha256':context['runtime']['decoding_sha256'],
        'source':context['source'],'greedy':runtime.GREEDY,'rows':1950,'worker_proof':proof,'process_identity_sha256':proof['worker_identity_sha256']}
    require(all(body.get(k)==v for k,v in expected.items()),'original DEV body worker/source/runtime/model/role bind')
    directory=Path(context['policy']['output_directory'])/owner.role
    require(Path(body_ref['path'])==directory/'DEV_BODY_COMPLETE.json' and Path(body['token_journal'])==directory/'TOKEN_JOURNAL.jsonl'
        and Path(body['attempts_journal'])==directory/'ATTEMPTS.jsonl','fixed original DEV output filenames')
    token=owner.reference(directory/'TOKEN_JOURNAL.jsonl',context['policy']['journal_bytes_upper'])
    attempts=owner.reference(directory/'ATTEMPTS.jsonl',context['policy']['journal_bytes_upper'])
    spec=owner.child['entry_specs'][owner.role]['bootstrap'];root=Path(spec['owned_directory'])
    refs={name:owner.reference(root/filename,replay.MAX_JOURNAL if name=='journal_ref' else replay.MAX_HEADER)
        for name,filename in (('header_ref','SERVICE_BOOTSTRAP_HEADER.json'),('journal_ref','SERVICE_BOOTSTRAP.jsonl'),('seal_ref','SERVICE_BOOTSTRAP_SEAL.json'))}
    state=replay.verify_service_bootstrap({'mode':'registered_scope','request_sha256':context['request_sha256'],'verification_binding':binding},spec,
        refs['header_ref'],refs['journal_ref'],refs['seal_ref'],reader=owner.reader,outer_preflight=owner.preflight)
    snapshot=state['derived_snapshot'];require(snapshot['clock_ended'] and not snapshot['pending'] and not snapshot['failed_operations'],
        'worker costs ended cleanly after original read/lease/handle cleanup')
    require(snapshot['last_monotonic']<=observed_at<=snapshot['last_monotonic']+spec['finalization_bounds']['wall_s_upper'],
            'external terminal observation bounds worker end/readback/exit suffix')
    return {'binding':binding,'body':body,'token_journal_ref':token,'attempts_journal_ref':attempts,'worker_service_refs':refs,
        'worker_service_snapshot':snapshot,'terminal':terminal,'terminal_observed_monotonic':observed_at}


def supervise(owner):
    require(type(owner) is DevSupervisor,'actual source DEV supervisor, no CE service impersonation')
    owned=retained=None;primary=None;terminal_reached=False
    try:
        owner.resource.launch()
        with update._span(owner.operation,'dev_supervisor_lifetime',dict(owner.spec['bootstrap']['operation_profiles']['dev_supervisor_lifetime'])):
            parent=owner.observe_current();spec=owner.build_launch(parent)
            permit=launch.NativeLaunchPermit(spec,owner,launch_verifier,owner.operation)
            owned=launch.launch_native(spec,permit,owner.operation)
            ready=owned.handshake();observed=worker.verify_worker_handle(owned.monitor,{'request_sha256':spec['request_sha256'],
                'scope_sha256':spec['scope_sha256'],'worker_identity_sha256':owned.monitor.identity_sha256},require_native=True)
            require(set(ready)=={'worker_proof'},'exact ready worker provenance envelope');proof=ready['worker_proof']
            same_worker_proof(observed,proof,spec)
            response=owned.request('read',{});require(response.get('body_completed') is True and response.get('native_qualified') is False
                and response.get('os_terminal_verified') is False,'complete fixed readonly body response')
            finished=owned.request('finish',{});require(finished=={'finished':True,'native_qualified':False},'exact fixed DEV finish response')
            physical=owned.terminal(spec['deadline_seconds']);require(physical['exit_code']==0,'same owned DEV process exit0')
            observed_at=time.perf_counter();terminal_reached=True
            retained=owned.monitor.retain_terminal()
            terminal=worker.verify_worker_handle(retained,{'request_sha256':spec['request_sha256'],'scope_sha256':spec['scope_sha256'],
                'worker_identity_sha256':proof['worker_identity_sha256']},require_native=True,terminal=True)
            require(terminal['os_identity']==physical['os_identity'] and terminal['exit_filetime_100ns']==physical['exit_filetime_100ns'],
                    'retained terminal clone equals original owned OS exit')
            result=read_worker(owner,response['raw_artifact'],proof,terminal,observed_at)
            owned.close();owned=None;retained.close();retained=None;owner.close_current()
        completion={'schema':COMPLETE,'status':'DEV_BODY_EXIT_VERIFIED_REQUIRES_COST_SEAL','read_id':owner.role,
            'request_sha256':owner.context['request_sha256'],'scope_sha256':owner.context['scope_sha256'],
            'context_ref':owner.child['context_refs'][owner.role],'scope_receipt_ref':owner.configuration['supervisor']['scope_receipt_ref'],
            'body_ref':response['raw_artifact'],'token_journal_ref':result['token_journal_ref'],'attempts_journal_ref':result['attempts_journal_ref'],
            'worker_proof':proof,'terminal':terminal,'terminal_observed_monotonic':observed_at,'verification_binding':result['binding'],
            'worker_service_refs':result['worker_service_refs'],'worker_service_snapshot':result['worker_service_snapshot'],
            'constructor_refs':{name:owner.reference(path,replay.MAX_HEADER) for name,path in (
                ('worker',Path(owner.child['entry_specs'][owner.role]['bootstrap']['owned_directory'])/'DEV_CONSTRUCTOR_PREPAYMENT.json'),
                ('supervisor',Path(owner.spec['bootstrap']['owned_directory'])/'DEV_SUPERVISOR_CONSTRUCTOR.json'))},
            'supervisor_header_ref':owner.reference(owner.bootstrap.header_path,replay.MAX_HEADER),
            'expected_supervisor_seal_path':str(owner.bootstrap.seal_path),'native_qualified':False,'learning_claim':False}
        reference=owner.publish(completion)
        owner.bootstrap.end();evidence=owner.bootstrap.inspect_evidence()
        owner.completion={'schema':COMPLETE,'status':'DEV_BODY_EXIT_COSTS_SEALED','read_id':owner.role,'completed':True,
            'completion_ref':reference,'supervisor_service_refs':evidence,'native_qualified':False,'learning_claim':False}
        return copy.deepcopy(owner.completion)
    except BaseException as error:primary=error;raise
    finally:
        failures=[]
        if owned is not None:
            if not terminal_reached:
                try:owned._stop_owned('DEV_supervisor_failure',primary)
                except BaseException as error:failures.append(error)
            try:owned.close()
            except BaseException as error:failures.append(error)
        if retained is not None:
            try:retained.close()
            except BaseException as error:failures.append(error)
        try:owner.close_current()
        except BaseException as error:failures.append(error)
        if failures:
            failure=failures[0]
            for extra in failures[1:]:raw._accounting(failure,extra)
            if primary is not None:raw._accounting(primary,failure);raise primary from failure
            raise failure


def selected_completion(read_id):
    require(type(_SOURCE_COMPLETIONS) is MappingProxyType and raw._hex(_SOURCE_COMPLETIONS_SHA256)
        and digest(dict(_SOURCE_COMPLETIONS))==_SOURCE_COMPLETIONS_SHA256 and read_id in entry.RUN_IDS,
        'source-selected DEV completions absent; closed before read')
    value=_SOURCE_COMPLETIONS.get(read_id);require(type(value) is dict,'source completed read not recorded')
    return value


def _publication_link(reader,journal_ref,binding,completion_ref,original_replay):
    """Link the exact completion bytes to an original pre-end paid return."""
    require(journal_ref['bytes']<=replay.MAX_JOURNAL,'bounded original publication journal')
    wrapped=files._decorated(reader,journal_ref,binding,'DEV_publication_linkage')
    with reader._overhead('DEV_publication_reference_linkage',work_bytes=64*journal_ref['bytes']):
        stream=reader._blocks(wrapped,binding,'DEV_publication_linkage')
        pending=b'';count=0;publication=[];reservation=end=None;head=None;primary=None
        try:
            for block in stream:
                pending+=block
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    require(0<len(line)<=replay.MAX_LINE and count<replay.MAX_EVENTS,'bounded original publication event')
                    row=raw._json(line);require(type(row.get('seq')) is int and row['seq']==count,'original publication event sequence')
                    payload=row['payload'];kind=row['kind']
                    if kind=='operation' and payload['operation']=='dev_supervisor_publication' and payload['phase']=='returned':
                        publication.append((row['seq'],payload['operation_id'],payload['fields'].get('reference')))
                    elif kind=='end_reservation':reservation=row['seq']
                    elif kind=='end':end=row['seq']
                    head=row['sha256'];count+=1
                require(len(pending)<=replay.MAX_LINE,'bounded original publication partial line')
            require(not pending and head==original_replay['derived_snapshot']['head']
                and count==len(original_replay['original_offsets']),'exact same independently replayed parent journal')
            require(len(publication)==1 and publication[0][2]==completion_ref and type(reservation) is int and type(end) is int
                and publication[0][0]<reservation<end,'selected completion is the original paid publication returned before parent end/seal')
        except BaseException as error:primary=error;raise
        finally:
            try:stream.close()
            except BaseException as failure:
                if primary is not None:raw._accounting(primary,failure);raise primary from failure
                raise
    return {'operation_id':publication[0][1],'publication_seq':publication[0][0],
        'end_reservation_seq':reservation,'end_seq':end,'journal_ref':copy.deepcopy(journal_ref)}


def verify_completed_dev(selection,*,reader,outer_preflight):
    require(type(selection) is dict and any(selection is value for value in (_SOURCE_COMPLETIONS or {}).values()),
            'exact source-selected DEV completion before IO')
    require(selected_completion(selection['read_id']) is selection,'immutable DEV completion selection')
    require(type(reader) is raw.RawEvidenceReader and callable(outer_preflight),'paid independent DEV completion reader')
    binding=selection['verification_binding'];raw.binding(binding)
    outer_preflight('completed_dev_pair',{'reader_bounds':dict(reader.remaining),'read_id':selection['read_id']})
    completion=files._read_json(reader,selection['completion_ref'],binding,'DEV_completion')
    context=files._read_json(reader,completion['context_ref'],binding,'DEV_context');runtime.validate_context(context)
    require(completion['schema']==COMPLETE and completion['status']=='DEV_BODY_EXIT_VERIFIED_REQUIRES_COST_SEAL'
        and completion['verification_binding']==binding==verification_binding(context,completion['worker_proof'])
        and completion['read_id']==selection['read_id']==context['read_id'],'source completed DEV logical/source binding')
    require(completion['supervisor_header_ref']==selection['supervisor_service_refs']['header_ref']
        and completion['expected_supervisor_seal_path']==selection['supervisor_service_refs']['seal_ref']['path'],
        'completion pairs only the original supervisor header and expected final cost seal')
    proof=completion['worker_proof'];terminal=completion['terminal']
    require(proof['schema']==worker.PROOF and proof['fresh_process'] is True and proof['source_verified'] is True
        and terminal['schema']==worker.TERMINAL and terminal['worker_identity_sha256']==proof['worker_identity_sha256']
        and terminal['os_identity']==proof['os_identity'] and terminal['exit_code']==0
        and terminal['request_sha256']==context['request_sha256'] and terminal['scope_sha256']==context['scope_sha256'],
        'source-pinned historical same-worker OS terminal; no fresh PID inference')
    replays={}
    constructors={}
    for name,refs,spec in (('worker',completion['worker_service_refs'],selection['worker_service_spec']),
                         ('supervisor',selection['supervisor_service_refs'],selection['supervisor_service_spec'])):
        record=files._read_json(reader,completion['constructor_refs'][name],binding,'DEV_constructor_'+name)
        require(record['schema']==('CE8-DEV-CONSTRUCTOR-PREPAYMENT-1' if name=='worker' else 'CE8-DEV-SUPERVISOR-CONSTRUCTOR-1')
            and record['status']=='PERMANENTLY_RESERVED' and record['model_authority'] is False
            and record['context_ref']==completion['context_ref'] and record['bootstrap_spec_sha256']==digest(spec)
            and set(record['bounds'])==set(services.COUNTERS)
            and all(type(v) is int and v>0 for k,v in record['bounds'].items() if k!='wall_seconds')
            and type(record['bounds']['wall_seconds']) in (int,float) and 0<record['bounds']['wall_seconds']<=120,
            'original permanent independent constructor costs retained without refund')
        constructors[name]=record['bounds']
        replays[name]=replay.verify_service_bootstrap({'mode':'registered_scope','request_sha256':context['request_sha256'],
            'verification_binding':binding},spec,refs['header_ref'],refs['journal_ref'],refs['seal_ref'],reader=reader,outer_preflight=outer_preflight)
        snapshot=replays[name]['derived_snapshot'];require(snapshot['clock_ended'] and not snapshot['pending'] and not snapshot['failed_operations'],
            'both original cost owners ended without missing/failed work')
    require(replays['worker']['derived_snapshot']==completion['worker_service_snapshot'],'worker cost record differs from actual replay')
    publication=_publication_link(reader,selection['supervisor_service_refs']['journal_ref'],binding,selection['completion_ref'],replays['supervisor'])
    last=replays['worker']['derived_snapshot']['last_monotonic']
    require(last<=completion['terminal_observed_monotonic']<=last+selection['worker_service_spec']['finalization_bounds']['wall_s_upper'],
            'source retained external terminal suffix bound')
    body=files._read_json(reader,completion['body_ref'],binding,'DEV_body')
    require(body['read_id']==context['read_id'] and body['worker_proof']==proof and body['rows']==1950
        and body['scope_sha256']==context['scope_sha256'] and body['source']==context['source'],'original source DEV body linkage')
    return {'schema':COMPLETE,'completed_pair_verified':True,'execution_reverified':False,'native_qualified':False,'evaluation_authorized':False,
        **{key:body[key] for key in ('read_id','model_name','read_index','scope_sha256','model_sha256','tokenizer_sha256','decoding_sha256','greedy','source')},
        **{key:completion[key] for key in ('context_ref','scope_receipt_ref','body_ref','token_journal_ref','attempts_journal_ref','worker_proof','verification_binding')},
        'completion_ref':selection['completion_ref'],'cost_seal_ref':selection['supervisor_service_refs']['seal_ref'],
        'service_replays':replays,'constructor_reservations':constructors,'publication_linkage':publication,
        'selection_sha256':digest(selection),'source_selection_sha256':_SOURCE_COMPLETIONS_SHA256}


def main():
    configuration=_configuration();role=entry._fixed_read_id(sys.argv[1:])
    services._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES|=frozenset((DevSupervisorPrepayment,))
    worker._REGISTERED_METADATA_VERIFIERS|=frozenset((metadata_verifier,))
    launch._REGISTERED_LAUNCH_VERIFIERS|=frozenset((launch_verifier,))
    launch._REGISTERED_OBSERVATION_FACTORIES|=frozenset((launch_observation,))
    owner=DevSupervisor(configuration,role)
    with redirect_stdout(sys.stderr):result=supervise(owner)
    print(canonical(result).decode('ascii'));return 0


if __name__=='__main__':raise SystemExit(importlib.import_module(__package__+'.ce_dev_supervisor').main())
