"""One genuine retained-parent actor qualification fixture; no model modules.

The exact old provider performs atomic Job assignment at target birth. This
parent observes CLR startup, retained target exit, child closure and lifetime IO.
Its own final process exit still needs the original tool owner's observation.
"""
import ctypes as c,hashlib,math,os,sys,threading,time
from pathlib import Path
import admission as a
import accounting as cost
Accounting=cost.Budget
def publish(owner,name,value,maximum=16384):
    raw=a.canonical(value,owner=owner,maximum=maximum)
    def write():
        with (a.OUTPUT/name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    owner.run('control',{'operation_attempts':3,'write_bytes':maximum},write)
    return a.ref(a.OUTPUT/name,raw,owner=owner)
def validate_bound_record(permit,name,value):
    ticket=permit.owner.before('control',operation_attempts=64,metadata_bytes=256)
    permit._static.envelopes_module.validate(permit._static.envelopes['schemas'][name],value)
    a.require(value['grant_ref']==permit.grant_ref,'same actual source-issued grant reference, not merely width-domain')
    ticket.finish();return None
def publish_birth(permit,case,identity):
    owner=permit.owner;build=owner.before('control',operation_attempts=1,metadata_bytes=1024)
    value={'identity':identity,'grant_ref':permit.grant_ref,'case_name':case,'target_manifest_ref':permit.packet['target_manifest_ref']}
    build.finish()
    validate_bound_record(permit,'ACTOR_'+case+'_RETAINED_BIRTH.json',value)
    a.require(identity['image_path']==permit.target['runtime_ref']['path'] and identity['argv'][5]==permit.grant_ref['sha256'],
      'exact fixed runtime image and actual fresh grant argv before smaller encoding')
    return publish(owner,'ACTOR_'+case+'_RETAINED_BIRTH.json',value,maximum=4096)
def target_argv(permit,case):
    permit.require();ticket=permit.owner.before('control',operation_attempts=1,metadata_bytes=512);a.require(case in a.CASE_NAMES,'fixed closed source case');t=permit.target
    argv=(t['runtime_ref']['path'],'exec',t['artifact_ref']['path'],'--qualification-probe',
      str(a.ROOT/'FIXTURE_GRANT.json'),permit.grant_ref['sha256'],str(a.ACTOR_OUTPUT/case));ticket.finish();return argv
def validate_probe(value,permit,case):
    permit.require();ticket=permit.owner.before('control',operation_attempts=32,metadata_bytes=1024);g=permit.grant
    a.require(type(value) is dict and value['schema']=='CE8-ACTOR-QUALIFICATION-PROBE-CLOSED-1'
      and value['grant_sha256']==permit.grant_ref['sha256'] and value['model_loads']==0 and value['optimizer_updates']==0
      and value['native_model_qualification'] is False and value['job_active_processes']==0
      and value['all_handles_closed'] is True and value['case_name']==case
      and value['child_exit_code']==(0 if case=='normal_exit' else 124) and value['refunds']==0
      and value['physical_origin_monotonic']==g['physical_origin_monotonic']
      and value['physical_deadline_monotonic']==g['physical_deadline_monotonic']
      and value['source_identity']==permit.packet['target_source_ref']
      and value['target_manifest_ref']==permit.packet['target_manifest_ref'], 'actual fixed actor probe source/root/closed-child evidence')
    a.require(value['selected']==a.CASE_SELECTED and type(value['used']) is dict and set(value['used'])==set(a.FIELDS)
      and all((type(value['used'][key]) in (int,float) if key=='wall_seconds' else type(value['used'][key]) is int) and math.isfinite(value['used'][key])
        and 0<=value['used'][key]<=a.CASE_SELECTED[key] for key in a.FIELDS),'actual target charges inside permanently retained case selection')
    a.require(type(value['actual_retained_child_pid']) is int and value['actual_retained_child_pid']>0
      and type(value['creation_filetime_100ns']) is int and value['creation_filetime_100ns']>0
      and type(value['exit_filetime_100ns']) is int and value['exit_filetime_100ns']>=value['creation_filetime_100ns'],
      'actual child retained birth/exit, never a claimed success flag alone')
    ticket.finish();return None
def validate_child_probe(value,permit,case):
    # New bounded callback-domain walk is explicitly priced before any work.
    ticket=permit.owner.before('control',operation_attempts=64,metadata_bytes=256)
    child=value['child_receipt']
    permit._static.envelopes_module.validate(permit._static.envelopes['schemas']['ACTOR_PROBE_CHILD_'+case],child)
    a.require(child['physical_origin_monotonic']==permit.grant['physical_origin_monotonic']
      and child['physical_deadline_monotonic']==permit.grant['physical_deadline_monotonic']
      and child['pid']==value['actual_retained_child_pid']
      and child['creation_filetime_100ns']==value['creation_filetime_100ns']
      and child['exit_filetime_100ns']==value['exit_filetime_100ns']
      and child['returncode']==value['child_exit_code'],'same original root and retained child timing/identity')
    if case=='timeout':
        a.require(permit.grant['physical_origin_monotonic']<child['timeout_planned_dispatch_monotonic']
          and 0<child['timeout_stop_monotonic']<=child['timeout_stop_returned_monotonic']<=child['nested_timeout_deadline']
          and child['nested_timeout_deadline']<=permit.grant['physical_deadline_monotonic']
          and 0<child['timeout_scheduled_due_milliseconds']<=1900,
          'strict original timeout request/return fence and bounded earlier dispatch; no lateness tolerance')
    ticket.finish();return child

def owned_cleanup(backend,action,final=False):
    a.require(not backend.cleanup_mode(),'no nested or leaked cleanup mode')
    backend._cleanup_local.active=final
    try:return action()
    finally:backend._cleanup_local.active=False

class OriginalOwnerLogs:
    """Fixed two-reader consumer of original positively selected log workspace.

    Every I/O dispatch and return checks the same original owner clock. No
    reader arms a tail, samples Native resources or acquires stop authority.
    Two <=1536-byte payload slots +5120 logical state use original8192.
    """
    def __init__(self,backend,directory):
        self.backend=backend;self.directory=directory
        # All partial fields/slots are retained before a lock, Path or Thread
        # constructor can fail; caller retained this object before __init__.
        self.threads=(c.py_object*2)(None,None)
        self.errors=(c.py_object*2)(None,None)
        self.secondary_close_errors=(c.py_object*2)(None,None)
        self.streams=(c.py_object*2)(None,None)
        self.blocks=(c.py_object*2)(None,None)
        self.log_identities=(c.py_object*2)(cost.OutputDescriptorIdentity(),cost.OutputDescriptorIdentity())
        self.sidecar_identities=(c.py_object*2)(cost.OutputDescriptorIdentity(),cost.OutputDescriptorIdentity())
        self.started=(c.c_bool*2)(False,False)
        self.ended=(c.c_bool*2)(False,False)
        self.close_attempted=(c.c_bool*2)(False,False)
        self.actual_io=(c.c_uint32*6)() # open/read/write/flush/fsync/close
        self.read=self.persisted=self.calls=0;self.paths=None;self.lock=None;self.names=None
        self.path_strings=None;self.sidecars=None;self.case=None
        self.block_log_writes=(c.c_int32*2)(-1,-1)
        self.sidecar_attempted=(c.c_bool*2)(False,False)
        self.sidecar_stages=(c.c_uint8*2)()
        self.sidecar_failed=(c.c_bool*2)(False,False)
        self.sidecar_written=(c.c_uint32*2)()
        self.sidecar_write_observed=(c.c_bool*2)(False,False)
        self.sidecar_bytes=(c.c_uint32*2)()
        self.sidecar_digests=(c.py_object*2)(None,None)
        self.sidecar_streams=(c.py_object*2)(None,None)
        self.sidecar_close_attempted=(c.c_bool*2)(False,False)
        self.sidecar_close_observed=(c.c_bool*2)(False,False)
        self._guard()
        plan=backend._creation_plan
        a.require(plan._root_owner is backend.owner and plan._static.creation_plan is plan,
          'same original prepared log input owner')
        for case,prepared_directory,paths,names,path_strings,sidecars in plan._log_inputs:
            if prepared_directory is directory:
                self.paths=paths;self.names=names;self.path_strings=path_strings;self.sidecars=sidecars;self.case=case;break
        a.require(self.paths is not None,'borrow exact same PREP log directory/paths by identity')
        self.lock=threading.Lock();self._guard()
    def _guard(self,closing=False):
        # Pure existing owner clock/liveness check. allow_failed permits only
        # original root/tail closure; it never creates or resets a deadline.
        self.backend.owner._tick(allow_failed=closing)
    def reserve(self,count,closing=False):
        a.require(type(count) is int and count==1,'one actual bounded log I/O source selection')
        with self.lock:
            a.require(self.calls+count<=(2048 if closing else 2032),'source-owned original log operations exhausted')
            self.calls+=count
    def perform(self,index,kind,action,*,closing=False,returned=None):
        self._guard(closing);self.reserve(1,closing=closing);self._guard(closing)
        with self.lock:self.actual_io[kind]+=1
        value=action()
        # Record exact returned resource/data ownership BEFORE a late guard.
        if returned is not None:returned(value)
        self._guard(closing);return value
    def retain_open(self,index,stream):
        self.streams[index]=stream;self.log_identities[index].capture(stream)
    def retain_read(self,index,block):
        self.blocks[index]=block;self.block_log_writes[index]=-1
        with self.lock:
            self.read+=len(block)
            a.require(len(block)<=1536 and self.read<=8388608,'source-owned original bounded log read selection')
    def retain_close(self,index,value):
        self.log_identities[index].close_returned=True;self.streams[index]=None
    def close_stream(self,index):
        stream=self.streams[index]
        if stream is None:return
        try:
            a.require(not self.close_attempted[index],'each original log stream close attempted at most once')
            # Admission and both guards belong to the SAME bounded secondary
            # diagnostic operation, not only the native stream.close call.
            def close():
                identity=self.log_identities[index]
                a.require(identity is not None,'initialized original log status before close')
                self.close_attempted[index]=True;identity.close_attempted=True
                try:return stream.close()
                except BaseException:identity.close_failed=True;raise
            self.perform(index,5,close,closing=True,returned=lambda value:self.retain_close(index,value))
        except BaseException as close_error:
            # One fixed diagnostic per reader. perform never runs its return
            # guard after an action raises, so actual-close/afterguard double
            # failure is not a reachable source path. Preserve first evidence.
            if self.secondary_close_errors[index] is None:
                name=type(close_error).__name__
                self.secondary_close_errors[index]=name if name.isascii() and len(name)<=128 else 'UNBOUNDED_CLOSE_ERROR_TYPE'
            cost.traceback.clear_frames(close_error.__traceback__)
            raise

    def reader(self,index,fd,name):
        block=None
        try:
            self.perform(index,0,lambda:open(self.path_strings[index],'xb',buffering=0),returned=lambda stream:self.retain_open(index,stream))
            while True:
                a.require(block is None and self.blocks[index] is None,'old block released before next read')
                block=self.perform(index,1,lambda:os.read(fd,1536),returned=lambda value:self.retain_read(index,value))
                if not block:
                    self.blocks[index]=None;block=None;break
                # The aggregate check and actual synchronous write share the
                # same lock. No competing reader can overspend persisted cap.
                self._guard();self.reserve(1);self._guard()
                with self.lock:
                    a.require(self.persisted+len(block)<=8192,'source-owned aggregate log byte ceiling')
                    self._guard();self.actual_io[2]+=1
                    written=self.streams[index].write(block)
                    self.block_log_writes[index]=written if type(written) is int and 0<=written<=len(block) else -2
                    a.require(type(written) is int and 0<=written<=len(block),'actual log write count domain')
                    self.persisted+=written
                    a.require(written==len(block),'source-owned full log write')
                    self._guard()
                self.blocks[index]=None;block=None
            # EOF closure consists of three distinct checked and counted I/O
            # attempts. A late return prevents dispatch of the following call.
            self.perform(index,3,lambda:self.streams[index].flush(),closing=True)
            self.perform(index,4,lambda:os.fsync(self.streams[index].fileno()),closing=True)
            self.close_stream(index)
        except BaseException as error:
            self.errors[index]=error
            # Returned late/unwritten payload remains in this original slot.
            # Reader records a signal; original control thread owns Native work.
        finally:
            if self.streams[index] is not None and not self.close_attempted[index]:
                try:self.close_stream(index)
                except BaseException as error:
                    if self.errors[index] is None:self.errors[index]=error
            self.ended[index]=True
    def publish_late_blocks(self):
        # Original controller calls only after readers are observed quiescent.
        a.require(all(thread is None or not thread.is_alive() for thread in self.threads),
          'same retained reader quiescence before late block disposition')
        first=None
        for index in range(2):
            block=self.blocks[index]
            if block is None or len(block)==0:continue
            a.require(type(block) is bytes and len(block)<=1536,'fixed original late block extent')
            a.require(not self.sidecar_attempted[index],'original late sidecar selection never retried')
            self.sidecar_attempted[index]=True;self.sidecar_bytes[index]=len(block)
            # Mark BEFORE selecting the complete source recipe. Refusal records
            # attempted disposition and retains original payload/partial files.
            try:
                self.backend.owner.tail_run(1,lambda:self._publish_sidecar(index,block))
                a.require(self.blocks[index] is block,'same original payload through complete tail return guard')
                self.sidecar_stages[index]=10;self.blocks[index]=None
            except BaseException as error:
                self.sidecar_failed[index]=True
                if first is None:first=error
        if first is not None:raise first
    def _publish_sidecar(self,index,block):
        owner=self.backend.owner
        scope=owner.tail_working_scope('sidecar'+str(index))
        readback=None;primary=None
        try:
            scope.phase('sidecar',8192)
            a.require(type(block) is bytes and len(block)<=1536 and type(self.sidecars[index]) is str
              and len(self.sidecars[index])<=320 and self.sidecars[index].isascii(),
              'bounded sidecar payload/native filename domain before any I/O')
            a.require(owner._tail_deadline is not None and self.sidecars is not None,
              'only already selected authenticated original tail')
            self.sidecar_stages[index]=1
            def checked(action,returned=None):
                owner._tick(allow_failed=True);value=action()
                if returned is not None:returned(value)
                try:owner._tick(allow_failed=True)
                except BaseException:
                    # Preserve the exact exception without retaining a readback
                    # data copy in this guard's traceback frame.
                    value=None;raise
                return value
            def created(stream):
                self.sidecar_streams[index]=stream;self.sidecar_stages[index]=2
                self.sidecar_identities[index].capture(stream)
            def wrote(value):
                a.require(type(value) is int and 0<=value<=len(block),'actual sidecar write count domain')
                self.sidecar_written[index]=value;self.sidecar_write_observed[index]=True;self.sidecar_stages[index]=3
            def closed(value):
                self.sidecar_identities[index].close_returned=True
                self.sidecar_streams[index]=None;self.sidecar_close_observed[index]=True
            primary=None;readback=None
            try:
                checked(lambda:open(self.sidecars[index],'xb+',buffering=0),created)
                written=checked(lambda:self.sidecar_streams[index].write(block),wrote)
                a.require(written==len(block),'full original late sidecar write required')
                checked(lambda:self.sidecar_streams[index].flush());self.sidecar_stages[index]=4
                checked(lambda:os.fsync(self.sidecar_streams[index].fileno()));self.sidecar_stages[index]=5
                checked(lambda:self.sidecar_streams[index].seek(0));self.sidecar_stages[index]=6
                readback=checked(lambda:self.sidecar_streams[index].read(len(block)+1))
                a.require(readback==block,'exact durable original late sidecar readback')
                readback=None
                self.sidecar_stages[index]=7
                # Exact readback equality was observed. Hash that identical
                # original block after releasing the extra readback allocation.
                digest=checked(lambda:hashlib.sha256(block).hexdigest())
                self.sidecar_digests[index]=digest;self.sidecar_stages[index]=8
                # Retain owned stream through all publication steps. A close is
                # attempted once and observed return clears custody before guard.
                def close():
                    identity=self.sidecar_identities[index]
                    a.require(identity is not None,'initialized original sidecar close status')
                    self.sidecar_close_attempted[index]=True;identity.close_attempted=True
                    try:return self.sidecar_streams[index].close()
                    except BaseException:identity.close_failed=True;raise
                checked(close,closed)
                self.sidecar_stages[index]=9
            except BaseException as error:primary=error;raise
            finally:
                readback=None
                if self.sidecar_streams[index] is not None and not self.sidecar_close_attempted[index]:
                    try:
                        def close():
                            identity=self.sidecar_identities[index]
                            a.require(identity is not None,'initialized original sidecar close status')
                            self.sidecar_close_attempted[index]=True;identity.close_attempted=True
                            try:return self.sidecar_streams[index].close()
                            except BaseException:identity.close_failed=True;raise
                        checked(close,closed)
                    except BaseException as error:
                        if primary is None:raise
                        primary.add_note('late sidecar unresolved close: '+type(error).__name__)
        finally:
            readback=None
            if primary is not None:cost.traceback.clear_frames(primary.__traceback__)
            scope.release()

    def disposition(self):
        if self.case not in a.CASE_NAMES or self.sidecars is None:return None
        return {'case':self.case,'slots':[{'index':index,'retained_bytes':0 if self.blocks[index] is None else len(self.blocks[index]),
          'reported_log_write':int(self.block_log_writes[index]),'attempted':bool(self.sidecar_attempted[index]),
          'stage':int(self.sidecar_stages[index]),'failed':bool(self.sidecar_failed[index]),
          'owned_stream':self.sidecar_streams[index] is not None,'sidecar_written':int(self.sidecar_written[index]),
          'sidecar_write_observed':bool(self.sidecar_write_observed[index]),
          'close_observed':bool(self.sidecar_close_observed[index]),
          'ref':None if self.sidecar_bytes[index]==0 else {'path':self.sidecars[index],
            'bytes':int(self.sidecar_bytes[index]),'sha256':self.sidecar_digests[index]},
          'durable':self.sidecar_stages[index]==10 and not self.sidecar_failed[index]}
          for index in range(2)]}
    def start(self):
        a.require(len(self.backend.read_fds)==2,'exact two original output pipes')
        for index,name in enumerate(('supervisor.stdout.log','supervisor.stderr.log')):
            try:
                self._guard()
                thread=threading.Thread(target=self.reader,args=(index,self.backend.read_fds[index],name),name=self.names[index],daemon=True)
                self.threads[index]=thread;self._guard();thread.start();self.started[index]=True;self._guard()
            except BaseException as error:self.errors[index]=error;raise
    def join_until(self,deadline):
        clock=self.backend.owner._clock
        a.require(deadline==self.backend.owner.deadline or deadline==self.backend.owner._tail_deadline,
          'join borrows only same authenticated original root or existing tail deadline')
        for thread in self.threads:
            if thread is not None and thread.ident is not None:thread.join(max(0,deadline-clock()))
        complete=all(thread is None or not thread.is_alive() for thread in self.threads)
        self.backend._logs_joined=complete
        a.require(clock()<deadline and complete,'original stream join unresolved at existing deadline')
        a.require(all(stream is None for stream in self.streams),'original owned log streams not closed')
        return complete
    def finish(self,deadline):
        self.join_until(deadline)
        for error in self.errors:
            if error is not None:raise error
        a.require(all(identity.capture_status=='COMPLETE' and identity.capture_error is None
          for identity in self.log_identities),'normal case requires complete original output diagnostics')
        a.require(all(block is None for block in self.blocks),'late/unwritten original blocks prevent pass')
        return {'read_bytes':self.read,'write_bytes':self.persisted,'operation_attempts':self.calls}


def observe_original_owner(backend,permit,case,directory,creation_plan):
    """Actual parent replacement; qualified backend primitives only.

    Existing same-object Budget/custody/permit/clock and the400 complete
    poll guards remain mandatory. No invocation of provider.run_owned or its
    log consumer survives this source path. Native evidence remains absent.
    """
    budget=permit.owner;clock=budget._clock;deadline=budget.deadline
    a.require(backend.owner is budget and deadline==permit.grant['physical_deadline_monotonic'],
      'same original owner and authenticated physical grant deadline')
    started=clock();terminal=None;identity=None
    try:
        budget._tick();backend.create_job();budget._tick()
        identity=backend.create_suspended(target_argv(permit,case),a.ROOT,creation_plan,permit=permit,case=case);budget._tick()
        publish_birth(permit,case,identity);budget._tick()
        # Retain the partial consumer before __init__/start can fail. Its
        # constructor is covered by the retained original log workspace.
        logs=object.__new__(OriginalOwnerLogs);backend._owned_logs=logs
        OriginalOwnerLogs.__init__(logs,backend,directory);logs.start();budget._tick()
        backend.resume();budget._tick()
        for _ in range(400):
            for error in logs.errors:
                if error is not None:raise error
            budget._tick();row=backend.poll();budget._tick()
            if row['supervisor_signaled'] and row['job_active_processes']==0:
                terminal=row;break
            time.sleep(min(.1,max(0,deadline-clock())))
        a.require(terminal is not None,'same original400 complete observation ceiling')
        output=logs.finish(deadline);budget._tick()
        terminal.update(same_retained_supervisor_birth=True,retained_supervisor_identity=identity,
          terminal_monotonic=clock(),log_accounting=output,native_qualified=False,execution_verified=False,
          all_owned_handles_closed=False,provider_wall_seconds=clock()-started)
        return terminal
    except BaseException as error:
        budget.mark_failed()
        # begin_final_tail is original source authentication: min(original
        # root deadline, first tail start+5). Never adds time to the root.
        try:budget.begin_final_tail()
        except BaseException as secondary:error.add_note('same original tail: '+type(secondary).__name__)
        try:owned_cleanup(backend,backend.stop,final=True)
        except BaseException as secondary:error.add_note('single original stop: '+type(secondary).__name__)
        raise
def close_checker(backend,budget,primary=None):
    first=None
    try:owned_cleanup(backend,backend.close,final=budget.fatal)
    except BaseException as error:
        first=error;budget.mark_failed()
        for action in (backend.release_deferred_handles,backend.stop,backend.close):
            try:owned_cleanup(backend,action,final=True)
            except BaseException as secondary:first.add_note('original checker cleanup: '+type(secondary).__name__)
    if first is not None:
        if primary is not None:primary.add_note('checker close: '+type(first).__name__)
        else:raise first
class OwnedLifecycle:
    """Retain original partial objects before constructor/return guards."""
    __slots__=('budget','custody','backends','roles','constructed','errors','count','lock','token','output_created','lease_created','lease_released','lease_status','token_match')
    def __init__(self,budget,custody):
        self.budget=budget;self.custody=custody
        self.backends=[None]*4;self.roles=[None]*4;self.constructed=[False]*4
        self.errors=[None]*4;self.count=0
        self.lock=a.OUTPUT/'OWNER_FIXTURE.lock';self.token=None
        self.output_created=False;self.lease_created=False;self.lease_released=False;self.lease_status='NOT_CREATED';self.token_match=None
    def construct(self,kind,role):
        a.require(self.count<4,'fixed four original backend lifetimes')
        ticket=self.budget.before('control',operation_attempts=1,metadata_bytes=64)
        slot=self.count;self.count+=1;self.roles[slot]=role
        obj=object.__new__(kind);self.backends[slot]=obj
        kind.__init__(obj);self.constructed[slot]=True
        ticket.finish();return obj
    def make_output(self):
        def create():
            a.OUTPUT.mkdir();self.output_created=True
        self.budget.run('control',{'operation_attempts':1},create)
        self.token=self.budget.run('control',{'operation_attempts':1,'metadata_bytes':128},lambda:os.urandom(32).hex())
        def lease():
            with self.lock.open('xb') as stream:
                self.lease_created=True;self.lease_status='OWNED_OPEN'
                a.require(stream.write(self.token.encode('ascii'))==64,'full same-token lease write')
                stream.flush();os.fsync(stream.fileno())
        self.budget.run('control',{'operation_attempts':3,'write_bytes':64},lease)
    def backends_closed(self):
        for index in range(self.count):
            obj=self.backends[index]
            if not hasattr(obj,'closure_observation'):return False
            logs=getattr(obj,'_owned_logs',None)
            if logs is not None and hasattr(logs,'blocks'):
                if any(block is not None and len(block)>0 for block in logs.blocks):return False
                if any(stream is not None for stream in logs.sidecar_streams):return False
            row=obj.closure_observation()
            if not row['all_owned_handles_closed'] or not row['closed_flag']:return False
            if row['job_created'] and not row['job_empty_observed']:return False
            if row['process_created'] and not row['process_terminal_observed']:return False
        return True
    def close_owned(self,primary):
        for index in range(self.count):
            obj=self.backends[index]
            if getattr(obj,'closed',False):continue
            for name in ('release_deferred_handles','stop'):
                try:owned_cleanup(obj,getattr(obj,name),final=True)
                except BaseException as error:
                    if self.errors[index] is None:self.errors[index]=type(error).__name__
                    primary.add_note('retained '+str(self.roles[index])+' cleanup '+name+': '+type(error).__name__)
            try:
                # One bounded join, followed by the existing source-owned
                # exact Job0/process wait verification. Unresolved streams or
                # a live Job retain original handles and lease for evidence.
                logs=getattr(obj,'_owned_logs',None)
                if logs is not None:logs.join_until(self.budget._tail_deadline)
                owned_cleanup(obj,obj.verify_job_empty,final=True)
                a.require(not obj._process_created or obj._process_terminal_observed,'same retained process exit before close')
                a.require(not obj._job_created or obj._job_empty_observed,'same retained Job0 before close')
                owned_cleanup(obj,obj.close,final=True)
            except BaseException as error:
                if self.errors[index] is None:self.errors[index]=type(error).__name__
                primary.add_note('retained '+str(self.roles[index])+' unresolved closure: '+type(error).__name__)
    def publish_late_blocks(self):
        candidate=None
        for index in range(self.count):
            logs=getattr(self.backends[index],'_owned_logs',None)
            if logs is None or not hasattr(logs,'blocks'):continue
            if any(block is not None and len(block)>0 for block in logs.blocks):
                a.require(candidate is None and self.roles[index] in a.CASE_NAMES,
                  'only current active case two original late blocks')
                candidate=logs
        if candidate is not None:candidate.publish_late_blocks()
    def release_lease(self,failed=False):
        if not self.lease_created:return
        def release():
            a.require(self.backends_closed() and self.custody._closed,'actual original Job/handles/pins close before lease release')
            with self.lock.open('rb') as stream:observed=stream.read(65)
            self.token_match=observed==self.token.encode('ascii')
            a.require(self.token_match,'same-token lease only; preserve mismatch')
            self.lock.unlink();self.lease_released=True;self.lease_status='RELEASED_SAME_TOKEN'
        if failed:self.budget.tail_run(2,release)
        else:self.budget.run('control',{'operation_attempts':2,'read_bytes':65,'metadata_bytes':1024},release)
    def failure_state(self):
        self.budget._tail_authority.authorize_data('lifecycle',self)
        rows=[]
        for index in range(self.count):
            obj=self.backends[index]
            observation=obj.closure_observation() if hasattr(obj,'closure_observation') else {
              'job_created':False,'job_empty_observed':False,'process_created':False,'process_terminal_observed':False,
              'all_owned_handles_closed':False,'closed_flag':False,'image_file_owned':False,'image_check_complete':False}
            rows.append({'role':self.roles[index],'construction_started':True,'construction_returned':self.constructed[index],
              **observation,'cleanup_errors':[] if self.errors[index] is None else [self.errors[index]],
              'observer':{'stop_selected':bool(getattr(obj,'_stop_started',False)),
                'stop_native_selected':bool(getattr(obj,'_stop_native_selected',False)),
                'stop_ffi_attempt_entered':bool(getattr(obj,'_stop_ffi_attempt_entered',False)),
                'stop_ffi_returned':bool(getattr(obj,'_stop_ffi_returned',False)),
                'stop_wrapper_finished':bool(getattr(obj,'_stop_wrapper_finished',False)),
                'stop_windows_error':getattr(obj,'_stop_windows_error',None),
                'stop_failed':getattr(obj,'_stop_error',None) is not None,
                'latest_native':{'api':getattr(obj,'_native_name',None),
                  'selected':bool(getattr(obj,'_native_selected',False)),
                  'ffi_attempt_entered':bool(getattr(obj,'_ffi_attempt_entered',False)),
                  'ffi_returned':bool(getattr(obj,'_ffi_returned',False)),
                  'afterguard_finished':bool(getattr(obj,'_native_afterguard_finished',False)),
                  'windows_error':getattr(obj,'_native_windows_error',None),
                  'error_type':None if getattr(obj,'_native_error',None) is None else type(obj._native_error).__name__,
                  'os_primitive_entry_observed':False},
                'latest_descriptor':{'selected':bool(getattr(obj,'_descriptor_selected',False)),
                  'call_attempt_entered':bool(getattr(obj,'_descriptor_entered',False)),
                  'returned':bool(getattr(obj,'_descriptor_returned',False)),
                  'afterguard_finished':bool(getattr(obj,'_descriptor_afterguard_finished',False)),
                  'error_type':None if getattr(obj,'_descriptor_error',None) is None else type(obj._descriptor_error).__name__},
                'descriptor_batch':None if not hasattr(obj,'_observation') else {
                  'selected_count':int(obj._observation.descriptor_selected_count),
                  'actual_attempts':int(obj._observation.descriptor_attempts),
                  'observed_returns':int(obj._observation.descriptor_return_count),
                  'attempt_mask':int(obj._observation.descriptor_attempt_mask),
                  'return_mask':int(obj._observation.descriptor_return_mask),
                  'afterguard_mask':int(obj._observation.descriptor_afterguard_mask),
                  'error_mask':int(obj._observation.descriptor_error_mask),
                  'tail_group_selected':bool(obj._observation.descriptor_tail_selected)},
                'log_consumer_retained':getattr(obj,'_owned_logs',None) is not None,
                'output_identities':None if getattr(obj,'_owned_logs',None) is None or not hasattr(obj._owned_logs,'log_identities') else {
                  'logs':[None if identity is None or identity.capture_status=='NOT_CAPTURED' else identity.value() for identity in obj._owned_logs.log_identities],
                  'sidecars':[None if identity is None or identity.capture_status=='NOT_CAPTURED' else identity.value() for identity in obj._owned_logs.sidecar_identities],
                  'secondary_close_errors':list(obj._owned_logs.secondary_close_errors)},
                'log_disposition':None if getattr(obj,'_owned_logs',None) is None or not hasattr(obj._owned_logs,'sidecar_attempted') else obj._owned_logs.disposition(),
                'log_readers_joined':bool(getattr(obj,'_logs_joined',False))}})
        return {'schema':'CE8-OWNED-LIFECYCLE-1','backends':rows,'custody_closed':bool(self.custody._closed),
          'owned_lease':{'created':self.lease_created,'released':self.lease_released,'status':self.lease_status,
            'path':str(self.lock),'token_match':self.token_match},'output_created':self.output_created}

def run_case(permit,budget,provider,base,case,lifecycle):
    permit.custody.require_live(before_info=lambda:budget.native('GetFileInformationByHandle'),
      before_namespace=lambda namespace:budget.before('control',operation_attempts=1,metadata_bytes=16384))
    a.require(case in a.CASE_NAMES and not budget._roles[case][0],'each exact case retained once')
    budget.before(role=case,**a.CASE_SELECTED) # immutable full subset; never refunded from small actual use
    budget.before(role='logs',operation_attempts=2048,read_bytes=8388608,write_bytes=8192,metadata_bytes=8192)
    directory=None
    for named,prepared_directory,paths,names,path_strings,sidecars in permit._static.creation_plan._log_inputs:
        if named==case:directory=prepared_directory;break
    a.require(directory is not None,'borrow exact PREP parent log directory')
    budget.run('control',{'operation_attempts':1},directory.mkdir)
    class ParentBackend(base):
        def call(self,name,*args,**kwargs):
            self.native_selection_started(name)
            try:
                cleanup=self.cleanup_mode();ticket=None
                if cleanup and name in ('WaitForSingleObject','QueryInformationJobObject'):
                    context=getattr(self,'_terminal_verification_context',None)
                    a.require(context is not None,'only source-owned original terminal verification context')
                    ticket=budget.verify_native(context,name,args,kwargs)
                else:
                    if cleanup:a.require(name in ('CloseHandle','TerminateJobObject','DeleteProcThreadAttributeList'),'fixed original cleanup API')
                    ticket=budget.native(name,cleanup=cleanup)
                self.native_selected(name)
            except BaseException as selection_error:
                self._native_error=selection_error
                raise
            try:
                a.require(self.calls<30000 or cleanup,'fixed parent native call ceiling')
                result=super().call(name,*args,**kwargs)
            except BaseException as primary:
                budget.mark_failed()
                if ticket is not None:
                    try:ticket.finish()
                    except BaseException as secondary:primary.add_note('original return guard: '+type(secondary).__name__)
                raise
            else:
                if ticket is not None:
                    try:ticket.finish()
                    except BaseException as guard_error:
                        self._native_error=guard_error;raise
            self._native_afterguard_finished=True
            return result
    backend=lifecycle.construct(ParentBackend,case);primary=None
    try:
        # No root environment dict/sort/join/buffer construction remains.
        terminal=observe_original_owner(backend,permit,case,directory,permit._static.creation_plan)
        a.require(terminal['supervisor_signaled'] is True and terminal['actual_supervisor_returncode']==0 and terminal['job_active_processes']==0
          and terminal['same_retained_supervisor_birth'] is True,'actual actor case exit and full descendant Job0')
        born,ended,kernel,user=provider.FileTime(),provider.FileTime(),provider.FileTime(),provider.FileTime()
        backend.call('GetProcessTimes',backend.process,c.byref(born),c.byref(ended),c.byref(kernel),c.byref(user))
        a.require(provider.filetime(born)==terminal['retained_supervisor_identity']['creation_filetime_100ns']
          and provider.filetime(ended)>=provider.filetime(born),'same retained actor case birth/exit')
        receipt_path=a.ACTOR_OUTPUT/case/'ACTOR_PROBE_CLOSED.json'
        budget.before(operation_attempts=1)
        a.require(receipt_path.is_file() and not receipt_path.is_symlink() and 0<receipt_path.stat().st_size<=16384,'bounded same-source case receipt')
        ticket=budget.before('control',operation_attempts=1,read_bytes=16385)
        with receipt_path.open('rb') as stream:raw=stream.read(16385)
        ticket.finish()
        a.require(len(raw)<=16384,'case receipt grew');value=a.strict(raw,owner=budget,maximum=16384);validate_probe(value,permit,case);child=validate_child_probe(value,permit,case)
        owned_cleanup(backend,backend.close);a.require(backend.closed,'case retained handles closed')
        result_ticket=budget.before('control',operation_attempts=1,metadata_bytes=1024)
        result={'case_name':case,'actor_receipt_ref':a.ref(receipt_path,raw,owner=budget),'actor_retained_terminal':terminal,
          'actor_exit_filetime_100ns':provider.filetime(ended),'parent_handles_closed':True,'target_selected':a.CASE_SELECTED,
          'actual_target_cost_ref':value.get('actual_cost_ref'),'actor_child_receipt':child}
        result_ticket.finish();return result
    except BaseException as error:
        primary=error;budget.mark_failed()
        try:budget.begin_final_tail()
        except BaseException as secondary:error.add_note('original final tail: '+type(secondary).__name__)
        raise
def main(permit=None):
    a.require(len(sys.argv)==1 and type(permit) is a.SourceIssuedFixturePermit,'fixed same-original-owner direct entry')
    permit.require();budget=permit.owner
    a.require(a.checked_owner(budget) is budget,'no second accounting owner or root reset')
    lifecycle=None
    def create_lifecycle():
        nonlocal lifecycle
        lifecycle=OwnedLifecycle(budget,permit.custody)
    try:
        budget.run('control',{'operation_attempts':1,'metadata_bytes':1024},create_lifecycle)
        lifecycle.make_output()
        provider,source,census=permit.prepared_modules
        base=source.backend_type(provider,max_members=8,owner=budget,census=census,creation_plan=permit._static.creation_plan)
        class CheckBackend(base):
            def call(self,name,*args,**kwargs):
                self.native_selection_started(name)
                try:
                    cleanup=self.cleanup_mode();ticket=None
                    if cleanup and name in ('WaitForSingleObject','QueryInformationJobObject'):
                        context=getattr(self,'_terminal_verification_context',None)
                        a.require(context is not None,'only source-owned original terminal verification context')
                        ticket=budget.verify_native(context,name,args,kwargs)
                    else:
                        if cleanup:a.require(name in ('CloseHandle','TerminateJobObject','DeleteProcThreadAttributeList'),'fixed original cleanup API')
                        ticket=budget.native(name,cleanup=cleanup)
                    self.native_selected(name)
                except BaseException as selection_error:
                    self._native_error=selection_error
                    raise
                try:result=super().call(name,*args,**kwargs)
                except BaseException as primary:
                    budget.mark_failed()
                    if ticket is not None:
                        try:ticket.finish()
                        except BaseException as secondary:primary.add_note('checker return guard: '+type(secondary).__name__)
                    raise
                if ticket is not None:
                    try:ticket.finish()
                    except BaseException as guard_error:
                        self._native_error=guard_error;raise
                self._native_afterguard_finished=True
                return result
        checker=lifecycle.construct(CheckBackend,'initial_checker')
        checker_primary=None
        try:
            # Fresh boot query selects the same one native attempt through
            # the ordinary retained call()/returnguard route, not a predebit.
            identity=checker.own_identity();a.require(identity['boot_identity']==permit.grant['physical_boot_identity'] and checker.available_ram()>=8589934592,'actual source boot and8GiB launch floor')
        except BaseException as error:checker_primary=error;budget.mark_failed();raise
        finally:close_checker(checker,budget,checker_primary)
        prepay_build=budget.before('control',operation_attempts=1,metadata_bytes=1024)
        prepay={'grant_ref':permit.grant_ref,'selected':a.BOUNDS,'partitions':a.partitions(),
          'case_names':list(a.CASE_NAMES),'target_manifest_ref':permit.packet['target_manifest_ref'],'models_loaded':0,'optimizer_updates':0,'refunds':0}
        prepay_build.finish();validate_bound_record(permit,'ROOT_PREPAYMENT.json',prepay)
        publish(budget,'ROOT_PREPAYMENT.json',prepay,maximum=4096)
        cases=[run_case(permit,budget,provider,base,case,lifecycle) for case in a.CASE_NAMES]
        checker=lifecycle.construct(CheckBackend,'final_checker')
        checker_primary=None
        try:own_io=checker.own_io()
        except BaseException as error:checker_primary=error;budget.mark_failed();raise
        finally:close_checker(checker,budget,checker_primary)
        root_read=sum(row['actor_retained_terminal']['job_cumulative_io']['ReadTransferCount'] for row in cases)
        root_write=sum(row['actor_retained_terminal']['job_cumulative_io']['WriteTransferCount'] for row in cases)
        a.require(own_io['ReadTransferCount']+root_read+65536<=a.BOUNDS['read_bytes'] and own_io['WriteTransferCount']+root_write+65536<=a.BOUNDS['write_bytes'],'physical own plus disjoint Job IO once')
        # Normal case disposals above use regular paid API calls. The one final
        # tail begins here, after BOTH cases, and never restarts per pin.
        budget.begin_final_tail()
        pin_release=a.release_original_pins(permit.custody,budget.pin_close)
        lifecycle.release_lease()
        budget.final_envelope(a.OUTPUT/'CLOSED_ACCOUNTING.json',lambda state:{'schema':'CE8-ACTOR-PARENT-TWO-CASE-LOGICAL-CLOSED-6','grant_ref':permit.grant_ref,
          'owner_state':state,'controller_retained_identity':identity,'cases':cases,'pending':[],'failed':[],'refunds':0,'original_runtime_pin_release':pin_release,
          'owned_lease_released':True,'physical_parent_IO':own_io,'physical_disjoint_Job_read':root_read,'physical_disjoint_Job_write':root_write,
          'physical_IO_not_additional_logical_debit':True,'model_loads':0,'optimizer_updates':0,'whole_model_qualification':False,
          'unknown':['ORIGINAL_TOOL_PARENT_FINAL_EXIT_AND_LAST_PUBLICATION_IO_AWAIT_EXTERNAL_OBSERVATION'],
          'physical_deadline_monotonic':budget.deadline},maximum=24576,
          validator=lambda value:permit._static.envelopes_module.validate(permit._static.envelopes['schemas']['CLOSED_ACCOUNTING.json'],value))
        budget.finish();return 0
    except BaseException as error:
        return fail_owned(permit,lifecycle,error)

def fail_owned(permit,lifecycle,error):
    budget=permit.owner;budget.mark_failed()
    try:budget.begin_final_tail()
    except BaseException as secondary:error.add_note('original tail: '+type(secondary).__name__)
    if lifecycle is not None:
        lifecycle.close_owned(error)
        try:lifecycle.publish_late_blocks()
        except BaseException as secondary:error.add_note('late block disposition: '+type(secondary).__name__)
    if not permit.custody._closed and not a._PIN_RELEASE_ATTEMPTED:
        try:a.release_original_pins(permit.custody,budget.pin_close)
        except BaseException as secondary:error.add_note('original pins: '+type(secondary).__name__)
    elif not permit.custody._closed:error.add_note('original pin close plan already attempted; unresolved pins retained')
    if lifecycle is not None and lifecycle.lease_created and not lifecycle.lease_released:
        try:lifecycle.release_lease(failed=True)
        except BaseException as secondary:
            lifecycle.lease_status='UNRESOLVED_PRESERVED'
            error.add_note('same-token lease preserved: '+type(secondary).__name__)
    try:
        budget.failure_record(error,a.ROOT/'FAILED_QUALIFICATION.json',lifecycle=None if lifecycle is None else lifecycle.failure_state,
          envelope=permit._static.failure_module)
    except BaseException as secondary:error.add_note('durable failure publication: '+type(secondary).__name__)
    raise error
if __name__=='__main__':raise SystemExit(main())
