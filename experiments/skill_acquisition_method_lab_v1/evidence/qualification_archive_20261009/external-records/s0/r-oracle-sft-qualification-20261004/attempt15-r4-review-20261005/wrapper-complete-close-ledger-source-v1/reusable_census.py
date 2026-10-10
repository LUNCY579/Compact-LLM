"""Source-bound fixed Job PID observation storage; no Native API at import.

Two raw query buffers are reused. Only failure/publication creates owned
evidence. A PID view expires at the next query and never grants process rights.
"""
import ctypes as c
import struct

HEADER_BYTES=8
MAX_OBSERVATION_SLOTS=128
MAX_QUERY_ATTEMPTS=2
MAX_BUFFER_BYTES=8+128*8
_HEADER=struct.Struct('<II')
_PID32=struct.Struct('<I')
_PID64=struct.Struct('<Q')

def require(value,message):
    if value is not True:raise RuntimeError(message)

class QuerySlot:
    __slots__=('buffer','returned','size','capacity','pointer_bytes','guard','captured','ok','error','returned_bytes','assigned','count','decoded','complete','retry_capacity','authority_accepted')
    def __init__(self):
        self.buffer=c.create_string_buffer(MAX_BUFFER_BYTES);self.returned=c.c_uint32()
        self.reset(0,0,0)
    def reset(self,size,pointer_bytes,guard):
        self.size=size;self.pointer_bytes=pointer_bytes;self.guard=guard
        self.capacity=(size-8)//pointer_bytes if pointer_bytes else 0
        self.captured=False;self.ok=False;self.error=0;self.returned_bytes=0;self.returned.value=0
        self.assigned=0;self.count=0;self.decoded=False;self.complete=False;self.retry_capacity=0;self.authority_accepted=False
        if size:c.memset(self.buffer,0,size)
    def record_native_return(self,ok,error,returned_bytes):
        # Backend's actual native return hook calls this before its afterguard.
        require(not self.captured,'one original raw query return')
        require(type(ok) in (int,bool) and type(error) is int and error>=0
          and type(returned_bytes) is int and 0<=returned_bytes<=0xffffffff,'actual raw return types')
        self.ok=bool(ok);self.error=error;self.returned_bytes=returned_bytes;self.captured=True

class PIDView:
    __slots__=('_storage','_generation')
    def __init__(self,storage,generation):
        object.__setattr__(self,'_storage',storage);object.__setattr__(self,'_generation',generation)
    def __setattr__(self,*args):raise RuntimeError('immutable current observation token')
    def _live(self):
        s=self._storage
        require(s.generation==self._generation and s.accepted and not s.failed and not s.in_query,
          'PID observation expired or refused; no mutable authority escapes')
        return s
    def __len__(self):return self._live().pid_count
    def __getitem__(self,index):
        s=self._live();require(type(index) is int,'bounded PID index only')
        if index<0:index+=s.pid_count
        if not 0<=index<s.pid_count:raise IndexError(index)
        return int(s.pids[index])
    def __iter__(self):
        index=0
        while index<len(self):
            yield self[index];index+=1
    def __contains__(self,value):
        for pid in self:
            if pid==value:return True
        return False

class CensusStorage:
    __slots__=('backend','guard','pointer_bytes','slots','pids','generation','attempts','pid_count','accepted','failed','in_query')
    def __init__(self,backend,guard):
        require(backend.pointer_bytes in (4,8) and type(guard) is int and 1<=guard<=10,'source-fixed PID pointer/authority guard')
        self.backend=backend;self.guard=guard;self.pointer_bytes=backend.pointer_bytes
        self.slots=(QuerySlot(),QuerySlot());self.pids=(c.c_uint64*128)()
        self.generation=0;self.attempts=0;self.pid_count=0;self.accepted=False;self.failed=False;self.in_query=False
    def _query(self,backend,guard):
        require(backend is self.backend and guard==self.guard and backend.pointer_bytes==self.pointer_bytes,
          'same source backend and unchanged census guard')
        require(not self.failed and not self.in_query,'no census reentry or failed-observation revival')
        self.generation+=1;self.attempts=0;self.pid_count=0;self.accepted=False;self.in_query=True
        capacity=guard;decoder=_PID64 if self.pointer_bytes==8 else _PID32
        try:
            for attempt in range(MAX_QUERY_ATTEMPTS):
                slot=self.slots[attempt];size=HEADER_BYTES+capacity*self.pointer_bytes
                require(8<=size<=MAX_BUFFER_BYTES,'bounded raw PID snapshot')
                slot.reset(size,self.pointer_bytes,guard);self.attempts=attempt+1
                backend.raw_job_pid_query_into(size,slot.buffer,slot.returned,slot)
                require(slot.captured,'raw native return not retained before boundary')
                assigned,count=_HEADER.unpack_from(slot.buffer)
                slot.assigned=assigned;slot.count=count;slot.decoded=True
                if not slot.ok:
                    require(slot.error==234,'Job census failed with recorded native error: '+str(slot.error))
                    require(assigned<=128 and count<=capacity,'Job census sizing data exceeds observation bound')
                    capacity=max(capacity+1,assigned,2*capacity);slot.retry_capacity=capacity
                    require(capacity<=128,'bounded Job observation exhausted')
                    continue
                require(count==assigned and count<=capacity,'complete Job census header does not fit raw observation')
                require(8+count*self.pointer_bytes<=slot.returned_bytes<=size,
                  'successful Job census returned extent does not cover complete PID rows')
                slot.complete=True
                require(count<=guard,'complete Job census violates unchanged process guard')
                for index in range(count):
                    pid=decoder.unpack_from(slot.buffer,8+index*self.pointer_bytes)[0]
                    require(0<pid<=0xffffffff,'complete Job PID identities invalid/duplicated')
                    for previous in range(index):
                        require(self.pids[previous]!=pid,'complete Job PID identities invalid/duplicated')
                    self.pids[index]=pid
                self.pid_count=count;slot.authority_accepted=True;self.accepted=True;self.in_query=False
                return None
            raise RuntimeError('Job census incomplete after one bounded retry; evidence retained')
        except BaseException as error:
            self.failed=True;self.in_query=False
            error.__dict__['_ce8_census_storage']=self
            raise
    def query(self,backend,guard):
        self._query(backend,guard)
        return PIDView(self,self.generation)
    def query_contains(self,backend,guard,pid):
        require(type(pid) is int and 0<pid<=0xffffffff,'source-validated resampling PID')
        self._query(backend,guard)
        # All query predicates have succeeded. The bool singleton never lends
        # mutable PID storage or allocates a fresh observation token.
        for index in range(self.pid_count):
            if self.pids[index]==pid:return True
        return False
    def snapshot(self):
        self.backend.owner._tail_authority.authorize_data('snapshot',self)
        # Caller owns/prices this bounded allocation before publication/refusal.
        rows=[]
        for index in range(self.attempts):
            self.backend.owner._tick(allow_failed=self.backend.owner._tail_work_active is not None)
            slot=self.slots[index]
            if slot.captured:
                assigned,count=_HEADER.unpack_from(slot.buffer)
            else:assigned=count=None
            row={'buffer_bytes':slot.size,'pointer_bytes':slot.pointer_bytes,'capacity':slot.capacity,
              'api_success':slot.ok if slot.captured else None,'actual_windows_error':slot.error if slot.captured else None,
              'api_returned_bytes':slot.returned_bytes if slot.captured else None,'process_guard':slot.guard,
              'assigned_header':assigned,'returned_count_header':count,'native_return_observed':slot.captured,
              'raw_buffer_hex':memoryview(slot.buffer).cast('B')[:slot.size].hex(),'complete':slot.complete,
              'authority_accepted':slot.authority_accepted}
            if slot.retry_capacity:row['retry_capacity']=slot.retry_capacity
            if slot.captured and slot.ok and slot.complete and count is not None and count<=slot.capacity:
                decoder=_PID64 if slot.pointer_bytes==8 else _PID32
                pids=[decoder.unpack_from(slot.buffer,8+i*slot.pointer_bytes)[0] for i in range(count)]
                duplicates=0
                for current in range(len(pids)):
                    self.backend.owner._tick(allow_failed=self.backend.owner._tail_work_active is not None)
                    if any(pids[previous]==pids[current] for previous in range(current)):duplicates+=1
                row.update(observed_pid_entries=pids,observed_unique_pid_count=len(pids)-duplicates,observed_duplicate_entries=duplicates)
                # Snapshot data is immutable after production. Both JSON
                # fields retain the same complete validated list; no duplicate
                # accepted-PID container is needed to preserve canonical data.
                if slot.authority_accepted:row['pids']=pids
            rows.append(row)
        self.backend.owner._tick(allow_failed=self.backend.owner._tail_work_active is not None)
        return rows

def prepare_storage(backend,guard):
    require(getattr(backend,'_reusable_census',None) is None,'one prepaid census storage constructor')
    value=CensusStorage(backend,guard);backend._reusable_census=value;return value
def query_complete(backend,guard):
    value=getattr(backend,'_reusable_census',None)
    require(type(value) is CensusStorage,'prepaid fixed census storage required')
    return value.query(backend,guard)
def query_contains(backend,guard,pid):
    value=getattr(backend,'_reusable_census',None)
    require(type(value) is CensusStorage,'prepaid fixed census storage required')
    return value.query_contains(backend,guard,pid)
def snapshot_evidence(backend):
    value=getattr(backend,'_reusable_census',None)
    require(type(value) is CensusStorage and not value.in_query,'source-owned nonrunning census evidence')
    return value.snapshot()
