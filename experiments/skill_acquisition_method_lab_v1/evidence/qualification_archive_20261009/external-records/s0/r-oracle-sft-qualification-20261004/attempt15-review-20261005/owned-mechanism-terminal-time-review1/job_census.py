"""Bounded raw Job PID-list decoding; observation never increases authority."""
import struct

HEADER_BYTES=8
MAX_OBSERVATION_SLOTS=128
MAX_QUERY_ATTEMPTS=2


def decode_snapshot(blob,pointer_bytes,guard,*,succeeded,error,returned_bytes):
    if pointer_bytes not in (4,8) or type(guard) is not int or not 1<=guard<=10:
        raise RuntimeError('source-fixed PID pointer/authority guard')
    if type(blob) is not bytes or len(blob)<8 or len(blob)>8+128*pointer_bytes:
        raise RuntimeError('bounded raw PID snapshot')
    assigned,count=struct.unpack_from('<II',blob)
    capacity=(len(blob)-8)//pointer_bytes
    row={'buffer_bytes':len(blob),'pointer_bytes':pointer_bytes,'capacity':capacity,
         'assigned_header':assigned,'returned_count_header':count,'api_success':succeeded,
         'actual_windows_error':error,'api_returned_bytes':returned_bytes,'process_guard':guard}
    if not succeeded:
        if error!=234:
            raise RuntimeError('Job census failed with recorded native error: '+str(error))
        if assigned>128 or count>capacity:
            raise RuntimeError('Job census sizing data exceeds observation bound')
        # Return sizing evidence only; no membership accepted from partial rows.
        row['complete']=False
        row['retry_capacity']=max(capacity+1,assigned,2*capacity)
        if row['retry_capacity']>128:
            raise RuntimeError('bounded Job observation exhausted')
        return row
    if count!=assigned or count>capacity:
        raise RuntimeError('complete Job census header does not fit raw observation')
    if type(returned_bytes) is not int or not 8+count*pointer_bytes<=returned_bytes<=len(blob):
        raise RuntimeError('successful Job census returned extent does not cover complete PID rows')
    width='<Q' if pointer_bytes==8 else '<I'
    pids=[struct.unpack_from(width,blob,8+i*pointer_bytes)[0] for i in range(count)]
    row.update(complete=True,observed_pid_entries=pids,observed_unique_pid_count=len(set(pids)),
               observed_duplicate_entries=len(pids)-len(set(pids)),authority_accepted=False)
    if count>guard:
        rejected=RuntimeError('complete Job census violates unchanged process guard')
        rejected.__dict__['_ce8_job_census_evidence']=[row]
        raise rejected
    if any(not 0<pid<=0xffffffff for pid in pids) or len(set(pids))!=len(pids):
        rejected=RuntimeError('complete Job PID identities invalid/duplicated')
        rejected.__dict__['_ce8_job_census_evidence']=[row]
        raise rejected
    row.update(pids=pids,authority_accepted=True)
    return row


def query_complete(backend,guard):
    capacity=guard
    evidence=[]
    for attempt in range(2):
        blob,ok,error,returned=backend.raw_job_pid_query(8+capacity*backend.pointer_bytes)
        # Preserve raw result before ANY rejection, including second-call
        # errors or malformed headers. No validated membership is implied.
        raw={'buffer_bytes':len(blob),'pointer_bytes':backend.pointer_bytes,'capacity':capacity,
             'api_success':ok,'actual_windows_error':error,'api_returned_bytes':returned,'process_guard':guard}
        if len(blob)>=8:
            raw['assigned_header'],raw['returned_count_header']=struct.unpack_from('<II',blob)
        evidence.append(raw)
        backend.last_job_census_evidence=list(evidence)
        try:row=decode_snapshot(blob,backend.pointer_bytes,guard,succeeded=ok,error=error,returned_bytes=returned)
        except BaseException as rejected:
            detailed=rejected.__dict__.get('_ce8_job_census_evidence')
            if detailed:evidence[-1].update(detailed[-1])
            backend.last_job_census_evidence=list(evidence)
            raise
        evidence[-1].update(row)
        backend.last_job_census_evidence=list(evidence)
        if row['complete']:
            return row['pids']
        if attempt==0:
            capacity=row['retry_capacity']
    raise RuntimeError('Job census incomplete after one bounded retry; evidence retained')
