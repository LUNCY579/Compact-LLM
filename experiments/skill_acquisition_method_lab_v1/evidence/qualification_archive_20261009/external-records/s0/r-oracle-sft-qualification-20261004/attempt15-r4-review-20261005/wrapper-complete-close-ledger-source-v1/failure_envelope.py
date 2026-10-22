"""Finite durable failure value builder. No IO, native API, or authority at import.

Caller selects its entire original failure envelope BEFORE calling build.
Raw query observations are lossless; diagnostic text is explicitly bounded.
"""
SCHEMA='CE8-DURABLE-OWNED-FAILURE-1'
from pathlib import Path
import math
EXPECTED_OUTPUT_ROOT=Path(__file__).resolve().parent/'native-one-attempt'
EXPECTED_LATE_PATHS={case:tuple(str(EXPECTED_OUTPUT_ROOT/('parent_'+case)/('late-block-'+str(index)+'.bin'))
  for index in range(2)) for case in ('normal_exit','timeout')}
MAXIMUM=32768
WORKSPACE=32768
ATTEMPTS=8
MAX_RAW_BYTES=1032
ROW_KEYS={'buffer_bytes','pointer_bytes','capacity','api_success','actual_windows_error',
 'api_returned_bytes','process_guard','assigned_header','returned_count_header',
 'native_return_observed','raw_buffer_hex','complete','authority_accepted'}
OPTIONAL_ROW_KEYS={'retry_capacity','observed_pid_entries','observed_unique_pid_count',
 'observed_duplicate_entries','pids'}
BACKEND_KEYS={'role','construction_started','construction_returned','job_created','process_created',
 'job_empty_observed','process_terminal_observed','all_owned_handles_closed','closed_flag','cleanup_errors','observer',
 'image_file_owned','image_check_complete'}
LIFECYCLE_KEYS={'schema','backends','custody_closed','owned_lease','output_created'}
LEASE_KEYS={'created','released','status','path','token_match'}

def require(value,message):
    if value is not True:raise RuntimeError(message)
def text(value,maximum):
    require(type(value) is str and len(value)<=maximum and value.isascii()
      and sum(2 if ch in ('"','\\','\b','\f','\n','\r','\t') else 6 if ord(ch)<32 else 1 for ch in value)<=maximum,
      'bounded source failure ASCII JSON-content extent')
    return value
def integer(value,maximum=0xffffffff,nullable=False):
    require((nullable and value is None) or (type(value) is int and 0<=value<=maximum),'bounded source failure integer')
    return value
def boolean(value,nullable=False):
    require(type(value) is bool or (nullable and value is None),'source failure boolean')
    return value
def rows(value,*,borrow=False):
    if borrow:
        require(type(value) is list and len(value)<=2,'original raw query cardinality before allocation')
        for row in value:
            require(type(row) is dict and ROW_KEYS<=set(row)<=ROW_KEYS|OPTIONAL_ROW_KEYS,'closed original row keys')
            size=integer(row['buffer_bytes'],MAX_RAW_BYTES)
            require(size>=8 and row['pointer_bytes'] in (4,8),'fixed original row ABI')
            raw=text(row['raw_buffer_hex'],2064)
            require(len(raw)==size*2 and all(ch in '0123456789abcdef' for ch in raw),'lossless raw hex domain')
            for key in ('capacity','process_guard'):integer(row[key],128)
            for key in ('actual_windows_error','api_returned_bytes','assigned_header','returned_count_header'):integer(row[key],nullable=True)
            for key in ('native_return_observed','complete','authority_accepted'):boolean(row[key])
            boolean(row['api_success'],nullable=True)
            for key in ('retry_capacity','observed_unique_pid_count','observed_duplicate_entries'):
                if key in row:integer(row[key],256 if key=='retry_capacity' else 128)
            for key in ('observed_pid_entries','pids'):
                if key in row:
                    require(type(row[key]) is list and len(row[key])<=128,'original descriptive PID slots')
                    for pid in row[key]:integer(pid,0xffffffffffffffff)
        return value
    require(type(value) is list and len(value)<=2,'only two original raw query attempts')
    result=[]
    for row in value:
        require(type(row) is dict and ROW_KEYS<=set(row)<=ROW_KEYS|OPTIONAL_ROW_KEYS,'closed raw query row schema')
        size=integer(row['buffer_bytes'],MAX_RAW_BYTES)
        require(size>=8 and row['pointer_bytes'] in (4,8),'bounded original query shape')
        raw=text(row['raw_buffer_hex'],MAX_RAW_BYTES*2)
        require(len(raw)==2*size and not set(raw)-set('0123456789abcdef'),'lossless original raw query bytes')
        out={'raw_buffer_hex':raw,'buffer_bytes':size,'pointer_bytes':row['pointer_bytes']}
        for key in ('capacity','process_guard'):out[key]=integer(row[key],128)
        for key in ('actual_windows_error','api_returned_bytes','assigned_header','returned_count_header'):
            out[key]=integer(row[key],nullable=True)
        for key in ('native_return_observed','complete','authority_accepted'):out[key]=boolean(row[key])
        out['api_success']=boolean(row['api_success'],nullable=True)
        for key in ('retry_capacity','observed_unique_pid_count','observed_duplicate_entries'):
            # Retry256 is descriptive evidence of a refused proposed next
            # capacity. It never increases the actual buffer/authority128.
            if key in row:out[key]=integer(row[key],256 if key=='retry_capacity' else 128)
        for key in ('observed_pid_entries','pids'):
            if key in row:
                require(type(row[key]) is list and len(row[key])<=128,'bounded original PID evidence')
                out[key]=[integer(pid,0xffffffffffffffff) for pid in row[key]]
        result.append(out)
    return value if borrow else result
def log_disposition(value,role,*,borrow=False):
    if value is None:return None
    if borrow:
        require(role in EXPECTED_LATE_PATHS and type(value) is dict and set(value)=={'case','slots'}
          and value['case']==role and type(value['slots']) is list and len(value['slots'])==2,'exact original case slots')
        keys={'index','retained_bytes','reported_log_write','attempted','stage','failed','owned_stream',
          'sidecar_written','sidecar_write_observed','ref','durable','close_observed'}
        for index,row in enumerate(value['slots']):
            require(type(row) is dict and set(row)==keys and row['index']==index,'original fixed slot shape')
            integer(row['retained_bytes'],3072);integer(row['sidecar_written'],3072);integer(row['stage'],10)
            require(type(row['reported_log_write']) is int and -2<=row['reported_log_write']<=3072,'reported original write domain')
            for key in ('attempted','failed','owned_stream','durable','close_observed','sidecar_write_observed'):boolean(row[key])
            ref=row['ref']
            if ref is not None:
                require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and ref['path']==EXPECTED_LATE_PATHS[role][index],'exact original sidecar path')
                text(ref['path'],1024);integer(ref['bytes'],3072)
                require(ref['sha256'] is None or (type(ref['sha256']) is str and len(ref['sha256'])==64
                  and all(ch in '0123456789abcdef' for ch in ref['sha256'])),'original digest domain')
            require(not row['durable'] or (row['attempted'] and not row['failed'] and not row['owned_stream']
              and row['close_observed'] and row['stage']==10 and row['retained_bytes']==0 and ref is not None
              and ref['bytes']>0 and row['sidecar_write_observed'] and row['sidecar_written']==ref['bytes']
              and ref['sha256'] is not None),'original complete durability predicate')
        return value
    require(role in EXPECTED_LATE_PATHS and type(value) is dict and set(value)=={'case','slots'}
      and value['case']==role and type(value['slots']) is list and len(value['slots'])==2,
      'only exact current-case two original late slots')
    out={'case':role,'slots':[]}
    keys={'index','retained_bytes','reported_log_write','attempted','stage','failed','owned_stream',
      'sidecar_written','sidecar_write_observed','ref','durable','close_observed'}
    for index,row in enumerate(value['slots']):
        require(type(row) is dict and set(row)==keys and row['index']==index,'closed ordered late slot schema')
        item={'index':index,'retained_bytes':integer(row['retained_bytes'],3072),
          'sidecar_written':integer(row['sidecar_written'],3072),'stage':integer(row['stage'],10)}
        count=row['reported_log_write']
        require(type(count) is int and -2<=count<=3072,'exact reported log write or unknown/invalid sentinel')
        item['reported_log_write']=count
        for key in ('attempted','failed','owned_stream','durable','close_observed','sidecar_write_observed'):item[key]=boolean(row[key])
        ref=row['ref']
        if ref is None:item['ref']=None
        else:
            require(type(ref) is dict and set(ref)=={'path','bytes','sha256'}
              and ref['path']==EXPECTED_LATE_PATHS[role][index],'only exact registered create-only late path')
            digest=ref['sha256']
            require(digest is None or (type(digest) is str and len(digest)==64
              and not set(digest)-set('0123456789abcdef')),'bounded actual late readback digest')
            item['ref']={'path':text(ref['path'],1024),'bytes':integer(ref['bytes'],3072),'sha256':digest}
        require(not item['durable'] or (item['attempted'] and not item['failed'] and not item['owned_stream']
          and item['close_observed'] and item['stage']==10 and item['retained_bytes']==0 and item['ref'] is not None
          and item['ref']['bytes']>0 and item['sidecar_write_observed'] and item['sidecar_written']==item['ref']['bytes']
          and item['ref']['sha256'] is not None),'complete exact guarded sidecar receipt before durability claim')
        out['slots'].append(item)
    return value if borrow else out

def output_identities(value):
    if value is None:return None
    require(type(value) is dict and set(value)=={'logs','sidecars','secondary_close_errors'},'fixed original output diagnostic groups')
    require(type(value['secondary_close_errors']) is list and len(value['secondary_close_errors'])==2,'two fixed secondary close slots')
    for item in value['secondary_close_errors']:
        if item is not None:text(item,128)
    for group in ('logs','sidecars'):
        rows=value[group]
        require(type(rows) is list and len(rows)==2,'two original diagnostic slots per group')
        for item in rows:
            if item is None:continue
            require(type(item) is dict and set(item)=={'fd','handle','capture_status','capture_error',
              'close_attempted','close_returned','close_failed','terminal_status','diagnostic_only'},
              'closed partial output descriptor diagnostic schema')
            integer(item['fd'],0x7fffffff,nullable=True)
            require(item['handle'] is None or (type(item['handle']) is int and -(1<<63)<=item['handle']<(1<<64)),
              'bounded original HANDLE or missing capture')
            require(item['capture_status'] in ('NOT_CAPTURED','FD_CAPTURED','COMPLETE','PARTIAL_FAILED'),
              'closed original capture status')
            if item['capture_error'] is not None:text(item['capture_error'],128)
            require(item['capture_status']!='COMPLETE' or (item['fd'] is not None and item['handle'] is not None
              and item['capture_error'] is None),'complete diagnostic capture shape')
            require(item['capture_status']!='PARTIAL_FAILED' or item['capture_error'] is not None,
              'failed diagnostic capture retains error')
            for key in ('close_attempted','close_returned','close_failed','diagnostic_only'):boolean(item[key])
            require(item['diagnostic_only'] is True and item['terminal_status']==
              ('OBSERVED_CLOSED' if item['close_returned'] else 'UNKNOWN' if item['close_attempted'] else 'NOT_ATTEMPTED'),
              'numeric identity does not prove live ownership or allow retry')
            require(not item['close_returned'] or (item['close_attempted'] and not item['close_failed']),
              'observed return requires actual attempt without close failure')
            require(not item['close_failed'] or (item['close_attempted'] and not item['close_returned']),
              'failed close has unknown terminal status and actual attempt')
    return value

def lifecycle_value(value,*,borrow=False):
    if value is None:
        return {'schema':'CE8-OWNED-LIFECYCLE-1','backends':[],
          'custody_closed':False,'owned_lease':{'created':False,'released':False,
          'status':'NOT_OBSERVED','path':'','token_match':None},'output_created':False}
    if borrow:
        require(type(value) is dict and set(value)==LIFECYCLE_KEYS and value['schema']=='CE8-OWNED-LIFECYCLE-1'
          and type(value['backends']) is list and len(value['backends'])<=4,'complete original generated lifecycle')
        boolean(value['custody_closed']);boolean(value['output_created'])
        roles=('initial_checker','normal_exit','timeout','final_checker')
        for slot,row in enumerate(value['backends']):
            require(type(row) is dict and set(row)==BACKEND_KEYS and row['role']==roles[slot],'exact original ordered backend')
            for key in BACKEND_KEYS-{'role','cleanup_errors','observer'}:boolean(row[key])
            errors=row['cleanup_errors'];require(type(errors) is list and len(errors)<=1,'one actual stored cleanup error per original backend')
            for name in errors:text(name,128)
            ob=row['observer'];require(type(ob) is dict and set(ob)=={'stop_selected','stop_native_selected','stop_ffi_attempt_entered','stop_ffi_returned',
              'stop_wrapper_finished','stop_windows_error','stop_failed','latest_native','latest_descriptor','descriptor_batch',
              'log_consumer_retained','log_readers_joined','log_disposition','output_identities'},'closed original observer')
            for key in ('stop_selected','stop_native_selected','stop_ffi_attempt_entered','stop_ffi_returned','stop_wrapper_finished',
              'stop_failed','log_consumer_retained','log_readers_joined'):boolean(ob[key])
            integer(ob['stop_windows_error'],nullable=True);log_disposition(ob['log_disposition'],row['role'],borrow=True)
            output_identities(ob['output_identities'])
            native=ob['latest_native'];require(type(native) is dict and set(native)=={'api','selected','ffi_attempt_entered','ffi_returned',
              'afterguard_finished','windows_error','error_type','os_primitive_entry_observed'} and native['os_primitive_entry_observed'] is False,'original native observation shape')
            for key in ('selected','ffi_attempt_entered','ffi_returned','afterguard_finished','os_primitive_entry_observed'):boolean(native[key])
            integer(native['windows_error'],nullable=True)
            for key in ('api','error_type'):
                if native[key] is not None:text(native[key],128)
            desc=ob['latest_descriptor'];require(type(desc) is dict and set(desc)=={'selected','call_attempt_entered','returned','afterguard_finished','error_type'},'original descriptor shape')
            for key in ('selected','call_attempt_entered','returned','afterguard_finished'):boolean(desc[key])
            if desc['error_type'] is not None:text(desc['error_type'],128)
            batch=ob['descriptor_batch']
            if batch is not None:
                require(type(batch) is dict and set(batch)=={'selected_count','actual_attempts','observed_returns','attempt_mask','return_mask','afterguard_mask','error_mask','tail_group_selected'},'original batch shape')
                for key in ('selected_count','actual_attempts','observed_returns','attempt_mask','return_mask','afterguard_mask','error_mask'):
                    integer(batch[key],5 if key in ('actual_attempts','observed_returns') else 31)
                boolean(batch['tail_group_selected'])
        lease=value['owned_lease'];require(type(lease) is dict and set(lease)==LEASE_KEYS,'original lease shape')
        for key in ('created','released'):boolean(lease[key])
        boolean(lease['token_match'],nullable=True);text(lease['status'],64);text(lease['path'],1024)
        require(not lease['released'] or (lease['created'] and lease['token_match'] is True),'same original lease release')
        return value
    require(type(value) is dict and LIFECYCLE_KEYS<=set(value)<=LIFECYCLE_KEYS|{'census_evidence'},'closed owned lifecycle schema')
    require(value['schema']=='CE8-OWNED-LIFECYCLE-1' and type(value['backends']) is list and len(value['backends'])<=4,'four retained backend slots')
    out={'schema':value['schema'],'backends':[],'custody_closed':boolean(value['custody_closed']),
      'output_created':boolean(value['output_created'])}
    allowed_roles=('initial_checker','normal_exit','timeout','final_checker')
    for slot,row in enumerate(value['backends']):
        require(type(row) is dict and set(row)==BACKEND_KEYS,'closed backend closure schema')
        require(row['role']==allowed_roles[slot],'only exact original ordered backend lifetimes')
        item={'role':text(row['role'],64)}
        for key in BACKEND_KEYS-{'role','cleanup_errors','observer'}:item[key]=boolean(row[key])
        observer=row['observer']
        require(type(observer) is dict and set(observer)=={'stop_selected','stop_native_selected','stop_ffi_attempt_entered','stop_ffi_returned',
          'stop_wrapper_finished','stop_windows_error','stop_failed','latest_native','latest_descriptor',
          'descriptor_batch','log_consumer_retained','log_readers_joined','log_disposition','output_identities'},'closed original observer evidence')
        item['observer']={key:boolean(observer[key]) for key in ('stop_selected','stop_native_selected','stop_ffi_attempt_entered',
          'stop_ffi_returned','stop_wrapper_finished','stop_failed','log_consumer_retained','log_readers_joined')}
        item['observer']['stop_windows_error']=integer(observer['stop_windows_error'],nullable=True)
        item['observer']['output_identities']=output_identities(observer['output_identities'])
        item['observer']['log_disposition']=log_disposition(observer['log_disposition'],row['role'],borrow=borrow)
        native=observer['latest_native']
        require(type(native) is dict and set(native)=={'api','selected','ffi_attempt_entered','ffi_returned',
          'afterguard_finished','windows_error','error_type','os_primitive_entry_observed'},'closed latest native attempt evidence')
        require(native['os_primitive_entry_observed'] is False,'FFI attempt does not prove OS entry')
        item['observer']['latest_native']={key:boolean(native[key]) for key in ('selected','ffi_attempt_entered','ffi_returned','afterguard_finished','os_primitive_entry_observed')}
        item['observer']['latest_native'].update(api=None if native['api'] is None else text(native['api'],128),
          windows_error=integer(native['windows_error'],nullable=True),
          error_type=None if native['error_type'] is None else text(native['error_type'],128))
        descriptor=observer['latest_descriptor']
        require(type(descriptor) is dict and set(descriptor)=={'selected','call_attempt_entered','returned','afterguard_finished','error_type'},
          'closed latest descriptor attempt evidence')
        item['observer']['latest_descriptor']={key:boolean(descriptor[key]) for key in ('selected','call_attempt_entered','returned','afterguard_finished')}
        item['observer']['latest_descriptor']['error_type']=None if descriptor['error_type'] is None else text(descriptor['error_type'],128)
        batch=observer['descriptor_batch']
        if batch is None:item['observer']['descriptor_batch']=None
        else:
            require(type(batch) is dict and set(batch)=={'selected_count','actual_attempts','observed_returns',
              'attempt_mask','return_mask','afterguard_mask','error_mask','tail_group_selected'},'closed fixed descriptor batch')
            item['observer']['descriptor_batch']={key:integer(batch[key],5 if key in ('actual_attempts','observed_returns') else 31)
              for key in ('selected_count','actual_attempts','observed_returns','attempt_mask','return_mask','afterguard_mask','error_mask')}
            item['observer']['descriptor_batch']['tail_group_selected']=boolean(batch['tail_group_selected'])
        require(type(row['cleanup_errors']) is list and len(row['cleanup_errors'])<=8,'bounded retained cleanup failures')
        item['cleanup_errors']=[text(name,128) for name in row['cleanup_errors']]
        out['backends'].append(item)
    lease=value['owned_lease'];require(type(lease) is dict and set(lease)==LEASE_KEYS,'closed same-token lease schema')
    out['owned_lease']={'created':boolean(lease['created']),'released':boolean(lease['released']),
      'status':text(lease['status'],64),'path':text(lease['path'],1024),'token_match':boolean(lease['token_match'],nullable=True)}
    require(not lease['released'] or (lease['created'] and lease['token_match'] is True),'only exact created lease release')
    return out
def describe(error,*,scope=None):
    if scope is not None:scope._owner._tail_authority.authorize_data('diagnostic',error)
    require(isinstance(error,BaseException),'retained original exception')
    # No arbitrary __str__/repr callback. Original exception object is retained
    # by caller; only bounded textual args/notes are serialized here.
    def diagnostic(value,maximum=256):
        # Preserve the previous escaped-prefix/truncation semantics without
        # materializing a full astral256-character expansion (up to2560 bytes).
        buffer=bytearray(maximum);used=0;extent=0;prefix_extent=0;stopped=False
        for index in range(min(len(value),maximum)):
            token=value[index].encode('ascii','backslashreplace').decode('ascii')
            prefix_extent+=len(token)
            for ch in token:
                size=2 if ch in ('"','\\','\b','\f','\n','\r','\t') else 6 if ord(ch)<32 else 1
                if stopped or extent+size>maximum:stopped=True;continue
                buffer[used]=ord(ch);used+=1;extent+=size
        selected=bytes(memoryview(buffer)[:used]).decode('ascii')
        buffer=None
        return {'text':selected,'truncated':len(value)>maximum or len(selected)<prefix_extent}
    args=[]
    for item in error.args[:4]:
        if type(item) is str:args.append(diagnostic(item))
        elif item is None or type(item) is bool or (type(item) is int and item.bit_length()<=64):
            args.append({'text':str(item)[:256],'truncated':len(str(item))>256})
        else:args.append({'type':diagnostic(type(item).__name__,128),'value_not_serialized':True})
    notes=getattr(error,'__notes__',())
    require(type(notes) in (list,tuple),'source exception notes container')
    return {'type':diagnostic(type(error).__name__,128),'module':diagnostic(type(error).__module__,128),
      'args':args,'additional_args':max(0,len(error.args)-4),
      'notes':[diagnostic(item) for item in notes[:8] if type(item) is str],
      'additional_notes':max(0,len(notes)-8)}
ACCOUNT_FIELDS=('operation_attempts','read_bytes','write_bytes','identity_hash_attempts','identity_hash_bytes','metadata_bytes','wall_seconds')
ACCOUNT_ROLES=('normal_exit','timeout','native_host','logs','final_tail','control')
ACCOUNT_KEYS={'selected','used','partition_used','failed','refunds','native_cleanup_attempts',
 'pin_cleanup_attempts','pin_cleanup_limit','tail_control_attempts_selected','tail_start','tail_deadline',
 'original_origin','original_deadline','complete_failure_envelope_selected_before_snapshot',
 'publication_operations_selected','publication_bytes_selected','publication_metadata_including_workspace_selected',
 'external_original_process_exit_required'}

def authenticate_pieces(value):
    """Authenticate string/scalar escape domains before JSON encoder work."""
    def visit(item,depth=0):
        require(depth<=24,'closed generated failure nesting domain')
        if type(item) is dict:
            require(len(item)<=128,'bounded generated object width')
            for key,member in item.items():visit(key,depth+1);visit(member,depth+1)
        elif type(item) in (list,tuple):
            require(len(item)<=240,'bounded generated list width')
            for member in item:visit(member,depth+1)
        elif type(item) is str:
            text(item,2064) # rawhex dominates, path1024/diagnostic256/key widths are smaller
        elif type(item) is float:
            require(math.isfinite(item) and len(repr(item))<=32,'finite source float piece domain')
        elif type(item) is int:require(item.bit_length()<=64,'uint64/signed diagnostic number domain')
        else:require(item is None or type(item) is bool,'closed generated scalar type')
    visit(value);return 2066 # largest quoted escaped string piece

def preflight(error,lifecycle,accounting_state,*,scope):
    """Closed schema preflight before output builder/diagnostic allocation.

    Trusted lifecycle/snapshot producers generate their bounded input under the
    selected build scope. Their own preallocation bound still needs source proof.
    """
    owner=scope.require_live();authority=owner._tail_authority
    require(scope._recipe=='failure','only original registered failure producer borrows views')
    authority.require_producer(owner,'failure',__import__('sys')._getframe(1))
    if callable(lifecycle):
        require(type(lifecycle) is __import__('types').MethodType and lifecycle.__func__.__code__ is authority._lifecycle
          and type(lifecycle.__self__) is authority._harness.OwnedLifecycle and lifecycle.__self__.budget is owner
          and lifecycle.__self__.custody is authority._custody,'exact original authenticated lifecycle/custody')
    require(isinstance(error,BaseException) and type(error.args) is tuple,'original exception domain')
    notes=getattr(error,'__notes__',())
    require(type(notes) in (list,tuple),'bounded diagnostic note source')
    require(type(accounting_state) is dict and set(accounting_state)==ACCOUNT_KEYS,'exact original accounting map')
    for name in ('selected','used'):
        vector=accounting_state[name]
        require(type(vector) is dict and set(vector)==set(ACCOUNT_FIELDS),'all-seven accounting fields')
        for key,value in vector.items():
            require(type(value) in (int,float) and math.isfinite(value) and value>=0
              and (value<=60 if key=='wall_seconds' else type(value) is int and value.bit_length()<=64),
              'finite bounded all-seven counter scalar domain')
    partition=accounting_state['partition_used']
    require(type(partition) is dict and set(partition)==set(ACCOUNT_ROLES),'exact six original accounting roles')
    for vector in partition.values():
        require(type(vector) is dict and set(vector)==set(ACCOUNT_FIELDS),'closed original role counters')
        for key,value in vector.items():
            require(type(value) in (int,float) and math.isfinite(value) and value>=0
              and (value<=60 if key=='wall_seconds' else type(value) is int and value.bit_length()<=64),
              'bounded original role scalar domain')
    for name in ('tail_start','tail_deadline','original_origin','original_deadline'):
        value=accounting_state[name]
        require(value is None or (type(value) in (int,float) and math.isfinite(value) and len(repr(value))<=32),
          'finite physical time domain')
    integer(accounting_state['native_cleanup_attempts'],16);integer(accounting_state['pin_cleanup_attempts'],240)
    integer(accounting_state['pin_cleanup_limit'],240);integer(accounting_state['tail_control_attempts_selected'],16)
    value=lifecycle() if callable(lifecycle) else lifecycle
    evidence=error.__dict__.get('_ce8_job_census_evidence');source='PRIMARY_CAPTURED_ROWS'
    if evidence is None:
        storage=error.__dict__.get('_ce8_census_storage')
        if storage is not None:
            require(type(storage) is authority._census.CensusStorage and storage.backend.owner is owner
              and not storage.in_query,'same original registered fixed census storage')
            evidence=storage.snapshot();source='ORIGINAL_RETAINED_STORAGE_SNAPSHOT'
        elif type(value) is dict and 'census_evidence' in value:evidence=value['census_evidence'];source='LIFECYCLE_RETAINED_ROWS'
        else:evidence=[];source='NO_QUERY_OBSERVATION_ATTACHED'
    # Existing closed validators enforce lifecycle4, raw2/PID128, exact case2
    # sidecars and all detailed flags/strings before final output construction.
    if value is not None:require(set(value)==LIFECYCLE_KEYS,'original generated lifecycle complete keys')
    validated=lifecycle_value(value,borrow=True);observations=rows(evidence,borrow=True)
    active=error.__dict__.get('_ce8_job_active_total')
    if active is not None:
        require(type(active) is dict and set(active)=={'active','total','scope'},'closed Job activity source')
        integer(active['active']);integer(active['total']);text(active['scope'],128)
    domain=(error,validated,observations,source,active,accounting_state,scope)
    value=evidence=storage=None
    return domain

def build_preflight(domain):
    require(type(domain) is tuple and len(domain)==7,'source preflight domain tuple')
    error,state,observations,source,active,accounting_state,scope=domain
    if active is not None:active={'active':active['active'],'total':active['total'],'scope':active['scope']}
    return {'schema':SCHEMA,'primary':describe(error,scope=scope),'lifecycle':state,'original_job_active_total':active,
      'original_query_observations':observations,'query_evidence_source':source,
      'accounting':accounting_state,'model_loads':0,'optimizer_updates':0,
      'native_model_qualification':False,'external_original_process_exit_required':True,
      'failure_publication_selected_before_snapshot':True,'refunds':0}

def build(error,lifecycle,accounting_state):
    value=lifecycle() if callable(lifecycle) else lifecycle
    state=lifecycle_value(value)
    evidence=error.__dict__.get('_ce8_job_census_evidence')
    source='PRIMARY_CAPTURED_ROWS'
    if evidence is None:
        storage=error.__dict__.get('_ce8_census_storage')
        if storage is not None:
            evidence=storage.snapshot();source='ORIGINAL_RETAINED_STORAGE_SNAPSHOT'
        elif type(value) is dict and 'census_evidence' in value:
            evidence=value['census_evidence'];source='LIFECYCLE_RETAINED_ROWS'
        else:evidence=[];source='NO_QUERY_OBSERVATION_ATTACHED'
    observations=rows(evidence)
    active=error.__dict__.get('_ce8_job_active_total')
    if active is not None:
        require(type(active) is dict and set(active)=={'active','total','scope'},'closed original Job activity evidence')
        active={'active':integer(active['active']),'total':integer(active['total']),'scope':text(active['scope'],128)}
    return {'schema':SCHEMA,'primary':describe(error),'lifecycle':state,'original_job_active_total':active,
      'original_query_observations':observations,'query_evidence_source':source,
      'accounting':accounting_state,'model_loads':0,'optimizer_updates':0,
      'native_model_qualification':False,'external_original_process_exit_required':True,
      'failure_publication_selected_before_snapshot':True,'refunds':0}
