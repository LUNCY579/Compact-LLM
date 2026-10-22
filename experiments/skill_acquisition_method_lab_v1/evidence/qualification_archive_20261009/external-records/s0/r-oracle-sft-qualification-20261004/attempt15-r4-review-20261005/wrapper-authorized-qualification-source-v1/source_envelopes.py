"""Pure source-field ASCII wire bounds; caller must prepay derivation/validation.

No serialization, file read, runtime API, owner or permission is created here.
Domains police the exact bounded record before the smaller encoder selection.
"""
import math
from collections.abc import Mapping
from types import MappingProxyType
class ImmutableDomain:
    __slots__=()
    def __setattr__(self,*args):raise RuntimeError('immutable source envelope domain')
    def __delattr__(self,*args):raise RuntimeError('immutable source envelope domain')

class UInt(ImmutableDomain):
    __slots__=('bits',)
    def __init__(self,bits):object.__setattr__(self,'bits',bits)
class FiniteFloat(ImmutableDomain):
    __slots__=()
class HexText(ImmutableDomain):
    __slots__=('length',)
    def __init__(self,length):object.__setattr__(self,'length',length)
class AsciiToken(ImmutableDomain):
    __slots__=('length',)
    def __init__(self,length):object.__setattr__(self,'length',length)
class HexAtMost(ImmutableDomain):
    __slots__=('length',)
    def __init__(self,length):object.__setattr__(self,'length',length)
class BoolValue(ImmutableDomain):
    __slots__=()
class AtMost(ImmutableDomain):
    __slots__=('item','count')
    def __init__(self,item,count):object.__setattr__(self,'item',item);object.__setattr__(self,'count',count)
class OptionalKeys(ImmutableDomain):
    __slots__=('required','optional')
    def __init__(self,required,optional):object.__setattr__(self,'required',required);object.__setattr__(self,'optional',optional)
F64=FiniteFloat();U32=UInt(32);U64=UInt(64)
def immutable_plan(value):
    if isinstance(value,Mapping):return MappingProxyType({key:immutable_plan(item) for key,item in value.items()})
    if type(value) in (list,tuple):return tuple(immutable_plan(item) for item in value)
    if type(value) is AtMost:return AtMost(immutable_plan(value.item),value.count)
    if type(value) is OptionalKeys:return OptionalKeys(immutable_plan(value.required),immutable_plan(value.optional))
    return value
def require(value,message):
    if value is not True:raise RuntimeError(message)
def string_length(value):
    require(type(value) is str and len(value)<=65536,'bounded source string')
    result=2
    for char in value:
        n=ord(char)
        result+=2 if char in '"\\\b\f\n\r\t' else 1 if 32<=n<127 else 6 if n<=65535 else 12
    return result
def canonical_length(value):
    ancestors=set();nodes=0
    def visit(item,depth):
        nonlocal nodes
        nodes+=1;require(nodes<=8192 and depth<=32,'bounded acyclic source record graph')
        if item is None:return 4
        if type(item) is bool:return 4 if item else 5
        if type(item) is int:
            require(-2**64<=item<=2**64-1,'bounded source integer');return len(str(item))
        if type(item) is float:
            require(math.isfinite(item),'finite source float');return len(repr(item))
        if type(item) is str:return string_length(item)
        require(isinstance(item,Mapping) or type(item) in (tuple,list),'registered source graph type')
        identity=id(item);require(identity not in ancestors,'cyclic source record');ancestors.add(identity)
        try:
            if isinstance(item,Mapping):
                require(all(type(key) is str for key in item),'JSON object string keys')
                return 2+max(0,len(item)-1)+sum(string_length(key)+1+visit(val,depth+1) for key,val in item.items())
            return 2+max(0,len(item)-1)+sum(visit(val,depth+1) for val in item)
        finally:ancestors.remove(identity)
    return visit(value,0)+1 # exact canonical LF
def upper(schema):
    def visit(item):
        if type(item) is UInt:return len(str(2**item.bits-1))
        if type(item) is FiniteFloat:return 32 # Python binary64 JSON repr <32 ASCII chars
        if type(item) is BoolValue:return 5
        if type(item) is HexText:return item.length+2
        if type(item) is AsciiToken:return item.length+2
        if type(item) is HexAtMost:return item.length+2
        if type(item) is AtMost:return 2+max(0,item.count-1)+item.count*visit(item.item)
        if type(item) is OptionalKeys:return visit(dict(item.required,**item.optional))
        if isinstance(item,Mapping):return 2+max(0,len(item)-1)+sum(string_length(k)+1+visit(v) for k,v in item.items())
        if type(item) in (tuple,list):return 2+max(0,len(item)-1)+sum(visit(v) for v in item)
        return canonical_length(item)-1
    return visit(schema)+1
def validate(schema,value):
    def visit(expected,actual,depth):
        require(depth<=32,'bounded source envelope depth')
        if type(expected) is UInt:
            require(type(actual) is int and 0<=actual<2**expected.bits,'unsigned source field');return
        if type(expected) is FiniteFloat:
            require(type(actual) in (int,float) and math.isfinite(actual) and len(str(actual))<=32,'finite32char source number');return
        if type(expected) is BoolValue:
            require(type(actual) is bool,'source boolean field');return
        if type(expected) is HexText:
            require(type(actual) is str and len(actual)==expected.length and not set(actual)-set('0123456789abcdef'),'fixed raw lowercase hex');return
        if type(expected) is AsciiToken:
            require(type(actual) is str and len(actual)<=expected.length and all(32<=ord(ch)<127 and ch not in '"\\' for ch in actual),'bounded unescaped ASCII source token');return
        if type(expected) is HexAtMost:
            require(type(actual) is str and len(actual)<=expected.length and len(actual)%2==0 and not set(actual)-set('0123456789abcdef'),'bounded raw lowercase hex');return
        if type(expected) is AtMost:
            require(type(actual) in (tuple,list) and len(actual)<=expected.count,'bounded source record array')
            for x in actual:visit(expected.item,x,depth+1)
            return
        if type(expected) is OptionalKeys:
            require(isinstance(actual,Mapping) and set(expected.required)<=set(actual)<=set(expected.required)|set(expected.optional),'exact source optional key set')
            for k,v in actual.items():visit(expected.required[k] if k in expected.required else expected.optional[k],v,depth+1)
            return
        if isinstance(expected,Mapping):
            require(isinstance(actual,Mapping) and set(actual)==set(expected),'exact source record keys')
            for k,v in expected.items():visit(v,actual[k],depth+1)
            return
        if type(expected) in (tuple,list):
            require(type(actual) in (tuple,list) and len(actual)==len(expected),'exact source sequence length')
            for x,y in zip(expected,actual):visit(x,y,depth+1)
            return
        require(type(actual) is type(expected) and actual==expected,'fixed source literal/identity')
    visit(schema,value,0);require(canonical_length(value)<=upper(schema),'source-domain canonical bound');return None
def _ref(path,max_bytes):return {'path':str(path),'bytes':UInt(max_bytes.bit_length()),'sha256':HexText(64)}
def derive_envelopes(grant_ref,target_manifest_ref,source_ref,runtime_ref,actor_output,bounds,partitions,artifact_path,case_selected):
    from pathlib import Path
    # Fixed exact references/paths and vectors are established in paid source
    # preparation. Only genuine future OS scalar values vary in these domains.
    grant_domain=_ref(grant_ref['path'],16384) # future root grant digest/size not fabricated in preparation
    prepay={'grant_ref':grant_domain,'selected':bounds,'partitions':partitions,'case_names':['normal_exit','timeout'],
      'target_manifest_ref':target_manifest_ref,'models_loaded':0,'optimizer_updates':0,'refunds':0}
    census=OptionalKeys({'buffer_bytes':U32,'pointer_bytes':U32,'capacity':U32,'api_success':BoolValue(),
      'actual_windows_error':U32,'api_returned_bytes':U32,'process_guard':8,'assigned_header':U32,
      'returned_count_header':U32,'native_return_observed':True,'raw_buffer_hex':HexAtMost(2064),'complete':BoolValue(),'authority_accepted':BoolValue()},
      {'retry_capacity':U32,'observed_pid_entries':AtMost(U32,8),'observed_unique_pid_count':U32,'observed_duplicate_entries':U32,'pids':AtMost(U32,8)})
    state={'selected':bounds,'used':{k:F64 if k=='wall_seconds' else U64 for k in bounds},
      'partition_used':{role:{k:F64 if k=='wall_seconds' else U64 for k in vector} for role,vector in partitions.items()},
      'pending':[],'failed':[],'refunds':0,'complete_final_envelope_selected_before_snapshot':True}
    schemas={'ROOT_PREPAYMENT.json':prepay};birth_schemas={};cases=[]
    for case in ('normal_exit','timeout'):
        directory=Path(actor_output)/case
        argv=(runtime_ref['path'],'exec',str(artifact_path),
          '--qualification-probe',grant_ref['path'],HexText(64),str(directory))
        identity={'pid':U32,'creation_filetime_100ns':U64,'image_path':runtime_ref['path'],'argv':argv}
        birth={'identity':identity,'grant_ref':grant_domain,'case_name':case,'target_manifest_ref':target_manifest_ref}
        birth_schemas[case]=birth;schemas['ACTOR_'+case+'_RETAINED_BIRTH.json']=birth
        io={k:U64 for k in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')}
        terminal={'supervisor_signaled':True,'actual_supervisor_returncode':0,'job_active_processes':0,'job_total_processes':U32,
          'job_cumulative_io':io,'job_census_evidence':AtMost(census,2),'same_retained_supervisor_birth':True,
          'retained_supervisor_identity':identity,'terminal_monotonic':F64,
          'log_accounting':{'read_bytes':U64,'write_bytes':U64,'operation_attempts':U64},
          'native_qualified':False,'execution_verified':False,'all_owned_handles_closed':False,'provider_wall_seconds':F64}
        timed=case=='timeout'
        child={'role':'QUALIFICATION_CHILD','expected_child_exit_code':124 if timed else 0,
          'nested_timeout_deadline':F64 if timed else 0,'timeout_stop_monotonic':F64 if timed else 0,
          'timeout_stop_returned_monotonic':F64 if timed else 0,'timeout_planned_dispatch_monotonic':F64 if timed else 0,
          'timeout_dispatch_reserve_milliseconds':100 if timed else 0,'timeout_scheduled_due_milliseconds':U32 if timed else 0,
          'timeout_observed_dispatch_lag_milliseconds':F64 if timed else 0,'timeout_callback_quiesced':True,
          'pid':U32,'creation_filetime_100ns':U64,'exit_filetime_100ns':U64,'returncode':124 if timed else 0,
          'job_active':0,'handles_closed':True,'job_read':U64,'job_write':U64,
          'physical_origin_monotonic':F64,'physical_deadline_monotonic':F64}
        schemas['ACTOR_PROBE_CHILD_'+case]=child
        cases.append({'case_name':case,'actor_receipt_ref':_ref(directory/'ACTOR_PROBE_CLOSED.json',16384),
          'actor_retained_terminal':terminal,'actor_exit_filetime_100ns':U64,'parent_handles_closed':True,
          'target_selected':case_selected,
          'actual_target_cost_ref':None,'actor_child_receipt':child})
    final={'schema':'CE8-ACTOR-PARENT-TWO-CASE-LOGICAL-CLOSED-6','grant_ref':grant_domain,
      'owner_state':state,'controller_retained_identity':{'pid':U32,'creation_filetime_100ns':U64,'boot_identity':AsciiToken(54)},'cases':cases,'pending':[],'failed':[],'refunds':0,
      'original_runtime_pin_release':{'closed_pins':U32,'all_original_handles_closed':True,'refunds':0},
      'owned_lease_released':True,'physical_parent_IO':io,'physical_disjoint_Job_read':U64,'physical_disjoint_Job_write':U64,
      'physical_IO_not_additional_logical_debit':True,'model_loads':0,'optimizer_updates':0,'whole_model_qualification':False,
      'unknown':['ORIGINAL_TOOL_PARENT_FINAL_EXIT_AND_LAST_PUBLICATION_IO_AWAIT_EXTERNAL_OBSERVATION'],
      'physical_deadline_monotonic':F64}
    schemas['CLOSED_ACCOUNTING.json']=final
    final_upper=upper(final)
    return {'schemas':schemas,'birth_schemas':birth_schemas,'prepayment_upper':upper(prepay),
      'birth_uppers':{k:upper(v) for k,v in birth_schemas.items()},'final_upper':final_upper,
      'final_status':'SOURCE_DOMAIN_BOUND_FITS' if final_upper<=24576 else 'SOURCE_DOMAIN_BOUND_NO_FIT',
      'owner_state_template':state,'case_templates':cases,'candidate_final_ceiling':24576,
      'required_before_small_birth_encoding':'Validate exact runtime image_path and full argv/source reference identities under a prepaid predicate scope.',
      'limitations':'Source-domain wire bounds only; no physical heap/CPU or Native qualification. Unexpected keys/domains refuse before smaller serialization.'}
