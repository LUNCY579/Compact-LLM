"""Source-only compatibility for the historical INFQ metadata schema.

Only fixed safe metadata and two consumer source files are read. References to
raw records, configurations, model inputs and reconstruction payloads are never
followed. This adapter cannot qualify arithmetic lineage or native admission.
"""
from pathlib import Path
import hashlib,json,math,os,stat

REPO=Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-synth-centralized-reasoning-v1')
PACKAGE=REPO/'experiments/rineya_synth_centralized_reasoning_v1'
REVIEW=Path('C:/Users/clclc/BusinessProject/rineya-infq-exposure-projection-metadata-review-20261009-01')
EVIDENCE=PACKAGE/'evidence/INFQ_exposure_projection_cpu_20261009'
OUTPUT=Path('C:/Users/clclc/BusinessProject/synth-INFQ-arithmetic-contract-adapter-20261009-01')
CONFIG_FILE='a26897bfbe36ce284221c6f4182c487d0d76287e708fe84de75f4e57fc2a9a79'
CONFIG_BODY='078443b1cd0e456f3ce345eed68a822b6b5b85f69e15519357278a810853eea2'
RESULT_BODY='c20c3db71c2528526354f87d17ea13c958c35ad0f2eb60037de4f94f800c25e6'
ROLES={
 'audit':(PACKAGE/'evidence/infq001_attempt1_independent_audit_20261004.json',26435,'b655ec5d883d398714564a0d72e2f987ec061489eaec803e49bb234513f7965e'),
 'report':(REVIEW/'REPORT.json',9778,'10cbfb177ecddd2c5c692e5e5cbe66a593b0cc4258cae214a1794c40d11a3fab'),
 'pin_index':(REVIEW/'PIN_INDEX.json',12383,'e3c594b38665dd28c2fe6886d929d1b0c1222f3bba14a9d8dcfef9ca8461c8a9'),
 'execution':(EVIDENCE/'ROOT_ACTUAL_EXECUTION001.json',1836,'1acdf1927a48c12052a343be904144f74c8a69007465a9b97ea2a94575fb63d9'),
 'receipt':(EVIDENCE/'actual_output_attempt003/PROJECTION_RECEIPT001.json',35783,'af248fabd78219f1c35b2b54a3f3713eee04eb5723543327b24f3b7741376a69'),
 'independent':(EVIDENCE/'independent_actual_review001/INDEPENDENT_ACTUAL_PROJECTION_AUDIT001.json',134599,'86e1f5baea216108b14b05a8c39b8e42ed6117f8fb25dd63c74daf6c6102ffaa')}
CONSUMERS={
 'arithmetic_evidence.py':'405d77d4e1810a01ce8f2e9ac7ce1587b13c6783a8ff8385fbd083a4a7fc5e4c',
 'arithmetic_measurement.py':'0b48569d5686c175c2ab9f8ea8b57437902ef18d979fd9eb91e446b61dc344fd'}
ROLE_CALLS={'zero_parity_base_forward':1,'zero_parity_adapter_forward':1,
 'inference_qualification_forward':78,'inference_qualification_bulk_forward':4}
ROLE_POSITIONS={'zero_parity_base_forward':419,'zero_parity_adapter_forward':419,
 'inference_qualification_forward':26544,'inference_qualification_bulk_forward':1245}
PARENTS=['balanced_parent_023','balanced_parent_038']

def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def strict_json(raw):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'duplicate metadata key');value[key]=item
        return value
    def finite(value):
        result=float(value);require(math.isfinite(result),'nonfinite metadata');return result
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite metadata')))
def sealed(value):
    require(type(value) is dict and value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),'metadata body seal')
def checked_integer(value,expected):return type(value) is int and value==expected

def original_infq_accounting(audit):
    """Exact legacy INFQ nested fields; no fabricated modern all_equal flag."""
    sealed(audit)
    require(checked_integer(audit.get('schema_version'),1) and audit.get('status')=='PASS_SHORT_SCOPED_NATIVE_INFERENCE_MECHANICS','original INFQ audit schema/status')
    closure=audit['hash_closure'];account=audit['accounting']
    require(closure.get('config_file_sha256')==CONFIG_FILE and closure.get('config_body_sha256')==CONFIG_BODY,'original nested config file/body binding')
    require(closure.get('preservation_receipt_sha256')=='fe49928806b6edd62b5c004afa5dde94e9d9d217f1026d6ecbee739db7644f67'
        and checked_integer(closure.get('raw_file_count'),95) and closure.get('source_and_artifact_bytes_unchanged_during_audit') is True,'original preservation/byte closure')
    expected={'actual_forward_calls':84,'actual_input_positions':28627,'returned_API_calls':403,
        'failed_API_calls':0,'pending_API_calls':0,'backwards':0,'optimizer_steps':0,
        'parity_forward_calls':2,'parity_input_positions':838,'qualification_forward_calls':82,
        'qualification_input_positions':27789,'individual_durable_token_stages':78,'generated_only_bulk_forwards':4}
    require(all(checked_integer(account.get(k),v) for k,v in expected.items()),'exact original INFQ observed accounting')
    ops=account['registered_operation_counts']
    require(all(checked_integer(ops.get(k),n) for k,n in ROLE_CALLS.items()) and len(ops)==14
        and all(type(n) is int and n>0 for n in ops.values()) and sum(ops.values())==403,'original14-operation/403 closure')
    return {'attempt_id':'INFQ001_attempt1','config_file_sha256':closure['config_file_sha256'],
        'config_body_sha256':closure['config_body_sha256'],'observed_forward_calls':account['actual_forward_calls'],
        'observed_input_positions':account['actual_input_positions'],'global_API_calls':account['returned_API_calls'],
        'original_raw_file_count':closure['raw_file_count'],'registered_operation_counts':ops,
        'accounting_source':'accounting.actual_forward_calls/actual_input_positions',
        'config_binding_source':'hash_closure.config_file_sha256/config_body_sha256'}

def _sanitized_contract(documents):
    """Pure shape seam for authored tests; authority exists only in fixed reads."""
    audit,report,index,execution,receipt,independent=(documents[k] for k in ROLES)
    observed=original_infq_accounting(audit)
    for value in (execution,receipt,independent):sealed(value)
    require(report.get('kind')=='INDEPENDENT_INFQ_EXPOSURE_PROJECTION_SAFE_METADATA_REVIEW001'
        and index.get('all13_named_pins_matched_before_report_sealing') is True and checked_integer(index.get('pin_count'),13),'fixed safe review/index schema')
    scoped=report['attributed_single_scope_closure']
    require(checked_integer(scoped.get('physical_forward_calls'),84) and checked_integer(scoped.get('input_positions'),28627)
        and checked_integer(scoped.get('selected_source_rows'),4) and checked_integer(scoped.get('shared_source_parents'),2)
        and scoped.get('source_parent_ids_from_committed_note')==PARENTS,'safe metadata four-row/two-parent accounting')
    require(execution.get('kind')=='ROOT_ACTUAL_INFQ_EXISTING_RECORD_CPU_PROJECTION001'
        and checked_integer(execution.get('exit_code'),0) and checked_integer(execution.get('calls'),84)
        and checked_integer(execution.get('input_positions'),28627) and checked_integer(execution.get('source_rows'),4)
        and execution.get('source_parents')==PARENTS and execution.get('source_recipe_reconstruction_not_independent_token_telemetry') is True,'fixed successful source-recipe execution metadata')
    require(independent.get('kind')=='INDEPENDENT_ACTUAL_INFQ84_SOURCE_RECIPE_PROJECTION_AUDIT001'
        and independent.get('accepted_scoped_projection') is True
        and independent.get('all105_original_bytes_stable') is True
        and independent.get('all84_prefixes_masks_hashes_globalIDs_timeline_costs_directly_rebuilt') is True
        and checked_integer(independent.get('physical_forward_calls'),84)
        and checked_integer(independent.get('physical_input_positions'),28627)
        and independent.get('role_calls')==ROLE_CALLS and independent.get('role_positions')==ROLE_POSITIONS
        and checked_integer(independent.get('source_rows'),4) and independent.get('source_parents')==PARENTS,'independent scoped source-recipe metadata joins')
    require(receipt.get('all95_original_byte_links_verified') is True and checked_integer(receipt.get('original_file_count'),105)
        and receipt.get('source_recipe_reconstruction_verified') is True
        and receipt.get('result_body_sha256')==independent.get('result_body_sha256')==RESULT_BODY
        and receipt['sha256']==independent.get('receipt_body_sha256'),'saved receipt/review result body joins without reading result')
    require(all(v.get('global_exposure_coverage') is False for v in (execution,receipt,independent))
        and independent.get('independent_pre_hook_input_telemetry_established') is False
        and independent.get('complete_retention_disjointness_or_membership_qualified') is False
        and execution.get('arithmetic_family_mapping')==independent.get('arithmetic_family_mapping')=='UNDEFINED','no promotion of historical recipe evidence')
    return {'kind':'INFQ_ARITHMETIC_SOURCE_CONTRACT_METADATA_ADAPTER001','schema_version':1,
        'status':'COMPATIBLE_SINGLE_HISTORICAL_SCOPE_METADATA_ONLY','original_schema_accounting':observed,
        'sanitized_source_recipe':{'source_rows':4,'source_parents':PARENTS,'role_calls':ROLE_CALLS,'role_input_positions':ROLE_POSITIONS,
            'result_body_sha256':RESULT_BODY,'input_values_are_independently_recorded_per_call':False},
        'qualified':False,'global_exposure_coverage':False,'canonical_arithmetic_membership':'UNKNOWN',
        'R_S_to_arithmetic_family_mapping':'UNDEFINED','authenticated_owner_chronology':'UNKNOWN',
        'complete_retention_disjointness_or_membership_qualified':False,'native_admission_authorized':False,
        'consumer_boundary':{'original_consumers_unchanged':True,'legacy_observed_model_input_calls_field_present':False,
            'modern_hash_closure_all_equal_field_synthesized':False,'full_lineage_result':'UNKNOWN_REFUSE'},
        'candidate_parsing_disclosure':'Original producer/auditor parsed whole candidate JSONL including prompt/target/spec fields; only seven identity fields extracted/emitted. This adapter reads none of those payloads.',
        'provenance_limit':'Saved launcher/source hashes do not prospectively establish executed launcher bytes; accepted raw/source-recipe closure is attributed to the fixed independent actual audit.',
        'pending_artifacts':['Complete actual Native7/Q003/other successful and failed partial exposure manifests.',
            'Source-qualified shared R/S to arithmetic canonical-family and model-input comparability proof.',
            'Authenticated original owner chronology and complete prospective roster/barrier evidence.',
            'Genuine scorer/all-domain grade authorities, qualified full M0 enrollment and current native model/runtime/resource evidence.']}

def _read_fixed(path,size,sha):
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and type(size) is int and 0<size<=32*1024**2,'fixed safe read role')
    with path.open('rb') as f:
        before=os.fstat(f.fileno());require(stat.S_ISREG(before.st_mode) and before.st_size==size,'fixed role physical size')
        raw=f.read(size+1);after=os.fstat(f.fileno())
    require(len(raw)==size and before.st_size==after.st_size and hashlib.sha256(raw).hexdigest()==sha,'fixed safe metadata/source byte binding')
    return raw
def load_fixed_contract():
    documents={k:strict_json(_read_fixed(*ref)) for k,ref in ROLES.items()}
    pins=documents['pin_index']['pins']
    for role in ('execution','receipt','independent'):
        path,size,sha=ROLES[role]
        require(sum(1 for pin in pins if Path(pin['path']).resolve()==path.resolve() and pin['bytes']==size and pin['sha256']==sha)==1,'independently selected metadata pin-index join')
    source_refs={}
    for name,sha in CONSUMERS.items():
        path=PACKAGE/'artifacts/arithmetic_measurement_completion_work/frozen_arithmetic_measurement002'/name
        raw=_read_fixed(path,path.stat().st_size,sha)
        source_refs[name]={'path':str(path),'bytes':len(raw),'sha256':sha}
    value=_sanitized_contract(documents)
    value['safe_metadata_references']={k:{'path':str(p),'bytes':n,'sha256':s} for k,(p,n,s) in ROLES.items()}
    value['unchanged_consumer_source_references']=source_refs
    value['raw_records_or_reference_payloads_followed']=False
    value['sha256']=digest(value)
    return value
def require_complete_lineage(value):
    require(value.get('qualified') is False and value.get('global_exposure_coverage') is False,'metadata adapter authority drift')
    raise ValueError('UNKNOWN: original INFQ recipe metadata alone cannot establish complete comparable lineage or owner chronology')
def write_metadata(value):
    path=OUTPUT/'INFQ_ARITHMETIC_CONTRACT_METADATA001.json'
    require(value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),'adapter result seal')
    raw=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    require(path.read_bytes()==raw,'metadata output readback')
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
if __name__=='__main__':print(json.dumps(write_metadata(load_fixed_contract()),sort_keys=True))
