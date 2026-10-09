"""Finite arithmetic measurement evidence validators; no arbitrary proof callbacks."""
from pathlib import Path
from types import ModuleType
import hashlib
import json

PACKAGE=next(p for p in Path(__file__).resolve().parents if p.name=='rineya_synth_centralized_reasoning_v1')
SCORER='02025fecbddd0fa882e67501c19556aad01795b608a33821f14201757fdc25b9'
LABELS='2992454925eecfabda4a6046d1d59ef78e718c1639d58714aa1404f90888f5c3'
POLICY='42ee9c26f0fe1ab42c368618a5c25385fe09dac0f5777bab92df5c34c5fa05b7'
SEMANTIC_REVIEW='3153dfad0f57e2c3d00782bef118296a6e0f661ed5a0a668ab01c4f982901710'
CATEGORIES={'available_training','available_evaluation','observed_consumed_training','observed_consumed_evaluation'}
KNOWN_PARTIAL_ATTEMPTS={'NQ001_attempt7','INFQ001_attempt1','BACKENDQ003_A','BACKENDQ003_B','SHORT_M0_BATCH05_ATTEMPT2'}
DOMAINS={'arithmetic','retrieval','answerability_abstention','dialogue','long_context'}

def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def read(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256','value_sha256'} and type(ref['bytes']) is int
        and 0<ref['bytes']<=32*1024**2,'selected evidence reference')
    p=Path(ref['path']);require(p.is_absolute() and p.is_file() and not p.is_symlink(),'selected evidence file')
    with p.open('rb') as f:raw=f.read(ref['bytes']+1)
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'actual evidence byte binding')
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate evidence key');out[k]=v
        return out
    v=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    require(digest(v)==ref['value_sha256'],'actual evidence value binding');return v
def evidence(request,purpose):
    require(type(request) is dict and set(request)=={'baseline_id','evidence'},'fixed validator request schema')
    v=read(request['evidence'])
    require(type(v) is dict and v.get('sha256')==digest({k:x for k,x in v.items() if k!='sha256'})
        and v.get('purpose')==purpose and v.get('baseline_id')==request['baseline_id']
        and v.get('fixture_only') is False,'sealed genuine evidence purpose/baseline; fixture receipts forbidden')
    return v
def scorer(request):
    v=evidence(request,'ARITHMETIC_SCORER_SOURCE_LABEL_POLICY_QUALIFIED')
    require(v.get('scorer_source_sha256')==SCORER and v.get('labels_file_sha256')==LABELS
        and v.get('policy_file_sha256')==POLICY,'exact previously reviewed arithmetic semantics/data')
    ref=v['independent_semantic_review'];require(ref['sha256']==SEMANTIC_REVIEW,'fixed independent6-control semantic review')
    review=read(ref);require(review.get('kind')=='INDEPENDENT_CLOSED_ARITHMETIC_GUARD_REVIEW001'
        and review.get('tests')==6 and review.get('errors')==review.get('failures')==review.get('exit_code')==0
        and review.get('new_concrete_blockers')==[] and review.get('source_pins',{}).get('arithmetic_guard.py')==SCORER
        and review.get('source_pins',{}).get('SYNTH_ARITHMETIC_GUARD_POLICY001.json')==POLICY
        and review.get('population_file_sha256')==LABELS,'actual pinned original CPU semantics review; not native authority')
    require(hashlib.sha256((PACKAGE/'arithmetic_guard.py').read_bytes()).hexdigest()==SCORER,'current exact arithmetic scorer')
    return v
def lineage(request):
    v=evidence(request,'ARITHMETIC_AVAILABLE_TRAIN_DEV_LINEAGE_DISJOINT')
    require(v.get('scope')=='AVAILABLE_SOURCES_AND_OBSERVED_CONSUMED_OCCURRENCES'
        and v.get('namespace_only_proof') is False and set(v.get('categories',{}))==CATEGORIES,'complete available and partial consumed exposure scope')
    profile=v.get('shared_commitment_profile')
    require(profile=={'family':'arithmetic_full_canonical_ast_json_v1','input':'ordered_model_input_token_ids_json_v1',
        'tokenizer_sha256':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
        'chat_template_sha256':'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'},'shared typed source-free family and model-input commitments; no unlike prompt hashes')
    path=PACKAGE/'arithmetic_guard.py';raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==SCORER,'fixed canonical arithmetic parser')
    guard=ModuleType('lineage_fixed_arithmetic_parser');guard.__file__=str(path);exec(compile(raw,str(path),'exec'),guard.__dict__)
    def derive(refs):
        require(type(refs) is list and bool(refs),'actual comparable original metadata witnesses required')
        families=set();inputs=set()
        for ref in refs:
            document=read(ref)
            require(document.get('kind')=='SOURCE_DERIVED_COMPARABLE_INPUT_LEDGER001'
                and document.get('commitment_profile')==profile and type(document.get('records')) is list and bool(document['records']),
                'source-aware comparable input ledger; unsupported R/S/arithmetic mapping remains undefined')
            for record in document['records']:
                require(record.get('original_input_reference') and record.get('original_semantic_reference'),
                    'invented comparable ledger cannot replace original input/semantic source projection')
                original_input=read(record['original_input_reference']);original_semantic=read(record['original_semantic_reference'])
                require(type(record.get('original_row_index')) is int and record['original_row_index']>=0,
                    'exact original source row locator')
                index=record['original_row_index'];input_row=original_input['rows'][index];semantic_row=original_semantic['rows'][index]
                require(input_row['row_id']==semantic_row['row_id']==record.get('row_id'), 'original input/semantic row projection join')
                actual_ids=input_row.get('prompt_ids',input_row.get('input_ids'))
                actual_expr=semantic_row.get('canonical_arithmetic_expression',semantic_row.get('parent',{}).get('canonical_expression'))
                require(actual_expr is not None,'UNKNOWN cross-schema family mapping; arithmetic-aware original projection absent')
                require(actual_ids==record.get('model_input_ids') and actual_expr==record.get('canonical_arithmetic_expression'),
                    'supplied commitments must reproduce exact original source values')
                require(type(record.get('canonical_arithmetic_expression')) in (dict,int)
                    and type(record.get('model_input_ids')) is list and record['model_input_ids']
                    and all(type(x) is int and 0<=x<151936 for x in record['model_input_ids']),
                    'actual canonical family and ordered model-input values required, not declared hashes')
                normalized=guard.normalize(record['canonical_arithmetic_expression'])
                families.add(digest(normalized));inputs.add(digest(record['model_input_ids']))
        return families,inputs
    attempts=v.get('attempt_commitments',[])
    require(KNOWN_PARTIAL_ATTEMPTS<={a.get('attempt_id') for a in attempts}
        and all(a.get('includes_failed_and_partial_prefixes') is True and a.get('observed_complete') is True
            and type(a.get('original_journal_references')) is list and a['original_journal_references'] for a in attempts),
        'all known native and failed partial exposures require original journal commitments')
    for attempt in attempts:
        if attempt['attempt_id']=='SHORT_M0_BATCH05_ATTEMPT2':require(attempt.get('observed_forward_calls')==253 and attempt.get('durable_final_rows')==19,'known failed253calls/19finalrows plus actual partial-prefix accounting')
        config=read(attempt['original_config_reference']);audit=read(attempt['original_audit_reference'])
        require(audit.get('config_body_sha256')==config.get('sha256') and audit.get('hash_closure',{}).get('all_equal') is True,
            'known attempted consumption must join original config/audit/source closure')
        require(attempt.get('observed_forward_calls')==audit.get('observed_forward_calls',audit.get('forward_calls'))
            and type(attempt.get('observed_forward_calls')) is int,'actual observed calls, never planned optimizer exposure counts')
        observed_families,observed_inputs=derive(attempt['original_journal_references'])
        require(observed_inputs==set(attempt.get('observed_model_input_hashes',[]))
            and observed_families==set(attempt.get('observed_canonical_family_hashes',[])),
            'actual failed/partial exposure commitment derivation differs')
    ledger_ref=v['guard_source_ledger_reference'];require(ledger_ref['sha256']=='c46f9ca13871e9bd9ab7695f0b79e729a9300d208b4d2bc9fce83ba92e56a252','fixed actual128 arithmetic source ledger')
    ledger=read(ledger_ref);families={p['expression_sha256'] for p in ledger['selected_parents']}
    plan_ref=v['guard_plan_reference'];require(plan_ref['sha256']=='81cf5f729e232710343a103cec1c34f4efb07acf5dca1ed32def76f9b5838c95','fixed actual512 arithmetic model-input plan')
    plan=read(plan_ref);inputs={digest(r['prompt_ids']) for b in plan['batches']['arithmetic_guard_DEV'] for r in b['rows']}
    prompt_ref=v['guard_prompt_reference'];require(prompt_ref['sha256']=='7e2ec66fd08c0a7c5fd56444e0d3fa86cbd0b93b40c3cf8ffad75512ff6d7d66','exact measured prompt-only input source')
    prompts=read(prompt_ref);require({digest(r['input_ids']) for r in prompts['rows']}==inputs,'actual plan/measured model input join')
    require(families==set(v.get('guard_family_hashes',[])) and inputs==set(v.get('guard_input_hashes',[])),
        'actual guard-side commitment derivation differs')
    require(len(families)==128 and len(inputs)>0,'actual arithmetic guard commitments required')
    for category in CATEGORIES:
        item=v['categories'][category]
        require(item.get('observed_complete') is True and item.get('incomparable_records')==0
            and item.get('includes_failed_partial_exposures') is True and item.get('source_references')
            and item.get('commitment_profile')==profile,'complete comparable original-source exclusion category')
        source_families,source_inputs=derive(item['source_references'])
        require(source_families==set(item['canonical_family_hashes']) and source_inputs==set(item['model_input_hashes']),
            'available/consumed commitment values must derive from actual selected source records')
        require(not families.intersection(source_families) and not inputs.intersection(source_inputs),
            'arithmetic family or model input overlaps available/consumed exposure')
    return v
def chronology(request,purpose):
    v=evidence(request,purpose)
    require(set(v.get('domain_ids',[]))==DOMAINS and v.get('candidate_outputs_obtained') is False,
        'exact accepted complete domain roster; no replacement by class labels')
    roster=read(v['roster_reference']);require({d['domain_id'] for d in roster['domains']}==DOMAINS,'actual frozen domain roster')
    events=[read(ref) for ref in v['event_references']]
    require(events and all(e.get('baseline_id')==request['baseline_id'] and e.get('original_source_observed') is True for e in events),
        'original sealed chronological source event witnesses')
    from datetime import datetime,timezone
    require(len({e['event'] for e in events})==len(events),'duplicate chronology event names forbidden')
    times={}
    for event in events:
        stamp=event['observed_utc'];require(type(stamp) is str and stamp.endswith('Z'),'typed UTC observation')
        instant=datetime.fromisoformat(stamp.replace('Z','+00:00'));require(instant.tzinfo==timezone.utc,'UTC chronology only')
        original=read(event['original_event_reference'])
        require(original.get('event')==event['event'] and original.get('observed_utc')==stamp
            and original.get('baseline_id')==request['baseline_id'],'actual original event source join')
        times[event['event']]=instant
    require('roster_frozen' in times and 'pre_treatment_owner_barrier' in times
        and times['roster_frozen']<=times['pre_treatment_owner_barrier'],'prospective roster before current owner barrier')
    barrier=next(e for e in events if e['event']=='pre_treatment_owner_barrier')
    require(barrier.get('treatment_started') is False and barrier.get('candidate_outputs_present') is False
        and barrier.get('observed_original_source_references'),'actual pre-treatment owner/source state; no future first-update requirement')
    for ref in barrier['observed_original_source_references']:read(ref)
    if purpose=='ENROLLMENT_BEFORE_TREATMENT':
        require('all_M0_scoring_complete' in times and times['all_M0_scoring_complete']<=times['pre_treatment_owner_barrier'],
            'all-domain M0 score completion before current treatment admission barrier')
    return v
def raw_capture(request):
    v=evidence(request,'ARITHMETIC_FIXED64_RAW_CAPTURE_QUALIFIED')
    path=PACKAGE/'arithmetic_raw_audit_v1/arithmetic_raw_auditor.py'
    require(path.is_file() and v.get('auditor_source_sha256')==hashlib.sha256(path.read_bytes()).hexdigest()==
        '59d534b6546a6926d88e9e9bf57ce306eb93f37a85bc3a93f2caf463d807bbba',
        'reviewed integrated arithmetic-specific original raw auditor required')
    module=ModuleType('fixed_arithmetic_raw_auditor');module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    actual=module.audit_existing(v['actual_audit_receipt'],read(v['actual_audit_request']))
    require(actual==read(v['actual_audit_receipt']) and actual.get('status')=='PASS_SCOPED_SHORT_M0_RAW_COLLECTION_MECHANICS_UNSCORED',
        'actual original raw-audit replay, no declared Boolean substitute')
    return v
def domain_scores(request):
    v=evidence(request,'ALL_DOMAIN_M0_SCORING_QUALIFIED')
    require(v.get('domain_id') in DOMAINS and v.get('source_qualification_reference') and v.get('independent_grade_reference'),
        'actual domain source and independent grade evidence')
    source=read(v['source_qualification_reference']);grade=read(v['independent_grade_reference'])
    require(source.get('qualified') is True and source.get('fixture_only') is False and grade.get('fixture_only') is False
        and grade.get('baseline_id')==request['baseline_id'] and grade.get('score_file_sha256')==v['score_file_sha256'],
        'genuine same-domain source qualification and complete M0 grade')
    scores=read(v['score_reference']);require(v['score_reference']['sha256']==v['score_file_sha256']
        and type(grade.get('rows')) is list and grade['rows'] and grade.get('complete_domain') is True,
        'actual independently graded complete row records and score file')
    derived={row['row_id']:row['correct'] for row in grade['rows']}
    require(len(derived)==len(grade['rows']) and all(type(x) is bool for x in derived.values())
        and all(scores.get(k) is x for k,x in derived.items()),'independent whole-response grades reproduce exact M0 map')
    if v['domain_id']=='dialogue':
        require(set(source.get('qualified_dimensions',[]))=={'adequacy','relevance','fluency','role_consistency'},'applicable exact four-dimension rubric; no unsupported remapping')
        raise ValueError('UNQUALIFIED dialogue authority: fixed independent human-rating audit provider not integrated')
    if v['domain_id']=='long_context':
        raise ValueError('UNQUALIFIED long-context endpoint: separate actual native source/collection/grade provider required')
    if v['domain_id'] in ('retrieval','answerability_abstention'):
        raise ValueError('UNQUALIFIED task endpoint: fixed original whole-response task grade audit provider required')
    if v['domain_id']=='arithmetic':
        require(grade.get('kind')=='INDEPENDENT_ARITHMETIC_WHOLE_RESPONSE_GRADE_AUDIT001'
            and grade.get('scorer_source_sha256')==SCORER,'fixed arithmetic whole-response grade audit')
    return v
def validate(purpose,request):
    if purpose=='ARITHMETIC_SCORER_SOURCE_LABEL_POLICY_QUALIFIED':return scorer(request)
    if purpose=='ARITHMETIC_AVAILABLE_TRAIN_DEV_LINEAGE_DISJOINT':return lineage(request)
    if purpose in ('ROSTER_FROZEN_BEFORE_CAPTURE','ENROLLMENT_BEFORE_TREATMENT'):return chronology(request,purpose)
    if purpose=='ARITHMETIC_FIXED64_RAW_CAPTURE_QUALIFIED':return raw_capture(request)
    if purpose=='ALL_DOMAIN_M0_SCORING_QUALIFIED':return domain_scores(request)
    raise ValueError('unregistered arithmetic evidence purpose')
