"""Conditional arithmetic grading/enrollment from selected audited evidence.

Staging rejects real use before any reads. Proofs are rerun from selected source,
not accepted from Boolean callback assertions. CPU actors have separate tokens.
"""
from copy import deepcopy
from pathlib import Path
from types import ModuleType
import hashlib
import importlib
import json
import math
import sys

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'arithmetic_measurement_v1'
PLAN_FILE_SHA256='81cf5f729e232710343a103cec1c34f4efb07acf5dca1ed32def76f9b5838c95'
LABEL_FILE_SHA256='2992454925eecfabda4a6046d1d59ef78e718c1639d58714aa1404f90888f5c3'
SCORER_SOURCE_SHA256='02025fecbddd0fa882e67501c19556aad01795b608a33821f14201757fdc25b9'
REF_FIELDS={'path','bytes','sha256','value_sha256'}

def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def seal(value):return {**value,'sha256':digest(value)}
def pairs(items):
    result={}
    for k,v in items:require(k not in result,'duplicate selected JSON field');result[k]=v
    return result
def finite(text):
    value=float(text);require(math.isfinite(value),'nonfinite selected JSON');return value
def parse(raw):return json.loads(raw.decode(),object_pairs_hook=pairs,parse_float=finite,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def read_ref(ref):
    require(type(ref) is dict and set(ref)==REF_FIELDS and type(ref['bytes']) is int and 0<ref['bytes']<=32*1024**2,'exact bounded selected JSON reference')
    path=Path(ref['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink(),'selected regular JSON file')
    with path.open('rb') as stream:raw=stream.read(ref['bytes']+1)
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'selected evidence byte drift')
    value=parse(raw);require(digest(value)==ref['value_sha256'],'selected evidence value drift');return value
def source(ref,pins):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and pins.get(ref['path'])==ref['sha256'],'source belongs to reviewed closure')
    path=Path(ref['path']);require(path.is_absolute() and path.suffix=='.py' and path.is_file() and not path.is_symlink() and type(ref['bytes']) is int and 0<ref['bytes']<=1024**2,'bounded selected validator source')
    with path.open('rb') as stream:raw=stream.read(ref['bytes']+1)
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'selected validator byte drift')
    m=ModuleType('arithmetic_selected_'+ref['sha256'][:16]);m.__file__=ref['path'];exec(compile(raw,m.__file__,'exec'),m.__dict__);return m
def proof(spec,pins,*,purpose,fixture_only,baseline_id):
    require(type(spec) is dict and set(spec)=={'source','function','request','result','purpose'} and spec['purpose']==purpose,'exact selected evidence purpose')
    if not fixture_only:
        expected_path=HERE/'arithmetic_evidence.py'
        require(Path(spec['source']['path']).resolve()==expected_path.resolve() and spec['function']=='validate',
                'fixed own evidence validators; arbitrary callbacks forbidden')
    request=read_ref(spec['request']);expected=read_ref(spec['result']);module=source(spec['source'],pins)
    function=getattr(module,spec['function'],None);require(callable(function),'selected independent validator function')
    actual=function(purpose,deepcopy(request)) if not fixture_only else function(**deepcopy(request))
    require(actual==expected and type(actual) is dict and actual.get('purpose')==purpose
            and actual.get('fixture_only') is fixture_only and actual.get('baseline_id')==baseline_id
            and actual.get('qualified') is True,'independent original evidence replay/source/baseline differs')
    return actual
def integrated():require(HERE==EXPECTED,'staged arithmetic measurement has no native grading or enrollment authority')

def _cap_factory():
    token=object()
    def get():return token
    def check(value):require(value is token,'private authored arithmetic measurement capability')
    return get,check
_CPU_capability,_check_cap=_cap_factory()

class Prepared:
    def __init__(self,value,*,capability):
        _check_cap(capability);require(value['config']['fixture_only'] is True,'CPU token cannot mint native context')
        self.value=deepcopy(value);self.sha256=digest(value)
def _value(prepared):require(type(prepared) is Prepared and digest(prepared.value)==prepared.sha256,'unaltered source-qualified measurement context');return deepcopy(prepared.value)
def _native_factory():
    token=object()
    class NativePrepared:
        def __init__(self,value,key):
            require(key is token and value['config']['fixture_only'] is False,'private source-qualified native mint')
            self.value=deepcopy(value);self.sha256=digest(value)
    def mint(value):return NativePrepared(value,token)
    def read(value):
        require(type(value) is NativePrepared and digest(value.value)==value.sha256,'native-qualified context; CPU contexts forbidden')
        return deepcopy(value.value)
    return mint,read
_mint_native,_native_value=_native_factory()

def _registration(config_ref,*,fixture_only,authority_anchors=None):
    if not fixture_only:
        require(type(authority_anchors) is dict and set(authority_anchors)=={'approval','review'},
                'independently selected external approval/review role anchors required before registration reads')
        selected_approval=read_ref(authority_anchors['approval']);selected_review=read_ref(authority_anchors['review'])
        require(selected_approval.get('kind')=='OWNER_ARITHMETIC_M0_SCORING_AUTHORIZATION001'
                and selected_review.get('kind')=='INDEPENDENT_ARITHMETIC_M0_MEASUREMENT_REVIEW001',
                'actual external metadata authority roles; labels cannot substitute')
    c=read_ref(config_ref)
    fields={'kind','fixture_only','baseline_id','source_pins','approval','review','plan','labels','policy','roster',
            'raw_capture_registration','capture_checks','scorer_check','disjointness_check','chronology_check','sha256'}
    require(type(c) is dict and set(c)==fields and c['kind']=='REGISTERED_ARITHMETIC_M0_MEASUREMENT001'
            and c['fixture_only'] is fixture_only and c['sha256']==digest({k:v for k,v in c.items() if k!='sha256'}),'exact arithmetic measurement registration')
    approval=read_ref(c['approval']);review=read_ref(c['review'])
    if not fixture_only:
        require(c['approval']==authority_anchors['approval'] and c['review']==authority_anchors['review']
                and approval==selected_approval and review==selected_review,'external authority references cannot come from self-selected registration')
    expected={'baseline_id':c['baseline_id'],'source_pins':c['source_pins'],'plan':c['plan'],'labels':c['labels'],
              'policy':c['policy'],'roster':c['roster'],'raw_capture_registration':c['raw_capture_registration'],
              'proofs_sha256':digest({k:c[k] for k in ('capture_checks','scorer_check','disjointness_check','chronology_check')})}
    require(approval.get('kind')=='OWNER_ARITHMETIC_M0_SCORING_AUTHORIZATION001' and approval.get('approved') is True
            and approval.get('fixture_only') is fixture_only and approval.get('binding')==expected
            and approval.get('permissions')=={'arithmetic_M0_scoring':True,'arithmetic_enrollment_contribution':True,
                'training':False,'protected_bank_access':False,'language_ratings':False}
            and type(approval.get('human_evidence_reference')) is str and approval['human_evidence_reference'],'scope-specific owner source/population/proof binding before captures')
    require(review.get('kind')=='INDEPENDENT_ARITHMETIC_M0_MEASUREMENT_REVIEW001' and review.get('accepted') is True
            and review.get('fixture_only') is fixture_only and review.get('binding')==expected
            and review.get('approval_file_sha256')==c['approval']['sha256'],'independently selected exact measurement review')
    for path,pin in c['source_pins'].items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,'reviewed measurement source bytes')
    if not fixture_only:
        require(c['plan']['sha256']==PLAN_FILE_SHA256 and c['labels']['sha256']==LABEL_FILE_SHA256
                and c['source_pins'].get(str(PACKAGE/'arithmetic_guard.py'))==SCORER_SOURCE_SHA256,'fixed reviewed actual arithmetic plan/labels/scorer pins')
    return c

def _prepare(config_ref,*,fixture_only,loader,authority_anchors=None):
    c=_registration(config_ref,fixture_only=fixture_only,authority_anchors=authority_anchors)
    # All evidence purposes/source/request/results are joined before raw outputs or grading.
    checks={}
    for name,purpose in (('disjointness_check','ARITHMETIC_AVAILABLE_TRAIN_DEV_LINEAGE_DISJOINT'),
                         ('chronology_check','ROSTER_FROZEN_BEFORE_CAPTURE'),
                         ('scorer_check','ARITHMETIC_SCORER_SOURCE_LABEL_POLICY_QUALIFIED')):
        checks[name]=proof(c[name],c['source_pins'],purpose=purpose,fixture_only=fixture_only,baseline_id=c['baseline_id'])
    require(len(c['capture_checks'])==8,'all8 original64 arithmetic capture checks')
    capture_evidence=[]
    for ordinal,spec in enumerate(c['capture_checks'],1):
        evidence=proof(spec,c['source_pins'],purpose='ARITHMETIC_FIXED64_RAW_CAPTURE_QUALIFIED',fixture_only=fixture_only,baseline_id=c['baseline_id'])
        require(evidence.get('ordinal')==ordinal and evidence.get('rows')==64 and evidence.get('source_kind')=='arithmetic_guard_DEV','exact ordered64 capture proof')
        capture_evidence.append(evidence)
    if not fixture_only:
        raw_registration=read_ref(c['raw_capture_registration'])
        require(len(raw_registration.get('capture_anchors',[]))==len(raw_registration.get('batch_configs',[]))==8,
                'all eight actual raw config/audit/preservation anchors before response import')
        for evidence,anchor,configuration in zip(capture_evidence,raw_registration['capture_anchors'],raw_registration['batch_configs']):
            require(evidence.get('audit_file_sha256')==anchor['expected_audit_sha256']
                and evidence.get('config_file_sha256')==anchor['expected_config_sha256']==configuration['sha256']
                and Path(evidence.get('preserved_directory','')).resolve()==Path(anchor['preserved_directory']).resolve()
                and evidence.get('plan_file_sha256')==c['plan']['sha256'], 'exact independently audited capture/config/preserved/plan joins')
    plan=read_ref(c['plan']);labels=read_ref(c['labels']);policy=read_ref(c['policy']);roster=read_ref(c['roster'])
    require(plan['rows']==512 and plan['parent_components']==128 and len(plan['batches']['arithmetic_guard_DEV'])==8,'exact512/128/8 arithmetic plan')
    rows=labels['rows'];expected=[r['row_id'] for b in plan['batches']['arithmetic_guard_DEV'] for r in b['rows']]
    require(len(rows)==512 and [r['row_id'] for r in rows]==expected and digest(expected)==plan['row_order_sha256'],'whole512 semantic labels/raw source order')
    domain={'domain_id':'arithmetic','class':'task','row_ids':expected,'component_ids':[r['parent_id'] for r in rows]}
    require(len(set(domain['component_ids']))==128 and type(roster.get('domains')) is list
            and any(d==domain for d in roster['domains']) and roster.get('candidate_outputs_used') is False,
            'complete frozen roster includes exact arithmetic membership before treatment')
    require(checks['scorer_check'].get('labels_file_sha256')==c['labels']['sha256']
            and checks['scorer_check'].get('scorer_source_sha256')==c['source_pins'][str(PACKAGE/'arithmetic_guard.py')]
            and checks['scorer_check'].get('policy_value_sha256')==c['policy']['value_sha256']
            and checks['disjointness_check'].get('plan_file_sha256')==c['plan']['sha256']
            and checks['chronology_check'].get('roster_file_sha256')==c['roster']['sha256'],'evidence binds actual scorer/labels/policy/lineage/roster')
    lineage=checks['disjointness_check']
    family_hashes={r['parent']['expression_sha256'] for r in rows}
    token_hashes={digest(row['prompt_ids']) for batch in plan['batches']['arithmetic_guard_DEV'] for row in batch['rows']}
    require(set(lineage.get('guard_family_hashes',[]))==family_hashes
            and set(lineage.get('guard_input_hashes',[]))==token_hashes,
            'guard-side source-free canonical family and shared ordered model-input commitments')
    imported=loader(c['raw_capture_registration'])
    require(imported.get('fixture_only') is fixture_only and len(imported['rows'])==512 and imported['domain']==domain
            and [r['row_id'] for r in imported['rows']]==expected,'all typed audited arithmetic M0 rows/source order')
    value={'config':c,'imported':imported,'labels':rows,'policy':policy,'domain':domain,'roster':roster,'checks':checks}
    return Prepared(value,capability=_CPU_capability()) if fixture_only else _mint_native(value)

def prepare_registered(config_ref,*,authority_anchors=None):
    integrated()
    if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
    pipeline=importlib.import_module('arithmetic_capture_v1.arithmetic_capture_pipeline')
    require(Path(pipeline.__file__).resolve()==PACKAGE/'arithmetic_capture_v1/arithmetic_capture_pipeline.py','reviewed arithmetic typed raw importer origin')
    return _prepare(config_ref,fixture_only=False,loader=pipeline.prepare_task_import,authority_anchors=authority_anchors)

def pre_capture_admission(config_ref,*,authority_anchors=None):
    integrated();c=_registration(config_ref,fixture_only=False,authority_anchors=authority_anchors)
    lineage=proof(c['disjointness_check'],c['source_pins'],purpose='ARITHMETIC_AVAILABLE_TRAIN_DEV_LINEAGE_DISJOINT',
        fixture_only=False,baseline_id=c['baseline_id'])
    chronology=proof(c['chronology_check'],c['source_pins'],purpose='ROSTER_FROZEN_BEFORE_CAPTURE',
        fixture_only=False,baseline_id=c['baseline_id'])
    roster=read_ref(c['roster'])
    require({d['domain_id'] for d in roster['domains']}=={'arithmetic','retrieval','answerability_abstention','dialogue','long_context'}
        and roster.get('candidate_outputs_used') is False,'exact complete prospectively frozen roster before capture')
    require(lineage.get('plan_file_sha256')==c['plan']['sha256'],'same source-aware disjoint capture population')
    return seal({'kind':'ARITHMETIC_PRECAPTURE_SOURCE_ADMISSION001','registration':deepcopy(config_ref),
        'baseline_id':c['baseline_id'],'roster_reference':c['roster'],'disjointness_evidence':lineage,'chronology_evidence':chronology,
        'native_model_runtime_resource_qualification_required':True,'native_launch_authorized':False})

def score_prepared(prepared):
    integrated();value=_native_value(prepared)
    guard=_fixed_scorer(value)
    return _score(value,guard)
def _fixed_scorer(value):
    path=PACKAGE/'arithmetic_guard.py';raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==SCORER_SOURCE_SHA256
            and value['config']['source_pins'].get(str(path))==SCORER_SOURCE_SHA256,'fixed reviewed arithmetic scorer bytes')
    module=ModuleType('fresh_source_arithmetic_scorer');module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__);return module
def _score(value,guard):
    scores={};details=[]
    for raw_row,label in zip(value['imported']['rows'],value['labels']):
        require(raw_row['row_id']==label['row_id'],'source qualified complete response order')
        raw=raw_row['raw'];require(raw['finish_reason'] in ('eos','generation_cap'),'unchanged full finish categories')
        grade=guard.score(label,raw['decoded_response'],value['policy'])
        try:answer=parse(raw['decoded_response'].encode());abstained=type(answer) is dict and set(answer)=={'answer'} and answer['answer'] is None
        except (ValueError,TypeError,RecursionError):abstained=False
        scores[label['row_id']]=grade['correct'];details.append({'row_id':label['row_id'],'finish_reason':raw['finish_reason'],
            'at_generation_cap':raw['finish_reason']=='generation_cap','abstained':abstained,**grade})
    return seal({'kind':'SOURCE_QUALIFIED_ARITHMETIC_M0_SCORES001','baseline_id':value['config']['baseline_id'],
        'domain':value['domain'],'scores':scores,'details':details,'qualification':value['checks']['scorer_check'],
        'fixture_only':value['config']['fixture_only'],'config_value_sha256':digest(value['config']),
        'raw_import_sha256':digest(value['imported']),'roster_reference':deepcopy(value['config']['roster']),
        'labels_reference':deepcopy(value['config']['labels']),'native_launch_authorized':False,'all_domain_retention_qualified':False})

def enroll_registered(prepared,scored,all_domain_score_ref,all_domain_proof_specs,*,enrollment_chronology_check=None):
    integrated();value=_native_value(prepared)
    require(scored==_score(value,_fixed_scorer(value)),'actual native score packet must reproduce whole responses/source context')
    proof(enrollment_chronology_check,value['config']['source_pins'],purpose='ENROLLMENT_BEFORE_TREATMENT',
        fixture_only=False,baseline_id=value['config']['baseline_id'])
    return _enroll(value,scored,all_domain_score_ref,all_domain_proof_specs)
def _enroll(value,scored,score_ref,specs,*,statistics=None):
    c=value['config'];domains=value['roster']['domains'];require(scored['baseline_id']==c['baseline_id'] and scored['domain']==value['domain'],'same complete M0 arithmetic contribution')
    require(scored.get('sha256')==digest({k:v for k,v in scored.items() if k!='sha256'})
            and scored.get('fixture_only') is c['fixture_only'] and scored.get('config_value_sha256')==digest(c)
            and scored.get('raw_import_sha256')==digest(value['imported'])
            and scored.get('roster_reference')==c['roster'] and scored.get('labels_reference')==c['labels'],
            'sealed source/config/raw/roster/labels-bound M0 score producer')
    require(set(specs)=={d['domain_id'] for d in domains},'all declared domain source/score qualification proofs required')
    bindings={}
    for domain in domains:
        evidence=proof(specs[domain['domain_id']],c['source_pins'],purpose='ALL_DOMAIN_M0_SCORING_QUALIFIED',fixture_only=c['fixture_only'],baseline_id=c['baseline_id'])
        require(evidence.get('domain_id')==domain['domain_id'] and evidence.get('roster_file_sha256')==c['roster']['sha256']
                and evidence.get('score_file_sha256')==score_ref['sha256'],'exact complete source/score/roster domain evidence')
        bindings[domain['domain_id']]={k:evidence[k] for k in ('qualified','scorer_sha256','labels_sha256','qualification_receipt_sha256')}
    scores=read_ref(score_ref);require(all(scores.get(r)==v for r,v in scored['scores'].items()),'same arithmetic M0 grades in all-domain score file')
    if statistics is None:
        if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
        path=PACKAGE/'dev_statistics.py';raw=path.read_bytes()
        require(c['source_pins'].get(str(path))==hashlib.sha256(raw).hexdigest(),'reviewed original statistics source bytes')
        require(hashlib.sha256(raw).hexdigest()=='2851ee7f7f2aa2399b7030a05b2bda14e85a8eff98ac23f71e7d918014d3059c'
            and hashlib.sha256((PACKAGE/'DEV_STATISTICS_POLICY.json').read_bytes()).hexdigest()=='9db137bfa6c0e0e07ec8161e4127ec0a2ec20bb2c224529face447986e876d11',
            'fixed original statistics implementation and policy')
        statistics=ModuleType('fresh_source_original_statistics');statistics.__file__=str(path)
        exec(compile(raw,str(path),'exec'),statistics.__dict__)
    result=statistics.freeze_guard_enrollment(domains,scores,baseline_id=c['baseline_id'],scorer_bindings=bindings)
    return seal({'kind':'REGISTERED_ARITHMETIC_ALL_DOMAIN_ENROLLMENT001','enrollment':result,
        'fixture_only':c['fixture_only'],'native_launch_authorized':False,'precision_qualified':False,
        'all_domain_retention_qualified':False,'chronology_evidence':value['checks']['chronology_check']})

def _CPU_prepare(config_ref,loader,*,capability):_check_cap(capability);return _prepare(config_ref,fixture_only=True,loader=loader)
def _CPU_score(prepared,scorer,*,capability):
    _check_cap(capability);value=_value(prepared);require(value['config']['fixture_only'] is True,'CPU prepared only');return _score(value,scorer)
def _CPU_enroll(prepared,scored,score_ref,specs,statistics,*,capability):
    _check_cap(capability);value=_value(prepared);require(value['config']['fixture_only'] is True,'CPU enrollment only');return _enroll(value,scored,score_ref,specs,statistics=statistics)
