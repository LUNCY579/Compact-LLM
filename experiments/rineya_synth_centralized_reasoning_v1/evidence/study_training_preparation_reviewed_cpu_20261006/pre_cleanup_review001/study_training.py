"""Distinct prospective STUDY composition using original training/resume math.

This ignored source location is not native dispatch authority. A future native
packet must be approved and independently reviewed before any training data,
model/framework/optimizer or checkpoint read. Existing qualification workers and
their STATE1/checkpoints are never used as STUDY state.
"""
from copy import deepcopy
from pathlib import Path
from types import ModuleType,FunctionType
import hashlib,importlib,json,math,sys

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'study_training_v1'
SCOPE='REGISTERED_STUDY_ONE_FRESH_NINE_UPDATE_CHUNK_V1'
CPU_SCOPE='AUTHORED_INERT_STUDY_CHUNK_COMPOSITION_ONLY'
PERMISSIONS={'training':True,'model':True,'optimizer':True,'scoring':False,'protected_bank_access':False,'paid_services':False,'promotion':False}
HELPERS=('study_schedule.py','study_native_chunk.py','study_torch_backend.py','native_loader.py','native_mechanics.py','native_precision.py','native_contract.py','checkpoint_successor_profile.py','dev_statistics.py')
REF_FIELDS={'path','bytes','sha256','value_sha256'}
FIELDS={'kind','scope','fixture_only','arm','chunk','source_pins','model_pins','resources','tolerances','approval','review','training_registration','training_qualification_check','m0_scoring_check','guard_enrollment_check','guard_enrollment','baseline_id','owned_boundary_source','owned_boundary_name','model_directory','output_directory','previous_resume','latest_terminal_costs','sha256'}
APPROVAL={'kind','scope','fixture_only','approved','permissions','arm','chunk','source_pins','training_registration_value_sha256','baseline_id','owned_boundary_source','owned_boundary_name','model_directory','output_directory','previous_resume_value_sha256','human_evidence_reference'}
REVIEW={'kind','scope','fixture_only','accepted','source_pins','approval_file_sha256','training_registration_value_sha256','training_qualification_check','m0_scoring_check','guard_enrollment_check','baseline_id'}

def require(ok,message):
    if not ok:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
def seal(value):return {**value,'sha256':digest(value)}
def _pairs(items):
    result={}
    for key,value in items:require(key not in result,'duplicate selected JSON key');result[key]=value
    return result
def _bad(value):raise ValueError('nonfinite selected JSON')
def read_json(ref):
    require(type(ref) is dict and set(ref)==REF_FIELDS,'exact selected JSON descriptor')
    path=Path(ref['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink() and type(ref['bytes']) is int and 0<ref['bytes']<=16*1024**2,'bounded selected JSON path')
    with path.open('rb') as stream:raw=stream.read(ref['bytes']+1)
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'selected JSON bytes unchanged')
    value=json.loads(raw,object_pairs_hook=_pairs,parse_constant=_bad);require(digest(value)==ref['value_sha256'],'selected JSON value unchanged');return value
def read_source(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'},'exact selected source descriptor')
    path=Path(ref['path']);require(path.is_absolute() and path.suffix=='.py' and path.is_file() and not path.is_symlink() and type(ref['bytes']) is int and 0<ref['bytes']<=1024**2,'bounded selected Python source')
    with path.open('rb') as stream:raw=stream.read(ref['bytes']+1)
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'selected source bytes unchanged');return raw
def source_pins():return {str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**{str(PACKAGE/n):hashlib.sha256((PACKAGE/n).read_bytes()).hexdigest() for n in HELPERS}}
def _helpers():
    if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
    modules={n:importlib.import_module(n) for n in ('study_schedule','study_native_chunk','study_torch_backend','native_loader','native_contract','dev_statistics','checkpoint_successor_profile')}
    for name,module in modules.items():require(Path(module.__file__).resolve()==PACKAGE/(name+'.py'),'original STUDY helper source origin')
    return modules
def _load_selected(ref,pins):
    require(pins.get(ref['path'])==ref['sha256'],'selected independent/owned source belongs to approved closure')
    raw=read_source(ref);module=ModuleType('selected_source_'+ref['sha256'][:16]);module.__file__=ref['path'];sys.modules[module.__name__]=module
    try:exec(compile(raw,ref['path'],'exec'),module.__dict__)
    except BaseException:sys.modules.pop(module.__name__,None);raise
    return module
def _independent_check(spec,pins,*,fixture_only,purpose,baseline_id):
    require(type(spec) is dict and set(spec)=={'source','function','request','expected_result','purpose'},'exact independently replayed evidence request')
    require(spec['purpose']==purpose and type(spec['function']) is str and spec['function'].isidentifier(),'independent evidence purpose/function')
    request=read_json(spec['request']);expected=read_json(spec['expected_result']);require(type(request) is dict,'independent audit keyword request')
    module=_load_selected(spec['source'],pins);function=getattr(module,spec['function'],None);require(callable(function),'selected original independent auditor callable')
    actual=function(**deepcopy(request));require(actual==expected,'original independent auditor replay differs from selected evidence')
    require(type(actual) is dict and actual.get('fixture_only') is fixture_only and actual.get('purpose')==purpose and actual.get('baseline_id')==baseline_id and actual.get('qualified') is True,'qualified independent purpose/baseline/source evidence')
    return deepcopy(actual)
def _cap_factory():
    token=object()
    def get():return token
    def check(value):require(value is token,'explicit private authored CPU capability')
    return get,check
_CPU_capability,_check_cap=_cap_factory()
class _CPUFixture:
    def __init__(self,registration,pools,records,guard_policy,*,capability):
        _check_cap(capability);require(registration.get('CPU_authored') is True,'explicit authored schedule')
        self.value=deepcopy({'registration':registration,'pools':pools,'records':records,'guard_policy':guard_policy});self.original=digest(self.value)
def _CPU_fixture(registration,pools,records,guard_policy,*,capability):return _CPUFixture(registration,pools,records,guard_policy,capability=capability)
def _fixture_value(fixture):require(type(fixture) is _CPUFixture and digest(fixture.value)==fixture.original,'unmodified authored source supplier');return deepcopy(fixture.value)
def _fixture_modules(modules,documents):
    schedule=ModuleType('explicit_authored_study_schedule');schedule.__dict__.update(modules['study_schedule'].__dict__)
    def inputs():return {},{'schedules':{a:[x for u in documents['registration']['arms'][a]['updates'] for x in u['occurrence_row_ids']] for a in ('M1','M2')}},deepcopy(documents['pools']),deepcopy(documents['records'])
    def validate(registration):
        require(registration==documents['registration'],'authored schedule changed');_,src,pools,records=inputs()
        for arm in ('M1','M2'):schedule.audit_chunks(arm,registration['arms'][arm],src['schedules'][arm],pools[arm],records)
    schedule._inputs=inputs;schedule.validate_registration=validate
    chunks=ModuleType('original_chunk_with_explicit_authored_supplier');chunks.__file__=str(PACKAGE/'study_native_chunk.py')
    exec(compile((PACKAGE/'study_native_chunk.py').read_bytes(),chunks.__file__,'exec'),chunks.__dict__);chunks.schedule=schedule
    return {**modules,'study_schedule':schedule,'study_native_chunk':chunks}
def _enrollment(modules,value,documents):
    stats=modules['dev_statistics']
    if documents is None:stats._validate_enrollment(value)
    else:
        freeze=stats.freeze_guard_enrollment;freeze=FunctionType(freeze.__code__,{**freeze.__globals__,'policy':lambda:deepcopy(documents['guard_policy'])},freeze.__name__,freeze.__defaults__,freeze.__closure__);freeze.__kwdefaults__=stats.freeze_guard_enrollment.__kwdefaults__
        check=stats._validate_enrollment;check=FunctionType(check.__code__,{**check.__globals__,'freeze_guard_enrollment':freeze},check.__name__,check.__defaults__,check.__closure__);check(value)
    require(value['status']=='FROZEN_M0_MEMBERSHIP' and not value['issues'] and not value['missing_m0_row_ids'] and value['candidate_outputs_used'] is False and set(value['scorer_bindings'])=={d['domain_id'] for d in value['domains']} and bool(value['scorer_bindings']) and all(x['qualified'] is True for x in value['scorer_bindings'].values()),'complete all-domain source-correct M0 enrollment required')
def _validate(config_path,anchors,*,fixture=None):
    fixture_only=fixture is not None
    require(type(anchors) is dict and set(anchors)=={'config_file_sha256','config_value_sha256'},'externally selected STUDY file/value anchors')
    path=Path(config_path).absolute();ref={'path':str(path),'bytes':path.stat().st_size,'sha256':anchors['config_file_sha256'],'value_sha256':anchors['config_value_sha256']};config=read_json(ref)
    require(type(config) is dict and set(config)==FIELDS and config['kind']=='REGISTERED_STUDY_TRAINING_CHUNK_V1' and config['scope']==SCOPE and config['fixture_only'] is fixture_only and config['sha256']==digest({k:v for k,v in config.items() if k!='sha256'}),'exact STUDY packet fixture/native separation')
    require(config['arm'] in ('M1','M2') and type(config['chunk']) is int and config['chunk'] in (1,2,3),'one adapted arm and one nine-update chunk')
    approval=read_json(config['approval']);require(type(approval) is dict and set(approval)==APPROVAL and approval['kind']=='OWNER_STUDY_TRAINING_CHUNK_AUTHORIZATION_V1' and approval['scope']==SCOPE and approval['fixture_only'] is fixture_only and approval['approved'] is True and approval['permissions']==PERMISSIONS and all(approval[k]==config[k] for k in ('arm','chunk','source_pins','baseline_id','owned_boundary_source','owned_boundary_name','model_directory','output_directory')) and approval['training_registration_value_sha256']==config['training_registration']['value_sha256'] and approval['previous_resume_value_sha256']==(None if config['previous_resume'] is None else config['previous_resume']['value_sha256']) and type(approval['human_evidence_reference']) is str and bool(approval['human_evidence_reference']),'exact approved STUDY scope before training data/checkpoint/assets')
    review=read_json(config['review']);require(type(review) is dict and set(review)==REVIEW and review['kind']=='INDEPENDENT_STUDY_TRAINING_CHUNK_REVIEW_V1' and review['scope']==SCOPE and review['fixture_only'] is fixture_only and review['accepted'] is True and review['approval_file_sha256']==config['approval']['sha256'] and review['training_registration_value_sha256']==config['training_registration']['value_sha256'] and all(review[k]==config[k] for k in ('source_pins','training_qualification_check','m0_scoring_check','guard_enrollment_check','baseline_id')),'independently reviewed exact STUDY packet before data/assets')
    pins=source_pins();require(all(config['source_pins'].get(k)==v for k,v in pins.items()),'current original STUDY source bytes')
    modules=_helpers();require(config['resources']==modules['native_contract'].RESOURCE_BOUNDS and config['tolerances']==modules['native_contract'].TOLERANCES and config['model_pins']==modules['native_loader'].ASSET_PINS,'unchanged frozen model/resources/numerical metadata')
    qualified=_independent_check(config['training_qualification_check'],config['source_pins'],fixture_only=fixture_only,purpose='STUDY_TRAINING_FULL27_AND_RESUME_QUALIFIED',baseline_id=config['baseline_id'])
    require(qualified.get('full27_qualified') is True and qualified.get('full_resumed9_qualified') is True and qualified.get('study_checkpoint_used') is False,'real sustained full27/resumed9 qualification; no qualification checkpoint initialization')
    m0=_independent_check(config['m0_scoring_check'],config['source_pins'],fixture_only=fixture_only,purpose='COMPLETE_M0_384_96_SCORED_BEFORE_TREATMENT',baseline_id=config['baseline_id'])
    require(m0.get('rows')==384 and m0.get('components')==96 and m0.get('candidate_outputs_used') is False,'qualified scored complete384/96 M0 before treatment')
    guards=_independent_check(config['guard_enrollment_check'],config['source_pins'],fixture_only=fixture_only,purpose='ALL_REQUIRED_GUARD_SCORERS_SOURCE_CORRECT_BEFORE_TREATMENT',baseline_id=config['baseline_id'])
    documents=_fixture_value(fixture) if fixture_only else None
    enrollment=read_json(config['guard_enrollment']);_enrollment(modules,enrollment,documents)
    require(enrollment['baseline_id']==config['baseline_id'] and guards['enrollment_sha256']==enrollment['sha256'],'independent source-correct guard enrollment binding')
    require(guards.get('scorer_bindings')==enrollment['scorer_bindings'] and set(guards.get('qualified_domain_ids',[]))=={d['domain_id'] for d in enrollment['domains']},'independent evidence covers exact all-domain scorer/source qualification set')
    if fixture_only:modules=_fixture_modules(modules,documents)
    registration=read_json(config['training_registration']);modules['study_schedule'].validate_registration(registration)
    require(registration['arms']['M1']['totals']['parent_exposures']==registration['arms']['M2']['totals']['parent_exposures'],'matched source-parent exposure vector')
    recipe=modules['study_native_chunk'].recipe(registration,config['arm'],modules['study_native_chunk'].FAKE_SCOPE if fixture_only else modules['study_native_chunk'].NATIVE_SCOPE)
    previous=latest=None
    if config['chunk']==1:require(config['previous_resume'] is config['latest_terminal_costs'] is None,'fresh unchanged base; no qualification checkpoint warmstart')
    else:
        require(config['previous_resume'] is not None and config['latest_terminal_costs'] is not None,'exact successful previous STUDY and latest costs required')
        previous=read_json(config['previous_resume']);latest=read_json(config['latest_terminal_costs']);modules['study_native_chunk']._validate_resume(previous,registration,config['arm'],config['chunk'],recipe,latest)
    output=Path(config['output_directory']);require(output.is_absolute() and output.parent.is_dir() and not output.exists(),'fresh selected STUDY output leaf')
    return {'config':config,'registration':registration,'recipe':recipe,'previous':previous,'latest':latest,'modules':modules,'documents':documents,'fixture_only':fixture_only,'independent_checks':{'training':qualified,'m0':m0,'guards':guards}}
class _NativeFactory:
    def __init__(self,prepared,boundaries):self.prepared,self.boundaries=prepared,boundaries;self.execution_scope=prepared['modules']['study_native_chunk'].NATIVE_SCOPE;self.backend=None;self.profiles=[]
    def __call__(self,*,call,guard,arm,seed,recipe_sha256,cohort_sha256):
        value=self.prepared;config=value['config'];modules=value['modules']
        require(arm==config['arm'] and seed==101 and recipe_sha256==value['recipe']['sha256'] and cohort_sha256==value['recipe']['cohort_sha256'],'same admitted STUDY arm/seed/recipe')
        guard('study_before_framework_import');torch,config_class,model_class=call('study_framework_import',self.boundaries['frameworks']);guard('study_after_framework_import')
        model,loader_receipt,hashes=modules['native_loader'].load_titan(torch,config_class,model_class,config['model_directory'],call,guard)
        self.boundaries['evidence']('study_original_loader',loader_receipt)
        def construct():return modules['study_torch_backend'].StudyTorchBackend(torch,model,call,guard,arm=arm,seed=seed,recipe_sha256=recipe_sha256,cohort_sha256=cohort_sha256,execution_scope=self.execution_scope,source_hashes=hashes)
        self.backend=call('study_original_backend_construct',construct)
        def checkpoint_profile():
            from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop
            return modules['checkpoint_successor_profile'].install(model,torch,checkpoint,early_stop_context=set_checkpoint_early_stop,expected_layers=28,call=call,guard=guard)
        profile=call('study_original_checkpoint_profile_install',checkpoint_profile);self.profiles.append(profile)
        self.boundaries['evidence']('study_full_checkpoint_profile',profile['snapshot']())
        return self.backend
    def close(self):
        errors=[]
        for profile in reversed(self.profiles):
            try:self.boundaries['evidence']('study_checkpoint_profile_final',profile['snapshot']())
            except BaseException as error:errors.append(error)
            try:profile['remove']()
            except BaseException as error:errors.append(error)
        self.backend=None;self.profiles=[]
        if errors:raise RuntimeError('owned STUDY checkpoint profile cleanup failed') from errors[0]
def _boundaries(prepared):
    config=prepared['config'];module=_load_selected(config['owned_boundary_source'],config['source_pins']);function=getattr(module,config['owned_boundary_name'],None);require(callable(function),'reviewed original owned boundary builder')
    value=function(deepcopy({k:v for k,v in prepared.items() if k not in ('modules','documents')}))
    fields={'frameworks','call','guard','preserve','admission','evidence','cleanup','terminal_audit'}
    require(type(value) is dict and set(value)==fields and all(callable(x) for x in value.values()),'exact owned admission/journal/resource/cleanup/audit boundaries');return value
def _run(prepared,factory,boundaries):
    modules=prepared['modules'];config=prepared['config'];_,_,pools,_=modules['study_schedule']._inputs()
    try:
        result=modules['study_native_chunk'].run_chunk(prepared['registration'],config['arm'],config['chunk'],factory,pools[config['arm']].__getitem__,boundaries['call'],boundaries['guard'],boundaries['preserve'],resume_descriptor=prepared['previous'],latest_terminal_costs=prepared['latest'],admission=boundaries['admission'])
    finally:
        try:
            if callable(getattr(factory,'close',None)):factory.close()
        finally:boundaries['cleanup']()
    terminal=boundaries['terminal_audit'](deepcopy(result));require(type(terminal) is dict,'independent owned terminal audit receipt')
    # Preserve exact scientific resume object; native/lifetime evidence stays outside it.
    return seal({'scope':CPU_SCOPE if prepared['fixture_only'] else SCOPE,'arm':config['arm'],'chunk':config['chunk'],'protocol_result':result,'terminal_audit':terminal,'independent_preconditions':prepared['independent_checks'],'fixture_only':prepared['fixture_only'],'native_PASS':False,'quality_gain_established':False,'automatic_resume_authorized':False})
def run_registered(config_path,external_anchors):
    require(HERE==EXPECTED,'staged STUDY worker has no native authority')
    prepared=_validate(config_path,external_anchors);boundaries=_boundaries(prepared);factory=_NativeFactory(prepared,boundaries);return _run(prepared,factory,boundaries)
def _CPU_run_registered(config_path,external_anchors,fixture,backend_factory,boundaries,*,capability):
    _check_cap(capability);prepared=_validate(config_path,external_anchors,fixture=fixture)
    require(callable(backend_factory) and type(boundaries) is dict,'explicit inert authored CPU actors')
    factory=backend_factory(prepared,boundaries);require(factory.execution_scope==prepared['modules']['study_native_chunk'].FAKE_SCOPE,'only original inert CPU protocol')
    return _run(prepared,factory,boundaries)
