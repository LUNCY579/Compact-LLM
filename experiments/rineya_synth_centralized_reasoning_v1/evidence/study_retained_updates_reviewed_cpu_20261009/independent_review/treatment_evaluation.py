"""Closed final-treatment raw collection/import preparation.

CPU fixture captures are authored, not model outputs. Complete source-bound
27/540 adapter protocol publications and original384/96 public prompt lineage
are joined before any grader-facing records exist. Original M0 importer and its
native M1/M2 rejection remain unchanged. Native producer/audit wiring is
conditional on a separately trusted packet; this stage cannot dispatch or score.
"""
from pathlib import Path
from types import ModuleType,FunctionType
from copy import deepcopy
import ast,hashlib,importlib,json,math,os,sys

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'treatment_evaluation_v1'
SCOPE='CPU_AUTHORED_TREATMENT_COLLECTION_IMPORT_FIXTURE_ONLY'
HELPERS=('final_adapter_inference.py','score_import.py','inference_component.py','study_schedule.py','study_native_chunk.py','native_loader.py','native_mechanics.py','native_precision.py','study_torch_backend.py','dev_statistics.py','native_contract.py','short_collection_worker.py')

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()

def seal(value):return {**value,'sha256':digest(value)}

def _factory():
    token=object()
    def get():return token
    def check(candidate):require(candidate is token,'private CPU treatment fixture capability')
    return get,check
_CPU_capability,_check_CPU_capability=_factory()

class _Prepared:
    def __init__(self,value,*,_capability):
        _check_CPU_capability(_capability);self.value=deepcopy(value);self.original=digest(value)

def _helpers():
    # This imports source-only helpers. Frameworks are not imported by them.
    if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
    modules={}
    for name in ('final_adapter_inference','score_import','inference_component'):
        module=importlib.import_module(name)
        require(Path(module.__file__).resolve()==PACKAGE/(name+'.py'),'original treatment helper source origin')
        modules[name]=module
    pins={str(PACKAGE/name):hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() for name in HELPERS}
    return modules,pins

def _shared_raw_join(adapter,expected,batch,ordinal,arm,*,fixture_only=True):
    """Execute the unchanged existing importer inner row contract, fixture only.

    This is no substitute for native batch audit or loader authority. The full
    original prepare_import native M1/M2 rejection is deliberately untouched.
    """
    tree=ast.parse(Path(adapter.__file__).read_bytes())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_import')
    wanted=ast.dump(ast.parse("zip(expected, batch['outputs'])",mode='eval').body,include_attributes=False)
    matches=[n for n in ast.walk(function) if isinstance(n,ast.For) and ast.dump(n.iter,include_attributes=False)==wanted]
    require(len(matches)==1,'one original shared row/prompt/finish contract')
    joined=[];namespace={**adapter.__dict__,'scope':adapter.FIXTURE if fixture_only else adapter.NATIVE,'expected':deepcopy(expected),'batch':deepcopy(batch),'ordinal':ordinal,'arm':arm,'joined':joined}
    exec(compile(ast.Module(body=[deepcopy(matches[0])],type_ignores=[]),adapter.__file__,'exec'),namespace)
    return joined

class _CPUSources:
    def __init__(self,training_registration,panel,finish,guard_policy,*,_capability):
        _check_CPU_capability(_capability)
        require(type(training_registration) is dict and training_registration.get('CPU_authored') is True and training_registration.get('sha256')==digest({k:v for k,v in training_registration.items() if k!='sha256'}),'authored-only CPU training schedule')
        require(panel.get('scope')=='CPU_SYNTHETIC_SCORE_IMPORT_FIXTURE' and len(panel['rows'])==384 and len({r['component_id'] for r in panel['rows']})==96 and all(r['parent_id'].startswith('fixture_') and r['component_id'].startswith('component_') for r in panel['rows']),'authored-only CPU384/96 panel namespace')
        require(type(guard_policy) is dict and guard_policy.get('CPU_authored') is True and guard_policy.get('retention',{}).get('required_domain_classes')==['task','language','long_context'],'authored-only CPU guard policy')
        self.value=deepcopy({'training_registration':training_registration,'panel':dict(panel),'finish':finish,'guard_policy':guard_policy});self.original=digest(self.value)

def _CPU_sources_fixture(training_registration,panel,finish,guard_policy,*,_capability):
    return _CPUSources(training_registration,panel,finish,guard_policy,_capability=_capability)

def _fixture_documents(sources):
    require(type(sources) is _CPUSources and digest(sources.value)==sources.original,'original authored CPU source capability')
    return deepcopy(sources.value)

def _panel_read(value,adapter):
    if value.get('fixture_only') is True:
        documents=value.get('fixture_documents');require(type(documents) is dict,'explicit authored panel source required')
        return deepcopy(documents['panel'])
    return adapter.load_frozen_panel(PACKAGE)

def _verify_chain(restore,registration,arm,chain,latest,*,fixture_documents=None):
    verifier=restore.verify_final_chain
    if fixture_documents is not None:
        require(fixture_documents['training_registration']==registration and registration.get('CPU_authored') is True and registration['sha256']==digest({k:v for k,v in registration.items() if k!='sha256'}),'same authored CPU schedule; real source schedule denied')
        # Execute the unchanged original final-chain function with its one
        # registration supplier replaced explicitly. Publication/checkpoint,
        # arm/binding/dose/cost verification functions remain original.
        supplier=ModuleType('CPU_authored_schedule_supplier');supplier.__dict__.update(restore.schedule.__dict__)
        def validate(doc):require(doc==fixture_documents['training_registration'],'authored CPU registration source drift')
        supplier.validate_registration=validate
        verifier=FunctionType(verifier.__code__,{**verifier.__globals__,'schedule':supplier},verifier.__name__,verifier.__defaults__,verifier.__closure__)
    try:return verifier(deepcopy(registration),arm,deepcopy(chain),deepcopy(latest))
    except (KeyError,TypeError,IndexError) as error:raise ValueError('incomplete original final-study chain structure') from error

def _CPU_prepare(registration,arm,chain,latest_terminal_costs,*,fixture_sources,_capability):
    _check_CPU_capability(_capability);require(arm in ('M1','M2'),'adapted same-arm treatment only')
    documents=_fixture_documents(fixture_sources)
    modules,pins=_helpers();restore=modules['final_adapter_inference'];adapter=modules['score_import']
    # Original verifier reads only CPU-authored fixture archive/publication bytes;
    # it does not deserialize tensors, instantiate a model or restore an adapter.
    descriptor,binding,recipe=_verify_chain(restore,registration,arm,chain,latest_terminal_costs,fixture_documents=documents)
    panel=deepcopy(documents['panel'])
    require(len(panel['rows'])==384 and len({row['component_id'] for row in panel['rows']})==96,'original complete384/96 treatment population')
    for row in panel['rows']:modules['inference_component'].validate_row(row['input'])
    policy=documents['finish'];require(policy['population']['row_order_sha256']==panel['row_order_sha256'] and policy['population']['scorer_sources']==panel['scorer_sources'],'authored finish/scorer/panel fixture binding')
    value={'scope':SCOPE,'arm':arm,'training_registration':deepcopy(registration),'chunk_chain':deepcopy(chain),'latest_terminal_costs':deepcopy(latest_terminal_costs),'checkpoint_descriptor':descriptor,'final_binding':binding,'recipe':recipe,'chain_sha256':digest(chain),'panel':panel,'source_pins':pins,'fixture_documents':documents,'finish_policy':{'path':'CPU_AUTHORED_POLICY_NOT_NATIVE','sha256':digest(policy),'body_sha256':policy['sha256']},'fixture_only':True,'native_launch_authorized':False,'scoring_authorized':False,'quality_gain_established':False}
    return _Prepared(value,_capability=_capability)

def _value(prepared,_capability):
    _check_CPU_capability(_capability);require(type(prepared) is _Prepared and digest(prepared.value)==prepared.original,'original unmodified CPU treatment plan')
    value=deepcopy(prepared.value)
    require(value['scope']==SCOPE and value['fixture_only'] is True and value['native_launch_authorized'] is value['scoring_authorized'] is value['quality_gain_established'] is False,'CPU plan never native capture/scoring authority')
    for path,pin in value['source_pins'].items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,'selected treatment helper source unchanged')
    require(digest(value['fixture_documents']['finish'])==value['finish_policy']['sha256'],'authored finish policy unchanged')
    return value

def _producer_body(value,assets_factory,call,guard,preserve,*,restore_function,component_type,validate_input,fixture_only):
    """Lazy final adapter restore -> unchanged greedy component collection.

    Native wiring supplies original restore_final/InferenceComponent/input check;
    the separate CPU capability path supplies inert actors for those boundaries.
    No loader/math is recreated, optimizer/RNG checkpoint state is never reused.
    """
    require(all(callable(x) for x in (assets_factory,call,guard,preserve,restore_function,component_type,validate_input)),'explicit lazy asset/journal/guard/preservation boundaries')
    modules,_=_helpers();restore=modules['final_adapter_inference'];adapter=modules['score_import']
    descriptor,binding,recipe=_verify_chain(restore,value['training_registration'],value['arm'],value['chunk_chain'],value['latest_terminal_costs'],fixture_documents=value.get('fixture_documents'))
    require((descriptor,binding,recipe)==(value['checkpoint_descriptor'],value['final_binding'],value['recipe']),'complete same-arm final chain before any asset factory')
    require(value['panel']==_panel_read(value,adapter),'whole384 panel before any asset factory')
    guard('treatment_before_asset_factory');assets=assets_factory(call=call,guard=guard);guard('treatment_after_asset_factory')
    require(type(assets) is dict and set(assets)=={'torch','model','reader','decoder','source_hashes'},'exact lazy inference asset bundle')
    restored=restore_function(assets['torch'],assets['model'],assets['reader'],call,guard,arm=value['arm'],registration=value['training_registration'],chunk_chain=value['chunk_chain'],latest_terminal_costs=value['latest_terminal_costs'],source_hashes=assets['source_hashes'])
    require(type(restored) is dict and restored.get('inference_allowed') is True and restored.get('arm')==value['arm'] and restored.get('checkpoint_sha256')==descriptor['sha256'] and restored.get('final_binding')==binding and restored.get('optimizer_restored') is restored.get('training_RNG_restored') is False,'successful exact final-factor restore only; no optimizer/RNG reuse')
    require(restored.get('CPU_fixture') is fixture_only,'fixture/native restore evidence cannot substitute')
    if not fixture_only:require(restored.get('real_final_study_chain_verified') is True,'original real-study chain restore required')
    rows=value['panel']['rows'];inputs=[row['input'] for row in rows]
    for row in inputs:validate_input(assets['model'],row,restored)
    max_positions=sum(sum(row['prompt_tokens']+step for step in range(32)) for row in inputs)
    component=component_type(assets['torch'],assets['model'],assets['decoder'],call,guard,preserve,max_attempted_forwards=384*32,max_input_positions=max_positions,context_cap=512,generation_cap=32,vocabulary_size=151936,device='cpu' if fixture_only else 'cuda:0')
    batches=[]
    for ordinal in range(1,7):
        expected=rows[(ordinal-1)*64:ordinal*64];packet=[row['input'] for row in expected];packet_hash=adapter.digest(packet)
        collection=component.collect(deepcopy(packet),source_packet_sha256=packet_hash)
        require(type(collection) is dict and len(collection['rows'])==64 and collection['source_packet_sha256']==packet_hash,'whole unchanged component treatment batch')
        batch={'ordinal':ordinal,'inputs':deepcopy(packet),'outputs':deepcopy(collection['rows']),'source_packet_sha256':packet_hash}
        _shared_raw_join(adapter,expected,batch,ordinal,value['arm'],fixture_only=fixture_only);batches.append(batch)
        preserve('treatment_batch_%02d'%ordinal,deepcopy(batch))
    return batches,restored

def _wired_native_producer(config_path,external_anchors):
    prepared=run_registered(config_path,external_anchors)
    value=deepcopy(prepared.value);modules,_=_helpers();restore=modules['final_adapter_inference'];component=modules['inference_component']
    boundaries=_factory_boundaries(value['registration_config'],prepared)
    batches,restored=_producer_body(value,boundaries['assets_factory'],boundaries['call'],boundaries['guard'],boundaries['preserve'],restore_function=restore.restore_final,component_type=component.InferenceComponent,validate_input=restore.validate_raw_input,fixture_only=False)
    collection=_collection_envelope(prepared,batches,restored,fixture_only=False)
    boundaries['preserve']('treatment_collection_complete',collection)
    return collection


def _CPU_collect(prepared,assets_factory,call,guard,preserve,*,restore_function,component_type,validate_input,_capability):
    """Same restore/collection ordering with authored inert actors only."""
    value=_value(prepared,_capability)
    batches,restored=_producer_body(value,assets_factory,call,guard,preserve,restore_function=restore_function,component_type=component_type,validate_input=validate_input,fixture_only=True)
    _value(prepared,_capability)
    return seal({'scope':SCOPE,'arm':value['arm'],'plan_sha256':prepared.original,'chain_sha256':value['chain_sha256'],'checkpoint_sha256':value['checkpoint_descriptor']['sha256'],'training_registration_sha256':value['training_registration']['sha256'],'restore_receipt':restored,'batches':batches,'fixture_only':True,'native_launch_authorized':False,'scoring_authorized':False,'quality_gain_established':False})

def _CPU_prepare_import(prepared,collection,*,_capability):
    value=_value(prepared,_capability);require(type(collection) is dict and collection.get('sha256')==digest({k:v for k,v in collection.items() if k!='sha256'}),'complete authored collection checksum')
    require(collection['scope']==SCOPE and collection['fixture_only'] is True and collection['native_launch_authorized'] is collection['scoring_authorized'] is collection['quality_gain_established'] is False,'synthetic collection never native evidence')
    require(collection['arm']==value['arm'] and collection['plan_sha256']==prepared.original and collection['chain_sha256']==value['chain_sha256'] and collection['checkpoint_sha256']==value['checkpoint_descriptor']['sha256'] and collection['training_registration_sha256']==value['training_registration']['sha256'],'same-arm final adapter/source/exposure chain and capture provenance')
    restored=collection['restore_receipt']
    require(type(restored) is dict and restored.get('arm')==value['arm'] and restored.get('checkpoint_sha256')==value['checkpoint_descriptor']['sha256'] and restored.get('final_binding')==value['final_binding'] and restored.get('inference_allowed') is True and restored.get('CPU_fixture') is True and restored.get('optimizer_restored') is restored.get('training_RNG_restored') is False,'same authored final-factor restore and disjoint fixture provenance')
    modules,_=_helpers();restore=modules['final_adapter_inference'];adapter=modules['score_import']
    descriptor,binding,recipe=_verify_chain(restore,value['training_registration'],value['arm'],value['chunk_chain'],value['latest_terminal_costs'],fixture_documents=value.get('fixture_documents'))
    require((descriptor,binding,recipe)==(value['checkpoint_descriptor'],value['final_binding'],value['recipe']),'same complete final chain reread; no rollback or byte drift')
    require(value['panel']==_panel_read(value,adapter),'same authored384 prompt/scorer metadata')
    batches=collection['batches'];require(type(batches) is list and len(batches)==6,'whole384 requires six complete64 treatment batches')
    joined=[]
    for ordinal,batch in enumerate(batches,1):
        expected=value['panel']['rows'][(ordinal-1)*64:ordinal*64]
        require(batch['ordinal']==ordinal and batch['inputs']==[row['input'] for row in expected] and len(batch['outputs'])==64 and batch['source_packet_sha256']==adapter.digest(batch['inputs']),'whole original treatment batch/prompt/order')
        joined.extend(_shared_raw_join(adapter,expected,batch,ordinal,value['arm']))
    require(len(joined)==384 and adapter.digest([row['row_id'] for row in joined])==value['panel']['row_order_sha256'],'complete original treatment row lineage')
    return seal({'scope':SCOPE,'arm':value['arm'],'status':'CPU_AUTHORED_TREATMENT_PRE_SCORE_JOIN_COMPLETE_NATIVE_GATE_CLOSED','rows':joined,'panel_metadata_sha256':value['panel']['panel_metadata_sha256'],'row_order_sha256':value['panel']['row_order_sha256'],'panel_files':value['panel']['panel_files'],'scorer_sources':value['panel']['scorer_sources'],'statistics_sources':value['panel']['statistics_sources'],'finish_policy':value['finish_policy'],'final_binding':binding,'checkpoint_descriptor':descriptor,'chain_sha256':value['chain_sha256'],'training_registration_sha256':value['training_registration']['sha256'],'collection_sha256':collection['sha256'],'fixture_only':True,'native_launch_authorized':False,'scoring_authorized':False,'quality_gain_established':False})


NATIVE_SCOPE='REGISTERED_FINAL_TREATMENT_RAW_COLLECTION_UNSCORED_V1'
PERMISSIONS={'model':True,'final_adapter_restore':True,'DEV_raw_collection':True,'optimizer':False,'training':False,'scoring':False,'protected_bank_access':False,'paid_services':False}
REG_FIELDS={'schema_version','kind','scope','fixture_only','arm','training_registration','chunk_chain','checkpoint_sha256','latest_terminal_costs','panel_registration_sha256','finish_policy','source_pins','model_pins','resources','tolerances','approval','adapter_qualification','m0_baseline','enrollment','factory_source','factory_name','output_directory','sha256'}
REF_FIELDS={'path','bytes','sha256','value_sha256'}
APPROVAL_FIELDS={'kind','scope','arm','fixture_only','approved','permissions','checkpoint_sha256','chain_sha256','source_pins','factory_source','factory_name','output_directory','human_evidence_reference'}

def _enrollment_check(stats,enrollment,documents):
    if documents is None:return stats._validate_enrollment(enrollment)
    # CPU arithmetic uses an explicit authored supplier; original globals stay intact.
    freeze=stats.freeze_guard_enrollment
    freeze=FunctionType(freeze.__code__,{**freeze.__globals__,'policy':lambda:deepcopy(documents['guard_policy'])},freeze.__name__,freeze.__defaults__,freeze.__closure__)
    freeze.__kwdefaults__=stats.freeze_guard_enrollment.__kwdefaults__
    check=stats._validate_enrollment
    check=FunctionType(check.__code__,{**check.__globals__,'freeze_guard_enrollment':freeze},check.__name__,check.__defaults__,check.__closure__)
    return check(enrollment)

def _selected_source(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'},'exact guard scorer source descriptor')
    path=Path(ref['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink() and type(ref['bytes']) is int and 0<ref['bytes']<=1024**2,'bounded selected guard scorer source')
    with path.open('rb') as stream:data=stream.read(ref['bytes']+1)
    require(len(data)==ref['bytes'] and hashlib.sha256(data).hexdigest()==ref['sha256'],'actual selected guard scorer source bytes')
    return ref['sha256']

def _pinned_json(ref):
    require(type(ref) is dict and set(ref)==REF_FIELDS,'exact externally selected treatment JSON descriptor')
    path=Path(ref['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink(),'selected treatment JSON path')
    require(type(ref['bytes']) is int and 0<ref['bytes']<=16*1024**2,'bounded selected treatment JSON')
    with path.open('rb') as stream:data=stream.read(ref['bytes']+1)
    require(len(data)==ref['bytes'] and hashlib.sha256(data).hexdigest()==ref['sha256'],'selected treatment JSON bytes unchanged')
    # Reuse the existing strict duplicate/nonfinite parser, no grader.
    modules,_=_helpers();value=modules['score_import']._json(data)
    require(digest(value)==ref['value_sha256'],'selected treatment JSON value unchanged')
    return value

def _source_closure():
    return {str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**{str(PACKAGE/name):hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() for name in HELPERS}}

def _m0_native_baseline(value,panel,adapter):
    fields={'kind','fixture_only','panel_registration_sha256','row_order_sha256','panel_metadata_sha256','batches','baseline_id','scorer_sources','scoring_registration','guard_scorer_evidence','sha256'}
    require(type(value) is dict and set(value)==fields and value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}) and value['kind']=='COMPLETE_SOURCE_BOUND_M0_BASELINE_BEFORE_TREATMENT_V1' and value['fixture_only'] is False,'actual complete M0 baseline proof required')
    require(value['panel_registration_sha256']==panel['sha256'] and value['row_order_sha256']==panel['row_order_sha256'] and value['panel_metadata_sha256']==panel['panel_metadata_sha256'] and value['scorer_sources']==panel['scorer_sources'],'same frozen whole384 M0 baseline/scorer population')
    require(type(value['batches']) is list and len(value['batches'])==6,'complete six independently audited M0 batches required')
    batches=[]
    for row in value['batches']:
        require(type(row) is dict and set(row)=={'preserved_directory','audit_path','expected_audit_sha256','expected_config_sha256'},'original independently pinned M0 loader anchors')
        batches.append(adapter.load_m0_batch(**row))
    joined=adapter.prepare_import(panel,batches,arm='M0')
    require(len(joined['rows'])==384,'original native M0 complete384 import')
    scoring=_pinned_json(value['scoring_registration'])
    require(scoring.get('kind')=='INDEPENDENT_COMPLETE_M0_SCORING_BEFORE_TREATMENT_V1' and scoring.get('fixture_only') is False and scoring.get('raw_import_sha256')==joined['sha256'] and scoring.get('baseline_id')==value['baseline_id'] and scoring.get('scorer_sources')==panel['scorer_sources'] and scoring.get('candidate_outputs_used') is False,'independently selected complete M0 scoring provenance before treatment')
    return value['baseline_id']

def _validate_selected(config_path,external_anchors,*,fixture_only,m0_fixture=None,for_collection=True,fixture_sources=None):
    require(type(external_anchors) is dict and set(external_anchors)=={'config_file_sha256','config_value_sha256'},'external treatment config file/value anchors required')
    path=Path(config_path).absolute();require(path.is_file() and not path.is_symlink(),'exact treatment registration path')
    config=_pinned_json({'path':str(path),'bytes':path.stat().st_size,'sha256':external_anchors['config_file_sha256'],'value_sha256':external_anchors['config_value_sha256']})
    require(type(config) is dict and set(config)==REG_FIELDS and config['schema_version']==1 and config['kind']=='REGISTERED_FINAL_TREATMENT_COLLECTION_V1' and config['scope']==NATIVE_SCOPE and config['fixture_only'] is fixture_only and config['sha256']==digest({k:v for k,v in config.items() if k!='sha256'}),'exact future treatment registration; fixture/native separation')
    require(config['arm'] in ('M1','M2') and config['source_pins']==_source_closure(),'exact selected treatment implementation/source closure')
    modules,_=_helpers();restore=modules['final_adapter_inference'];adapter=modules['score_import'];native=importlib.import_module('native_contract');stats=importlib.import_module('dev_statistics')
    require(config['resources']==native.RESOURCE_BOUNDS and config['tolerances']==native.TOLERANCES and config['model_pins']==restore.loader.ASSET_PINS,'unchanged original resource/numerical/model metadata pins')
    approval=_pinned_json(config['approval'])
    require(type(approval) is dict and set(approval)==APPROVAL_FIELDS and approval.get('kind')=='OWNER_APPROVED_FINAL_TREATMENT_COLLECTION_V1' and approval.get('scope')==NATIVE_SCOPE and approval.get('arm')==config['arm'] and approval.get('fixture_only') is fixture_only and approval.get('approved') is True and approval.get('permissions')==PERMISSIONS and approval.get('source_pins')==config['source_pins'] and approval.get('factory_source')==config['factory_source'] and approval.get('factory_name')==config['factory_name'] and approval.get('output_directory')==config['output_directory'] and type(approval.get('human_evidence_reference')) is str and bool(approval['human_evidence_reference']) and approval.get('chain_sha256')==config['chunk_chain'].get('value_sha256') and approval.get('checkpoint_sha256')==config['checkpoint_sha256'],'explicit exact treatment approval before checkpoint/panel/asset reads')
    documents=_fixture_documents(fixture_sources) if fixture_only else None
    training=_pinned_json(config['training_registration']);chain=_pinned_json(config['chunk_chain']);latest=_pinned_json(config['latest_terminal_costs'])
    require(type(chain) is list and len(chain)==3 and chain[-1]['envelope']['backend_descriptor']['sha256']==config['checkpoint_sha256'],'declared exact approved checkpoint before archive/panel reads')
    descriptor,binding,recipe=_verify_chain(restore,training,config['arm'],chain,latest,fixture_documents=documents)
    if not fixture_only:require(recipe.get('retained_update_policy')==restore.chunks.producer_policy(),'production treatment requires all27 original retained updates')
    panel=deepcopy(documents['panel']) if fixture_only else adapter.load_frozen_panel(PACKAGE)
    require(config['panel_registration_sha256']==panel['sha256'],'same declared whole384/96 public or authored population')
    finish=_pinned_json(config['finish_policy'])
    if fixture_only:require(finish==documents['finish'],'same authored finish policy fixture')
    else:require(config['finish_policy']['sha256']==hashlib.sha256((PACKAGE/'DEV_FINISH_POLICY_001.json').read_bytes()).hexdigest(),'original native finish policy byte pin')
    require(finish['population']['row_order_sha256']==panel['row_order_sha256'] and finish['population']['scorer_sources']==panel['scorer_sources'],'same finish policy and fixed scorer metadata')
    require(approval.get('checkpoint_sha256')==descriptor['sha256'] and approval.get('chain_sha256')==digest(chain),'approved exact final checkpoint/chain binding')
    qualification=_pinned_json(config['adapter_qualification'])
    required={'kind','fixture_only','arm','checkpoint_sha256','chain_sha256','training_registration_sha256','final_binding','source_pins','model_pins','resources','tolerances','restore_receipt','raw_collection_mechanics_verified','scoring_authorized','evidence','sha256'}
    require(type(qualification) is dict and set(qualification)==required and qualification['sha256']==digest({k:v for k,v in qualification.items() if k!='sha256'}) and qualification['kind']=='INDEPENDENT_FINAL_ADAPTER_COLLECTION_QUALIFICATION_V1' and qualification['fixture_only'] is fixture_only,'independent final-adapter qualification schema; historical zeroB/STATE1 scope denied')
    require(qualification['arm']==config['arm'] and qualification['checkpoint_sha256']==descriptor['sha256'] and qualification['chain_sha256']==digest(chain) and qualification['training_registration_sha256']==training['sha256'] and qualification['final_binding']==binding and qualification['source_pins']==config['source_pins'] and qualification['model_pins']==config['model_pins'] and qualification['resources']==config['resources'] and qualification['tolerances']==config['tolerances'] and qualification['raw_collection_mechanics_verified'] is True and qualification['scoring_authorized'] is False,'same source-bound final27/540 qualified adapter/collection scope')
    restored=qualification['restore_receipt'];_restore_join(restored,config['arm'],descriptor,binding,fixture_only=fixture_only)
    require(type(qualification['evidence']) is dict and set(qualification['evidence'])=={'independent_audit','preservation'},'selected final-adapter qualification original evidence anchors')
    audit=_pinned_json(qualification['evidence']['independent_audit']);preservation=_pinned_json(qualification['evidence']['preservation'])
    require(audit.get('kind')=='INDEPENDENT_FINAL_ADAPTER_CAPTURE_AUDIT_V1' and audit.get('fixture_only') is fixture_only and audit.get('arm')==config['arm'] and audit.get('checkpoint_sha256')==descriptor['sha256'] and audit.get('chain_sha256')==digest(chain) and audit.get('restore_receipt')==restored and audit.get('raw_collection_mechanics_verified') is True and audit.get('source_pins')==config['source_pins'] and audit.get('resources')==config['resources'] and audit.get('tolerances')==config['tolerances'] and audit.get('preservation_value_sha256')==digest(preservation),'original independent final-adapter restore/trace/resource/preservation audit joins')
    require(preservation.get('kind')=='PRESERVED_FINAL_ADAPTER_CAPTURE_QUALIFICATION_V1' and preservation.get('fixture_only') is fixture_only and preservation.get('arm')==config['arm'] and preservation.get('checkpoint_sha256')==descriptor['sha256'] and preservation.get('source_pins')==config['source_pins'] and type(preservation.get('raw_files')) is dict and bool(preservation['raw_files']),'original final-adapter qualification raw/source preservation joins')
    for ref in preservation['raw_files'].values():_pinned_json(ref)
    m0=_pinned_json(config['m0_baseline'])
    if fixture_only:
        require(callable(m0_fixture) and m0.get('fixture_only') is True,'explicit authored complete-M0 fixture supplier')
        baseline_id=m0_fixture(m0,panel)
    else:
        require(m0_fixture is None,'no CPU supplier may enter native baseline validation');baseline_id=_m0_native_baseline(m0,panel,adapter)
    enrollment=_pinned_json(config['enrollment']);_enrollment_check(stats,enrollment,documents)
    require(enrollment['status']=='FROZEN_M0_MEMBERSHIP' and enrollment['issues']==[] and enrollment['missing_m0_row_ids']==[] and enrollment['candidate_outputs_used'] is False and enrollment['baseline_id']==baseline_id and set(enrollment['scorer_bindings'])=={d['domain_id'] for d in enrollment['domains']} and bool(enrollment['scorer_bindings']) and all(row['qualified'] is True for row in enrollment['scorer_bindings'].values()),'complete qualified source-correct M0 membership before treatment')
    domain_evidence=m0.get('guard_scorer_evidence');require(type(domain_evidence) is dict and set(domain_evidence)==set(enrollment['scorer_bindings']),'exact independently bound guard domain scorer evidence')
    for domain,bound in enrollment['scorer_bindings'].items():
        proof=_pinned_json(domain_evidence[domain])
        require(proof.get('domain_id')==domain and proof.get('baseline_id')==baseline_id and proof.get('fixture_only') is fixture_only and proof.get('source_correct_membership_scorer_binding')==bound and proof.get('scorer_source_sha256')==bound['scorer_sha256'] and proof.get('labels_sha256')==bound['labels_sha256'] and proof.get('qualification_receipt_sha256')==bound['qualification_receipt_sha256'],'actual selected guard scorer/source/label qualification binding')
        evidence=proof.get('evidence');require(type(evidence) is dict and set(evidence)=={'scorer_source','labels','qualification'},'selected guard source/labels/qualification evidence required')
        require(_selected_source(evidence['scorer_source'])==bound['scorer_sha256'] and evidence['labels']['sha256']==bound['labels_sha256'] and evidence['qualification']['sha256']==bound['qualification_receipt_sha256'],'guard bindings select actual source and receipt byte pins')
        labels=_pinned_json(evidence['labels']);qualified=_pinned_json(evidence['qualification'])
        require(labels.get('domain_id')==domain and labels.get('fixture_only') is fixture_only and qualified.get('kind')=='INDEPENDENT_GUARD_SCORER_QUALIFICATION_V1' and qualified.get('domain_id')==domain and qualified.get('fixture_only') is fixture_only and qualified.get('qualified') is True and qualified.get('scorer_source_sha256')==bound['scorer_sha256'] and qualified.get('labels_sha256')==bound['labels_sha256'],'selected domain scorer qualification/source/labels joins')
    factory=config['factory_source'];require(type(factory) is dict and set(factory)=={'path','bytes','sha256'} and type(config['factory_name']) is str and config['factory_name'].isidentifier(),'finite selected future owned factory source/name')
    factory_path=Path(factory['path']);require(factory_path.is_absolute() and factory_path.suffix=='.py' and factory_path.is_file() and not factory_path.is_symlink(),'selected future factory source')
    require(factory_path.stat().st_size==factory['bytes'] and 0<factory['bytes']<=1024**2 and hashlib.sha256(factory_path.read_bytes()).hexdigest()==factory['sha256'],'exact future asset/guard/journal factory source bytes')
    output=Path(config['output_directory']);require(output.is_absolute() and output.parent.is_dir() and output.parent.resolve()==output.parent,'externally selected treatment output parent')
    require((not output.exists()) if for_collection else (output.is_dir() and not output.is_symlink()),'fresh collection leaf or exact existing audit leaf')
    value={'scope':SCOPE if fixture_only else NATIVE_SCOPE,'arm':config['arm'],'training_registration':training,'chunk_chain':chain,'latest_terminal_costs':latest,'checkpoint_descriptor':descriptor,'final_binding':binding,'recipe':recipe,'chain_sha256':digest(chain),'panel':panel,'source_pins':config['source_pins'],'finish_policy':{'path':config['finish_policy']['path'],'sha256':config['finish_policy']['sha256'],'body_sha256':finish['sha256']},'fixture_documents':documents,'registration_config':config,'registration_external_anchors':deepcopy(external_anchors),'registration_path':str(path),'baseline_id':baseline_id,'enrollment_sha256':enrollment['sha256'],'fixture_only':fixture_only,'native_launch_authorized':False,'scoring_authorized':False,'quality_gain_established':False}
    return _Prepared(value,_capability=_CPU_capability())

def _restore_join(restored,arm,descriptor,binding,*,fixture_only):
    require(type(restored) is dict and restored.get('arm')==arm and restored.get('checkpoint_sha256')==descriptor['sha256'] and restored.get('final_binding')==binding and restored.get('inference_allowed') is True and restored.get('CPU_fixture') is fixture_only and restored.get('optimizer_restored') is restored.get('training_RNG_restored') is False and restored.get('native_launch_authorized') is False,'exact completed final-factor restore receipt; no permission transfer')
    if not fixture_only:require(restored.get('real_final_study_chain_verified') is True and restored.get('status')=='RESTORED_FINAL_ADAPTER_INFERENCE_UNQUALIFIED','original same real final adapter restore receipt')

def _factory_boundaries(config,prepared):
    # Called only after all external source/approval/qualification/enrollment joins.
    descriptor=config['factory_source'];path=Path(descriptor['path']);data=path.read_bytes()
    require(len(data)==descriptor['bytes'] and hashlib.sha256(data).hexdigest()==descriptor['sha256'],'selected lazy factory source unchanged after admission')
    module=ModuleType('selected_treatment_owned_factory');module.__file__=str(path)
    exec(compile(data,str(path),'exec'),module.__dict__);factory=module.__dict__.get(config['factory_name']);require(callable(factory),'selected owned factory callable')
    boundaries=factory(deepcopy(prepared.value))
    require(type(boundaries) is dict and set(boundaries)=={'assets_factory','call','guard','preserve'} and all(callable(v) for v in boundaries.values()),'exact owned lazy asset/resource/journal/preservation boundaries')
    return boundaries

def _collection_envelope(prepared,batches,restored,*,fixture_only):
    value=prepared.value
    return seal({'scope':SCOPE if fixture_only else NATIVE_SCOPE,'arm':value['arm'],'plan_sha256':prepared.original,'chain_sha256':value['chain_sha256'],'checkpoint_sha256':value['checkpoint_descriptor']['sha256'],'training_registration_sha256':value['training_registration']['sha256'],'restore_receipt':deepcopy(restored),'batches':deepcopy(batches),'fixture_only':fixture_only,'native_launch_authorized':False,'scoring_authorized':False,'quality_gain_established':False})

def run_registered(config_path=None,external_anchors=None,*,for_collection=True,**legacy):
    require(HERE==EXPECTED,'staged treatment collector has no native authority')
    return _validate_selected(config_path,external_anchors,fixture_only=False,for_collection=for_collection)

def _CPU_registered_fixture(config_path,external_anchors,*,m0_fixture,fixture_sources,_capability,for_collection=True):
    _check_CPU_capability(_capability)
    return _validate_selected(config_path,external_anchors,fixture_only=True,m0_fixture=m0_fixture,for_collection=for_collection,fixture_sources=fixture_sources)

def audit_raw_collection(config_path,external_anchors,collection_ref,*,expected_raw_files):
    """Independent original-byte/final-chain/384 raw lineage audit; no grading.

    This verifies preserved capture structure/provenance only. The selected
    qualification and owned terminal audit retain tensor/resource authority.
    """
    prepared=run_registered(config_path,external_anchors,for_collection=False)
    return _raw_consumer(prepared,_pinned_json(collection_ref),expected_raw_files,fixture_only=False)

def _raw_consumer(prepared,collection,expected_raw_files,*,fixture_only):
    value=deepcopy(prepared.value);modules,_=_helpers();restore=modules['final_adapter_inference'];adapter=modules['score_import']
    require(digest(value)==prepared.original,'original registered treatment plan unchanged')
    require(type(collection) is dict and collection.get('sha256')==digest({k:v for k,v in collection.items() if k!='sha256'}) and collection['scope']==(SCOPE if fixture_only else NATIVE_SCOPE) and collection['fixture_only'] is fixture_only,'same registered actual/synthetic collection scope/checksum')
    require(collection['arm']==value['arm'] and collection['plan_sha256']==prepared.original and collection['chain_sha256']==value['chain_sha256'] and collection['checkpoint_sha256']==value['checkpoint_descriptor']['sha256'] and collection['training_registration_sha256']==value['training_registration']['sha256'] and collection['native_launch_authorized'] is collection['scoring_authorized'] is collection['quality_gain_established'] is False,'same registered final chain/capture provenance; no permission claim')
    descriptor,binding,recipe=_verify_chain(restore,value['training_registration'],value['arm'],value['chunk_chain'],value['latest_terminal_costs'],fixture_documents=value.get('fixture_documents'))
    require((descriptor,binding,recipe)==(value['checkpoint_descriptor'],value['final_binding'],value['recipe']),'independent final publication/bytes/dose reread')
    _restore_join(collection['restore_receipt'],value['arm'],descriptor,binding,fixture_only=fixture_only)
    require(value['panel']==_panel_read(value,adapter),'independent384 scorer/prompt supplier reread')
    batches=collection['batches'];require(type(batches) is list and len(batches)==6,'independent whole384 requires six64 batches')
    require(type(expected_raw_files) is list and len(expected_raw_files)==384,'exact original384 preserved raw file anchors')
    joined=[];cursor=0
    for ordinal,batch in enumerate(batches,1):
        expected=value['panel']['rows'][(ordinal-1)*64:ordinal*64]
        require(batch['ordinal']==ordinal and batch['inputs']==[r['input'] for r in expected] and len(batch['outputs'])==64 and batch['source_packet_sha256']==adapter.digest(batch['inputs']),'same independent frozen treatment batch/order')
        rows=_shared_raw_join(adapter,expected,batch,ordinal,value['arm'],fixture_only=fixture_only)
        for row in rows:
            raw=row['raw'];require(_pinned_json(expected_raw_files[cursor])==raw,'actual preserved final raw row bytes/terminal join');cursor+=1
            trace=raw['trace'];require(type(trace) is list and len(trace)==len(raw['raw_generated_ids'])==len(raw['selected_logprobs']),'whole generated trace/IDs/LP population')
            for index,step in enumerate(trace):
                require(type(step) is dict and set(step)=={'token','logprob','prefix_length'} and type(step['token']) is int and step['token']==raw['raw_generated_ids'][index] and type(step['logprob']) in (int,float) and math.isfinite(step['logprob']) and step['logprob']<=0 and step['logprob']==raw['selected_logprobs'][index] and type(step['prefix_length']) is int and step['prefix_length']==row['input']['prompt_tokens']+index,'exact generated token/LP/prefix lineage')
        joined.extend(rows)
    require(len(joined)==384 and adapter.digest([r['row_id'] for r in joined])==value['panel']['row_order_sha256'],'independent whole original384/96 row lineage')
    return seal({'scope':SCOPE if fixture_only else NATIVE_SCOPE,'arm':value['arm'],'status':'RAW_TREATMENT_SOURCE_LINEAGE_VERIFIED_UNSCORED','rows':joined,'checkpoint_sha256':descriptor['sha256'],'final_binding':binding,'scorer_sources':value['panel']['scorer_sources'],'finish_policy':value['finish_policy'],'baseline_id':value.get('baseline_id'),'enrollment_sha256':value.get('enrollment_sha256'),'fixture_only':fixture_only,'native_PASS':False,'native_launch_authorized':False,'scoring_authorized':False,'quality_gain_established':False})

def _CPU_raw_consumer_fixture(prepared,collection,raw_files,*,_capability):
    _check_CPU_capability(_capability);require(prepared.value['fixture_only'] is True,'only authored CPU raw consumer fixture')
    return _raw_consumer(prepared,collection,raw_files,fixture_only=True)
