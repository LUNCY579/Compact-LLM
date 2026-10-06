"""Conditional task768 M0 capture routing and audited pre-score import.

The original short collector already owns64-row collection. This adapter does
not relabel the TRAIN-only native-long qualification as a long guard producer,
nor qualify a language scorer. Staging rejects native use before any reads.
"""
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
from types import ModuleType
import ast,hashlib,importlib,json,sys

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'task_guard_pipeline_v1'
SCOPE='REGISTERED_TASK768_M0_RAW_CAPTURE_AND_UNSCORED_IMPORT_V1'
DOMAIN='synth_task_retention001'
KINDS=('retrieval','instruction_following','answerability_abstention')
REF={'path','bytes','sha256','value_sha256'}
CONFIG={'kind','scope','fixture_only','source_pins','collector_source_pins','model_pins','approval','review','population','batch_configs','capture_anchors','row_order_sha256','sha256'}
PERMISSION={'M0_task_raw_collection':True,'task_raw_import':True,'scoring':False,'enrollment':False,'M1_M2':False,'long_context':False,'language':False,'training':False,'protected_bank_access':False,'paid_services':False}
def require(ok,message):
    if not ok:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
def seal(value):return {**value,'sha256':digest(value)}
def pairs(items):
    out={}
    for k,v in items:require(k not in out,'duplicate selected task JSON');out[k]=v
    return out
def bad(value):raise ValueError('nonfinite task JSON')
def read_ref(ref):
    require(type(ref) is dict and set(ref)==REF,'exact selected task JSON descriptor')
    path=Path(ref['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink() and type(ref['bytes']) is int and 0<ref['bytes']<=16*1024**2,'bounded selected original task JSON')
    with path.open('rb') as stream:raw=stream.read(ref['bytes']+1)
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'task original byte descriptor')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad);require(digest(value)==ref['value_sha256'],'task original value descriptor');return value
def _helpers():
    if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
    modules={name:importlib.import_module(name) for name in ('score_import','inference_component')}
    for name,module in modules.items():require(Path(module.__file__).resolve()==PACKAGE/(name+'.py'),'original task helper source origin')
    return modules
def _cap_factory():
    token=object()
    def get():return token
    def check(value):require(value is token,'private authored task pipeline capability')
    return get,check
_CPU_capability,_check_cap=_cap_factory()

def validate_registration(config,*,fixture_only):
    require(type(config) is dict and set(config)==CONFIG and config['kind']=='REGISTERED_TASK_GUARD_M0_PIPELINE_V1' and config['scope']==SCOPE and config['fixture_only'] is fixture_only and config['sha256']==digest({k:v for k,v in config.items() if k!='sha256'}),'exact task-only M0 registration')
    approval=read_ref(config['approval']);review=read_ref(config['review'])
    fields={'kind','scope','fixture_only','approved','permissions','source_pins','collector_source_pins','model_pins','population_value_sha256','batch_config_sha256s','capture_anchors_sha256','human_evidence_reference'}
    require(type(approval) is dict and set(approval)==fields and approval['kind']=='OWNER_TASK_GUARD_M0_PIPELINE_AUTHORIZATION_V1' and approval['scope']==SCOPE and approval['fixture_only'] is fixture_only and approval['approved'] is True and approval['permissions']==PERMISSION and approval['source_pins']==config['source_pins'] and approval['collector_source_pins']==config['collector_source_pins'] and approval['model_pins']==config['model_pins'] and approval['capture_anchors_sha256']==digest(config['capture_anchors']) and approval['population_value_sha256']==config['population']['value_sha256'] and approval['batch_config_sha256s']==[r['sha256'] for r in config['batch_configs']] and type(approval['human_evidence_reference']) is str and bool(approval['human_evidence_reference']),'exact separate task-only permission before population/captures/frameworks')
    require(type(review) is dict and set(review)=={'kind','scope','fixture_only','accepted','source_pins','collector_source_pins','model_pins','approval_file_sha256','population_value_sha256','batch_config_sha256s','capture_anchors_sha256'} and review['kind']=='INDEPENDENT_TASK_GUARD_M0_PIPELINE_SOURCE_REVIEW_V1' and review['scope']==SCOPE and review['fixture_only'] is fixture_only and review['accepted'] is True and review['source_pins']==config['source_pins'] and review['collector_source_pins']==config['collector_source_pins'] and review['model_pins']==config['model_pins'] and review['capture_anchors_sha256']==approval['capture_anchors_sha256'] and review['approval_file_sha256']==config['approval']['sha256'] and review['population_value_sha256']==config['population']['value_sha256'] and review['batch_config_sha256s']==approval['batch_config_sha256s'],'independently reviewed exact task population/dose/source before reads')
    if not fixture_only:
        required={str(PACKAGE/name) for name in ('score_import.py','short_collection_contract.py','short_collection_worker.py','short_collection_supervisor.py','inference_component.py')}|{str(Path(__file__).resolve())}
        require(required<=set(config['source_pins']),'complete original task producer/importer source closure before helpers')
    for path,pin in config['source_pins'].items():require(Path(path).suffix=='.py' and hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,'selected task source bytes unchanged')
    for path,pin in config['collector_source_pins'].items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,'exact approved original collector byte closure')
    require(len(config['batch_configs'])==12 and len(config['capture_anchors']) in (0,12),'original twelve fixed64 task scopes')
    return config

def population(value,validate_row):
    require(type(value) is dict and set(value)=={'rows','metadata','domain'} and len(value['rows'])==len(value['metadata'])==768,'whole768 original prompt/metadata population')
    rows=value['rows'];meta=value['metadata'];domain=value['domain'];ids=[r['row_id'] for r in rows]
    require(len(set(ids))==768 and ids==[m['row_id'] for m in meta],'unique ordered original task metadata/token rows')
    require(type(domain) is dict and set(domain)=={'domain_id','class','row_ids','component_ids'} and domain['domain_id']==DOMAIN and domain['class']=='task' and domain['row_ids']==ids and domain['component_ids']==[m['component_id'] for m in meta],'original task domain and exact whole component membership')
    require(Counter(m['mechanism'] for m in meta)=={k:256 for k in KINDS} and len(set(domain['component_ids']))==192,'original task three mechanisms/192world quotas')
    groups=defaultdict(list)
    for row,item in zip(rows,meta):
        validate_row(row);require(type(item) is dict and set(item)=={'row_id','component_id','mechanism','variant','rendering'},'task metadata includes no oracle/labels');groups[item['component_id']].append(item)
    variants={}
    for group in groups.values():
        require(len(group)==4 and len({m['mechanism'] for m in group})==1 and {m['rendering'] for m in group}=={'json_lines','typed_ledger'} and len({m['variant'] for m in group})==2 and len({(m['variant'],m['rendering']) for m in group})==4,'all task world variant/rendering pairs')
        kind=group[0]['mechanism'];selected={m['variant'] for m in group};require(kind not in variants or variants[kind]==selected,'same original per-mechanism variant inventory');variants[kind]=selected
    return deepcopy(value)

def _row_code(adapter):
    function=next(n for n in ast.parse(Path(adapter.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='prepare_import');wanted=ast.dump(ast.parse("zip(expected, batch['outputs'])",mode='eval').body,include_attributes=False);loops=[n for n in ast.walk(function) if isinstance(n,ast.For) and ast.dump(n.iter,include_attributes=False)==wanted];require(len(loops)==1,'one unchanged original raw row loop');return compile(ast.Module(body=[deepcopy(loops[0])],type_ignores=[]),adapter.__file__,'exec')
def import_loaded(config,source,batches,adapter,*,fixture_only):
    require(len(batches)==12,'whole768 requires twelve complete audited64 task batches');joined=[];references=[];code=_row_code(adapter);model_pins=None
    for ordinal,batch in enumerate(batches,1):
        rows=source['rows'][(ordinal-1)*64:ordinal*64];metadata=source['metadata'][(ordinal-1)*64:ordinal*64]
        require(batch['arm']=='M0' and batch['source_kind']=='task_guard_DEV' and type(batch['ordinal']) is int and batch['ordinal']==ordinal and batch['inputs']==rows and len(batch['outputs'])==64 and batch['source_packet_sha256']==adapter.digest(rows) and batch['audit_passed'] is True,'same original audited task-only M0 whole64 source/order')
        if not fixture_only:require(type(batch) is adapter._LoadedM0Batch,'actual original independently pinned M0 loader handle, not Boolean batch flags')
        refs=batch['references'];require(refs['model_pins']==config['model_pins'] and refs['source_pins']==config['collector_source_pins'],'exact approved model/source closure all task captures');references.append(deepcopy(refs));require(model_pins is None or model_pins==refs['model_pins'],'same unchanged frozen M0 assets all task batches');model_pins=refs['model_pins']
        expected=[{'row_id':r['row_id'],'input':r,'guard_metadata':m} for r,m in zip(rows,metadata)]
        namespace={**adapter.__dict__,'FIXTURE':'CPU_FAKE_COMPONENT' if fixture_only else adapter.FIXTURE,'scope':'CPU_FAKE_COMPONENT' if fixture_only else adapter.NATIVE,'expected':expected,'batch':batch,'ordinal':ordinal,'arm':'M0','joined':joined};exec(code,namespace)
    require(len(joined)==768 and digest([r['row_id'] for r in joined])==config['row_order_sha256'],'complete768 original task order')
    return seal({'scope':SCOPE,'arm':'M0','rows':joined,'domain':source['domain'],'batch_references':references,'row_order_sha256':config['row_order_sha256'],'fixture_only':fixture_only,'native_PASS':False,'scoring_authorized':False,'enrollment_authorized':False,'language_qualification_established':False,'long_context_qualification_established':False,'quality_gain_established':False})

def selected_batch_configs(config,source):
    values=[]
    for ordinal,ref in enumerate(config['batch_configs'],1):
        selected=read_ref(ref);rows=source['rows'][(ordinal-1)*64:ordinal*64]
        require(selected['batch_selector']=={'source_kind':'task_guard_DEV','ordinal':ordinal} and selected['rows']==rows and selected['source_packet_sha256']==digest(rows) and selected['model_role']=='fresh_dense_M0_zero_B_only_no_STATE1_or_study_checkpoint' and selected['model_pins']==config['model_pins'] and selected['source_pins']==config['collector_source_pins'],'exact original selected task64 source/model/population')
        require(selected['dose']['rows']==64 and selected['dose']['context_cap']==512 and selected['dose']['maximum_generated_tokens']==32,'unchanged original context512/64/gen32 dose');values.append(selected)
    return values
def _prepare(config_ref,*,fixture_only,loader=None):
    config=validate_registration(read_ref(config_ref),fixture_only=fixture_only);helpers=_helpers();source=population(read_ref(config['population']),helpers['inference_component'].validate_row)
    require(digest([r['row_id'] for r in source['rows']])==config['row_order_sha256'],'registered full task order')
    selected_batch_configs(config,source)
    require(len(config['capture_anchors'])==12,'independently audited twelve task capture anchors required')
    batches=[]
    for ordinal,anchor in enumerate(config['capture_anchors'],1):
        require(type(anchor) is dict and set(anchor)=={'preserved_directory','audit_path','expected_audit_sha256','expected_config_sha256'} and anchor['expected_config_sha256']==config['batch_configs'][ordinal-1]['sha256'],'exact ordered approved capture/config anchor before loader')
        batch=(helpers['score_import'].load_m0_batch(**anchor) if loader is None else loader(anchor));require(batch['references']['config']['sha256']==anchor['expected_config_sha256'] and batch['references']['audit']['sha256']==anchor['expected_audit_sha256'],'same approved independent audit/config receipts');batches.append(batch)
    if not fixture_only:
        for batch in batches:require(batch==helpers['score_import'].load_m0_batch(**batch._anchor),'original native loader handle recheck before import')
    return import_loaded(config,source,batches,helpers['score_import'],fixture_only=fixture_only)
def prepare_task_import(config_ref):
    require(HERE==EXPECTED,'staged task pipeline has no native import authority');return _prepare(config_ref,fixture_only=False)
def dispatch_task_batch(config_ref,ordinal,output_directory):
    require(HERE==EXPECTED,'staged task pipeline has no native capture authority');config=validate_registration(read_ref(config_ref),fixture_only=False)
    require(type(ordinal) is int and 1<=ordinal<=12,'one original task64 batch ordinal')
    helpers=_helpers();source=population(read_ref(config['population']),helpers['inference_component'].validate_row);selected_batch_configs(config,source)
    if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
    original=importlib.import_module('short_collection_supervisor');require(Path(original.__file__).resolve()==PACKAGE/'short_collection_supervisor.py','original qualified short capture dispatcher source origin')
    return original.dispatch(config['batch_configs'][ordinal-1]['path'],output_directory)
def prepare_long_import(*args,**kwargs):raise ValueError('long2304 DEV capture/decoder/owned worker and audited batch loader absent; TRAIN qualification cannot substitute')
def _CPU_prepare(config_ref,loader,*,capability):
    _check_cap(capability);require(callable(loader),'explicit authored captured batch supplier');return _prepare(config_ref,fixture_only=True,loader=loader)
