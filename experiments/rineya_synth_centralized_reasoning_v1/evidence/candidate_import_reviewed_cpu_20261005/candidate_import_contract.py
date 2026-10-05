"""CPU synthetic candidate joins only. Native interfaces remain unavailable.

No model/framework/checkpoint execution/decoder/grader/process API exists here.
Source-faithful synthetic fixtures prove metadata/byte-join mechanics only.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import hashlib,json,math,stat,sys,weakref
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
sys.path.insert(0,str(PACKAGE));import score_import as legacy
FIXTURE='CPU_SYNTHETIC_CANDIDATE_IMPORT_FIXTURE'
STUDY='SYNTH_STUDY_CHUNK_PROTOCOL_V1';ARCHIVE='SYNTH_STUDY_TORCH_BOUNDARY_V1';CPU_STUDY='CPU_PROTOCOL_FAKE'
MAX_JSON=16*1024**2;MAX_ARCHIVE=32*1024**2;MAX_LINE=65536;MAX_JSONL=32*1024**2
SPEC=PACKAGE/'evidence/candidate_import_spec_reviewed_20261005/PROPOSED_SCHEMAS.json'
SPEC_SHA='60a7dbee449231716a3362472cb7c5683f87009a42f07973ec78e77dacef3f45'
FIELDS={};_TOKEN=object();_HANDLES=weakref.WeakKeyDictionary()
def require(value,message):
    if not value:raise ValueError(message)
def canonical(value,study=False):return json.dumps(value,ensure_ascii=not study,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def study_digest(value):return hashlib.sha256(canonical(value,True).encode('utf-8')).hexdigest()
def seal(value,study=False):
    value=deepcopy(value);value.pop('sha256',None);value['sha256']=(study_digest if study else digest)(value);return value
def hashpin(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def pairs(items):
    result={}
    for key,value in items:require(key not in result,'duplicate JSON key');result[key]=value
    return result
def floating(text):
    value=float(text);require(math.isfinite(value),'finite JSON exponent');return value
def loads(raw):
    try:return json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_float=floating,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON constant')))
    except (UnicodeError,RecursionError) as error:raise ValueError('invalid bounded JSON') from error
def regular(path,root=None):
    path=Path(path);require(path.is_absolute(),'absolute selected file path');meta=path.lstat()
    require(stat.S_ISREG(meta.st_mode) and not path.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400),'regular non-reparse file')
    for parent in path.parents:
        m=parent.lstat();require(not parent.is_symlink() and not(getattr(m,'st_file_attributes',0)&0x400),'non-reparse file ancestors')
    if root is not None:require(path.resolve().is_relative_to(Path(root).resolve()),'owned fixture pointer cannot escape')
    return path,meta
def file_bytes(path,limit=MAX_JSON,root=None):
    path,before=regular(path,root);require(type(limit) is int and 0<limit<=MAX_ARCHIVE and 0<before.st_size<=limit,'bounded nonempty selected file')
    with path.open('rb') as stream:raw=stream.read(limit+1)
    require(len(raw)==before.st_size==path.stat().st_size,'complete stable selected bytes');return raw
def read_json(path,limit=MAX_JSON):return loads(file_bytes(Path(path).absolute(),limit))
def read_jsonl(path,max_lines=200000):
    path,before=regular(Path(path).absolute());require(before.st_size<=MAX_JSONL,'bounded JSONL file');values=[];total=0
    with path.open('rb') as stream:
        while True:
            raw=stream.readline(MAX_LINE+1)
            if not raw:break
            total+=len(raw);require(raw.endswith(b'\n') and len(raw)<=MAX_LINE and len(values)<max_lines,'complete bounded JSONL line/count')
            values.append(loads(raw))
    require(total==before.st_size==path.stat().st_size,'stable complete JSONL bytes');return values
def raw_file_anchor(path):
    path=Path(path).absolute();raw=file_bytes(path,MAX_ARCHIVE)
    return {'path':str(path),'bytes':len(raw),'file_sha256':hashlib.sha256(raw).hexdigest()}
CONVENTIONS={'ASCII_SEALED_EXCLUDE_OWN_SHA256','ASCII_UNSEALED_FULL_DOCUMENT','UTF8_STUDY_SEALED_EXCLUDE_OWN_SHA256','UTF8_STUDY_UNSEALED_FULL_DOCUMENT'}
def body_hash(value,convention):
    require(convention in CONVENTIONS,'known producer-role JSON body convention')
    sealed='SEALED_EXCLUDE' in convention
    if sealed:
        require(type(value) is dict and hashpin(value.get('sha256')),'actual sealed producer checksum required')
        body={k:v for k,v in value.items() if k!='sha256'}
    else:body=value
    result=(study_digest if convention.startswith('UTF8') else digest)(body)
    if sealed:require(value['sha256']==result,'producer body checksum matches role convention')
    return result
def json_anchor(path,convention):
    anchor=raw_file_anchor(path);value=loads(file_bytes(Path(path).absolute()))
    return {**anchor,'body_sha256':body_hash(value,convention),'body_hash_convention':convention}
def fields(value,names,label):require(type(value) is dict and set(value)==set(names),'exact '+label+' field schema')
def checked_seal(value,study=False):
    require(type(value) is dict and hashpin(value.get('sha256')) and value['sha256']==(study_digest if study else digest)({k:v for k,v in value.items() if k!='sha256'}),'exact canonical body seal')
def _spec():
    raw=file_bytes(SPEC);require(hashlib.sha256(raw).hexdigest()==SPEC_SHA,'exact reviewed candidate specification bytes')
    return {k:set(v['fields']) for k,v in loads(raw)['schemas'].items()}
FIELDS=_spec()
class ReadSet:
    def __init__(self,root):self.root=Path(root).resolve();self.files={};self.rosters={}
    def read(self,path,expected,limit=MAX_ARCHIVE):
        raw=file_bytes(path,limit,self.root);anchor={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        require(hashpin(expected) and anchor['sha256']==expected,'original externally selected file SHA')
        key=str(Path(path).absolute());require(key not in self.files or self.files[key]==anchor,'never refresh a first-read byte baseline')
        self.files.setdefault(key,anchor);directory=Path(path).absolute().parent;name=str(directory)
        roster=sorted(p.name for p in directory.iterdir());require(name not in self.rosters or self.rosters[name]==roster,'immutable owned directory roster')
        self.rosters.setdefault(name,roster);return raw
    def anchor(self,anchor,*,json_document=False):
        fields(anchor,FIELDS['JsonDocumentAnchor' if json_document else 'PlainFileAnchor'],'selected anchor')
        require(type(anchor['bytes']) is int and anchor['bytes']>0,'positive nonboolean byte count')
        raw=self.read(anchor['path'],anchor['file_sha256'],MAX_JSON if json_document else MAX_ARCHIVE)
        require(len(raw)==anchor['bytes'],'exact selected file byte count')
        if not json_document:return raw
        value=loads(raw);require(body_hash(value,anchor['body_hash_convention'])==anchor['body_sha256'],'selected body role/hash');return value
    def verify(self):
        for path,anchor in self.files.items():
            raw=file_bytes(path,MAX_ARCHIVE,self.root);require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==anchor,'entire original selected read-set stable')
        for path,roster in self.rosters.items():require(sorted(p.name for p in Path(path).iterdir())==roster,'entire selected directory roster stable')
class _Handle:
    __slots__=('_value','_hash','_readset','_kind','__weakref__')
    def __init__(self,value,kind,readset=None,*,token):
        require(token is _TOKEN,'fixture loader factory required');self._value=deepcopy(value);self._hash=digest(self._value);self._kind=kind;self._readset=readset
        maps=None if readset is None else digest({'files':readset.files,'rosters':readset.rosters})
        _HANDLES[self]=(self._hash,kind,readset,maps)
    def __copy__(self):raise ValueError('factory handles cannot be copied')
    def __deepcopy__(self,memo):raise ValueError('factory handles cannot be copied')
    def __reduce__(self):raise ValueError('factory handles cannot be serialized')
def value(handle,kind):
    require(type(handle) is _Handle and handle in _HANDLES,'exact registered fixture factory instance; cloned attributes are not a handle')
    original,original_kind,readset,maps=_HANDLES[handle]
    require(handle._kind==kind==original_kind and digest(handle._value)==handle._hash==original and handle._readset is readset,
        'unmodified exact fixture factory body/context/anchors')
    if readset is not None:
        require(digest({'files':readset.files,'rosters':readset.rosters})==maps,'immutable factory first-read baseline')
        readset.verify()
    return deepcopy(handle._value)
def load_native_selection(*args,**kwargs):raise ValueError('native study/collector audit interfaces absent; native loading and authority closed before reads')
def load_native_candidate_batch(*args,**kwargs):raise ValueError('native study/collector audit interfaces absent; native loading and authority closed before reads')
def rates():return [float(Fraction(1,20000)),float(Fraction(1,10000))]+[float(Fraction(1,10000)*Fraction(27-i,25)) for i in range(3,28)]
def binding(registration,arm,completed):
    require(type(completed) is int and completed in (9,18,27),'exact complete nine-update endpoint')
    updates=registration['arms'][arm]['updates'][:completed];ids=[r for u in updates for r in u['occurrence_row_ids']]
    return {'completed_updates':completed,'consumed_occurrences':completed*20,'input_tokens':sum(u['input_tokens'] for u in updates),
        'supervised_tokens':sum(u['supervised_tokens'] for u in updates),'consumed_order_sha256':study_digest(ids),
        'next_update_ordinal':completed+1,'learning_rate':rates()[completed-1]}
def study_recipe(registration,arm):
    def source(name):return hashlib.sha256(file_bytes(PACKAGE/name)).hexdigest()
    return seal({'schema_version':1,'registration_sha256':registration['sha256'],'arm':arm,'seed':101,
        'cohort_sha256':registration['arms'][arm]['occurrence_order_sha256'],'source_pins':deepcopy(registration['source_and_decision_pins']),
        'execution_scope':CPU_STUDY,'chunk_source_sha256':source('study_native_chunk.py'),'backend_source_sha256':source('study_torch_backend.py')},study=True)
PROTOCOL={'gradient_reset':27,'forward':540,'completion_loss':540,'backward':540,'finite_gradient_clip':27,'optimizer_step':27,
    'checkpoint_save':3,'checkpoint_reload':2,'checkpoint_publish':3,'terminal_costs_publish':3,'checkpoint_promote':3}
def counts(n):return {'attempted':n,'returned':n,'failed':0,'pending':0}
def counters(values):
    require(type(values) is dict,'exact conserved operation dictionary')
    for name,item in values.items():
        fields(item,counts(0),'operation counts');require(type(name) is str and name and all(type(v) is int and v>=0 for v in item.values())
            and item['attempted']==item['returned'] and item['failed']==item['pending']==0,'all actual operation returns conserved; no failed/pending')
def costs(cost,b,ordinal,posted=False):
    fields(cost,{'schema_version','scope','protocol','positions','primitive_operations'},'study costs')
    require(type(cost['schema_version']) is int and cost['schema_version']==1 and cost['scope']==STUDY and set(cost['protocol'])==set(PROTOCOL),'study-only exact protocol cost roster')
    counters(cost['protocol']);counters(cost['primitive_operations'])
    n=ordinal
    expected={'gradient_reset':9*n,'forward':180*n,'completion_loss':180*n,'backward':180*n,'finite_gradient_clip':9*n,'optimizer_step':9*n,
        'checkpoint_save':n,'checkpoint_reload':n-1,'checkpoint_publish':n-int(posted),'terminal_costs_publish':n-int(posted),'checkpoint_promote':n-int(posted)}
    require(cost['protocol']=={k:counts(v) for k,v in expected.items()},'exact full cumulative study protocol work')
    require(cost['positions']=={'input_attempted':b['input_tokens'],'input_returned':b['input_tokens'],
        'supervised_attempted':b['supervised_tokens'],'supervised_returned':b['supervised_tokens']},'exact complete cumulative token dose')
def project(cost,operation):
    result=deepcopy(cost);result['protocol'][operation]['attempted']+=1;result['protocol'][operation]['returned']+=1
    for suffix in ('write','verify'):
        item=result['primitive_operations'].setdefault('chunk_'+operation+'_'+suffix,counts(0));item['attempted']+=1;item['returned']+=1
    return result
def _publication(publication,payload,readset):
    fields(publication,{'path','bytes','sha256','payload_sha256','durable'},'study durable publication')
    require(publication['durable'] is True and type(publication['bytes']) is int and publication['bytes']>0
        and publication['payload_sha256']==study_digest(payload),'exact actual study publication body')
    raw=readset.read(publication['path'],publication['sha256'],MAX_JSON);require(len(raw)==publication['bytes']
        and canonical(loads(raw),True)==canonical(payload,True),'exact original study publication bytes/payload')
def validate_study_chain(registration,arm,chain,latest,*,root):
    checked_seal(registration,True);require(registration.get('kind')=='CPU_SYNTHETIC_STUDY_SCHEDULE_FIXTURE' and registration.get('seed')==101
        and arm in ('M1','M2') and set(registration['arms'])=={'M1','M2'},'explicit disjoint synthetic study registration')
    for name,info in registration['arms'].items():
        updates=info['updates'];require(len(updates)==27,'complete unchanged27-update schedule')
        for i,u in enumerate(updates,1):require(u['update_ordinal']==i and len(u['occurrence_row_ids'])==20 and all(type(r) is str for r in u['occurrence_row_ids'])
            and type(u['input_tokens']) is int and type(u['supervised_tokens']) is int and 0<u['supervised_tokens']<=u['input_tokens']
            and type(u['learning_rate']) is float and u['learning_rate']==rates()[i-1],'exact whole20 update/order/dose/global LR')
        require(info['occurrence_order_sha256']==study_digest([r for u in updates for r in u['occurrence_row_ids']]),'exact full cohort order')
    require(type(chain) is list and len(chain)==3,'all three source-shaped study resume records')
    recipe=study_recipe(registration,arm);reads=ReadSet(root);previous=None;start_hash=None
    resume_fields={'schema_version','kind','status','envelope','publication','durable_terminal','terminal_publication','promotion','promotion_publication','terminal_costs','sha256'}
    envelope_fields={'schema_version','kind','arm','seed','chunk_ordinal','recipe','binding','backend_descriptor','posted_save_costs','native_resume_qualified','training_authorized','sha256'}
    for ordinal,resume in enumerate(chain,1):
        fields(resume,resume_fields,'study resume');checked_seal(resume,True)
        require(type(resume['schema_version']) is int and resume['schema_version']==1 and resume['kind']==STUDY
            and resume['status']=='SUCCESSFUL_STUDY_CHUNK_PROTOCOL_ONLY','qualification/shared archive cannot substitute study protocol')
        envelope=resume['envelope'];fields(envelope,envelope_fields,'study envelope');checked_seal(envelope,True)
        b=binding(registration,arm,ordinal*9)
        require(type(envelope['schema_version']) is int and envelope['schema_version']==1 and envelope['kind']==STUDY and envelope['arm']==arm
            and type(envelope['seed']) is int and envelope['seed']==101 and type(envelope['chunk_ordinal']) is int and envelope['chunk_ordinal']==ordinal
            and envelope['recipe']==recipe and envelope['binding']==b and envelope['training_authorized'] is envelope['native_resume_qualified'] is False,
            'exact study recipe/ordinal/binding/closed original labels')
        cp=envelope['backend_descriptor'];fields(cp,FIELDS['CheckpointDescriptor'],'archive descriptor')
        require(type(cp['schema_version']) is int and cp['schema_version']==1 and cp['kind']==ARCHIVE and cp['metadata_sha256']==study_digest(cp['metadata']),'exact shared backend descriptor metadata')
        meta=cp['metadata'];fields(meta,FIELDS['StudyCheckpointMetadata'],'study archive metadata')
        require(type(meta['schema_version']) is int and meta['schema_version']==1 and meta['kind']==ARCHIVE and meta['experiment']=='SYNTH-CRS-001'
            and meta['scope']==CPU_STUDY and meta['arm']==arm and type(meta['seed']) is int and meta['seed']==101 and meta['recipe_sha256']==recipe['sha256']
            and meta['cohort_sha256']==recipe['cohort_sha256'] and meta['backend_source_sha256']==recipe['backend_source_sha256']
            and meta['binding']==b and hashpin(meta['fresh_start_attestation_sha256'])
            and meta['adapter_parameter_order']==list(adapter_shapes())
            and meta['adapter_inventory_sha256']==study_digest(adapter_inventory())
            and all(meta[k] is False for k in ('native_launch_authorized','training_authorized','native_resume_qualified','qualification_STATE1_used')),'exact archive recipe/cohort/source/start/order/binding fixture scope')
        require(start_hash is None or meta['fresh_start_attestation_sha256']==start_hash,'same selected fresh zero-B/base/seed attestation across all three chunks')
        start_hash=meta['fresh_start_attestation_sha256']
        require(type(cp['bytes']) is int and 0<cp['bytes']<=MAX_ARCHIVE and 'state1' not in str(cp['path']).lower()
            and Path(cp['path']).suffix=='.json','inert fixture archive JSON only; never deserialize checkpoint')
        raw=reads.read(cp['path'],cp['sha256']);require(len(raw)==cp['bytes'],'exact inert fixture checkpoint bytes')
        payload=loads(raw);fields(payload,{'kind','metadata','factor_byte_observation'},'inert fixture checkpoint metadata container')
        require(payload['kind']=='CPU_INERT_CHECKPOINT_METADATA_ONLY' and payload['metadata']==meta,
            'exact inert archive bytes embed the selected metadata; no descriptor-only substitution')
        factor_observations({role:payload['factor_byte_observation'] for role in ('archive','resident_before_capture','resident_after_capture')})
        costs(envelope['posted_save_costs'],b,ordinal,True);_publication(resume['publication'],envelope,reads)
        terminal=resume['durable_terminal'];fields(terminal,{'envelope_sha256','costs','costs_sha256'},'durable terminal')
        want=project(project(envelope['posted_save_costs'],'checkpoint_publish'),'terminal_costs_publish')
        require(terminal['envelope_sha256']==envelope['sha256'] and terminal['costs']==want and terminal['costs_sha256']==study_digest(want),'exact post-save/publication/terminal projection')
        _publication(resume['terminal_publication'],terminal,reads)
        promotion=resume['promotion'];fields(promotion,{'envelope_sha256','durable_terminal_sha256','terminal_costs'},'study promotion')
        want=project(want,'checkpoint_promote')
        require(promotion=={'envelope_sha256':envelope['sha256'],'durable_terminal_sha256':study_digest(terminal),'terminal_costs':want}
            and resume['terminal_costs']==want,'exact promotion/latest cumulative costs; no rollback')
        _publication(resume['promotion_publication'],promotion,reads);costs(want,b,ordinal)
        if previous is not None:
            require(all(k in want['primitive_operations'] and all(want['primitive_operations'][k][t]>=v for t,v in c.items())
                for k,c in previous['primitive_operations'].items()),'prior primitive costs cannot disappear')
        previous=want
    require(latest==chain[-1]['terminal_costs'],'authoritative exact final latest costs')
    reads.verify();final=chain[-1]['envelope']
    return {'descriptor':deepcopy(final['backend_descriptor']),'binding':deepcopy(final['binding']),'recipe':recipe,'reads':reads,
        'archive_factor_observation':deepcopy(payload['factor_byte_observation'])}
def fixture_population(panel,prompts,tokens,refs,source_refs,statistics_refs):
    registered=legacy._panel(deepcopy(panel),deepcopy(prompts),deepcopy(tokens),deepcopy(refs),deepcopy(source_refs),deepcopy(statistics_refs),legacy.FIXTURE)
    return _Handle(dict(registered),'population',token=_TOKEN)
def fixture_selection(population,registration,arm,chain,latest,*,root):
    pop=value(population,'population');checked=validate_study_chain(registration,arm,chain,latest,root=root)
    identity=seal({'kind':FIXTURE,'arm':arm,'training_seed':101,'study_registration_body_sha256':registration['sha256'],
        'study_recipe_sha256':checked['recipe']['sha256'],'cohort_sha256':checked['recipe']['cohort_sha256'],
        'final_descriptor_sha256':study_digest(checked['descriptor']),'final_checkpoint_file_sha256':checked['descriptor']['sha256'],
        'final_binding':checked['binding'],'panel_metadata_sha256':pop['panel_metadata_sha256'],'row_order_sha256':pop['row_order_sha256']})
    body={'kind':FIXTURE,'arm':arm,'identity':identity,'population_sha256':pop['sha256'],'registration':deepcopy(registration),
        'chain':deepcopy(chain),'latest':deepcopy(latest),'root':str(Path(root).resolve()),'archive_factor_observation':checked['archive_factor_observation']}
    return _Handle(body,'selection',checked['reads'],token=_TOKEN)
def adapter_shapes():
    return {f'model.layers.{layer}.self_attn.{projection}.lora_{factor}':shape for layer in range(28)
        for projection,factor,shape in (('q_proj','A',[8,1024]),('q_proj','B',[2048,8]),('v_proj','A',[8,1024]),('v_proj','B',[1024,8]))}
def adapter_inventory():
    return [{'name':name,'shape':shape,'dtype':'float32','device':'cpu','requires_grad':True,'parameters':math.prod(shape)}
        for name,shape in adapter_shapes().items()]
def restore_receipt(receipt,selection):
    fields(receipt,FIELDS['FactorRestoreReceipt'],'unchanged successful unsealed factor restore receipt')
    require(receipt['status']=='RESTORED_FINAL_ADAPTER_INFERENCE_UNQUALIFIED' and receipt['scope']=='CPU_RANDOM_TINY_FINAL_ADAPTER_INFERENCE_FIXTURE'
        and receipt['arm']==selection['arm'] and receipt['CPU_fixture'] is True and receipt['real_final_study_chain_verified'] is False
        and receipt['adapter_tensors_copied']==112 and receipt['checkpoint_sha256']==selection['identity']['final_checkpoint_file_sha256']
        and receipt['final_binding']==selection['identity']['final_binding'],'exact successful fixture factor-only restore identity')
    require(all(receipt[k] is True for k in ('inference_allowed','caller_RNG_preserved','base_unchanged','evaluation_mode_preserved','no_grad_required_for_all_raw_forwards'))
        and all(receipt[k] is False for k in ('native_launch_authorized','training_authorized','study_checkpoint_qualification_established','optimizer_restored','training_RNG_restored','confirmation_eligible')),
        'helper remains unqualified and all optimizer/training/launch authority closed')
    counters(receipt['operation_costs']);require(receipt['operation_costs'].get('final_adapter_parameter_copy')==counts(112),'all112 actual fixture factor copy returns')
    return digest(receipt)
def factor_observations(values):
    fields(values,FIELDS['FactorByteObservations'],'archive/before/after factor observations');shapes=adapter_shapes();order=list(shapes);expected=None
    for role,observation in values.items():
        fields(observation,FIELDS['FactorByteObservation'],'factor observation');checked_seal(observation)
        require(observation['parameter_order']==order and set(observation['parameters'])==set(shapes)
            and observation['hash_recipe']=='IEEE754_FP32_LITTLE_ENDIAN_CONTIGUOUS_ROW_MAJOR_RAW_BYTES_V1'
            and type(observation['maximum_hash_read_bytes']) is int and 0<observation['maximum_hash_read_bytes']<=2*1024**2
            and observation['total_factor_bytes']==4587520,'exact source112 FP32 order/recipe/byte totals')
        for name,shape in shapes.items():
            item=observation['parameters'][name];fields(item,{'shape','dtype','bytes','sha256'},'factor byte descriptor')
            require(item['shape']==shape and item['dtype']=='float32' and type(item['bytes']) is int
                and item['bytes']==4*math.prod(shape) and hashpin(item['sha256']),'exact FP32 source factor shape/byte/hash')
        require(expected is None or observation['parameters']==expected,'archive/copied/after factor bytes identical');expected=observation['parameters']
def raw_receipt(raw,row,packet,call_n,positions,max_positions):
    fields(raw,FIELDS['InferenceRawReceipt'],'successful raw receipt')
    require(type(raw['schema_version']) is int and raw['schema_version']==1 and raw['implementation']=='synth-prompt-only-inference-component-1'
        and raw['scope']==legacy.FIXTURE and raw['status']=='RAW_RESPONSE_COLLECTED_UNSCORED','explicit unscored fixture raw producer scope/status')
    require(raw['row_id']==row['row_id'] and raw['prompt_sha256']==row['prompt_sha256'] and raw['prompt_ids_sha256']==digest(row['prompt_ids'])
        and raw['input_row_sha256']==digest(row) and raw['source_packet_sha256']==packet and raw['prompt_tokens']==row['prompt_tokens'],'exact raw/prompt/input/whole packet identity')
    require(raw['context_cap']==512 and raw['maximum_generated_tokens']==32 and raw['decoding']=='greedy_no_cache_no_grad'
        and all(raw[k] is False for k in ('padding','packing','prompt_truncated','oracle_inputs_read','capability_correctness_scored','native_launch_authorized','native_qualification_established','confirmation_eligible')),
        'raw fixed policy and all authority closed')
    ids=raw['raw_generated_ids'];trace=raw['trace'];lps=raw['selected_logprobs']
    require(type(ids) is list and 1<=len(ids)<=32 and len(ids)==len(trace)==len(lps)
        and all(type(t) is int and 0<=t<151936 for t in ids) and not any(t in (151643,151645) for t in ids[:-1]),'typed first-EOS generated IDs and trace population')
    for step,(token,lp,t) in enumerate(zip(ids,lps,trace)):
        fields(t,FIELDS['RawTraceStep'],'raw trace step');require(type(lp) in (int,float) and math.isfinite(lp) and lp<=0
            and t['token']==token and type(t['token']) is int and type(t['logprob']) in (int,float) and math.isfinite(t['logprob'])
            and t['logprob']<=0 and t['logprob']==lp and type(t['prefix_length']) is int
            and t['prefix_length']==row['prompt_tokens']+step,'finite exact LP and zero-based pre-generation prefix')
        call_n+=1;positions+=t['prefix_length']
    eos=ids[-1] in (151643,151645)
    require(raw['finish_reason']==('eos' if eos else 'generation_cap') and raw['truncated'] is(not eos)
        and raw['eos_token_id']==(ids[-1] if eos else None) and (eos or len(ids)==32),'exact EOS/cap finish without filtering')
    require(type(raw['decoded_response']) is str and len(raw['decoded_response'].encode('utf-8'))<=1024**2,'bounded generated-only fixture response; never decode or grade')
    c=raw['costs'];require(c=={'attempted_forwards':call_n,'input_positions':{'attempted':positions,'returned':positions,'failed':0,'pending':0},
        'maximum_attempted_forwards':2048,'maximum_input_positions':max_positions,'operations':{'inference_forward':counts(call_n)}},'exact complete cumulative capture cost arithmetic')
    preservation=raw['parameter_preservation'];fields(preservation,{'parameter_aliases','objects_pointers_dtypes_flags_gradients_preserved',
        'mutation_counters_preserved','independent_parameter_byte_verification_required'},'parameter preservation evidence')
    require(type(preservation['parameter_aliases']) is int and preservation['parameter_aliases']==423
        and all(preservation[k] is True for k in ('objects_pointers_dtypes_flags_gradients_preserved','mutation_counters_preserved','independent_parameter_byte_verification_required')),
        'parameter metadata is incomplete without factor/base evidence')
    return call_n,positions
def fixture_batch(selection,ordinal,inputs,outputs,restore,factors,refs,*,root):
    selected=value(selection,'selection');require(type(ordinal) is int and 1<=ordinal<=6,'fixed batch ordinal1..6')
    fields(refs,{'config','audit'},'externally selected postcapture fixture batch references');reads=ReadSet(root)
    require(all(refs[k].get('body_hash_convention')=='ASCII_SEALED_EXCLUDE_OWN_SHA256' for k in ('config','audit')),
        'config/audit anchor convention fixed by fixture producer role')
    config=reads.anchor(refs['config'],json_document=True);audit=reads.anchor(refs['audit'],json_document=True)
    fields(config,{'schema_version','kind','arm','ordinal','candidate_identity_sha256','inputs','source_packet_sha256','restore_receipt_sha256','sha256'},'fixture candidate config')
    fields(audit,{'schema_version','kind','arm','ordinal','candidate_identity_sha256','outputs','config_file_sha256','restore_receipt_sha256','factor_observations_sha256','sha256'},'fixture candidate audit')
    require(config.get('kind')==FIXTURE and audit.get('kind')==FIXTURE and config.get('ordinal')==audit.get('ordinal')==ordinal
        and type(config['schema_version']) is int and config['schema_version']==1 and type(audit['schema_version']) is int and audit['schema_version']==1
        and config['arm']==audit['arm']==selected['arm'] and config['source_packet_sha256']==digest(inputs)
        and config.get('candidate_identity_sha256')==audit.get('candidate_identity_sha256')==selected['identity']['sha256']
        and config.get('inputs')==inputs and audit.get('outputs')==outputs and audit.get('config_file_sha256')==refs['config']['file_sha256'],
        'actual selected config/audit fixture bytes and common candidate joins')
    rpin=restore_receipt(restore,selected);factor_observations(factors)
    require(factors['archive']==selected['archive_factor_observation'],'exact selected inert archive factor observation; self-consistent alternate maps forbidden')
    require(config.get('restore_receipt_sha256')==audit.get('restore_receipt_sha256')==rpin
        and audit.get('factor_observations_sha256')==digest(factors),'selected actual restore/factor observation joins')
    body={'kind':FIXTURE,'arm':selected['arm'],'ordinal':ordinal,'candidate_identity_sha256':selected['identity']['sha256'],
        'inputs':deepcopy(inputs),'outputs':deepcopy(outputs),'restore':deepcopy(restore),'factors':deepcopy(factors),'refs':deepcopy(refs)}
    reads.verify();return _Handle(body,'batch',reads,token=_TOKEN)
def fixture_import_selection(selection,batches,*,root):
    selected=value(selection,'selection');require(type(batches) is list and len(batches)==6,'postcapture manifest exact six batches')
    anchors=[]
    for ordinal,handle in enumerate(batches,1):
        batch=value(handle,'batch');require(batch['ordinal']==ordinal and batch['candidate_identity_sha256']==selected['identity']['sha256'],'ordered same candidate six selected batch anchors')
        anchors.append({'batch_ordinal':ordinal,**deepcopy(batch['refs'])})
    return _Handle({'kind':FIXTURE,'candidate_identity_sha256':selected['identity']['sha256'],'selection_sha256':digest(selected),'batch_anchors':anchors},'import_selection',token=_TOKEN)
def prepare_candidate_import(population,batches,*,selection,import_selection):
    pop=value(population,'population');selected=value(selection,'selection');post=value(import_selection,'import_selection')
    require(selected['population_sha256']==pop['sha256'] and post['selection_sha256']==digest(selected)
        and post['candidate_identity_sha256']==selected['identity']['sha256'],'two-stage immutable population/candidate selections')
    validate_study_chain(selected['registration'],selected['arm'],selected['chain'],selected['latest'],root=selected['root'])
    require(type(batches) is list and len(batches)==6,'all six complete fixed64 batches before output')
    joined=[]
    for ordinal,handle in enumerate(batches,1):
        b=value(handle,'batch');expected=pop['rows'][(ordinal-1)*64:ordinal*64]
        require(b['ordinal']==ordinal and b['arm']==selected['arm'] and b['candidate_identity_sha256']==selected['identity']['sha256']
            and post['batch_anchors'][ordinal-1]=={'batch_ordinal':ordinal,**b['refs']},'exact independently selected ordered audit/config/candidate')
        require(b['inputs']==[r['input'] for r in expected] and len(b['outputs'])==64,'whole64 original prompt/order/source slice')
        restore_receipt(b['restore'],selected);factor_observations(b['factors'])
        require(b['factors']['archive']==selected['archive_factor_observation'],'every batch keeps exact selected archive factor bytes')
        packet=digest(b['inputs']);n=positions=0;cap=sum(32*r['prompt_tokens']+496 for r in b['inputs'])
        for metadata,inp,raw in zip(expected,b['inputs'],b['outputs']):
            n,positions=raw_receipt(raw,inp,packet,n,positions,cap);joined.append({**deepcopy(metadata),'arm':selected['arm'],'batch_ordinal':ordinal,'raw':deepcopy(raw)})
        require(n<=2048 and positions<=cap,'whole capture cost within fixed dose')
    require(len(joined)==len({r['row_id'] for r in joined})==384 and digest([r['row_id'] for r in joined])==pop['row_order_sha256'],
        'complete384 unique original order; no resampling/dropping')
    require(len({r['component_id'] for r in joined})==96 and Counter(r['skill'] for r in joined)=={'R':128,'S':128,'RS':128},'all96/32 components and four original condition/rendering rows')
    for h,k in ((population,'population'),(selection,'selection'),(import_selection,'import_selection')):value(h,k)
    for h in batches:value(h,'batch')
    return seal({'schema_version':1,'kind':FIXTURE,'status':'PASS_CPU_COMPLETE384_CANDIDATE_IMPORT_MECHANICS_ONLY','arm':selected['arm'],
        'rows':joined,'population_registration_sha256':pop['sha256'],'panel_metadata_sha256':pop['panel_metadata_sha256'],
        'row_order_sha256':pop['row_order_sha256'],'candidate_identity':selected['identity'],'selection_sha256':digest(selected),
        'import_selection_sha256':digest(post),'scorer_sources':pop['scorer_sources'],'statistics_sources':pop['statistics_sources'],
        'native_launch_authorized':False,'scoring_authorized':False,'training_authorized':False,'retention_certified':False,
        'confirmation_eligible':False,'protected_bank_access':False,'paid_services':False,'quality_gain_established':False,
        'native_candidate_import_admitted':False,'study_completed':False})
