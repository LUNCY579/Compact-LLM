"""Authored CPU guard import/enrollment composition; native admission is absent.

Task uses the original short raw-row contract and future M0 byte-bound loader.
Long changes only that contract's context assertion to the source row envelope.
No task/long panels, native captures, labels, TRAIN, tokenizer or model are read
by public entrypoints. Fixture scoring is authored exact-string arithmetic, not
qualification of any task, long-context or human language preservation grader.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import ast, hashlib, importlib, json, math, sys

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
SCOPE='CPU_AUTHORED_GUARD_IMPORT_ENROLLMENT_FIXTURE_ONLY'
TASK_DOMAIN='synth_task_retention001'
MECHANISMS=('retrieval','instruction_following','answerability_abstention')
FAMILIES=('three_fact_keyed_join','latest_authorized_record_selection')
PLACEMENTS=('early','middle','late','distributed')
ENVELOPES={992:1024,2016:2048,4064:4096}
LONG_VARIANTS={'answerable_neutral':(True,False),'answerable_adversarial':(True,True),
               'unanswerable_neutral':(False,False),'unanswerable_adversarial':(False,True)}
INPUT_FIELDS={'row_id','prompt_sha256','prompt_ids','prompt_tokens','attention_mask'}

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()

def seal(value):return {**value,'sha256':digest(value)}

def _factory():
    token=object()
    def get():return token
    def check(value):require(value is token,'private authored CPU guard capability')
    return get,check
_CPU_capability,_check_capability=_factory()

def _helpers():
    if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
    modules={name:importlib.import_module(name) for name in ('score_import','dev_statistics')}
    for name,module in modules.items():require(Path(module.__file__).resolve()==PACKAGE/(name+'.py'),'original guard helper source origin')
    return modules

def _closed_native():
    require(False,'native guard collection/import source/scorer registration absent; gate closed before reads')

def prepare_task_import(*args,**kwargs):
    _closed_native()

def prepare_long_import(*args,**kwargs):
    _closed_native()

def _original_native_task_batches(batch_anchors):
    """Future reviewed admission must replace the closed gate before this loader.

    This retains the actual original independently pinned M0 batch loader; CPU
    fixture paths never call it and arbitrary authored audit flags cannot enter.
    """
    _closed_native()
    adapter=_helpers()['score_import']
    return [adapter.load_m0_batch(**anchor) for anchor in batch_anchors]

class _Fixture:
    def __init__(self,value,*,_capability):
        _check_capability(_capability);self.value=deepcopy(value);self.original=digest(value)

def _value(fixture,_capability):
    _check_capability(_capability)
    require(type(fixture) is _Fixture and digest(fixture.value)==fixture.original,'original unmodified authored guard fixture')
    return deepcopy(fixture.value)

def _population(kind,rows,metadata,declared_domains):
    count=768 if kind=='task' else 2304
    require(kind in ('task','long') and type(rows) is list and type(metadata) is list and len(rows)==len(metadata)==count,'complete original guard row population')
    require(type(declared_domains) is list and bool(declared_domains),'externally frozen declared guard domain IDs required')
    expected_class='task' if kind=='task' else 'long_context'
    ids=[r['row_id'] for r in rows]
    require(len(set(ids))==count and ids==[m['row_id'] for m in metadata],'exact unique ordered guard metadata/input rows')
    require(all(type(r) is dict and set(r)==INPUT_FIELDS and r['row_id'].startswith('fixture_') for r in rows),'authored prompt-only guard inputs')
    domains={};component_domain={}
    for domain in declared_domains:
        require(type(domain) is dict and set(domain)=={'domain_id','class','row_ids','component_ids'} and type(domain['domain_id']) is str and bool(domain['domain_id'].strip()) and domain['class']==expected_class and type(domain['row_ids']) is list and type(domain['component_ids']) is list and len(domain['row_ids'])==len(domain['component_ids']),'exact externally frozen guard domain metadata')
        require(domain['domain_id'] not in domains,'duplicate declared guard domain ID')
        domains[domain['domain_id']]=domain
        for row,component in zip(domain['row_ids'],domain['component_ids']):
            require(component not in component_domain or component_domain[component]==domain['domain_id'],'joined guard world cannot cross declared domains')
            component_domain[component]=domain['domain_id']
    declared=[(row,component) for domain in declared_domains for row,component in zip(domain['row_ids'],domain['component_ids'])]
    require(len(declared)==count and len({row for row,_ in declared})==count and dict(declared)=={m['row_id']:m['component_id'] for m in metadata},'declared domain covers every original guard row/component exactly')
    fields=({'row_id','component_id','mechanism','variant','rendering'} if kind=='task' else {'row_id','component_id','family','placement','missing_index','total_sequence_envelope','variant','answerable','adversarial'})
    require(all(type(m) is dict and set(m)==fields and m['component_id'].startswith('fixture_component_') for m in metadata) and len({m['component_id'] for m in metadata})==192,'authored192 original guard world components and exact metadata')
    groups=defaultdict(list)
    for row,meta in zip(rows,metadata):
        require(type(row['prompt_ids']) is list and row['prompt_ids'] and all(type(t) is int and 0<=t<151936 for t in row['prompt_ids']) and type(row['prompt_tokens']) is int and row['prompt_tokens']==len(row['prompt_ids']) and row['attention_mask']==[1]*row['prompt_tokens'],'whole typed guard prompt/token/mask arrays')
        require(type(row['prompt_sha256']) is str and len(row['prompt_sha256'])==64 and all(c in '0123456789abcdef' for c in row['prompt_sha256']),'guard prompt identity')
        groups[meta['component_id']].append((row,meta))
    if kind=='task':
        require(set(domains)=={TASK_DOMAIN},'original task guard domain ID retained')
        require(Counter(m['mechanism'] for m in metadata)=={name:256 for name in MECHANISMS},'original three task category quotas')
        variants={}
        for group in groups.values():
            require(len(group)==4 and len({m['mechanism'] for _,m in group})==1 and {m['rendering'] for _,m in group}=={'json_lines','typed_ledger'} and len({m['variant'] for _,m in group})==2 and len({(m['variant'],m['rendering']) for _,m in group})==4,'task whole world variant/rendering strata')
            mechanism=group[0][1]['mechanism'];selected={m['variant'] for _,m in group}
            require(mechanism not in variants or variants[mechanism]==selected,'task original per-category variant inventory retained');variants[mechanism]=selected
            require(all(0<r['prompt_tokens']<=480 for r,_ in group),'unchanged task context512 reserve32')
    else:
        require(TASK_DOMAIN not in domains,'long requires separately frozen external domain IDs')
        counts=Counter();missing_counts=Counter()
        for group in groups.values():
            require(len(group)==12 and len({(m['family'],m['placement'],m['missing_index']) for _,m in group})==1,'long whole twelve-derivative world metadata')
            family,placement,missing=group[0][1]['family'],group[0][1]['placement'],group[0][1]['missing_index']
            require(family in FAMILIES and placement in PLACEMENTS and type(missing) is int and missing in (0,1,2),'original long family/placement/missing-index strata')
            counts[(family,placement)]+=1
            missing_counts[(family,placement,missing)]+=1
            require({(r['prompt_tokens'],m['variant']) for r,m in group}=={(length,variant) for length in ENVELOPES for variant in LONG_VARIANTS},'all original long lengths/answerability/adversarial variants retained')
            require(all(m['total_sequence_envelope']==ENVELOPES[r['prompt_tokens']] and (m['answerable'],m['adversarial'])==LONG_VARIANTS[m['variant']] and type(m['answerable']) is type(m['adversarial']) is bool for r,m in group),'exact original long envelope and variant flags')
        require(counts=={(family,placement):24 for family in FAMILIES for placement in PLACEMENTS},'original long family/placement world quotas')
        require(missing_counts=={(family,placement,missing):8 for family in FAMILIES for placement in PLACEMENTS for missing in (0,1,2)},'original long missing-index world strata retained')
    return {'kind':kind,'rows':rows,'metadata':metadata,'declared_domains':declared_domains,'row_order_sha256':digest(ids),'fixture_only':True}

def _CPU_source(kind,rows,metadata,declared_domains,*,_capability):
    _check_capability(_capability)
    return _Fixture(_population(kind,deepcopy(rows),deepcopy(metadata),deepcopy(declared_domains)),_capability=_capability)

def _CPU_capture(source,batches,*,_capability):
    value=_value(source,_capability)
    return _Fixture({'source_sha256':source.original,'kind':value['kind'],'batches':batches,'fixture_only':True},_capability=_capability)

def _original_row_contract(adapter):
    """Compile the original shared loop once, adapting only context metadata."""
    function=next(n for n in ast.parse(Path(adapter.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='prepare_import')
    wanted=ast.dump(ast.parse("zip(expected, batch['outputs'])",mode='eval').body,include_attributes=False)
    loops=[n for n in ast.walk(function) if isinstance(n,ast.For) and ast.dump(n.iter,include_attributes=False)==wanted]
    require(len(loops)==1,'one unchanged original raw row loop')
    loop=deepcopy(loops[0]);changed=0
    for node in ast.walk(loop):
        if isinstance(node,ast.Compare) and ast.dump(node.left,include_attributes=False)==ast.dump(ast.parse("output['context_cap']",mode='eval').body,include_attributes=False):
            require(len(node.ops)==1 and isinstance(node.ops[0],ast.Eq) and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==512,'exact sole original context512 comparison')
            node.comparators=[ast.Name(id='context_cap',ctx=ast.Load())];changed+=1
    require(changed==1,'only source-bound context comparison adapted')
    return compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),adapter.__file__,'exec')

def _original_raw_rows(adapter,contract,expected,batch,ordinal,context_cap):
    joined=[];namespace={**adapter.__dict__,'scope':adapter.FIXTURE,'expected':expected,'batch':batch,'ordinal':ordinal,'arm':'M0','joined':joined,'context_cap':context_cap}
    exec(contract,namespace)
    return joined

def _trace(raw,input_row):
    trace=raw.get('trace');ids=raw['raw_generated_ids'];lp=raw.get('selected_logprobs')
    require(type(trace) is list and type(lp) is list and len(trace)==len(ids)==len(lp),'whole raw guard trace/token/logprob population')
    for index,step in enumerate(trace):
        require(type(step) is dict and set(step)=={'token','logprob','prefix_length'} and type(step['token']) is int and step['token']==ids[index] and type(step['logprob']) in (int,float) and math.isfinite(step['logprob']) and step['logprob']<=0 and step['logprob']==lp[index] and type(step['prefix_length']) is int and step['prefix_length']==input_row['prompt_tokens']+index,'exact guard token/logprob/prefix lineage')

def _CPU_import(source,capture,*,_capability):
    value=_value(source,_capability);captured=_value(capture,_capability)
    require(captured['fixture_only'] is True and captured['source_sha256']==source.original and captured['kind']==value['kind'],'same authored source and raw capture binding')
    adapter=_helpers()['score_import'];contract=_original_row_contract(adapter);batches=captured['batches'];kind=value['kind'];number=12 if kind=='task' else 36
    require(type(batches) is list and len(batches)==number,'complete original guard64 batch roster')
    joined=[]
    for ordinal,batch in enumerate(batches,1):
        offset=(ordinal-1)*64;rows=value['rows'][offset:offset+64];meta=value['metadata'][offset:offset+64]
        require(type(batch) is dict and set(batch)=={'ordinal','source_kind','inputs','outputs','source_packet_sha256','fixture_only'} and batch['fixture_only'] is True and type(batch['ordinal']) is int and batch['ordinal']==ordinal and batch['source_kind']==('task_guard_DEV' if kind=='task' else 'long_guard_DEV'),'same exact guard batch ordinal/source scope')
        require(batch['inputs']==rows and type(batch['outputs']) is list and len(batch['outputs'])==64 and batch['source_packet_sha256']==adapter.digest(rows),'whole original guard batch input/output population and order')
        for row,metadata,raw in zip(rows,meta,batch['outputs']):
            cap=512 if kind=='task' else ENVELOPES[row['prompt_tokens']]
            joined.extend(_original_raw_rows(adapter,contract,[{'row_id':row['row_id'],'input':row,'guard_metadata':metadata}],{**batch,'outputs':[raw]},ordinal,cap));_trace(raw,row)
    require(len(joined)==len(value['rows']) and digest([r['row_id'] for r in joined])==value['row_order_sha256'],'whole original guard roster/order retained')
    return _Fixture({'scope':SCOPE,'kind':kind,'source_sha256':source.original,'rows':joined,'declared_domains':value['declared_domains'],'row_order_sha256':value['row_order_sha256'],'fixture_only':True,'native_launch_authorized':False,'scoring_authorized':False,'quality_gain_established':False},_capability=_capability)

def _authored_ref(ref,*,source=False):
    fields={'path','bytes','sha256'} if source else {'path','bytes','sha256','value_sha256'}
    require(type(ref) is dict and set(ref)==fields and type(ref['bytes']) is int and 0<ref['bytes']<=16*1024**2,'bounded authored guard evidence descriptor')
    path=Path(ref['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(HERE),'authored guard evidence stays in fixture workspace')
    with path.open('rb') as stream:raw=stream.read(ref['bytes']+1)
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'actual authored guard evidence byte identity')
    if source:return raw
    value=_helpers()['score_import']._json(raw);require(digest(value)==ref['value_sha256'],'authored guard evidence value identity');return value

def _CPU_enrollment(imports,evidence,*,baseline_id,guard_policy,_capability):
    _check_capability(_capability)
    require(type(imports) is list and imports and type(baseline_id) is str and baseline_id.startswith('CPU_AUTHORED_') and type(guard_policy) is dict and guard_policy.get('CPU_authored') is True and guard_policy.get('retention',{}).get('required_domain_classes')==['task','language','long_context'],'explicit authored guard baseline/policy only')
    domains=[];raw_by_id={};source_bindings={}
    for packet in imports:
        value=_value(packet,_capability)
        require(value.get('scope')==SCOPE and value['fixture_only'] is True and value['native_launch_authorized'] is value['scoring_authorized'] is value['quality_gain_established'] is False,'authored imported guard population only')
        for row in value['rows']:
            require(row['row_id'] not in raw_by_id,'guard populations cannot duplicate rows');raw_by_id[row['row_id']]=row['raw']['decoded_response']
        for domain in value['declared_domains']:
            require(domain['domain_id'] not in source_bindings,'exact distinct externally declared guard domain IDs');domains.append(domain);source_bindings[domain['domain_id']]=value['source_sha256']
    require(type(evidence) is dict and set(evidence)==set(source_bindings),'exact complete selected guard domain scorer evidence')
    scores={};bindings={}
    for domain in domains:
        name=domain['domain_id'];proof=evidence[name]
        require(type(proof) is dict and set(proof)=={'source_sha256','scorer_source','labels','qualification'} and proof['source_sha256']==source_bindings[name],'same selected guard source before authored scoring')
        code=_authored_ref(proof['scorer_source'],source=True);labels=_authored_ref(proof['labels']);qualification=_authored_ref(proof['qualification'])
        require(labels.get('kind')=='CPU_AUTHORED_GUARD_LABELS' and labels.get('fixture_only') is True and labels.get('domain_id')==name and labels.get('source_sha256')==source_bindings[name] and type(labels.get('rows')) is dict and set(labels['rows'])==set(domain['row_ids']) and all(type(v) is str for v in labels['rows'].values()),'whole selected authored domain/source/label population')
        require(qualification.get('kind')=='CPU_AUTHORED_GUARD_SCORER_QUALIFICATION_FIXTURE' and qualification.get('fixture_only') is True and qualification.get('domain_id')==name and qualification.get('source_sha256')==source_bindings[name] and qualification.get('scorer_source_sha256')==proof['scorer_source']['sha256'] and qualification.get('labels_sha256')==proof['labels']['sha256'] and qualification.get('qualified') is True and qualification.get('native_grader_qualified') is False,'same authored source/label/scorer qualification; no native authority')
        namespace={};exec(compile(code,proof['scorer_source']['path'],'exec'),namespace);scorer=namespace.get('score_fixture')
        require(callable(scorer),'selected authored fixture scorer callable')
        controls=qualification.get('controls');require(type(controls) is list and len(controls)==2 and {c['expected'] for c in controls}=={True,False},'authored positive and negative scorer controls')
        for control in controls:
            require(type(control) is dict and set(control)=={'response','label','expected'} and type(control['expected']) is bool and scorer(control['response'],control['label']) is control['expected'],'selected authored scorer qualification reproduces')
        for row in domain['row_ids']:
            score=scorer(raw_by_id[row],labels['rows'][row]);require(type(score) is bool,'exact authored scorer bool');scores[row]=score
        bindings[name]={'scorer_sha256':proof['scorer_source']['sha256'],'labels_sha256':proof['labels']['sha256'],'qualification_receipt_sha256':proof['qualification']['sha256'],'qualified':True}
    stats=_helpers()['dev_statistics'];freeze=stats.freeze_guard_enrollment
    freeze=FunctionType(freeze.__code__,{**freeze.__globals__,'policy':lambda:deepcopy(guard_policy)},freeze.__name__,freeze.__defaults__,freeze.__closure__);freeze.__kwdefaults__=stats.freeze_guard_enrollment.__kwdefaults__
    enrollment=freeze(deepcopy(domains),scores,baseline_id=baseline_id,scorer_bindings=bindings)
    return seal({'scope':SCOPE,'enrollment':enrollment,'source_bindings':source_bindings,'fixture_only':True,'native_membership_admitted':False,'native_launch_authorized':False,'scoring_authorized':False,'quality_gain_established':False,'human_language_qualification_established':False})
