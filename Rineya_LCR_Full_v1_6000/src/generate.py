"""Deterministic, world-first LCR generator. No models, network or evaluation inputs.

This builds complete *candidate* chat rows. Native token fields stay null unless a
separate hash-locked native census measures them. Word sizing is NOT tokenization.
Gold answers use the symbolic world; check.py independently parses the final prose.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, random, re
from collections import Counter
from pathlib import Path
from dataclasses import dataclass

VERSION = '1.1.0-full-candidate'
SPLITS = ('train','dev','holdout')
TASKS = ('alias_resolution','causal_disambiguation','rule_exception','count_with_exclusion',
         'temporal_correlation','object_state_tracking','latest_verified','multi_condition')
REASONS = ('entity_absent','attribute_absent','retracted_or_superseded','presupposition_contradicted')
# These partitions are allocated BEFORE any world or text is constructed.
PARTITIONS = {
 'train':dict(depth=2, prefix=['Bel','Cor','Dal','Fen','Gal','Har','Kel','Lom'],
  domain='courier archive',unit='parcel',question_unit='consignment',place='station',question_place='depot',
  materials=['linen','cedar','copper','canvas','paper','wax','twine','felt'],
  prose_id='dispatch_active'),
 'dev':dict(depth=3,prefix=['Mer','Ner','Pel','Quel','Rav','Sel','Tor','Vel'],
  domain='conservatory dossier',unit='specimen',question_unit='sample',place='pavilion',question_place='enclosure',
  materials=['moss','pollen','peat','loam','pumice','bark','reed','humus'],
  prose_id='curator_apposition'),
 'holdout':dict(depth=4,prefix=['Wen','Yal','Zer','Bri','Dru','Gri','Kro','Sha'],
  domain='theatre chronicle',unit='prop',question_unit='stagepiece',place='gallery',question_place='hall',
  materials=['velvet','satin','plaster','ribbon','gauze','tulle','brocade','pigment'],
  prose_id='stage_possession')
}
COLORS = ['red','blue','green','black']
COLOR_Q = ['crimson','azure','verdant','ebony']
SITES = ['north','south','east','west']
SITE_Q = ['northern','southern','eastern','western']
FLAGS = ['warm','dry','stable','sealed','quiet','smooth','intact','clear']
ORDS = ['first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','eleventh','twelfth']
UNITS = ['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve',
         'thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
TENS = ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
SYSTEM = ('Use only the supplied record. Resolve descriptive names and follow explicit rules, '
          'chronology and withdrawals. Give one exact short answer. Use UNKNOWN alone when '
          'the requested result cannot be established. Do not explain.')
TASK_STEMS = {
 'alias_resolution':('Identify the custodian','Name the final caretaker','Identify the eventual guardian'),
 'causal_disambiguation':('Identify the qualifying cause','Name the responsible source','Identify the qualifying origin'),
 'rule_exception':('Identify the required material','Name the governing requirement','Identify the applicable material'),
 'count_with_exclusion':('Calculate the eligible quantity','Calculate the eligible amount','Calculate the admissible quantity'),
 'temporal_correlation':('Identify the next activity after the chime','Name the activity succeeding the chime',
                         'Identify the successor to the chime'),
 'object_state_tracking':('Identify the inspection condition','Name the condition upon inspection',
                          'Identify the recorded inspection condition'),
 'latest_verified':('Identify the latest endorsed label','Name the newest endorsed label','Identify the most recent endorsed label'),
 'multi_condition':('Identify the qualifying person','Name the person meeting every requirement','Identify the fully qualified person')
}

def dump(path:Path,obj:object)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def number(n:int)->str:
    if not 0<=n<100: raise ValueError(n)
    return UNITS[n] if n<20 else TENS[n//10]+(' '+UNITS[n%10] if n%10 else '')

def joined(seq:list[str])->str:
    return seq[0] if len(seq)==1 else ', '.join(seq[:-1])+' and '+seq[-1]

def words(text:str)->list[str]: return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+",text)

def names_for(seed:int,split:str,count:int)->list[str]:
    """Split-disjoint invented names; assignment does not depend on the answer class."""
    r=random.Random(seed^0x73165)
    starts=PARTITIONS[split]['prefix']
    mids=['a','e','i','o','u','ae','ia','io']
    tails=['lan','ren','vin','dor','sen','mar','len','nor','tel','rin','las','mon','var','tis','nel','rus']
    pool=[a+b+c for a,b,c in itertools.product(starts,mids,tails)]
    return r.sample(pool,count)

@dataclass
class Sentence:
    text:str
    role:str='distractor'
    case:str|None=None
    atom:str|None=None


def selector(name:str,color:int,site:int,split:str)->str:
    p=PARTITIONS[split];c=COLORS[color];s=SITES[site]
    if split=='train': return f'{name} handles {c} {p["unit"]}s at {s} {p["place"]}s.'
    if split=='dev': return f'{name}, assigned {c} {p["unit"]}s, works inside {s} {p["place"]}s.'
    return f'{s.capitalize()} galleries place {c} {p["unit"]}s under {name}.'


def question(task:str,color:int,site:int,split:str,template:int,difficulty:str="hard")->str:
    p=PARTITIONS[split]
    stem=TASK_STEMS[task][template]
    if difficulty=='medium':
        return f'{stem} for the {COLORS[color]} {p["unit"]} from the {SITES[site]} {p["place"]}.'
    return f'{stem} for the {COLOR_Q[color]} {p["question_unit"]} from the {SITE_Q[site]} {p["question_place"]}.'


def new_case(task:str,root:str,available_names:list[str],depth:int,r:random.Random,reason:str|None)->dict:
    """Constructive reference fields are sampled before rendering any text."""
    c={'root':root,'task':task,'depth':depth,'reason':reason,'summary':False}
    pool=iter(available_names)
    get=lambda:next(pool)
    if task=='alias_resolution':
        c['chain']=[root]+[get() for _ in range(depth)]
        c['custodian']=get()
        if reason=='attribute_absent':c['custodian']=None
    elif task in ('causal_disambiguation','multi_condition'):
        c['required']=r.sample(FLAGS,depth)
        c['candidates']=[]
        for j in range(3):
            features=list(c['required'])
            if j:features.remove(c['required'][j-1])
            if j==0 and reason=='attribute_absent':features.remove(c['required'][-1])
            c['candidates'].append({'name':get(),'features':features,'withheld':j==0 and reason=='attribute_absent'})
        r.shuffle(c['candidates'])
    elif task=='rule_exception':
        c['default']=get()
        c['statuses']=r.sample(FLAGS,depth)
        c['rules']=[{'rank':i+1,'status':tag,'value':get()} for i,tag in enumerate(c['statuses'])]
        if reason=='attribute_absent':c['rules'][-1]['value']=None
    elif task=='count_with_exclusion':
        c['batches']=[]
        for i in range(depth):
            qty=r.randint(9,24);excluded=r.randint(1,qty-2)
            c['batches'].append({'name':get(),'total':qty,'excluded':excluded})
        if reason=='attribute_absent':c['batches'][r.randrange(depth)]['total']=None
    elif task=='temporal_correlation':
        c['activities']=[get() for _ in range(depth+2)]
        c['chime_index']=r.randrange(depth+1)
        # Relations are emitted out of chronological order; depth is real and active.
        c['edges']=list(zip(c['activities'][:-1],c['activities'][1:]))
        if reason=='attribute_absent':
            c['edges'].remove((c['activities'][c['chime_index']],c['activities'][c['chime_index']+1]))
    elif task=='object_state_tracking':
        c['checkpoint']=depth
        c['updates']=[{'time':i+1,'value':get()} for i in range(depth+1)]
        if reason=='attribute_absent':
            # Earlier state cannot survive as a misleading available answer.
            for item in c['updates']:
                if item['time']<=c['checkpoint']:item['value']=None
    elif task=='latest_verified':
        c['updates']=[]
        chosen=r.randint(1,depth)
        for i in range(1,depth+3):
            verified=i<=chosen
            c['updates'].append({'time':i,'value':get(),'verified':verified})
        if reason=='attribute_absent':c['updates'][chosen-1]['value']=None
    else:raise ValueError(task)
    return c


def reference(case:dict)->str:
    """World-first reference; no access to rendered text or independent checker."""
    if case['reason'] in ('retracted_or_superseded','presupposition_contradicted'):return 'UNKNOWN'
    if case.get('summary'):return str(case['result'])
    t=case['task']
    if t=='alias_resolution':return case['custodian'] or 'UNKNOWN'
    if t in ('causal_disambiguation','multi_condition'):
        good=[x['name'] for x in case['candidates'] if set(case['required'])<=set(x['features'])]
        return good[0] if len(good)==1 else 'UNKNOWN'
    if t=='rule_exception':
        rules=[x for x in case['rules'] if x['status'] in case['statuses']]
        last=max(rules,key=lambda x:x['rank'])
        return last['value'] or 'UNKNOWN'
    if t=='count_with_exclusion':
        if any(x['total'] is None for x in case['batches']):return 'UNKNOWN'
        return str(sum(x['total']-x['excluded'] for x in case['batches']))
    if t=='temporal_correlation':
        key=case['activities'][case['chime_index']]
        after=[b for a,b in case['edges'] if a==key]
        return after[0] if len(after)==1 else 'UNKNOWN'
    if t=='object_state_tracking':
        available=[x for x in case['updates'] if x['time']<=case['checkpoint']]
        return max(available,key=lambda x:x['time'])['value'] or 'UNKNOWN'
    if t=='latest_verified':
        allowed=[x for x in case['updates'] if x['verified']]
        return max(allowed,key=lambda x:x['time'])['value'] or 'UNKNOWN'
    raise ValueError(t)


def render_case(c:dict,split:str,r:random.Random)->list[Sentence]:
    """Render factual prose. The checker has separately written parsing patterns."""
    root=c['root'];t=c['task'];out=[]
    def emit(s,atom):out.append(Sentence(s,'case',root,atom))
    emit(selector(root,c['color'],c['site'],split),'selector')
    if c.get('summary'):
        v=str(c['result'])
        text={
          'alias_resolution':f'{root} entrusts {v} with custody.',
          'causal_disambiguation':f'{root} identifies {v} as its sole qualifying cause.',
          'rule_exception':f'{root} requires {v} after applying all exceptions.',
          'count_with_exclusion':f'{root} retains {number(int(v)) if t=='count_with_exclusion' else v} eligible pieces after all exclusions.',
          'temporal_correlation':f'{root} schedules {v} immediately after its bell.',
          'object_state_tracking':f'{root} records {v} as its checkpoint state.',
          'latest_verified':f'{root} endorses {v} as its newest verified entry.',
          'multi_condition':f'{root} selects {v} as its sole fully qualified applicant.'}[t]
        emit(text,'summary')
        return out
    if t=='alias_resolution':
        for a,b in zip(c['chain'][:-1],c['chain'][1:]):
            if split=='train':emit(f'{a} also signs as {b}.','alias')
            elif split=='dev':emit(f'{b} serves as another identity used by {a}.','alias')
            else:emit(f'Signatures reading {b} belong to {a}.','alias')
        if c['custodian']:
            emit(f'{c["chain"][-1]} entrusted {c["custodian"]} with custody.','custody')
        else:emit(f'{c["chain"][-1]} completed routine correspondence.','unrelated')
    elif t in ('causal_disambiguation','multi_condition'):
        kind='causes' if t=='causal_disambiguation' else 'applicants'
        emit(f'{root} accepts only {kind} exhibiting {joined(c["required"])} together.','criterion')
        for cand in c['candidates']:
            stated=joined(cand['features'])
            if cand['withheld']:
                emit(f'{cand["name"]} has documented {stated} during {root} review, with other traits unrecorded.','features')
            else:
                emit(f'{cand["name"]} exhibits exactly {stated} during {root} review.','features')
        emit(f'{root} treats this candidate roster as complete.','completeness')
    elif t=='rule_exception':
        emit(f'{root} ordinarily requires {c["default"]}.','default')
        emit(f'{root} carries {joined(c["statuses"])} status.','statuses')
        for rule in c['rules']:
            value=rule['value'] or 'an unspecified material'
            emit(f'{root} applies level {number(rule["rank"])} exception {rule["status"]}, requiring {value}.','exception')
        emit(f'{root} gives higher numbered qualifying exceptions priority.','precedence')
    elif t=='count_with_exclusion':
        for b in c['batches']:
            amount=number(b['total']) if b['total'] is not None else 'an unstated quantity of'
            emit(f'{root} received {amount} pieces in batch {b["name"]}, including {number(b["excluded"])} disqualified pieces.','batch')
        emit(f'{root} lists every batch once, with no shared pieces.','completeness')
    elif t=='temporal_correlation':
        acts=list(c['activities']);r.shuffle(acts)
        emit(f'{root} records activities {joined(acts)}.','activities')
        emit(f'{root} rings its bell during {c["activities"][c["chime_index"]]}.','chime')
        edges=list(c['edges']);r.shuffle(edges)
        for a,b in edges:emit(f'{b} immediately followed {a} during {root} proceedings.','precedence')
    elif t=='object_state_tracking':
        emit(f'{root} uses checkpoint {ORDS[c["checkpoint"]-1]}.','checkpoint')
        upd=list(c['updates']);r.shuffle(upd)
        for x in upd:
            state=x['value'] or 'an unrecorded state'
            emit(f'{root} became {state} at observation {ORDS[x["time"]-1]}.','state')
        emit(f'{root} keeps each state until another observation changes it.','persistence')
    elif t=='latest_verified':
        upd=list(c['updates']);r.shuffle(upd)
        for x in upd:
            value=x['value'] or 'an unrecorded label'
            stamp='verified' if x['verified'] else 'unverified'
            emit(f'{root} logged {value} at edition {ORDS[x["time"]-1]}, marked {stamp}.','version')
        emit(f'{root} ranks editions by their stated order, not paragraph placement.','ordering')
    if c['reason']=='retracted_or_superseded':
        emit(f'{root} withdrew its entire operative account without replacement.','retraction')
    elif c['reason']=='presupposition_contradicted':
        emit(f'{root} never issued this item, despite its preliminary assignment.','contradiction')
    return out


def dependency_graph(c:dict,selected:bool=True)->dict:
    """Actual typed query-operation graph; repeated operators are explicitly disclosed."""
    t=c['task'];d=c['depth'];nodes=[{'id':'q','kind':'selector_join'}];edges=[]
    if t=='alias_resolution':
        prev='q'
        for i in range(d):
            nid=f'a{i}';nodes.append({'id':nid,'kind':'alias_edge'});edges.append([prev,nid]);prev=nid
        nodes.append({'id':'z','kind':'custody_read'});edges.append([prev,'z'])
    elif t in ('causal_disambiguation','multi_condition'):
        for i in range(d):
            nid=f'f{i}';nodes.append({'id':nid,'kind':'required_feature'});edges.append(['q',nid]);edges.append([nid,'z'])
        nodes.append({'id':'z','kind':'unique_conjunction'})
    elif t=='rule_exception':
        prev='q'
        for i in range(d):
            nid=f'r{i}';nodes.append({'id':nid,'kind':'applicable_priority_rule'});edges.append([prev,nid]);prev=nid
        nodes.append({'id':'z','kind':'max_priority_read'});edges.append([prev,'z'])
    elif t=='count_with_exclusion':
        for i in range(d):
            for kind in ('received','excluded'):
                nid=f'{kind}{i}';nodes.append({'id':nid,'kind':kind});edges.append(['q',nid]);edges.append([nid,f'b{i}'])
            nodes.append({'id':f'b{i}','kind':'set_difference'});edges.append([f'b{i}','z'])
        nodes.append({'id':'z','kind':'disjoint_union_cardinality'})
    elif t=='temporal_correlation':
        for i in range(d+2):nodes.append({'id':f'e{i}','kind':'activity'})
        edges.append(['q',f'e{c["chime_index"]}'])
        for i in range(d+1):edges.append([f'e{i}',f'e{i+1}'])
        nodes.append({'id':'z','kind':'immediate_successor'});edges.append([f'e{c["chime_index"]}','z'])
    else:
        n=d+1 if t=='object_state_tracking' else d+2
        for i in range(n):
            nid=f'e{i}';nodes.append({'id':nid,'kind':'dated_observation' if t=='object_state_tracking' else 'dated_endorsement'})
            edges.append(['q',nid]);edges.append([nid,'z'])
        nodes.append({'id':'z','kind':'latest_before_cutoff' if t=='object_state_tracking' else 'latest_verified'})
    # This is the task layout, not a claim that absent rows contain positive evidence.
    return {'task':t,'nodes':nodes,'edges':edges,'scope':'registered_case_layout; semantic operator families shared'}


def filler_sentence(r:random.Random,split:str,names:list[str],index:int)->Sentence:
    p=PARTITIONS[split];a,b=r.sample(names,2);m=r.choice(p['materials']);direction=r.choice(SITES)
    templates={
      'train':[
        '{a} checked {m} wrapping beside {direction} shelving and passed unused stock to {b}.',
        '{a} counted spare {m} supplies while {b} examined a separate delivery with worn packaging.',
        '{a} sent {m} samples toward {direction} storage after {b} confirmed that the corridor was clear.',
        '{a} retained the {m} ledger because {b} needed its earlier inspection notes for a different route.',
        '{a} repaired {m} covers while {b} compared dispatch estimates against the reserve inventory.',
        '{a} reported a delayed {m} delivery, but {b} completed the unrelated inspection before closing.'
      ],
      'dev':[
        'Beside the {direction} beds, {a} examined {m} while {b} prepared an unrelated conservation note.',
        'During a survey of {m}, {a} consulted {b} about drainage records stored in the {direction} annex.',
        'For the {m} collection, {a} revised the watering estimate; {b} retained the previous field sketch.',
        'After comparing {m} samples, {a} asked {b} to inspect the shaded benches in the {direction} enclosure.',
        'A separate {m} observation brought {a} and {b} to the {direction} beds before the routine survey.',
        'With {b} recording the moisture readings, {a} moved surplus {m} away from the {direction} doorway.'
      ],
      'holdout':[
        'The {direction} rehearsal paired {a} with {b} to inspect {m} scenery before the dress run.',
        'Spare {m} remained under {a} while {b} reviewed stage clearances along the {direction} passage.',
        'Lighting checks occupied {a}; meanwhile {b} counted {m} remnants near the {direction} curtain.',
        'A {m} repair delayed {a}, so {b} arranged a separate rehearsal at the {direction} entrance.',
        'Props made with {m} reached {a} after {b} had finished marking the {direction} stage boundary.',
        'The {m} inventory named {a} as examiner and {b} as witness for a separate {direction} rehearsal.'
      ]}
    return Sentence(r.choice(templates[split]).format(a=a,b=b,m=m,direction=direction),'background',None,f'background_{index}')


def arrange(critical:list[Sentence],others:list[Sentence],position:str,r:random.Random)->list[Sentence]:
    """Place complete sentences; never truncate. Critical blocks are 1 or 3 fragments."""
    r.shuffle(others)
    if position=='early':return critical+others
    if position=='late':return others+critical
    if position=='middle':
        n=len(others)//2
        return others[:n]+critical+others[n:]
    # Exactly three required fragments when possible, distributed in record thirds.
    n=len(critical)
    cuts=[0,max(1,n//3),max(2,2*n//3),n]
    chunks=[critical[cuts[i]:cuts[i+1]] for i in range(3)]
    m=len(others);ocuts=[0,m//3,2*m//3,m]
    result=[]
    for i in range(3):result+=chunks[i]+others[ocuts[i]:ocuts[i+1]]
    return result


def make_row(plan:dict,row_number:int)->tuple[dict,dict]:
    seed=int(plan['scenario_seed']);r=random.Random(seed);split=plan['split'];p=PARTITIONS[split]
    task=plan['task_type'];pool=names_for(seed,split,180)
    colors=r.sample(range(4),4);sites=r.sample(range(4),4)
    c0,c1,c2,c3=colors;s0,s1,s2,s3=sites
    if 'question_color' in plan:
        c0=int(plan['question_color']);s0=int(plan['question_site'])
        colors=[c0]+[x for x in colors if x!=c0];sites=[s0]+[x for x in sites if x!=s0]
        c0,c1,c2,c3=colors;s0,s1,s2,s3=sites
    actual_reason=plan['unanswerable_reason']
    # A hidden construction reason is drawn for every row, independent of its question.
    paired_reason=actual_reason or r.choice(REASONS)
    primary_reason=actual_reason if actual_reason!='entity_absent' else None
    companion_reason=paired_reason if actual_reason is None and paired_reason!='entity_absent' else None
    primary=new_case(task,pool[0],pool[4:30],p['depth'],r,primary_reason)
    companion=new_case(task,pool[1],pool[32:58],p['depth'],r,companion_reason)
    primary.update(color=c0,site=s0);companion.update(color=c1,site=s1)
    # Marginal colors and sites do not disclose a missing pair; only their binding does.
    if actual_reason=='entity_absent':primary['site'],companion['site']=s1,s0
    nval=lambda: r.randint(8,62) if task=='count_with_exclusion' else pool[r.randint(65,79)]
    near1={'root':pool[2],'task':task,'color':c0,'site':s2,'depth':p['depth'],
           'summary':True,'reason':None,'result':nval()}
    near2={'root':pool[3],'task':task,'color':c2,'site':s0,'depth':p['depth'],
           'summary':True,'reason':None,'result':nval()}
    cases=[primary,companion,near1,near2]
    gold='UNKNOWN' if actual_reason=='entity_absent' else reference(primary)
    if (gold=='UNKNOWN')!=(actual_reason is not None):raise AssertionError('Construction answerability mismatch')
    rendered=[render_case(c,split,r) for c in cases]
    focus=rendered[0]
    if gold != 'UNKNOWN':
        # Keep every case fact; move the decisive assertion, not its text, to the last proof chunk.
        decisive=[];rest=[]
        for st in focus:
            decisive_hit=(st.atom in ('custody','features','exception','batch','precedence','state','version')
                and re.search(r'\b'+re.escape(gold)+r'\b',st.text))
            # Numeric counts depend on every batch; selecting the last batch suffices for a second necessary chunk.
            if task=='count_with_exclusion':decisive_hit=st is next((x for x in reversed(focus) if x.atom=='batch'),None)
            if decisive_hit:decisive.append(st)
            else:rest.append(st)
        focus=rest+decisive
    other=[s for group in rendered[1:] for s in group]
    # All records carry forecasts. The counterfactual label denotes a verified contradicted forecast,
    # not an exclusive phrase or a claim about pretrained-model prior probabilities.
    forecast=str(r.randint(65,95)) if task=='count_with_exclusion' else pool[80]
    if forecast==gold:forecast=str(int(forecast)+1) if task=='count_with_exclusion' else pool[82]
    other.append(Sentence(f'An initial forecast associated {pool[0]} with {forecast}, pending investigation.','forecast'))
    other.append(Sentence(f'An initial forecast associated {pool[1]} with {pool[81] if task!="count_with_exclusion" else number(r.randint(65,95))}, pending investigation.','forecast'))
    dnames=pool[90:108]
    for i,name in enumerate(dnames[:10]):
        material=p['materials'][i%len(p['materials'])]
        other.append(Sentence(f'{name} catalogued spare {material}.','background',None,f'distractor_{i}'))
    # Genuine prose additions. Targets are WORDS only and explicitly not asserted as native bands.
    target={'short':r.randint(235,275),'medium':r.randint(480,540),'long':r.randint(815,890)}[plan['length_band']]
    intro=(f'This {p["domain"]} lists every assignment. Bell means chime; inspection means checkpoint. '
           'Withdrawals invalidate accounts. Forecasts remain provisional. Capitalized terms name people, materials, states or activities.')
    current=len(words(intro+' '+ ' '.join(s.text for s in focus+other)))
    extra=0
    while current<target:
        s=filler_sentence(r,split,dnames,extra);other.append(s);current+=len(words(s.text));extra+=1
    # Early means the END of all required case evidence is inside the first quarter.
    # If complete-sentence padding is needed, keep it and report the actual word size;
    # NEVER pretend that an unmeasured native length band has passed.
    # Defined near-miss conclusions flank the case record. Their selectors remain
    # in the body; selecting the first/last task-valued assertion is not sufficient.
    prefix=next(st for st in other if st.case==near1['root'] and st.atom=='summary')
    suffix=next(st for st in other if st.case==near2['root'] and st.atom=='summary')
    other=[st for st in other if st is not prefix and st is not suffix]
    boundary_chars=len(prefix.text)+len(suffix.text)+2
    position_extra=0
    if plan['needle_position'] in ('early','middle','late'):
        width=len('\n'.join(st.text for st in focus))
        if plan['needle_position']=='early':minimum_chars=math.ceil((len(intro)+3+len(prefix.text)+width)/0.245)
        elif plan['needle_position']=='late':minimum_chars=math.ceil((width+len(suffix.text)+1)/0.245)
        else:minimum_chars=math.ceil(width/0.22)
        while len(intro)+2+boundary_chars+sum(len(st.text)+1 for st in focus+other)<minimum_chars:
            st=filler_sentence(r,split,dnames,extra+position_extra);other.append(st);position_extra+=1
    # Keep a placement anchor for absence without inventing answer-bearing evidence.
    arranged=[prefix]+arrange(focus,other,plan['needle_position'],r)+[suffix]
    content=intro+'\n\n';sentences=[];critical_ranges=[]
    for s in arranged:
        start=len(content);content+=s.text;end=len(content);content+='\n'
        info={'start':start,'end':end,'role':s.role,'case':s.case,'atom':s.atom}
        sentences.append(info)
        if s.case==primary['root']:critical_ranges.append([start,end])
    context=content.rstrip('\n')
    qtemplate=r.randrange(3)
    if 'question_template' in plan:qtemplate=int(plan['question_template'])
    q=question(task,c0,s0,split,qtemplate,plan["difficulty"])
    user=context+'\n\nQuestion: '+q
    # Coalesce neighboring factual sentences into at most three exact user-text spans.
    merged=[]
    for a,b in critical_ranges:
        if merged and a<=merged[-1][1]+1:merged[-1][1]=b
        else:merged.append([a,b])
    spans=[] if gold=='UNKNOWN' else merged
    evidence=' '.join(user[a:b] for a,b in spans)
    overlap=sorted(set(w.lower() for w in words(q)) & set(w.lower() for w in words(evidence)))
    supported_entity_names=set(re.findall(r'\b[A-Z][a-z]+\b',evidence))
    real_names=set(pool)&set(words(context))
    distractors=sorted(real_names-supported_entity_names)
    sig=hashlib.sha256(str(seed).encode()).hexdigest()[:8]
    rid=f'lcr_v1_{task}_{sig}';gid=f'lcr_v1_scenario_{sig}'
    meta={
      'task_type':task,'variant':plan['variant'],'needle_position':plan['needle_position'],
      'needle_char_spans':spans,'related_char_spans':merged if gold=='UNKNOWN' else [],
      'retraction_char_spans':[[x['start'],x['end']] for x in sentences if x['case']==primary['root'] and x['atom']=='retraction'],
      'contradiction_char_spans':[[x['start'],x['end']] for x in sentences if x['case']==primary['root'] and x['atom']=='contradiction'],
      'context_char_span':[0,len(context)],'question_char_span':[len(context)+12,len(user)],
      'length_band':plan['length_band'],'length_band_status':'INTENDED_NOT_NATIVE_MEASURED',
      'difficulty':plan['difficulty'],'difficulty_status':'REQUESTED; see measured hard checks',
      'distractor_entities':len(distractors),'near_miss_candidates':2,
      'near_miss_criteria':['requested descriptive colour','requested compass location','defined task result'],
      'near_miss_entities':[near1['root'],near2['root']],
      'unanswerable_reason':actual_reason,'answer_type':'literal_unknown' if gold=='UNKNOWN' else ('integer' if task=='count_with_exclusion' else 'entity_name'),
      'context_tokens':None,'total_tokens':None,'rendered_prompt_tokens':None,'rendered_total_tokens':None,
      'completion_tokens':None,'unused_capacity':None,'native_token_status':'BLOCKED_MISSING_HASH_LOCKED_ASSETS',
      'context_words':len(words(context)),'context_characters':len(context),'complete_sentences_added_for_position':position_extra,
      'verified_by':None,'scenario_seed':seed,'template_id':f'{split}_{task}_{p["prose_id"]}_{plan["difficulty"]}_q{qtemplate}',
      'partition_profile':split,'registered_operator_depth':p['depth'],'question_mode':'synonym_binding' if plan['difficulty']=='hard' else 'literal_descriptor_binding',
      'lexical_question_positive_evidence_overlap':overlap,
      'positive_evidence_position_applicable':gold!='UNKNOWN',
      'placement_scope':'positive evidence' if gold!='UNKNOWN' else 'related case / absence neighborhood; NOT positive evidence',
      'forecast':forecast,'counterfactual_scope':'contradicted in-record provisional forecast' if plan['variant']=='counterfactual' else None,
      'source':'original synthetic world','license':'CC0-1.0 dedication intent',
      'human_review_status':'NOT_INDEPENDENTLY_HUMAN_REVIEWED','version':VERSION}
    row={'id':rid,'group_id':gid,'split':split,'messages':[{'role':'system','content':SYSTEM},
         {'role':'user','content':user},{'role':'assistant','content':gold}],'metadata':meta}
    world={'id':rid,'group_id':gid,'split':split,'allocation_id':plan['allocation_id'],
           'query':{'task':task,'color':c0,'site':s0},'cases':cases,'gold':gold,
           'sentences':sentences,'entity_names':sorted(real_names),'distractor_names':distractors,
           'dependency_graph':dependency_graph(primary),'generator_version':VERSION}
    return row,world


def build(root:Path,limit:int|None=None)->dict:
    plan=[json.loads(x) for x in (root/'allocation/plan.jsonl').read_text().splitlines()]
    if limit is not None:plan=plan[:limit]
    # Freeze query selectors before rendering. One stable stem per task avoids
    # accidental label-specific wording; sixteen colour/location pairs remain.
    # Each block of four contains every colour and every location exactly once.
    groups={}
    for row in plan:
        key=(row['split'],row['difficulty'],row['variant']=='unanswerable')
        groups.setdefault(key,[]).append(row)
    pairs=[(i,(i+j)%4) for j in range(4) for i in range(4)]
    for key,cohort in sorted(groups.items()):
        cohort.sort(key=lambda x:hashlib.sha256(('question-balance-v1:'+x['allocation_id']).encode()).hexdigest())
        for i,row in enumerate(cohort):
            row['question_color'],row['question_site']=pairs[i%16]
            row['question_template']=0
    for folder in ['data','worlds','reports']:(root/folder).mkdir(exist_ok=True)
    dump(root/'provenance/QUESTION_ALLOCATION.json',{'version':'query-balance-v1','assigned_before_rendering':True,
       'rule':'Within split/difficulty/answerability cohorts, deterministic shuffled order receives balanced Latin-cycle selectors. Stable task-specific stem; no row filtering.',
       'scope':'Pre-model data development. Previous DEV surface imbalance prompted a whole-corpus design revision; DEV/holdout are not sealed model benchmarks.',
       'assignments':[{'allocation_id':x['allocation_id'],'color':x['question_color'],'site':x['question_site'],'template':x['question_template']} for x in plan]})
    # Write a frozen actual partition registry before opening any data output.
    dump(root/'provenance/PARTITIONS.json',{'version':VERSION,'assigned_before_generation':True,
       'profiles':PARTITIONS,'shared':{'language':'English function words and syntax',
       'semantics':'all eight task families; typed operator primitives shared'},
       'structural_scope':'different case-layout depth per split, with concrete graphs in worlds; not a claim of new mathematical operators',
       'external_evaluation_nonoverlap':'NOT_ESTABLISHED; no evaluation corpus is opened'})
    rows=[];worlds=[];ids=set();groups=set()
    for i,p in enumerate(plan):
        row,w=make_row(p,i)
        if row['id'] in ids or row['group_id'] in groups:raise AssertionError('ID collision')
        ids.add(row['id']);groups.add(row['group_id']);rows.append(row);worlds.append(w)
    for split in SPLITS:
        for folder,items in [('data',rows),('worlds',worlds)]:
            with (root/folder/(split+'.jsonl')).open('w',encoding='utf-8',newline='\n') as f:
                for item in items:
                    if item['split']==split:f.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
    stats={'rows_generated':len(rows),'counts':dict(Counter(x['split'] for x in rows)),
       'gold_unknown':sum(x['messages'][-1]['content']=='UNKNOWN' for x in rows),
       'model_calls':0,'native_tokenizer_calls':0,'evaluation_rows_opened':0,
       'rows_dropped_for_balance':0,'rows_truncated':0,'status':'CONTENT_GENERATED_VALIDATION_PENDING'}
    dump(root/'reports/generation.json',stats);return stats

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--limit',type=int)
    a=ap.parse_args();print(json.dumps(build(a.root,a.limit),indent=2))
