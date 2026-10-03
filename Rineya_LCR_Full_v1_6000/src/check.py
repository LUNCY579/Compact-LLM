"""Independent text-only label derivation for LCR full candidate.

Does NOT import generate.py, read worlds, consult generator gold fields, or use
row.metadata to solve. It infers the task and selector from the written question,
parses prose into relations, then solves independently. Shared grammar/authorship
is disclosed; this is algorithmic independence, not independent human review.
"""
from __future__ import annotations
import argparse,json,re,functools
from collections import defaultdict,deque
from pathlib import Path

VERSION='lcr_text_checker_1.0.0'
N=r'[A-Z][a-z]+'
COLOR={'crimson':'red','azure':'blue','verdant':'green','ebony':'black','red':'red','blue':'blue','green':'green','black':'black'}
DIRECTION={'northern':'north','southern':'south','eastern':'east','western':'west','north':'north','south':'south','east':'east','west':'west'}
ORDINAL={w:i+1 for i,w in enumerate(('first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','eleventh','twelfth'))}
ONES={w:i for i,w in enumerate(('zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen'))}
TENS={'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90}
FLAGS={'warm','dry','stable','sealed','quiet','smooth','intact','clear'}

class DataError(ValueError):pass

def quantity(text:str)->int:
    parts=text.split()
    if len(parts)==1 and parts[0] in ONES:return ONES[parts[0]]
    if len(parts) in (1,2) and parts[0] in TENS:
        if len(parts)==2 and not 0<ONES.get(parts[1],99)<10:raise DataError('bad number')
        return TENS[parts[0]]+(ONES[parts[1]] if len(parts)==2 else 0)
    raise DataError(f'Invalid written number: {text!r}')

def features(text:str)->set[str]:
    result=set(re.findall(r'[a-z]+',text)) - {'and'}
    if not result<=FLAGS:raise DataError('unrecognized feature')
    return result

def query_from_text(text:str)->dict:
    if text.count('\n\nQuestion: ')!=1:raise DataError('Expected exactly one question boundary')
    record,q=text.split('\n\nQuestion: ')
    m=re.search(r' for the (crimson|azure|verdant|ebony|red|blue|green|black) (consignment|sample|stagepiece|parcel|specimen|prop) from the (northern|southern|eastern|western|north|south|east|west) (depot|enclosure|hall|station|pavilion|gallery)\.$',q)
    if not m:raise DataError('Unknown query selector grammar')
    stem=q[:m.start()]
    if any(x in stem for x in ('custodian','caretaker','guardian')):t='alias_resolution'
    elif any(x in stem for x in ('cause','responsible source','qualifying origin')):t='causal_disambiguation'
    elif any(x in stem for x in ('required material','governing requirement','applicable material')):t='rule_exception'
    elif any(x in stem for x in ('eligible quantity','eligible amount','admissible quantity')):t='count_with_exclusion'
    elif 'chime' in stem:t='temporal_correlation'
    elif 'inspection' in stem:t='object_state_tracking'
    elif 'endorsed label' in stem:t='latest_verified'
    elif any(x in stem for x in ('qualifying person','every requirement','qualified person')):t='multi_condition'
    else:raise DataError(f'Unknown task stem: {stem}')
    return {'record':record,'question':q,'task':t,'color':COLOR[m[1]],'site':DIRECTION[m[3]],'unit':m[2]}


@functools.lru_cache(maxsize=16)
def parse_record(record:str)->dict:
    # Each map contains only independently parsed model-visible facts.
    d={'selectors':{},'aliases':defaultdict(set),'custody':{},'required':{},'candidate_features':defaultdict(dict),
       'incomplete_features':defaultdict(set),'defaults':{},'statuses':{},'rules':defaultdict(list),'batches':defaultdict(list),'chimes':{},
       'activities':{},'edges':defaultdict(set),'cutoffs':{},'states':defaultdict(list),'versions':defaultdict(list),
       'withdrawn':set(),'nonissued':set(),'summaries':{},'parsed_facts':0}
    for line in record.splitlines():
        s=line.strip()
        if not s:continue
        m=re.fullmatch(fr'({N}) handles (red|blue|green|black) (?:parcels|specimens|props) at (north|south|east|west) (?:stations|pavilions|galleries)\.',s)
        if not m:
            m=re.fullmatch(fr'({N}), assigned (red|blue|green|black) specimens, works inside (north|south|east|west) pavilions\.',s)
        if m:
            key=(m[2],m[3]);d['selectors'].setdefault(key,[]).append(m[1]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'(North|South|East|West) galleries place (red|blue|green|black) props under ({N})\.',s)
        if m:
            d['selectors'].setdefault((m[2],m[1].lower()),[]).append(m[3]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) also signs as ({N})\.',s)
        if m:d['aliases'][m[1]].add(m[2]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) serves as another identity used by ({N})\.',s)
        if m:d['aliases'][m[2]].add(m[1]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'Signatures reading ({N}) belong to ({N})\.',s)
        if m:d['aliases'][m[2]].add(m[1]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) entrusted ({N}) with custody\.',s)
        if m:d['custody'][m[1]]=m[2];d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) accepts only (causes|applicants) exhibiting ([a-z, ]+) together\.',s)
        if m:d['required'][m[1]]=features(m[3]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) exhibits exactly ([a-z, ]+) during ({N}) review\.',s)
        if m:d['candidate_features'][m[3]][m[1]]=features(m[2]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) has documented ([a-z, ]+) during ({N}) review, with other traits unrecorded\.',s)
        if m:
            d['candidate_features'][m[3]][m[1]]=features(m[2]);d['incomplete_features'][m[3]].add(m[1]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) ordinarily requires ({N})\.',s)
        if m:d['defaults'][m[1]]=m[2];d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) carries ([a-z, ]+) status\.',s)
        if m:d['statuses'][m[1]]=features(m[2]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) applies level ([a-z ]+) exception ([a-z]+), requiring ({N}|an unspecified material)\.',s)
        if m:d['rules'][m[1]].append((quantity(m[2]),m[3],None if m[4]=='an unspecified material' else m[4]));d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) received ([a-z ]+) pieces in batch ({N}), including ([a-z ]+) disqualified pieces\.',s)
        if m:
            total=None if m[2]=='an unstated quantity of' else quantity(m[2])
            d['batches'][m[1]].append((m[3],total,quantity(m[4])));d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) records activities (.+)\.',s)
        if m:d['activities'][m[1]]=set(re.findall(N,m[2]));d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) rings its bell during ({N})\.',s)
        if m:d['chimes'][m[1]]=m[2];d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) immediately followed ({N}) during ({N}) proceedings\.',s)
        if m:d['edges'][m[3]].add((m[2],m[1]));d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) uses checkpoint ([a-z]+)\.',s)
        if m:d['cutoffs'][m[1]]=ORDINAL[m[2]];d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) became ({N}|an unrecorded state) at observation ([a-z]+)\.',s)
        if m:d['states'][m[1]].append((ORDINAL[m[3]],None if m[2]=='an unrecorded state' else m[2]));d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) logged ({N}|an unrecorded label) at edition ([a-z]+), marked (verified|unverified)\.',s)
        if m:d['versions'][m[1]].append((ORDINAL[m[3]],m[4]=='verified',None if m[2]=='an unrecorded label' else m[2]));d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) withdrew its entire operative account without replacement\.',s)
        if m:d['withdrawn'].add(m[1]);d['parsed_facts']+=1;continue
        m=re.fullmatch(fr'({N}) never issued this item, despite its preliminary assignment\.',s)
        if m:d['nonissued'].add(m[1]);d['parsed_facts']+=1;continue
        # Separate summary grammar for genuinely defined near-miss cases.
        patterns=[
         ('alias_resolution',fr'({N}) entrusts ({N}) with custody\.'),
         ('causal_disambiguation',fr'({N}) identifies ({N}) as its sole qualifying cause\.'),
         ('rule_exception',fr'({N}) requires ({N}) after applying all exceptions\.'),
         ('count_with_exclusion',fr'({N}) retains ([a-z ]+) eligible pieces after all exclusions\.'),
         ('temporal_correlation',fr'({N}) schedules ({N}) immediately after its bell\.'),
         ('object_state_tracking',fr'({N}) records ({N}) as its checkpoint state\.'),
         ('latest_verified',fr'({N}) endorses ({N}) as its newest verified entry\.'),
         ('multi_condition',fr'({N}) selects ({N}) as its sole fully qualified applicant\.')]
        for task,pat in patterns:
            m=re.fullmatch(pat,s)
            if m:
                val=str(quantity(m[2])) if task=='count_with_exclusion' else m[2]
                d['summaries'][(m[1],task)]=val;d['parsed_facts']+=1;break
    return d


def solve(text:str)->dict:
    query=query_from_text(text);d=parse_record(query['record']);task=query['task']
    roots=d['selectors'].get((query['color'],query['site']),[])
    result={'answer':'UNKNOWN','reason':None,'task':task,'root':None,'parsed_facts':d['parsed_facts'],
            'candidate_results':{},'selector':{'color':query['color'],'site':query['site']}}
    if not roots:result['reason']='entity_absent';return result
    if len(roots)!=1:raise DataError('Nonunique selector')
    root=roots[0];result['root']=root
    if root in d['nonissued']:result['reason']='presupposition_contradicted';return result
    if root in d['withdrawn']:result['reason']='retracted_or_superseded';return result
    if (root,task) in d['summaries']:
        result['answer']=d['summaries'][(root,task)];return result
    value=None
    if task=='alias_resolution':
        seen=set();queue=deque([root]);answers=set()
        while queue:
            a=queue.popleft()
            if a in seen:continue
            seen.add(a)
            if a in d['custody']:answers.add(d['custody'][a])
            queue.extend(d['aliases'].get(a,set())-seen)
        if len(answers)==1:value=next(iter(answers))
        if len(answers)>1:raise DataError('Conflicting custody')
    elif task in ('causal_disambiguation','multi_condition'):
        req=d['required'].get(root)
        if req:
            survivors=set(d['candidate_features'][root])
            # Successive intersections, rather than the generator's inclusion predicate.
            for feature in req:
                survivors &= {who for who,fs in d['candidate_features'][root].items() if feature in fs}
            if len(survivors)==1 and not (d['incomplete_features'][root]-survivors):value=next(iter(survivors))
            if len(survivors)>1:raise DataError('Nonunique qualifying candidate')
    elif task=='rule_exception':
        value=d['defaults'].get(root)
        for priority,flag,material in sorted(d['rules'][root]):
            if flag in d['statuses'].get(root,set()):value=material
    elif task=='count_with_exclusion':
        # Materialize disjoint item sets and remove excluded prefix subsets.
        items=set();seen_batches=set();missing=False
        if not d['batches'][root]:missing=True
        for label,total,bad in d['batches'][root]:
            if label in seen_batches:raise DataError('Duplicate batch')
            seen_batches.add(label)
            if total is None:missing=True;continue
            if not 0<=bad<=total:raise DataError('Invalid exclusions')
            batch={(label,i) for i in range(total)}
            batch-={(label,i) for i in range(bad)}
            items|=batch
        if not missing:value=str(len(items))
    elif task=='temporal_correlation':
        bell=d['chimes'].get(root);acts=d['activities'].get(root,set())
        successors=defaultdict(set);indeg={a:0 for a in acts}
        for a,b in d['edges'][root]:
            if a not in acts or b not in acts:raise DataError('Unknown activity')
            successors[a].add(b);indeg[b]+=1
        # Ensure acyclic valid precedence before answering the explicit immediate-successor relation.
        pending=deque(a for a,deg in indeg.items() if deg==0);visited=0
        while pending:
            a=pending.popleft();visited+=1
            for b in successors[a]:
                indeg[b]-=1
                if indeg[b]==0:pending.append(b)
        if visited!=len(acts):raise DataError('Cyclic schedule')
        after=successors.get(bell,set())
        if len(after)==1:value=next(iter(after))
        elif len(after)>1:raise DataError('Conflicting immediate successor')
    elif task=='object_state_tracking':
        cutoff=d['cutoffs'].get(root)
        if cutoff is not None:
            seen_times=set()
            for time,state in sorted(d['states'][root]):
                if time in seen_times:raise DataError('Duplicate state time')
                seen_times.add(time)
                if time<=cutoff:value=state
    elif task=='latest_verified':
        seen_times=set()
        for time,verified,label in sorted(d['versions'][root]):
            if time in seen_times:raise DataError('Duplicate edition')
            seen_times.add(time)
            if verified:value=label
    if value is not None:result['answer']=str(value)
    else:result['reason']='attribute_absent'
    return result


def validate_row(row:dict)->dict:
    msgs=row.get('messages',[])
    if [m.get('role') for m in msgs]!=['system','user','assistant']:raise DataError('Invalid chat roles')
    if any(not isinstance(m.get('content'),str) or not m['content'] for m in msgs):raise DataError('Invalid chat text')
    receipt=solve(msgs[1]['content']);expected=msgs[2]['content']
    if receipt['answer']!=expected:raise DataError(f"Answer disagreement: {row.get('id')} gold={expected!r} parsed={receipt['answer']!r}")
    metadata=row.get('metadata',{})
    if receipt['reason']!=metadata.get('unanswerable_reason'):raise DataError('Reason disagreement')
    if receipt['task']!=metadata.get('task_type'):raise DataError('Task-name disagreement')
    text=msgs[1]['content'];context=receipt and query_from_text(text)['record']
    for field in ('needle_char_spans','related_char_spans','retraction_char_spans','contradiction_char_spans'):
        prev=-1
        for span in metadata.get(field,[]):
            if not (isinstance(span,list) and len(span)==2 and all(type(v)is int for v in span)):raise DataError('Malformed span')
            a,b=span
            if not 0<=a<b<=len(context) or a<prev:raise DataError('Invalid or unordered span')
            prev=b
    if expected=='UNKNOWN' and metadata.get('needle_char_spans'):raise DataError('False positive evidence for UNKNOWN')
    if expected!='UNKNOWN' and not metadata.get('needle_char_spans'):raise DataError('Missing supported evidence')
    qspan=metadata['question_char_span']
    if text[qspan[0]:qspan[1]]!=query_from_text(text)['question']:raise DataError('Question offset mismatch')
    if len(expected.split())>8:raise DataError('Answer exceeds eight words')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('jsonl',type=Path);a=p.parse_args();n=0
    for line in a.jsonl.read_text().splitlines():validate_row(json.loads(line));n+=1
    print(json.dumps({'rows_verified':n,'disagreements':0,'checker':VERSION}))
