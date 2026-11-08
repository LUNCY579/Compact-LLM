from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib,json
base=Path('C:/Users/clclc/BusinessProject/Titan-Joint-Quality-Pilot-v2-runs/data-001')
source=Path('C:/Users/clclc/BusinessProject/Titan-Joint-Quality-Pilot-v2/titan_joint_quality_v1/data.py')
expected={'train.jsonl':'6c82a98a4d66c64d15aa813dc1afa3f20898996c00f2d8590158a8eec8e1b11d','pilot.prompts.jsonl':'1ba852d8b0bb78d42c543a876e24a38987284095ff17f3ec9887bd7882d46bf8','pilot.references.jsonl':'51b3b96d8e557ddcf046efbb7a8afc81d1b49bbadd1289fb41023a7936aced2d'}
buffers={name:(base/name).read_bytes() for name in expected}
assert all(hashlib.sha256(buffers[name]).hexdigest()==sha for name,sha in expected.items())
assert hashlib.sha256(source.read_bytes()).hexdigest()=='dd4a39762f805b4d2a99d9c7416cbf454c2d99244ca2f42e4e73faa17d4dba1b'
def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result
train=[json.loads(line,object_pairs_hook=unique) for line in buffers['train.jsonl'].decode().splitlines()]
prompts={row['case_id']:row['messages'] for row in [json.loads(line,object_pairs_hook=unique) for line in buffers['pilot.prompts.jsonl'].decode().splitlines()]}
refs=[json.loads(line,object_pairs_hook=unique) for line in buffers['pilot.references.jsonl'].decode().splitlines()]
pilot=[{**row,'messages':prompts[row['case_id']]} for row in refs]
headers=defaultdict(set);coverage=Counter();issues=[];notes=Counter();example=None;copy=Counter();round1=Counter()
def read_public(messages):
    assert len(messages)==2 and [m['role'] for m in messages]==['system','user']
    body=messages[1]['content'];assert body.count('TASK_JSON\n')==body.count('\nEND_TASK')==1
    prefix,payload=body.split('TASK_JSON\n',1);raw,tail=payload.split('\nEND_TASK',1);assert not tail
    task=json.loads(raw,object_pairs_hook=unique)
    headers[task['kind']].add(messages[0]['content'])
    return task,prefix.strip()
def truth(t):
    k=t['kind'];q=t['query']
    if k=='program':
        names=sorted(t['initial']);basis={name:[int(name==other) for other in names]+[0] for name in names}
        alt=dict(t['initial'])
        for command in t['instructions']:
            dst,op,arg=command['dst'],command['op'],command['arg']
            if op=='add':basis[dst][-1]+=arg;alt[dst]+=arg
            elif op=='multiply':basis[dst]=[coefficient*arg for coefficient in basis[dst]];alt[dst]*=arg
            elif op=='add_from':basis[dst]=[a+b for a,b in zip(basis[dst],basis[arg])];alt[dst]+=t['initial'][arg]
            else:raise ValueError('uncovered program primitive')
        values=[t['initial'][name] for name in names]+[1]
        answer=sum(coefficient*value for coefficient,value in zip(basis[q],values))
        notes['program_initial_source_alternative_differs']+=alt[q]!=answer
        return str(answer)
    if k=='dependencies':
        memo=dict(t['leaves'])
        def value(name,trail=()):
            if name in memo:return memo[name]
            assert name not in trail
            left,op,right=t['expressions'][name];a,b=value(left,trail+(name,)),value(right,trail+(name,))
            memo[name]={'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b}[op]()
            return memo[name]
        return str(value(q))
    if k=='item_events':
        matches=[(index,event['bin']) for index,event in enumerate(t['events']) if event['item']==q]
        notes['item_queries_with_multiple_overwrites']+=len(matches)>1
        return max(matches)[1] if matches else t['initial'].get(q,'UNKNOWN')
    if k=='fsm':
        states=sorted(t['transitions']);vector={state:int(state==t['initial']) for state in states}
        assert 0<=q['after_symbols']<=len(t['symbols'])
        notes['fsm_zero_symbol_queries']+=q['after_symbols']==0
        for symbol in t['symbols'][:q['after_symbols']]:
            vector={dest:sum(vector[origin] for origin in states if t['transitions'][origin][symbol]==dest) for dest in states}
        final=[state for state,count in vector.items() if count];assert len(final)==1
        return final[0]
    if k=='baskets':return str(sum(Fraction(term['quantity'])*term['price'] for term in t['baskets'][q]))
    if k=='sharing':
        g=t['groups'][q['group']];assert g['count']>=0 and g['people']>0
        share=Fraction(g['count'],g['people']);whole=share.numerator//share.denominator
        return str(whole if q['component']=='quotient' else g['count']-g['people']*whole)
    if k=='authority':
        matching=[f['value'] for f in t['facts'] if (f['entity'],f['attribute'],f['source'])==(q['entity'],q['attribute'],t['authoritative_source'])]
        notes['authority_conflicting_authoritative_queries']+=len(set(matching))>1
        notes['authority_missing_authoritative_queries']+=not matching
        return matching[0] if matching and len(set(matching))==1 else 'UNKNOWN'
    if k=='horn':
        inferred=set(t['facts']);new=True
        while new:
            additions={rule['conclusion'] for rule in t['rules'] if all(premise in inferred for premise in rule['premises'])}-inferred
            new=bool(additions);inferred|=additions
        notes['horn_multi_premise_rules']+=sum(len(rule['premises'])>1 for rule in t['rules'])
        return 'SUPPORTED' if q in inferred else 'UNKNOWN'
    if k=='latest_record':
        records=[record for record in t['records'] if record['entity']==q['entity']]
        if not records:return 'UNKNOWN'
        maximum=max(record['timestamp'] for record in records);chosen=[record for record in records if record['timestamp']==maximum]
        notes['latest_equal_maximum_timestamp_queries']+=len(chosen)>1
        assert len(chosen)==1
        return str(chosen[0][q['field']])
    if k=='alias_join':
        code=t['aliases'][q['alias']];matches=[record for record in t['records'] if record['code']==code]
        notes['alias_nonunique_record_queries']+=len(matches)!=1
        assert len(matches)==1
        return matches[0][q['field']]
    if k=='field_transform':
        assert len(q['keys'])==len(set(q['keys']))==2
        selected=[t['fields'][key] for key in q['keys']]
        transform=str.upper if q['case']=='upper' else str.lower
        return q['separator'].join(map(transform,selected))
    if k=='filtered_roster':
        labels=[record['label'] for record in t['records'] if record['tier']==q['tier'] and not(record['score']<q['minimum'])]
        notes['roster_duplicate_selected_label_queries']+=len(labels)!=len(set(labels))
        notes['roster_empty_selected_queries']+=not labels
        return ','.join(sorted(labels)[::(-1 if q['order']=='desc' else 1)]) if labels else 'NONE'
    raise ValueError('uncovered kind')
formats={'Return the answer alone with no surrounding text.':'plain','Return exactly one JSON object with one string field named "answer".':'json','Return exactly <answer>VALUE</answer> with no surrounding text.':'tag'}
for split,rows in [('train',train),('pilot',pilot)]:
    for line,row in enumerate(rows,1):
        task,prefix=read_public(row['messages']);answer=truth(task);coverage[split+':'+task['kind']]+=1
        if answer!=row['canonical_answer']:issues.append({'case_id':row['case_id'],'predicted':answer,'gold':row['canonical_answer']})
        assert formats[prefix]==row['requested_format']
        target=answer if row['requested_format']=='plain' else json.dumps({'answer':answer},separators=(',',':'),ensure_ascii=False) if row['requested_format']=='json' else '<answer>'+answer+'</answer>'
        assert target==row['target']
        if task['kind']=='program' and example is None:
            example={'case_id':row['case_id'],'train_jsonl_line':line,'query':task['query'],'initial':task['initial'],'gold':answer,'instructions':task['instructions']}
        if task['kind']=='horn':
            pred='SUPPORTED' if task['query'] in task['facts'] else 'UNKNOWN'
            once=set(task['facts'])|{rule['conclusion'] for rule in task['rules'] if all(atom in task['facts'] for atom in rule['premises'])}
            pred1='SUPPORTED' if task['query'] in once else 'UNKNOWN'
            for counter,prediction in [(copy,pred),(round1,pred1)]:
                counter['correct']+=prediction==row['canonical_answer'];counter['false_abstention']+=prediction=='UNKNOWN' and row['canonical_answer']!='UNKNOWN';counter['unsupported']+=prediction=='SUPPORTED' and row['canonical_answer']=='UNKNOWN'
print(json.dumps({'scope':'EXISTING_OPEN_ROWS_FRESH_PUBLIC_PARSE_NO_PACKAGE_TEACHERS','coverage':dict(coverage),'truth_mismatches':issues,'public_wrapper_and_target_mismatches':0,'unique_system_wordings_per_kind':{kind:len(values) for kind,values in headers.items()},'semantic_corner_coverage':dict(notes),'horn_copy_fact_floor':dict(copy),'horn_one_simultaneous_round_floor':dict(round1),'program_wording_example':example,'hashes':{'data.py':hashlib.sha256(source.read_bytes()).hexdigest(),**{name:hashlib.sha256(blob).hexdigest() for name,blob in buffers.items()}},'model_calls':0,'bank_generated':False,'files_written':False}))
