#!/usr/bin/env python3
"""Independent text-side solvers and release validator (stdlib only).
Checkers read the model-visible prompt; no generator answers, seeds, latent worlds,
or metadata are passed into solve_prompt(). Natural-language target checks confirm
slot faithfulness to this controlled grammar, not human-level stylistic quality.
"""
from __future__ import annotations
import argparse, collections, gzip, hashlib, itertools, json, re
from pathlib import Path
from typing import Any

if not __debug__:
    raise RuntimeError("Run verification without Python -O/-OO; validation assertions must remain enabled.")

def parts(prompt: str):
    before, question=prompt.rsplit('\nQUESTION\n',1)
    records=before.split('\nRECORDS\n',1)[1].rsplit('\nEND RECORDS',1)[0]
    return records,question

def one(pattern: str,text: str):
    m=re.search(pattern,text)
    if not m:raise ValueError(f'Missing required text: {pattern}')
    return m.groups()

def atom(text: str):
    return ('!' + text[4:]) if text.startswith('not ') else text

def solve_deduction(s: str,q: str) -> str:
    who,prop=one(r'Determine the status of "(\w+) is (\w+)"',q)
    known=set(re.findall(r'Fact: (\w+) is (not \w+|\w+)\.',s));known={(a,atom(b)) for a,b in known}
    rules=[([atom(x.strip()) for x in a.split(' and ')],atom(b)) for a,b in re.findall(r'Rule: if something is (.*?), then it is (not \w+|\w+)\.',s)]
    entities={a for a,b in known}
    while True:
        additions={(e,out) for e in entities for conditions,out in rules if all((e,c) in known for c in conditions)}
        if additions<=known:break
        known|=additions
    pos=(who,prop) in known;neg=(who,'!'+prop) in known
    return 'BOTH' if pos and neg else 'SUPPORTED' if pos else 'REFUTED' if neg else 'UNKNOWN'

def solve_relational_join(s: str,q: str) -> str:
    start,rels=one(r'Start at (\w+)\. Follow in order: ([\w, ]+)\.',q)
    edges=collections.defaultdict(set)
    for a,b,c in re.findall(r'Registry: (\w+) --(\w+)--> (\w+)\.',s):edges[a,b].add(c)
    cur=start
    for rel in rels.split(', '):
        options=edges[cur,rel]
        if len(options)!=1:return 'UNKNOWN'
        cur=next(iter(options))
    return cur

def solve_detective(s: str,q: str) -> str:
    candidates=one(r'Candidates: ([\w, ]+)\.',q)[0].split(', ')
    room,minute=one(r'Incident: the object was moved from (\w+) at minute (\d+)\.',s);minute=int(minute)
    requirements=dict(re.findall(r'Access rule: entering (\w+) required an? (\w+) badge\.',s))
    (thread,)=one(r'Trace report: the mover wore an? (\w+) scarf\.',s)
    present={p for p,rr,a,b in re.findall(r'Presence: (\w+) was in (\w+) from minute (\d+) through minute (\d+), inclusive\.',s) if rr==room and int(a)<=minute<=int(b)}
    badge_classes=dict(re.findall(r'Badge class: (\w+) was (\w+)\.',s))
    admitted={p for badge,p in re.findall(r'Badge assignment: (\w+) belonged to (\w+)\.',s) if badge_classes.get(badge)==requirements[room]}
    clothed={p for p,color in re.findall(r'Clothing: (\w+) wore an? (\w+) scarf\.',s) if color==thread}
    valid=set(candidates)&present&admitted&clothed
    return next(iter(valid)) if len(valid)==1 else 'UNKNOWN'

def solve_story_state(s: str,q: str) -> str:
    obj,cutoff=one(r'Where was (\w+) at minute (\d+)\?',q);cutoff=int(cutoff)
    aliases=dict(re.findall(r'Alias: (\w+) refers to (\w+)\.',s))
    def root(x):
        seen=set()
        while x in aliases:
            if x in seen:raise ValueError('cyclic alias')
            seen.add(x);x=aliases[x]
        return x
    moves=[]
    for t,p,o,loc in re.findall(r'At minute (\d+), (\w+) placed (\w+) in (\w+)\.',s):
        if root(o)==root(obj) and int(t)<=cutoff:moves.append((int(t),loc))
    return max(moves)[1] if moves else 'UNKNOWN'

def solve_document_reconciliation(s: str,q: str) -> str:
    project,cutoff=one(r'value for project (\w+) at revision snapshot (\d+)\?',q)
    docs=[(int(v),x) for p,v,st,x in re.findall(r'Document: project (\w+); revision (\d+); status (\w+); value (\w+)\.',s) if p==project and st=='APPROVED' and int(v)<=int(cutoff)]
    return max(docs)[1] if docs else 'UNKNOWN'

def solve_signed_ledger(s: str,q: str) -> str:
    (account,)=one(r'balance of account (\w+)\.',q)
    openings=[int(x) for a,x in re.findall(r'Opening: account (\w+) held (-?\d+) units\.',s) if a==account]
    if not openings:return 'UNKNOWN'
    if len(openings)!=1:raise ValueError('nonunique opening')
    seen={};total=openings[0]
    for rid,acc,delta,st in re.findall(r'Entry: receipt (\w+); account (\w+); delta ([+-]\d+); state (\w+)\.',s):
        if rid in seen and seen[rid]!=(acc,delta,st):raise ValueError('conflicting receipt')
        if rid not in seen and acc==account and st=='POSTED':total+=int(delta)
        seen[rid]=(acc,delta,st)
    return str(total)

def solve_temporal_order(s: str,q: str) -> str:
    ns=one(r'Order these events: ([\w, ]+)\.',q)[0].split(', ')
    constraints={(a,b) for a,b in re.findall(r'Constraint: (\w+) must occur before (\w+)\.',s) if a in ns and b in ns}
    solutions=[]
    for perm in itertools.permutations(ns):
        pos={x:i for i,x in enumerate(perm)}
        if all(pos[a]<pos[b] for a,b in constraints):solutions.append(perm)
        if len(solutions)>1:return 'UNKNOWN'
    return ' > '.join(solutions[0]) if solutions else 'IMPOSSIBLE'

def solve_truth_tellers(s: str,q: str) -> str:
    nstext,target=one(r'Speakers: ([\w, ]+)\. Determine the type of (\w+)\.',q);ns=nstext.split(', ')
    (count,)=one(r'Population fact: exactly (\d+) of these speakers are honest\.',s);count=int(count)
    statements={p:t for p,t in re.findall(r'Statement: (\w+) says "([^"]+)"\.',s) if p in ns}
    if set(statements)!=set(ns):raise ValueError('incomplete speaker statements')
    def statement_true(text: str,honest: set[str]) -> bool:
        if m:=re.fullmatch(r'(\w+) is honest',text):return m[1] in honest
        if m:=re.fullmatch(r'(\w+) is a liar',text):return m[1] not in honest
        if m:=re.fullmatch(r'(\w+) and (\w+) have the same type',text):return (m[1] in honest)==(m[2] in honest)
        if m:=re.fullmatch(r'(\w+) and (\w+) have different types',text):return (m[1] in honest)!=(m[2] in honest)
        if m:=re.fullmatch(r'(\w+) and (\w+) are both honest',text):return {m[1],m[2]}<=honest
        if m:=re.fullmatch(r'at least one of (\w+) and (\w+) is honest',text):return bool({m[1],m[2]}&honest)
        raise ValueError(f'unknown statement: {text}')
    possible=set()
    if count<=len(ns):
        for combo in itertools.combinations(ns,count):
            h=set(combo)
            if all(statement_true(statements[p],h)==(p in h) for p in ns):possible.add(target in h)
    return 'INCONSISTENT' if not possible else 'UNKNOWN' if len(possible)==2 else 'HONEST' if True in possible else 'LIAR'

def solve_hanoi(s: str,q: str) -> str:
    n=int(one(r'Solve the (\d+)-disk Hanoi instance\.',q)[0])
    initial={int(d):p for d,p in re.findall(r'Initial disk: (\d+) is on peg ([ABC])\.',s)}
    goal={int(d):p for d,p in re.findall(r'Goal disk: (\d+) is on peg ([ABC])\.',s)}
    if set(initial)!=set(range(1,n+1)) or set(goal)!=set(initial):raise ValueError('incomplete Hanoi state')
    # Checker uses explicit peg stacks and reverse BFS, unlike forward tuple BFS in generator.
    def to_stacks(d):return tuple(tuple(i for i in range(n,0,-1) if d[i]==p) for p in 'ABC')
    start=to_stacks(initial);end=to_stacks(goal)
    def neighbors(st):
        for a,peg in enumerate(st):
            if not peg:continue
            disk=peg[-1]
            for b,other in enumerate(st):
                if a==b or (other and other[-1]<disk):continue
                new=list(st);new[a]=peg[:-1];new[b]=other+(disk,)
                yield tuple(new),(disk,'ABC'[a],'ABC'[b])
    distance={end:0};queue=collections.deque([end])
    while queue:
        st=queue.popleft()
        for new,_ in neighbors(st):
            if new not in distance:distance[new]=distance[st]+1;queue.append(new)
    d=distance[start]
    if d==0:return 'moves=0; next=NONE'
    options=[mv for new,mv in neighbors(start) if distance[new]==d-1]
    disk,a,b=min(options)
    return f'moves={d}; next={disk}:{a}->{b}'

def solve_weighted_routes(s: str,q: str) -> str:
    start,goal=one(r'least-cost route from (\w+) to (\w+)\.',q)
    edges=[(a,b,int(w)) for a,b,w in re.findall(r'Road: (\w+) -> (\w+); cost (\d+)\.',s)]
    nodes={x for a,b,w in edges for x in (a,b)}|{start,goal};best={start:(0,(start,))}
    # Bellman-Ford with lexical tie-breaking, independent of generator's Dijkstra.
    for _ in range(len(nodes)-1):
        nxt=dict(best)
        for a,b,w in edges:
            if w<=0:raise ValueError('road weights must be positive')
            if a in best:
                cand=(best[a][0]+w,best[a][1]+(b,))
                if b not in nxt or cand<nxt[b]:nxt[b]=cand
        if nxt==best:break
        best=nxt
    if goal not in best:return 'UNREACHABLE'
    return f'cost={best[goal][0]}; path='+' -> '.join(best[goal][1])

def solve_counterfactual(s: str,q: str) -> str:
    intervention,forced,target=one(r'Under intervention do\((\w+)=(0|1)\), is (\w+) ON or OFF\?',q)
    inputs={x:int(v) for x,v in re.findall(r'Input: (\w+) = (0|1)\.',s)}
    gates={out:(a,op,b) for out,a,op,b in re.findall(r'Gate: (\w+) = (\w+) (AND|OR|XOR) (\w+)\.',s)}
    memo={intervention:int(forced)};active=set()
    def evaluate(x):
        if x in memo:return memo[x]
        if x in active:raise ValueError('cycle in Boolean system')
        active.add(x)
        if x in inputs:v=inputs[x]
        else:
            a,op,b=gates[x];va,vb=evaluate(a),evaluate(b)
            if op=='AND':v=int(bool(va) and bool(vb))
            elif op=='OR':v=int(bool(va) or bool(vb))
            else:v=int(bool(va)!=bool(vb))
        active.remove(x);memo[x]=v;return v
    return 'ON' if evaluate(target) else 'OFF'

def solve_program_trace(s: str,q: str) -> str:
    (target,)=one(r'final value of counter (\w+)\?',q)
    state={v:int(x) for v,x in re.findall(r'Initial counter: (\w+) = (-?\d+)\.',s)}
    steps=sorted((int(i),t) for i,t in re.findall(r'Step (\d+): ([^\n.]+)\.',s))
    if len({i for i,t in steps})!=len(steps):raise ValueError('duplicate step index')
    if [i for i,t in steps]!=list(range(1,len(steps)+1)):raise ValueError('non-contiguous steps')
    for i,text in steps:
        z=text.split();op=z[0]
        if op=='SET':state[z[1]]=int(z[2])
        elif op=='ADD':state[z[1]]=state[z[1]]+int(z[2])
        elif op=='MUL':state[z[1]]=state[z[1]]*int(z[2])
        elif op=='SWAP':a,b=z[1:];old=state[a];state[a]=state[b];state[b]=old
        elif op=='IFGT':
            if z[3]!='ADD':raise ValueError('bad conditional')
            if state[z[1]]>int(z[2]):state[z[4]]+=int(z[5])
        else:raise ValueError('unrecognized instruction')
    return str(state[target])

def solve_grounded_summary(s: str,q: str) -> str:
    (team,)=one(r'central idea and current outcome for (\w+)\.',q)
    goals=dict(re.findall(r'Goal record for (\w+): The group set out to ([^.]+)\.',s))
    obstacles=dict(re.findall(r'Obstacle record for (\w+): The work was delayed because ([^.]+)\.',s))
    decisions=[(int(v),a) for p,v,st,a in re.findall(r'Decision record for (\w+): revision (\d+); status (\w+); action ([^.]+)\.',s) if p==team and st=='ADOPTED']
    outcomes={p:(st,a) for p,st,a in re.findall(r'Outcome record for (\w+): status (\w+); report ([^.]+)\.',s)}
    if not decisions or team not in goals or team not in obstacles or team not in outcomes:raise ValueError('summary evidence incomplete')
    final=outcomes[team][1] if outcomes[team][0]=='CONFIRMED' else 'the final outcome is still unconfirmed'
    return f'The {team} team aimed to {goals[team]}, but {obstacles[team]}. It chose to {max(decisions)[1]}; {final}.'

def solve_dialogue_memory(s: str,q: str) -> str:
    (target,)=one(r'current preferences of (\w+)\?',q)
    vals={}
    for t,p,v,f in re.findall(r'Turn (\d+): (\w+) said, "Please use (\w+) for my (\w+)\."',s):
        if p==target and (f not in vals or int(t)>vals[f][0]):vals[f]=(int(t),v)
    return '; '.join(f'{f}={vals[f][1] if f in vals else "UNKNOWN"}' for f in ['venue','format','day'])

def solve_communication(s: str,q: str) -> str:
    (mode,)=one(r'Write the (\w+) reply\.',q)
    p=dict(re.findall(r'Approved detail: (\w+) = ([^.]+)\.',s))
    name,event=p['recipient'],p['event']
    if mode=='DECLINE':return 'Hi '+name+', thank you for inviting me to '+event+'. I cannot attend on '+p['unavailable']+', but I am available on '+p['alternative']+'. Would that work for you?'
    if mode=='CLARIFY':return 'Hi '+name+', I can help with '+event+'. Could you confirm the '+p['missing']+' before I make a commitment?'
    if mode=='CORRECT':return 'Hi '+name+', a quick correction: '+event+' is scheduled for '+p['confirmed']+', not '+p['old']+'. Thank you for updating your plans.'
    if mode=='UPDATE':return 'Hi '+name+', thank you for coordinating '+event+'. I plan to '+p['action']+' by '+p['deadline']+'; that work is not completed yet.'
    raise ValueError('unrecognized communication mode')

def solve_source_boundaries(s: str,q: str) -> str:
    item,field=one(r'For item (\w+), what is its (\w+)\?',q)
    matches={v for i,f,v in re.findall(r'Verified register: item (\w+); field (\w+); value (\w+)\.',s) if i==item and f==field}
    return next(iter(matches)) if len(matches)==1 else 'UNKNOWN'

SOLVERS={name[6:]:fn for name,fn in list(globals().items()) if name.startswith('solve_')}

def solve_prompt(task: str,prompt: str) -> str:
    s,q=parts(prompt)
    if task not in SOLVERS:raise ValueError(f'Unknown task family: {task}')
    return SOLVERS[task](s,q)

def iter_jsonl(path: Path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        for line_number,line in enumerate(f,1):
            try:yield json.loads(line)
            except Exception as e:raise ValueError(f'{path}:{line_number}: invalid JSON') from e

def sha(text: str):return hashlib.sha256(text.encode()).hexdigest()
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'),sort_keys=True)

def validate_row(row: dict[str,Any],meta: dict[str,Any]) -> None:
    assert row['id']==meta['id'],'id mismatch'
    assert set(row)=={'id','messages'},'unexpected training columns'
    msgs=row['messages'];assert [m['role'] for m in msgs]==['system','user','assistant'],'invalid roles'
    assert all(set(m)=={'role','content'} for m in msgs),'unexpected message fields'
    assert all(isinstance(m['content'],str) and m['content'].strip() for m in msgs),'empty text'
    user=msgs[1]['content'];answer=msgs[2]['content']
    assert answer==solve_prompt(meta['family'],user),'reference mismatch against text oracle'
    assert answer==meta['reference_answer'],'metadata answer mismatch'
    assert sha(user)==meta['prompt_sha256'],'prompt hash mismatch'
    assert sha(compact(row))==meta['record_sha256'],'record hash mismatch'
    assert len(user.split())==meta['actual_user_words'],'word count mismatch'
    assert all('['+x+']' in user for x in meta['core_record_ids']),'missing evidence record'
    assert 'token_count' in meta and meta['token_count'] is None,'cannot claim unmeasured tokens'
    assert row['id'] not in user,'record id leaked into prompt'
    assert 'reference_answer' not in user and 'required_slots' not in user,'audit metadata leaked'
    if meta['family']=='grounded_summary':assert len(answer.split())<=80 and answer.count('.')<=2,'summary length'
    if meta['family']=='communication':assert len(answer.split())<=65 and sum(answer.count(c) for c in '.?!')<=3,'communication length'

def validate_release(root: Path) -> dict[str,Any]:
    seen_ids=set();seen_prompts=set();seen_cases=set();counts={};matrix=collections.defaultdict(collections.Counter)
    total=0
    for path in sorted((root/'data').glob('*.jsonl.gz')):
        split=path.name.split('.')[0];meta_path=root/'metadata'/path.name
        metas=list(iter_jsonl(meta_path));rows=iter_jsonl(path);count=0
        for row,meta in itertools.zip_longest(rows,metas):
            assert row is not None and meta is not None,'metadata/data lengths differ'
            validate_row(row,meta)
            assert meta['split']==split,'split mismatch'
            assert row['id'] not in seen_ids,'row/group overlap across splits'
            assert meta['prompt_sha256'] not in seen_prompts,'exact prompt overlap'
            assert meta['case_sha256'] not in seen_cases,'exact canonical case overlap'
            seen_ids.add(row['id']);seen_prompts.add(meta['prompt_sha256']);seen_cases.add(meta['case_sha256']);count+=1;total+=1
            if meta['family']=='deduction':matrix[meta['reference_answer']][meta['position_mode']]+=1
        counts[split]=count;print(f'checked {split}: {count}',flush=True)
    report={'status':'PASS','verified_rows':total,'split_counts':counts,'checks':[
        'JSON/schema and roles','independent text-side answer recomputation','reference and prompt hashes',
        'record/group uniqueness','exact prompt disjointness','exact canonical case disjointness',
        'evidence-record presence','length constraints for generated responses','no audit fields in training columns'],
        'deduction_label_by_position':{a:dict(b) for a,b in matrix.items()},
        'not_checked':['real model token lengths','human prose quality','model efficacy','external or private benchmark overlap','near-duplicate semantic templates']}
    (root/'reports'/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(validate_release(a.root),indent=2))
