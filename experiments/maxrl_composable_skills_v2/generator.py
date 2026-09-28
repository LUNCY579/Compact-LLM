"""Composable finite worlds and per-stage counterfactual positive/negative controls."""
from __future__ import annotations
import copy
import json
import random
from itertools import combinations
from .contracts import SKILLS,FIELDS,canonical,digest,signature,validate
from .oracle import answers,derive,reachable_sums
from .verifier import score


def world(rng: random.Random,level: int):
    groups=3 if level==0 else 4 if level==1 else 5
    n=3*groups
    indices=rng.sample(range(n),n)
    groupset=[indices[i*3:i*3+3] for i in range(groups)]
    base=rng.sample(range(1,36),groups)
    final=rng.sample(range(2,37),groups-1)+[-rng.randint(1,12)]
    initial=[0]*n;desired=[0]*n
    for g,members in enumerate(groupset):
        for i in members:initial[i]=base[g];desired[i]=final[g]
    # Diverse replay histories; terminal group equivalence permits multiple certificates.
    state=initial[:];ops=[]
    for _ in range(n*(level+1)):
        op=rng.choice(('add','add','copy','swap'));i,j=rng.sample(range(n),2)
        if op=='add':
            x=rng.choice([v for v in range(-9,10) if v]);state[i]+=x
        elif op=='copy':x=j;state[i]=state[j]
        else:x=j;state[i],state[j]=state[j],state[i]
        ops.append([op,i,x])
    for i in rng.sample(range(n),n):
        change=desired[i]-state[i]
        if change:ops.append(['add',i,change]);state[i]+=change
    links={};requested=[]
    for i in indices:
        leaf='r'+format(rng.getrandbits(28),'x');links[leaf]=i
        terminal=leaf
        for _ in range(level+1):
            alias='a'+format(rng.getrandbits(28),'x');links[alias]=terminal;terminal=alias
        if any(i in group[:2] for group in groupset):requested.append(terminal)
    rng.shuffle(requested)
    # Dense but non-total DAG: tied ranks leave alternative legal orders.
    order=rng.sample(range(n),n);rank={v:j//2 for j,v in enumerate(order)}
    edges=[[u,v] for u in range(n) for v in range(n) if rank[u]<rank[v]]
    w={'initial':initial,'lookup':{'links':links,'requested':requested},
       'updates':{'operations':ops,'cutoff':0},'precedence':edges}
    return w


def view(w:dict,sig:str,rng:random.Random,level:int):
    s={'initial':w['initial'][:]}
    for k in sig:
        if k!='S':s[FIELDS[k]]=copy.deepcopy(w[FIELDS[k]])
    if sig=='S': # oracle derive requires no particular active field until selection is bound
        candidates=set(range(len(s['initial'])));values=s['initial']
    else:candidates,values=derive(s)
    if 'S' in sig:
        # Choose at most one item per score class so a same-valued spare remains.
        representatives={}
        for i in sorted(candidates):representatives.setdefault(values[i],[]).append(i)
        k=2 if level==0 else min(3,len(representatives))
        if len(representatives)<k:return None
        chosen_values=rng.sample(sorted(representatives),k)
        s['selection']={'k':k,'target':sum(chosen_values)}
    validate(s)
    return s


def controls(s:dict,refs:list[str]):
    """For EACH active skill, change only that stage. Old answer must fail;
    the changed problem must still have an independently recovered valid answer.
    This is local dependency evidence, not proof of algorithmic difficulty.
    """
    candidates,values=derive(s)
    for original in refs:
        answer=json.loads(original)['answer']
        if type(answer) is not list:continue
        collected=[]
        for skill in signature(s):
            proposals=[]
            if skill=='R':
                # Replace one requested item by an otherwise-excluded equal-valued item.
                for i in answer:
                    for j in range(len(values)):
                        if j in candidates or values[j]!=values[i]:continue
                        q=copy.deepcopy(s)
                        for key,v in list(q['lookup']['links'].items()):
                            if type(v) is int and v==i:q['lookup']['links'][key]=j
                        proposals.append(q)
            elif skill=='U':
                # A selected item is made ineligible. A redundant equal-valued witness
                # supplies an alternative when selection must preserve the target.
                for i in answer:
                    q=copy.deepcopy(s)
                    q['updates']['operations'].append(['set',i,q['updates']['cutoff']-1])
                    proposals.append(q)
            elif skill=='S':
                possible=sorted(reachable_sums(s)-{s['selection']['target']})
                for target in possible[:2]+possible[-2:]:
                    q=copy.deepcopy(s);q['selection']['target']=target;proposals.append(q)
            else:
                # A single-stage relation change, not advertised as one-edge minimality.
                if len(answer)>=2:
                    q=copy.deepcopy(s);q['precedence']=[[answer[-1],answer[0]]];proposals.append(q)
            accepted=None
            for q in proposals:
                validate(q)
                if score(q,original):continue
                alt=answers(q,1)[0]
                if '"UNKNOWN"' not in alt and score(q,alt)==1:
                    accepted={'skill':skill,'spec':q,'transferred_response':original,'transferred_reward':0,
                              'positive_response':alt,'positive_reward':1};break
            if accepted is None:break
            collected.append(accepted)
        if len(collected)==len(signature(s)):
            return original,collected
    return None


def boundary_pair(s:dict):
    """Matched feasible/impossible selection targets INSIDE achievable min/max.

    Independent enumeration proves infeasibility; no hidden reference is needed
    by the verifier. Targets are neighbors in numeric space, not out-of-range cues.
    """
    sums=sorted(reachable_sums(s))
    gaps=[(a,b) for a,b in zip(sums,sums[1:]) if b>a+1]
    if not gaps:return None
    a,b=gaps[len(gaps)//2]
    yes=copy.deepcopy(s);no=copy.deepcopy(s)
    yes['selection']['target']=a
    no['selection']['target']=a+1
    ry=answers(yes);rn=answers(no)
    if '"UNKNOWN"' in ry[0] or rn!=['{"answer":"UNKNOWN"}']:return None
    assert all(score(yes,x)==1 for x in ry) and score(no,rn[0])==1
    return yes,no,ry,rn
