"""Independent public-spec search. Uses closure, symbolic replay, and enumeration.

Does not import the reward verifier or construction witnesses. References are
not model rollouts. No potentially valid certificate is rejected for being unlisted.
"""
from __future__ import annotations
from itertools import combinations
from .contracts import validate,canonical


def derive(s):
    n=len(s['initial'])
    candidates=set(range(n))
    if 'lookup' in s:
        closure={k:v for k,v in s['lookup']['links'].items() if type(v) is int}
        for _ in range(len(s['lookup']['links'])):
            for k,v in s['lookup']['links'].items():
                if type(v) is str and v in closure:closure[k]=closure[v]
        if any(q not in closure for q in s['lookup']['requested']):return None,[]
        candidates={closure[q] for q in s['lookup']['requested']}
    # (initial-register provenance or constant, accumulated offset)
    expressions=[(i,0) for i in range(n)]
    if 'updates' in s:
        for op,i,x in s['updates']['operations']:
            if op=='add':expressions[i]=(expressions[i][0],expressions[i][1]+x)
            elif op=='set':expressions[i]=(None,x)
            elif op=='copy':expressions[i]=expressions[x]
            else:expressions[i],expressions[x]=expressions[x],expressions[i]
    values=[(s['initial'][j] if j is not None else 0)+b for j,b in expressions]
    if 'updates' in s:candidates={i for i in candidates if values[i]>=s['updates']['cutoff']}
    return candidates,values


def orders(items,edges,limit):
    out=[];remaining=set(items);requirements={v:{u for u,w in edges if w==v and u in remaining} for v in remaining}
    def expand(prefix,left):
        if len(out)>=limit:return
        if not left:out.append(prefix);return
        for v in sorted(left):
            if not(requirements[v]&left):expand(prefix+[v],left-{v})
    expand([],remaining)
    return out


def answers(s: dict,limit: int=3) -> list[str]:
    validate(s)
    candidates,values=derive(s)
    if not candidates:return [canonical({'answer':'UNKNOWN'})]
    groups=[tuple(sorted(candidates))] if 'selection' not in s else combinations(sorted(candidates),s['selection']['k'])
    result=[]
    for group in groups:
        if 'selection' in s and sum(values[i] for i in group)!=s['selection']['target']:continue
        for seq in orders(group,s.get('precedence',[]),limit-len(result)):
            result.append(canonical({'answer':seq}))
            if len(result)>=limit:return result
    return result or [canonical({'answer':'UNKNOWN'})]


def reachable_sums(s):
    candidates,values=derive(s)
    if not candidates or 'selection' not in s:return set()
    return {sum(values[i] for i in group) for group in combinations(sorted(candidates),s['selection']['k'])}
