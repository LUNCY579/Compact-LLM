"""Binary acceptance of ANY legal certificate; independent of construction/search."""
from __future__ import annotations
from .contracts import validate, strict_json, intlist, require, VERSION


def concrete(s):
    values=s['initial'][:]; candidates=set(range(len(values)))
    if 'lookup' in s:
        links=s['lookup']['links']; candidates=set()
        for alias in s['lookup']['requested']:
            seen=set();node=alias
            while type(node) is str:
                if node in seen or node not in links:return None,values
                seen.add(node);node=links[node]
            candidates.add(node)
    if 'updates' in s:
        for op,i,x in s['updates']['operations']:
            if op=='add':values[i]+=x
            elif op=='set':values[i]=x
            elif op=='copy':values[i]=values[x]
            else:values[i],values[x]=values[x],values[i]
        candidates={i for i in candidates if values[i]>=s['updates']['cutoff']}
    return candidates,values


def feasible(s,candidates,values):
    """Cardinality-sum DP; acyclic precedence guarantees an order for each subset."""
    if not candidates:return False
    if 'selection' not in s:return True
    k=s['selection']['k']; target=s['selection']['target']
    reachable=[set() for _ in range(k+1)];reachable[0].add(0)
    for i in sorted(candidates):
        for count in range(k,0,-1):
            reachable[count].update(t+values[i] for t in reachable[count-1])
    return target in reachable[k]


def score(spec: dict,response: str) -> int:
    validate(spec) # corrupt metadata is a host error, never an ordinary model failure
    try:
        obj=strict_json(response,4096)
        if type(obj) is not dict or set(obj)!={'answer'}:return 0
        answer=obj['answer']
        candidates,values=concrete(spec)
        if type(answer) is str:
            return int(answer=='UNKNOWN' and not feasible(spec,candidates,values))
        if candidates is None or not intlist(answer,0,len(values)-1) or not answer or len(answer)!=len(set(answer)):return 0
        selected=set(answer)
        if not selected<=candidates:return 0
        if 'selection' in spec:
            q=spec['selection']
            if len(answer)!=q['k'] or sum(values[i] for i in answer)!=q['target']:return 0
        elif selected!=candidates:return 0
        if 'precedence' in spec:
            position={v:i for i,v in enumerate(answer)}
            if any(position[u]>=position[v] for u,v in spec['precedence'] if u in selected and v in selected):return 0
        return 1
    except (ValueError,TypeError,UnicodeError,RecursionError,OverflowError):
        return 0


def compute_score(data_source: str,solution_str: str,ground_truth: str,extra_info=None) -> float:
    payload=strict_json(ground_truth)
    require(type(payload) is dict and set(payload)=={'version','kind','spec'} and payload['version']==VERSION,'verifier payload/version')
    if payload['kind']=='composition':
        require(data_source=='rineya_maxrl_comp_v2','data source mismatch')
        return float(score(payload['spec'],solution_str))
    if payload['kind']=='legacy':
        from experiments.maxrl_verifiable_data_v1.verifier import score as legacy_score
        q=payload['spec'];require(type(q) is dict and set(q)=={'family','problem'},'legacy schema')
        require(data_source=='rineya_maxrl_comp_v2/legacy/'+q['family'],'legacy data source mismatch')
        return float(legacy_score(q['family'],q['problem'],solution_str))
    raise ValueError('unknown verifier kind')
