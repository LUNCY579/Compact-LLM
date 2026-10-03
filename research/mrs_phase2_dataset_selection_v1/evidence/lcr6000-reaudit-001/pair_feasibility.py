"""Exact integer transportation feasibility of each fixed-marginal pair table; no dataset writes."""
import collections, hashlib, json, math
from pathlib import Path
R=Path(__file__).resolve().parent
balance=json.loads((R/'balance_reaudit.json').read_text())
def solve(cells):
    left=sorted({c['left'] for c in cells});right=sorted({c['right'] for c in cells})
    rows={a:sum(c['actual'] for c in cells if c['left']==a) for a in left}
    cols={b:sum(c['actual'] for c in cells if c['right']==b) for b in right}
    bounds={(c['left'],c['right']):(math.ceil(c['proportional_expectation']*.97-1e-10),math.floor(c['proportional_expectation']*1.03+1e-10)) for c in cells}
    needs={a:rows[a]-sum(bounds[a,b][0] for b in right) for a in left}
    wants={b:cols[b]-sum(bounds[a,b][0] for a in left) for b in right}
    if any(x<0 for x in (*needs.values(),*wants.values())) or any(lo>hi for lo,hi in bounds.values()):return {'feasible':False,'reason':'lower/upper bound conflict'}
    caps=collections.defaultdict(dict)
    def add(a,b,n):caps[a][b]=n;caps[b][a]=0
    for a,n in needs.items():add('S',('L',a),n)
    for (a,b),(lo,hi) in bounds.items():add(('L',a),('R',b),hi-lo)
    for b,n in wants.items():add(('R',b),'T',n)
    flow=0
    while True:
        parent={'S':None};queue=collections.deque(['S'])
        while queue and 'T' not in parent:
            u=queue.popleft()
            for v,cap in caps[u].items():
                if cap>0 and v not in parent:parent[v]=u;queue.append(v)
        if 'T' not in parent:break
        path=[];v='T'
        while parent[v] is not None:path.append((parent[v],v));v=parent[v]
        extra=min(caps[u][v] for u,v in path)
        for u,v in path:caps[u][v]-=extra;caps[v][u]+=extra
        flow+=extra
    ok=flow==sum(needs.values())
    witness=[{'left':a,'right':b,'count':bounds[a,b][0]+caps[('R',b)][('L',a)]} for a,b in bounds] if ok else None
    return {'feasible':ok,'required_residual_flow':sum(needs.values()),'achieved_flow':flow,'witness':witness}
result={'scope':'Each global/within-band pair separately, preserving both realized marginals. Exact integer max-flow with +/-3% relative-count cell bounds. Does not prove a single six-way table can realize every pair witness simultaneously.',
        'global':{p:solve(cs) for p,cs in balance['pair_tables'].items()},
        'within_band':{b:{p:solve(cs) for p,cs in ts.items()} for b,ts in balance['within_band_pair_tables'].items()}}
result['all_pair_tables_individually_feasible']=all(v['feasible'] for v in result['global'].values()) and all(v['feasible'] for ts in result['within_band'].values() for v in ts.values())
result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with (R/'balance_pair_feasibility.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps({'all_pair_tables_individually_feasible':result['all_pair_tables_individually_feasible'],'failures':[(b,p) for b,ts in result['within_band'].items() for p,v in ts.items() if not v['feasible']]}))
