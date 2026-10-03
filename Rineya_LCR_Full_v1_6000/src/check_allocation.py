"""Independent stdlib census of the shipped plan. No tokenizer, model or corpus use."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math
from collections import Counter
from fractions import Fraction
from pathlib import Path


def audit(plan: Path, contract: Path, expected_sha256: str) -> dict:
    raw=plan.read_bytes()
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected_sha256:raise ValueError('Allocation hash mismatch')
    if b'\r' in raw:raise ValueError('Non-LF line endings')
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    cfg=json.loads(contract.read_text(encoding='utf-8'))
    margins=cfg['marginals'];n=cfg['total_rows']
    if len(rows)!=n:raise ValueError('Wrong planned row count')
    if any('messages' in r for r in rows):raise ValueError('A plan must not masquerade as chat data')
    for key in ('allocation_id','prospective_group_id'):
        if len(set(r[key] for r in rows))!=n:raise ValueError(f'Duplicate {key}')
    for dim,expected in margins.items():
        if dict(Counter(r[dim] for r in rows))!=expected:raise ValueError(f'Marginal mismatch: {dim}')
    def proportional(items):
        ans=Fraction(n)
        for dim,value in items.items():ans*=Fraction(margins[dim][value],n)
        return ans
    global_cells=0;max_abs=Fraction(0);max_rel=Fraction(0)
    for a,b in itertools.combinations(margins,2):
        count=Counter((r[a],r[b]) for r in rows)
        for av,bv in itertools.product(margins[a],margins[b]):
            e=proportional({a:av,b:bv});v=count[av,bv]
            if not math.floor(e)<=v<=math.ceil(e):raise ValueError('Pair bound failure')
            global_cells+=1;max_abs=max(max_abs,abs(v-e));max_rel=max(max_rel,abs(v-e)/e)
    conditional_cells=0;conditional_max_rel=Fraction(0)
    for band in margins['length_band']:
        subset=[r for r in rows if r['length_band']==band]
        for a,b in itertools.combinations([d for d in margins if d!='length_band'],2):
            count=Counter((r[a],r[b]) for r in subset)
            for av,bv in itertools.product(margins[a],margins[b]):
                e=proportional({'length_band':band,a:av,b:bv});v=count[av,bv]
                if not math.floor(e)<=v<=math.ceil(e):raise ValueError('Within-band pair bound failure')
                conditional_cells+=1;conditional_max_rel=max(conditional_max_rel,abs(v-e)/e)
    dims=list(margins);counts=Counter(tuple(r[d] for d in dims) for r in rows)
    for vals in itertools.product(*(margins[d] for d in dims)):
        e=proportional(dict(zip(dims,vals)));v=counts[vals]
        if not math.floor(e)<=v<=math.ceil(e):raise ValueError('Six-way cell bound failure')
    reasons=Counter(r['unanswerable_reason'] for r in rows if r['variant']=='unanswerable')
    if dict(reasons)!=cfg['unanswerable_reason_counts']:raise ValueError('Unknown reason quota failure')
    if any(r['unanswerable_reason'] is not None for r in rows if r['variant']!='unanswerable'):
        raise ValueError('Supported row has unknown mechanism')
    by_reason={}
    for split,band in itertools.product(margins['split'],margins['length_band']):
        subset=[r for r in rows if r['split']==split and r['length_band']==band and r['variant']=='unanswerable']
        dist=Counter(r['unanswerable_reason'] for r in subset)
        if len(set(dist.values()))!=1 or len(dist)!=4:raise ValueError('Unknown mechanism split/band imbalance')
        by_reason[split+'__'+band]=dict(dist)
    return {'status':'ALLOCATION_CENSUS_PASS_ONLY','planned_rows':n,'corpus_rows':0,
        'allocation_sha256':actual,'global_pair_tables':15,'global_pair_cells_checked':global_cells,
        'global_pair_max_absolute_error':float(max_abs),'global_pair_max_relative_error':float(max_rel),
        'within_band_pair_tables':30,'within_band_pair_cells_checked':conditional_cells,
        'within_band_pair_max_relative_error':float(conditional_max_rel),
        'six_way_cells_checked':math.prod(len(v) for v in margins.values()),
        'unknown_reasons_by_split_and_band':by_reason,
        'does_not_certify':['native token lengths','semantic correctness','needle placement','difficulty',
            'structural partitions','shortcut resistance','external exclusion','corpus acceptance']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    lock=json.loads((args.root/'ALLOCATION_LOCK.json').read_text(encoding='utf-8'))
    print(json.dumps(audit(args.root/'allocation/plan.jsonl',args.root/'source/ATTACHED_CONTRACT.json',lock['sha256']),indent=2))
