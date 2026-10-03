"""Matched held-out layout, core-preserving short, and causal-evidence probes."""
from __future__ import annotations
import argparse
import copy
import random
from collections import defaultdict
from pathlib import Path
from .common import FAMILIES, Writer, digest, dumps, read_jsonl, write_json, sha_file, loads
from .tasks import solve
from .text import render, parse_prompt, canonical_case, LAYOUTS
from .oracle import audit_answer, verify_solution
from .rewards import score_response


def changed_world(source: dict) -> dict:
    w=copy.deepcopy(source);f=w['family'];q=w['query'];r=w['records'];old=solve(w)[0]
    if f=='ledger_replay':
        next(e for e in r if e['kind']=='initial' and e['account']==q['account'])['amount'] += 1
    elif f=='register_machine':
        tick=max(x.get('tick',0) for x in r)+1;q['cutoff']=tick
        r.append(dict(kind='add',tick=tick,register=q['register'],other=next(x for x in 'XYZ' if x!=q['register']),value=1,threshold=0))
    elif f in ('temporal_route','continuous_replan'):
        if old.get('status')=='IMPOSSIBLE':q.update(start=q['goal'],waypoints=[])
        elif old['cost']==0:q['goal']=next(x for x in q['nodes'] if x!=q['start'])
        else:
            for e in r:e['cost']*=2
    elif f=='prerequisite_plan':
        if old.get('status')=='IMPOSSIBLE':q.update(goal_on=q['initial'],goal_off=((1<<q['bits'])-1)^q['initial'])
        elif old['cost']==0:
            q.update(goal_on=((1<<q['bits'])-1)^q['initial'],goal_off=q['initial'])
        else:
            for e in r:e['cost']*=2
    elif f=='deadline_schedule':
        if old.get('status')=='IMPOSSIBLE':
            for e in r:e['deadline']=999
        else:
            for e in r:e['duration']+=1
    elif f=='budget_selection':
        if old.get('status')=='UNKNOWN' or old.get('value')==0:
            for e in r:
                if e['kind']=='limit':e.update(status='approved',value=100)
                else:e.update(cost=1,weight=1)
        else:
            for e in r:
                if e['kind']=='item':e['value']*=2
    else:
        tick=max(x['tick'] for x in r);q['start']='Zprobe0'
        for i,key in enumerate(q['path']):
            tick+=1
            value=f'Zprobe{i+1}' if i<len(q['path'])-1 else 'S_changed'
            r.append(dict(kind='field',entity=f'Zprobe{i}',key=key,value=value,tick=tick,status='approved'))
        q['cutoff']=tick
    cert=audit_answer(w)
    if verify_solution(w,old,cert)['correct']:
        raise AssertionError('Perturbation did not invalidate the old reference')
    return w


def build_probes(root: Path, parents_per_family: int = 8) -> dict:
    if not 1 <= parents_per_family <= 32:raise ValueError('Invalid parent count')
    if (root/'probes').exists():raise FileExistsError('Probes already exist')
    byfamily=defaultdict(list)
    for row in read_jsonl(root/'metadata/test_iid.jsonl.gz'):byfamily[row['family']].append(row)
    n=parents=0
    with Writer(root/'probes/views.jsonl.gz') as data, Writer(root/'probes/metadata.jsonl.gz') as metadata:
        for family in FAMILIES:
            candidates=byfamily[family]
            k=min(parents_per_family,len(candidates))
            indices=sorted({round(i*(len(candidates)-1)/(k-1)) if k>1 else 0 for i in range(k)})
            for index in indices:
                source=candidates[index];parents+=1
                world=source['world'];changed=changed_world(world)
                seed=digest([source['id'],'matched-probes-v1'])
                long_records=None
                for variant in ('short',*LAYOUTS,'evidence_edit'):
                    w=changed if variant=='evidence_edit' else world
                    tier=0 if variant=='short' else 3
                    layout=variant if variant in LAYOUTS else 'distributed'
                    prompt,positions=render(w,random.Random(seed),tier,layout)
                    parsed=parse_prompt(prompt);answer=solve(parsed)[0];cert=audit_answer(parsed)
                    if not verify_solution(parsed,answer,cert)['correct']:raise AssertionError('Probe reference mismatch')
                    if variant in LAYOUTS:
                        records=sorted(line for line in prompt[1]['content'].splitlines() if line.startswith('Record '))
                        if long_records is not None and records!=long_records:raise AssertionError('Matched layouts changed record multiset')
                        long_records=records
                    if variant!='evidence_edit' and not verify_solution(parsed,source['reference'],cert)['correct']:raise AssertionError('Layout changed answer')
                    rid=source['id']+'-probe-'+variant
                    data.write(dict(id=rid,group_id=source['group_id'],family=family,split='test_iid',variant=variant,prompt=prompt))
                    metadata.write(dict(id=rid,group_id=source['group_id'],family=family,split='test_iid',variant=variant,
                                        world=parsed,reference=answer,certificate=cert,prompt_sha256=digest(prompt),layout=positions))
                    n+=1
    report=dict(status='PASS',views=n,parent_worlds=parents,variants=['short',*LAYOUTS,'evidence_edit'],
                additional_independent_worlds=0,heldout_only=True,model_calls=0,
                long_layout_record_multisets_equal=True,all_evidence_edits_invalidate_original_reference=True,
                limitation='Evidence edits may change query or task parameters as well as records; not all are one-record minimal interventions.')
    write_json(root/'reports/probe_report.json',report)
    files={str(p.relative_to(root)).replace('\\','/'):{'sha256':sha_file(p),'bytes':p.stat().st_size}
           for p in sorted((root/'probes').rglob('*')) if p.is_file()}
    write_json(root/'PROBE_MANIFEST.json',{'files':files})
    return report


def evaluate_probes(root: Path, predictions: Path) -> dict:
    manifest=loads((root/'PROBE_MANIFEST.json').read_text())
    for rel,entry in manifest['files'].items():
        path=(root/rel).resolve()
        if not path.is_relative_to(root.resolve()) or sha_file(path)!=entry['sha256']:raise ValueError('Probe integrity failed')
    refs={r['id']:r for r in read_jsonl(root/'probes/metadata.jsonl.gz')}
    pred={}
    for row in read_jsonl(predictions):
        if set(row)!={'id','prediction'} or not isinstance(row['id'], str) or row['id'] not in refs or row['id'] in pred or not isinstance(row['prediction'],str):raise ValueError('Invalid or duplicate probe prediction')
        pred[row['id']]=row['prediction']
    groups=defaultdict(dict);byvariant=defaultdict(list)
    for rid,r in refs.items():
        score=score_response(r['world'],pred.get(rid,''),certificate=r['certificate'])['correct']
        groups[r['group_id']][r['variant']]=(score,rid in pred)
        byvariant[r['variant']].append(score)
    paired={}
    for variant in (*LAYOUTS,'evidence_edit'):
        eligible=[v for v in groups.values() if v['short'][1] and v[variant][1]]
        paired[variant]=dict(complete_pairs=len(eligible),long_or_edited_wins=sum(not x['short'][0] and x[variant][0] for x in eligible),
                             long_or_edited_losses=sum(x['short'][0] and not x[variant][0] for x in eligible),
                             both_correct=sum(x['short'][0] and x[variant][0] for x in eligible))
    return dict(parent_worlds=len(groups),views_expected=len(refs),views_received=len(pred),
                accuracy_by_variant_missing_as_wrong={k:sum(v)/len(v) for k,v in byvariant.items()},paired=paired,
                independent_unit='parent world, not a view',model_calls=0,
                scope='layout robustness and task-evidence responsiveness; not model performance unless supplied predictions are real model outputs')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--predictions',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.predictions:
        if a.output is None or a.output.exists():raise ValueError('A new --output is required for scoring')
        result=evaluate_probes(a.root,a.predictions);write_json(a.output,result)
    else:result=build_probes(a.root)
    print(dumps(result))


if __name__=='__main__':main()
