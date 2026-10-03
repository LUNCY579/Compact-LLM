#!/usr/bin/env python3
"""Score external predictions without invoking a model.
Exact-task accuracy is separated from lexical slot diagnostics for free-form tasks.
Slot coverage is NOT a semantic or stylistic correctness score.
"""
from __future__ import annotations
import argparse,collections,json,re
from pathlib import Path
from verify import iter_jsonl

def normalize(x):return ' '.join(x.strip().split())
def score(predictions: Path,metadata: Path,allow_partial: bool=False):
    metas={m['id']:m for m in iter_jsonl(metadata)};pred={}
    for row in iter_jsonl(predictions):
        if row['id'] in pred:raise ValueError('duplicate prediction id')
        if row['id'] not in metas:raise ValueError('unknown prediction id')
        if not isinstance(row.get('prediction'),str):raise ValueError('prediction must be a string')
        pred[row['id']]=row['prediction']
    missing=set(metas)-set(pred)
    if missing and not allow_partial:raise ValueError(f'{len(missing)} predictions missing; use --allow-partial only for explicit partial reports')
    stats=collections.defaultdict(lambda:{'n':0,'correct':0,'slot_hits':0,'slot_total':0,'kind':None})
    for rid,text in pred.items():
        m=metas[rid];z=stats[m['family']];z['n']+=1;z['kind']=m['answer_category']
        if m['answer_category']=='exact':z['correct']+=normalize(text)==normalize(m['reference_answer'])
        else:
            for key,value in m['required_slots'].items():
                if key=='mode':continue
                z['slot_total']+=1;z['slot_hits']+=normalize(value).casefold() in normalize(text).casefold()
    report={'coverage':len(pred)/max(1,len(metas)),'predictions':len(pred),'expected':len(metas),'by_family':{},
      'warning':'Free-form lexical slot coverage misses paraphrases and does not detect every unsupported claim. Human/rubric review is required.'}
    exact=[]
    for family,z in stats.items():
        if z['kind']=='exact':
            accuracy=z['correct']/z['n'];exact.append(accuracy);report['by_family'][family]={'n':z['n'],'exact_accuracy':accuracy}
        else:report['by_family'][family]={'n':z['n'],'lexical_slot_coverage_only':z['slot_hits']/max(1,z['slot_total'])}
    report['macro_exact_accuracy_on_reported_exact_families']=sum(exact)/len(exact) if exact else None
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();result=score(a.predictions,a.metadata,a.allow_partial);text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text)
