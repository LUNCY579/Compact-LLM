#!/usr/bin/env python3
"""Create matched evaluation-only context-layout views from test_iid worlds.
These are derived controls, NOT additional independent test cases.
Full-length views share exactly the same record multiset; only layout differs.
"""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,random,re
from pathlib import Path
from verify import iter_jsonl,parts,solve_prompt
from forge import FAMILIES,SYSTEM,compact,digest,gzip_write

def relayout(user: str,core_ids: list[str],mode: str):
    s,q=parts(user);head=user.split('\nRECORDS\n',1)[0]
    rows=s.strip().split('\n\n');core=[];noise=[]
    for line in rows:
        rid=re.match(r'\[(R\d+)\]',line).group(1)
        (core if rid in core_ids else noise).append(line)
    if mode=='short':
        # All original core records survive. Retain a little nearby noise as well.
        selected=core+noise[:4]
    elif mode=='early':selected=core+noise
    elif mode=='late':selected=noise+core
    elif mode=='middle':
        mid=len(noise)//2;selected=noise[:mid]+core+noise[mid:]
    elif mode=='distributed':
        selected=list(noise)
        for i,line in enumerate(core):selected.insert(round(i*len(selected)/max(1,len(core))),line)
    else:raise ValueError(mode)
    return head+'\nRECORDS\n'+'\n\n'.join(selected)+'\n\nEND RECORDS\n\nQUESTION\n'+q

def build(root: Path):
    metas={m['id']:m for m in iter_jsonl(root/'metadata/test_iid.jsonl.gz')};picked=collections.Counter();rows=[];mm=[]
    for row in iter_jsonl(root/'data/test_iid.jsonl.gz'):
        m=metas[row['id']];fam=m['family']
        if picked[fam]>=2 or m['requested_user_words']!=4096:continue
        if row['messages'][-1]['content'] in ('UNKNOWN','INCONSISTENT','IMPOSSIBLE','UNREACHABLE'):continue
        picked[fam]+=1
        for mode in ['short','early','middle','late','distributed']:
            user=relayout(row['messages'][1]['content'],m['core_record_ids'],mode)
            answer=solve_prompt(fam,user)
            assert answer==row['messages'][-1]['content'],'layout altered answer'
            rid=row['id']+'-probe-'+mode
            rows.append({'id':rid,'messages':[row['messages'][0],{'role':'user','content':user},row['messages'][-1]]})
            mm.append({'id':rid,'parent_id':row['id'],'group_id':row['id'],'split':'test_iid_derived_probe',
              'family':fam,'layout':mode,'reference_answer':answer,'answer_category':m['answer_category'],
              'required_slots':m['required_slots'],'user_words':len(user.split()),'token_count':None,
              'prompt_sha256':digest(user),'independent_sample':False})
    assert set(picked)==set(FAMILIES) and all(v==2 for v in picked.values()),picked
    out=root/'probes';out.mkdir(exist_ok=True)
    for filename,objects in [('layout_views.jsonl.gz',rows),('layout_metadata.jsonl.gz',mm)]:
        with gzip_write(out/filename) as f:
            for x in objects:f.write((compact(x)+'\n').encode())
    report={'derived_views':len(rows),'underlying_test_iid_worlds':sum(picked.values()),'views_per_world':5,
      'full_length_views':'identical record multiset, different ordering','short_view':'same complete core records, reduced distractors',
      'reference_invariance':'PASS','training_use':'PROHIBITED: these descend from test_iid',
      'statistics':'Use paired/group-level analysis; 160 views are only 32 independent worlds.'}
    (root/'reports/probe_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);build(p.parse_args().root)
