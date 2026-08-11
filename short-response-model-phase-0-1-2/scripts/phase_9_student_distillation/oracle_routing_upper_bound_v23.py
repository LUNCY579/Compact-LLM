from __future__ import annotations

"""Compute best-of-candidate oracle routing upper bounds for v2.3 diagnostics."""
import argparse, json
from pathlib import Path
from typing import Any

def read_jsonl(p:Path)->list[dict[str,Any]]:return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def direct_correct_map(p:Path)->dict[str,bool]:return {str(r.get('id')):bool(r.get('correct')) for r in read_jsonl(p)}
def rag_correct(r:dict[str,Any])->bool:
    if str(r.get('condition'))!='real_rag':return False
    k=str(r.get('case_class') or '')
    if k=='supported':return bool(r.get('supported_answer_correct'))
    if k=='unsupported':return bool(r.get('abstained'))
    if k=='conflict':return bool(r.get('conflict_safe_response'))
    if k=='social':return bool(r.get('social_usable'))
    return False
def rag_correct_map(p:Path)->dict[str,bool]:return {str(r.get('id')):rag_correct(r) for r in read_jsonl(p) if str(r.get('condition'))=='real_rag'}
def union(maps:dict[str,dict[str,bool]])->dict[str,Any]:
    ids=sorted(set().union(*(set(m) for m in maps.values())));per={n:sum(bool(m.get(i)) for i in ids) for n,m in maps.items()};u=sum(any(bool(m.get(i)) for m in maps.values()) for i in ids)
    return {'case_count':len(ids),'candidate_correct_counts':per,'candidate_accuracies':{n:(c/len(ids) if ids else None) for n,c in per.items()},'oracle_union_correct':u,'oracle_union_accuracy':u/len(ids) if ids else None,'oracle_gain_over_best_single':(u-max(per.values()))/len(ids) if ids and per else None}
def parse_specs(specs:list[str],rag:bool)->dict[str,dict[str,bool]]:
    fn=rag_correct_map if rag else direct_correct_map;out={}
    for s in specs:
        n,p=s.split('=',1);out[n]=fn(Path(p))
    return out
def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--direct',action='append',default=[]);ap.add_argument('--rag',action='append',default=[]);ap.add_argument('--output-json',type=Path,required=True);ap.add_argument('--output-md',type=Path,required=True);a=ap.parse_args();out={'schema_version':'2.3'}
    if a.direct:out['direct']=union(parse_specs(a.direct,False))
    if a.rag:out['rag']=union(parse_specs(a.rag,True))
    a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8');lines=['# v2.3 Oracle Routing Upper Bound','']
    for k,v in out.items():
        if k=='schema_version':continue
        lines += [f'## {k}', '', f"- Cases: **{v['case_count']}**",f"- Oracle union accuracy: **{v['oracle_union_accuracy']}**",f"- Gain over best single: **{v['oracle_gain_over_best_single']}**",'', 'Candidate accuracies:']+[f"- {n}: `{x}`" for n,x in v['candidate_accuracies'].items()]+['']
    a.output_md.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
