from __future__ import annotations

"""Gold-verified residual mining and oracle-union audit for v2.3.

Teacher disagreement is not enough. A row enters a residual curriculum only
when the frozen greedy anchor is wrong and at least one discovery teacher is
verified correct against the packaged deterministic gold/evaluator contract.
Cases every model misses are archived as unresolved instead of becoming noisy
pseudo-labels.
"""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(p:Path)->list[dict[str,Any]]:return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def write_jsonl(p:Path,rows:list[dict[str,Any]])->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
def write_json(p:Path,o:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def direct_map(p:Path)->dict[str,dict[str,Any]]:return {str(r.get('id')):r for r in read_jsonl(p)}
def rag_map(p:Path)->dict[str,dict[str,Any]]:
    rows=[r for r in read_jsonl(p) if str(r.get('condition'))=='real_rag'];return {str(r.get('id')):r for r in rows}
def rag_correct(r:dict[str,Any])->bool:
    k=str(r.get('case_class') or '')
    if k=='supported':return bool(r.get('supported_answer_correct'))
    if k=='unsupported':return bool(r.get('abstained'))
    if k=='conflict':return bool(r.get('conflict_safe_response'))
    if k=='social':return bool(r.get('social_usable'))
    return False
def add_meta(row:dict[str,Any],**fields:Any)->dict[str,Any]:
    x=json.loads(json.dumps(row));m=dict(x.get('metadata') or {});m.update(fields);x['metadata']=m;return x
def rate(n:int,d:int)->float|None:return n/d if d else None

def with_assistant_training_target(row:dict[str,Any])->dict[str,Any]:
    """Make original Phase 9 preservation rows CompletionDataset-ready.

    Original reviewed splits are often prompt-only (system + user) with gold in
    `approved_answer` / `answer`. Residual corpora already include the assistant
    turn; preservation replay must append that target before SFT.
    """
    out=json.loads(json.dumps(row))
    messages=[dict(m) for m in (out.get('messages') or [])]
    if messages and messages[-1].get('role')=='assistant':
        content=str(messages[-1].get('content') or '').strip()
        if not content:raise ValueError(f"{out.get('id')}: empty assistant training target")
        out['messages']=messages;out.setdefault('reference',content);return out
    answer=str(out.get('approved_answer') or out.get('answer') or out.get('reference') or '').strip()
    if not answer:raise ValueError(f"{out.get('id')}: missing approved_answer/answer for assistant training target")
    if not messages:raise ValueError(f"{out.get('id')}: missing prompt messages for assistant training target")
    messages.append({'role':'assistant','content':answer});out['messages']=messages;out['reference']=answer;return out

def choose_preservation(rows:list[dict[str,Any]],results:dict[str,dict[str,Any]],limit:int,seed:int)->list[dict[str,Any]]:
    correct=[]
    for row in rows:
        rr=results.get(str(row.get('id')))
        if rr and rag_correct(rr):correct.append(row)
    def cls(r:dict[str,Any])->str:
        d=str(r.get('decision') or '');m=str(r.get('mode') or '');c=str(r.get('category') or '')
        if d=='social' or 'social' in m:return 'social'
        if d=='conflict' or 'conflict' in c:return 'conflict'
        if d=='abstain':return 'boundary'
        if d=='answer' and (r.get('procedures') or 'procedure' in m):return 'procedure_answer'
        return 'supported_answer'
    quota={'supported_answer':0.50,'procedure_answer':0.25,'social':0.10,'boundary':0.08,'conflict':0.07}
    b=defaultdict(list);rng=random.Random(seed)
    for r in correct:b[cls(r)].append(r)
    for xs in b.values():rng.shuffle(xs)
    out=[]
    for k,f in quota.items():out.extend(b[k][:min(len(b[k]),round(limit*f))])
    used={str(r.get('id')) for r in out};rest=[r for r in correct if str(r.get('id')) not in used];rng.shuffle(rest);out.extend(rest[:max(0,limit-len(out))]);rng.shuffle(out)
    return [add_meta(with_assistant_training_target(r),v23_role='preservation',v23_anchor_kl_multiplier=1.0,v23_preservation_class=cls(r)) for r in out[:limit]]

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--direct-benchmark',type=Path,required=True);ap.add_argument('--direct-anchor-results',type=Path,required=True)
    ap.add_argument('--direct-teacher-a-results',type=Path,required=True);ap.add_argument('--direct-teacher-b-results',type=Path,required=True);ap.add_argument('--direct-teacher-procedure-results',type=Path,required=True)
    ap.add_argument('--procedure-benchmark',type=Path,required=True);ap.add_argument('--procedure-anchor-results',type=Path,required=True);ap.add_argument('--procedure-teacher-b-results',type=Path,required=True);ap.add_argument('--procedure-teacher-procedure-results',type=Path,required=True)
    ap.add_argument('--preservation-benchmark',type=Path,required=True);ap.add_argument('--preservation-anchor-results',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--preservation-limit',type=int,default=2400);ap.add_argument('--seed',type=int,default=23061);a=ap.parse_args()
    direct_rows=read_jsonl(a.direct_benchmark);drow={str(r.get('id')):r for r in direct_rows};da=direct_map(a.direct_anchor_results);teachers={'track_a':direct_map(a.direct_teacher_a_results),'track_b':direct_map(a.direct_teacher_b_results),'procedure':direct_map(a.direct_teacher_procedure_results)}
    residual_r=[];unresolved_r=[];recoverable=0;anchor_correct=0;teacher_counts=Counter();union_correct=0;answerable_n=0
    for rid,row in drow.items():
        if str(row.get('decision'))!='answer':continue
        answerable_n+=1;ac=bool((da.get(rid) or {}).get('correct'));anchor_correct+=int(ac);good=[name for name,m in teachers.items() if bool((m.get(rid) or {}).get('correct'))]
        for g in good:teacher_counts[g]+=1
        union=ac or bool(good);union_correct+=int(union)
        if (not ac) and good:
            recoverable+=1;residual_r.append(add_meta(row,v23_role='residual_reasoning',v23_anchor_kl_multiplier=0.35,v23_verified_teachers=good,v23_anchor_failed=True))
        elif (not ac) and not good:unresolved_r.append(row)
    proc_rows=read_jsonl(a.procedure_benchmark);prow={str(r.get('id')):r for r in proc_rows};pa=rag_map(a.procedure_anchor_results);pt={'track_b':rag_map(a.procedure_teacher_b_results),'procedure':rag_map(a.procedure_teacher_procedure_results)}
    residual_p=[];unresolved_p=[];p_answerable=0;p_anchor_correct=0;p_union=0;p_teacher=Counter()
    for rid,row in prow.items():
        if str(row.get('decision'))!='answer':continue
        p_answerable+=1;ac=rag_correct(pa.get(rid) or {});p_anchor_correct+=int(ac);good=[name for name,m in pt.items() if rag_correct(m.get(rid) or {})]
        for g in good:p_teacher[g]+=1
        p_union+=int(ac or bool(good))
        if (not ac) and good:residual_p.append(add_meta(row,v23_role='residual_procedure',v23_anchor_kl_multiplier=0.30,v23_verified_teachers=good,v23_anchor_failed=True))
        elif (not ac) and not good:unresolved_p.append(row)
    preservation=choose_preservation(read_jsonl(a.preservation_benchmark),rag_map(a.preservation_anchor_results),a.preservation_limit,a.seed)
    a.output_root.mkdir(parents=True,exist_ok=True)
    write_jsonl(a.output_root/'reasoning_residual.reviewed.jsonl',residual_r);write_jsonl(a.output_root/'procedure_residual.reviewed.jsonl',residual_p);write_jsonl(a.output_root/'reasoning_unresolved.jsonl',unresolved_r);write_jsonl(a.output_root/'procedure_unresolved.jsonl',unresolved_p);write_jsonl(a.output_root/'preservation_replay.reviewed.jsonl',preservation)
    audit={'schema_version':'2.3','direct':{'answerable_cases':answerable_n,'anchor_correct':anchor_correct,'anchor_accuracy':rate(anchor_correct,answerable_n),'teacher_correct_counts':dict(teacher_counts),'oracle_union_correct':union_correct,'oracle_union_accuracy':rate(union_correct,answerable_n),'recoverable_anchor_failures':recoverable,'residual_rows':len(residual_r),'unresolved_anchor_failures':len(unresolved_r)},'procedural_rag':{'answerable_cases':p_answerable,'anchor_correct':p_anchor_correct,'anchor_accuracy':rate(p_anchor_correct,p_answerable),'teacher_correct_counts':dict(p_teacher),'oracle_union_correct':p_union,'oracle_union_accuracy':rate(p_union,p_answerable),'recoverable_anchor_failures':len(residual_p),'residual_rows':len(residual_p),'unresolved_anchor_failures':len(unresolved_p)},'preservation':{'requested':a.preservation_limit,'verified_anchor_correct_rows':len(preservation),'class_counts':dict(Counter((r.get('metadata') or {}).get('v23_preservation_class') for r in preservation))},'mining_rule':'anchor wrong AND at least one teacher gold-verified correct; targets always come from deterministic gold, never teacher free-form output'}
    write_json(a.output_root/'RESIDUAL_MINING_AUDIT.json',audit)
    md=['# v2.3 Residual Mining / Oracle Union Audit','',f"Direct answerable anchor: **{audit['direct']['anchor_accuracy']}**",f"Direct oracle union: **{audit['direct']['oracle_union_accuracy']}**",f"Direct recoverable residual rows: **{len(residual_r)}**",f"Direct unresolved anchor failures: **{len(unresolved_r)}**",'',f"Procedural anchor: **{audit['procedural_rag']['anchor_accuracy']}**",f"Procedural oracle union: **{audit['procedural_rag']['oracle_union_accuracy']}**",f"Procedural residual rows: **{len(residual_p)}**",f"Procedural unresolved anchor failures: **{len(unresolved_p)}**",'',f"Verified preservation replay rows: **{len(preservation)}**",'', 'Only gold-verified teacher successes become residual training rows.']
    (a.output_root/'RESIDUAL_MINING_AUDIT.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2,sort_keys=True))
if __name__=='__main__':main()
