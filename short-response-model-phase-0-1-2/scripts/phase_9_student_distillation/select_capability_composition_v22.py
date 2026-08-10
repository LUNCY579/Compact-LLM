from __future__ import annotations

"""Build the v2.2 capability matrix, Pareto frontier, and DEV selection."""

import argparse, json
from pathlib import Path
from typing import Any

PROTECTED=["procedure_correctness","supported_answer_correctness","evidence_faithfulness","correct_abstention_rate","conflict_safe_handling_rate","social_usable_rate"]


def load(path:str|Path)->dict[str,Any]: return json.loads(Path(path).read_text(encoding='utf-8'))
def num(d:dict[str,Any],k:str)->float|None:
    v=d.get(k)
    try:return None if v is None else float(v)
    except (TypeError,ValueError):return None

def real(summary:dict[str,Any])->dict[str,Any]:
    c=summary.get('conditions')
    if isinstance(c,dict) and isinstance(c.get('real_rag'),dict): return c['real_rag']
    return summary

def rel_loss(before:float|None,after:float|None)->float:
    if before is None or after is None: return 1.0
    if before<=0: return max(0.0, (before or 0.0)-(after or 0.0))
    return max(0.0,1.0-after/before)

def positive(before:float|None,after:float|None)->float:
    if before is None or after is None:return 0.0
    return max(0.0,after-before)

def candidate_packet(entry:dict[str,Any],base:dict[str,Any],cfg:dict[str,Any])->dict[str,Any]:
    md=load(entry['multidomain_summary']); v212=load(entry['v212_summary']); orig=load(entry['original_rag_summary']); proc=load(entry['procedural_summary'])
    bmd=load(base['multidomain_summary']); bv212=load(base['v212_summary']); borig=load(base['original_rag_summary']); bproc=load(base['procedural_summary'])
    o=real(orig); bo=real(borig); p=real(proc); bp=real(bproc)
    invalid=[]
    for label,s in [('multidomain',md),('v212',v212),('original',orig),('procedural',proc)]:
        if not bool(s.get('evaluation_valid',True)): invalid.append(f'{label}_invalid')
    for key in ('overall_correctness','macro_domain_correctness','hard_correctness','abstention_correctness','short_usable_rate'):
        if num(md,key) is None: invalid.append(f'missing_multidomain_metric:{key}')
    for key in ('overall_correctness','macro_family_correctness','hard_correctness'):
        if num(v212,key) is None: invalid.append(f'missing_v212_metric:{key}')
    if num(p,'procedure_correctness') is None or num(bp,'procedure_correctness') is None:
        invalid.append('missing_procedural_challenge_metric:procedure_correctness')
    for key in PROTECTED:
        if num(o,key) is None or num(bo,key) is None: invalid.append(f'missing_protected_metric:{key}')
    md_keys=['overall_correctness','macro_domain_correctness','hard_correctness']
    md_gain=sum(positive(num(bmd,k),num(md,k)) for k in md_keys)/len(md_keys)
    cont_gain=sum(positive(num(bv212,k),num(v212,k)) for k in ['overall_correctness','macro_family_correctness','hard_correctness'])/3.0
    proc_gain=positive(num(bp,'procedure_correctness'),num(p,'procedure_correctness'))
    protected={k:{'baseline':num(bo,k),'candidate':num(o,k),'relative_loss':rel_loss(num(bo,k),num(o,k)),'delta':(num(o,k)-num(bo,k)) if num(o,k) is not None and num(bo,k) is not None else None} for k in PROTECTED}
    md_abst_loss=rel_loss(num(bmd,'abstention_correctness'),num(md,'abstention_correctness'))
    short_loss=rel_loss(num(bmd,'short_usable_rate'),num(md,'short_usable_rate'))
    policies={}
    for name,pc in cfg['selection']['policies'].items():
        benefit=float(pc.get('multidomain_gain_weight',1))*md_gain + float(pc.get('v212_continuity_gain_weight',0))*cont_gain + float(pc.get('procedural_challenge_gain_weight',0))*proc_gain
        preservation=float(pc.get('original_preservation_weight',1))*sum(v['relative_loss'] for v in protected.values()) + float(pc.get('reasoning_abstention_weight',0))*md_abst_loss + float(pc.get('shortness_weight',0))*short_loss
        policies[name]={'benefit':benefit,'preservation_cost':preservation,'opportunity_margin':benefit-preservation}
    return {'name':entry['name'],'track':entry.get('track'),'model':entry.get('model'),'adapter':entry.get('adapter'),'technical_invalidity':sorted(set(invalid)),'multidomain':{'overall':num(md,'overall_correctness'),'macro_domain':num(md,'macro_domain_correctness'),'hard':num(md,'hard_correctness'),'abstention':num(md,'abstention_correctness'),'short':num(md,'short_usable_rate'),'gain':md_gain},'v212':{'overall':num(v212,'overall_correctness'),'hard':num(v212,'hard_correctness'),'gain':cont_gain},'original':{k:num(o,k) for k in PROTECTED},'original_preservation':protected,'procedural_challenge':{'baseline':num(bp,'procedure_correctness'),'candidate':num(p,'procedure_correctness'),'gain':proc_gain},'policies':policies}

def dominates(a:dict[str,Any],b:dict[str,Any])->bool:
    keys=[('multidomain','overall'),('multidomain','hard'),('original','supported_answer_correctness'),('original','procedure_correctness'),('original','evidence_faithfulness'),('original','correct_abstention_rate'),('procedural_challenge','candidate')]
    av=[];bv=[]
    for x,y in keys:
        va=a[x].get(y); vb=b[x].get(y)
        if va is None or vb is None:return False
        av.append(float(va)); bv.append(float(vb))
    return all(x>=y for x,y in zip(av,bv)) and any(x>y for x,y in zip(av,bv))

def select(cfg:dict[str,Any],manifest:dict[str,Any])->dict[str,Any]:
    entries=manifest['candidates']; base=next((x for x in entries if x['name']=='frozen_anchor'),None)
    if base is None: raise ValueError('candidates manifest must include frozen_anchor')
    packets=[candidate_packet(e,base,cfg) for e in entries]
    official=cfg['selection']['official_policy']
    base_packet=next(p for p in packets if p['name']=='frozen_anchor')
    if base_packet['technical_invalidity']:
        return {'schema_version':'2.2','status':'INVALID_BASELINE_EVIDENCE','official_policy':official,'selected_candidate':'frozen_anchor','selected_model':base_packet.get('model'),'selected_adapter':base_packet.get('adapter'),'policy_selections':{name:'frozen_anchor' for name in cfg['selection']['policies']},'policy_sensitivity_disagreement':False,'pareto_frontier':[],'candidates':packets,'null_metric_policy':'Frozen-anchor preservation evidence is incomplete; official capability-composition selection is blocked.'}
    valid=[p for p in packets if not p['technical_invalidity']]
    frontier=[p['name'] for p in valid if not any(dominates(other,p) for other in valid if other is not p)]
    # Frozen anchor is always eligible at margin 0; trained/composed candidates need positive margin.
    eligible=[p for p in valid if p['name']=='frozen_anchor' or p['policies'][official]['opportunity_margin']>0]
    winner=max(eligible,key=lambda p:(p['policies'][official]['opportunity_margin'],p['multidomain']['gain'])) if eligible else next(p for p in packets if p['name']=='frozen_anchor')
    selections={}
    for pol in cfg['selection']['policies']:
        e=[p for p in valid if p['name']=='frozen_anchor' or p['policies'][pol]['opportunity_margin']>0]
        selections[pol]=max(e,key=lambda p:(p['policies'][pol]['opportunity_margin'],p['multidomain']['gain']))['name'] if e else 'frozen_anchor'
    disagree=len(set(selections.values()))>1
    status='CAPABILITY_COMPOSITION_CANDIDATE_SELECTED' if winner['name']!='frozen_anchor' else 'FROZEN_ANCHOR_RETAINED'
    if disagree: status='POLICY_SENSITIVITY_REVIEW_REQUIRED'
    return {'schema_version':'2.2','status':status,'official_policy':official,'selected_candidate':winner['name'],'selected_model':winner.get('model'),'selected_adapter':winner.get('adapter'),'policy_selections':selections,'policy_sensitivity_disagreement':disagree,'pareto_frontier':frontier,'candidates':packets,'null_metric_policy':'Missing/null protected Real-RAG metrics invalidate official selection; they never become zero preservation cost.'}

def markdown(out:dict[str,Any])->str:
    lines=['# v2.2 Capability Composition DEV Selection','',f"**Status:** `{out['status']}`",f"**Selected:** `{out['selected_candidate']}`",f"**Official policy:** `{out['official_policy']}`",'', '| Candidate | Track | Multi-domain | Hard | Orig proc | Supported | Faithful | Abstain | Proc challenge | Margin | Invalid |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in out['candidates']:
        m=c['multidomain'];o=c['original'];p=c['procedural_challenge'];pol=c['policies'].get(out['official_policy'],{})
        lines.append(f"| {c['name']} | {c.get('track')} | {m['overall']} | {m['hard']} | {o['procedure_correctness']} | {o['supported_answer_correctness']} | {o['evidence_faithfulness']} | {o['correct_abstention_rate']} | {p['candidate']} | {pol.get('opportunity_margin')} | {', '.join(c['technical_invalidity'])} |")
    lines += ['',f"Pareto frontier: `{', '.join(out['pareto_frontier'])}`",f"Policy selections: `{out['policy_selections']}`",'', 'Protected metrics that are missing are treated as invalid evidence, not as zero regression.']
    return '\n'.join(lines)+'\n'

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--candidates-manifest',type=Path,required=True); p.add_argument('--output-json',type=Path,required=True); p.add_argument('--output-md',type=Path,required=True); a=p.parse_args()
    out=select(load(a.config),load(a.candidates_manifest)); a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8'); a.output_md.write_text(markdown(out),encoding='utf-8'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
