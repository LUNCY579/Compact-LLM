from __future__ import annotations

"""v2.3 residual capability matrix, Pareto frontier, and DEV/OOD selection."""
import argparse, json
from pathlib import Path
from typing import Any

PROTECTED=['procedure_correctness','supported_answer_correctness','evidence_faithfulness','correct_abstention_rate','conflict_safe_handling_rate','social_usable_rate']
PRIMARY_DIRECT=['answerable_correctness','answerable_hard_correctness','macro_answerable_domain_correctness','abstention_correctness','false_abstention_rate','short_usable_rate']

def load(p:str|Path)->dict[str,Any]:return json.loads(Path(p).read_text(encoding='utf-8'))
def num(d:dict[str,Any],k:str)->float|None:
    try:
        v=d.get(k);return None if v is None else float(v)
    except (TypeError,ValueError):return None
def real(s:dict[str,Any])->dict[str,Any]:
    c=s.get('conditions');return c.get('real_rag',{}) if isinstance(c,dict) else s
def rel_loss(before:float|None,after:float|None)->float:
    if before is None or after is None:return 1.0
    if before<=0:return max(0.0,(before or 0)-(after or 0))
    return max(0.0,1-after/before)
def positive(before:float|None,after:float|None)->float:
    return 0.0 if before is None or after is None else max(0.0,after-before)
def increase(before:float|None,after:float|None)->float:
    return 1.0 if before is None or after is None else max(0.0,after-before)

def packet(entry:dict[str,Any],base:dict[str,Any],cfg:dict[str,Any])->dict[str,Any]:
    m=load(entry['evaluation_manifest']);bm=load(base['evaluation_manifest'])
    d=load(m['v23_direct_summary']);bd=load(bm['v23_direct_summary']);o=load(m['v23_direct_ood_summary']);bo=load(bm['v23_direct_ood_summary'])
    orig=real(load(m['original_rag_summary']));borig=real(load(bm['original_rag_summary']))
    hp=real(load(m['historical_procedural_summary']));bhp=real(load(bm['historical_procedural_summary']))
    vp=real(load(m['v23_procedural_summary']));bvp=real(load(bm['v23_procedural_summary']))
    vpo=real(load(m['v23_procedural_ood_summary']));bvpo=real(load(bm['v23_procedural_ood_summary']))
    invalid=[]
    for label,s in [('direct',d),('direct_ood',o)]:
        if not bool(s.get('evaluation_valid',True)):invalid.append(f'{label}_invalid')
        for k in PRIMARY_DIRECT:
            if num(s,k) is None:invalid.append(f'missing_{label}:{k}')
    for k in PROTECTED:
        if num(orig,k) is None or num(borig,k) is None:invalid.append(f'missing_protected:{k}')
    for label,s,bs in [('historical_proc',hp,bhp),('v23_proc',vp,bvp),('v23_proc_ood',vpo,bvpo)]:
        if num(s,'procedure_correctness') is None or num(bs,'procedure_correctness') is None:invalid.append(f'missing_{label}:procedure_correctness')
    protected={k:{'baseline':num(borig,k),'candidate':num(orig,k),'relative_loss':rel_loss(num(borig,k),num(orig,k)),'delta':None if num(orig,k) is None or num(borig,k) is None else num(orig,k)-num(borig,k)} for k in PROTECTED}
    dev_gain=sum(positive(num(bd,k),num(d,k)) for k in ('answerable_correctness','answerable_hard_correctness','macro_answerable_domain_correctness'))/3
    ood_gain=sum(positive(num(bo,k),num(o,k)) for k in ('answerable_correctness','answerable_hard_correctness','macro_answerable_domain_correctness'))/3
    proc_gain=(positive(num(bvp,'procedure_correctness'),num(vp,'procedure_correctness'))+positive(num(bvpo,'procedure_correctness'),num(vpo,'procedure_correctness')))/2
    hist_gain=positive(num(bhp,'procedure_correctness'),num(hp,'procedure_correctness'))
    false_abs_cost=(increase(num(bd,'false_abstention_rate'),num(d,'false_abstention_rate'))+increase(num(bo,'false_abstention_rate'),num(o,'false_abstention_rate')))/2
    abst_loss=(rel_loss(num(bd,'abstention_correctness'),num(d,'abstention_correctness'))+rel_loss(num(bo,'abstention_correctness'),num(o,'abstention_correctness')))/2
    short_loss=(rel_loss(num(bd,'short_usable_rate'),num(d,'short_usable_rate'))+rel_loss(num(bo,'short_usable_rate'),num(o,'short_usable_rate')))/2
    protected_loss_sum=sum(x['relative_loss'] for x in protected.values())
    unweighted_preservation={'original_protected_relative_loss_sum':protected_loss_sum,'false_abstention_cost':false_abs_cost,'abstention_correctness_loss':abst_loss,'shortness_loss':short_loss}
    policies={}
    for name,pc in cfg['selection']['policies'].items():
        benefit=float(pc['direct_dev_answerable_gain_weight'])*dev_gain+float(pc['direct_ood_answerable_gain_weight'])*ood_gain+float(pc['procedure_v23_gain_weight'])*proc_gain+float(pc['historical_procedure_challenge_gain_weight'])*hist_gain
        w_orig=float(pc['original_preservation_weight']);w_fa=float(pc['false_abstention_cost_weight']);w_abs=float(pc['new_abstention_loss_weight']);w_short=float(pc['shortness_loss_weight'])
        weighted={'original_preservation':w_orig*protected_loss_sum,'false_abstention':w_fa*false_abs_cost,'abstention_correctness_loss':w_abs*abst_loss,'shortness_loss':w_short*short_loss}
        preservation=sum(weighted.values())
        policies[name]={'benefit':benefit,'preservation_cost':preservation,'opportunity_margin':benefit-preservation,'preservation_cost_decomposition':{'unweighted':unweighted_preservation,'weighted':weighted}}
    return {'name':entry['name'],'track':entry.get('track'),'model':entry.get('model'),'adapter':entry.get('adapter'),'technical_invalidity':sorted(set(invalid)),'direct':{'answerable':num(d,'answerable_correctness'),'hard_answerable':num(d,'answerable_hard_correctness'),'macro_answerable_domain':num(d,'macro_answerable_domain_correctness'),'abstention':num(d,'abstention_correctness'),'false_abstention':num(d,'false_abstention_rate'),'answer_coverage':num(d,'answer_coverage'),'short':num(d,'short_usable_rate'),'gain':dev_gain},'direct_ood':{'answerable':num(o,'answerable_correctness'),'hard_answerable':num(o,'answerable_hard_correctness'),'macro_answerable_domain':num(o,'macro_answerable_domain_correctness'),'abstention':num(o,'abstention_correctness'),'false_abstention':num(o,'false_abstention_rate'),'answer_coverage':num(o,'answer_coverage'),'short':num(o,'short_usable_rate'),'gain':ood_gain},'original':{k:num(orig,k) for k in PROTECTED},'original_preservation':protected,'v23_procedure':{'dev':num(vp,'procedure_correctness'),'ood':num(vpo,'procedure_correctness'),'gain':proc_gain},'historical_procedure':{'baseline':num(bhp,'procedure_correctness'),'candidate':num(hp,'procedure_correctness'),'gain':hist_gain},'policies':policies}

def dominates(a:dict[str,Any],b:dict[str,Any])->bool:
    vals=lambda p:[p['direct']['answerable'],p['direct_ood']['answerable'],p['v23_procedure']['dev'],p['v23_procedure']['ood'],p['original']['supported_answer_correctness'],p['original']['procedure_correctness'],p['original']['evidence_faithfulness'],p['original']['correct_abstention_rate'],p['historical_procedure']['candidate']]
    av,bv=vals(a),vals(b)
    if any(x is None for x in av+bv):return False
    return all(float(x)>=float(y) for x,y in zip(av,bv)) and any(float(x)>float(y) for x,y in zip(av,bv))

def select(cfg:dict[str,Any],manifest:dict[str,Any])->dict[str,Any]:
    entries=manifest['candidates'];base=next((x for x in entries if x['name']=='greedy_anchor'),None)
    if base is None:raise ValueError('candidates manifest requires greedy_anchor')
    ps=[packet(e,base,cfg) for e in entries];official=cfg['selection']['official_policy'];bp=next(p for p in ps if p['name']=='greedy_anchor')
    if bp['technical_invalidity']:
        return {'schema_version':'2.3','status':'INVALID_BASELINE_EVIDENCE','official_policy':official,'selected_candidate':'greedy_anchor','selected_model':bp.get('model'),'selected_adapter':bp.get('adapter'),'policy_selections':{k:'greedy_anchor' for k in cfg['selection']['policies']},'policy_sensitivity_disagreement':False,'pareto_frontier':[],'candidates':ps}
    valid=[p for p in ps if not p['technical_invalidity']];frontier=[p['name'] for p in valid if not any(dominates(q,p) for q in valid if q is not p)]
    sels={}
    for pol in cfg['selection']['policies']:
        elig=[p for p in valid if p['name']=='greedy_anchor' or p['policies'][pol]['opportunity_margin']>0]
        sels[pol]=max(elig,key=lambda p:(p['policies'][pol]['opportunity_margin'],p['direct_ood']['gain'],p['direct']['gain']))['name'] if elig else 'greedy_anchor'
    winner=next(p for p in ps if p['name']==sels[official]);dis=len(set(sels.values()))>1;status='RESIDUAL_CAPABILITY_CANDIDATE_SELECTED' if winner['name']!='greedy_anchor' else 'GREEDY_ANCHOR_RETAINED'
    if dis:status='POLICY_SENSITIVITY_REVIEW_REQUIRED'
    return {'schema_version':'2.3','status':status,'official_policy':official,'selected_candidate':winner['name'],'selected_model':winner.get('model'),'selected_adapter':winner.get('adapter'),'policy_selections':sels,'policy_sensitivity_disagreement':dis,'pareto_frontier':frontier,'candidates':ps,'sealed_policy':'v2.2 sealed data is consumed and excluded from v2.3 selection. Fresh v2.3 sealed data may run only after this DEV/OOD selection is frozen.'}

def markdown(out:dict[str,Any])->str:
    lines=['# v2.3 Preservation-Constrained Residual Capability Selection','',f"**Status:** `{out['status']}`",f"**Selected:** `{out['selected_candidate']}`",f"**Official policy:** `{out['official_policy']}`",'', '| Candidate | Track | Direct ans | OOD ans | V23 proc DEV | V23 proc OOD | Orig proc | Supported | Faithful | Abstain | Hist proc | Margin | Invalid |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in out['candidates']:
        p=c['policies'].get(out['official_policy'],{});lines.append(f"| {c['name']} | {c.get('track')} | {c['direct']['answerable']} | {c['direct_ood']['answerable']} | {c['v23_procedure']['dev']} | {c['v23_procedure']['ood']} | {c['original']['procedure_correctness']} | {c['original']['supported_answer_correctness']} | {c['original']['evidence_faithfulness']} | {c['original']['correct_abstention_rate']} | {c['historical_procedure']['candidate']} | {p.get('opportunity_margin')} | {', '.join(c['technical_invalidity'])} |")
    lines += ['',f"Pareto frontier: `{', '.join(out['pareto_frontier'])}`",f"Policy selections: `{out['policy_selections']}`",'',out.get('sealed_policy','')]
    return '\n'.join(lines)+'\n'
def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--candidates-manifest',type=Path,required=True);ap.add_argument('--output-json',type=Path,required=True);ap.add_argument('--output-md',type=Path,required=True);a=ap.parse_args();out=select(load(a.config),load(a.candidates_manifest));a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8');a.output_md.write_text(markdown(out),encoding='utf-8');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
