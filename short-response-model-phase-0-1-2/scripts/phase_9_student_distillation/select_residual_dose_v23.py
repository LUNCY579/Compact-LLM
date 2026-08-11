from __future__ import annotations

"""Select a capability dose from probe evaluations using the v2.3 policy."""
import argparse, json
from pathlib import Path
from typing import Any
from scripts.phase_9_student_distillation.select_residual_capability_v23 import load, packet

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--candidates-manifest',type=Path,required=True);ap.add_argument('--output-json',type=Path,required=True);ap.add_argument('--output-md',type=Path,required=True);a=ap.parse_args();cfg=load(a.config);man=load(a.candidates_manifest);base=next(x for x in man['candidates'] if x['name']=='greedy_anchor');ps=[packet(e,base,cfg) for e in man['candidates']];pol=cfg['selection']['official_policy'];valid=[p for p in ps if not p['technical_invalidity']]
    eligible=[p for p in valid if p['name']=='greedy_anchor' or p['policies'][pol]['opportunity_margin']>0];winner=max(eligible,key=lambda p:(p['policies'][pol]['opportunity_margin'],p['direct_ood']['gain'],p['direct']['gain']))
    source=next(e for e in man['candidates'] if e['name']==winner['name']);out={'schema_version':'2.3','status':'DOSE_SELECTED' if winner['name']!='greedy_anchor' else 'ZERO_DOSE_RETAINED','selected_candidate':winner['name'],'selected_weight':source.get('weight',0.0),'selected_model':source.get('model'),'official_policy':pol,'candidates':ps}
    a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8');lines=['# v2.3 Capability-Dose Selection','',f"Selected: **{out['selected_candidate']}**",f"Weight: **{out['selected_weight']}**",'', '| Candidate | Weight | Direct answerable | OOD answerable | Orig procedure | Supported | Faithful | Margin |','|---|---:|---:|---:|---:|---:|---:|---:|']
    byname={e['name']:e for e in man['candidates']}
    for p in ps:
        e=byname[p['name']];lines.append(f"| {p['name']} | {e.get('weight',0.0)} | {p['direct']['answerable']} | {p['direct_ood']['answerable']} | {p['original']['procedure_correctness']} | {p['original']['supported_answer_correctness']} | {p['original']['evidence_faithfulness']} | {p['policies'][pol]['opportunity_margin']} |")
    a.output_md.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
