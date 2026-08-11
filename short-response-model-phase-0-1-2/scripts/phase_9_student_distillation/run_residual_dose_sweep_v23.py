from __future__ import annotations

"""Automated coarse+refined capability-dose sweep for one v2.3 residual LoRA."""
import argparse, json, math, shutil, subprocess, sys
from pathlib import Path
from typing import Any


def run(args:list[str])->None:print('+',' '.join(args),flush=True);subprocess.run(args,check=True)
def load(p:Path)->dict[str,Any]:return json.loads(p.read_text(encoding='utf-8'))
def tag(w:float)->str:return f'{w:.2f}'.replace('.','p')
def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--label',required=True);ap.add_argument('--base-model',required=True);ap.add_argument('--adapter',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True)
    for n in ('original-benchmark','original-db','original-calibration','historical-procedural-benchmark','historical-procedural-db','historical-procedural-calibration','v23-rag-benchmark','v23-rag-ood-benchmark','v23-rag-db','v23-rag-calibration','v23-direct-benchmark','v23-direct-ood-benchmark'):
        ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--maximum-evidence-tokens',type=int,default=384);ap.add_argument('--require-cuda',action='store_true');ap.add_argument('--reuse-existing',action='store_true');ap.add_argument('--retain-models',action='store_true');a=ap.parse_args();cfg=load(a.config);a.output_root.mkdir(parents=True,exist_ok=True);py=sys.executable
    def eval_candidate(name:str,model:str,weight:float)->dict[str,Any]:
        root=a.output_root/name;ev=root/'eval';manifest=ev/'candidate_evaluation_manifest.json'
        if not (a.reuse_existing and manifest.exists()):
            args=[py,'scripts/phase_9_student_distillation/evaluate_candidate_v23.py','--model',model,'--output-root',str(ev),'--original-benchmark',str(a.original_benchmark),'--original-db',str(a.original_db),'--original-calibration',str(a.original_calibration),'--historical-procedural-benchmark',str(a.historical_procedural_benchmark),'--historical-procedural-db',str(a.historical_procedural_db),'--historical-procedural-calibration',str(a.historical_procedural_calibration),'--v23-rag-benchmark',str(a.v23_rag_benchmark),'--v23-rag-ood-benchmark',str(a.v23_rag_ood_benchmark),'--v23-rag-db',str(a.v23_rag_db),'--v23-rag-calibration',str(a.v23_rag_calibration),'--v23-direct-benchmark',str(a.v23_direct_benchmark),'--v23-direct-ood-benchmark',str(a.v23_direct_ood_benchmark),'--maximum-evidence-tokens',str(a.maximum_evidence_tokens),'--smoke','--resume']
            if a.require_cuda:args.append('--require-cuda')
            run(args)
        return {'name':name,'track':f'dose_{a.label}','model':model,'adapter':None,'weight':weight,'evaluation_manifest':str(manifest)}
    entries=[eval_candidate('greedy_anchor',a.base_model,0.0)]
    evaluated={0.0:entries[0]}
    def evaluate_weight(w:float)->dict[str,Any]:
        w=round(float(w),6)
        if w in evaluated:return evaluated[w]
        root=a.output_root/f'weight_{tag(w)}';model=root/'model';spec=root/'composition.json';manifest=model/'v23_composition_manifest.json';root.mkdir(parents=True,exist_ok=True)
        if not (a.reuse_existing and manifest.exists()):
            spec.write_text(json.dumps({'schema_version':'2.3','ingredients':[{'name':a.label,'adapter':str(a.adapter),'weight':w}]},indent=2)+'\n',encoding='utf-8');run([py,'scripts/phase_9_student_distillation/compose_residual_adapters_v23.py','--base-model',a.base_model,'--spec',str(spec),'--output-dir',str(model),'--force'])
        e=eval_candidate(f'{a.label}_dose_{tag(w)}',str(model),w);evaluated[w]=e;entries.append(e);return e
    coarse=[float(x) for x in cfg['dose_calibration']['coarse_weights']]
    for w in coarse:evaluate_weight(w)
    def select_now(suffix:str)->dict[str,Any]:
        man=a.output_root/f'candidates_{suffix}.json';man.write_text(json.dumps({'schema_version':'2.3','candidates':entries},indent=2)+'\n',encoding='utf-8');oj=a.output_root/f'dose_selection_{suffix}.json';om=a.output_root/f'DOSE_SELECTION_{suffix.upper()}.md';run([py,'scripts/phase_9_student_distillation/select_residual_dose_v23.py','--config',str(a.config),'--candidates-manifest',str(man),'--output-json',str(oj),'--output-md',str(om)]);return load(oj)
    first=select_now('coarse');best=float(first.get('selected_weight') or 0.0)
    if best>0:
        rad=float(cfg['dose_calibration']['refinement_radius']);step=float(cfg['dose_calibration']['refinement_step']);lo=max(step,best-rad);hi=min(1.0,best+rad);n=int(math.floor((hi-lo)/step+1e-9))+1
        for i in range(n+1):
            w=round(lo+i*step,6)
            if w<=1.0+1e-9:evaluate_weight(min(1.0,w))
    final=select_now('final');winner=str(final['selected_candidate']);selected=next(e for e in entries if e['name']==winner);result={'schema_version':'2.3','label':a.label,'selected_candidate':winner,'selected_weight':float(selected.get('weight') or 0.0),'selected_model':selected['model'],'source_adapter':str(a.adapter),'coarse_selection':first,'final_selection':final};(a.output_root/'DOSE_SWEEP_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8');(a.output_root/'DOSE_SWEEP_RESULT.md').write_text(f"# v2.3 {a.label} Capability-Dose Sweep\n\nSelected weight: **{result['selected_weight']}**\n\nSelected model: `{result['selected_model']}`\n\nZero dose means the frozen greedy anchor remained better than every tested residual dose.\n",encoding='utf-8')
    if not a.retain_models and not bool(cfg['dose_calibration'].get('retain_nonwinning_models',False)):
        selected_path=Path(str(selected['model'])).resolve()
        for w,e in evaluated.items():
            if w==0:continue
            p=Path(str(e['model']))
            if p.exists() and p.resolve()!=selected_path:shutil.rmtree(p,ignore_errors=True)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
