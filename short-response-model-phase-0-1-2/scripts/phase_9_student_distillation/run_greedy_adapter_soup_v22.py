from __future__ import annotations

"""Greedy v2.2 adapter soup search with probe-first evaluation.

At each round, every remaining adapter is tentatively added at each configured
weight. The exact static LoRA delta sum is materialized, evaluated on bounded
DEV probes, and retained only when the official v2.2 opportunity margin
improves. The final accepted ingredient set is then re-composed and evaluated
on full DEV. No TEST data is used.
"""

import argparse, json, shutil, subprocess, sys
from pathlib import Path
from typing import Any


def run(cmd:list[str])->None:
    print('+',' '.join(cmd),flush=True); subprocess.run(cmd,check=True)

def parse_adapter(value:str)->tuple[str,str]:
    if '=' not in value: raise argparse.ArgumentTypeError('adapter must be NAME=PATH')
    name,path=value.split('=',1); return name.strip(),path.strip()

def load(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding='utf-8'))

def eval_candidate(py:str, model:str, out:Path, a:argparse.Namespace, probe:bool)->Path:
    if probe:
        probe_root=a.output_root/'probes'
        original_benchmark=probe_root/'original.jsonl'
        procedural_benchmark=probe_root/'procedural.jsonl'
        multidomain_benchmark=probe_root/'multidomain.jsonl'
        v212_benchmark=probe_root/'v212.jsonl'
    else:
        original_benchmark=a.original_benchmark; procedural_benchmark=a.procedural_benchmark; multidomain_benchmark=a.multidomain_benchmark; v212_benchmark=a.v212_benchmark
    cmd=[py,'scripts/phase_9_student_distillation/evaluate_candidate_v22.py','--model',model,'--output-root',str(out),'--original-benchmark',str(original_benchmark),'--original-db',str(a.original_db),'--original-calibration',str(a.original_calibration),'--procedural-benchmark',str(procedural_benchmark),'--procedural-db',str(a.procedural_db),'--procedural-calibration',str(a.procedural_calibration),'--multidomain-benchmark',str(multidomain_benchmark),'--v212-benchmark',str(v212_benchmark),'--maximum-evidence-tokens',str(a.maximum_evidence_tokens),'--resume']
    if a.require_cuda: cmd.append('--require-cuda')
    if not probe: cmd.append('--research-ready-rag')
    run(cmd); return out/'candidate_evaluation_manifest.json'

def candidate_entry(name:str,track:str,model:str,manifest:Path)->dict[str,Any]:
    m=load(manifest); return {'name':name,'track':track,'model':model,'adapter':None,'multidomain_summary':m['multidomain_summary'],'v212_summary':m['v212_summary'],'original_rag_summary':m['original_rag_summary'],'procedural_summary':m['procedural_summary']}

def margin(py:str,cfg:Path,base_entry:dict[str,Any],trial_entry:dict[str,Any],root:Path,tag:str)->float:
    man=root/f'{tag}_candidates.json'; outj=root/f'{tag}_selection.json'; outm=root/f'{tag}_selection.md'; man.write_text(json.dumps({'candidates':[base_entry,trial_entry]},indent=2)+'\n',encoding='utf-8')
    run([py,'scripts/phase_9_student_distillation/select_capability_composition_v22.py','--config',str(cfg),'--candidates-manifest',str(man),'--output-json',str(outj),'--output-md',str(outm)])
    o=load(outj); c=next(x for x in o['candidates'] if x['name']==trial_entry['name']); return float(c['policies'][o['official_policy']]['opportunity_margin']) if not c['technical_invalidity'] else -1e9

def compose(py:str,base:str,ingredients:list[dict[str,Any]],out:Path,spec_path:Path)->None:
    spec_path.parent.mkdir(parents=True,exist_ok=True); spec_path.write_text(json.dumps({'schema_version':'2.2','ingredients':ingredients},indent=2)+'\n',encoding='utf-8')
    run([py,'scripts/phase_9_student_distillation/compose_lora_adapters_v22.py','--base-model',base,'--spec',str(spec_path),'--output-dir',str(out),'--force'])

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--base-model',required=True); p.add_argument('--adapter',action='append',type=parse_adapter,required=True); p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--original-benchmark',type=Path,required=True); p.add_argument('--original-db',type=Path,required=True); p.add_argument('--original-calibration',type=Path,required=True); p.add_argument('--procedural-benchmark',type=Path,required=True); p.add_argument('--procedural-db',type=Path,required=True); p.add_argument('--procedural-calibration',type=Path,required=True); p.add_argument('--multidomain-benchmark',type=Path,required=True); p.add_argument('--v212-benchmark',type=Path,required=True); p.add_argument('--maximum-evidence-tokens',type=int,default=384); p.add_argument('--require-cuda',action='store_true'); a=p.parse_args()
    py=sys.executable; a.output_root.mkdir(parents=True,exist_ok=True); cfg=load(a.config); g=cfg['tracks']['D_composition']['greedy']; weights=[float(x) for x in g['candidate_weights']]; max_rounds=int(g['max_rounds']); min_imp=float(g['minimum_improvement'])
    probe_root=a.output_root/'probes'
    run([py,'scripts/phase_9_student_distillation/build_capability_probe_v22.py','--original',str(a.original_benchmark),'--procedural',str(a.procedural_benchmark),'--multidomain',str(a.multidomain_benchmark),'--v212',str(a.v212_benchmark),'--output-root',str(probe_root),'--original-count',str(g['probe_original_rag_cases']),'--procedural-count',str(g['probe_procedural_cases']),'--multidomain-count',str(g['probe_multidomain_cases']),'--v212-count','180'])
    base_eval=eval_candidate(py,a.base_model,a.output_root/'probe_frozen_anchor',a,True); base_entry=candidate_entry('frozen_anchor','baseline',a.base_model,base_eval)
    remaining={n:p for n,p in a.adapter}; accepted:list[dict[str,Any]]=[]; current_margin=0.0; history=[]
    for rnd in range(1,max_rounds+1):
        best=None
        for name,path in sorted(remaining.items()):
            for weight in weights:
                ingredients=accepted+[{'name':name,'adapter':path,'weight':weight}]; tag=f'r{rnd}_{name}_w{str(weight).replace(".","p")}'; trial_root=a.output_root/'trials'/tag; model=trial_root/'model'; compose(py,a.base_model,ingredients,model,trial_root/'composition.json'); ev=eval_candidate(py,str(model),trial_root/'eval',a,True); entry=candidate_entry(tag,'greedy_probe',str(model),ev); score=margin(py,a.config,base_entry,entry,trial_root,'score'); history.append({'round':rnd,'candidate':name,'weight':weight,'margin':score})
                # The trial checkpoint is only needed long enough to produce DEV probe summaries.
                # Delete it immediately so the greedy search cannot fill the user's disk with
                # many full materialized models; accepted mixtures are recomposed from their
                # ingredient list in later rounds and once more for the final full-DEV readout.
                shutil.rmtree(model, ignore_errors=True)
                if best is None or score>best['margin']: best={'name':name,'path':path,'weight':weight,'margin':score,'ingredients':ingredients,'trial_root':str(trial_root)}
        if best is None or best['margin'] <= current_margin + min_imp: break
        accepted=best['ingredients']; current_margin=float(best['margin']); remaining.pop(best['name'],None)
        if not remaining: break
    final_model=a.output_root/'final_model'
    if accepted: compose(py,a.base_model,accepted,final_model,a.output_root/'final_composition.json')
    else:
        final_model=Path(a.base_model)
    full_eval=eval_candidate(py,str(final_model),a.output_root/'full_dev_eval',a,False)
    result={'schema_version':'2.2','accepted_ingredients':accepted,'probe_margin':current_margin,'history':history,'final_model':str(final_model),'full_dev_evaluation_manifest':str(full_eval)}
    (a.output_root/'GREEDY_SOUP_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# v2.2 Greedy Adapter Soup','',f"Accepted ingredients: `{accepted}`",f"Probe margin: `{current_margin}`",f"Final model: `{final_model}`",'', 'Search used DEV probes only; the final accepted mixture is evaluated on full DEV.']
    (a.output_root/'GREEDY_SOUP_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
