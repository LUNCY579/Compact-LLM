from __future__ import annotations

"""Run every v2.2 DEV/TEST evaluation axis for one candidate.

This wrapper exists so the weighted, greedy, and layer-separated composition
tracks use exactly the same evaluation contract as the single-adapter tracks.
"""

import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any


def run(args:list[str])->None:
    print('+', ' '.join(args), flush=True)
    subprocess.run(args, check=True)


def maybe_flag(args:list[str], flag:str, enabled:bool)->None:
    if enabled: args.append(flag)


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True); p.add_argument('--adapter',default=''); p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--original-benchmark',type=Path,required=True); p.add_argument('--original-db',type=Path,required=True); p.add_argument('--original-calibration',type=Path,required=True)
    p.add_argument('--procedural-benchmark',type=Path,required=True); p.add_argument('--procedural-db',type=Path,required=True); p.add_argument('--procedural-calibration',type=Path,required=True)
    p.add_argument('--multidomain-benchmark',type=Path,required=True); p.add_argument('--v212-benchmark',type=Path,required=True)
    p.add_argument('--maximum-evidence-tokens',type=int,default=384); p.add_argument('--require-cuda',action='store_true'); p.add_argument('--research-ready-rag',action='store_true'); p.add_argument('--resume',action='store_true')
    p.add_argument('--limit-original',type=int); p.add_argument('--limit-procedural',type=int); p.add_argument('--limit-multidomain',type=int); p.add_argument('--limit-v212',type=int)
    a=p.parse_args(); py=sys.executable; a.output_root.mkdir(parents=True,exist_ok=True)
    common_adapter=[] if not a.adapter else ['--adapter',a.adapter]
    for label,bench,db,cal,limit in [
        ('original_rag',a.original_benchmark,a.original_db,a.original_calibration,a.limit_original),
        ('procedural_rag',a.procedural_benchmark,a.procedural_db,a.procedural_calibration,a.limit_procedural),
    ]:
        args=[py,'scripts/phase_9_student_distillation/evaluate_rag.py','--model',a.model,*common_adapter,'--benchmark',str(bench),'--output-dir',str(a.output_root/label),'--knowledge-db',str(db),'--gate-calibration',str(cal),'--conditions','oracle_rag,real_rag','--retrieval-selection-policy','query_aware_v21','--maximum-evidence-tokens',str(a.maximum_evidence_tokens),'--max-new-tokens','32','--minimum-cases-per-condition','1','--minimum-supported-cases','1','--minimum-unsupported-cases','1']
        if a.resume: args.append('--resume')
        if a.require_cuda: args.append('--require-cuda')
        if a.research_ready_rag and limit is None: args.append('--require-research-ready')
        elif limit is not None: args += ['--limit',str(limit),'--smoke-test']
        run(args)
    for script,label,bench,limit,mincases in [
        ('evaluate_multidomain_reasoning_v22.py','multidomain',a.multidomain_benchmark,a.limit_multidomain,500),
        ('evaluate_general_reasoning_v212.py','v212',a.v212_benchmark,a.limit_v212,100),
    ]:
        args=[py,f'scripts/phase_9_student_distillation/{script}','--model',a.model,*common_adapter,'--benchmark',str(bench),'--output-dir',str(a.output_root/label),'--max-new-tokens','32']
        if a.resume: args.append('--resume')
        if a.require_cuda: args.append('--require-cuda')
        if limit is not None: args += ['--limit',str(limit),'--minimum-cases','1']
        else: args += ['--minimum-cases',str(mincases)]
        run(args)
    manifest={
        'schema_version':'2.2','model':a.model,'adapter':a.adapter or None,
        'original_rag_summary':str(a.output_root/'original_rag/summary.json'),
        'procedural_summary':str(a.output_root/'procedural_rag/summary.json'),
        'multidomain_summary':str(a.output_root/'multidomain/summary.json'),
        'v212_summary':str(a.output_root/'v212/summary.json'),
    }
    (a.output_root/'candidate_evaluation_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=='__main__': main()
