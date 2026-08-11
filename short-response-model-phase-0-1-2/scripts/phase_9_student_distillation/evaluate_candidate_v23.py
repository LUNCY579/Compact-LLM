from __future__ import annotations

"""Evaluate one v2.3 candidate on every DEV/OOD axis with frozen retrieval calibrations."""
import argparse, json, subprocess, sys
from pathlib import Path


def run(args:list[str])->None:
    print('+',' '.join(args),flush=True);subprocess.run(args,check=True)

def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--adapter',default='');p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--original-benchmark',type=Path,required=True);p.add_argument('--original-db',type=Path,required=True);p.add_argument('--original-calibration',type=Path,required=True)
    p.add_argument('--historical-procedural-benchmark',type=Path,required=True);p.add_argument('--historical-procedural-db',type=Path,required=True);p.add_argument('--historical-procedural-calibration',type=Path,required=True)
    p.add_argument('--v23-rag-benchmark',type=Path,required=True);p.add_argument('--v23-rag-ood-benchmark',type=Path,required=True);p.add_argument('--v23-rag-db',type=Path,required=True);p.add_argument('--v23-rag-calibration',type=Path,required=True)
    p.add_argument('--v23-direct-benchmark',type=Path,required=True);p.add_argument('--v23-direct-ood-benchmark',type=Path,required=True)
    p.add_argument('--v212-benchmark',type=Path);p.add_argument('--v22-benchmark',type=Path)
    p.add_argument('--maximum-evidence-tokens',type=int,default=384);p.add_argument('--require-cuda',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--research-ready-rag',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--limit-original',type=int);p.add_argument('--limit-historical-procedural',type=int);p.add_argument('--limit-v23-rag',type=int);p.add_argument('--limit-v23-rag-ood',type=int);p.add_argument('--limit-v23-direct',type=int);p.add_argument('--limit-v23-direct-ood',type=int)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True);py=sys.executable;adapter=[] if not a.adapter else ['--adapter',a.adapter]
    rag_axes=[
        ('original_rag',a.original_benchmark,a.original_db,a.original_calibration,a.limit_original,'query_aware_v21'),
        ('historical_procedural_rag',a.historical_procedural_benchmark,a.historical_procedural_db,a.historical_procedural_calibration,a.limit_historical_procedural,'query_aware_v21'),
        ('v23_procedural_rag',a.v23_rag_benchmark,a.v23_rag_db,a.v23_rag_calibration,a.limit_v23_rag,'query_aware_v23'),
        ('v23_procedural_rag_ood',a.v23_rag_ood_benchmark,a.v23_rag_db,a.v23_rag_calibration,a.limit_v23_rag_ood,'query_aware_v23'),
    ]
    for label,bench,db,cal,limit,policy in rag_axes:
        args=[py,'scripts/phase_9_student_distillation/evaluate_rag.py','--model',a.model,*adapter,'--benchmark',str(bench),'--output-dir',str(a.output_root/label),'--knowledge-db',str(db),'--gate-calibration',str(cal),'--conditions','oracle_rag,real_rag','--retrieval-selection-policy',policy,'--maximum-evidence-tokens',str(a.maximum_evidence_tokens),'--max-new-tokens','32','--minimum-cases-per-condition','1','--minimum-supported-cases','1','--minimum-unsupported-cases','1']
        if a.resume:args.append('--resume')
        if a.require_cuda:args.append('--require-cuda')
        if limit is not None:args += ['--limit',str(limit),'--smoke-test']
        elif a.smoke:args.append('--smoke-test')
        elif a.research_ready_rag:args.append('--require-research-ready')
        run(args)
    for label,bench,limit in [('v23_direct',a.v23_direct_benchmark,a.limit_v23_direct),('v23_direct_ood',a.v23_direct_ood_benchmark,a.limit_v23_direct_ood)]:
        args=[py,'scripts/phase_9_student_distillation/evaluate_residual_reasoning_v23.py','--model',a.model,*adapter,'--benchmark',str(bench),'--output-dir',str(a.output_root/label),'--minimum-cases','1' if (a.smoke or limit is not None) else '500','--max-new-tokens','32']
        if limit is not None:args += ['--limit',str(limit)]
        if a.resume:args.append('--resume')
        if a.require_cuda:args.append('--require-cuda')
        run(args)
    optional={}
    if a.v212_benchmark:
        args=[py,'scripts/phase_9_student_distillation/evaluate_general_reasoning_v212.py','--model',a.model,*adapter,'--benchmark',str(a.v212_benchmark),'--output-dir',str(a.output_root/'v212'),'--minimum-cases','1' if a.smoke else '100','--max-new-tokens','32']
        if a.resume:args.append('--resume')
        if a.require_cuda:args.append('--require-cuda')
        run(args);optional['v212_summary']=str(a.output_root/'v212/summary.json')
    if a.v22_benchmark:
        args=[py,'scripts/phase_9_student_distillation/evaluate_multidomain_reasoning_v22.py','--model',a.model,*adapter,'--benchmark',str(a.v22_benchmark),'--output-dir',str(a.output_root/'v22'),'--minimum-cases','1' if a.smoke else '500','--max-new-tokens','32']
        if a.resume:args.append('--resume')
        if a.require_cuda:args.append('--require-cuda')
        run(args);optional['v22_summary']=str(a.output_root/'v22/summary.json')
    manifest={'schema_version':'2.3','model':a.model,'adapter':a.adapter or None,'original_rag_summary':str(a.output_root/'original_rag/summary.json'),'historical_procedural_summary':str(a.output_root/'historical_procedural_rag/summary.json'),'v23_procedural_summary':str(a.output_root/'v23_procedural_rag/summary.json'),'v23_procedural_ood_summary':str(a.output_root/'v23_procedural_rag_ood/summary.json'),'v23_direct_summary':str(a.output_root/'v23_direct/summary.json'),'v23_direct_ood_summary':str(a.output_root/'v23_direct_ood/summary.json'),**optional}
    (a.output_root/'candidate_evaluation_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=='__main__':main()
