"""Bind observed branch documents/receipts without importing their native claims."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOTS = {
    'comp_rs': Path(r'C:/Users/clclc/BusinessProject/ChatBubble-comp-rs-amendment-1'),
    'central_mimo': Path(r'C:/Users/clclc/BusinessProject/ChatBubble-post-training-Mimo'),
    'rvs': Path(r'C:/Users/clclc/BusinessProject/ChatBubble-rvs001'),
}
FILES = {
    'comp_rs': ['experiments/maxrl_comp002_pretraining_v1/COMP_RS_001_AMENDMENT_1.md',
                'experiments/maxrl_comp002_pretraining_v1/evidence/S0_Q1_FP32.json',
                'experiments/maxrl_comp002_pretraining_v1/evidence/S0_COMPETENCE.json'],
    'central_mimo': ['experiments/post_training_mimo_centralized_v1/CURRENT_BEST_RECIPE.json',
                     'experiments/post_training_mimo_centralized_v1/METHOD_STATUS_MATRIX.csv',
                     'experiments/post_training_mimo_centralized_v1/receipts/LAPTOP_NATIVE_STAGE_005_RESULTS.json',
                     'experiments/post_training_mimo_centralized_v1/receipts/CAPTURE_GEOMETRY_D1_NATIVE_RESULT.json',
                     'experiments/post_training_mimo_centralized_v1/receipts/NATIVE_BACKEND_QUAL_V1_PREWORKER_ADMISSION.json',
                     'experiments/post_training_mimo_centralized_v1/receipts/train_scope_v1/TRAIN_SCOPE_RECEIPT.json'],
    'rvs': ['experiments/rineya_verified_resampling_v1/runs/pilot-validation-20260929/evidence/NATIVE_STATUS.json'],
}
PAPERS = {
    'MaxRL_v3': 'https://arxiv.org/html/2602.02710v3',
    'GRPO': 'https://arxiv.org/abs/2402.03300',
    'DAPO': 'https://arxiv.org/html/2503.14476',
    'NGU': 'https://arxiv.org/html/2609.13443',
    'Wanda': 'https://arxiv.org/html/2306.11695',
    'F_Wanda': 'https://arxiv.org/html/2608.00481',
    'EWC': 'https://arxiv.org/abs/1612.00796',
    'tangent_arithmetic': 'https://arxiv.org/html/2305.12827',
    'MiMo_MOPD': 'https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD',
}


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise SystemExit('source evidence exists (create-only)')
    branches={}
    for key,root in ROOTS.items():
        records={}
        for relative in FILES[key]:
            path=root/relative
            raw=path.read_bytes()
            records[relative]={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            if path.suffix=='.json':
                value=json.loads(raw)
                records[relative]['status_fields']={name:value[name] for name in
                    ('status','outcome','passed','completed','model_loads','completions','acquisition_effect','retention_effect') if name in value}
        branches[key]={'root':str(root),'head':git(root,'rev-parse','HEAD'),
                       'branch':git(root,'branch','--show-current'),
                       'dirty_paths':git(root,'status','--porcelain','--untracked-files=all').splitlines(),
                       'files':records}
    receipt={'kind':'read-only branch/source evidence snapshot','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'branches':branches,'primary_paper_urls':PAPERS,
             'interpretation':'Mechanics, proposals, qualified execution and efficacy are distinct. No method promotion is inferred from this snapshot.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({key:{'head':value['head'],'files':len(value['files'])} for key,value in branches.items()}))


if __name__=='__main__':
    main()
