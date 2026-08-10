from __future__ import annotations

"""Materialize base + optional input LoRA as the immutable v2.2 anchor checkpoint."""

import argparse, sys, json, shutil, uuid
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import sha256_tree, write_json


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--student',required=True); p.add_argument('--adapter',type=Path); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--force',action='store_true'); a=p.parse_args()
    if a.output_dir.exists():
        if not a.force: raise FileExistsError(f'Output exists: {a.output_dir}')
        shutil.rmtree(a.output_dir)
    a.output_dir.parent.mkdir(parents=True,exist_ok=True)
    staging=a.output_dir.parent/f'.{a.output_dir.name}.tmp-{uuid.uuid4().hex}'; staging.mkdir(parents=True)
    tok=AutoTokenizer.from_pretrained(a.student,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.student,trust_remote_code=True,torch_dtype='auto',device_map=None,low_cpu_mem_usage=True)
    if a.adapter is not None:
        if not a.adapter.exists(): raise FileNotFoundError(a.adapter)
        model=PeftModel.from_pretrained(model,str(a.adapter),is_trainable=False)
        model=model.merge_and_unload(safe_merge=True)
    model.save_pretrained(staging,safe_serialization=True); tok.save_pretrained(staging)
    manifest={'schema_version':'2.2','kind':'frozen_materialized_student_anchor','student':a.student,'student_tree_sha256':sha256_tree(Path(a.student)),'input_adapter':str(a.adapter) if a.adapter else None,'input_adapter_tree_sha256':sha256_tree(a.adapter) if a.adapter else None}
    write_json(staging/'v22_anchor_manifest.json',manifest)
    staging.replace(a.output_dir)
    print(json.dumps(manifest,indent=2,sort_keys=True)); print(f'Anchor: {a.output_dir}')
if __name__=='__main__': main()
