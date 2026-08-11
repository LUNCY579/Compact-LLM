from __future__ import annotations

"""Materialize one or more weighted v2.3 residual LoRA deltas into a dense checkpoint."""
import argparse, json, shutil, tempfile, uuid
from pathlib import Path
from typing import Any

from scripts.common.experiment_utils import sha256_tree, write_json
from scripts.phase_9_student_distillation.compose_lora_adapters_v22 import load_spec, scale_adapter

def main()->None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    p=argparse.ArgumentParser();p.add_argument('--base-model',required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--force',action='store_true');a=p.parse_args();spec=load_spec(a.spec)
    if a.output_dir.exists():
        if not a.force:raise FileExistsError(a.output_dir)
        shutil.rmtree(a.output_dir)
    a.output_dir.parent.mkdir(parents=True,exist_ok=True);tok=AutoTokenizer.from_pretrained(a.base_model,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token;model=AutoModelForCausalLM.from_pretrained(a.base_model,trust_remote_code=True,torch_dtype='auto',device_map=None,low_cpu_mem_usage=True);records=[]
    with tempfile.TemporaryDirectory(prefix='v23_residual_compose_') as td:
        root=Path(td)
        for i,item in enumerate(spec['ingredients']):
            src=Path(str(item['adapter']));w=float(item.get('weight',1.0));scaled=root/f'adapter_{i:02d}';rec=scale_adapter(src,scaled,w);rec['name']=str(item.get('name') or src.name);records.append(rec);model=PeftModel.from_pretrained(model,str(scaled),is_trainable=False).merge_and_unload(safe_merge=True)
    staging=a.output_dir.parent/f'.{a.output_dir.name}.tmp-{uuid.uuid4().hex}';staging.mkdir(parents=True);model.save_pretrained(staging,safe_serialization=True);tok.save_pretrained(staging);manifest={'schema_version':'2.3','kind':'preservation_constrained_residual_composition','base_model':a.base_model,'base_tree_sha256':sha256_tree(Path(a.base_model)),'spec':str(a.spec),'ingredients':records,'formula':'W = W_greedy_anchor + sum(weight_i * Delta_residual_i)','inference_extra_adapter_cost':0};write_json(staging/'v23_composition_manifest.json',manifest);staging.replace(a.output_dir);print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=='__main__':main()
