from __future__ import annotations

"""Exact static weighted composition of LoRA deltas for v2.2.

Each LoRA was trained from the same frozen anchor.  Scaling one lora_B tensor by
w scales that adapter's delta by w.  Sequential merge_and_unload therefore
materializes:

    W = W_anchor + sum_i w_i * Delta_i

without requiring all adapters to remain resident at inference time.
"""

import argparse, sys, json, shutil, tempfile, uuid
from pathlib import Path
from typing import Any

import torch
try:
    from safetensors.torch import load_file as safe_load, save_file as safe_save
except ImportError:  # pragma: no cover
    safe_load=safe_save=None

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import sha256_tree, write_json


def load_spec(path:Path)->dict[str,Any]:
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data.get('ingredients'),list) or not data['ingredients']:
        raise ValueError('composition spec requires non-empty ingredients')
    return data


def scale_adapter(src:Path,dst:Path,weight:float)->dict[str,Any]:
    shutil.copytree(src,dst)
    sf=dst/'adapter_model.safetensors'; bf=dst/'adapter_model.bin'
    scaled=0
    if sf.exists():
        if safe_load is None: raise RuntimeError('safetensors is required for .safetensors adapters')
        tensors=safe_load(str(sf),device='cpu')
        out={}
        for key,val in tensors.items():
            if 'lora_B' in key:
                val=val*float(weight); scaled+=1
            out[key]=val
        safe_save(out,str(sf))
    elif bf.exists():
        tensors=torch.load(bf,map_location='cpu')
        if not isinstance(tensors,dict): raise RuntimeError(f'Unexpected adapter state format: {bf}')
        for key in list(tensors):
            if 'lora_B' in key and torch.is_tensor(tensors[key]):
                tensors[key]=tensors[key]*float(weight); scaled+=1
        torch.save(tensors,bf)
    else:
        raise FileNotFoundError(f'No adapter weight file under {src}')
    if scaled<=0: raise RuntimeError(f'No lora_B tensors found under {src}; this composer only supports LoRA-family adapters')
    return {'source':str(src),'weight':float(weight),'scaled_lora_B_tensors':scaled,'source_tree_sha256':sha256_tree(src)}


def main()->None:
    # Keep heavyweight training/runtime dependencies lazy so dataset and
    # composition-tensor unit tests can run in lightweight CI environments.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    p=argparse.ArgumentParser(); p.add_argument('--base-model',required=True); p.add_argument('--spec',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--force',action='store_true'); a=p.parse_args()
    spec=load_spec(a.spec)
    if a.output_dir.exists():
        if not a.force: raise FileExistsError(f'Output exists: {a.output_dir}')
        shutil.rmtree(a.output_dir)
    a.output_dir.parent.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(a.base_model,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.base_model,trust_remote_code=True,torch_dtype='auto',device_map=None,low_cpu_mem_usage=True)
    records=[]
    with tempfile.TemporaryDirectory(prefix='v22_lora_compose_') as td:
        root=Path(td)
        for idx,item in enumerate(spec['ingredients']):
            src=Path(str(item['adapter'])); weight=float(item.get('weight',1.0))
            if not src.exists(): raise FileNotFoundError(src)
            scaled=root/f'adapter_{idx:02d}'
            rec=scale_adapter(src,scaled,weight); rec['name']=str(item.get('name') or src.name); records.append(rec)
            wrapped=PeftModel.from_pretrained(model,str(scaled),is_trainable=False)
            model=wrapped.merge_and_unload(safe_merge=True)
    staging=a.output_dir.parent/f'.{a.output_dir.name}.tmp-{uuid.uuid4().hex}'
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    model.save_pretrained(staging,safe_serialization=True); tok.save_pretrained(staging)
    manifest={'schema_version':'2.2','kind':'exact_static_weighted_lora_composition','base_model':a.base_model,'base_tree_sha256':sha256_tree(Path(a.base_model)),'spec':str(a.spec),'ingredients':records,'formula':'W = W_anchor + sum(weight_i * Delta_i)','inference_extra_adapter_cost':0}
    write_json(staging/'v22_composition_manifest.json',manifest); staging.replace(a.output_dir)
    print(json.dumps(manifest,indent=2,sort_keys=True)); print(f'Composed model: {a.output_dir}')
if __name__=='__main__': main()
