from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from scripts.common.experiment_utils import read_jsonl, write_json, environment
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model-id',default='Qwen/Qwen3-0.6B'); p.add_argument('--sft-adapter',type=Path,required=True); p.add_argument('--train-file',type=Path,default=ROOT/'data/preferences/train.jsonl'); p.add_argument('--dev-file',type=Path,default=ROOT/'data/preferences/dev.jsonl'); p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/phase_3_dpo'); p.add_argument('--max-steps',type=int,default=200); p.add_argument('--seed',type=int,default=11); a=p.parse_args()
 tok=AutoTokenizer.from_pretrained(a.model_id); tok.pad_token=tok.pad_token or tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(a.model_id,torch_dtype='auto',device_map='auto')
 model=PeftModel.from_pretrained(base,str(a.sft_adapter),is_trainable=True)
 def ds(path): return Dataset.from_list([{k:r[k] for k in ('prompt','chosen','rejected')} for r in read_jsonl(path)])
 cfg=DPOConfig(output_dir=str(a.output_dir/'checkpoints'),max_steps=a.max_steps,per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=5e-6,logging_steps=5,save_steps=100,eval_steps=100,eval_strategy='steps',seed=a.seed,report_to=[])
 trainer=DPOTrainer(model=model,args=cfg,processing_class=tok,train_dataset=ds(a.train_file),eval_dataset=ds(a.dev_file))
 metrics=trainer.train().metrics; out=a.output_dir/'adapter'; out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); tok.save_pretrained(out)
 write_json(a.output_dir/'training_manifest.json',{'model_id':a.model_id,'sft_adapter':str(a.sft_adapter),'seed':a.seed,'metrics':metrics,'environment':environment()})
 print('Saved',out)
if __name__=='__main__': main()
