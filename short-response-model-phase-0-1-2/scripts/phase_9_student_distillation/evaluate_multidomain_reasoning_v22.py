from __future__ import annotations

"""Evaluate v2.2 direct multi-domain reasoning with concise deterministic gold."""

import argparse, json, re, statistics, time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
except ImportError:
    PeftModel=None  # type: ignore


def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open('r',encoding='utf-8-sig') as f: return [json.loads(x) for x in f if x.strip()]

def append_jsonl(path:Path,row:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n'); f.flush()

def norm(text:str)->str:
    text=(text or '').strip().casefold(); text=re.sub(r'^[\s"\'`]+|[\s"\'`]+$','',text); text=re.sub(r'[.!?]+$','',text); return re.sub(r'\s+',' ',text)

def correct(answer:str,row:dict[str,Any])->bool:
    gold=[str(row.get('approved_answer') or row.get('answer') or '')]+[str(x) for x in (row.get('answer_aliases') or [])]
    a=norm(answer); return any(a==norm(g) for g in gold if norm(g))

def chat(tok:Any,messages:list[dict[str,Any]])->str:
    kw={'tokenize':False,'add_generation_prompt':True}
    try:return tok.apply_chat_template(messages,enable_thinking=False,**kw)
    except TypeError:return tok.apply_chat_template(messages,**kw)

def build(model_id:str,adapter:Path|None)->tuple[Any,Any]:
    tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True,torch_dtype='auto',device_map='auto',low_cpu_mem_usage=True)
    if adapter is not None:
        if PeftModel is None: raise RuntimeError('peft is required for adapter evaluation')
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model.eval(); return model,tok

def generate(model:Any,tok:Any,messages:list[dict[str,Any]],max_seq:int,max_new:int)->tuple[str,int,float]:
    prompt=chat(tok,messages); enc=tok(prompt,return_tensors='pt',add_special_tokens=False)
    if int(enc['input_ids'].shape[1])>max_seq: raise RuntimeError(f"prompt {enc['input_ids'].shape[1]} > {max_seq}")
    dev=next(model.parameters()).device; enc={k:v.to(dev) for k,v in enc.items()}; start=time.perf_counter()
    with torch.inference_mode(): out=model.generate(**enc,max_new_tokens=max_new,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    elapsed=time.perf_counter()-start; gen=out[0,enc['input_ids'].shape[1]:]; return tok.decode(gen,skip_special_tokens=True).strip(),int(gen.numel()),elapsed

def rate(values:list[bool])->float|None: return sum(values)/len(values) if values else None

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--adapter',type=Path); p.add_argument('--benchmark',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--max-sequence-length',type=int,default=512); p.add_argument('--max-new-tokens',type=int,default=32); p.add_argument('--limit',type=int); p.add_argument('--resume',action='store_true'); p.add_argument('--require-cuda',action='store_true'); p.add_argument('--minimum-cases',type=int,default=500); a=p.parse_args()
    if a.require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA is required but unavailable')
    rows=read_jsonl(a.benchmark); rows=rows[:a.limit] if a.limit else rows; a.output_dir.mkdir(parents=True,exist_ok=True); rp=a.output_dir/'results.jsonl'
    done={}
    if a.resume and rp.exists(): done={str(r.get('id')):r for r in read_jsonl(rp)}
    elif rp.exists(): rp.unlink()
    model,tok=build(a.model,a.adapter); results=[]
    for idx,row in enumerate(rows,1):
        rid=str(row.get('id') or idx)
        if rid in done: results.append(done[rid]); continue
        msgs=[dict(m) for m in (row.get('messages') or [])]
        if not msgs or msgs[-1].get('role')!='assistant': raise ValueError(f'{rid}: benchmark target missing')
        ans,toks,sec=generate(model,tok,msgs[:-1],a.max_sequence_length,a.max_new_tokens)
        pack={'id':rid,'domain':row.get('domain'),'reasoning_family':row.get('reasoning_family'),'difficulty':row.get('difficulty'),'decision':row.get('decision'),'answer':ans,'gold':row.get('approved_answer'),'correct':correct(ans,row),'generated_tokens':toks,'generation_seconds':sec,'short_usable':bool(ans.strip()) and toks<=a.max_new_tokens}
        append_jsonl(rp,pack); results.append(pack)
        if idx%200==0: print(f'evaluated {idx}/{len(rows)}')
    by_domain=defaultdict(list); by_family=defaultdict(list); by_diff=defaultdict(list); by_dec=defaultdict(list)
    for r in results:
        val=bool(r['correct']); by_domain[str(r.get('domain'))].append(val); by_family[str(r.get('reasoning_family'))].append(val); by_diff[str(r.get('difficulty'))].append(val); by_dec[str(r.get('decision'))].append(val)
    domain={k:rate(v) for k,v in sorted(by_domain.items())}; family={k:rate(v) for k,v in sorted(by_family.items())}; diff={k:rate(v) for k,v in sorted(by_diff.items())}; dec={k:rate(v) for k,v in sorted(by_dec.items())}
    overall=rate([bool(r['correct']) for r in results]) or 0.0; macro_domain=sum(v for v in domain.values() if v is not None)/len(domain) if domain else 0.0; macro_family=sum(v for v in family.values() if v is not None)/len(family) if family else 0.0
    summary={'schema_version':'2.2','benchmark':str(a.benchmark),'adapter':str(a.adapter) if a.adapter else None,'case_count':len(results),'evaluation_valid':len(results)>=a.minimum_cases and len(domain)>=20,'overall_correctness':overall,'macro_domain_correctness':macro_domain,'macro_family_correctness':macro_family,'hard_correctness':diff.get('hard'),'moderate_correctness':diff.get('moderate'),'easy_correctness':diff.get('easy'),'abstention_correctness':dec.get('abstain'),'answer_correctness':dec.get('answer'),'short_usable_rate':rate([bool(r['short_usable']) for r in results]),'mean_generated_tokens':statistics.mean([int(r['generated_tokens']) for r in results]) if results else None,'domain_correctness':domain,'family_correctness':family,'difficulty_correctness':diff,'decision_correctness':dec}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=['# v2.2 Multi-Domain Reasoning Evaluation','',f'- Cases: **{len(results)}**',f'- Overall: **{overall:.4f}**',f'- Domain macro: **{macro_domain:.4f}**',f'- Family macro: **{macro_family:.4f}**',f"- Hard: **{float(diff.get('hard') or 0):.4f}**",f"- Abstention: **{float(dec.get('abstain') or 0):.4f}**",'', '## Domain correctness','']+[f'- {k}: `{v}`' for k,v in domain.items()]
    (a.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
