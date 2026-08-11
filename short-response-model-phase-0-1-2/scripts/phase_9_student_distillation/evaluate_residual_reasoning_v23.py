from __future__ import annotations

"""Evaluate v2.3 direct residual reasoning with explicit answerability metrics."""
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

ABSTAIN="I don't know based on the available information."

def read_jsonl(p:Path)->list[dict[str,Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def append_jsonl(p:Path,row:dict[str,Any])->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8",newline="\n") as f:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n");f.flush()
def norm(t:str)->str:
    t=(t or "").strip().casefold();t=re.sub(r'^[\s"\'`]+|[\s"\'`]+$','',t);t=re.sub(r'[.!?]+$','',t);return re.sub(r'\s+',' ',t)
def is_abstain(t:str)->bool:
    n=norm(t);return n==norm(ABSTAIN) or n.startswith("i don't know") or n.startswith("i do not know")
def correct(ans:str,row:dict[str,Any])->bool:
    gold=[str(row.get("approved_answer") or row.get("answer") or "")]+[str(x) for x in (row.get("answer_aliases") or [])];a=norm(ans);return any(a==norm(g) for g in gold if norm(g))
def chat(tok:Any,msgs:list[dict[str,Any]])->str:
    kw={"tokenize":False,"add_generation_prompt":True}
    try:return tok.apply_chat_template(msgs,enable_thinking=False,**kw)
    except TypeError:return tok.apply_chat_template(msgs,**kw)
def build(model_id:str,adapter:Path|None)->tuple[Any,Any]:
    tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True,torch_dtype="auto",device_map="auto",low_cpu_mem_usage=True)
    if adapter:
        if PeftModel is None:raise RuntimeError("peft is required for adapter evaluation")
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model.eval();return model,tok
def generate(model:Any,tok:Any,msgs:list[dict[str,Any]],max_seq:int,max_new:int)->tuple[str,int,float]:
    prompt=chat(tok,msgs);enc=tok(prompt,return_tensors="pt",add_special_tokens=False)
    if int(enc["input_ids"].shape[1])>max_seq:raise RuntimeError(f"prompt {enc['input_ids'].shape[1]} > {max_seq}")
    dev=next(model.parameters()).device;enc={k:v.to(dev) for k,v in enc.items()};start=time.perf_counter()
    with torch.inference_mode():out=model.generate(**enc,max_new_tokens=max_new,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    sec=time.perf_counter()-start;gen=out[0,enc["input_ids"].shape[1]:];return tok.decode(gen,skip_special_tokens=True).strip(),int(gen.numel()),sec
def rate(xs:list[bool])->float|None:return sum(xs)/len(xs) if xs else None

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--adapter",type=Path);ap.add_argument("--benchmark",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--max-sequence-length",type=int,default=768);ap.add_argument("--max-new-tokens",type=int,default=32);ap.add_argument("--limit",type=int);ap.add_argument("--resume",action="store_true");ap.add_argument("--require-cuda",action="store_true");ap.add_argument("--minimum-cases",type=int,default=500);a=ap.parse_args()
    if a.require_cuda and not torch.cuda.is_available():raise RuntimeError("CUDA is required but unavailable")
    rows=read_jsonl(a.benchmark);rows=rows[:a.limit] if a.limit else rows;a.output_dir.mkdir(parents=True,exist_ok=True);rp=a.output_dir/"results.jsonl"
    done={}
    if a.resume and rp.exists():done={str(r.get("id")):r for r in read_jsonl(rp)}
    elif rp.exists():rp.unlink()
    model,tok=build(a.model,a.adapter);results=[]
    for idx,row in enumerate(rows,1):
        rid=str(row.get("id") or idx)
        if rid in done:results.append(done[rid]);continue
        msgs=[dict(m) for m in (row.get("messages") or [])]
        if not msgs or msgs[-1].get("role")!="assistant":raise ValueError(f"{rid}: benchmark target missing")
        ans,toks,sec=generate(model,tok,msgs[:-1],a.max_sequence_length,a.max_new_tokens);abst=is_abstain(ans);decision=str(row.get("decision") or "answer");ok=correct(ans,row)
        rec={"id":rid,"domain":row.get("domain"),"reasoning_family":row.get("reasoning_family"),"surface_format":row.get("surface_format"),"difficulty":row.get("difficulty"),"decision":decision,"answer":ans,"gold":row.get("approved_answer"),"correct":ok,"abstained":abst,"false_abstention":bool(decision=="answer" and abst),"answered_when_answerable":bool(decision=="answer" and not abst),"generated_tokens":toks,"generation_seconds":sec,"short_usable":bool(ans.strip()) and toks<=a.max_new_tokens}
        append_jsonl(rp,rec);results.append(rec)
        if idx%250==0:print(f"evaluated {idx}/{len(rows)}")
    by_domain=defaultdict(list);by_family=defaultdict(list);by_diff=defaultdict(list);by_dec=defaultdict(list);ans_domain=defaultdict(list)
    for r in results:
        val=bool(r["correct"]);by_domain[str(r.get("domain"))].append(val);by_family[str(r.get("reasoning_family"))].append(val);by_diff[str(r.get("difficulty"))].append(val);by_dec[str(r.get("decision"))].append(val)
        if r.get("decision")=="answer":ans_domain[str(r.get("domain"))].append(val)
    domain={k:rate(v) for k,v in sorted(by_domain.items())};family={k:rate(v) for k,v in sorted(by_family.items())};diff={k:rate(v) for k,v in sorted(by_diff.items())};dec={k:rate(v) for k,v in sorted(by_dec.items())};ansdom={k:rate(v) for k,v in sorted(ans_domain.items())}
    answerable=[r for r in results if r.get("decision")=="answer"];unanswerable=[r for r in results if r.get("decision")!="answer"];hard_answerable=[r for r in answerable if r.get("difficulty")=="hard"]
    overall=rate([bool(r["correct"]) for r in results]) or 0.0
    summary={"schema_version":"2.3","benchmark":str(a.benchmark),"adapter":str(a.adapter) if a.adapter else None,"case_count":len(results),"answerable_case_count":len(answerable),"unanswerable_case_count":len(unanswerable),"evaluation_valid":len(results)>=a.minimum_cases and len(domain)>=20,"overall_correctness":overall,"answerable_correctness":rate([bool(r["correct"]) for r in answerable]),"answerable_hard_correctness":rate([bool(r["correct"]) for r in hard_answerable]),"macro_answerable_domain_correctness":sum(v for v in ansdom.values() if v is not None)/len(ansdom) if ansdom else None,"abstention_correctness":rate([bool(r["correct"]) for r in unanswerable]),"false_abstention_rate":rate([bool(r["false_abstention"]) for r in answerable]),"answer_coverage":rate([bool(r["answered_when_answerable"]) for r in answerable]),"short_usable_rate":rate([bool(r["short_usable"]) for r in results]),"hard_correctness":diff.get("hard"),"macro_domain_correctness":sum(v for v in domain.values() if v is not None)/len(domain) if domain else None,"macro_family_correctness":sum(v for v in family.values() if v is not None)/len(family) if family else None,"mean_generated_tokens":statistics.mean([int(r["generated_tokens"]) for r in results]) if results else None,"domain_correctness":domain,"answerable_domain_correctness":ansdom,"family_correctness":family,"difficulty_correctness":diff,"decision_correctness":dec}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=["# v2.3 Residual Reasoning Evaluation","",f"- Cases: **{len(results)}**",f"- Overall: **{overall:.4f}**",f"- Answerable: **{float(summary['answerable_correctness'] or 0):.4f}**",f"- Answerable hard: **{float(summary['answerable_hard_correctness'] or 0):.4f}**",f"- Macro answerable domain: **{float(summary['macro_answerable_domain_correctness'] or 0):.4f}**",f"- Abstention correctness: **{float(summary['abstention_correctness'] or 0):.4f}**",f"- False abstention on answerable cases: **{float(summary['false_abstention_rate'] or 0):.4f}**",f"- Answer coverage: **{float(summary['answer_coverage'] or 0):.4f}**"]
    (a.output_dir/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__":main()
