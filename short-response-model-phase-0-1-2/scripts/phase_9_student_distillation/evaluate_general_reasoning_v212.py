from __future__ import annotations

"""Evaluate the v2.1.2 direct general-reasoning benchmark.

This evaluator is intentionally separate from evaluate_rag.py.  v2.1.2 measures
reasoning over information already present in the prompt; it does not use a
retrieval database.  The output remains short-response oriented and is scored
against deterministic gold answers.
"""

import argparse
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out=[]
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")
        f.flush()


def normalize(text: str) -> str:
    text=(text or "").strip().casefold()
    text=re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    text=re.sub(r"[.!?]+$", "", text)
    text=re.sub(r"\s+", " ", text)
    return text


def answer_correct(answer: str, row: dict[str, Any]) -> bool:
    candidates=[str(row.get("approved_answer") or row.get("answer") or "")]
    candidates += [str(x) for x in (row.get("answer_aliases") or [])]
    a=normalize(answer)
    return any(a == normalize(g) for g in candidates if normalize(g))


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    kwargs={"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_model(model_id: str, adapter: Path | None) -> tuple[Any, Any]:
    tok=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True
    )
    if adapter is not None:
        if PeftModel is None:
            raise RuntimeError("peft is required to evaluate an adapter")
        model=PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, tok


def generate(model: Any, tok: Any, messages: list[dict[str, Any]], max_sequence_length: int, max_new_tokens: int) -> tuple[str, int, float]:
    prompt=apply_chat_template(tok, messages)
    encoded=tok(prompt, return_tensors="pt", add_special_tokens=False)
    if int(encoded["input_ids"].shape[1]) > max_sequence_length:
        raise RuntimeError(f"Prompt requires {encoded['input_ids'].shape[1]} tokens > max_sequence_length={max_sequence_length}")
    device=next(model.parameters()).device
    encoded={k:v.to(device) for k,v in encoded.items()}
    started=time.perf_counter()
    with torch.inference_mode():
        out=model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    elapsed=time.perf_counter()-started
    gen=out[0, encoded["input_ids"].shape[1]:]
    text=tok.decode(gen, skip_special_tokens=True).strip()
    return text, int(gen.numel()), elapsed


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--minimum-cases", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args=parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    rows=read_jsonl(args.benchmark)
    if args.limit is not None: rows=rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path=args.output_dir/"results.jsonl"
    done={}
    if args.resume and results_path.exists():
        for r in read_jsonl(results_path): done[str(r.get("id"))]=r
    elif results_path.exists():
        results_path.unlink()

    model,tok=build_model(args.model,args.adapter)
    results=[]
    for idx,row in enumerate(rows,1):
        rid=str(row.get("id") or idx)
        if rid in done:
            results.append(done[rid]); continue
        messages=[dict(m) for m in (row.get("messages") or [])]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"{rid}: benchmark row must end with assistant gold target")
        answer,tokens,elapsed=generate(model,tok,messages[:-1],args.max_sequence_length,args.max_new_tokens)
        correct=answer_correct(answer,row)
        packet={
            "id":rid,
            "reasoning_family":row.get("reasoning_family"),
            "difficulty":row.get("difficulty"),
            "decision":row.get("decision"),
            "answer":answer,
            "gold":row.get("approved_answer"),
            "correct":correct,
            "generated_tokens":tokens,
            "generation_seconds":elapsed,
            "short_usable":bool(answer.strip()) and tokens <= args.max_new_tokens,
        }
        append_jsonl(results_path,packet)
        results.append(packet)
        if idx % 100 == 0:
            print(f"evaluated {idx}/{len(rows)}")

    by_family=defaultdict(list); by_diff=defaultdict(list); by_decision=defaultdict(list)
    for r in results:
        by_family[str(r.get("reasoning_family"))].append(bool(r["correct"]))
        by_diff[str(r.get("difficulty"))].append(bool(r["correct"]))
        by_decision[str(r.get("decision"))].append(bool(r["correct"]))
    family_acc={k:sum(v)/len(v) for k,v in sorted(by_family.items())}
    diff_acc={k:sum(v)/len(v) for k,v in sorted(by_diff.items())}
    decision_acc={k:sum(v)/len(v) for k,v in sorted(by_decision.items())}
    overall=sum(bool(r["correct"]) for r in results)/len(results) if results else 0.0
    macro=sum(family_acc.values())/len(family_acc) if family_acc else 0.0
    summary={
        "schema_version":"2.1.2",
        "benchmark":str(args.benchmark),
        "case_count":len(results),
        "evaluation_valid":len(results) >= args.minimum_cases,
        "overall_correctness":overall,
        "macro_family_correctness":macro,
        "hard_correctness":diff_acc.get("hard"),
        "moderate_correctness":diff_acc.get("moderate"),
        "easy_correctness":diff_acc.get("easy"),
        "abstention_correctness":decision_acc.get("abstain"),
        "answer_correctness":decision_acc.get("answer"),
        "short_usable_rate":sum(bool(r["short_usable"]) for r in results)/len(results) if results else 0.0,
        "mean_generated_tokens":statistics.mean([int(r["generated_tokens"]) for r in results]) if results else None,
        "family_correctness":family_acc,
        "difficulty_correctness":diff_acc,
        "decision_correctness":decision_acc,
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=["# v2.1.2 General Reasoning Evaluation","",f"- Cases: **{len(results)}**",f"- Overall correctness: **{overall:.4f}**",f"- Family macro correctness: **{macro:.4f}**",f"- Hard correctness: **{(diff_acc.get('hard') or 0.0):.4f}**",f"- Abstention correctness: **{(decision_acc.get('abstain') or 0.0):.4f}**",""]
    (args.output_dir/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__ == "__main__": main()
