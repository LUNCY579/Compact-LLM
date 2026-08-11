from __future__ import annotations

"""CPU smoke eval over specialist-mean dose candidates."""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ABSTAIN = "I don't know based on the available information."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def norm(text: str) -> str:
    text = (text or "").strip().casefold()
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    text = re.sub(r"[.!?]+$", "", text)
    return re.sub(r"\s+", " ", text)


def correct(answer: str, row: dict[str, Any]) -> bool:
    gold = [str(row.get("approved_answer") or row.get("answer") or "")] + [
        str(x) for x in (row.get("answer_aliases") or [])
    ]
    a = norm(answer)
    return any(a == norm(g) for g in gold if norm(g))


def is_abstain(text: str) -> bool:
    n = norm(text)
    return n == norm(ABSTAIN) or n.startswith("i don't know") or n.startswith("i do not know")


def chat(tok: Any, messages: list[dict[str, Any]]) -> str:
    kw = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(messages, **kw)


def load_model(model_dir: Path) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def generate(model: Any, tok: Any, messages: list[dict[str, Any]], max_new: int) -> tuple[str, int, float]:
    prompt = chat(tok, messages)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    gen = out[0, enc["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip(), int(gen.numel()), elapsed


def rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_one(model_dir: Path, benchmark: Path, limit: int, max_new: int) -> dict[str, Any]:
    rows = read_jsonl(benchmark)[:limit]
    model, tok = load_model(model_dir)
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        msgs = [dict(m) for m in (row.get("messages") or [])]
        if not msgs or msgs[-1].get("role") != "assistant":
            raise ValueError(f"row missing assistant target: {row.get('id')}")
        answer, toks, sec = generate(model, tok, msgs[:-1], max_new)
        decision = str(row.get("decision") or "answer")
        results.append(
            {
                "id": str(row.get("id") or idx),
                "decision": decision,
                "correct": correct(answer, row),
                "abstained": is_abstain(answer),
                "answer": answer,
                "gold": row.get("approved_answer"),
                "generated_tokens": toks,
                "generation_seconds": sec,
            }
        )
        if idx % 10 == 0:
            print(f"  {model_dir.name}: {idx}/{len(rows)}", flush=True)
    del model
    answerable = [r for r in results if r["decision"] == "answer"]
    abstain = [r for r in results if r["decision"] != "answer"]
    return {
        "model": str(model_dir),
        "benchmark": str(benchmark),
        "case_count": len(results),
        "overall_correctness": rate([bool(r["correct"]) for r in results]),
        "answerable_correctness": rate([bool(r["correct"]) for r in answerable]),
        "abstention_correctness": rate([bool(r["correct"]) for r in abstain]),
        "false_abstention_rate": rate([bool(r["abstained"]) for r in answerable]),
        "mean_generation_seconds": statistics.mean(r["generation_seconds"] for r in results) if results else None,
        "results": results,
    }


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "dose_sweep.json")
    p.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--max-new-tokens", type=int, default=32)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    official = (ROOT / cfg["official_root"]).resolve()
    v22 = (ROOT / cfg["v22_run_root"]).resolve()
    build = json.loads((args.artifacts / "DOSE_BUILD_SUMMARY.json").read_text(encoding="utf-8"))

    candidates = {
        "greedy_soup": str(v22 / cfg["ingredients"]["greedy_soup"]),
        **build["doses"],
    }
    benchmarks = {
        name: official / rel for name, rel in cfg["smoke_benchmarks"].items() if (official / rel).exists()
    }
    out_root = args.artifacts / "smoke_eval"
    out_root.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, Any] = {"limit": args.limit, "device": "cpu", "benchmarks": {}}

    for bench_name, bench_path in benchmarks.items():
        print(f"\n=== {bench_name} ===", flush=True)
        matrix["benchmarks"][bench_name] = {}
        for cand_name, cand_path in candidates.items():
            path = Path(cand_path)
            print(f"candidate {cand_name}", flush=True)
            packet = evaluate_one(path, bench_path, args.limit, args.max_new_tokens)
            results = packet.pop("results")
            dest = out_root / bench_name / cand_name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "summary.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (dest / "results.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in results) + "\n",
                encoding="utf-8",
            )
            matrix["benchmarks"][bench_name][cand_name] = packet

    (out_root / "SMOKE_COMPARISON.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Specialist-mean dose-sweep smoke comparison",
        "",
        f"- Device: CPU",
        f"- Limit: {args.limit}",
        f"- Merge backend: official Phase 4 `merge_delta_soup.py`",
        "",
        "| Benchmark | Candidate | Overall | Answerable | Abstention |",
        "|---|---|---:|---:|---:|",
    ]
    for bench_name, cands in matrix["benchmarks"].items():
        for cand_name, packet in cands.items():
            lines.append(
                f"| {bench_name} | {cand_name} | {packet.get('overall_correctness')} | "
                f"{packet.get('answerable_correctness')} | {packet.get('abstention_correctness')} |"
            )
    (out_root / "SMOKE_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(matrix, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
