from __future__ import annotations

"""CPU eval for soup candidates (smoke or full official DEV)."""

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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


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


def summarize(results: list[dict[str, Any]], model_dir: Path, benchmark: Path) -> dict[str, Any]:
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
        "mean_generation_seconds": statistics.mean(float(r["generation_seconds"]) for r in results) if results else None,
    }


def evaluate_one(
    model_dir: Path,
    benchmark: Path,
    dest: Path,
    *,
    limit: int | None,
    max_new: int,
    resume: bool,
) -> dict[str, Any]:
    rows = read_jsonl(benchmark)
    if limit is not None:
        rows = rows[:limit]
    dest.mkdir(parents=True, exist_ok=True)
    results_path = dest / "results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if resume and results_path.exists():
        done = {str(r.get("id")): r for r in read_jsonl(results_path)}
    elif results_path.exists():
        results_path.unlink()

    need_model = any(str(row.get("id") or idx) not in done for idx, row in enumerate(rows, 1))
    model = tok = None
    if need_model:
        model, tok = load_model(model_dir)

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        rid = str(row.get("id") or idx)
        if rid in done:
            results.append(done[rid])
            continue
        assert model is not None and tok is not None
        msgs = [dict(m) for m in (row.get("messages") or [])]
        if not msgs or msgs[-1].get("role") != "assistant":
            raise ValueError(f"row missing assistant target: {rid}")
        answer, toks, sec = generate(model, tok, msgs[:-1], max_new)
        decision = str(row.get("decision") or "answer")
        pack = {
            "id": rid,
            "decision": decision,
            "correct": correct(answer, row),
            "abstained": is_abstain(answer),
            "answer": answer,
            "gold": row.get("approved_answer"),
            "domain": row.get("domain"),
            "difficulty": row.get("difficulty"),
            "generated_tokens": toks,
            "generation_seconds": sec,
        }
        append_jsonl(results_path, pack)
        results.append(pack)
        if idx % 50 == 0 or idx == len(rows):
            print(f"  {model_dir.name}: {idx}/{len(rows)} (done={len(results)})", flush=True)

    if model is not None:
        del model
    packet = summarize(results, model_dir, benchmark)
    (dest / "summary.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packet


def write_comparison(out_root: Path, matrix: dict[str, Any], limit: int | None, mode: str) -> None:
    json_name = "FULL_COMPARISON.json" if mode == "full" else "SMOKE_COMPARISON.json"
    md_name = "FULL_COMPARISON.md" if mode == "full" else "SMOKE_COMPARISON.md"
    (out_root / json_name).write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    title = "Soup micro-experiment FULL DEV comparison" if mode == "full" else "Soup micro-experiment smoke comparison"
    lines = [
        f"# {title}",
        "",
        f"- Device: CPU (CUDA hidden)",
        f"- Mode: {mode}",
        f"- Limit: {'ALL' if limit is None else limit}",
        "",
        "| Benchmark | Candidate | Cases | Overall | Answerable | Abstention |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for bench_name, cands in matrix["benchmarks"].items():
        for cand_name, packet in cands.items():
            lines.append(
                f"| {bench_name} | {cand_name} | {packet.get('case_count')} | {packet.get('overall_correctness')} | "
                f"{packet.get('answerable_correctness')} | {packet.get('abstention_correctness')} |"
            )
    (out_root / md_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default_soups.json")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--limit", type=int, default=None, help="Row cap per benchmark. Omit with --full for complete DEV.")
    parser.add_argument("--full", action="store_true", help="Use configs.full_benchmarks and write under full_eval/.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark", action="append", dest="benchmarks")
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    official = (ROOT / cfg["official_root"]).resolve()
    v22 = (ROOT / cfg["v22_run_root"]).resolve()

    mode = "full" if args.full else "smoke"
    bench_map = cfg["full_benchmarks"] if args.full else cfg["smoke_benchmarks"]
    if args.full and args.limit is not None:
        raise SystemExit("Do not pass --limit with --full; full mode uses complete DEV sets.")
    limit = None if args.full else (48 if args.limit is None else args.limit)

    all_candidates = {
        "frozen_anchor": v22 / cfg["ingredients"]["frozen_anchor"],
        "greedy_soup": v22 / cfg["ingredients"]["greedy_soup"],
        "anchor_greedy_uniform": args.artifacts / "soups" / "anchor_greedy_uniform",
        "specialists_lora_uniform": args.artifacts / "soups" / "specialists_lora_uniform",
        "kitchen_sink_full_uniform": args.artifacts / "soups" / "kitchen_sink_full_uniform",
    }
    present = {name: path for name, path in all_candidates.items() if path.exists()}
    if args.candidates:
        missing = [name for name in args.candidates if name not in present]
        if missing:
            raise SystemExit(f"Missing/unknown candidates: {missing}. Present={sorted(present)}")
        present = {name: present[name] for name in args.candidates}
    if len(present) < 1:
        raise SystemExit(f"Need soups built first. Present={list(present)}")

    benchmarks = {name: official / rel for name, rel in bench_map.items() if (official / rel).exists()}
    if args.benchmarks:
        benchmarks = {name: benchmarks[name] for name in args.benchmarks if name in benchmarks}
        if not benchmarks:
            raise SystemExit(f"No matching benchmarks. Available: {sorted(bench_map)}")

    out_root = args.artifacts / ("full_eval" if args.full else "smoke_eval")
    out_root.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, Any] = {
        "mode": mode,
        "limit": limit,
        "device": "cpu",
        "resume": bool(args.resume),
        "candidates": {name: str(path) for name, path in present.items()},
        "benchmarks": {},
    }

    for bench_name, bench_path in benchmarks.items():
        total = len(read_jsonl(bench_path))
        target = total if limit is None else min(limit, total)
        print(f"\n=== benchmark {bench_name} ({bench_path.name}, target_rows={target}) ===", flush=True)
        matrix["benchmarks"][bench_name] = {}
        for cand_name, cand_path in present.items():
            print(f"candidate {cand_name} <- {cand_path}", flush=True)
            dest = out_root / bench_name / cand_name
            packet = evaluate_one(
                cand_path,
                bench_path,
                dest,
                limit=limit,
                max_new=args.max_new_tokens,
                resume=bool(args.resume),
            )
            matrix["benchmarks"][bench_name][cand_name] = packet
            write_comparison(out_root, matrix, limit, mode)

    write_comparison(out_root, matrix, limit, mode)
    print(json.dumps(matrix, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
