from __future__ import annotations

"""Automate the 8-mode toxic-exclusion × procedure-toggle soup matrix.

Modes = 4 exclusion policies × 2 procedure toggles.
Each mode: materialize included adapters, optionally add layer_separated,
uniform-average dense checkpoints on CPU, smoke/full-eval, update PROGRESS_REPORT.md.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ABSTAIN = "I don't know based on the available information."
# One densified adapter is ~1.3GB; soup write needs a full staging copy too.
MIN_FREE_BYTES_FOR_SOUP = int(3.5 * 1024**3)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or ROOT))


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return int(usage.free)


def ensure_disk_space(path: Path, *, need: int = MIN_FREE_BYTES_FOR_SOUP, context: str) -> None:
    free = free_bytes(path)
    if free < need:
        raise RuntimeError(
            f"{context}: only {free / 1024**3:.2f} GiB free under {path.drive or path}; "
            f"need at least {need / 1024**3:.2f} GiB. Shared materialize cache lives under "
            f"_materialized/; delete leftover modes/*/materialized duplicates if present."
        )


def reclaim_per_mode_materialized(modes_root: Path, shared_root: Path) -> dict[str, Any]:
    """Collapse duplicated per-mode densified adapters into one shared cache.

    Older runs wrote a full copy under each mode. That exhausts disk (~8 modes ×
    ~5–7 × 1.3GB). Keep one shared copy per adapter name and delete the rest.
    """
    shared_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    removed = 0
    freed = 0
    if not modes_root.exists():
        return {"moved": moved, "removed": removed, "approx_freed_bytes": freed}
    for mode_dir in sorted(p for p in modes_root.iterdir() if p.is_dir()):
        local = mode_dir / "materialized"
        if not local.exists():
            continue
        for item in sorted(p for p in local.iterdir() if p.is_dir()):
            dest = shared_root / item.name
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            if (dest / "model.safetensors").exists():
                shutil.rmtree(item, ignore_errors=True)
                removed += 1
                freed += size
            else:
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.move(str(item), str(dest))
                moved += 1
        # Remove empty / leftover local materialized trees and failed soup staging.
        shutil.rmtree(local, ignore_errors=True)
        for staging in mode_dir.glob(".soup.tmp-*"):
            size = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
            shutil.rmtree(staging, ignore_errors=True)
            freed += size
            removed += 1
    return {"moved": moved, "removed": removed, "approx_freed_bytes": freed}

def enumerate_modes(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    modes: list[dict[str, Any]] = []
    for ex in cfg["exclusion_modes"]:
        for proc in cfg["procedure_modes"]:
            mode_id = f"{ex['id']}__{proc['id']}"
            modes.append(
                {
                    "mode_id": mode_id,
                    "exclusion_id": ex["id"],
                    "exclude": list(ex["exclude"]),
                    "include_procedure": bool(proc["include_procedure"]),
                    "description": f"{ex['description']}; procedure={'yes' if proc['include_procedure'] else 'no'}",
                }
            )
    return modes


def resolve_mode_ingredients(cfg: dict[str, Any], v22: Path, mode: dict[str, Any]) -> dict[str, Any]:
    excluded = set(mode["exclude"])
    adapters: list[tuple[str, Path]] = []
    full_models: list[tuple[str, Path]] = []

    for rel in cfg["always_include"]["adapters"]:
        path = v22 / rel
        name = path.parent.name if path.name == "adapter" else path.name
        adapters.append((name, path))
    for rel in cfg["always_include"].get("full_models") or []:
        path = v22 / rel
        full_models.append((path.name, path))

    for toxic_name, meta in cfg["toxic_candidates"].items():
        if toxic_name in excluded:
            continue
        path = v22 / meta["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        if meta["kind"] == "adapter":
            adapters.append((toxic_name, path))
        else:
            full_models.append((toxic_name, path))

    if mode["include_procedure"]:
        proc = cfg["optional_procedure"]
        path = v22 / proc["path"]
        adapters.append((proc["name"], path))

    # Deduplicate by name while preserving order.
    seen: set[str] = set()
    uniq_adapters: list[tuple[str, Path]] = []
    for name, path in adapters:
        if name in seen:
            continue
        seen.add(name)
        uniq_adapters.append((name, path))
    uniq_full: list[tuple[str, Path]] = []
    for name, path in full_models:
        if name in seen:
            continue
        seen.add(name)
        uniq_full.append((name, path))

    return {
        "adapters": [{"name": n, "path": str(p)} for n, p in uniq_adapters],
        "full_models": [{"name": n, "path": str(p)} for n, p in uniq_full],
        "excluded": sorted(excluded),
        "include_procedure": mode["include_procedure"],
    }


def materialize_adapter(base: Path, adapter: Path, dest: Path, reuse: bool) -> Path:
    if dest.exists() and reuse and (dest / "model.safetensors").exists():
        return dest
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "materialize_adapter_cpu.py"),
        "--base-model",
        str(base),
        "--adapter",
        str(adapter),
        "--output-dir",
        str(dest),
        "--force",
    ]
    run(cmd)
    return dest


def build_mode_soup(
    *,
    cfg: dict[str, Any],
    default_cfg: dict[str, Any],
    v22: Path,
    mode: dict[str, Any],
    mode_root: Path,
    shared_mat_root: Path,
    reuse: bool,
) -> Path:
    soup_dir = mode_root / "soup"
    manifest_path = mode_root / "mode_manifest.json"
    if soup_dir.exists() and reuse and (soup_dir / "model.safetensors").exists() and manifest_path.exists():
        print(f"reuse soup {soup_dir}", flush=True)
        return soup_dir

    base = v22 / default_cfg["ingredients"]["frozen_anchor"]
    ingredients = resolve_mode_ingredients(cfg, v22, mode)
    # Shared densify cache across all 8 modes — never per-mode copies.
    shared_mat_root.mkdir(parents=True, exist_ok=True)
    dense_paths: list[Path] = []
    for item in ingredients["adapters"]:
        dense_paths.append(
            materialize_adapter(base, Path(item["path"]), shared_mat_root / item["name"], reuse=reuse)
        )
    for item in ingredients["full_models"]:
        dense_paths.append(Path(item["path"]))

    if len(dense_paths) < 2:
        raise RuntimeError(f"{mode['mode_id']}: need >=2 ingredients to average, got {len(dense_paths)}")

    # Drop legacy per-mode densify trees before the soup write (needs ~1.3GB staging).
    legacy = mode_root / "materialized"
    if legacy.exists():
        print(f"removing legacy per-mode materialize cache {legacy}", flush=True)
        shutil.rmtree(legacy, ignore_errors=True)
    ensure_disk_space(mode_root, context=f"{mode['mode_id']} soup write")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "average_checkpoints_cpu.py"),
        "--output-dir",
        str(soup_dir),
        "--force",
    ]
    for path in dense_paths:
        cmd.extend(["--model", str(path)])
    run(cmd)

    write_json(
        manifest_path,
        {
            "mode": mode,
            "ingredients": ingredients,
            "dense_checkpoints_averaged": [str(p) for p in dense_paths],
            "shared_materialized_root": str(shared_mat_root),
            "soup": str(soup_dir),
            "built_at": now(),
            "formula": "W = (1/N) * sum densified_ingredients",
        },
    )
    return soup_dir

# ---- lightweight CPU eval (same gold contract as smoke_eval_cpu) ----

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


def evaluate_model(
    model_dir: Path,
    benchmark: Path,
    dest: Path,
    *,
    limit: int | None,
    resume: bool,
    max_new: int = 32,
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

    need = any(str(row.get("id") or i) not in done for i, row in enumerate(rows, 1))
    model = tok = None
    if need:
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
            raise ValueError(f"missing assistant target: {rid}")
        answer, toks, sec = generate(model, tok, msgs[:-1], max_new)
        pack = {
            "id": rid,
            "decision": str(row.get("decision") or "answer"),
            "correct": correct(answer, row),
            "abstained": is_abstain(answer),
            "answer": answer,
            "gold": row.get("approved_answer"),
            "generated_tokens": toks,
            "generation_seconds": sec,
        }
        append_jsonl(results_path, pack)
        results.append(pack)
        if idx % 50 == 0 or idx == len(rows):
            print(f"  eval {model_dir.name}: {idx}/{len(rows)}", flush=True)
    if model is not None:
        del model
    answerable = [r for r in results if r["decision"] == "answer"]
    abstain = [r for r in results if r["decision"] != "answer"]
    summary = {
        "model": str(model_dir),
        "benchmark": str(benchmark),
        "case_count": len(results),
        "overall_correctness": rate([bool(r["correct"]) for r in results]),
        "answerable_correctness": rate([bool(r["correct"]) for r in answerable]),
        "abstention_correctness": rate([bool(r["correct"]) for r in abstain]),
        "false_abstention_rate": rate([bool(r["abstained"]) for r in answerable]),
        "mean_generation_seconds": statistics.mean(r["generation_seconds"] for r in results) if results else None,
    }
    write_json(dest / "summary.json", summary)
    return summary


def render_report(state: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Exclusion-matrix soup progress report",
        "",
        f"- Updated: `{state.get('updated_at')}`",
        f"- Device: CPU",
        f"- Eval limit: `{state.get('limit')}`",
        f"- ReuseExisting: `{state.get('reuse_existing')}`",
        "",
        "## Mode matrix (8)",
        "",
        "| Mode | Excluded | Procedure | Status | Soup | Best overall (bench) |",
        "|---|---|---|---|---|---|",
    ]
    for mode in state["modes"]:
        mid = mode["mode_id"]
        entry = state["results"].get(mid) or {}
        status = entry.get("status", "pending")
        soup = entry.get("soup", "")
        best = ""
        if entry.get("benchmarks"):
            parts = []
            for b, pkt in entry["benchmarks"].items():
                parts.append(f"{b}:{pkt.get('overall_correctness')}")
            best = "; ".join(parts)
        lines.append(
            f"| `{mid}` | {', '.join(mode['exclude']) or '(none of toxic trio)'} | "
            f"{'yes' if mode['include_procedure'] else 'no'} | {status} | `{soup}` | {best} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Toxic trio from v2.2 DEV selection margins: `track_a_broad`, `specialist_general`, `layer_separated`.",
            "- `specialist_procedure` is toggled independently because it alone scored on the procedural challenge.",
            "- Always kept: Track B, abstention, grounding, social.",
            "- Merge: densify included LoRAs onto frozen_anchor, optionally add layer_separated, uniform-average dense checkpoints.",
            "",
        ]
    )
    # Detailed per-mode blocks
    for mode in state["modes"]:
        mid = mode["mode_id"]
        entry = state["results"].get(mid)
        if not entry:
            continue
        lines.append(f"## `{mid}`")
        lines.append("")
        lines.append(f"- Status: **{entry.get('status')}**")
        lines.append(f"- Description: {mode['description']}")
        if entry.get("ingredients"):
            adapters = [x["name"] for x in entry["ingredients"].get("adapters", [])]
            fulls = [x["name"] for x in entry["ingredients"].get("full_models", [])]
            lines.append(f"- Adapters: {', '.join(adapters)}")
            lines.append(f"- Full models: {', '.join(fulls) if fulls else '(none)'}")
        if entry.get("benchmarks"):
            lines.append("")
            lines.append("| Benchmark | Cases | Overall | Answerable | Abstention |")
            lines.append("|---|---:|---:|---:|---:|")
            for b, pkt in entry["benchmarks"].items():
                lines.append(
                    f"| {b} | {pkt.get('case_count')} | {pkt.get('overall_correctness')} | "
                    f"{pkt.get('answerable_correctness')} | {pkt.get('abstention_correctness')} |"
                )
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description="Run 8-mode toxic-exclusion soup matrix")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "exclusion_matrix.json")
    parser.add_argument("--default-config", type=Path, default=ROOT / "configs" / "default_soups.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "exclusion_matrix")
    parser.add_argument("--ReuseExisting", action="store_true", help="Resume soups/evals that already exist.")
    parser.add_argument("--full", action="store_true", help="Use full official DEV benchmarks.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", action="append", dest="modes", help="Optional subset of mode_id values.")
    args = parser.parse_args()

    cfg = load_json(args.config)
    default_cfg = load_json(args.default_config)
    official = (ROOT / cfg["official_root"]).resolve()
    v22 = (ROOT / cfg["v22_run_root"]).resolve()
    modes = enumerate_modes(cfg)
    if args.modes:
        modes = [m for m in modes if m["mode_id"] in set(args.modes)]
        if not modes:
            raise SystemExit("No matching --mode values")

    if args.full and args.limit is not None:
        raise SystemExit("Do not pass --limit with --full")
    if args.full:
        bench_map = default_cfg["full_benchmarks"]
        limit = None
    else:
        bench_map = default_cfg["smoke_benchmarks"]
        limit = 48 if args.limit is None else args.limit
    benchmarks = {name: official / rel for name, rel in bench_map.items() if (official / rel).exists()}

    args.output_root.mkdir(parents=True, exist_ok=True)
    shared_mat_root = args.output_root / "_materialized"
    reclaim = reclaim_per_mode_materialized(args.output_root / "modes", shared_mat_root)
    if reclaim["moved"] or reclaim["removed"]:
        print(
            f"reclaimed duplicated materializations: moved={reclaim['moved']} "
            f"removed={reclaim['removed']} ~{reclaim['approx_freed_bytes'] / 1024**3:.2f} GiB",
            flush=True,
        )
    ensure_disk_space(args.output_root, context="exclusion matrix startup")
    state_path = args.output_root / "EXCLUSION_MATRIX_STATE.json"
    report_path = args.output_root / "PROGRESS_REPORT.md"
    state: dict[str, Any] = {
        "updated_at": now(),
        "reuse_existing": bool(args.ReuseExisting),
        "limit": limit,
        "full": bool(args.full),
        "modes": modes,
        "shared_materialized_root": str(shared_mat_root),
        "reclaim": reclaim,
        "results": {},
    }
    if args.ReuseExisting and state_path.exists():
        prev = load_json(state_path)
        state["results"] = prev.get("results") or {}

    # baselines for comparison (greedy + frozen) once
    baselines = {
        "frozen_anchor": v22 / default_cfg["ingredients"]["frozen_anchor"],
        "greedy_soup": v22 / default_cfg["ingredients"]["greedy_soup"],
    }
    for base_name, base_path in baselines.items():
        key = f"baseline::{base_name}"
        entry = state["results"].get(key) or {"status": "pending", "benchmarks": {}}
        if entry.get("status") == "complete" and args.ReuseExisting:
            print(f"reuse baseline eval {base_name}", flush=True)
        else:
            print(f"\n=== baseline {base_name} ===", flush=True)
            entry = {"status": "running", "soup": str(base_path), "benchmarks": {}, "started_at": now()}
            for bname, bpath in benchmarks.items():
                dest = args.output_root / "_baselines" / base_name / bname
                entry["benchmarks"][bname] = evaluate_model(
                    base_path, bpath, dest, limit=limit, resume=bool(args.ReuseExisting)
                )
                state["results"][key] = entry
                state["updated_at"] = now()
                write_json(state_path, state)
                render_report(state, report_path)
            entry["status"] = "complete"
            entry["finished_at"] = now()
            state["results"][key] = entry
            state["updated_at"] = now()
            write_json(state_path, state)
            render_report(state, report_path)

    for mode in modes:
        mid = mode["mode_id"]
        print(f"\n=== mode {mid} ===", flush=True)
        mode_root = args.output_root / "modes" / mid
        mode_root.mkdir(parents=True, exist_ok=True)
        entry = state["results"].get(mid) or {"status": "pending"}
        if entry.get("status") == "complete" and args.ReuseExisting:
            print(f"reuse complete mode {mid}", flush=True)
            continue
        entry = {
            "status": "building",
            "mode": mode,
            "ingredients": resolve_mode_ingredients(cfg, v22, mode),
            "started_at": now(),
            "benchmarks": {},
        }
        state["results"][mid] = entry
        state["updated_at"] = now()
        write_json(state_path, state)
        render_report(state, report_path)

        soup = build_mode_soup(
            cfg=cfg,
            default_cfg=default_cfg,
            v22=v22,
            mode=mode,
            mode_root=mode_root,
            shared_mat_root=shared_mat_root,
            reuse=bool(args.ReuseExisting),
        )
        entry["status"] = "evaluating"
        entry["soup"] = str(soup)
        state["results"][mid] = entry
        state["updated_at"] = now()
        write_json(state_path, state)
        render_report(state, report_path)

        for bname, bpath in benchmarks.items():
            dest = mode_root / "eval" / bname
            entry["benchmarks"][bname] = evaluate_model(
                soup, bpath, dest, limit=limit, resume=bool(args.ReuseExisting)
            )
            state["results"][mid] = entry
            state["updated_at"] = now()
            write_json(state_path, state)
            render_report(state, report_path)

        entry["status"] = "complete"
        entry["finished_at"] = now()
        state["results"][mid] = entry
        state["updated_at"] = now()
        write_json(state_path, state)
        render_report(state, report_path)

    state["updated_at"] = now()
    write_json(state_path, state)
    render_report(state, report_path)
    print(f"Done. Report: {report_path}")


if __name__ == "__main__":
    main()
