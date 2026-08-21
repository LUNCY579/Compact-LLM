#!/usr/bin/env python3
"""CPU dual (chat DEV + RAG DEV) for R2 baseline and CrossDepth hybrids.

CUDA_VISIBLE_DEVICES is forced empty. Showcase reports from run_lab.py are reused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
PHASE = REPO / "short-response-model-phase-0-1-2"
DEMO_EVAL = PHASE / "scripts" / "phase_1_dense_sft" / "evaluate_model.py"
RAG_EVAL = PHASE / "scripts" / "phase_9_student_distillation" / "evaluate_rag.py"
CHAT_DEV = PHASE / "data" / "dev" / "dev.jsonl"
RAG_DEV = (
    PHASE
    / "data"
    / "phase_9_student_distillation"
    / "rag_reasoning_v3_large"
    / "reviewed_splits"
    / "dev.reviewed.jsonl"
)
QUALITY = PHASE / "configs" / "quality_evaluation.json"
KDB = PHASE / "artifacts" / "teacher_dual_sft_v1" / "rag_v3.sqlite3"
CAL = PHASE / "artifacts" / "teacher_dual_sft_v1" / "rag_v3_calibration_query_aware_v21.json"

SCORE_WEIGHTS = {
    "chat_quality": 0.30,
    "task_adequacy": 0.15,
    "oracle_supported": 0.20,
    "real_supported": 0.15,
    "real_procedure": 0.10,
    "oracle_abstain": 0.10,
}

QWEN3_COMPARATORS = {
    "Qwen3 Nimbus (α=0.5 Aegis)": {"dual": 0.7814, "showcase": "24/64"},
    "Qwen3 Lyra": {"dual": 0.7445, "showcase": "28/64"},
    "Qwen3 Halo": {"dual": 0.7381, "showcase": "30/64"},
    "Qwen3 Halo Uncensored": {"dual": 0.6823, "showcase": "31/64"},
    "Qwen3 late_rineya / Pulse 1.6": {"dual": 0.7561, "showcase": "31/64"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fnum(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def dual_score(metrics: dict[str, float | None]) -> float | None:
    parts: list[float] = []
    for key, weight in SCORE_WEIGHTS.items():
        value = metrics.get(key)
        if value is None:
            return None
        parts.append(weight * float(value))
    return sum(parts)


def fmt4(value: object) -> str:
    number = fnum(value)
    return "—" if number is None else f"{number:.4f}"


def run_checked(command: list[str], env: dict[str, str]) -> None:
    print(f"[{utc_now()}] $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=str(REPO), env=env)


def weights_ok(path: Path) -> bool:
    return path.is_dir() and any(
        p.is_file() and p.suffix in {".safetensors", ".bin"} for p in path.rglob("*")
    )


def chat_metrics(summary_path: Path) -> dict[str, float | None]:
    chat = load_json(summary_path)
    dims = chat.get("mean_quality_dimensions") or {}
    return {
        "chat_quality": fnum(chat.get("mean_quality_score")),
        "task_adequacy": fnum(dims.get("task_adequacy")),
        "median_reply_tokens": fnum(chat.get("median_reply_tokens")),
        "duplicate_reply_rate": fnum(chat.get("duplicate_reply_rate")),
    }


def rag_metrics(summary_path: Path) -> dict[str, float | None]:
    rag = load_json(summary_path)
    cond = rag.get("conditions") or {}
    oracle = cond.get("oracle_rag") or {}
    real = cond.get("real_rag") or {}
    return {
        "oracle_supported": fnum(oracle.get("supported_answer_correctness")),
        "oracle_abstain": fnum(oracle.get("correct_abstention_rate")),
        "oracle_procedure": fnum(oracle.get("procedure_correctness")),
        "real_supported": fnum(real.get("supported_answer_correctness")),
        "real_abstain": fnum(real.get("correct_abstention_rate")),
        "real_procedure": fnum(real.get("procedure_correctness")),
    }


def showcase_from_report(name: str, report: dict[str, Any]) -> dict[str, Any]:
    run = report["runs"][0]
    pack = run.get("pack_metrics") or {}
    hits = int(round(float(run["strict_exact"]) * int(run["key_items"])))
    return {
        "name": name,
        "hits": hits,
        "items": run["key_items"],
        "strict_exact": run["strict_exact"],
        "unanswerable_accuracy": pack.get("unanswerable_accuracy"),
        "counterfactual_pair_both_correct_and_changed": pack.get(
            "counterfactual_pair_both_correct_and_changed"
        ),
        "short_response_rate_le_12_words": run.get("short_response_rate_le_12_words"),
    }


def run_chat(python: str, model: Path, out_dir: Path, env: dict[str, str], dtype: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            python,
            str(DEMO_EVAL),
            "--model-id",
            str(model),
            "--benchmark",
            str(CHAT_DEV),
            "--output-dir",
            str(out_dir),
            "--quality-config",
            str(QUALITY),
            "--max-new-tokens",
            "32",
            "--response-control",
            "eos_only",
            "--torch-dtype",
            dtype,
            "--resume",
        ],
        env,
    )


def run_rag(python: str, model: Path, out_dir: Path, env: dict[str, str], dtype: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            python,
            str(RAG_EVAL),
            "--model",
            str(model),
            "--benchmark",
            str(RAG_DEV),
            "--output-dir",
            str(out_dir),
            "--knowledge-db",
            str(KDB),
            "--gate-calibration",
            str(CAL),
            "--conditions",
            "oracle_rag,real_rag",
            "--retrieval-selection-policy",
            "query_aware_v21",
            "--response-control",
            "eos_only",
            "--max-new-tokens",
            "32",
            "--require-research-ready",
            "--torch-dtype",
            dtype,
            "--resume",
        ],
        env,
    )


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R2 CrossDepth — CPU showcase + dual",
        "",
        f"- Generated: `{payload['generated_at']}`",
        "- Device: **CPU** (`CUDA_VISIBLE_DEVICES` empty).",
        "- Scaffold: local copy of R2 (`rineya-core-en32k-28L`, vocab 32768).",
        "- Showcase: greedy, 24 tok, 4096, seed 17, thinking off, bf16.",
        "- Dual: Phase 1 chat DEV (510) + Phase 9 RAG DEV (752×2).",
        "- Qwen3 family rows are **cross-scaffold references**, not paired.",
        "",
        "## Showcase",
        "",
        "| Name | Hits / 64 | Strict exact | Unanswerable | CF pair |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("showcase_rows") or []:
        lines.append(
            "| `{name}` | {hits}/{items} | {strict:.4f} | {unans} | {cf} |".format(
                name=row["name"],
                hits=row["hits"],
                items=row["items"],
                strict=float(row["strict_exact"]),
                unans=fmt4(row.get("unanswerable_accuracy")),
                cf=fmt4(row.get("counterfactual_pair_both_correct_and_changed")),
            )
        )
    lines.extend(
        [
            "",
            "## Dual composite",
            "",
            "| Model | Dual | Chat Q | Task adequacy | Oracle supp | Real supp | Oracle abstain | Real proc | Showcase |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, ref in QWEN3_COMPARATORS.items():
        lines.append(
            "| {name} (Qwen3 ref) | {dual} | — | — | — | — | — | — | {sc} |".format(
                name=name,
                dual=fmt4(ref.get("dual")),
                sc=ref.get("showcase") or "—",
            )
        )
    for row in payload.get("dual_rows") or []:
        merged = {**(row.get("chat") or {}), **(row.get("rag") or {})}
        lines.append(
            "| `{name}` | {dual} | {cq} | {ta} | {os} | {rs} | {oa} | {rp} | {sc} |".format(
                name=row["name"],
                dual=fmt4(dual_score(merged)),
                cq=fmt4(merged.get("chat_quality")),
                ta=fmt4(merged.get("task_adequacy")),
                os=fmt4(merged.get("oracle_supported")),
                rs=fmt4(merged.get("real_supported")),
                oa=fmt4(merged.get("oracle_abstain")),
                rp=fmt4(merged.get("real_procedure")),
                sc=row.get("showcase") or "—",
            )
        )
    lines.extend(
        [
            "",
            payload.get("note")
            or "Lab copies only. Not vaulted to Rineya_Published_Cores.",
            "",
        ]
    )
    return "\n".join(lines)


def model_catalog() -> dict[str, Path]:
    hybrids = LAB / "artifacts" / "hybrids"
    return {
        "r2_baseline": LAB / "scaffolds" / "rineya-core-en32k-28L",
        "r2_x_aegis": hybrids / "r2_x_aegis",
        "r2_x_v1_5e4_1e5": hybrids / "r2_x_v1_5e4_1e5",
        "r2_x_v1_5e4_v2": hybrids / "r2_x_v1_5e4_v2",
        "r2_x_v1_5e4_v2_a1": hybrids / "r2_x_v1_5e4_v2_a1",
        "r2_late_rineya": hybrids / "r2_late_rineya",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["dual"],
        choices=("dual", "chat", "rag"),
    )
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYTHONUTF8", "1")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    python = sys.executable
    catalog = model_catalog()
    names = args.models or list(catalog)
    missing = [name for name in names if name not in catalog]
    if missing:
        raise SystemExit(f"Unknown models: {missing}")
    for name in names:
        path = catalog[name]
        if not weights_ok(path):
            raise SystemExit(f"Missing weights for {name}: {path}")

    out_root = LAB / "outputs" / "dual_cpu"
    out_root.mkdir(parents=True, exist_ok=True)

    showcase_rows: list[dict[str, Any]] = []
    showcase_dir = LAB / "outputs" / "showcase"
    for name in ["r2_baseline", *list(catalog.keys())]:
        report_path = showcase_dir / f"{name}_report.json"
        if report_path.is_file() and name not in {row["name"] for row in showcase_rows}:
            showcase_rows.append(showcase_from_report(name, load_json(report_path)))

    dual_rows: list[dict[str, Any]] = []
    need_chat = "dual" in args.stages or "chat" in args.stages
    need_rag = "dual" in args.stages or "rag" in args.stages
    for name in names:
        path = catalog[name]
        chat_dir = out_root / name / "chat_dev"
        rag_dir = out_root / name / "rag_dev"
        chat_summary = chat_dir / "summary.json"
        rag_summary = rag_dir / "summary.json"
        print(f"[{utc_now()}] === dual {name} (CPU) ===", flush=True)
        if need_chat and not (args.skip_existing and chat_summary.is_file()):
            run_chat(python, path, chat_dir, env, args.torch_dtype)
        if need_rag and not (args.skip_existing and rag_summary.is_file()):
            run_rag(python, path, rag_dir, env, args.torch_dtype)
        row: dict[str, Any] = {
            "name": name,
            "model_path": str(path),
            "chat": chat_metrics(chat_summary) if chat_summary.is_file() else {},
            "rag": rag_metrics(rag_summary) if rag_summary.is_file() else {},
        }
        sc = next((item for item in showcase_rows if item["name"] == name), None)
        if sc is not None:
            row["showcase"] = f"{sc['hits']}/{sc['items']}"
        dual_rows.append(row)

    payload = {
        "generated_at": utc_now(),
        "showcase_rows": showcase_rows,
        "dual_rows": dual_rows,
        "qwen3_comparators": QWEN3_COMPARATORS,
        "note": (
            "Lab copies only; not vaulted. Qwen3 published-core numbers are cross-scaffold "
            "references (vocab 151936 vs R2 32768)."
        ),
    }
    write_json(out_root / "scoreboard.json", payload)
    (out_root / "scoreboard.md").write_text(render_md(payload), encoding="utf-8")
    combined = LAB / "outputs" / "scoreboard.md"
    combined.write_text(render_md(payload), encoding="utf-8")
    write_json(LAB / "outputs" / "scoreboard.json", payload)
    print(f"[{utc_now()}] Wrote {out_root / 'scoreboard.md'} and {combined}", flush=True)


if __name__ == "__main__":
    main()
