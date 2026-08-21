#!/usr/bin/env python3
"""CPU-only showcase + dual (chat DEV + RAG DEV) for late_rineya and soup-lab global_best.

Keeps the Halo-pair GPU soup-lab alone: CUDA_VISIBLE_DEVICES is forced empty.
Does not rename or vault anything.
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
DEMO = REPO / "Rineya_Long_Context_Demo_v1"
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

LATE_RINEYA = LAB / "artifacts" / "alpha_recipes" / "late_rineya"
GLOBAL_BEST = (
    REPO
    / "Rineya_Layerwise_Soup_Lab_v1.0"
    / "runs"
    / "qwen3_halo_uncensored"
    / "candidate_cache"
    / "cand_d1b08a8faf299ce4"
)
SOUP_STATE = (
    REPO
    / "Rineya_Layerwise_Soup_Lab_v1.0"
    / "runs"
    / "qwen3_halo_uncensored"
    / "state"
    / "run_state.json"
)

SCORE_WEIGHTS = {
    "chat_quality": 0.30,
    "task_adequacy": 0.15,
    "oracle_supported": 0.20,
    "real_supported": 0.15,
    "real_procedure": 0.10,
    "oracle_abstain": 0.10,
}

# Recorded dual-skill DEV rows for side-by-side (phase_evals scoreboard).
COMPARATORS = {
    "Aegis 19L": {
        "chat_quality": 0.8593539651416122,
        "task_adequacy": 0.49776470588235294,
        "oracle_supported": 1.0,
        "real_supported": 0.7358490566037735,
        "real_procedure": 0.9533333333333334,
        "oracle_abstain": 1.0,
        "dual": 0.8382,
        "showcase": "4/64",
    },
    "Nimbus (qwen3_x_aegis)": {
        "chat_quality": 0.8602,
        "task_adequacy": 0.5032,
        "oracle_supported": 0.8906,
        "real_supported": 0.6736,
        "real_procedure": 0.8600,
        "oracle_abstain": 0.8273,
        "dual": 0.7814,
        "showcase": "24/64",
    },
    "Lyra (qwen3_x_v1_5e4_1e5)": {
        "chat_quality": 0.8667,
        "task_adequacy": 0.5289,
        "oracle_supported": 0.8415,
        "real_supported": 0.6283,
        "real_procedure": 0.7000,
        "oracle_abstain": 0.7266,
        "dual": 0.7445,
        "showcase": "28/64",
    },
    "Halo (qwen3_x_v1_5e4_v2)": {
        "chat_quality": 0.8619,
        "task_adequacy": 0.5117,
        "oracle_supported": 0.8472,
        "real_supported": 0.6283,
        "real_procedure": 0.7000,
        "oracle_abstain": 0.6906,
        "dual": 0.7381,
        "showcase": "30/64",
    },
    "Pulse": {
        "chat_quality": 0.7984,
        "task_adequacy": None,
        "oracle_supported": None,
        "real_supported": None,
        "real_procedure": None,
        "oracle_abstain": None,
        "dual": None,
        "showcase": "0/64",
    },
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


def resolve_global_best() -> tuple[Path, str]:
    """Return (model_path, note). Prefer current soup-lab alias if it still matches."""
    note = "soup-lab alias global_best = cand_d1b08a8faf299ce4 (uniform α=0.40)"
    if SOUP_STATE.is_file():
        state = load_json(SOUP_STATE)
        aliases = state.get("aliases") or {}
        candidate_id = str(aliases.get("global_best") or "")
        if candidate_id:
            path = (
                REPO
                / "Rineya_Layerwise_Soup_Lab_v1.0"
                / "runs"
                / "qwen3_halo_uncensored"
                / "candidate_cache"
                / candidate_id
            )
            if weights_ok(path):
                note = f"soup-lab alias global_best = {candidate_id}"
                return path, note
            raise SystemExit(f"global_best alias {candidate_id} has no weights at {path}")
    if weights_ok(GLOBAL_BEST):
        return GLOBAL_BEST, note
    raise SystemExit(f"Missing global_best weights: {GLOBAL_BEST}")


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


def showcase_row(name: str, report: dict[str, Any], source: str) -> dict[str, Any]:
    run = report["runs"][0]
    pack = run.get("pack_metrics") or {}
    hits = int(round(float(run["strict_exact"]) * int(run["key_items"])))
    return {
        "name": name,
        "source": source,
        "hits": hits,
        "items": run["key_items"],
        "strict_exact": run["strict_exact"],
        "unanswerable_accuracy": pack.get("unanswerable_accuracy"),
        "counterfactual_pair_both_correct_and_changed": pack.get(
            "counterfactual_pair_both_correct_and_changed"
        ),
        "short_response_rate_le_12_words": run.get("short_response_rate_le_12_words"),
    }


def run_showcase(python: str, model: Path, name: str, out_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = out_dir / f"{name}.jsonl"
    report_json = out_dir / f"{name}_report.json"
    report_md = out_dir / f"{name}_report.md"
    run_checked(
        [
            python,
            str(DEMO / "scripts" / "run_local_hf.py"),
            "--model",
            str(model),
            "--input",
            str(DEMO / "data" / "showcase_prompts.jsonl"),
            "--output",
            str(pred),
            "--max-new-tokens",
            "24",
            "--max-context",
            "4096",
            "--seed",
            "17",
            "--device-map",
            "cpu",
            "--torch-dtype",
            "bfloat16",
            "--resume",
            "--disable-thinking",
        ],
        env,
    )
    run_checked(
        [
            python,
            str(DEMO / "scripts" / "evaluate_predictions.py"),
            "--key",
            str(DEMO / "data" / "showcase_answer_key.jsonl"),
            "--predictions",
            str(pred),
            "--json-report",
            str(report_json),
            "--markdown-report",
            str(report_md),
        ],
        env,
    )
    return load_json(report_json)


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
        "# late_rineya + global_best — CPU dual + showcase fill-in",
        "",
        f"- Generated: `{payload['generated_at']}`",
        "- Device: **CPU** (`CUDA_VISIBLE_DEVICES` empty) so the Halo-pair soup-lab can keep the GPU.",
        "- Showcase: greedy, 24 tok, 4096, seed 17, thinking off, bf16.",
        "- Dual: Phase 1 chat DEV (510) + Phase 9 RAG DEV (752×2), same weights as family scoreboard.",
        f"- global_best note: {payload.get('global_best_note')}",
        "",
        "## Showcase",
        "",
        "| Name | Hits / 64 | Strict exact | Unanswerable | CF pair | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("showcase_rows") or []:
        lines.append(
            "| `{name}` | {hits}/{items} | {strict:.4f} | {unans} | {cf} | {source} |".format(
                name=row["name"],
                hits=row["hits"],
                items=row["items"],
                strict=float(row["strict_exact"]),
                unans=fmt4(row.get("unanswerable_accuracy")),
                cf=fmt4(row.get("counterfactual_pair_both_correct_and_changed")),
                source=row["source"],
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
    for name, ref in COMPARATORS.items():
        lines.append(
            "| `{name}` | {dual} | {cq} | {ta} | {os} | {rs} | {oa} | {rp} | {sc} |".format(
                name=name,
                dual=fmt4(ref.get("dual")),
                cq=fmt4(ref.get("chat_quality")),
                ta=fmt4(ref.get("task_adequacy")),
                os=fmt4(ref.get("oracle_supported")),
                rs=fmt4(ref.get("real_supported")),
                oa=fmt4(ref.get("oracle_abstain")),
                rp=fmt4(ref.get("real_procedure")),
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
    lines.extend(["", payload.get("note") or "", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["showcase", "dual"],
        choices=("showcase", "dual", "chat", "rag"),
        help="showcase = 64 long-context; dual = chat+rag; or chat/rag alone.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["global_best", "late_rineya"],
        choices=("global_best", "late_rineya"),
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--skip-existing-dual", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYTHONUTF8", "1")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    python = sys.executable
    out_root = LAB / "outputs" / "lab_fillins_cpu"
    out_root.mkdir(parents=True, exist_ok=True)

    gb_path, gb_note = resolve_global_best()
    models = {
        "global_best": (gb_path, gb_note),
        "late_rineya": (LATE_RINEYA, "CrossDepth alpha recipe late_rineya (Halo donor, non-uniform α)"),
    }
    for name in args.models:
        path, _ = models[name]
        if not weights_ok(path):
            raise SystemExit(f"Missing weights for {name}: {path}")
        print(f"[{utc_now()}] {name} -> {path}", flush=True)

    showcase_rows: list[dict[str, Any]] = []
    # Known recorded showcase for late_rineya (already scored).
    recorded_showcase = {
        "late_rineya": {
            "name": "late_rineya",
            "source": "prior alpha_recipes sweep (reused; not re-scored unless --models forces)",
            "hits": 31,
            "items": 64,
            "strict_exact": 0.484375,
            "unanswerable_accuracy": 0.125,
            "counterfactual_pair_both_correct_and_changed": 0.4375,
            "short_response_rate_le_12_words": None,
        }
    }

    if "showcase" in args.stages:
        for name in args.models:
            path, note = models[name]
            if name == "late_rineya" and "global_best" in args.models:
                # Default path: only re-score global_best; reuse late_rineya 31/64.
                showcase_rows.append(recorded_showcase["late_rineya"])
                continue
            print(f"[{utc_now()}] === showcase {name} (CPU) ===", flush=True)
            report = run_showcase(python, path, name, out_root / "showcase", env)
            showcase_rows.append(showcase_row(name, report, note))

    dual_rows: list[dict[str, Any]] = []
    need_chat = "dual" in args.stages or "chat" in args.stages
    need_rag = "dual" in args.stages or "rag" in args.stages
    if need_chat or need_rag:
        for name in args.models:
            path, note = models[name]
            chat_dir = out_root / name / "chat_dev"
            rag_dir = out_root / name / "rag_dev"
            chat_summary = chat_dir / "summary.json"
            rag_summary = rag_dir / "summary.json"
            if args.skip_existing_dual and chat_summary.is_file() and rag_summary.is_file():
                print(f"[{utc_now()}] skip existing dual for {name}", flush=True)
            else:
                print(f"[{utc_now()}] === dual {name} (CPU) ===", flush=True)
                if need_chat and not (args.skip_existing_dual and chat_summary.is_file()):
                    run_chat(python, path, chat_dir, env, args.torch_dtype)
                if need_rag and not (args.skip_existing_dual and rag_summary.is_file()):
                    run_rag(python, path, rag_dir, env, args.torch_dtype)
            row: dict[str, Any] = {
                "name": name,
                "model_path": str(path),
                "note": note,
                "chat": chat_metrics(chat_summary) if chat_summary.is_file() else {},
                "rag": rag_metrics(rag_summary) if rag_summary.is_file() else {},
            }
            sc = next((r for r in showcase_rows if r["name"] == name), None)
            if sc is None and name in recorded_showcase:
                sc = recorded_showcase[name]
            if sc is not None:
                row["showcase"] = f"{sc['hits']}/{sc['items']}"
            dual_rows.append(row)

    payload = {
        "generated_at": utc_now(),
        "global_best_note": gb_note,
        "showcase_rows": showcase_rows,
        "dual_rows": dual_rows,
        "comparators": COMPARATORS,
        "note": (
            "Lab fill-in only. global_best is the soup-lab uniform global α winner so far; "
            "layer_influence probes can score higher on chat but do not replace the alias until "
            "layer_search/freeze. late_rineya/global_best are not vault cores."
        ),
    }
    write_json(out_root / "scoreboard.json", payload)
    (out_root / "scoreboard.md").write_text(render_md(payload), encoding="utf-8")
    print(f"[{utc_now()}] Wrote {out_root / 'scoreboard.md'}", flush=True)
    print(json.dumps({"showcase_rows": showcase_rows, "dual_rows": dual_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
