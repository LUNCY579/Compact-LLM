#!/usr/bin/env python3
"""Chat-quality and RAG-supportedness eval for the three 28L Qwen3 cross-depth hybrids."""

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
VAULT_HALO = REPO / "Rineya_Published_Cores" / "Rineya-Halo-1.2.0-devfreeze"
VAULT_FOR = {
    "qwen3_x_aegis": ("Nimbus", REPO / "Rineya_Published_Cores" / "Rineya-Nimbus-1.3.0-devfreeze"),
    "qwen3_x_v1_5e4_1e5": ("Lyra", REPO / "Rineya_Published_Cores" / "Rineya-Lyra-1.4.0-devfreeze"),
    "qwen3_x_v1_5e4_v2": ("Halo", VAULT_HALO),
}
THREE = ("qwen3_x_aegis", "qwen3_x_v1_5e4_1e5", "qwen3_x_v1_5e4_v2")
REPLACE = ("qwen3_replace_aegis", "qwen3_replace_v1_5e4_1e5", "qwen3_replace_v1_5e4_v2")
KNOWN = THREE + REPLACE
SCORE_WEIGHTS = {
    "chat_quality": 0.30,
    "task_adequacy": 0.15,
    "oracle_supported": 0.20,
    "real_supported": 0.15,
    "real_procedure": 0.10,
    "oracle_abstain": 0.10,
}
AEGIS_V1_5E4 = {
    "label": "Aegis 19L (teacher dual-SFT v1 5e-4, PEFT on Pulse)",
    "source": "short-response-model-phase-0-1-2/artifacts/teacher_dual_sft_v1/lr_sweep/SELECTION_REPORT.md",
    "chat_quality": 0.8593539651416122,
    "task_adequacy": 0.49776470588235294,
    "oracle_supported": 1.0,
    "real_supported": 0.7358490566037735,
    "real_procedure": 0.9533333333333334,
    "oracle_abstain": 1.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_checked(command: list[str], env: dict[str, str]) -> None:
    print(f"[{utc_now()}] $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=str(REPO), env=env)


def fnum(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def drop_weight_files(target: Path) -> list[str]:
    removed: list[str] = []
    if not target.exists():
        return removed
    for path in target.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".safetensors", ".bin"} or path.name == "tokenizer.json":
            path.unlink()
            removed.append(str(path.relative_to(target)))
    return removed


def weights_present(target: Path) -> bool:
    if not target.exists():
        return False
    return any(path.is_file() and path.suffix in {".safetensors", ".bin"} for path in target.rglob("*"))


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


def dual_score(metrics: dict[str, float | None]) -> float | None:
    parts: list[float] = []
    for key, weight in SCORE_WEIGHTS.items():
        value = metrics.get(key)
        if value is None:
            return None
        parts.append(weight * value)
    return sum(parts)


def fmt4(value: object) -> str:
    number = fnum(value)
    return "—" if number is None else f"{number:.4f}"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# 28L hybrid chat + RAG eval",
        "",
        f"- Generated: `{payload['generated_at']}`",
        "- Protocol: Phase 1 chat quality (`dev.jsonl`, 32 tokens, eos_only) and Phase 9 RAG",
        "  (`dev.reviewed.jsonl`, oracle_rag + real_rag, query_aware_v21). Thinking disabled.",
        "- Same dual-skill DEV protocol as teacher dual-SFT. Not sealed TEST.",
        "",
        "| Model | Dual | Chat Q | Task adequacy | Oracle supp | Real supp | Oracle abstain | Real proc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ref = payload.get("aegis_reference") or {}
    if ref:
        lines.append(
            "| `{label}` | {dual} | {cq} | {ta} | {os} | {rs} | {oa} | {rp} |".format(
                label=ref.get("label", "Aegis 19L"),
                dual=fmt4(dual_score(ref)),
                cq=fmt4(ref.get("chat_quality")),
                ta=fmt4(ref.get("task_adequacy")),
                os=fmt4(ref.get("oracle_supported")),
                rs=fmt4(ref.get("real_supported")),
                oa=fmt4(ref.get("oracle_abstain")),
                rp=fmt4(ref.get("real_procedure")),
            )
        )
    for row in payload["rows"]:
        merged = {**(row.get("chat") or {}), **(row.get("rag") or {})}
        lines.append(
            "| `{name}` | {dual} | {cq} | {ta} | {os} | {rs} | {oa} | {rp} |".format(
                name=row["name"],
                dual=fmt4(dual_score(merged)),
                cq=fmt4(merged.get("chat_quality")),
                ta=fmt4(merged.get("task_adequacy")),
                os=fmt4(merged.get("oracle_supported")),
                rs=fmt4(merged.get("real_supported")),
                oa=fmt4(merged.get("oracle_abstain")),
                rp=fmt4(merged.get("real_procedure")),
            )
        )
    lines.extend(["", payload.get("note") or "", ""])
    return "\n".join(lines)


def write_scoreboard(outputs: Path, rows: list[dict[str, Any]]) -> None:
    scoreboard_path = outputs / "scoreboard.json"
    if scoreboard_path.is_file():
        old = load_json(scoreboard_path).get("rows") or []
        updated = {row["name"]: row for row in rows}
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in old:
            name = str(row.get("name") or "")
            if name in updated:
                merged.append(updated[name])
                seen.add(name)
            else:
                merged.append(row)
        for row in rows:
            if row["name"] not in seen:
                merged.append(row)
        rows = merged
    payload = {
        "generated_at": utc_now(),
        "lab": "Rineya_Qwen3_CrossDepth_Soup_Mini_v1/phase_evals",
        "aegis_reference": AEGIS_V1_5E4,
        "rows": rows,
        "note": (
            "50/50 cores: Nimbus / Lyra / Halo. Replace-kept (alpha=1.0) rows are lab-only until recorded. "
            "Aegis row is the prior 19L PEFT dual-SFT v1 5e-4 DEV freeze, not a 28L hybrid."
        ),
    }
    (outputs / "scoreboard.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (outputs / "scoreboard.md").write_text(render_md(payload), encoding="utf-8")
    print(f"[{utc_now()}] Wrote {outputs / 'scoreboard.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=list(THREE))
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--skip-rematerialize", action="store_true")
    parser.add_argument("--keep-lab-weights", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUTF8", "1")
    eval_env = os.environ.copy()
    eval_env["PYTHONUTF8"] = "1"
    if not eval_env.get("CUDA_VISIBLE_DEVICES") or eval_env.get("CUDA_VISIBLE_DEVICES") == "-1":
        eval_env["CUDA_VISIBLE_DEVICES"] = "0"
    eval_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cpu_env = eval_env.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = "-1"

    outputs = LAB / "outputs" / "phase_evals"
    outputs.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    rows: list[dict[str, Any]] = []

    for name in args.only:
        if name not in KNOWN:
            raise SystemExit(f"Unknown hybrid: {name}")

        vault = VAULT_FOR.get(name)
        use_vault = vault is not None and (vault[1] / "model.safetensors").is_file()
        if use_vault:
            recorded_as, model_path = vault
            print(f"[{utc_now()}] {name}: using recorded {recorded_as} vault {model_path}", flush=True)
        else:
            recorded_as = None
            model_path = LAB / "artifacts" / name
            if not args.skip_rematerialize and not weights_present(model_path):
                run_checked(
                    [
                        python,
                        str(LAB / "scripts" / "run_lab.py"),
                        "--only",
                        name,
                        "--materialize-only",
                    ],
                    cpu_env,
                )
            if not (model_path / "config.json").exists() or not weights_present(model_path):
                raise SystemExit(f"Missing hybrid checkpoint: {model_path}")

        chat_dir = outputs / name / "chat_dev"
        rag_dir = outputs / name / "rag_dev"
        print(f"[{utc_now()}] === phase eval {name} @ {model_path} ===", flush=True)
        run_checked(
            [
                python,
                str(DEMO_EVAL),
                "--model-id",
                str(model_path),
                "--benchmark",
                str(CHAT_DEV),
                "--output-dir",
                str(chat_dir),
                "--quality-config",
                str(QUALITY),
                "--max-new-tokens",
                "32",
                "--response-control",
                "eos_only",
                "--require-cuda",
                "--torch-dtype",
                args.torch_dtype,
                "--resume",
            ],
            eval_env,
        )
        run_checked(
            [
                python,
                str(RAG_EVAL),
                "--model",
                str(model_path),
                "--benchmark",
                str(RAG_DEV),
                "--output-dir",
                str(rag_dir),
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
                "--require-cuda",
                "--require-research-ready",
                "--torch-dtype",
                args.torch_dtype,
                "--resume",
            ],
            eval_env,
        )
        rows.append(
            {
                "name": name,
                "model_path": str(model_path),
                "recorded_as": recorded_as,
                "chat": chat_metrics(chat_dir / "summary.json"),
                "rag": rag_metrics(rag_dir / "summary.json"),
            }
        )
        write_scoreboard(outputs, rows)
        if not args.keep_lab_weights and not use_vault:
            removed = drop_weight_files(model_path)
            print(f"[{utc_now()}] {name}: dropped lab hybrid weights {removed}", flush=True)


if __name__ == "__main__":
    main()
