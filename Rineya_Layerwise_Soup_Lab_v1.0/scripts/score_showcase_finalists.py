#!/usr/bin/env python3
"""Score soup-lab finalists on the frozen 64-example showcase (descriptive only)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rineya_soup_lab.config import load_config
from rineya_soup_lab.gpu import wait_for_python_gpu
from rineya_soup_lab.model.layer_map import MergeRecipe
from rineya_soup_lab.model.sources import resolve_model_sources
from rineya_soup_lab.candidates import CandidateManager
from rineya_soup_lab.run_context import RunContext
from rineya_soup_lab.utils import read_json, write_json


REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "Rineya_Long_Context_Demo_v1"
DEFAULT_ALIASES = ("uniform", "global_best", "layerwise_seed", "layerwise_best")
RECORDED = {
    "halo": {"hits": 30, "items": 64, "strict_exact": 0.46875, "source": "vault Halo α=0.5"},
    "halo_uncensored": {
        "hits": 31,
        "items": 64,
        "strict_exact": 0.484375,
        "source": "vault Halo Uncensored α=1.0",
    },
    "late_rineya": {
        "hits": 31,
        "items": 64,
        "strict_exact": 0.484375,
        "source": "hand-recipe α sweep; not a soup-lab candidate",
    },
    "qwen3_06b": {
        "hits": 14,
        "items": 64,
        "strict_exact": 0.21875,
        "source": "official Qwen3-0.6B bf16",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hits_from_run(run: dict[str, Any]) -> int:
    return int(round(float(run["strict_exact"]) * int(run["key_items"])))


def row_from_run(name: str, run: dict[str, Any], source: str) -> dict[str, Any]:
    pack = run.get("pack_metrics") or {}
    return {
        "name": name,
        "source": source,
        "hits": hits_from_run(run),
        "items": run["key_items"],
        "strict_exact": run["strict_exact"],
        "unanswerable_accuracy": pack.get("unanswerable_accuracy"),
        "counterfactual_pair_both_correct_and_changed": pack.get(
            "counterfactual_pair_both_correct_and_changed"
        ),
        "short_response_rate_le_12_words": run.get("short_response_rate_le_12_words"),
        "coverage": run.get("coverage"),
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 × Halo Uncensored layerwise soup — showcase (descriptive)",
        "",
        "Coefficients were selected on chat DEV v4.5. This table is **not** used to pick α.",
        "",
        f"- Generated: `{payload['generated_at']}`",
        "- Protocol: greedy, 24 new tokens, context 4096, seed 17, thinking off, bf16",
        "- Recorded Halo 30/64 and Halo Uncensored 31/64 are vault reports, not restitched.",
        "",
        "| Name | Hits / 64 | Strict exact | Unanswerable | CF pair | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| `{name}` | {hits}/{items} | {strict:.4f} | {unans:.4f} | {cf:.4f} | {source} |".format(
                name=row["name"],
                hits=row["hits"],
                items=row["items"],
                strict=float(row["strict_exact"]),
                unans=float(row["unanswerable_accuracy"] or 0.0),
                cf=float(row["counterfactual_pair_both_correct_and_changed"] or 0.0),
                source=row["source"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Showcase-score soup-lab finalists.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aliases", nargs="+", default=list(DEFAULT_ALIASES))
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUTF8", "1")
    config = load_config(args.config)
    context = RunContext.create(config, existing_run=args.run_dir)
    sources = resolve_model_sources(context)
    manager = CandidateManager(context, sources)

    python = sys.executable
    prompts = DEMO / "data" / "showcase_prompts.jsonl"
    key = DEMO / "data" / "showcase_answer_key.jsonl"
    run_script = DEMO / "scripts" / "run_local_hf.py"
    eval_script = DEMO / "scripts" / "evaluate_predictions.py"
    out_dir = context.path("evaluations", "showcase")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONUTF8"] = "1"

    rows: list[dict[str, Any]] = []
    for name, recorded in RECORDED.items():
        rows.append(
            {
                "name": name,
                "source": recorded["source"],
                "hits": recorded["hits"],
                "items": recorded["items"],
                "strict_exact": recorded["strict_exact"],
                "unanswerable_accuracy": None,
                "counterfactual_pair_both_correct_and_changed": None,
                "short_response_rate_le_12_words": None,
                "coverage": 1.0,
            }
        )

    aliases = context.state.payload.get("aliases") or {}
    for alias in args.aliases:
        if alias not in aliases:
            print(f"[{utc_now()}] skip missing alias {alias}", flush=True)
            continue
        candidate_id = str(aliases[alias])
        record = context.state.candidate(candidate_id)
        recipe = MergeRecipe.from_dict(record["recipe"])
        print(f"[{utc_now()}] showcase {alias} ({candidate_id})", flush=True)
        model_path = manager.model_path_for(recipe, retain=True)
        pred_path = out_dir / f"{alias}.jsonl"
        report_json = out_dir / f"{alias}_report.json"
        report_md = out_dir / f"{alias}_report.md"
        wait_for_python_gpu(60)
        decode = [
            python,
            str(run_script),
            "--model",
            str(model_path),
            "--input",
            str(prompts),
            "--output",
            str(pred_path),
            "--max-new-tokens",
            "24",
            "--max-context",
            "4096",
            "--seed",
            "17",
            "--device-map",
            "auto",
            "--torch-dtype",
            "bfloat16",
            "--resume",
            "--disable-thinking",
        ]
        print(f"[{utc_now()}] $ {' '.join(decode)}", flush=True)
        subprocess.run(decode, check=True, cwd=str(REPO), env=env)
        subprocess.run(
            [
                python,
                str(eval_script),
                "--key",
                str(key),
                "--predictions",
                str(pred_path),
                "--json-report",
                str(report_json),
                "--markdown-report",
                str(report_md),
            ],
            check=True,
            cwd=str(REPO),
            env=env,
        )
        report = read_json(report_json)
        rows.append(row_from_run(alias, report["runs"][0], source=f"soup-lab alias {alias}"))

    payload = {
        "generated_at": utc_now(),
        "run_dir": str(context.root),
        "note": "Showcase is descriptive. Chat DEV v4.5 selected coefficients.",
        "rows": rows,
    }
    write_json(out_dir / "scoreboard.json", payload)
    (out_dir / "scoreboard.md").write_text(render_md(payload), encoding="utf-8")
    print(f"[{utc_now()}] wrote {out_dir / 'scoreboard.md'}", flush=True)


if __name__ == "__main__":
    main()
