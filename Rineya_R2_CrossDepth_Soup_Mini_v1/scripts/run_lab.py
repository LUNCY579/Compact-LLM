#!/usr/bin/env python3
"""Rematerialize 19L donors, stitch R2 28L hybrids, score the frozen 64-row showcase. CPU only."""

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
DEMO = REPO / "Rineya_Long_Context_Demo_v1"
PHASE = REPO / "short-response-model-phase-0-1-2"
MERGE_SCRIPT = PHASE / "scripts" / "phase_4_model_soups" / "merge_delta_soup.py"
STITCH_SCRIPT = REPO / "Rineya_Qwen3_CrossDepth_Soup_Mini_v1" / "scripts" / "stitch_cross_depth.py"
RUN_SCRIPT = DEMO / "scripts" / "run_local_hf.py"
EVAL_SCRIPT = DEMO / "scripts" / "evaluate_predictions.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    return path.resolve()


def ok_prediction_count(path: Path) -> int:
    return sum(1 for row in load_jsonl(path) if row.get("status", "ok") == "ok")


def run_checked(command: list[str], env: dict[str, str]) -> None:
    print(f"[{utc_now()}] $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=str(REPO), env=env)


def weights_present(target: Path) -> bool:
    if not target.exists():
        return False
    return any(
        path.is_file() and path.suffix in {".safetensors", ".bin"} for path in target.rglob("*")
    )


def hits_from_run(run: dict[str, Any]) -> int:
    return int(round(float(run["strict_exact"]) * int(run["key_items"])))


def scoreboard_row(name: str, kind: str, round_name: str, run: dict[str, Any]) -> dict[str, Any]:
    pack = run.get("pack_metrics") or {}
    return {
        "name": name,
        "kind": kind,
        "round": round_name,
        "strict_exact": run["strict_exact"],
        "hits": hits_from_run(run),
        "items": run["key_items"],
        "unanswerable_accuracy": pack.get("unanswerable_accuracy"),
        "counterfactual_pair_both_correct_and_changed": pack.get(
            "counterfactual_pair_both_correct_and_changed"
        ),
        "short_response_rate_le_12_words": run.get("short_response_rate_le_12_words"),
        "coverage": run.get("coverage"),
    }


def render_scoreboard_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R2 cross-depth soup scoreboard",
        "",
        "CPU-only lab. Scaffold is R2 (28L, vocab 32768). Same Phase-2B kept-layer stitch as published Qwen3 cores.",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Protocol: greedy, `{payload['eval']['max_new_tokens']}` new tokens, "
        f"context `{payload['eval']['max_context']}`, seed `{payload['eval']['seed']}`, "
        "thinking disabled, device CPU",
        "- Qwen3 published numbers below are **cross-scaffold references**, not paired scores.",
        "",
        "## Qwen3 published family (reference only)",
        "",
        "| Model | Hits / 64 | Dual (if recorded) |",
        "|---|---:|---:|",
    ]
    refs = payload.get("qwen3_published_reference") or {}
    for key, label in (
        ("qwen3_06b_showcase", "Qwen3-0.6B"),
        ("nimbus_qwen3_x_aegis", "Nimbus (Qwen3 × Aegis α=0.5)"),
        ("lyra", "Lyra"),
        ("halo", "Halo"),
        ("halo_uncensored", "Halo Uncensored"),
        ("late_rineya_pulse_1_6", "late_rineya / Pulse 1.6"),
    ):
        ref = refs.get(key) or {}
        hits = ref.get("hits")
        items = ref.get("items", 64)
        dual = ref.get("dual")
        dual_s = "—" if dual is None else f"{float(dual):.4f}"
        if hits is None:
            continue
        lines.append(f"| {label} | {hits}/{items} | {dual_s} |")
    lines.extend(
        [
            "",
            "## R2 lab runs",
            "",
            "| Name | Round | Kind | Hits / 64 | Strict exact | Unanswerable | Counterfactual-and-changed | ≤12-word rate |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| `{name}` | {round} | {kind} | {hits}/{items} | {strict:.4f} | {unans:.4f} | {cf:.4f} | {short:.4f} |".format(
                name=row["name"],
                round=row["round"],
                kind=row["kind"],
                hits=row["hits"],
                items=row["items"],
                strict=float(row["strict_exact"]),
                unans=float(row["unanswerable_accuracy"] or 0.0),
                cf=float(row["counterfactual_pair_both_correct_and_changed"] or 0.0),
                short=float(row["short_response_rate_le_12_words"] or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def cpu_env() -> dict[str, str]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYTHONUTF8", "1")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch and showcase R2 CrossDepth hybrids on CPU.")
    parser.add_argument("--config", type=Path, default=LAB / "configs" / "hybrids_r2.json")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--skip-showcase", action="store_true")
    parser.add_argument("--stitch-only", action="store_true")
    args = parser.parse_args()

    env = cpu_env()
    config = load_json(args.config)
    ingredients = {
        name: resolve_repo_path(path) for name, path in config.get("ingredients", {}).items()
    }
    scaffold = resolve_repo_path(config["qwen3_model"])
    pulse = resolve_repo_path(config["pulse_base"])
    aegis = resolve_repo_path(config["aegis_model"])
    layer_map = (LAB / config["layer_map"]).resolve()
    eval_cfg = config["eval"]
    prompts = resolve_repo_path(eval_cfg["input"])
    key = resolve_repo_path(eval_cfg["key"])
    expected_items = 64 if args.limit is None else args.limit

    outputs = LAB / "outputs"
    artifacts = LAB / "artifacts"
    donors_root = artifacts / "donors"
    hybrids_root = artifacts / "hybrids"
    outputs.mkdir(parents=True, exist_ok=True)
    donors_root.mkdir(parents=True, exist_ok=True)
    hybrids_root.mkdir(parents=True, exist_ok=True)

    runs = list(config["runs"])
    if args.only:
        wanted = set(args.only)
        missing = wanted - {run["name"] for run in runs}
        if missing:
            raise SystemExit(f"Unknown hybrid names: {sorted(missing)}")
        runs = [run for run in runs if run["name"] in wanted]

    for path, label in (
        (scaffold, "R2 scaffold"),
        (pulse, "Pulse"),
        (aegis, "Aegis"),
        (layer_map, "layer map"),
        (STITCH_SCRIPT, "stitch script"),
    ):
        if path.is_dir():
            if not (path / "config.json").exists():
                raise SystemExit(f"Missing {label}: {path}")
        elif not path.exists():
            raise SystemExit(f"Missing {label}: {path}")

    python = sys.executable
    for spec in runs:
        name = spec["name"]
        kind = spec["kind"]
        pred_path = outputs / "showcase" / f"{name}.jsonl"
        report_json = outputs / "showcase" / f"{name}_report.json"
        report_md = outputs / "showcase" / f"{name}_report.md"
        hybrid_dir = hybrids_root / name
        complete = (not args.force_eval) and ok_prediction_count(pred_path) >= expected_items

        print(f"[{utc_now()}] === {name} ({kind}) ===", flush=True)

        if kind == "vault":
            donor_path = aegis
        elif kind == "soup":
            donor_path = donors_root / spec["donor_name"]
            members = [ingredients[member] for member in spec["members"]]
            for adapter in members:
                if not (adapter / "adapter_config.json").exists():
                    raise SystemExit(f"Missing adapter: {adapter}")
            if not weights_present(donor_path):
                donor_path.mkdir(parents=True, exist_ok=True)
                run_checked(
                    [
                        python,
                        str(MERGE_SCRIPT),
                        "--model-id",
                        str(pulse),
                        "--adapters",
                        *[str(path) for path in members],
                        "--output-dir",
                        str(donor_path),
                        "--overwrite",
                    ],
                    env,
                )
        else:
            raise SystemExit(f"{name}: unknown kind {kind}")

        if not weights_present(hybrid_dir):
            stitch_cmd = [
                python,
                str(STITCH_SCRIPT),
                "--qwen3",
                str(scaffold),
                "--donor",
                str(donor_path),
                "--output-dir",
                str(hybrid_dir),
                "--layer-map",
                str(layer_map),
                "--overwrite",
            ]
            alpha_json = spec.get("alpha_json")
            if alpha_json:
                stitch_cmd.extend(["--alpha-json", str((LAB / str(alpha_json)).resolve())])
            else:
                stitch_cmd.extend(["--alpha", str(spec.get("alpha", 0.5))])
            run_checked(stitch_cmd, env)

        if args.stitch_only:
            print(f"[{utc_now()}] {name}: stitch complete, skip showcase", flush=True)
            continue

        if args.skip_showcase:
            continue

        pred_path.parent.mkdir(parents=True, exist_ok=True)
        if not complete:
            decode_cmd = [
                python,
                str(RUN_SCRIPT),
                "--model",
                str(hybrid_dir),
                "--input",
                str(prompts),
                "--output",
                str(pred_path),
                "--max-new-tokens",
                str(eval_cfg["max_new_tokens"]),
                "--max-context",
                str(eval_cfg["max_context"]),
                "--seed",
                str(eval_cfg["seed"]),
                "--device-map",
                "cpu",
                "--torch-dtype",
                "bfloat16",
                "--resume",
            ]
            if eval_cfg.get("disable_thinking"):
                decode_cmd.append("--disable-thinking")
            if args.limit is not None:
                decode_cmd.extend(["--limit", str(args.limit)])
            run_checked(decode_cmd, env)
        else:
            print(f"[{utc_now()}] {name}: skip decode ({ok_prediction_count(pred_path)} ok rows)", flush=True)

        if ok_prediction_count(pred_path) < expected_items:
            raise SystemExit(
                f"{name}: expected {expected_items} ok predictions, found {ok_prediction_count(pred_path)}"
            )

        run_checked(
            [
                python,
                str(EVAL_SCRIPT),
                "--key",
                str(key),
                "--predictions",
                str(pred_path),
                "--json-report",
                str(report_json),
                "--markdown-report",
                str(report_md),
            ],
            env,
        )

    if args.stitch_only or args.skip_showcase:
        print(f"[{utc_now()}] stitch/materialize complete; scoreboard unchanged", flush=True)
        return

    # Baseline R2 scaffold showcase (no stitch).
    baseline_name = "r2_baseline"
    baseline_pred = outputs / "showcase" / f"{baseline_name}.jsonl"
    baseline_json = outputs / "showcase" / f"{baseline_name}_report.json"
    baseline_md = outputs / "showcase" / f"{baseline_name}_report.md"
    baseline_pred.parent.mkdir(parents=True, exist_ok=True)
    if args.force_eval or ok_prediction_count(baseline_pred) < expected_items:
        print(f"[{utc_now()}] === {baseline_name} (scaffold) ===", flush=True)
        decode_cmd = [
            python,
            str(RUN_SCRIPT),
            "--model",
            str(scaffold),
            "--input",
            str(prompts),
            "--output",
            str(baseline_pred),
            "--max-new-tokens",
            str(eval_cfg["max_new_tokens"]),
            "--max-context",
            str(eval_cfg["max_context"]),
            "--seed",
            str(eval_cfg["seed"]),
            "--device-map",
            "cpu",
            "--torch-dtype",
            "bfloat16",
            "--resume",
        ]
        if eval_cfg.get("disable_thinking"):
            decode_cmd.append("--disable-thinking")
        if args.limit is not None:
            decode_cmd.extend(["--limit", str(args.limit)])
        run_checked(decode_cmd, env)
        run_checked(
            [
                python,
                str(EVAL_SCRIPT),
                "--key",
                str(key),
                "--predictions",
                str(baseline_pred),
                "--json-report",
                str(baseline_json),
                "--markdown-report",
                str(baseline_md),
            ],
            env,
        )

    rows: list[dict[str, Any]] = []
    if baseline_json.is_file():
        run = load_json(baseline_json)["runs"][0]
        rows.append(scoreboard_row(baseline_name, "scaffold", "baseline", run))
    for spec in config["runs"]:
        report_path = outputs / "showcase" / f"{spec['name']}_report.json"
        if not report_path.is_file():
            continue
        run = load_json(report_path)["runs"][0]
        rows.append(scoreboard_row(spec["name"], spec["kind"], spec["round"], run))

    payload = {
        "generated_at": utc_now(),
        "lab": "Rineya_R2_CrossDepth_Soup_Mini_v1",
        "eval": eval_cfg,
        "qwen3_published_reference": config.get("qwen3_published_reference", {}),
        "rows": rows,
        "note": (
            "64-row showcase on R2 tokenizer. Qwen3 family numbers are cross-scaffold "
            "references only. Hybrid weights kept under artifacts/hybrids/."
        ),
    }
    (outputs / "scoreboard.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (outputs / "scoreboard.md").write_text(render_scoreboard_md(payload), encoding="utf-8")
    print(f"[{utc_now()}] Wrote {outputs / 'scoreboard.md'}", flush=True)


if __name__ == "__main__":
    main()
