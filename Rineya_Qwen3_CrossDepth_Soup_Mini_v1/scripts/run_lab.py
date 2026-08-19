#!/usr/bin/env python3
"""Rematerialize 19L winners, stitch 28L Qwen3 hybrids, score the frozen 64-row showcase."""

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
STITCH_SCRIPT = LAB / "scripts" / "stitch_cross_depth.py"
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
        "# Qwen3 cross-depth soup scoreboard",
        "",
        "Curiosity-only. Frozen 64-row showcase. 28-layer Qwen3 geometry. Not a recorded core.",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Protocol: greedy, `{payload['eval']['max_new_tokens']}` new tokens, "
        f"context `{payload['eval']['max_context']}`, seed `{payload['eval']['seed']}`, "
        "thinking disabled",
        "",
        "## References",
        "",
        "| Model | Hits / 64 | Strict exact | Unanswerable | Counterfactual-and-changed | ≤12-word rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, ref in payload["references"].items():
        lines.append(
            "| {name} | {hits}/{items} | {strict:.4f} | {unans:.4f} | {cf:.4f} | {short:.4f} |".format(
                name=name,
                hits=ref["hits"],
                items=ref["items"],
                strict=ref["strict_exact"],
                unans=ref["unanswerable_accuracy"],
                cf=ref["counterfactual_pair_both_correct_and_changed"],
                short=ref["short_response_rate_le_12_words"],
            )
        )
    lines.extend(
        [
            "",
            "## Lab runs",
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
                strict=row["strict_exact"],
                unans=float(row["unanswerable_accuracy"] or 0.0),
                cf=float(row["counterfactual_pair_both_correct_and_changed"] or 0.0),
                short=float(row["short_response_rate_le_12_words"] or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen3 cross-depth soup mini-lab.")
    parser.add_argument("--config", type=Path, default=LAB / "configs" / "hybrids.json")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Rematerialize hybrid weights without showcase decode. Implies --keep-weights.",
    )
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Decode dtype. bfloat16 is safer for 28L fp32 checkpoints on 8GB GPUs.",
    )
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("PYTHONUTF8", "1")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "-1")
    env["PYTHONUTF8"] = "1"

    config = load_json(args.config)
    ingredients = {
        name: resolve_repo_path(path) for name, path in config.get("ingredients", {}).items()
    }
    qwen3 = resolve_repo_path(config["qwen3_model"])
    pulse = resolve_repo_path(config["pulse_base"])
    aegis = resolve_repo_path(config["aegis_model"])
    layer_map = resolve_repo_path(config["layer_map"]) if Path(config["layer_map"]).is_absolute() else (LAB / config["layer_map"]).resolve()
    eval_cfg = config["eval"]
    prompts = resolve_repo_path(eval_cfg["input"])
    key = resolve_repo_path(eval_cfg["key"])
    expected_items = 64 if args.limit is None else args.limit

    outputs = LAB / "outputs"
    artifacts = LAB / "artifacts"
    donors_root = artifacts / "donors"
    outputs.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    donors_root.mkdir(parents=True, exist_ok=True)

    runs = list(config["runs"])
    if args.only:
        wanted = set(args.only)
        missing = wanted - {run["name"] for run in runs}
        if missing:
            raise SystemExit(f"Unknown hybrid names: {sorted(missing)}")
        runs = [run for run in runs if run["name"] in wanted]

    for path, label in ((qwen3, "Qwen3-0.6B"), (pulse, "Pulse"), (aegis, "Aegis"), (layer_map, "layer map")):
        if path.is_dir():
            if not (path / "config.json").exists():
                raise SystemExit(f"Missing {label}: {path}")
        elif not path.exists():
            raise SystemExit(f"Missing {label}: {path}")

    python = sys.executable
    for spec in runs:
        name = spec["name"]
        kind = spec["kind"]
        pred_path = outputs / f"{name}.jsonl"
        report_json = outputs / f"{name}_report.json"
        report_md = outputs / f"{name}_report.md"
        hybrid_dir = artifacts / name
        complete = (not args.force_eval) and ok_prediction_count(pred_path) >= expected_items

        print(f"[{utc_now()}] === {name} ({kind}) ===", flush=True)

        donor_path: Path
        donor_is_temp = False
        need_hybrid = (args.materialize_only or not complete) and not weights_present(hybrid_dir)
        if kind == "vault":
            donor_path = aegis
        elif kind == "soup":
            donor_path = donors_root / spec["donor_name"]
            donor_is_temp = True
            members = [ingredients[member] for member in spec["members"]]
            for adapter in members:
                if not (adapter / "adapter_config.json").exists():
                    raise SystemExit(f"Missing adapter: {adapter}")
            if need_hybrid and not weights_present(donor_path):
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

        if need_hybrid:
            run_checked(
                [
                    python,
                    str(STITCH_SCRIPT),
                    "--qwen3",
                    str(qwen3),
                    "--donor",
                    str(donor_path),
                    "--output-dir",
                    str(hybrid_dir),
                    "--layer-map",
                    str(layer_map),
                    "--alpha",
                    str(spec.get("alpha", 0.5)),
                    "--overwrite",
                ],
                env,
            )

        if args.materialize_only:
            if not weights_present(hybrid_dir):
                raise SystemExit(f"{name}: materialize-only produced no weights at {hybrid_dir}")
            if donor_is_temp:
                removed_donor = drop_weight_files(donor_path)
                print(f"[{utc_now()}] {name}: dropped 19L donor weights {removed_donor}", flush=True)
            print(f"[{utc_now()}] {name}: materialized, skip showcase decode", flush=True)
            continue

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
                args.device_map,
                "--resume",
            ]
            if eval_cfg.get("disable_thinking"):
                decode_cmd.append("--disable-thinking")
            if args.torch_dtype != "auto":
                decode_cmd.extend(["--torch-dtype", args.torch_dtype])
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

        if not args.keep_weights:
            removed_hybrid = drop_weight_files(hybrid_dir)
            print(f"[{utc_now()}] {name}: dropped hybrid weights {removed_hybrid}", flush=True)
            if donor_is_temp:
                removed_donor = drop_weight_files(donor_path)
                print(f"[{utc_now()}] {name}: dropped 19L donor weights {removed_donor}", flush=True)

    if args.materialize_only:
        print(f"[{utc_now()}] materialize-only complete; scoreboard unchanged", flush=True)
        return

    rows: list[dict[str, Any]] = []
    for spec in runs:
        report = load_json(outputs / f"{spec['name']}_report.json")
        run = report["runs"][0]
        rows.append(scoreboard_row(spec["name"], spec["kind"], spec["round"], run))

    scoreboard_path = outputs / "scoreboard.json"
    if scoreboard_path.is_file() and args.only:
        old_rows = load_json(scoreboard_path).get("rows") or []
        updated = {row["name"]: row for row in rows}
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in old_rows:
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
        "lab": "Rineya_Qwen3_CrossDepth_Soup_Mini_v1",
        "eval": eval_cfg,
        "references": config.get("references", {}),
        "rows": rows,
        "note": "64-row showcase is a noisy selection signal. Curiosity only; not a recorded core.",
    }
    (outputs / "scoreboard.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (outputs / "scoreboard.md").write_text(render_scoreboard_md(payload), encoding="utf-8")
    print(f"[{utc_now()}] Wrote {outputs / 'scoreboard.md'}", flush=True)


if __name__ == "__main__":
    main()
