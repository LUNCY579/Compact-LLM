#!/usr/bin/env python3
"""Validate teacher dual-SFT adapters, merge designed soups, score the frozen 64-row showcase."""

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
VALIDATE_SCRIPT = PHASE / "scripts" / "phase_4_model_soups" / "validate_ingredients.py"
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


def drop_soup_weights(soup_dir: Path) -> list[str]:
    removed: list[str] = []
    if not soup_dir.exists():
        return removed
    for path in soup_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".safetensors", ".bin"} or path.name == "tokenizer.json":
            path.unlink()
            removed.append(str(path.relative_to(soup_dir)))
    return removed


def soup_weights_present(soup_dir: Path) -> bool:
    if not soup_dir.exists():
        return False
    return any(
        path.is_file() and (path.suffix in {".safetensors", ".bin"})
        for path in soup_dir.rglob("*")
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
        "# Teacher soup mini-lab scoreboard",
        "",
        "Curiosity-only. Frozen 64-row showcase. Not a recorded core.",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Protocol: greedy, `{payload['eval']['max_new_tokens']}` new tokens, "
        f"context `{payload['eval']['max_context']}`, seed `{payload['eval']['seed']}`",
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
    parser = argparse.ArgumentParser(description="Run the teacher dual-SFT soup mini-lab.")
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB / "configs" / "combinations.json",
    )
    parser.add_argument("--only", nargs="+", help="Run only these combination names")
    parser.add_argument("--limit", type=int, help="Decode only the first N showcase rows")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--device-map", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("PYTHONUTF8", "1")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "-1")
    env["PYTHONUTF8"] = "1"

    config = load_json(args.config)
    ingredients = {
        name: resolve_repo_path(path) for name, path in config["ingredients"].items()
    }
    pulse = resolve_repo_path(config["base_model"])
    eval_cfg = config["eval"]
    prompts = resolve_repo_path(eval_cfg["input"])
    key = resolve_repo_path(eval_cfg["key"])
    expected_items = 64 if args.limit is None else args.limit

    outputs = LAB / "outputs"
    artifacts = LAB / "artifacts"
    outputs.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    runs = list(config["runs"])
    if args.only:
        wanted = set(args.only)
        missing = wanted - {run["name"] for run in runs}
        if missing:
            raise SystemExit(f"Unknown combination names: {sorted(missing)}")
        runs = [run for run in runs if run["name"] in wanted]

    unique_adapters = sorted({ingredients[name] for run in runs for name in run["members"]})
    for adapter in unique_adapters:
        if not (adapter / "adapter_config.json").exists():
            raise SystemExit(f"Missing adapter: {adapter}")
    if not (pulse / "config.json").exists():
        raise SystemExit(f"Missing Pulse base: {pulse}")

    print(f"[{utc_now()}] Validating {len(unique_adapters)} adapters against Pulse", flush=True)
    run_checked(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            *[str(path) for path in unique_adapters],
            "--out",
            str(outputs / "compatibility.json"),
        ],
        env,
    )

    python = sys.executable
    for spec in runs:
        name = spec["name"]
        kind = spec["kind"]
        members = [ingredients[member] for member in spec["members"]]
        pred_path = outputs / f"{name}.jsonl"
        report_json = outputs / f"{name}_report.json"
        report_md = outputs / f"{name}_report.md"
        soup_dir = artifacts / name
        complete = (not args.force_eval) and ok_prediction_count(pred_path) >= expected_items

        print(f"[{utc_now()}] === {name} ({kind}) ===", flush=True)

        model_path = pulse
        adapter_path: Path | None = None
        if kind == "peft":
            if len(members) != 1:
                raise SystemExit(f"{name}: peft runs must have exactly one member")
            adapter_path = members[0]
        elif kind == "soup":
            model_path = soup_dir
            need_merge = not complete and (
                not (soup_dir / "soup_manifest.json").exists() or not soup_weights_present(soup_dir)
            )
            if need_merge:
                soup_dir.mkdir(parents=True, exist_ok=True)
                merge_cmd = [
                    python,
                    str(MERGE_SCRIPT),
                    "--model-id",
                    str(pulse),
                    "--adapters",
                    *[str(path) for path in members],
                    "--output-dir",
                    str(soup_dir),
                    "--overwrite",
                ]
                run_checked(merge_cmd, env)
            elif complete:
                print(f"[{utc_now()}] {name}: predictions complete; skip merge", flush=True)
            else:
                print(f"[{utc_now()}] {name}: reusing existing soup weights", flush=True)
        else:
            raise SystemExit(f"{name}: unknown kind {kind}")

        if not complete:
            decode_cmd = [
                python,
                str(RUN_SCRIPT),
                "--model",
                str(model_path),
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
            if adapter_path is not None:
                decode_cmd.extend(["--adapter", str(adapter_path)])
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

        if kind == "soup" and not args.keep_weights:
            removed = drop_soup_weights(soup_dir)
            print(
                f"[{utc_now()}] {name}: dropped soup weights {removed}; kept soup_manifest.json",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    for spec in runs:
        report = load_json(outputs / f"{spec['name']}_report.json")
        run = report["runs"][0]
        rows.append(scoreboard_row(spec["name"], spec["kind"], spec["round"], run))

    payload = {
        "generated_at": utc_now(),
        "lab": "Rineya_Teacher_Soup_Mini_v1",
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
