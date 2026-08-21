#!/usr/bin/env python3
"""Sequential Halo-donor kept-layer α sweep: CPU stitch, one GPU bf16 showcase at a time."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
HYBRIDS = LAB / "configs" / "hybrids.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def cpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONUTF8"] = "1"
    return env


def gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONUTF8"] = "1"
    return env


def run_checked(command: list[str], env: dict[str, str]) -> None:
    print(f"[{utc_now()}] $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=str(REPO), env=env)


def competing_python_gpu() -> list[str]:
    """Return nvidia-smi rows for other Python/CUDA jobs. Ignore desktop C+G overlays."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []
    ours = str(os.getpid())
    hits: list[str] = []
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        lower = text.lower()
        if "python" not in lower and "pytorch" not in lower:
            continue
        pid = text.split(",", 1)[0].strip()
        if pid == ours:
            continue
        hits.append(text)
    return hits


def wait_for_gpu(poll_seconds: int = 60) -> None:
    while True:
        hits = competing_python_gpu()
        if not hits:
            print(f"[{utc_now()}] GPU clear of other Python jobs", flush=True)
            return
        print(
            f"[{utc_now()}] waiting {poll_seconds}s for GPU; other Python jobs:\n"
            + "\n".join(hits),
            flush=True,
        )
        time.sleep(poll_seconds)


def hits_from_run(run: dict[str, Any]) -> int:
    return int(round(float(run["strict_exact"]) * int(run["key_items"])))


def row_from_run(name: str, family: str, mean_alpha: float, run: dict[str, Any], source: str) -> dict[str, Any]:
    pack = run.get("pack_metrics") or {}
    return {
        "name": name,
        "family": family,
        "mean_alpha": mean_alpha,
        "source": source,
        "strict_exact": run["strict_exact"],
        "hits": hits_from_run(run),
        "items": run["key_items"],
        "unanswerable_accuracy": pack.get("unanswerable_accuracy"),
        "counterfactual_pair_both_correct_and_changed": pack.get(
            "counterfactual_pair_both_correct_and_changed"
        ),
        "short_response_rate_le_12_words": run.get("short_response_rate_le_12_words"),
        "coverage": run.get("coverage"),
        "kept_weights": False,
    }


def render_scoreboard_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Halo-donor kept-layer α recipes",
        "",
        "Curiosity-only. Same 19 kept layers / 9 dropped Qwen3 layers as Halo. "
        "Donor is Halo’s v1@5e-4 + v2@1e-4 soup. Frozen 64-example showcase.",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Protocol: greedy, `{payload['eval']['max_new_tokens']}` new tokens, "
        f"context `{payload['eval']['max_context']}`, seed `{payload['eval']['seed']}`, "
        "thinking off, bf16",
        f"- Halo reference: `{payload['halo_hits']}/{payload['halo_items']}`",
        "- Keep weights if hits ≥ Halo. ±1 hit is noise; +2 hits is discuss-only, not a vault core.",
        "",
        "| Name | Family | Mean α | Hits / 64 | Strict exact | Unanswerable | CF pair | Source |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| `{name}` | {family} | {mean:.4f} | {hits}/{items} | {strict:.4f} | {unans:.4f} | {cf:.4f} | {source} |".format(
                name=row["name"],
                family=row["family"],
                mean=float(row["mean_alpha"]),
                hits=row["hits"],
                items=row["items"],
                strict=float(row["strict_exact"]),
                unans=float(row["unanswerable_accuracy"] or 0.0),
                cf=float(row["counterfactual_pair_both_correct_and_changed"] or 0.0),
                source=row["source"],
            )
        )
    best = payload.get("best")
    if best:
        lines.extend(["", f"**Best so far:** `{best['name']}` at {best['hits']}/{best['items']}."])
    lines.append("")
    return "\n".join(lines)


def write_scoreboard(out_dir: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            int(row["hits"]),
            float(row["counterfactual_pair_both_correct_and_changed"] or 0.0),
            float(row["unanswerable_accuracy"] or 0.0),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    payload = {
        "generated_at": utc_now(),
        "lab": "Rineya_Qwen3_CrossDepth_Soup_Mini_v1/alpha_recipes",
        "eval": config["eval"],
        "halo_hits": config["halo_hits"],
        "halo_items": config["halo_items"],
        "keep_weights_min_hits": config["keep_weights_min_hits"],
        "rows": rows,
        "best": {"name": best["name"], "hits": best["hits"], "items": best["items"]} if best else None,
        "note": "Not a recorded core. 64 examples is a noisy selection signal.",
    }
    write_json(out_dir / "scoreboard.json", payload)
    (out_dir / "scoreboard.md").write_text(render_scoreboard_md(payload), encoding="utf-8")
    print(f"[{utc_now()}] Wrote {out_dir / 'scoreboard.md'}", flush=True)


def materialize_donor(
    python: str,
    hybrids: dict[str, Any],
    donor_name: str,
    members: list[str],
    donor_path: Path,
) -> None:
    ingredients = {name: resolve_repo_path(path) for name, path in hybrids["ingredients"].items()}
    pulse = resolve_repo_path(hybrids["pulse_base"])
    adapters = [ingredients[member] for member in members]
    for adapter in adapters:
        if not (adapter / "adapter_config.json").exists():
            raise SystemExit(f"Missing adapter: {adapter}")
    if weights_present(donor_path):
        print(f"[{utc_now()}] donor {donor_name}: reuse {donor_path}", flush=True)
        return
    donor_path.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            python,
            str(MERGE_SCRIPT),
            "--model-id",
            str(pulse),
            "--adapters",
            *[str(path) for path in adapters],
            "--output-dir",
            str(donor_path),
            "--overwrite",
        ],
        cpu_env(),
    )


def copy_reuse_report(src: Path, dest: Path, run_name: str) -> dict[str, Any]:
    report = load_json(src)
    write_json(dest, report)
    print(f"[{utc_now()}] {run_name}: reused {src}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Halo-donor kept-layer α recipes.")
    parser.add_argument("--config", type=Path, default=LAB / "configs" / "alpha_recipes.json")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    config = load_json(args.config)
    hybrids = load_json(HYBRIDS)
    recipes = list(config["recipes"])
    if args.only:
        wanted = set(args.only)
        missing = wanted - {recipe["name"] for recipe in recipes}
        if missing:
            raise SystemExit(f"Unknown recipe names: {sorted(missing)}")
        recipes = [recipe for recipe in recipes if recipe["name"] in wanted]

    qwen3 = resolve_repo_path(config["qwen3_model"])
    layer_map = (LAB / config["layer_map"]).resolve()
    eval_cfg = config["eval"]
    prompts = resolve_repo_path(eval_cfg["input"])
    key = resolve_repo_path(eval_cfg["key"])
    expected = 64
    min_keep = int(config["keep_weights_min_hits"])

    out_dir = LAB / "outputs" / "alpha_recipes"
    artifacts = LAB / "artifacts" / "alpha_recipes"
    donors_root = LAB / "artifacts" / "donors"
    alpha_tmp = LAB / "artifacts" / "alpha_recipes" / "_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    donors_root.mkdir(parents=True, exist_ok=True)
    alpha_tmp.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    donor_path = donors_root / str(config["donor"])
    materialize_donor(python, hybrids, str(config["donor"]), list(config["members"]), donor_path)

    rows: list[dict[str, Any]] = []
    scoreboard_path = out_dir / "scoreboard.json"
    if scoreboard_path.is_file() and args.only:
        rows = list(load_json(scoreboard_path).get("rows") or [])

    def upsert(row: dict[str, Any]) -> None:
        updated = [item for item in rows if item["name"] != row["name"]]
        updated.append(row)
        rows.clear()
        rows.extend(updated)
        write_scoreboard(out_dir, config, rows)

    for spec in recipes:
        name = spec["name"]
        family = spec["family"]
        mean_alpha = float(spec["mean_alpha"])
        pred_path = out_dir / f"{name}.jsonl"
        report_json = out_dir / f"{name}_report.json"
        report_md = out_dir / f"{name}_report.md"
        hybrid_dir = artifacts / name
        reuse = spec.get("reuse_report")
        print(f"[{utc_now()}] === {name} ({family}) ===", flush=True)

        if reuse:
            src = LAB / reuse if not Path(reuse).is_absolute() else Path(reuse)
            if not src.is_file():
                raise SystemExit(f"{name}: missing reuse report {src}")
            report = copy_reuse_report(src, report_json, name)
            row = row_from_run(name, family, mean_alpha, report["runs"][0], source=str(reuse))
            row["kept_weights"] = False
            upsert(row)
            continue

        complete = (not args.force_eval) and ok_prediction_count(pred_path) >= expected
        vector_path = alpha_tmp / f"{name}.json"
        write_json(
            vector_path,
            {
                "name": name,
                "mean_alpha": mean_alpha,
                "rineya_alphas": spec["rineya_alphas"],
            },
        )

        if not complete and not weights_present(hybrid_dir):
            if hybrid_dir.exists():
                for leftover in hybrid_dir.glob(".tmp*"):
                    try:
                        leftover.unlink()
                    except OSError as exc:
                        print(
                            f"[{utc_now()}] {name}: leftover {leftover.name} still locked; {exc}",
                            flush=True,
                        )
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
                    "--alpha-json",
                    str(vector_path),
                    "--overwrite",
                ],
                cpu_env(),
            )

        if not complete:
            if not args.skip_gpu_check:
                wait_for_gpu()
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
                "--torch-dtype",
                str(eval_cfg.get("torch_dtype", "bfloat16")),
                "--resume",
            ]
            if eval_cfg.get("disable_thinking"):
                decode_cmd.append("--disable-thinking")
            run_checked(decode_cmd, gpu_env())
        else:
            print(
                f"[{utc_now()}] {name}: skip decode ({ok_prediction_count(pred_path)} ok rows)",
                flush=True,
            )

        if ok_prediction_count(pred_path) < expected:
            raise SystemExit(
                f"{name}: expected {expected} ok predictions, found {ok_prediction_count(pred_path)}"
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
            cpu_env(),
        )
        report = load_json(report_json)
        row = row_from_run(name, family, mean_alpha, report["runs"][0], source="this_sweep")
        keep = args.keep_weights or int(row["hits"]) >= min_keep
        row["kept_weights"] = keep
        if keep:
            print(f"[{utc_now()}] {name}: keeping hybrid weights ({row['hits']} hits)", flush=True)
        else:
            removed = drop_weight_files(hybrid_dir)
            print(f"[{utc_now()}] {name}: dropped hybrid weights {removed}", flush=True)
        upsert(row)

    drop_weight_files(donor_path)
    print(f"[{utc_now()}] dropped rematerialized 19L donor weights at {donor_path}", flush=True)
    write_scoreboard(out_dir, config, rows)


if __name__ == "__main__":
    main()
