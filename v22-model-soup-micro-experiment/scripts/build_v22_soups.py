from __future__ import annotations

"""Build the three micro-experiment soups on CPU from the sealed v2.2 run."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = (ROOT / "../short-response-model-phase-0-1-2").resolve()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(v22: Path, rel: str) -> Path:
    path = v22 / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def build_anchor_greedy(cfg: dict[str, Any], v22: Path, out_root: Path, force: bool) -> Path:
    dest = out_root / "soups" / "anchor_greedy_uniform"
    if dest.exists() and not force:
        print(f"reuse {dest}")
        return dest
    models = [
        resolve(v22, cfg["ingredients"]["frozen_anchor"]),
        resolve(v22, cfg["ingredients"]["greedy_soup"]),
    ]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "average_checkpoints_cpu.py"),
        "--output-dir",
        str(dest),
        "--force",
    ]
    for model in models:
        cmd.extend(["--model", str(model)])
    run(cmd)
    return dest


def build_specialists_lora(cfg: dict[str, Any], v22: Path, out_root: Path, force: bool) -> Path:
    dest = out_root / "soups" / "specialists_lora_uniform"
    if dest.exists() and not force:
        print(f"reuse {dest}")
        return dest
    adapters = [resolve(v22, rel) for rel in cfg["ingredients"]["specialist_adapters"]]
    weight = 1.0 / len(adapters)
    ingredients = [
        {"name": path.parent.name if path.name == "adapter" else path.name, "adapter": str(path), "weight": weight}
        for path in adapters
    ]
    spec = {
        "schema_version": "2.2",
        "kind": "micro_experiment_uniform_specialist_lora_soup",
        "ingredients": ingredients,
        "formula": "W = W_anchor + (1/N) * sum Delta_i",
    }
    spec_path = out_root / "specs" / "specialists_lora_uniform.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    base = resolve(v22, cfg["ingredients"]["frozen_anchor"])
    # Reuse official exact LoRA composer; keep CUDA invisible so merge stays on CPU.
    env_python = sys.executable
    cmd = [
        env_python,
        str(OFFICIAL / "scripts" / "phase_9_student_distillation" / "compose_lora_adapters_v22.py"),
        "--base-model",
        str(base),
        "--spec",
        str(spec_path),
        "--output-dir",
        str(dest),
        "--force",
    ]
    import os

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(OFFICIAL)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(OFFICIAL), env=env)
    return dest


def materialize(base: Path, adapter: Path, dest: Path, force: bool) -> Path:
    if dest.exists() and not force:
        return dest
    run(
        [
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
    )
    return dest


def build_kitchen_sink(cfg: dict[str, Any], v22: Path, out_root: Path, force: bool) -> Path:
    dest = out_root / "soups" / "kitchen_sink_full_uniform"
    if dest.exists() and not force:
        print(f"reuse {dest}")
        return dest
    base = resolve(v22, cfg["ingredients"]["frozen_anchor"])
    greedy = resolve(v22, cfg["ingredients"]["greedy_soup"])
    mat_root = out_root / "materialized"
    mat_root.mkdir(parents=True, exist_ok=True)
    ingredients = [base, greedy]
    for rel in cfg["ingredients"]["specialist_adapters"]:
        adapter = resolve(v22, rel)
        name = adapter.parent.name if adapter.name == "adapter" else adapter.name
        ingredients.append(materialize(base, adapter, mat_root / name, force=force))
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "average_checkpoints_cpu.py"),
        "--output-dir",
        str(dest),
        "--force",
    ]
    for model in ingredients:
        cmd.extend(["--model", str(model)])
    run(cmd)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default_soups.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-kitchen-sink",
        action="store_true",
        help="Skip the larger full-checkpoint kitchen-sink soup (still builds the first two).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    v22 = (ROOT / cfg["v22_run_root"]).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Force CPU for this whole builder process tree.
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    built = {
        "anchor_greedy_uniform": str(build_anchor_greedy(cfg, v22, args.output_root, args.force)),
        "specialists_lora_uniform": str(build_specialists_lora(cfg, v22, args.output_root, args.force)),
    }
    if not args.skip_kitchen_sink:
        built["kitchen_sink_full_uniform"] = str(build_kitchen_sink(cfg, v22, args.output_root, args.force))
    summary = {
        "config": str(args.config),
        "v22_run_root": str(v22),
        "soups": built,
        "note": "CPU-only construction; does not modify official Phase artifacts.",
    }
    out = args.output_root / "SOUP_BUILD_SUMMARY.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
