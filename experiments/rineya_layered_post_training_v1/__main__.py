"""Run from repository root: python -m experiments.rineya_layered_post_training_v1."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .config import ARMS, Config, require
from .data import fixtures, load_rows
from .engine import confirm, run
from .runtime import tiny_runtime
from .storage import native_preflight, native_runtime


def main():
    parser = argparse.ArgumentParser(description="RLP-001: explicit post-training and retention-constrained soups")
    parser.add_argument("command", choices=("plan", "smoke", "preflight", "run", "confirm", "compare"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="confirmation only: use the tiny fixture model")
    parser.add_argument("--reports", type=Path, nargs="+")
    args = parser.parse_args()
    cfg = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    if args.arm:
        cfg = replace(cfg, arm=args.arm)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.command == "plan":
        result = {"config": asdict(cfg), "config_hash": cfg.fingerprint, "schedule": cfg.schedule(), "note": "Equal actor steps are NOT equal FLOPs; compare actual rollout/forward-token/critic/evaluation costs."}
    elif args.command == "smoke":
        import torch
        torch.set_num_threads(1)
        cfg = replace(cfg, arm=args.arm or "full", actor_steps=28, critic_steps=6, learning_rate=.005, critic_lr=.01, max_prompt_tokens=8, max_new_tokens=2, max_context_tokens=16, min_eval_groups=2, required_suites=("acquisition", "retention"), bootstrap_samples=400)
        require(args.output is not None, "--output required")
        result = run(tiny_runtime(cfg), fixtures(), args.output)
    elif args.command == "preflight":
        require(args.model_dir is not None and args.output is not None, "--model-dir and existing --output directory required")
        result = native_preflight(args.model_dir, cfg, args.output)
    elif args.command in {"run", "confirm"}:
        require(args.output is not None, "--output required")
        if args.command == "confirm":
            cfg = Config.from_dict(json.loads((args.output / "config.json").read_text()))
        if args.synthetic:
            require(args.command == "confirm", "--synthetic is only for confirming a smoke run")
            import torch
            torch.set_num_threads(1)
            runtime, rows = tiny_runtime(cfg), fixtures()
        else:
            require(args.data is not None and args.model_dir is not None, "--data and --model-dir required")
            require(args.command != "run" or args.allow_training, "explicit --allow-training required")
            runtime = native_runtime(args.model_dir, cfg, args.output.parent, allow_model_load=args.allow_model_load)
            rows = load_rows(args.data)
        result = run(runtime, rows, args.output) if args.command == "run" else confirm(runtime, rows, args.output)
    else:
        require(args.reports and len(args.reports) >= 2, "provide >=2 run_report.json files")
        runs = [json.loads(p.read_text()) for p in args.reports]
        manifests = [json.loads((p.parent / "data_manifest.json").read_text()) for p in args.reports]
        configs = [Config.from_dict(json.loads((p.parent / "config.json").read_text())) for p in args.reports]
        require(len({json.dumps(m, sort_keys=True) for m in manifests}) == 1, "comparison datasets differ")
        protected = ("seed", "max_prompt_tokens", "max_new_tokens", "max_context_tokens", "required_suites", "train_last_layers", "trainable_fraction_cap")
        require(all(all(getattr(c, k) == getattr(configs[0], k) for k in protected) for c in configs), "unmatched seed/inference/capacity settings")
        keys = ("processed_tokens", "generated_tokens", "actor_steps", "critic_steps")
        ranges = {k: [min(r["budget"][k] for r in runs), max(r["budget"][k] for r in runs)] for k in keys}
        matched = all(hi == 0 or lo / hi >= .9 for lo, hi in ranges.values())
        result = {"status": "TOKEN_PROXY_MATCHED_WITHIN_10_PERCENT_NOT_FLOP_CERTIFIED" if matched else "NOT_COMPUTE_MATCHED", "cost_ranges": ranges, "arms": [{k: r[k] for k in ("arm", "selected_checkpoint", "selected_scores", "terminal_scores", "status")} for r in runs], "note": "Descriptive comparison only. Native gains require locked confirmation and at least three independently trained seeds."}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
