#!/usr/bin/env python3
"""P0/P1/P2 driver for the Supa-A1.1 FWA lab. One model on GPU at a time."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from rineya_fwa.arms import apply_arm, count_fwa_tensors, count_trainable  # noqa: E402
from rineya_fwa.data import build_lab_mix  # noqa: E402
from rineya_fwa.eval_interference import score_interference  # noqa: E402
from rineya_fwa.eval_position import build_position_suite, score_suite  # noqa: E402
from rineya_fwa.loading import freeze_backbone, load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS, CONFIGS, DATA, REPORTS  # noqa: E402
from rineya_fwa.profile import profile_forward  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402
from rineya_fwa.train import load_adapter, train_arm  # noqa: E402


def _load_yaml(name: str) -> dict:
    path = CONFIGS / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_p1() -> dict:
    cfg = _load_yaml("pilot_falcon2_r16.yaml")
    mix = build_lab_mix(
        train_rows=int(cfg["data"]["train_rows"]),
        dev_rows=int(cfg["data"]["dev_rows"]),
        strata=list(cfg["data"]["strata"]),
        seed=int(cfg["training"]["seed"]),
        dest=DATA / "unsealed",
    )
    out = ARTIFACTS / "p1_pilot" / "arm_C" / "seed_11"
    seq = int(cfg["training"]["max_sequence_length"])
    steps = int(cfg["training"]["max_steps"])
    try:
        man = train_arm(
            arm="C",
            layers=list(cfg["fwa"]["layers"]),
            train_file=DATA / "unsealed" / "train.jsonl",
            output_dir=out,
            seed=int(cfg["training"]["seed"]),
            rank=int(cfg["fwa"]["rank"]),
            max_steps=steps,
            max_sequence_length=seq,
            learning_rate=float(cfg["training"]["learning_rate"]),
            grad_accum=int(cfg["training"]["gradient_accumulation_steps"]),
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        man = train_arm(
            arm="C",
            layers=list(cfg["fwa"]["layers"]),
            train_file=DATA / "unsealed" / "train.jsonl",
            output_dir=out,
            seed=int(cfg["training"]["seed"]),
            rank=int(cfg["fwa"]["rank"]),
            max_steps=max(16, steps // 2),
            max_sequence_length=max(256, seq // 2),
            learning_rate=float(cfg["training"]["learning_rate"]),
            grad_accum=int(cfg["training"]["gradient_accumulation_steps"]),
        )
        man["oom_fallback"] = True
    payload = {"mix": mix, "training": man}
    write_json(ARTIFACTS / "p1_pilot" / "summary.json", payload, config=cfg)
    return payload


def _eval_arm(arm: str, layers: list[int], rank: int, adapter: Path | None, lengths: list[int]) -> dict:
    import torch

    device = _device()
    tokenizer = load_tokenizer()
    model = load_supa(device=device, eval_mode=True)
    freeze_backbone(model)
    handle = apply_arm(model, arm, layers=layers, rank=rank, init_alpha=0.0)
    if handle is not None and adapter is not None and adapter.is_file():
        load_adapter(handle, adapter)
    model.eval()
    items = build_position_suite(lengths=lengths, n_per_cell=1, seed=11)
    try:
        pos = score_suite(model, tokenizer, items)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        items = build_position_suite(lengths=[256], n_per_cell=1, seed=11)
        pos = score_suite(model, tokenizer, items)
        pos["oom_fallback"] = True
    inter = score_interference(model, tokenizer)
    resources = []
    for seq in [256, 512, 1024]:
        try:
            resources.append(profile_forward(model, tokenizer, handle, seq_len=seq, device=device))
        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
    result = {
        "arm": arm,
        "trainable_parameters": count_trainable(handle),
        "fwa_parameters": count_fwa_tensors(handle),
        "position": {k: v for k, v in pos.items() if k != "rows"},
        "position_rows": pos.get("rows"),
        "interference": {k: v for k, v in inter.items() if k != "rows"},
        "interference_rows": inter.get("rows"),
        "resources": resources,
    }
    if handle is not None:
        handle.uninstall()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_p2() -> dict:
    cfg = _load_yaml("arms_abcd_p2.yaml")
    mix = build_lab_mix(
        train_rows=int(cfg["data"]["train_rows"]),
        dev_rows=int(cfg["data"]["dev_rows"]),
        strata=list(cfg["data"]["strata"]),
        seed=11,
        dest=DATA / "unsealed",
    )
    layers = list(cfg["fwa"]["layers"])
    rank = int(cfg["fwa"]["rank"])
    seeds = list(cfg["training"]["seeds"])
    lengths = [256, 512]
    train_summaries = []
    evals = []
    for seed in seeds:
        for arm in ("B", "C", "D"):
            out = ARTIFACTS / "p2_abcd" / f"arm_{arm}" / f"seed_{seed}"
            try:
                man = train_arm(
                    arm=arm,
                    layers=layers,
                    train_file=DATA / "unsealed" / "train.jsonl",
                    output_dir=out,
                    seed=seed,
                    rank=rank,
                    max_steps=int(cfg["training"]["max_steps"]),
                    max_sequence_length=int(cfg["training"]["max_sequence_length"]),
                    learning_rate=float(cfg["training"]["learning_rate"]),
                    grad_accum=int(cfg["training"]["gradient_accumulation_steps"]),
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                man = train_arm(
                    arm=arm,
                    layers=layers,
                    train_file=DATA / "unsealed" / "train.jsonl",
                    output_dir=out,
                    seed=seed,
                    rank=rank,
                    max_steps=max(16, int(cfg["training"]["max_steps"]) // 2),
                    max_sequence_length=512,
                    learning_rate=float(cfg["training"]["learning_rate"]),
                    grad_accum=int(cfg["training"]["gradient_accumulation_steps"]),
                )
                man["oom_fallback"] = True
            train_summaries.append(man)
            adapter = Path(man["adapter_path"]) if man.get("adapter_path") else None
            ev = _eval_arm(arm, layers, rank, adapter, lengths)
            ev["seed"] = seed
            evals.append(ev)

    # Arm A once (eval-only)
    ev_a = _eval_arm("A", layers, rank, None, lengths)
    ev_a["seed"] = None
    evals.append(ev_a)

    payload = {"mix": mix, "training": train_summaries, "eval": evals}
    write_json(ARTIFACTS / "p2_abcd" / "summary.json", payload, config=cfg)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("p1", "p2", "all"), default="all")
    args = p.parse_args()
    try:
        if args.phase in ("p1", "all"):
            print(json.dumps({"p1": run_p1()}, default=str)[:2000])
        if args.phase in ("p2", "all"):
            print(json.dumps({"p2_keys": list(run_p2().keys())}))
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
