#!/usr/bin/env python3
"""Preflight or build the frozen 1.0625 adapter-dose transfer trial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from rineya_tgsa.historical_dose import build_dose_trial, load_dose_registration, preflight_dose


def resolve_defaults(config: Path):
    value = load_dose_registration(config)
    resolve = lambda key: (REPO / value[key]["locator_from_retrofit_repo"]).resolve()
    output = lambda key: (LAB / value["outputs"][key]).resolve()
    return resolve("parent"), resolve("base"), resolve("adapter"), output("control"), output("candidate"), output("evidence")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "build", "smoke-load"))
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB / "configs" / "historical_dose_a11_1p0625_trial.json",
    )
    args = parser.parse_args()
    config = args.config.resolve(strict=True)
    parent, base, adapter, control, candidate, evidence = resolve_defaults(config)
    if args.command == "preflight":
        print(json.dumps(preflight_dose(config, parent, base, adapter), sort_keys=True))
        return 0
    if args.command == "build":
        print(
            json.dumps(
                build_dose_trial(
                    config, parent, base, adapter, control, candidate, evidence
                ),
                sort_keys=True,
            )
        )
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    registration = load_dose_registration(config)
    tokenizer = AutoTokenizer.from_pretrained(
        candidate, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        candidate,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map=None,
        attn_implementation="eager",
    ).eval()
    required_class = registration["parent"]["required_class"]
    if type(model).__name__ != required_class:
        raise RuntimeError(f"loaded {type(model).__name__}, expected {required_class}")
    encoded = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits[:, -1]
    if not torch.isfinite(logits.float()).all():
        raise RuntimeError("fresh-load dose smoke produced non-finite logits")
    result = {
        "status": "PASS",
        "class": type(model).__name__,
        "dtype": str(next(model.parameters()).dtype),
        "scale": registration["recipe"]["candidate_scale"],
        "abstain_token_id": int(model.config.abstain_token_id),
        "abstain_logit_bias": float(model.config.abstain_logit_bias),
        "top_token_id": int(logits.argmax(-1).item()),
    }
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "fresh_process_smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
