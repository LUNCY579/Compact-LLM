#!/usr/bin/env python3
"""Preflight, build, verify, or smoke-load the frozen historical soup trial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from rineya_tgsa.historical_soup import (
    build_candidate,
    load_registration,
    preflight,
    verify_candidate,
)


def _defaults(config_path: Path) -> tuple[Path, Path, Path, Path]:
    registration = load_registration(config_path)
    parent = (REPO / registration["parent"]["locator_from_retrofit_repo"]).resolve()
    ingredient = (REPO / registration["ingredient"]["locator_from_retrofit_repo"]).resolve()
    artifact = (LAB / registration["candidate"]["artifact_root"]).resolve()
    evidence = (LAB / registration["candidate"]["evidence_root"]).resolve()
    return parent, ingredient, artifact, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "build", "verify", "smoke-load"))
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB / "configs" / "historical_soup_a11_baya_midpoint_trial.json",
    )
    parser.add_argument("--parent-dir", type=Path)
    parser.add_argument("--ingredient-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()

    config = args.config.resolve(strict=True)
    default_parent, default_ingredient, default_artifact, default_evidence = _defaults(config)
    parent = (args.parent_dir or default_parent).resolve()
    ingredient = (args.ingredient_dir or default_ingredient).resolve()
    artifact = (args.artifact_dir or default_artifact).resolve()
    evidence = (args.evidence_dir or default_evidence).resolve()

    if args.command == "preflight":
        report = preflight(config, parent, ingredient)
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "preflight.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "build":
        print(json.dumps(build_candidate(config, parent, ingredient, artifact, evidence), sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_candidate(config, artifact), sort_keys=True))
        return 0

    registration = load_registration(config)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        artifact, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        artifact,
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
        raise RuntimeError("fresh-load smoke produced non-finite logits")
    result = {
        "status": "PASS",
        "class": type(model).__name__,
        "dtype": str(next(model.parameters()).dtype),
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
