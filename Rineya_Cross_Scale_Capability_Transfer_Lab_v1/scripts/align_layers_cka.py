"""Exploratory D4: linear CKA monotonic depth alignment on calibration prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.cka import monotonic_alignment, pairwise_cka  # noqa: E402
from rineya_cross_scale.hashing import atomic_write_json, sha256_file  # noqa: E402
from rineya_cross_scale.leakage import guard_non_sealed_path  # noqa: E402


def _prompts(path: Path) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line)["prompt"])
    return rows


def _hidden_layers(model_path: Path, prompts: list[str], *, device: str, max_items: int) -> list[torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    collected: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for prompt in prompts[:max_items]:
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True)
            states = [state[0, -1, :].cpu() for state in output.hidden_states[1:]]
            collected.append(states)
    del model
    layers = []
    for index in range(len(collected[0])):
        layers.append(torch.stack([row[index] for row in collected], dim=0))
    return layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-items", type=int, default=32)
    args = parser.parse_args()
    calibration = guard_non_sealed_path(args.calibration)
    prompts = _prompts(calibration)
    target = _hidden_layers(args.target, prompts, device=args.device, max_items=args.max_items)
    donor = _hidden_layers(args.donor, prompts, device=args.device, max_items=args.max_items)
    similarity = pairwise_cka(target, donor)
    alignment = monotonic_alignment(similarity)
    payload = {
        "schema_version": "2.0",
        "exploratory": True,
        "calibration_sha256": sha256_file(calibration),
        "target": str(args.target.resolve()),
        "donor": str(args.donor.resolve()),
        "similarity": similarity.tolist(),
        **alignment,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"mapping": alignment["mapping"], "score": alignment["score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
