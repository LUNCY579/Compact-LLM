"""v3 E1c — the shape-matched random-donor null that E1 should have run.

E1's synthetic donor drew a sorted-uniform top-r spectrum and then rescaled it by
an independent uniform(2, 12) factor per tensor. That randomises the donor's
top-r energy fraction over a six-fold range, which no real weight matrix does.
It therefore understates how much two *arbitrary* donors agree through the P4
operator, and it made the real 4B/8B agreement look like a large lift.

The correct null holds the donor's shape fixed and randomises only its content:
draw an iid Gaussian matrix at each true donor tensor shape, pass it through the
identical low-rank estimator and the identical operator, and measure the same
cross-donor cosine. Independent draw per tensor, per side.

If two random matrices of the right shapes agree as strongly as Qwen3-4B and
Qwen3-8B do, then the agreement is a property of the operator and the shapes,
not of the donors.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, atomic_write_text  # noqa: E402
from rineya_cross_scale.spectral_projection import _low_rank_spectrum  # noqa: E402
from rineya_cross_scale.v3_transport import (  # noqa: E402
    delta_coordinates_from_profile,
    donor_profile,
    global_cosine,
)

RANK = 8
STRENGTH = 0.10


def donor_shapes(config: dict[str, Any]) -> dict[str, tuple[int, int]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    head_dim = int(config["head_dim"])
    query = int(config["num_attention_heads"]) * head_dim
    key_value = int(config["num_key_value_heads"]) * head_dim
    return {
        "self_attn.q_proj": (query, hidden),
        "self_attn.k_proj": (key_value, hidden),
        "self_attn.v_proj": (key_value, hidden),
        "self_attn.o_proj": (hidden, query),
        "mlp.gate_proj": (intermediate, hidden),
        "mlp.up_proj": (intermediate, hidden),
        "mlp.down_proj": (hidden, intermediate),
    }


def family_of(key: str, shapes: dict[str, tuple[int, int]]) -> str:
    for family in shapes:
        if family in key:
            return family
    raise KeyError(key)


def gaussian_profile(shape: tuple[int, int], generator: torch.Generator) -> list[float]:
    """Normalised top-r spectrum of an iid Gaussian matrix of the given shape."""
    matrix = torch.randn(shape, generator=generator, dtype=torch.float32)
    _, singular, _ = _low_rank_spectrum(matrix, RANK, seed=17, iterations=1)
    frobenius = float(torch.linalg.vector_norm(matrix))
    values = [float(v) / frobenius for v in singular]
    del matrix, singular
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E1c shape-matched donor null.")
    parser.add_argument("--projected-root", type=Path,
                        default=LAB_ROOT / "local_models" / "projected" / "P4_D2_W4")
    parser.add_argument("--architecture", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "02_architecture")
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "e1_channel_capacity")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 42])
    args = parser.parse_args()

    started = time.perf_counter()

    def load(name: str) -> dict[str, dict[str, Any]]:
        records = json.loads(
            (args.projected_root / name / "tensor_projection_map.json").read_text("utf-8"))
        return {r["target_key"]: r["stats"] for r in records if r["stats"]}

    stats4 = load("qwen4b_spectral_rank8_s010")
    stats8 = load("qwen8b_spectral_rank8_s010")
    keys = sorted(set(stats4) & set(stats8))

    configs = {
        name: json.loads((args.architecture / f"{name}.json").read_text("utf-8"))
        for name in ("qwen4b", "qwen8b")
    }
    shapes = {name: donor_shapes(configs[name]["config"]) for name in configs}

    real4 = {k: donor_profile(stats4[k]) for k in keys}
    real8 = {k: donor_profile(stats8[k]) for k in keys}

    def build(profiles: dict[str, list[float]]) -> dict[str, Any]:
        return {k: delta_coordinates_from_profile(stats4[k], profiles[k], STRENGTH) for k in keys}

    real_cosine = global_cosine(build(real4), build(real8), keys)

    arms: list[dict[str, Any]] = [{
        "arm": "real_4b_vs_real_8b",
        "cosine": real_cosine,
        "note": "the v1 headline comparison",
    }]

    for seed in args.seeds:
        left = torch.Generator().manual_seed(seed)
        right = torch.Generator().manual_seed(seed + 5000)
        profiles4: dict[str, list[float]] = {}
        profiles8: dict[str, list[float]] = {}
        for position, key in enumerate(keys):
            profiles4[key] = gaussian_profile(shapes["qwen4b"][family_of(key, shapes["qwen4b"])], left)
            profiles8[key] = gaussian_profile(shapes["qwen8b"][family_of(key, shapes["qwen8b"])], right)
            if position % 20 == 19:
                gc.collect()
        arms.append({
            "arm": f"shape_matched_gaussian_donors_seed{seed}",
            "cosine": global_cosine(build(profiles4), build(profiles8), keys),
            "note": "iid Gaussian matrices at the true 4B/8B tensor shapes, independent draw per tensor",
        })
        print(f"  seed {seed}: {arms[-1]['cosine']:.4f}", flush=True)

    null_values = [a["cosine"] for a in arms if a["arm"].startswith("shape_matched")]
    null_mean = sum(null_values) / len(null_values)

    payload = {
        "schema_version": "1.0",
        "experiment": "v3_e1c_shape_matched_donor_null",
        "protocol": "amendment_v3_transport_diagnosis.md 3.0.0-diagnostic",
        "supersedes": "the two_random_donors arms in channel_capacity.json",
        "rank": RANK,
        "strength": STRENGTH,
        "tensors": len(keys),
        "donor_shapes": {name: {k: list(v) for k, v in shape.items()} for name, shape in shapes.items()},
        "arms": arms,
        "verdict": {
            "real_cosine": real_cosine,
            "shape_matched_null_mean": null_mean,
            "lift_over_null": real_cosine - null_mean,
            "real_exceeds_null": real_cosine > null_mean,
            "HT2_agreement_is_operator_induced": real_cosine <= null_mean,
        },
        "resources": {"wall_seconds": time.perf_counter() - started, "models_loaded": 0},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "shape_matched_donor_null.json", payload)

    lines = [
        "# v3 E1c — The Shape-Matched Random-Donor Null",
        "",
        "E1's synthetic donor randomised the top-r energy fraction over a six-fold range per",
        "tensor, which no real weight matrix does. This is the null that holds shape fixed and",
        "randomises only content.",
        "",
        "| Arm | cos(delta_a, delta_b) |",
        "|---|---:|",
    ]
    for arm in arms:
        lines.append(f"| {arm['arm']} | {arm['cosine']:.4f} |")
    lines += [
        "",
        f"Real Qwen3-4B vs Qwen3-8B: **{real_cosine:.4f}**.",
        f"Shape-matched Gaussian donors: **{null_mean:.4f}** (mean of {len(null_values)} seed pairs).",
        f"Lift of the real donors over the null: **{real_cosine - null_mean:+.4f}**.",
        "",
        "## Verdict",
        "",
    ]
    if real_cosine <= null_mean:
        lines += [
            "The real cross-donor agreement does **not** exceed what two random matrices of the",
            "same shapes produce through this operator. It is in fact slightly lower.",
            "",
            "**HT2 holds and E1's Section-2 conclusion is retracted.** The v1 headline statistic",
            "(cosine 0.932 between the projected 4B and 8B deltas, read as evidence that the donor",
            "signal is not random) is a property of the P4 operator and of the donor tensor shapes.",
            "It is not evidence of a shared cross-scale direction, because donors that share nothing",
            "but their shapes reproduce it.",
            "",
            "The mechanism is the one E1 already established. The delta's coordinates are",
            "`a_i = strength * ||W_target|| * (p_donor_i - p_target_i)`, where `p` is a normalised",
            "top-r spectrum. The normalised top-r spectrum of a large matrix is governed mostly by",
            "its aspect ratio, so any two donors of these shapes produce nearly the same `p`, hence",
            "nearly the same `a`, hence a high cosine.",
        ]
    else:
        lines += [
            "The real cross-donor agreement exceeds the shape-matched null, so a donor-specific",
            "component survives this stronger control.",
        ]
    atomic_write_text(args.output / "shape_matched_donor_null.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
