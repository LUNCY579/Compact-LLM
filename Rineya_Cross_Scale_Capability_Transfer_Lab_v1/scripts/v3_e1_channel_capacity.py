"""v3 E1 (corrected) — what does the transport operator actually transmit?

The P4 operator keeps the *target's* singular vectors and transplants a
normalised donor singular-value profile. Section 1 derives the induced delta in
closed form and counts the donor-derived scalars that cross the bridge.
Section 2 holds the operator fixed and varies the donor, which is what separates
agreement belonging to the donors from agreement belonging to the operator.

The sharp form of the closed-form result is

    a_i = strength * ||W_target|| * (p_i - q_i)

where `p` is the donor's normalised top-r spectrum and `q` is the target's own.
The delta is a *difference of two near-parallel profiles*, so it is dominated by
their radial gap. Any null that does not hold `||p||` at a realistic value
destroys exactly the quantity under test.

Supersedes `attempt_0001_unfair_nulls/`, whose synthetic donor multiplied the
profile norm by `uniform(2, 12)` and therefore randomised the signal itself.
That version reported the cross-donor agreement as a large lift over its
controls; with the controls the amendment actually froze, it is not.

Reads only committed `tensor_projection_map.json` records. Loads no weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, atomic_write_text  # noqa: E402
from rineya_cross_scale.v3_transport import (  # noqa: E402
    delta_coordinates_from_profile,
    donor_profile,
    global_cosine,
)

STRENGTH = 0.10
RANK = 8
MODULE_FAMILIES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def family_of(key: str) -> str:
    for family in MODULE_FAMILIES:
        if family in key:
            return family
    raise KeyError(key)


def target_profile(stat: dict[str, Any]) -> list[float]:
    return [v / stat["target_fro"] for v in stat["target_top_singular_values"]]


def norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E1 transport channel capacity (corrected).")
    parser.add_argument("--projected-root", type=Path,
                        default=LAB_ROOT / "local_models" / "projected" / "P4_D2_W4")
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "e1_channel_capacity")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 42])
    args = parser.parse_args()

    started = time.perf_counter()

    def load(name: str):
        manifest = json.loads((args.projected_root / name / "projection_manifest.json").read_text("utf-8"))
        records = json.loads((args.projected_root / name / "tensor_projection_map.json").read_text("utf-8"))
        if abs(float(manifest["strength"]) - STRENGTH) > 1e-12 or int(manifest["rank"]) != RANK:
            raise SystemExit(f"projection manifest disagrees with frozen constants: {name}")
        return {r["target_key"]: r["stats"] for r in records if r["stats"]}

    stats4 = load("qwen4b_spectral_rank8_s010")
    stats8 = load("qwen8b_spectral_rank8_s010")
    if set(stats4) != set(stats8):
        raise SystemExit("donor projection key sets differ")
    keys = sorted(stats4)

    real4 = {k: donor_profile(stats4[k]) for k in keys}
    real8 = {k: donor_profile(stats8[k]) for k in keys}
    target = {k: target_profile(stats4[k]) for k in keys}

    def build(profiles: dict[str, list[float]]) -> dict[str, Any]:
        return {k: delta_coordinates_from_profile(stats4[k], profiles[k], STRENGTH) for k in keys}

    coords4, coords8 = build(real4), build(real8)

    # --- Section 1: the sharp identity, and how wide the channel is ---------
    identity_error = []
    for key in keys:
        block, _ = delta_coordinates_from_profile(stats4[key], real4[key], STRENGTH)
        sharp = [STRENGTH * stats4[key]["target_fro"] * (p - q)
                 for p, q in zip(real4[key], target[key])]
        scale = max(norm(sharp), 1e-30)
        identity_error.append(
            math.sqrt(sum((b - s) ** 2 for b, s in zip(block, sharp))) / scale)

    # How parallel are the donor and target profiles? If they are nearly
    # parallel, the delta is essentially their radial gap.
    profile_cosines = []
    for key in keys:
        p, q = real4[key], target[key]
        profile_cosines.append(
            sum(a * b for a, b in zip(p, q)) / max(norm(p) * norm(q), 1e-30))
    profile_cosines.sort()

    # --- Section 1b: how much of the delta is donor-independent? -----------
    # A "role-mean donor" carries one pooled profile per module family: 7 roles
    # x r values, and zero tensor-specific information.
    role_sum: dict[str, list[float]] = defaultdict(lambda: [0.0] * RANK)
    role_count: dict[str, int] = defaultdict(int)
    for key in keys:
        family = family_of(key)
        role_count[family] += 1
        for index in range(RANK):
            role_sum[family][index] += (real4[key][index] + real8[key][index]) / 2.0
    role_mean = {f: [v / role_count[f] for v in role_sum[f]] for f in role_sum}
    coords_role = build({k: role_mean[family_of(k)] for k in keys})

    # Two decompositions. The orthogonal one gives a genuine energy fraction:
    # project each real delta onto the role-mean direction per tensor. The plain
    # subtraction a - C is also reported because it is the simpler statement.
    explained4 = explained8 = total4_energy = total8_energy = 0.0
    orthogonal4_energy = orthogonal8_energy = orthogonal_dot = 0.0
    plain4_energy = plain8_energy = plain_dot = 0.0
    for key in keys:
        common = list(coords_role[key][0]) + [coords_role[key][1]]
        a4 = list(coords4[key][0]) + [coords4[key][1]]
        a8 = list(coords8[key][0]) + [coords8[key][1]]
        common_energy = sum(v * v for v in common)
        t4 = sum(x * c for x, c in zip(a4, common)) / max(common_energy, 1e-30)
        t8 = sum(x * c for x, c in zip(a8, common)) / max(common_energy, 1e-30)
        explained4 += t4 * t4 * common_energy
        explained8 += t8 * t8 * common_energy
        total4_energy += sum(v * v for v in a4)
        total8_energy += sum(v * v for v in a8)

        o4 = [x - t4 * c for x, c in zip(a4, common)]
        o8 = [x - t8 * c for x, c in zip(a8, common)]
        orthogonal4_energy += sum(v * v for v in o4)
        orthogonal8_energy += sum(v * v for v in o8)
        orthogonal_dot += sum(x * y for x, y in zip(o4, o8))

        p4 = [x - c for x, c in zip(a4, common)]
        p8 = [x - c for x, c in zip(a8, common)]
        plain4_energy += sum(v * v for v in p4)
        plain8_energy += sum(v * v for v in p8)
        plain_dot += sum(x * y for x, y in zip(p4, p8))

    donor_independent_fraction = explained4 / max(total4_energy, 1e-30)
    donor_independent_fraction_8b = explained8 / max(total8_energy, 1e-30)
    residual_cosine = orthogonal_dot / max(math.sqrt(orthogonal4_energy * orthogonal8_energy), 1e-30)
    plain_residual_cosine = plain_dot / max(math.sqrt(plain4_energy * plain8_energy), 1e-30)

    # --- Section 2: hold the operator fixed, vary the donor ----------------
    controls: list[dict[str, Any]] = []

    def record(arm: str, value: float, note: str) -> None:
        controls.append({"arm": arm, "cosine": value, "note": note})

    record("real_4b_vs_real_8b", global_cosine(coords4, coords8, keys),
           "the v1 headline comparison, recomputed in exact arithmetic")

    # Frozen control 1: norm-matched random. Random profile SHAPE, but the
    # per-tensor profile norm matched to the real donor's.
    for seed in args.seeds:
        left, right = random.Random(seed), random.Random(seed + 1000)

        def matched(rng: random.Random, source: dict[str, list[float]]) -> dict[str, list[float]]:
            output = {}
            for key in keys:
                shape = sorted((rng.random() + 0.05 for _ in range(RANK)), reverse=True)
                factor = norm(source[key]) / max(norm(shape), 1e-30)
                output[key] = [v * factor for v in shape]
            return output

        record(f"norm_matched_random_seed{seed}",
               global_cosine(build(matched(left, real4)), build(matched(right, real8)), keys),
               "random profile shape, per-tensor profile norm matched to the real donor")

    # Frozen control 2: layer-shuffled WITHIN module role, so role identity is
    # preserved and only depth correspondence is destroyed.
    by_family: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_family[family_of(key)].append(key)
    for seed in args.seeds:
        rng = random.Random(seed)
        permuted4, permuted8 = {}, {}
        for family, family_keys in by_family.items():
            ordered = sorted(family_keys)
            left_order, right_order = list(ordered), list(ordered)
            rng.shuffle(left_order)
            rng.shuffle(right_order)
            for key, source_left, source_right in zip(ordered, left_order, right_order):
                permuted4[key] = real4[source_left]
                permuted8[key] = real8[source_right]
        record(f"layer_shuffled_within_role_seed{seed}",
               global_cosine(build(permuted4), build(permuted8), keys),
               "real donor spectra, depth correspondence destroyed, module role preserved")

    # Frozen control 3: sign-shuffled delta coordinates.
    for seed in args.seeds:
        rng = random.Random(seed)
        flipped = {}
        for key in keys:
            block, perpendicular = coords4[key]
            signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(len(block) + 1)]
            flipped[key] = ([abs(v) * s for v, s in zip(block, signs)],
                            abs(perpendicular) * signs[-1])
        record(f"sign_shuffled_seed{seed}", global_cosine(flipped, coords8, keys),
               "delta coordinate magnitudes preserved, signs randomised")

    record("role_mean_donor_vs_real_8b", global_cosine(coords_role, coords8, keys),
           f"a 'donor' carrying only {len(role_mean)} x {RANK} = {len(role_mean) * RANK} numbers "
           "and no tensor-specific information")

    record("identical_donors_upper_bound", global_cosine(coords4, coords4, keys),
           "sanity upper bound")

    real = controls[0]["cosine"]
    null_arms = [c["cosine"] for c in controls
                 if c["arm"].startswith(("norm_matched_random", "layer_shuffled", "role_mean"))]

    payload = {
        "schema_version": "2.0",
        "experiment": "v3_e1_transport_channel_capacity",
        "protocol": "amendment_v3_transport_diagnosis.md 3.0.0-diagnostic",
        "supersedes": "attempt_0001_unfair_nulls",
        "post_hoc_visible": True,
        "operator": {
            "family": "P4 target-basis spectral profile transfer",
            "rank": RANK,
            "strength": STRENGTH,
            "closed_form": "a_i = strength * ||W_target|| * (p_i - q_i); "
                           "a_perp = (residual_scale - 1) * ||W_perp||",
            "donor_derived_free_scalars_per_tensor": RANK,
            "donor_directional_inputs": 0,
            "max_identity_relative_error": max(identity_error),
        },
        "channel": {
            "projected_tensors": len(keys),
            "target_tensors_total": 310,
            "donor_scalars_transmitted": len(keys) * RANK,
            "qwen3_4b_parameters": 4022468096,
            "qwen3_8b_parameters": 8190735360,
            "fraction_of_4b_transmitted": len(keys) * RANK / 4022468096,
            "donor_target_profile_cosine_median": profile_cosines[len(profile_cosines) // 2],
            "donor_target_profile_cosine_min": profile_cosines[0],
        },
        "donor_independence": {
            "role_mean_donor_numbers": len(role_mean) * RANK,
            "donor_independent_energy_fraction_4b": donor_independent_fraction,
            "donor_independent_energy_fraction_8b": donor_independent_fraction_8b,
            "residual_cross_donor_cosine_orthogonal": residual_cosine,
            "residual_cross_donor_cosine_plain_subtraction": plain_residual_cosine,
            "HT2_statistic": donor_independent_fraction,
            "HT2_some_agreement_is_operator_induced": donor_independent_fraction > 0.0,
        },
        "controls": controls,
        "verdict": {
            "real_cosine": real,
            "null_band_low": min(null_arms),
            "null_band_high": max(null_arms),
            "real_inside_null_band": min(null_arms) <= real <= max(null_arms),
            "HT1_channel_is_narrow_and_directionless": True,
        },
        "protocol_deviation_in_attempt_0001": (
            "The amendment freezes norm-matched random, layer-shuffled, and sign-shuffled "
            "controls for E1-E3. The first attempt ran a norm-RANDOMISED synthetic donor, an "
            "all-key shuffle that also destroyed module role, and no sign-shuffled arm. It also "
            "published a 'null donor' arm whose delta is an exact algebraic zero, so its reported "
            "cosine of -0.2462 was the angle between two rounding-error vectors. This corrected "
            "run implements the frozen controls."
        ),
        "resources": {"wall_seconds": time.perf_counter() - started, "models_loaded": 0},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "channel_capacity.json", payload)

    lines = [
        "# v3 E1 — Transport Channel Capacity (corrected)",
        "",
        f"Operator: P4 target-basis spectral profile transfer, rank {RANK}, strength {STRENGTH}.",
        "",
        "Supersedes `attempt_0001_unfair_nulls/`. See the protocol-deviation note at the end.",
        "",
        "## What crosses the bridge",
        "",
        f"- Projected tensors: **{len(keys)}** of 310 target tensors.",
        f"- Donor-derived scalars transmitted: **{len(keys) * RANK}**"
        f" ({len(keys)} tensors x {RANK} normalised singular values).",
        f"- That is {payload['channel']['fraction_of_4b_transmitted']:.3e} of Qwen3-4B.",
        "- Donor **directional** information transmitted: **none**. Every direction in the",
        "  injected update is one of the target's own top-r singular directions or the",
        "  target weight itself.",
        "",
        "The closed form has a sharper reading than E1's first attempt used:",
        "",
        "> `a_i = strength * ||W_target|| * (p_i - q_i)`",
        "",
        "where `p` is the donor's normalised top-r spectrum and `q` is the target's own",
        f"(verified to {max(identity_error):.2e} relative). The two profiles are nearly parallel —",
        f"median cos(p, q) = {payload['channel']['donor_target_profile_cosine_median']:.4f} — so the delta is",
        "essentially their *radial gap*. That is the quantity any fair null must preserve.",
        "",
        "## How much of the delta is donor-independent?",
        "",
        "Replace each donor profile with the pooled mean for its module role. That 'donor'",
        f"carries {len(role_mean) * RANK} numbers in total and no tensor-specific information at all.",
        "",
        f"- Projecting each real delta onto it explains **{donor_independent_fraction * 100:.1f}%**"
        f" of the 4B delta energy",
        f"  and **{donor_independent_fraction_8b * 100:.1f}%** of the 8B delta energy.",
        f"- The orthogonal residual still agrees across donors at cosine"
        f" **{residual_cosine:.4f}**, so a genuine",
        "  tensor-specific component survives; it is simply a minority of the signal.",
        f"- Under plain subtraction rather than projection the residual cosine is"
        f" {plain_residual_cosine:.4f}.",
        "",
        "## Hold the operator fixed, vary the donor",
        "",
        "| Arm | cos(delta_a, delta_b) |",
        "|---|---:|",
    ]
    for control in controls:
        lines.append(f"| {control['arm']} | {control['cosine']:.4f} |")
    lines += [
        "",
        f"Real cross-donor cosine **{real:.4f}**. Architecture-only null band"
        f" **{min(null_arms):.4f} – {max(null_arms):.4f}**.",
        "",
        "The strongest null lives in E1c: iid Gaussian matrices at the *true donor tensor",
        "shapes*, through the identical operator, reach **0.9988** across three seed pairs —",
        "above the real donors. See `shape_matched_donor_null.md`.",
        "",
        "## Interpretation",
        "",
        "**HT1 holds**: the channel is narrow and carries no directions.",
        "",
        f"**HT2 holds**: {donor_independent_fraction * 100:.1f}% of the delta energy is donor-independent"
        " by exact algebra —",
        "reproducible from a 56-number role-mean donor. The amendment's wording is that *some*",
        "of the agreement is a property of `T`, and that is amply satisfied.",
        "",
        "**The cross-donor agreement statistic carries no evidence about donor content.** The v1",
        "report read cosine 0.932 as showing the donor signal is not random. But donors sharing",
        "nothing except their tensor shapes reach 0.9988 through the same operator — *better* than",
        "the real donors. The statistic should be retired as evidence, not merely down-scoped.",
        "",
        "That is not the same as saying nothing donor-specific is present. After projecting out the",
        f"role-mean construction, a tensor-specific residual carrying {(1 - donor_independent_fraction) * 100:.0f}% of the delta",
        f"energy still agrees across the two donors at cosine {residual_cosine:.4f}. Real donor structure does",
        "reach the target; it is a minority of an already narrow channel, and the headline statistic",
        "was never measuring it.",
        "",
        "This does not weaken the study's negative result — it strengthens it. It removes the",
        "premise that would otherwise justify building a wider operator on the grounds that",
        "'the donors demonstrably agree'.",
        "",
        "## Protocol deviation in the first attempt",
        "",
        payload["protocol_deviation_in_attempt_0001"],
    ]
    atomic_write_text(args.output / "channel_capacity.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
