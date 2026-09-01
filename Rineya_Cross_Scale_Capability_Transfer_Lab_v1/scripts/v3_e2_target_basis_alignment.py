"""v3 E2/E3 — is the transported donor direction pointing anywhere useful?

The target model has its own known-good adaptation direction on disk:
`Delta_Supa = Supa - Qwen3-0.6B`. This experiment asks whether any transported
donor direction resembles it, and whether the subspace `P4` writes into is a
subspace the target's own successful adaptation actually uses.

Every statistic is reported beside norm-matched-random, sign-shuffled, and
layer-shuffled controls. Results are aggregated globally, by relative-depth
bucket (E3), and by module family (E3).

Streams one tensor at a time from memory-mapped safetensors. Never holds two
checkpoints in memory. CPU and FP32 throughout.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.checkpoint import CheckpointReader  # noqa: E402
from rineya_cross_scale.hashing import atomic_write_json, atomic_write_text  # noqa: E402
from rineya_cross_scale.spectral_projection import _low_rank_spectrum, _seed  # noqa: E402

RANK = 8
SEED = 17
DEPTH_BUCKETS = (("early", 0, 8), ("middle", 9, 18), ("late", 19, 27))
MODULE_FAMILIES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def layer_of(key: str) -> int | None:
    parts = key.split(".")
    if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2])
    return None


def bucket_of(layer: int | None) -> str:
    if layer is None:
        return "non_layer"
    for name, low, high in DEPTH_BUCKETS:
        if low <= layer <= high:
            return name
    return "non_layer"


def family_of(key: str) -> str:
    for family in MODULE_FAMILIES:
        if family in key:
            return family
    return "other"


class Pair:
    """Streaming accumulator for a global cosine and sign agreement."""

    __slots__ = ("dot", "x2", "y2", "agree", "count")

    def __init__(self) -> None:
        self.dot = self.x2 = self.y2 = 0.0
        self.agree = self.count = 0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.dot += float(torch.dot(x, y))
        self.x2 += float(torch.dot(x, x))
        self.y2 += float(torch.dot(y, y))
        self.agree += int((torch.sign(x) == torch.sign(y)).sum())
        self.count += x.numel()

    def result(self) -> dict[str, Any]:
        return {
            "cosine": self.dot / max(math.sqrt(self.x2 * self.y2), 1e-30),
            "sign_agreement": self.agree / max(1, self.count),
            "norm_ratio": math.sqrt(self.x2) / max(math.sqrt(self.y2), 1e-30),
            "coordinates": self.count,
        }


def effective_rank(singular: torch.Tensor) -> float:
    """exp(entropy of the normalised singular-value distribution)."""
    values = singular[singular > 0]
    if values.numel() == 0:
        return 0.0
    probabilities = values / values.sum()
    entropy = float(-(probabilities * probabilities.log()).sum())
    return math.exp(entropy)


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E2/E3 target-basis alignment.")
    parser.add_argument("--deltas-root", type=Path, default=LAB_ROOT / "local_models" / "deltas")
    parser.add_argument("--reference", type=Path,
                        default=Path("C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
                                     "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"))
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis")
    parser.add_argument("--subspace-rank", type=int, default=RANK)
    args = parser.parse_args()

    started = time.perf_counter()
    generator = torch.Generator().manual_seed(SEED)

    donors = {
        "delta_4b": args.deltas_root / "P4_D2_W4_4B_s010",
        "delta_8b": args.deltas_root / "P4_D2_W4_8B_s010",
        "delta_mean": args.deltas_root / "P4_D2_W4_MEAN_s010",
        "delta_p5_4b": args.deltas_root / "P5_D2_WDIR_4B_s010",
    }
    supa_path = args.deltas_root / "SUPA_MINUS_QWEN06"

    global_pairs: dict[str, Pair] = defaultdict(Pair)
    bucket_pairs: dict[tuple[str, str], Pair] = defaultdict(Pair)
    family_pairs: dict[tuple[str, str], Pair] = defaultdict(Pair)
    subspace_rows: list[dict[str, Any]] = []
    layer_shuffle_pairs: dict[str, Pair] = defaultdict(Pair)

    with CheckpointReader(supa_path) as supa, CheckpointReader(args.reference) as reference:
        readers = {name: CheckpointReader(path).__enter__() for name, path in donors.items()}
        try:
            keys = [k for k in supa.keys if k in set(readers["delta_4b"].keys)]
            two_dim = [k for k in keys if len(supa.descriptor(k).shape) == 2 and layer_of(k) is not None]
            # Layer-shuffled control: pair donor tensor at layer L with the Supa
            # delta of the SAME module family at a different layer.
            by_family: dict[str, list[str]] = defaultdict(list)
            for key in two_dim:
                by_family[family_of(key)].append(key)
            shuffled_partner: dict[str, str] = {}
            for family, family_keys in by_family.items():
                ordered = sorted(family_keys, key=lambda k: layer_of(k) or 0)
                rotation = max(1, len(ordered) // 2)
                for index, key in enumerate(ordered):
                    shuffled_partner[key] = ordered[(index + rotation) % len(ordered)]

            two_dim_set = set(two_dim)
            for position, key in enumerate(keys):
                supa_delta = supa.get_tensor(key).float().reshape(-1)
                if float(torch.dot(supa_delta, supa_delta)) == 0.0:
                    del supa_delta
                    continue
                layer = layer_of(key)
                bucket, family = bucket_of(layer), family_of(key)

                for name, reader in readers.items():
                    donor_delta = reader.get_tensor(key).float().reshape(-1)
                    if float(torch.dot(donor_delta, donor_delta)) == 0.0:
                        del donor_delta
                        continue
                    global_pairs[name].add(donor_delta, supa_delta)
                    bucket_pairs[(name, bucket)].add(donor_delta, supa_delta)
                    family_pairs[(name, family)].add(donor_delta, supa_delta)

                    if name == "delta_mean":
                        # Controls, evaluated against the same Supa direction.
                        matched = torch.randn(donor_delta.shape, generator=generator)
                        matched *= torch.linalg.vector_norm(donor_delta) / torch.linalg.vector_norm(matched)
                        global_pairs["control_norm_matched_random"].add(matched, supa_delta)
                        signs = torch.randint(0, 2, donor_delta.shape, generator=generator) * 2 - 1
                        global_pairs["control_sign_shuffled"].add(donor_delta.abs() * signs, supa_delta)
                        del matched, signs
                        partner = shuffled_partner.get(key)
                        if partner is not None and partner != key:
                            other = supa.get_tensor(partner).float().reshape(-1)
                            if other.numel() == donor_delta.numel():
                                layer_shuffle_pairs["control_layer_shuffled"].add(donor_delta, other)
                            del other
                    del donor_delta

                # Where does Supa's own adaptation live relative to the subspace
                # that P4 writes into?
                if key in two_dim_set:
                    weight = reference.get_tensor(key).float()
                    u, s, v = _low_rank_spectrum(
                        weight, args.subspace_rank, seed=_seed(key, "target"), iterations=1
                    )
                    supa_matrix = supa_delta.reshape(weight.shape)
                    coefficients = torch.einsum("mr,mn,nr->r", u, supa_matrix, v)
                    supa_energy = float(torch.dot(supa_delta, supa_delta))
                    subspace_energy = float(torch.dot(coefficients, coefficients))
                    weight_flat = weight.reshape(-1)
                    weight_energy = float(torch.dot(weight_flat, weight_flat))
                    parallel = float(torch.dot(supa_delta, weight_flat)) ** 2 / max(weight_energy, 1e-30)
                    rows, columns = weight.shape
                    subspace_rows.append({
                        "key": key, "layer": layer, "bucket": bucket, "family": family,
                        "supa_delta_energy": supa_energy,
                        "energy_in_p4_subspace": subspace_energy,
                        "fraction_in_p4_subspace": subspace_energy / max(supa_energy, 1e-30),
                        "fraction_parallel_to_weight": parallel / max(supa_energy, 1e-30),
                        "random_expectation": args.subspace_rank / (rows * columns),
                        "supa_relative_norm": math.sqrt(supa_energy) / max(math.sqrt(weight_energy), 1e-30),
                    })
                    del weight, u, s, v, supa_matrix, coefficients, weight_flat
                del supa_delta
                if position % 20 == 19:
                    gc.collect()
        finally:
            for reader in readers.values():
                reader.__exit__(None, None, None)

    # Effective rank of the two directions, sampled on the largest MLP tensors.
    rank_rows = []
    with CheckpointReader(supa_path) as supa, CheckpointReader(donors["delta_mean"]) as mean_reader:
        for layer in (0, 7, 14, 21, 27):
            key = f"model.layers.{layer}.mlp.down_proj.weight"
            supa_matrix = supa.get_tensor(key).float()
            mean_matrix = mean_reader.get_tensor(key).float()
            rank_rows.append({
                "key": key,
                "supa_effective_rank": effective_rank(torch.linalg.svdvals(supa_matrix)),
                "donor_mean_effective_rank": effective_rank(torch.linalg.svdvals(mean_matrix)),
                "full_rank": min(supa_matrix.shape),
            })
            del supa_matrix, mean_matrix
            gc.collect()

    payload = {
        "schema_version": "1.0",
        "experiment": "v3_e2_target_basis_alignment",
        "protocol": "amendment_v3_transport_diagnosis.md 3.0.0-diagnostic",
        "reference": str(args.reference),
        "supa_direction": str(supa_path),
        "global": {name: pair.result() for name, pair in sorted(global_pairs.items())},
        "layer_shuffled_control": {n: p.result() for n, p in layer_shuffle_pairs.items()},
        "by_depth_bucket": {
            f"{name}::{bucket}": pair.result() for (name, bucket), pair in sorted(bucket_pairs.items())
        },
        "by_module_family": {
            f"{name}::{family}": pair.result() for (name, family), pair in sorted(family_pairs.items())
        },
        "supa_energy_in_p4_subspace": {
            "tensors": len(subspace_rows),
            "subspace_rank": args.subspace_rank,
            "mean_fraction_in_p4_subspace": sum(r["fraction_in_p4_subspace"] for r in subspace_rows)
            / max(1, len(subspace_rows)),
            "mean_random_expectation": sum(r["random_expectation"] for r in subspace_rows)
            / max(1, len(subspace_rows)),
            "mean_fraction_parallel_to_weight": sum(r["fraction_parallel_to_weight"] for r in subspace_rows)
            / max(1, len(subspace_rows)),
            "mean_supa_relative_norm": sum(r["supa_relative_norm"] for r in subspace_rows)
            / max(1, len(subspace_rows)),
            "per_tensor": subspace_rows,
        },
        "effective_rank": rank_rows,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "models_loaded": 0,
            "checkpoint_readers_opened": len(donors) + 2,
        },
    }

    destination = args.output / "e2_target_basis_alignment"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "target_basis_alignment.json", payload)

    enriched = payload["supa_energy_in_p4_subspace"]
    lines = [
        "# v3 E2 — Target-Basis Alignment of the Transported Direction",
        "",
        "Reference frame: `Delta_Supa = Supa-A1.1 - Qwen3-0.6B`, the target model's own",
        "adaptation direction, measured over tensors where it is non-zero.",
        "",
        "## Does any transported direction resemble the target's own adaptation?",
        "",
        "| Direction | cos vs Delta_Supa | sign agreement | ||delta|| / ||Delta_Supa|| |",
        "|---|---:|---:|---:|",
    ]
    for name, result in sorted(payload["global"].items()):
        lines.append(
            f"| {name} | {result['cosine']:+.4f} | {result['sign_agreement']:.4f} |"
            f" {result['norm_ratio']:.4f} |"
        )
    for name, result in payload["layer_shuffled_control"].items():
        lines.append(
            f"| {name} | {result['cosine']:+.4f} | {result['sign_agreement']:.4f} |"
            f" {result['norm_ratio']:.4f} |"
        )
    lines += [
        "",
        "## Does Supa's own adaptation use the subspace P4 writes into?",
        "",
        f"- Tensors measured: {enriched['tensors']}, subspace rank {enriched['subspace_rank']}.",
        f"- Mean fraction of `Delta_Supa` energy inside P4's rank-{enriched['subspace_rank']}"
        f" target subspace: **{enriched['mean_fraction_in_p4_subspace']:.4e}**.",
        f"- Chance level for an isotropic direction: {enriched['mean_random_expectation']:.4e}.",
        f"- Enrichment over chance: **{enriched['mean_fraction_in_p4_subspace'] / max(enriched['mean_random_expectation'], 1e-30):.1f}x**.",
        f"- Mean fraction parallel to the weight itself: {enriched['mean_fraction_parallel_to_weight']:.4e}.",
        f"- Mean `||Delta_Supa|| / ||W||`: {enriched['mean_supa_relative_norm']:.4e}.",
        "",
        "## Effective rank",
        "",
        "| Tensor | Delta_Supa | donor mean | full rank |",
        "|---|---:|---:|---:|",
    ]
    for row in rank_rows:
        lines.append(
            f"| {row['key']} | {row['supa_effective_rank']:.1f} |"
            f" {row['donor_mean_effective_rank']:.1f} | {row['full_rank']} |"
        )
    atomic_write_text(destination / "target_basis_alignment.md", "\n".join(lines) + "\n")

    localization = {
        "schema_version": "1.0",
        "experiment": "v3_e3_localization",
        "by_depth_bucket": payload["by_depth_bucket"],
        "by_module_family": payload["by_module_family"],
        "depth_buckets": [{"name": n, "first": a, "last": b} for n, a, b in DEPTH_BUCKETS],
    }
    localization_dir = args.output / "e3_localization"
    localization_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(localization_dir / "localization.json", localization)

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
