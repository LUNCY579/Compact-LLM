"""v3 E4 (analysis) — the cross-model functional alignment atlas.

Consumes the sparse captures written by `v3_e4_capture_representations.py` and
builds the depth x model similarity tables the amendment calls for:

  1. representation similarity between every model pair at matched depths;
  2. logit-space relationships (cosine, symmetric KL, top-1 agreement, margin);
  3. the decisive quantity — whether a transported donor's *representation
     change* relative to Qwen3-0.6B points the same way as Supa's own
     representation change relative to Qwen3-0.6B.

(3) is reported beside a prompt-shuffled control, which is what separates a
genuine per-prompt alignment from a constant bias direction that would score
high for trivial reasons.

Pure NumPy over cached summaries. Loads no model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, atomic_write_text  # noqa: E402

BASE = "qwen06"
TARGET = "supa"
SEED = 17


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = (a * b).sum(axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.maximum(denominator, 1e-30)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two sample-by-feature matrices."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    left = np.linalg.norm(x.T @ x, ord="fro")
    right = np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / max(left * right, 1e-30))


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def logit_relationship(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    centered_a = a32 - a32.mean(axis=-1, keepdims=True)
    centered_b = b32 - b32.mean(axis=-1, keepdims=True)
    log_a, log_b = log_softmax(a32), log_softmax(b32)
    prob_a, prob_b = np.exp(log_a), np.exp(log_b)
    symmetric_kl = ((prob_a * (log_a - log_b)).sum(axis=-1)
                    + (prob_b * (log_b - log_a)).sum(axis=-1))
    return {
        "logit_cosine": float(row_cosine(centered_a, centered_b).mean()),
        "symmetric_kl": float(symmetric_kl.mean()),
        "symmetric_kl_max": float(symmetric_kl.max()),
        "top1_agreement": float((a32.argmax(-1) == b32.argmax(-1)).mean()),
    }


def margin(values: np.ndarray) -> float:
    ordered = np.sort(values.astype(np.float32), axis=-1)
    return float((ordered[:, -1] - ordered[:, -2]).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E4 functional alignment atlas.")
    parser.add_argument("--captures", type=Path,
                        default=LAB_ROOT / "local_models" / "v3_activations")
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "e4_functional_atlas")
    args = parser.parse_args()

    started = time.perf_counter()
    index = json.loads((args.captures / "capture_index.json").read_text("utf-8"))
    slots = index["models"][0]["capture_slots"]
    pooling = index.get("pooling", "last")
    names = [m["model"] for m in index["models"]]

    hidden = {n: np.load(args.captures / f"{n}.npz")["hidden"] for n in names}
    logits = {n: np.load(args.captures / f"{n}.npz")["logits"] for n in names}
    prompts = hidden[BASE].shape[0]

    # --- 1. representation similarity at matched depths -------------------
    representation: dict[str, dict[str, Any]] = {}
    for left, right in combinations(names, 2):
        per_depth = []
        for slot in range(len(slots)):
            a, b = hidden[left][:, slot, :], hidden[right][:, slot, :]
            per_depth.append({
                "slot": slots[slot],
                "mean_cosine": float(row_cosine(a, b).mean()),
                "linear_cka": linear_cka(a, b),
                "relative_distance": float(
                    (np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1), 1e-30)).mean()
                ),
            })
        representation[f"{left}__vs__{right}"] = {"depths": per_depth}

    # --- 2. logit-space relationships -------------------------------------
    logit_space = {
        f"{left}__vs__{right}": logit_relationship(logits[left], logits[right])
        for left, right in combinations(names, 2)
    }
    margins = {n: margin(logits[n]) for n in names}

    # --- 3. representation-change alignment (the decisive table) ----------
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(prompts)
    while np.any(permutation == np.arange(prompts)):
        permutation = rng.permutation(prompts)

    supa_change = hidden[TARGET] - hidden[BASE]
    alignment: dict[str, Any] = {}
    for name in names:
        if name in (BASE, TARGET):
            continue
        change = hidden[name] - hidden[BASE]
        per_depth = []
        for slot in range(len(slots)):
            donor_change = change[:, slot, :]
            target_change = supa_change[:, slot, :]
            aligned = row_cosine(donor_change, target_change)
            shuffled = row_cosine(donor_change, target_change[permutation])
            random_direction = rng.standard_normal(donor_change.shape).astype(np.float32)
            per_depth.append({
                "slot": slots[slot],
                "mean_cosine_vs_supa_change": float(aligned.mean()),
                "std_cosine_vs_supa_change": float(aligned.std()),
                "control_prompt_shuffled": float(shuffled.mean()),
                "control_random_direction": float(row_cosine(random_direction, target_change).mean()),
                "change_relative_magnitude": float(
                    (np.linalg.norm(donor_change, axis=-1)
                     / np.maximum(np.linalg.norm(hidden[BASE][:, slot, :], axis=-1), 1e-30)).mean()
                ),
                "supa_change_relative_magnitude": float(
                    (np.linalg.norm(target_change, axis=-1)
                     / np.maximum(np.linalg.norm(hidden[BASE][:, slot, :], axis=-1), 1e-30)).mean()
                ),
            })
        alignment[name] = {"depths": per_depth}

    payload = {
        "schema_version": "1.0",
        "experiment": "v3_e4_functional_alignment_atlas",
        "protocol": "amendment_v3_transport_diagnosis.md 3.0.0-diagnostic",
        "split": "calibration",
        "pooling": pooling,
        "dataset_sha256": index["dataset_sha256"],
        "prompts": prompts,
        "capture_slots": slots,
        "models": {m["model"]: m["model_digest"] for m in index["models"]},
        "representation_similarity": representation,
        "logit_space": logit_space,
        "top1_margin": margins,
        "representation_change_alignment": alignment,
        "resources": {"wall_seconds": time.perf_counter() - started, "models_loaded": 0},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "functional_atlas.json", payload)

    def table(title: str, header: list[str], rows: list[list[str]]) -> list[str]:
        out = [f"## {title}", "", "| " + " | ".join(header) + " |",
               "|" + "|".join(["---"] * len(header)) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return out + [""]

    lines = [
        "# v3 E4 — Cross-Model Functional Alignment Atlas",
        "",
        f"{prompts} frozen calibration prompts, prompts only, CPU float32.",
        f"Token pooling: **{pooling}**"
        + (" (the final prompt token, which is what produces the next token)."
           if pooling == "last" else
           " (mean over all prompt positions, a robustness check on the final-token view)."),
        f"Depth slots: {', '.join(slots)}.",
        "",
        "Donor-side rows (Qwen3-4B, Qwen3-8B) are absent: those checkpoints are no longer",
        "resident on this host. See `protocol/atlas_models.json`.",
        "",
    ]

    lines += table(
        f"Representation similarity vs the reference base (mean cosine, {pooling} pooling)",
        ["pair"] + slots,
        [[pair] + [f"{d['mean_cosine']:.4f}" for d in data["depths"]]
         for pair, data in representation.items() if pair.startswith(f"{BASE}__vs__")],
    )
    lines += table(
        "Linear CKA vs the reference base",
        ["pair"] + slots,
        [[pair] + [f"{d['linear_cka']:.4f}" for d in data["depths"]]
         for pair, data in representation.items() if pair.startswith(f"{BASE}__vs__")],
    )
    lines += table(
        "Logit space",
        ["pair", "logit cosine", "symmetric KL", "top-1 agreement"],
        [[pair, f"{v['logit_cosine']:.4f}", f"{v['symmetric_kl']:.4f}", f"{v['top1_agreement']:.4f}"]
         for pair, v in logit_space.items()],
    )

    lines += ["## Representation-change alignment (the decisive table)", "",
              "For each model X: does `h_X - h_qwen06` point where `h_supa - h_qwen06` points?", ""]
    for name, data in alignment.items():
        lines += table(
            f"{name}",
            ["depth", "cos vs Supa change", "prompt-shuffled control", "random control",
             "||change|| / ||h||", "Supa ||change|| / ||h||"],
            [[d["slot"], f"{d['mean_cosine_vs_supa_change']:+.4f}",
              f"{d['control_prompt_shuffled']:+.4f}", f"{d['control_random_direction']:+.4f}",
              f"{d['change_relative_magnitude']:.4e}", f"{d['supa_change_relative_magnitude']:.4e}"]
             for d in data["depths"]],
        )

    atomic_write_text(args.output / "functional_atlas.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
