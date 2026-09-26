"""v3 E5 (analysis) — the cross-scale functional alignment atlas.

Donor width (2560 / 4096) differs from target width (1024), so raw cosine is
undefined across scales. The atlas therefore uses measures that do not depend on
a shared basis:

  * linear CKA over every (donor depth x target depth) pair — the depth x depth
    surface v1 and v2 never measured;
  * representational similarity analysis, correlating each model's prompt x
    prompt similarity matrix with every other model's;
  * logit-space agreement over the shared vocabulary, which *is* directly
    comparable because all five models use an identical tokenizer.

Every statistic is reported beside a prompt-shuffled control. That control is
what separates real per-prompt structure from a constant offset, and v3 E4
already showed the two can disagree.

The decision rule was frozen in amendment v3.1 before this ran.

Pure NumPy over cached summaries. Loads no model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, atomic_write_text  # noqa: E402

SEED = 17
CKA_LIFT_THRESHOLD = 0.10  # frozen in amendment v3.1


def centre(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x, y = centre(x.astype(np.float64)), centre(y.astype(np.float64))
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    left = np.linalg.norm(x.T @ x, ord="fro")
    right = np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / max(left * right, 1e-30))


def rsa(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation between the two models' prompt x prompt geometries."""
    def gram(z: np.ndarray) -> np.ndarray:
        z = centre(z.astype(np.float64))
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-30)
        return z @ z.T

    a, b = gram(x), gram(y)
    upper = np.triu_indices_from(a, k=1)
    va, vb = a[upper], b[upper]

    def rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(v), dtype=np.float64)
        return ranks

    ra, rb = rank(va), rank(vb)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / max(np.linalg.norm(ra) * np.linalg.norm(rb), 1e-30))


def log_softmax(v: np.ndarray) -> np.ndarray:
    shifted = v - v.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def logit_relationship(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    ca = a32 - a32.mean(axis=-1, keepdims=True)
    cb = b32 - b32.mean(axis=-1, keepdims=True)
    cosine = (ca * cb).sum(-1) / np.maximum(
        np.linalg.norm(ca, axis=-1) * np.linalg.norm(cb, axis=-1), 1e-30)
    la, lb = log_softmax(a32), log_softmax(b32)
    pa, pb = np.exp(la), np.exp(lb)
    skl = (pa * (la - lb)).sum(-1) + (pb * (lb - la)).sum(-1)
    return {
        "logit_cosine": float(cosine.mean()),
        "symmetric_kl": float(skl.mean()),
        "top1_agreement": float((a32.argmax(-1) == b32.argmax(-1)).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E5 cross-scale atlas.")
    parser.add_argument("--donors", type=Path,
                        default=LAB_ROOT / "local_models" / "v3_donor_activations")
    parser.add_argument("--targets", type=Path,
                        default=LAB_ROOT / "local_models" / "v3_activations")
    parser.add_argument("--pooling", default="last")
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "e5_cross_scale_atlas")
    args = parser.parse_args()

    started = time.perf_counter()
    donor_index = json.loads(
        (args.donors / f"donor_capture_index_{args.pooling}.json").read_text("utf-8"))
    target_index = json.loads((args.targets / "capture_index.json").read_text("utf-8"))

    models: dict[str, dict[str, Any]] = {}
    for record in donor_index["models"]:
        data = np.load(args.donors / f"{record['model']}_{args.pooling}.npz")
        models[record["model"]] = {
            "hidden": data["hidden"], "logits": data["logits"],
            "slots": record["capture_slots"], "role": "donor",
            "width": record["hidden_size"],
        }
    for record in target_index["models"]:
        if record["model"] not in ("qwen06", "supa"):
            continue
        data = np.load(args.targets / f"{record['model']}.npz")
        models[record["model"]] = {
            "hidden": data["hidden"], "logits": data["logits"],
            "slots": record["capture_slots"], "role": "target",
            "width": record["hidden_size"],
        }

    prompts = models["qwen06"]["hidden"].shape[0]
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(prompts)
    while np.any(permutation == np.arange(prompts)):
        permutation = rng.permutation(prompts)

    # --- depth x depth CKA surfaces ---------------------------------------
    pairs = [("qwen4b", "qwen06"), ("qwen8b", "qwen06"),
             ("qwen4b", "supa"), ("qwen8b", "supa"), ("qwen4b", "qwen8b")]
    surfaces: dict[str, Any] = {}
    for left, right in pairs:
        a, b = models[left], models[right]
        matrix, shuffled = [], []
        for i in range(len(a["slots"])):
            row, row_shuffled = [], []
            for j in range(len(b["slots"])):
                x, y = a["hidden"][:, i, :], b["hidden"][:, j, :]
                row.append(linear_cka(x, y))
                row_shuffled.append(linear_cka(x, y[permutation]))
            matrix.append(row)
            shuffled.append(row_shuffled)
        matched = [matrix[k][k] for k in range(min(len(a["slots"]), len(b["slots"])))]
        matched_shuffled = [shuffled[k][k] for k in range(len(matched))]
        surfaces[f"{left}__vs__{right}"] = {
            "donor_slots": a["slots"], "target_slots": b["slots"],
            "cka": matrix, "cka_prompt_shuffled": shuffled,
            "matched_relative_depth_cka": matched,
            "matched_relative_depth_cka_shuffled": matched_shuffled,
            "matched_lift": [m - s for m, s in zip(matched, matched_shuffled)],
            "rsa_matched": [rsa(a["hidden"][:, k, :], b["hidden"][:, k, :])
                            for k in range(len(matched))],
            "rsa_matched_shuffled": [rsa(a["hidden"][:, k, :], b["hidden"][:, k, :][permutation])
                                     for k in range(len(matched))],
        }

    # --- logit space over the shared vocabulary ---------------------------
    logit_pairs = {}
    names = list(models)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            logit_pairs[f"{left}__vs__{right}"] = logit_relationship(
                models[left]["logits"], models[right]["logits"])

    # --- frozen decision rule --------------------------------------------
    donor_target_lift = max(
        max(surfaces[f"{d}__vs__{t}"]["matched_lift"])
        for d in ("qwen4b", "qwen8b") for t in ("qwen06", "supa"))
    donor_donor_lift = max(surfaces["qwen4b__vs__qwen8b"]["matched_lift"])
    e6_authorised = donor_target_lift >= CKA_LIFT_THRESHOLD

    payload = {
        "schema_version": "1.0",
        "experiment": "v3_e5_cross_scale_functional_atlas",
        "protocol": "amendment_v3_1_donor_side.md 3.1.0-diagnostic",
        "gate0": donor_index["gate0"],
        "split": "calibration",
        "pooling": args.pooling,
        "prompts": prompts,
        "widths": {n: m["width"] for n, m in models.items()},
        "depth_surfaces": surfaces,
        "logit_space": logit_pairs,
        "decision": {
            "cka_lift_threshold": CKA_LIFT_THRESHOLD,
            "best_donor_target_matched_lift": donor_target_lift,
            "best_donor_donor_matched_lift": donor_donor_lift,
            "E6_authorised": bool(e6_authorised),
            "outcome": ("donor-target structure present; E6 authorised" if e6_authorised
                        else "outcome D: donor consensus is not portable to the target basis"),
        },
        "resources": {"wall_seconds": time.perf_counter() - started, "models_loaded": 0},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / f"cross_scale_atlas_{args.pooling}.json", payload)

    lines = [
        f"# v3 E5 — Cross-Scale Functional Alignment Atlas ({args.pooling} pooling)",
        "",
        f"{prompts} frozen calibration prompts, prompts only. Donors run through the",
        f"validated streaming forward (Gate 0 {donor_index['gate0']}, worst relative error",
        f"{donor_index['gate0_worst_relative_error']:.2e}). Widths:",
        "  " + ", ".join(f"{n} {m['width']}" for n, m in models.items()) + ".",
        "",
        "Cross-width cosine is undefined, so the atlas uses linear CKA and RSA, both",
        "basis-free, plus logit agreement over the shared vocabulary.",
        "",
    ]
    for pair, data in surfaces.items():
        lines += [f"## {pair} — depth x depth linear CKA", "",
                  "| donor depth \\ target depth | " + " | ".join(data["target_slots"]) + " |",
                  "|---" * (len(data["target_slots"]) + 1) + "|"]
        for i, slot in enumerate(data["donor_slots"]):
            lines.append(f"| {slot} | " + " | ".join(f"{v:.4f}" for v in data["cka"][i]) + " |")
        lines += ["", "Matched relative depth, against the prompt-shuffled control:", "",
                  "| depth | CKA | shuffled | lift | RSA | RSA shuffled |", "|---|---:|---:|---:|---:|---:|"]
        for k in range(len(data["matched_relative_depth_cka"])):
            lines.append(
                f"| {k} | {data['matched_relative_depth_cka'][k]:.4f} | "
                f"{data['matched_relative_depth_cka_shuffled'][k]:.4f} | "
                f"{data['matched_lift'][k]:+.4f} | {data['rsa_matched'][k]:+.4f} | "
                f"{data['rsa_matched_shuffled'][k]:+.4f} |")
        lines.append("")

    lines += ["## Logit space (shared vocabulary)", "",
              "| pair | logit cosine | symmetric KL | top-1 agreement |", "|---|---:|---:|---:|"]
    for pair, value in logit_pairs.items():
        lines.append(f"| {pair} | {value['logit_cosine']:.4f} | "
                     f"{value['symmetric_kl']:.4f} | {value['top1_agreement']:.4f} |")

    lines += ["", "## Frozen decision", "",
              f"- Best donor->target matched-depth CKA lift over control: **{donor_target_lift:+.4f}**",
              f"- Best donor->donor matched-depth CKA lift over control: **{donor_donor_lift:+.4f}**",
              f"- Threshold from amendment v3.1: {CKA_LIFT_THRESHOLD:+.2f}",
              "",
              f"**{payload['decision']['outcome']}**"]
    atomic_write_text(args.output / f"cross_scale_atlas_{args.pooling}.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
