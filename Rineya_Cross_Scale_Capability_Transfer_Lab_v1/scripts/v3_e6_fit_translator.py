"""v3 E6 — the low-rank cross-scale functional translator.

E5 showed donor and target residual streams share strong prompt-specific
structure at matched relative depth (CKA 0.82-0.96 against controls at 0.03-0.21)
while the transported *parameter* direction was orthogonal to everything
(+0.0002). E6 asks the question that decides between outcomes B and D:

  Can a small linear map carry donor representations into the target basis, and
  does the donor then explain anything the target's own base does not?

Two fits per matched depth:

  1. `z_target ~ A z_donor` by ridge and by orthogonal Procrustes, at ranks
     4/8/16 and full. Held-out R-squared on a **prompt-disjoint** split, because
     tokens inside a prompt are not independent.
  2. The incremental test: `z_supa ~ [z_qwen06, z_donor]` against
     `z_supa ~ z_qwen06`. If the donor adds nothing beyond the target's own base,
     there is nothing to transfer no matter how well (1) fits.

Amendment v3.1 froze the sample-size hazard and the holdout policy before this
ran. CPU least squares on cached activations; no gradient step, no training.
"""

from __future__ import annotations

import argparse
import gc
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
RANKS = (4, 8, 16)
RIDGE = 1e-2
# v1 fitted its only basis map on the vocabulary embeddings and applied it at
# every depth. These are its held-out cosines, the bar E6 has to clear.
V1_BASELINES = {"ridge_4b": 0.335, "procrustes_4b": 0.536, "procrustes_8b": 0.275}


def standardise(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True) + 1e-6
    return [(train - mean) / scale] + [(o - mean) / scale for o in others]


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += alpha * np.trace(gram) / gram.shape[0]
    return np.linalg.solve(gram, x.T @ y)


def procrustes_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    return u @ vt


def low_rank(matrix: np.ndarray, rank: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]


def scores(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    residual = ((prediction - truth) ** 2).sum()
    total = ((truth - truth.mean(axis=0, keepdims=True)) ** 2).sum()
    cosine = ((prediction * truth).sum(-1)
              / np.maximum(np.linalg.norm(prediction, axis=-1)
                           * np.linalg.norm(truth, axis=-1), 1e-30))
    return {"r2": float(1.0 - residual / max(total, 1e-30)),
            "mean_cosine": float(cosine.mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E6 translator fit.")
    parser.add_argument("--states", type=Path,
                        default=LAB_ROOT / "local_models" / "v3_token_states")
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "e6_functional_translator")
    parser.add_argument("--train-prompts", type=int, default=40)
    args = parser.parse_args()

    started = time.perf_counter()
    index = json.loads((args.states / "token_state_index.json").read_text("utf-8"))
    data = {r["model"]: np.load(args.states / f"{r['model']}.npz") for r in index["models"]}
    states = {name: blob["states"] for name, blob in data.items()}
    prompt_index = data["qwen06"]["prompt_index"]
    prompts = int(prompt_index.max()) + 1

    rng = np.random.default_rng(SEED)
    order = rng.permutation(prompts)
    train_prompts = set(order[: args.train_prompts].tolist())
    train_mask = np.isin(prompt_index, list(train_prompts))
    test_mask = ~train_mask
    depths = states["qwen06"].shape[1]

    results: list[dict[str, Any]] = []
    for donor in ("qwen4b", "qwen8b"):
        for target in ("qwen06", "supa"):
            for depth in range(depths):
                x_all = states[donor][:, depth, :].astype(np.float64)
                y_all = states[target][:, depth, :].astype(np.float64)
                x_tr, x_te = standardise(x_all[train_mask], x_all[test_mask])
                y_tr, y_te = standardise(y_all[train_mask], y_all[test_mask])

                row: dict[str, Any] = {
                    "donor": donor, "target": target, "depth": depth,
                    "train_tokens": int(train_mask.sum()), "test_tokens": int(test_mask.sum()),
                    "donor_width": x_all.shape[1], "target_width": y_all.shape[1],
                }
                full = ridge_fit(x_tr, y_tr, RIDGE)
                row["ridge_full"] = scores(x_te @ full, y_te)
                for rank in RANKS:
                    row[f"ridge_rank{rank}"] = scores(x_te @ low_rank(full, rank), y_te)
                orthogonal = procrustes_fit(x_tr, y_tr)
                row["procrustes"] = scores(x_te @ orthogonal, y_te)

                # Control: a donor whose token rows are shuffled. Holds all
                # marginal statistics and destroys the token correspondence.
                shuffled = rng.permutation(x_tr.shape[0])
                control = ridge_fit(x_tr[shuffled], y_tr, RIDGE)
                row["control_token_shuffled_ridge"] = scores(x_te @ control, y_te)

                results.append(row)
                print(f"  {donor}->{target} depth {depth}: ridge R2 "
                      f"{row['ridge_full']['r2']:+.4f} cos {row['ridge_full']['mean_cosine']:+.4f} | "
                      f"rank8 R2 {row['ridge_rank8']['r2']:+.4f} | "
                      f"shuffled R2 {row['control_token_shuffled_ridge']['r2']:+.4f}", flush=True)
                del x_all, y_all, x_tr, x_te, y_tr, y_te, full, orthogonal, control
                gc.collect()

    # --- the incremental test: does the donor add anything over the base? ---
    incremental = []
    for donor in ("qwen4b", "qwen8b"):
        for depth in range(depths):
            base = states["qwen06"][:, depth, :].astype(np.float64)
            extra = states[donor][:, depth, :].astype(np.float64)
            truth = states["supa"][:, depth, :].astype(np.float64)
            b_tr, b_te = standardise(base[train_mask], base[test_mask])
            e_tr, e_te = standardise(extra[train_mask], extra[test_mask])
            t_tr, t_te = standardise(truth[train_mask], truth[test_mask])

            base_only = scores(b_te @ ridge_fit(b_tr, t_tr, RIDGE), t_te)
            joint = scores(np.hstack([b_te, e_te])
                           @ ridge_fit(np.hstack([b_tr, e_tr]), t_tr, RIDGE), t_te)
            # Width-matched control: pad the base with random features so the
            # joint fit's extra columns cannot flatter it by dimension alone.
            noise_tr = rng.standard_normal(e_tr.shape)
            noise_te = rng.standard_normal(e_te.shape)
            padded = scores(np.hstack([b_te, noise_te])
                            @ ridge_fit(np.hstack([b_tr, noise_tr]), t_tr, RIDGE), t_te)
            incremental.append({
                "donor": donor, "depth": depth,
                "base_only_r2": base_only["r2"], "base_plus_donor_r2": joint["r2"],
                "base_plus_random_r2": padded["r2"],
                "donor_increment": joint["r2"] - base_only["r2"],
                "random_increment": padded["r2"] - base_only["r2"],
                "increment_over_control": (joint["r2"] - base_only["r2"])
                                          - (padded["r2"] - base_only["r2"]),
            })
            print(f"  incremental {donor} depth {depth}: base {base_only['r2']:+.4f} "
                  f"-> +donor {joint['r2']:+.4f} (+{joint['r2'] - base_only['r2']:.4f}) "
                  f"vs +random {padded['r2']:+.4f}", flush=True)
            del base, extra, truth, b_tr, b_te, e_tr, e_te, t_tr, t_te
            gc.collect()

    best_translator = max(r["ridge_full"]["mean_cosine"] for r in results)
    best_increment = max(r["increment_over_control"] for r in incremental)

    payload = {
        "schema_version": "1.0",
        "experiment": "v3_e6_low_rank_functional_translator",
        "protocol": "amendment_v3_1_donor_side.md 3.1.0-diagnostic",
        "split": "calibration", "holdout": "prompt-disjoint",
        "train_prompts": args.train_prompts, "test_prompts": prompts - args.train_prompts,
        "total_tokens": int(prompt_index.shape[0]),
        "ridge_alpha": RIDGE, "ranks": list(RANKS), "seed": SEED,
        "v1_embedding_fitted_baselines": V1_BASELINES,
        "translator": results,
        "incremental": incremental,
        "summary": {
            "best_heldout_mean_cosine": best_translator,
            "v1_best_baseline_cosine": max(V1_BASELINES.values()),
            "beats_v1_baseline": best_translator > max(V1_BASELINES.values()),
            "best_donor_increment_over_control": best_increment,
        },
        "resources": {"wall_seconds": time.perf_counter() - started, "models_loaded": 0},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "translator.json", payload)

    lines = [
        "# v3 E6 — Low-Rank Cross-Scale Functional Translator",
        "",
        f"{payload['total_tokens']} token positions, {args.train_prompts} training prompts and",
        f"{prompts - args.train_prompts} held-out prompts. The holdout is prompt-disjoint, not",
        "token-disjoint, because tokens inside a prompt are not independent.",
        "",
        "## Can a linear map carry donor representations into the target basis?",
        "",
        "| donor -> target | depth | ridge R2 | ridge cos | rank-8 R2 | rank-16 R2 | Procrustes cos | token-shuffled R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['donor']} -> {row['target']} | {row['depth']} | "
            f"{row['ridge_full']['r2']:+.4f} | {row['ridge_full']['mean_cosine']:+.4f} | "
            f"{row['ridge_rank8']['r2']:+.4f} | {row['ridge_rank16']['r2']:+.4f} | "
            f"{row['procrustes']['mean_cosine']:+.4f} | "
            f"{row['control_token_shuffled_ridge']['r2']:+.4f} |")
    lines += [
        "",
        f"Best held-out mean cosine: **{best_translator:.4f}**. v1's embedding-fitted maps reached",
        f"{V1_BASELINES['ridge_4b']} (ridge) and {V1_BASELINES['procrustes_4b']} / "
        f"{V1_BASELINES['procrustes_8b']} (Procrustes).",
        "",
        "## Does the donor explain anything the target's own base does not?",
        "",
        "Predicting Supa's representation from Qwen3-0.6B alone, versus from Qwen3-0.6B plus the",
        "donor. The random column adds the same number of features as the donor, so any gain from",
        "sheer dimensionality is subtracted.",
        "",
        "| donor | depth | base only R2 | + donor | + random | donor gain | gain over control |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in incremental:
        lines.append(
            f"| {row['donor']} | {row['depth']} | {row['base_only_r2']:+.4f} | "
            f"{row['base_plus_donor_r2']:+.4f} | {row['base_plus_random_r2']:+.4f} | "
            f"{row['donor_increment']:+.4f} | {row['increment_over_control']:+.4f} |")
    lines += ["", f"Best donor increment over the width-matched control: "
                  f"**{best_increment:+.4f}**."]
    atomic_write_text(args.output / "translator.md", "\n".join(lines) + "\n")
    print("\n".join(lines[-14:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
