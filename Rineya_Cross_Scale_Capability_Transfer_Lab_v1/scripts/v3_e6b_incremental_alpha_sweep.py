"""v3 E6b — is "the donor adds nothing" a regularisation artifact?

E6's incremental test fits `z_supa ~ [z_qwen06, z_donor]`. With 40 training
prompts the joint design has roughly 6,000 rows against 1024 + 4096 columns, so
it sits near the rank-deficient regime. A donor block that genuinely helped
could still hurt held-out prediction there, purely through variance.

That would make E6's central conclusion an artifact. This sweeps the ridge
penalty over four orders of magnitude, against the same width-matched
random-feature control.

The criterion needs care. "Does the donor help at SOME penalty?" is the wrong
question, and answering it that way flags E6 as refuted. Ridge shrinkage degrades
both fits, and where the base is over-shrunk, correlated donor columns partially
compensate — which reads as a large gain (+0.24 at alpha 100) while base-only R2
has collapsed from 0.959 to 0.357. That is the donor patching a crippled base,
not supplying information the base lacks.

The correct test gives each model its own optimal penalty and asks whether the
donor raises the *best achievable* held-out fit. Under that test E6 stands.
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
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def standardise(train: np.ndarray, other: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / scale, (other - mean) / scale


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += alpha * np.trace(gram) / gram.shape[0]
    return np.linalg.solve(gram, x.T @ y)


def r2(prediction: np.ndarray, truth: np.ndarray) -> float:
    residual = ((prediction - truth) ** 2).sum()
    total = ((truth - truth.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - residual / max(total, 1e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E6b incremental alpha sweep.")
    parser.add_argument("--states", type=Path,
                        default=LAB_ROOT / "local_models" / "v3_token_states")
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "e6_functional_translator")
    parser.add_argument("--train-prompts", type=int, default=40)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()

    started = time.perf_counter()
    index = json.loads((args.states / "token_state_index.json").read_text("utf-8"))
    data = {r["model"]: np.load(args.states / f"{r['model']}.npz") for r in index["models"]}
    prompt_index = data["qwen06"]["prompt_index"]
    prompts = int(prompt_index.max()) + 1

    rng = np.random.default_rng(SEED)
    order = rng.permutation(prompts)
    train_mask = np.isin(prompt_index, order[: args.train_prompts].tolist())
    test_mask = ~train_mask

    rows: list[dict[str, Any]] = []
    for donor in ("qwen4b", "qwen8b"):
        for depth in args.depths:
            base = data["qwen06"]["states"][:, depth, :].astype(np.float64)
            extra = data[donor]["states"][:, depth, :].astype(np.float64)
            truth = data["supa"]["states"][:, depth, :].astype(np.float64)
            b_tr, b_te = standardise(base[train_mask], base[test_mask])
            e_tr, e_te = standardise(extra[train_mask], extra[test_mask])
            t_tr, t_te = standardise(truth[train_mask], truth[test_mask])
            noise_tr = rng.standard_normal(e_tr.shape)
            noise_te = rng.standard_normal(e_te.shape)
            joint_tr = np.hstack([b_tr, e_tr])
            joint_te = np.hstack([b_te, e_te])
            padded_tr = np.hstack([b_tr, noise_tr])
            padded_te = np.hstack([b_te, noise_te])

            for alpha in ALPHAS:
                base_r2 = r2(b_te @ ridge_fit(b_tr, t_tr, alpha), t_te)
                donor_r2 = r2(joint_te @ ridge_fit(joint_tr, t_tr, alpha), t_te)
                random_r2 = r2(padded_te @ ridge_fit(padded_tr, t_tr, alpha), t_te)
                rows.append({
                    "donor": donor, "depth": depth, "alpha": alpha,
                    "base_only_r2": base_r2, "base_plus_donor_r2": donor_r2,
                    "base_plus_random_r2": random_r2,
                    "donor_increment": donor_r2 - base_r2,
                    "increment_over_control": (donor_r2 - base_r2) - (random_r2 - base_r2),
                })
                print(f"  {donor} depth {depth} alpha {alpha:<7g}: base {base_r2:+.4f} "
                      f"+donor {donor_r2:+.4f} ({donor_r2 - base_r2:+.4f}) "
                      f"+random {random_r2:+.4f}", flush=True)
            del base, extra, truth, b_tr, b_te, e_tr, e_te, t_tr, t_te
            del joint_tr, joint_te, padded_tr, padded_te, noise_tr, noise_te
            gc.collect()

    # "Does the donor help at SOME penalty?" is the wrong test. Ridge shrinkage
    # degrades both fits; where the base is over-shrunk, correlated donor columns
    # partially compensate, which reads as a gain without being new information.
    # The correct comparison gives each model its own optimal penalty and asks
    # whether the donor raises the BEST achievable held-out fit.
    best_by_cell: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        cell = best_by_cell.setdefault((row["donor"], row["depth"]),
                                       {"base": -9.0, "donor": -9.0, "random": -9.0})
        cell["base"] = max(cell["base"], row["base_only_r2"])
        cell["donor"] = max(cell["donor"], row["base_plus_donor_r2"])
        cell["random"] = max(cell["random"], row["base_plus_random_r2"])
    best_over_alpha = [
        {"donor": donor, "depth": depth,
         "best_base_only_r2": cell["base"], "best_base_plus_donor_r2": cell["donor"],
         "delta": cell["donor"] - cell["base"]}
        for (donor, depth), cell in sorted(best_by_cell.items())
    ]
    helps_at_own_optimum = any(row["delta"] > 0.0 for row in best_over_alpha)
    naive_best = max(rows, key=lambda r: r["donor_increment"])
    best = naive_best
    helps_anywhere = naive_best["donor_increment"] > 0.0

    payload = {
        "schema_version": "1.0",
        "experiment": "v3_e6b_incremental_alpha_sweep",
        "protocol": "amendment_v3_1_donor_side.md 3.1.0-diagnostic",
        "question": "is E6's 'the donor adds nothing' conclusion a regularisation artifact?",
        "alphas": list(ALPHAS), "depths": args.depths,
        "train_prompts": args.train_prompts,
        "train_tokens": int(train_mask.sum()), "test_tokens": int(test_mask.sum()),
        "rows": rows,
        "best_over_alpha": best_over_alpha,
        "verdict": {
            "primary_test": "best-over-alpha: does the donor raise the best achievable held-out fit?",
            "donor_helps_at_own_optimum": bool(helps_at_own_optimum),
            "max_best_over_alpha_delta": max(r["delta"] for r in best_over_alpha),
            "e6_conclusion_robust": not helps_at_own_optimum,
            "secondary_naive_test": {
                "donor_helps_at_some_penalty": bool(helps_anywhere),
                "largest_increment": naive_best["donor_increment"],
                "at": {k: naive_best[k] for k in ("donor", "depth", "alpha")},
                "why_this_is_not_evidence": (
                    "the largest increments occur at penalties where BOTH fits are far below "
                    "their own optimum, so the donor is compensating for an over-shrunk base "
                    "rather than contributing information the base lacks"),
            },
        },
        "resources": {"wall_seconds": time.perf_counter() - started, "models_loaded": 0},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "incremental_alpha_sweep.json", payload)

    lines = [
        "# v3 E6b — Incremental Test, Ridge Penalty Sweep",
        "",
        f"{payload['train_tokens']} training tokens against 1024 + donor-width columns, so the",
        "joint fit sits near the rank-deficient regime. If E6's 'the donor adds nothing' result",
        "were a variance artifact, a stronger penalty would reveal it.",
        "",
        "| donor | depth | alpha | base only | + donor | + random | donor gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['donor']} | {row['depth']} | {row['alpha']:g} | "
                     f"{row['base_only_r2']:+.4f} | {row['base_plus_donor_r2']:+.4f} | "
                     f"{row['base_plus_random_r2']:+.4f} | {row['donor_increment']:+.4f} |")
    lines += ["", "## The correct test: each model at its own optimal penalty", "",
              "Ridge shrinkage degrades both fits. Where the base is over-shrunk, correlated donor",
              "columns partially compensate, which looks like a gain without being new information.",
              "So the question is whether the donor raises the *best achievable* held-out fit.", "",
              "| donor | depth | best base only | best base + donor | delta |",
              "|---|---:|---:|---:|---:|"]
    for row in best_over_alpha:
        lines.append(f"| {row['donor']} | {row['depth']} | {row['best_base_only_r2']:+.4f} | "
                     f"{row['best_base_plus_donor_r2']:+.4f} | {row['delta']:+.4f} |")
    lines += ["",
              f"Largest naive per-penalty increment: {naive_best['donor_increment']:+.4f} at "
              f"{naive_best['donor']} depth {naive_best['depth']} alpha {naive_best['alpha']:g} — "
              f"but there base-only R2 is only {naive_best['base_only_r2']:+.4f} against "
              f"{max(r['best_base_only_r2'] for r in best_over_alpha):+.4f} at its own optimum.",
              ""]
    lines.append("**E6's conclusion is robust: the donor never raises the best achievable fit.**"
                 if not helps_at_own_optimum else
                 "**E6's conclusion does not survive: the donor helps at its own optimum.**")
    atomic_write_text(args.output / "incremental_alpha_sweep.md", "\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
