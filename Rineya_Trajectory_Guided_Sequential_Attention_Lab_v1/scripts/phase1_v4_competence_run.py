"""Stage 5: produce the FROZEN competence report for the selected v4 regime.

The pilot screens regimes; this produces the gate artifact. It records every
input the gate needs — including per-family accuracy and distinct-prediction
counts, which the pilot does not collect — and writes a machine-readable
GO / NO_GO that the confirmatory runner consumes.

Development-only. The regime was selected arm-blind, on the minimum held-out
accuracy across all three arms, before this ran.

## Resume

A nine-cell run at this scale is four hours, and it has twice been lost whole --
once to GPU contention, once to a machine reboot -- because results existed only
in memory until the final write. Each (arm, seed) cell is therefore written to
its own file the moment it finishes, and a restart skips the cells it already
has.

Reuse is fail-closed. Every cell records a fingerprint of the regime that
produced it: band, data volume, update budget, model shape, optimizer settings,
split seeds, and split generation order. A cell whose fingerprint does not match
the current invocation is REFUSED rather than skipped or silently mixed in,
because a competence report assembled from cells measured under two different
regimes would certify neither of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
from collections import Counter

LAB = pathlib.Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

import torch

from tgsa.phase1_competence import (
    ArmCompetence,
    CompetenceThresholds,
    evaluate_competence,
)
from tgsa.phase1_data import (
    SyntheticVocabulary,
    derive_seed,
    deterministic_batch_stream,
    fixed_batches,
    generate_split,
)
from tgsa.phase1_model import TinyLMConfig, answer_logits, build_model, masked_answer_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = ("cascade", "history_control", "learned_delta")
VOCAB = SyntheticVocabulary(domain_size=32, max_hops=8, max_abs_step=8)
PRIMARY = ("pointer_chase", "state_update")

BANDS = {
  "easy_nontrivial": {
    "direct_lookup": {"min_nodes": 5, "max_nodes": 8, "min_hops": 1, "max_hops": 1},
    "pointer_chase": {"min_nodes": 5, "max_nodes": 8, "min_hops": 2, "max_hops": 2,
                      "reject_trivial": True},
    "state_update": {"min_steps": 2, "max_steps": 3, "reject_trivial": True},
  },
  "medium": {
    "direct_lookup": {"min_nodes": 8, "max_nodes": 12, "min_hops": 1, "max_hops": 1},
    "pointer_chase": {"min_nodes": 8, "max_nodes": 12, "min_hops": 2, "max_hops": 3,
                      "reject_trivial": True},
    "state_update": {"min_steps": 3, "max_steps": 4, "reject_trivial": True},
  },
}


# v2 adds the compute backend. A v1 cell recorded none, so it cannot be checked
# and is refused rather than guessed at.
CELL_SCHEMA_VERSION = "tgsa.phase1-competence-cell.v2"


def compute_backend() -> dict[str, object]:
    """What actually executed the measurement.

    Twenty thousand optimizer steps through a different BLAS backend is a
    different measurement. This is the one input to `train_one` that comes from
    the environment rather than from an argument, so it is the one the caller
    cannot see in the command line.
    """

    backend: dict[str, object] = {
        "device_type": DEVICE.type,
        "torch": torch.__version__,
    }
    if DEVICE.type == "cuda":
        try:
            backend["gpu_name"] = torch.cuda.get_device_name(DEVICE)
            backend["capability"] = list(torch.cuda.get_device_capability(DEVICE))
        except Exception as error:  # noqa: BLE001 - provenance must not be fatal
            backend["gpu_name"] = f"unavailable: {type(error).__name__}"
    return backend


def regime_fingerprint(args, train_seed: int) -> str:
    """Everything that changes what a cell measures.

    A cell may only be reused by an invocation that would have produced it.
    Seeds are deliberately EXCLUDED: adding seed 4 later must not invalidate the
    cells for 1, 2 and 3, and each cell already records which seed it is. Every
    other knob is included, because changing any of them changes the measurement
    -- including the compute backend, which is the only one that comes from the
    environment rather than the command line and so is the one a caller cannot
    see going wrong.
    """

    payload = {
        # The backend is part of what a cell measures, not merely provenance.
        "backend": compute_backend(),
        "band": args.band,
        "band_spec": BANDS[args.band],
        "train_examples": args.train_examples,
        "holdout": args.holdout,
        "steps": args.steps,
        "embed_dim": args.embed_dim,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_split_seed": train_seed,
        "holdout_split_seed": args.holdout_split_seed,
        "split_generation_order": ["holdout", "train"],
        "vocabulary": VOCAB.to_dict(),
        "schema": CELL_SCHEMA_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_cell_atomically(path: pathlib.Path, payload: dict) -> None:
    """A half-written cell is worse than a missing one, because it gets reused."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def load_cell(path: pathlib.Path, fingerprint: str, arm: str, seed: int):
    """Return a reusable cell, or None if absent. Refuse anything ambiguous."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "{0} is not valid JSON ({1}); delete it and re-measure rather than "
            "guessing what it contained".format(path.name, error)
        ) from error
    if payload.get("schema_version") != CELL_SCHEMA_VERSION:
        raise ValueError(
            "{0} has schema {1!r}, expected {2!r}; delete the cell directory and "
            "re-measure".format(path.name, payload.get("schema_version"),
                                CELL_SCHEMA_VERSION)
        )
    if payload.get("regime_fingerprint") != fingerprint:
        recorded = payload.get("backend")
        current = compute_backend()
        detail = ""
        if isinstance(recorded, dict) and recorded != current:
            detail = f" Its backend was {recorded}, this run's is {current}."
        raise ValueError(
            "{0} was measured under a different regime ({1} != {2}).{3} A competence "
            "report assembled from cells measured under two regimes certifies "
            "neither. Use a fresh --cell-dir, or delete this one".format(
                path.name, str(payload.get("regime_fingerprint"))[:12],
                fingerprint[:12], detail)
        )
    cell = payload.get("run")
    if not isinstance(cell, dict):
        raise ValueError("{0} has no run payload".format(path.name))
    if cell.get("arm") != arm or cell.get("seed") != seed:
        raise ValueError(
            "{0} holds {1!r}/{2!r} but is named for {3!r}/{4!r}; the cell "
            "directory is inconsistent".format(
                path.name, cell.get("arm"), cell.get("seed"), arm, seed)
        )
    return cell


def make_split(band: str, count: int, seed: int, forbidden=None):
    per_family = max(1, count // 3)
    return generate_split(
        split="train",
        count_by_family={"direct_lookup": per_family, "pointer_chase": per_family,
                         "state_update": per_family},
        dataset_seed=seed, vocab=VOCAB, generation_spec={"families": BANDS[band]},
        forbidden_signatures=forbidden,
    )


@torch.no_grad()
def measure(model, examples, max_len: int) -> dict:
    usable = [e for e in examples if len(e.tokens) - 1 <= max_len]
    correct = total = 0
    nll = 0.0
    predicted: list[int] = []
    per_family_hit: dict[str, list[float]] = {}
    for batch in fixed_batches(usable, batch_size=256):
        gpu = batch.to(DEVICE)
        out = model(gpu.input_ids, key_padding_mask=gpu.key_padding_mask)
        logits = answer_logits(out.logits, gpu.answer_input_indices).float()
        hits = (logits.argmax(-1) == gpu.answer_tokens)
        correct += int(hits.sum())
        nll += float(-torch.log_softmax(logits, -1).gather(1, gpu.answer_tokens[:, None]).sum())
        predicted.extend(logits.argmax(-1).tolist())
        for index, family in enumerate(batch.families):
            per_family_hit.setdefault(family, []).append(float(hits[index]))
        total += len(gpu.answer_tokens)
    return {
        "accuracy": correct / total,
        "answer_nll": nll / total,
        "distinct_predictions": len(set(predicted)),
        "per_family_accuracy": {k: sum(v) / len(v) for k, v in sorted(per_family_hit.items())},
        "examples": total,
    }


def train_one(arm: str, train, holdout, seed: int, max_len: int, steps: int,
              embed_dim: int, layers: int, heads: int, batch_size: int, lr: float) -> dict:
    config = TinyLMConfig(
        vocab_size=VOCAB.vocab_size, max_sequence_length=max_len, embed_dim=embed_dim,
        num_heads=heads, num_layers=layers, ffn_multiplier=4.0, dropout=0.0, bias=False,
        trajectory_rank=16, arm=arm, tie_embeddings=True,
    )
    model = build_model(config, initialization_seed=derive_seed(seed, "model-initialization")).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    stream = deterministic_batch_stream(train, batch_size=batch_size, seed=derive_seed(seed, "training-order"))
    warmup = max(1, steps // 20)
    model.train()
    curve: list[tuple[int, float]] = []
    for step in range(steps):
        for group in optimizer.param_groups:
            if step < warmup:
                group["lr"] = lr * (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, steps - warmup)
                group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * progress))
        _, batch = next(stream)
        batch = batch.to(DEVICE)
        out = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
        loss = masked_answer_loss(out.logits, batch.labels, batch.loss_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            curve.append((step, float(loss.detach())))
    model.eval()
    seen = measure(model, train[:1500], max_len)
    held = measure(model, holdout, max_len)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"arm": arm, "seed": seed, "seen": seen, "holdout": held,
            "final_train_loss": curve[-1][1], "loss_curve": curve}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--train-examples", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=int, default=3000)
    parser.add_argument("--train-split-seed", type=int, default=1_000_000)
    parser.add_argument("--holdout-split-seed", type=int, default=987_654_321)
    parser.add_argument("--band", default="easy_nontrivial")
    parser.add_argument("--cell-dir", type=pathlib.Path, default=None,
                        help="per-cell results for resume; defaults to <output>.cells")
    parser.add_argument("--no-resume", action="store_true",
                        help="recompute every cell even if one is already on disk")
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]

    # Explicit, recorded, process-independent. The pilot derived this from
    # builtin hash(), which is salted per process, so its exact dataset draw
    # was not reproducible from its own record. This one is.
    train_seed = derive_seed(args.train_split_seed, "pilot-train-split", args.band)
    # The HOLDOUT is generated first, then train excludes it. Each split excludes
    # the signatures already taken, so whichever is generated first is the only
    # unbiased draw. Generating train first consumed the entire 7,680-item
    # non-trivial two-step `state_update` space and left the holdout with
    # three-step items only -- a competence measurement on a harder distribution
    # than the confirmatory test set it is certifying.
    holdout = make_split(args.band, args.holdout, seed=args.holdout_split_seed)
    train = make_split(args.band, args.train_examples, seed=train_seed,
                       forbidden={e.latent_signature for e in holdout})
    assert not ({e.latent_signature for e in train}
                & {e.latent_signature for e in holdout}), "train/holdout leak"
    max_len = max(len(e.tokens) - 1 for e in train + holdout)

    train_major = Counter(e.answer_token for e in train).most_common(1)[0][0]
    answers = [e.answer_token for e in holdout]
    trivial = {
        "uniform": 1.0 / VOCAB.domain_size,
        "eval_majority": Counter(answers).most_common(1)[0][1] / len(answers),
        "train_majority": sum(1 for a in answers if a == train_major) / len(answers),
    }
    print(f"device={DEVICE}  train={len(train)}  holdout={len(holdout)}  max_len={max_len}", flush=True)
    print(f"trivial baselines: {trivial}", flush=True)

    cell_dir = args.cell_dir or args.output.with_name(args.output.stem + ".cells")
    fingerprint = regime_fingerprint(args, train_seed)
    print(f"cells: {cell_dir}  fingerprint={fingerprint[:12]}", flush=True)

    runs = []
    resumed: list[str] = []
    for arm in ARMS:
        for seed in seeds:
            cell_path = cell_dir / f"{arm}__seed{seed}.json"
            cached = None if args.no_resume else load_cell(cell_path, fingerprint, arm, seed)
            if cached is not None:
                runs.append(cached)
                resumed.append(f"{arm}:{seed}")
                print(f"  {arm:16s} seed={seed} "
                      f"holdout={cached['holdout']['accuracy']:.4f} "
                      f"seen={cached['seen']['accuracy']:.4f} "
                      f"nll={cached['holdout']['answer_nll']:.3f}  [resumed]", flush=True)
                continue
            result = train_one(arm, train, holdout, seed, max_len, args.steps,
                               args.embed_dim, args.layers, args.heads,
                               args.batch_size, args.lr)
            write_cell_atomically(cell_path, {
                "schema_version": CELL_SCHEMA_VERSION,
                "regime_fingerprint": fingerprint,
                # Recorded as well as fingerprinted, so a refusal can say WHAT
                # differs instead of only that the digests disagree.
                "backend": compute_backend(),
                "run": result,
            })
            runs.append(result)
            print(f"  {arm:16s} seed={seed} holdout={result['holdout']['accuracy']:.4f} "
                  f"seen={result['seen']['accuracy']:.4f} nll={result['holdout']['answer_nll']:.3f}",
                  flush=True)

    arms = []
    for arm in ARMS:
        subset = [r for r in runs if r["arm"] == arm]
        families = sorted({f for r in subset for f in r["holdout"]["per_family_accuracy"]})
        arms.append(ArmCompetence(
            arm=arm,
            holdout_accuracy=statistics.mean(r["holdout"]["accuracy"] for r in subset),
            seen_accuracy=statistics.mean(r["seen"]["accuracy"] for r in subset),
            answer_nll=statistics.mean(r["holdout"]["answer_nll"] for r in subset),
            distinct_predictions=int(min(r["holdout"]["distinct_predictions"] for r in subset)),
            per_family_accuracy={
                f: statistics.mean(r["holdout"]["per_family_accuracy"].get(f, 0.0) for r in subset)
                for f in families
            },
            per_seed_holdout_accuracy=[r["holdout"]["accuracy"] for r in subset],
        ))

    thresholds = CompetenceThresholds(
        minimum_margin_over_trivial=0.030,
        minimum_absolute_accuracy=0.070,
        maximum_answer_nll_fraction_of_uniform=0.97,
        minimum_distinct_predictions=4,
        maximum_family_accuracy=0.95,
        maximum_generalization_gap=0.10,
        maximum_seed_coefficient_of_variation=0.35,
        minimum_seed_runs=3,
    )
    report = evaluate_competence(
        arms, thresholds=thresholds, trivial_baselines=trivial,
        uniform_answer_nll=math.log(VOCAB.domain_size), primary_families=PRIMARY,
    )
    report["regime"] = {
        "band": args.band, "band_spec": BANDS[args.band],
        "train_examples": args.train_examples, "steps": args.steps,
        "embed_dim": args.embed_dim, "num_layers": args.layers, "num_heads": args.heads,
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "development_seeds": seeds,
        "train_split_seed": train_seed,
        "holdout_split_seed": args.holdout_split_seed,
        "holdout_examples": len(holdout),
        "split_generation_order": ["holdout", "train"],
        "regime_fingerprint": fingerprint,
        # Every other artifact in this repository records the backend -- the
        # screening pilot, the study runner, the doctor. The one artifact that
        # GATES a multi-hour confirmatory matrix, and gets SHA-256 bound into the
        # registered config, did not.
        "backend": compute_backend(),
    }
    report["runs"] = runs
    report["provenance"] = {
        "cell_dir": str(cell_dir),
        "resumed_cells": sorted(resumed),
        "computed_cells": sorted(
            key for key in (f"{r['arm']}:{r['seed']}" for r in runs)
            if key not in set(resumed)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    print(f"VERDICT: {report['verdict']}")
    for name, ok in report["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for reason in report["reasons"]:
        print(f"  reason: {reason}")
    print(f"wrote {args.output}")
    return 0 if report["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
