"""Stages 1+2: positive controls and the development-only capacity/budget pilot.

Development only. Nothing here touches a confirmatory bank, and the selection
rule is ARM-BLIND: configurations are scored by the MINIMUM absolute competence
across all three arms, never by a between-arm difference.

The v3 postmortem showed pure memorization -- train 0.236, in-band held-out
0.045 against a 0.041 majority baseline -- so the primary pilot dimension is
training-set size, not model width. More capacity on 5,000 examples would
memorise harder, not generalise.

Every run reports absolute competence against the registered trivial baselines,
so a configuration cannot look good merely by being compared to another arm.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass

LAB = pathlib.Path(r"C:\Users\clclc\BusinessProject\Chat-Bubble-v4\Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1")
sys.path.insert(0, str(LAB))

import torch

from tgsa.phase1_data import (
    SyntheticVocabulary,
    collate_examples,
    derive_seed,
    deterministic_batch_stream,
    fixed_batches,
    generate_split,
)
from tgsa.phase1_model import TinyLMConfig, answer_logits, build_model, masked_answer_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = ("cascade", "history_control", "learned_delta")


@dataclass(frozen=True, slots=True)
class Regime:
    """One development configuration. Recorded before execution."""

    name: str
    train_examples: int
    steps: int
    embed_dim: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float
    band: str  # which difficulty band the generator uses
    trajectory_rank: int = 16
    ffn_multiplier: float = 4.0


BANDS: dict[str, dict] = {
    # Genuinely nontrivial: fixed-point / cancellation shortcuts are rejected.
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
    "hard": {
        "direct_lookup": {"min_nodes": 10, "max_nodes": 20, "min_hops": 1, "max_hops": 1},
        "pointer_chase": {"min_nodes": 10, "max_nodes": 20, "min_hops": 2, "max_hops": 4,
                          "reject_trivial": True},
        "state_update": {"min_steps": 3, "max_steps": 7, "reject_trivial": True},
    },
}

VOCAB = SyntheticVocabulary(domain_size=32, max_hops=8, max_abs_step=8)


def make_split(band: str, count: int, seed: int, forbidden: set[str] | None = None):
    per_family = max(1, count // 3)
    return generate_split(
        split="train",
        count_by_family={"direct_lookup": per_family, "pointer_chase": per_family,
                         "state_update": per_family},
        dataset_seed=seed,
        vocab=VOCAB,
        generation_spec={"families": BANDS[band]},
        forbidden_signatures=forbidden,
    )


def baselines(train, evaluation) -> dict[str, float]:
    train_major = Counter(e.answer_token for e in train).most_common(1)[0][0]
    answers = [e.answer_token for e in evaluation]
    return {
        "uniform": 1.0 / VOCAB.domain_size,
        "eval_majority": Counter(answers).most_common(1)[0][1] / len(answers),
        "train_majority": sum(1 for a in answers if a == train_major) / len(answers),
    }


@torch.no_grad()
def evaluate(model, examples, max_len: int) -> tuple[float, float]:
    usable = [e for e in examples if len(e.tokens) - 1 <= max_len]
    correct = total = 0
    nll = 0.0
    for batch in fixed_batches(usable, batch_size=256):
        batch = batch.to(DEVICE)
        out = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
        logits = answer_logits(out.logits, batch.answer_input_indices).float()
        correct += int((logits.argmax(-1) == batch.answer_tokens).sum())
        nll += float(-torch.log_softmax(logits, -1).gather(1, batch.answer_tokens[:, None]).sum())
        total += len(batch.answer_tokens)
    return correct / max(total, 1), nll / max(total, 1)


def train_arm(regime: Regime, arm: str, train, holdout, seed: int, max_len: int) -> dict:
    config = TinyLMConfig(
        vocab_size=VOCAB.vocab_size, max_sequence_length=max_len,
        embed_dim=regime.embed_dim, num_heads=regime.num_heads,
        num_layers=regime.num_layers, ffn_multiplier=regime.ffn_multiplier,
        dropout=0.0, bias=False, trajectory_rank=regime.trajectory_rank,
        arm=arm, tie_embeddings=True,
    )
    model = build_model(config, initialization_seed=derive_seed(seed, "model-initialization")).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=regime.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.01)
    total = sum(p.numel() for p in model.parameters())
    stream = deterministic_batch_stream(train, batch_size=regime.batch_size,
                                        seed=derive_seed(seed, "training-order"))
    warmup = max(1, regime.steps // 20)
    model.train()
    losses = []
    for step in range(regime.steps):
        for group in optimizer.param_groups:
            if step < warmup:
                group["lr"] = regime.learning_rate * (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, regime.steps - warmup)
                group["lr"] = regime.learning_rate * (0.5 * (1 + math.cos(math.pi * progress)))
        _, batch = next(stream)
        batch = batch.to(DEVICE)
        out = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
        loss = masked_answer_loss(out.logits, batch.labels, batch.loss_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % max(1, regime.steps // 10) == 0 or step == regime.steps - 1:
            losses.append((step, float(loss)))

    model.eval()
    seen_accuracy, _ = evaluate(model, train[:1000], max_len)
    holdout_accuracy, holdout_nll = evaluate(model, holdout, max_len)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "arm": arm, "seed": seed, "parameters": total,
        "train_seen_accuracy": seen_accuracy,
        "holdout_accuracy": holdout_accuracy,
        "holdout_nll": holdout_nll,
        "loss_curve": losses,
    }


def run_regime(regime: Regime, seeds: list[int]) -> dict:
    # Not builtin hash(): str hashing is salted per process, so regimes run in
    # separate processes silently drew different training sets.
    train = make_split(regime.band, regime.train_examples,
                       seed=derive_seed(1_000_000, "pilot-train-split", regime.band))
    signatures = {e.latent_signature for e in train}
    holdout = make_split(regime.band, 1500, seed=987_654_321, forbidden=signatures)
    max_len = max(len(e.tokens) - 1 for e in train + holdout)
    base = baselines(train, holdout)

    started = time.time()
    results = [train_arm(regime, arm, train, holdout, seed, max_len)
               for arm in ARMS for seed in seeds]
    by_arm = {arm: [r for r in results if r["arm"] == arm] for arm in ARMS}
    arm_holdout = {a: statistics.mean(r["holdout_accuracy"] for r in v) for a, v in by_arm.items()}
    arm_seen = {a: statistics.mean(r["train_seen_accuracy"] for r in v) for a, v in by_arm.items()}
    arm_nll = {a: statistics.mean(r["holdout_nll"] for r in v) for a, v in by_arm.items()}

    strongest_trivial = max(base.values())
    # ARM-BLIND competence: the weakest arm must clear the bar.
    min_holdout = min(arm_holdout.values())
    return {
        "regime": asdict(regime),
        "baselines": base,
        "strongest_trivial_baseline": strongest_trivial,
        "arm_holdout_accuracy": arm_holdout,
        "arm_seen_accuracy": arm_seen,
        "arm_holdout_nll": arm_nll,
        "uniform_nll": math.log(VOCAB.domain_size),
        "min_arm_holdout_accuracy": min_holdout,
        "competence_margin": min_holdout - strongest_trivial,
        "all_arms_beat_trivial": min_holdout > strongest_trivial,
        "all_arms_beat_uniform_nll": max(arm_nll.values()) < math.log(VOCAB.domain_size),
        "memorization_gap": statistics.mean(arm_seen.values()) - statistics.mean(arm_holdout.values()),
        "parameters": results[0]["parameters"],
        "elapsed_minutes": (time.time() - started) / 60.0,
        "runs": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--stage", default="pilot")
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]

    # Grid recorded BEFORE execution. Data volume first, capacity second.
    grid = [
        Regime("v3_replica",       5_000,  2_500,  96, 2, 6, 64, 1e-3, "hard"),
        Regime("data_50k",        50_000,  2_500,  96, 2, 6, 64, 1e-3, "hard"),
        Regime("data_200k",      200_000,  2_500,  96, 2, 6, 64, 1e-3, "hard"),
        Regime("data_200k_long", 200_000, 10_000,  96, 2, 6, 64, 1e-3, "hard"),
        Regime("easy_5k",          5_000,  2_500,  96, 2, 6, 64, 1e-3, "easy_nontrivial"),
        Regime("easy_50k",        50_000,  2_500,  96, 2, 6, 64, 1e-3, "easy_nontrivial"),
        Regime("easy_50k_long",   50_000, 10_000,  96, 2, 6, 64, 1e-3, "easy_nontrivial"),
        Regime("medium_50k_long", 50_000, 10_000,  96, 2, 6, 64, 1e-3, "medium"),
        Regime("easy_50k_wide",   50_000, 10_000, 192, 4, 6, 64, 1e-3, "easy_nontrivial"),
    ]
    print(f"device={DEVICE}  regimes={len(grid)}  seeds={seeds}", flush=True)
    print(f"{'regime':18s} {'params':>8s} {'trivial':>8s} {'min_arm':>8s} {'margin':>8s} "
          f"{'memgap':>8s} {'nll':>7s} {'GO':>4s} {'min':>6s}", flush=True)

    out = []
    for regime in grid:
        result = run_regime(regime, seeds)
        out.append(result)
        print(f"{regime.name:18s} {result['parameters']:8d} "
              f"{result['strongest_trivial_baseline']:8.4f} "
              f"{result['min_arm_holdout_accuracy']:8.4f} "
              f"{result['competence_margin']:+8.4f} "
              f"{result['memorization_gap']:8.4f} "
              f"{max(result['arm_holdout_nll'].values()):7.3f} "
              f"{'YES' if result['all_arms_beat_trivial'] else 'no':>4s} "
              f"{result['elapsed_minutes']:6.1f}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"schema_version": "tgsa.phase1-v4-pilot.v1", "device": str(DEVICE),
             "selection_rule": "arm-blind: minimum holdout accuracy across all three arms",
             "uniform_nll": math.log(VOCAB.domain_size), "results": out}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
