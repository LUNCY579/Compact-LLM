"""Stage 6: derive the v4 registered config from the frozen artifacts.

The v3 config was hand-written, which is fine when the design is chosen by
argument. v4's design is chosen by *evidence* — the competence report fixes the
band, the data volume, the update budget and the model, and the power report
fixes the seed count — so writing those numbers out by hand would break the only
property that makes them defensible: that they came from somewhere.

This script reads both artifacts, refuses to proceed unless the competence
verdict is GO, copies the certified regime verbatim, and records a SHA-256 of
each input in the config's ``registration`` block. Re-running it against the
same inputs is a no-op; running it against different inputs produces a config
whose provenance says so.

What it deliberately does NOT touch: the ``analysis`` block. The estimator, the
primary gate, the hypothesis thresholds, the guardrail margin, and the
falsifier are carried over from v3 unchanged. v3's statistical design was not
what failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from tgsa.phase1_competence import require_go
from tgsa.phase1_data import SyntheticVocabulary, generate_split

# Carried over from v3 without modification. RULE 1: the statistical pipeline is
# not redesigned without a demonstrated defect, and none was demonstrated.
V3_ANALYSIS_KEYS = (
    "primary_families",
    "guardrail_family",
    "bootstrap_samples",
    "confidence",
    "bootstrap_seed",
    "minimum_history_delta",
    "minimum_cascade_delta",
    "minimum_alignment_drop",
    "direct_lookup_noninferiority_margin",
    "alignment_condition",
    "minimum_derangement_coverage",
    "primary_gate",
    "interval_estimator",
    "sensitivity_estimators",
    "powered_for_history_delta",
    "powered_for_cascade_delta",
    "powered_for_alignment_drop",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure_sequence_length(band_spec: dict, counts: dict[str, int], seed: int,
                            vocab: SyntheticVocabulary) -> int:
    """The longest input any split can present. Guessing this truncates data."""

    examples = generate_split(
        split="train", count_by_family=counts, dataset_seed=seed, vocab=vocab,
        generation_spec={"families": band_spec},
    )
    return max(len(example.tokens) - 1 for example in examples)


def build(competence_path: Path, power_path: Path, v3_config_path: Path,
          *, study_id: str, dataset_seed: int, test_per_family: int,
          dev_per_family: int, length_margin: int,
          allow_development_only: bool = False) -> dict:
    # fails closed on NO_GO, on a superseded schema, and -- unless the caller is
    # the rehearsal -- on a report marked development_only
    competence = require_go(competence_path,
                            allow_development_only=allow_development_only)
    power = json.loads(power_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_config_path.read_text(encoding="utf-8"))

    regime = competence["regime"]

    # The gate artifact must certify the distribution the study will actually
    # evaluate on. Competence runs that generated their TRAINING split first drew
    # their holdout from what was left, which on this band means the holdout is
    # missing an entire difficulty stratum. Such a report is conservative -- its
    # holdout is harder -- so it is fine as evidence that a regime FAILS, but it
    # must not be the certificate a confirmatory matrix is dispatched on.
    if regime.get("split_generation_order") != ["holdout", "train"]:
        raise ValueError(
            f"competence report was produced with split_generation_order="
            f"{regime.get('split_generation_order')!r}; a confirmatory config "
            f"requires ['holdout', 'train'] so the certified distribution matches "
            f"the one the study evaluates on. Re-measure this regime with the "
            f"corrected harness -- per the registered ladder, re-measuring the "
            f"same regime after a documented harness fix is a correction and does "
            f"not consume a rung"
        )

    seed_count = power.get("recommended_seed_count")
    if not seed_count:
        raise ValueError(
            "power report recommends no adequate seed count; the design is not "
            "powered and must not be registered"
        )

    vocabulary = dict(v3["dataset"]["vocabulary"])
    vocab = SyntheticVocabulary(**vocabulary)
    band_spec = regime["band_spec"]
    per_family = int(regime["train_examples"]) // 3

    # The competence evidence was collected on splits of exactly these sizes;
    # a longer sequence in test than in train would be an untested regime.
    longest = max(
        measure_sequence_length(band_spec, {k: per_family for k in band_spec},
                                regime["train_split_seed"], vocab),
        measure_sequence_length(band_spec, {k: test_per_family for k in band_spec},
                                dataset_seed, vocab),
    )

    seeds = [101 * (index + 1) for index in range(int(seed_count))]

    config: dict = {
        "schema_version": v3["schema_version"],
        "study_id": study_id,
        "confirmatory": True,
        "competence_report": competence_path.name,
        "expected_base_branch": v3["expected_base_branch"],
        "expected_base_commit": v3["expected_base_commit"],
        "claim_scope": v3["claim_scope"],
        "arms": list(v3["arms"]),
        "seeds": seeds,
        "dataset": {
            "seed": dataset_seed,
            "vocabulary": vocabulary,
            "families": band_spec,
            # Evaluation splits are generated FIRST. Each split excludes the
            # signatures already taken, so whichever runs first is the only
            # unbiased draw. Generating train first on this band consumed the
            # entire 7,680-item non-trivial two-step `state_update` space and
            # left dev and test with three-step items only -- an unintended
            # difficulty shift in a study whose premise is that they are in-band.
            "split_generation_order": ["test", "dev", "train"],
            # No split_overrides. v3 made dev and test HARDER than train, which
            # stacks difficulty extrapolation on top of the mechanism contrast;
            # with no arm competent, the result was uninterpretable. v4 tests
            # in-band -- same difficulty, novel instances -- so the contrast is
            # about the mechanism and nothing else. Difficulty response remains
            # available as the registered NON-GATING secondary analysis.
            "splits": {
                "train": {family: per_family for family in band_spec},
                "dev": {family: dev_per_family for family in band_spec},
                "test": {family: test_per_family for family in band_spec},
            },
        },
        "model": {
            "max_sequence_length": longest + length_margin,
            "embed_dim": regime["embed_dim"],
            "num_heads": regime["num_heads"],
            "num_layers": regime["num_layers"],
            "ffn_multiplier": v3["model"]["ffn_multiplier"],
            "dropout": v3["model"]["dropout"],
            "bias": v3["model"]["bias"],
            "trajectory_rank": v3["model"]["trajectory_rank"],
            "tie_embeddings": v3["model"]["tie_embeddings"],
        },
        "training": {
            **v3["training"],
            "steps": regime["steps"],
            "batch_size": regime["batch_size"],
            "learning_rate": regime["learning_rate"],
        },
        "evaluation": dict(v3["evaluation"]),
        "analysis": {key: v3["analysis"][key] for key in V3_ANALYSIS_KEYS},
        "runtime": dict(v3["runtime"]),
        "registration": {
            "preregistration": "PHASE1_V4_PREREGISTRATION.md",
            "supersedes_study_id": v3["study_id"],
            "supersedes_reason": (
                "v3 executed cleanly and returned `inconclusive` because no arm was "
                "competent: all three memorized 5,000 training examples and scored at "
                "the majority baseline on held-out data. v4 changes ONLY the regime in "
                "which the contrast is run -- difficulty band, training-set size, and "
                "update budget -- plus a competence gate that must return GO before the "
                "matrix may be dispatched. The mechanism, arms, controls, falsifier, "
                "estimator, primary gate, and hypothesis thresholds are unchanged."
            ),
            "competence_report_sha256": digest(competence_path),
            "power_report_sha256": digest(power_path),
            "competence_verdict": competence["verdict"],
            "seed_count_source": "power report recommended_seed_count",
        },
    }

    # The marker propagates. A development-only report can only produce a
    # development-only config, whose runner is the only thing that will accept
    # a development-only report. Neither end launders it.
    if competence.get("regime", {}).get("development_only"):
        config["development_only"] = True
        config["registration"]["development_only_reason"] = (
            "built from a competence report marked development_only; this "
            "configuration certifies nothing and exists to exercise the pipeline"
        )
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competence", type=Path, required=True)
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--v3-config", type=Path,
                        default=LAB / "configs" / "phase1_identifiability_v3.json")
    parser.add_argument("--output", type=Path,
                        default=LAB / "configs" / "phase1_identifiability_v4.json")
    parser.add_argument("--study-id", default="tgsa-phase1-identifiability-v4-2026-08-30")
    parser.add_argument("--dataset-seed", type=int, default=4_150_927)
    parser.add_argument("--test-per-family", type=int, default=2000)
    parser.add_argument("--dev-per-family", type=int, default=1000)
    parser.add_argument("--length-margin", type=int, default=4)
    parser.add_argument(
        "--allow-development-only", action="store_true",
        help="accept a report marked development_only and mark the config to "
             "match; for the pipeline rehearsal only",
    )
    args = parser.parse_args()

    config = build(
        args.competence, args.power, args.v3_config,
        study_id=args.study_id, dataset_seed=args.dataset_seed,
        test_per_family=args.test_per_family, dev_per_family=args.dev_per_family,
        length_margin=args.length_margin,
        allow_development_only=args.allow_development_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"  seeds            {len(config['seeds'])}")
    print(f"  train / family   {list(config['dataset']['splits']['train'].values())[0]}")
    print(f"  steps            {config['training']['steps']}")
    print(f"  max_seq_length   {config['model']['max_sequence_length']}")
    print(f"  competence       {config['registration']['competence_verdict']} "
          f"({config['registration']['competence_report_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
