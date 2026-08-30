"""Whichever split is generated first is the only unbiased one.

Splits are generated in sequence and each excludes the signatures already taken,
so every split after the first is a draw from the complement of its predecessors.
Generating `train` first therefore biases the evaluation splits whenever train
saturates any sub-region of the generator's space.

That is not hypothetical. On the v4 `easy_nontrivial` band the non-trivial
two-step `state_update` space is exactly 32 x 16 x 15 = 7,680 items, and a
200,000-example train split consumes every one of them, leaving dev and test
containing only three-step items -- in a study whose premise is that dev and test
are in-band.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from tgsa.phase1_data import materialize_dataset

# A deliberately small space so the two-step region is exhaustible in a fast
# test: 8 starts x 4 non-zero steps x 3 non-cancelling second steps = 96 items.
SMALL = {
    "seed": 4242,
    "vocabulary": {"domain_size": 8, "max_hops": 4, "max_abs_step": 2},
    "families": {
        "direct_lookup": {"min_nodes": 4, "max_nodes": 5, "min_hops": 1, "max_hops": 1},
        "pointer_chase": {"min_nodes": 4, "max_nodes": 5, "min_hops": 2, "max_hops": 2,
                          "reject_trivial": True},
        "state_update": {"min_steps": 2, "max_steps": 3, "reject_trivial": True},
    },
    "splits": {
        "train": {"direct_lookup": 40, "pointer_chase": 60, "state_update": 200},
        "dev": {"direct_lookup": 10, "pointer_chase": 10, "state_update": 30},
        "test": {"direct_lookup": 10, "pointer_chase": 10, "state_update": 60},
    },
}


def _build(tmp_path: Path, order=None) -> dict[str, collections.Counter]:
    config = dict(SMALL)
    if order is not None:
        config = {**SMALL, "split_generation_order": order}
    materialize_dataset(dataset_config=config, output_dir=tmp_path, force=True)
    result = {}
    for split in ("train", "dev", "test"):
        rows = [json.loads(line) for line in
                (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        result[split] = collections.Counter(
            row["stratum"] for row in rows if row["family"] == "state_update")
    return result


def _two_step_fraction(counter: collections.Counter) -> float:
    total = sum(counter.values())
    return sum(v for k, v in counter.items() if ":s2:" in k) / total


def test_train_first_depletes_the_evaluation_splits(tmp_path: Path) -> None:
    """The defect, reproduced directionally.

    The generator draws step counts uniformly from {2, 3}, so an unbiased test
    split is about half two-step. Generating train first drains the smaller
    two-step region and the test split inherits the shortfall.

    This fixture is small enough to run fast, so it shows depletion rather than
    total starvation. At the real v4 scale the depletion is total: a 200,000
    example train split consumes all 7,680 two-step items and the test split
    contains none.
    """

    train_first = _build(tmp_path / "train_first")
    eval_first = _build(tmp_path / "eval_first", order=["test", "dev", "train"])

    assert _two_step_fraction(eval_first["test"]) > 0.35, (
        f"unbiased draw should be near half two-step, got {dict(eval_first['test'])}"
    )
    assert _two_step_fraction(train_first["test"]) < _two_step_fraction(eval_first["test"]) / 2, (
        f"train-first {dict(train_first['test'])} vs "
        f"eval-first {dict(eval_first['test'])}"
    )


def test_evaluation_first_keeps_the_test_split_in_band(tmp_path: Path) -> None:
    """The fix. Test is drawn from the full space and keeps both step counts."""

    strata = _build(tmp_path, order=["test", "dev", "train"])
    assert any(":s2:" in k for k in strata["test"]), dict(strata["test"])
    assert any(":s3:" in k for k in strata["test"]), dict(strata["test"])
    # Train still gets the remainder, which is the correct direction to lose data.
    assert sum(strata["train"].values()) == 200


def test_the_default_order_is_unchanged(tmp_path: Path) -> None:
    """An existing configuration must regenerate byte-identically."""

    without = _build(tmp_path / "a")
    with_explicit = _build(tmp_path / "b", order=["train", "dev", "test"])
    assert without == with_explicit


def test_the_manifest_records_which_split_was_unbiased(tmp_path: Path) -> None:
    manifest = materialize_dataset(
        dataset_config={**SMALL, "split_generation_order": ["test", "dev", "train"]},
        output_dir=tmp_path, force=True,
    )
    assert manifest["split_generation_order"] == ["test", "dev", "train"]
    assert manifest["unbiased_split"] == "test"


def test_a_default_manifest_records_the_original_order(tmp_path: Path) -> None:
    manifest = materialize_dataset(dataset_config=SMALL, output_dir=tmp_path, force=True)
    assert manifest["split_generation_order"] == ["train", "dev", "test"]
    assert manifest["unbiased_split"] == "train"


@pytest.mark.parametrize("order", [
    ["train", "test"],
    ["train", "dev", "test", "train"],
    ["train", "dev", "holdout"],
    "train,dev,test",
])
def test_a_malformed_order_is_refused(tmp_path: Path, order) -> None:
    with pytest.raises((ValueError, TypeError)):
        materialize_dataset(
            dataset_config={**SMALL, "split_generation_order": order},
            output_dir=tmp_path, force=True,
        )


def test_cross_split_disjointness_holds_under_either_order(tmp_path: Path) -> None:
    """Reordering must not weaken the leakage guarantee."""

    for index, order in enumerate((["train", "dev", "test"], ["test", "dev", "train"])):
        target = tmp_path / str(index)
        materialize_dataset(
            dataset_config={**SMALL, "split_generation_order": order},
            output_dir=target, force=True,
        )
        signatures = {}
        for split in ("train", "dev", "test"):
            rows = [json.loads(line) for line in
                    (target / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            signatures[split] = {row["latent_signature"] for row in rows}
        assert not signatures["train"] & signatures["test"]
        assert not signatures["train"] & signatures["dev"]
        assert not signatures["dev"] & signatures["test"]
