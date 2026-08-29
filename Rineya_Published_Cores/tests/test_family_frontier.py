from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

VAULT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, VAULT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PUBLISH = load_module("publish_family_frontier", "_publish_family_frontier.py")
RUNTIME = load_module("family_system_runtime", "_family_system_runtime.py")


def test_frontier_names_scores_and_arm_counts_are_exact() -> None:
    expected = [
        ("Supa", 48, 128, 176, 9),
        ("Baya", 46, 125, 171, 3),
        ("Nimbus DualScale PEFT Gate45", 45, 124, 169, 2),
        ("Supa-Baya Dense Gate42", 42, 124, 166, 2),
    ]
    observed = [
        (
            spec.codename,
            spec.results["showcase64"][0],
            spec.results["stability192"][0],
            spec.results["all256"][0],
            spec.arm_count,
        )
        for spec in PUBLISH.SPECS
    ]
    assert observed == expected
    assert all(spec.kind != "dense_model" for spec in PUBLISH.SPECS)


@pytest.mark.parametrize("spec", PUBLISH.SPECS, ids=lambda spec: spec.codename)
def test_every_replayable_split_matches_frozen_output(spec) -> None:
    for evidence in spec.evidence.values():
        if not evidence.inputs:
            continue
        arm_rows = {
            arm: RUNTIME.load_jsonl(PUBLISH.relative(path))
            for arm, path in evidence.inputs.items()
        }
        routed = RUNTIME.route_predictions(
            {"system_id": spec.system_id, "routing": spec.routing}, arm_rows
        )
        expected = RUNTIME.load_jsonl(PUBLISH.relative(evidence.output))
        assert RUNTIME.canonical_prediction_hash(
            routed
        ) == RUNTIME.canonical_prediction_hash(expected)


def test_dense_gate42_is_not_the_single_peft_42_arm() -> None:
    spec = PUBLISH.SPECS[3]
    assert spec.kind == "two_dense_core_unknown_gate"
    assert spec.arms["supa_dense_low"]["base"] == "Rineya-Supa-1.7.0-devfreeze"
    assert spec.arms["baya_dense_high"]["base"] == "Rineya-Baya-1.7.0-devfreeze"
    assert spec.results["all256"] == (166, 256)


def test_system_catalog_keeps_dense_catalog_separate() -> None:
    cards = {
        spec.codename: {
            "core_id": spec.system_id,
            "results": {
                name: {"hits": hits, "items": items}
                for name, (hits, items) in spec.results.items()
            },
        }
        for spec in PUBLISH.SPECS
    }
    catalog = PUBLISH.system_catalog(cards, "now")
    assert catalog["dense_core_catalog"] == "../CATALOG.json"
    assert len(catalog["systems"]) == 4
    assert catalog["naming"]["Supa"].startswith("48/64")
    assert catalog["legacy_dense_anchors"][0]["actual_showcase_hits"] == 40


def test_runtime_rejects_missing_or_misaligned_arms() -> None:
    system = {
        "system_id": "x",
        "routing": {
            "initial_arm": "low",
            "steps": [{"op": "select_if_arm_unknown", "arm": "high"}],
        },
    }
    low = [{"id": "a", "prediction": "answer", "status": "ok"}]
    high = [{"id": "a", "prediction": "UNKNOWN", "status": "ok"}]
    assert (
        RUNTIME.route_predictions(system, {"low": low, "high": high})[0]["prediction"]
        == "UNKNOWN"
    )
    with pytest.raises(KeyError):
        RUNTIME.route_predictions(system, {"high": high})
    with pytest.raises(ValueError):
        RUNTIME.route_predictions(
            system,
            {"low": low, "high": [{"id": "b", "prediction": "UNKNOWN"}]},
        )
