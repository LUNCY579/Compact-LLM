from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from rineya_significance_lab.config import load_config
from rineya_significance_lab.manifest import sha256_file

LAB_ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, LAB_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_family_config_is_exactly_published_secondary_qwen3_cores() -> None:
    config = load_config(LAB_ROOT / "configs" / "published_family_showcase64_v2.yaml")
    assert set(config["cores"]) == {"halo_uncensored", "lyra", "nimbus", "pulse"}
    assert [row["id"] for row in config["conditions"]] == ["B0", "B2", "B3"]
    assert config["analysis"]["prohibit_cross_core_pooling"] is True
    assert config["excluded_from_active_scope"] == ["Rineya/R2 checkpoints", "Rineya Aegis"]


def test_frozen_part1_source_is_complete_and_answer_blind() -> None:
    path = (
        LAB_ROOT
        / "results"
        / "runs"
        / "halo_family_showcase64_v1_part1"
        / "complete_cores.jsonl"
    )
    assert sha256_file(path) == "c6c644ae7c95810d7aafa2ddaeee941db9474a6678ee1cf784f168de463969ea"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 256
    assert {(row["core"], row["condition"]) for row in rows} == {
        ("halo_uncensored", "B0"),
        ("halo_uncensored", "B3"),
        ("lyra", "B0"),
        ("lyra", "B3"),
    }
    assert len({(row["core"], row["case_id"], row["condition"]) for row in rows}) == 256
    forbidden = {"accepted_answers", "answer", "correct", "reference_answer"}
    assert not (set().union(*(set(row) for row in rows)) & forbidden)


def test_family_runner_conditions_are_config_driven() -> None:
    module = _module("family_runner_v2", "run_published_family_core_showcase64.py")
    config = load_config(LAB_ROOT / "configs" / "published_family_showcase64_v2.yaml")
    conditions = module._condition_map(config)
    assert conditions["B0"].policy == "full_cache"
    assert conditions["B2"].exponent == 0.0
    assert conditions["B3"].exponent == 1.0


def test_exact_sign_and_pack_pairing_use_scenario_as_unit() -> None:
    module = _module("family_finalizer_v2", "finalize_published_family_showcase64.py")
    assert module.exact_sign_p(0, 8) == 0.0078125
    assert module.exact_sign_p(2, 2) == 1.0
    rows = []
    for scenario, candidate, reference in (("s1", [True, True], [False, False]), ("s2", [False, False], [True, True])):
        for index, value in enumerate(candidate):
            rows.append(
                {
                    "core": "lyra",
                    "scenario_id": scenario,
                    "case_id": f"{scenario}-{index}",
                    "condition": "B3",
                    "correct": value,
                }
            )
        for index, value in enumerate(reference):
            rows.append(
                {
                    "core": "lyra",
                    "scenario_id": scenario,
                    "case_id": f"{scenario}-{index}",
                    "condition": "B2",
                    "correct": value,
                }
            )
    result = module.paired_stats(rows, core="lyra", candidate="B3", reference="B2")
    assert result["case_unit"]["wins"] == 2
    assert result["case_unit"]["losses"] == 2
    assert result["scenario_pack_unit"]["wins"] == 1
    assert result["scenario_pack_unit"]["losses"] == 1
    assert result["scenario_pack_unit"]["p_value_two_sided"] == 1.0
