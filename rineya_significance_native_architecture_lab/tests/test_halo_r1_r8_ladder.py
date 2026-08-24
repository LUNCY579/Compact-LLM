from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def _module():
    path = LAB / "scripts" / "build_halo_r1_r8_ladder.py"
    spec = importlib.util.spec_from_file_location("build_halo_r1_r8_ladder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ladder_matrix_has_declared_independent_structure() -> None:
    import yaml

    module = _module()
    config = yaml.safe_load((LAB / "configs" / "halo_r1_r8_ladder.yaml").read_text(encoding="utf-8"))
    cases = module.generate_ladder_cases(config)

    assert len(cases) == 1176
    assert len({case.case_id for case in cases}) == 1176
    assert len({case.metadata["cluster_id"] for case in cases}) == 84
    assert Counter(case.family for case in cases) == {
        "R1": 490,
        "R2": 98,
        "R3": 98,
        "R4": 98,
        "R5": 98,
        "R6": 98,
        "R7": 98,
        "R8": 98,
    }
    assert {case.metadata["context_layout"] for case in cases} == {
        "all_early",
        "all_middle",
        "all_late",
        "early_late",
        "early_middle_late",
        "clustered",
        "uniformly_distributed",
        "random",
    }


def test_ladder_covers_r1_cardinalities_and_r2_hops() -> None:
    import yaml

    module = _module()
    config = yaml.safe_load((LAB / "configs" / "halo_r1_r8_ladder.yaml").read_text(encoding="utf-8"))
    cases = module.generate_ladder_cases(config)
    assert {int(case.metadata["required_fact_count"]) for case in cases if case.family == "R1"} == {
        2,
        3,
        4,
        6,
        8,
    }
    assert {int(case.oracle["hops"]) for case in cases if case.family == "R2"} == set(range(2, 9))


def test_every_ladder_answer_has_independent_agreement() -> None:
    import yaml

    module = _module()
    config = yaml.safe_load((LAB / "configs" / "halo_r1_r8_ladder.yaml").read_text(encoding="utf-8"))
    for case in module.generate_ladder_cases(config):
        oracle, rendered = module._verified_answer(case)
        assert oracle == rendered == case.answer
