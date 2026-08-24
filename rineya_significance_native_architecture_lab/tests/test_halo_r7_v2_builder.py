from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def _module():
    path = LAB / "scripts" / "build_halo_r7_v2.py"
    spec = importlib.util.spec_from_file_location("build_halo_r7_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r7_v2_matrix_and_screen_are_prospectively_balanced() -> None:
    import yaml

    module = _module()
    config = yaml.safe_load((LAB / "configs" / "halo_r7_v2.yaml").read_text(encoding="utf-8"))
    cases = module.generate_cases(config)
    selected = module._screen_cases(cases, dict(config["benchmark"]))
    assert len(cases) == 80
    assert len({case.case_id for case in cases}) == 80
    assert len({case.metadata["cluster_id"] for case in cases}) == 8
    assert Counter(case.answer for case in cases) == {"YES": 40, "NO": 40}
    assert Counter(case.answer for case in selected) == {"YES": 4, "NO": 4}
    assert len({case.metadata["cluster_id"] for case in selected}) == 8
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


def test_committed_r7_v2_views_are_separate_and_exactly_annotated() -> None:
    root = LAB / "results" / "benchmarks" / "halo_r7_v2"
    assert root.is_dir()
    inference = [json.loads(line) for line in (root / "inference.jsonl").read_text().splitlines()]
    evaluation = [json.loads(line) for line in (root / "evaluation.jsonl").read_text().splitlines()]
    assert len(inference) == len(evaluation) == 80
    assert not any(
        "answer" in row or "oracle" in row or "evidence_annotation" in row for row in inference
    )
    assert all(len(row["evidence_annotation"]["facts"]) == 10 for row in evaluation)
    assert all(row["verification"]["deletion_minimal_exact_evidence"] for row in evaluation)
    assert {row["case_id"] for row in inference} == {row["case_id"] for row in evaluation}
    audit = json.loads((root / "design_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["frozen_v1"]["status"] == "INVALID_UNINFORMATIVE_FROZEN"
