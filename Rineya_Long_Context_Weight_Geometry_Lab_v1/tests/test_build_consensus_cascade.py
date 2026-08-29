from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_consensus_cascade.py"
)
SPEC = importlib.util.spec_from_file_location("build_consensus_cascade", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(
    row_id: str,
    prediction: str,
    *,
    status: str = "ok",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "status": status,
        "prediction": prediction,
        **extra,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Answer: The Aurora!  ", "aurora"),
        ("Final answer: 'A Sunday.'", "sunday"),
        ("\u201cUNKNOWN\u201d.", "unknown"),
        ("Talia---Frost", "talia---frost"),
    ],
)
def test_normalize_is_stable_for_gate_comparisons(raw: str, expected: str) -> None:
    assert MODULE.normalize(raw) == expected


def test_cascade_applies_each_stage_in_order_and_records_every_decision() -> None:
    primary = [
        row("a", "Primary A"),
        row("b", "Primary B"),
        row("c", "Answer: UNKNOWN."),
        row("d", "Primary D"),
        row("e", "Primary E"),
        row("f", "Primary F"),
    ]
    override_1 = [
        row("a", "Override A"),
        row("b", "\u201cUNKNOWN\u201d!"),
        row("c", "Recovery is forbidden"),
        row("d", "UNKNOWN."),
        row("e", "Override E"),
        row("f", "Override F"),
    ]
    override_2 = [
        row("a", "Still not unknown"),
        row("b", "Cannot recover B"),
        row("c", "Cannot recover C"),
        row("d", "Answer: UNKNOWN"),
        row("e", "Still not unknown"),
        row("f", "Still not unknown"),
    ]
    consensus_1_left = [
        row("a", "Answer: The Aurora!"),
        row("b", "Rescued B"),
        row("c", "Rescued C"),
        row("d", "Rescued D"),
        row("e", "Left E"),
        row("f", "First"),
    ]
    consensus_1_right = [
        row("a", "\u201caurora\u201d"),
        row("b", "Rescued B"),
        row("c", "Rescued C"),
        row("d", "Rescued D"),
        row("e", "Right E"),
        row("f", "Answer: first."),
    ]
    consensus_2_left = [
        row("a", "Later A"),
        row("b", "Later B"),
        row("c", "Later C"),
        row("d", "Later D"),
        row("e", "Answer: UNKNOWN."),
        row("f", "Answer: The Second."),
    ]
    consensus_2_right = [
        row("a", "Different A"),
        row("b", "Later B"),
        row("c", "Later C"),
        row("d", "Later D"),
        row("e", "unknown!"),
        row("f", "second"),
    ]

    result, summary = MODULE.build_cascade(
        primary,
        [
            ("unknown_override_1", override_1),
            ("unknown_override_2", override_2),
        ],
        [
            (
                "consensus_1_left",
                consensus_1_left,
                "consensus_1_right",
                consensus_1_right,
            ),
            (
                "consensus_2_left",
                consensus_2_left,
                "consensus_2_right",
                consensus_2_right,
            ),
        ],
    )

    assert [item["id"] for item in result] == ["a", "b", "c", "d", "e", "f"]
    assert [item["prediction"] for item in result] == [
        "Answer: The Aurora!",
        "\u201cUNKNOWN\u201d!",
        "Answer: UNKNOWN.",
        "Answer: UNKNOWN",
        "Primary E",
        "Answer: The Second.",
    ]
    assert [item["selected_arm"] for item in result] == [
        "consensus_1_left",
        "unknown_override_1",
        "primary",
        "unknown_override_2",
        "primary",
        "consensus_2_left",
    ]
    assert all(item["status"] == "ok" for item in result)
    assert all(item["cascade_method"] for item in result)

    by_id = {item["id"]: item for item in result}
    assert by_id["a"]["selection_trace"] == [
        {
            "stage": "primary",
            "arm": "primary",
            "selected": True,
            "normalized_prediction": "primary",
        },
        {
            "stage": "unknown_override",
            "arm": "unknown_override_1",
            "selected": False,
            "reason": "not_unknown",
            "normalized_prediction": "override",
        },
        {
            "stage": "unknown_override",
            "arm": "unknown_override_2",
            "selected": False,
            "reason": "not_unknown",
            "normalized_prediction": "still not unknown",
        },
        {
            "stage": "consensus",
            "index": 1,
            "left_arm": "consensus_1_left",
            "right_arm": "consensus_1_right",
            "selected": True,
            "selected_arm": "consensus_1_left",
            "reason": "non_unknown_agreement",
            "left_normalized_prediction": "aurora",
            "right_normalized_prediction": "aurora",
        },
        {
            "stage": "consensus",
            "index": 2,
            "left_arm": "consensus_2_left",
            "right_arm": "consensus_2_right",
            "selected": False,
            "selected_arm": None,
            "reason": "disagreement",
            "left_normalized_prediction": "later",
            "right_normalized_prediction": "different",
        },
    ]

    # Consensus cannot recover any current UNKNOWN, even from unanimous arms.
    for row_id in ("b", "c", "d"):
        consensus_trace = [
            decision
            for decision in by_id[row_id]["selection_trace"]
            if decision["stage"] == "consensus"
        ]
        assert [decision["reason"] for decision in consensus_trace] == [
            "current_unknown",
            "current_unknown",
        ]
        assert not any(decision["selected"] for decision in consensus_trace)

    b_first_override = by_id["b"]["selection_trace"][1]
    assert b_first_override == {
        "stage": "unknown_override",
        "arm": "unknown_override_1",
        "selected": True,
        "reason": "normalized_unknown",
        "normalized_prediction": "unknown",
    }
    d_override_trace = by_id["d"]["selection_trace"][1:3]
    assert [decision["selected"] for decision in d_override_trace] == [True, True]

    e_final_decision = by_id["e"]["selection_trace"][-1]
    assert e_final_decision["reason"] == "unknown_agreement_rejected"
    assert e_final_decision["selected"] is False

    # Both consensus pairs select f; the later pair deterministically wins.
    f_consensus_trace = by_id["f"]["selection_trace"][-2:]
    assert [decision["selected"] for decision in f_consensus_trace] == [True, True]
    assert summary == {
        "rows": 6,
        "unknown_override_selections": {
            "unknown_override_1": 2,
            "unknown_override_2": 1,
        },
        "consensus_pair_selections": {
            "consensus_1": 2,
            "consensus_2": 1,
        },
        "final_selected_arm_counts": {
            "consensus_1_left": 1,
            "consensus_2_left": 1,
            "primary": 2,
            "unknown_override_1": 1,
            "unknown_override_2": 1,
        },
    }


def test_arms_are_aligned_by_id_and_output_keeps_primary_order() -> None:
    primary = [row("a", "one"), row("b", "two")]
    override = [row("b", "UNKNOWN"), row("a", "not unknown")]
    left = [row("b", "Bee"), row("a", "Aye")]
    right = [row("a", "Answer: Aye"), row("b", "Answer: Bee")]

    result, _ = MODULE.build_cascade(
        primary,
        [("override", override)],
        [("left", left, "right", right)],
    )

    assert [item["id"] for item in result] == ["a", "b"]
    assert [item["prediction"] for item in result] == ["Aye", "UNKNOWN"]
    assert [item["selected_arm"] for item in result] == ["left", "override"]


def test_every_arm_requires_identical_id_coverage() -> None:
    primary = [row("a", "one"), row("b", "two")]

    with pytest.raises(
        ValueError,
        match="(?i)same ids|id sequence|match primary ids",
    ):
        MODULE.build_cascade(
            primary,
            [("override", [row("a", "x")])],
            [],
        )


@pytest.mark.parametrize(
    "bad_rows",
    [
        [row("a", "one"), row("a", "two")],
        [row("a", "one", status="error")],
    ],
)
def test_load_jsonl_rejects_duplicate_ids_and_non_ok_statuses(
    tmp_path: Path,
    bad_rows: list[dict[str, Any]],
) -> None:
    source = tmp_path / "bad.jsonl"
    write_jsonl(source, bad_rows)

    with pytest.raises(ValueError, match="(?i)duplicate|non-ok|status"):
        MODULE.load_jsonl(source)


def test_build_cascade_validates_direct_call_rows_too() -> None:
    with pytest.raises(ValueError, match="(?i)duplicate"):
        MODULE.build_cascade(
            [row("same", "one"), row("same", "two")],
            [],
            [],
        )

    with pytest.raises(ValueError, match="(?i)non-ok|status"):
        MODULE.build_cascade(
            [row("a", "one")],
            [("bad", [row("a", "two", status="failed")])],
            [],
        )


def test_build_cascade_never_reads_or_copies_answer_key_fields() -> None:
    forbidden = {"answer", "answer_key", "gold", "reference_answer"}

    class AnswerKeyTripwire(dict[str, Any]):
        def __getitem__(self, key: str) -> Any:
            if key in forbidden:
                raise AssertionError(f"answer key field was read: {key}")
            return super().__getitem__(key)

        def get(self, key: str, default: Any = None) -> Any:
            if key in forbidden:
                raise AssertionError(f"answer key field was read: {key}")
            return super().get(key, default)

    def keyed(prediction: str) -> AnswerKeyTripwire:
        return AnswerKeyTripwire(
            row(
                "a",
                prediction,
                answer="secret",
                answer_key="secret",
                gold="secret",
                reference_answer="secret",
            )
        )

    result, _ = MODULE.build_cascade(
        [keyed("public prediction")],
        [("override", [keyed("not unknown")])],
        [
            (
                "left",
                [keyed("Answer: Public consensus")],
                "right",
                [keyed("The public consensus!")],
            )
        ],
    )

    assert result[0]["prediction"] == "Answer: Public consensus"
    assert forbidden.isdisjoint(result[0])


def test_cli_writes_hashed_provenance_and_deterministic_labels(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    override = tmp_path / "override.jsonl"
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    answer_key = tmp_path / "answer_key.jsonl"
    output = tmp_path / "cascade.jsonl"
    write_jsonl(primary, [row("a", "A"), row("b", "B")])
    write_jsonl(override, [row("a", "UNKNOWN"), row("b", "not unknown")])
    write_jsonl(left, [row("a", "Rescue"), row("b", "Answer: Winner")])
    write_jsonl(right, [row("a", "Rescue"), row("b", "The winner!")])
    answer_key.write_text("this nearby canary must never be read", encoding="utf-8")
    answer_key_before = answer_key.read_bytes()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--primary",
            str(primary),
            "--unknown-override",
            str(override),
            "--consensus-pair",
            str(left),
            str(right),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert answer_key.read_bytes() == answer_key_before
    written = read_jsonl(output)
    assert [item["prediction"] for item in written] == ["UNKNOWN", "Answer: Winner"]
    assert [item["selected_arm"] for item in written] == [
        "unknown_override_1",
        "consensus_1_left",
    ]

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert manifest["method"]
    assert manifest["answer_key_accessed"] is False
    assert manifest["primary"] == {
        "label": "primary",
        "path": str(primary.resolve()),
        "sha256": sha256(primary),
        "rows": 2,
    }
    assert manifest["unknown_overrides"] == [
        {
            "label": "unknown_override_1",
            "path": str(override.resolve()),
            "sha256": sha256(override),
            "rows": 2,
        }
    ]
    assert manifest["consensus_pairs"] == [
        {
            "index": 1,
            "left": {
                "label": "consensus_1_left",
                "path": str(left.resolve()),
                "sha256": sha256(left),
                "rows": 2,
            },
            "right": {
                "label": "consensus_1_right",
                "path": str(right.resolve()),
                "sha256": sha256(right),
                "rows": 2,
            },
        }
    ]
    assert manifest["output"] == {
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "rows": 2,
    }
    assert manifest["summary"] == {
        "rows": 2,
        "unknown_override_selections": {"unknown_override_1": 1},
        "consensus_pair_selections": {"consensus_1": 1},
        "final_selected_arm_counts": {
            "consensus_1_left": 1,
            "unknown_override_1": 1,
        },
    }
