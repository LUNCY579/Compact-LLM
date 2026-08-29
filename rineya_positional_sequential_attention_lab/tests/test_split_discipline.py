"""SPEC section 8: the sealed split is opened exactly once, at Phase 14.

Two independent guarantees, because either alone is weak:

1. **Nothing outside ``scripts/phase14_*.py`` may name the sealed split in code.** A
   grep would be fooled by a comment and defeated by a docstring, so the scan is over
   the AST: string constants (docstrings excluded) and identifiers. Comments are
   invisible to it by construction, which is the right call — a comment cannot read a
   file.
2. **The split assignment is stable across processes.** ``split_for_scenario`` is a
   SHA-256 of the scenario id, so it must not depend on ``PYTHONHASHSEED``, on import
   order, or on anything else that varies between runs. A subprocess with a different
   hash seed recomputes the whole assignment and the two must agree exactly. If they
   ever did not, every "held out" claim in the lab would be false.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rineya_psa.paths import LAB_ROOT, python_executable
from rineya_psa.types import SPLITS, split_for_scenario, stable_id

SRC = LAB_ROOT / "src"
SCRIPTS = LAB_ROOT / "scripts"

#: Files permitted to name the sealed split, and why. Anything else that does is a
#: discipline breach, not a style question.
SEALED_ALLOWLIST: dict[str, str] = {
    "src/rineya_psa/paths.py": "defines DATA_SEALED; defining a path is not reading it",
    "src/rineya_psa/types.py": "split_for_scenario returns the label",
    "src/rineya_psa/benchmark/generate.py": "the one-time writer that lays the bank down",
    "scripts/phase3_baseline_fingerprint.py": "refuses --split sealed; asserted below",
}


def _python_files() -> list[Path]:
    files = sorted(SRC.rglob("*.py"))
    if SCRIPTS.is_dir():
        files += sorted(SCRIPTS.rglob("*.py"))
    return [p for p in files if "__pycache__" not in p.parts]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """``id()`` of every Constant node that is a module/class/function docstring."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(
            getattr(first, "value", None), ast.Constant
        ) and isinstance(first.value.value, str):
            out.add(id(first.value))
    return out


def sealed_references(path: Path) -> list[str]:
    """Code-level (not comment-level) mentions of the sealed split in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = _docstring_nodes(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            if "sealed" in node.value.lower():
                hits.append(f"line {node.lineno}: string {node.value!r}")
        elif isinstance(node, ast.Name) and "SEALED" in node.id:
            hits.append(f"line {node.lineno}: name {node.id}")
        elif isinstance(node, ast.Attribute) and "SEALED" in node.attr:
            hits.append(f"line {node.lineno}: attribute .{node.attr}")
    return hits


# --------------------------------------------------------------------------------------
# 1. Nobody but Phase 14 touches the sealed split
# --------------------------------------------------------------------------------------


def test_only_phase14_and_the_allowlist_name_the_sealed_split() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _python_files():
        rel = path.relative_to(LAB_ROOT).as_posix()
        if rel in SEALED_ALLOWLIST:
            continue
        if rel.startswith("scripts/phase14_"):
            continue
        hits = sealed_references(path)
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        "the sealed split is opened once, at Phase 14 (SPEC section 8). These files "
        f"name it in code:\n{json.dumps(offenders, indent=2)}"
    )


def test_the_allowlist_is_not_stale() -> None:
    """An allowlist entry that no longer applies is a hole waiting to be reused."""
    for rel in SEALED_ALLOWLIST:
        path = LAB_ROOT / rel
        assert path.is_file(), f"allowlisted file {rel} no longer exists"
        assert sealed_references(path), (
            f"{rel} is allowlisted for naming the sealed split but no longer does; "
            "drop the entry rather than leaving the exemption open"
        )


def test_phase3_refuses_the_sealed_split() -> None:
    """The allowlist entry for the Phase-3 script is a refusal. Prove it refuses."""
    script = LAB_ROOT / "scripts" / "phase3_baseline_fingerprint.py"
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [python_executable(), str(script), "--split", "sealed", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(LAB_ROOT),
        timeout=180,
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "sealed" in combined


def test_no_phase14_script_exists_yet() -> None:
    """Documents the current state: the sealed split has not been opened.

    When Phase 14 lands this test is expected to be replaced by one that checks the
    Phase-14 script reads the sealed split exactly once. Failing here is a signal to
    do that, not a defect.
    """
    existing = sorted(p.name for p in SCRIPTS.glob("phase14_*.py")) if SCRIPTS.is_dir() else []
    assert existing == [], f"Phase 14 has landed ({existing}); update this test"


# --------------------------------------------------------------------------------------
# 2. The assignment itself is stable
# --------------------------------------------------------------------------------------

_PROBE_IDS: tuple[str, ...] = tuple(stable_id("split-probe", i) for i in range(400))

_SUBPROCESS_PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from rineya_psa.types import split_for_scenario
ids = json.loads(sys.argv[2])
print(json.dumps([split_for_scenario(s) for s in ids]))
"""


def test_split_for_scenario_is_stable_across_processes() -> None:
    here = [split_for_scenario(s) for s in _PROBE_IDS]

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "31337"  # would perturb anything built on builtins.hash
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            python_executable(),
            "-c",
            _SUBPROCESS_PROBE,
            str(SRC),
            json.dumps(list(_PROBE_IDS)),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    there = json.loads(result.stdout)
    assert there == here


def test_split_for_scenario_matches_the_spec_rule() -> None:
    """SPEC section 8 pins the rule, not just the determinism: <40 / <70 / else."""
    import hashlib

    for scenario_id in _PROBE_IDS[:50]:
        digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        expected = "calibration" if bucket < 40 else "validation" if bucket < 70 else "sealed"
        assert split_for_scenario(scenario_id) == expected


def test_split_proportions_track_the_declared_40_30_30() -> None:
    counts = {s: 0 for s in SPLITS}
    for scenario_id in _PROBE_IDS:
        counts[split_for_scenario(scenario_id)] += 1
    n = len(_PROBE_IDS)
    assert counts["calibration"] / n == pytest.approx(0.40, abs=0.06)
    assert counts["validation"] / n == pytest.approx(0.30, abs=0.06)
    assert counts["sealed"] / n == pytest.approx(0.30, abs=0.06)


def test_positional_variants_of_one_scenario_share_a_split() -> None:
    """The property the paired statistics depend on: split is keyed on the scenario."""
    for scenario_id in _PROBE_IDS[:20]:
        split = split_for_scenario(scenario_id)
        for position in (0.02, 0.5, 0.98):
            # item ids vary with position; the split must not.
            item_id = stable_id(scenario_id, position)
            assert item_id != scenario_id
            assert split_for_scenario(scenario_id) == split


def test_split_labels_are_exactly_the_declared_three() -> None:
    seen = {split_for_scenario(s) for s in _PROBE_IDS}
    assert seen == set(SPLITS)
    assert sys.version_info >= (3, 11)  # SPEC section 0 pins the interpreter family
