"""A script that reads results and writes a summary must REFUSE to write when it found nothing.

**Why this test exists.** The peer session re-ran a tabulator it expected to be a no-op. The
script searched a stale root (completed arms had been archived elsewhere), found MISSING for all
three arms, and then **wrote its empty result over the existing record**, destroying it. Two
independent defects compounding:

  1. a stale search root -- the two-evidence-directory trap, on the READ side;
  2. an **unconditional write** -- it wrote whatever it had, including nothing.

It was recoverable only because the file had been pushed to a remote. **The records directory is
not a git repository, so there is no local undo.**

> **A lookup failure must never be able to destroy the thing it failed to look up.**

These scorers were audited empirically against that shape -- pointed at an empty root, and at a
root where the bank exists but every `predictions.jsonl` is zero-length -- and all refused. This
test pins that behaviour so it cannot regress silently, because the property is invisible in
normal use: every passing run exercises the path where data IS found.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    ROOT / "probe" / "score_mpos.py",
    ROOT / "analysis" / "cue_differential.py",
    ROOT / "analysis" / "position_selection.py",
]


def _run(script: Path, root: Path, out: Path):
    return subprocess.run([sys.executable, str(script), "--root", str(root), "--out", str(out)],
                          capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_refuses_to_write_on_empty_root(script, tmp_path):
    """Nothing to read at all -- the plainest form of the defect."""
    out = tmp_path / "result.json"
    _run(script, tmp_path, out)
    assert not out.exists(), (
        f"{script.name} wrote {out.name} despite an empty root. A lookup failure must never "
        f"be able to destroy the thing it failed to look up.")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_refuses_to_overwrite_an_existing_record_on_empty_input(script, tmp_path):
    """The destructive form: an output file already exists and must survive a failed read."""
    out = tmp_path / "result.json"
    sentinel = {"record": "PRE-EXISTING", "do_not_destroy": True}
    out.write_text(json.dumps(sentinel), encoding="utf-8")
    _run(script, tmp_path, out)
    assert json.loads(out.read_text(encoding="utf-8")) == sentinel, (
        f"{script.name} OVERWROTE an existing record after finding no input. This is the exact "
        f"failure that destroyed a published frontier record in a paired session.")


def test_refuses_when_predictions_exist_but_are_empty(tmp_path):
    """The subtler form: directories and files present, zero rows inside.

    A guard that only checks `path.exists()` passes here and then summarises nothing.
    """
    bank = (Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
            / "mpos002-20260919" / "MPOS_PROBE.jsonl")
    if not bank.exists():
        pytest.skip("bank not present on this machine")
    (tmp_path / "MPOS_PROBE.jsonl").write_text(bank.read_text(encoding="utf-8"),
                                               encoding="utf-8")
    for d in ("gen-B", "gen-a100", "gen-lwf_l0", "gen-sparse", "gen-s10", "gen-s50"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "predictions.jsonl").write_text("", encoding="utf-8")

    out = tmp_path / "result.json"
    sentinel = {"record": "PRE-EXISTING"}
    out.write_text(json.dumps(sentinel), encoding="utf-8")
    _run(ROOT / "probe" / "score_mpos.py", tmp_path, out)
    assert json.loads(out.read_text(encoding="utf-8")) == sentinel, (
        "score_mpos.py summarised zero-length predictions instead of refusing")
