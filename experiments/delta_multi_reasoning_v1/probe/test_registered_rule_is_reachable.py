"""A registered rule that no scoring path can reach governs nothing — unless the path is SEALED.

**Why this file exists.** GATEFIX-001 was registered at 2026-09-20T14:11:46Z and replaced GATE
v2's `references_retained` conjunct with a three-valued verdict. DOSELORA-001's arms were scored
at 17:49:46Z — **3h38m later, under the two-valued rule it replaced** — because the prereg
changed prose and nothing changed code.

> **A prereg cannot assert its own implementation. Something has to test it.**

## The doctrine this file got WRONG on its first version

Version 1 said: *"fix the scoring path, or withdraw the registration."* **That instruction would
have destroyed evidence.** `retention_highdose_lora_resume_v1/report.py` is pinned in an 11-file
`PRELAUNCH_BINDING.json` **and** is `RESULTS.json`'s `reporter_sha256`, matching byte-for-byte.
Editing it breaks the custody chain of the only DOSELORA result that exists — the recorded
reporter hash would no longer be reproducible from the working tree and the arm becomes
unverifiable.

The paired session caught this before anyone acted on it. There is a third case:

> **A scoring module that is SEALED EVIDENCE of a completed run cannot be fixed and must not be.
> It is excluded by PROVENANCE, never by convenience.**

## Why this does not reopen the xfail loophole

The exclusion is **computed from the artifacts**, not asserted by whoever finds the test
inconvenient. A module is sealed iff its sha256 appears under **any** `*_sha256` key, at any
depth, in any `PRELAUNCH_BINDING.json` or `RESULTS.json`. You cannot exempt a module by editing
this test; you would have to forge a receipt.

**The harvest is STRUCTURAL, and version 2 got that wrong too.** It enumerated exactly two field
names — `code_files_sha256` and `reporter_sha256` — and treated every other hash field as
absent. The tree also carries `external_code_files_sha256` (pinning code in *another* worktree)
and `pilot_code_sha256`, both of which are code bindings. **Enumerating field names is the same
scope error the sealed-evidence doctrine exists to prevent, one level down**: absence from the
two fields I happened to list was being read as absence from every receipt. Harvesting by shape
raised the sealed set from 50 hashes to **152** and surfaced a **fifth** module neither session
had seen — `binding_mixture_v1/report.py`, sealed by a third session's `RESULTS.json`.

**Both sets grow over time.** New receipts appear as other sessions complete runs, so a module
classified LIVE today can be SEALED tomorrow. The classification is a function of the record at
the moment it is computed, not a fixed property — which is why it is recomputed on every run
rather than cached to disk.

**And the exclusion makes the test STRONGER, not weaker.** A module that is both sealed and
superseded is a finding in its own right — it says *a result on the record was produced under a
rule that has since been replaced*. That earns its own status, `SEALED_UNDER_SUPERSEDED_RULE`,
reported separately so the red count is the actionable one.

## This test is EXPECTED TO FAIL while a live path is inert

A reachability test that passes on the day it is written tests nothing. Do not skip or xfail it
to green the suite — wire the rule into the **unsealed** paths, or withdraw the registration.

Companion to `test_refuse_empty.py`, which pins a property invisible in normal use for the same
reason: every passing run exercises the path where the thing works.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"

SEARCH_ROOTS = (
    BASE / "ChatBubble-dose-param" / "research",
    BASE / "ChatBubble-dmrf" / "experiments",
)

# APPLIES the conjunct = thresholds the endpoint, rather than merely printing it.
APPLIES = re.compile(r"references_retained\"?\]?\s*>=", re.I)

# GATEFIX-001's verdict is three-valued and interval-based.
THREE_VALUED = ("CERTIFIED_PASS", "CERTIFIED_FAIL", "UNCERTIFIABLE",
                "clopper", "cp_lower", "interval_contains", "retention_verdict")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _sealed_hashes() -> dict[str, str]:
    """Every code hash a committed receipt pins. Computed from artifacts, not from a list."""
    sealed: dict[str, str] = {}
    if not RECORDS.is_dir():
        return sealed
    hexish = re.compile(r"^[0-9a-f]{64}$")

    def harvest(obj, src: str, key: str = "") -> None:
        """Any *_sha256 field, at any depth. Enumerating field NAMES is how this leaks."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                harvest(v, src, k)
        elif isinstance(obj, list):
            for v in obj:
                harvest(v, src, key)
        elif isinstance(obj, str) and key.lower().endswith("sha256") and hexish.match(obj):
            sealed.setdefault(obj, f"{src}:{key}")

    for name in ("PRELAUNCH_BINDING.json", "RESULTS.json"):
        for f in RECORDS.rglob(name):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            harvest(d, f"{f.parent.name}/{f.name}")
    return sealed


@lru_cache(maxsize=1)
def _modules() -> tuple[tuple[Path, str, str], ...]:
    """(path, status, provenance) for every module applying the conjunct."""
    out = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.py")):
            if p.name.startswith("test_"):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not APPLIES.search(text):
                continue
            seal = _sealed_hashes().get(_sha(p))
            has_rule = any(t.lower() in text.lower() for t in THREE_VALUED)
            if seal and not has_rule:
                out.append((p, "SEALED_UNDER_SUPERSEDED_RULE", seal))
            elif seal:
                out.append((p, "SEALED_WITH_RULE", seal))
            else:
                out.append((p, "LIVE", ""))
    return tuple(out)


def test_the_test_can_find_its_subject():
    assert _modules(), (
        "No module applies the references_retained conjunct. Either the scoring path moved or "
        "SEARCH_ROOTS is stale. A reachability test that cannot find its subject is a check "
        "whose scope cannot reach its claim — fix the roots rather than trusting the pass.")


def test_sealed_modules_are_reported_not_fixed():
    """Sealed-and-superseded is a FINDING, not a failure. It must never be silently green."""
    sealed = [(p, prov) for p, st, prov in _modules() if st == "SEALED_UNDER_SUPERSEDED_RULE"]
    for p, prov in sealed:
        print(f"\nSEALED_UNDER_SUPERSEDED_RULE: {p}\n    pinned by {prov}\n"
              f"    A result on the record was produced under a rule that has since been "
              f"replaced. This module CANNOT be edited without breaking that result's custody "
              f"chain. It is excluded by provenance, and the exclusion is the finding.")
    assert True  # reporting only; the assertion lives in the LIVE test below


@pytest.mark.parametrize(
    "module,status,provenance",
    [m for m in _modules() if m[1] == "LIVE"] or [(None, "NONE", "")],
    ids=lambda v: v.name if isinstance(v, Path) else str(v))
def test_live_scoring_paths_implement_the_registered_rule(module, status, provenance):
    """Every UNSEALED module applying the conjunct must implement GATEFIX-001."""
    if module is None:
        pytest.skip("no live (unsealed) module applies the conjunct")
    text = module.read_text(encoding="utf-8", errors="replace")
    found = [t for t in THREE_VALUED if t.lower() in text.lower()]
    assert found, (
        f"\n{module}\napplies the references_retained conjunct, is NOT sealed by any receipt, "
        f"and carries none of the three-valued vocabulary {THREE_VALUED}.\n\n"
        f"GATEFIX-001 (probe/GATEFIX001_PREREG.md, committed 2026-09-20T14:11:46Z) replaced the "
        f"binary verdict with CERTIFIED_PASS / UNCERTIFIABLE / CERTIFIED_FAIL, and "
        f"SCOREFIX-001 registered the reconciliation. This module still applies the superseded "
        f"rule on a LIVE path, so future scoring reports a verdict the registration withdrew.\n\n"
        f"`probe/retention_verdict.py` is the single implementation — import it rather than "
        f"writing a fifth copy; the three report.py copies have already diverged from each "
        f"other.\n\n"
        f"Do NOT xfail this to green the suite. Sealed modules are excluded automatically by "
        f"provenance; this one is not sealed, so it is genuinely inert.")
