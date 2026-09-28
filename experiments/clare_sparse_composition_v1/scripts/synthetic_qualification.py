"""Phase 2 synthetic qualification: run the offline suite and measure the core invariants directly.

    python scripts/synthetic_qualification.py --out qualification/SYNTHETIC_QUALIFICATION.json

Writes one JSON report (LF, create-only): every test's outcome (via pytest's junit XML), plus a
direct measurement on a tiny 4-skill problem of the invariants CLARE-RINEYA-001 relies on
(exact budgets, zero support overlap, zero dot product, frozen deltas unchanged, exact discovery
reset, base weights unchanged, learned != random supports at identical capacity). CPU only.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "tests")]


def run_suite() -> dict:
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                               "-W", "ignore::UserWarning", f"--junitxml={xml}", "tests"],
                              cwd=HERE, capture_output=True, text=True)
        tests = []
        for case in ET.parse(xml).getroot().iter("testcase"):
            outcome = "passed"
            for tag in ("failure", "error", "skipped"):
                if case.find(tag) is not None:
                    outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
            tests.append({"test": f"{case.get('classname')}::{case.get('name')}", "outcome": outcome})
    counts = {k: sum(t["outcome"] == k for t in tests) for k in ("passed", "failed", "error", "skipped")}
    return {"returncode": proc.returncode, "seconds": round(time.time() - t0, 1), "counts": counts,
            "tests": tests, "summary_line": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""}


def measure() -> list[dict]:
    from clare.allocation import ArmSpec, run_arm
    from clare.geometry import pairwise
    from clare.schedule import budgets
    from clare.training import base_checksum
    from tinymodel import RECIPE, SKILLS, problem

    checks = []

    def check(name, value, ok):
        checks.append({"check": name, "measured": value, "status": "PASS" if ok else "FAIL"})

    order = tuple(SKILLS)
    model, fp, host, data, loss_fn = problem(hidden=8, inter=16)
    base0 = base_checksum(model)
    sched = [b.budget for b in budgets(fp.total, order, "0.9")]
    res = {a: run_arm(ArmSpec(a, order, rho="0.9"), host, data, loss_fn, RECIPE)
           for a in ("D_random_disjoint", "E_learned_disjoint")}
    for a, r in res.items():
        got = [x.mask.count() for x in r.records]
        check(f"{a}: mask budgets == schedule {sched}", got, got == sched)
        geo = pairwise(r.deltas())
        inter = max(g["support_intersection"] for g in geo)
        dot = max(abs(g["dot"]) for g in geo)
        check(f"{a}: max pairwise support intersection", inter, inter == 0)
        check(f"{a}: max |parameter dot product|", dot, dot == 0.0)
        same = all(x.frozen_checksum_before == x.frozen_checksum_after for x in r.records)
        check(f"{a}: frozen deltas unchanged by every later stage", same, same)
    resets = [x.discovery["effective_checksum_before"] == x.discovery["effective_checksum_after_reset"]
              for x in res["E_learned_disjoint"].records]
    check("E: discovery reset restores the exact pre-discovery effective weights (per skill)", resets, all(resets))
    zeros = [x.discovery["selected_with_zero_importance"] for x in res["E_learned_disjoint"].records]
    check("E: selected coordinates with zero importance (0 = selection fully learned)", zeros, all(z == 0 for z in zeros))
    differ = res["D_random_disjoint"].records[0].mask != res["E_learned_disjoint"].records[0].mask
    check("D and E choose different supports at identical capacity", differ, differ)
    check("base weights unchanged after both arms", base_checksum(model) == base0, base_checksum(model) == base0)
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    import numpy
    import torch

    from clare.io import write_json
    from clare.manifests import code_digest

    suite = run_suite()
    checks = measure()
    ok = suite["returncode"] == 0 and all(c["status"] == "PASS" for c in checks)
    report = {"experiment": "CLARE-RINEYA-001", "phase": "2-synthetic-qualification",
              "status": "PASS" if ok else "FAIL", "code_digest": code_digest(),
              "runtime": {"python": platform.python_version(), "executable": sys.executable,
                          "torch": torch.__version__, "numpy": numpy.__version__, "device": "cpu"},
              "suite": suite, "direct_checks": checks,
              "scope": "tiny synthetic models only; says nothing about the native 596M model or about skills"}
    write_json(args.out, report)
    print(report["status"], suite["counts"], sum(c["status"] == "PASS" for c in checks), "/", len(checks), "direct checks")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
