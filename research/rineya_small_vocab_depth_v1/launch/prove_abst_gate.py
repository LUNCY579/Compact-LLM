"""Prove the FINALIZED ABST-001 launch gate can pass AND fail (ABST001_PREREG.md §9-10; gates must be able to
fail and pass). Runs only the hash-checking head of launch/abst_run.sh -- never the run -- in a scratch copy:
  P0  launch_abst.ps1's embedded expected sha256s equal abst_run.sh and smallvd_runner.py on disk
  P1  the gate head of abst_run.sh as committed          -> exit 0 (every gated file hashes to its registered value)
  F1  the same head with a hash list whose FIRST entry is flipped (and HASHES_SHA pointing at it) -> exit 3
  F2  the committed hash list but a wrong HASHES_SHA      -> exit 3 (the list verifies itself first)
  F3  the committed head against a hash list that names a gated file absent from disk -> exit 3
Writes nothing outside --scratch. usage: python -B launch/prove_abst_gate.py --scratch NEWDIR --log NEWFILE"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
BASH = r"C:\Program Files\Git\bin\bash.exe"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def head_of(run_sh: str) -> str:
    """abst_run.sh up to (not including) the result-exists check: the hash gate only."""
    cut = run_sh.index('[ -e "$A/result" ]')
    return run_sh[:cut] + 'echo "GATE PASSED" | tee -a "$LOG"\nexit 0\n'


def run_head(text: str, where: Path, log: Path) -> int:
    s = where / f"gate_{len(list(where.glob('gate_*.sh')))}.sh"
    s.write_bytes(text.encode("utf-8"))
    return subprocess.run([BASH, s.as_posix(), log.as_posix()], capture_output=True, text=True, timeout=1800).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.scratch.exists() or a.log.exists():
        raise SystemExit("--scratch and --log are create-only")
    a.scratch.mkdir(parents=True)
    L = PKG / "launch"
    run_sh = (L / "abst_run.sh").read_bytes().decode("utf-8")
    ps1 = (L / "launch_abst.ps1").read_bytes().decode("utf-8")
    if "__HASHES_SHA__" in run_sh or "__RUN_SH_SHA__" in ps1 or "__RUNNER_SHA__" in ps1:
        raise SystemExit("the chain is not finalized")
    res = {}
    exp_run = re.search(r'\$expectedRunSh = "([0-9a-f]{64})"', ps1).group(1)
    exp_runner = re.search(r'\$expectedRunner = "([0-9a-f]{64})"', ps1).group(1)
    res["P0_ps1_expected_equal_disk"] = exp_run == sha(L / "abst_run.sh") and exp_runner == sha(L / "smallvd_runner.py")
    hs_line = re.search(r"^HASHES_SHA=([0-9a-f]{64})$", run_sh, re.M)
    res["P0_run_sh_pins_the_list"] = bool(hs_line) and hs_line.group(1) == sha(L / "abst_hashes.txt")
    head = head_of(run_sh)
    pkg_posix = "/" + PKG.as_posix()[0].lower() + PKG.as_posix()[2:]
    pkg_line = re.search(r"^PKG=(.*)$", head, re.M).group(1)
    substituted = pkg_line != pkg_posix          # the proof always tests the package it sits in (a no-op when real)
    head = head.replace(f"PKG={pkg_line}", f"PKG={pkg_posix}")
    res["pkg_line_substituted"] = substituted
    log = a.scratch / "gate.log"
    res["P1_committed_gate_exit"] = run_head(head, a.scratch, log)
    # F1: a flipped first entry, self-consistent list sha, so the failure must come from sha256sum -c
    lines = (L / "abst_hashes.txt").read_bytes().decode("utf-8").splitlines()
    h0, rest = lines[0].split("  ", 1)
    bad = a.scratch / "abst_hashes_flipped.txt"
    bad.write_bytes(("\n".join([("0" if h0[0] != "0" else "1") + h0[1:] + "  " + rest] + lines[1:]) + "\n").encode("utf-8"))
    f1 = head.replace("launch/abst_hashes.txt", bad.as_posix()).replace(hs_line.group(1), sha(bad))
    res["F1_flipped_entry_exit"] = run_head(f1, a.scratch, log)
    # F2: the right list, a wrong pinned sha
    f2 = head.replace(hs_line.group(1), "f" * 64)
    res["F2_wrong_list_sha_exit"] = run_head(f2, a.scratch, log)
    # F3: the list names a gated file that is absent (strict mode must refuse)
    gone = a.scratch / "abst_hashes_missing.txt"
    gone.write_bytes(("\n".join(lines + ["0" * 64 + "  " + (a.scratch / "no_such_file.py").as_posix()]) + "\n").encode("utf-8"))
    f3 = head.replace("launch/abst_hashes.txt", gone.as_posix()).replace(hs_line.group(1), sha(gone))
    res["F3_missing_file_exit"] = run_head(f3, a.scratch, log)
    checks = {"P0 ps1 pins abst_run.sh and smallvd_runner.py": res["P0_ps1_expected_equal_disk"],
              "the committed PKG line is this package (no substitution)": not substituted,
              "P0 abst_run.sh pins abst_hashes.txt": res["P0_run_sh_pins_the_list"],
              "P1 committed gate passes (exit 0)": res["P1_committed_gate_exit"] == 0,
              "F1 flipped entry refused (exit 3)": res["F1_flipped_entry_exit"] == 3,
              "F2 wrong list sha refused (exit 3)": res["F2_wrong_list_sha_exit"] == 3,
              "F3 missing gated file refused (exit 3)": res["F3_missing_file_exit"] == 3}
    out = {"schema": 1, "record": "ABST_GATE_PROOF", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "abst_run_sh_sha256": sha(L / "abst_run.sh"), "abst_hashes_sha256": sha(L / "abst_hashes.txt"),
           "launch_abst_ps1_sha256": sha(L / "launch_abst.ps1"), "gated_lines": len(lines),
           "results": res, "checks": checks, "all_pass": all(checks.values()),
           "gate_log_tail": log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:] if log.exists() else []}
    with open(a.log, "x", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"checks": checks, "all_pass": out["all_pass"]}, indent=1))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
