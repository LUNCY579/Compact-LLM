"""SMALLVD-001 gates G1-G3 -- build A1/A2/A3 ONE AT A TIME, verify, test, record, delete the weights.

  A1  V2 vocabulary (65,536 rows), all 28 blocks
  A2  teacher vocabulary (151,936 rows), the registered 18-block set (report_layer_study.py §4 rule)
  A3  V2 vocabulary + the SAME 18-block set (never re-selected)

Per arm: surgery.build_student -> surgery.verify_built -> surgery.reload_checks -> pytest
tests/test_surgery.py (SMALLVD_ARMS=<this arm>; includes the forward-equivalence, virtual-student
and save/reload tests and, for depth-pruned arms, the wrong-layer-map negative control). The
SURGERY.json, verification results, test log and sidecar hashes go to the records tree; the WEIGHTS
are deleted from the scratch build dir afterwards (they rebuild exactly from the teacher + the
committed builder + the committed id map; the data-section sha256 is the identity of record).
Peak disk is one arm (<= 1.02 GB). The lease is held for builds as well as tests.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "rab001_exploration"))
import gpu_lease  # noqa: E402
import hostbudget  # noqa: E402
import param_accounting as pa  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--vocab-dir", required=True, type=Path, help="the V2 candidate directory")
    ap.add_argument("--layer-report", required=True, type=Path, help="LAYER_REPORT.json (selected set)")
    ap.add_argument("--build-root", required=True, type=Path, help="scratch dir for transient weights")
    ap.add_argument("--records-out", required=True, type=Path)
    ap.add_argument("--arms", default="A1,A2,A3")
    ap.add_argument("--expect-depth", type=int, default=18)
    ap.add_argument("--resume", action="store_true",
                    help="skip arms whose ARM_GATES.json already records PASS; the summary NAMES them resumed")
    ap.add_argument("--dry-run", action="store_true",
                    help="plumbing check on a TINY random teacher (< 50M params, asserted): no lease")
    a = ap.parse_args(argv)
    rep = json.loads(a.layer_report.read_text(encoding="utf-8"))
    kept18 = rep["selected_layer_set"]["kept"]
    if len(kept18) != a.expect_depth:
        raise SystemExit(f"selected set has {len(kept18)} blocks, expected {a.expect_depth}")
    arms = {"A1": (a.vocab_dir, None), "A2": (None, kept18), "A3": (a.vocab_dir, kept18)}
    a.records_out.mkdir(parents=True, exist_ok=True)
    cfg0 = json.loads((a.teacher / "config.json").read_text(encoding="utf-8"))
    n0 = pa.analytic(cfg0, cfg0["vocab_size"], cfg0["num_hidden_layers"])["unique_physical_parameters"]
    if a.dry_run:
        if n0 >= 50_000_000:
            raise SystemExit(f"--dry-run is only for tiny teachers (got {n0:,} params)")
        need, token = 0.8, None
    else:
        need = 3.6
        hostbudget.require(need, "SMALLVD-001 surgery arms")
        token = gpu_lease.acquire(timeout_s=7200, label="smallvd001:surgery")
    import surgery
    summary = {}
    try:
        if token:
            gpu_lease.set_workers(token, [os.getpid()])
        # A0: the prereg's "three counts agree on the teacher" claim was registered on HEADER = ANALYTIC
        # alone; the TORCH count had never been run on the teacher. Measure it here, on the same code
        # path the arms use, and refuse to build anything if it disagrees.
        a0 = surgery.reload_checks(a.teacher)
        header_n = pa.account_header(pa.read_header(a.teacher / "model.safetensors")[0], cfg0)["unique_physical_parameters"]
        a0["header_unique_parameters"], a0["analytic_unique_parameters"] = header_n, n0
        if not (a0["unique_params_torch"] == a0["unique_params_by_storage"] == header_n == n0):
            raise SystemExit(f"A0 parameter counts disagree: {a0}")
        with open(a.records_out / f"A0_THREE_COUNTS_{int(time.time())}.json", "x", encoding="utf-8", newline="\n") as f:
            json.dump({"schema": 1, "record": "SMALLVD_A0_THREE_COUNTS", "teacher": str(a.teacher), **a0}, f, indent=1)
        print(f"A0: header = analytic = torch = by-storage = {n0:,}", flush=True)
        for name in a.arms.split(","):
            tok_dir, kept = arms[name]
            out = a.build_root / name
            rec = a.records_out / name
            if rec.exists():
                prev = rec / "ARM_GATES.json"
                if a.resume and prev.exists() and json.loads(prev.read_text(encoding="utf-8"))["G1_G2_G3"] == "PASS":
                    g = json.loads(prev.read_text(encoding="utf-8"))
                    summary[name] = {k: g[k] for k in ("G1_G2_G3", "unique_parameters", "out_file_bytes",
                                                       "out_DATA_SECTION_sha256", "pytest_tail")}
                    summary[name]["mode"] = "resumed"
                    continue
                raise SystemExit(f"{rec} exists and is not a recorded PASS (create-only)")
            if out.exists():                 # scratch build dir from an interrupted try: ours, transient
                shutil.rmtree(out)
            hostbudget.require(need, f"SMALLVD-001 build {name}")
            t0 = time.time()
            man = surgery.build_student(a.teacher, out, kept_layers=kept, tok_dir=tok_dir, arm=name,
                                        label=f"smallvd001_{name.lower()}")
            checks = surgery.verify_built(out, a.teacher)
            reload_ = surgery.reload_checks(out)
            env = dict(os.environ, SMALLVD_ARMS=str(out), SMALLVD_TEACHER=str(a.teacher),
                       PYTHONIOENCODING="utf-8")
            t1 = time.time()
            p = subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q",
                                str(HERE / "tests" / "test_surgery.py")], cwd=str(HERE), env=env,
                               capture_output=True, text=True, timeout=7200)
            rec.mkdir(parents=True)
            for f in out.iterdir():
                if f.name != "model.safetensors":
                    shutil.copyfile(f, rec / f.name)
            (rec / "PYTEST_test_surgery.log").write_text(p.stdout[-20000:] + "\n" + p.stderr[-5000:],
                                                         encoding="utf-8", newline="\n")
            result = {"schema": 1, "record": "SMALLVD_ARM_GATES", "arm": name,
                      "verify_built": checks, "reload_checks": reload_, "pytest_returncode": p.returncode,
                      "pytest_tail": p.stdout.strip().splitlines()[-3:],
                      "build_and_verify_seconds": round(t1 - t0, 1), "test_seconds": round(time.time() - t1, 1),
                      "weights_deleted_after_recording": True,
                      "rebuild": f"surgery.build_student(teacher, out, kept_layers={kept}, tok_dir={tok_dir})",
                      "out_DATA_SECTION_sha256": man["out_DATA_SECTION_sha256"],
                      "out_file_bytes": man["out_file_bytes"],
                      "unique_parameters": checks["unique_parameters"],
                      "G1_G2_G3": "PASS" if p.returncode == 0 else "FAIL"}
            with open(rec / "ARM_GATES.json", "x", encoding="utf-8", newline="\n") as f:
                json.dump(result, f, indent=1, allow_nan=False)
            (out / "model.safetensors").unlink()
            summary[name] = {k: result[k] for k in ("G1_G2_G3", "unique_parameters", "out_file_bytes",
                                                    "out_DATA_SECTION_sha256", "pytest_tail")}
            summary[name]["mode"] = "computed"
            print(f"{name}: {result['G1_G2_G3']} params {checks['unique_parameters']:,} bytes "
                  f"{man['out_file_bytes']:,} | {result['pytest_tail']}", flush=True)
            if p.returncode != 0:
                break
        with open(a.records_out / f"SURGERY_SUMMARY_{int(time.time())}.json", "x", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=1)
        return 0 if all(v["G1_G2_G3"] == "PASS" for v in summary.values()) else 1
    finally:
        if token:
            gpu_lease.release(token)


if __name__ == "__main__":
    raise SystemExit(main())
