"""DPACQ-001: acquisition on the four scaled LoRA rungs, registered in DPACQ_PREREG.md.

Holds the card lease for the whole battery rather than per arm -- dropping it between arms lets
another lane interleave mid-experiment.

WHAT IS REUSED RATHER THAN REWRITTEN. The probe is `process_probe.py`, unmodified, the same
script and the same `implementation_digest a329eb7bb06f` that produced every comparator. Only
this driver is new. A second scoring implementation would make the contrast a contrast of
implementations.

COMPARABILITY, fixed by the prereg before any score:
  --instrument select-views-001   bank 7dd53f2a, NOT the dose bank 45674251
  --max-new-tokens 128            the probe's DEFAULT IS 96; all five comparators used 128
Scoring a rung on the wrong bank would reproduce exactly the label-for-bank error I made while
auditing someone else for scope.

The probe is CREATE-ONLY (line 263: refuses if the result exists). It does NOT skip, so this
battery cannot report success having computed nothing -- the failure mode found in the
DOSEPARAM builder. A rerun over an existing artifact raises rather than passing.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

import psutil

REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO.parent / "rineya-dataset-retention-records"
PROBE = REPO / "experiments/rab001_exploration/process_probe.py"
PY = REPO.parent / ".venv-rab001-cu126/Scripts/python.exe"
ARMS = RECORDS / "doseparam001-20260922/arms"
OUT = REPO / "experiments/architecture_benchmark_001/evidence/retention-design-v1"
LOCK = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"

RUNGS = [(1, "rung1-b0.1551407", 0.4616, 40), (2, "rung2-b0.3102815", 0.9232, 37),
         (3, "rung3-b0.4654222", 1.3848, 31), (4, "rung4-b0.6205630", 1.8464, 30)]
MAX_NEW = 128                      # the comparators' value; the probe's default 96 would break it
INSTRUMENT = "select-views-001"    # bank 7dd53f2a -- the ACQUISITION bank
RAM_GATE = 3.0                     # the probe's OWN hostbudget.require(3.0), used UNCHANGED


def free_gib():
    return psutil.virtual_memory().available / 2 ** 30


def foreign_cuda_job():
    """A real model job that is not mine. Matches on RSS, not on a name substring."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                       "--format=csv,noheader"], text=True, timeout=30)
    except Exception:
        return None
    mine = {os.getpid()} | {c.pid for c in psutil.Process(os.getpid()).children(recursive=True)}
    for tok in out.split():
        if not tok.strip().isdigit():
            continue
        pid = int(tok)
        if pid in mine:
            continue
        try:
            p = psutil.Process(pid)
            if "python" in p.name().lower() and p.memory_info().rss >= 0.5 * 2 ** 30:
                return pid
        except Exception:
            continue
    return None


def main():
    if LOCK.exists():
        d = json.loads(LOCK.read_text(encoding="utf-8"))
        raise SystemExit(f"REFUSED: lease held by {d.get('job') or d.get('label')} pid {d.get('pid')}")
    fp = foreign_cuda_job()
    if fp:
        raise SystemExit(f"REFUSED: foreign CUDA model job on the card, pid {fp}")
    if free_gib() < RAM_GATE:
        raise SystemExit(f"REFUSED: host RAM {free_gib():.2f} GiB below the probe's own "
                         f"{RAM_GATE} GiB gate")

    token = uuid.uuid4().hex
    with LOCK.open("x", encoding="utf-8", newline="\n") as f:
        json.dump({"pid": os.getpid(), "token": token, "created": time.time(),
                   "worker_pids": [], "job": "DPACQ-001 acquisition battery",
                   "label": "dpacq001:acquisition:battery",
                   "orphan_after_unix": time.time() + 900,
                   "staleness_rule": ("reclaimable ONLY if holder pid dead AND worker_pids empty "
                                      "AND orphan_after_unix passed AND no live process carries "
                                      "this label")}, f)
    print(f"lease taken. RAM {free_gib():.2f} GiB, gate {RAM_GATE}. "
          f"instrument {INSTRUMENT}, max_new_tokens {MAX_NEW}", flush=True)
    done, failed = [], []
    try:
        for n, arm, norm, dense_ret in RUNGS:
            label = f"doseparam001_rung{n}_select"
            dst = OUT / f"PROCESS_{label}.json"
            if dst.exists():
                print(f"  rung {n}: artifact already exists -- probe would REFUSE; skipping "
                      f"this driver's call, NOT counting it as scored", flush=True)
                done.append((n, None))
                continue
            free = free_gib()
            if free < RAM_GATE:
                print(f"  rung {n}: REFUSED, host RAM {free:.2f} GiB < {RAM_GATE}", flush=True)
                failed.append((n, f"ram {free:.2f}"))
                continue
            cmd = [str(PY), str(PROBE), "--model", str(ARMS / arm), "--label", label,
                   "--device", "cuda", "--max-new-tokens", str(MAX_NEW),
                   "--instrument", INSTRUMENT]
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            if rc.returncode != 0:
                print(f"  rung {n}: rc={rc.returncode} in {time.time()-t0:.0f}s", flush=True)
                print((rc.stderr or rc.stdout or "")[-1200:], flush=True)
                failed.append((n, f"rc {rc.returncode}"))
                continue
            d = json.loads(dst.read_text(encoding="utf-8"))
            acq = d["aggregate"]["canonical_correct"]
            bank = str(d.get("views_sha256", ""))[:8]
            mnt = d.get("max_new_tokens")
            ok = (bank == "7dd53f2a" and mnt == MAX_NEW and d.get("views") == 384)
            print(f"  rung {n}  norm {norm}  acq {acq}/384  dense {dense_ret and ''}"
                  f"  bank {bank}  mnt {mnt}  comparable={ok}  {time.time()-t0:.0f}s", flush=True)
            if not ok:
                failed.append((n, f"COMPARABILITY bank={bank} mnt={mnt}"))
            done.append((n, acq))
    finally:
        if LOCK.exists():
            try:
                if json.loads(LOCK.read_text(encoding="utf-8")).get("token") == token:
                    LOCK.unlink()
                    print("lease released", flush=True)
            except Exception:
                pass
    scored = [x for x in done if x[1] is not None]
    print(f"\nBATTERY {'COMPLETE' if len(scored) == 4 and not failed else 'INCOMPLETE'} "
          f"{len(scored)}/4 scored, {len(failed)} failed  {failed}", flush=True)
    return 0 if (len(scored) == 4 and not failed) else 2


if __name__ == "__main__":
    raise SystemExit(main())
