"""ARMPERP-001 battery: the THREE legs registered in ARMPERP_PREREG.md section 5.

RETENTION IS NOT RUN. Excluded by the prereg in advance, with the reason: across all 28 pairs
of this family the maximum discordance on the n=47 endpoint is 3 rows, against the >= 6 an
exact two-sided McNemar needs to reach p<0.05. It could not resolve this arm whatever it scores.
Its absence is registered, not a post-hoc drop.

Invocations are COPIED from indep005-20260921/run_indep5.sh rather than reconstructed -- the
comparators were produced by those exact strings. process_probe's default max_new_tokens is 96
and it is LEFT ALONE: this family is the 96-token family, and DPACQ's 128 would break it.
"""
from pathlib import Path
import json, os, shutil, subprocess, sys, tempfile, time, uuid
import psutil

REPO = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-dose-param")
REC = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
PY = Path(r"C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe")
ARM = REC / "armperp001-20260923/arm-perp"
OUT = REC / "armperp001-20260923/evidence"
EVID = REPO / "experiments/architecture_benchmark_001/evidence/retention-design-v1"
LOCK = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
LABEL = "armperp001_perp"
RAM_GATE = 3.0


def free_gib():
    return psutil.virtual_memory().available / 2 ** 30


def foreign_cuda_job():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                       "--format=csv,noheader"], text=True, timeout=30)
    except Exception:
        return None
    mine = {os.getpid()} | {c.pid for c in psutil.Process(os.getpid()).children(recursive=True)}
    for tok in out.split():
        if not tok.strip().isdigit() or int(tok) in mine:
            continue
        try:
            p = psutil.Process(int(tok))
            if "python" in p.name().lower() and p.memory_info().rss >= 0.5 * 2 ** 30:
                return int(tok)
        except Exception:
            continue
    return None


def main():
    if LOCK.exists():
        d = json.loads(LOCK.read_text(encoding="utf-8"))
        raise SystemExit(f"REFUSED: lease held by {d.get('job')} pid {d.get('pid')}")
    if (fp := foreign_cuda_job()):
        raise SystemExit(f"REFUSED: foreign CUDA model job on the card, pid {fp}")
    if free_gib() < RAM_GATE:
        raise SystemExit(f"REFUSED: host RAM {free_gib():.2f} GiB below the probe's {RAM_GATE}")
    if not (ARM / "model.safetensors").exists():
        raise SystemExit(f"REFUSED: arm not built at {ARM}")
    OUT.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex
    with LOCK.open("x", encoding="utf-8", newline="\n") as f:
        json.dump({"pid": os.getpid(), "token": token, "created": time.time(), "worker_pids": [],
                   "job": "ARMPERP-001 battery", "label": "armperp001:battery",
                   "orphan_after_unix": time.time() + 900,
                   "staleness_rule": "holder pid dead AND worker_pids empty AND orphan_after_unix "
                                     "passed AND no live process carries this label"}, f)
    print(f"lease taken. RAM {free_gib():.2f} GiB. arm {ARM}", flush=True)

    hd = OUT / f"HSKILL2_{LABEL}.json.d"
    legs = [
        ("acquisition", EVID / f"PROCESS_{LABEL}.json",
         [str(PY), "experiments/rab001_exploration/process_probe.py", "--model", str(ARM),
          "--label", LABEL, "--device", "cuda", "--instrument", "select-views-001"]),
        ("longctx", EVID / f"LONGCTX_{LABEL}.json",
         [str(PY), "experiments/rab001_exploration/long_context_probe.py", "--model", str(ARM),
          "--label", LABEL, "--device", "cuda"]),
        ("hskill", OUT / f"HSKILL2_{LABEL}.json",
         [str(PY), str(REC / "hskill-views-002/hskill_probe_v2.py"), "--model", str(ARM),
          "--label", LABEL, "--instrument", str(REC / "hskill-views-002/HSKILL_VIEWS_V2.jsonl"),
          "--out", str(hd), "--device", "cuda"]),
    ]
    failed = []
    try:
        for name, dst, cmd in legs:
            if dst.exists():
                print(f"  {name}: artifact exists -- NOT counting as scored", flush=True)
                failed.append((name, "pre-existing artifact"))
                continue
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            if rc.returncode != 0:
                print(f"  {name}: rc={rc.returncode} in {time.time()-t0:.0f}s", flush=True)
                print((rc.stderr or rc.stdout or "")[-1500:], flush=True)
                failed.append((name, f"rc {rc.returncode}"))
                continue
            if name == "hskill" and (hd / f"HSKILL2_{LABEL}.json").exists():
                shutil.move(str(hd / f"HSKILL2_{LABEL}.json"), str(dst))
                try:
                    hd.rmdir()
                except OSError:
                    pass
            print(f"  {name}: ok {time.time()-t0:.0f}s -> {dst.name}", flush=True)
    finally:
        if LOCK.exists():
            try:
                if json.loads(LOCK.read_text(encoding="utf-8")).get("token") == token:
                    LOCK.unlink(); print("lease released", flush=True)
            except Exception:
                pass

    for p in EVID.glob(f"*{LABEL}*.json"):
        shutil.copy2(p, OUT / p.name)
    print(f"\nBATTERY {'COMPLETE' if not failed else 'INCOMPLETE'} 3/3, {len(failed)} failed {failed}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
