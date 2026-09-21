"""Integrity check: was TH_s perturbed by the GPU contention it was generated under?

TH_s ran at 6.28 s/row against SCALED_T's clean 1.69 — ~29 min of another lane's job on the
same card. The contended arm is the one that scored HIGHER, so the risk is not neutral: if
contention perturbed generation it perturbed the arm carrying the +8.

This regenerates a prefix of TH_s on a quiet card and compares byte-for-byte against the
stored rows. It is an INTEGRITY check of an existing arm, not a new endpoint: it introduces no
instrument, scores nothing, and can only BLOCK a contrast, never create one.

Divergence on any row => the contrast is BLOCKED, not caveated.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"
OUT = RECORDS / "compose4b001-20260921"
Q = RECORDS / "qwen001-20260918"
PY = BASE / "Chat-Bubble/.venv/Scripts/python.exe"
MODEL = "C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
LOCK = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
ROWS = 48


def gpu_pids():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=30)
    return {int(x.strip()) for x in out.split() if x.strip().isdigit()}


def foreign_on_card():
    import psutil
    mine = {os.getpid()} | {c.pid for c in psutil.Process(os.getpid()).children(recursive=True)}
    for pid in gpu_pids() - mine:
        try:
            p = psutil.Process(pid)
            if "python" in p.name().lower() and p.memory_info().rss >= 0.5 * 2**30:
                return pid
        except Exception:
            continue
    return None


def main():
    stored = [json.loads(l) for l in (OUT / "gen-th_s/predictions.jsonl").open(encoding="utf-8") if l.strip()]
    if len(stored) != 384:
        raise SystemExit(f"stored TH_s has {len(stored)} rows, expected 384")

    if LOCK.exists():
        raise SystemExit("REFUSED: lease held")
    if foreign_on_card():
        raise SystemExit("REFUSED: foreign CUDA job on the card")
    token = uuid.uuid4().hex
    with LOCK.open("x", encoding="utf-8", newline="\n") as f:
        json.dump({"pid": os.getpid(), "token": token, "created": time.time(),
                   "worker_pids": [], "job": "COMPOSE4B-001 determinism check (short)",
                   "staleness_rule": "measure the card, not this list"}, f)
    try:
        out = OUT / "recheck-th_s"
        if out.exists():
            import shutil; shutil.rmtree(out)
        cmd = [str(PY), str(Q / "stage2_generate.py"), "--model", MODEL, "--label", "th_s_recheck",
               "--prompts", str(Q / "render-process/prompts.jsonl"), "--out", str(out),
               "--max-new-tokens", "96", "--dtype", "bf16", "--four-bit",
               "--adapter", str(OUT / "arm-th-s/adapter"), "--rows", str(ROWS)]
        t0 = time.time()
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode:
            print((rc.stderr or "")[-1500:])
            raise SystemExit("regeneration failed")
        fresh = [json.loads(l) for l in (out / "predictions.jsonl").open(encoding="utf-8") if l.strip()]
    finally:
        if LOCK.exists():
            try:
                if json.loads(LOCK.read_text(encoding="utf-8")).get("token") == token:
                    LOCK.unlink(); print("lease released", flush=True)
            except Exception:
                pass

    by_id = {r["id"]: r for r in stored}
    checked, diffs = 0, []
    for r in fresh:
        rid = r["id"]
        if rid not in by_id:
            diffs.append({"id": rid, "why": "id absent from the stored run"}); continue
        checked += 1
        a, b = r.get("prediction"), by_id[rid].get("prediction")
        if a != b:
            diffs.append({"id": rid, "fresh": str(a)[:200], "stored": str(b)[:200]})

    # POSITIVE CONTROL: the comparison must be capable of detecting a difference.
    probe = dict(fresh[0]); probe["prediction"] = str(probe.get("prediction")) + "_MUTATED"
    detects = probe["prediction"] != by_id[fresh[0]["id"]].get("prediction")

    result = {
        "what": "regenerate a TH_s prefix on a quiet card, compare byte-for-byte to the stored run",
        "why": "TH_s was generated under ~29 min of GPU contention and is the arm carrying the +8",
        "rows_regenerated": len(fresh), "rows_compared": checked,
        "identical": len(diffs) == 0, "divergences": diffs[:10], "n_divergences": len(diffs),
        "positive_control_detects_a_mutation": detects,
        "wall_seconds": round(time.time() - t0, 1),
        "verdict": ("CONTRAST STANDS - generation reproduced exactly on a quiet card"
                    if not diffs and detects and checked == len(fresh)
                    else "CONTRAST BLOCKED - regeneration diverged or the check was vacuous"),
        "model_execution": True, "quality_claim": False,
    }
    (OUT / "DETERMINISM_CHECK.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("rows_compared", "identical", "n_divergences",
                       "positive_control_detects_a_mutation", "wall_seconds", "verdict")}, indent=2))
    return 0 if not diffs and detects else 2


if __name__ == "__main__":
    raise SystemExit(main())
