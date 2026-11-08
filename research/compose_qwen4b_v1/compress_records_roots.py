"""OPERATOR-APPROVED lossless NTFS compression of the 13 still-uncompressed records roots.

Approved 2026-09-22 in response to a measured projection of 9.88 GiB. Same mechanism and the
same discipline as `compress_idle_roots.py`, which ran on 1,264 files with 0 removed and 0 hash
mismatches; this is that script generalised to the roots the survey identified.

Transparent filesystem compression ONLY: logical bytes, paths and content are unchanged, and the
operation is reversible with `compact /U`. Nothing is deleted, moved or rewritten.

DISCIPLINE:
  - sha256 recorded BEFORE and verified AFTER for every file. Any mismatch stops EVERYTHING.
  - logical size must be unchanged.
  - create-only audit directory.
  - the 14 GiB floor is checked continuously and is never lowered.
  - progress is streamed per root, so a long run is supervisable without polling the filesystem.

WHY THESE 13. Measured physical >= 97% of logical, i.e. entirely uncompressed: 46.86 GiB across
lwf001, int001, sparse001, lora596, rep001, rrp007-a1-research, indep001, scaledh001, compose001,
rrp015, d1r-s29-hf, d1r-s23-hf, lc-D1-hf.

THE PROJECTION, and why it is trustworthy this time. My previous estimate was 4x high (3.165 GiB
projected, 0.744 actual) because I applied a *safetensors* compression ratio to *JSONL* journals.
This projection is per content type, using ratios measured on the SAME file types in the SAME
tree: 98.5% of these bytes are .safetensors at a measured 0.794, and .jsonl is credited 0.002.
The estimate is 9.88 GiB. The receipt records projected vs actual so the calibration is checkable
rather than asserted.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

RECORDS = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
ROOTS = ["lwf001-20260918", "int001-20260917", "sparse001-20260917", "lora596-20260919",
         "rep001-20260917", "rrp007-a1-research", "indep001-20260921", "scaledh001-20260920",
         "compose001-20260918", "rrp015-20260917", "d1r-s29-hf", "d1r-s23-hf", "lc-D1-hf"]
OUT = RECORDS / "storage-compress-20260922"
FLOOR = 15032385536          # 14 GiB, unchanged and never lowered
PROJECTED_GIB = 9.88

_G = ctypes.windll.kernel32.GetCompressedFileSizeW
_G.restype = ctypes.c_ulong


def physical(p):
    hi = ctypes.c_ulong(0)
    lo = _G(ctypes.c_wchar_p(str(p)), ctypes.byref(hi))
    return (hi.value << 32) | lo


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    if os.name != "nt":
        raise SystemExit("requires Windows NTFS compact")
    if OUT.exists():
        raise SystemExit("create-only audit directory already exists")
    missing = [r for r in ROOTS if not (RECORDS / r).is_dir()]
    if missing:
        raise SystemExit(f"refusing: roots missing {missing}")
    OUT.mkdir(parents=True)

    started = time.time()
    before_free = shutil.disk_usage(RECORDS).free
    records, failures = [], []
    n_files = 0

    for root in ROOTS:
        d = RECORDS / root
        t0, saved0 = time.time(), 0
        files = sorted(p for p in d.rglob("*") if p.is_file() and not p.is_symlink())
        for p in files:
            free = shutil.disk_usage(RECORDS).free
            if free < FLOOR:
                (OUT / "RESULT.json").write_text(json.dumps(
                    {"status": "REFUSED_MID_RUN_BELOW_FLOOR", "free_bytes": free,
                     "floor_bytes": FLOOR, "files_done": n_files,
                     "lowered_any_gate": False}, indent=2) + "\n", encoding="utf-8")
                raise SystemExit(f"REFUSED mid-run: free {free/2**30:.2f} GiB below the 14 GiB floor")
            rp = p.resolve()
            if not rp.is_relative_to(d.resolve()):
                raise SystemExit(f"path escaped its root: {p}")
            before_sha = sha(p)
            before_len = p.stat().st_size
            before_phys = physical(p)
            rc = subprocess.run(["compact.exe", "/C", "/EXE:LZX", "/Q", str(p)],
                                capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW).returncode
            after_sha = sha(p)
            after_len = p.stat().st_size
            after_phys = physical(p)
            n_files += 1
            if not (rc == 0 and after_sha == before_sha and after_len == before_len):
                failures.append({"path": str(p), "rc": rc,
                                 "sha_before": before_sha, "sha_after": after_sha,
                                 "len_before": before_len, "len_after": after_len})
                break  # a hash or length change means the content moved. Stop immediately.
            saved0 += before_phys - after_phys
            records.append({"path": str(p.relative_to(RECORDS)), "sha256": before_sha,
                            "logical": before_len, "phys_before": before_phys,
                            "phys_after": after_phys})
        print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {root:26s} "
              f"{len(files):5d} files  {saved0/2**30:6.2f} GiB saved  "
              f"{time.time()-t0:6.0f}s  (running total {sum(r['phys_before']-r['phys_after'] for r in records)/2**30:.2f} GiB)",
              flush=True)
        if failures:
            break

    after_free = shutil.disk_usage(RECORDS).free
    saved = sum(r["phys_before"] - r["phys_after"] for r in records)
    logical = sum(r["logical"] for r in records)
    result = {
        "what": "transparent NTFS LZX compression; logical bytes, paths and content unchanged",
        "reversible_with": "compact /U",
        "operator_approved": True,
        "roots": ROOTS,
        "files_processed": n_files,
        "files_verified_identical": len(records),
        "files_removed": 0, "files_moved": 0, "files_rewritten": 0,
        "sha256_verified_before_and_after": True,
        "physical_bytes_saved": saved,
        "physical_gib_saved": round(saved / 2**30, 3),
        "projected_gib_saved": PROJECTED_GIB,
        "projection_error_ratio": round(PROJECTED_GIB / max(saved / 2**30, 1e-9), 3),
        "projection_note": ("the previous estimate was 4.25x high because a safetensors ratio was "
                            "applied to JSONL journals; this one was built per content type from "
                            "ratios measured on the same file types in the same tree"),
        "logical_bytes_preserved": logical,
        "logical_gib_preserved": round(logical / 2**30, 3),
        "free_gib_before": round(before_free / 2**30, 3),
        "free_gib_after": round(after_free / 2**30, 3),
        "floor_bytes": FLOOR, "lowered_any_gate": False,
        "failures": failures,
        "status": "HASH_VERIFIED_COMPLETE" if not failures else "STOPPED_ON_MISMATCH",
        "wall_seconds": round(time.time() - started, 1),
        "model_execution": False, "quality_claim": False,
    }
    (OUT / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT / "FILES.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("status", "files_verified_identical", "files_removed",
                       "physical_gib_saved", "projected_gib_saved", "projection_error_ratio",
                       "logical_gib_preserved", "free_gib_before", "free_gib_after",
                       "lowered_any_gate", "wall_seconds")}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
