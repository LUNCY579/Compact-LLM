"""Operator-approved lossless NTFS compression of three idle records roots.

Transparent filesystem compression only: logical bytes, paths and content are unchanged, and
the operation is reversible with `compact /U`. This is the mechanism the programme already
validated on 664 DOSEINT-001 files.

DISCIPLINE, matching compress_evidence.py:
  - sha256 recorded BEFORE and verified AFTER for every file. Any mismatch stops everything.
  - logical size must be unchanged.
  - create-only audit directory; nothing is ever deleted, moved or rewritten.
  - the 14 GiB floor is checked continuously and never lowered.

WHY THESE THREE. process-a1-journal-{002,s23,s29}: 5.5 days idle, the oldest large roots in the
tree, 3.516 GiB logical each and 0.000 GiB saved (entirely uncompressed). Approved by the
operator. No file here is deleted -- if any of these turn out to be receipt-pinned evidence,
compression preserves them byte-for-byte anyway, which is precisely why compression rather than
deletion is the sanctioned route.
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
ROOTS = ["process-a1-journal-002", "process-a1-journal-s23", "process-a1-journal-s29"]
OUT = RECORDS / "storage-compress-20260921"
FLOOR = 15032385536

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
    OUT.mkdir(parents=True)

    started = time.time()
    before_free = shutil.disk_usage(RECORDS).free
    records, failures = [], []
    n_files = 0

    for root in ROOTS:
        d = RECORDS / root
        if not d.is_dir():
            failures.append({"root": root, "why": "missing"})
            continue
        files = sorted(p for p in d.rglob("*") if p.is_file() and not p.is_symlink())
        for p in files:
            if shutil.disk_usage(RECORDS).free < FLOOR:
                raise SystemExit("REFUSED mid-run: free disk fell below the 14 GiB floor")
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
            ok = (rc == 0 and after_sha == before_sha and after_len == before_len)
            n_files += 1
            if not ok:
                failures.append({"path": str(p), "rc": rc,
                                 "sha_before": before_sha, "sha_after": after_sha,
                                 "len_before": before_len, "len_after": after_len})
                # A hash or length change means the content moved. Stop immediately.
                break
            records.append({"path": str(p.relative_to(RECORDS)), "sha256": before_sha,
                            "logical": before_len, "phys_before": before_phys,
                            "phys_after": after_phys})
        if failures:
            break

    after_free = shutil.disk_usage(RECORDS).free
    saved = sum(r["phys_before"] - r["phys_after"] for r in records)
    result = {
        "what": "transparent NTFS LZX compression; logical bytes, paths and content unchanged",
        "reversible_with": "compact /U",
        "roots": ROOTS,
        "files_processed": n_files,
        "files_verified_identical": len(records),
        "files_removed": 0, "files_moved": 0, "files_rewritten": 0,
        "sha256_verified_before_and_after": True,
        "physical_bytes_saved": saved,
        "physical_gib_saved": round(saved / 2**30, 3),
        "logical_bytes_preserved": sum(r["logical"] for r in records),
        "free_before": before_free, "free_after": after_free,
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
                       "physical_gib_saved", "free_gib_before", "free_gib_after",
                       "lowered_any_gate", "wall_seconds")}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
