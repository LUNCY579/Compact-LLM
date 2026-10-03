"""Lossless filesystem compression of this pilot's four immutable checkpoints.

No source weights, model exports, checkpoint contents, manifests, paths or other
jobs are changed. Every tensor file must match its sealed manifest before and
after. Per-batch receipts survive interruption; this does not delete evidence.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def physical_size(path):
    high = ctypes.c_ulong(0)
    function = ctypes.WinDLL("kernel32", use_last_error=True).GetCompressedFileSizeW
    function.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    function.restype = ctypes.c_ulong
    ctypes.set_last_error(0)
    low = function(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    return (high.value << 32) | low


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise ValueError("this operation is Windows NTFS-specific")
    root = args.records.resolve(strict=True)
    if root.name != "rrp015-20260917":
        raise ValueError("only the explicitly owned RRP-015 pilot is in scope")
    args.out.mkdir(parents=True, exist_ok=False)
    entries = []
    for arm, final in (("control", "checkpoint-00028"), ("rehearsal", "checkpoint-00055")):
        for name in ("checkpoint-00002", final):
            directory = (root / f"{arm}-journal/runs/train-001/microfit" / name).resolve(strict=True)
            if not directory.is_relative_to(root) or directory.is_symlink():
                raise ValueError("unexpected checkpoint path")
            manifest = json.loads((directory / "manifest.json").read_text())
            expected = manifest.pop("sha256")
            canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode()
            if hashlib.sha256(canonical).hexdigest() != expected or manifest["status"] != "COMPLETE":
                raise ValueError("invalid or incomplete checkpoint manifest")
            if len(manifest["entries"]) != 220:
                raise ValueError("unexpected tensor inventory")
            for entry in manifest["entries"]:
                path = (directory / entry["file"]).resolve(strict=True)
                if not path.is_relative_to(directory) or path.is_symlink() or not path.is_file():
                    raise ValueError("unexpected tensor path")
                if sha(path) != entry["sha256"]:
                    raise ValueError(f"pre-existing tensor corruption: {path}")
                entries.append({"path": str(path), "sha256": entry["sha256"],
                                "logical_bytes": path.stat().st_size,
                                "physical_before": physical_size(path)})
    write_new(args.out / "INVENTORY.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                           "root": str(root), "files": entries})
    started = time.monotonic()
    for index in range(0, len(entries), 20):
        batch = entries[index:index + 20]
        result = subprocess.run(["compact.exe", "/C", "/EXE:LZX", "/Q",
                                 *[entry["path"] for entry in batch]],
                                capture_output=True, text=True, timeout=120,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        for entry in batch:
            path = Path(entry["path"])
            entry["sha256_after"] = sha(path)
            entry["physical_after"] = physical_size(path)
        valid = all(entry["sha256_after"] == entry["sha256"] for entry in batch)
        write_new(args.out / f"batch-{index // 20:03d}.json", {
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
            "all_content_hashes_unchanged": valid, "files": batch})
        if not valid or result.returncode:
            raise RuntimeError("compression/hash verification failed; inspect retained batch receipt")
        print(f"verified {min(index+20, len(entries))}/{len(entries)}", flush=True)
    summary = {"status": "COMPLETE_BYTES_UNCHANGED", "files": len(entries),
               "logical_bytes": sum(e["logical_bytes"] for e in entries),
               "physical_before": sum(e["physical_before"] for e in entries),
               "physical_after": sum(e["physical_after"] for e in entries),
               "bytes_reclaimed": sum(e["physical_before"] - e["physical_after"] for e in entries),
               "wall_seconds": time.monotonic() - started,
               "model_calls": 0, "files_deleted": 0,
               "limitations": "Storage mechanics only; no new resume or model-quality qualification."}
    write_new(args.out / "SUMMARY.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
