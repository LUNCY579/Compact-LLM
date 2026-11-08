"""Read-only restart gate; never create a model/process/actual human allowance."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import psutil
from .run_apparatus import PACKAGE, ROOT, sha
from .bulk_restart_supervisor import gpu_inventory
from .full_generation_restart_authority import validate_human_extension


def main(name):
    path = PACKAGE / "full_generation_restart_admission_v1.json"
    admission = json.loads(path.read_text(encoding="utf-8"))
    review = json.loads((PACKAGE / admission["review_receipt"]).read_text(encoding="utf-8"))
    reasons = []
    try:
        validate_human_extension(admission)
    except PermissionError as error:
        reasons.append(str(error))
    mismatches = [relative for relative, expected in admission["dispatch_hashes"].items()
                  if sha(ROOT / relative) != expected]
    assets = [relative for relative, expected in admission["checkpoint_files"].items()
              if sha(Path(admission["source"]) / relative) != expected]
    inventory = gpu_inventory()
    available = psutil.virtual_memory().available
    events = [json.loads(line) for line in (PACKAGE / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    spent = sum(row["native_worker_seconds"] for row in events if row.get("event") == "native_completed")
    size = sum(q.stat().st_size for q in PACKAGE.rglob("*") if q.is_file())
    disk = shutil.disk_usage(PACKAGE).free
    if mismatches or assets: reasons.append("SOURCE_OR_ASSET_MISMATCH")
    if review["status"] != "PASS_BULK_RESTART_FP32_DISPATCH" or review["admission_sha256"] != sha(path): reasons.append("EXACT_REVIEW_REQUIRED")
    if sys.executable != admission["interpreter"]: reasons.append("INTERPRETER_MISMATCH")
    if available < admission["parent_host_available_floor_bytes"]: reasons.append("HOST_RAM_FLOOR")
    if inventory["free_bytes"] < admission["gpu_initial_free_floor_bytes"]: reasons.append("GPU_FREE_FLOOR")
    if inventory["uuid"] != admission["gpu_uuid"] or inventory["driver_version"] != admission["gpu_driver_version"]: reasons.append("GPU_IDENTITY_CHANGED")
    if spent + admission["hard_deadline_seconds"] + 1800 > 21600: reasons.append("BUDGET_RESERVE")
    if disk < 15*1024**3 or size > 1536*1024**2: reasons.append("DISK_RESERVE")
    if (PACKAGE / "native_worker.lock").exists(): reasons.append("LIVE_OR_UNKNOWN_NATIVE_LOCK")
    if (PACKAGE / "runs/native-bulk-restart-001").exists(): reasons.append("EXCLUSIVE_OUTPUT_EXISTS")
    report = {"status": "READY_PARENT_GATES_NOT_NATIVE_QUALIFIED" if not reasons else "NO_DISPATCH",
              "reasons": reasons, "source_mismatches": mismatches, "asset_mismatches": assets,
              "admission_sha256": sha(path), "GPU": inventory, "available_host_ram_bytes": available,
              "campaign_bytes": size, "host_disk_free_bytes": disk, "native_seconds_used": round(spent,3),
              "native_seconds_remaining": round(21600-spent,3), "native_worker_created": False,
              "model_loads": 0, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    with (PACKAGE / name).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report))
    return 0 if not reasons else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="FULL_GENERATION_RESTART_READINESS_001.json")
    raise SystemExit(main(parser.parse_args().name))
