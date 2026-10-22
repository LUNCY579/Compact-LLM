"""Read-only CPU causal replay resource and freeze check; never load a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import psutil
from .run_apparatus import ROOT, PACKAGE, sha
from .causal_replay_extended_authority import validate_human_extension


def main(name):
    admission_path = PACKAGE / "cpu_causal_replay_admission_v4.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    review = json.loads((PACKAGE / admission["review_receipt"]).read_text(encoding="utf-8"))
    mismatches = [p for p, digest in admission["dispatch_hashes"].items() if sha(ROOT / p) != digest]
    assets = [p for p, digest in admission["checkpoint_files"].items() if sha(Path(admission["source"]) / p) != digest]
    ledger = [json.loads(line) for line in (PACKAGE / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    spent = sum(e["native_worker_seconds"] for e in ledger if e.get("event") == "native_completed")
    available = psutil.virtual_memory().available
    disk_free = shutil.disk_usage(PACKAGE).free
    campaign_bytes = sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file())
    reasons = []
    try:
        validate_human_extension(admission)
    except PermissionError as error:
        reasons.append(str(error))
    if mismatches or assets: reasons.append("SOURCE_OR_ASSET_HASH_MISMATCH")
    if review["status"] != "PASS_CPU_CAUSAL_REPLAY_DISPATCH" or review["admission_sha256"] != sha(admission_path): reasons.append("EXACT_REVIEW_REQUIRED")
    if sys.executable != admission["interpreter"]: reasons.append("INTERPRETER_MISMATCH")
    if available < admission["parent_host_available_floor_bytes"]: reasons.append("CPU_PARENT_RAM_FLOOR")
    if disk_free < 15*1024**3 or campaign_bytes > 1536*1024**2: reasons.append("DISK_RESERVE")
    if spent+admission["hard_deadline_seconds"]+1800 > 21600: reasons.append("NATIVE_BUDGET_RESERVE")
    if (PACKAGE / "native_worker.lock").exists(): reasons.append("NATIVE_LOCK_PRESENT")
    if (PACKAGE / "runs/native-cpu-causal-replay-003").exists(): reasons.append("EXCLUSIVE_OUTPUT_ALREADY_EXISTS")
    report = {"status": "READY_CPU_CAUSAL_REPLAY_PARENT_GATES_NOT_NATIVE_QUALIFIED" if not reasons else "NO_DISPATCH",
              "reasons": reasons, "source_mismatches": mismatches, "asset_mismatches": assets,
              "available_host_ram_bytes": available, "required_parent_ram_bytes": admission["parent_host_available_floor_bytes"],
              "disk_free_bytes": disk_free, "campaign_bytes": campaign_bytes,
              "native_seconds_used": spent, "native_seconds_remaining": 21600-spent,
              "admission_sha256": sha(admission_path), "native_worker_created": False,
              "model_loads": 0, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    with (PACKAGE / name).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report))
    return 0 if not reasons else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="CPU_CAUSAL_REPLAY_READINESS_001.json")
    raise SystemExit(main(parser.parse_args().name))
