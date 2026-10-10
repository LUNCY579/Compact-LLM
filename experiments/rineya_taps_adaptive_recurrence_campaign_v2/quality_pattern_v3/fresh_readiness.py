"""Exact fresh parent preflight, no model worker creation."""
import argparse
from datetime import datetime, timezone
import json
from .fresh_scope import V3, SOURCE, sha
from .fresh_supervisor import check_parent


def main(name):
    if not name.startswith("FRESH_READINESS_") or not name.endswith(".json") or "/" in name or "\\" in name:
        raise ValueError("exclusive V3 readiness filename required")
    target = V3 / name
    if target.exists():
        raise FileExistsError("preserve earlier live preflight evidence")
    admission_path = V3 / "fresh_admission_v2.json"
    packet = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "model_loads": 0, "worker_started": False}
    try:
        admission, inventory = check_parent(SOURCE.resolve(), V3 / "runs/native-fresh-open-replication-001", admission_path)
        packet.update(status="READY_FRESH_PARENT_GATES", admission_sha256=sha(admission_path), GPU=inventory,
                      study_id=admission["study_id"])
    except Exception as error:
        packet.update(status="REFUSED_BEFORE_MODEL_WORKER", error=type(error).__name__ + ": " + str(error))
    with target.open("x", encoding="utf-8") as stream:
        json.dump(packet, stream, indent=2); stream.write("\n")
    print(json.dumps(packet))
    return 0 if packet["status"] == "READY_FRESH_PARENT_GATES" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    raise SystemExit(main(parser.parse_args().name))
