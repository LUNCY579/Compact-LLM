"""Write Q2's hash-selected paired R/S calibration IDs before model output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.maxrl_comp002_pretraining_v1.preflight import digest

SALT = "COMP-RS-001-Q2-SOURCE-SMOKE|"
MANIFEST_SHA = "86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--asset-receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("selection exists (create-only)")
    data = args.data.resolve(strict=True)
    if sha256(data / "manifest.json") != MANIFEST_SHA:
        raise SystemExit("dataset manifest drift")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for relative in ("canonical/calibration.jsonl", "policy/calibration.jsonl"):
        if sha256(data / relative) != manifest["files"][relative]["sha256"]:
            raise SystemExit(f"calibration file drift: {relative}")
    asset = json.loads(args.asset_receipt.read_text(encoding="utf-8"))
    if asset["stage"] != "Q1_ASSET_CANDIDATE":
        raise SystemExit("asset candidate receipt missing")
    rows = {}
    with (data / "canonical/calibration.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["tag"] == "core" and row["skills"] in (["R"], ["S"]):
                parent = row["parent_id"]
                skill = row["skills"][0]
                rows.setdefault(parent, {})[skill] = {"id": row["id"], "parent_id": parent,
                                                     "prompt_hash": row["prompt_hash"], "skill": skill}
    if len(rows) != 150 or any(set(by_skill) != {"R", "S"} for by_skill in rows.values()):
        raise SystemExit("calibration R/S parent alignment mismatch")
    selected = sorted(rows, key=lambda parent: hashlib.sha256((SALT + parent).encode()).hexdigest())[:8]
    scoped = [rows[parent][skill] for parent in selected for skill in ("R", "S")]
    policy_ids = set()
    with (data / "policy/calibration.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            policy_ids.add(json.loads(line)["id"])
    if len(scoped) != 16 or not {r["id"] for r in scoped} <= policy_ids:
        raise SystemExit("selected policy IDs missing")
    receipt = {"experiment": "COMP-RS-001", "stage": "Q2_SOURCE_SMOKE_SELECTION",
               "dataset_manifest_sha256": MANIFEST_SHA,
               "canonical_calibration_sha256": sha256(data / "canonical/calibration.jsonl"),
               "policy_calibration_sha256": sha256(data / "policy/calibration.jsonl"),
               "source_weight_sha256": asset["assets"]["model.safetensors"]["sha256"],
               "salt": SALT, "parent_worlds": selected, "rows": scoped,
               "selection_digest": digest(scoped),
               "group_size": 8, "max_new_tokens": 128,
               "sampler": {"do_sample": True, "temperature": 1.0, "top_p": 1.0,
                           "top_k": 0, "repetition_penalty": 1.0},
               "max_attempts": 128, "max_generated_tokens": 16384,
               "model_outputs_seen": False, "training_authorized": False,
               "note": "Paired source feasibility pilot only; no TEST/CHALLENGE, guard or training read."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"parents": len(selected), "rows": len(scoped), "digest": receipt["selection_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
