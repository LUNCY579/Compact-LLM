"""Freeze exposed R/S/RS DEV identities and world clustering before model output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.maxrl_comp002_pretraining_v1.preflight import digest

SIGNATURES = ("R", "S", "RS")
MANIFEST_SHA = "86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("scope receipt exists (create-only)")
    data = args.data.resolve(strict=True)
    if sha256(data / "manifest.json") != MANIFEST_SHA:
        raise SystemExit("dataset manifest drift")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    canonical = data / "canonical/dev.jsonl"
    policy = data / "policy/dev.jsonl"
    for name, path in (("canonical/dev.jsonl", canonical), ("policy/dev.jsonl", policy)):
        if sha256(path) != manifest["files"][name]["sha256"]:
            raise SystemExit(f"DEV file drift: {name}")
    groups = {signature: {"ids": [], "parents": []} for signature in SIGNATURES}
    all_ids = set()
    with canonical.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in all_ids:
                raise SystemExit("duplicate DEV ID")
            all_ids.add(row["id"])
            sig = "".join(row["skills"])
            if row["tag"] == "core" and sig in groups:
                groups[sig]["ids"].append(row["id"])
                groups[sig]["parents"].append(row["parent_id"])
    if len(all_ids) != 4150:
        raise SystemExit("DEV count mismatch")
    policy_ids = set()
    with policy.open("r", encoding="utf-8") as stream:
        for line in stream:
            policy_ids.add(json.loads(line)["id"])
    if policy_ids != all_ids:
        raise SystemExit("policy/canonical DEV ID mismatch")
    sets = {signature: set(group["parents"]) for signature, group in groups.items()}
    if any(len(groups[sig]["ids"]) != 250 or len(sets[sig]) != 250 for sig in SIGNATURES):
        raise SystemExit("R/S/RS parent count mismatch")
    if not sets["R"] == sets["S"] == sets["RS"]:
        raise SystemExit("R/S/RS world alignment mismatch")
    receipt = {"experiment": "COMP-RS-001", "stage": "Q1_EXPOSED_DEV_SCOPE",
               "dataset_manifest_sha256": MANIFEST_SHA,
               "canonical_dev_sha256": sha256(canonical), "policy_dev_sha256": sha256(policy),
               "signatures": {sig: {"rows": len(groups[sig]["ids"]), "parents": len(sets[sig]),
                                    "ids_sha256": digest(sorted(groups[sig]["ids"])),
                                    "parents_sha256": digest(sorted(sets[sig]))}
                              for sig in SIGNATURES},
               "shared_parent_worlds_R_S_RS": len(sets["R"]),
               "test_challenge_read": False, "model_outputs_read": False,
               "note": "The 250 exposed DEV parent worlds each have aligned atomic R, atomic S and dependent RS views."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"shared_parents": receipt["shared_parent_worlds_R_S_RS"],
                      "rows_per_signature": {k: v["rows"] for k, v in receipt["signatures"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
