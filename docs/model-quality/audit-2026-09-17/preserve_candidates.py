"""Inventory and mark surviving research candidates without duplicating weights."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[3]
RECORDS = REPO.parent / "rineya-dataset-retention-records"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    candidates = []
    for seed, directory in ((17, "lc-D1-hf"), (23, "d1r-s23-hf"), (29, "d1r-s29-hf")):
        folder = RECORDS / directory
        receipt = json.loads((folder / "MATERIALIZATION.json").read_text(encoding="utf-8"))
        weight = folder / "model.safetensors"
        actual = sha(weight)
        if actual != receipt["out_weight_sha256"]:
            raise ValueError(f"stored D1 seed {seed} weights differ from receipt")
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in folder.iterdir() if p.is_file() and p.name != "PRESERVE_RESEARCH_CANDIDATE.json"}
        entry = {
            "candidate_id": f"Supa-D1-seed-{seed}-research",
            "selection_seed": seed, "weights_directory": str(folder),
            "weight_sha256": actual, "files": files,
            "materialization_weight_hash_verified": True,
            "source_weight_sha256": receipt["source_weight_sha256"],
            "checkpoint_manifest_sha256_as_recorded": receipt["checkpoint_manifest_sha256"],
            "full_optimizer_checkpoint_present": (Path(receipt["checkpoint"]) / "manifest.json").is_file(),
            "precision": "BF16 stored export; historical evaluation via FP32 DenseReference",
            "status": "PRESERVED_EXPERIMENTAL_CANDIDATE_NOT_RELEASE",
            "do_not_automatically_reclaim_weights": True,
            "no_published_core_registry_change": True,
        }
        marker = folder / "PRESERVE_RESEARCH_CANDIDATE.json"
        if marker.exists():
            if json.loads(marker.read_text()) != entry:
                raise ValueError("existing preservation marker differs")
        else:
            write_new(marker, entry)
        candidates.append(entry)
    original = json.loads((RECORDS / "dose-D1-journal/runs/train-002/microfit/result.json").read_text())
    reconstruction = json.loads((RECORDS / "lc-D1-journal/runs/train-001/microfit/result.json").read_text())
    if original["final_state_sha256"] != reconstruction["final_state_sha256"]:
        raise ValueError("D1 seed17 retained reconstruction is not state-identical")
    value = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": candidates,
        "seed17_original_and_reconstruction_final_state_match": original["final_state_sha256"],
        "current_pilot_checkpoints": {
            arm: str(RECORDS / f"rrp015-20260917/{arm}-journal/runs/train-001/microfit/{checkpoint}")
            for arm, checkpoint in (("control", "checkpoint-00028"), ("rehearsal", "checkpoint-00055"))
        },
        "scope": "Hash verification and local preservation markers only; no model calls, copying, publication or release",
        "limitations": [
            "Historical source/target scores remain exploratory; preservation is not model promotion.",
            "Old D1 examples may overlap the new SELECT-VIEWS instrument; do not directly compare banks.",
            "M4 and asymmetric-dose run directories inspected have no surviving weight files; results and recipes remain.",
            "The markers express preservation policy; they are not filesystem access controls or a second backup.",
        ],
    }
    write_new(Path(__file__).parent / "PRESERVED_CANDIDATES.json", value)
    print(json.dumps({"verified_surviving_D1_models": 3,
                      "seed17_original_state_reproduced": True,
                      "new_weight_bytes": 0, "published": False}))


if __name__ == "__main__":
    main()
