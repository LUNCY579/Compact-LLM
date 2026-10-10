"""Project authorized local inventory metadata; never follow inventory paths."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

OUT = Path(__file__).resolve().parent
BASE = Path("C:/Users/clclc/BusinessProject/Rineya-Dataset/backup_20261010")
NAMES = ("INVENTORY_METADATA001.json", "INVENTORY_METADATA002.json", "INVENTORY_METADATA003.json")
TOP_FIELDS = ("schema", "status", "source", "destination", "namespace", "roots", "counts",
              "scope", "restore_scope", "training_eligible")
FILE_FIELDS = ("path", "bytes", "sha256", "status", "role", "backup_role", "owner",
               "training_eligible")


def protected(path):
    parts = [part.lower() for part in PurePosixPath(str(path).replace("\\", "/")).parts]
    return any(part in {"check", "final", "evaluation", "evaluations", "eval", "spent",
                        "private-bank", "private_bank", "sealed", "weights"}
               or part.startswith("sealed_") or part.startswith("private_bank")
               for part in parts)


projections = []
for name in NAMES:
    source = BASE / name
    if source.stat().st_size > 8388608:
        raise RuntimeError("Authorized metadata size limit exceeded")
    raw = source.read_bytes()
    inventory = json.loads(raw.decode("utf-8-sig"))
    files = [{key: entry[key] for key in FILE_FIELDS if key in entry}
             for entry in inventory.get("files", []) if not protected(entry.get("path", ""))]
    by_root = defaultdict(lambda: {"files": 0, "bytes": 0, "roles": Counter()})
    for entry in files:
        root = PurePosixPath(entry["path"]).parts[0]
        by_root[root]["files"] += 1
        by_root[root]["bytes"] += entry.get("bytes", 0)
        by_root[root]["roles"][entry.get("backup_role", entry.get("role", "not_recorded"))] += 1
    projections.append({
        "metadata_source_path": str(source), "metadata_source_bytes": len(raw),
        "metadata_source_sha256": hashlib.sha256(raw).hexdigest(),
        "top_level_schema": {key: type(value).__name__ for key, value in inventory.items()},
        "metadata_fields": {key: inventory[key] for key in TOP_FIELDS if key in inventory},
        "file_inventory_entry_count": len(inventory.get("files", [])),
        "projected_nonprotected_file_count": len(files),
        "protected_metadata_entries_suppressed": len(inventory.get("files", [])) - len(files),
        "files": files, "by_root": dict(by_root),
        "exclusion_reason_counts": dict(Counter(str(x.get("reason")) for x in inventory.get("excluded", []))),
        "file_entries_with_sha256": sum("sha256" in entry for entry in files),
        "file_entries_with_owner": sum("owner" in entry for entry in files),
    })

script = BASE / "prepare_snapshot001.py"
raw = script.read_bytes()
result = {"source_directory": str(BASE),
          "immediate_names": sorted(path.name for path in BASE.iterdir()),
          "inventories": projections,
          "static_script": {"path": str(script), "bytes": len(raw),
              "sha256": hashlib.sha256(raw).hexdigest(), "executed": False,
              "planned_output_namespace": "rineya-dataset-snapshots",
              "planned_output_directory": str(BASE / "prepared"),
              "planned_source_inventory": str(BASE / "INVENTORY_METADATA003.json")},
          "inventory_paths_followed": 0, "source_payload_files_opened": 0}
(OUT / "local_metadata_projection.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
summary = [{key: item[key] for key in ("metadata_source_path", "metadata_source_bytes",
    "metadata_source_sha256", "metadata_fields", "file_inventory_entry_count", "by_root",
    "file_entries_with_sha256", "file_entries_with_owner", "exclusion_reason_counts")}
    for item in projections]
print(json.dumps({"inventories": summary, "static_script": result["static_script"]},
                 ensure_ascii=False, indent=2))
