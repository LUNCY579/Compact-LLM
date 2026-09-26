from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, sha256_file  # noqa: E402


def main() -> int:
    source = LAB_ROOT / "data" / "evaluation" / "mergedev.jsonl"
    destination = LAB_ROOT / "data" / "evaluation" / "mergedev_fast.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    selected = []
    seen_families = set()
    paraphrase_scenario = None
    for row in rows:
        family = row["family"]
        if family == "paraphrase_robustness":
            if paraphrase_scenario is None:
                paraphrase_scenario = row["scenario_id"]
            if row["scenario_id"] == paraphrase_scenario:
                selected.append(row)
            continue
        if family not in seen_families:
            selected.append(row)
            seen_families.add(family)
    selected.sort(key=lambda row: row["case_id"])
    families = Counter(row["family"] for row in selected)
    expected_families = {row["family"] for row in rows}
    if set(families) != expected_families or families["paraphrase_robustness"] != 3:
        raise RuntimeError("balanced fast subset construction failed")
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "1.0",
        "status": "FROZEN_BEFORE_CANDIDATE_SCREENING",
        "source": "mergedev.jsonl",
        "source_sha256": sha256_file(source),
        "selection": "first case per family plus all three variants of the first paraphrase scenario",
        "items": len(selected),
        "scenarios": len({row["scenario_id"] for row in selected}),
        "families": dict(sorted(families.items())),
        "dataset_sha256": sha256_file(destination),
    }
    atomic_write_json(
        LAB_ROOT / "cross_scale_evidence" / "00_protocol" / "mergedev_fast_freeze.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
