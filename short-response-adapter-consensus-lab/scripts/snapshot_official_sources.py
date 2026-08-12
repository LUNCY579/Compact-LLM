from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.discovery.provenance import compare_source_manifests, write_source_manifest
from consensus_lab.io import read_json
from consensus_lab.paths import LabPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("before", "after"), required=True)
    parser.add_argument("--paths", type=Path)
    args = parser.parse_args()
    paths = LabPaths.from_config(args.paths) if args.paths else LabPaths.discover(LAB_ROOT)
    output = LAB_ROOT / f"artifacts/manifests/official_sources_{args.label}.json"
    packet = write_source_manifest(paths, output)
    result: dict[str, object] = {"output": str(output), "manifest_sha256": packet["manifest_sha256"]}
    if args.label == "after":
        before_path = LAB_ROOT / "artifacts/manifests/official_sources_before.json"
        if not before_path.is_file():
            raise FileNotFoundError("The before manifest must exist before creating an after manifest")
        changed = compare_source_manifests(read_json(before_path), packet)
        result["changed_sources"] = changed
        if changed:
            raise RuntimeError("Protected official artifacts changed: " + ", ".join(changed))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

