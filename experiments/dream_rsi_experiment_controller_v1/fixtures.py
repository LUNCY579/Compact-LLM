"""Generated synthetic test worlds. None contains a Rineya measurement."""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

from .replay import Plan
from .schema import digest, identity_keys, write_new


def outcome(capability: float | None = 0.5, retention: float | None = 0.8,
            status: str = "measured", failure: str = "none") -> dict:
    return {"status": status, "qualified": status == "measured", "protocol_valid": True,
            "capability": capability, "retention": retention, "gpu_seconds": 2.0,
            "wall_seconds": 3.0, "peak_host_gib": 4.0, "peak_vram_gib": 3.0,
            "parameter_delta": 0, "latency_ratio": 1.0, "failure": failure}


def make_world(key: str = "fixture", seed: int = 1) -> dict:
    rng = random.Random(seed)
    templates = [
        [(0.54, 0.80), (0.55, 0.80), (0.55, 0.80), (0.54, 0.80)],
        [(None, None), (0.58, 0.80), (0.62, 0.80), (0.65, 0.80)],
        [(0.90, 0.35), (0.95, 0.35), (0.57, 0.79), (0.61, 0.80)],
        [(0.48, 0.82), (0.50, 0.82), (0.56, 0.82), (0.70, 0.82)],
    ]
    rng.shuffle(templates)
    nodes = []
    for branch, template in enumerate(templates):
        parent = "root"
        for depth, (cap, ret) in enumerate(template):
            key_id = f"fixture-source-{branch}-{depth}"
            measured = (outcome(None, None, "failed", "implementation") if cap is None else
                        outcome(round(max(0, min(1, cap + rng.uniform(-0.02, 0.02))), 6), ret))
            nodes.append({"id": key_id, "parent": parent, "order": len(nodes) + 1,
                          "context": ["root"] if parent == "root" else ["root", parent],
                          "outcome": measured})
            parent = key_id
    identity = {"checkpoint_sha256": digest([key, "synthetic-checkpoint"]),
                "protocol_sha256": digest("synthetic-protocol-v1"),
                "dataset_sha256": digest([key, "synthetic-data"]),
                "study_family": key, "task_family": key, "lineage_family": key}
    return {"schema": "rineya-dream-world-v1", "id": key, "kind": "synthetic",
            "closed": True, "sealed": False, "identity": identity,
            "leakage_keys": list(identity_keys(identity)),
            "baseline": outcome(), "nodes": nodes, "admission": None}


def create_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for split, size in (("train", 3), ("validation", 2), ("test", 5)):
        for index in range(size):
            key = f"synthetic-{split}-{index}"
            world = make_world(key, len(entries) * 137 + 17)
            filename = key + ".json"
            write_new(root / filename, world)
            entries.append({"id": key, "path": filename, "sha256": digest(world),
                            "split": split, "leakage_keys": world["leakage_keys"]})
    write_new(root / "catalog.json", {"schema": "rineya-dream-catalog-v1", "kind": "synthetic", "worlds": entries})
    write_new(root / "plan.json", asdict(Plan()))
    return root / "catalog.json"
