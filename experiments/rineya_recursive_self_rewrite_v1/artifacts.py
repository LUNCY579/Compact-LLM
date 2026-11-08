"""Private, exclusive run bundles with replay and recompiled-dataset verification."""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from pathlib import Path

from .core import Task, VERSION, canonical, digest, enroll, loads, require, source_hash
from .data import compile_dataset
from .pipeline import discovery_report

FILES = ("catalog.private.json", "discovery.private.json", "rewrites.private.json", "datasets.private.json")


def write_run(path, tasks, ancestors, discovery, rewrites, datasets):
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    objects = [dict(tasks=[asdict(t) for t in tasks], ancestors=ancestors), discovery, rewrites, datasets]
    hashes = {}
    for name, value in zip(FILES, objects):
        raw = (canonical(value) + "\n").encode()
        with open(root / name, "xb") as stream:
            os.chmod(root / name, 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        hashes[name] = hashlib.sha256(raw).hexdigest()
    all_origins = [r["origin"] for r in discovery["records"]]
    all_origins += [call["origin"] for rewrite in rewrites for call in rewrite["calls"]]
    manifest = dict(schema=VERSION, code_hash=source_hash(), files=hashes,
                    evidence="FIXTURE_ONLY" if all(origin == "fixture" for origin in all_origins) else "UNQUALIFIED_NATIVE",
                    summary=discovery_report(discovery), status="COMPLETE")
    with open(root / "manifest.json", "x", encoding="utf-8") as stream:
        os.chmod(root / "manifest.json", 0o600)
        stream.write(canonical(manifest) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def verify_run(path):
    root = Path(path)
    require(root.is_dir() and not root.is_symlink(), "invalid run directory")
    require({p.name for p in root.iterdir()} == set(FILES) | {"manifest.json"}, "incomplete/unexpected bundle files")
    require(all(not (root / name).is_symlink() for name in (*FILES, "manifest.json")), "symlink in bundle")
    manifest = loads((root / "manifest.json").read_text())
    require(manifest["schema"] == VERSION and manifest["status"] == "COMPLETE", "manifest status")
    require(manifest["code_hash"] == source_hash() and set(manifest["files"]) == set(FILES), "manifest drift")
    values = []
    for name in FILES:
        raw = (root / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == manifest["files"][name], "file hash mismatch")
        values.append(loads(raw.decode("utf-8")))
    catalog, discovery, rewrites, datasets = values
    tasks = [Task.from_dict(t) for t in catalog["tasks"]]
    enroll(tasks, catalog["ancestors"])
    require(set(discovery["task_keys"]) == {t.key for t in tasks if t.split == "train"}, "discovery enrollment")
    require(discovery_report(discovery) == manifest["summary"], "summary mismatch")
    require(bool(datasets), "no compiled datasets")
    for dataset in datasets:
        recomputed = compile_dataset(tasks, catalog["ancestors"], discovery["records"], rewrites,
                                     arm=dataset["arm"], parent_cap=dataset["parent_cap"],
                                     source_ids=dataset["selected_source_ids"])
        require(recomputed == dataset, "recompiled dataset mismatch")
    return dict(verified=True, evidence=manifest["evidence"], source_hash=manifest["code_hash"],
                source_attempts=len(discovery["records"]),
                rewrite_attempts=sum(len(r["records"]) for r in rewrites),
                dataset_actions={d["arm"]: len(d["rows"]) for d in datasets})
