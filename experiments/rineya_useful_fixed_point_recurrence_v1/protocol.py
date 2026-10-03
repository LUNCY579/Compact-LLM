"""Split admission, immutable output, paired outcomes and qualification preflight."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
from collections import defaultdict
from typing import Iterable

BASE_SHA = "03db218ceebc61a2bc04f2482cf7e094045e01c1"
CHECKPOINT_SHA = "a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    # Exclusive writes prevent silent overwrites of evidence or checkpoints.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def new_run(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def admit_splits(rows: Iterable[dict]) -> list[dict]:
    """Validate all rows TOGETHER before separating TRAIN / DEV / RETENTION.

    Exact normalized prompt duplicates across splits are rejected, as are reused
    cluster IDs. This is a minimum check, not semantic contamination detection.
    """
    rows = list(rows)
    seen_ids, prompts, clusters = set(), {}, {}
    if not rows:
        raise ValueError("empty dataset")
    for row in rows:
        split = row.get("split")
        if split not in {"TRAIN", "DEV", "RETENTION"}:
            raise ValueError("only unsealed TRAIN, DEV, RETENTION rows admitted")
        key, cluster, prompt = (row.get(k) for k in ("case_id", "cluster_id", "prompt"))
        if not all(isinstance(v, str) and v.strip() for v in (key, cluster, prompt)) or key in seen_ids:
            raise ValueError("unique IDs, cluster IDs and nonempty prompts required")
        seen_ids.add(key)
        normalized = " ".join(prompt.split()).casefold()
        if normalized in prompts and prompts[normalized] != split:
            raise ValueError("cross-split duplicate prompt")
        if cluster in clusters and clusters[cluster] != split:
            raise ValueError("cross-split cluster leakage")
        prompts[normalized], clusters[cluster] = split, split
    return rows


def paired_report(rows: Iterable[dict], *, bootstrap_samples: int = 2000, seed: int = 17) -> dict:
    """One arm/seed/split comparison per call; paired CLUSTER bootstrap.

    Pass independent seeds separately. Fractional scores are rejected: success
    must be defined by the frozen task scorer before forming these records.
    """
    rows = list(rows)
    if not rows or type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("nonempty comparison and positive bootstrap count required")
    keys, groups, seen = set(), defaultdict(list), set()
    for row in rows:
        keys.add((row["arm"], row["seed"], row["split"]))
        if row["split"] not in {"DEV", "RETENTION"} or row["case_id"] in seen:
            raise ValueError("unique unsealed evaluation cases required")
        if not isinstance(row.get("cluster_id"), str) or not row["cluster_id"]:
            raise ValueError("cluster ID required")
        if type(row["baseline_correct"]) is not bool or type(row["candidate_correct"]) is not bool:
            raise ValueError("binary correctness must be explicit, not inferred from score")
        seen.add(row["case_id"])
        groups[row["cluster_id"]].append(int(row["candidate_correct"]) - int(row["baseline_correct"]))
    if len(keys) != 1:
        raise ValueError("do not pool arms, seeds, or splits")
    corrections = sum(not r["baseline_correct"] and r["candidate_correct"] for r in rows)
    damages = sum(r["baseline_correct"] and not r["candidate_correct"] for r in rows)
    keys_list = sorted(groups)
    samples, rng = [], random.Random(seed)
    if len(groups) > 1:
        for _ in range(bootstrap_samples):
            draw = [v for k in rng.choices(keys_list, k=len(groups)) for v in groups[k]]
            samples.append(sum(draw) / len(draw))
        samples.sort()
    ci = None if not samples else [samples[int(.025 * (len(samples)-1))], samples[int(.975 * (len(samples)-1))]]
    return {"cases": len(rows), "clusters": len(groups), "corrections": corrections, "damages": damages,
            "net_correction_rate": (corrections-damages)/len(rows), "paired_cluster_bootstrap_95": ci,
            "baseline_accuracy": sum(r["baseline_correct"] for r in rows)/len(rows),
            "candidate_accuracy": sum(r["candidate_correct"] for r in rows)/len(rows),
            "ci_warning": "descriptive DEV interval, not independent confirmation; single-cluster CI omitted"}


def verify_bundle(root: Path, pins: dict[str, str], *, exact: bool = True) -> None:
    """All local model files must be pinned; no symlinks, traversal or pickle."""
    if not root.is_dir() or not pins or root.is_symlink():
        raise ValueError("existing directory and nonempty hash manifest required")
    root = root.resolve()
    actual = set()
    if exact:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("symlinks forbidden in pinned bundle")
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
    if exact and actual != set(pins):
        raise ValueError("bundle file set differs from pinned manifest")
    for name, expected in pins.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid pinned path or digest")
        path = root / relative
        if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts)+1)):
            raise ValueError("symlink in pinned path")
        if not path.is_file() or path.suffix in {".pt", ".pth", ".bin", ".pkl", ".pickle"}:
            raise ValueError("missing file or unsafe serialized artifact")
        if sha256(path) != expected:
            raise ValueError(f"hash mismatch: {name}")


def preflight(permit: dict) -> dict:
    """No imports of Transformers, network, model load, or training in preflight."""
    reasons = []
    for key in ("allow_native_screen", "laptop_pause_cleared"):
        if permit.get(key) is not True:
            reasons.append(f"{key} is not explicitly true")
    if permit.get("experiment_id") != "FP-RINEYA-001" or permit.get("base_sha") != BASE_SHA:
        reasons.append("experiment/base identity mismatch")
    if permit.get("checkpoint_files", {}).get("model.safetensors") != CHECKPOINT_SHA:
        reasons.append("canonical Supa checkpoint unbound")
    for name in ("checkpoint_files", "source_files", "runtime_versions", "host_qualification", "dataset"):
        if not isinstance(permit.get(name), dict) or not permit[name]:
            reasons.append(f"missing {name}")
    if type(permit.get("applications")) is not int or permit.get("applications") != 2 or permit.get("device") != "cpu" or permit.get("dtype") != "float32":
        reasons.append("native bridge remains two-total-application CPU FP32")
    alpha = permit.get("alpha")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 2:
        reasons.append("fixed DEV scale must be explicitly bound")
    return {"status": "BLOCKED" if reasons else "READY_FOR_FILE_AND_RUNTIME_VALIDATION",
            "reasons": reasons, "native_loads": 0, "native_optimizer_steps": 0,
            "qualification_is_not_automatic": True}
