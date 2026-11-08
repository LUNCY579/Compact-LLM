"""Phase 1 driver: audit the panel, classify every pair, write the five artifacts.

Outputs (each with a sibling ``*.provenance.json``)::

    manifests/model_manifest.json
    manifests/tensor_hashes.csv
    results/model_pairwise_distance.csv     all pairs, aggregated over shared tensors
    results/weight_delta_by_layer.csv       m vs base, per tensor and per (layer, family)
    results/weight_delta_by_head.csv        m vs base, per (layer, head_kind, head)

Resumable and idempotent. Per-model audits and per-pair deltas are cached under
``manifests/.audit_cache/`` keyed on a cheap directory fingerprint (names, sizes,
mtimes) plus the numeric parameters of the run; a second invocation re-reads the
cache, finds every artifact already byte-correct, and writes nothing. Touching a
checkpoint invalidates only that model and the pairs it participates in.

Run as::

    PYTHONPATH=src python -m rineya_psa.model_audit.run_audit
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import functools
import importlib
import io
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from rineya_psa import paths
from rineya_psa.model_audit import distances as dist
from rineya_psa.model_audit.hashing import (
    CHUNK_BYTES,
    collect_tensor_locations,
    directory_fingerprint,
    discover_model_files,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from rineya_psa.model_audit.inspect_ckpt import (
    AUDIT_SCHEMA_VERSION,
    CheckpointAudit,
    PairClassification,
    audit_checkpoint,
    classify_pair,
)
from rineya_psa.types import Arch, ModelSpec

MANIFEST_NAME = "model_manifest.json"
TENSOR_HASHES_NAME = "tensor_hashes.csv"
PAIRWISE_NAME = "model_pairwise_distance.csv"
BY_LAYER_NAME = "weight_delta_by_layer.csv"
BY_HEAD_NAME = "weight_delta_by_head.csv"

CACHE_DIRNAME = ".audit_cache"
CACHE_SCHEMA_VERSION = "1.0"

PAIRWISE_SCOPE = "all_shared_tensors"

_METRIC_COLUMNS: tuple[str, ...] = (
    "n_params",
    "frobenius",
    "base_frobenius",
    "other_frobenius",
    "rel_frobenius",
    "cosine",
    "max_abs_delta",
    "max_abs_base",
    "rmse",
    "frac_changed",
    "frac_changed_rel",
)


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


def _fmt(value: Any) -> str:
    """CSV cell text. Floats keep full round-trip precision; ``None`` is empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _rows_to_csv(columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_fmt(row.get(c)) for c in columns])
    return buf.getvalue()


def pair_order(panel: Sequence[str], base_key: str) -> list[tuple[str, str]]:
    """All unordered pairs, canonically oriented with the base model first.

    Relative Frobenius and the base norm are asymmetric, so the pair's orientation
    is not free: fixing it here means the per-layer and per-head tables never have
    to flip a payload they cannot legitimately flip.
    """
    index = {k: i for i, k in enumerate(panel)}
    pairs = [(b, a) if b == base_key else (a, b) for a, b in combinations(panel, 2)]
    pairs.sort(key=lambda p: (p[0] != base_key, index[p[0]], index[p[1]]))
    return pairs


def provenance_path(artifact: Path) -> Path:
    """Sibling provenance file for an artifact, per SPEC.md 9."""
    return artifact.with_suffix(artifact.suffix + ".provenance.json")


def _write_if_changed(path: Path, text: str) -> bool:
    """Write only when the bytes would actually change. Returns True if written."""
    paths.assert_readonly_target(path)
    data = text.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def _git_info(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["git", *args],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except Exception:  # noqa: BLE001 - provenance must never break the run
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "torch", "transformers", "safetensors", "pandas", "pyyaml"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
        except Exception:  # noqa: BLE001 - metadata lookups are best effort
            out[name] = None
    return out


def _dtype_policy(panel: Sequence[str]) -> dict[str, Any]:
    """Record the run-dtype approximation SPEC.md §4 requires the manifest to carry.

    ``loading`` imports torch, so it is imported lazily and the policy degrades to
    "unknown" rather than dragging a CUDA context into a CPU-only audit host.
    """
    try:
        from rineya_psa import loading
    except Exception as exc:  # noqa: BLE001 - torch may be absent on an analysis box
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {"available": True, "models": {}}
    for key in panel:
        try:
            out["models"][key] = {
                "store_dtype": paths.model_spec(key).store_dtype,
                "run_dtype": str(loading.run_dtype(key)).removeprefix("torch."),
                "is_approximation": bool(loading.dtype_is_approximation(key)),
            }
        except Exception as exc:  # noqa: BLE001
            out["models"][key] = {"error": f"{type(exc).__name__}: {exc}"}
    out["note"] = (
        "is_approximation=true means the lab runs the model in a narrower dtype than "
        "the checkpoint stores (Halo: float32 store, bfloat16 run, per SPEC.md 4). "
        "All Phase 1 weight distances are computed from the stored bytes in float64 "
        "and are unaffected."
    )
    return out


def environment_stamp(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """SPEC.md section 9's stamp, from ``statistics/provenance.py`` when importable.

    That module is owned by another phase and may not exist on every checkout, so a
    local equivalent (git, packages, interpreter, platform) stands in and the choice
    is recorded in ``stamp_source``. Either way the stamp is *not* what decides
    whether a provenance file gets rewritten -- see ``idempotency_key`` below --
    because it carries wall-clock times and a git dirty-path list that this very run
    mutates.
    """
    try:
        from rineya_psa.statistics.provenance import stamp

        return dict(stamp(config, repo_root=paths.REPO_ROOT)), (
            "rineya_psa.statistics.provenance.stamp"
        )
    except Exception as exc:  # noqa: BLE001 - the canonical stamp is optional
        return (
            {
                "git": _git_info(paths.REPO_ROOT),
                "packages": _package_versions(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            },
            "rineya_psa.model_audit.run_audit (local fallback)",
        )


def _provenance_payload(
    artifact: str,
    artifact_text: str,
    *,
    config: Mapping[str, Any],
    audits: Mapping[str, CheckpointAudit],
    stamp: Mapping[str, Any],
    stamp_source: str,
) -> dict[str, Any]:
    model_sha: dict[str, Any] = {}
    for key, audit in audits.items():
        model_sha[key] = {
            "path": audit.path,
            "dir_fingerprint": audit.dir_fingerprint,
            "config_normalized_sha256": audit.config_normalized_sha256,
            "tokenizer_identity_sha256": audit.tokenizer["identity_sha256"],
            "files": {
                f["name"]: f["sha256"] for f in audit.files if f["name"] in audit.model_files
            },
        }
    # Everything in ``stable`` is a function of the inputs, never of the clock.
    stable = {
        "artifact": artifact,
        "artifact_sha256": sha256_bytes(artifact_text.encode("utf-8")),
        "producer": "rineya_psa.model_audit.run_audit",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "config": dict(config),
        "config_hash": sha256_json(config),
        # The audit is a pure function of bytes on disk; no RNG is consulted.
        "seeds": {},
        "model_sha256": model_sha,
    }
    return {
        **stable,
        "idempotency_key": sha256_json(stable),
        "stamp": dict(stamp),
        "stamp_source": stamp_source,
    }


def _write_provenance(artifact_path: Path, payload: Mapping[str, Any]) -> bool:
    """Write the sibling provenance, but only when something other than the clock moved.

    The comparison is on ``idempotency_key`` rather than on the whole body because
    the environment stamp legitimately changes between runs (timestamps, and a git
    dirty-path list that includes the artifacts being written).
    """
    prov_path = provenance_path(artifact_path)
    body = dict(payload)
    if prov_path.exists():
        try:
            existing = json.loads(prov_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("idempotency_key") == body.get(
            "idempotency_key"
        ):
            return False
    body_with_time = {**body, "generated_at": _dt.datetime.now(_dt.UTC).isoformat()}
    paths.assert_readonly_target(prov_path)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(
        json.dumps(body_with_time, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #


@functools.cache
def _module_digest(module_name: str) -> str:
    module = importlib.import_module(module_name)
    source = getattr(module, "__file__", None)
    return sha256_file(source) if source and Path(source).is_file() else "unavailable"


def code_fingerprint(*module_names: str) -> str:
    """Digest of the source that produces a cached result.

    A directory fingerprint alone is not a sufficient cache key: editing the audit
    and re-running would happily serve a result computed by the previous version.
    Folding the producing modules' own bytes into the key means a code change
    invalidates exactly the caches that depend on it, and nothing else.
    """
    return sha256_json({name: _module_digest(name) for name in module_names})


AUDIT_CODE_MODULES: tuple[str, ...] = (
    "rineya_psa.model_audit.hashing",
    "rineya_psa.model_audit.inspect_ckpt",
)
DELTA_CODE_MODULES: tuple[str, ...] = (
    "rineya_psa.model_audit.hashing",
    "rineya_psa.model_audit.distances",
)


_CACHE_GITIGNORE = (
    "# Derived from the checkpoints on disk by rineya_psa.model_audit.run_audit.\n"
    "# Regenerated automatically; never a source of truth. Not for version control.\n"
    "*\n"
)


def _prepare_cache_dir(cache_dir: Path) -> None:
    """Create the cache directory and make it self-ignoring.

    The cache holds a few MB of derived JSON per panel. Marking it here keeps it out
    of git without requiring an edit to the repository's shared ``.gitignore``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / ".gitignore"
    if not marker.exists():
        marker.write_text(_CACHE_GITIGNORE, encoding="utf-8")


def _cache_read(path: Path, expected_key: str) -> Any | None:
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict) or blob.get("cache_key") != expected_key:
        return None
    return blob.get("payload")


def _cache_write(path: Path, cache_key: str, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cache_key": cache_key, "payload": payload}, ensure_ascii=False),
        encoding="utf-8",
    )


def load_or_audit(
    key: str,
    *,
    spec: ModelSpec,
    cache_dir: Path,
    force: bool = False,
    probe_transformers: bool = True,
    chunk_size: int = CHUNK_BYTES,
) -> tuple[CheckpointAudit, bool]:
    """Return ``(audit, from_cache)``, recomputing only when the directory moved."""
    fingerprint = directory_fingerprint(spec.path)
    cache_key = sha256_json(
        {
            "v": CACHE_SCHEMA_VERSION,
            "schema": AUDIT_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "path": str(Path(spec.path).resolve()),
            "probe": probe_transformers,
            "code": code_fingerprint(*AUDIT_CODE_MODULES),
        }
    )
    cache_path = cache_dir / f"audit_{key}.json"
    if not force:
        cached = _cache_read(cache_path, cache_key)
        if cached is not None:
            return CheckpointAudit.from_json(cached), True
    audit = audit_checkpoint(
        key, spec=spec, chunk_size=chunk_size, probe_transformers=probe_transformers
    )
    _cache_write(cache_path, cache_key, audit.to_json())
    return audit, False


def _pair_cache_key(
    a: CheckpointAudit, b: CheckpointAudit, *, rel_tol: float, chunk_bytes: int, per_head: bool
) -> str:
    return sha256_json(
        {
            "v": CACHE_SCHEMA_VERSION,
            "a": a.dir_fingerprint,
            "b": b.dir_fingerprint,
            "rel_tol": rel_tol,
            "chunk_bytes": chunk_bytes,
            "per_head": per_head,
            "code": code_fingerprint(*DELTA_CODE_MODULES),
        }
    )


def _pair_delta_to_json(pd: dist.PairDelta) -> dict[str, Any]:
    return {
        "base_key": pd.base_key,
        "other_key": pd.other_key,
        "rel_tol": pd.rel_tol,
        "orientation": pd.orientation.to_json() if pd.orientation else None,
        "overall": pd.overall_metrics(),
        "skipped": list(pd.skipped),
        "per_tensor": [
            {
                "name": td.key.name,
                "layer": td.key.layer,
                "family": td.key.family,
                "param": td.key.param,
                "shape": list(td.shape),
                "base_dtype": td.base_dtype,
                "other_dtype": td.other_dtype,
                "metrics": td.metrics(),
                "layout": (
                    {
                        "axis": td.layout.axis,
                        "n_heads": td.layout.n_heads,
                        "head_dim": td.layout.head_dim,
                        "space": td.layout.space,
                        "evidence": td.layout.evidence,
                    }
                    if td.layout
                    else None
                ),
                "heads": {str(h): acc.metrics() for h, acc in sorted(td.heads.items())},
            }
            for td in pd.per_tensor
        ],
        "by_layer_family": [
            {"layer": layer, "family": family, "metrics": acc.metrics()}
            for (layer, family), acc in sorted(
                pd.by_layer_family.items(),
                key=lambda kv: (kv[0][0], dist.family_sort_key(kv[0][1])),
            )
        ],
    }


def load_or_compute_pair(
    base: CheckpointAudit,
    other: CheckpointAudit,
    *,
    cache_dir: Path,
    arch: Arch | None,
    per_head: bool,
    rel_tol: float,
    chunk_bytes: int,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    cache_key = _pair_cache_key(
        base, other, rel_tol=rel_tol, chunk_bytes=chunk_bytes, per_head=per_head
    )
    cache_path = cache_dir / f"delta_{base.key}__{other.key}.json"
    if not force:
        cached = _cache_read(cache_path, cache_key)
        if cached is not None:
            return cached, True

    base_locs = collect_tensor_locations(discover_model_files(base.path))
    other_locs = collect_tensor_locations(discover_model_files(other.path))
    pd = dist.pair_deltas(
        base_locs,
        other_locs,
        base_key=base.key,
        other_key=other.key,
        arch=arch,
        rel_tol=rel_tol,
        chunk_bytes=chunk_bytes,
        per_head=per_head,
    )
    payload = _pair_delta_to_json(pd)
    _cache_write(cache_path, cache_key, payload)
    return payload, False


# --------------------------------------------------------------------------- #
# artifact builders
# --------------------------------------------------------------------------- #

TENSOR_HASH_COLUMNS: tuple[str, ...] = (
    "model_key",
    "file",
    "tensor",
    "layer",
    "family",
    "param",
    "dtype",
    "shape",
    "n_params",
    "nbytes",
    "sha256",
)


def build_tensor_hashes_csv(audits: Mapping[str, CheckpointAudit], panel: Sequence[str]) -> str:
    rows: list[dict[str, Any]] = []
    for key in panel:
        audit = audits[key]
        for t in audit.tensors:
            tk = dist.parse_tensor_name(t.name)
            rows.append(
                {
                    "model_key": key,
                    "file": t.file,
                    "tensor": t.name,
                    "layer": tk.layer,
                    "family": tk.family,
                    "param": tk.param,
                    "dtype": t.st_dtype,
                    "shape": "x".join(str(d) for d in t.shape),
                    "n_params": t.n_params,
                    "nbytes": t.nbytes,
                    "sha256": t.sha256,
                }
            )
    return _rows_to_csv(TENSOR_HASH_COLUMNS, rows)


PAIRWISE_COLUMNS: tuple[str, ...] = (
    "model_a",
    "model_b",
    "scope",
    "pair_class",
    "delta_comparable",
    "weights_identical",
    "configs_identical",
    "schema_identical",
    "arch_dims_identical",
    "dtype_differs",
    "dtype_a",
    "dtype_b",
    "architectures_differ",
    "tokenizer_identical",
    "tokenizer_semantically_identical",
    "tokenizer_vocab_identical",
    "tokenizer_diff",
    "n_shared_tensors",
    "n_identical_tensors",
    "n_only_in_a",
    "n_only_in_b",
    "tied_lm_head_convention",
    *_METRIC_COLUMNS,
    "rel_tol",
    "n_skipped_tensors",
    "n_config_diffs",
    "config_diff",
    "behavioural_config_diff",
    "only_in_a",
    "only_in_b",
    "reason",
)


def build_pairwise_csv(
    classifications: Sequence[PairClassification],
    deltas: Mapping[tuple[str, str], dict[str, Any]],
    audits: Mapping[str, CheckpointAudit],
) -> str:
    rows: list[dict[str, Any]] = []
    for pc in classifications:
        payload = deltas.get((pc.model_a, pc.model_b))
        metrics = (payload or {}).get("overall", {})
        rows.append(
            {
                **pc.to_json(),
                "scope": PAIRWISE_SCOPE,
                "dtype_a": ",".join(audits[pc.model_a].tensor_dtypes),
                "dtype_b": ",".join(audits[pc.model_b].tensor_dtypes),
                "n_only_in_a": len(pc.only_in_a),
                "n_only_in_b": len(pc.only_in_b),
                "only_in_a": ";".join(pc.only_in_a),
                "only_in_b": ";".join(pc.only_in_b),
                "tokenizer_diff": ";".join(pc.tokenizer_diff),
                "n_config_diffs": len(pc.config_diff),
                "config_diff": ";".join(pc.config_diff),
                "behavioural_config_diff": ";".join(pc.behavioural_config_diff),
                "rel_tol": (payload or {}).get("rel_tol"),
                "n_skipped_tensors": len((payload or {}).get("skipped", [])),
                **{c: metrics.get(c) for c in _METRIC_COLUMNS},
            }
        )
    return _rows_to_csv(PAIRWISE_COLUMNS, rows)


BY_LAYER_COLUMNS: tuple[str, ...] = (
    "row_kind",
    "model_key",
    "base_key",
    "layer",
    "family",
    "param",
    "tensor",
    "n_tensors",
    "shape",
    "base_dtype",
    "other_dtype",
    *_METRIC_COLUMNS,
)


def build_by_layer_csv(
    base_key: str, ordered: Sequence[str], deltas: Mapping[tuple[str, str], dict[str, Any]]
) -> str:
    rows: list[dict[str, Any]] = []
    for other_key in ordered:
        payload = deltas.get((base_key, other_key))
        if payload is None:
            continue
        counts: dict[tuple[int, str], int] = {}
        for t in payload["per_tensor"]:
            counts[(t["layer"], t["family"])] = counts.get((t["layer"], t["family"]), 0) + 1
            rows.append(
                {
                    "row_kind": "tensor",
                    "model_key": other_key,
                    "base_key": base_key,
                    "layer": t["layer"],
                    "family": t["family"],
                    "param": t["param"],
                    "tensor": t["name"],
                    "n_tensors": 1,
                    "shape": "x".join(str(d) for d in t["shape"]),
                    "base_dtype": t["base_dtype"],
                    "other_dtype": t["other_dtype"],
                    **{c: t["metrics"].get(c) for c in _METRIC_COLUMNS},
                }
            )
        for group in payload["by_layer_family"]:
            rows.append(
                {
                    "row_kind": "layer_family",
                    "model_key": other_key,
                    "base_key": base_key,
                    "layer": group["layer"],
                    "family": group["family"],
                    "param": "",
                    "tensor": "",
                    "n_tensors": counts.get((group["layer"], group["family"]), 0),
                    "shape": "",
                    "base_dtype": "",
                    "other_dtype": "",
                    **{c: group["metrics"].get(c) for c in _METRIC_COLUMNS},
                }
            )
    return _rows_to_csv(BY_LAYER_COLUMNS, rows)


BY_HEAD_COLUMNS: tuple[str, ...] = (
    "model_key",
    "base_key",
    "layer",
    "head_kind",
    "head_space",
    "head_index",
    "head_axis",
    "head_dim",
    "n_heads",
    "axis_evidence",
    "tensor",
    *_METRIC_COLUMNS,
)


def build_by_head_csv(
    base_key: str, ordered: Sequence[str], deltas: Mapping[tuple[str, str], dict[str, Any]]
) -> str:
    rows: list[dict[str, Any]] = []
    for other_key in ordered:
        payload = deltas.get((base_key, other_key))
        if payload is None:
            continue
        for t in payload["per_tensor"]:
            layout = t.get("layout")
            if not layout or not t.get("heads"):
                continue
            for head_index, metrics in sorted(t["heads"].items(), key=lambda kv: int(kv[0])):
                rows.append(
                    {
                        "model_key": other_key,
                        "base_key": base_key,
                        "layer": t["layer"],
                        "head_kind": t["family"],
                        "head_space": layout["space"],
                        "head_index": int(head_index),
                        "head_axis": "rows" if layout["axis"] == 0 else "cols",
                        "head_dim": layout["head_dim"],
                        "n_heads": layout["n_heads"],
                        "axis_evidence": layout["evidence"],
                        "tensor": t["name"],
                        **{c: metrics.get(c) for c in _METRIC_COLUMNS},
                    }
                )
    return _rows_to_csv(BY_HEAD_COLUMNS, rows)


def build_manifest(
    audits: Mapping[str, CheckpointAudit],
    panel: Sequence[str],
    base_key: str,
    classifications: Sequence[PairClassification],
    deltas: Mapping[tuple[str, str], dict[str, Any]],
    config: Mapping[str, Any],
) -> str:
    pairs = []
    for pc in classifications:
        payload = deltas.get((pc.model_a, pc.model_b))
        pairs.append(
            {
                **pc.to_json(),
                "distance": (payload or {}).get("overall"),
                "skipped_tensors": (payload or {}).get("skipped", []),
            }
        )
    orientation = {
        other: (deltas[(base_key, other)] or {}).get("orientation")
        for other in panel
        if (base_key, other) in deltas
    }
    body = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "base_key": base_key,
        "panel": list(panel),
        "lab_root": str(paths.LAB_ROOT),
        "repo_root": str(paths.REPO_ROOT),
        "config": dict(config),
        "dtype_policy": _dtype_policy(panel),
        "head_decomposition": {
            "families": list(dist.HEAD_KINDS),
            "orientation_by_pair": orientation,
            "note": (
                "q_proj/k_proj/v_proj split their row (output) axis; o_proj splits its "
                "column (input) axis. Measured from shapes, not assumed."
            ),
        },
        "families": {
            "aggregation": list(dist.FAMILIES),
            "extra": list(dist.EXTRA_FAMILIES),
        },
        "models": {k: audits[k].to_json() for k in panel},
        "pairs": pairs,
        "warnings": {k: audits[k].warnings for k in panel if audits[k].warnings},
    }
    return json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run_audit(
    *,
    models: Mapping[str, ModelSpec] | None = None,
    panel: Sequence[str] | None = None,
    base_key: str | None = None,
    out_root: Path | None = None,
    manifests_dir: Path | None = None,
    results_dir: Path | None = None,
    rel_tol: float = dist.DEFAULT_REL_TOL,
    chunk_bytes: int = dist.DEFAULT_CHUNK_BYTES,
    compute_distances: bool = True,
    per_head: bool = True,
    force: bool = False,
    probe_transformers: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    """Audit the panel and write the artifacts. Returns a summary dict."""
    models = dict(models or paths.MODELS)
    panel = list(panel or [k for k in paths.PANEL if k in models])
    base_key = base_key or (paths.BASE_KEY if paths.BASE_KEY in panel else panel[0])
    unknown = [k for k in panel if k not in models]
    if unknown:
        raise KeyError(f"panel refers to unknown model keys: {unknown}")
    if base_key not in panel:
        raise KeyError(f"base_key {base_key!r} is not in the panel {panel}")

    if out_root is not None:
        manifests_dir = manifests_dir or Path(out_root) / "manifests"
        results_dir = results_dir or Path(out_root) / "results"
    manifests_dir = Path(manifests_dir or paths.MANIFESTS)
    results_dir = Path(results_dir or paths.RESULTS)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = manifests_dir / CACHE_DIRNAME
    _prepare_cache_dir(cache_dir)

    def say(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    config = {
        "base_key": base_key,
        "panel": list(panel),
        "rel_tol": rel_tol,
        "chunk_bytes": chunk_bytes,
        "compute_distances": compute_distances,
        "per_head": per_head,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
    }

    # -- per-model audits ---------------------------------------------------- #
    audits: dict[str, CheckpointAudit] = {}
    for i, key in enumerate(panel, start=1):
        t0 = time.perf_counter()
        audit, cached = load_or_audit(
            key,
            spec=models[key],
            cache_dir=cache_dir,
            force=force,
            probe_transformers=probe_transformers,
        )
        audits[key] = audit
        say(
            f"[audit {i}/{len(panel)}] {key:<9} "
            f"files={len(audit.files):<3} "
            f"size={audit.total_bytes / 2**30:6.2f}GiB "
            f"tensors={audit.tensor_count:<4} "
            f"params={audit.param_count_unique:>12,} "
            f"nonemb={audit.param_count_non_embedding:>12,} "
            f"dtype={','.join(audit.tensor_dtypes):<5} "
            f"cfg={audit.config_normalized_sha256[:8]} "
            f"{'cached' if cached else 'read  '} "
            f"{time.perf_counter() - t0:6.1f}s"
        )
        for w in audit.warnings:
            say(f"            ! {key}: {w}")

    # -- pairwise ------------------------------------------------------------ #
    pair_keys = pair_order(panel, base_key)
    classifications = [classify_pair(audits[a], audits[b]) for a, b in pair_keys]
    by_pair = {(c.model_a, c.model_b): c for c in classifications}

    deltas: dict[tuple[str, str], dict[str, Any]] = {}
    if compute_distances:
        base_arch: Arch | None
        try:
            base_arch = audits[base_key].arch()
        except ValueError:
            base_arch = None
        for i, (a, b) in enumerate(pair_keys, start=1):
            pc = by_pair[(a, b)]
            if not pc.delta_comparable:
                say(f"[delta {i}/{len(pair_keys)}] {a} vs {b}: {pc.pair_class}, skipped")
                continue
            want_heads = per_head and a == base_key
            t0 = time.perf_counter()
            payload, cached = load_or_compute_pair(
                audits[a],
                audits[b],
                cache_dir=cache_dir,
                arch=base_arch if want_heads else None,
                per_head=want_heads,
                rel_tol=rel_tol,
                chunk_bytes=chunk_bytes,
                force=force,
            )
            deltas[(a, b)] = payload
            m = payload["overall"]
            say(
                f"[delta {i}/{len(pair_keys)}] {a:<9} vs {b:<9} "
                f"{pc.pair_class:<28} "
                f"relF={m['rel_frobenius']:.6f} "
                f"cos={m['cosine']:.6f} "
                f"changed={m['frac_changed']:.4f} "
                f"heads={'y' if want_heads else 'n'} "
                f"{'cached' if cached else 'read  '} "
                f"{time.perf_counter() - t0:6.1f}s"
            )

    # ``pair_order`` guarantees the base sits first in every pair it appears in, so
    # the base-relative metrics in these payloads are already the right way round.
    base_anchored = {k: v for k, v in deltas.items() if k[0] == base_key}
    others = [k for k in panel if k != base_key]

    # -- write ---------------------------------------------------------------- #
    artifacts: dict[Path, str] = {
        manifests_dir / MANIFEST_NAME: build_manifest(
            audits, panel, base_key, classifications, deltas, config
        ),
        manifests_dir / TENSOR_HASHES_NAME: build_tensor_hashes_csv(audits, panel),
        results_dir / PAIRWISE_NAME: build_pairwise_csv(classifications, deltas, audits),
        results_dir / BY_LAYER_NAME: build_by_layer_csv(base_key, others, base_anchored),
        results_dir / BY_HEAD_NAME: build_by_head_csv(base_key, others, base_anchored),
    }
    stamp, stamp_source = environment_stamp(config)
    written: list[str] = []
    for path, text in artifacts.items():
        changed = _write_if_changed(path, text)
        prov = _write_provenance(
            path,
            _provenance_payload(
                path.name,
                text,
                config=config,
                audits=audits,
                stamp=stamp,
                stamp_source=stamp_source,
            ),
        )
        if changed or prov:
            written.append(path.name)
        say(f"[write] {path.name:<32} {'updated' if changed else 'unchanged'}")

    return {
        "audits": audits,
        "classifications": classifications,
        "deltas": deltas,
        "artifacts": [str(p) for p in artifacts],
        "written": written,
        "manifests_dir": str(manifests_dir),
        "results_dir": str(results_dir),
        "base_key": base_key,
        "panel": panel,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rineya_psa.model_audit.run_audit",
        description="Phase 1 CPU-side structural audit of the Rineya Qwen3-0.6B panel.",
    )
    p.add_argument("--models", default=None, help="comma-separated subset of the panel")
    p.add_argument("--base", default=None, help=f"reference model (default {paths.BASE_KEY})")
    p.add_argument(
        "--out-root", default=None, type=Path, help="write under OUT/manifests and OUT/results"
    )
    p.add_argument("--rel-tol", default=dist.DEFAULT_REL_TOL, type=float)
    p.add_argument("--chunk-mb", default=dist.DEFAULT_CHUNK_BYTES // 2**20, type=int)
    p.add_argument("--no-distances", action="store_true")
    p.add_argument("--no-per-head", action="store_true")
    p.add_argument("--no-probe", action="store_true", help="skip the transformers attention probe")
    p.add_argument("--force", action="store_true", help="ignore the cache and re-read every byte")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    panel = [k.strip() for k in args.models.split(",") if k.strip()] if args.models else None
    t0 = time.perf_counter()
    summary = run_audit(
        panel=panel,
        base_key=args.base,
        out_root=args.out_root,
        rel_tol=args.rel_tol,
        chunk_bytes=args.chunk_mb * 2**20,
        compute_distances=not args.no_distances,
        per_head=not args.no_per_head,
        force=args.force,
        probe_transformers=not args.no_probe,
        quiet=args.quiet,
    )
    if not args.quiet:
        print(
            f"[done] {len(summary['panel'])} models, "
            f"{len(summary['classifications'])} pairs, "
            f"{len(summary['written'])} artifacts updated, "
            f"{time.perf_counter() - t0:.1f}s"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
