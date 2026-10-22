"""Paired exact statistics and a fail-closed PROCESS artifact reader (stdlib only).

Positive net means the candidate wins over the reference. The reported threshold
is the smallest significant absolute net at *this pair's fixed discordance*;
it is not a prospective power calculation. The independent 20-view floor applies
only to structural acquisition claims across training draws.

PROBE_WEIGHT_BINDING.json is an execution receipt, never reconstructed from an
old result. All paths may be absolute or relative to the binding file. Sidecar
map keys are relative to arm_dir. Missing execution receipts block comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping


BLOCKED = "BLOCKED_ON_COMPARABILITY"
RELATIONSHIPS = ("same_training_draw", "cross_training_draw")
SIDE_CARS = ("config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.json", "merges.txt")


class ComparabilityError(ValueError):
    """No endpoint may be reported after this exception."""


def _count(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _alpha(value: float) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("alpha must lie strictly between zero and one")
    result = Fraction(str(value))
    if not 0 < result < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return result


def exact_mcnemar_fraction(wins: int, losses: int) -> Fraction:
    """Two-sided conditional Binomial(nd, 1/2), doubled smaller tail."""
    nd = _count(wins, "wins") + _count(losses, "losses")
    if nd == 0:
        return Fraction(1)
    numerator = 2 * sum(math.comb(nd, k) for k in range(min(wins, losses) + 1))
    return min(Fraction(1), Fraction(numerator, 1 << nd))


def exact_mcnemar(wins: int, losses: int) -> float:
    return float(exact_mcnemar_fraction(wins, losses))


def minimum_detectable_net(nd: int, alpha: float = 0.05) -> int | None:
    """Minimum |net| with p <= alpha at fixed nd, respecting integer parity.

    None means no attainable split of these discordant rows is significant.
    In particular, nd=0 has p=1 and no finite threshold at that discordance.
    """
    _count(nd, "nd")
    threshold = _alpha(alpha)
    for net in range(nd % 2, nd + 1, 2):
        if exact_mcnemar_fraction((nd + net) // 2, (nd - net) // 2) <= threshold:
            return net
    return None


def paired_boolean_summary(reference: Mapping[str, bool], candidate: Mapping[str, bool],
                           *, relationship: str, alpha: float = 0.05) -> dict[str, Any]:
    """Analyze already comparable paired rows; this function does no file gating.

    Artifact consumers must use compare_acquisition(), which gates provenance
    before extracting any endpoints. This primitive also supports other paired
    boolean endpoints, whose domain-specific structural floors belong outside it.
    """
    if relationship not in RELATIONSHIPS:
        raise ValueError(f"relationship must be one of {RELATIONSHIPS}")
    threshold = _alpha(alpha)
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("paired maps must have identical nonempty row IDs")
    if any(not isinstance(k, str) or not k for k in reference):
        raise ValueError("paired row IDs must be nonempty strings")
    if any(type(v) is not bool for rows in (reference, candidate) for v in rows.values()):
        raise ValueError("paired row values must be booleans, never truth-coerced")
    wins = sum(candidate[k] and not reference[k] for k in reference)
    losses = sum(reference[k] and not candidate[k] for k in reference)
    p = exact_mcnemar_fraction(wins, losses)
    nd = wins + losses
    return {
        "n": len(reference), "reference_correct": sum(reference.values()),
        "candidate_correct": sum(candidate.values()), "wins": wins, "losses": losses,
        "net": wins - losses, "nd": nd, "exact_p": float(p),
        "alpha": float(threshold), "nominally_significant": p <= threshold,
        "minimum_significant_abs_net_at_observed_nd": minimum_detectable_net(nd, alpha),
        "threshold_structure": "fixed observed within-pair discordance; not prospective power",
        "relationship": relationship,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ComparabilityError(reason)


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(result, dict), f"JSON object required: {path.name}")
    return result


def _sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), f"invalid or missing SHA256: {label}")
    return value


def _path(value: Any, base: Path, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"missing path: {label}")
    result = Path(value)
    return (result if result.is_absolute() else base / result).resolve()


def _bound_file(binding: dict[str, Any], base: Path, stem: str) -> Path:
    path = _path(binding.get(f"{stem}_path"), base, stem)
    expected = _sha(binding.get(f"{stem}_sha256"), stem)
    _require(path.is_file(), f"missing bound file: {stem}")
    _require(sha256_file(path) == expected, f"bound file hash mismatch: {stem}")
    return path


def _load_bound_probe(raw_path: Path, binding_path: Path, kind: str) -> tuple[dict, dict, set[str], dict]:
    """Validate evidence identities and complete row coverage without reporting endpoints."""
    _require(kind in ("process", "retention", "long_context"), "unsupported probe kind")
    binding = _read_json(binding_path)
    base = binding_path.parent
    _require(binding.get("schema") == 1, "unsupported binding schema")
    _require(binding.get("kind") == kind, "binding kind differs from requested probe")
    _require(binding.get("status") == "COMPLETE" and binding.get("complete") is True,
             "probe execution is not completed successfully")
    _require(type(binding.get("returncode")) is int and binding["returncode"] == 0,
             "probe return code is missing or nonzero")
    _require(_bound_file(binding, base, "raw_result") == raw_path.resolve(),
             "binding points to a different raw probe artifact")
    raw = _read_json(raw_path)
    if kind == "process":
        _require(raw.get("record") == "PROCESS_TRANSITION_PROBE", "raw artifact is not PROCESS")
        _require(raw.get("checkpoint") is None, "process probe unexpectedly applied a checkpoint")
    else:
        expected_kind = {"retention": "exploratory_typed_retention_probe_only",
                         "long_context": "exploratory_long_context_retention_probe_only"}[kind]
        _require(raw.get("evidence_kind") == expected_kind, "raw evidence kind differs from requested probe")
        if kind == "long_context":
            _require(raw.get("checkpoint_applied") is None, "long-context probe unexpectedly applied a checkpoint")
    arm_dir = _path(binding.get("arm_dir"), base, "arm_dir")
    model_field = {"process": "model_dir", "retention": "target", "long_context": "model_directory"}[kind]
    model_dir = raw.get(model_field)
    if kind == "retention" and isinstance(model_dir, str) and model_dir.startswith("hf:"):
        model_dir = model_dir[3:]
    _require(_path(model_dir, base, f"raw.{model_field}") == arm_dir,
             "raw model directory differs from the bound arm")
    arm_path = _bound_file(binding, base, "arm_receipt")
    materialization_path = _bound_file(binding, base, "materialization")
    _require(arm_path == arm_dir / "ARM.json", "arm receipt must be arm_dir/ARM.json")
    _require(materialization_path == arm_dir / "MATERIALIZATION.json",
             "materialization receipt must be arm_dir/MATERIALIZATION.json")
    arm = _read_json(arm_path)
    materialization = _read_json(materialization_path)
    provenance = {}
    for key in ("source_weight_sha256", "checkpoint_manifest_sha256"):
        provenance[key] = _sha(arm.get(key), f"ARM {key}")
        _require(provenance[key] == _sha(materialization.get(key), f"MATERIALIZATION {key}"),
                 f"arm receipts disagree on training provenance: {key}")
    weight_sha = _sha(binding.get("weight_sha256"), "weight_sha256")
    for value, label in ((binding.get("weight_sha256_before"), "weight before"),
                         (binding.get("weight_sha256_after"), "weight after"),
                         (arm.get("arm_weight_sha256"), "ARM weight"),
                         (materialization.get("out_weight_sha256"), "MATERIALIZATION weight")):
        _require(_sha(value, label) == weight_sha, f"weight identity mismatch: {label}")
    if "weight_sha256" in raw:
        _require(raw["weight_sha256"] == weight_sha, "raw weight identity mismatch")
    weight_path = _path(binding.get("weight_path"), base, "weight_path")
    _require(weight_path == arm_dir / "model.safetensors", "weight must be arm_dir/model.safetensors")
    if weight_path.exists():
        _require(weight_path.is_file() and sha256_file(weight_path) == weight_sha,
                 "surviving weight file hash mismatch")

    before = binding.get("sidecar_sha256_before")
    after = binding.get("sidecar_sha256_after")
    _require(isinstance(before, dict) and isinstance(after, dict) and before == after,
             "sidecar hashes missing or changed during probing")
    _require(set(SIDE_CARS).issubset(before), "required model sidecar snapshots missing")
    for name, digest in before.items():
        sidecar = (arm_dir / name).resolve()
        _require(isinstance(name, str) and sidecar.is_relative_to(arm_dir),
                 "sidecar path escapes arm directory")
        _require(sidecar.is_file() and sha256_file(sidecar) == _sha(digest, name),
                 f"sidecar hash mismatch: {name}")

    _bound_file(binding, base, "script")
    _bound_file(binding, base, "wrapper")
    supervisor = _read_json(_bound_file(binding, base, "supervisor"))
    terminal = _read_json(_bound_file(binding, base, "terminal"))
    _require(supervisor.get("status") == "EXITED" and
             type(supervisor.get("returncode")) is int and supervisor["returncode"] == 0,
             "supervisor did not establish a successful exit")
    _require(terminal.get("status") == "COMPLETED" and terminal.get("complete") is True and
             type(terminal.get("system_exit_code")) is int and terminal["system_exit_code"] == 0 and
             "failure" in terminal and terminal["failure"] is None and
             terminal.get("script_unchanged") is True and
             terminal.get("script_sha256_after") == binding["script_sha256"] and
             "script_hash_error" in terminal and terminal["script_hash_error"] is None,
             "probe terminal is incomplete, failed, or script identity changed")
    original_path = raw_path.resolve()
    if "original_raw_result_path" in binding or "original_raw_result_sha256" in binding:
        original_path = _bound_file(binding, base, "original_raw_result")
        _require(binding["original_raw_result_sha256"] == binding["raw_result_sha256"],
                 "copied raw result differs from original raw bytes")
    _require(_path(terminal.get("result"), base, "terminal.result") == original_path,
             "probe terminal points to a different raw artifact")
    views_path = _bound_file(binding, base, "views")
    views_sha = binding["views_sha256"]
    if kind != "long_context":
        views_field = "probe_sha256" if kind == "retention" else "views_sha256"
        _require(_sha(raw.get(views_field), f"raw {views_field}") == views_sha,
                 "raw views hash differs from actual views bytes")
    implementation_field = "reference_digest" if kind == "retention" else "implementation_digest"
    implementation = _sha(raw.get(implementation_field), f"raw {implementation_field}")
    _require(implementation == _sha(binding.get("implementation_digest"), "binding implementation_digest"),
             "raw implementation digest differs from binding")
    view_ids: set[str] = set()
    with views_path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            view = json.loads(line)
            row_id = view.get("id")
            _require(isinstance(row_id, str) and bool(row_id) and row_id not in view_ids,
                     "views have missing or duplicate row IDs")
            view_ids.add(row_id)
    _require(bool(view_ids), "views bank is empty")
    count_field = {"process": "views", "retention": "probe_rows", "long_context": "rows_total"}[kind]
    _require(type(raw.get(count_field)) is int and raw[count_field] == len(view_ids),
             "raw views count differs from the bound bank")
    inputs_before = binding.get("additional_input_sha256_before")
    inputs_after = binding.get("additional_input_sha256_after")
    _require(isinstance(inputs_before, dict) and isinstance(inputs_after, dict) and
             inputs_before == inputs_after, "additional inputs missing or changed during probing")
    minimum_inputs = {"process": 0, "retention": 2, "long_context": 1}[kind]
    _require(len(inputs_before) >= minimum_inputs, "required additional probe inputs missing")
    additional_paths = []
    for name, digest in inputs_before.items():
        path = _path(name, base, "additional input")
        _require(path.is_file() and sha256_file(path) == _sha(digest, "additional input"),
                 "additional probe input hash mismatch")
        additional_paths.append(path)
    if kind == "process":
        _rows(raw, view_ids)
    elif kind == "retention":
        saved = _read_json(_bound_file(binding, base, "retention_predictions"))
        _require(saved.get("scope") == "UNCHANGED_LEGACY_GENERATOR_RETURN" and
                 saved.get("scorer_modified") is False,
                 "retention predictions are not an unchanged generator capture")
        predictions = saved.get("predictions")
        _require(isinstance(predictions, list) and type(saved.get("rows")) is int and
                 saved["rows"] == len(view_ids) == len(predictions),
                 "retention generator capture is incomplete")
        ids = [row.get("id") for row in predictions]
        _require(all(isinstance(row_id, str) for row_id in ids) and len(ids) == len(set(ids)) and
                 set(ids) == view_ids, "retention prediction IDs differ from the complete panel")
    elif kind == "long_context":
        _require(len(additional_paths) == 1, "long-context requires exactly one bound answer key")
        key_ids = []
        with additional_paths[0].open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    key_ids.append(json.loads(line).get("id"))
        _require(all(isinstance(row_id, str) for row_id in key_ids) and
                 len(key_ids) == len(set(key_ids)) and set(key_ids) == view_ids,
                 "long-context answer key IDs differ from the complete views bank")
        _require(isinstance(raw.get("predictions"), dict) and
                 isinstance(raw.get("per_row_correct"), dict) and
                 set(raw["predictions"]) == set(raw["per_row_correct"]) == view_ids,
                 "long-context prediction/score IDs differ from the complete views bank")
    return raw, binding, view_ids, provenance


def validate_probe_binding(raw_path: str | Path, binding_path: str | Path,
                           kind: str) -> tuple[dict, dict, set[str]]:
    """Validate one probe's custody; return raw, enriched binding, and full row IDs.

    Raises ComparabilityError without reporting endpoints. The returned binding
    adds training_identity (source/checkpoint hashes agreed by both arm receipts).
    Multi-arm callers must still compare views, implementation, script, wrapper,
    generation settings, additional inputs, and training identity across arms
    before reporting any endpoint. Endpoint computation is left to the caller.
    """
    try:
        raw, binding, ids, provenance = _load_bound_probe(
            Path(raw_path).resolve(), Path(binding_path).resolve(), kind)
        return raw, {**binding, "training_identity": provenance}, ids
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ComparabilityError(str(exc)) from exc


def _load_bound_process(raw_path: Path, binding_path: Path) -> tuple[dict, dict, set[str], dict]:
    return _load_bound_probe(raw_path, binding_path, "process")


def _rows(raw: dict, expected_ids: set[str]) -> dict[str, bool]:
    predictions = raw.get("predictions")
    _require(isinstance(predictions, list), "raw predictions list missing")
    result: dict[str, bool] = {}
    for row in predictions:
        _require(isinstance(row, dict), "prediction is not an object")
        row_id = row.get("id")
        _require(isinstance(row_id, str) and bool(row_id) and row_id not in result,
                 "missing or duplicate prediction ID")
        _require(type(row.get("canonical")) is bool, "canonical correctness is not boolean")
        result[row_id] = row["canonical"]
    _require(result.keys() == expected_ids, "prediction IDs differ from the complete views bank")
    return result


def compare_acquisition(reference_raw: str | Path, candidate_raw: str | Path, *,
                        relationship: str, reference_binding: str | Path | None = None,
                        candidate_binding: str | Path | None = None,
                        alpha: float = 0.05) -> dict[str, Any]:
    """Return endpoints only after every provenance and paired-row gate passes.

    Historical artifacts without contemporaneous weight bindings cannot be made
    comparable by synthesizing a binding now. On any malformed/missing evidence,
    return only BLOCKED_ON_COMPARABILITY and reasons, without endpoint numbers.
    """
    try:
        _require(relationship in RELATIONSHIPS, "explicit valid training relationship required")
        paths = (Path(reference_raw).resolve(), Path(candidate_raw).resolve())
        bindings = (Path(reference_binding).resolve() if reference_binding else
                    paths[0].parent / "PROBE_WEIGHT_BINDING.json",
                    Path(candidate_binding).resolve() if candidate_binding else
                    paths[1].parent / "PROBE_WEIGHT_BINDING.json")
        left, lb, li, lp = _load_bound_process(paths[0], bindings[0])
        right, rb, ri, rp = _load_bound_process(paths[1], bindings[1])
        for key in ("views_sha256", "implementation_digest", "instrument", "max_new_tokens",
                    "numerics", "runtime"):
            _require(key in left and key in right and left[key] == right[key],
                     f"arms differ on required comparability field: {key}")
        _require(li == ri, "bound banks have different row IDs")
        for key in ("script_sha256", "wrapper_sha256"):
            _require(lb[key] == rb[key], f"arms used different probe code: {key}")
        if relationship == "same_training_draw":
            for key in ("source_weight_sha256", "checkpoint_manifest_sha256"):
                _require(lp[key] == rp[key], f"same_training_draw provenance differs: {key}")
        # Extraction occurs only after both complete provenance gates succeed.
        reference = _rows(left, li)
        candidate = _rows(right, ri)
        summary = paired_boolean_summary(reference, candidate, relationship=relationship, alpha=alpha)
        cross_draw = relationship == "cross_training_draw"
        summary["cross_training_draw_abs_net_floor"] = 20 if cross_draw else None
        summary["cross_training_draw_floor_cleared"] = abs(summary["net"]) >= 20 if cross_draw else None
        summary["nominal_structural_acquisition_claim_eligible"] = (
            summary["nominally_significant"] and (not cross_draw or abs(summary["net"]) >= 20))
        return {"status": "COMPARABLE", "endpoint": "acquisition_canonical", "report_only": True,
                "quality_claim": False, "release_permitted": False,
                "selection_permitted": False, "paired": summary,
                "views_sha256": lb["views_sha256"],
                "implementation_digest": lb["implementation_digest"],
                "reference_raw_sha256": lb["raw_result_sha256"],
                "candidate_raw_sha256": rb["raw_result_sha256"],
                "reference_weight_sha256": lb["weight_sha256"],
                "candidate_weight_sha256": rb["weight_sha256"],
                "reference_binding_sha256": sha256_file(bindings[0]),
                "candidate_binding_sha256": sha256_file(bindings[1])}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {"status": BLOCKED, "reasons": [str(exc)], "report_only": True,
                "quality_claim": False, "release_permitted": False, "selection_permitted": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--candidate-raw", type=Path, required=True)
    parser.add_argument("--reference-binding", type=Path)
    parser.add_argument("--candidate-binding", type=Path)
    parser.add_argument("--relationship", choices=RELATIONSHIPS, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    result = compare_acquisition(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "COMPARABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
