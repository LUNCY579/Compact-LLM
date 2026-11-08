# ruff: noqa: TRY301 -- the Git proof is intentionally normalized at one boundary.
"""Fail-closed Phase-C qualification gate and Phase-D authentication.

The gate deliberately treats every JSON boolean as an assertion to verify, not as
evidence.  It follows the committed preregistration root through the supplied
manifests, reloads every persisted native plan, recomputes the held-out loss
summary, and only then evaluates the prespecified paired statistics.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from .hashing import canonical_json_hash, sha256_bytes, sha256_file
from .integration import to_native_update_plan
from .manifest import atomic_write_json, utc_now
from .persistence import load_plan
from .statistics import paired_comparison
from .topk import budget_to_k

CANONICAL_SEEDS: Final = (11, 17, 23, 29, 31)
CANONICAL_BUDGETS: Final = (0.0001, 0.0005, 0.001, 0.005, 0.01)
CANONICAL_ARMS: Final = (
    "no_update_frozen",
    "random_sparse_native",
    "gradient_topk_sparse_native",
    "target_only_sparse_native",
    "coordinate_shuffled_peer_pcsm",
    "layer_shuffled_peer_pcsm",
    "real_peer_pcsm_bootstrap",
    "real_peer_pcsm_calibrated",
)
PLANNED_ARMS: Final = CANONICAL_ARMS[1:]
PEER_CAPACITY_ARMS: Final = (
    "coordinate_shuffled_peer_pcsm",
    "layer_shuffled_peer_pcsm",
    "real_peer_pcsm_bootstrap",
    "real_peer_pcsm_calibrated",
)
EXPECTED_RUN_CELL_COUNT: Final = 200
EXPECTED_PLAN_CELL_COUNT: Final = 175
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class GateInputs:
    """All evidence required by the Phase-C gate."""

    run: Path
    plans: Path
    preregistration: Path
    budget: Path
    sample: Path
    split: Path
    gradient_summary: Path
    gradient_cache: Path


@dataclass(frozen=True, slots=True)
class LoadedPlanEvidence:
    """Small immutable projection of a fully loaded persisted plan."""

    base_fingerprint: str
    target_name: str
    shape: tuple[int, ...]
    k: int
    support_fraction: float
    plan_hash: str
    support_hash: str
    metadata: Mapping[str, object]
    support_stats: Mapping[str, object]


PlanEvidenceLoader = Callable[[Path, str], LoadedPlanEvidence]
CommitVerifier = Callable[[Path], Mapping[str, object]]


class _Reasons:
    """Deduplicated bounded reason collector for usable STOP artifacts."""

    def __init__(self, limit: int = 300) -> None:
        self._values: list[str] = []
        self._seen: set[str] = set()
        self._limit = limit
        self._omitted = 0

    def add(self, value: str) -> None:
        if value in self._seen:
            return
        self._seen.add(value)
        if len(self._values) < self._limit:
            self._values.append(value)
        else:
            self._omitted += 1

    def values(self) -> list[str]:
        result = list(self._values)
        if self._omitted:
            result.append(f"{self._omitted} additional audit failures omitted")
        return result

    def __bool__(self) -> bool:
        return bool(self._values or self._omitted)


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable canonical JSON: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _self_hash_valid(value: Mapping[str, Any], key: str) -> bool:
    declared = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    return (
        isinstance(declared, str)
        and _SHA256.fullmatch(declared) is not None
        and declared == canonical_json_hash(payload)
    )


def gate_self_hash_valid(value: Mapping[str, Any]) -> bool:
    """Return whether a gate artifact has a valid canonical self-seal."""

    return _self_hash_valid(value, "gate_sha256")


def _file_record(path: Path) -> dict[str, object]:
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def _default_plan_loader(path: Path, base_fingerprint: str) -> LoadedPlanEvidence:
    plan = load_plan(path, expected_base_fingerprint=base_fingerprint)
    if len(plan.entries) != 1:
        raise ValueError("qualification plans must contain exactly one target entry")
    entry = plan.entries[0]
    native = to_native_update_plan(plan)
    return LoadedPlanEvidence(
        base_fingerprint=plan.base_fingerprint,
        target_name=entry.name,
        shape=tuple(entry.shape),
        k=entry.k,
        support_fraction=float(cast(float, entry.support_fraction)),
        plan_hash=native.plan_hash,
        support_hash=native.support_hash,
        metadata=dict(plan.metadata),
        support_stats=plan.support_stats(),
    )


def _committed_file_evidence(path: Path) -> Mapping[str, object]:
    """Prove that *path* is byte-identical to the version committed at HEAD."""

    def run_text(directory: Path, *arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        return str(result.stdout)

    def run_bytes(directory: Path, *arguments: str) -> bytes:
        result = subprocess.run(
            ["git", *arguments],
            cwd=directory,
            check=True,
            capture_output=True,
            text=False,
        )
        return bytes(result.stdout)

    try:
        root_text = run_text(path.parent, "rev-parse", "--show-toplevel")
        root = Path(root_text.strip()).resolve()
        relative = path.resolve().relative_to(root).as_posix()
        run_text(root, "ls-files", "--error-unmatch", "--", relative)
        commit_text = run_text(root, "log", "-1", "--format=%H", "--", relative)
        commit = commit_text.strip()
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("preregistration has no full Git commit identity")
        run_text(root, "merge-base", "--is-ancestor", commit, "HEAD")
        committed = run_bytes(root, "show", f"{commit}:{relative}")
        if committed != path.read_bytes():
            raise ValueError("working file differs from its last committed bytes")
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        message = f"preregistration is not committed at HEAD: {error}"
        raise ValueError(message) from error
    return {
        "verified": True,
        "commit": commit,
        "repository_relative_path": relative,
        "blob_sha256": sha256_bytes(committed),
    }


def _canonical_cells(arms: Sequence[str]) -> list[dict[str, object]]:
    return [
        {"seed": seed, "budget_fraction": fraction, "arm": arm}
        for seed in CANONICAL_SEEDS
        for fraction in CANONICAL_BUDGETS
        for arm in arms
    ]


def _cell_key(record: Mapping[str, Any]) -> tuple[int, float, str]:
    seed = record.get("seed")
    fraction = record.get("budget_fraction")
    arm = record.get("arm")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError("budget_fraction must be numeric")
    if not isinstance(arm, str):
        raise TypeError("arm must be a string")
    return seed, float(fraction), arm


def _index_records(
    value: object,
    *,
    expected: set[tuple[int, float, str]],
    label: str,
    expected_count: int,
    reasons: _Reasons,
) -> dict[tuple[int, float, str], dict[str, Any]]:
    if not isinstance(value, list):
        reasons.add(f"{label} records are not a list")
        return {}
    if len(value) != expected_count:
        reasons.add(
            f"{label} has {len(value)} records; exact count is {expected_count}"
        )
    grouped: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            reasons.add(f"{label} record {index} is not an object")
            continue
        try:
            grouped[_cell_key(raw)].append(raw)
        except (TypeError, ValueError) as error:
            reasons.add(f"{label} record {index} has an invalid cell key: {error}")
    duplicate = sorted(key for key, records in grouped.items() if len(records) != 1)
    if duplicate:
        reasons.add(f"{label} has {len(duplicate)} duplicate cells")
    missing = expected.difference(grouped)
    extra = set(grouped).difference(expected)
    if missing:
        reasons.add(f"{label} is missing {len(missing)} canonical cells")
    if extra:
        reasons.add(f"{label} has {len(extra)} noncanonical cells")
    return {
        key: records[0]
        for key, records in grouped.items()
        if key in expected and len(records) == 1
    }


def _is_zero(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == 0.0
    )


def _same_float(first: object, second: float, *, tolerance: float = 1e-12) -> bool:
    return (
        not isinstance(first, bool)
        and isinstance(first, (int, float))
        and math.isfinite(float(first))
        and math.isclose(float(first), second, rel_tol=tolerance, abs_tol=tolerance)
    )


def _expected_data_signature(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    ordered = [
        {
            "row_id": str(row["row_id"]),
            "content_sha256": str(row["content_sha256"]),
            "token_count": int(row["token_count"]),
            "assistant_token_count": int(row["assistant_token_count"]),
        }
        for row in rows
    ]
    return {"examples": ordered, "signature_sha256": canonical_json_hash(ordered)}


def _validate_access_log(
    artifact: Mapping[str, Any], label: str, reasons: _Reasons
) -> None:
    log = artifact.get("access_log")
    if not isinstance(log, list) or not log:
        reasons.add(f"{label} has no auditable access log")
        return
    if artifact.get("access_log_sha256") != canonical_json_hash(log):
        reasons.add(f"{label} access-log hash is invalid")
    for index, entry in enumerate(log):
        if not isinstance(entry, dict):
            reasons.add(f"{label} access-log entry {index} is malformed")
            continue
        if entry.get("sealed") is not False:
            reasons.add(f"{label} access-log entry {index} is not sealed=false")
        if str(entry.get("split", "")).lower() not in {"train", "validation"}:
            reasons.add(f"{label} access log includes a forbidden split")


def _validate_exact_access_logs(
    plans: Mapping[str, Any],
    run: Mapping[str, Any],
    inputs: GateInputs,
    reasons: _Reasons,
) -> None:
    expected_planner = [
        {
            "event": "cache_access",
            "purpose": "META_TRAIN gradient cache",
            "split": "train",
            "sealed": False,
        },
        {
            "event": "cache_access",
            "purpose": "MASK_CAL gradient cache",
            "split": "train",
            "sealed": False,
        },
    ]
    if plans.get("access_log") != expected_planner:
        reasons.add(
            "plan-result access log is not the exact preregistered event sequence"
        )
    run_log = run.get("access_log")
    if not isinstance(run_log, list) or len(run_log) != 2:
        reasons.add("run access log is not the exact two-event sequence")
        return
    validation, artifacts = run_log
    if (
        not isinstance(validation, dict)
        or validation.get("event") != "data_access"
        or validation.get("purpose") != "ONE_STEP_DEV held-out loss only"
        or validation.get("split") != "validation"
        or validation.get("sealed") is not False
        or "validation" not in Path(str(validation.get("path", ""))).parts
    ):
        reasons.add("run validation-access event is malformed or not ONE_STEP_DEV-only")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("event") != "artifact_access"
        or artifacts.get("purpose") != "fixed plan and calibration provenance"
        or artifacts.get("split") != "train"
        or artifacts.get("sealed") is not False
        or artifacts.get("paths")
        != [
            str(inputs.plans.resolve()),
            str(inputs.gradient_summary.resolve()),
            str(inputs.gradient_cache.resolve()),
        ]
    ):
        reasons.add("run artifact-access event is malformed or incomplete")


def _validate_preregistration_access_log(
    preregistration: Mapping[str, Any], reasons: _Reasons
) -> None:
    log = preregistration.get("access_log")
    if (
        not isinstance(log, list)
        or len(log) != 1
        or preregistration.get("access_log_sha256") != canonical_json_hash(log)
    ):
        reasons.add("preregistration access log is absent or has an invalid seal")
        return
    entry = log[0]
    if (
        not isinstance(entry, dict)
        or entry.get("event") != "data_access"
        or entry.get("purpose")
        != "freeze ONE_STEP_DEV token IDs and labels before qualification"
        or entry.get("split") != "validation"
        or entry.get("sealed") is not False
        or "validation" not in Path(str(entry.get("path", ""))).parts
    ):
        reasons.add("preregistration data-access event is not the exact freeze event")


def _validate_file_binding(
    frozen: object,
    path: Path,
    *,
    self_key: str,
    live: Mapping[str, Any],
    label: str,
    reasons: _Reasons,
) -> None:
    if not isinstance(frozen, dict):
        reasons.add(f"preregistration lacks the {label} file binding")
        return
    if frozen.get("sha256") != sha256_file(path):
        reasons.add(f"preregistration {label} file hash differs")
    if frozen.get("size") != path.stat().st_size:
        reasons.add(f"preregistration {label} file size differs")
    if frozen.get("self_sha256") != live.get(self_key):
        reasons.add(f"preregistration {label} self-seal differs")


def _validate_source_tree(
    preregistration: Mapping[str, Any], reasons: _Reasons
) -> None:
    lab_root = Path(__file__).resolve().parents[2]
    source_paths = [
        *sorted((lab_root / "src" / "rineya_pcsm").glob("*.py")),
        lab_root / "scripts" / "freeze_phase_c_preregistration.py",
        lab_root / "scripts" / "build_phase_c_plans.py",
        lab_root / "scripts" / "run_phase_c.py",
        lab_root / "scripts" / "audit_phase_c_gate.py",
        lab_root / "scripts" / "train_phase_d.py",
    ]
    expected = {path.relative_to(lab_root).as_posix(): path for path in source_paths}
    frozen = preregistration.get("source_files")
    if not isinstance(frozen, dict) or set(frozen) != set(expected):
        reasons.add("preregistered source file set is not the exact gate producer tree")
        return
    source_hashes: dict[str, str] = {}
    for name, path in expected.items():
        record = frozen.get(name)
        if not isinstance(record, dict):
            reasons.add(f"preregistered source record is malformed: {name}")
            continue
        digest = sha256_file(path)
        source_hashes[name] = digest
        if (
            record.get("sha256") != digest
            or record.get("size") != path.stat().st_size
            or Path(str(record.get("path", ""))).resolve() != path.resolve()
        ):
            reasons.add(f"preregistered source file changed: {name}")
    if preregistration.get("source_tree_sha256") != canonical_json_hash(source_hashes):
        reasons.add("preregistered source-tree hash is invalid")


def _validate_layer_permutations(
    preregistration: Mapping[str, Any], reasons: _Reasons
) -> dict[int, dict[str, Any]]:
    raw = preregistration.get("layer_depth_permutations")
    result: dict[int, dict[str, Any]] = {}
    target = str(preregistration.get("target", {}).get("parameter_name", ""))
    expected_names = {
        f"model.layers.{layer}.mlp.down_proj.weight"
        for layer in range(28)
        if f"model.layers.{layer}.mlp.down_proj.weight" != target
    }
    expected_depths = {int(name.split(".")[2]) for name in expected_names}
    if not isinstance(raw, dict) or set(raw) != {str(seed) for seed in CANONICAL_SEEDS}:
        reasons.add("preregistered layer permutations do not cover exactly five seeds")
        return result
    for seed in CANONICAL_SEEDS:
        record = raw.get(str(seed))
        if not isinstance(record, dict):
            reasons.add(f"seed {seed} layer permutation is malformed")
            continue
        mapping = record.get("mapping")
        valid_mapping = isinstance(mapping, dict) and set(mapping) == expected_names
        values: list[int] = []
        if valid_mapping and isinstance(mapping, dict):
            valid_mapping = all(
                not isinstance(value, bool) and isinstance(value, int)
                for value in mapping.values()
            )
            values = [int(value) for value in mapping.values()] if valid_mapping else []
        identity = {name: int(name.split(".")[2]) for name in sorted(expected_names)}
        if (
            not valid_mapping
            or set(values) != expected_depths
            or len(values) != len(set(values))
            or mapping == identity
            or record.get("bijection") is not True
            or record.get("nonidentity") is not True
            or record.get("mapping_sha256") != canonical_json_hash(mapping)
        ):
            reasons.add(
                f"seed {seed} layer permutation is not a bound nonidentity bijection"
            )
            continue
        result[seed] = record
    return result


def _validate_evaluation(
    record: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
    expected_signature: str,
    label: str,
    reasons: _Reasons,
) -> float | None:
    if record.get("dev_data_signature") != expected_signature:
        reasons.add(f"{label} has the wrong held-out data signature")
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        reasons.add(f"{label} has no evaluation object")
        return None
    examples = evaluation.get("examples")
    if not isinstance(examples, list):
        reasons.add(f"{label} has no per-example loss evidence")
        return None
    expected = {str(row["row_id"]): row for row in expected_rows}
    observed: dict[str, dict[str, Any]] = {}
    for example in examples:
        if not isinstance(example, dict) or not isinstance(example.get("row_id"), str):
            reasons.add(f"{label} contains a malformed evaluation example")
            continue
        row_id = str(example["row_id"])
        if row_id in observed:
            reasons.add(f"{label} repeats evaluation row {row_id}")
        observed[row_id] = example
    if set(observed) != set(expected):
        reasons.add(f"{label} evaluation row IDs differ from frozen ONE_STEP_DEV")
        return None
    weighted = 0.0
    supervised_total = 0
    input_total = 0
    for row_id, frozen in expected.items():
        example = observed[row_id]
        supervised = example.get("supervised_tokens")
        tokens = example.get("tokens")
        loss = example.get("loss")
        if supervised != int(frozen["assistant_token_count"]):
            reasons.add(f"{label} supervised-token count changed for {row_id}")
            continue
        if tokens != int(frozen["token_count"]):
            reasons.add(f"{label} input-token count changed for {row_id}")
            continue
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
        ):
            reasons.add(f"{label} has invalid NLL for {row_id}")
            continue
        weighted += float(loss) * int(supervised)
        supervised_total += int(supervised)
        input_total += int(tokens)
    if supervised_total <= 0:
        return None
    recomputed = weighted / supervised_total
    if not _same_float(evaluation.get("mean_nll"), recomputed, tolerance=1e-10):
        reasons.add(f"{label} mean NLL is not the token-weighted recomputation")
    if evaluation.get("supervised_tokens") != supervised_total:
        reasons.add(f"{label} total supervised-token count is inconsistent")
    if record.get("evaluation_tokens") != input_total:
        reasons.add(f"{label} evaluated input-token count is inconsistent")
    return recomputed


def _validate_delta(
    record: Mapping[str, Any],
    *,
    k: int,
    plan_hash: str | None,
    support_hash: str | None,
    target_name: str,
    label: str,
    reasons: _Reasons,
) -> None:
    before = record.get("unplanned_parameter_hash_before")
    after = record.get("unplanned_parameter_hash_after")
    if (
        not isinstance(before, str)
        or _SHA256.fullmatch(before) is None
        or before != after
        or record.get("unplanned_parameter_hash_unchanged") is not True
    ):
        reasons.add(f"{label} does not prove bitwise-frozen unplanned parameters")
    delta = record.get("delta_audit")
    if not isinstance(delta, dict):
        reasons.add(f"{label} has no independent delta audit")
        return
    if delta.get("valid") is not True or delta.get("planned_k") != k:
        reasons.add(f"{label} delta validity/planned K is inconsistent")
    if k == 0:
        if delta.get("changed_union_size") != 0:
            reasons.add(f"{label} no-update baseline changed native coordinates")
        return
    exact = {
        "support_union_size": k,
        "outside_support_changed_count": 0,
    }
    if any(delta.get(key) != value for key, value in exact.items()):
        reasons.add(f"{label} delta support union or leakage count is inconsistent")
    if any(
        not _is_zero(delta.get(key))
        for key in (
            "outside_support_delta_l2",
            "outside_support_delta_max_abs",
        )
    ):
        reasons.add(f"{label} has nonzero outside-support delta")
    if delta.get("leakage_examples") != []:
        reasons.add(f"{label} has outside-support leakage examples")
    if delta.get("plan_hash") != plan_hash or delta.get("support_hash") != support_hash:
        reasons.add(f"{label} delta audit is bound to a different plan/support")
    changed = delta.get("changed_union_size")
    selected = delta.get("selected_changed_count")
    if (
        isinstance(changed, bool)
        or not isinstance(changed, int)
        or isinstance(selected, bool)
        or not isinstance(selected, int)
        or changed != selected
        or not 0 <= changed <= k
    ):
        reasons.add(f"{label} changed-coordinate counts are internally inconsistent")
    selected_l2 = delta.get("selected_delta_l2")
    if (
        isinstance(selected_l2, bool)
        or not isinstance(selected_l2, (int, float))
        or not math.isfinite(float(selected_l2))
        or not _same_float(delta.get("total_delta_l2"), float(selected_l2))
    ):
        reasons.add(f"{label} aggregate delta norms are internally inconsistent")
    total_native = delta.get("total_native_coordinates")
    if (
        isinstance(total_native, bool)
        or not isinstance(total_native, int)
        or total_native < k
        or not _same_float(delta.get("support_union_fraction"), k / total_native)
    ):
        reasons.add(f"{label} support-union fraction is internally inconsistent")
    tensors = delta.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != 1:
        reasons.add(f"{label} delta audit does not contain exactly one target tensor")
        return
    tensor = tensors[0]
    if not isinstance(tensor, dict):
        reasons.add(f"{label} target delta evidence is malformed")
        return
    if (
        tensor.get("name") != target_name
        or tensor.get("selected_k") != k
        or tensor.get("changed_union_size") != changed
        or tensor.get("selected_changed_count") != selected
        or tensor.get("outside_support_changed_count") != 0
        or not _is_zero(tensor.get("outside_support_delta_l2"))
        or not _is_zero(tensor.get("outside_support_delta_max_abs"))
    ):
        reasons.add(f"{label} target delta evidence contradicts its aggregate")


def _plan_target_record(
    metadata: Mapping[str, object], target_name: str
) -> Mapping[str, Any] | None:
    records = metadata.get("target_plan_records")
    if not isinstance(records, dict) or set(records) != {target_name}:
        return None
    record = records[target_name]
    return record if isinstance(record, dict) else None


def _validate_attention(
    record: Mapping[str, Any], target_name: str, label: str, reasons: _Reasons
) -> None:
    attention = record.get("attention")
    if not isinstance(attention, dict):
        reasons.add(f"{label} attention evidence is malformed")
        return
    peer_names = attention.get("peer_names")
    alphas = attention.get("alphas")
    if not isinstance(peer_names, list) or not isinstance(alphas, list):
        reasons.add(f"{label} omits peer names or attention alphas")
        return
    if (
        target_name in peer_names
        or len(peer_names) != len(set(peer_names))
        or len(peer_names) != len(alphas)
        or not peer_names
        or attention.get("self_alpha") != 0.0
    ):
        reasons.add(f"{label} violates same-role no-self peer attention")
        return
    try:
        numeric = [float(value) for value in alphas]
    except (TypeError, ValueError):
        reasons.add(f"{label} has nonnumeric attention alphas")
        return
    if not all(
        math.isfinite(value) and value >= 0.0 for value in numeric
    ) or not math.isclose(sum(numeric), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        reasons.add(f"{label} attention alphas are not a probability distribution")


def _paired_summary(
    shuffled: Sequence[float], real: Sequence[float], *, seed: int
) -> dict[str, object]:
    comparison = paired_comparison(shuffled, real, seed=seed)
    raw_differences = cast(list[float], comparison["differences"])
    differences = [float(value) for value in raw_differences]
    wins = sum(value > 0.0 for value in differences)
    losses = sum(value < 0.0 for value in differences)
    ties = len(differences) - wins - losses
    mean = statistics.fmean(differences)
    deviation = statistics.stdev(differences)
    if deviation == 0.0:
        paired_dz: float | str = (
            "positive_infinity"
            if mean > 0.0
            else "negative_infinity"
            if mean < 0.0
            else "undefined_zero_variance"
        )
    else:
        paired_dz = mean / deviation
    return {
        **comparison,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "paired_d_z": paired_dz,
    }


def _authorization_payload(result: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: result[key]
        for key in (
            "schema_version",
            "analysis_class",
            "stage",
            "verdict",
            "passed",
            "reasons",
            "canonical_shape",
            "statistics",
            "evidence",
            "root_of_trust",
            "sealed_test_read",
            "pilot_can_advance",
        )
    }


def _stop_result(
    inputs: GateInputs,
    reasons: _Reasons,
    *,
    evidence: Mapping[str, object] | None = None,
    root_of_trust: Mapping[str, object] | None = None,
    statistics_result: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "2.0",
        "created_at_utc": utc_now(),
        "analysis_class": "PRE_SPECIFIED_CONFIRMATORY",
        "stage": "qualification",
        "verdict": "STOP_NO_PHASE_D",
        "passed": False,
        "reasons": reasons.values(),
        "canonical_shape": {
            "seeds": list(CANONICAL_SEEDS),
            "budget_fractions": list(CANONICAL_BUDGETS),
            "arms": list(CANONICAL_ARMS),
            "run_cells": EXPECTED_RUN_CELL_COUNT,
            "plan_cells": EXPECTED_PLAN_CELL_COUNT,
        },
        "statistics": dict(statistics_result or {}),
        "evidence": dict(evidence or {}),
        "root_of_trust": dict(root_of_trust or {}),
        "sealed_test_read": False,
        "pilot_can_advance": False,
    }
    result["authorization_sha256"] = canonical_json_hash(_authorization_payload(result))
    result["gate_sha256"] = canonical_json_hash(result)
    return result


def _audit_phase_c_gate(
    inputs: GateInputs,
    *,
    plan_evidence_loader: PlanEvidenceLoader = _default_plan_loader,
    commit_verifier: CommitVerifier = _committed_file_evidence,
) -> dict[str, Any]:
    """Audit all qualification evidence and return a self-sealed PASS or STOP."""

    reasons = _Reasons()
    try:
        artifacts = {
            "run": _load_mapping(inputs.run, "run"),
            "plans": _load_mapping(inputs.plans, "plan result"),
            "preregistration": _load_mapping(inputs.preregistration, "preregistration"),
            "budget": _load_mapping(inputs.budget, "budget"),
            "sample": _load_mapping(inputs.sample, "sample manifest"),
            "split": _load_mapping(inputs.split, "split manifest"),
            "gradient_summary": _load_mapping(
                inputs.gradient_summary, "gradient summary"
            ),
        }
        cache_manifests = {
            section: _load_mapping(
                inputs.gradient_cache / f"{section}.json", f"{section} cache"
            )
            for section in ("meta_train", "mask_cal")
        }
        evidence: dict[str, object] = {
            name: _file_record(path)
            for name, path in (
                ("run", inputs.run),
                ("plans", inputs.plans),
                ("preregistration", inputs.preregistration),
                ("budget", inputs.budget),
                ("sample", inputs.sample),
                ("split", inputs.split),
                ("gradient_summary", inputs.gradient_summary),
            )
        }
        evidence["gradient_cache"] = {
            section: {
                "manifest": _file_record(inputs.gradient_cache / f"{section}.json"),
                "tensor": _file_record(
                    inputs.gradient_cache / f"{section}.safetensors"
                ),
            }
            for section in ("meta_train", "mask_cal")
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        reasons.add(f"evidence loading failed closed: {type(error).__name__}: {error}")
        return _stop_result(inputs, reasons)

    run = artifacts["run"]
    plans = artifacts["plans"]
    prereg = artifacts["preregistration"]
    budget = artifacts["budget"]
    sample = artifacts["sample"]
    split = artifacts["split"]
    gradient_summary = artifacts["gradient_summary"]
    self_seals = (
        (run, "run_sha256", "run"),
        (plans, "result_sha256", "plan result"),
        (prereg, "preregistration_sha256", "preregistration"),
        (budget, "budget_manifest_sha256", "budget"),
        (sample, "sample_manifest_sha256", "sample"),
        (split, "split_manifest_sha256", "split"),
        (gradient_summary, "summary_sha256", "gradient summary"),
    )
    for value, seal_key, label in self_seals:
        if not _self_hash_valid(value, seal_key):
            reasons.add(f"{label} self-seal is invalid")
    for section, manifest in cache_manifests.items():
        if not _self_hash_valid(manifest, "cache_sha256"):
            reasons.add(f"{section} cache self-seal is invalid")

    try:
        committed = dict(commit_verifier(inputs.preregistration))
    except (OSError, TypeError, ValueError) as error:
        committed = {"verified": False}
        reasons.add(str(error))
    if committed.get("verified") is not True:
        reasons.add("committed preregistration root of trust is unproven")
    root_of_trust: dict[str, object] = {
        "kind": "committed_immutable_preregistration",
        **committed,
        "preregistration_self_sha256": prereg.get("preregistration_sha256"),
    }

    for artifact, label in (
        (run, "run"),
        (plans, "plan result"),
        (prereg, "preregistration"),
        (budget, "budget"),
        (sample, "sample"),
        (gradient_summary, "gradient summary"),
    ):
        if artifact.get("sealed_test_read") is not False:
            reasons.add(f"{label} does not prove sealed_test_read=false")
    sealed = split.get("sealed")
    if (
        not isinstance(sealed, dict)
        or sealed.get("source") != "test"
        or sealed.get("read_during_split_build") is not False
    ):
        reasons.add("split manifest does not prove the test split remained sealed")
    _validate_access_log(run, "run", reasons)
    _validate_access_log(plans, "plan result", reasons)
    _validate_access_log(prereg, "preregistration", reasons)
    _validate_exact_access_logs(plans, run, inputs, reasons)
    _validate_preregistration_access_log(prereg, reasons)

    if prereg.get("schema_version") != "2.0":
        reasons.add("preregistration schema is not exactly 2.0")
    if prereg.get("analysis_class") != "PRE_SPECIFIED_CONFIRMATORY":
        reasons.add("preregistration analysis class is not confirmatory")
    if prereg.get("root_of_trust") != "committed immutable preregistration artifact":
        reasons.add("preregistration root-of-trust declaration is wrong")
    canonical_gate_policy = {
        "layer_permutation_test_required": True,
        "paired_bootstrap_ci_must_exclude_zero": True,
        "pilot_can_advance": False,
        "real_over_shuffled_seed_wins": "at_least_4_of_5",
        "requires_complete_qualification": True,
    }
    if (
        budget.get("advancement_gate") != canonical_gate_policy
        or prereg.get("advancement_gate") != canonical_gate_policy
    ):
        reasons.add("advancement policy differs from the committed canonical gate")
    if prereg.get("access_policy") != {
        "selection_reads": ["train", "validation"],
        "sealed_split": "test",
        "sealed_read_permitted": False,
        "one_step_dev_may_influence_plans_or_lr": False,
    }:
        reasons.add("preregistered sealed-data access policy is not canonical")
    _validate_source_tree(prereg, reasons)
    if run.get("schema_version") != "2.0" or plans.get("schema_version") != "2.0":
        reasons.add("run and plan-result must both use evidence schema 2.0")
    if run.get("status") != "complete" or run.get("stage") != "qualification":
        reasons.add("run is not a complete qualification run")
    if plans.get("status") != "complete" or plans.get("stage") != "qualification":
        reasons.add("plan result is not a complete qualification plan set")

    qualification = budget.get("qualification")
    if not isinstance(qualification, dict):
        reasons.add("budget has no qualification section")
        qualification = {}
    if qualification.get("seeds") != list(CANONICAL_SEEDS):
        reasons.add("qualification seed grid is not exactly 11/17/23/29/31")
    if qualification.get("budget_fractions") != list(CANONICAL_BUDGETS):
        reasons.add("qualification budget grid is not exactly the five frozen budgets")
    if qualification.get("arms") != list(CANONICAL_ARMS):
        reasons.add("qualification arm grid is not exactly the eight frozen arms")
    if prereg.get("seeds") != list(CANONICAL_SEEDS):
        reasons.add("preregistration seed grid differs from the canonical grid")
    if prereg.get("budget_fractions") != list(CANONICAL_BUDGETS):
        reasons.add("preregistration budget grid differs from the canonical grid")
    if prereg.get("arms") != list(CANONICAL_ARMS):
        reasons.add("preregistration arm grid differs from the canonical grid")

    expected_run_cells = _canonical_cells(CANONICAL_ARMS)
    expected_plan_cells = _canonical_cells(PLANNED_ARMS)
    if (
        prereg.get("expected_run_cells") != expected_run_cells
        or prereg.get("expected_run_cell_count") != EXPECTED_RUN_CELL_COUNT
    ):
        reasons.add("preregistered run-cell shape is not the canonical 200 cells")
    if (
        prereg.get("expected_plan_cells") != expected_plan_cells
        or prereg.get("expected_plan_cell_count") != EXPECTED_PLAN_CELL_COUNT
    ):
        reasons.add("preregistered plan-cell shape is not the canonical 175 cells")
    expected_run_hash = canonical_json_hash(expected_run_cells)
    expected_plan_hash = canonical_json_hash(expected_plan_cells)

    try:
        _validate_file_binding(
            prereg.get("budget"),
            inputs.budget,
            self_key="budget_manifest_sha256",
            live=budget,
            label="budget",
            reasons=reasons,
        )
        _validate_file_binding(
            prereg.get("sample_manifest"),
            inputs.sample,
            self_key="sample_manifest_sha256",
            live=sample,
            label="sample",
            reasons=reasons,
        )
        _validate_file_binding(
            prereg.get("split_manifest"),
            inputs.split,
            self_key="split_manifest_sha256",
            live=split,
            label="split",
            reasons=reasons,
        )
        _validate_file_binding(
            prereg.get("gradient_summary"),
            inputs.gradient_summary,
            self_key="summary_sha256",
            live=gradient_summary,
            label="gradient summary",
            reasons=reasons,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        reasons.add(f"preregistered file binding is malformed: {error}")

    budget_file_sha = sha256_file(inputs.budget)
    plans_file_sha = sha256_file(inputs.plans)
    prereg_file_sha = sha256_file(inputs.preregistration)
    sample_file_sha = sha256_file(inputs.sample)
    split_file_sha = sha256_file(inputs.split)
    summary_file_sha = sha256_file(inputs.gradient_summary)
    base_weight = str(budget.get("base_model", {}).get("weight_sha256", ""))
    base_config = str(budget.get("base_model", {}).get("config_sha256", ""))
    target = prereg.get("target")
    target_name = (
        str(target.get("parameter_name", "")) if isinstance(target, dict) else ""
    )
    target_shape = tuple(target.get("shape", ())) if isinstance(target, dict) else ()
    target_numel = target.get("numel") if isinstance(target, dict) else None
    if (
        target_name != budget.get("target_parameter")
        or target_shape != (1024, 3072)
        or target_numel != 1024 * 3072
        or prereg.get("model", {}).get("weight_sha256") != base_weight
        or prereg.get("model", {}).get("config_sha256") != base_config
    ):
        reasons.add("base/target identity differs across budget and preregistration")
    if budget.get("sample_manifest", {}).get("sha256") != sample_file_sha:
        reasons.add("budget is not bound to the supplied sample manifest")
    if sample.get("tokenizer_sha256") != prereg.get("model", {}).get(
        "tokenizer_sha256"
    ):
        reasons.add("sample tokenizer differs from preregistration")
    prereg_model = prereg.get("model")
    tokenizer_artifacts = (
        prereg_model.get("tokenizer_artifacts")
        if isinstance(prereg_model, dict)
        else None
    )
    expected_tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    }
    tokenizer_hashes: dict[str, str] = {}
    if (
        not isinstance(tokenizer_artifacts, dict)
        or set(tokenizer_artifacts) != expected_tokenizer_names
    ):
        reasons.add("preregistration does not bind the exact tokenizer artifact set")
        tokenizer_artifacts = {}
    for name in sorted(expected_tokenizer_names):
        record = tokenizer_artifacts.get(name)
        if not isinstance(record, dict):
            continue
        path = Path(str(record.get("path", "")))
        try:
            digest = sha256_file(path)
            size = path.stat().st_size
        except OSError as error:
            reasons.add(f"tokenizer artifact is unreadable ({name}): {error}")
            continue
        tokenizer_hashes[name] = digest
        if record.get("sha256") != digest or record.get("size") != size:
            reasons.add(f"tokenizer artifact differs from preregistration: {name}")
    tokenizer_artifacts_hash = canonical_json_hash(tokenizer_hashes)
    if (
        not isinstance(prereg_model, dict)
        or prereg_model.get("tokenizer_artifacts_sha256") != tokenizer_artifacts_hash
        or prereg_model.get("tokenizer_sha256")
        != tokenizer_hashes.get("tokenizer.json")
    ):
        reasons.add("tokenizer artifact aggregate identity is inconsistent")

    optimizer = budget.get("native_optimizer")
    if not isinstance(optimizer, dict):
        optimizer = {}
        reasons.add("native optimizer configuration is absent")
    if prereg.get("optimizer") != optimizer:
        reasons.add("preregistered optimizer differs from the budget")
    optimizer_signature: dict[str, object] = {
        "kind": optimizer.get("kind"),
        "betas": optimizer.get("betas"),
        "epsilon": optimizer.get("epsilon"),
        "weight_decay": optimizer.get("weight_decay"),
        "learning_rate_grid": optimizer.get("learning_rate_grid"),
        "selection_split": optimizer.get("selection_split"),
        "step_count": 1,
    }
    optimizer_signature["signature_sha256"] = canonical_json_hash(optimizer_signature)
    optimizer_hash = str(optimizer_signature["signature_sha256"])
    lr_grid_raw = optimizer.get("learning_rate_grid")
    lr_grid = (
        [float(value) for value in lr_grid_raw] if isinstance(lr_grid_raw, list) else []
    )
    if (
        optimizer.get("kind") != "ExactSupportAdamW"
        or optimizer.get("selection_split") != "META_TRAIN_only"
        or len(lr_grid) != len(set(lr_grid))
        or not lr_grid
    ):
        reasons.add(
            "optimizer kind, LR grid, or selection split is not frozen correctly"
        )
    lr_grid_hash = canonical_json_hash(lr_grid_raw)

    data_signature: dict[str, object]
    frozen_dev = qualification.get("one_step_dev")
    if not isinstance(frozen_dev, list) or not frozen_dev:
        reasons.add("frozen ONE_STEP_DEV data is absent")
        frozen_dev = []
        data_signature = {"examples": [], "signature_sha256": ""}
    else:
        try:
            data_signature = _expected_data_signature(frozen_dev)
        except (KeyError, TypeError, ValueError) as error:
            reasons.add(f"frozen ONE_STEP_DEV data is malformed: {error}")
            frozen_dev = []
            data_signature = {"examples": [], "signature_sha256": ""}
    if prereg.get("one_step_dev") != frozen_dev:
        reasons.add("preregistered ONE_STEP_DEV rows differ from the budget")
    maximum_tokens = prereg.get("maximum_tokens")
    if (
        isinstance(maximum_tokens, bool)
        or not isinstance(maximum_tokens, int)
        or maximum_tokens <= 0
        or sample.get("maximum_tokens") != maximum_tokens
    ):
        reasons.add("maximum token limit differs across preregistration and sample")
    encoded_dev = prereg.get("encoded_one_step_dev")
    encoded_signature = prereg.get("encoded_data_signature_sha256")
    if (
        not isinstance(encoded_dev, list)
        or len(encoded_dev) != len(frozen_dev)
        or encoded_signature != canonical_json_hash(encoded_dev)
    ):
        reasons.add("encoded ONE_STEP_DEV aggregate signature is invalid")
        encoded_dev = []
    for frozen, encoded in zip(frozen_dev, encoded_dev, strict=True):
        if (
            not isinstance(encoded, dict)
            or encoded.get("row_id") != frozen.get("row_id")
            or encoded.get("token_count") != frozen.get("token_count")
            or encoded.get("assistant_token_count")
            != frozen.get("assistant_token_count")
            or not isinstance(encoded.get("input_ids_sha256"), str)
            or _SHA256.fullmatch(str(encoded.get("input_ids_sha256"))) is None
            or not isinstance(encoded.get("labels_sha256"), str)
            or _SHA256.fullmatch(str(encoded.get("labels_sha256"))) is None
        ):
            reasons.add(
                "encoded ONE_STEP_DEV row evidence differs from frozen metadata"
            )
            break
    encoded_by_stage = prereg.get("encoded_one_step_dev_by_stage")
    signatures_by_stage = prereg.get("encoded_data_signature_sha256_by_stage")
    if (
        not isinstance(encoded_by_stage, dict)
        or not isinstance(signatures_by_stage, dict)
        or set(encoded_by_stage) != {"pilot", "qualification"}
        or set(signatures_by_stage) != {"pilot", "qualification"}
    ):
        reasons.add("stage-specific encoded data maps are absent or noncanonical")
    else:
        for stage_name in ("pilot", "qualification"):
            stage_rows = encoded_by_stage[stage_name]
            stage_budget = budget.get(stage_name)
            frozen_stage_rows = (
                stage_budget.get("one_step_dev", [])
                if isinstance(stage_budget, dict)
                else []
            )
            if (
                not isinstance(stage_rows, list)
                or signatures_by_stage[stage_name] != canonical_json_hash(stage_rows)
                or len(stage_rows) != len(frozen_stage_rows)
                or any(
                    not isinstance(encoded, dict)
                    or encoded.get("row_id") != frozen.get("row_id")
                    or encoded.get("token_count") != frozen.get("token_count")
                    or encoded.get("assistant_token_count")
                    != frozen.get("assistant_token_count")
                    or not isinstance(encoded.get("input_ids_sha256"), str)
                    or _SHA256.fullmatch(str(encoded.get("input_ids_sha256"))) is None
                    or not isinstance(encoded.get("labels_sha256"), str)
                    or _SHA256.fullmatch(str(encoded.get("labels_sha256"))) is None
                    for frozen, encoded in zip(
                        frozen_stage_rows, stage_rows, strict=True
                    )
                )
            ):
                reasons.add(f"{stage_name} encoded data map is invalid")
        if (
            encoded_by_stage["qualification"] != encoded_dev
            or signatures_by_stage["qualification"] != encoded_signature
        ):
            reasons.add("qualification encoded-data aliases differ from stage map")
    sample_sections = sample.get("sections")
    sample_dev = (
        sample_sections.get("one_step_dev", [])
        if isinstance(sample_sections, dict)
        else []
    )
    sample_dev_by_id = {
        row.get("row_id"): row for row in sample_dev if isinstance(row, dict)
    }
    if any(
        sample_dev_by_id.get(row.get("row_id")) != row
        for row in frozen_dev
        if isinstance(row, dict)
    ):
        reasons.add("qualification ONE_STEP_DEV is not an exact subset of the sample")
    split_dev = split.get("one_step_dev")
    split_ids: set[str] = set()
    if isinstance(split_dev, dict):
        for component in split_dev.get("components", []):
            if isinstance(component, dict):
                split_ids.update(str(value) for value in component.get("row_ids", []))
    if any(str(row.get("row_id")) not in split_ids for row in frozen_dev):
        reasons.add("qualification ONE_STEP_DEV rows are absent from the frozen split")
    if sample.get("split_manifest", {}).get("sha256") != split_file_sha:
        reasons.add("sample manifest is not bound to the supplied split manifest")

    cache_file_records: dict[str, dict[str, object]] = {}
    prereg_cache = prereg.get("gradient_cache")
    summary_cache = gradient_summary.get("cache")
    budget_calibration = budget.get("calibration")
    for section, manifest in cache_manifests.items():
        manifest_path = inputs.gradient_cache / f"{section}.json"
        tensor_path = inputs.gradient_cache / f"{section}.safetensors"
        manifest_file_sha = sha256_file(manifest_path)
        tensor_file_sha = sha256_file(tensor_path)
        cache_file_records[section] = {
            "manifest_sha256": manifest_file_sha,
            "tensor_sha256": tensor_file_sha,
            "manifest_self_sha256": manifest.get("cache_sha256"),
            "row_ids": [
                row.get("row_id")
                for row in manifest.get("report", {}).get("examples", [])
                if isinstance(row, dict)
            ],
        }
        frozen_cache = (
            prereg_cache.get(section) if isinstance(prereg_cache, dict) else None
        )
        if not isinstance(frozen_cache, dict):
            reasons.add(f"preregistration lacks {section} gradient-cache evidence")
            continue
        if (
            frozen_cache.get("manifest", {}).get("sha256") != manifest_file_sha
            or frozen_cache.get("manifest", {}).get("size")
            != manifest_path.stat().st_size
            or frozen_cache.get("manifest_self_sha256") != manifest.get("cache_sha256")
            or frozen_cache.get("tensor", {}).get("sha256") != tensor_file_sha
            or frozen_cache.get("tensor", {}).get("size") != tensor_path.stat().st_size
        ):
            reasons.add(f"{section} cache files differ from preregistration")
        report_rows = cache_file_records[section]["row_ids"]
        expected_rows = (
            [row.get("row_id") for row in budget_calibration.get(section, [])]
            if isinstance(budget_calibration, dict)
            else []
        )
        if report_rows != expected_rows or frozen_cache.get("row_ids") != expected_rows:
            reasons.add(f"{section} cache rows differ from the frozen calibration rows")
        if (
            manifest.get("name") != section
            or manifest.get("target") != target_name
            or manifest.get("model", {}).get("weight_sha256") != base_weight
            or manifest.get("model", {}).get("config_sha256") != base_config
            or manifest.get("tensor", {}).get("sha256") != tensor_file_sha
            or manifest.get("tensor", {}).get("shape") != list(target_shape)
            or manifest.get("tensor", {}).get("dtype") != "torch.float32"
        ):
            reasons.add(f"{section} cache identity/shape/provenance is inconsistent")
        summary_section = (
            summary_cache.get(section) if isinstance(summary_cache, dict) else None
        )
        if (
            not isinstance(summary_section, dict)
            or summary_section.get("manifest_sha256") != manifest.get("cache_sha256")
            or summary_section.get("tensor_sha256") != tensor_file_sha
            or summary_section.get("report") != manifest.get("report")
        ):
            reasons.add(f"gradient summary contradicts the {section} cache manifest")
    if (
        gradient_summary.get("status") != "complete"
        or gradient_summary.get("base_weight_sha256") != base_weight
        or gradient_summary.get("budget_manifest_sha256") != budget_file_sha
        or gradient_summary.get("target") != target_name
        or gradient_summary.get("native_weight_delta") != "bitwise_zero"
        or set(gradient_summary.get("computed_sections", [])).union(
            gradient_summary.get("resumed_sections", [])
        )
        != {"meta_train", "mask_cal"}
    ):
        reasons.add("gradient summary is incomplete or has inconsistent provenance")

    layer_permutations = _validate_layer_permutations(prereg, reasons)

    expected_run_keys = {
        (seed, fraction, arm)
        for seed in CANONICAL_SEEDS
        for fraction in CANONICAL_BUDGETS
        for arm in CANONICAL_ARMS
    }
    expected_plan_keys = {
        (seed, fraction, arm)
        for seed in CANONICAL_SEEDS
        for fraction in CANONICAL_BUDGETS
        for arm in PLANNED_ARMS
    }
    run_records = _index_records(
        run.get("records"),
        expected=expected_run_keys,
        label="run",
        expected_count=EXPECTED_RUN_CELL_COUNT,
        reasons=reasons,
    )
    plan_records = _index_records(
        plans.get("records"),
        expected=expected_plan_keys,
        label="plan result",
        expected_count=EXPECTED_PLAN_CELL_COUNT,
        reasons=reasons,
    )
    for record in run_records.values():
        if not _self_hash_valid(record, "record_sha256"):
            reasons.add("one or more run record self-seals are invalid")
            break
    for record in plan_records.values():
        if not _self_hash_valid(record, "record_sha256"):
            reasons.add("one or more plan record self-seals are invalid")
            break

    shared_provenance: dict[str, object] = {
        "preregistration_file_sha256": prereg_file_sha,
        "preregistration_self_sha256": prereg.get("preregistration_sha256"),
        "budget_file_sha256": budget_file_sha,
        "budget_self_sha256": budget.get("budget_manifest_sha256"),
        "sample_manifest_sha256": sample_file_sha,
        "split_manifest_sha256": split_file_sha,
        "model_weight_sha256": base_weight,
        "model_config_sha256": base_config,
        "tokenizer_sha256": prereg.get("model", {}).get("tokenizer_sha256"),
    }
    provenance_expectations = (
        (
            plans,
            "plan result",
            {**shared_provenance, "target": target_name, "role": "mlp.down_proj"},
        ),
        (
            run,
            "run",
            {
                **shared_provenance,
                "gradient_summary_sha256": summary_file_sha,
                "optimizer_signature": optimizer_signature,
                "data_signature": data_signature,
                "tokenizer_artifacts": tokenizer_artifacts,
                "tokenizer_artifacts_sha256": tokenizer_artifacts_hash,
                "encoded_data_signature_sha256": encoded_signature,
                "maximum_tokens": maximum_tokens,
            },
        ),
    )
    for artifact, label, expected_provenance in provenance_expectations:
        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict):
            reasons.add(f"{label} has no schema-v2 provenance")
            continue
        for field_name, expected_value in expected_provenance.items():
            if provenance.get(field_name) != expected_value:
                reasons.add(f"{label} provenance field {field_name} differs")
        if provenance.get("gradient_cache") != cache_file_records:
            reasons.add(f"{label} gradient-cache provenance differs")
    plan_provenance = plans.get("provenance", {})
    run_provenance = run.get("provenance", {})
    if isinstance(plan_provenance, dict) and (
        plan_provenance.get("expected_plan_cells_sha256") != expected_plan_hash
    ):
        reasons.add("plan result is not bound to the canonical 175-cell plan grid")
    if isinstance(run_provenance, dict) and (
        run_provenance.get("expected_run_cells_sha256") != expected_run_hash
    ):
        reasons.add("run is not bound to the canonical 200-cell run grid")
    if run.get("budget_manifest_sha256") != budget_file_sha:
        reasons.add("run is not bound to the supplied budget file")
    if run.get("plan_result_sha256") != plans_file_sha:
        reasons.add("run is not bound to the supplied plan-result file")
    if run.get("base_weight_sha256") != base_weight:
        reasons.add("run base weight differs from preregistration")
    if plans.get("base_fingerprint") != base_weight:
        reasons.add("plan-result base weight differs from preregistration")
    if (
        not isinstance(run_provenance, dict)
        or run_provenance.get("optimizer_signature") != optimizer_signature
    ):
        reasons.add("run optimizer signature is absent or inconsistent")
    if (
        not isinstance(run_provenance, dict)
        or run_provenance.get("data_signature") != data_signature
    ):
        reasons.add("run held-out data signature is absent or inconsistent")
    environment_signature = (
        run_provenance.get("environment_signature")
        if isinstance(run_provenance, dict)
        else None
    )
    if not isinstance(environment_signature, dict) or not _self_hash_valid(
        environment_signature, "signature_sha256"
    ):
        reasons.add("run environment signature is absent or invalid")
    if not isinstance(run_provenance, dict) or (
        run_provenance.get("device") != run.get("device")
    ):
        reasons.add("run device differs from its environment-bound provenance")

    loaded_by_cell: dict[tuple[int, float, str], LoadedPlanEvidence] = {}
    plan_directories: set[Path] = set()
    capacity_signatures: dict[tuple[int, float], set[tuple[object, ...]]] = defaultdict(
        set
    )
    for plan_key in sorted(expected_plan_keys):
        plan_record = plan_records.get(plan_key)
        run_record = run_records.get(plan_key)
        if plan_record is None or run_record is None:
            continue
        seed, fraction, arm = plan_key
        label = f"seed={seed}/budget={fraction}/arm={arm}"
        expected_k = budget_to_k(int(target_numel or 0), fraction)
        frozen_exact_k = prereg.get("exact_k", {}).get(str(fraction))
        support_stats = plan_record.get("support_stats")
        if (
            plan_record.get("stage") != "qualification"
            or plan_record.get("k") != expected_k
            or frozen_exact_k != expected_k
            or not isinstance(support_stats, dict)
            or support_stats.get("union_k") != expected_k
            or support_stats.get("per_target_k") != {target_name: expected_k}
            or support_stats.get("total_numel") != target_numel
            or not _same_float(
                support_stats.get("union_fraction"), expected_k / int(target_numel or 1)
            )
        ):
            reasons.add(f"{label} plan K/union differs from the frozen exact support")
        directory_raw = plan_record.get("plan_directory")
        if not isinstance(directory_raw, str):
            reasons.add(f"{label} plan directory is absent")
            continue
        directory = Path(directory_raw).resolve()
        if directory in plan_directories:
            reasons.add(f"{label} reuses another cell's plan directory")
        plan_directories.add(directory)
        lower_parts = {part.lower() for part in directory.parts}
        if "qualification" not in lower_parts or "pilot" in lower_parts:
            reasons.add(f"{label} plan path is not a qualification path")
        manifest_path = directory / "plan.json"
        tensor_path = directory / "tensors.safetensors"
        try:
            actual_manifest_sha = sha256_file(manifest_path)
            actual_tensor_sha = sha256_file(tensor_path)
        except OSError as error:
            reasons.add(f"{label} persisted plan files are unreadable: {error}")
            continue
        if (
            plan_record.get("plan_manifest_sha256") != actual_manifest_sha
            or plan_record.get("plan_tensor_sha256") != actual_tensor_sha
            or run_record.get("plan_manifest_sha256") != actual_manifest_sha
            or run_record.get("plan_tensor_sha256") != actual_tensor_sha
        ):
            reasons.add(f"{label} persisted plan file hashes differ across evidence")
        try:
            loaded = plan_evidence_loader(directory, base_weight)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            reasons.add(f"{label} persisted plan reload failed: {error}")
            continue
        loaded_by_cell[plan_key] = loaded
        if (
            loaded.base_fingerprint != base_weight
            or loaded.target_name != target_name
            or loaded.shape != target_shape
            or loaded.k != expected_k
            or not _same_float(loaded.support_fraction, fraction)
            or loaded.plan_hash != plan_record.get("plan_hash")
            or loaded.support_hash != plan_record.get("support_hash")
            or loaded.plan_hash != run_record.get("plan_hash")
            or loaded.support_hash != run_record.get("support_hash")
            or dict(loaded.support_stats) != support_stats
        ):
            reasons.add(f"{label} loaded native plan/support identity differs")
        metadata = dict(loaded.metadata)
        if metadata != plan_record.get("metadata"):
            reasons.add(f"{label} plan metadata differs from the persisted plan")
        target_record = _plan_target_record(metadata, target_name)
        if (
            target_record is None
            or target_record.get("exact_k") != expected_k
            or target_record.get("shape") != list(target_shape)
            or target_record.get("semantic_role") != "mlp.down_proj"
            or not _same_float(target_record.get("requested_fraction"), fraction)
        ):
            reasons.add(f"{label} target plan record is internally inconsistent")
            target_record = {}
        generator_config = target_record.get("generator_config", {})
        attention_config = metadata.get("attention_config", {})
        frozen_generator = budget.get("generator", {})
        if arm not in {"random_sparse_native", "gradient_topk_sparse_native"}:
            bounds = (
                frozen_generator.get("stabilizer_bounds", [])
                if isinstance(frozen_generator, dict)
                else []
            )
            if (
                not isinstance(generator_config, dict)
                or generator_config.get("descriptor_dimension")
                != frozen_generator.get("descriptor_dimension")
                or generator_config.get("stabilizer_lower")
                != (bounds[0] if len(bounds) == 2 else None)
                or generator_config.get("stabilizer_upper")
                != (bounds[1] if len(bounds) == 2 else None)
                or metadata.get("generator_learning_rate")
                != frozen_generator.get("calibration_learning_rate")
            ):
                reasons.add(f"{label} generator configuration differs from the budget")
        if arm in PEER_CAPACITY_ARMS and (
            not isinstance(attention_config, dict)
            or attention_config.get("descriptor_dimension")
            != frozen_generator.get("descriptor_dimension")
        ):
            reasons.add(f"{label} attention configuration differs from the budget")
        generator_count = metadata.get("generator_parameter_count", 0)
        attention_count = metadata.get("attention_parameter_count", 0)
        valid_counts = (
            not isinstance(generator_count, bool)
            and isinstance(generator_count, int)
            and not isinstance(attention_count, bool)
            and isinstance(attention_count, int)
        )
        if (
            not isinstance(generator_count, bool)
            and isinstance(generator_count, int)
            and not isinstance(attention_count, bool)
            and isinstance(attention_count, int)
        ):
            generator_count_int = generator_count
            attention_count_int = attention_count
        else:
            generator_count_int = -1
            attention_count_int = -1
        if (
            plan_record.get("generator_architecture_sha256")
            != canonical_json_hash(generator_config)
            or plan_record.get("attention_architecture_sha256")
            != canonical_json_hash(attention_config)
            or plan_record.get("generator_parameter_count") != generator_count
            or plan_record.get("attention_parameter_count") != attention_count
            or not valid_counts
            or plan_record.get("total_generator_capacity")
            != generator_count_int + attention_count_int
        ):
            reasons.add(f"{label} generator/attention architecture capacity differs")
        if arm in PEER_CAPACITY_ARMS:
            _validate_attention(plan_record, target_name, label, reasons)
            capacity_signatures[(seed, fraction)].add(
                (
                    plan_record.get("generator_architecture_sha256"),
                    generator_count,
                    plan_record.get("attention_architecture_sha256"),
                    attention_count,
                    plan_record.get("total_generator_capacity"),
                )
            )
        steps = (
            int(budget.get("generator", {}).get("bootstrap_steps", -1))
            if arm == "real_peer_pcsm_bootstrap"
            else int(budget.get("generator", {}).get("calibration_steps", -1))
            if arm
            in {
                "target_only_sparse_native",
                "coordinate_shuffled_peer_pcsm",
                "layer_shuffled_peer_pcsm",
                "real_peer_pcsm_calibrated",
            }
            else 0
        )
        if arm not in {"random_sparse_native", "gradient_topk_sparse_native"} and (
            metadata.get("steps") != steps
            or metadata.get("generator_learning_rate")
            != frozen_generator.get("calibration_learning_rate")
            or metadata.get("structural_regularizer")
            != frozen_generator.get("structural_regularizer_primary")
            or metadata.get("structural_proxy_learning_rate")
            != frozen_generator.get("structural_proxy_learning_rate")
            or metadata.get("topk_ste_temperature")
            != frozen_generator.get("topk_ste_temperature")
            or metadata.get("support_frozen_once_after_calibration") is not True
        ):
            reasons.add(f"{label} calibrated generator controls differ from budget")
        trace = plan_record.get("calibration_trace")
        meta_compute = plan_record.get("meta_compute")
        trace_steps = (
            [entry.get("step") for entry in trace if isinstance(entry, dict)]
            if isinstance(trace, list)
            else []
        )
        if (
            not isinstance(trace, list)
            or trace_steps != [float(step) for step in range(steps)]
            or not isinstance(meta_compute, dict)
            or meta_compute.get("calibration_steps") != steps
            or meta_compute.get("backward_count") != steps
            or meta_compute.get("trace_length") != steps
            or meta_compute.get("meta_train") != cache_file_records["meta_train"]
            or meta_compute.get("mask_cal") != cache_file_records["mask_cal"]
        ):
            reasons.add(f"{label} meta-step sequence or cache provenance differs")
        if arm == "layer_shuffled_peer_pcsm":
            frozen_permutation = layer_permutations.get(seed)
            if (
                frozen_permutation is None
                or plan_record.get("layer_depth_permutation") != frozen_permutation
                or metadata.get("layer_depth_assignment")
                != frozen_permutation.get("mapping")
                or metadata.get("layer_depth_assignment_sha256")
                != frozen_permutation.get("mapping_sha256")
                or metadata.get("layer_depth_assignment_nonidentity") is not True
            ):
                reasons.add(f"{label} layer permutation differs from preregistration")
    for cell, signatures in capacity_signatures.items():
        if len(signatures) != 1:
            reasons.add(
                f"seed={cell[0]}/budget={cell[1]} peer controls have unequal capacity"
            )

    losses: dict[tuple[int, float, str], float] = {}
    for run_key in sorted(expected_run_keys):
        run_record_current = run_records.get(run_key)
        if run_record_current is None:
            continue
        seed, fraction, arm = run_key
        label = f"seed={seed}/budget={fraction}/arm={arm}"
        expected_k = (
            0
            if arm == "no_update_frozen"
            else budget_to_k(int(target_numel or 0), fraction)
        )
        if run_record_current.get("k") != expected_k:
            reasons.add(f"{label} run K differs from the frozen exact support")
        if (
            run_record_current.get("preregistration_file_sha256") != prereg_file_sha
            or run_record_current.get("dev_data_signature")
            != data_signature["signature_sha256"]
            or run_record_current.get("optimizer_config_sha256") != optimizer_hash
            or run_record_current.get("base_hash_verified") is not True
            or run_record_current.get("access_log_sha256")
            != run.get("access_log_sha256")
            or run_record_current.get("learning_rate_grid_sha256") != lr_grid_hash
            or run_record_current.get("encoded_data_signature_sha256")
            != encoded_signature
            or run_record_current.get("maximum_tokens") != maximum_tokens
        ):
            reasons.add(f"{label} common provenance is absent or inconsistent")
        loss = _validate_evaluation(
            run_record_current,
            frozen_dev,
            str(data_signature["signature_sha256"]),
            label,
            reasons,
        )
        if loss is not None:
            losses[run_key] = loss
        if arm == "no_update_frozen":
            if run_record_current.get("learning_rate") != 0.0:
                reasons.add(f"{label} baseline has a nonzero learning rate")
            full_before = run_record_current.get("full_parameter_hash_before")
            full_after = run_record_current.get("full_parameter_hash_after")
            if (
                not isinstance(full_before, str)
                or _SHA256.fullmatch(full_before) is None
                or full_before != full_after
            ):
                reasons.add(f"{label} baseline full-model bitwise-zero proof failed")
            _validate_delta(
                run_record_current,
                k=0,
                plan_hash=None,
                support_hash=None,
                target_name=target_name,
                label=label,
                reasons=reasons,
            )
            continue
        current_plan_record = plan_records.get(run_key)
        run_loaded = loaded_by_cell.get(run_key)
        if current_plan_record is None or run_loaded is None:
            continue
        if (
            run_record_current.get("plan_record_sha256")
            != current_plan_record.get("record_sha256")
            or run_record_current.get("plan_manifest_sha256")
            != current_plan_record.get("plan_manifest_sha256")
            or run_record_current.get("plan_tensor_sha256")
            != current_plan_record.get("plan_tensor_sha256")
        ):
            reasons.add(f"{label} run is not bound to its plan-result record/files")
        predictions = run_record_current.get("meta_lr_predictions")
        prediction_grid: dict[float, float] = {}
        if isinstance(predictions, dict):
            try:
                prediction_grid = {
                    float(key_value): float(prediction)
                    for key_value, prediction in predictions.items()
                }
            except (TypeError, ValueError):
                prediction_grid = {}
        selected_lr = run_record_current.get("learning_rate")
        expected_selected = (
            min(lr_grid, key=lambda value: (prediction_grid[value], value))
            if set(prediction_grid) == set(lr_grid)
            and all(math.isfinite(value) for value in prediction_grid.values())
            else None
        )
        if (
            set(prediction_grid) != set(lr_grid)
            or selected_lr != expected_selected
            or run_record_current.get("learning_rate_grid_sha256") != lr_grid_hash
            or run_record_current.get("learning_rate_selection_split")
            != "META_TRAIN_only"
            or run_record_current.get("learning_rate_source_cache_sha256")
            != cache_file_records["meta_train"]["manifest_sha256"]
        ):
            reasons.add(
                f"{label} LR grid/selection is incomplete or not META_TRAIN-only"
            )
        if (
            run_record_current.get("optimizer_state_bytes") != expected_k * 16
            or run_record_current.get("checkpoint_delta_bytes_estimate")
            != expected_k * 10
        ):
            reasons.add(f"{label} sparse optimizer/checkpoint budget is inconsistent")
        _validate_delta(
            run_record_current,
            k=expected_k,
            plan_hash=run_loaded.plan_hash,
            support_hash=run_loaded.support_hash,
            target_name=target_name,
            label=label,
            reasons=reasons,
        )

    statistics_result: dict[str, object] = {
        "difference_definition": "shuffled_loss_minus_real_loss; positive favors real",
        "seed_count": 5,
        "budget_count_per_seed": 5,
        "aggregation": "unweighted mean across the five preregistered budgets per seed",
        "small_n_caveat": {
            "n": 5,
            "minimum_attainable_two_sided_exact_sign_flip_p_value": 0.0625,
            "conventional_p_below_0p05_attainable": False,
            "interpretation": (
                "The gate uses prespecified wins and a paired bootstrap CI; with n=5, "
                "a two-sided exact sign-flip test cannot establish p<0.05."
            ),
        },
    }
    required_stats_keys = {
        (seed, fraction, arm)
        for seed in CANONICAL_SEEDS
        for fraction in CANONICAL_BUDGETS
        for arm in (
            "coordinate_shuffled_peer_pcsm",
            "layer_shuffled_peer_pcsm",
            "real_peer_pcsm_calibrated",
        )
    }
    if required_stats_keys.issubset(losses):
        real_by_seed = [
            statistics.fmean(
                losses[(seed, fraction, "real_peer_pcsm_calibrated")]
                for fraction in CANONICAL_BUDGETS
            )
            for seed in CANONICAL_SEEDS
        ]
        coordinate_by_seed = [
            statistics.fmean(
                losses[(seed, fraction, "coordinate_shuffled_peer_pcsm")]
                for fraction in CANONICAL_BUDGETS
            )
            for seed in CANONICAL_SEEDS
        ]
        layer_by_seed = [
            statistics.fmean(
                losses[(seed, fraction, "layer_shuffled_peer_pcsm")]
                for fraction in CANONICAL_BUDGETS
            )
            for seed in CANONICAL_SEEDS
        ]
        coordinate = _paired_summary(coordinate_by_seed, real_by_seed, seed=20260827)
        layer = _paired_summary(layer_by_seed, real_by_seed, seed=20260828)
        statistics_result.update(
            {
                "per_seed_real_loss": dict(
                    zip(CANONICAL_SEEDS, real_by_seed, strict=True)
                ),
                "coordinate_shuffled_minus_real": coordinate,
                "layer_shuffled_minus_real": layer,
            }
        )
        coordinate_ci = cast(dict[str, object], coordinate["confidence_interval"])
        layer_ci = cast(dict[str, object], layer["confidence_interval"])
        coordinate_wins = cast(int, coordinate["wins"])
        layer_wins = cast(int, layer["wins"])
        if coordinate_wins < 4 or float(cast(float, coordinate_ci["low"])) <= 0.0:
            reasons.add(
                "real peers fail the 4/5 wins plus positive paired-bootstrap CI gate "
                "against coordinate-shuffled peers"
            )
        if layer_wins < 4 or float(cast(float, layer_ci["low"])) <= 0.0:
            reasons.add(
                "real peers fail the 4/5 wins plus positive paired-bootstrap CI gate "
                "against layer-shuffled peers"
            )
    else:
        reasons.add(
            "paired gate statistics cannot be computed from all 75 required cells"
        )

    if reasons:
        return _stop_result(
            inputs,
            reasons,
            evidence=evidence,
            root_of_trust=root_of_trust,
            statistics_result=statistics_result,
        )
    result = _stop_result(
        inputs,
        reasons,
        evidence=evidence,
        root_of_trust=root_of_trust,
        statistics_result=statistics_result,
    )
    result["passed"] = True
    result["verdict"] = "PASS_ADVANCE_TO_PHASE_D"
    result["authorization_sha256"] = canonical_json_hash(_authorization_payload(result))
    result.pop("gate_sha256")
    result["gate_sha256"] = canonical_json_hash(result)
    return result


def audit_phase_c_gate(
    inputs: GateInputs,
    *,
    plan_evidence_loader: PlanEvidenceLoader = _default_plan_loader,
    commit_verifier: CommitVerifier = _committed_file_evidence,
) -> dict[str, Any]:
    """Audit all evidence; unexpected malformed input becomes a sealed STOP."""

    try:
        return _audit_phase_c_gate(
            inputs,
            plan_evidence_loader=plan_evidence_loader,
            commit_verifier=commit_verifier,
        )
    # Authorization must fail closed even for an unanticipated malformed type.
    except Exception as error:
        reasons = _Reasons()
        reasons.add(
            f"evidence validation failed closed: {type(error).__name__}: {error}"
        )
        return _stop_result(inputs, reasons)


def write_gate_artifact(path: Path, result: Mapping[str, Any]) -> None:
    """Atomically persist a previously audited gate artifact."""

    if not gate_self_hash_valid(result):
        raise ValueError("refusing to persist a gate with an invalid self-seal")
    atomic_write_json(path, dict(result))


def authenticate_phase_d_gate(
    gate_path: Path,
    inputs: GateInputs,
    *,
    plan_evidence_loader: PlanEvidenceLoader = _default_plan_loader,
    commit_verifier: CommitVerifier = _committed_file_evidence,
) -> dict[str, Any]:
    """Re-audit Phase-C and authenticate its gate before any Phase-D code runs."""

    gate = _load_mapping(gate_path, "Phase-C gate")
    if not gate_self_hash_valid(gate):
        raise RuntimeError("Phase D blocked: gate self-hash is invalid")
    if (
        gate.get("schema_version") != "2.0"
        or gate.get("stage") != "qualification"
        or gate.get("analysis_class") != "PRE_SPECIFIED_CONFIRMATORY"
        or gate.get("verdict") != "PASS_ADVANCE_TO_PHASE_D"
        or gate.get("passed") is not True
        or gate.get("pilot_can_advance") is not False
        or gate.get("sealed_test_read") is not False
    ):
        raise RuntimeError(
            "Phase D blocked: gate envelope is not an authorized qualification"
        )
    fresh = audit_phase_c_gate(
        inputs,
        plan_evidence_loader=plan_evidence_loader,
        commit_verifier=commit_verifier,
    )
    if fresh.get("passed") is not True:
        raise RuntimeError("Phase D blocked: fresh Phase-C evidence audit did not pass")
    if (
        gate.get("authorization_sha256") != fresh.get("authorization_sha256")
        or gate.get("evidence") != fresh.get("evidence")
        or gate.get("root_of_trust") != fresh.get("root_of_trust")
    ):
        raise RuntimeError("Phase D blocked: gate is forged or evidence has changed")
    return gate


__all__ = [
    "CANONICAL_ARMS",
    "CANONICAL_BUDGETS",
    "CANONICAL_SEEDS",
    "GateInputs",
    "LoadedPlanEvidence",
    "audit_phase_c_gate",
    "authenticate_phase_d_gate",
    "gate_self_hash_valid",
    "write_gate_artifact",
]
