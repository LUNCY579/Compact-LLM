"""Fail-closed, model-free provenance gates. Digests authenticate bytes, not claims."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path


class Blocked(ValueError):
    """An input is not scientifically comparable or not safely executable."""


def require(condition, message):
    if not condition:
        raise Blocked("BLOCKED_ON_COMPARABILITY: " + message)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_hash(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            name + " missing/invalid SHA256")


def write_once(path, value):
    """Never rewrite evidence, including after a failed or partial attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
    return file_digest(path)


def verify_file(path, expected):
    valid_hash(expected, "file digest")
    require(Path(path).is_file(), "missing artifact: " + str(path))
    require(file_digest(path) == expected, "artifact bytes differ: " + str(path))


def implementation_digest(root, paths):
    require(bool(paths) and len(set(paths)) == len(paths), "empty/duplicate source inventory")
    root = Path(root).resolve()
    inventory = {}
    for name in sorted(paths):
        target = (root / name).resolve()
        require(target.is_relative_to(root), "source path escapes root")
        inventory[name] = file_digest(target)
    return {"files": inventory, "sha256": digest(inventory)}


def verify_source(root, expected_commit, clean=True):
    require(isinstance(expected_commit, str) and re.fullmatch(r"[0-9a-f]{40}", expected_commit),
            "missing source commit")
    def git(*args):
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        require(done.returncode == 0, "git source check failed")
        return done.stdout.strip()
    require(git("rev-parse", "HEAD") == expected_commit, "stale source commit")
    if clean:
        require(not git("status", "--porcelain=v1", "--untracked-files=all"),
                "dirty/untracked source or evidence")
    return expected_commit


def validate_calibration(binding, forbidden_row_ids=(), forbidden_parent_ids=()):
    """Role labels alone are insufficient; check frozen ID/parent deny-lists too."""
    valid_hash(binding.get("dataset_sha256"), "calibration dataset")
    valid_hash(binding.get("calibration_sha256"), "calibration digest")
    require(binding.get("role") == "open_calibration", "sealed/noncalibration fitting data")
    rows = binding.get("rows", [])
    require(bool(rows), "empty calibration")
    require(binding["calibration_sha256"] == digest(rows), "calibration row digest differs")
    seen, groups = set(), {"fit": set(), "map_check": set()}
    for row in rows:
        rid, parent, split = row.get("id"), row.get("parent_id"), row.get("split")
        require(isinstance(rid, str) and rid and rid not in seen, "duplicate/missing calibration ID")
        require(isinstance(parent, str) and parent, "missing calibration parent")
        require(split in groups and row.get("open") is True, "sealed/invalid row role")
        require(rid not in forbidden_row_ids and parent not in forbidden_parent_ids,
                "sealed evaluation example or parent used for transport")
        seen.add(rid)
        groups[split].add(parent)
    require(all(groups.values()), "fit and heldout map_check both required")
    require(not groups["fit"] & groups["map_check"], "fit/map_check parent overlap")
    return groups


CAPTURE_KEYS = ("model_sha256", "tokenizer_sha256", "dataset_sha256", "calibration_sha256",
                "layer_map_sha256", "prompt_positions_sha256", "implementation_digest",
                "runtime_binding_sha256", "anchor_checkpoint_sha256")


def validate_capture(capture, expected):
    for key in CAPTURE_KEYS:
        valid_hash(expected.get(key), key)
        require(capture.get(key) == expected[key], "capture " + key + " differs")
    require(capture.get("role") in ("student_base", "donor_base", "donor_skilled"), "capture role")
    require(capture.get("role") == expected.get("role"), "capture checkpoint role differs")
    if capture["role"] in ("student_base", "donor_base"):
        require(capture["anchor_checkpoint_sha256"] == capture["model_sha256"], "base capture anchor differs")
    require(capture.get("layer_ids") == expected.get("layer_ids") and bool(capture.get("layer_ids")),
            "wrong capture layer IDs")
    require(capture.get("pooling") == "last_nonpad_prompt_token", "wrong token pooling")
    require(capture.get("eval_mode") is True and capture.get("optimization_steps") == 0,
            "capture was not frozen eval-only")
    require(capture.get("dtype") == "float32" and capture.get("padding_excluded") is True,
            "capture precision/padding protocol")


COMMON_KEYS = ("views_sha256", "implementation_digest", "dataset_sha256", "tokenizer_sha256",
               "model_binding_sha256", "base_checkpoint_sha256", "training_rows_sha256",
               "parent_partition_sha256", "evaluation_bank_sha256")
ARM_KEYS = ("skill_delta_sha256", "probe_weight_sha256", "donor_checkpoint_sha256",
            "layer_map_sha256", "calibration_sha256", "basis_sha256", "basis_run_id",
            "rank_budget", "parameter_budget", "optimizer_steps", "training_run_id", "alpha")
COMPOSITION_KEYS = ("skill_id", "independent_training", "target_modules")


def validate_modules(modules):
    require(isinstance(modules, list) and bool(modules)
            and all(isinstance(name, str) and name for name in modules)
            and modules == sorted(set(modules)), "target modules must be sorted, unique and nonempty")


def validate_arm(metadata, expected):
    """Compare each arm to its own frozen binding; unequal deltas are expected."""
    for key in COMMON_KEYS + ARM_KEYS:
        require(key in expected and expected[key] is not None, "registration missing " + key)
        if key.endswith("sha256") or key == "implementation_digest":
            valid_hash(expected[key], key)
        require(metadata.get(key) == expected[key], "arm " + key + " differs")
    require(metadata.get("completed") is True, "arm incomplete")
    for key in ("rank_budget", "parameter_budget"):
        require(type(metadata[key]) is int and metadata[key] > 0, "invalid arm " + key)
    require(type(metadata["optimizer_steps"]) is int and metadata["optimizer_steps"] >= 0,
            "invalid optimizer steps")
    require(type(metadata["alpha"]) in (int, float) and math.isfinite(metadata["alpha"])
            and metadata["alpha"] >= 0, "invalid interpolation alpha")
    for key in ("training_run_id", "basis_run_id"):
        require(isinstance(metadata[key], str) and metadata[key], "missing arm " + key)
    return True


def validate_norm_control(actual, control, *, relative_tolerance=1e-8):
    require(0 <= relative_tolerance <= 1e-6, "norm tolerance was weakened")
    for key in ("rank_budget", "parameter_budget", "target_modules", "optimizer_steps", "training_rows_sha256"):
        require(key in actual and key in control and actual[key] == control[key], "norm control " + key)
    for item in (actual, control):
        require(type(item.get("norm")) in (int, float) and math.isfinite(item["norm"]) and item["norm"] > 0,
                "invalid norm control norm")
        validate_modules(item["target_modules"])
        valid_hash(item["training_rows_sha256"], "norm control training rows")
        require(all(type(item[key]) is int and item[key] > 0 for key in ("rank_budget", "parameter_budget")),
                "invalid norm control budget")
        require(type(item["optimizer_steps"]) is int and item["optimizer_steps"] >= 0,
                "invalid norm control optimizer steps")
    require(math.isclose(actual["norm"], control["norm"], rel_tol=relative_tolerance, abs_tol=0),
            "norm control mismatch")
    require(actual.get("effective_ranks") == control.get("effective_ranks")
            and isinstance(actual.get("effective_ranks"), dict) and bool(actual["effective_ranks"]),
            "effective rank mismatch (projection rank growth needs separate control)")
    require(set(actual["effective_ranks"]) == set(actual["target_modules"])
            and all(type(rank) is int and rank >= 0 for rank in actual["effective_ranks"].values()),
            "effective rank scope/values differ")


def validate_composition(left, right):
    for key in ("base_checkpoint_sha256", "tokenizer_sha256", "model_binding_sha256", "target_modules"):
        require(key in left and key in right and left[key] == right[key], "incompatible composition " + key)
    for item in (left, right):
        valid_hash(item.get("skill_delta_sha256"), "composition delta")
        validate_modules(item.get("target_modules"))
        require(item.get("independent_training") is True, "composition skill not independently trained")
    require(left.get("training_run_id") and right.get("training_run_id")
            and left["training_run_id"] != right["training_run_id"], "composition uses same training draw")
    require(left.get("skill_id") and right.get("skill_id") and left["skill_id"] != right["skill_id"],
            "composition requires distinct skills")


def validate_basis(basis, expected_run_id, expected_digest):
    valid_hash(expected_digest, "protected basis")
    require(basis.get("sha256") == expected_digest and basis.get("run_id") == expected_run_id,
            "projection basis from another run")
    require(basis.get("source_kind") in ("independent_skill_delta", "retention_sensitive_delta", "retention_activation"),
            "unsupported protected basis source")
    require(type(basis.get("rank")) is int and basis["rank"] > 0, "invalid protected rank")
    require(bool(basis.get("scope")) and bool(basis.get("normalization")), "incomplete basis scope")


def complete_child(returncode, child):
    require(returncode == 0 and child.get("status") == "COMPLETE" and child.get("completed") is True,
            "wrapper cannot report COMPLETE after failed child")
    return {"status": "COMPLETE", "child_verified": True}


def paired_report(left, right, *, cross_training_draw):
    """left minus right; exact two-sided discordance test, no scalar-score inference."""
    require(bool(left) and set(left) == set(right), "paired row IDs differ; intersections forbidden")
    require(all(type(v) is bool for v in [*left.values(), *right.values()]), "nonboolean endpoint")
    b = sum(left[k] and not right[k] for k in left)
    c = sum(right[k] and not left[k] for k in left)
    nd, net = b + c, b - c
    p = min(Fraction(1), Fraction(2 * sum(math.comb(nd, k) for k in range(min(b, c) + 1)), 2**nd))
    return {"net": net, "nd": nd, "exact_p": str(p), "p": float(p), "rows": len(left),
            "cross_training_draw": cross_training_draw,
            "cross_draw_floor_cleared": abs(net) >= 20 if cross_training_draw else None,
            "directional_claim_floor_met": not cross_training_draw or abs(net) >= 20}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def compare_endpoint_files(left, right, left_expected, right_expected, *, cross_training_draw, source_root):
    """Gate a frozen output receipt and population BEFORE reading endpoint values.

    Expected bindings come from the qualified run/output registry, not payload
    sidecars. Each additionally pins endpoint_sha256, endpoint_row_ids, their
    digest, and source_commit. These are trusted custody inputs, not signatures.
    """
    validate_arm(left["metadata"], left_expected)
    validate_arm(right["metadata"], right_expected)
    for key in COMMON_KEYS:
        require(left["metadata"][key] == right["metadata"][key], "comparison " + key + " differs")
    require(left_expected.get("source_commit") == right_expected.get("source_commit"),
            "comparison source commits differ")
    verify_source(source_root, left_expected.get("source_commit"), clean=True)
    derived_cross_draw = left["metadata"]["training_run_id"] != right["metadata"]["training_run_id"]
    require(type(cross_training_draw) is bool and cross_training_draw == derived_cross_draw,
            "cross-training-draw classification differs from registered training ancestry")
    for item, expected in ((left, left_expected), (right, right_expected)):
        valid_hash(expected.get("endpoint_sha256"), "registered endpoint payload")
        require(item.get("sha256") == expected["endpoint_sha256"], "endpoint digest is not registered")
        rows = expected.get("endpoint_row_ids")
        require(isinstance(rows, list) and bool(rows) and all(isinstance(row, str) and row for row in rows)
                and len(rows) == len(set(rows)), "invalid registered endpoint population")
        valid_hash(expected.get("endpoint_row_ids_sha256"), "registered endpoint row IDs")
        require(digest(rows) == expected["endpoint_row_ids_sha256"], "endpoint population digest differs")
    require(left_expected["endpoint_row_ids"] == right_expected["endpoint_row_ids"],
            "registered endpoint populations/order differ")
    for item, expected in ((left, left_expected), (right, right_expected)):
        verify_file(item["path"], expected["endpoint_sha256"])
    outputs = []
    for item, expected in ((left, left_expected), (right, right_expected)):
        value = json.loads(Path(item["path"]).read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
        require(isinstance(value, dict) and list(value) == expected["endpoint_row_ids"],
                "endpoint rows/order differ from frozen population")
        outputs.append(value)
    return paired_report(*outputs, cross_training_draw=derived_cross_draw)


def composition_losses(a_alone, b_alone, a_composed, b_composed):
    """Prospective loss in percentage points; negative means gain in composition."""
    for alone, composed in ((a_alone, a_composed), (b_alone, b_composed)):
        paired_report(alone, composed, cross_training_draw=False)
    score = lambda values: 100 * sum(values.values()) / len(values)
    return {"I_A_pp": score(a_alone) - score(a_composed),
            "I_B_pp": score(b_alone) - score(b_composed),
            "A_paired": paired_report(a_alone, a_composed, cross_training_draw=False),
            "B_paired": paired_report(b_alone, b_composed, cross_training_draw=False)}
