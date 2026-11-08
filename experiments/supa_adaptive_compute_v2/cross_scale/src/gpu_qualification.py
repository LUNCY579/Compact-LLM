"""Prospective Track B FP32 runtime comparisons on prompt-only calibration data."""
from __future__ import annotations

import torch
import math
import re
from torch import nn
from torch.nn import functional as F

from shared.hashing import canonical_sha256
from shared.legacy import legacy_graft
from .delta_alignment import donor_delta, finite, frozen, normalized_loss


def qualification_fixtures(rows, config):
    rule = config["qualification"]
    if (rule["fixture_count"] != 8 or rule["fixture_rule"] !=
            "first_sorted_per_family_alternating_fit_holdout"):
        raise ValueError("unregistered cross-scale GPU fixture rule")
    families = sorted({row["family"] for row in rows})
    if len(families) < 8:
        raise ValueError("GPU qualification requires eight calibration families")
    selected = []
    for index, family in enumerate(families[:8]):
        partition = "fit" if index % 2 == 0 else "holdout"
        candidates = sorted((row for row in rows if row["family"] == family and
            row["alignment_partition"] == partition), key=lambda row: row["example_id"])
        if not candidates:
            raise ValueError("calibration fixture family lacks its registered partition")
        selected.append(candidates[0])
    if len({row["scenario_id"] for row in selected}) != 8:
        raise ValueError("qualification fixtures must be distinct scenarios")
    return selected


def tensor_agreement(reference, actual, thresholds):
    """FP64 CPU diagnostic of FP32 tensors; zero reference handled explicitly."""
    if reference.shape != actual.shape or not reference.numel():
        raise ValueError("qualification comparison shape mismatch")
    reference = finite(reference.detach().cpu().double())
    actual = finite(actual.detach().cpu().double())
    error = actual-reference
    target_norm = reference.norm()
    relative = float(error.norm()/target_norm.clamp_min(1e-12))
    absolute = float(error.abs().max())
    # Bias/gradient vectors are one row; activations retain their token rows.
    left, right = reference.reshape(-1, reference.shape[-1]), actual.reshape(-1, actual.shape[-1])
    nonzero = (left.norm(dim=-1) > 1e-12) | (right.norm(dim=-1) > 1e-12)
    cosine = float(F.cosine_similarity(left[nonzero], right[nonzero], dim=-1).mean()) if nonzero.any() else 1.0
    return {"shape": list(reference.shape), "relative_l2": relative,
            "maximum_absolute_error": absolute, "mean_token_cosine": cosine,
            "pass": relative <= thresholds["maximum_relative_l2"] and
                absolute <= thresholds["maximum_absolute_error"] and
                cosine >= thresholds["minimum_mean_token_cosine"]}


def bridge_gradients(initial, layer, rope, prompt, config, token_normalized):
    """One real full-prompt backward, never a training run or efficacy selection."""
    device = next(layer.parameters()).device
    frozen(layer); frozen(rope)
    up = nn.Parameter(initial.up.detach().to(device).clone())
    bias = nn.Parameter(initial.up_bias.detach().to(device).clone())
    small, big, target = (prompt[key].detach().to(device).unsqueeze(0)
                          for key in ("small", "big", "delta"))
    projected = legacy_graft().normalize(small)@up+bias
    delta = donor_delta(layer, rope, projected, config=config)
    loss = (normalized_loss(delta, target, token_normalized=token_normalized) +
            config["state_anchor_weight"]*normalized_loss(projected, big, token_normalized=token_normalized) +
            config["delta_cosine_weight"]*(1-F.cosine_similarity(delta, target, dim=-1)).mean())
    finite(loss).backward()
    frozen(layer); frozen(rope)
    return {"up": finite(up.grad).detach().cpu().clone(),
            "bias": finite(bias.grad).detach().cpu().clone(),
            "loss": float(loss.detach())}


def fixture_manifest(rows):
    return {"sha256": canonical_sha256(rows), "rows": rows,
            "answer_fields_consumed": False, "generation_count": 0}


def validate_qualification_report(report, config, calibration_rows):
    """Recompute coverage and numerical admission independently of saved pass flags."""
    from .donor_layers import strip_calibration_rows, DONOR_RECORD
    from shared.hashing import sha256_file
    import json
    selected = qualification_fixtures(calibration_rows, config)
    expected_rows = [{**row, "alignment_partition": selected[index]["alignment_partition"]}
                     for index, row in enumerate(strip_calibration_rows(selected))]
    if report.get("fixtures") != fixture_manifest(expected_rows):
        raise ValueError("qualification fixture identity, partition or prompt differs from frozen calibration")
    thresholds = config["qualification"]
    if report.get("thresholds") != thresholds:
        raise ValueError("qualification thresholds differ from registration")
    if (report.get("qualification_pass") is not True or report.get("generation_count") != 0 or
            any(report.get(key) is not True for key in (
                "same_gpu_donor_delta_repeat_bit_identical", "same_gpu_streamed_capture_repeat_bit_identical",
                "same_gpu_bridge_gradient_repeat_bit_identical"))):
        raise ValueError("qualification lacks complete passing repeat evidence")
    strata = ("all_tokens", "ordinary_tokens", "special_tokens", "final_prompt_token")
    expected = set()
    for name, depths in (("supa", config["supa_layers"]), ("donor", sorted({
            depth for index in config["donor_layers"] for depth in (index-1, index)}))):
        for depth in depths:
            for row in selected:
                expected.update((f"{name}_L{depth}_{row['example_id']}", stratum) for stratum in strata)
    for supa_index in config["supa_layers"]:
        for donor_index in config["donor_layers"]:
            for row in selected:
                if row["alignment_partition"] == "holdout":
                    expected.update((f"S{supa_index}_D{donor_index}_B0_delta_{row['example_id']}", stratum)
                                    for stratum in strata)
    for donor_index in config["donor_layers"]:
        for objective in thresholds["bridge_gradient_objectives"]:
            for key in ("up", "bias"):
                expected.add((f"S{thresholds['gradient_supa_layer']}_D{donor_index}_{objective}_gradient_{key}", "all"))
    actual = [(row.get("label"), row.get("stratum")) for row in report.get("comparisons", [])]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("qualification comparison coverage is incomplete, duplicated or unregistered")
    for row in report["comparisons"]:
        if row.get("tokens") == 0:
            if row["stratum"] != "special_tokens" or set(row) != {"label", "stratum", "tokens", "pass"} or row["pass"] is not True:
                raise ValueError("only explicitly empty special-token strata may omit metrics")
            continue
        shape = row.get("shape", [])
        if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError("qualification matrix shape is invalid")
        expected_width = (1024*2560 if row["label"].endswith("gradient_up") else
                          1024 if row["label"].startswith("supa_") else 2560)
        if shape[1] != expected_width:
            raise ValueError("qualification residual/bridge dimensions differ from registered backbones")
        values = [row.get(key) for key in ("relative_l2", "maximum_absolute_error", "mean_token_cosine")]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("qualification metric is nonfinite or missing")
        relative, absolute, cosine = values
        if (not 0 <= relative <= thresholds["maximum_relative_l2"] or
                not 0 <= absolute <= thresholds["maximum_absolute_error"] or
                not thresholds["minimum_mean_token_cosine"] <= cosine <= 1.000001 or row.get("pass") is not True):
            raise ValueError("qualification numerical values do not satisfy registered admission")
        if row["stratum"] in ("final_prompt_token", "all") and shape[0] != 1:
            raise ValueError("qualification final-token/gradient row shape is invalid")
    indexed = {(row["label"], row["stratum"]): row for row in report["comparisons"]}
    for label, stratum in expected:
        if stratum != "all_tokens":
            continue
        all_count = indexed[label, "all_tokens"]["shape"][0]
        ordinary_count = indexed[label, "ordinary_tokens"]["shape"][0]
        special = indexed[label, "special_tokens"]
        special_count = 0 if special.get("tokens") == 0 else special["shape"][0]
        widths = {indexed[label, name]["shape"][1] for name in strata if "shape" in indexed[label, name]}
        if all_count != ordinary_count+special_count or len(widths) != 1:
            raise ValueError("qualification token strata do not partition one residual stream")
    integrity = report.get("layer_integrity", [])
    if (len(integrity) != len(config["donor_layers"]) or
            {row.get("donor_layer") for row in integrity} != set(config["donor_layers"])):
        raise ValueError("qualification donor freeze coverage incomplete")
    for row in integrity:
        if (row.get("CPU_GPU_parameter_bytes_equal") is not True or row.get("frozen_unchanged") is not True or
                row.get("backbone_gradients") != 0 or row.get("trainable_backbone_parameters") != 0 or
                any(not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key])
                    for key in ("state_sha256", "cpu_rope_sha256", "gpu_rope_sha256"))):
            raise ValueError("qualification backbone freeze/hash evidence invalid")
    if report.get("supa_files", {}).get("model.safetensors") != config["checkpoint_sha256"]:
        raise ValueError("qualification used another Supa checkpoint")
    identity = report.get("donor_identity", {})
    record = json.loads(DONOR_RECORD.read_text(encoding="utf-8"))
    weights = record["weights"] if isinstance(record["weights"], list) else record["weights"]["files"]
    expected_files = {row["name"]: row["sha256"] for row in weights}
    expected_files["config.json"] = record["config"]["sha256"]
    if (identity.get("revision") != config["donor_revision"] or
            identity.get("record_sha256") != sha256_file(DONOR_RECORD) or identity.get("files") != expected_files):
        raise ValueError("qualification used another donor checkpoint")
    return True
