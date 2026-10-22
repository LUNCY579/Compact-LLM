"""Bounded stdlib delta-control mechanics; no model, checkpoint reader or dispatch.

The fixture arithmetic does not qualify an FP32 artifact or establish model gain.
"""
from __future__ import annotations

import hashlib
import json
import math

SPARSE_PARENT_SHA256 = "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284"
MAX_ELEMENTS_PER_TENSOR = 10_000
MAX_TOTAL_ELEMENTS = 40_000
ALLOWED_FRAMES = frozenset(("toy", "new_relative_delta"))
ARM_NAMES = ("dense_new_delta", "top25_new_delta", "per_tensor_norm_matched_new_delta")
NORM_REL_TOLERANCE = 1e-12
NORM_ABS_TOLERANCE = 1e-12


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_vectors(tensors):
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("a nonempty named dictionary of tiny fixture vectors is required")
    total = 0
    validated = {}
    for name, values in tensors.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tensor names must be nonempty strings")
        if "lora_a" in name.lower() or "lora_b" in name.lower():
            raise ValueError("raw LoRA factors are not effective parameter deltas")
        if not isinstance(values, (list, tuple)):
            raise ValueError("only explicit stdlib fixture vectors are accepted")
        length = len(values)
        if length <= 0 or length > MAX_ELEMENTS_PER_TENSOR or length % 4:
            raise ValueError("tensor length must be positive, divisible by four and <=10000")
        total += length
        if total > MAX_TOTAL_ELEMENTS:
            raise ValueError("total fixture element cap exceeded")
        copied = []
        for value in values:
            if type(value) not in (int, float):
                raise ValueError("fixture elements must be finite float/int values, never bool")
            try:
                finite = math.isfinite(float(value))
            except (OverflowError, ValueError):
                finite = False
            if not finite:
                raise ValueError("fixture element is not finite in the declared scalar arithmetic")
            copied.append(value)
        validated[name] = copied
    return validated, total


def _norm(values):
    value = math.hypot(*(float(element) for element in values))
    if not math.isfinite(value):
        raise ValueError("finite fixture norm cannot be established")
    return value


def build_controls(tensors, *, frame, parent_sha256=None):
    """Return exactly three transforms of an explicit toy/new-relative delta.

    ``new_relative_delta`` is a prospective caller declaration, not independently
    verified model provenance. It must name the exact immediate Titan-Sparse
    parent. There is no file/tensor loader or native artifact constructor here.
    """
    if not isinstance(frame, str) or frame not in ALLOWED_FRAMES:
        raise ValueError("absolute checkpoints, source whole weights and adapter matrices are not delta frames")
    if frame == "new_relative_delta" and parent_sha256 != SPARSE_PARENT_SHA256:
        raise ValueError("prospective new delta must be relative to the exact Titan-Sparse parent")
    if frame == "toy" and parent_sha256 not in (None, SPARSE_PARENT_SHA256):
        raise ValueError("a toy fixture cannot substitute a different prospective parent")
    vectors, total = _validate_vectors(tensors)
    source_before = _digest(tensors)
    dense, top25, norm_matched, tensor_receipts = {}, {}, {}, {}
    for name in sorted(vectors):
        vector = vectors[name]
        keep = len(vector) // 4
        # Rank the ORIGINAL numbers: large integer distinctions must not be
        # erased by a preliminary float conversion. Equal magnitudes use the
        # smallest flat coordinate first, including positive/negative ties.
        selected = sorted(range(len(vector)), key=lambda index: (-abs(vector[index]), index))[:keep]
        support = set(selected)
        pruned = [value if index in support else 0 for index, value in enumerate(vector)]
        full_norm, pruned_norm = _norm(vector), _norm(pruned)
        alpha = 0.0 if full_norm == 0.0 else pruned_norm / full_norm
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("per-tensor norm-matching coefficient is invalid")
        scaled = [alpha * value for value in vector]
        if not all(math.isfinite(float(value)) for value in scaled):
            raise ValueError("uniform norm-matched fixture transform is nonfinite")
        scaled_norm = _norm(scaled)
        equal = math.isclose(scaled_norm, pruned_norm,
                            rel_tol=NORM_REL_TOLERANCE, abs_tol=NORM_ABS_TOLERANCE)
        if not equal:
            raise ValueError("fixture norm equality failed; no native tolerance is inferred")
        dense[name] = list(vector)
        top25[name] = pruned
        norm_matched[name] = scaled
        tensor_receipts[name] = {"elements": len(vector), "selected_elements": keep,
            "selected_flat_indices": selected, "dense_norm": full_norm,
            "top25_norm": pruned_norm, "uniform_norm": scaled_norm, "alpha_t": alpha,
            "zero_tensor": full_norm == 0.0, "matched_norm_verified_on_fixture": equal}
    source_after = _digest(tensors)
    if source_before != source_after:
        raise ValueError("input fixture vectors changed during construction")
    arms = {ARM_NAMES[0]: dense, ARM_NAMES[1]: top25, ARM_NAMES[2]: norm_matched}
    return {"status": "MECHANICAL_NO_MODEL_GAIN", "frame": frame,
            "parent_sha256": parent_sha256, "source_frame_independently_verified": False,
            "fixture_scalar_arithmetic": "Python finite scalar arithmetic; not native FP32 artifact qualification",
            "native_authorized": False, "training_authorized": False,
            "model_calls": 0, "optimizer_steps": 0, "model_quality_gain": False,
            "max_elements_per_tensor": MAX_ELEMENTS_PER_TENSOR,
            "max_total_elements": MAX_TOTAL_ELEMENTS, "total_elements": total,
            "source_before_sha256": source_before, "source_after_sha256": source_after,
            "source_unchanged": True, "arms": arms, "arm_names": list(ARM_NAMES),
            "tensor_receipts": tensor_receipts,
            "arm_sha256": {name: _digest(value) for name, value in arms.items()},
            "tie_rule": "descending absolute value, then ascending flat index",
            "norm_fixture_tolerance": {"relative": NORM_REL_TOLERANCE, "absolute": NORM_ABS_TOLERANCE},
            "native_future_proposal": prospective_recipe()}


def prospective_recipe():
    """Describe a three-arm owner-reviewed question, never an executable grant."""
    return {"status": "PROSPECTIVE_UNQUALIFIED_RECIPE_ONLY",
        "immediate_parent_sha256": SPARSE_PARENT_SHA256,
        "required_input": "new joint absolute trained checkpoint plus its exact immediate Titan-Sparse parent",
        "conversion": "new_delta = new_joint_absolute_checkpoint - immediate_Titan_Sparse_parent, exactly once",
        "construction": "each arm = immediate_Titan_Sparse_parent + one transform of new_delta",
        "inherited_delta_rule": "Titan-Sparse already contains its inherited update; never add that update a second time",
        "prohibited_inputs": ["source whole weights labeled as a delta", "raw LoRA A/B matrices",
                              "another parent's delta", "BF16 copy treated as the FP32 source"],
        "operator_envelope": {"arms": list(ARM_NAMES), "arm_count": 3,
                              "free_parameter_sweep": False, "extra_training": False},
        "remaining_owner_reviewed_gates": [
            "exact new checkpoint and immediate-parent source/ancestry/representation proof",
            "native runtime and construction precision qualification without re-masking inherited support",
            "matched exposure, optimizer/update, teacher and evaluation contracts",
            "complete measured cost including failures, all controls, finalization and resources",
            "prospective source-correct retention, answerability, content/format and statistical gates",
            "independent training-draw replication and new evaluation population where required"],
        "native_command": None, "native_authorized": False, "training_authorized": False,
        "protected_data_authorized": False, "model_quality_gain_established": False}
