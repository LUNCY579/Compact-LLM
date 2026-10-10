"""Prospective qualification helpers; no loader, optimizer, or CLI wiring.

Calling these helpers on a fixture establishes fixture mechanics only. Native
use requires separate approval, the exact eligible source, and a frozen packet.
Raw-source parity records must be captured BEFORE selected-parameter casting,
with the same pinned precision/attention policy. Exact full-logit equality is
required; a failure calls for a new prospective design, not a relaxed tolerance.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import time
import torch

from .data import score
from .objectives import reference_nll, response_logits
from .plan import arms, checkpoint_manifest, digest_json, hash_file
from .teacher import tensor_digest


def _gate_rows(rows, task, adaptation_rows=()):
    """Exactly eight fresh TRAIN gate families, separate from optimization."""
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("Freeze exactly eight teacher-gate TRAIN rows")
    ids, parents, prompts = set(), set(), set()
    for row in rows:
        if (row.get("split"), row.get("admitted_for"), row.get("purpose"), row.get("task")) != (
                "train", "train", "acquisition", task):
            raise ValueError("Teacher gate permits only admitted TRAIN acquisition rows")
        if not all(isinstance(row.get(k), str) and row[k] for k in (
                "id", "parent_id", "prompt", "reference", "answer", "scorer")):
            raise ValueError("Incomplete teacher-gate row")
        prompt = " ".join(row["prompt"].split())
        if row["id"] in ids or row["parent_id"] in parents or prompt in prompts:
            raise ValueError("Teacher-gate rows need eight distinct families and prompts")
        ids.add(row["id"])
        parents.add(row["parent_id"])
        prompts.add(prompt)
        if not score(row["reference"], row["answer"], row["scorer"]):
            raise ValueError("Teacher-gate reference fails its verified scorer")
    for row in adaptation_rows:
        if (row.get("id") in ids or row.get("parent_id") in parents
                or " ".join(row.get("prompt", "").split()) in prompts):
            raise ValueError("Teacher-gate families cannot enter optimization/evaluation")
    return rows


@contextmanager
def _no_update_scope(engine, grad=False):
    """Restore selected parameters, buffers and banks, including on exceptions.

    Frozen parameters are hashed, never duplicated into an extra 0.6B bank. An
    illegal frozen-parameter mutation fails qualification and requires unloading
    that in-memory model; the immutable checkpoint is never written.
    """
    if engine.model.training:
        raise ValueError("Qualification requires eval mode with dropout disabled")
    if engine.bank._busy:
        raise RuntimeError("Qualification cannot overlap a graph or teacher swap")
    parameter_digest = tensor_digest(engine.model.named_parameters())
    buffer_digest = tensor_digest(engine.model.named_buffers())
    selected = engine.bank.snapshot()
    buffers = {n: b.detach().cpu().clone() for n, b in engine.model.named_buffers()}
    initial = {n: p.clone() for n, p in engine.bank.initial.items()}
    teacher = {n: p.clone() for n, p in engine.bank.teacher.items()}
    bank_digest = (tensor_digest(initial.items()), tensor_digest(teacher.items()))
    error = None
    try:
        with torch.enable_grad() if grad else torch.no_grad():
            yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        mutated = (parameter_digest != tensor_digest(engine.model.named_parameters())
                   or buffer_digest != tensor_digest(engine.model.named_buffers())
                   or bank_digest != (tensor_digest(engine.bank.initial.items()),
                                      tensor_digest(engine.bank.teacher.items())))
        with torch.no_grad():
            for name, parameter in engine.bank.selected.items():
                parameter.copy_(selected[name])
            for name, buffer in engine.model.named_buffers():
                buffer.copy_(buffers[name])
        engine.bank.initial = initial
        engine.bank.teacher = teacher
        if mutated:
            reason = "No-update qualification observed parameter/buffer/bank mutation"
            if parameter_digest != tensor_digest(engine.model.named_parameters()):
                reason += "; frozen parameters changed: discard the loaded model"
            if error is not None:
                error.add_note(reason)
            else:
                raise RuntimeError(reason)


def prepare_parity_probes(engine, gate_rows):
    """Prepare clean original prompts and a full-budget synthetic no-update probe.

    IDs padding extends the first gate prompt using EOS, never reference text or
    DEV. This is an allocation/dispatch probe, not a capability example.
    """
    _gate_rows(gate_rows, engine.config["task"])
    probes = []
    for row in gate_rows[:2]:
        ids = engine.ids(row, False)
        if len(ids) > engine.config["max_sequence_tokens"]:
            raise ValueError("Parity prompt exceeds frozen context budget")
        probes.append({"id": "clean:" + row["id"], "kind": "clean_prompt",
                       "input_ids": ids, "input_sha256": digest_json(ids), "privileged": False})
    ids = list(probes[0]["input_ids"])
    ids.extend([engine.eos] * (engine.config["max_sequence_tokens"] - len(ids)))
    probes.append({"id": "max_shape", "kind": "max_shape_no_update", "input_ids": ids,
                   "input_sha256": digest_json(ids), "privileged": False})
    return probes


def capture_unprivileged_parity(engine, probes):
    """Capture exact full-vocabulary logit hashes without any optimizer object.

    Capture once on the raw source, then again after selected FP32 casting under
    the same precision policy. Native resource guards must surround both calls.
    """
    if not isinstance(probes, list) or not probes or len({p["id"] for p in probes}) != len(probes):
        raise ValueError("Nonempty unique parity probes required")
    records = []
    with _no_update_scope(engine):
        for probe in probes:
            ids = probe["input_ids"]
            if probe.get("privileged") is not False or probe.get("input_sha256") != digest_json(ids):
                raise ValueError("Only hash-pinned unprivileged parity inputs are allowed")
            logits = engine.forward(ids)
            if (list(logits.shape) != [1, len(ids), engine.model.config.vocab_size]
                    or not torch.isfinite(logits).all()):
                raise RuntimeError("Parity requires finite full-vocabulary logits")
            records.append({"id": probe["id"], "kind": probe["kind"],
                            "input_sha256": probe["input_sha256"], "shape": list(logits.shape),
                            "logits_sha256": tensor_digest([("logits", logits)])})
    return {"records": records, "all_unprivileged": True, "parameters_unchanged": True,
            "buffers_unchanged": True, "optimizer_objects_created": 0, "training_updates": 0}


def capture_raw_source_parity(model, config, probes):
    """Capture the raw BF16 source BEFORE FP32 selection, without TeacherBank.

    The model must already be locally loaded and in eval mode. This helper never
    loads a checkpoint, creates a bank/optimizer, or changes parameter dtypes.
    Caller supplies the same pinned fixed_precision scope and resource floors
    used by the subsequent Engine parity capture.
    """
    if model.training or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Raw-source parity requires eval mode and clear gradients")
    if not isinstance(probes, list) or not probes or len({p["id"] for p in probes}) != len(probes):
        raise ValueError("Nonempty unique raw-source parity probes required")
    device = torch.device(config["device"])
    before = tensor_digest(model.named_parameters())
    buffer_before = tensor_digest(model.named_buffers())
    buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    records, error = [], None
    try:
        with torch.no_grad():
            for probe in probes:
                ids = probe["input_ids"]
                if (probe.get("privileged") is not False or probe.get("input_sha256") != digest_json(ids)
                        or not ids or len(ids) > config["max_sequence_tokens"]
                        or any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in ids)):
                    raise ValueError("Only bounded hash-pinned unprivileged raw-source inputs are allowed")
                inputs = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    logits = model(input_ids=inputs, use_cache=False, return_dict=True).logits
                if (list(logits.shape) != [1, len(ids), model.config.vocab_size]
                        or not torch.isfinite(logits).all()):
                    raise RuntimeError("Raw-source parity requires finite full-vocabulary logits")
                records.append({"id": probe["id"], "kind": probe["kind"],
                                "input_sha256": probe["input_sha256"], "shape": list(logits.shape),
                                "logits_sha256": tensor_digest([("logits", logits)])})
    except BaseException as exc:
        error = exc
        raise
    finally:
        mutated = before != tensor_digest(model.named_parameters()) or buffer_before != tensor_digest(model.named_buffers())
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
        if mutated:
            reason = "Raw-source parity observed mutation; discard loaded model if parameters changed"
            if error is not None:
                error.add_note(reason)
            else:
                raise RuntimeError(reason)
    return {"records": records, "all_unprivileged": True, "parameters_unchanged": True,
            "buffers_unchanged": True, "raw_parameters_sha256": before,
            "optimizer_objects_created": 0, "training_updates": 0}


def check_loaded_source_parity(engine, probes, raw_source_report):
    """Require exact raw-source/selected-cast parity, including the maximum shape."""
    if not all(raw_source_report.get(k) is True for k in (
            "all_unprivileged", "parameters_unchanged", "buffers_unchanged")):
        raise ValueError("Raw source lacks unprivileged no-update parity evidence")
    current = capture_unprivileged_parity(engine, probes)
    if current["records"] != raw_source_report.get("records"):
        raise ValueError("Raw-source full logits changed after selected casting/load")
    if not any(r["kind"] == "max_shape_no_update"
               and r["shape"][1] == engine.config["max_sequence_tokens"] for r in current["records"]):
        raise ValueError("Missing frozen maximum-shape no-update parity check")
    return dict(current, source_inference_parity=True, max_shape_no_update=True,
                equality="exact_full_logits", native_source_eligibility_established=False)


def maximum_shape_backward(engine, probes):
    """Exercise full-budget SFT backward with final <=32 response tokens; no step.

    These are synthetic EOS-padding labels, not capability or acquisition data.
    The first-row five-arm qualification remains separately required. CUDA
    floors/stage capture must surround this helper in future approved orchestration.
    """
    candidates = [p for p in probes if p.get("kind") == "max_shape_no_update"]
    if len(candidates) != 1:
        raise ValueError("One full-budget unprivileged maximum-shape probe required")
    probe = candidates[0]
    ids = probe["input_ids"]
    if (probe.get("privileged") is not False or probe.get("input_sha256") != digest_json(ids)
            or len(ids) != engine.config["max_sequence_tokens"]):
        raise ValueError("Maximum-shape backward must use the exact frozen context budget")
    response_length = min(32, engine.config["max_new_tokens"])
    if len(ids) <= response_length:
        raise ValueError("Maximum-shape backward needs nonempty clean prefix")
    engine.model.zero_grad(set_to_none=True)
    try:
        with _no_update_scope(engine, grad=True):
            with engine.bank.student_graph():
                logits = response_logits(engine.forward(ids), len(ids) - response_length, response_length)
                targets = torch.tensor([ids[-response_length:]], dtype=torch.long, device=engine.device)
                loss = reference_nll(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite maximum-shape objective")
                loss.backward()
            grads = [p.grad for p in engine.bank.selected.values()]
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise RuntimeError("Missing/nonfinite maximum-shape selected gradient")
            norm = math.sqrt(sum(float(g.double().square().sum()) for g in grads))
            result = {"maximum_shape_tokens": len(ids), "response_tokens": response_length,
                      "vocabulary_size": engine.model.config.vocab_size,
                      "loss": float(loss.detach()), "gradient_l2": norm,
                      "parameters_unchanged": True, "buffers_unchanged": True,
                      "max_shape_backward_no_update": True, "training_updates": 0,
                      "optimizer_objects_created": 0, "evidence": "MECHANICAL_SYNTHETIC_SHAPE_ONLY"}
    finally:
        engine.model.zero_grad(set_to_none=True)
    result["gradients_cleared"] = all(p.grad is None for p in engine.model.parameters())
    return result


def qualify_teacher_competence(engine, gate_rows, adaptation_rows=()):
    """Prospective initial teacher gate on eight TRAIN-only rows.

    Compare privileged teacher and clean source student greedily on the same
    frozen rows. Pass means >=7/8 teacher successes, >=2 paired wins, zero paired
    harms. No DEV, protected bank, optimizer, parameter update or retuning occurs.
    These rows must be excluded from optimization and later efficacy cohorts.
    """
    _gate_rows(gate_rows, engine.config["task"], adaptation_rows)
    if tensor_digest(engine.bank.initial.items()) != tensor_digest(engine.bank.teacher.items()):
        raise ValueError("Prospective teacher gate must start at the initial source")
    for row in gate_rows:
        for privileged in (False, True):
            if len(engine.ids(row, privileged)) + engine.config["max_new_tokens"] > engine.config["max_sequence_tokens"]:
                raise ValueError("Complete teacher-gate prompt exceeds frozen budget")
    records = []
    with _no_update_scope(engine):
        for row in gate_rows:
            seed = int(digest_json([engine.config["seed"], row["id"], "teacher-gate"])[:8], 16)
            with engine.bank.use("initial"):
                student_ids, student_stop = engine.generate(engine.ids(row, False), seed, greedy=True)
            with engine.bank.use("teacher"):
                teacher_ids, teacher_stop = engine.generate(engine.ids(row, True), seed, greedy=True)
            student_text = engine.tokenizer.decode(student_ids, skip_special_tokens=True)
            teacher_text = engine.tokenizer.decode(teacher_ids, skip_special_tokens=True)
            records.append({"id": row["id"], "parent_id": row["parent_id"],
                            "student_correct": score(student_text, row["answer"], row["scorer"]),
                            "teacher_correct": score(teacher_text, row["answer"], row["scorer"]),
                            "student_prediction": student_text, "teacher_prediction": teacher_text,
                            "student_stop_reason": student_stop, "teacher_stop_reason": teacher_stop,
                            "student_response_tokens": len(student_ids), "teacher_response_tokens": len(teacher_ids),
                            "student_prefix_sha256": digest_json(engine.ids(row, False)),
                            "teacher_prefix_sha256": digest_json(engine.ids(row, True))})
    teacher_correct = sum(r["teacher_correct"] for r in records)
    student_correct = sum(r["student_correct"] for r in records)
    wins = sum(r["teacher_correct"] and not r["student_correct"] for r in records)
    harms = sum(r["student_correct"] and not r["teacher_correct"] for r in records)
    passed = teacher_correct >= 7 and wins >= 2 and harms == 0
    return {"status": "TEACHER_GATE_PASSED" if passed else "TEACHER_GATE_FAILED",
            "evidence": "TRAIN_GATE_BEHAVIOR_ONLY", "gate_rows": 8,
            "teacher_correct": teacher_correct, "student_correct": student_correct,
            "paired_wins": wins, "paired_harms": harms, "paired_net_wins": wins - harms,
            "required_teacher_correct": 7, "required_paired_wins": 2, "allowed_paired_harms": 0,
            "pass": passed, "records": records, "parameters_unchanged": True,
            "buffers_unchanged": True, "banks_unchanged": True, "training_updates": 0,
            "native_source_eligibility_established": False, "promotion_allowed": False}


def snapshot_source_custody(checkpoint, dataset):
    checkpoint, dataset = Path(checkpoint), Path(dataset)
    if checkpoint.is_symlink() or dataset.is_symlink():
        raise ValueError("Custody requires materialized source/data paths")
    return {"checkpoint_path": str(checkpoint.resolve()), "dataset_path": str(dataset.resolve()),
            "checkpoint_sha256": checkpoint_manifest(checkpoint)["sha256"],
            "dataset_sha256": hash_file(dataset)}


def recheck_source_custody(before):
    after = snapshot_source_custody(before["checkpoint_path"], before["dataset_path"])
    if before != after:
        raise RuntimeError("Source/data custody changed; qualification invalid")
    return {"source_custody_unchanged": True, "before": before, "after": after}


def capture_resource_stage(device, stage, min_free_cuda_gib=None):
    """Record CUDA allocated/reserved peaks without resetting load evidence.

    Future caller must record load_before/load_after/max_shape_after and place
    floors before each stage. CPU host peak accounting is explicitly pending.
    """
    if device not in {"cpu", "cuda"} or not isinstance(stage, str) or not stage:
        raise ValueError("Explicit device and resource stage required")
    if min_free_cuda_gib is not None and (isinstance(min_free_cuda_gib, bool)
            or not isinstance(min_free_cuda_gib, (int, float))
            or not math.isfinite(min_free_cuda_gib) or min_free_cuda_gib <= 0):
        raise ValueError("Positive finite CUDA free-memory floor required")
    record = {"device": device, "stage": stage, "monotonic_seconds": time.monotonic(),
              "host_peak_accounting": "PENDING", "peak_stats_reset_by_helper": False}
    if device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested BF16 CUDA runtime unavailable; no fallback")
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        record.update(free_bytes=free, total_bytes=total,
                      allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), min_free_cuda_gib=min_free_cuda_gib)
        if min_free_cuda_gib is not None and free < min_free_cuda_gib * 1024**3:
            raise RuntimeError("CUDA free-memory floor violated; no recipe fallback")
    return record


def validate_native_evidence(report, expected_identity, expected_source, expected_initial_digest):
    """Fail closed on source, loaded state, math dispatch and mechanical evidence.

    This validator cannot manufacture missing native evidence. The legacy CLI
    does not call it; approved orchestration and resource integration are pending.
    """
    if (not isinstance(expected_source, dict) or expected_source.get("eligible") is not True
            or not expected_source.get("source_id") or not expected_source.get("checkpoint_sha256")):
        raise ValueError("Freeze the exact eligible-source attestation first")
    if expected_identity.get("checkpoint_sha256") != expected_source["checkpoint_sha256"]:
        raise ValueError("Identity and eligible-source checkpoint differ")
    if report.get("identity") != expected_identity or report.get("source") != expected_source:
        raise ValueError("Qualification source/runtime/config identity differs")
    if report.get("initial_parameters_sha256") != expected_initial_digest:
        raise ValueError("Loaded initial parameter hash differs from qualification/source")
    if report.get("training_updates") != 0 or report.get("optimizer_objects_created") != 0:
        raise ValueError("Qualification must create no optimizer and perform no updates")
    events = report.get("backend_events", [])
    if not any(isinstance(e, str) and "scaled_dot_product" in e and "attention_math" in e for e in events):
        raise ValueError("Observed math SDPA dispatch evidence required")
    required_checks = ("parameters_unchanged", "buffers_unchanged", "teacher_initial_identity",
                       "full_vocabulary", "teacher_detached", "common_response_prefix",
                       "source_inference_parity", "max_shape_no_update", "max_shape_backward_no_update",
                       "source_custody_unchanged")
    if not all(report.get("checks", {}).get(k) is True for k in required_checks):
        raise ValueError("Missing no-update/parity/custody/full-vocabulary mechanical evidence")
    diagnostics = report.get("diagnostics", [])
    required_arms = {a.name for a in arms((0.,))}
    if len(diagnostics) != 5 or {d.get("arm_id") for d in diagnostics} != required_arms:
        raise ValueError("Qualification needs both rollout sources/directions and SFT exactly once")
    for diagnostic in diagnostics:
        if any(isinstance(diagnostic.get(k), bool) or not isinstance(diagnostic.get(k), (int, float))
               or not math.isfinite(diagnostic[k]) for k in ("loss", "gradient_l2")):
            raise ValueError("Nonfinite/missing mechanical objective or gradient")
        if diagnostic["gradient_l2"] < 0 or type(diagnostic.get("response_tokens")) is not int or diagnostic["response_tokens"] < 1:
            raise ValueError("Invalid mechanical gradient/response evidence")
    return {"mechanical_evidence_validated": True, "native_campaign_authorized": False,
            "teacher_competence_required_separately": True, "resource_integration": "PENDING"}
