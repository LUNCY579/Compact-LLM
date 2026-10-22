"""Disposable runtime mechanics checks, teacher competence screen, and restore."""
from __future__ import annotations

from pathlib import Path

import torch

from .backend import tensor_hash
from .contracts import require, sealed, step_seed, write_new
from .data import verify
from .objectives import dense_kl, projected_kl


def teacher_screen(backend, tasks):
    """Public-prompt competence only; caps and wrong answers remain in denominator."""
    dev = [t for t in tasks if t.role == "teacher_dev"]
    require(bool(dev), "teacher-dev cases required")
    if backend.plan.design == "selected":
        require(len(dev) == 16 and len({t.parent for t in dev}) == 16,
                "selected qualification freezes exactly 16 teacher-dev parents")
    student_ok, teacher_ok = 0, 0
    outputs = []
    for t in dev:
        seed = step_seed(0, t.key, 0, "teacher-qualification")
        s = backend.sample(t.public(), "student", seed, greedy=True, purpose="teacher_competence_screen")
        q = backend.sample(t.public(), "teacher", seed, greedy=True, purpose="teacher_competence_screen")
        sc, qc = verify(t, s.text), verify(t, q.text)
        student_ok += sc
        teacher_ok += qc
        outputs.append({"key": t.key, "parent": t.parent, "source": t.source,
                        "student": s.text, "teacher": q.text, "student_correct": sc,
                        "teacher_correct": qc, "student_terminated": s.terminated,
                        "teacher_terminated": q.terminated,
                        "student_response_tokens": len(s.response_ids),
                        "teacher_response_tokens": len(q.response_ids)})
    accuracy, advantage = teacher_ok / len(dev), (teacher_ok - student_ok) / len(dev)
    passed = (accuracy >= backend.plan.teacher_min_accuracy
              and advantage >= backend.plan.teacher_min_advantage)
    return {"status": "PASS_TEACHER_SCREEN" if passed else "FAIL_TEACHER_SCREEN",
            "teacher_dev_n": len(dev), "teacher_accuracy": accuracy,
            "student_accuracy": student_ok / len(dev), "teacher_advantage": advantage,
            "outputs": outputs, "capped_attempts_included": True, "efficacy_claim": False}


def probe_logits(backend, task, rollout):
    model = backend.activate("student")
    with torch.no_grad():
        h = backend.hidden(model, rollout.prompt_ids, rollout.response_ids[:2], False, "student",
                           problem=task.public(), purpose="save_reload_logits_probe")
        return model.lm_head(h).detach().cpu()


def qualify(backend, tasks, output: Path):
    """Keep primary and finalization failures, including restoration failures."""
    output = Path(output)
    existed = output.exists()
    try:
        return _qualify(backend, tasks, output)
    except BaseException as exc:
        if not existed and output.is_dir():
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "restoration_receipt_present": (output / "RESTORED.json").is_file()}
            name = "FINALIZATION_FAILED.json" if (output / "FAILED.json").exists() else "FAILED.json"
            write_new(output / name, failure)
            if not (output / "CONSUMED_PROMPTS_INCOMPLETE.json").exists():
                try:
                    backend.write_prompt_manifest(output / "CONSUMED_PROMPTS_INCOMPLETE.json", complete=False)
                except BaseException as manifest_exc:
                    manifest_name = ("FINALIZATION_MANIFEST_FAILED.json" if (output / "MANIFEST_FAILED.json").exists()
                                     else "MANIFEST_FAILED.json")
                    write_new(output / manifest_name,
                              {"type": type(manifest_exc).__name__, "message": str(manifest_exc)})
        raise


def _qualify(backend, tasks, output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = backend.snapshot()
    original_version = backend.version
    frozen = backend.frozen_hashes()
    backend.bind_prompt_catalog(tasks)
    backend.meter.phase = "qualification"
    try:
        screen = teacher_screen(backend, tasks)
        write_new(output / "TEACHER_SCREEN.json", sealed(screen))
        require(screen["status"] == "PASS_TEACHER_SCREEN",
                "teacher competence/capability-gap screen failed")
        t = next(t for t in tasks if t.role == "train")
        r = backend.sample(t.public(), "student", 813, purpose="kl_probe_rollout")
        teacher = backend.activate("teacher")
        with torch.no_grad():
            th = backend.hidden(teacher, r.prompt_ids, r.response_ids[:2], False, "teacher",
                                problem=t.public(), purpose="kl_value_gradient_probe").cpu()
        student = backend.activate("student")
        with torch.no_grad():
            sh = backend.hidden(student, r.prompt_ids, r.response_ids[:2], False, "student",
                                problem=t.public(), purpose="kl_value_gradient_probe")
        errors = {}
        for direction in ("forward", "reverse"):
            h = sh.detach().clone().requires_grad_(True)
            zt = torch.nn.functional.linear(th, teacher.lm_head.weight).detach().to(h.device)
            reference = dense_kl(torch.nn.functional.linear(h, student.lm_head.weight), zt, direction)
            expected = torch.autograd.grad(reference, h)[0]
            actual_loss = projected_kl(h, student.lm_head.weight, th, teacher.lm_head.weight, direction, 1)
            actual = torch.autograd.grad(actual_loss, h)[0]
            # BF16 matmul has rounding; strict FP64 gradient checks live in tests.
            require(torch.allclose(actual_loss, reference, atol=2e-4, rtol=2e-4), "native KL value parity")
            require(torch.allclose(actual, expected, atol=2e-3, rtol=2e-2), "native KL gradient parity")
            errors[direction] = {"loss_abs": float((actual_loss-reference).detach().abs()),
                                 "hidden_gradient_max_abs": float((actual-expected).abs().max())}
        for direction in ("forward", "reverse"):
            backend.restore(original)
            backend.version = original_version
            opt = torch.optim.AdamW(list(backend.params.values()), lr=backend.plan.learning_rates[0],
                                    weight_decay=backend.plan.weight_decay)
            r = backend.sample(t.public(), "student", 813, purpose="disposable_" + direction + "_rollout")
            loss = backend.distill(t, r, direction, "student", purpose="disposable_" + direction + "_backward")
            loss.backward()
            backend.gradient_step(opt)
            require(tensor_hash(backend.snapshot()) != tensor_hash(original), "no scoped optimizer update")
        saved_hash = tensor_hash(backend.snapshot())
        before_logits = probe_logits(backend, t, r)
        backend.save(output / "disposable-checkpoint")
        backend.restore(original)
        backend.load(output / "disposable-checkpoint")
        require(tensor_hash(backend.snapshot()) == saved_hash, "save/reload bitwise adapter mismatch")
        require(torch.equal(probe_logits(backend, t, r), before_logits), "save/reload native-logit mismatch")
        require(backend.frozen_hashes() == frozen, "frozen parameter changed during qualification")
        result = {**screen, "status": "PASS_SCOPED_MECHANICS_AND_TEACHER_SCREEN", "parity_errors": errors,
                  "frozen_hashes": frozen, "save_reload_bitwise": True, "save_reload_logits_bitwise": True,
                  "disposable_optimizer_updates": 2, "treatment_optimizer_updates": 0,
                  "evidence_kind": backend.evidence_kind,
                  "meter": backend.meter.snapshot()}
    except BaseException as exc:
        write_new(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        try:
            backend.write_prompt_manifest(output / "CONSUMED_PROMPTS_INCOMPLETE.json", complete=False)
        except BaseException as manifest_exc:
            write_new(output / "MANIFEST_FAILED.json",
                      {"type": type(manifest_exc).__name__, "message": str(manifest_exc)})
        raise
    finally:
        backend.restore(original)
        backend.version = original_version
        require(tensor_hash(backend.snapshot()) == tensor_hash(original), "qualification source restoration failed")
        write_new(output / "RESTORED.json", sealed({"adapter_sha256": tensor_hash(original),
                  "policy_version": original_version, "optimizer_reused_for_treatment": False}))
    result["source_restored_bitwise"] = True
    result["consumed_prompt_manifest"] = backend.write_prompt_manifest(output / "CONSUMED_PROMPTS.json", complete=True)
    write_new(output / "RESULT.json", sealed(result))
    return result
