"""Disposable runtime mechanics checks, teacher competence screen, and restore."""
from __future__ import annotations

from pathlib import Path

import torch

from .backend import tensor_hash
from .contracts import require, sealed, step_seed, write_new
from .data import verify
from .objectives import dense_kl, projected_kl


def qualify(backend, tasks, output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = backend.snapshot()
    original_version = backend.version
    frozen = backend.frozen_hashes()
    backend.meter.phase = "qualification"
    try:
        dev = [t for t in tasks if t.role == "teacher_dev"]
        require(bool(dev), "teacher-dev cases required")
        student_ok, teacher_ok = 0, 0
        outputs = []
        for t in dev:
            s = backend.sample(t.public(), "student", step_seed(0, t.key, 0, "teacher-qualification"), greedy=True)
            q = backend.sample(t.public(), "teacher", step_seed(0, t.key, 0, "teacher-qualification"), greedy=True)
            student_ok += verify(t, s.text)
            teacher_ok += verify(t, q.text)
            outputs.append({"key": t.key, "student": s.text, "teacher": q.text})
        accuracy, advantage = teacher_ok/len(dev), (teacher_ok-student_ok)/len(dev)
        require(accuracy >= backend.plan.teacher_min_accuracy and advantage >= backend.plan.teacher_min_advantage,
                "teacher competence/capability-gap screen failed")
        t = next(t for t in tasks if t.role == "train")
        r = backend.sample(t.public(), "student", 813)
        teacher = backend.activate("teacher")
        with torch.no_grad():
            th = backend.hidden(teacher, r.prompt_ids, r.response_ids[:2], False, "teacher").cpu()
        student = backend.activate("student")
        with torch.no_grad():
            sh = backend.hidden(student, r.prompt_ids, r.response_ids[:2], False, "student")
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
            r = backend.sample(t.public(), "student", 813)
            loss = backend.distill(t, r, direction, "student")
            loss.backward()
            backend.gradient_step(opt)
            require(tensor_hash(backend.snapshot()) != tensor_hash(original), "no scoped optimizer update")
        saved_hash = tensor_hash(backend.snapshot())
        backend.save(output / "disposable-checkpoint")
        backend.restore(original)
        backend.load(output / "disposable-checkpoint")
        require(tensor_hash(backend.snapshot()) == saved_hash, "save/reload bitwise adapter mismatch")
        require(backend.frozen_hashes() == frozen, "frozen parameter changed during qualification")
        result = {"status": "PASS_SCOPED_MECHANICS_AND_TEACHER_SCREEN", "teacher_dev_n": len(dev),
                  "teacher_accuracy": accuracy, "student_accuracy": student_ok/len(dev),
                  "teacher_advantage": advantage, "outputs": outputs, "parity_errors": errors,
                  "frozen_hashes": frozen, "save_reload_bitwise": True, "efficacy_claim": False,
                  "meter": backend.meter.snapshot()}
        write_new(output / "RESULT.json", sealed(result))
        return result
    except Exception as exc:
        write_new(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        backend.restore(original)
        backend.version = original_version
