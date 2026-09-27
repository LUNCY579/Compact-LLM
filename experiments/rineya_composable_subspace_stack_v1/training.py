"""Stage-2 tangent teacher and ordinary-student distillation mechanics.

A functional module must return (prediction, hidden). No model download, native
loader, optimizer auto-resume, or authorization. This is a DELTA-inspired
reference implementation with a squared Taylor-residual curvature surrogate,
not a claim of faithfully reproducing the paper's complete training recipe.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.func import functional_call, jvp
from .core import require


@dataclass(frozen=True)
class TrainingConfig:
    teacher_steps: int = 8
    student_steps: int = 8
    lr: float = 0.01
    curvature_weight: float = 0.1
    distillation_weight: float = 1.0
    max_trainable_elements: int = 262_144
    grad_clip: float = 1.0

    def validate(self):
        for n in (self.teacher_steps, self.student_steps):
            require(type(n) is int and 1 <= n <= 256, "bounded positive training steps required")
        require(type(self.max_trainable_elements) is int and 0 < self.max_trainable_elements <= 4_194_304, "trainable budget invalid")
        vals = torch.tensor([self.lr, self.curvature_weight, self.distillation_weight, self.grad_clip])
        require(bool(torch.isfinite(vals).all()), "nonfinite training settings")
        require(0 < self.lr <= 0.1 and self.curvature_weight >= 0 and self.distillation_weight >= 0 and self.grad_clip > 0, "training settings out of range")


class FunctionalSkill:
    def __init__(self, module, parameter_names, max_elements=262_144):
        require(not module.training, "deterministic eval mode required; dropout disabled")
        require(parameter_names and len(parameter_names) == len(set(parameter_names)), "unique trainable parameter names required")
        self.module = module
        self.base = {n: p.detach().clone() for n, p in module.named_parameters()}
        self.buffers = {n: b.detach().clone() for n, b in module.named_buffers()}
        self.names = tuple(parameter_names)
        require(set(self.names) <= set(self.base), "unknown trainable parameter")
        require(sum(self.base[n].numel() for n in self.names) <= max_elements, "trainable budget exceeded")
        require(all(self.base[n].dtype in (torch.float32, torch.float64) for n in self.names), "FP32/FP64 tangent parameters required")

    def zero(self):
        return tuple(torch.zeros_like(self.base[n]) for n in self.names)

    def call(self, delta, x):
        require(len(delta) == len(self.names), "delta parameter count mismatch")
        require(all(d.shape == self.base[n].shape for n, d in zip(self.names, delta)), "delta parameter shape mismatch")
        params = dict(self.base)
        params.update({n: self.base[n]+d for n, d in zip(self.names, delta)})
        # Cloned buffers forbid training-time state leaking between teacher/student.
        output = functional_call(self.module, (params, {n: b.clone() for n, b in self.buffers.items()}), (x,), strict=True)
        require(isinstance(output, tuple) and len(output) == 2, "model must return (prediction, hidden)")
        require(all(torch.is_tensor(t) and bool(torch.isfinite(t).all()) for t in output), "invalid model output")
        return output

    def tangent(self, delta, x):
        base_out, directional = jvp(lambda ds: self.call(ds, x), (self.zero(),), (tuple(delta),))
        return tuple(y+dy for y, dy in zip(base_out, directional))

    def qualify(self, x, *, epsilon=1e-3, tolerance=0.03):
        require(epsilon > 0 and tolerance > 0, "qualification parameters invalid")
        direction = tuple(torch.ones_like(p)/max(p.numel(), 1)**0.5 for p in self.zero())
        base, exact = jvp(lambda ds: self.call(ds, x), (self.zero(),), (direction,))
        plus = self.call(tuple(epsilon*d for d in direction), x)
        minus = self.call(tuple(-epsilon*d for d in direction), x)
        errors = []
        for j, hi, lo in zip(exact, plus, minus):
            fd = (hi-lo)/(2*epsilon)
            err = torch.linalg.vector_norm(fd-j)/max(float(torch.linalg.vector_norm(j)), 1e-7)
            errors.append(float(err.detach()))
        require(max(errors) <= tolerance, "JVP/central-difference qualification failed")
        require(all(torch.equal(self.base[n], p.detach()) for n, p in self.module.named_parameters()), "base model mutated")
        return {"jvp_relative_errors": errors, "passed": True, "native_rineya_qualified": False}


def distill_skill(module, parameter_names, batches, loss_fn, config=TrainingConfig()):
    """Run bounded teacher/student optimizer steps on caller-supplied TRAIN data.

    The caller must bind/authorize any native model separately. CHECK data is
    rejected here; no synthetic receipt can authorize a native host run.
    """
    config.validate()
    require(batches and all(b.get("split") == "TRAIN" for b in batches), "training accepts TRAIN-only batches")
    engine = FunctionalSkill(module, parameter_names, config.max_trainable_elements)
    qualification = engine.qualify(batches[0]["x"])
    teacher = [torch.nn.Parameter(d) for d in engine.zero()]
    student = [torch.nn.Parameter(d) for d in engine.zero()]
    history = {"teacher": [], "student": []}
    def optimize(params, steps, kind):
        opt = torch.optim.Adam(params, lr=config.lr)
        for step in range(steps):
            batch = batches[step % len(batches)]
            opt.zero_grad(set_to_none=True)
            if kind == "teacher":
                linear = engine.tangent(params, batch["x"])
                nonlinear = engine.call(params, batch["x"])
                curvature = sum(torch.mean((a-b)**2) for a, b in zip(nonlinear, linear))
                loss = loss_fn(linear[0], batch["y"]) + config.curvature_weight*curvature
            else:
                with torch.no_grad():
                    target = engine.tangent(tuple(p.detach() for p in teacher), batch["x"])
                prediction, hidden = engine.call(params, batch["x"])
                loss = loss_fn(prediction, batch["y"]) + config.distillation_weight*torch.mean((hidden-target[1])**2)
            require(loss.ndim == 0 and bool(torch.isfinite(loss)), "invalid training loss")
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params), "missing/nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip, error_if_nonfinite=True)
            opt.step()
            require(all(bool(torch.isfinite(p).all()) for p in params), "nonfinite trained delta")
            history[kind].append(float(loss.detach()))
    optimize(teacher, config.teacher_steps, "teacher")
    optimize(student, config.student_steps, "student")
    require(all(torch.equal(engine.base[n], p.detach()) for n, p in module.named_parameters()), "base changed during distillation")
    return {"teacher_delta": dict(zip(engine.names, (p.detach().clone() for p in teacher))),
            "student_delta": dict(zip(engine.names, (p.detach().clone() for p in student))),
            "history": history, "qualification": qualification,
            "optimizer_steps": config.teacher_steps+config.student_steps,
            "native_rineya_qualified": False, "release_permitted": False}


def ordinary_skill(module, parameter_names, batches, loss_fn, config=TrainingConfig()):
    """Matched student-update supervised control; no tangent teacher overhead."""
    config.validate()
    require(batches and all(b.get("split") == "TRAIN" for b in batches), "TRAIN-only batches required")
    engine = FunctionalSkill(module, parameter_names, config.max_trainable_elements)
    parameters = [torch.nn.Parameter(d) for d in engine.zero()]
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    history = []
    for step in range(config.student_steps):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = engine.call(parameters, batch["x"])
        loss = loss_fn(prediction, batch["y"])
        require(loss.ndim == 0 and bool(torch.isfinite(loss)), "invalid ordinary loss")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), "ordinary gradients invalid")
        torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in parameters), "ordinary delta invalid")
        history.append(float(loss.detach()))
    return {"delta": dict(zip(engine.names, (p.detach().clone() for p in parameters))),
            "loss_history": history, "optimizer_steps": config.student_steps}


def train_composition_study(module, parameter_names, skill_batches, loss_fn,
                            coefficients, arm, *, training_config=TrainingConfig(), compose_settings):
    """Two-stage experiment: ordinary/tangent/distilled skills -> same merger.

    Each skill starts from the identical immutable module. Exactly the same TRAIN
    batches reach ordinary and distilled students. Optimizer-step accounting
    exposes teacher overhead; this is NOT a total-compute-matched comparison.
    """
    from .core import compose
    require(2 <= len(skill_batches) <= 4 and len(coefficients) == len(skill_batches), "skill/coefficient count mismatch")
    names = sorted(skill_batches)
    controls, trained = {}, {}
    for name in names:
        controls[name] = ordinary_skill(module, parameter_names, skill_batches[name], loss_fn, training_config)
        trained[name] = distill_skill(module, parameter_names, skill_batches[name], loss_fn, training_config)
    base = dict(module.named_parameters())
    merged = {}
    for variant in ("ordinary", "tangent", "distilled"):
        matrices = {}
        for key in parameter_names:
            w = base[key].detach().cpu().numpy()
            if variant == "ordinary":
                ds = [controls[n]["delta"][key].cpu().numpy() for n in names]
            else:
                field = "teacher_delta" if variant == "tangent" else "student_delta"
                ds = [trained[n][field][key].cpu().numpy() for n in names]
            matrices[key] = compose(w, ds, coefficients, arm, **compose_settings)
        merged[variant] = matrices
    return {"skill_order": names, "ordinary": controls, "distilled": trained, "compositions": merged,
            "total_compute_matched": False, "student_updates_matched": True, "native_rineya_qualified": False}
