"""No-optimizer diagnostics of already-trained Titan updates at fixed prefixes."""
from __future__ import annotations
import torch
from .controls import composition_controls, head_tail, norm_match
from .data import Row
from .metrics import interaction, response_metrics
from .tangent import Delta, FrozenAnchor


def prefix(row: Row, device="cpu") -> dict:
    first = next(i for i, x in enumerate(row.labels) if x != -100)
    ids = torch.tensor([row.input_ids[:first]], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def qualify(anchor: FrozenAnchor, row: Row, *, epsilons=(0.05, 0.025),
            max_relative_error=0.05) -> dict:
    """Zero identity and nonzero JVP finite differences. No training or accuracy gate."""
    device = next(iter(anchor.params.values())).device
    x = prefix(row, device)
    gen = torch.Generator(device=device).manual_seed(1789)
    d = {k: torch.randn(p.shape, dtype=p.dtype, device=device, generator=gen) * 0.001
         for k, p in anchor.params.items() if k in anchor.scope}
    with torch.no_grad():
        zero = anchor.zero()
        base = anchor.evaluate(zero, **x)[:, -1]
        zero_linear = anchor.evaluate(zero, mode="linear", **x)[:, -1]
        linear = anchor.evaluate(d, mode="linear", **x)[:, -1]
        j = (linear - base).double()
        if j.norm() == 0:
            raise ValueError("qualification direction has zero observed response")
        errors = []
        for eps in epsilons:
            plus = anchor.evaluate({k: eps * v for k, v in d.items()}, **x)[:, -1]
            minus = anchor.evaluate({k: -eps * v for k, v in d.items()}, **x)[:, -1]
            fd = (plus.double() - minus.double()) / (2 * eps)
            errors.append(float((fd - j).norm() / j.norm()))
    # Forward AD passing does not qualify reverse-mode gradients through the JVP.
    trainable = {k: v.detach().requires_grad_(True) for k, v in d.items()}
    probe_logits = anchor.evaluate(trainable, mode="linear", **x)[:, -1]
    target = next(v for v in row.labels if v != -100)
    grads = torch.autograd.grad(probe_logits[0, target], tuple(trainable.values()))
    gradient_ok = all(torch.isfinite(g).all() for g in grads) and any(g.count_nonzero() for g in grads)
    identity = torch.equal(base, zero_linear)
    anchor.assert_unchanged()
    ok = identity and bool(gradient_ok) and all(e <= max_relative_error for e in errors)
    return {"status": "QUALIFICATION_PASS" if ok else "QUALIFICATION_FAIL",
            "zero_delta_bitwise_identity": identity, "reverse_through_jvp_finite_nonzero": bool(gradient_ok),
            "backward_calls": 1, "epsilons": list(epsilons),
            "finite_difference_relative_errors": errors,
            "pre_registered_max_relative_error": max_relative_error,
            "scope_parameter_tensors": len(anchor.scope), "optimizer_steps": 0,
            "meaning": "one-prefix backend mechanics, NOT task efficacy or all-input parity"}


def historical_diagnostics(anchor: FrozenAnchor, a: Delta, rows: list[Row],
                           b: Delta | None = None) -> dict:
    anchor.validate(a)
    if b is not None:
        anchor.validate(b)
    if not rows or any(r.split != "diagnostic" for r in rows):
        raise ValueError("only explicitly developmental diagnostic rows are eligible")
    head, tail = head_tail(a, 0.25)
    arms = {"full": a, "head25": head, "tail75": tail,
            "dense_head_norm": norm_match(a, head)}
    if b is not None:
        arms.update(composition_controls(a, b))
    results = []
    device = next(iter(a.values())).device
    with torch.no_grad():
        for row in rows:
            x = prefix(row, device)
            base = anchor.evaluate(anchor.zero(), **x)[:, -1]
            nonlinear, linear = {}, {}
            for name, d in arms.items():
                nonlinear[name] = anchor.evaluate(d, **x)[:, -1]
                linear[name] = anchor.evaluate(d, mode="linear", **x)[:, -1]
            target = next(x for x in row.labels if x != -100)
            measured = {name: {**response_metrics(base, nonlinear[name], linear[name]),
                               "reference_first_token": target,
                               "predicted_first_token": int(nonlinear[name].argmax()),
                               "reference_logprob_change": float(
                                   nonlinear[name].log_softmax(-1)[0, target]
                                   - base.log_softmax(-1)[0, target])}
                        for name in arms}
            item = {"id": row.id, "parent_id": row.parent_id, "task": row.task, "arms": measured,
                    "head_tail_nonlinear_interaction_rms": interaction(
                        base, nonlinear["head25"], nonlinear["tail75"], nonlinear["full"])}
            if b is not None:
                item["matched_sum_vs_attenuated_a_rms"] = float(
                    (nonlinear["matched_sum"] - nonlinear["attenuated_a"]).double().square().mean().sqrt())
            results.append(item)
    anchor.assert_unchanged()
    return {"status": "DIAGNOSTIC_COMPLETE", "rows": results, "optimizer_steps": 0,
            "quality_claim": False, "mechanism_established": False,
            "warning": "output energy is not harm; head/tail are not independent skills"}
