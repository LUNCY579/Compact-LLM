"""Frozen routing replay, actual whole-call execution, and labelled mechanics demos."""
from dataclasses import asdict
import json
import time
import numpy as np
import torch
from torch import nn
from .artifacts import validate_rows, write_json
from .control import ControlConfig, digest, CountController, quantile_bias
from .routing import Expert, RoutingConfig, PoolRouter, diagnostic
from .moe import MoEConfig, SparseLoRALinear


def build_pool(spec):
    if spec.get("schema") != 1:
        raise ValueError("unsupported pool schema")
    return PoolRouter([Expert(**e) for e in spec["experts"]],
                      RoutingConfig(**spec.get("routing", {})),
                      ControlConfig(**spec.get("control", {})))


def calibrate(spec, rows):
    validate_rows(rows)
    if not rows or any(r["split"] not in {"train", "calibration"} for r in rows):
        raise ValueError("calibration cannot read development/confirmation rows")
    pool = build_pool(spec)
    window = int(spec.get("window_size", 32))
    if window < 1:
        raise ValueError("positive window size required")
    logs = []
    for start in range(0, len(rows), window):
        pool.controller.begin("calibration")
        try:
            for row in rows[start:start + window]:
                pool.choose(row["family"], row["quality"], row["predicted_cost"],
                            row.get("applicable"), record=True)
            logs.append(pool.controller.commit())
        except BaseException:
            pool.controller.abort()
            raise
    return {"state": pool.state_dict(), "calibration_sha256": digest(rows),
            "windows": logs, "native_qualified": False,
            "quality_sources": sorted({digest(r.get("quality_source", "unspecified")) for r in rows}),
            "calibration_identities": [{"family": r["family"], "parent_id": r["parent_id"],
                                        "payload_sha256": digest(r["payload"])} for r in rows]}


def exact_state(text, expected):
    try:
        value = json.loads(text.strip())
        return float(digest(value) == digest(expected))
    except (ValueError, TypeError):
        return 0.0


def evaluate(spec, rows, checkpoint, split="dev", backend=None, evidence_kind="synthetic"):
    validate_rows(rows)
    if split not in {"dev", "confirm"} or not rows or any(r["split"] != split for r in rows):
        raise ValueError("evaluate exactly one held-out split")
    sources = {digest(r.get("quality_source", "unspecified")) for r in rows}
    if not sources.issubset(set(checkpoint["quality_sources"])):
        raise ValueError("quality estimator changed since calibration")
    prior = checkpoint["calibration_identities"]
    seen_parents = {(r["family"], r["parent_id"]) for r in prior}
    seen_payloads = {r["payload_sha256"] for r in prior}
    if any((r["family"], r["parent_id"]) in seen_parents or digest(r["payload"]) in seen_payloads for r in rows):
        raise ValueError("held-out evaluation overlaps calibration parents/payloads")
    pool = build_pool(spec)
    pool.controller.load_state_dict(checkpoint["state"])
    before = pool.state_dict()
    records, outcomes, picks, masks = [], [], [], []
    for row in rows:
        start = time.perf_counter()
        choice = pool.choose(row["family"], row["quality"], row["predicted_cost"], row.get("applicable"))
        index = choice["expert_index"]
        response = None
        if backend is None:
            if evidence_kind not in {"synthetic", "native_replay"}:
                raise ValueError("replay cannot claim native execution")
            success = row["outcomes"][index]
            elapsed = row["observed_cost"][index]
            outcomes.append(row["outcomes"])
        else:
            feature_cost = row["quality_feature_seconds"]
            if not np.isfinite(feature_cost) or feature_cost < 0:
                raise ValueError("native execution must account for quality-feature cost")
            response = backend(choice["expert"], choice["strength"],
                               {"prompt": row["payload"]["prompt"],
                                "max_new_tokens": spec.get("max_new_tokens", 128)})
            elapsed = time.perf_counter() - start + feature_cost
            success = exact_state(response["text"], row["expected_state"])
        if not np.isfinite(success) or not 0 <= success <= 1 or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid outcome/cost")
        picks.append(index)
        masks.append(choice["eligible"])
        records.append({"id": row["id"], "parent_id": row["parent_id"], "family": row["family"],
                        "seed": row.get("seed", 0), "stratum": row.get("stratum", "all"),
                        "payload_sha256": digest(row["payload"]),
                        "success": float(success), "elapsed_seconds": float(elapsed),
                        "choice": choice, "response": response})
    if pool.state_dict() != before:
        raise RuntimeError("evaluation mutated the controller")
    return {"schema": 1, "evidence_kind": "native_execution" if backend else evidence_kind,
            "native_qualified": False, "evaluation_split": split,
            "end_to_end_timing": backend is not None,
            "data_sha256": digest(rows), "pool_sha256": digest(spec["experts"]),
            "budget": {"calls_per_example": 1, "max_new_tokens": spec.get("max_new_tokens", 128)},
            "spec_sha256": digest(spec), "controller_sha256": before["sha256"],
            "quality_predictions": "supplied from development-qualified estimator; audit required",
            "mean_success": float(np.mean([r["success"] for r in records])),
            "mean_cost_seconds": float(np.mean([r["elapsed_seconds"] for r in records])),
            "override_fraction": float(np.mean([r["choice"]["overrode_unbiased"] for r in records])),
            "diagnostic": diagnostic(outcomes, picks, masks) if backend is None else None,
            "records": records}


def synthetic_rows(split, count=192, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        # Constructed opportunities, misses, and no-expert-solves competence gaps.
        quality = rng.uniform(0.68, 0.82, 4)
        quality[0] += 0.03
        truth = rng.binomial(1, np.clip(quality + rng.normal(0, 0.08, 4), 0, 1))
        if i % 13 == 0:
            truth[:] = 0
        applicable = [True, True, True, i % 5 != 0]
        rows.append({"id": f"{split}-{i}", "parent_id": f"{split}-p{i // 2}",
                     "family": "transition", "split": split,
                     "payload": {"prompt": f"SYNTHETIC {split} state {i}"},
                     "quality": quality.tolist(), "predicted_cost": [1.0, 1.1, 0.9, 1.2],
                     "observed_cost": [1.0, 1.1, 0.9, 1.2], "outcomes": truth.tolist(),
                     "applicable": applicable})
    return rows


def synthetic_spec(method="id", target_policy="prior"):
    return {"schema": 1,
            "experts": [asdict(Expert(f"expert_{i}", "transition", digest({"fixture": i}), True,
                                      prior=[0.5, 0.2, 0.2, 0.1][i])) for i in range(4)],
            "routing": asdict(RoutingConfig(target_policy=target_policy)),
            "control": asdict(ControlConfig(method=method)), "window_size": 32,
            "qualification": "SYNTHETIC_REGISTRY_NOT_FOR_NATIVE_USE"}


def routing_demo(out):
    calibration, dev = synthetic_rows("calibration", seed=1), synthetic_rows("dev", seed=2)
    validate_rows(calibration + dev)
    summary = []
    for target in ("uniform", "prior"):
        for method in ("none", "sign", "integral", "id"):
            spec = synthetic_spec(method, target)
            checkpoint = calibrate(spec, calibration)
            result = evaluate(spec, dev, checkpoint)
            tag = f"{target}_{method}"
            write_json(out / f"{tag}_state.json", checkpoint)
            write_json(out / f"{tag}_report.json", result)
            summary.append({"arm": tag, "success": result["mean_success"],
                            "override_fraction": result["override_fraction"]})
    return {"status": "SYNTHETIC_MECHANICS_ONLY", "native_executed": False, "arms": summary}


def tiny_moe(method="id", auxiliary=False, seed=0, steps=20):
    """Actual CPU optimizer steps on a constructed regression task, not an LLM."""
    if steps < 1:
        raise ValueError("positive steps required")
    torch.manual_seed(seed)
    model = SparseLoRALinear(nn.Linear(8, 6), MoEConfig(auxiliary_weight=0.02 if auxiliary else 0),
                            ControlConfig(method="none" if auxiliary else method), "tiny-fixture")
    base_before = {k: v.detach().clone() for k, v in model.base.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.03)
    x = torch.randn(32, 8)
    target = x[:, :6] * (1 + (x[:, :1] > 0).float())
    losses, logs = [], []
    for _ in range(steps):
        model.begin_step()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = nn.functional.mse_loss(pred, target) + model.last_auxiliary
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        logs.append(model.end_step())
        losses.append(float(loss.detach()))
    immutable = all(torch.equal(v, base_before[k]) for k, v in model.base.state_dict().items())
    if not immutable:
        raise RuntimeError("frozen projection changed")
    model.eval()
    with torch.no_grad():
        evaluation = float(nn.functional.mse_loss(model(x), target))
    return {"status": "SYNTHETIC_OPTIMIZER_MECHANICS_ONLY", "native_executed": False,
            "method": "auxiliary" if auxiliary else method, "seed": seed, "steps": steps,
            "initial_loss": losses[0], "last_training_loss": losses[-1], "fixture_mse": evaluation,
            "frozen_base_unchanged": immutable, "losses": losses, "control": logs,
            "controller_state": model.controller.state_dict()}


def paper_demo(seed=0, windows=40):
    rng = np.random.default_rng(seed)
    batches = [rng.uniform(0, 1, (96, 16)) + np.arange(16)[None, :] * 0.025
               for _ in range(windows)]
    histories = {}
    for method in ("none", "sign", "integral", "id", "quantile"):
        controller = CountController([str(i) for i in range(16)],
                                     ControlConfig(method=method if method != "quantile" else "none",
                                                   bias_limit=None, min_expected=0))
        trace = []
        for scores in batches:
            choice = np.argsort(-(scores + controller.bias), kind="stable", axis=1)[:, :3]
            counts = np.bincount(choice.ravel(), minlength=16)
            target = len(scores) * 3 / 16
            trace.append(float((counts.max() - target) / target))
            if method == "quantile":
                controller.bias = quantile_bias(scores, controller.bias, 3)
            else:
                controller.begin()
                controller.observe(counts, [target] * 16)
                controller.commit()
        histories[method] = trace
    return {"status": "SYNTHETIC_CONTROL_DYNAMICS_NOT_PAPER_REPRODUCTION",
            "native_executed": False, "seed": seed, "max_overload_traces": histories}
