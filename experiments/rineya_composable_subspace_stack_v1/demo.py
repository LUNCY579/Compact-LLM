"""Tiny reproducible end-to-end mechanics demo; NEVER native efficacy evidence."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from . import EXPERIMENT
from .core import compose, design, geometry
from .protocol import default_plan, write_new, file_hash, digest, freeze, audit_splits, control_recipes
from .routing import PromptFeature, fit_router, route_and_predict, routing_regret
from .evaluation import compare, factorial_effects, select_on_cal, confirm_check


def run_demo(output, seed=1729):
    import torch
    from .training import distill_skill, TrainingConfig
    output = Path(output)
    output.mkdir(exist_ok=False)
    write_new(output/"INCOMPLETE.json", {"fixture_only": True})
    rng = np.random.default_rng(seed)
    base = np.diag(np.linspace(4, 1, 8))
    ds = [np.diag([0.0, 0.3, 0.0, 0.2, 0, 0, 0, 0]),
          np.diag([0.0, 0.3, 0.2, 0.0, 0, 0, 0, 0])]
    plan = default_plan()
    plan["settings"].update(rank=2, protected_rank=1, shared_rank=2)
    artifacts, reports, arrays = {}, {}, {}
    for arm in design():
        result, report = compose(base, ds, plan["coefficients"], arm, **plan["settings"])
        path = output/(arm.name+".npz")
        np.savez(path, delta=result)
        artifacts[arm.name] = file_hash(path)
        reports[arm.name] = report
        arrays[arm.name] = result
    control_reports = {}
    for name, control in control_recipes(plan).items():
        if control["kind"] == "endpoint":
            source = control["source"]
            result = np.zeros_like(base) if source == "supa" else ds[plan["donors"].index(source)].copy()
        else:
            result = sum(a*d for a, d in zip(control["coefficients"], ds))
        path = output/(name+".npz")
        np.savez(path, delta=result)
        artifacts[name] = file_hash(path)
        arrays[name] = result
        control_reports[name] = control
    ledgers = {split: [{"id": f"{split}-{i}", "parent_id": f"{split}-parent-{i}",
                        "prompt": f"synthetic {split} example {i}"} for i in range(12)]
               for split in ("TRAIN", "CAL", "CHECK")}
    audit = audit_splits(ledgers)
    frozen = freeze(plan, artifacts, cal_sha256=digest(ledgers["CAL"]),
                    check_sha256=digest(ledgers["CHECK"]), audit_sha256=digest(audit))
    write_new(output/"plan.json", plan)
    write_new(output/"freeze.json", frozen)
    def runs(split):
        x = rng.normal(size=(12, 8)).astype(np.float32)
        wanted = base + 0.5*(ds[0]+ds[1])
        answer = x @ wanted.astype(np.float32).T
        out = {}
        for name, delta in arrays.items():
            losses = np.mean((x@(base+delta).astype(np.float32).T-answer)**2, axis=1)
            rows = [{"id": row["id"], "parent_id": row["parent_id"],
                     "domain": "toy_skill" if i < 6 else "toy_retention",
                     "role": "capability" if i < 6 else "retention", "correct": bool(losses[i] < 0.005),
                     "loss": float(losses[i])} for i, row in enumerate(ledgers[split])]
            meta = {"experiment": EXPERIMENT, "arm": name, "split": split,
                    "plan_sha256": digest(plan), "dataset_sha256": frozen["cal_sha256" if split == "CAL" else "check_sha256"],
                    "scorer_sha256": digest("synthetic_mse_threshold_0.005"), "runtime_sha256": digest("synthetic_numpy"),
                    "decoding_sha256": digest("no_decoder_synthetic_matrix"), "tokenizer_sha256": digest("no_tokenizer"),
                    "freeze_sha256": digest(frozen), "artifact_sha256": artifacts[name], "evaluation_dtype": "float32"}
            out[name] = {"schema": 1, "metadata": meta, "rows": rows, "fixture_only": True}
        return out
    cal = runs("CAL")
    selection = select_on_cal(cal["base"], [cal[a.name] for a in design()], frozen)
    write_new(output/"selection.json", selection)
    # Never manufacture a positive candidate when the screen finds none.
    confirmation = {"status": "NOT_RUN_NO_SELECTED_CANDIDATE"}
    if selection["selected"] is not None:
        check = runs("CHECK")
        confirmation = confirm_check(check[selection["selected"]], frozen, selection)
    features = [PromptFeature("cal-route-a", "cal-route-pa", digest("train route a"), (1., 0.)),
                PromptFeature("cal-route-b", "cal-route-pb", digest("train route b"), (0., 1.))]
    router = fit_router(features, ["a", "b"], {"a": [1., 0.], "b": [0., 1.]},
                        split="CAL", feature_receipt=digest("synthetic_features"))
    routed = route_and_predict(router, [PromptFeature("q", "pq", digest("query"), (1., 0.))], ["query"],
                               lambda prompt, coefficients: float(np.linalg.norm(sum(a*d for a, d in zip(coefficients, ds)))))
    torch.manual_seed(seed)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4, bias=False)
        def forward(self, x):
            hidden = torch.tanh(self.layer(x))
            return hidden, hidden
    toy = Toy().double().eval()
    x = torch.tensor(rng.normal(size=(8, 4)), dtype=torch.float64)
    batch = {"x": x, "y": 0.5*torch.tanh(x), "split": "TRAIN"}
    training = distill_skill(toy, ["layer.weight"], [batch], torch.nn.functional.mse_loss,
                            TrainingConfig(teacher_steps=4, student_steps=4))
    np.savez(output/"distilled_skill.npz", **{k: v.numpy() for k, v in training["student_delta"].items()})
    report = {"status": "SYNTHETIC_MECHANICS_ONLY", "seed": seed, "arms": reports, "controls": control_reports,
              "geometry": geometry(ds), "split_audit": audit,
              "full_vs_average": compare(cal["P0S0C0"], cal["P1S1C1"]),
              "factorial": factorial_effects({k: v for k, v in cal.items() if k.startswith("P")}),
              "selection": selection, "confirmation": confirmation, "routing": routed,
              "routing_regret": routing_regret([[0., 1.], [1., 0.]], [0., 0.]),
              "training_history": training["history"], "jvp_qualification": training["qualification"],
              "synthetic_optimizer_steps": training["optimizer_steps"],
              "native_model_loads": 0, "native_optimizer_steps": 0,
              "native_rineya_qualified": False, "release_permitted": False}
    write_new(output/"report.json", report)
    (output/"INCOMPLETE.json").unlink()
    return report
