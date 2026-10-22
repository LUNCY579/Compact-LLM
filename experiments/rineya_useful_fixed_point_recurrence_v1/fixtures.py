"""Public analytic/synthetic fixtures, NOT Rineya benchmarks or paper results."""
from __future__ import annotations

import copy
import platform
from pathlib import Path
import torch
from torch import nn
from .core import InjectionInterface, finite_recurrence, kv_geometry
from .diagnostics import trajectory_metrics
from .training import TrainingBudget, fit_interface
from .protocol import new_run, write_json


def run_fixtures(out: Path) -> dict:
    out = new_run(out)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cases = {}
    start = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[0.5, 0.5]])
    functions = {
        "contractive_to_target": lambda h: .5*h+.5*target,
        "contractive_to_wrong": lambda h: .5*h-.5*target,
        "identity_wrong": lambda h: h,
        "oscillation": lambda h: -h,
        "zero_scaled": lambda h: .5*h+.5*target,
    }
    for name, core in functions.items():
        h, records = start.clone(), []
        alpha = 0.0 if name == "zero_scaled" else 1.0
        for _ in range(8):
            proposal = core(h)
            y = h + alpha*(proposal-h)
            records.append(trajectory_metrics(h, h, proposal, y, target, start))
            h = y
        cases[name] = {"trajectory": records, "target_distance": float(torch.linalg.vector_norm(h-target)),
                       "synthetic_target_reached": bool(torch.linalg.vector_norm(h-target) < .02)}
    fits = []
    for seed in (17, 29):
        torch.manual_seed(seed)
        backbone = nn.Linear(8, 8, bias=False).eval().requires_grad_(False)
        # Near-identity frozen fixture, deliberately not an LLM.
        with torch.no_grad():
            backbone.weight.copy_(.8*torch.eye(8))
        states, anchors = torch.randn(12, 8), torch.randn(12, 8)
        targets = .75*states+.25*anchors
        initial = InjectionInterface(8, rank=2, mode="additive")
        initial_state = copy.deepcopy(initial.state_dict())
        for mode in ("additive", "orthogonal"):
            interface = InjectionInterface(8, rank=2, mode=mode)
            interface.load_state_dict(initial_state)
            def loss(step):
                prediction = finite_recurrence(backbone, states, anchors, steps=1, interface=interface)
                return (prediction-targets).square().mean(), states.shape[0]
            before = float(loss(0)[0].detach())
            result = fit_interface(interface, backbone, loss, train_split="TRAIN",
                                   budget=TrainingBudget(max_steps=12, learning_rate=.01))
            result.update(seed=seed, mode=mode, loss_before=before, loss_after=float(loss(0)[0].detach()),
                          parameter_count=sum(p.numel() for p in interface.parameters()))
            fits.append(result)
    result = {"status": "CPU_FIXTURES_PASS_NATIVE_NOT_RUN", "experiment_id": "FP-RINEYA-001",
              "evidence_kind": "analytic systems and synthetic regression only",
              "native_checkpoint_loads": 0, "native_optimizer_steps": 0,
              "synthetic_optimizer_steps": sum(r["optimizer_steps"] for r in fits),
              "python": platform.python_version(), "torch": torch.__version__,
              "trajectories": cases, "matched_training": fits,
              "hypothetical_kv_28L_R2": kv_geometry(28, 2, 2)}
    # Assert the instrument distinguishes useful, wrong and suppressed stationarity.
    assert cases["contractive_to_target"]["synthetic_target_reached"]
    assert not cases["identity_wrong"]["synthetic_target_reached"]
    assert cases["zero_scaled"]["trajectory"][-1]["suppressed_update_fraction"] == 1
    write_json(out / "fixtures.json", result)
    return result
