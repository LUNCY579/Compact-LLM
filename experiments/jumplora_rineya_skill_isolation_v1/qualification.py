"""JLR-QUAL-001 mechanical qualification and bounded CPU micro-run."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .checkpoint import assert_state_equal, clone_state, state_digest
from .jump_lora import (
    EarlyFrozenMaskLoRA,
    JumpLoRALinear,
    JumpSchedule,
    LoRALinear,
    MagnitudePrunedLoRA,
    build_adapter,
    deterministic_topk_mask,
    symmetric_jump_relu,
    threshold_for_mask,
)
from .metrics import support_counts
from .resources import assert_no_sealed_paths, resource_snapshot, reset_cuda_peak_stats


QUALIFICATION_ID = "JLR-QUAL-001"
EXPERIMENT_ID = "jumplora_rineya_skill_isolation_v1"


@dataclass(frozen=True)
class MicroRun:
    arm: str
    seed: int
    final_loss: float
    support: dict[str, int | float]
    trainable_parameters: int
    state_digest: str
    output_digest: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _digest_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], check=True,
                                capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    model_config = root / "Rineya_Published_Cores" / "Rineya-Pulse-1.6.0-devfreeze" / "config.json"
    experiment_config = Path(__file__).with_name("config.json")
    return {
        "repository_url": "https://github.com/LUNCY579/ChatBubble.git",
        "base_commit": _git(root, "rev-parse", "origin/main"),
        "branch": _git(root, "branch", "--show-current"),
        "code_commit_at_qualification": _git(root, "rev-parse", "HEAD"),
        "target_remote_branch": "research/jumplora-rineya-skill-isolation-v1",
        "target_remote_sha_before_work": "7d9b4fd450bc56405f6cec41459a7e83df7958ae",
        "model_checkpoint": {
            "status": "NOT_LOADED",
            "checkpoint_identity": None,
            "checkpoint_sha256": None,
            "reference_config_path": str(model_config.relative_to(root)).replace("\\", "/"),
            "reference_config_sha256": _sha256_file(model_config),
        },
        "tokenizer": {"status": "NOT_LOADED", "identity": None, "sha256": None},
        "datasets": {
            "status": "NOT_ACCESSED",
            "train": {"status": "NOT_ACCESSED", "count": None, "sha256": None},
            "dev": {"status": "NOT_ACCESSED", "count": None, "sha256": None},
            "sealed": {"status": "FORBIDDEN_AND_NOT_ACCESSED", "count": None, "sha256": None},
        },
        "precision": "torch.float32 synthetic qualification tensors",
        "seeds": [1, 3, 4, 5, 17, 23, 20260925],
        "config_path": str(experiment_config.relative_to(root)).replace("\\", "/"),
        "config_sha256": _sha256_file(experiment_config),
        "runtime": {"python": sys.version, "torch": torch.__version__},
        "train_eval_counts": {
            "rineq_train": None,
            "rineq_dev": None,
            "rineq_sealed": None,
            "synthetic_examples_per_micro_run": 24,
            "synthetic_optimizer_steps_per_arm": 10,
        },
    }


def _check(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        detail = fn()
        return {"name": name, "status": "PASS", "detail": detail}
    except Exception as exc:  # qualification receipt must identify the first failure
        return {"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def _base_template(seed: int = 20260925) -> dict[str, torch.Tensor]:
    seed_everything(seed)
    base = nn.Linear(8, 8, bias=False)
    return clone_state(base)


def _make_base(state: dict[str, torch.Tensor]) -> nn.Linear:
    base = nn.Linear(8, 8, bias=False)
    base.load_state_dict(state)
    return base


def _micro_data(seed: int = 20260925) -> tuple[torch.Tensor, torch.Tensor]:
    seed_everything(seed + 1)
    x = torch.randn(24, 8)
    target_delta = torch.randn(8, 8) * 0.08
    base = torch.zeros(8, 8)
    y = x @ (base + target_delta).T
    return x, y


def train_micro_arm(arm: str, base_state: dict[str, torch.Tensor], *, seed: int,
                    reference_active_count: int | None = None,
                    steps: int = 10) -> tuple[MicroRun, LoRALinear]:
    """Train only a tiny synthetic linear task; never loads a checkpoint/dataset."""
    seed_everything(seed)
    base = _make_base(base_state)
    schedule = JumpSchedule(2, 8, epsilon=1.0e-3)
    adapter = build_adapter(base, arm, rank=2, alpha=4.0, schedule=schedule)
    x, y = _micro_data()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.15)
    paper_initial_count = min(adapter.dense_delta().numel(),
                              adapter.lora_A.numel() + adapter.lora_B.numel())
    if isinstance(adapter, JumpLoRALinear):
        adapter.set_step(0)
    for step in range(steps):
        if isinstance(adapter, JumpLoRALinear):
            if step == adapter.schedule.s_start:
                adapter.initialize_threshold(paper_initial_count)
            adapter.set_step(step)
        if isinstance(adapter, EarlyFrozenMaskLoRA) and step == 2:
            count = reference_active_count if reference_active_count is not None else paper_initial_count
            adapter.set_frozen_mask(deterministic_topk_mask(adapter.dense_delta(), count))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(adapter(x, step), y)
        if not bool(torch.isfinite(loss)):
            raise AssertionError(f"nonfinite loss in {arm} at step {step}")
        loss.backward()
        for name, parameter in adapter.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError(f"nonfinite gradient in {arm}: {name}")
        optimizer.step()
    if isinstance(adapter, MagnitudePrunedLoRA) and not isinstance(adapter, EarlyFrozenMaskLoRA):
        if reference_active_count is None:
            raise ValueError("J1 requires a frozen J3 reference active count")
        adapter.set_frozen_mask(deterministic_topk_mask(adapter.dense_delta(), reference_active_count))
    output = adapter(x, steps)
    support = support_counts(adapter.support_mask())
    return MicroRun(
        arm=str(arm).upper(), seed=seed, final_loss=float(loss.detach()),
        support=support,
        trainable_parameters=sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        state_digest=state_digest(adapter), output_digest=_digest_tensor(output),
    ), adapter


def check_dense_state_equivalence() -> dict[str, Any]:
    seed_everything(1)
    base = nn.Linear(5, 7, bias=False)
    adapter = JumpLoRALinear(base, rank=2, alpha=4.0, schedule=JumpSchedule(2, 5), threshold=0.2)
    with torch.no_grad():
        adapter.lora_A.normal_()
        adapter.lora_B.normal_()
    x = torch.randn(4, 5)
    adapter.set_step(0)
    dense = adapter.base(x) + torch.nn.functional.linear(x, adapter.dense_delta())
    actual = adapter(x)
    torch.testing.assert_close(actual, dense, rtol=0.0, atol=0.0)
    return {"gamma": adapter.gamma, "max_abs_error": 0.0}


def check_deterministic_masks() -> dict[str, Any]:
    values = torch.tensor([[4.0, -4.0, 1.0], [2.0, 3.0, -3.0]])
    first = deterministic_topk_mask(values, 4)
    second = deterministic_topk_mask(values, 4)
    assert torch.equal(first, second)
    assert int(first.sum()) == 4
    return {"active": int(first.sum()), "mask": first.to(dtype=torch.int8).tolist()}


def check_threshold_ste() -> dict[str, Any]:
    # Values at both signed threshold bands make the tau gradient hand-checkable.
    x = torch.tensor([0.55, -0.55, 0.1], requires_grad=True)
    tau = torch.tensor(0.5, requires_grad=True)
    y = symmetric_jump_relu(x, tau, epsilon=0.2)
    y.sum().backward()
    # Positive x contributes -tau/epsilon and negative x contributes +tau/epsilon.
    expected = -0.5 / 0.2 + 0.5 / 0.2
    assert abs(float(tau.grad) - expected) < 1.0e-6
    torch.testing.assert_close(y, torch.tensor([0.55, -0.55, 0.0]), rtol=0.0, atol=0.0)
    return {"tau_grad": float(tau.grad), "expected_tau_grad": expected,
            "x_grad": x.grad.tolist()}


def check_matched_density() -> dict[str, Any]:
    base_state = _base_template()
    j3_run, _ = train_micro_arm("J3", base_state, seed=17)
    j1_run, _ = train_micro_arm("J1", base_state, seed=17,
                                reference_active_count=int(j3_run.support["active"]))
    assert j1_run.support["active"] == j3_run.support["active"]
    assert j1_run.support["total"] == j3_run.support["total"]
    return {"j1": j1_run.support, "j3": j3_run.support}


def check_merge_unmerge_and_serialization() -> dict[str, Any]:
    seed_everything(3)
    base = nn.Linear(4, 4, bias=False)
    adapter = LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():
        adapter.lora_B.normal_(0.0, 0.1)
    x = torch.randn(3, 4)
    before = adapter(x).detach()
    state_before = clone_state(adapter)
    with adapter.merged():
        merged = adapter(x).detach()
    torch.testing.assert_close(before, adapter(x), rtol=0.0, atol=0.0)
    torch.testing.assert_close(before, merged, rtol=1.0e-5, atol=1.0e-6)
    assert_state_equal(adapter, state_before)
    blob = io.BytesIO()
    torch.save(adapter.state_dict(), blob)
    blob.seek(0)
    copy = LoRALinear(nn.Linear(4, 4, bias=False), rank=2, alpha=4.0)
    copy.load_state_dict(torch.load(blob, weights_only=True))
    torch.testing.assert_close(adapter(x), copy(x), rtol=0.0, atol=0.0)
    return {"state_digest": state_digest(adapter), "roundtrip": "PASS"}


def check_baseline_load_and_disabled_path() -> dict[str, Any]:
    base_state = _base_template(4)
    first = _make_base(base_state)
    before = state_digest(first)
    checkpoint = io.BytesIO()
    torch.save(first.state_dict(), checkpoint)
    checkpoint.seek(0)
    second = nn.Linear(8, 8, bias=False)
    second.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert state_digest(second) == before
    adapter = LoRALinear(_make_base(base_state), rank=2, alpha=4.0)
    x = torch.randn(2, 8)
    torch.testing.assert_close(adapter(x), first(x), rtol=0.0, atol=0.0)
    return {"baseline_digest": before, "disabled_path": "exact"}


def check_counts_and_finiteness() -> dict[str, Any]:
    base = nn.Linear(6, 5, bias=False)
    dense = LoRALinear(base, rank=2, alpha=4.0)
    jump = JumpLoRALinear(nn.Linear(6, 5, bias=False), rank=2, alpha=4.0,
                          schedule=JumpSchedule(1, 3), threshold=0.1)
    assert sum(p.numel() for p in dense.parameters() if p.requires_grad) == 2 * (6 + 5)
    assert sum(p.numel() for p in jump.parameters() if p.requires_grad) == 2 * (6 + 5) + 1
    assert all(bool(torch.isfinite(p).all()) for p in jump.parameters())
    return {"dense_trainable": 22, "jump_trainable": 23}


def check_reproducibility() -> dict[str, Any]:
    base_state = _base_template(5)
    first, _ = train_micro_arm("J3", base_state, seed=23)
    second, _ = train_micro_arm("J3", base_state, seed=23)
    assert asdict(first) == asdict(second)
    return {"state_digest": first.state_digest, "output_digest": first.output_digest}


def check_paired_ids_and_sealed_guard() -> dict[str, Any]:
    from .metrics import validate_paired_ids
    rows = [{"id": f"dev-{i}"} for i in range(4)]
    assert validate_paired_ids({"J0": rows, "J1": list(rows)}) == [r["id"] for r in rows]
    try:
        validate_paired_ids({"J0": rows, "J1": rows[:-1]})
    except ValueError:
        pass
    else:
        raise AssertionError("paired ID mismatch was not rejected")
    assert_no_sealed_paths(["experiments/jumplora_rineya_skill_isolation_v1/tests"])
    try:
        assert_no_sealed_paths(["data/sealed/dev.jsonl"])
    except AssertionError:
        pass
    else:
        raise AssertionError("sealed path was not rejected")
    return {"paired_ids": "PASS", "sealed_access": "NONE"}


def run_qualification(output_path: Path | None = None) -> dict[str, Any]:
    """Run JLR-QUAL-001 and optionally write its immutable JSON receipt."""
    torch.set_num_threads(1)
    reset_cuda_peak_stats()
    before = resource_snapshot()
    checks = [
        _check("gamma_dense_state_equivalence", check_dense_state_equivalence),
        _check("deterministic_masks", check_deterministic_masks),
        _check("threshold_ste_gradient", check_threshold_ste),
        _check("matched_density_j1_j3", check_matched_density),
        _check("merge_unmerge_serialization", check_merge_unmerge_and_serialization),
        _check("baseline_load_and_disabled_path", check_baseline_load_and_disabled_path),
        _check("parameter_counts_and_finiteness", check_counts_and_finiteness),
        _check("seed_reproducibility", check_reproducibility),
        _check("paired_ids_and_no_sealed_paths", check_paired_ids_and_sealed_guard),
    ]
    after = resource_snapshot()
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "qualification_id": QUALIFICATION_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "result_classification": "MECHANICAL_ONLY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_occurred": True,
        "training_scope": "bounded synthetic CPU micro-run only; no Rineya/full experiment training",
        "sealed_access": False,
        "sealed_paths_touched": [],
        "model_loaded": False,
        "source_weight_files_read": False,
        "checks": checks,
        "resource_before": before,
        "resource_after": after,
        "resource_peak_measured": {
            "cuda_peak_memory_allocated_bytes": after.get("cuda_peak_memory_allocated_bytes"),
            "cuda_peak_memory_reserved_bytes": after.get("cuda_peak_memory_reserved_bytes"),
            "host_ram_low_water_mark_bytes": min(value for value in (
                before.get("host_ram_available_bytes"), after.get("host_ram_available_bytes")
            ) if value is not None) if any(value is not None for value in (
                before.get("host_ram_available_bytes"), after.get("host_ram_available_bytes")
            )) else None,
        },
        "paper_source": "https://arxiv.org/abs/2604.16171v3",
        "provenance": _provenance(),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
                               encoding="utf-8", newline="\n")
    return receipt


if __name__ == "__main__":
    result = run_qualification()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
