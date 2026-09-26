"""CPU evidence only: these small synthetic adapters are not the experiment model."""
import importlib.util
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from slcmi import checkpoint_resume
from slcmi.checkpoint_resume import fingerprint
from slcmi.training import train
from test_training_safety import TinyAdapter, assert_restored, row

spec = importlib.util.spec_from_file_location("baseline_training", HERE / "baseline_training.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class Interrupted(RuntimeError):
    pass


class StochasticAdapter(TinyAdapter):
    def __init__(self, *args, fail_train_call=None, fail_validation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_calls = 0
        self.fail_train_call = fail_train_call
        self.fail_validation = fail_validation

    def forward(self, **kwargs):
        out = super().forward(**kwargs)
        # Exercise all three CPU RNG streams, including during validation.
        factor = torch.rand(()) + random.random() + float(np.random.random()) + 1
        out.loss = out.loss * factor
        if torch.is_grad_enabled():
            self.train_calls += 1
            if self.train_calls == self.fail_train_call:
                raise Interrupted("mid accumulation")
        elif self.fail_validation:
            raise Interrupted("during validation after consuming RNG")
        return out


DATA = [row(1), row(2), row(0), row(0), row(-1), row(1), row(3), row(1)]
EVAL = [row(2), row(1)]


def identity(data=DATA, eval_set=EVAL, **overrides):
    return {
        "recipe_sha256": fingerprint({"fixture": "unchanged CPU tiny adapter recipe"}),
        "train_data_sha256": fingerprint(data), "eval_data_sha256": fingerprint(eval_set),
        "base_checkpoint_sha256": fingerprint("tiny fixture frozen one"),
        "adapter_initialization_sha256": fingerprint("tiny fixture LoRA ones"),
        **overrides,
    }


def options(path=None, **overrides):
    args = dict(device="cpu", max_steps=7, grad_accum=2, lr=1e-4,
                warmup_ratio=0.4, eval_set=EVAL, eval_every=2, log=lambda _: None,
                checkpoint_dir=path, checkpoint_fingerprints=identity() if path else None)
    args.update(overrides)
    return args


def seed():
    random.seed(311)
    np.random.seed(711)


def load(path):
    return torch.load(path / "latest.pt", map_location="cpu", weights_only=True)


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()
                if k not in {"elapsed_seconds", "wall_seconds", "steps_per_second", "applied_updates_per_second"}}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, tuple):
        return tuple(canonical(v) for v in value)
    return value


def exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    else:
        assert left == right


def stop_at(event, attempt):
    def callback(record):
        if record["event"] == event and record["step"] == attempt:
            raise Interrupted(f"after {event} {attempt}")
    return callback


def uninterrupted(path, dtype=torch.bfloat16):
    seed()
    model = StochasticAdapter(dtype=dtype).eval()
    result = train(model, DATA, **options(path))
    return model, result, load(path)


@pytest.mark.parametrize("boundary,step", [("optimizer_attempt", 1), ("optimizer_attempt", 2),
                                           ("optimizer_attempt", 3), ("optimizer_attempt", 7),
                                           ("validation", 2), ("validation", 7)])
def test_exact_resume_after_committed_update_skip_or_validation(tmp_path, boundary, step):
    full_model, full_result, full_state = uninterrupted(tmp_path / "full")
    seed()
    partial = StochasticAdapter().eval()
    with pytest.raises(Interrupted):
        train(partial, DATA, **options(tmp_path / "split", progress_callback=stop_at(boundary, step)))
    assert_restored(partial)
    saved = load(tmp_path / "split")
    assert saved["progress"]["completed_attempts"] == step
    if boundary == "optimizer_attempt" and step in (2, 7):
        assert saved["progress"]["pending_validation"] == step
    del partial
    # Model construction and unrelated process work consume RNG before restoration.
    random.random(); np.random.random(); torch.rand(9)
    resumed = StochasticAdapter().eval()
    events = []
    result = train(resumed, DATA, **options(tmp_path / "split", resume_from_checkpoint=True, progress_callback=events.append))
    exact(resumed.lora_A.weight, full_model.lora_A.weight)
    exact(canonical(result), canonical(full_result))
    exact(canonical(load(tmp_path / "split")), canonical(full_state))
    assert all(r["step"] > step for r in events if r["event"] == "optimizer_attempt")
    assert len(result["trajectory"]) == 7
    assert [v["step"] for v in result["validation"]] == [2, 4, 6, 7]
    assert result["steps_skipped_by_safety_check"] == 4  # two zero and two inf attempts
    assert_restored(resumed)


@pytest.mark.parametrize("fail_train_call", [1, 2, 3, 4, 9])
def test_mid_accumulation_restarts_from_prior_boundary_with_rng(tmp_path, fail_train_call):
    full_model, full_result, full_state = uninterrupted(tmp_path / "full")
    seed()
    partial = StochasticAdapter(fail_train_call=fail_train_call).eval()
    with pytest.raises(Interrupted, match="mid accumulation"):
        train(partial, DATA, **options(tmp_path / "split"))
    done = load(tmp_path / "split")["progress"]["completed_attempts"]
    assert done == (fail_train_call - 1) // 2
    resumed = StochasticAdapter().eval()
    result = train(resumed, DATA, **options(tmp_path / "split", resume_from_checkpoint=True))
    exact(resumed.lora_A.weight, full_model.lora_A.weight)
    exact(canonical(result), canonical(full_result))
    exact(canonical(load(tmp_path / "split")), canonical(full_state))


def test_validation_interruption_replays_validation_without_reapplying_update(tmp_path):
    full_model, full_result, full_state = uninterrupted(tmp_path / "full")
    seed()
    partial = StochasticAdapter(fail_validation=True).eval()
    with pytest.raises(Interrupted, match="during validation"):
        train(partial, DATA, **options(tmp_path / "split"))
    saved = load(tmp_path / "split")
    assert saved["progress"]["pending_validation"] == 2
    assert saved["progress"]["evals"] == []
    events = []
    resumed = StochasticAdapter().eval()
    result = train(resumed, DATA, **options(tmp_path / "split", resume_from_checkpoint=True, progress_callback=events.append))
    assert events[0]["event"] == "validation" and events[0]["step"] == 2
    assert events[1]["event"] == "optimizer_attempt" and events[1]["step"] == 3
    exact(resumed.lora_A.weight, full_model.lora_A.weight)
    exact(canonical(result), canonical(full_result))
    exact(canonical(load(tmp_path / "split")), canonical(full_state))


def test_actual_cpu_gradscaler_state_and_overflow_resume_are_exact(tmp_path, monkeypatch):
    original_scaler = torch.amp.GradScaler
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: original_scaler("cpu", init_scale=kwargs["init_scale"], enabled=True))
    full_model, full_result, full_state = uninterrupted(tmp_path / "full", dtype=torch.float16)
    seed()
    partial = StochasticAdapter(dtype=torch.float16).eval()
    with pytest.raises(Interrupted):
        train(partial, DATA, **options(tmp_path / "split", progress_callback=stop_at("optimizer_attempt", 3)))
    before = load(tmp_path / "split")
    assert before["scaler"]["scale"] == 2048
    assert before["progress"]["skipped_steps"] == 1
    assert before["progress"]["safety_skipped_steps"] == 1
    resumed = StochasticAdapter(dtype=torch.float16).eval()
    result = train(resumed, DATA, **options(tmp_path / "split", resume_from_checkpoint=True))
    assert result["steps_skipped_by_scaler"] == 2
    assert result["steps_skipped_by_safety_check"] == 2
    exact(resumed.lora_A.weight, full_model.lora_A.weight)
    exact(canonical(result), canonical(full_result))
    exact(canonical(load(tmp_path / "split")), canonical(full_state))


def test_checkpointing_preserves_original_live_loop_math(tmp_path):
    seed()
    original_model = StochasticAdapter().eval()
    args = options()
    for name in ("checkpoint_dir", "checkpoint_fingerprints"):
        del args[name]
    original_result = baseline.train(original_model, DATA, **args)
    model, result, _ = uninterrupted(tmp_path / "full")
    exact(model.lora_A.weight, original_model.lora_A.weight)
    exact(canonical(result), canonical(original_result))


@pytest.mark.parametrize("change", ["lr", "max_steps", "grad_accum", "eval_every", "seed", "data_order", "eval_order",
                                  "recipe_sha256", "base_checkpoint_sha256", "adapter_initialization_sha256"])
def test_changed_recipe_or_order_is_refused_before_model_mutation(tmp_path, change):
    uninterrupted(tmp_path / "full")
    data = list(DATA)
    args = options(tmp_path / "full", resume_from_checkpoint=True)
    if change in ("lr", "max_steps", "grad_accum", "eval_every", "seed"):
        args[change] = {"lr": 2e-4, "max_steps": 8, "grad_accum": 1, "eval_every": 3, "seed": 18}[change]
    elif change == "data_order":
        data.reverse()
        args["checkpoint_fingerprints"] = identity(data=data)
    elif change == "eval_order":
        args["eval_set"] = list(reversed(EVAL))
        args["checkpoint_fingerprints"] = identity(eval_set=args["eval_set"])
    else:
        args["checkpoint_fingerprints"] = identity(**{change: "f" * 64})
    model = StochasticAdapter().eval()
    original = model.lora_A.weight.detach().clone()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        train(model, data, **args)
    assert torch.equal(model.lora_A.weight, original)
    assert_restored(model)


def test_caller_data_hash_must_match_actual_data(tmp_path):
    with pytest.raises(ValueError, match="caller train data fingerprint"):
        train(StochasticAdapter().eval(), list(reversed(DATA)), **options(tmp_path / "bad"))
    assert not (tmp_path / "bad" / "latest.pt").exists()


def test_saved_model_tensors_are_only_trainable_lora_and_load_is_weights_only(tmp_path, monkeypatch):
    _, _, state = uninterrupted(tmp_path / "full")
    assert set(state["adapter"]) == {"lora_A.weight"}
    assert "frozen" not in state["adapter"]
    assert state["trainable_names"] == ["lora_A.weight"]
    assert state["optimizer"]["state"]
    assert state["scheduler"]["last_epoch"] == state["progress"]["applied_steps"]
    assert "torch_cpu" in state["rng"] and state["rng"]["torch_cuda"] == []
    original_load = torch.load
    observed = []
    def checked_load(*args, **kwargs):
        observed.append(kwargs)
        assert kwargs["weights_only"] is True
        assert kwargs["map_location"] == "cpu"
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", checked_load)
    train(StochasticAdapter().eval(), DATA, **options(tmp_path / "full", resume_from_checkpoint=True))
    assert len(observed) == 1


def test_failed_atomic_replace_keeps_previous_committed_checkpoint(tmp_path, monkeypatch):
    seed()
    original_replace = checkpoint_resume.os.replace
    calls = 0
    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replace failure")
        return original_replace(source, destination)
    monkeypatch.setattr(checkpoint_resume.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected replace"):
        train(StochasticAdapter().eval(), DATA, **options(tmp_path / "split"))
    assert load(tmp_path / "split")["progress"]["completed_attempts"] == 0
    assert list((tmp_path / "split").glob("*.tmp")) == []
    monkeypatch.setattr(checkpoint_resume.os, "replace", original_replace)
    resumed = StochasticAdapter().eval()
    result = train(resumed, DATA, **options(tmp_path / "split", resume_from_checkpoint=True))
    full, expected, state = uninterrupted(tmp_path / "full")
    exact(resumed.lora_A.weight, full.lora_A.weight)
    exact(canonical(result), canonical(expected))
    exact(canonical(load(tmp_path / "split")), canonical(state))


def test_existing_checkpoint_requires_explicit_resume(tmp_path):
    uninterrupted(tmp_path / "full")
    with pytest.raises(FileExistsError, match="explicitly resume"):
        train(StochasticAdapter().eval(), DATA, **options(tmp_path / "full"))


def test_completed_checkpoint_is_idempotent_and_does_not_emit_duplicates(tmp_path):
    model, result, saved = uninterrupted(tmp_path / "full")
    events = []
    restored = StochasticAdapter().eval()
    again = train(restored, DATA, **options(tmp_path / "full", resume_from_checkpoint=True, progress_callback=events.append))
    assert events == []
    assert restored.train_calls == 0
    exact(restored.lora_A.weight, model.lora_A.weight)
    exact(canonical(again), canonical(result))
    exact(load(tmp_path / "full"), saved)


@pytest.mark.parametrize("tamper", ["cursor", "applied_steps", "pending_validation", "applied_answer_tokens", "finite_nonzero_steps",
                                  "scheduler_epoch", "optimizer_lr", "cpu_rng", "scaler"])
def test_invalid_checkpoint_accounting_is_refused(tmp_path, tamper):
    uninterrupted(tmp_path / "full")
    payload = load(tmp_path / "full")
    if tamper == "scheduler_epoch":
        payload["scheduler"]["last_epoch"] = 999
    elif tamper == "optimizer_lr":
        payload["optimizer"]["param_groups"][0]["lr"] = 9.0
    elif tamper == "cpu_rng":
        payload["rng"]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
    elif tamper == "scaler":
        payload["scaler"] = {"scale": 999}
    else:
        payload["progress"][tamper] = 999
    torch.save(payload, tmp_path / "full" / "latest.pt")
    model = StochasticAdapter().eval()
    original = model.lora_A.weight.detach().clone()
    with pytest.raises(ValueError, match="checkpoint"):
        train(model, DATA, **options(tmp_path / "full", resume_from_checkpoint=True))
    assert torch.equal(model.lora_A.weight, original)


@pytest.mark.parametrize("interruption", ["interrupt", "mid_accumulation"])
def test_real_process_exit_and_fresh_process_resume_are_exact(tmp_path, interruption):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    def run(mode, directory, code=0):
        process = subprocess.run([sys.executable, str(HERE / "cpu_process_worker.py"), mode, str(directory)],
                                 cwd=HERE, env=env, capture_output=True, text=True, timeout=60)
        assert process.returncode == code, process.stdout + process.stderr
    run("uninterrupted", tmp_path / "full")
    run(interruption, tmp_path / "split", code=37)
    partial = load(tmp_path / "split")
    assert partial["progress"]["completed_attempts"] == (2 if interruption == "interrupt" else 1)
    run("resume", tmp_path / "split")
    exact(canonical(load(tmp_path / "full")), canonical(load(tmp_path / "split")))
    full = torch.load(tmp_path / "full" / "final_result.pt", weights_only=True)
    resumed = torch.load(tmp_path / "split" / "final_result.pt", weights_only=True)
    exact(canonical(full), canonical(resumed))


@pytest.mark.parametrize("edge", ["underflow", "growth_saturation"])
def test_scaler_validation_matches_actual_float32_numerical_edges(edge):
    initial_scale = 4096.0 if edge == "underflow" else float(2 ** 126)
    overflowing = edge == "underflow"
    attempts = 170 if overflowing else 4
    def setup():
        param = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([param], lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cpu", init_scale=initial_scale, growth_interval=1)
        return param, optimizer, scheduler, scaler
    param, optimizer, scheduler, scaler = setup()
    trajectory = []
    for _ in range(attempts):
        optimizer.zero_grad(set_to_none=True)
        before = scaler.get_scale()
        scaler.scale(param.sum()).backward()
        if overflowing:
            param.grad.fill_(float("inf"))
        scaler.step(optimizer)
        scaler.update()
        if not overflowing:
            scheduler.step()
        trajectory.append({"scaler_scale_before": before, "scaler_scale_after": scaler.get_scale(),
                           "step_skipped_by_scaler": overflowing, "optimizer_update_applied": not overflowing,
                           "lr_used": 1e-4, "lr": 1e-4})
    assert scaler.get_scale() == (0.0 if overflowing else float(2 ** 127))
    payload = {"progress": {"applied_steps": 0 if overflowing else attempts, "trajectory": trajectory},
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}
    fresh, fresh_optimizer, fresh_scheduler, fresh_scaler = setup()
    checkpoint_resume.validate_optimizer_states(payload, named_params=[("lora_A.weight", fresh)],
                                                optimizer=fresh_optimizer, scheduler=fresh_scheduler, scaler=fresh_scaler)
