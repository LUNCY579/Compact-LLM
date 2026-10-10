import json
import math
from types import SimpleNamespace

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.control import BlockRecurrence, TAPSConfig
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.observer import (
    PassiveBlock, last_token_stats, observed_block, readout_stats,
)
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import probe_native as probe
from experiments.anchor_routed_middle_recurrence_v1.src.armr.recurrence import FixedBlock


class Layer(torch.nn.Module):
    def __init__(self, index):
        super().__init__(); self.index = index; self.calls = 0

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        return hidden_states * (.97 + self.index * .001) + self.index * .001


class Host(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(64).reshape(16, 4).float() / 64)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer(i) for i in range(28)])
        self.config = SimpleNamespace(layer_types=["full_attention"] * 28)
        self.eval().requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        hidden = self.embed(input_ids)
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0)
        for layer in self.model.layers:
            hidden = layer(hidden, position_ids=pos, attention_mask=attention_mask)
        return SimpleNamespace(logits=hidden @ self.embed.weight.T, past_key_values=None)


def snapshot(model):
    return [("forward" in layer.__dict__, layer.__dict__.get("forward"), len(layer._forward_pre_hooks),
             "_taps_owner" in layer.__dict__) for layer in model.model.layers], len(model._forward_pre_hooks)


@pytest.mark.parametrize("extra", [0, 1])
def test_observer_is_passive_and_stages_precede_or_follow_actual_extra(extra):
    model = Host(); ids = torch.tensor([[1, 2, 3]]); events = []
    before = snapshot(model)
    with torch.inference_mode(), BlockRecurrence(model, extra_loops=extra, config=TAPSConfig(scheduler_mode="unit")):
        baseline = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    start_calls = model.model.layers[12].calls
    def sink(value):
        events.append((value, model.model.layers[12].calls - start_calls))
    with torch.inference_mode(), observed_block(model, extra_loops=extra, sample_id="input", sink=sink) as block:
        observed = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    assert torch.equal(baseline, observed)
    assert snapshot(model) == before and block._observer_previous is None
    assert [event["stage"] for event, _ in events] == ["ORDINARY_PRE_EXTRA"] + (["UNIT_PROPOSAL_POST_EXTRA"] if extra else [])
    assert [calls for _, calls in events] == [1] + ([2] if extra else [])
    for event, _ in events:
        assert event["target_independent"] and event["preceding_gold_tokens"] == 0
        assert not event["teacher_conditioned"] and "gold" not in event and "scorer" not in event
    assert block.observer_seconds > 0


def test_observer_class_does_not_override_forward_entry_or_restoration():
    assert "_forward" not in PassiveBlock.__dict__ and "_entry" not in PassiveBlock.__dict__
    assert "__enter__" not in PassiveBlock.__dict__ and "__exit__" not in PassiveBlock.__dict__


def test_shared_worker_invoker_keeps_real_forward_inside_wrapper_context(monkeypatch):
    # CPU analytic host: replace only GPU timing and recorder, not the operator.
    monkeypatch.setattr(probe, "install_gpu_timing", lambda *args: None)
    model = Host(); ids = torch.tensor([[1, 2]]); events = []; values = []
    before = probe.hook_snapshot(model)
    class FixtureRecorder:
        def forward(self, host, prefix, context):
            mode = context["mode"]
            if mode not in ("INTACT", "RESTORED_INTACT"):
                assert any("_taps_owner" in layer.__dict__ or "_armr_owner" in layer.__dict__ for layer in host.model.layers)
            return host(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=False).logits[0, -1]
    for mode in probe.QUAL_MODES:
        value, _ = probe.invoke_prefix(model, ids, mode, {"case_id": "fixture"}, FixtureRecorder(),
                                       lambda: {}, lambda: None, FixedBlock, events.append)
        values.append(value)
        assert probe.hook_snapshot(model) == before
    assert all(torch.equal(values[0], values[i]) for i in (1, 2, 6))
    assert all(torch.equal(values[3], values[i]) for i in (4, 5))
    assert [event["stage"] for event in events] == ["ORDINARY_PRE_EXTRA", "ORDINARY_PRE_EXTRA", "UNIT_PROPOSAL_POST_EXTRA"]
    def failure(_): raise RuntimeError("shared forced failure")
    with pytest.raises(RuntimeError, match="forced failure"):
        probe.invoke_prefix(model, ids, "OBSERVED_UNIT", {"case_id": "fixture"}, FixtureRecorder(),
                            lambda: {}, lambda: None, FixedBlock, failure)
    assert probe.hook_snapshot(model) == before


def test_forced_observer_exception_restores_hooks_owner_and_state():
    model = Host(); ids = torch.tensor([[1, 2]]); before = snapshot(model)
    with torch.inference_mode():
        intact = model(input_ids=ids, use_cache=False).logits
    def fail(_):
        raise RuntimeError("forced observer failure")
    with pytest.raises(RuntimeError, match="forced observer"):
        with torch.inference_mode(), observed_block(model, extra_loops=1, sample_id="x", sink=fail) as block:
            model(input_ids=ids, use_cache=False)
    assert snapshot(model) == before and block._observer_previous is None
    with torch.inference_mode():
        assert torch.equal(intact, model(input_ids=ids, use_cache=False).logits)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("kwargs", [{"use_cache": True}, {"use_cache": False, "past_key_values": object()}])
def test_cache_rejects_before_any_layer_or_feature(kwargs):
    model = Host(); events = []; before = snapshot(model)
    with pytest.raises(RuntimeError, match="cache"):
        with torch.inference_mode(), observed_block(model, extra_loops=1, sample_id="x", sink=events.append):
            model(input_ids=torch.tensor([[1]]), **kwargs)
    assert not events and sum(layer.calls for layer in model.model.layers) == 0 and snapshot(model) == before


def test_prefix_moments_and_last_delta_reset_no_cross_input_state():
    model = Host(); events = []
    with torch.inference_mode(), observed_block(model, extra_loops=1, sample_id="same", sink=events.append):
        for ids in (torch.tensor([[1]]), torch.tensor([[2, 3]])):
            model(input_ids=ids, use_cache=False)
    assert [event["iteration"] for event in events] == [0, 1, 0, 1]
    assert events[0]["statistics"]["previous_delta_l2"] is None
    assert events[2]["statistics"]["previous_delta_l2"] is None


def test_zero_denominators_are_explicit_null_and_nonfinite_is_rejected():
    zero = torch.zeros(1, 2, 4)
    stats, delta = last_token_stats(zero, zero, zero)
    assert stats["state_cosine"] is None and stats["delta_over_old_l2"] is None
    assert stats["state_cosine_denominator"] == 0 and stats["finite"]
    stats, _ = last_token_stats(zero, zero, zero, delta)
    assert stats["delta_cosine_previous"] is None and stats["delta_over_previous_l2"] is None
    with pytest.raises(FloatingPointError):
        last_token_stats(zero, torch.full_like(zero, math.nan), zero)


def test_readout_uses_actual_unknown_rival_including_special_and_is_shift_invariant():
    logits = torch.tensor([-.2, 1., 3., 2.], dtype=torch.float32); saved = logits.clone()
    value = readout_stats(logits, unknown_id=3)
    assert value["rival_token_id"] == 2 and value["UNKNOWN_minus_rival_margin"] == -1
    assert value["top1_token_id"] == 2 and value["top2_margin"] == 1
    shifted = readout_stats(logits + 10., unknown_id=3)
    assert shifted["UNKNOWN_minus_rival_margin"] == value["UNKNOWN_minus_rival_margin"]
    assert shifted["entropy_nats"] == pytest.approx(value["entropy_nats"], abs=1e-6)
    assert torch.equal(logits, saved) and not value["logits_changed"]
    with pytest.raises(FloatingPointError):
        readout_stats(torch.tensor([0., math.inf, 1.]), unknown_id=0)


def fixture_population():
    bank, tokens = [], []
    for i in range(192):
        prompt = "input " + str(i)
        bank.append({"case_id": str(i), "split": "DEVELOPMENT", "raw_prompt": prompt,
                     "raw_prompt_sha256": probe.text_sha(prompt)})
        tokens.append({**bank[-1], "rendered_prompt": prompt, "rendered_prompt_sha256": probe.text_sha(prompt),
                       "input_token_ids": [1, 2], "input_tokens": 2})
    return bank, tokens


def test_population_is_complete_original_only_and_labels_fail_closed():
    bank, tokens = fixture_population(); rows = probe.project_population(bank, tokens)
    assert len(rows) == 192 and "gold" not in rows[0] and "category" not in rows[0]
    with pytest.raises(RuntimeError, match="192"):
        probe.project_population(bank[:-1], tokens)
    bank[0]["gold"] = "2"
    with pytest.raises(PermissionError, match="label"):
        probe.project_population(bank, tokens)
    del bank[0]["gold"]; tokens[0]["split"] = "OPEN_VALIDATION"
    with pytest.raises(RuntimeError, match="split"):
        probe.project_population(bank, tokens)


def test_actual_prompt_only_packet_schema_before_any_asset_loader(monkeypatch):
    def forbidden_loader(*args):
        raise AssertionError("model/source loader must not be invoked by metadata reader")
    monkeypatch.setattr(probe, "admitted_source", forbidden_loader)
    token_packet = json.loads((probe.EXPORT / "DEVELOPMENT_TOKEN_BINDING.json").read_text())
    admission = {"source": token_packet["canonical_source"]}
    for key, filename, value in (
        ("dev_bank", "DEVELOPMENT_BANK.json", probe.BANK_SHA),
        ("dev_token_binding", "DEVELOPMENT_TOKEN_BINDING.json", probe.TOKENS_SHA),
        ("dev_export_manifest", "MANIFEST.json", probe.EXPORT_SHA),
    ):
        admission[key + "_path"] = str(probe.EXPORT / filename)
        admission[key + "_sha256"] = value
    rows = probe.read_population(admission)
    assert len(rows) == 192 and sum(row["prefix_tokens"] for row in rows) == 24608
    assert max(row["prefix_tokens"] for row in rows) == 232
    assert all(not {"gold", "category", "parent_id", "answerability"}.intersection(row) for row in rows)
    admission["dev_bank_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="export"):
        probe.read_population(admission)


def test_exact_counts_and_full_forecast_refuse_incomplete_cost_or_budget():
    counts = probe.count_contract()
    assert counts["qualification_full_attempts"] == 7 * 8 + 7 + 2 == 65
    assert counts["model_API_attempts"] == 65 + 2 + 192 * 2 == 451
    assert counts["completed_forwards"] == 65 - 1 + 384 == 448
    assert counts["completed_prefix_positions"] == 7 * 295 + 8 * 357 + 2 * 24608 == 54137
    assert probe.complete_forecast([.1], 100)["admitted"]
    assert not probe.complete_forecast([2.], 100)["admitted"]
    for times in ([], [math.nan], [0.]):
        with pytest.raises(RuntimeError):
            probe.complete_forecast(times, 0)
    with pytest.raises(RuntimeError):
        probe.complete_forecast([.1], math.nan)


def test_parent_clock_rejects_missing_future_nonfinite_or_wrong_budget(monkeypatch):
    admission = {"hard_deadline_seconds": 900, "internal_reserve_seconds": 120,
                 "global_finalization_reserve_seconds": 1800}
    monkeypatch.setattr(probe.time, "perf_counter", lambda: 200.)
    monkeypatch.delenv(probe.CLOCK_ENV, raising=False)
    with pytest.raises(RuntimeError): probe.parent_clock(admission)
    for value in ("nan", "inf", "201", "-1000", "invalid"):
        monkeypatch.setenv(probe.CLOCK_ENV, value)
        with pytest.raises(RuntimeError): probe.parent_clock(admission)
    monkeypatch.setenv(probe.CLOCK_ENV, "100")
    assert probe.parent_clock(admission) == 100
    with pytest.raises(RuntimeError): probe.parent_clock({**admission, "internal_reserve_seconds": 0})


def test_actual_inherited_source_closure_all_read_git_bound_dependencies():
    # Exactly the source/config/receipts read by admitted_source; no assets loaded.
    paths = [probe.REPO / relative for relative in probe.INHERITED_METADATA]
    for root in (probe.ARMR / "src", probe.ARMR / "scripts", probe.LAB / "src"):
        paths.extend(root.rglob("*.py"))
    hashes = {path.relative_to(probe.REPO).as_posix(): probe.sha(path) for path in paths}
    assert any(relative.startswith("experiments/anchor_routed_middle_recurrence_v1/scripts/") for relative in hashes)
    assert len(probe.INHERITED_METADATA) == 3
    probe.source_closure(hashes)
    wrong = dict(hashes)
    first = next(iter(probe.INHERITED_METADATA)); wrong[first] = "0" * 64
    with pytest.raises(RuntimeError, match="closure"):
        probe.source_closure(wrong)


def test_closure_rejects_unlisted_metadata_and_dataset_without_opening(monkeypatch):
    # Existing .gitignore proves the branch rejects before calling its hash reader.
    def forbidden_read(path):
        raise AssertionError("unadmitted path must not be read")
    monkeypatch.setattr(probe, "sha", forbidden_read)
    with pytest.raises(RuntimeError, match="closure"):
        probe.source_closure({".gitignore": "0" * 64})
    # Admission of scripts does not admit a similarly located arbitrary JSON.
    with pytest.raises(RuntimeError, match="closure"):
        probe.source_closure({"experiments/anchor_routed_middle_recurrence_v1/AGENTS.md": "0" * 64})
    # Synthetic unlisted paths: avoid any filesystem access to dataset surfaces.
    monkeypatch.setattr(probe.Path, "resolve", lambda self, **kwargs: self)
    for relative in ("Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/unlisted_fixture.jsonl",
                     "experiments/anchor_routed_middle_recurrence_v1/configs/unlisted_fixture.json"):
        with pytest.raises(RuntimeError, match="closure"):
            probe.source_closure({relative: "0" * 64})


def test_recorder_preserves_failed_attempt_unknown_and_precompute_cache(monkeypatch):
    class MemoryJournal:
        def __init__(self): self.rows = []
        def append(self, row): self.rows.append(row)
    journal = MemoryJournal(); recorder = probe.Recorder(journal, probe.time.perf_counter(), lambda: {}, lambda: None)
    def failure(*args, **kwargs): raise RuntimeError("expected partial")
    monkeypatch.setattr(probe, "forward_logits", failure)
    with pytest.raises(RuntimeError, match="partial"):
        recorder.forward(None, torch.tensor([[1, 2]]), {"phase": "QUALIFICATION"})
    assert recorder.counts()["completed_forwards"] == 0 and recorder.API == 1
    assert journal.rows[-1]["partial_compute"] == "UNKNOWN" and "traceback" in journal.rows[-1]
    model = Host()
    with torch.inference_mode(), observed_block(model, extra_loops=1, sample_id="cache", sink=None) as block:
        recorder.reject_cache(model, torch.tensor([[1]]), {"use_cache": False, "past_key_values": object()}, block)
    assert recorder.cache == 1 and journal.rows[-1]["model_compute_started"] is False


def test_recorder_counts_completed_compute_when_later_resource_guard_refuses(monkeypatch):
    rows = []
    journal = SimpleNamespace(append=rows.append)
    checks = []
    def guard():
        checks.append(1)
        if len(checks) > 1: raise MemoryError("after-forward reserve")
        return {}
    recorder = probe.Recorder(journal, probe.time.perf_counter(), guard, lambda: None)
    monkeypatch.setattr(probe, "forward_logits", lambda *a, **k: torch.tensor([1., 2.]))
    with pytest.raises(MemoryError, match="reserve"):
        recorder.forward(None, torch.tensor([[1, 2]]), {"phase": "TASK"})
    assert recorder.completed == 1 and recorder.positions == 2 and recorder.partial == 0
    assert rows[-1]["event"] == "FORWARD_COMPLETED" and rows[-1]["resources"] is None
    assert rows[-1]["post_forward_guard_error"] == "after-forward reserve"
