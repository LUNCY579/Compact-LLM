from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.control import BlockRecurrence, TAPSConfig
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.gpu_native import evaluate_case
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.observer import readout_stats
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.opposed_gate_generation import (
    GenerationSession, PilotHandle, _hook_snapshot, minimal_probability,
)
from experiments.anchor_routed_middle_recurrence_v1.src.armr.metrics import output_flags


class Layer(torch.nn.Module):
    def __init__(self, index):
        super().__init__(); self.index = index; self.calls = 0; self.fail = False

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("forced layer failure")
        return hidden_states + ({12: 1., 13: 2.}.get(self.index, 0.))


class Host(torch.nn.Module):
    """Analytic operator changes the first channel; the head reads that value."""
    def __init__(self, sequences=None, cache=False, nonfinite=False):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer(i) for i in range(28)])
        self.config = SimpleNamespace(layer_types=["full_attention"] * 28, hidden_size=4,
            intermediate_size=8, num_attention_heads=1, head_dim=4, num_key_value_heads=1,
            num_hidden_layers=28, vocab_size=10)
        self.generation_config = SimpleNamespace(eos_token_id=[9, 8])
        self.seen, self.sequences = [], sequences
        self.cache, self.nonfinite = cache, nonfinite
        self.eval().requires_grad_(False)

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        assert use_cache is False and logits_to_keep == 1
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
        positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
        for layer in self.model.layers:
            hidden = layer(hidden, position_ids=positions, position_embeddings=None, attention_mask=attention_mask)
        k = int(float(hidden[0, 0, 0]) > 5.)
        self.seen.append((input_ids[0].tolist(), k))
        if self.sequences is not None:
            sequence = self.sequences[k]
            token = sequence[min(input_ids.shape[1] - 2, len(sequence) - 1)]
        elif input_ids.shape[1] == 2:
            token = 4 if k else 3
        else:
            token = ({4: 6, 6: 9} if k else {3: 5, 5: 9}).get(int(input_ids[0, -1]), 7)
        logits = torch.zeros(1, 1, 10)
        logits[0, -1, token] = float("nan") if self.nonfinite else 10.
        return SimpleNamespace(logits=logits, past_key_values=object() if self.cache else None)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "USER:" + messages[0]["content"]

    def __call__(self, rendered, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return " ".join(str(x) for x in ids if x not in (8, 9))


IDS = torch.tensor([[1, 2]])


def session(model=None, api_cap=100, guard=None, **kwargs):
    return GenerationSession(model or Host(), Tokenizer(), run_id="fixture", source_sha256="a" * 64,
        resident_state_sha256="b" * 64, tokenizer_sha256="c" * 64, api_cap=api_cap,
        guard=guard or (lambda: {}), synchronize=lambda: None, enforce_device=False, **kwargs)


def reference(model, k, cap):
    case = {"case_id": "logging_only", "family": "fixture", "scenario_id": "fixture",
            "prompt": "original", "max_new_tokens": cap, "scorer": {"unused": True}, "format_scorer": None}
    with torch.inference_mode(), BlockRecurrence(model, extra_loops=k, config=TAPSConfig(scheduler_mode="unit")) as wrapper:
        return evaluate_case(model, Tokenizer(), case, {"max_new_tokens": cap, "max_prompt_seconds": 900, "top_k": 3},
            wrapper, lambda *a: {"score": None}, output_flags, lambda *a: {}, enforce_device=False,
            synchronize=lambda: None)


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("use_pilot", [False, True])
def test_selected_own_path_matches_inherited_greedy_and_split_call_accounting(k, use_pilot):
    expected, old_first = reference(Host(), k, 8)
    s = session(); before = _hook_snapshot(s.model)
    pilot = s.make_pilot(IDS) if use_pilot else None
    row, first = s.generate(IDS, k=k, pilot=pilot)
    for key in ("token_ids", "response", "termination", "eos_emitted", "truncated", "safety_stopped"):
        assert row[key] == expected[key]
    assert torch.equal(first, old_first)
    assert row["token_ids"] == ([4, 6, 9] if k else [3, 5, 9])
    assert row["full_model_calls"] == 3 + (k if use_pilot else 0)
    assert row["completed_token_positions"] == 9 + (2 * k if use_pilot else 0)
    assert row["extra_layer_applications"] == 6 * k
    assert s.completed_forwards == row["full_model_calls"] and s.completed_positions == row["completed_token_positions"]
    assert _hook_snapshot(s.model) == before
    assert all(p.grad is None and not p.requires_grad for p in s.model.parameters())
    selected = s.model.seen[1:] if use_pilot else s.model.seen
    if k:
        assert [x[0] for x in selected] == [[1, 2], [1, 2, 4], [1, 2, 4, 6]]
    else:
        assert [x[0] for x in selected] == ([[1, 2, 3], [1, 2, 3, 5]] if use_pilot else [[1, 2], [1, 2, 3], [1, 2, 3, 5]])


def test_pilot_not_a_selected_token_or_shared_between_opposed_arms():
    s = session(); p0, p1 = s.make_pilot(IDS), s.make_pilot(IDS)
    assert p0 is not p1 and p0.logits_sha256 == p1.logits_sha256 and p0.probability == p1.probability
    row0, _ = s.generate(IDS, k=0, pilot=p0)
    row1, _ = s.generate(IDS, k=1, pilot=p1)
    assert row0["token_ids"] == [3, 5, 9] and row1["token_ids"] == [4, 6, 9]
    assert row0["full_model_calls"] + row1["full_model_calls"] == 7
    assert len(s.model.seen) == 7


@pytest.mark.parametrize("k", [0, 1])
def test_first_eos_includes_one_generated_token_and_no_continuation(k):
    s = session(Host(sequences={0: (9,), 1: (9,)})); pilot = s.make_pilot(IDS)
    row, _ = s.generate(IDS, k=k, pilot=pilot)
    assert row["token_ids"] == [9] and row["eos_emitted"] and row["response"] == ""
    assert row["full_model_calls"] == 1 + k and row["selected_generation_physical_calls"] == k


@pytest.mark.parametrize("k", [0, 1])
def test_cap_uses_no_unused_next_forward_and_preserves_repetition_stop(k):
    s = session(Host(sequences={0: (3, 4, 5), 1: (4, 5, 6)})); pilot = s.make_pilot(IDS)
    row, _ = s.generate(IDS, k=k, pilot=pilot, max_new_tokens=2)
    assert len(row["token_ids"]) == 2 and row["truncated"] and not row["safety_stopped"]
    assert row["full_model_calls"] == 2 + k
    repeated = session(Host(sequences={0: (3,), 1: (4,)}))
    row, _ = repeated.generate(IDS, k=k, max_new_tokens=8)
    assert row["termination"] == "repetition_loop" and row["safety_stopped"] and len(row["token_ids"]) == 6


def test_probability_exact_observer_expression_equality_and_finite_rejection():
    z = torch.tensor([0., 0.])
    assert minimal_probability(z) == .5 == readout_stats(z, unknown_id=0)["top1_probability"]
    z = torch.linspace(-3, 2, 151936)
    assert minimal_probability(z) == readout_stats(z)["top1_probability"]
    for bad in (torch.tensor([float("nan"), 0.]), torch.tensor([float("inf"), 0.])):
        with pytest.raises(FloatingPointError): minimal_probability(bad)
    s = session(Host(nonfinite=True)); before = _hook_snapshot(s.model)
    with pytest.raises(FloatingPointError): s.make_pilot(IDS)
    assert s.failed_forwards == 1 and _hook_snapshot(s.model) == before and not s._pilots


def test_forged_serialized_reused_wrong_prefix_and_session_handles_rejected():
    s = session(); pilot = s.make_pilot(IDS)
    forged = replace(pilot)
    with pytest.raises(ValueError, match="forged"):
        s.generate(IDS, k=0, pilot=forged)
    other = session()
    with pytest.raises(ValueError, match="session"):
        other.generate(IDS, k=0, pilot=pilot)
    with pytest.raises(ValueError, match="binding"):
        s.generate(torch.tensor([[1, 3]]), k=0, pilot=pilot)
    with pytest.raises(ValueError, match="consumed"):
        s.generate(IDS, k=0, pilot=pilot)
    fresh = s.make_pilot(IDS); s.generate(IDS, k=0, pilot=fresh)
    with pytest.raises(ValueError, match="consumed"):
        s.generate(IDS, k=1, pilot=fresh)


def test_changed_pilot_metadata_model_state_and_frozen_firewall():
    s = session(); pilot = s.make_pilot(IDS)
    object.__setattr__(pilot, "probability", 0.01)
    with pytest.raises(ValueError, match="binding"):
        s.generate(IDS, k=0, pilot=pilot)
    s = session(); pilot = s.make_pilot(IDS)
    with torch.no_grad(): s.model.parameter.add_(1.)
    with pytest.raises(RuntimeError, match="binding"):
        s.generate(IDS, k=1, pilot=pilot)
    s = session(); s.model.train()
    with pytest.raises(RuntimeError, match="frozen"):
        s.make_pilot(IDS)
    s = session(); pilot = s.make_pilot(IDS)
    s.model.config.abstain_logit_bias = 100.
    with pytest.raises(RuntimeError, match="binding"):
        s.generate(IDS, k=0, pilot=pilot)
    s = session(); pilot = s.make_pilot(IDS)
    s.tokenizer = Tokenizer()
    with pytest.raises(RuntimeError, match="binding"):
        s.generate(IDS, k=0, pilot=pilot)


def test_layer_and_trajectory_failures_restore_and_keep_partial_unknown():
    s = session(); pilot = s.make_pilot(IDS); before = _hook_snapshot(s.model)
    s.model.model.layers[13].fail = True
    with pytest.raises(RuntimeError, match="layer failure"):
        s.generate(IDS, k=1, pilot=pilot)
    assert _hook_snapshot(s.model) == before and s.failed_forwards == 1
    assert [x for x in s.events if x["event"] == "FORWARD_FAILED"][-1]["partial_compute"] == "UNKNOWN"
    with pytest.raises(ValueError, match="consumed"):
        s.generate(IDS, k=0, pilot=pilot)
    def fail(_): raise RuntimeError("forced trajectory failure")
    s = session(trajectory_sink=fail); before = _hook_snapshot(s.model)
    with pytest.raises(RuntimeError, match="trajectory"):
        s.generate(IDS, k=1)
    assert _hook_snapshot(s.model) == before


def test_returned_cache_and_native_device_guards_fail_without_false_complete():
    s = session(Host(cache=True)); before = _hook_snapshot(s.model)
    with pytest.raises(RuntimeError, match="cache"):
        s.generate(IDS, k=0)
    assert _hook_snapshot(s.model) == before and s.completed_forwards == 0
    with pytest.raises(RuntimeError, match="cuda:0"):
        GenerationSession(Host(), Tokenizer(), run_id="native", source_sha256="a" * 64,
            resident_state_sha256="b" * 64, tokenizer_sha256="c" * 64, api_cap=5,
            guard=lambda: {}, synchronize=lambda: None)


def test_label_free_api_shape_caps_and_actual_head_index_bounds():
    s = session()
    with pytest.raises(TypeError): s.generate(IDS, k=0, scorer={"gold": "answer"})
    with pytest.raises(ValueError): s.make_pilot({"input_ids": IDS, "gold": "x"})
    with pytest.raises(ValueError): s.generate(IDS, k=True)
    with pytest.raises(ValueError): s.generate(IDS, k=0, max_new_tokens=9)
    with pytest.raises(ValueError): s.generate(torch.tensor([[1, 10]]), k=0)
    with pytest.raises(ValueError): s.generate(torch.ones(1, 356, dtype=torch.long), k=0, max_new_tokens=8)
    assert s.API_attempts == 0


def test_split_depth_flops_include_intact_pilot_and_every_actual_head():
    s = session(); pilot = s.make_pilot(IDS)
    row, _ = s.generate(IDS, k=1, pilot=pilot)
    c = s.model.config
    P = 2 * (4 * 4 + 2 * 4 * 4 + 4 * 4 + 3 * 4 * 8); H = 2 * 4 * 10
    assert row["generation_dense_matmul_flops_estimate"] == P * (28 * 2 + 30 * 9) + H * 4
    assert row["pilot_dense_matmul_flops_estimate"] == P * 28 * 2 + H
    assert row["generation_dense_matmul_flops_estimate"] == row["pilot_dense_matmul_flops_estimate"] + row["selected_dense_matmul_flops_estimate"]


def test_unit_does_not_stop_on_discarded_pilot_eos():
    s = session(Host(sequences={0: (9,), 1: (4, 6, 9)})); p = s.make_pilot(IDS)
    row, _ = s.generate(IDS, k=1, pilot=p)
    assert row["token_ids"] == [4, 6, 9] and row["full_model_calls"] == 4


def test_api_cap_and_postforward_resource_failure_keep_actual_counts_and_restore():
    s = session(api_cap=1); p = s.make_pilot(IDS); before = _hook_snapshot(s.model)
    with pytest.raises(RuntimeError, match="API cap"):
        s.generate(IDS, k=1, pilot=p)
    assert s.API_attempts == s.completed_forwards == 1 and len(s.model.seen) == 1
    assert _hook_snapshot(s.model) == before
    model = Host()
    def guard():
        if model.seen: raise MemoryError("after completed forward")
        return {}
    s = session(model, guard=guard); before = _hook_snapshot(model)
    with pytest.raises(MemoryError, match="completed"):
        s.generate(IDS, k=0)
    assert s.completed_forwards == 1 and s.failed_forwards == 0 and s.completed_positions == 2
    assert _hook_snapshot(model) == before
    complete = [x for x in s.events if x["event"] == "FORWARD_COMPLETED"][-1]
    assert complete["resources"] is None and complete["post_forward_guard_error"]
