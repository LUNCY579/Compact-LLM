from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.gpu_native import (
    GIB, encode_prefix, evaluate_case, forward_logits, one_shot_anchors,
    require_cuda_tensor, validate_admission, install_gpu_timing,
)


class Tokenizer:
    def __init__(self):
        self.rendered = []
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "USER:" + messages[0]["content"]
    def __call__(self, rendered, **kwargs):
        assert kwargs == {"return_tensors": "pt"}
        self.rendered.append(rendered)
        return {"input_ids": torch.tensor([[1, 2]])}
    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return " ".join(str(token) for token in ids if token != 9)


class FakeModel(torch.nn.Module):
    def __init__(self, sequence=(3, 4, 9), cache=False):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.sequence, self.seen, self.cache = sequence, [], cache
        self.config = SimpleNamespace(hidden_size=4, intermediate_size=8, num_attention_heads=1,
                                      head_dim=4, num_key_value_heads=1, num_hidden_layers=3, vocab_size=10)
        self.generation_config = SimpleNamespace(eos_token_id=[9, 8])
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        assert use_cache is False and logits_to_keep == 1 and torch.equal(attention_mask, torch.ones_like(input_ids))
        assert input_ids.device == attention_mask.device == self.parameter.device
        self.seen.append(input_ids.clone())
        for layer in self.model.layers:
            layer(input_ids)
        token = self.sequence[min(input_ids.shape[1] - 2, len(self.sequence) - 1)]
        logits = torch.zeros(1, 1, 10)
        logits[0, -1, token] = 10
        return SimpleNamespace(logits=logits, past_key_values=object() if self.cache else None)


def flags(text, ids):
    return {"repetition_loop": False, "script_corruption": False, "control_count": 0,
            "replacement_count": 0, "invalid": False}


def metrics(base, first, topk, gold):
    assert gold is None
    return {"logit_cosine": 1., "kl_base_to_arm": 0.}


def case(cap=24):
    return {"case_id": "fake", "family": "analytic", "scenario_id": "parent",
            "prompt": "question", "max_new_tokens": cap,
            "scorer": {"type": "exact", "answers": ["3 4"]}, "format_scorer": None}


def wrapper():
    return SimpleNamespace(records=[], calls=0, min_rms_ratio=1., max_rms_ratio=1.,
                           capture=True, extra_layer_applications=0)


def score(text, spec):
    return {"score": int(text in spec["answers"])}


CONFIG = {"max_new_tokens": 24, "max_prompt_seconds": 900, "top_k": 10}


def test_fake_greedy_eos_full_prefix_and_device_tensor_creation_match_contract():
    model, tokenizer, calls = FakeModel(), Tokenizer(), []
    with torch.inference_mode(), one_shot_anchors(model):
        row, first = evaluate_case(model, tokenizer, case(), CONFIG, wrapper(), score, flags, metrics,
                                   enforce_device=False, synchronize=lambda: None, forward_sink=calls.append)
    assert row["token_ids"] == [3, 4, 9] and row["response"] == "3 4" and row["score"] == 1
    assert row["termination"] == "eos" and row["gold_nll"] is None and row["gold_answer"] is None
    assert [prefix.tolist() for prefix in model.seen] == [[[1, 2]], [[1, 2, 3]], [[1, 2, 3, 4]]]
    assert row["processed_token_positions"] == 9 and row["full_model_calls_including_gold_nll"] == 3
    assert len([event for event in calls if event["event"] == "FORWARD_COMPLETED"]) == 3
    assert len([event for event in calls if event["event"] == "FORWARD_STARTED"]) == 3
    assert tokenizer.rendered == ["USER:question"]
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_case_cap_uses_no_extra_forward_and_retains_truncation():
    model = FakeModel(sequence=(3, 4, 5))
    row, _ = evaluate_case(model, Tokenizer(), case(2), CONFIG, wrapper(), score, flags, metrics,
                            enforce_device=False)
    assert row["token_ids"] == [3, 4] and len(model.seen) == 2
    assert row["termination"] == "token_cap" and row["truncated"] and not row["safety_stopped"]


def test_native_device_guard_and_returned_cache_fail_before_false_pass():
    with pytest.raises(RuntimeError, match="cuda:0"):
        require_cuda_tensor(torch.ones(1))
    with pytest.raises(RuntimeError, match="cuda:0"):
        forward_logits(FakeModel(), torch.tensor([[1, 2]]))
    with pytest.raises(RuntimeError, match="cache"):
        forward_logits(FakeModel(cache=True), torch.tensor([[1, 2]]), enforce_device=False)


def test_corruption_termination_does_not_read_target_or_compute_another_prefix():
    class TargetForbidden(dict):
        def __getitem__(self, key):
            if key in ("gold", "answer", "canonical_answer"):
                raise AssertionError("future target read")
            return super().__getitem__(key)
    def corrupted(text, ids):
        return {**flags(text, ids), "script_corruption": True}
    model = FakeModel()
    row, _ = evaluate_case(model, Tokenizer(), TargetForbidden(case()), CONFIG, wrapper(), score,
                           corrupted, metrics, enforce_device=False)
    assert row["safety_stopped"] and row["termination"] == "script_or_unicode_corruption" and len(model.seen) == 1


def test_anchor_failure_restores_hooks_and_resource_contract_has_no_cpu_fallback(tmp_path):
    model = FakeModel()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
    with pytest.raises(IndexError), one_shot_anchors(model):
        pass
    assert not model._forward_pre_hooks and not model._forward_hooks
    source = tmp_path / "canonical"
    binding = {"shapes": {"max_forward_prefix_tokens": 2248}, "checkpoint_files": {"model": "hash"}}
    admission = {"status": "READY_FOR_GPU_FP32_DISPATCH", "device": "cuda:0", "dtype": "float32",
                 "attn_implementation": "eager", "gpu_allocated_ceiling_bytes": 5 * GIB,
                 "gpu_global_free_reserve_bytes": GIB, "gpu_initial_free_floor_bytes": 6 * GIB,
                 "worker_host_available_floor_bytes": 6 * GIB, "worker_rss_ceiling_bytes": 6 * GIB,
                 "source": str(source), "training": False, "native_test": False,
                 "maximum_shapes": binding["shapes"], "checkpoint_files": binding["checkpoint_files"]}
    validate_admission(admission, binding, source)
    admission["device"] = "cpu"
    with pytest.raises(RuntimeError, match="contract"):
        validate_admission(admission, binding, source)


@pytest.mark.parametrize("cap", [2, 24])
def test_successor_greedy_body_matches_pinned_generation_only_inherited_body(monkeypatch, cap):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"))
    monkeypatch.syspath_prepend(str(root / "experiments/anchor_routed_middle_recurrence_v1/src"))
    from armr.evaluation import evaluate_case as inherited
    from armr.metrics import output_flags, logit_metrics
    from rineya_cross_scale.evaluation.scoring import score_response
    from experiments.rineya_taps_adaptive_recurrence_v1.diagnostics import generation_only_evaluator
    old_model, new_model = FakeModel(), FakeModel()
    config = {**CONFIG, "max_rss_gib": 6}
    old, old_logits = generation_only_evaluator(inherited)(old_model, Tokenizer(), case(cap), config, wrapper())
    new, new_logits = evaluate_case(new_model, Tokenizer(), case(cap), config, wrapper(), score_response,
                                    output_flags, logit_metrics, enforce_device=False)
    for key in ("token_ids", "response", "termination", "score", "format_score", "gold_nll", "gold_answer",
                "eos_emitted", "truncated", "safety_stopped", "processed_token_positions",
                "generation_dense_matmul_flops_estimate", "invalid"):
        assert old[key] == new[key]
    assert torch.equal(old_logits, new_logits)


def test_gpu_timing_proxy_preserves_math_and_does_not_nest_over_prefix_resets():
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.control import BlockRecurrence, TAPSConfig
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_control import Host
    model, syncs = Host(), []
    ids = torch.tensor([[1, 4]])
    with torch.inference_mode(), BlockRecurrence(model, (3, 4), config=TAPSConfig()) as reference:
        expected = model(ids, use_cache=False)
    wrapper = BlockRecurrence(model, (3, 4), config=TAPSConfig())
    with torch.inference_mode(), wrapper:
        install_gpu_timing(wrapper, lambda: syncs.append(1), require_cuda=False)
        for _ in range(2):
            actual = model(ids, use_cache=False)
            assert torch.equal(actual, expected)
        assert len(syncs) == 22 and wrapper.gpu_conditioning_checks == 6
    assert not model._forward_pre_hooks and not wrapper._gpu_timing_ready
    assert wrapper._gpu_conditioning is None
    assert all(not layer._forward_pre_hooks for layer in model.model.layers)


def test_cache_object_guard_and_failed_forward_attempt_are_recorded():
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.control import BlockRecurrence
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_control import Host
    model = Host()
    with torch.inference_mode(), BlockRecurrence(model, (3, 4)) as wrapper:
        with pytest.raises(RuntimeError, match="cache"):
            model(torch.tensor([[1, 4]]), use_cache=False, past_key_values=object())
        assert wrapper.forward_count == 0
    events = []
    with pytest.raises(RuntimeError, match="cache"):
        evaluate_case(FakeModel(cache=True), Tokenizer(), case(), CONFIG, globals()["wrapper"](), score, flags, metrics,
                      enforce_device=False, forward_sink=events.append)
    assert [event["event"] for event in events] == ["FORWARD_STARTED", "FORWARD_FAILED"]
    assert events[-1]["completed"] is False and events[-1]["partial_compute"] == "UNKNOWN"


def test_partial_loader_resource_sample_survives_without_returned_model(monkeypatch):
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import gpu_native
    monkeypatch.setattr(gpu_native.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(gpu_native, "gpu_resources", lambda: {"gpu_peak_allocated_bytes": 123})
    assert gpu_native.final_resource_snapshot() == {"gpu_peak_allocated_bytes": 123}
    monkeypatch.setattr(gpu_native.torch.cuda, "is_initialized", lambda: False)
    assert gpu_native.final_resource_snapshot() is None
