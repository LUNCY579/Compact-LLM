from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.control import (
    BlockRecurrence, TAPSScheduler, TAPSConfig,
)
from experiments.rineya_taps_adaptive_recurrence_v1.integration import (
    BlockRecurrence as ReferenceBlock,
)


class Layer(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i, self.seen, self.fail = i, [], False

    def forward(self, hidden_states, **kwargs):
        self.seen.append(kwargs)
        if self.fail:
            raise RuntimeError("injected layer failure")
        return hidden_states * (.9 + .01 * self.i) + .01 * self.i


class Host(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 4)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(128).reshape(32, 4).float() / 128)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer(i) for i in range(6)])
        self.config = SimpleNamespace(layer_types=["full_attention"] * 6)
        self.eval().requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embed(input_ids)
        positions = torch.arange(input_ids.shape[1])
        for layer in self.model.layers:
            hidden = layer(hidden, position_ids=positions, attention_mask=attention_mask)
        return hidden


RANDOM = TAPSConfig(scheduler_mode="random", random_pool=(.8, .8, 1., 1.2), seed=41)


def scheduler_stream(config=RANDOM, arm="A5", name="case"):
    scheduler = TAPSScheduler(config, arm_id=arm)
    outputs = []
    for prefix in range(30):
        scheduler.reset_moments()
        delta = torch.full((1, 1 + prefix % 3, 4), .5)
        first, event = scheduler.observe(delta, (name,), force_unit=True)
        assert float(first[0]) == 1 and scheduler.state.step == 1
        alpha, event = scheduler.observe(delta, (name,))
        assert scheduler.state.step == 2 and int(event["random_draw_index"][0]) == prefix + 1
        outputs.append(float(alpha[0]))
    return outputs, scheduler


def test_random_stream_survives_prefix_reset_and_binds_actual_seed_arm_sample():
    values, _ = scheduler_stream()
    assert values == scheduler_stream()[0] and len(set(values)) > 1
    assert values != scheduler_stream(replace(RANDOM, seed=42))[0]
    assert values != scheduler_stream(arm="other")[0]
    assert values != scheduler_stream(name="other")[0]


def test_full_reset_restarts_stream_but_prefix_reset_forbids_new_example():
    values, scheduler = scheduler_stream()
    scheduler.reset_moments()
    with pytest.raises(ValueError, match="identity"):
        scheduler.observe(torch.ones(1, 4), ("different",))
    scheduler.reset()
    for prefix, expected in enumerate(values):
        scheduler.reset_moments()
        alpha, _ = scheduler.observe(torch.ones(1, 4), ("case",))
        assert float(alpha[0]) == expected


def test_rejected_delta_does_not_consume_rng_and_force_unit_consumes_no_draw():
    scheduler = TAPSScheduler(RANDOM, arm_id="A5")
    with pytest.raises(FloatingPointError):
        scheduler.observe(torch.full((1, 4), float("nan")), ("case",))
    assert scheduler.random_draws == []
    for _ in range(3):
        scheduler.observe(torch.ones(1, 4), ("case",), force_unit=True)
    assert scheduler.random_draws == [0]
    alpha, _ = scheduler.observe(torch.ones(1, 4), ("case",))
    assert float(alpha[0]) == scheduler_stream()[0][0]


def test_random_batch_stream_is_permutation_and_order_independent():
    config = RANDOM
    batch = TAPSScheduler(config, arm_id="A5")
    reverse = TAPSScheduler(config, arm_id="A5")
    separate = [TAPSScheduler(config, arm_id="A5") for _ in range(2)]
    for _ in range(8):
        for scheduler in (batch, reverse, *separate):
            scheduler.reset_moments()
        alpha, _ = batch.observe(torch.ones(2, 4), ("a", "b"))
        rev, _ = reverse.observe(torch.ones(2, 4), ("b", "a"))
        parts = torch.cat([s.observe(torch.ones(1, 4), (name,))[0]
                           for s, name in zip(separate, ("a", "b"))])
        assert torch.equal(alpha, rev.flip(0)) and torch.equal(alpha, parts)


@pytest.mark.parametrize("config", [TAPSConfig(), TAPSConfig(scheduler_mode="unit"),
                                      TAPSConfig(scheduler_mode="fixed", fixed_alpha=.8)])
@pytest.mark.parametrize("extra", [0, 1])
def test_reference_numerics_and_policy_independent_zero_budget(config, extra):
    model = Host()
    with torch.inference_mode():
        for prefix in (torch.tensor([[1, 4]]), torch.tensor([[1, 4, 6]])):
            with ReferenceBlock(model, (3, 4), extra, config=config):
                expected = model(prefix, use_cache=False)
            with BlockRecurrence(model, (3, 4), extra, config=config):
                actual = model(prefix, use_cache=False)
            assert torch.equal(actual, expected)
        prefix = torch.tensor([[1, 4]])
        baseline = model(prefix, use_cache=False)
        with BlockRecurrence(model, (3, 4), 0, config=RANDOM) as wrapper:
            assert torch.equal(model(prefix, use_cache=False), baseline)
            assert wrapper._controller.random_draws == [0]


def test_wrapper_random_prefix_stream_reset_moments_positions_and_reentry():
    model, events = Host(), []
    wrapper = BlockRecurrence(model, (3, 4), 1, config=RANDOM, arm_id="A5",
                              sample_id="case", sink=events.append)
    def evaluate():
        start = len(events)
        with torch.inference_mode(), wrapper:
            for i in range(30):
                ids = torch.tensor([[1, 4] + [6] * (i % 3)])
                mask = torch.ones_like(ids)
                model(ids, attention_mask=mask, use_cache=False)
                assert wrapper.scheduler.state.step == 2
                for index in (3, 4):
                    first, extra = model.model.layers[index].seen[-2:]
                    assert first["position_ids"] is extra["position_ids"]
                    assert first["attention_mask"] is extra["attention_mask"]
        return [e["alpha"] for e in events[start:] if e["iteration"] == 1]
    values = evaluate()
    assert values == scheduler_stream()[0] == evaluate()
    assert wrapper.random_binding == {"seed": 41, "arm_id": "A5", "sample_id": "case", "prefix_stream": True}
    assert wrapper._controller.state.mean is None


def test_layer_failure_restores_host_and_reentry_reproduces_first_random_draw():
    model = Host()
    original = model.model.layers[4].forward
    wrapper = BlockRecurrence(model, (3, 4), 1, config=RANDOM, arm_id="A5", sample_id="case")
    model.model.layers[4].fail = True
    with pytest.raises(RuntimeError, match="injected"):
        with torch.inference_mode(), wrapper:
            model(torch.tensor([[1, 4]]), use_cache=False)
    assert model.model.layers[4].forward == original and not model._forward_pre_hooks
    assert all("_taps_owner" not in layer.__dict__ for layer in model.model.layers)
    model.model.layers[4].fail = False
    with torch.inference_mode(), wrapper:
        model(torch.tensor([[1, 4]]), use_cache=False)
        assert wrapper.records[-1]["alpha"] == scheduler_stream()[0][0]


def test_inherited_cache_and_overlap_guards_remain_active():
    model = Host()
    with torch.inference_mode(), BlockRecurrence(model, (3, 4), 1) as wrapper:
        with pytest.raises(RuntimeError, match="cache"):
            model(torch.tensor([[1, 4]]), use_cache=True)
        assert wrapper.calls == 0
        with pytest.raises(RuntimeError, match="overlap"):
            with BlockRecurrence(model, (4, 5), 1):
                pass


def binding_fixture(tmp_path, monkeypatch):
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import binding
    lab = tmp_path / "lab"
    data = lab / "data/evaluation"
    data.mkdir(parents=True)
    for name, receipt_name in (("mergedev_fast.jsonl", "HISTORY"),
                               ("supa_layer_ablation_smoke.jsonl", "SMOKE_HISTORY")):
        case = {"case_id": name, "family": "choice", "scenario_id": "parent-" + name,
                "prompt": "Answer YES or NO", "prompt_sha256": binding.text_sha("Answer YES or NO"),
                "max_new_tokens": 3, "scorer": {"type": "choice", "value": "YES", "allowed": ["YES", "NO"]}}
        path = data / name
        path.write_text(json.dumps(case) + "\n", encoding="utf-8")
        receipt = tmp_path / (receipt_name + ".json")
        receipt.write_text(json.dumps({"dataset_sha256": binding.sha(path)}), encoding="utf-8")
        monkeypatch.setattr(binding, receipt_name, receipt)
    monkeypatch.setattr(binding, "LAB", lab)
    return binding, data


def test_metadata_binding_shape_formula_and_exposure_flags(tmp_path, monkeypatch):
    binding, _ = binding_fixture(tmp_path, monkeypatch)
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "TEMPLATE:" + messages[0]["content"]
        def __call__(self, rendered, **kwargs):
            assert rendered == "TEMPLATE:Answer YES or NO" and kwargs == {"return_tensors": "pt"}
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    scorer = SimpleNamespace(score_response=lambda response, config: {"score": float(response == config["value"])})
    rows, datasets, audits = binding.bind_cases(Tokenizer(), {"max_new_tokens": 24}, scorer)
    assert {row["split"] for row in rows} == {"DEV", "SMOKE"}
    for row in rows:
        assert row["max_forward_prefix_tokens"] == 7 and row["maximum_model_calls"] == 3
        assert row["maximum_processed_prefix_positions"] == 18
        assert row["audit"]["human_review"] == "NOT_REVIEWED" and not row["audit"]["solver_verified"]
        assert row["audit"]["target_strings_found_in_prompt"] == ["YES"]
        assert row["audit"]["prompt_copy_score"] == 0
    assert datasets["DEV"]["items"] == 1 and len(audits["normalized_exact_prompt_duplicates"]) == 1


def test_metadata_binding_refuses_changed_data_before_tokenizer_access(tmp_path, monkeypatch):
    binding, data = binding_fixture(tmp_path, monkeypatch)
    with (data / "mergedev_fast.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    class NoTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise AssertionError("unadmitted data reached tokenizer")
    with pytest.raises(RuntimeError, match="data changed"):
        binding.bind_cases(NoTokenizer(), {"max_new_tokens": 24}, None)
