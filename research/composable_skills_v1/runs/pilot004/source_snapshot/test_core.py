import pytest
from .core import bank, exact, paired, rate_at, validate_banks


def test_paired_cannot_hide_forgetting_in_net_gain():
    result = paired({"a": True, "b": False, "c": False}, {"a": False, "b": True, "c": True})
    assert result["net"] == 1 and result["lost"] == 1 and result["retained"] == 0


def test_paired_requires_identical_instruments():
    with pytest.raises(ValueError):
        paired({"a": True}, {"b": True})


def test_no_json_boolean_numeric_or_unterminated_false_positive():
    assert not exact('{"A":true}', '{"A":1}', "ledger_replay")
    assert not exact('{"A":1.0}', '{"A":1}', "ledger_replay")
    assert not exact("UNKNOWN", "UNKNOWN", "binding", False)
    assert exact(' { "A" : 1 } ', '{"A":1}', "ledger_replay")


def test_schedule_is_token_based_sqrt_decay():
    assert rate_at(1, 10, 100) == .5
    assert rate_at(1, 80, 100) == 1
    assert rate_at(1, 90, 100) == pytest.approx(.1 + .9 * (1 - 2 ** -.5))
    assert rate_at(1, 100, 100) == pytest.approx(.1)
    with pytest.raises(ValueError):
        rate_at(1, 0, 100)


def test_world_pair_changes_answer_and_necessity_not_question():
    rows = bank(301, 2, "anchor")
    binding = [r for r in rows if r["family"] == "binding" and r["world_id"].endswith("0000")]
    assert len(binding) == 4
    assert binding[0]["reference"] != binding[2]["reference"]
    assert binding[1]["reference"] == binding[3]["reference"] == "UNKNOWN"
    assert len({r["prompt"][1]["content"].split("\n")[-1] for r in binding}) == 1


def test_bank_leakage_checks_reject_semantic_overlap():
    anchor = bank(301, 2, "anchor")
    dev = bank(302, 2, "dev")
    assert validate_banks(anchor, dev)["template_overlap"] == 0
    dev[0]["semantic_key"] = anchor[0]["semantic_key"]
    with pytest.raises(ValueError, match="semantic_key"):
        validate_banks(anchor, dev)


def test_binding_target_not_always_first_or_same_position():
    rows = bank(301, 28, "anchor")
    answerable = [r for r in rows if r["family"] == "binding" and r["answerable"] and not r["long"]]
    positions = {r["prompt"][1]["content"].split("\n")[0].split(". ").index(
        next(s for s in r["prompt"][1]["content"].split("\n")[0].split(". ") if s.startswith(r["reference"] + " is"))) for r in answerable}
    assert positions == {0, 1, 2, 3}


def test_independent_kl_has_no_teacher_gradient_and_learns_old_distribution():
    import torch
    from .pilot import kl_loss
    teacher = torch.tensor([[2., -1., 0.]], requires_grad=True)
    identical = teacher.detach().clone().requires_grad_()
    loss = kl_loss(identical, teacher)
    assert abs(loss.item()) < 1e-7
    loss.backward()
    assert teacher.grad is None
    assert identical.grad.abs().max() < 1e-7
    student = torch.tensor([[-1., 2., 0.]], requires_grad=True)
    loss = kl_loss(student, teacher)
    loss.backward()
    assert loss > 0 and student.grad[0, 0] < 0 and student.grad[0, 1] > 0


def test_incomplete_baseline_cannot_authorize_training(tmp_path):
    from .core import write_new
    from .pilot import require_complete
    write_new(tmp_path / "baseline/qualification.json", {"passed": True})
    with pytest.raises(FileNotFoundError):
        require_complete(tmp_path, "baseline")
    write_new(tmp_path / "baseline/COMPLETE.json", {"stage": "baseline", "artifacts": {}})
    write_new(tmp_path / "baseline/supervisor.json", {"status": "FAILED", "child_returncode": 1})
    with pytest.raises(ValueError):
        require_complete(tmp_path, "baseline")


def test_duplicate_predictions_cannot_silently_collapse():
    from .pilot import correctness
    with pytest.raises(ValueError):
        correctness([{"id": "a", "correct": True}, {"id": "a", "correct": False}])


def test_native_qwen_selected_logit_api_matches_cached_hidden_path():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(cfg).eval()
    ids = torch.tensor([[1, 4, 2, 7, 3]])
    positions = torch.tensor([2, 3])
    with torch.no_grad():
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, positions]
        cached = torch.nn.functional.linear(hidden, model.get_output_embeddings().weight)
        actual = model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits[0]
    torch.testing.assert_close(cached, actual, atol=1e-5, rtol=0)
