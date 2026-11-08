"""Prompt-only projection/cap tests with a tiny analytic tokenizer."""
import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.prepare_fresh_data import prompt_projection, token_projection


class FakeTokenizer:
    def __init__(self, count=2): self.count = count
    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["role"] == "user"
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "CHAT:" + messages[0]["content"]
    def __call__(self, rendered): return {"input_ids": [7] * self.count}


def test_labels_are_projected_out_and_render_bound():
    raw = prompt_projection([{"case_id": "opaque", "prompt": "text", "gold": "secret", "family": "hidden"}])
    assert set(raw[0]) == {"case_id", "raw_prompt", "raw_prompt_sha256"}
    row = token_projection(raw, FakeTokenizer())[0]
    assert row["rendered_prompt"] == "CHAT:text"
    assert row["input_tokens"] == 2
    assert "gold" not in row and "family" not in row


def test_bound_exceeding_case_is_not_truncated():
    raw = prompt_projection([{"case_id": "opaque", "prompt": "text"}])
    with pytest.raises(ValueError): token_projection(raw, FakeTokenizer(351))
    raw[0]["gold"] = "secret"
    with pytest.raises(PermissionError): token_projection(raw, FakeTokenizer())


def test_duplicate_ids_are_not_removed():
    with pytest.raises(ValueError):
        prompt_projection([{"case_id": "same", "prompt": "a"}, {"case_id": "same", "prompt": "b"}])


def test_stale_original_prompt_hash_rejected_before_tokenizer():
    raw = prompt_projection([{"case_id": "opaque", "prompt": "original"}])
    raw[0]["raw_prompt"] = "changed"
    with pytest.raises(PermissionError):
        token_projection(raw, FakeTokenizer())
