from __future__ import annotations

from rineya_state_lab.training.tokenization import tokenize_conversation


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": [ord(char) % 251 + 1 for char in text]}


def test_assistant_only_labels_and_target_preservation() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "x" * 100},
        {"role": "assistant", "content": "answer"},
    ]
    result = tokenize_conversation(tokenizer, messages, max_length=32, max_target_tokens=16)
    assert len(result.input_ids) <= 32
    assert result.target_tokens > 0
    assert all(label == -100 for label in result.labels[: result.prompt_tokens])
    assert all(label != -100 for label in result.labels[result.prompt_tokens :])
    assert result.truncated_prompt_tokens > 0
