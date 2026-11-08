"""Identical response-token prefixes across roles, with no target retokenization."""
from __future__ import annotations

from .contracts import Problem, Task, require

TEMPLATE_VERSION = "assistant-privilege-v1"
TRANSITION = "\nNow solve the original problem independently. Continue your own attempt:\n"


class Views:
    def __init__(self, tokenizer, eos_ids: tuple[int, ...], max_context: int):
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.max_context = max_context
        require(bool(eos_ids) and tokenizer.eos_token_id == eos_ids[0], "EOS contract")

    def content(self, text: str) -> tuple[int, ...]:
        ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        specials = set(getattr(self.tokenizer, "all_special_ids", self.eos_ids))
        require(not specials.intersection(ids), "special/control token in task content")
        return ids

    def student(self, problem: Problem) -> tuple[int, ...]:
        require(type(problem) is Problem, "public Problem DTO required")
        self.content(problem.text)
        ids = tuple(self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem.text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False))
        require(bool(ids), "empty chat template")
        return ids

    def privileged(self, task: Task, *, selection: bool = False) -> tuple[int, ...]:
        require(task.split == ("train_selection" if selection else "train"),
                "privileged guidance restricted to TRAIN/explicit selection probe")
        # Student response IDs are appended later, untouched, after this prefix.
        return self.student(task.public()) + self.content("\nVerified solution:\n" + task.solution + TRANSITION)

    def rewriter(self, problem: Problem, original_text: str) -> tuple[int, ...]:
        require(type(problem) is Problem, "rewriter must not receive Task/gold")
        prompt = Problem(problem.key, "Rewrite the attempt below into a shorter, correct, self-contained "
                         "solution. Preserve the problem's answer format. Do not refer to an earlier attempt.\n"
                         "Problem:\n" + problem.text + "\nAttempt:\n" + original_text)
        return self.student(prompt)

    def target(self, text: str) -> tuple[int, ...]:
        ids = self.content(text)
        require(bool(ids), "empty supervised target")
        return ids + (self.eos_ids[0],)

    def fit(self, prefix: tuple[int, ...], response_length: int) -> None:
        require(bool(prefix) and response_length > 0
                and len(prefix) + response_length <= self.max_context, "context overflow; truncation forbidden")

    def response(self, ids: tuple[int, ...], terminated: bool) -> None:
        require(bool(ids), "empty response")
        positions = [i for i, t in enumerate(ids) if t in self.eos_ids]
        require(positions == ([len(ids) - 1] if terminated else []), "EOS/termination mismatch")
