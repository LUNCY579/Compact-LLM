from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..data.views import IndexedRow, choose_index_rows, read_at_offset, scan_jsonl_index, select_messages
from .tokenization import TokenizedConversation, tokenize_conversation


class JsonlChatDataset(Dataset):
    """Memory-light random-access JSONL dataset with a small per-process tokenization cache."""

    def __init__(
        self,
        path: str | Path,
        *,
        tokenizer: Any,
        view: str,
        max_length: int,
        max_target_tokens: int | None,
        seed: int,
        max_examples: int | None = None,
        one_per_problem: bool = False,
        preserve_trajectory_order: bool = False,
        inverse_problem_weight: bool = False,
        train_on_prompt: bool = False,
        cache_size: int = 2048,
    ) -> None:
        self.path = Path(path).resolve()
        self.tokenizer = tokenizer
        self.view = view
        self.max_length = int(max_length)
        self.max_target_tokens = int(max_target_tokens) if max_target_tokens is not None else None
        self.train_on_prompt = bool(train_on_prompt)
        index = scan_jsonl_index(self.path)
        self.rows = choose_index_rows(
            index,
            seed=seed,
            max_examples=max_examples,
            one_per_problem=one_per_problem,
            preserve_trajectory_order=preserve_trajectory_order,
        )
        self.inverse_problem_weight = inverse_problem_weight
        self.cache_size = max(0, int(cache_size))
        self._cache: collections.OrderedDict[int, dict[str, Any]] = collections.OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _sample_weight(self, row: IndexedRow) -> float:
        if self.inverse_problem_weight and row.total_steps:
            return 1.0 / float(row.total_steps)
        return 1.0

    def _load(self, index: int) -> dict[str, Any]:
        if index in self._cache:
            value = self._cache.pop(index)
            self._cache[index] = value
            return value
        row = self.rows[index]
        record = read_at_offset(self.path, row.offset)
        messages = select_messages(record, self.view)
        encoded: TokenizedConversation = tokenize_conversation(
            self.tokenizer,
            messages,
            max_length=self.max_length,
            max_target_tokens=self.max_target_tokens,
            train_on_prompt=self.train_on_prompt,
        )
        value = {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "labels": encoded.labels,
            "sample_weight": self._sample_weight(row),
            "prompt_tokens": encoded.prompt_tokens,
            "target_tokens": encoded.target_tokens,
            "truncated_prompt_tokens": encoded.truncated_prompt_tokens,
            "truncated_target_tokens": encoded.truncated_target_tokens,
        }
        if self.cache_size:
            self._cache[index] = value
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        value = self._load(index)
        # Keep accounting fields out of the model batch.
        return {
            "input_ids": list(value["input_ids"]),
            "attention_mask": list(value["attention_mask"]),
            "labels": list(value["labels"]),
            "sample_weight": float(value["sample_weight"]),
        }

    def exposure_stats(self, *, exact: bool, sample_size: int = 2048) -> dict[str, Any]:
        row_count = len(self.rows)
        unique_problems = len({row.problem_id for row in self.rows})
        if row_count == 0:
            return {
                "rows": 0,
                "unique_problems": 0,
                "token_count_mode": "exact",
                "input_tokens": 0,
                "target_tokens": 0,
            }
        if exact:
            indexes = range(row_count)
            scale = 1.0
            mode = "exact"
        else:
            size = min(row_count, max(1, int(sample_size)))
            # Deterministic approximately uniform positions.
            indexes = sorted({min(row_count - 1, int(i * row_count / size)) for i in range(size)})
            scale = row_count / len(indexes)
            mode = f"estimated_from_{len(indexes)}_rows"
        prompt_tokens = target_tokens = truncated_prompt = truncated_target = 0
        for index in indexes:
            value = self._load(index)
            prompt_tokens += int(value["prompt_tokens"])
            target_tokens += int(value["target_tokens"])
            truncated_prompt += int(value["truncated_prompt_tokens"])
            truncated_target += int(value["truncated_target_tokens"])
        return {
            "rows": row_count,
            "unique_problems": unique_problems,
            "token_count_mode": mode,
            "input_tokens": round(prompt_tokens * scale),
            "target_tokens": round(target_tokens * scale),
            "truncated_prompt_tokens": round(truncated_prompt * scale),
            "truncated_target_tokens": round(truncated_target * scale),
            "mean_input_tokens": (prompt_tokens + target_tokens) / len(indexes),
            "mean_target_tokens": target_tokens / len(indexes),
        }


class AssistantOnlyCollator:
    def __init__(self, tokenizer: Any, *, pad_to_multiple_of: int | None = 8) -> None:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError("Tokenizer needs a pad_token_id or eos_token_id")
        self.pad_token_id = int(pad_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        batch_size = len(features)
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        weights = torch.ones((batch_size,), dtype=torch.float32)
        for row_index, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row_index, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row_index, :length] = torch.tensor(feature["attention_mask"], dtype=torch.long)
            labels[row_index, :length] = torch.tensor(feature["labels"], dtype=torch.long)
            weights[row_index] = float(feature.get("sample_weight", 1.0))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_weight": weights,
        }
