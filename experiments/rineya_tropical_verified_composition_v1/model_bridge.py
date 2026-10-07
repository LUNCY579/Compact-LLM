"""Object-bound causal-LM bridge. Never downloads, deserializes or selects weights.

The training host must supply qualified tokenization and generation callbacks;
response masks, EOS behavior and canonical Markov prompts are host responsibilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

import torch
from torch import Tensor, nn

from .domain import State, Task, root, step, success
from .objectives import causal_response_logps
from .runtime import Ledger, Sample


@dataclass(frozen=True)
class EncodedAction:
    input_ids: Tensor
    attention_mask: Tensor
    response_mask: Tensor


class CausalActor(nn.Module):
    def __init__(self, model: nn.Module,
                 encode_action: Callable[[Task, State, str], EncodedAction],
                 generate_action: Callable[[Task, State, random.Random, int], Sample]):
        super().__init__()
        self.model = model
        self.encode_action = encode_action
        self.generate_action = generate_action

    def sample(self, task: Task, state: State, rng: random.Random,
               max_new_tokens: int) -> Sample:
        return self.generate_action(task, state, rng, max_new_tokens)

    def action_logp(self, task: Task, state: State, action: str,
                    ledger: Ledger | None = None, training: bool = False) -> tuple[Tensor, Tensor]:
        encoded = self.encode_action(task, state, action)
        if encoded.input_ids.ndim != 2 or encoded.input_ids.shape[0] != 1:
            raise ValueError("Native bridge expects one untruncated action example")
        count = int(encoded.response_mask.sum().item())
        if count <= 0:
            raise ValueError("Native encoder omitted the response")
        if ledger is not None:
            name = "supervised_tokens" if training else "scored_tokens"
            ledger.require(**{name: count})
            ledger.spend(**{name: count})
        output = self.model(input_ids=encoded.input_ids,
                            attention_mask=encoded.attention_mask,
                            use_cache=False, return_dict=True)
        logps, counts = causal_response_logps(output.logits, encoded.input_ids,
                                              encoded.attention_mask, encoded.response_mask)
        return logps[0], counts[0]

    def score(self, task: Task, state: State, action: str,
              ledger: Ledger | None = None) -> float:
        with torch.no_grad():
            return float(self.action_logp(task, state, action, ledger)[0])

    def paths(self, task: Task, paths: list[tuple[str, ...]], ledger: Ledger
              ) -> tuple[Tensor, Tensor]:
        if task.role not in ("train", "replay_train"):
            raise ValueError("Cannot optimize on evaluation inputs")
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Empty or duplicate candidate paths")
        logps, lengths = [], []
        for actions in paths:
            state, terms, counts = root(task), [], []
            for action in actions:
                following = step(task, state, action)
                value, count = self.action_logp(task, state, action, ledger, training=True)
                terms.append(value)
                counts.append(count)
                state = following
            if not terms or not success(task, state):
                raise ValueError("Only complete verifier-accepted paths may train")
            logps.append(torch.stack(terms).sum())
            lengths.append(torch.stack(counts).sum())
        return torch.stack(logps), torch.stack(lengths)
