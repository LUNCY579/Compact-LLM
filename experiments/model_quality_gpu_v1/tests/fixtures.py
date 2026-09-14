"""Synthetic fixtures only. Never counterfeit a native tokenizer or model receipt."""
from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from experiments.model_quality_gpu_v1.config import Design
from experiments.model_quality_gpu_v1.data import compile_plan


class ByteTokenizer:
    eos_token, eos_token_id, is_fast = "<|im_end|>", 2, True

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            if text.startswith(self.eos_token):
                result.append(2); text = text[len(self.eos_token):]
            else:
                c, text = text[0], text[1:]
                result.extend(b + 10 for b in c.encode("utf-8"))
        return result

    def decode(self, ids, skip_special_tokens=False):
        out = bytearray()
        for i in ids:
            if i == 2:
                if not skip_special_tokens:
                    out.extend(self.eos_token.encode())
            elif i >= 10:
                out.append(i - 10)
        return out.decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        value = ""
        for message in messages:
            value += message["role"] + ":" + message["content"] + self.eos_token + "\n"
        if add_generation_prompt:
            value += "assistant:"
        return self.encode(value) if tokenize else value


class TinyBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(266, 8)
        self.layers = nn.ModuleList(nn.Linear(8, 8, bias=False) for _ in range(4))
        self.norm = nn.LayerNorm(8)
        self.checkpointed = False

    def forward(self, input_ids, **kwargs):
        h = self.embed_tokens(input_ids)
        # Include cross-token dependence, not just independent token classifiers.
        h = h.cumsum(1) / torch.arange(1, h.shape[1] + 1, device=h.device)[None, :, None]
        for layer in self.layers:
            def block(x, layer=layer):
                return x + layer(x).tanh()
            h = checkpoint(block, h, use_reentrant=False) if self.checkpointed and self.training else block(h)
        return SimpleNamespace(last_hidden_state=self.norm(h))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBody()
        self.lm_head = nn.Linear(8, 266, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        assert gradient_checkpointing_kwargs == {"use_reentrant": False}
        self.model.checkpointed = True

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        ids = input_ids if past_key_values is None else torch.cat((past_key_values, input_ids), 1)
        hidden = self.model(ids).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden)[:, -input_ids.shape[1]:],
                               past_key_values=ids if use_cache else None)


def row(i, split="train"):
    return {"id": f"{split}-{i}", "template_family": f"{split}-family-{i}",
            "messages": [{"role": "user", "content": f"{split} task {chr(65+i)}: reply seven."},
                         {"role": "assistant", "content": "seven"}],
            "evaluation": {"scorer": "exact", "answers": ["seven"], "answerable": True}}


def synthetic_plan(**changes):
    d = Design(processed_budget=440, min_budget_fraction=0.7, max_sequence=160,
               update_tokens=160, trainable_tail=2, ce_chunk=7, checkpoint_every=1)
    d = replace(d, **changes)
    return compile_plan([row(i) for i in range(12)], [row(0, "dev")], [row(0, "excluded")],
                        ByteTokenizer(), d, {}, {"scope": "SYNTHETIC"}, synthetic=True)
