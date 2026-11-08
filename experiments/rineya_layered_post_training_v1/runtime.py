"""Real PyTorch objectives; PRLC's audited GRPO primitive is reused unchanged.

Native training uses a fixed dense projection subset, not averaged LoRA factors.
Single-device, one unpadded sequence at a time; no remote code or downloads.
"""
from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from experiments.progressive_rl_curriculum_v1.objectives import advantages, policy_loss
from .config import Config, digest, require
from .data import Row, score
from .soup import state_hash


class Budget:
    def __init__(self, config):
        self.config, self.start = config, time.monotonic()
        self.counts = {"processed_tokens": 0, "generated_tokens": 0, "rollout_calls": 0, "actor_steps": 0, "critic_steps": 0}
        self.by_phase = {}
        self.phase = "initialization"

    def charge(self, **counts):
        require(time.monotonic() - self.start <= self.config.wall_seconds, "wall budget exhausted")
        limits = {"processed_tokens": self.config.processed_token_budget, "generated_tokens": self.config.generated_token_budget, "rollout_calls": self.config.rollout_budget, "actor_steps": self.config.actor_steps, "critic_steps": self.config.critic_steps + self.config.actor_steps}
        require(all(k in limits and type(v) is int and v >= 0 for k, v in counts.items()), "invalid budget charge")
        require(all(self.counts[k] + v <= limits[k] for k, v in counts.items()), "compute budget exhausted before operation")
        for k, v in counts.items():
            self.counts[k] += v
            phase = self.by_phase.setdefault(self.phase, {})
            phase[k] = phase.get(k, 0) + v

    def report(self):
        return {**self.counts, "elapsed_seconds": time.monotonic() - self.start, "by_phase": copy.deepcopy(self.by_phase)}


@dataclass(frozen=True)
class Episode:
    prompt: tuple[int, ...]
    response: tuple[int, ...]
    old_logps: tuple[float, ...]
    text: str
    terminated: bool
    version: int


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    require(beta > 0 and math.isfinite(beta), "invalid DPO beta")
    for x in (chosen, rejected, ref_chosen, ref_rejected):
        require(x.ndim == 1 and x.numel() > 0 and bool(torch.isfinite(x).all()), "invalid response logprobs")
    margin = chosen.sum() - rejected.sum() - (ref_chosen.sum() - ref_rejected.sum()).detach()
    return -F.logsigmoid(beta * margin)


def prefix_features(hidden, prompt_length, response_length):
    """h[p-1+t] predicts action t: never include the action being predicted."""
    require(hidden.ndim == 3 and hidden.shape[0] == 1 and prompt_length > 0 and response_length > 0, "invalid prefix features")
    require(hidden.shape[1] >= prompt_length + response_length - 1, "short hidden sequence")
    return hidden[0, prompt_length - 1:prompt_length + response_length - 1]


def critic_objective(predictions, terminal_return):
    require(predictions.ndim == 1 and predictions.numel() > 0 and terminal_return in (0., 1.), "invalid critic target")
    require(bool(torch.isfinite(predictions).all()), "nonfinite critic")
    return .5 * (predictions - terminal_return).square().mean()


class Runtime:
    def __init__(self, model, tokenizer, config: Config, names, identity, *, device="cpu", native=False):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.device, self.native, self.version = device, native, 0
        self.names = tuple(sorted(names))
        require(bool(self.names) and len(set(self.names)) == len(self.names), "empty/duplicate trainable names")
        require(not any("lora_" in n for n in self.names), "RLP native path requires effective dense weights")
        parameters = dict(model.named_parameters())
        require(set(self.names) <= set(parameters), "unknown trainable name")
        for name, p in parameters.items():
            p.requires_grad_(name in self.names)
        self.identity = {**identity, "trainable_names": list(self.names), "capacity": sum(parameters[n].numel() for n in self.names)}
        self.identity_hash = digest(self.identity)
        self.model.to(device).eval()
        self.reference = copy.deepcopy(model).cpu().eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference_hash = state_hash(self.state())
        self.budget = Budget(config)
        self.parity = {"checks": 0, "max_abs_error": 0.}
        self.critic = torch.nn.Linear(model.config.hidden_size, 1, dtype=torch.float32).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr, betas=(.9, .95))
        self.reset_optimizer()

    def state(self):
        params = dict(self.model.named_parameters())
        return {n: params[n].detach().cpu().clone() for n in self.names}

    def restore(self, state):
        require(set(state) == set(self.names), "checkpoint capacity mismatch")
        params = dict(self.model.named_parameters())
        for name, value in state.items():
            require(value.shape == params[name].shape and value.dtype == params[name].dtype and bool(torch.isfinite(value).all()), "invalid checkpoint tensor")
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value.to(self.device))
        self.version += 1  # Invalidate every pre-merge trajectory, even at alpha=0.
        self.reset_optimizer()  # Adam moments are not meaningful for merged weights.

    def reset_optimizer(self):
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.config.learning_rate, weight_decay=0.)

    def anchor_reference(self):
        params = dict(self.reference.named_parameters())
        state = self.state()
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value)
        self.reference_hash = state_hash(state)

    def prompt_ids(self, text):
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
        else:
            ids = tok.encode(text, add_special_tokens=True)
        require(0 < len(ids) <= self.config.max_prompt_tokens, "prompt length outside qualified range; no truncation")
        return tuple(ids)

    def response_ids(self, text):
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        require(bool(ids), "empty supervised response")
        eos = self.tokenizer.eos_token_id
        require(eos not in ids[:-1], "embedded EOS in response")
        if ids[-1] != eos:
            ids.append(eos)
        return tuple(ids)

    def logps(self, prompt, response, *, reference=False, features=False):
        require(bool(prompt) and bool(response) and len(prompt) + len(response) <= self.config.max_context_tokens, "context overflow or empty tokens")
        model = self.reference if reference else self.model
        device = "cpu" if reference else self.device
        ids = torch.tensor([[*prompt, *response]], dtype=torch.long, device=device)
        self.budget.charge(processed_tokens=ids.numel())
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=features)
        logits = out.logits[0, len(prompt) - 1:-1].float()
        targets = torch.tensor(response, device=device)
        logps = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
        require(bool(torch.isfinite(logps).all()), "nonfinite selected-token log probabilities")
        if features:
            return logps, prefix_features(out.hidden_states[-1], len(prompt), len(response))
        return logps

    def collect(self, prompt_text, count, seed):
        """No gold argument. Temperature=1, no top-k/top-p/length warpers."""
        require(type(count) is int and count > 0, "invalid rollout count")
        prompt = self.prompt_ids(prompt_text)
        require(len(prompt) + self.config.max_new_tokens <= self.config.max_context_tokens, "rollout context overflow")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        episodes = []
        with torch.no_grad():
            for _ in range(count):
                self.budget.charge(rollout_calls=1)
                response, behavior, cache = [], [], None
                for _ in range(self.config.max_new_tokens):
                    tokens = [*prompt, *response] if cache is None else [response[-1]]
                    self.budget.charge(processed_tokens=len(tokens), generated_tokens=1)
                    inp = torch.tensor([tokens], device=self.device)
                    kwargs = {"input_ids": inp, "attention_mask": torch.ones((1, len(prompt) + len(response)), dtype=torch.long, device=self.device), "use_cache": self.native}
                    if cache is not None:
                        kwargs["past_key_values"] = cache
                    out = self.model(**kwargs)
                    cache = getattr(out, "past_key_values", None)
                    lp = out.logits[0, -1].float().log_softmax(-1)
                    require(bool(torch.isfinite(lp).all()), "nonfinite sampling distribution")
                    token = int(torch.multinomial(lp.exp(), 1, generator=generator))
                    response.append(token)
                    behavior.append(float(lp[token]))
                    if token == self.tokenizer.eos_token_id:
                        break
                recomputed = self.logps(prompt, tuple(response))
                error = float((recomputed - torch.tensor(behavior, device=self.device)).abs().max())
                self.parity["checks"] += 1
                self.parity["max_abs_error"] = max(self.parity["max_abs_error"], error)
                require(error <= self.config.logprob_tolerance, f"sampler/trainer parity failed: {error}")
                episodes.append(Episode(prompt, tuple(response), tuple(behavior), self.tokenizer.decode(response, skip_special_tokens=True), response[-1] == self.tokenizer.eos_token_id, self.version))
        return episodes

    def ce(self, row, text=None):
        return -self.logps(self.prompt_ids(row.prompt), self.response_ids(row.target if text is None else text)).mean()

    def step(self, loss, *, critic=False):
        require(loss.ndim == 0 and bool(torch.isfinite(loss)), "nonfinite objective")
        self.budget.charge(**{"critic_steps" if critic else "actor_steps": 1})
        optimizer = self.critic_optimizer if critic else self.optimizer
        parameters = list(self.critic.parameters()) if critic else [p for p in self.model.parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in parameters), "nonfinite parameters; abort run")
        if not critic:
            self.version += 1
        return float(loss.detach())

    def sft(self, row, replay=None):
        loss = self.ce(row)
        if replay is not None and self.config.replay_weight:
            loss = loss + self.config.replay_weight * self.ce(replay)
        return self.step(loss)

    def dpo(self, row, replay=None):
        require(row.rejected is not None, "DPO requires curated rejected response")
        p = self.prompt_ids(row.prompt)
        chosen, rejected = self.response_ids(row.target), self.response_ids(row.rejected)
        with torch.no_grad():
            rc = self.logps(p, chosen, reference=True).to(self.device)
            rr = self.logps(p, rejected, reference=True).to(self.device)
        loss = dpo_loss(self.logps(p, chosen), self.logps(p, rejected), rc, rr, self.config.dpo_beta)
        if replay is not None and self.config.replay_weight:
            loss = loss + self.config.replay_weight * self.ce(replay)
        return self.step(loss)

    def critic_fit(self, episodes, rewards):
        require(len(episodes) == len(rewards) and bool(episodes), "invalid critic batch")
        losses = []
        for ep, reward in zip(episodes, rewards):
            require(ep.terminated and reward in (0., 1.), "critic requires complete labeled returns")
            with torch.no_grad():
                _, h = self.logps(ep.prompt, ep.response, features=True)
            losses.append(critic_objective(self.critic(h.detach().float()).squeeze(-1), reward))
        return self.step(torch.stack(losses).mean(), critic=True)

    def rl(self, episodes, rewards, replay=None, *, use_critic=False):
        require(len(episodes) == len(rewards) and len(episodes) >= 2, "invalid RL group")
        aa = advantages(rewards)
        losses = []
        for ep, reward, a in zip(episodes, rewards, aa):
            require(ep.version == self.version and ep.terminated, "stale/truncated on-policy episode")
            current, h = self.logps(ep.prompt, ep.response, features=True)
            old = torch.tensor(ep.old_logps, device=self.device)
            require(float((current.detach() - old).abs().max()) <= self.config.logprob_tolerance, "on-policy likelihood mismatch")
            with torch.no_grad():
                ref = self.logps(ep.prompt, ep.response, reference=True).to(self.device)
            if use_critic:
                # Reuse PRLC's clipped loss with each token as a length-one item.
                a_tensor = reward - self.critic(h.detach().float()).squeeze(-1).detach()
                shape = (-1, 1)
            else:
                a_tensor = torch.tensor([a], device=self.device, dtype=current.dtype)
                shape = (1, -1)
            losses.append(policy_loss(current.reshape(shape), old.reshape(shape), ref.reshape(shape), a_tensor, torch.ones_like(current.reshape(shape), dtype=torch.bool), epsilon=self.config.clip_epsilon, beta=self.config.kl_beta))
        loss = torch.stack(losses).mean()
        if replay is not None and self.config.replay_weight:
            loss = loss + self.config.replay_weight * self.ce(replay)
        result = self.step(loss)
        if use_critic:
            self.critic_fit(episodes, rewards)
        return result


class TinyTokenizer:
    eos_token_id, pad_token_id, chat_template = 1, 0, None

    def encode(self, text, add_special_tokens=True):
        if not add_special_tokens:
            require(text.strip() in set("01234"), "tiny response must be one digit")
            return [5 + int(text)]
        m = re.fullmatch(r"(sum|keep) ([0-4]) ([0-4])", text)
        require(m is not None, "tiny tokenizer accepts fixtures only")
        return [2, 5 + int(m[2]), 5 + int(m[3]), 3 if m[1] == "sum" else 4]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(x - 5) for x in ids if 5 <= x <= 9)


class TinyLM(torch.nn.Module):
    """Trainable causal GRU. Grammar logits make every sample terminate in 2 tokens."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=24)
        self.embedding = torch.nn.Embedding(10, 24)
        self.gru = torch.nn.GRU(24, 24, batch_first=True)
        self.head = torch.nn.Linear(24, 10)

    def forward(self, input_ids, **kwargs):
        h, _ = self.gru(self.embedding(input_ids))
        logits = self.head(h)
        is_prompt_end = (input_ids == 3) | (input_ids == 4)
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        allowed[..., 5:] = is_prompt_end[..., None]
        allowed[..., 1] = ~is_prompt_end
        logits = logits.masked_fill(~allowed, -100.)
        return SimpleNamespace(logits=logits, hidden_states=(h,), past_key_values=None)


def tiny_runtime(config):
    torch.manual_seed(config.seed)
    model = TinyLM()
    return Runtime(model, TinyTokenizer(), config, [n for n, _ in model.named_parameters()], {"kind": "synthetic_torch", "fixture": "mod5-gru-v1", "initial_weights": state_hash(dict(model.named_parameters()))})
