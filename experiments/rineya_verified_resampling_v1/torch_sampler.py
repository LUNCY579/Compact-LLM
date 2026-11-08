"""Explicit temperature sampling; torch is imported only when this module is used."""
from __future__ import annotations

import time
from typing import Callable
from .contracts import require


class TorchSampler:
    def __init__(self, model, torch, device: str, guard: Callable[[], None] = lambda: None):
        self.model, self.torch, self.device, self.guard = model, torch, device, guard

    def logits(self, prefix: list[int]):
        t = self.torch
        self.guard()
        with t.inference_mode():
            result = self.model(t.tensor([prefix], dtype=t.long, device=self.device),
                                use_cache=False, logits_to_keep=1).logits[0, -1].float()
        require(bool(t.isfinite(result).all()), "nonfinite logits")
        return result

    def sample(self, prompt: list[int], *, seed: int, temperature: float, max_tokens: int,
               greedy: bool, cache: bool, eos: tuple[int, ...], emit: Callable) -> dict:
        t = self.torch
        require(prompt and 0 < max_tokens <= 128, "invalid sampler input")
        generator = t.Generator(device=self.device).manual_seed(seed)
        generated, raw_lps, policy_lps, past, positions = [], [], [], None, 0
        started = time.monotonic()
        with t.inference_mode():
            for _ in range(max_tokens):
                self.guard()
                current = prompt + generated if not cache or past is None else [generated[-1]]
                ids = t.tensor([current], dtype=t.long, device=self.device)
                kwargs = {"use_cache": cache, "logits_to_keep": 1,
                          "attention_mask": t.ones((1, len(prompt)+len(generated)), dtype=t.long, device=self.device)}
                if cache and past is not None:
                    kwargs["past_key_values"] = past
                positions += len(current)
                result = self.model(ids, **kwargs)
                scores = result.logits[0, -1].float()
                require(bool(t.isfinite(scores).all()), "nonfinite logits")
                if cache:
                    past = result.past_key_values
                    require(past is not None, "model did not return a cache")
                raw = scores.log_softmax(-1)
                policy = (scores / temperature).log_softmax(-1)
                tok = int(scores.argmax().item()) if greedy else int(t.multinomial(policy.exp(), 1, generator=generator).item())
                raw_lp, policy_lp = float(raw[tok]), 0.0 if greedy else float(policy[tok])
                generated.append(tok)
                raw_lps.append(raw_lp)
                policy_lps.append(policy_lp)
                emit(tok, raw_lp, policy_lp)
                del result, scores, raw, policy
                if tok in eos:
                    break
        self.guard()
        return {"token_ids": generated, "raw_logprobs": raw_lps, "policy_logprobs": policy_lps,
                "stop": "eos" if generated[-1] in eos else "length",
                "seconds": time.monotonic()-started, "forward_positions": positions}

    def qualify(self, prompt: list[int], config: dict, eos: tuple[int, ...]) -> dict:
        """Reference path recomputes full prefixes; no benchmark answer is accessed."""
        args = dict(seed=config["seed"], temperature=config["temperature"],
                    max_tokens=min(32, config["max_new_tokens"]), eos=eos, emit=lambda *_: None)
        first = self.sample(prompt, greedy=False, cache=config["cache"], **args)
        repeat = self.sample(prompt, greedy=False, cache=config["cache"], **args)
        require(first["token_ids"] == repeat["token_ids"] and first["policy_logprobs"] == repeat["policy_logprobs"],
                "seeded repetition mismatch")
        errors, policy_errors, prefix = [], [], list(prompt)
        for tok, online, online_policy in zip(first["token_ids"], first["raw_logprobs"], first["policy_logprobs"]):
            score = self.logits(prefix)
            errors.append(abs(float(score.log_softmax(-1)[tok])-online))
            policy_errors.append(abs(float((score/config["temperature"]).log_softmax(-1)[tok])-online_policy))
            prefix.append(tok)
        for values in (errors, policy_errors):
            require(max(values) <= 0.03125 and sum(values)/len(values) <= 0.01,
                    "cached/full-prefix likelihood mismatch")
        greedy = self.sample(prompt, greedy=True, cache=config["cache"], **args)
        reference = self.sample(prompt, greedy=True, cache=False, **args)
        require(greedy["token_ids"] == reference["token_ids"], "cached/full-prefix greedy mismatch")
        return {"seeded_repeat_exact": True, "greedy_reference_exact": True,
                "raw_logprob_max_error": max(errors), "policy_logprob_max_error": max(policy_errors),
                "qualified_sample_tokens": first["token_ids"], "sample_stop": first["stop"],
                "configured_max_new_tokens": config["max_new_tokens"],
                "qualification_trace_cap": args["max_tokens"],
                "note": "short semantic-neutral trace plus separate maximum-context allocation probe"}
