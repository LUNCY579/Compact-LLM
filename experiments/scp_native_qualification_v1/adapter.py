"""Executable host-port boundary, exercised ONLY with explicit scripted fixtures.

No live transport can be attached in this release. Keeping the native gate closed
is intentional: these mechanics do not certify an MQ worker, a tokenizer or a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from .contract import canonical, digest, loads, require, refuse_native


class Reservations:
    """Unit bridge to unchanged RRP Journal.reserve; not a replacement journal.

The owning future controller must hold the MQ lease, start the journal first,
and settle ONLY from an observed terminal supervisor receipt. Reservations here
are worst-case forwards; completed/attempted forwards remain a distinct measure.
"""
    def __init__(self, persistent_reserve):
        require(callable(persistent_reserve), "persistent reservation callback required")
        self.reserve = persistent_reserve
        self.requests = self.allowance = self.forward_allowance = 0
        self.qualifications = 0

    def qualification(self):
        require(self.qualifications == 0 and self.requests == 0, "qualification must occur once before generation")
        self.reserve({"evaluation_and_qualification_model_calls": 6})
        self.qualifications = 1
        self.forward_allowance += 6

    def generation(self, cap):
        require(type(cap) is int and cap in (32, 64, 288), "unregistered request cap")
        require(self.qualifications == 1, "qualification reservation required first")
        require(self.requests < 12 and self.allowance + cap <= 928, "request/token envelope exhausted")
        self.reserve({"evaluation_and_qualification_model_calls": cap,
                      "generated_tokens_all_candidates": cap})
        self.requests += 1
        self.allowance += cap
        self.forward_allowance += cap


@dataclass(frozen=True)
class ScriptedTransport:
    """Literal full-vocabulary logits; cannot load or call a pretrained model."""
    steps: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        require(type(self.steps) is tuple and self.steps and all(type(s) is tuple for s in self.steps),
                "literal tuple-of-tuples fixture script required")

    def step(self, ids, cache):
        index = 0 if cache is None else cache
        require(index < len(self.steps), "fixture script exhausted")
        return self.steps[index], index + 1


class FixtureTokenizer:
    """Deliberately not a Qwen tokenizer; never yields native-tokenizer evidence."""
    def render_ids(self, messages):
        return [1] + [2] * sum(len(m["content"].split()) for m in messages)

    def decode(self, ids):
        return " ".join(f"<token:{i}>" for i in ids)


class HostAdapter:
    """Cache-isolated, reservation-first decoder boundary. Fixture transport only."""
    def __init__(self, transport, tokenizer, eos_ids, reservations, guard, emit, *, native=False):
        if native is not False:
            refuse_native()
        require(type(transport) is ScriptedTransport and type(tokenizer) is FixtureTokenizer,
                "live/arbitrary host callbacks are not qualified; scripted fixtures only")
        require(type(eos_ids) is list and eos_ids and all(type(i) is int and i >= 0 for i in eos_ids)
                and len(eos_ids) == len(set(eos_ids)), "full generation-config EOS list required")
        require(type(reservations) is Reservations and callable(guard) and callable(emit), "missing boundary observers")
        self.transport, self.tokenizer = transport, tokenizer
        self.eos, self.reservations, self.guard, self.emit = frozenset(eos_ids), reservations, guard, emit
        self.failed = False
        self.attempted_forwards = 0

    def check_context(self, messages, cap):
        require(not self.failed, "failed adapter is terminal; no retries")
        require(type(cap) is int and cap in (32, 64, 288), "unregistered output cap")
        require(type(messages) is list and len(messages) == 2, "exact system/user messages required")
        for role, message in zip(("system", "user"), messages):
            require(type(message) is dict and set(message) == {"role", "content"}
                    and message["role"] == role and type(message["content"]) is str
                    and bool(message["content"].strip()), "label-bearing or malformed message")
        ids = self.tokenizer.render_ids(messages)
        require(type(ids) is list and ids and all(type(i) is int and i >= 0 for i in ids), "invalid input IDs")
        require(len(ids) + cap <= 2048, "OVERLENGTH: no truncation or candidate shortening")
        return ids

    def generate(self, messages, cap):
        require(not self.failed, "failed adapter is terminal; no retries")
        request = {"messages": loads(canonical(messages)), "cap": cap}
        request_sha = digest(request)
        reserved = False
        try:
            ids = self.check_context(messages, cap)
            self.reservations.generation(cap)
            reserved = True
            self.emit({"event": "RESERVED", "request_sha256": request_sha, **request,
                       "classification": "SOFTWARE_FIXTURE_ONLY"})
            generated, cache = [], None  # Independent cache for EVERY request/candidate.
            current = ids
            vocab = None
            for index in range(cap):
                self.guard()
                self.emit({"event": "FORWARD_RESERVED", "request_sha256": request_sha, "index": index})
                self.attempted_forwards += 1
                logits, cache = self.transport.step(current, cache)
                require(type(logits) is tuple and logits, "full-vocabulary logits missing")
                require(all(type(v) in (int, float) and math.isfinite(v) for v in logits), "nonfinite full-vocabulary logits")
                require(vocab is None or len(logits) == vocab, "vocabulary changed during decode")
                vocab = len(logits)
                require(all(i < vocab for i in self.eos), "EOS outside vocabulary")
                token = max(range(vocab), key=logits.__getitem__)
                generated.append(token)
                self.emit({"event": "TOKEN", "request_sha256": request_sha, "index": index, "token_id": token})
                if token in self.eos:
                    break
                current = [token]
            terminated = generated[-1] in self.eos
            result = {"text": self.tokenizer.decode(generated[:-1] if terminated else generated),
                      "input_tokens": len(ids), "output_tokens": len(generated), "terminated": terminated,
                      "generated_ids_including_terminal": generated, "actual_forward_calls": len(generated),
                      "full_vocab_checked": vocab, "eos_ids": sorted(self.eos), "request_sha256": request_sha,
                      "classification": "SOFTWARE_FIXTURE_ONLY", "pretrained_calls": 0}
            self.emit({"event": "COMPLETED", **result})
            return result
        except BaseException as exc:
            self.failed = True
            # An event-sink failure must propagate; never manufacture a completed record.
            self.emit({"event": "FAILED", "request_sha256": request_sha, "reserved": reserved,
                       "exception_type": type(exc).__name__})
            raise


def resource_gate(snapshot, *, before_load, max_rss, now):
    """Validate an observer value; supplied JSON is NOT an independently observed host."""
    require(type(before_load) is bool and type(max_rss) is int and max_rss > 0, "invalid resource policy")
    require(type(snapshot) is dict and set(snapshot) == {"available_host_bytes", "rss_bytes", "gpu0", "time_unix"},
            "resource observation incomplete")
    for key in ("available_host_bytes", "rss_bytes"):
        require(type(snapshot[key]) is int and snapshot[key] >= 0, "invalid resource value")
    gpu = snapshot["gpu0"]
    require(type(gpu) is dict and set(gpu) == {"free", "total"}
            and all(type(v) is int and v >= 0 for v in gpu.values()) and 0 < gpu["total"] >= gpu["free"],
            "CUDA observation absent/invalid")
    require(type(now) in (int, float) and math.isfinite(now), "invalid observer clock")
    stamp = snapshot["time_unix"]
    require(type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= now - stamp <= 5, "stale/future resource observation")
    require(snapshot["available_host_bytes"] >= (4.6 if before_load else 1.5) * 2**30, "host floor refused")
    require(gpu["free"] >= (3.0 if before_load else 0.5) * 2**30, "GPU floor refused")
    require(snapshot["rss_bytes"] <= max_rss, "process RSS ceiling exceeded")
    return {"classification": "RESOURCE_VALUE_CHECK_ONLY", "native_execution_permitted": False}
