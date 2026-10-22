"""Label-free, prospective K0/K1 greedy adapter; imports never load assets.

The coordinator must bind/qualify the actual model, corpus and complete budget.
``enforce_device=False`` is exclusively for analytic CPU fixture hosts.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import time
import traceback

import torch

from ..control import BlockRecurrence, TAPSConfig
from ..gpu_native import forward_logits, install_gpu_timing, one_shot_anchors, require_cuda_tensor
from experiments.anchor_routed_middle_recurrence_v1.src.armr.metrics import output_flags


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def minimal_probability(logits):
    """Exactly observer da6's kernel expression, not a LSE reconstruction."""
    if logits.ndim != 1 or logits.numel() < 2 or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("finite nonempty full-head vector required")
    z = logits.detach().float()
    logp = z.log_softmax(0)
    p_vector = logp.exp()
    p = float(p_vector[z.argmax()])
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise FloatingPointError("actual FP32 probability outside finite[0,1]")
    return p


@dataclass(frozen=True, eq=False)
class PilotHandle:
    """Opaque once-only receipt; no logit tensor is exposed to a policy."""
    probability: float
    prefix_ids_sha256: str
    API_attempt: int
    logits_sha256: str
    run_id: str
    source_sha256: str
    resident_state_sha256: str
    tokenizer_sha256: str


@dataclass(frozen=True)
class _Binding:
    run_id: str
    source_sha256: str
    resident_state_sha256: str
    tokenizer_sha256: str


def _hook_snapshot(model):
    return (len(model._forward_pre_hooks), len(model._forward_hooks), tuple(
        (len(layer._forward_pre_hooks), len(layer._forward_hooks), "forward" in layer.__dict__,
         id(layer.__dict__["forward"]) if "forward" in layer.__dict__ else None,
         "_taps_owner" in layer.__dict__, "_armr_owner" in layer.__dict__)
        for layer in model.model.layers))


class GenerationSession:
    """Own one actual model/tokenizer instance and a bounded physical call ledger.

    There are no case dictionaries, labels, scorer callbacks or policy objects.
    The outer worker supplies input-only K and logs case IDs separately.
    """
    def __init__(self, model, tokenizer, *, run_id, source_sha256, resident_state_sha256,
                 tokenizer_sha256, api_cap, guard, synchronize=None, event_sink=None,
                 trajectory_sink=None, enforce_device=True, maximum_prefix_tokens=357,
                 output_flags_fn=None):
        if (type(api_cap) is not int or api_cap < 1 or type(enforce_device) is not bool
                or type(maximum_prefix_tokens) is not int or not 1 <= maximum_prefix_tokens <= 357):
            raise ValueError("registered positive call/shape limits and Boolean device contract required")
        if not isinstance(run_id, str) or not run_id or not callable(guard):
            raise ValueError("run identity and mandatory resource/deadline guard required")
        for value in (source_sha256, resident_state_sha256, tokenizer_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("exact lowercase SHA256 bindings required")
        self.model, self.tokenizer = model, tokenizer
        self._model_instance, self._tokenizer_instance = model, tokenizer
        self._binding = _Binding(run_id, source_sha256, resident_state_sha256, tokenizer_sha256)
        self.api_cap, self.enforce_device, self.maximum_prefix_tokens = api_cap, enforce_device, maximum_prefix_tokens
        self.guard, self.synchronize, self.event_sink = guard, synchronize, event_sink
        self.trajectory_sink = trajectory_sink
        self._flags = output_flags if output_flags_fn is None else output_flags_fn
        if enforce_device and output_flags_fn is not None:
            raise ValueError("native output flags must use the unchanged inherited implementation")
        if not enforce_device and type(model).__name__ == "Qwen3AbstainForCausalLM":
            raise RuntimeError("CPU testing flag cannot admit an actual canonical model")
        self._device = next(model.parameters()).device
        if enforce_device:
            if synchronize is None:
                raise ValueError("native synchronized timings are required")
            for name, value in list(model.named_parameters()) + list(model.named_buffers()):
                require_cuda_tensor(value, name)
                if value.is_floating_point() and value.dtype != torch.float32:
                    raise RuntimeError("native floating parameters/buffers must be FP32")
        elif self._device.type != "cpu":
            raise RuntimeError("testing flag is for analytic CPU hosts only")
        self._signature = self._model_signature()
        self._configuration = self._configuration_signature()
        self._methods = self._method_signature()
        self._eos = self._eos_ids()
        if enforce_device and self._eos != {151643, 151645}:
            raise RuntimeError("canonical EOS contract differs")
        self.API_attempts = self.completed_forwards = self.failed_forwards = self.completed_positions = 0
        self.events, self._pilots = [], {}
        self._check_frozen()

    def _model_signature(self):
        return tuple((name, id(value), value._version, tuple(value.shape), value.dtype, value.device)
                     for name, value in list(self.model.named_parameters()) + list(self.model.named_buffers()))

    def _configuration_signature(self):
        config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else vars(self.model.config)
        return json.dumps(config, sort_keys=True, default=str, allow_nan=False)

    def _method_signature(self):
        return (getattr(self.model.forward, "__func__", self.model.forward),
                getattr(self.tokenizer.decode, "__func__", self.tokenizer.decode))

    def _eos_ids(self):
        values = self.model.generation_config.eos_token_id
        values = values if isinstance(values, list) else [values]
        if not values or any(type(x) is not int or not 0 <= x < self.model.config.vocab_size for x in values):
            raise RuntimeError("actual EOS/head-index contract invalid")
        return set(values)

    def _check_frozen(self):
        if (any(module.training for module in self.model.modules())
                or self.model is not self._model_instance or self.tokenizer is not self._tokenizer_instance
                or any(p.requires_grad or p.grad is not None for p in self.model.parameters())
                or getattr(self.model, "is_gradient_checkpointing", False) or hasattr(self.model, "_orig_mod")
                or self._model_signature() != self._signature or self._eos_ids() != self._eos
                or self._configuration_signature() != self._configuration or self._method_signature() != self._methods):
            raise RuntimeError("frozen model/state/eval/EOS binding changed")

    def _guard(self):
        self._check_frozen()
        return self.guard()

    def _emit(self, value):
        self.events.append(value)
        if self.event_sink is not None:
            self.event_sink(value)

    def _ids(self, input_ids):
        if (not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long
                or input_ids.ndim != 2 or input_ids.shape[0] != 1
                or not 1 <= input_ids.shape[1] <= self.maximum_prefix_tokens or input_ids.device != self._device):
            raise ValueError("same-device nonempty batch1 original long IDs required")
        if self.enforce_device:
            require_cuda_tensor(input_ids, "original prefix")
        values = tuple(input_ids[0].tolist())
        if any(not 0 <= value < self.model.config.vocab_size for value in values):
            raise ValueError("input ID outside actual embedding/head rows")
        return input_ids.detach().clone(), values

    @staticmethod
    def _started(value):
        now = time.perf_counter()
        if value is None:
            return now
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value > now:
            raise ValueError("case start clock must be finite and nonfuture")
        return float(value)

    @contextmanager
    def _scope(self, k):
        snapshot = _hook_snapshot(self.model)
        wrapper = BlockRecurrence(self.model, extra_loops=k, config=TAPSConfig(scheduler_mode="unit"),
                                  sample_id="raw_generation", arm_id="unit", guard=self._guard,
                                  sink=self.trajectory_sink)
        try:
            with torch.inference_mode(), one_shot_anchors(self.model), wrapper:
                install_gpu_timing(wrapper, self.synchronize or (lambda: None), require_cuda=self.enforce_device)
                yield wrapper
        finally:
            if _hook_snapshot(self.model) != snapshot:
                raise RuntimeError("generation forward/hook/owner restoration failed")

    def _forward(self, ids, *, k, stage, calls):
        sample = self._guard()
        if self.API_attempts >= self.api_cap:
            raise RuntimeError("registered physical model-API cap exhausted")
        self.API_attempts += 1
        before = time.perf_counter()
        record = {"API_attempt": self.API_attempts, "stage": stage, "K": k,
                  "prefix_tokens": ids.shape[1], "prefix_ids_sha256": ids_digest(tuple(ids[0].tolist())),
                  "depth": self.model.config.num_hidden_layers + 2 * k}
        self._emit({"event": "FORWARD_STARTED", **record, "completed": False, "resources": sample})
        try:
            logits = forward_logits(self.model, ids, enforce_device=self.enforce_device, synchronize=self.synchronize)
        except Exception as error:
            self.failed_forwards += 1
            self._emit({"event": "FORWARD_FAILED", **record, "completed": False,
                        "elapsed_seconds": time.perf_counter() - before, "partial_compute": "UNKNOWN",
                        "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            raise
        record.update(elapsed_seconds=time.perf_counter() - before, completed=True)
        self.completed_forwards += 1; self.completed_positions += ids.shape[1]; calls.append(record)
        # Completed compute is retained even if the following resource guard stops.
        try:
            resource = self._guard()
        except Exception as error:
            self._emit({"event": "FORWARD_COMPLETED", **record, "resources": None,
                        "post_forward_guard_error": str(error)})
            raise
        self._emit({"event": "FORWARD_COMPLETED", **record, "resources": resource, "post_forward_guard_error": None})
        return logits

    @staticmethod
    def _tensor_hash(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def make_pilot(self, input_ids, *, started_perf_counter=None):
        started = self._started(started_perf_counter)
        ids, values = self._ids(input_ids); calls = []
        with self._scope(0):
            logits = self._forward(ids, k=0, stage="PILOT", calls=calls)
        probability_started = time.perf_counter()
        probability = minimal_probability(logits)
        if self.synchronize:
            self.synchronize()
        seconds = time.perf_counter() - probability_started
        handle = PilotHandle(probability, ids_digest(values), calls[0]["API_attempt"], self._tensor_hash(logits),
                             self._binding.run_id, self._binding.source_sha256,
                             self._binding.resident_state_sha256, self._binding.tokenizer_sha256)
        self._pilots[handle] = {"snapshot": tuple(handle.__dict__.values()), "model": self.model,
                               "tokenizer": self.tokenizer, "signature": self._signature, "binding": self._binding,
                               "ids": values, "logits": logits, "calls": calls, "started": started,
                               "probability_seconds": seconds, "consumed": False}
        self._emit({"event": "PILOT_COMMITTED", "API_attempt": handle.API_attempt,
                    "prefix_ids_sha256": handle.prefix_ids_sha256, "logits_sha256": handle.logits_sha256,
                    "probability": probability, "probability_seconds": seconds})
        return handle

    def _claim(self, pilot, values):
        if type(pilot) is not PilotHandle or pilot not in self._pilots:
            raise ValueError("forged, serialized or different-session pilot handle")
        record = self._pilots[pilot]
        if record["consumed"]:
            raise ValueError("pilot handle already consumed")
        record["consumed"] = True
        if (record["snapshot"] != tuple(pilot.__dict__.values()) or record["ids"] != values
                or pilot.prefix_ids_sha256 != ids_digest(values) or record["binding"] != self._binding
                or record["model"] is not self.model or record["tokenizer"] is not self.tokenizer
                or record["signature"] != self._model_signature()):
            record["logits"] = None
            raise ValueError("pilot model/source/state/tokenizer/prefix binding differs")
        return record

    def _flops(self, calls):
        c = self.model.config
        h, intermediate = c.hidden_size, c.intermediate_size
        q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
        per_layer = 2 * (h * q + 2 * h * kv + q * h + 3 * h * intermediate)
        head = 2 * h * c.vocab_size
        return int(sum(x["prefix_tokens"] * x["depth"] * per_layer + head for x in calls))

    def generate(self, input_ids, *, k, pilot=None, max_new_tokens=8, qualification=False,
                 max_prompt_seconds=900., started_perf_counter=None):
        """Return raw row and selected first logits; no scorer or policy inputs."""
        if type(k) is not int or k not in (0, 1) or type(qualification) is not bool:
            raise ValueError("case-level K0/1 and Boolean qualification required")
        cap = 24 if qualification else 8
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= cap:
            raise ValueError("scientific8/qualification24 token cap exceeded")
        if not isinstance(max_prompt_seconds, (int, float)) or not math.isfinite(max_prompt_seconds) or max_prompt_seconds <= 0:
            raise ValueError("registered positive per-case timeout required")
        started = self._started(started_perf_counter)
        ids, values = self._ids(input_ids)
        if ids.shape[1] + max_new_tokens - 1 > self.maximum_prefix_tokens:
            raise ValueError("generation exceeds registered full-prefix envelope")
        self._check_frozen()
        record = self._claim(pilot, values) if pilot is not None else None
        if record is not None:
            if started_perf_counter is not None and started != record["started"]:
                raise ValueError("pilot case clock cannot be replaced")
            started = record["started"]
        calls = [] if record is None else list(record["calls"])
        pilot_logits = None if record is None else record["logits"]
        if record is not None:
            record["logits"] = None  # Registry cannot retain a consumed GPU vector.
        if time.perf_counter() - started > max_prompt_seconds:
            raise TimeoutError("registered case timeout before generation")
        tokens, finish, text = [], "token_cap", ""
        selected_calls = []
        try:
            with self._scope(k) as wrapper:
                if record is not None and k == 0:
                    logits = pilot_logits
                    self._emit({"event": "PILOT_FIRST_LOGITS_REUSED", "API_attempt": pilot.API_attempt,
                                "prefix_ids_sha256": pilot.prefix_ids_sha256, "extra_model_calls": 0})
                else:
                    logits = self._forward(ids, k=k, stage="GENERATION", calls=selected_calls)
                first = logits.clone()
                prefill_seconds = time.perf_counter() - started
                decode_started = time.perf_counter()
                for step in range(max_new_tokens):
                    token = int(logits.argmax().item())
                    tokens.append(token)
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    flags = self._flags(text, tokens)
                    if token in self._eos:
                        finish = "eos"; break
                    if flags["repetition_loop"]:
                        finish = "repetition_loop"; break
                    if flags["script_corruption"] or flags["control_count"] or flags["replacement_count"] >= 3:
                        finish = "script_or_unicode_corruption"; break
                    if time.perf_counter() - started > max_prompt_seconds:
                        finish = "timeout"; break
                    self._guard()
                    if step + 1 < max_new_tokens:
                        ids = torch.cat((ids, torch.tensor([[token]], dtype=ids.dtype, device=ids.device)), dim=1)
                        logits = self._forward(ids, k=k, stage="GENERATION", calls=selected_calls)
                if self.synchronize:
                    self.synchronize()
                decode_seconds = time.perf_counter() - decode_started
                minimum_rms, maximum_rms = wrapper.min_rms_ratio, wrapper.max_rms_ratio
            all_calls = calls + selected_calls
            length, original_length = len(tokens), len(values)
            positions = length * original_length + length * (length - 1) // 2
            expected_calls = length + (k if record is not None else 0)
            expected_positions = positions + (k * original_length if record is not None else 0)
            if len(all_calls) != expected_calls or sum(x["prefix_tokens"] for x in all_calls) != expected_positions:
                raise RuntimeError("actual pilot/selected call or position ledger differs")
            self._check_frozen()
            return {"response": text, "token_ids": tokens, **self._flags(text, tokens),
                    "K": k, "alpha": 1., "termination": finish, "eos_emitted": finish == "eos",
                    "truncated": finish == "token_cap", "safety_stopped": finish not in ("eos", "token_cap"),
                    "pilot_used": record is not None, "logical_first_reused": record is not None and k == 0,
                    "pilot_probability": None if record is None else pilot.probability,
                    "pilot_logits_sha256": None if record is None else pilot.logits_sha256,
                    "selected_first_logits_sha256": self._tensor_hash(first),
                    "full_model_calls": len(all_calls), "completed_token_positions": expected_positions,
                    "initial_block_applications": len(all_calls), "extra_layer_applications": 2 * k * length,
                    "pilot_model_calls": int(record is not None), "selected_generation_physical_calls": len(selected_calls),
                    "pilot_positions": 0 if record is None else original_length,
                    "selected_generation_physical_positions": sum(x["prefix_tokens"] for x in selected_calls),
                    "generation_dense_matmul_flops_estimate": self._flops(all_calls),
                    "pilot_dense_matmul_flops_estimate": 0 if record is None else self._flops(record["calls"]),
                    "selected_dense_matmul_flops_estimate": self._flops(selected_calls),
                    "physical_API_attempts": [x["API_attempt"] for x in all_calls],
                    "physical_forward_seconds": sum(x["elapsed_seconds"] for x in all_calls),
                    "probability_seconds": 0. if record is None else record["probability_seconds"],
                    "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
                    "wall_seconds": time.perf_counter() - started,
                    "trajectory_min_rms_ratio": minimum_rms, "trajectory_max_rms_ratio": maximum_rms,
                    "timing_scope": "instrumented full case; pilot/probability and caller work since supplied start included",
                    "generation_cap": max_new_tokens, "state_adjoints": 0, "gold_calls": 0}, first
        except Exception as error:
            self._emit({"event": "GENERATION_FAILED", "K": k, "completed_generation": False,
                        "pilot_used": record is not None, "completed_physical_API_attempts": [x["API_attempt"] for x in calls + selected_calls],
                        "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            raise
