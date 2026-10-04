"""Injected prompt-only greedy collection; no framework import or launch API.

The two-repeat/bulk control follows native_worker.capture's generated-only
numerical profile. Passing this component control never expands attempt 7's
short native qualification or admits study collection. The caller owns resource
guards, durable preservation, tokenizer/model bindings and launch authorization.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

VERSION = "synth-prompt-only-inference-component-1"
EOS_IDS = frozenset((151645, 151643))
ROW_FIELDS = frozenset(("row_id", "prompt_sha256", "prompt_ids", "prompt_tokens", "attention_mask"))
MAX_LOGPROB_ERROR, MEAN_LOGPROB_ERROR = 0.03125, 0.01


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_row(row, *, context_cap=512, generation_cap=32, vocabulary_size=151936):
    """Accept only the prepared public token-row allowlist, never labels/targets."""
    require(type(row) is dict and set(row) == ROW_FIELDS, "prompt-only row fields; oracle/target metadata forbidden")
    require(_hash(row["row_id"]) and _hash(row["prompt_sha256"]), "source row/prompt hashes")
    ids, mask = row["prompt_ids"], row["attention_mask"]
    require(type(ids) is list and bool(ids) and all(type(x) is int and 0 <= x < vocabulary_size for x in ids),
            "typed in-vocabulary prompt IDs")
    require(151643 not in ids, "padding token forbidden in prompt")
    require(type(row["prompt_tokens"]) is int and row["prompt_tokens"] == len(ids), "prompt token count")
    require(type(mask) is list and len(mask) == len(ids) and all(type(x) is int and x == 1 for x in mask),
            "exact all-one unpadded attention mask")
    require(len(ids) + generation_cap <= context_cap, "prompt plus full generation reservation exceeds context window")
    return deepcopy(row)


class InferenceComponent:
    """Greedy, batch-one, no-cache/no-grad component with explicit total dose caps.

    ``call(operation, function)``, ``guard(stage)`` and ``preserve(name, JSON)``
    are mandatory injected boundaries. Preservation must finish synchronously
    before it returns. ``tokenizer.decode(ids, skip_special_tokens=True)`` decodes
    only generated IDs; a CpuTokenizer's already-bound ``_engine`` can be used.
    No model is constructed, no precision wrappers installed, no eval/train mode
    changed, and no optimizer, backward or CUDA initialization API is called.
    """
    def __init__(self, torch, model, tokenizer, call, guard, preserve, *,
                 max_attempted_forwards, max_input_positions, context_cap=512,
                 generation_cap=32, vocabulary_size=151936, device="cpu"):
        require(type(context_cap) is int and 1 <= context_cap <= 512, "context cap must stay within 512")
        require(type(generation_cap) is int and 1 <= generation_cap <= 32, "generation cap must stay within 32")
        require(type(vocabulary_size) is int and vocabulary_size == 151936, "exact Titan vocabulary required")
        require(type(max_attempted_forwards) is int and max_attempted_forwards > 0, "explicit attempted-forward cap")
        require(type(max_input_positions) is int and max_input_positions > 0, "explicit input-position cap")
        require(device in ("cpu", "cuda:0"), "explicit CPU fixture or cuda:0 device")
        require(all(callable(x) for x in (call, guard, preserve, getattr(tokenizer, "decode", None))),
                "injected call/guard/preservation/decoder boundaries required")
        require(getattr(model, "training", None) is False, "model must already be in evaluation mode")
        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.call, self.guard, self.preserve = call, guard, preserve
        self.context_cap, self.generation_cap, self.vocabulary_size = context_cap, generation_cap, vocabulary_size
        self.max_attempted_forwards, self.max_input_positions = max_attempted_forwards, max_input_positions
        self.device = device
        self.scope = ("CPU_FAKE_COMPONENT" if device == "cpu" and
                      getattr(torch, "_synth_fake_component", False) is True else "UNQUALIFIED_NATIVE_IMPLEMENTATION")
        self.operations = {}
        self.positions = {"attempted": 0, "returned": 0, "failed": 0, "pending": 0}
        self.row_cursor = 0
        self.last_receipt = None
        self.halted = False
        self.snapshot = self._snapshot_parameters()

    def _snapshot_parameters(self):
        named = list(self.model.named_parameters(remove_duplicate=False))
        require(bool(named) and len({name for name, _ in named}) == len(named), "nonempty unique parameter names")
        result = {}
        for name, parameter in named:
            require(self.torch.is_tensor(parameter) and str(parameter.device) == self.device,
                    "resident parameter tensor/device")
            require(parameter.dtype in (self.torch.bfloat16, self.torch.float32), "BF16/FP32 parameter storage")
            require(type(parameter.requires_grad) is bool, "typed parameter requires_grad")
            # Existing gradients may belong to a loaded adapter state. Preserve
            # their identity and metadata rather than zeroing them for inference.
            gradient = parameter.grad
            require(gradient is None or (self.torch.is_tensor(gradient) and not gradient.requires_grad),
                    "invalid existing gradient tensor")
            result[name] = {"object": parameter, "data_ptr": parameter.data_ptr(), "dtype": parameter.dtype,
                            "shape": tuple(parameter.shape), "device": str(parameter.device),
                            "requires_grad": parameter.requires_grad, "version": getattr(parameter, "_version", None),
                            "gradient": gradient,
                            "gradient_state": None if gradient is None else self._tensor_state(gradient)}
        return result

    @staticmethod
    def _tensor_state(tensor):
        return (tensor.data_ptr(), tensor.dtype, tuple(tensor.shape), str(tensor.device),
                tensor.requires_grad, getattr(tensor, "_version", None))

    def verify_unchanged(self):
        """Verify resident objects/storage/flags/gradients and normal mutation counters.

        This does not replace the launch owner's independent base-byte hashing:
        illicit writes through ``.data`` can bypass Torch mutation counters.
        """
        named = list(self.model.named_parameters(remove_duplicate=False))
        require(len(named) == len(self.snapshot) and {n for n, _ in named} == set(self.snapshot),
                "parameter registration changed")
        require(getattr(self.model, "training", None) is False, "model evaluation mode changed")
        for name, parameter in named:
            expected = self.snapshot[name]
            require(parameter is expected["object"] and
                    self._tensor_state(parameter) == (expected["data_ptr"], expected["dtype"], expected["shape"],
                        expected["device"], expected["requires_grad"], expected["version"]),
                    "parameter identity/storage/dtype/flags/version changed: " + name)
            require(parameter.grad is expected["gradient"] and
                    (parameter.grad is None or self._tensor_state(parameter.grad) == expected["gradient_state"]),
                    "parameter gradient changed: " + name)
        return {"parameter_aliases": len(self.snapshot), "objects_pointers_dtypes_flags_gradients_preserved": True,
                "mutation_counters_preserved": True, "independent_parameter_byte_verification_required": True}

    def costs(self):
        return deepcopy({"operations": self.operations, "input_positions": self.positions,
                         "attempted_forwards": sum(x["attempted"] for x in self.operations.values()),
                         "maximum_attempted_forwards": self.max_attempted_forwards,
                         "maximum_input_positions": self.max_input_positions})

    def _forward(self, prefix, operation):
        require(not self.halted, "component halted after a failure; no further forward authorized")
        require(type(prefix) is list and 0 < len(prefix) <= self.context_cap and
                all(type(x) is int and 0 <= x < self.vocabulary_size for x in prefix), "bounded single token prefix")
        self.verify_unchanged()
        self.guard("before_" + operation)
        require(self.costs()["attempted_forwards"] < self.max_attempted_forwards and
                self.positions["attempted"] + len(prefix) <= self.max_input_positions, "attempted-forward/input-position budget exhausted")
        torch = self.torch
        ids = torch.tensor([list(prefix)], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        require(torch.is_tensor(ids) and tuple(ids.shape) == (1, len(prefix)) and ids.dtype == torch.long and
                str(ids.device) == self.device and ids.tolist() == [prefix], "exact unaltered input tensor")
        require(torch.is_tensor(mask) and tuple(mask.shape) == tuple(ids.shape) and mask.dtype == torch.long and
                str(mask.device) == self.device and mask.tolist() == [[1] * len(prefix)], "exact tensor attention mask")
        counts = self.operations.setdefault(operation, dict.fromkeys(("attempted", "returned", "failed", "pending"), 0))
        # Count the supplied call boundary even if its before/after journal guard
        # or persistence fails. The journal remains authoritative about that
        # wrapper; this ledger also distinguishes actual model returns.
        counts["attempted"] += 1
        counts["pending"] += 1
        self.positions["attempted"] += len(prefix)
        self.positions["pending"] += len(prefix)
        invoked, returned = False, False

        def forward():
            nonlocal invoked, returned
            require(not invoked, "supplied callable must invoke forward exactly once")
            invoked = True
            with torch.no_grad():
                output = self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            logits = output.logits
            returned = True
            return logits

        try:
            logits = self.call(operation, forward)
            require(invoked and returned, "supplied call omitted the actual model forward")
        except BaseException:
            counts["pending"] -= 1
            counts["returned" if returned else "failed"] += 1
            self.positions["pending"] -= len(prefix)
            self.positions["returned" if returned else "failed"] += len(prefix)
            self.verify_unchanged()
            raise
        counts["pending"] -= 1
        counts["returned"] += 1
        self.positions["pending"] -= len(prefix)
        self.positions["returned"] += len(prefix)
        require(torch.is_tensor(logits) and tuple(logits.shape) == (1, len(prefix), self.vocabulary_size) and
                logits.dtype == torch.float32 and str(logits.device) == self.device and not logits.requires_grad and
                bool(torch.isfinite(logits).all()), "full batch-one FP32 finite no-grad logits required")
        self.verify_unchanged()
        self.guard("after_" + operation)
        return logits

    def _selected(self, logits, position):
        row = logits[0, position]
        token = row.argmax().item()
        require(type(token) is int and 0 <= token < self.vocabulary_size, "typed generated argmax token")
        normalized = self.torch.log_softmax(row, dim=-1)
        require(self.torch.is_tensor(normalized) and normalized.dtype == self.torch.float32 and
                tuple(normalized.shape) == (self.vocabulary_size,) and bool(self.torch.isfinite(normalized).all()),
                "FP32 finite log-softmax row required")
        logprob = normalized[token].item()
        require(type(logprob) in (int, float) and math.isfinite(logprob) and logprob <= 0, "finite selected log probability")
        return token, float(logprob)

    def _trace(self, row, operation, trace, step_preserve=None):
        prefix = list(row["prompt_ids"])
        for _ in range(self.generation_cap):
            logits = self._forward(prefix, operation)
            token, logprob = self._selected(logits, -1)
            trace.append({"token": token, "logprob": logprob, "prefix_length": len(prefix)})
            if step_preserve is not None:
                step_preserve(deepcopy(trace[-1]))
            prefix.append(token)
            del logits
            if token in EOS_IDS:
                break
        return trace

    def _identity(self, row, source_packet_sha256):
        return {"row_id": row["row_id"], "prompt_sha256": row["prompt_sha256"],
                "prompt_ids_sha256": digest(row["prompt_ids"]), "input_row_sha256": digest(row),
                "source_packet_sha256": source_packet_sha256}

    def _receipt(self, row, source_packet_sha256, trace, status):
        ids = [step["token"] for step in trace]
        finished = bool(ids) and ids[-1] in EOS_IDS
        return {"schema_version": 1, "implementation": VERSION, "scope": self.scope, "status": status,
                **self._identity(row, source_packet_sha256), "prompt_tokens": row["prompt_tokens"],
                "raw_generated_ids": ids, "selected_logprobs": [step["logprob"] for step in trace],
                "trace": deepcopy(trace), "finish_reason": "eos" if finished else "generation_cap",
                "truncated": not finished, "eos_token_id": ids[-1] if finished else None,
                "context_cap": self.context_cap, "maximum_generated_tokens": self.generation_cap,
                "decoding": "greedy_no_cache_no_grad", "padding": False, "packing": False,
                "prompt_truncated": False, "oracle_inputs_read": False, "capability_correctness_scored": False,
                "native_launch_authorized": False, "native_qualification_established": False,
                "confirmation_eligible": False, "costs": self.costs()}

    def _persist(self, name, receipt):
        self.last_receipt = deepcopy(receipt)
        try:
            self.preserve(name, deepcopy(receipt))
        except BaseException:
            self.halted = True
            raise

    def collect(self, rows, *, source_packet_sha256):
        """Prevalidate the whole ordered packet; preserve each output before next row.

        Exceptions stop collection immediately. Completed/failure raw receipts
        are retained via the sink and ``last_receipt``; attempted costs remain
        available through ``costs()`` even when the sink itself fails.
        """
        require(not self.halted, "component halted after a failure")
        require(type(rows) is list and 0 < len(rows) <= 4096 and _hash(source_packet_sha256), "bounded rows/source hash")
        validated = [validate_row(row, context_cap=self.context_cap, generation_cap=self.generation_cap,
                                  vocabulary_size=self.vocabulary_size) for row in rows]
        require(len({row["row_id"] for row in validated}) == len(validated), "duplicate source row identity")
        results = []
        for row in validated:
            self.row_cursor += 1
            trace = []
            try:
                self._trace(row, "inference_forward", trace)
                receipt = self._receipt(row, source_packet_sha256, trace, "RAW_RESPONSE_COLLECTED_UNSCORED")
                decoded = self.tokenizer.decode(list(receipt["raw_generated_ids"]), skip_special_tokens=True)
                require(type(decoded) is str and len(decoded.encode("utf-8")) <= 1024 * 1024, "bounded decoded response string")
                receipt["decoded_response"] = decoded
                receipt["parameter_preservation"] = self.verify_unchanged()
            except BaseException as error:
                self.halted = True
                receipt = self._receipt(row, source_packet_sha256, trace, "COLLECTION_FAILED")
                receipt.update({"finish_reason": "error", "truncated": True, "decoded_response": None,
                                "error": {"type": type(error).__name__, "message": str(error)}})
                self._persist("inference_failure_%04d" % self.row_cursor, receipt)
                raise
            self._persist("inference_row_%04d" % self.row_cursor, receipt)
            # A sink can run arbitrary code. Verify again before any next row.
            try:
                self.verify_unchanged()
            except BaseException:
                self.halted = True
                raise
            results.append(deepcopy(receipt))
        return {"schema_version": 1, "scope": self.scope, "rows": results, "costs": self.costs(),
                "source_packet_sha256": source_packet_sha256, "native_launch_authorized": False,
                "native_qualification_established": False, "confirmation_eligible": False}

    def qualify(self, row, *, source_packet_sha256):
        """Optional generated-only two-repeat/prefix-bulk numerical control.

        Exact trace equality and native_worker.capture's unchanged 0.03125 max /
        0.01 mean selected-LP thresholds are separate from collection. This is
        component evidence; CUDA execution still needs its own approved scope.
        """
        require(not self.halted, "component halted after a failure")
        require(_hash(source_packet_sha256), "source packet hash")
        row = validate_row(row, context_cap=self.context_cap, generation_cap=self.generation_cap,
                           vocabulary_size=self.vocabulary_size)
        evidence = {"schema_version": 1, "implementation": VERSION, "scope": self.scope,
                    **self._identity(row, source_packet_sha256), "traces": [[], []],
                    "profile": "two_exact_greedy_repeats_and_generated_only_prefix_bulk",
                    "selected_logprob_max_error_limit": MAX_LOGPROB_ERROR,
                    "selected_logprob_mean_error_limit": MEAN_LOGPROB_ERROR,
                    "gold_continuation_supplied": False, "capability_correctness_scored": False,
                    "native_launch_authorized": False, "native_qualification_established": False,
                    "collection_scope_qualified": False, "confirmation_eligible": False}
        self.row_cursor += 1
        try:
            for repeat, trace in enumerate(evidence["traces"]):
                def preserve_step(value, repeat=repeat, trace=trace):
                    self._persist("inference_q%04d_r%d_t%02d" % (self.row_cursor, repeat, len(trace) - 1),
                                  {"scope": self.scope, **self._identity(row, source_packet_sha256),
                                   "repeat": repeat, "step": len(trace) - 1, **value,
                                   "native_qualification_established": False, "costs": self.costs()})
                self._trace(row, "inference_qualification_forward", trace, preserve_step)
            evidence["repeat_exact"] = evidence["traces"][0] == evidence["traces"][1]
            require(evidence["repeat_exact"], "repeat greedy capture differs")
            trace = evidence["traces"][0]
            prefix = row["prompt_ids"] + [step["token"] for step in trace[:-1]]
            logits = self._forward(prefix, "inference_qualification_bulk_forward")
            errors, argmax_differences = [], []
            evidence["selected_logprob_errors"], evidence["argmax_differences"] = errors, argmax_differences
            for step, observed in enumerate(trace):
                token, logprob = self._selected(logits, len(row["prompt_ids"]) - 1 + step)
                if token != observed["token"]:
                    argmax_differences.append(step)
                # Bulk LP must score the originally selected token, even when
                # the bulk argmax disagrees. Preserve that diagnostic faithfully.
                if token != observed["token"]:
                    lp = self.torch.log_softmax(logits[0, len(row["prompt_ids"]) - 1 + step], dim=-1)
                    logprob = float(lp[observed["token"]].item())
                    require(math.isfinite(logprob) and logprob <= 0, "finite bulk selected log probability")
                errors.append(abs(logprob - observed["logprob"]))
            del logits
            evidence.update({"generated_tokens_per_repeat": len(trace), "prefix_bulk_argmax_equal": not argmax_differences,
                             "selected_logprob_max_error": max(errors),
                             "selected_logprob_mean_error": sum(errors) / len(errors)})
            require(not argmax_differences, "prefix/bulk argmax differs")
            require(max(errors) <= MAX_LOGPROB_ERROR and sum(errors) / len(errors) <= MEAN_LOGPROB_ERROR,
                    "prefix/bulk numerical gate failed")
            evidence.update({"status": "COMPONENT_NUMERICAL_CONTROL_PASSED",
                             "parameter_preservation": self.verify_unchanged(), "costs": self.costs()})
        except BaseException as error:
            self.halted = True
            evidence.update({"status": "COMPONENT_NUMERICAL_CONTROL_FAILED", "costs": self.costs(),
                             "error": {"type": type(error).__name__, "message": str(error)}})
            self._persist("inference_qualification_failure_%04d" % self.row_cursor, evidence)
            raise
        self._persist("inference_qualification_%04d" % self.row_cursor, evidence)
        try:
            self.verify_unchanged()
        except BaseException:
            self.halted = True
            raise
        return deepcopy(evidence)
