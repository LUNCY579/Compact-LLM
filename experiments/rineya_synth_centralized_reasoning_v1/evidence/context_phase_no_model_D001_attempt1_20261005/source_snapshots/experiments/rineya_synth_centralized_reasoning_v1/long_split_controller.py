"""CPU-fake split mechanics only; no native registration, worker or grader.

Fixture identity hashes are declarations, never native provenance authority.
Greedy EOS is preserved; predeclared non-EOS prefix stress is a distinct mode.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import long_inference_component as component

HERE = Path(__file__).resolve().parent
STAGED = HERE.name == "long_context_qualification_work" and HERE.parent.name == "artifacts"
PACKAGE = HERE.parents[1] if STAGED else HERE

FIXTURE = "CPU_FAKE_LONG_SPLIT_MECHANICS"
PACKET_SHA = "208f3533fdc22e01cd345e8a746be21cad67681da734814629e64426d497edfe"
PACKET_BODY_SHA = "4a58b2c32357533da723cd1fb46b0d938eff11f014c70104784f250fe20b6fed"
COMPONENT_SHA = "5816724569a4e5a553428e6a05a0297fb928bbdf3de84c714566b40dba93d327"
IDENTITY_KEYS = {"source_sha256", "model_sha256", "runtime_profile_sha256", "zero_B_sha256", "initial_rng_sha256"}
ROLE_ORDER = ("trace1", "trace2", "bulk")


def require(value, message):
    if not value: raise ValueError(message)


def seal(value): return {**value, "sha256": component.digest(value)}


def checksum(value):
    require(type(value) is dict and value.get("sha256") == component.digest({k: v for k, v in value.items() if k != "sha256"}), "immutable capsule checksum")


def hashed(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def recipe(packet, cap, declared_identity, *, prefix_stress_ids=None):
    checksum(packet)
    require(packet["sha256"] == PACKET_BODY_SHA and cap in (1024, 2048, 4096), "reviewed TRAIN packet/envelope")
    item = next(x for x in packet["rows"] if x["context_cap"] == cap)
    row = component.validate_row(item["row"], context_cap=cap)
    short = component.validate_row(packet["short_parity_row"])
    require(short["prompt_tokens"] == 419 and packet["two_parity_positions"] == 838, "bound419 TRAIN parity; no390 substitution")
    require(item["source_split"] == "train" and item["source_arm"] == "M1" and item["source_skill"] == "R"
            and component.digest(item["messages"]) == row["prompt_sha256"], "TRAIN-only messages binding")
    require(type(declared_identity) is dict and set(declared_identity) == IDENTITY_KEYS
            and all(hashed(v) for v in declared_identity.values()), "explicit CPU fixture identity declarations")
    if prefix_stress_ids is not None:
        require(type(prefix_stress_ids) is list and len(prefix_stress_ids) == 31
                and all(type(t) is int and 0 <= t < 151936 and t not in (151643, 151645) for t in prefix_stress_ids), "31 typed predeclared non-EOS prefix tokens")
        require(any(row["prompt_ids"][i:i+31] == prefix_stress_ids for i in range(len(row["prompt_ids"])-30)), "stress prefix must be a contiguous bound TRAIN-input token span")
    trace_positions = 32*row["prompt_tokens"]+496+838
    bulk_positions = row["prompt_tokens"]+31+838
    require(max(trace_positions, bulk_positions) <= 134144, "unchanged per-scope position ceiling")
    return seal({"scope": FIXTURE, "packet_file_sha256": PACKET_SHA, "packet_body_sha256": PACKET_BODY_SHA,
        "component_sha256": COMPONENT_SHA, "context_cap": cap, "row": row, "short_parity_row": short,
        "prompt_hash_conventions": {"short": "SHA256_RENDERED_UTF8_CHAT_TEMPLATE", "long": "SHA256_CANONICAL_MESSAGES_JSON"},
        "expected_fixture_identity": deepcopy(declared_identity), "trace_scope_forwards": 34,
        "trace_scope_positions": trace_positions, "bulk_scope_forwards": 3, "bulk_scope_positions": bulk_positions,
        "per_scope_hard_position_ceiling": 134144, "per_scope_seconds": 600,
        "prefix_stress_ids": deepcopy(prefix_stress_ids), "native_launch_authorized": False,
        "native_provenance_independently_verified": False, "actual_grading_authorized": False})


class Controller:
    """One fresh injected CPU component per scope, with append-only capsule lineage."""
    def __init__(self, frozen, factory, preserve, guard, *, clock=time.monotonic):
        checksum(frozen)
        require(frozen.get("scope") == FIXTURE, "native split driver unregistered")
        require(all(callable(x) for x in (factory, preserve, guard, clock)), "explicit fixture boundaries")
        require(hashlib.sha256(Path(component.__file__).read_bytes()).hexdigest() == COMPONENT_SHA, "reviewed derived component byte pin")
        packet_path = PACKAGE/"evidence/long_train_prompts_cpu_20261005/TRAIN_PROMPTS_CPU.json"
        raw = packet_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == PACKET_SHA, "bound TRAIN packet file bytes")
        expected = recipe(json.loads(raw), frozen["context_cap"], frozen["expected_fixture_identity"], prefix_stress_ids=frozen["prefix_stress_ids"])
        require(frozen == expected, "exact registered CPU recipe/dose/row binding; no resealed expansion")
        self.frozen, self.factory, self.preserve, self.guard, self.clock = deepcopy(frozen), factory, preserve, guard, clock
        self.capsules, self.engines, self.nonces, self.published_hashes = [], [], set(), []
        self.halted = False
        self.recipe_published_sha256 = frozen["sha256"]

    @property
    def latest_sha256(self): return self.published_hashes[-1] if self.published_hashes else self.frozen["sha256"]

    def reference(self):
        return {cap["role"]: cap["sha256"] for cap in self.capsules if cap["role"] in ("trace1", "trace2")}

    def _identity(self, session):
        value = session["identity"]()
        require(type(value) is dict and set(value) == IDENTITY_KEYS | {"scope_nonce", "fixture"}
                and value["fixture"] is True and type(value["scope_nonce"]) is str and bool(value["scope_nonce"]), "fixture-only fresh scope identity")
        require({k: value[k] for k in IDENTITY_KEYS} == self.frozen["expected_fixture_identity"], "source/model/zero-B/runtime/RNG fixture identity mismatch")
        return deepcopy(value)

    def _dose_issues(self, engine, role):
        if engine is None: return []
        maximum = self.frozen["bulk_scope_forwards"] if role == "bulk" else self.frozen["trace_scope_forwards"]
        positions = self.frozen["bulk_scope_positions"] if role == "bulk" else self.frozen["trace_scope_positions"]
        issues = []
        if (engine.max_attempted_forwards, engine.max_input_positions, engine.context_cap, engine.generation_cap) != (maximum, positions, self.frozen["context_cap"], 32):
            issues.append("mutable component limits differ from frozen role")
        costs = engine.costs()
        if costs["attempted_forwards"] > maximum: issues.append("frozen role forward ceiling exceeded")
        if costs["input_positions"]["attempted"] > positions: issues.append("frozen role position ceiling exceeded")
        if costs["input_positions"]["attempted"] > 134144: issues.append("hard134144 position ceiling exceeded")
        allowed = {"CPU_fixture_parity_base_forward", "CPU_fixture_parity_zero_B_forward",
                   "long_prefix_bulk_forward" if role == "bulk" else "declared_non_EOS_prefix_stress_forward" if role == "prefix_stress" else "long_greedy_trace_forward"}
        if set(costs["operations"]) - allowed: issues.append("undeclared role operation")
        return issues

    def _exact_roster(self, engine, role, trace):
        require(not self._dose_issues(engine, role), "frozen role/source dose changed")
        row = self.frozen["row"]
        require(type(trace) is list and 1 <= len(trace) <= 32, "complete actual trace roster")
        for index, step in enumerate(trace):
            require(set(step) == {"token", "logprob", "prefix_length"} and type(step["token"]) is int
                    and 0 <= step["token"] < 151936 and math.isfinite(step["logprob"]) and step["logprob"] <= 0
                    and step["prefix_length"] == row["prompt_tokens"]+index, "exact trace token/LP/prefix roster")
        prefix = self.frozen["short_parity_row"]["prompt_ids"]
        expected = [prefix, prefix]
        operation = "long_prefix_bulk_forward" if role == "bulk" else "declared_non_EOS_prefix_stress_forward" if role == "prefix_stress" else "long_greedy_trace_forward"
        if role == "prefix_stress":
            require(len(trace) == 32, "complete declared32 prefix stress")
            expected += [row["prompt_ids"]+self.frozen["prefix_stress_ids"][:i] for i in range(32)]
            count = 32
        else:
            require(not any(step["token"] in component.EOS_IDS for step in trace[:-1])
                    and (len(trace) == 32 or trace[-1]["token"] in component.EOS_IDS), "complete EOS-respecting greedy trace")
            tokens = [step["token"] for step in trace]
            expected += [row["prompt_ids"]+tokens[:-1]] if role == "bulk" else [row["prompt_ids"]+tokens[:i] for i in range(len(trace))]
            count = 1 if role == "bulk" else len(trace)
        counts = lambda n: {"attempted": n, "returned": n, "failed": 0, "pending": 0}
        operations = {"CPU_fixture_parity_base_forward": counts(1), "CPU_fixture_parity_zero_B_forward": counts(1), operation: counts(count)}
        costs = engine.costs()
        require(engine.model.calls == expected and costs["operations"] == operations
                and costs["attempted_forwards"] == len(expected)
                and costs["input_positions"] == {"attempted": sum(map(len, expected)), "returned": sum(map(len, expected)), "failed": 0, "pending": 0},
                "exact frozen role operation/physical prefix roster; metered extra call rejected")

    def run(self, role, *, expected_latest_sha256, references=None):
        require(not self.halted, "split controller halted; no implicit retry")
        checksum(self.frozen)
        require(self.frozen["sha256"] == self.recipe_published_sha256, "published recipe changed")
        for cap, published in zip(self.capsules, self.published_hashes):
            checksum(cap); require(cap["sha256"] == published, "published immutable trace/cost capsule changed")
        require(len(self.capsules) == len(self.published_hashes), "published capsule roster changed")
        require(expected_latest_sha256 == self.latest_sha256, "rollback or stale cumulative lineage")
        expected_role = ROLE_ORDER[len(self.capsules)] if len(self.capsules) < 3 else "prefix_stress"
        require(role == expected_role and len(self.capsules) < 4, "declared append-only scope order")
        if role == "prefix_stress": require(self.frozen["prefix_stress_ids"] is not None, "separate prefix stress not predeclared")
        if role == "bulk": require(references == self.reference(), "exact immutable trace1/trace2 reference hashes")
        start = self.clock()
        engine = None
        def guard(stage):
            require(0 <= self.clock()-start <= 600, "unchanged600s per-scope deadline")
            require(not self._dose_issues(engine, role), "frozen role/source dose changed during scope")
            self.guard(role, stage)
        try:
            guard("before_fresh_factory")
            session = self.factory(deepcopy(self.frozen), role, guard)
            engine = session["component"]
            require(isinstance(engine, component.InferenceComponent) and engine.scope == "CPU_FAKE_COMPONENT"
                    and engine.device == "cpu", "CPU fake component only; native/model driver closed")
            require(all(engine is not old and engine.model is not old.model for old in self.engines), "fresh component/model required; no reused scope")
            require(hasattr(engine.model, "calls") and engine.model.calls == [] and engine.costs()["attempted_forwards"] == 0,
                    "fresh CPU fixture; no hidden factory warmup forwards")
            engine.guard = guard
            identity = self._identity(session)
            require(identity["scope_nonce"] not in self.nonces, "fresh scope nonce required")
            self.engines.append(engine); self.nonces.add(identity["scope_nonce"])
            maximum = self.frozen["bulk_scope_forwards"] if role == "bulk" else self.frozen["trace_scope_forwards"]
            positions = self.frozen["bulk_scope_positions"] if role == "bulk" else self.frozen["trace_scope_positions"]
            require(engine.max_attempted_forwards == maximum and engine.max_input_positions == positions
                    and engine.context_cap == self.frozen["context_cap"] and engine.generation_cap == 32,
                    "exact scope dose/context; no enlarged fixture budget")
            # Two explicit same-fake-model calls exercise parity accounting only.
            # They do not prove physical dense-base/zero-adapter parity.
            parity = []
            for name in ("CPU_fixture_parity_base_forward", "CPU_fixture_parity_zero_B_forward"):
                guard("before_"+name)
                logits = engine._forward(self.frozen["short_parity_row"]["prompt_ids"], name)
                parity.append(engine._selected(logits, -1)); del logits
            require(parity[0] == parity[1], "CPU fixture parity differs")
            row, trace, comparison = self.frozen["row"], [], None
            if role in ("trace1", "trace2"):
                def persist_step(step):
                    guard("before_token_preservation")
                    self.preserve(role+"_token_%02d"%len(trace), seal({"scope": FIXTURE, "role": role,
                        "recipe_sha256": self.frozen["sha256"], "identity": identity, "step": deepcopy(step),
                        "complete_trace_prefix": deepcopy(trace), "costs": engine.costs()}))
                engine._trace(row, "long_greedy_trace_forward", trace, persist_step)
            elif role == "bulk":
                first, second = self.capsules[:2]
                checksum(first); checksum(second)
                require(first["trace"] == second["trace"] and first["identity_common"] == second["identity_common"], "both complete greedy repeats must match")
                trace = deepcopy(first["trace"])
                logits = engine._forward(row["prompt_ids"]+[x["token"] for x in trace[:-1]], "long_prefix_bulk_forward")
                errors, differences = [], []
                for step, observed in enumerate(trace):
                    token, lp = engine._selected(logits, row["prompt_tokens"]-1+step)
                    if token != observed["token"]:
                        differences.append(step)
                        lp = float(engine.torch.log_softmax(logits[0, row["prompt_tokens"]-1+step], dim=-1)[observed["token"]].item())
                        require(math.isfinite(lp) and lp <= 0, "finite originally selected bulk LP")
                    errors.append(abs(lp-observed["logprob"]))
                del logits
                comparison = {"compared_actual_steps": len(trace), "argmax_differences": differences,
                    "maximum_selected_LP_error": max(errors), "mean_selected_LP_error": sum(errors)/len(errors)}
                require(not differences and max(errors) <= .03125 and sum(errors)/len(errors) <= .01, "unchanged bulk argmax/numerical gates")
            else:
                fixed = self.frozen["prefix_stress_ids"]
                for step in range(32):
                    logits = engine._forward(row["prompt_ids"]+fixed[:step], "declared_non_EOS_prefix_stress_forward")
                    token, lp = engine._selected(logits, -1); del logits
                    trace.append({"token": token, "logprob": lp, "prefix_length": row["prompt_tokens"]+step})
                # Selected EOS is recorded but never used to fabricate a greedy prefix.
            guard("after_scope_math")
            require(self._identity(session) == identity, "fixture identity changed during scope")
            engine.verify_unchanged()
            self._exact_roster(engine, role, trace)
            costs = engine.costs()
            require(len(engine.model.calls) == costs["attempted_forwards"]
                    and sum(len(ids) for ids in engine.model.calls) == costs["input_positions"]["attempted"],
                    "every actual CPU fake forward/position counted; no hidden extra calls")
            previous = self.capsules[-1]["cumulative_costs"] if self.capsules else {"attempted_forwards": 0, "input_positions": 0}
            cumulative = {"attempted_forwards": previous["attempted_forwards"]+costs["attempted_forwards"],
                          "input_positions": previous["input_positions"]+costs["input_positions"]["attempted"]}
            result = seal({"scope": FIXTURE, "role": role, "recipe_sha256": self.frozen["sha256"],
                "previous_capsule_sha256": self.latest_sha256, "identity_common": {k: identity[k] for k in IDENTITY_KEYS},
                "scope_nonce": identity["scope_nonce"], "trace": trace, "comparison": comparison, "scope_costs": costs,
                "cumulative_costs": cumulative, "greedy_EOS_respected": role != "prefix_stress",
                "greedy_finish_reason": None if role == "prefix_stress" else "eos" if trace[-1]["token"] in component.EOS_IDS else "generation_cap",
                "prefix_stress_selected_EOS_observed": role == "prefix_stress" and any(step["token"] in component.EOS_IDS for step in trace),
                "full32_greedy_coverage": role != "prefix_stress" and len(trace) == 32,
                "stress_is_greedy_trace": False, "native_provenance_verified": False,
                "native_zero_B_parity_proven": False, "native_launch_authorized": False,
                "native_qualification_established": False, "actual_grading_performed": False})
            self.preserve(role+"_complete", deepcopy(result))
            guard("after_scope_capsule_preservation")
            require(self._identity(session) == identity, "sink changed fixture identity")
            engine.verify_unchanged()
            self._exact_roster(engine, role, trace)
            require(engine.costs() == costs and len(engine.model.calls) == costs["attempted_forwards"]
                    and sum(len(ids) for ids in engine.model.calls) == costs["input_positions"]["attempted"],
                    "sink changed actual scope call/cost ledger")
            self.capsules.append(deepcopy(result))
            self.published_hashes.append(result["sha256"])
            return deepcopy(result)
        except BaseException as error:
            self.halted = True
            if engine is not None: engine.halted = True
            failure = seal({"scope": FIXTURE, "role": role, "recipe_sha256": self.frozen["sha256"],
                "previous_capsule_sha256": self.latest_sha256, "partial_trace": deepcopy(locals().get("trace", [])),
                "actual_scope_costs": None if engine is None else engine.costs(),
                "actual_observed_CPU_fixture_forwards": None if engine is None else len(getattr(engine.model, "calls", [])),
                "actual_observed_CPU_fixture_positions": None if engine is None else sum(len(ids) for ids in getattr(engine.model, "calls", [])),
                "actual_observed_CPU_fixture_prefix_hashes": None if engine is None else [component.digest(ids) for ids in getattr(engine.model, "calls", [])],
                "actual_observed_CPU_fixture_prefix_lengths": None if engine is None else [len(ids) for ids in getattr(engine.model, "calls", [])],
                "frozen_role_dose_violations": self._dose_issues(engine, role),
                "error": {"type": type(error).__name__, "message": str(error)}, "native_launch_authorized": False})
            self.preserve(role+"_failed", failure)
            raise
