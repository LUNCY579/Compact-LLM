"""Deferred native reader qualification. Top-level imports are stdlib only.

Reviewed D2 containment primitives are imported read-only. This namespace owns
new serving math/controls and never calls D2's launch contract or validator.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import threading
import time
import traceback
import types
import uuid

from . import native
from .operations import OperationsStore, host_state_path, canonical
from .reader import (CONTRACT, EOS_IDS, PACKAGE, SCOPE, ReaderBlocked, digest,
                     likelihood_measurement, manual_greedy, qualifier_plan,
                     read_json, require, require_same_prefix, sha, validate_request,
                     verify_qualification)

ROOT = PACKAGE.parents[1]
GIB = 2 ** 30
# Explicit adapters to the exact source-pinned, reviewed operations primitives.
collect_telemetry = native.collect_telemetry
admission_reasons = native.admission_reasons
job_process_ids = native.job_process_ids
job_member_witness = native.job_member_witness
owned_job_rss_snapshot = native.owned_job_rss_snapshot
resume_suspended_primary_thread = native.resume_suspended_primary_thread
duplicate_owned_job_query_handle = native.duplicate_owned_job_query_handle
drain_owned_job = native.drain_owned_job
read_event_journal = native.read_event_journal
committed_snapshot = native.committed_snapshot
write_new = native.write_new
safe_numbers = native.safe_numbers
verify_sources = native.verify_sources
verify_assets = native.verify_assets


def load_contract(expected_sha256):
    require(sha(CONTRACT) == expected_sha256, "reader contract drift")
    cfg = read_json(CONTRACT)
    require(cfg["scope"] == SCOPE and cfg["historical_bulk_capture"]["status"] == "FAILED_UNRESOLVED",
            "distinct reader scope/old bulk failure required")
    require(cfg["historical_bulk_capture"]["tolerances"] == {"max_abs": .03125, "mean_abs": .01}, "numerical limits changed")
    require(cfg["limits"] == {"worker_rss_gib": 2.0, "host_reserve_gib": 1.25, "preworker_host_gib": 3.3125,
            "gpu_preload_free_gib": 4.0, "gpu_runtime_free_gib": .5, "disk_free_gib": 4.0,
            "wall_seconds": 600, "watchdog_poll_seconds": .1}, "reader resource limits changed")
    qualifier_plan(cfg)
    return cfg


def binding(cfg, contract_sha256, commit):
    return {"source_commit": commit, "contract_sha256": contract_sha256,
            "reader_source_sha256": cfg["source_sha256"]["experiments/rineya_centralized_reasoning_system_v1/reader.py"],
            "execution_source_sha256": cfg["source_sha256"]["experiments/rineya_centralized_reasoning_system_v1/reader_execution.py"],
            "native_helper_sha256": cfg["source_sha256"]["experiments/rineya_centralized_reasoning_system_v1/native.py"],
            "model_sha256": cfg["asset_sha256"]["model.safetensors"], "config_sha256": cfg["asset_sha256"]["config.json"],
            "tokenizer_sha256": cfg["asset_sha256"]["tokenizer.json"], "template_sha256": cfg["chat_template_sha256"],
            "runtime_sha256": digest(cfg["runtime"])}


def review_matches(review, expected):
    require(type(review) is dict and review.get("status") == "APPROVED_FOR_READER_QUALIFICATION"
            and review.get("binding") == expected, "independent exact reader review missing/mismatched")
    reviewer = review.get("reviewer")
    require(type(reviewer) is str and reviewer.strip().rstrip("/").rsplit("/", 1)[-1] != "runtime_audit",
            "reader builder cannot approve its contract")


def preflight(contract_sha256, review_path=None):
    reasons, observations = [], {}
    try:
        cfg = load_contract(contract_sha256)
        source = committed_snapshot(); observations["source"] = source
        if source["dirty"]: reasons.append("clean committed reader source required")
        verify_sources(cfg); verify_assets(cfg)
        observations["runtime"] = native.runtime_metadata()
        require(observations["runtime"] == cfg["runtime"], "reader pinned runtime drift")
        require(os.name == "nt", "Windows reviewed containment required")
        review_matches(read_json(review_path) if review_path else None, binding(cfg, contract_sha256, source["commit"]))
        observations["host"] = collect_telemetry(desktop_identities=cfg["gpu_process_registry"])
        reasons.extend(admission_reasons(cfg, observations["host"]))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        reasons.append(str(error))
    return {"status": "BLOCKED" if reasons else "READY_READER_QUALIFICATION_ONLY", "reasons": reasons,
            "observations": observations, "reader_qualified": False, "training_qualified": False,
            "native_launches_this_preflight": 0}


def _mask_sha(length):
    return hashlib.sha256(b"".join(struct.pack("<H", 0 if column <= row else 65407)
                                   for row in range(length) for column in range(length))).hexdigest()


class ReaderSession:
    """One resident frozen checkpoint. Never mixes hidden state/KV across calls."""
    def __init__(self, torch, model, tokenizer, accounting, guard, event, cfg, run_binding):
        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.accounting, self.guard, self.event, self.cfg = accounting, guard, event, cfg
        self.binding = run_binding
        self.records, self.witness, self.stage, self.keep = [], {}, "", 1
        from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import tensor_digest
        self.tensor_digest = lambda value: tensor_digest(torch, value)
        self.hooks = [model.model.layers[0].register_forward_pre_hook(self._decoder, with_kwargs=True),
                      model.model.rotary_emb.register_forward_pre_hook(self._rotary, with_kwargs=True),
                      model.model.norm.register_forward_hook(self._norm)]

    def close(self):
        for hook in self.hooks: hook.remove()

    def _decoder(self, _module, _args, kwargs):
        mask = kwargs.get("attention_mask"); length = self.length
        require(mask is not None and tuple(mask.shape) == (1, 1, length, length)
                and mask.dtype == self.torch.bfloat16 and kwargs.get("past_key_values") is None,
                "actual causal mask/cache shape mismatch")
        actual = self.tensor_digest(mask)
        require(actual == _mask_sha(length), "actual causal mask raw bytes differ")
        self.witness["mask"] = {"shape": list(mask.shape), "dtype": "bfloat16", "raw_sha256": actual,
                "rows": length, "columns": length, "allowed_uint16": 0, "forbidden_uint16": 65407,
                "allowed_condition": "column <= row", "byte_order": "little_endian"}

    def _rotary(self, _module, args, kwargs):
        positions = args[1] if len(args) > 1 else kwargs.get("position_ids")
        require(positions is not None and positions.dtype == self.torch.long
                and positions.cpu().tolist() == [list(range(self.length))], "actual rotary position mismatch")
        self.witness["positions"] = {"shape": [1, self.length], "dtype": "int64",
                                    "ids": positions.cpu().tolist(), "raw_sha256": self.tensor_digest(positions)}

    def _norm(self, _module, _args, output):
        require(tuple(output.shape) == (1, self.length, 1024) and output.dtype == self.torch.bfloat16,
                "normalized hidden shape/dtype mismatch")
        self.witness["hidden"] = output[:, -self.keep:].detach().clone()

    def _record(self, ids, logits):
        t = self.torch
        require(tuple(logits.shape) == (1, self.keep, 151936) and logits.dtype == t.bfloat16
                and bool(t.isfinite(logits).all()), "nonfinite reader logits/shape/dtype")
        hidden = self.witness.pop("hidden")
        self._last_hidden = hidden
        require(bool(t.isfinite(hidden).all()) and set(self.witness) == {"mask", "positions"}, "incomplete passive witness")
        row = logits[0, -1].float(); token = int(row.argmax())
        require(float(row[token]) == float(row.max()), "raw argmax mismatch")
        record = {"stage": self.stage, "input_ids": list(ids), "input_ids_sha256": digest(ids),
                  "input_length": len(ids), "logits_to_keep": self.keep, "logits_shape": list(logits.shape),
                  "raw_logits_sha256": self.tensor_digest(logits), "raw_hidden_sha256": self.tensor_digest(hidden),
                  "argmax_id": token, "lowest_id_argmax_verified": True,
                  "selected_raw_logit": float(row[token]), "selected_model_logprob": float(row.log_softmax(-1)[token]),
                  "witnesses": dict(self.witness)}
        self.records.append(record); self.event("reader_forward_measurement", record=record)
        return record

    def forward(self, ids, keep=1, stage="reader", *, wrapped=True):
        require(type(ids) is list and 1 <= len(ids) <= 576 and type(keep) is int and 1 <= keep <= len(ids), "reader forward envelope")
        require(self.accounting.snapshot()["operations"]["forward"]["attempted"] < 278, "reader qualification forward budget")
        self.length, self.keep, self.stage, self.witness = len(ids), keep, stage, {}
        self.guard("before_" + stage)
        tensor = self.torch.tensor([ids], device="cuda", dtype=self.torch.long)
        kwargs = {"input_ids": tensor, "attention_mask": self.torch.ones_like(tensor),
                  "position_ids": self.torch.arange(len(ids), device="cuda")[None, :],
                  "cache_position": self.torch.arange(len(ids), device="cuda"),
                  "use_cache": False, "logits_to_keep": keep}
        # Both routes call stock Qwen; wrapper adds only request/schema checks.
        if wrapped:
            require(kwargs["use_cache"] is False and kwargs["logits_to_keep"] == keep, "no-op wrapper scope")
        output = self.accounting.call("forward", lambda: self.model(**kwargs).logits,
                                      stage=stage, input_positions=len(ids))
        record = self._record(ids, output)
        self.guard("after_" + stage)
        return record, output

    def decode(self, prompt, cap, stage):
        def forward(ids, keep, suffix):
            return self.forward(ids, keep, stage + "_" + suffix)[0]
        rollout = manual_greedy({"input_ids": prompt, "attention_mask": [1] * len(prompt), "max_new_tokens": cap}, forward)
        rollout.update(raw_text=self.tokenizer.decode(rollout["generated_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                       visible_text=self.tokenizer.decode(rollout["visible_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        return rollout

    def stock_generate(self, prompt, cap, stage):
        """Observe stock generate without replacing its forward/selection code."""
        from transformers import GenerationConfig
        t, captured, policy = self.torch, [], {}
        def before(_module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            require(ids is not None and tuple(ids.shape)[0] == 1 and kwargs.get("use_cache") is False
                    and kwargs.get("past_key_values") is None, "stock reference left no-cache batch1 path")
            self.length, self.keep, self.stage, self.witness = ids.shape[1], 1, stage + "_" + str(len(captured)), {}
            require(kwargs.get("logits_to_keep") == 1, "stock logits-retention changed")
            require(kwargs.get("attention_mask").cpu().tolist() == [[1] * self.length]
                    and kwargs.get("position_ids").cpu().tolist() == [list(range(self.length))], "stock input mask/position mismatch")
            self.guard("before_" + self.stage)
            require(self.accounting.snapshot()["operations"]["forward"]["attempted"] < 278, "stock forward budget")
            self._stock_call = self.accounting.begin("forward", stage=self.stage, input_positions=self.length)
            self._stock_ids = ids[0].cpu().tolist()
        def after(_module, _args, output):
            self.accounting.returned(self._stock_call)
            captured.append(self._record(self._stock_ids, output.logits)); self.guard("after_" + self.stage)
        original_processors, original_stops = self.model._get_logits_processor, self.model._get_stopping_criteria
        def processors(*args, **kwargs):
            value = original_processors(*args, **kwargs); policy["processors"] = [type(x).__name__ for x in value]
            require(not value, "active stock logits processor changes serving policy")
            return value
        def stops(*args, **kwargs):
            value = original_stops(*args, **kwargs); policy["stops"] = [type(x).__name__ for x in value]
            require(set(policy["stops"]) == {"MaxLengthCriteria", "EosTokenCriteria"}, "unregistered stopping criterion")
            return value
        self.model._get_logits_processor, self.model._get_stopping_criteria = processors, stops
        hooks = [self.model.register_forward_pre_hook(before, with_kwargs=True), self.model.register_forward_hook(after)]
        try:
            config = GenerationConfig(**self.cfg["stock_generation_config"], max_new_tokens=cap)
            ids = t.tensor([prompt], device="cuda", dtype=t.long)
            result = self.model.generate(input_ids=ids, attention_mask=t.ones_like(ids), generation_config=config,
                                         return_dict_in_generate=True, output_logits=True, output_scores=False,
                                         synced_gpus=False)
            generated = result.sequences[0, len(prompt):].cpu().tolist()
            require(len(generated) == len(captured) == len(result.logits) and 1 <= len(generated) <= cap,
                    "stock generation coverage mismatch")
            for token, record, raw in zip(generated, captured, result.logits):
                require(raw.dtype == t.float32 and bool(t.equal(raw, raw.to(t.bfloat16).float())), "stock raw FP32 copy not BF16-preserving")
                require(token == record["argmax_id"] and self.tensor_digest(raw.to(t.bfloat16)[:, None, :]) == record["raw_logits_sha256"],
                        "stock raw logits/token extraction mismatch")
            visible = generated[:-1] if generated[-1] in EOS_IDS else generated
            return {"generated_ids": generated, "steps": captured, "stop_reason": "eos" if generated[-1] in EOS_IDS else "cap",
                    "raw_text": self.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    "visible_text": self.tokenizer.decode(visible, skip_special_tokens=False, clean_up_tokenization_spaces=False), "policy": policy}
        except BaseException as error:
            call = getattr(self, "_stock_call", None)
            if call in self.accounting.pending and self.accounting.stack[-1] == call:
                self.accounting.failed(call, error)
            raise
        finally:
            for hook in hooks: hook.remove()
            self.model._get_logits_processor, self.model._get_stopping_criteria = original_processors, original_stops

    def serve(self, request, qualification):
        verify_qualification(qualification, self.binding)
        request = validate_request(request)
        return self.decode(request["input_ids"], request["max_new_tokens"], "qualified_serving")


def qualify_session(session, cfg):
    plan = qualifier_plan(cfg)
    blocks = [{**block, "status": "UNATTEMPTED", "actual_forwards": 0, "actual_positions": 0} for block in plan["blocks"]]
    checks, rollouts, bulk = {}, {}, {}
    def block(index, action):
        before = session.accounting.snapshot(); blocks[index]["status"] = "RUNNING"
        try:
            value = action(); blocks[index]["status"] = "COMPLETE"; return value
        finally:
            after = session.accounting.snapshot()
            blocks[index]["actual_forwards"] = after["operations"]["forward"]["attempted"] - before["operations"]["forward"]["attempted"]
            blocks[index]["actual_positions"] = after["forward_input_positions_attempted"] - before["forward_input_positions_attempted"]
            require(blocks[index]["actual_forwards"] <= blocks[index]["max_forwards"]
                    and blocks[index]["actual_positions"] <= blocks[index]["max_input_positions"], "qualification block budget exceeded")
    session.event("qualification_plan", blocks=blocks)
    try:
        def parity():
            for length in (128, 512, 576):
                ids = cfg["historical_prompt_ids"] if length == 128 else cfg["long_prompt_ids"] if length == 512 else cfg["long_prompt_ids"] + cfg["mechanical_continuation_ids"]
                direct, _ = session.forward(ids, stage="direct_" + str(length), wrapped=False)
                owned, _ = session.forward(ids, stage="wrapper_" + str(length))
                require(direct["raw_logits_sha256"] == owned["raw_logits_sha256"] and direct["raw_hidden_sha256"] == owned["raw_hidden_sha256"], "no-op raw parity failed")
            checks["direct_wrapper_parity"] = True
        block(0, parity)
        def natural(prompt, cap, label, frozen=None):
            owned = session.decode(prompt, cap, label + "_owned")
            if frozen is not None: require(owned["generated_ids"] == frozen, "historical natural output drift")
            stock = session.stock_generate(prompt, cap, label + "_stock")
            require(owned["generated_ids"] == stock["generated_ids"] and owned["stop_reason"] == stock["stop_reason"]
                    and owned["raw_text"] == stock["raw_text"] and owned["visible_text"] == stock["visible_text"], "independent stock decoding mismatch")
            for a, b in zip(owned["steps"], stock["steps"]):
                require(a["raw_logits_sha256"] == b["raw_logits_sha256"] and a["raw_hidden_sha256"] == b["raw_hidden_sha256"], "stock raw forward parity mismatch")
            replay = [session.forward(prompt + owned["generated_ids"][:i], stage=label + "_replay_" + str(i))[0]
                      for i in range(len(owned["generated_ids"]))]
            require(require_same_prefix(owned["steps"], replay)["passed"], "exact-prefix likelihood mismatch")
            rollouts[label] = {"owned": owned, "stock": stock, "replay": replay}
        block(1, lambda: natural(cfg["historical_prompt_ids"], 3, "historical", cfg["historical_generated_ids"]))
        block(2, lambda: natural(cfg["long_prompt_ids"], 64, "long"))
        block(3, lambda: [session.forward(ids, stage="FORCED_MECHANICAL_NOT_GENERATED_" + str(i))
                         for i, ids in enumerate(plan["forced_mechanical_prefixes"])])
        def bulk_diagnostics():
            for label, prompt in (("historical", cfg["historical_prompt_ids"]), ("long", cfg["long_prompt_ids"])):
                generated = rollouts[label]["owned"]["generated_ids"]
                record, logits = session.forward(prompt + generated, len(generated) + 1, label + "_bulk_DIAGNOSTIC_ONLY")
                target = session.torch.tensor(generated, device="cuda")[:, None]
                lps = logits[0, :-1].float().log_softmax(-1).gather(1, target).squeeze(1).cpu().tolist()
                bulk[label] = {"record": record, "comparison": likelihood_measurement(rollouts[label]["owned"]["model_logprobs"], lps),
                               "operation_eligible": False, "old_gate_relabelled": False}
                if label == "historical":
                    require(record["raw_logits_sha256"] == cfg["historical_bulk_logits_sha256"], "historical failed bulk endpoint drift")
        block(4, bulk_diagnostics)
        # Implement the four equal-shape operations directly for fixed positions.
        def suffix_controls():
            ids = cfg["long_prompt_ids"] + cfg["mechanical_continuation_ids"]
            baseline, logits = session.forward(ids, 4, "max576_baseline")
            baseline_logits = logits.detach().clone()
            baseline_hidden = session._last_hidden.clone()
            baseline_hidden_hashes = []
            # Passive norm witness from each forward is hashed over keep4. The
            # logits protected prefixes are independently compared byte-for-byte.
            for start in (573, 574, 575):
                mutated = list(ids)
                for i in range(start, 576): mutated[i] = (mutated[i] + 1) % 151936
                record, other = session.forward(mutated, 4, "max576_mutate_" + str(start))
                count = start - 572
                require(session.tensor_digest(baseline_logits[:, :count]) == session.tensor_digest(other[:, :count]), "max576 protected logits causal failure")
                require(session.tensor_digest(baseline_hidden[:, :count]) == session.tensor_digest(session._last_hidden[:, :count]),
                        "max576 protected normalized hidden causal failure")
                baseline_hidden_hashes.append({"mutation_start": start, "protected_positions": list(range(572, start)),
                                              "left_logits_sha256": session.tensor_digest(baseline_logits[:, :count]),
                                              "right_logits_sha256": session.tensor_digest(other[:, :count]),
                                              "left_hidden_sha256": session.tensor_digest(baseline_hidden[:, :count]),
                                              "right_hidden_sha256": session.tensor_digest(session._last_hidden[:, :count]), "record": record})
            checks["maximum_shape_causality"] = baseline_hidden_hashes
        block(5, suffix_controls)
        checks.update(stock_generation_parity=True, exact_prefix_replay=True, sustained_mechanical_shapes=True,
                      native_natural_eos_observed=any(r["owned"]["stop_reason"] == "eos" for r in rollouts.values()))
        require(checks["native_natural_eos_observed"], "no native natural EOS observed")
        return {"blocks": blocks, "checks": checks, "rollouts": rollouts, "bulk_diagnostics": bulk,
                "natural_cap_stop_observed": any(r["owned"]["stop_reason"] == "cap" for r in rollouts.values()),
                "bulk_capture_qualified": False, "training_qualified": False}
    finally:
        session.event("qualification_blocks_final", blocks=blocks)


def worker(args, cfg):
    require(os.name == "nt", "registered reader worker is Windows-only")
    output = args.output.resolve()
    require(output.is_relative_to(PACKAGE / "results") and args.go_path.resolve().parent == output, "reader worker path escape")
    deadline = time.monotonic() + 5
    while not args.go_path.exists() and time.monotonic() < deadline: time.sleep(.02)
    go = read_json(args.go_path); query = go.get("own_job_query_handle"); consumed = False
    try:
        require(all(go.get(k) == v for k, v in {"contract_sha256": args.contract_sha256, "owner": args.owner, "fence": args.fence}.items()), "reader GO mismatch")
        import psutil
        parents = psutil.Process().parents()
        require(go["supervisor_pid"] in [p.pid for p in parents] and go["worker_tree_pid"] in [os.getpid()] + [p.pid for p in parents], "reader supervised ancestry mismatch")
        command = psutil.Process(go["supervisor_pid"]).cmdline()
        require("--qualify" in command and "experiments.rineya_centralized_reasoning_system_v1.reader" in command, "direct reader worker unsupported")
        consumed = type(query) is int and query > 0
        containment = native.current_windows_job_witness(query, go["query_handle_target_pid"], go["query_handle_target_creation_time"])
    finally:
        if not consumed and type(query) is int and query > 0: native.close_unconsumed_query_handle(query)
    source = committed_snapshot(); require(not source["dirty"], "reader worker dirty source")
    run_binding = binding(cfg, args.contract_sha256, source["commit"])
    review_matches(read_json(args.review), run_binding)
    verify_sources(cfg); require(native.runtime_metadata() == cfg["runtime"], "reader runtime drift")
    store = OperationsStore(args.host_db, output / "receipts")
    started, result, session = time.monotonic(), {"status": "RUNNING_READER_QUALIFICATION", "scope": SCOPE,
                  "binding": run_binding, "model_loads": 0, "resources": [], "bulk_capture_qualified": False,
                  "reader_qualified": False, "training_qualified": False}, None
    def event(kind, **detail):
        store._require_lease(args.owner, args.fence)
        with (output / "EVENTS.jsonl").open("ab") as stream:
            stream.write(canonical({"kind": kind, "time": time.time(), **safe_numbers(detail)}) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting, tensor_digest
    accounting = CallAccounting(event, maximum_optimizer_calls=0)
    try:
        write_new(output / "CONTAINMENT_RECEIPT.json", containment)
        event("framework_import_attempt")
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        from experiments.post_training_mimo_centralized_v1.scripts import stream_checkpoint
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.device_count() == 1
                and torch.version.cuda == "12.6", "reader CUDA/BF16 runtime mismatch")
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        write_new(output / "RUNTIME_RECEIPT.json", {"binding": run_binding, "runtime": native.runtime_metadata(),
                   "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0), "dtype": "bfloat16",
                   "attention": "eager", "cache": False, "adapters": False, "allow_tf32_fixed": False,
                   "reduction_flags_changed": False, "reduction_flags": {name: {"available": hasattr(torch.backends.cuda.matmul, name),
                        "value": getattr(torch.backends.cuda.matmul, name, None)} for name in
                   ("allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction",
                    "allow_bf16_reduced_precision_reduction_split_k", "allow_fp16_reduced_precision_reduction_split_k", "allow_fp16_accumulation")}})
        process = psutil.Process()
        def guard(stage, preload=False):
            store._require_lease(args.owner, args.fence)
            require(committed_snapshot() == source and time.monotonic() - started <= 600, "reader source/wall drift")
            torch.cuda.synchronize(); free, _ = torch.cuda.mem_get_info()
            sample = {"stage": stage, "rss_gib": process.memory_info().rss / GIB, "host_available_gib": psutil.virtual_memory().available / GIB,
                      "gpu_free_gib": free / GIB, "disk_free_gib": psutil.disk_usage(str(output)).free / GIB,
                      "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / GIB, "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / GIB}
            result["resources"].append(sample); event("resource", **sample)
            require(sample["rss_gib"] <= 2 and sample["host_available_gib"] >= 1.25 and sample["disk_free_gib"] >= 4
                    and sample["gpu_free_gib"] >= (4 if preload else .5), "reader RAM/GPU/disk limits")
            if preload: require(sample["host_available_gib"] >= 1.25 + max(0, 2 - sample["rss_gib"]), "reader dynamic preload budget")
        guard("preload", True); verify_assets(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"], local_files_only=True, trust_remote_code=False)
        require(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == cfg["chat_template_sha256"]
                and tokenizer.eos_token_id == 151645 and tokenizer.pad_token_id == 151643, "reader tokenizer/template identity")
        raw_text = "Qualification: 7 + 5 = 12."
        require(tokenizer.encode(raw_text, add_special_tokens=False) == cfg["tokenizer_roundtrip_ids"]
                and tokenizer.decode(cfg["tokenizer_roundtrip_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False) == raw_text,
                "independent tokenizer raw roundtrip drift")
        user = [{"role": "user", "content": cfg["mechanical_user_message"]}]
        rendered = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.apply_chat_template(user, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        require(tokenizer.encode(rendered, add_special_tokens=False) == ids and cfg["historical_prompt_ids"][-len(ids):] == ids,
                "tokenizer rendered bytes/native template IDs mismatch")
        write_new(output / "TOKENIZER_RECEIPT.json", {"rendered_utf8_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "token_ids": ids,
                   "tokenizer_sha256": cfg["asset_sha256"]["tokenizer.json"], "template_sha256": cfg["chat_template_sha256"], "enable_thinking": False})
        model_cfg = Qwen3Config.from_pretrained(cfg["model_dir"], local_files_only=True); model_cfg._attn_implementation = "eager"
        prior = torch.get_default_dtype(); torch.cuda.reset_peak_memory_stats()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda:0"): model = accounting.call("model_construct", lambda: Qwen3ForCausalLM(model_cfg), stage="reader_construct")
        finally: torch.set_default_dtype(prior)
        model.eval().requires_grad_(False)
        hashes, details = accounting.call("checkpoint_load", lambda: stream_checkpoint.load_streamed(torch, model,
                         Path(cfg["model_dir"]) / "model.safetensors", accounting, tensor_digest), stage="reader_load")
        result["model_loads"] = 1
        require(model.lm_head.weight is model.model.embed_tokens.weight and sum(p.numel() for p in model.parameters()) == 596049920, "reader model tying/count")
        write_new(output / "MODEL_RECEIPT.json", {"binding": run_binding, "tensor_sha256": hashes, "loader": details})
        session = ReaderSession(torch, model, tokenizer, accounting, guard, event, cfg, run_binding)
        with torch.no_grad(): result.update(qualify_session(session, cfg))
        for name, parameter in model.named_parameters(): require(not parameter.requires_grad and parameter.grad is None
            and tensor_digest(torch, parameter) == hashes[name], "reader frozen base mutation")
        verify_sources(cfg); verify_assets(cfg); guard("complete")
        result["status"] = "COMPLETE_READER_QUALIFICATION_WORKER_ONLY"
        result["base_unchanged"] = True
    except BaseException as error:
        result.update(status="FAILED_READER_QUALIFICATION", error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
    finally:
        if session: session.close()
        result["accounting"] = accounting.snapshot(); result["wall_seconds"] = time.monotonic() - started
        write_new(output / "READER_QUAL_RESULT.json", safe_numbers(result)); store.close()
    return 0 if result["status"] == "COMPLETE_READER_QUALIFICATION_WORKER_ONLY" else 2


def validate_qualification_result(result, cfg, commit, events):
    require(result.get("status") == "COMPLETE_READER_QUALIFICATION_WORKER_ONLY"
            and result.get("scope") == SCOPE and result.get("binding", {}).get("source_commit") == commit
            and result.get("base_unchanged") is True and result.get("model_loads") == 1,
            "complete current exact-model reader result required")
    require(result.get("bulk_capture_qualified") is False and result.get("training_qualified") is False,
            "old bulk/training gates must remain closed")
    rollouts = result.get("rollouts", {})
    require(set(rollouts) == {"historical", "long"}, "natural rollout coverage missing")
    require(rollouts["historical"]["owned"]["generated_ids"] == cfg["historical_generated_ids"], "frozen historical output drift")
    natural_n = len(rollouts["long"]["owned"]["generated_ids"])
    require(1 <= natural_n <= 64, "long natural count invalid")
    for label, prompt in (("historical", cfg["historical_prompt_ids"]), ("long", cfg["long_prompt_ids"])):
        owned, stock, replay = (rollouts[label][k] for k in ("owned", "stock", "replay"))
        require(owned["generated_ids"] == stock["generated_ids"] and owned["stop_reason"] == stock["stop_reason"], "stock natural token/stop mismatch")
        require(len(owned["steps"]) == len(stock["steps"]) == len(replay) == len(owned["generated_ids"]), "natural/replay denominator mismatch")
        require(require_same_prefix(owned["steps"], replay)["passed"], "same-prefix replay extraction invalid")
        require(stock["policy"]["processors"] == [] and set(stock["policy"]["stops"]) == {"MaxLengthCriteria", "EosTokenCriteria"},
                "independent stock effective generation policy mismatch")
        for index, (a, b) in enumerate(zip(owned["steps"], stock["steps"])):
            require(a["input_ids"] == prompt + owned["generated_ids"][:index]
                    and a["raw_logits_sha256"] == b["raw_logits_sha256"] and a["raw_hidden_sha256"] == b["raw_hidden_sha256"],
                    "stock exact-prefix raw witness mismatch")
    require(any(r["owned"]["stop_reason"] == "eos" for r in rollouts.values()), "natural EOS not observed")
    counts = result["accounting"]
    expected_forward = 86 + 3 * natural_n
    expected_positions = 41900 + natural_n + 3 * sum(512 + i for i in range(natural_n))
    require(counts["last_call_id"] == 312 + expected_forward and counts["pending_calls"] == []
            and counts["forward_input_positions_attempted"] == counts["forward_input_positions_returned"] == expected_positions,
            "reader actual call/position ledger mismatch")
    for operation, count in {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310,
                             "forward": expected_forward, "backward": 0, "optimizer_step": 0,
                             "checkpoint_save": 0, "checkpoint_reload": 0}.items():
        require(counts["operations"][operation] == {"attempted": count, "returned": count, "failed": 0, "pending": 0}
                and all(type(v) is int for v in counts["operations"][operation].values()), "reader primitive ledger incomplete")
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import verify_call_journal
    adapted = [{"event": e["kind"], **{k: v for k, v in e.items() if k not in {"kind", "time"}}} for e in events]
    verify_call_journal(adapted, snapshot=counts, maximum_optimizer_calls=0)
    resources = [{k: v for k, v in e.items() if k not in {"kind", "time"}} for e in events if e["kind"] == "resource"]
    require(resources == result["resources"] and resources[-1]["stage"] == "complete", "reader durable resources incomplete")
    measured = [e["record"] for e in events if e["kind"] == "reader_forward_measurement"]
    require(len(measured) == expected_forward, "all actual reader forwards require durable witness")
    for record in measured:
        length = record["input_length"]
        require(record["input_ids_sha256"] == digest(record["input_ids"]) and len(record["input_ids"]) == length
                and record["witnesses"]["mask"]["raw_sha256"] == _mask_sha(length)
                and record["witnesses"]["positions"]["ids"] == [list(range(length))], "reader durable geometry witness corrupt")
    require(len(result["blocks"]) == 6 and all(b["status"] == "COMPLETE" for b in result["blocks"]), "planned qualification blocks incomplete")
    causal = result["checks"]["maximum_shape_causality"]
    require(len(causal) == 3 and all(c["left_logits_sha256"] == c["right_logits_sha256"]
            and c["left_hidden_sha256"] == c["right_hidden_sha256"] for c in causal), "max576 raw causal comparisons incomplete")
    return {"reader_scope": SCOPE, "actual_forwards": expected_forward, "actual_input_positions": expected_positions,
            "natural_eos_observed": True, "maximum_mechanical_shape": 576, "bulk_capture_qualified": False,
            "training_qualified": False, "durable_journal_verified": True}


# Adapted from the exact reviewed native.supervise source; only namespace,
# result filename and external completion validator differ. Original is untouched.
def _supervise_reader_worker(args, cfg):
    """Own one model-worker process and enforce actual external telemetry."""
    import psutil
    require(Path(args.host_db).resolve() == host_state_path().resolve(), "one canonical host-wide database required")
    report = preflight(contract_sha256=args.contract_sha256, review_path=args.review)
    require(report["status"] == "READY_READER_QUALIFICATION_ONLY", "preflight blocked: " + repr(report["reasons"]))
    time.sleep(2)
    second = collect_telemetry(desktop_identities=cfg["gpu_process_registry"])
    require(not admission_reasons(cfg, second), "second current pre-worker observation failed")
    output = args.output.resolve()
    require(output.is_relative_to(PACKAGE / "results") and not output.exists(), "new CRS results child required")
    output.mkdir(parents=True)
    store = OperationsStore(args.host_db, output / "receipts")
    owner = "CRS-READER-" + uuid.uuid4().hex
    process = job = None
    tree_exit = None
    lease = None
    started = time.monotonic()
    stopped = threading.Event()
    telemetry_state = {"snapshot": second, "error": None}
    outcome, reason = "FAILED", None
    worker_dispatched = False
    launch_attempted = False
    reservation_admitted = False
    try:
        store.register_job(args.job_key, {"contract_sha256": args.contract_sha256,
                           "source_commit": report["observations"]["source"]["commit"]}, args.epoch)
        store.ready(args.job_key)
        lease = store.acquire_lease(owner, 120)
        store.reserve(args.job_key, owner, lease["fence"], 600)
        reservation_admitted = True
        command = [sys.executable, "-B", "-m", "experiments.rineya_centralized_reasoning_system_v1.reader",
                   "--worker", "--contract-sha256", args.contract_sha256, "--review", str(args.review.resolve()),
                   "--output", str(output), "--go-path", str(output / "GO.json"), "--host-db", str(args.host_db),
                   "--owner", owner, "--fence", str(lease["fence"])]
        write_new(output / "LAUNCH_INTENT.json", {"command": command, "preflight": report,
                  "second_observation": second, "job_key": args.job_key, "lease": lease})
        from experiments.opallo_native_collector_v1.supervisor import WindowsJob
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            write_new(output / "LAUNCH_ATTEMPT.json", {"job_key": args.job_key, "owner": owner,
                       "fence": lease["fence"], "command_sha256": hashlib.sha256(canonical(command)).hexdigest(),
                       "attempted_at": time.time(), "outcome": "PENDING_SUBPROCESS_CREATION"})
            launch_attempted = True
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                     env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                              CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1"),
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x4)
            worker_dispatched = True
            observed = psutil.Process(process.pid)
            store.mark_running(args.job_key, owner, lease["fence"], {"pid": process.pid,
                "creation_time": str(observed.create_time()), "command_hash": hashlib.sha256(canonical(command)).hexdigest()})
            job = WindowsJob(process)
            before_resume = job_member_witness(job, cfg)
            require(job_process_ids(job) == [process.pid] and not observed.children(recursive=True),
                    "worker must be suspended before any child can form")
            resumed = resume_suspended_primary_thread(process)
            deadline = time.monotonic() + 3
            members = []
            while time.monotonic() < deadline:
                members = job_member_witness(job, cfg)
                if len(members) >= 2:
                    break
                time.sleep(.02)
            require(len(members) >= 2, "venv runtime child membership not observed before GO")
            runtime_targets = [m for m in members if Path(m["executable"]).resolve() == Path(sys._base_executable).resolve()]
            require(len(runtime_targets) == 1, "one exact CPython worker target required for Job query handle")
            query_transfer = duplicate_owned_job_query_handle(job, runtime_targets[0])
            write_new(output / "TREE_CONTAINMENT.json", {"before_resume": before_resume,
                       "resume": resumed, "before_go_members": members, "all_members_in_exact_owned_job": True,
                       "query_handle_transfer": query_transfer})
            write_new(output / "GO.json", {"contract_sha256": args.contract_sha256, "owner": owner, "fence": lease["fence"],
                       "supervisor_pid": os.getpid(), "worker_tree_pid": process.pid, **query_transfer})
            def gpu_watch():
                while not stopped.is_set():
                    querying_live = False
                    try:
                        allowed = job_process_ids(job)
                        if not allowed:
                            if process.poll() is not None:
                                telemetry_state["owned_exit_observed"] = True
                                return
                            stopped.wait(.02)
                            continue
                        querying_live = True
                        telemetry_state["snapshot"] = collect_telemetry(output, allowed, cfg["gpu_process_registry"])
                    except BaseException as exc:
                        if stopped.is_set() and not querying_live:
                            return
                        telemetry_state["error"] = str(exc)
                        break
                    stopped.wait(.5)
            watcher = threading.Thread(target=gpu_watch, daemon=True); watcher.start()
            last_heartbeat, peak_rss = time.monotonic(), 0
            with (output / "WATCHDOG.jsonl").open("xb") as audit:
                while process.poll() is None:
                    store._require_lease(owner, lease["fence"])
                    owned = owned_job_rss_snapshot(job, cfg)
                    members = owned["members"]
                    elapsed = time.monotonic() - started
                    rss = owned["rss_bytes"]
                    peak_rss = max(peak_rss, rss)
                    host_free = psutil.virtual_memory().available / GIB
                    sample = telemetry_state["snapshot"]
                    if elapsed >= 600: reason = "WALL_LIMIT"
                    elif rss > 2 * GIB or host_free < 1.25: reason = "PROCESS_TREE_RAM_LIMIT"
                    elif telemetry_state["error"] or time.time() - sample["sampled_at"] > 5: reason = "TELEMETRY_UNAVAILABLE"
                    elif sample["gpu_free_gib"] < .5 or sample["disk_free_gib"] < 4: reason = "GPU_DISK_FLOOR"
                    elif sample["unresolved_gpu_pids"] or sample["competing_gpu_pids"] or sample["unclassified_gpu_pids"]: reason = "GPU_PROCESS_CONFLICT"
                    elif stdout.tell() + stderr.tell() > 16 * 1024 ** 2: reason = "LOG_LIMIT"
                    audit.write(canonical({"elapsed": elapsed, "rss": rss, "host_free_gib": host_free,
                                "gpu_snapshot": sample, "owned_job_members": members,
                                "confirmed_owned_exit_races": owned["confirmed_exit_races"], "reason": reason}) + b"\n")
                    audit.flush(); os.fsync(audit.fileno())
                    if reason:
                        job.kill(); break
                    if time.monotonic() - last_heartbeat >= 10:
                        store.heartbeat_lease(owner, lease["fence"], 120)
                        last_heartbeat = time.monotonic()
                    stopped.wait(.1)
            code = process.wait(timeout=10)
            require(code == 0 and reason is None, "contained worker failed: " + str(reason or code))
            stopped.set()
            watcher.join(timeout=5)
            require(not watcher.is_alive(), "GPU telemetry worker did not stop")
            require(telemetry_state["error"] is None, "live GPU telemetry failure retained")
            require(job_process_ids(job) == [], "launcher exit left an owned descendant active")
            result = read_json(output / "READER_QUAL_RESULT.json")
            events = read_event_journal(output / "EVENTS.jsonl")
            completion = validate_qualification_result(result, cfg, report["observations"]["source"]["commit"], events)
            completion["events_sha256"] = sha(output / "EVENTS.jsonl")
            write_new(output / "COMPLETION_VERIFICATION.json", completion)
            verify_sources(cfg); verify_assets(cfg)
            require(not committed_snapshot()["dirty"], "source dirty after worker")
            outcome = "SUCCEEDED"
    except BaseException as exc:
        reason = reason or type(exc).__name__ + ": " + str(exc)
    finally:
        stopped.set()
        if job:
            try:
                tree_exit = drain_owned_job(job)
            except BaseException as exc:
                reason = reason or "owned tree exit unverified: " + str(exc)
                outcome = "OUTCOME_UNKNOWN"
            finally:
                job.close()
        if process is not None and process.poll() is None:
            # This process is the caller-owned, recorded worker only.
            process.kill(); process.wait(timeout=10)
        elapsed = time.monotonic() - started
        uncertain_creation = launch_attempted and not worker_dispatched
        if uncertain_creation:
            outcome = "OUTCOME_UNKNOWN"
        receipt = {"status": outcome, "reason": reason, "wall_seconds": elapsed,
                   "launch_attempted": launch_attempted, "worker_dispatched": worker_dispatched,
                   "subprocess_creation_outcome_unknown": uncertain_creation,
                   "owned_tree_exit": tree_exit,
                   "exit_code": process.returncode if process else None,
                   "peak_process_tree_rss_bytes": locals().get("peak_rss"),
                   "watchdog_poll_seconds": .1, "reader_qualified": False, "training_qualified": False}
        try:
            write_new(output / "SUPERVISOR_RESULT.json", receipt)
            if lease is not None and reservation_admitted:
                row = store.get_job(args.job_key)
                if (uncertain_creation or (worker_dispatched and row["worker"] is None)
                        or (worker_dispatched and not tree_exit)):
                    store.mark_unknown(args.job_key, "subprocess creation/identity response unavailable; reconcile before retry")
                    # Popen may have created a worker before losing its response.
                    # No process identity exists to prove exit or zero cost.
                    # Preserve the full reservation and held model lease.
                    print(json.dumps(receipt), flush=True)
                    return 2
                proof = {"observer": "CRS_READER_SUPERVISOR_V1", "request_hash": row["request_hash"],
                         "owner": owner, "fence": lease["fence"],
                         "elapsed_native_seconds": elapsed if worker_dispatched else 0}
                if worker_dispatched:
                    require(process is not None and process.poll() is not None, "worker exit observation missing")
                    proof.update(worker_identity=row["worker"], process_exited=True,
                                 exit_observed_at=time.time(), owned_tree_exited=True,
                                 owned_tree_exit=tree_exit)
                else:
                    require(not launch_attempted, "attempted launch cannot receive zero-cost no-dispatch proof")
                    proof.update(never_dispatched=True, launch_attempted=False)
                proof_ref = store.write_receipt("native_outcome_observation", args.job_key, proof)
                store.reconcile(args.job_key, outcome, elapsed if worker_dispatched else 0,
                                {"supervisor_receipt_sha256": sha(output / "SUPERVISOR_RESULT.json"),
                                 "native_outcome_proof_ref": proof_ref}, owner=owner, fence=lease["fence"])
            if lease is not None:
                store.release_lease(owner, lease["fence"])
        except BaseException:
            # Lost publication/reconciliation leaves reservation conservative.
            if lease is not None and reservation_admitted and store.get_job(args.job_key)["state"] in {"RESERVED", "RUNNING", "OUTCOME_UNKNOWN"}:
                store.mark_unknown(args.job_key, "supervisor publication/reconciliation failed")
            raise
        finally:
            store.close()
    print(json.dumps(receipt), flush=True)
    return 0 if outcome == "SUCCEEDED" else 2


def supervise(args, cfg):
    require(args.review is not None and args.output is not None and args.job_key and args.epoch,
            "review/output/durable reader job and epoch required")
    result = _supervise_reader_worker(args, cfg)
    if result == 0:
        output = args.output.resolve()
        worker_result = read_json(output / "READER_QUAL_RESULT.json")
        supervisor_result = read_json(output / "SUPERVISOR_RESULT.json")
        require(supervisor_result["status"] == "SUCCEEDED" and supervisor_result["owned_tree_exit"]["owned_tree_exited"], "complete supervisor prerequisite")
        require(args.review.resolve().is_relative_to(PACKAGE / "results"), "immutable independent review must be retained under results")
        publication = {"scope": SCOPE, "status": "PASS_EXACT_PREFIX_SERVING_576_SCOPED", "binding": worker_result["binding"],
                       "supervisor_fully_verified": True, "natural_eos_observed": True,
                       "bulk_capture_qualified": False, "training_qualified": False,
                       "worker_result": {"path": str(output / "READER_QUAL_RESULT.json"), "sha256": sha(output / "READER_QUAL_RESULT.json")},
                       "supervisor_result": {"path": str(output / "SUPERVISOR_RESULT.json"), "sha256": sha(output / "SUPERVISOR_RESULT.json")},
                       "independent_review": {"path": str(args.review.resolve()), "sha256": sha(args.review)}}
        write_new(output / "READER_QUALIFICATION_RECEIPT.json", publication)
    return result
