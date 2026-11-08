"""First-time prospective raw fresh K0/K1 worker; coordinator-only dispatch.

No labels, correctness scorer, bank generator, optimizer or cached answers are
read by this worker. Its native loader is reachable only after exact new scope,
source, review, runtime and prompt-only packet checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import inspect
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import psutil
import torch

from ..binding import ARMR, LAB, REPO, admitted_source, sha, text_sha
from ..control import BlockRecurrence, TAPSConfig
from ..gpu_native import (GIB, encode_prefix, evaluate_case, final_resource_snapshot, forward_logits,
                         gpu_resources, install_gpu_timing, one_shot_anchors,
                         require_cuda_tensor, resource_guard)
from .fresh_scope import FIXED_CONTRACT, STUDY_ID, LANE, verify_dispatch
from .opposed_gate_generation import GenerationSession
from .opposed_gate_policy import (CONFIDENT, UNCERTAIN, GUARD, Decision, decide,
                                  header_cue, matched_random_decisions)
from .probe_native import Recorder as PrefixRecorder, QUAL_MODES, invoke_prefix
from .observer import observed_block, readout_stats
from experiments.rineya_taps_adaptive_recurrence_v1.safety import Journal, digest, write_json

V3 = Path(__file__).resolve().parent
CAMPAIGN = V3.parent
INPUTS = V3 / "data/fresh_open_replication_001/model_inputs"
PARENT_CLOCK = "FRESH_PARENT_STARTED_PERF_COUNTER"
PATHS = ("K0", "UNIT", CONFIDENT, UNCERTAIN, GUARD, "HEADER_CUE", "RANDOM17", "RANDOM29", "PRIMARY_REPEAT")
HELPERS = {"gpu_native.py": "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f",
           "control.py": "298d0a7064ff562c498ecc50d13c2f23cd9bdf1df6b1d616d7204bac24a49ed3"}


def count_contract():
    return {"scientific_cases": 432, "preservation_cases": 24, "items": 456,
            "paths_per_item": 9, "task_physical_rows": 4104, "task_API_cap": 33744,
            "qualification_API_cap": 852, "model_API_cap": 34596,
            "qualification_cache_rejections": 2, "qualification_expected_partials": 2,
            "scientific_output_cap": 8, "qualification_output_cap": 24,
            "maximum_prefix_tokens": 357, "state_adjoints": 0, "gold_calls": 0}


def parent_clock():
    try:
        started = float(os.environ[PARENT_CLOCK])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("fresh parent clock missing or nonnumeric") from error
    if not math.isfinite(started) or not 0 <= time.perf_counter() - started < 15600:
        raise RuntimeError("fresh parent clock nonfinite/future/expired")
    return started


def disk_guard():
    if shutil.disk_usage(CAMPAIGN).free < 15 * GIB:
        raise MemoryError("unchanged host disk reserve unavailable")
    if sum(p.stat().st_size for p in CAMPAIGN.rglob("*") if p.is_file()) > 1536 * 1024**2:
        raise MemoryError("campaign2GiB must retain512MiB evidence reserve")


def project_inputs(raw, token_packet, count, source, cohort):
    raw_keys = {"case_id", "raw_prompt", "raw_prompt_sha256"}
    token_keys = raw_keys | {"rendered_prompt", "rendered_prompt_sha256", "input_token_ids", "input_tokens"}
    if (set(token_packet) != {"status", "canonical_source", "model_loads", "rows"}
            or token_packet["status"] != "FROZEN_FRESH_PROMPT_ONLY_TOKEN_BINDING"
            or token_packet["canonical_source"] != str(source) or token_packet["model_loads"] != 0):
        raise PermissionError("exact fresh tokenizer source/no-model schema required")
    tokens = token_packet["rows"]
    if (len(raw) != count or len(tokens) != count or any(set(x) != raw_keys for x in raw)
            or any(set(x) != token_keys for x in tokens)):
        raise PermissionError("complete fresh prompt-only rows required; metadata/labels forbidden")
    by = {x["case_id"]: x for x in tokens}
    if len(by) != count or len({x["case_id"] for x in raw}) != count or set(by) != {x["case_id"] for x in raw}:
        raise RuntimeError("duplicate/missing fresh original inputs")
    projected = []
    for row in raw:
        t = by[row["case_id"]]; ids = t["input_token_ids"]
        if (type(row["case_id"]) is not str or not row["case_id"] or type(row["raw_prompt"]) is not str
                or text_sha(row["raw_prompt"]) != row["raw_prompt_sha256"]
                or row["raw_prompt"] != t["raw_prompt"] or row["raw_prompt_sha256"] != t["raw_prompt_sha256"]
                or text_sha(t["rendered_prompt"]) != t["rendered_prompt_sha256"]
                or not 1 <= len(ids) <= 350 or t["input_tokens"] != len(ids)
                or any(type(x) is not int or not 0 <= x < 151936 for x in ids)):
            raise RuntimeError("fresh original prompt/rendered ID/hash/shape binding differs")
        projected.append({**row, **{k: t[k] for k in token_keys - raw_keys},
                          "cohort": cohort, "prefix_ids_sha256": digest(ids)})
    return projected


def read_inputs(admission, source):
    specs = (("scientific_bank", "SCIENTIFIC_PROMPTS.json"),
             ("scientific_token_binding", "SCIENTIFIC_TOKEN_BINDING.json"),
             ("preservation_bank", "PRESERVATION_PROMPTS.json"),
             ("preservation_token_binding", "PRESERVATION_TOKEN_BINDING.json"),
             ("prompt_export_manifest", "MODEL_EXPORT_MANIFEST.json"))
    packets, hashes = {}, {}
    for key, name in specs:
        path = Path(admission[key + "_path"]).resolve(strict=True)
        if path != (INPUTS / name).resolve(strict=True) or sha(path) != admission[key + "_sha256"]:
            raise PermissionError("exact source-bound fresh export path/hash differs")
        packets[key] = json.loads(path.read_text(encoding="utf-8")); hashes[name] = sha(path)
    m = packets["prompt_export_manifest"]
    expected = {"status": "FROZEN_FRESH_OPEN_MODEL_EXPORT", "scientific_cases": 432,
                "preservation_cases": 24, "maximum_input_tokens": 350, "maximum_forward_prefix_tokens": 357,
                "labels_excluded": True, "tokenizer_source": str(source), "model_loads": 0}
    if any(m.get(k) != value for k, value in expected.items()) or m.get("files") != {k: v for k, v in hashes.items() if k != "MODEL_EXPORT_MANIFEST.json"}:
        raise RuntimeError("fresh export manifest/science/cap/no-label binding differs")
    science = project_inputs(packets["scientific_bank"], packets["scientific_token_binding"], 432, source, "SCIENTIFIC")
    panel = project_inputs(packets["preservation_bank"], packets["preservation_token_binding"], 24, source, "PRESERVATION")
    if len({x["case_id"] for x in science + panel}) != 456:
        raise RuntimeError("scientific/preservation item identity overlap")
    return science + panel


def validate_tokenizer_identity(manifest, binding, runtime, source, interpreter):
    identity = manifest.get("tokenizer_identity", {})
    if (identity.get("qualified_manifest_sha256") != "b0b08bb354ce7bc48c760f59a87a7e7d8b8ba432aedddee3e05b55a0f0221131"
            or identity.get("checkpoint_files") != binding["checkpoint_files"]
            or manifest.get("checkpoint_files_after") != binding["checkpoint_files"]
            or identity.get("qwen_implementation_sha256") != runtime["qwen_implementation_sha256"]
            or identity.get("canonical_source") != str(source) or identity.get("interpreter") != interpreter
            or identity.get("class") != binding["tokenizer_class"] or identity.get("vocabulary") != binding["tokenizer_length"]
            or any(identity.get("runtime", {}).get(k) != runtime[k] for k in ("python", "torch", "transformers", "tokenizers"))):
        raise RuntimeError("fresh tokenizer before/after assets/runtime/class binding differs")
    return identity


class GlobalLedger:
    """Maps helper/session/reference local indexes to one physical API ledger."""
    def __init__(self, journal, started, cap=34596):
        self.journal, self.started, self.cap = journal, started, cap
        self.attempts = 0; self.mapping = {}; self.terminals = set()
        self.events = []; self.completed = []; self.failed = []; self.cache = []

    def accept(self, event, backend, context=None):
        kind = event["event"]
        value = {**event, **(context or {}), "backend": backend,
                 "parent_elapsed_seconds": time.perf_counter() - self.started,
                 "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        local = event.get("API_attempt", event.get("full_model_call"))
        if kind in ("FORWARD_STARTED", "CACHE_REJECTION_STARTED"):
            if self.attempts >= self.cap or (backend, local) in self.mapping:
                raise RuntimeError("global physical API cap/unique start binding failed")
            self.attempts += 1; self.mapping[(backend, local)] = self.attempts
        if kind in ("FORWARD_STARTED", "FORWARD_COMPLETED", "FORWARD_FAILED", "CACHE_REJECTION_STARTED", "CACHE_REJECTED"):
            if (backend, local) not in self.mapping:
                raise RuntimeError("physical terminal lacks its unique global start")
            value["local_API_attempt"] = local; value["API_attempt"] = self.mapping[(backend, local)]
        elif local is not None and (backend, local) in self.mapping:
            value["local_API_attempt"] = local; value["API_attempt"] = self.mapping[(backend, local)]
        if kind in ("FORWARD_COMPLETED", "FORWARD_FAILED", "CACHE_REJECTED"):
            if (backend, local) in self.terminals: raise RuntimeError("duplicate physical terminal")
            self.terminals.add((backend, local))
        if kind == "FORWARD_COMPLETED": self.completed.append(value)
        elif kind == "FORWARD_FAILED": self.failed.append(value)
        elif kind == "CACHE_REJECTED": self.cache.append(value)
        self.events.append(value); self.journal.append(value)
        return value

    def counts(self):
        return {"model_API_attempts": self.attempts, "completed_forwards": len(self.completed),
                "full_forward_attempts": self.attempts - len(self.cache),
                "failed_forwards": len(self.failed), "cache_rejections": len(self.cache),
                "completed_prefix_positions": sum(x["prefix_tokens"] for x in self.completed)}


def full_forecast(warm_times, observed_nonforward, parser_seconds, elapsed):
    if (not warm_times or any(not math.isfinite(x) or x <= 0 for x in warm_times)
            or not observed_nonforward or any(not math.isfinite(x) or x < 0 for x in observed_nonforward)
            or not math.isfinite(parser_seconds) or parser_seconds < 0 or not math.isfinite(elapsed) or elapsed < 0):
        raise RuntimeError("fresh complete forward/nonforward timing evidence missing")
    maximum = max(warm_times)
    nonforward = 1.5 * (max(observed_nonforward) + parser_seconds)
    seconds = 33744 * maximum * 1.5 + 4104 * nonforward
    return {"task_API_cap": 33744, "task_rows": 4104, "forward_factor": 1.5,
            "max_observed_warm_forward_seconds": maximum, "nonforward_per_row_bound_seconds": nonforward,
            "nonforward_scope": "fresh complete-SMOKE overhead plus all-input prompt-only parsing maximum;1.5factor",
            "remaining_forward_seconds": 33744 * maximum * 1.5,
            "remaining_nonforward_seconds": 4104 * nonforward, "remaining_forecast_seconds": seconds,
            "elapsed_seconds": elapsed, "internal_reserve_seconds": 1800, "parent_deadline_seconds": 15600,
            "admitted": elapsed + seconds + 1800 <= 15600,
            "limitation": "conservative measured extrapolation, not a mathematical bound on arbitrary future host stalls"}


def _repeat_signature(row):
    return tuple(row[k] for k in ("token_ids", "response", "termination", "K", "recognized", "certificate",
                                  "pilot_probability", "selected_first_logits_sha256", "pilot_logits_sha256"))


def execute_item(row, index, driver, random17, random29, sink):
    """Gate API uses raw prompt/p only; opaque identifiers are added afterward."""
    results, pilots = {}, {}
    for path in PATHS:
        if hasattr(driver, "begin_path"): driver.begin_path(row, index, path)
        started = time.perf_counter(); pilot = None
        policy = GUARD if path == "PRIMARY_REPEAT" else path
        if path in (CONFIDENT, UNCERTAIN):
            pilot = driver.make_pilot(row, started)
            decision = decide(policy, row["raw_prompt"], pilot.probability)
            pilots[path] = pilot
        elif path in (GUARD, "PRIMARY_REPEAT"):
            decision = decide(GUARD, row["raw_prompt"])
        elif path == "HEADER_CUE": decision = header_cue(row["raw_prompt"])
        elif path == "RANDOM17": decision = Decision(random17, False, "MATCHED_RANDOM_K")
        elif path == "RANDOM29": decision = Decision(random29, False, "MATCHED_RANDOM_K")
        else: decision = Decision(int(path == "UNIT"), False, "FIXED_K_CONTROL")
        output = driver.generate(row, decision.k, pilot, started)
        output.update(case_id=row["case_id"], cohort=row["cohort"], item_index=index, path=path,
                      policy=policy, recognized=decision.recognized, certificate=decision.certificate,
                      route_seed=17 if path == "RANDOM17" else 29 if path == "RANDOM29" else None,
                      prefix_ids_sha256=row["prefix_ids_sha256"], physical_run=True,
                      primary=path == GUARD, primary_repeat=path == "PRIMARY_REPEAT")
        results[path] = output; sink(output)
        if output["safety_stopped"]:
            raise RuntimeError("fresh generation safety stop; retain incomplete matrix")
        if path == UNCERTAIN:
            a, b = pilots[CONFIDENT], pilots[UNCERTAIN]
            if a.logits_sha256 != b.logits_sha256 or a.probability != b.probability or results[CONFIDENT]["K"] + output["K"] != 1:
                raise RuntimeError("independent actual confidence pilots/complementarity differ")
        if path == "PRIMARY_REPEAT" and _repeat_signature(output) != _repeat_signature(results[GUARD]):
            raise RuntimeError("actual independent primary-repeat token/route/certificate parity failed")
    return results


class NativeDriver:
    def __init__(self, session): self.session = session
    def _ids(self, row): return torch.tensor([row["input_token_ids"]], dtype=torch.long, device="cuda:0")
    def make_pilot(self, row, started): return self.session.make_pilot(self._ids(row), started_perf_counter=started)
    def generate(self, row, k, pilot, started):
        result, _ = self.session.generate(self._ids(row), k=k, pilot=pilot, max_new_tokens=8,
                                          started_perf_counter=started)
        return result


def qualify_max_driver(session, max_ids, context):
    """Nine physical maximum attempts, one expected partial, no targets."""
    _, first0 = session.generate(max_ids, k=0, max_new_tokens=1, qualification=True)
    _, first1 = session.generate(max_ids, k=1, max_new_tokens=1, qualification=True)
    for k, expected in ((0, first0), (1, first1)):
        p = session.make_pilot(max_ids)
        _, first = session.generate(max_ids, k=k, pilot=p, max_new_tokens=1, qualification=True)
        if not torch.equal(first, expected): raise RuntimeError("driver max357 pilot/selected identity failed")
    p = session.make_pilot(max_ids)
    original_sink = session.trajectory_sink
    def driver_fail(_): raise RuntimeError("EXPECTED_FRESH_DRIVER_EXCEPTION")
    session.trajectory_sink = driver_fail
    try: session.generate(max_ids, k=1, pilot=p, max_new_tokens=1, qualification=True)
    except RuntimeError as error:
        if str(error) != "EXPECTED_FRESH_DRIVER_EXCEPTION": raise
    else: raise RuntimeError("driver forced exception absent")
    finally: session.trajectory_sink = original_sink
    for k, expected in ((0, first0), (1, first1)):
        _, first = session.generate(max_ids, k=k, max_new_tokens=1, qualification=True)
        if not torch.equal(first, expected): raise RuntimeError("driver exception restore failed")
    return {"intact_logits_sha256": GenerationSession._tensor_hash(first0),
            "unit_logits_sha256": GenerationSession._tensor_hash(first1), "forced_exception_restoration": True}


def run(source, out, admission_path):
    source, out, admission_path = Path(source).resolve(), Path(out).resolve(), Path(admission_path).resolve(strict=True)
    if V3 not in out.parents: raise PermissionError("fresh output must be exclusive and V3-local")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "FAIL", "study_id": STUDY_ID, "lane": LANE, "native_model_loads": 0,
               "model_load_attempts": 0, "training": False, "state_adjoints": 0, "gold_calls": 0}
    started = model = ledger = None
    try:
        admission = verify_dispatch(admission_path); started = parent_clock()
        if str(source) != admission["source"]: raise PermissionError("exact canonical model source differs")
        for name, expected in HELPERS.items():
            if sha(CAMPAIGN / name) != expected: raise RuntimeError("qualified immutable helper changed")
        population = read_inputs(admission, source)
        binding_path = Path(admission["model_binding_path"]).resolve(strict=True)
        audit_path = Path(admission["smoke_prompt_audit_path"]).resolve(strict=True)
        if (binding_path != (CAMPAIGN / "runs/binding-001/MODEL_DATA_BINDING.json").resolve(strict=True)
                or audit_path != (CAMPAIGN / "runs/binding-001/PROMPT_AUDIT.json").resolve(strict=True)
                or sha(binding_path) != admission["model_binding_sha256"] or sha(audit_path) != admission["smoke_prompt_audit_sha256"]):
            raise RuntimeError("exact model/SMOKE metadata binding differs")
        binding = json.loads(binding_path.read_text()); audit = json.loads(audit_path.read_text())
        smoke = [{k: x[k] for k in ("case_id", "raw_prompt", "rendered_prompt", "input_token_ids")}
                 for x in audit if x["split"] == "SMOKE"]
        if len(smoke) != 8: raise RuntimeError("all8 exact SMOKE inputs required")
        historical, inherited = admitted_source(source)
        if sha(LAB / "data/evaluation/supa_layer_ablation_smoke.jsonl") != binding["datasets"]["SMOKE"]["sha256"]:
            raise RuntimeError("original SMOKE source changed")
        disk_guard()
        cache = out / "hf_cache"
        os.environ.update(HF_HOME=str(cache), HF_MODULES_CACHE=str(cache / "modules"), HF_HUB_CACHE=str(cache / "hub"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        runtime = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                   "qwen_implementation_sha256": sha(inspect.getfile(Qwen3DecoderLayer)), "cuda": torch.version.cuda,
                   "tokenizers": metadata.version("tokenizers")}
        if any(runtime[k] != historical[k] for k in ("python", "torch", "transformers", "qwen_implementation_sha256")):
            raise RuntimeError("qualified exact runtime changed")
        token_identity = validate_tokenizer_identity(json.loads(Path(admission["prompt_export_manifest_path"]).read_text()),
                                                     binding, runtime, source, sys.executable)
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or psutil.virtual_memory().available < 6 * GIB:
            raise MemoryError("original initial host/device/deterministic contract unavailable")
        torch.cuda.set_device(0); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        initial_resources = gpu_resources()
        if initial_resources["gpu_free_bytes"] < 6 * GIB: raise MemoryError("initial GPU free6GiB floor unavailable")
        runtime.update(CPU_threads=torch.get_num_threads(), dtype="float32", attention="eager", TF32=False,
                       CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"], device_name=torch.cuda.get_device_name(0))
        def guard():
            if time.perf_counter() - started >= 13800: raise TimeoutError("fresh body deadline;1800 evidence reserve retained")
            return resource_guard()
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False, cache_dir=cache / "hub")
        if type(tokenizer).__name__ != token_identity["class"] or len(tokenizer) != token_identity["vocabulary"]:
            raise RuntimeError("actual tokenizer class/vocabulary differs")
        for row in population + smoke:
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": row["raw_prompt"]}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
            if rendered != row["rendered_prompt"] or tokenizer(rendered, return_tensors="pt")["input_ids"][0].tolist() != row["input_token_ids"]:
                raise RuntimeError("actual original rendering/token binding changed")
        write_json(out / "population.json", population)
        write_json(out / "registration.json", {"study_id": STUDY_ID, "lane": LANE, "admission_sha256": sha(admission_path),
                   "count_contract": count_contract(), "runtime": runtime, "inherited_source_hashes": inherited,
                   "initial_resources": initial_resources, "primary_policy": GUARD, "DEV_K1_rate": 160 / 192,
                   "paths": list(PATHS), "generation": "raw own trajectory; EOS included inT", "labels_in_worker": False,
                   "hard_deadline_seconds": 15600, "body_deadline_seconds": 13800, "internal_reserve_seconds": 1800})
        guard(); receipt["model_load_attempts"] = 1
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=True,
                    dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True,
                    device_map={"": "cuda:0"}, cache_dir=cache / "hub").eval().requires_grad_(False)
        receipt["native_model_loads"] = 1; receipt["after_load_resources"] = guard()
        code = Path(inspect.getfile(type(model))).resolve()
        if (type(model).__name__ != "Qwen3AbstainForCausalLM" or cache not in code.parents
                or sha(code) != historical["checkpoint_files_before"]["modeling_supa_a11.py"]
                or model.config.abstain_token_id != 51378 or model.config.abstain_logit_bias != 5.156249985098839):
            raise RuntimeError("exact custom class/head/source changed")
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            require_cuda_tensor(value, name)
            if value.is_floating_point() and value.dtype != torch.float32: raise RuntimeError("all floating model tensors must beFP32")
        sys.path[:0] = [str(ARMR / "src"), str(LAB / "src")]
        from armr.provenance import state_hash, checkpoint_files
        from armr.recurrence import FixedBlock
        from armr.metrics import output_flags
        before_state, before_files = state_hash(model), checkpoint_files(source)
        if before_state != historical["state_hash_before"] or before_files != admission["checkpoint_files"]:
            raise RuntimeError("actual residentb23/checkpoint identity differs")
        receipt.update(before_state_sha256=before_state, checkpoint_files_before=before_files, loaded_custom_code_sha256=sha(code))
        synchronize = lambda: torch.cuda.synchronize(0)
        nonforward, parser_times = [], []
        for row in population:
            begin = time.perf_counter(); decide(GUARD, row["raw_prompt"]); header_cue(row["raw_prompt"])
            parser_times.append(time.perf_counter() - begin)
        with Journal(out / "forwards.jsonl") as fj, Journal(out / "trajectories.jsonl") as tj, Journal(out / "outcomes.jsonl") as outcomes:
            ledger = GlobalLedger(fj, started)
            class PrefixJournal:
                def append(self, value): ledger.accept(value, "prefix", {"phase": "QUALIFICATION_PREFIX"})
            prefix = PrefixRecorder(PrefixJournal(), started, guard, synchronize)
            def qualifier(ids, mode, context, sink=None):
                return invoke_prefix(model, ids, mode, context, prefix, guard, synchronize, FixedBlock, sink)[0]
            parity = []
            longest = max(population, key=lambda x: x["input_tokens"])
            filler = tokenizer.encode(".", add_special_tokens=False)[0]
            max_ids = torch.tensor([longest["input_token_ids"] + [filler] * (357 - longest["input_tokens"])], device="cuda:0")
            for row in smoke + [{"case_id": "MAX357", "input_token_ids": max_ids[0].tolist()}]:
                ids = torch.tensor([row["input_token_ids"]], device="cuda:0")
                context = {"case_id": row["case_id"], "phase": "QUALIFICATION"}
                values = [qualifier(ids, mode, context) for mode in QUAL_MODES]
                if not (all(torch.equal(values[0], values[i]) for i in (1, 2, 6)) and all(torch.equal(values[3], values[i]) for i in (4, 5))):
                    raise RuntimeError("fresh old first-prefix parity failed")
                parity.append({"case_id": row["case_id"], "hashes": [GenerationSession._tensor_hash(v) for v in values]})
            def prefix_fail(_): raise RuntimeError("EXPECTED_FRESH_PREFIX_EXCEPTION")
            try: qualifier(max_ids, "OBSERVED_UNIT", {"case_id": "MAX357", "expected_forced_exception": True}, prefix_fail)
            except RuntimeError as error:
                if str(error) != "EXPECTED_FRESH_PREFIX_EXCEPTION": raise
            else: raise RuntimeError("fresh prefix exception absent")
            restored = qualifier(max_ids, "RESTORED_INTACT", {"case_id": "MAX357"})
            if GenerationSession._tensor_hash(restored) != parity[-1]["hashes"][0]: raise RuntimeError("prefix restore changed")
            for kwargs in ({"use_cache": True}, {"use_cache": False, "past_key_values": object()}):
                with torch.inference_mode(), observed_block(model, extra_loops=1, sample_id="cache", sink=None) as wrapper:
                    prefix.reject_cache(model, max_ids, kwargs, wrapper)
            if prefix.API != 67 or prefix.completed != 64 or prefix.partial != 1 or prefix.cache != 2:
                raise RuntimeError("exact new first-prefix mechanics count failed")
            context = {"phase": "QUALIFICATION_DRIVER"}
            session = GenerationSession(model, tokenizer, run_id=STUDY_ID, source_sha256=digest(before_files),
                resident_state_sha256=before_state, tokenizer_sha256=before_files["tokenizer.json"], api_cap=34596,
                guard=guard, synchronize=synchronize, event_sink=lambda e: ledger.accept(e, "session", context),
                trajectory_sink=lambda e: tj.append({**context, **e}))
            def reference(row, k, backend, ids=None):
                before = time.perf_counter(); prior_seconds = sum(x.get("forward_seconds", x.get("elapsed_seconds", 0.)) for x in ledger.completed)
                dummy = {"case_id": row["case_id"], "family": "NOT_SCORED", "scenario_id": "NOT_SCORED", "prompt": row["raw_prompt"],
                         "max_new_tokens": 24, "scorer": {"type": "NOT_SCORED"}, "format_scorer": None}
                with torch.inference_mode(), BlockRecurrence(model, extra_loops=k, config=TAPSConfig(scheduler_mode="unit"), guard=guard) as w:
                    install_gpu_timing(w, synchronize)
                    result, first = evaluate_case(model, tokenizer, dummy, {**historical["config"], "max_new_tokens": 24}, w,
                        lambda *a: {"score": None}, output_flags, lambda *a: {}, synchronize=synchronize, guard=guard,
                        forward_sink=lambda e: ledger.accept(e, backend, {"phase": "QUALIFICATION_GENERATION", "K": k}))
                now_seconds = sum(x.get("forward_seconds", x.get("elapsed_seconds", 0.)) for x in ledger.completed)
                nonforward.append(max(0., time.perf_counter() - before - (now_seconds - prior_seconds)))
                return result, first
            full_parity, qualification_policy_decisions = [], []
            for row in smoke:
                a0, first0 = reference(row, 0, "ref0:" + row["case_id"])
                unit, first1 = reference(row, 1, "ref1:" + row["case_id"])
                ids = torch.tensor([row["input_token_ids"]], device="cuda:0")
                for k, base, first in ((0, a0, first0), (1, unit, first1)):
                    context.update(case_id=row["case_id"], selected_K=k)
                    begin = time.perf_counter(); p = session.make_pilot(ids, started_perf_counter=begin)
                    if p.logits_sha256 != GenerationSession._tensor_hash(first0) or p.probability != readout_stats(first0)["top1_probability"]:
                        raise RuntimeError("actual native intact-pilot/minimal probability identity failed")
                    got, logits = session.generate(ids, k=k, pilot=p, max_new_tokens=24, qualification=True)
                    for key in ("token_ids", "response", "termination", "eos_emitted", "truncated", "safety_stopped"):
                        if got[key] != base[key]: raise RuntimeError("actual full own-path/reuse parity failed:" + key)
                    if not torch.equal(logits, first): raise RuntimeError("selected first native logits differ")
                    nonforward.append(max(0., time.perf_counter() - begin - got["physical_forward_seconds"]))
                full_parity.append({"case_id": row["case_id"], "K0_tokens": a0["token_ids"], "unit_tokens": unit["token_ids"],
                                    "score_status": "NOT_COLLECTED_NOT_SCORED"})
                qualification_policy_decisions.append({"case_id": row["case_id"], "actual_intact_probability": p.probability,
                    "CONFIDENT": asdict(decide(CONFIDENT, row["raw_prompt"], p.probability)),
                    "UNCERTAIN": asdict(decide(UNCERTAIN, row["raw_prompt"], p.probability)),
                    "GUARD": asdict(decide(GUARD, row["raw_prompt"])), "HEADER": asdict(header_cue(row["raw_prompt"]))})
            max_before = ledger.attempts
            context.clear(); context.update(case_id="MAX357", phase="QUALIFICATION_MAX_DRIVER")
            max_driver_proof = qualify_max_driver(session, max_ids, context)
            if ledger.attempts - max_before != 9 or ledger.attempts > 852 or len(ledger.failed) != 2 or len(ledger.cache) != 2:
                raise RuntimeError("fresh complete qualification cap/partial/cache contract differs")
            synchronize(); after_qual = state_hash(model)
            if (after_qual != before_state or checkpoint_files(source) != before_files or sha(code) != receipt["loaded_custom_code_sha256"]
                    or any(p.requires_grad or p.grad is not None for p in model.parameters())):
                raise RuntimeError("qualification mutation or parameter-gradient failure")
            verify_dispatch(admission_path)
            qualification = {"status": "PASS_FRESH_K01_NATIVE_MECHANICS", "counts": ledger.counts(), "prefix_counts": prefix.counts(),
                             "prefix_parity": parity, "full_generation_parity": full_parity, "max_driver_API_attempts": 9,
                             "actual_input_policy_mapping": qualification_policy_decisions,
                             "minimal_probability_native_parity": True,
                             "max_driver_proof": max_driver_proof,
                             "before_state_sha256": before_state, "after_state_sha256": after_qual, "checkpoint_files": before_files,
                             "source_unchanged": True, "parameter_gradients": "NONE", "resources": guard(), "maximum_prefix_tokens": 357,
                             "forced_exception_restoration": True, "cache_contracts": [[True, False], [False, True]],
                             "L0_L2_and_conditioning": "unchanged native one-shot and CUDA mask/arange/replay identity guards",
                             "score_status": "NOT_COLLECTED_NOT_SCORED"}
            write_json(out / "qualification.json", qualification)
            warm = [x.get("forward_seconds", x.get("elapsed_seconds")) for x in ledger.completed[1:]]
            forecast = full_forecast(warm, nonforward, max(parser_times), time.perf_counter() - started)
            write_json(out / "matrix_forecast.json", forecast)
            if not forecast["admitted"]: raise RuntimeError("entire fresh all-control forecast refuses15600")
            random17 = matched_random_decisions(160 / 192, 456, 17); random29 = matched_random_decisions(160 / 192, 456, 29)
            write_json(out / "random_routes.json", {"rate": 160 / 192, "seed17": random17, "seed29": random29,
                       "order_ids_sha256": digest([x["case_id"] for x in population]), "scope": "case-level; no per-prefix redraw"})
            rows = []
            class Driver(NativeDriver):
                def begin_path(self, row, index, path):
                    context.clear(); context.update(phase="TASK", case_id=row["case_id"], cohort=row["cohort"],
                                                     item_index=index, path=path)
                def make_pilot(self, row, begun):
                    return super().make_pilot(row, begun)
                def generate(self, row, k, pilot, begun):
                    result = super().generate(row, k, pilot, begun)
                    result["local_physical_API_attempts"] = result["physical_API_attempts"]
                    result["physical_API_attempts"] = [ledger.mapping[("session", x)] for x in result["local_physical_API_attempts"]]
                    return result
            driver = Driver(session)
            for index, row in enumerate(population):
                execute_item(row, index, driver, random17[index], random29[index], lambda x: (outcomes.append(x), rows.append(x)))
                if index % 16 == 0: disk_guard()
            synchronize(); after_state, after_files = state_hash(model), checkpoint_files(source)
            verify_dispatch(admission_path)
            if after_state != before_state or after_files != before_files or sha(code) != receipt["loaded_custom_code_sha256"]:
                raise RuntimeError("fresh final source/assets/resident mutation")
            if len(rows) != 4104 or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise RuntimeError("complete raw matrix/gradient proof failed")
            task_completed = [x for x in ledger.completed if x.get("phase") == "TASK"]
            task_ids = [x for row in rows for x in row["physical_API_attempts"]]
            if (len(task_ids) != len(set(task_ids)) or set(task_ids) != {x["API_attempt"] for x in task_completed}
                    or sum(row["completed_token_positions"] for row in rows) != sum(x["prefix_tokens"] for x in task_completed)
                    or ledger.attempts > 34596 or len(ledger.failed) != 2 or len(ledger.cache) != 2):
                raise RuntimeError("complete task/global physical calls/positions/partial ledger proof failed")
        receipt.update(status="FRESH_K01_RAW_MATRIX_COMPLETE_NOT_ANALYZED", complete=True, physical_rows=4104,
                       scientific_rows=3888, preservation_rows=216, after_state_sha256=after_state,
                       checkpoint_files_after=after_files, source_unchanged=True, parameter_gradients="NONE", trainable_parameters=0,
                       runtime=runtime, dispatch_hashes=admission["dispatch_hashes"], final_resources=guard())
    except Exception as error:
        receipt.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), complete=False)
    finally:
        if ledger is not None: receipt["counts"] = ledger.counts()
        receipt["parent_elapsed_seconds"] = None if started is None else time.perf_counter() - started
        cuda_fault = any(x in receipt.get("error", "").lower() for x in ("cuda error", "illegal memory access", "illegal address", "device-side assert"))
        receipt["final_resources_on_stop"] = None if cuda_fault else final_resource_snapshot()
        receipt["CUDA_state_after_failure"] = "UNKNOWN_NO_RECOVERY_QUERY" if cuda_fault else None
        write_json(out / "receipt.json", receipt)
    return 0 if receipt["status"] == "FRESH_K01_RAW_MATRIX_COMPLETE_NOT_ANALYZED" else 2


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "out", "admission"): parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); return run(args.source, args.out, args.admission)


if __name__ == "__main__": raise SystemExit(main())
