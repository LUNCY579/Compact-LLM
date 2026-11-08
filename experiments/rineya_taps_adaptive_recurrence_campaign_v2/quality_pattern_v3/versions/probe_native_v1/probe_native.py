"""Prospective passive first-prefix GPU probe; coordinator-only dispatch."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import inspect
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
from ..gpu_native import (GIB, final_resource_snapshot, forward_logits, gpu_resources,
                         install_gpu_timing, one_shot_anchors, require_cuda_tensor, resource_guard)
from .observer import SCHEMA, observed_block, readout_stats
from experiments.rineya_taps_adaptive_recurrence_v1.safety import Journal, digest, write_json

CAMPAIGN = Path(__file__).resolve().parents[1]
STUDY = Path(__file__).resolve().parent
EXPORT = STUDY / "data/dev_prefix_probe_001"
LANE = "GPU_FP32_PASSIVE_INPUT_FEATURE_PROBE"
CLOCK_ENV = "PASSIVE_PROBE_PARENT_STARTED_PERF_COUNTER"
GPU_SHA = "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f"
CONTROL_SHA = "298d0a7064ff562c498ecc50d13c2f23cd9bdf1df6b1d616d7204bac24a49ed3"
BANK_SHA = "681a3e77e3d797e2faebcff3971dbb365a4479ebdac017bf3dcfcb48081ba03f"
TOKENS_SHA = "c34948026ebaf332111ce635a473b1489ef0d2f3932100f9e0d3cdd6a08ce6d2"
EXPORT_SHA = "f35dc01c54154c51fa6284807245767f01282128c23a09307a2485483951530b"
QUAL_MODES = ("INTACT", "UNOBSERVED_K0", "OBSERVED_K0", "LEGACY_UNIT",
              "UNOBSERVED_UNIT", "OBSERVED_UNIT", "RESTORED_INTACT")


def count_contract():
    return {"cases": 192, "parents": 96, "task_full_forwards": 384,
            "qualification_smoke_full_attempts": 56, "qualification_max_full_attempts": 9,
            "qualification_full_attempts": 65, "qualification_completed_forwards": 64,
            "qualification_API_attempts": 67, "cache_rejections": 2, "expected_partial": 1,
            "model_API_attempts": 451, "full_forward_attempts": 449,
            "completed_forwards": 448, "state_adjoints": 0, "task_readouts": 384,
            "task_feature_events": 576, "maximum_prefix_tokens": 357,
            "qualification_completed_prefix_positions": 4921,
            "task_completed_prefix_positions": 49216, "completed_prefix_positions": 54137}


def parent_clock(admission):
    if (admission.get("hard_deadline_seconds") != 900
            or admission.get("internal_reserve_seconds") != 120
            or admission.get("global_finalization_reserve_seconds") != 1800):
        raise RuntimeError("passive probe deadline/reserves differ")
    try:
        started = float(os.environ[CLOCK_ENV])
    except (KeyError, ValueError, TypeError) as error:
        raise RuntimeError("passive parent clock absent or nonnumeric") from error
    if not math.isfinite(started) or not 0 <= time.perf_counter() - started < 900:
        raise RuntimeError("passive parent clock future/nonfinite/expired")
    return started


def disk_guard():
    if shutil.disk_usage(CAMPAIGN).free < 15 * GIB:
        raise MemoryError("unchanged host disk reserve unavailable")
    used = sum(path.stat().st_size for path in CAMPAIGN.rglob("*") if path.is_file())
    if used > 1536 * 1024**2:
        raise MemoryError("campaign2GiB must retain512MiB evidence reserve")


def source_closure(hashes):
    for relative, expected in hashes.items():
        path = (REPO / relative).resolve(strict=True)
        source = relative.endswith(".py") and relative.startswith((
            "experiments/rineya_taps_adaptive_recurrence_v1/",
            "experiments/anchor_routed_middle_recurrence_v1/src/",
            "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/"))
        data = relative in (
            "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl",
            "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/mergedev_fast.jsonl")
        if not (CAMPAIGN in path.parents or source or data) or sha(path) != expected:
            raise RuntimeError("passive source closure differs: " + relative)


def read_population(admission):
    """Only the exact192 DEV export; project away labels before model access."""
    expected = (("dev_bank", EXPORT / "DEVELOPMENT_BANK.json", BANK_SHA),
                ("dev_token_binding", EXPORT / "DEVELOPMENT_TOKEN_BINDING.json", TOKENS_SHA),
                ("dev_export_manifest", EXPORT / "MANIFEST.json", EXPORT_SHA))
    packets = []
    for key, exact, pinned in expected:
        path = Path(admission[key + "_path"]).resolve(strict=True)
        if path != exact.resolve(strict=True) or sha(path) != admission[key + "_sha256"] or sha(path) != pinned:
            raise PermissionError("passive exact DEVELOPMENT export differs")
        packets.append(json.loads(path.read_text(encoding="utf-8")))
    bank, token_packet, manifest = packets
    if (manifest.get("status") != "PROMPT_ONLY_EXPOSED_DEV_EXPORT"
            or manifest.get("bank_sha256") != BANK_SHA or manifest.get("token_binding_sha256") != TOKENS_SHA
            or manifest.get("case_count") != 192 or manifest.get("parent_units") != 96
            or manifest.get("allowed_split") != "DEVELOPMENT" or manifest.get("model_loads") != 0):
        raise RuntimeError("passive DEV-only export manifest differs")
    if (token_packet.get("status") != "PROMPT_ONLY_NATIVE_NOT_MEASURED"
            or token_packet.get("native_model_loads") != 0
            or token_packet.get("canonical_source") != admission["source"]):
        raise RuntimeError("passive token source/no-model binding differs")
    tokens = token_packet["rows"]
    return project_population(bank, tokens)


def project_population(bank, tokens):
    bank_keys = {"case_id", "split", "raw_prompt", "raw_prompt_sha256"}
    token_keys = bank_keys | {"rendered_prompt", "rendered_prompt_sha256", "input_token_ids", "input_tokens"}
    if any(set(row) != bank_keys for row in bank) or any(set(row) != token_keys for row in tokens):
        raise PermissionError("passive packet includes label/metadata or misses exact prompt-only schema")
    by = {row["case_id"]: row for row in tokens}
    if len(bank) != 192 or len(tokens) != 192 or len(by) != 192 or len({row["case_id"] for row in bank}) != 192:
        raise RuntimeError("passive complete192 unique cells required")
    if set(by) != {row["case_id"] for row in bank}:
        raise RuntimeError("passive exact192 population differs")
    rows = []
    for raw in bank:
        token = by[raw["case_id"]]
        ids = token["input_token_ids"]
        if (raw["split"] != "DEVELOPMENT" or token["split"] != "DEVELOPMENT"
                or raw["raw_prompt"] != token["raw_prompt"] or text_sha(raw["raw_prompt"]) != raw["raw_prompt_sha256"]
                or raw["raw_prompt_sha256"] != token["raw_prompt_sha256"]
                or text_sha(token["rendered_prompt"]) != token["rendered_prompt_sha256"] or token["input_tokens"] != len(ids)
                or not 1 <= len(ids) <= 232 or any(type(x) is not int or not 0 <= x < 151936 for x in ids)):
            raise RuntimeError("passive original input/split/shape binding differs")
        rows.append({"case_id": raw["case_id"], "split": "DEVELOPMENT", "prompt": raw["raw_prompt"],
                     "prompt_sha256": raw["raw_prompt_sha256"], "rendered_prompt": token["rendered_prompt"],
                     "rendered_prompt_sha256": text_sha(token["rendered_prompt"]),
                     "input_token_ids": ids, "prefix_ids_sha256": digest(ids), "prefix_tokens": len(ids)})
    return rows


def complete_forecast(warm_times, elapsed):
    if (not warm_times or any(not math.isfinite(x) or x <= 0 for x in warm_times)
            or not math.isfinite(elapsed) or elapsed < 0):
        raise RuntimeError("passive fresh max357/SMOKE timings missing")
    maximum = max(warm_times)
    estimate = 384 * maximum * 1.5
    return {"remaining_calls": 384, "max_observed_warm_forward_seconds": maximum,
            "forecast_factor": 1.5, "remaining_forecast_seconds": estimate,
            "elapsed_seconds": elapsed, "internal_reserve_seconds": 120,
            "parent_deadline_seconds": 900, "admitted": elapsed + estimate + 120 <= 900,
            "scope": "all192 DEV two actual first-prefix forwards; cold/setup already charged"}


def tensor_hash(value):
    return __import__("hashlib").sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def hook_snapshot(model):
    return (len(model._forward_pre_hooks), len(model._forward_hooks), tuple(
        (len(layer._forward_pre_hooks), len(layer._forward_hooks), "forward" in layer.__dict__,
         id(layer.__dict__["forward"]) if "forward" in layer.__dict__ else None,
         "_taps_owner" in layer.__dict__, "_armr_owner" in layer.__dict__)
        for layer in model.model.layers))


def invoke_prefix(model, ids, mode, context, recorder, guard, synchronize, legacy_class, feature_sink=None):
    """Shared qualification/task context path; no device bypass in the recorder."""
    before_hooks = hook_snapshot(model)
    if mode not in QUAL_MODES:
        raise ValueError("passive mode outside registered qualification/task arms")
    if mode in ("INTACT", "RESTORED_INTACT"):
        wrapper = nullcontext()
    elif mode == "LEGACY_UNIT":
        wrapper = legacy_class(model, [12, 13], applications=2)
    elif mode.startswith("OBSERVED"):
        wrapper = observed_block(model, extra_loops=int(mode.endswith("UNIT")),
            sample_id=context["case_id"], sink=feature_sink, guard=guard)
    else:
        wrapper = BlockRecurrence(model, extra_loops=int(mode.endswith("UNIT")),
            config=TAPSConfig(scheduler_mode="unit"), sample_id=context["case_id"], guard=guard)
    try:
        with torch.inference_mode(), wrapper:
            if hasattr(wrapper, "originals"):
                install_gpu_timing(wrapper, synchronize)
            result = recorder.forward(model, ids, {**context, "mode": mode,
                "extra_loops": int(mode.endswith("UNIT")), "native_layers": [12, 13]})
    finally:
        if hook_snapshot(model) != before_hooks:
            raise RuntimeError("passive forward/hook restoration differs")
    if any("_taps_owner" in layer.__dict__ or "_armr_owner" in layer.__dict__ for layer in model.model.layers):
        raise RuntimeError("passive wrapper ownership leaked")
    return result, getattr(wrapper, "observer_seconds", 0.)


class Recorder:
    def __init__(self, journal, started, guard, synchronize):
        self.journal, self.started, self.guard, self.synchronize = journal, started, guard, synchronize
        self.API = self.attempts = self.completed = self.partial = self.cache = self.positions = 0
        self.times = []

    def append(self, value):
        self.journal.append({**value, "parent_elapsed_seconds": time.perf_counter() - self.started,
                             "timestamp_utc": datetime.now(timezone.utc).isoformat()})

    def forward(self, model, ids, context):
        self.guard()
        if self.API >= 451:
            raise RuntimeError("passive API hard cap exceeded")
        self.API += 1; self.attempts += 1
        before = time.perf_counter()
        base = {**context, "API_attempt": self.API, "prefix_tokens": ids.shape[1],
                "prefix_ids_sha256": digest(ids[0].tolist())}
        self.append({"event": "FORWARD_STARTED", **base, "completed": False})
        try:
            logits = forward_logits(model, ids, synchronize=self.synchronize)
        except Exception as error:
            self.partial += 1
            self.append({"event": "FORWARD_FAILED", **base, "completed": False,
                         "expected_forced_exception": context.get("expected_forced_exception", False),
                         "forward_seconds": time.perf_counter() - before, "partial_compute": "UNKNOWN",
                         "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            raise
        elapsed = time.perf_counter() - before
        self.completed += 1; self.positions += ids.shape[1]; self.times.append(elapsed)
        sample, guard_error = None, None
        try:
            sample = self.guard()
        except Exception as error:
            guard_error = error
        self.append({"event": "FORWARD_COMPLETED", **base, "completed": True,
                     "forward_seconds": elapsed, "resources": sample, "logits_sha256": tensor_hash(logits),
                     "post_forward_guard_error": None if guard_error is None else str(guard_error)})
        if guard_error is not None:
            raise guard_error
        return logits

    def reject_cache(self, model, ids, kwargs, wrapper):
        self.guard()
        if self.API >= 451:
            raise RuntimeError("passive API hard cap exceeded")
        self.API += 1
        before = time.perf_counter()
        base = {"API_attempt": self.API, "use_cache": kwargs["use_cache"],
                "cache_object_present": "past_key_values" in kwargs, "task_measurement": False}
        self.append({"event": "CACHE_REJECTION_STARTED", **base})
        try:
            model(input_ids=ids[:, :1], **kwargs)
        except RuntimeError as error:
            if "cache" not in str(error) or wrapper.forward_count != 0:
                raise
            self.cache += 1
            self.append({"event": "CACHE_REJECTED", **base, "completed": False,
                         "model_compute_started": False, "forward_seconds": time.perf_counter() - before})
        else:
            raise RuntimeError("passive cache rejection failed")

    def counts(self):
        return {"model_API_attempts": self.API, "full_forward_attempts": self.attempts,
                "completed_forwards": self.completed, "expected_partial": self.partial,
                "cache_rejections": self.cache, "completed_prefix_positions": self.positions, "state_adjoints": 0}


def validate_admission(admission, source):
    required = {"status": "READY_FOR_PASSIVE_PROBE_FP32_DISPATCH", "lane": LANE,
                "study_id": "V3-PASSIVE-PROBE-001",
                "device": "cuda:0", "dtype": "float32", "attn_implementation": "eager",
                "hard_deadline_seconds": 900, "internal_reserve_seconds": 120,
                "global_finalization_reserve_seconds": 1800, "gpu_initial_free_floor_bytes": 6 * GIB,
                "gpu_allocated_ceiling_bytes": 5 * GIB, "gpu_global_free_reserve_bytes": GIB,
                "worker_host_available_floor_bytes": 6 * GIB, "worker_rss_ceiling_bytes": 6 * GIB,
                "training": False, "native_test": False}
    if any(admission.get(k) != v for k, v in required.items()) or admission.get("source") != str(source):
        raise RuntimeError("passive source/lane/resource admission differs")
    if admission.get("count_contract") != count_contract():
        raise RuntimeError("passive exact count contract differs")
    for path, expected in ((CAMPAIGN / "gpu_native.py", GPU_SHA), (CAMPAIGN / "control.py", CONTROL_SHA)):
        if sha(path) != expected:
            raise RuntimeError("passive frozen helper changed")


def run(source, out, admission_path):
    source, out, admission_path = Path(source).resolve(), Path(out).resolve(), Path(admission_path).resolve(strict=True)
    if STUDY not in out.parents or CAMPAIGN not in admission_path.parents:
        raise PermissionError("passive output/admission must remain campaign-local")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "FAIL", "lane": LANE, "model_load_attempts": 0, "native_model_loads": 0,
               "training": False, "native_test": False, "state_adjoints": 0, "efficacy_claim": False}
    model, started, recorder = None, None, None
    try:
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        validate_admission(admission, source)
        review_path = (CAMPAIGN / admission["review_receipt"]).resolve(strict=True)
        if CAMPAIGN not in review_path.parents:
            raise PermissionError("passive review path outside campaign")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("status") != "PASS_PASSIVE_PROBE_FP32_DISPATCH" or review.get("admission_sha256") != sha(admission_path):
            raise RuntimeError("passive independent review/admission binding differs")
        source_closure(admission["dispatch_hashes"])
        # Coordinator-owned new authority helper; import is intentionally deferred.
        from .authority import validate_probe_scope
        validate_probe_scope(admission)
        started = parent_clock(admission)
        population = read_population(admission)
        binding_path = Path(admission["model_binding_path"]).resolve(strict=True)
        audit_path = Path(admission["smoke_prompt_audit_path"]).resolve(strict=True)
        if (binding_path != (CAMPAIGN / "runs/binding-001/MODEL_DATA_BINDING.json").resolve(strict=True)
                or audit_path != (CAMPAIGN / "runs/binding-001/PROMPT_AUDIT.json").resolve(strict=True)
                or sha(binding_path) != admission["model_binding_sha256"]
                or sha(audit_path) != admission["smoke_prompt_audit_sha256"]):
            raise RuntimeError("passive exact model/SMOKE binding differs")
        binding, audit = (json.loads(path.read_text(encoding="utf-8")) for path in (binding_path, audit_path))
        if not isinstance(audit, list):
            raise RuntimeError("passive inherited prompt audit schema differs")
        smoke = [row for row in audit if row["split"] == "SMOKE"]
        if len(smoke) != 8 or len({row["case_id"] for row in smoke}) != 8:
            raise RuntimeError("passive all8 SMOKE binding required")
        historical, inherited_hashes = admitted_source(source)
        smoke_path = LAB / "data/evaluation/supa_layer_ablation_smoke.jsonl"
        if sha(smoke_path) != binding["datasets"]["SMOKE"]["sha256"]:
            raise RuntimeError("passive exact inherited SMOKE source differs")
        disk_guard()
        cache = out / "hf_cache"
        os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"), HF_MODULES_CACHE=str(cache / "modules"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        runtime = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                   "qwen_implementation_sha256": sha(inspect.getfile(Qwen3DecoderLayer)), "cuda": torch.version.cuda,
                   "dtype": "float32", "attn_implementation": "eager", "CPU_threads": 4}
        for key in ("python", "torch", "transformers", "qwen_implementation_sha256"):
            if runtime[key] != historical[key]:
                raise RuntimeError("passive exact qualified runtime differs")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("passive deterministic CUBLAS environment not bound before imports")
        if not torch.cuda.is_available() or psutil.virtual_memory().available < 6 * GIB:
            raise MemoryError("passive initial host/GPU worker floor unavailable")
        torch.cuda.set_device(0); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        runtime.update(CPU_threads=torch.get_num_threads(), device_name=torch.cuda.get_device_name(0),
                       tf32=False, CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                       CUDA_LAUNCH_BLOCKING=os.environ.get("CUDA_LAUNCH_BLOCKING"))
        initial_resources = gpu_resources()
        if initial_resources["gpu_free_bytes"] < 6 * GIB:
            raise MemoryError("passive initial GPU free6GiB floor unavailable")
        def guard():
            if time.perf_counter() - started >= 780:
                raise TimeoutError("passive body deadline reached; preserve120s evidence reserve")
            return resource_guard()
        guard()
        write_json(out / "registration.json", {"lane": LANE, "study_id": "V3-PASSIVE-PROBE-001", "schema": SCHEMA, "runtime": runtime,
                   "admission_sha256": sha(admission_path), "count_contract": count_contract(),
                   "source_hashes": inherited_hashes, "data_sha256": admission["dev_bank_sha256"],
                   "token_binding_sha256": admission["dev_token_binding_sha256"],
                   "initial_resources": initial_resources, "hard_deadline_seconds": 900, "body_deadline_seconds": 780,
                   "internal_reserve_seconds": 120, "global_finalization_reserve_seconds": 1800,
                   "native_layers": [12, 13], "extra_loops": [0, 1], "alpha": 1., "seed": 17,
                   "prefix_kind": "ORIGINAL_INPUT_ONLY", "generation": False, "teacher_conditioned": False,
                   "training": False, "state_adjoints": 0, "readout_changed": False, "controller_routing": False})
        write_json(out / "population.json", population)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False, cache_dir=cache / "hub")
        for row in population + smoke:
            prompt = row.get("prompt", row.get("raw_prompt"))
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
            if rendered != row["rendered_prompt"] or tokenizer(rendered, return_tensors="pt")["input_ids"][0].tolist() != row["input_token_ids"]:
                raise RuntimeError("passive exact rendered original IDs differ")
        guard(); receipt["model_load_attempts"] = 1
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=True,
                    dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True,
                    device_map={"": "cuda:0"}, cache_dir=cache / "hub").eval().requires_grad_(False)
        receipt["native_model_loads"] = 1; receipt["after_load_resources"] = guard()
        code = Path(inspect.getfile(type(model))).resolve()
        if (type(model).__name__ != "Qwen3AbstainForCausalLM" or cache not in code.parents
                or sha(code) != historical["checkpoint_files_before"]["modeling_supa_a11.py"]
                or model.config.abstain_token_id != 51378 or model.config.abstain_logit_bias != 5.156249985098839
                or model.config.vocab_size != 151936):
            raise RuntimeError("passive exact custom head/code differs")
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            require_cuda_tensor(value, name)
            if value.is_floating_point() and value.dtype != torch.float32:
                raise RuntimeError("passive model floating tensor notFP32")
        sys.path[:0] = [str(ARMR / "src"), str(LAB / "src")]
        from armr.provenance import checkpoint_files, state_hash
        from armr.recurrence import FixedBlock
        before_files, before_state = checkpoint_files(source), state_hash(model)
        if before_state != historical["state_hash_before"] or before_files != binding["checkpoint_files"]:
            raise RuntimeError("passive exact loaded asset/state binding differs")
        receipt.update(before_state_sha256=before_state, loaded_custom_code_sha256=sha(code),
                       checkpoint_files_before=before_files)
        synchronize = lambda: torch.cuda.synchronize(0)
        task_readouts = task_features = 0
        with Journal(out / "forwards.jsonl") as forwards, Journal(out / "feature_events.jsonl") as features, Journal(out / "readouts.jsonl") as readouts, one_shot_anchors(model):
            recorder = Recorder(forwards, started, guard, synchronize)
            def invoke(ids, mode, context, feature_sink=None):
                return invoke_prefix(model, ids, mode, context, recorder, guard, synchronize, FixedBlock, feature_sink)
            def parity(ids, context):
                values, observer_times = [], []
                for mode in QUAL_MODES:
                    def feature_sink(event):
                        features.append({**context, "mode": mode, **event, "task_measurement": False,
                                         "prefix_kind": "SYNTHETIC_MAX357_NO_TARGET" if context["split"] == "MECHANICAL" else "ORIGINAL_INPUT_ONLY"})
                    logits, observer_time = invoke(ids, mode, context, feature_sink)
                    values.append(logits)
                    observer_times.append(observer_time)
                if not (all(torch.equal(values[0], values[i]) for i in (1, 2, 6))
                        and all(torch.equal(values[3], values[i]) for i in (4, 5))):
                    raise RuntimeError("passive byte-exact intact/unit/legacy/parity failed")
                return {"context": context, "modes": list(QUAL_MODES), "logits_sha256": [tensor_hash(v) for v in values],
                        "observer_seconds": observer_times}
            parity_proofs = []
            for row in smoke:
                ids = torch.tensor([row["input_token_ids"]], dtype=torch.long, device="cuda:0")
                parity_proofs.append(parity(ids, {"case_id": row["case_id"], "split": "SMOKE", "phase": "QUALIFICATION"}))
            maximal = max(population, key=lambda row: row["prefix_tokens"])
            filler = tokenizer.encode(".", add_special_tokens=False)[0]
            max_ids = torch.tensor([maximal["input_token_ids"] + [filler] * (357 - maximal["prefix_tokens"])], device="cuda:0")
            max_context = {"case_id": "SYNTHETIC_MAX357_NO_TARGET", "split": "MECHANICAL", "phase": "QUALIFICATION"}
            parity_proofs.append(parity(max_ids, max_context))
            def forced(_event):
                raise RuntimeError("EXPECTED_PASSIVE_FORCED_EXCEPTION")
            try:
                invoke(max_ids, "OBSERVED_UNIT", {**max_context, "expected_forced_exception": True}, forced)
            except RuntimeError as error:
                if str(error) != "EXPECTED_PASSIVE_FORCED_EXCEPTION":
                    raise
            else:
                raise RuntimeError("passive forced exception did not execute")
            restored, _ = invoke(max_ids, "RESTORED_INTACT", {**max_context, "after_forced_exception": True})
            if tensor_hash(restored) != parity_proofs[-1]["logits_sha256"][0]:
                raise RuntimeError("passive exception restoration differs")
            for kwargs in ({"use_cache": True}, {"use_cache": False, "past_key_values": object()}):
                with torch.inference_mode(), observed_block(model, extra_loops=1, sample_id="cache", sink=None) as wrapper:
                    recorder.reject_cache(model, max_ids, kwargs, wrapper)
            synchronize()
            after_qualification_state = state_hash(model)
            if (after_qualification_state != before_state or checkpoint_files(source) != before_files
                    or sha(code) != receipt["loaded_custom_code_sha256"]
                    or any(module.training for module in model.modules())
                    or any(p.requires_grad or p.grad is not None for p in model.parameters())):
                raise RuntimeError("passive qualification mutated frozen model/assets")
            source_closure(admission["dispatch_hashes"])
            if recorder.counts() != {
                "model_API_attempts": 67, "full_forward_attempts": 65, "completed_forwards": 64,
                "expected_partial": 1, "cache_rejections": 2, "completed_prefix_positions": 4921, "state_adjoints": 0}:
                raise RuntimeError("passive exact qualification counts differ")
            write_json(out / "qualification.json", {"status": "PASS_PASSIVE_INPUT_FEATURE_MECHANICS", "counts": recorder.counts(),
                       "parity_proofs": parity_proofs, "maximum_prefix_tokens": 357, "resources": guard(),
                       "before_state_sha256": before_state, "after_state_sha256": after_qualification_state,
                       "checkpoint_files_before": before_files, "checkpoint_files_after": checkpoint_files(source),
                       "dispatch_hashes": admission["dispatch_hashes"],
                       "checkpoint_files_unchanged": True, "source_unchanged": True, "parameter_gradients": "NONE",
                       "forced_exception_restoration": True, "cache_rejections": 2, "L0_L2_one_shot": True,
                       "mask_positions": "frozen GPU helper CUDA arange/same replay conditioning",
                       "head_and_operator_unchanged": True, "observer_output_identity": True})
            forecast = complete_forecast(recorder.times[1:], time.perf_counter() - started)
            write_json(out / "matrix_forecast.json", forecast)
            if not forecast["admitted"]:
                raise RuntimeError("complete passive192 matrix forecast refuses900s budget")
            disk_guard()
            for row in population:
                ids = torch.tensor([row["input_token_ids"]], dtype=torch.long, device="cuda:0")
                for mode in ("OBSERVED_K0", "OBSERVED_UNIT"):
                    context = {"case_id": row["case_id"], "split": "DEVELOPMENT",
                               "phase": "TASK", "mode": mode, "prefix_ids_sha256": row["prefix_ids_sha256"],
                               "prompt_sha256": row["prompt_sha256"], "rendered_prompt_sha256": row["rendered_prompt_sha256"]}
                    def feature_sink(event):
                        nonlocal task_features
                        features.append({**context, **event, "task_measurement": True})
                        task_features += 1
                    logits, observer_seconds = invoke(ids, mode, context, feature_sink)
                    before = time.perf_counter()
                    value = readout_stats(logits)
                    synchronize()
                    readout_seconds = time.perf_counter() - before
                    readouts.append({**context, **value, "prefix_kind": "ORIGINAL_INPUT_ONLY", "preceding_gold_tokens": 0,
                                     "teacher_conditioned": False, "observer_seconds": observer_seconds,
                                     "readout_observer_seconds": readout_seconds, "extra_model_calls_for_statistics": 0})
                    task_readouts += 1
            synchronize()
            after_state, after_files = state_hash(model), checkpoint_files(source)
            source_closure(admission["dispatch_hashes"])
            if (after_state != before_state or after_files != before_files
                    or sha(code) != receipt["loaded_custom_code_sha256"]
                    or any(module.training for module in model.modules())
                    or any(p.requires_grad or p.grad is not None for p in model.parameters())):
                raise RuntimeError("passive final state/assets/source/parameter firewall failed")
            expected_positions = sum(len(row["input_token_ids"]) for row in smoke) * 7 + 8 * 357 + sum(row["prefix_tokens"] for row in population) * 2
            if (recorder.API != 451 or recorder.completed != 448 or recorder.attempts != 449
                    or recorder.partial != 1 or recorder.cache != 2 or recorder.positions != expected_positions or expected_positions != 54137
                    or task_readouts != 384 or task_features != 576):
                raise RuntimeError("passive complete count/position/event matrix differs")
        receipt.update(status="PASSIVE_INPUT_FEATURE_PROBE_COMPLETE_NOT_EFFICACY", counts=recorder.counts(),
                       task_readouts=task_readouts, task_feature_events=task_features, cases=192,
                       after_state_sha256=after_state, checkpoint_files_after=after_files,
                       checkpoint_files_unchanged=True, source_unchanged=True,
                       parameter_gradients="NONE", trainable_parameters=0, runtime=runtime,
                       final_resources=resource_guard(), dispatch_hashes=admission["dispatch_hashes"], complete=True)
    except Exception as error:
        receipt.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), complete=False)
    finally:
        if recorder is not None:
            receipt["counts"] = recorder.counts()
        receipt["parent_elapsed_seconds"] = None if started is None else time.perf_counter() - started
        cuda_fault = any(term in receipt.get("error", "").lower() for term in ("illegal memory access", "illegal address", "device-side assert"))
        receipt["final_resources_on_stop"] = None if cuda_fault else final_resource_snapshot()
        receipt["CUDA_state_after_failure"] = "UNKNOWN_NO_RECOVERY_QUERY" if cuda_fault else None
        write_json(out / "receipt.json", receipt)
    return 0 if receipt["status"] == "PASSIVE_INPUT_FEATURE_PROBE_COMPLETE_NOT_EFFICACY" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--admission", required=True)
    args = parser.parse_args()
    return run(args.source, args.out, args.admission)


if __name__ == "__main__":
    raise SystemExit(main())
