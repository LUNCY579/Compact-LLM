"""288-case bulk successor; deterministic policies run once per case.

No replacement of inherited readers or guards. The coordinator alone launches
this worker through reviewed containment; importing it loads no model assets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import psutil
import torch

from .binding import ARMR, LAB, REPO, admitted_source, sha
from .control import BlockRecurrence, TAPSConfig
from .gpu_native import (GIB, GRID as REFERENCE_GRID, SEEDS, encode_prefix, evaluate_case,
                         final_resource_snapshot, forward_logits, gpu_resources,
                         install_gpu_timing, one_shot_anchors, require_cuda_tensor,
                         resource_guard)
from experiments.rineya_taps_adaptive_recurrence_v1.integration import BlockRecurrence as ReferenceBlock
from experiments.rineya_taps_adaptive_recurrence_v1.safety import Journal, digest, write_json

PACKAGE = Path(__file__).resolve().parent
GPU_HELPER_SHA256 = "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f"
STATIC_PREFIX_MEAN_ALPHA = 1.1473425965417514
GRID = tuple(sorted((*REFERENCE_GRID, STATIC_PREFIX_MEAN_ALPHA)))


def bank_module():
    from . import bulk_bank
    return bulk_bank


def count_forecast():
    return {"bulk_cases": 288, "parents": 144, "deterministic_policies": 9,
            "deterministic_evaluations_per_case": 9, "A5_evaluations_per_case": 2,
            "bulk_physical_evaluations": 3168, "SMOKE_physical_evaluations": 128,
            "physical_evaluations": 3296, "generation_call_cap": 28416,
            "maximum_shape_forward_attempts": 18, "cache_rejection_attempts": 2,
            "model_API_attempt_cap": 28436, "oracle_nonphysical_rows": 288,
            "validation_frozen_alias_nonphysical_rows": 96}


def deterministic_order(raw, parent_indices, variant_indices):
    labels = ["A0", "A1", *(f"A2:{alpha:g}" for alpha in GRID), "A3"]
    slot = 2 * parent_indices[raw["parent_id"]] + variant_indices[(raw["parent_id"], raw["variant_id"])]
    offset = slot % len(labels)
    rotated = labels[offset:] + labels[:offset]
    return list(reversed(rotated)) if (slot // len(labels)) % 2 else rotated


def read_bulk_packet(admission):
    bulk_bank = bank_module()
    """Read one reviewed OPEN packet; no dataset-directory scan or substitutions."""
    path = Path(admission["bulk_bank_path"]).resolve(strict=True)
    if PACKAGE not in path.parents or path.name != "bank.json" or not path.parent.name.startswith("bulk-bank-"):
        raise PermissionError("only the exact campaign OPEN bank packet is admitted")
    if sha(path) != admission["bulk_bank_sha256"]:
        raise RuntimeError("OPEN bank bytes differ from reviewed admission")
    expected_files = {"manifest.json": admission["bulk_manifest_sha256"],
                      "audit.json": admission["bulk_audit_sha256"],
                      "positive_controls.json": admission["bulk_positive_controls_sha256"]}
    for name, expected in expected_files.items():
        if sha(path.parent / name) != expected:
            raise RuntimeError("OPEN preparation packet differs: " + name)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((path.parent / "audit.json").read_text(encoding="utf-8"))
    if (manifest.get("generator_version") != bulk_bank.VERSION or manifest.get("training") is not False
            or manifest.get("files", {}).get("bank.json") != sha(path)
            or manifest.get("source_sha256") != admission["bulk_source_sha256"]
            or sha(PACKAGE / "bulk_bank.py") != admission["bulk_source_sha256"]
            or manifest.get("status") != "PASS_PREPARATION_ONLY" or audit.get("status") != "PASS_PREPARATION_ONLY"
            or audit.get("failures") != [] or audit.get("positive_control_correct") != 288):
        raise RuntimeError("OPEN provenance/positive-control/shortcut apparatus gate differs")
    normalizer_hash = sha(PACKAGE / "open_bank.py")
    if manifest.get("normalizer_source_sha256") != normalizer_hash or admission.get("bulk_normalizer_source_sha256") != normalizer_hash:
        raise RuntimeError("bulk reused strict normalizer source binding differs")
    rows = json.loads(path.read_text(encoding="utf-8"))
    positive = json.loads((path.parent / "positive_controls.json").read_text(encoding="utf-8"))
    if len(positive) != 288 or {row["id"] for row in positive} != {row["id"] for row in rows} or not all(row.get("correct") is True for row in positive):
        raise RuntimeError("bulk complete positive-control receipt differs")
    parents = Counter(row["parent_id"] for row in rows)
    if len(rows) != 288 or len({row["id"] for row in rows}) != 288 or len(parents) != 144 or set(parents.values()) != {2}:
        raise RuntimeError("OPEN packet count/parent identity differs")
    dev = [row for row in rows if row.get("split") == "DEVELOPMENT"]
    validation = [row for row in rows if row.get("split") == "OPEN_VALIDATION"]
    if len(dev) != 192 or len(validation) != 96:
        raise PermissionError("OPEN split contract differs")
    # Parent worlds are disjoint; shared generator code/families remain explicit
    # dependence, not renamed to imply independent source confirmation.
    if {row["parent_id"] for row in dev} & {row["parent_id"] for row in validation}:
        raise PermissionError("bulk parent worlds cross selection and validation")
    if len({row["category"] for row in rows}) != 6:
        raise RuntimeError("bulk six-category contract differs")
    for parent in parents:
        variants = [row for row in rows if row["parent_id"] == parent]
        if len({row["variant_id"] for row in variants}) != 2 or len({row["category"] for row in variants}) != 1:
            raise RuntimeError("bulk matched variant/parent category identity differs")
    for row in rows:
        provenance = row.get("provenance", {})
        if (provenance.get("historical_parents") != []
                or provenance.get("construction") != "independently seeded synthetic parent; shared authored generator/template"
                or provenance.get("generator_version") != bulk_bank.VERSION or provenance.get("nonce") != manifest["nonce"]
                or provenance.get("seed") != 17
                or row["shape"]["max_output_tokens"] != 8 or row["shape"]["target_max_input_tokens"] != 350):
            raise PermissionError("OPEN input is not a source-admitted fresh capped world")
    return rows, manifest


def adapt_bulk_case(raw):
    return {"case_id": raw["id"], "family": raw["category"], "scenario_id": raw["parent_id"],
            "prompt": raw["prompt"], "max_new_tokens": 8,
            "scorer": {"mode": "task", "case": raw}, "format_scorer": {"mode": "format", "case": raw}}


def strict_score(response, spec):
    bulk_bank = bank_module()
    result = bulk_bank.score_response(response, spec["case"])
    return {"score": int(result["correct"] if spec["mode"] == "task" else result["format_valid"])}


def bulk_diagnostics(response, raw):
    bulk_bank = bank_module()
    normalized = bulk_bank.normalize_output(response)
    unknown = normalized == "UNKNOWN"
    unanswerable = raw["answerability"] == "unanswerable"
    return {"normalized_response": normalized, "answerability": raw["answerability"],
            "source_status": raw["source_status"], "canonical_UNKNOWN": unknown,
            "literal_UNKNOWN": response.strip() == "UNKNOWN",
            "false_abstention_on_answerable": not unanswerable and unknown,
            "correct_canonical_abstention": unanswerable and unknown,
            "unsupported_canonical_answer_on_unanswerable": (not unknown if normalized is not None else None) if unanswerable else False,
            "semantic_abstention_beyond_canonical_UNKNOWN": "NOT_ASSESSED",
            "semantic_reasoning": "structured scalar contract only; human reasoning assessment NOT_RUN"}


def freeze_development(rows, traces, bank_sha, token_sha, source_sha, expected_dev_ids=None):
    # Refuse validation before reading any score or alpha field.
    if not rows or any(row.get("split") != "DEVELOPMENT" for row in rows):
        raise PermissionError("only OPEN development outcomes may select fixed alpha")
    if any(event.get("split") != "DEVELOPMENT" or event.get("arm") != "A3" for event in traces):
        raise PermissionError("random pool admits development generation A3 only")
    fixed_rows = [row for row in rows if row["arm"] == "A2"]
    cells = {(row["case_id"], row["seed"], row["fixed_alpha"]) for row in fixed_rows}
    case_ids = {row["case_id"] for row in rows}
    if (len(case_ids) != 192 or len(fixed_rows) != 192 * len(GRID)
            or len(cells) != len(fixed_rows)
            or cells != {(case_id, 17, alpha) for case_id in case_ids for alpha in GRID}
            or expected_dev_ids is not None and case_ids != set(expected_dev_ids)):
        raise RuntimeError("complete OPEN development fixed grid required before freeze")
    adaptive_rows = [row for row in rows if row["arm"] == "A3"]
    adaptive_cells = {(row["case_id"], row["seed"]) for row in adaptive_rows}
    if len(adaptive_rows) != 192 or adaptive_cells != {(case_id, 17) for case_id in case_ids}:
        raise RuntimeError("complete unique OPEN development A3 generations required for pool")
    allowed_cells = {cell: row for cell, row in zip(((row["case_id"], row["seed"]) for row in adaptive_rows), adaptive_rows)}
    grouped = {cell: [] for cell in allowed_cells}
    for event in traces:
        cell = (event.get("case_id"), event.get("seed"))
        if (cell not in grouped or event.get("configured_seed") != event.get("seed")
                or event.get("evidence_kind") != "native_trajectory"):
            raise PermissionError("OPEN random pool trace provenance differs from development generation")
        grouped[cell].append(event)
    for cell, row in allowed_cells.items():
        if not isinstance(row.get("token_ids"), list) or not row["token_ids"]:
            raise RuntimeError("OPEN A3 generation token evidence missing")
        expected = [(index, iteration) for index in range(1, len(row["token_ids"]) + 1) for iteration in (0, 1)]
        if [(event.get("forward_index"), event.get("iteration")) for event in grouped[cell]] != expected:
            raise RuntimeError("OPEN A3 prefix/iteration coverage differs from completed generation")
    # Equal categories; two semantic variants average within their parent first.
    scores = {}
    for alpha in GRID:
        by_parent = defaultdict(list); parent_category = {}
        for row in fixed_rows:
            if row["fixed_alpha"] != alpha: continue
            by_parent[row["parent_id"]].append(row["score"])
            parent_category[row["parent_id"]] = row["category"]
        categories = defaultdict(list)
        for parent, values in by_parent.items():
            if len(values) != 2: raise RuntimeError("bulk fixed policy lacks two variants per parent")
            categories[parent_category[parent]].append(statistics.mean(values))
        if len(categories) != 6: raise RuntimeError("bulk fixed calibration lacks six categories")
        scores[alpha] = statistics.mean(statistics.mean(values) for values in categories.values())
    best = max(GRID, key=lambda alpha: (scores[alpha], -alpha))
    pool = [min(1.2, max(.8, event["alpha"])) for event in traces if event.get("iteration") == 1]
    if not pool:
        raise RuntimeError("OPEN A5 cannot freeze an empty generation-derived pool")
    result = {"status": "FROZEN_BULK_DEVELOPMENT", "alpha": best, "pool": pool,
              "config": asdict(TAPSConfig()), "bank_sha256": bank_sha, "token_binding_sha256": token_sha,
              "runner_sha256": source_sha, "selection_case_ids": sorted(case_ids),
              "selection_parent_ids": sorted({row["parent_id"] for row in fixed_rows}),
              "pool_weighting": "generated prefixes; one deterministic A3 evaluation per case; longer responses contribute more",
              "deterministic_seed": 17, "A5_actual_seeds": [17, 29],
              "OPEN_VALIDATION_native_exposure_at_freeze": False, "controller_candidate_revision": 0}
    return {**result, "freeze_sha256": digest(result)}


def admit_validation(freeze, raw, bank_sha, token_sha, source_sha):
    payload = {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    if (freeze.get("status") != "FROZEN_BULK_DEVELOPMENT" or freeze.get("freeze_sha256") != digest(payload)
            or freeze.get("bank_sha256") != bank_sha or freeze.get("token_binding_sha256") != token_sha
            or freeze.get("runner_sha256") != source_sha or raw.get("split") != "OPEN_VALIDATION"
            or raw["id"] in freeze["selection_case_ids"] or raw["parent_id"] in freeze["selection_parent_ids"]):
        raise PermissionError("OPEN validation requires unchanged source/data/DEV freeze and disjoint case")


def validate_admission(admission, binding, source):
    expected = {"status": "READY_FOR_BULK_FP32_DISPATCH", "device": "cuda:0", "dtype": "float32",
                "attn_implementation": "eager", "gpu_allocated_ceiling_bytes": 5 * GIB,
                "gpu_global_free_reserve_bytes": GIB, "gpu_initial_free_floor_bytes": 6 * GIB,
                "worker_host_available_floor_bytes": 6 * GIB, "worker_rss_ceiling_bytes": 6 * GIB,
                "training": False, "native_test": False, "max_new_tokens": 8, "max_bulk_input_tokens": 350}
    if any(admission.get(key) != value for key, value in expected.items()):
        raise RuntimeError("OPEN data/scorer/cap/device/resource admission differs")
    if admission.get("source") != str(source) or admission.get("checkpoint_files") != binding["checkpoint_files"]:
        raise RuntimeError("OPEN canonical assets differ")
    if sha(PACKAGE / "gpu_native.py") != GPU_HELPER_SHA256 or admission.get("gpu_helpers_sha256") != GPU_HELPER_SHA256:
        raise RuntimeError("qualified GPU helpers changed; fresh source review required")


def run(source, out, admission_path):
    source, out, admission_path = Path(source).resolve(), Path(out).resolve(), Path(admission_path).resolve(strict=True)
    if PACKAGE not in out.parents or PACKAGE not in admission_path.parents:
        raise PermissionError("OPEN output/admission must remain campaign-local")
    out.mkdir(parents=True, exist_ok=False)
    started, model = time.perf_counter(), None
    receipt = {"status": "FAIL", "lane": "BULK_GPU_FP32_INTERVENTION", "native_model_loads": 0,
               "training": False, "native_test": False, "controller_candidate_revision": 0,
               "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        validate_admission(admission, binding, source)
        review_path = (PACKAGE / admission["review_receipt"]).resolve(strict=True)
        if PACKAGE not in review_path.parents:
            raise PermissionError("OPEN independent review must be campaign-local")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("status") != "PASS_BULK_FP32_DISPATCH" or review.get("admission_sha256") != sha(admission_path):
            raise RuntimeError("OPEN independent review is not bound to admission")
        for relative, expected in admission["dispatch_hashes"].items():
            path = (REPO / relative).resolve(strict=True)
            allowed = PACKAGE in path.parents or (relative.endswith(".py") and relative.startswith((
                "experiments/rineya_taps_adaptive_recurrence_v1/", "experiments/anchor_routed_middle_recurrence_v1/src/",
                "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/"))) or relative == "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl"
            if not allowed or sha(path) != expected:
                raise RuntimeError("OPEN source binding differs: " + relative)
        raw_cases, packet = read_bulk_packet(admission)
        token_path = Path(admission["bulk_token_binding_path"]).resolve(strict=True)
        if PACKAGE not in token_path.parents or sha(token_path) != admission["bulk_token_binding_sha256"]:
            raise RuntimeError("OPEN native tokenizer shape packet differs")
        token_binding = json.loads(token_path.read_text(encoding="utf-8"))
        if token_binding.get("bank_sha256") != admission["bulk_bank_sha256"]:
            raise RuntimeError("OPEN token packet does not bind this bank")
        token_rows = {row["case_id"]: row for row in token_binding["rows"]}
        if len(token_binding["rows"]) != 288 or set(token_rows) != {raw["id"] for raw in raw_cases}:
            raise RuntimeError("bulk tokenizer packet lacks complete288 unique cases")
        historical, source_hashes = admitted_source(source)
        cache = out / "hf_cache"
        os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"), HF_MODULES_CACHE=str(cache / "modules"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        if (platform.python_version() != historical["python"] or torch.__version__ != historical["torch"]
                or transformers.__version__ != historical["transformers"]
                or sha(inspect.getfile(Qwen3DecoderLayer)) != historical["qwen_implementation_sha256"]):
            raise RuntimeError("OPEN exact qualified runtime differs")
        if not torch.cuda.is_available() or psutil.virtual_memory().available < 6 * GIB:
            raise MemoryError("OPEN GPU or host6GiB worker floor unavailable")
        torch.cuda.set_device(0); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        if gpu_resources()["gpu_free_bytes"] < 6 * GIB:
            raise MemoryError("OPEN initial GPU free6GiB floor unavailable")
        old_config = {**historical["config"], "device": "cuda:0", "dtype": "float32", "max_rss_gib": 6}
        config = {**old_config, "max_new_tokens": 8}
        write_json(out / "registration.json", {"lane": "BULK_GPU_FP32_INTERVENTION", "admission_sha256": sha(admission_path),
                   "bank_sha256": admission["bulk_bank_sha256"], "token_binding_sha256": sha(token_path),
                   "scorer_sha256": sha(PACKAGE / "bulk_bank.py"), "config": config, "old_smoke_config": old_config,
                   "grid": GRID, "deterministic_seeds": [17], "A5_seeds": SEEDS, "native_layers": [12, 13], "extra_loops": [0, 1],
                   "dynamics_derived_static_alpha": STATIC_PREFIX_MEAN_ALPHA,
                   "static_alpha_source": "completed original A3 prefix mean; analyses/gpu-reference-002/REPORT.json; frozen prospectively",
                   "taps": asdict(TAPSConfig()), "source_hashes": source_hashes,
                   "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                               "transformers": transformers.__version__, "cuda": torch.version.cuda,
                               "device_name": torch.cuda.get_device_name(0), "tf32": False,
                               "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
                   "gold_nll_status": "NOT_COLLECTED_CAUSAL_GUARD",
                   "mask_positions": "batch1; no padding; full prefix; automatic arange; no KV",
                   "deterministic_order": "nine-policy rotation/reversal by sorted parent and variant index, independent of outputs",
                   "A5_order": "after DEV pool freeze; actual seed17 then29; phase limitation retained",
                   "count_forecast": count_forecast(),
                   "interventions": ["bulkdata288/144parents", "strictfullresponse scorer", "generationcap8", "deterministic policies once"],
                   "training": False, "native_test": False, "controller_candidate_revision": 0})
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False, cache_dir=cache / "hub")
        for raw in raw_cases:
            ids = encode_prefix(tokenizer, adapt_bulk_case(raw), torch.device("cpu"))[0].tolist()
            bound = token_rows[raw["id"]]
            if ids != bound["input_token_ids"] or not 1 <= len(ids) <= 350 or bound["max_new_tokens"] != 8:
                raise RuntimeError("OPEN source-rendered token IDs/caps differ")
        receipt["model_load_attempts"] = 1
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=True,
                    dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True,
                    device_map={"": "cuda:0"}, cache_dir=cache / "hub").eval().requires_grad_(False)
        receipt["native_model_loads"] = 1
        receipt["after_load_resources"] = resource_guard()
        code = Path(inspect.getfile(type(model))).resolve()
        if (type(model).__name__ != "Qwen3AbstainForCausalLM" or cache not in code.parents
                or sha(code) != historical["checkpoint_files_before"]["modeling_supa_a11.py"]
                or model.config.abstain_token_id != 51378 or model.config.abstain_logit_bias != 5.156249985098839):
            raise RuntimeError("OPEN exact custom head/code identity differs")
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            require_cuda_tensor(tensor, name)
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise RuntimeError("OPEN model floating tensor is notFP32")
        sys.path[:0] = [str(ARMR / "src"), str(LAB / "src")]
        from armr.provenance import cases_from, checkpoint_files, state_hash
        from armr.recurrence import FixedBlock
        from armr.hooks import Reapplication
        from armr.metrics import output_flags, logit_metrics
        from armr.reporting import summarize as smoke_gate
        from rineya_cross_scale.evaluation.scoring import score_response as old_score
        before_files, before_state = checkpoint_files(source), state_hash(model)
        if before_state != historical["state_hash_before"]:
            raise RuntimeError("OPEN loaded state differs from exact qualified FP32 state")
        receipt.update(loaded_state_sha256=before_state, loaded_custom_code_path=str(code), loaded_custom_code_sha256=sha(code))
        smoke_path = LAB / "data/evaluation" / old_config["smoke_file"]
        if sha(smoke_path) != binding["datasets"]["SMOKE"]["sha256"]:
            raise RuntimeError("inherited qualification SMOKE changed")
        smoke = cases_from(smoke_path)  # unchanged reader, only its allowlisted SMOKE
        dev = [raw for raw in raw_cases if raw["split"] == "DEVELOPMENT"]
        validation = [raw for raw in raw_cases if raw["split"] == "OPEN_VALIDATION"]
        parent_indices = {parent: index for index, parent in enumerate(sorted({raw["parent_id"] for raw in raw_cases}))}
        variant_indices = {}
        for parent in parent_indices:
            for index, variant in enumerate(sorted(raw["variant_id"] for raw in raw_cases if raw["parent_id"] == parent)):
                variant_indices[(parent, variant)] = index
        taps, synchronize = TAPSConfig(), lambda: torch.cuda.synchronize(0)
        unit = replace(taps, scheduler_mode="unit")
        arms = [("A0", unit, 0), ("A1", unit, 1)] + [("A2", replace(taps, scheduler_mode="fixed", fixed_alpha=alpha), 1) for alpha in GRID] + [("A3", taps, 1)]
        rows, pool_traces, bases, times = [], [], {}, []
        maximum_ids = torch.tensor([max(token_rows.values(), key=lambda item: len(item["input_token_ids"]))["input_token_ids"]], device="cuda:0")
        filler = tokenizer.encode(".", add_special_tokens=False)[0]
        maximum_ids = torch.cat((maximum_ids, torch.full((1, 357 - maximum_ids.shape[1]), filler, dtype=torch.long, device="cuda:0")), dim=1)
        with Journal(out / "outcomes.jsonl") as outcomes, Journal(out / "trajectories.jsonl") as trajectory, Journal(out / "forwards.jsonl") as forwards, Journal(out / "readout_pairs.jsonl") as readout_pairs, one_shot_anchors(model):
            def evaluate(case, arm, cfg, extra, seed, split, kind="new", raw=None, policy_order=None, order_index=None):
                cfg = replace(cfg, seed=seed)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                case_started = time.perf_counter()
                def sink(event):
                    event = {**event, "case_id": case["case_id"], "split": split, "seed": seed, "configured_seed": seed,
                             "arm": arm, "extra_loops": extra, "fixed_alpha": cfg.fixed_alpha if cfg.scheduler_mode == "fixed" else None}
                    trajectory.append(event)
                    if split == "DEVELOPMENT" and arm == "A3": pool_traces.append(event)
                guard = lambda: resource_guard(case_started, config["max_prompt_seconds"])
                if kind == "legacy":
                    wrapper = FixedBlock(model, [12, 13], applications=2) if extra else Reapplication(model)
                else:
                    cls = ReferenceBlock if kind == "reference" else BlockRecurrence
                    kwargs = {"arm_id": arm} if kind == "new" else {}
                    wrapper = cls(model, extra_loops=extra, config=cfg, sample_id=case["case_id"], sink=sink, guard=guard, **kwargs)
                with torch.inference_mode(), wrapper:
                    install_gpu_timing(wrapper, synchronize)
                    def record_forward(event):
                        if event.get("completed"): times.append(event["forward_seconds"])
                        forwards.append({**event, "arm": arm, "seed": seed, "split": split, "extra_loops": extra})
                    def deferred_metrics(_a, _b, _k, _gold):
                        return {"logit_cosine": None, "kl_base_to_arm": None,
                                "max_abs_logit_delta": None, "top1_agreement": None,
                                "topk_overlap": None, "margin_change": None,
                                "gold_token": None, "gold_rank": None, "baseline_gold_rank": None}
                    row, logits = evaluate_case(model, tokenizer, case, old_config if raw is None else config, wrapper,
                        old_score if raw is None else strict_score, output_flags, logit_metrics if raw is None else deferred_metrics,
                        bases.get((split, 17 if arm == "A5" else seed, case["case_id"])), synchronize=synchronize, guard=guard, forward_sink=record_forward)
                    row.update(arm=arm, seed=seed, configured_seed=seed, split=split, extra_loops=extra,
                               fixed_alpha=cfg.fixed_alpha if cfg.scheduler_mode == "fixed" else None,
                               physical_run=True, deployable=True, lane="BULK_GPU_FP32_INTERVENTION")
                    if raw is not None:
                        row.update(parent_id=raw["parent_id"], generator_family=raw["generator_family"],
                                   category=raw["category"], variant_id=raw["variant_id"], difficulty_stratum=raw["difficulty_stratum"],
                                   generator_variant=raw["generator_variant"], pair_kind=raw["pair_kind"],
                                   deterministic_policy_order=policy_order, policy_order_index=order_index,
                                   paired_readout_status="separate readout_pairs journal; no extra model forward",
                                   answer_kind=raw["answer_kind"], format_valid=bool(row["format_score"]),
                                   **bulk_diagnostics(row["response"], raw))
                    outcomes.append(row)
                    if row["safety_stopped"]: raise RuntimeError("OPEN native safety stop; retain incomplete matrix")
                return row, logits
            maximum_count = 0
            def maximum_forward():
                nonlocal maximum_count
                maximum_count += 1; begin = time.perf_counter()
                forwards.append({"event": "MAXIMUM_FORWARD_STARTED", "attempt": maximum_count, "prefix_tokens": 357,
                                 "completed": False, "task_measurement": False})
                try:
                    result = forward_logits(model, maximum_ids, synchronize=synchronize)
                except Exception as error:
                    forwards.append({"event": "MAXIMUM_FORWARD_FAILED", "attempt": maximum_count, "prefix_tokens": 357,
                                     "completed": False, "partial_compute": "UNKNOWN", "forward_seconds": time.perf_counter() - begin,
                                     "error": f"{type(error).__name__}: {error}", "resources": gpu_resources()}); raise
                elapsed = time.perf_counter() - begin; times.append(elapsed)
                forwards.append({"event": "MAXIMUM_FORWARD_COMPLETED", "attempt": maximum_count, "prefix_tokens": 357,
                                 "completed": True, "forward_seconds": elapsed, "resources": resource_guard()})
                return result
            for case in smoke:
                for extra in (0, 1):
                    old, a = evaluate(case, "qualification_legacy", unit, extra, 17, "SMOKE", "legacy")
                    new, b = evaluate(case, "qualification_new", unit, extra, 17, "SMOKE")
                    if old["token_ids"] != new["token_ids"] or not torch.equal(a, b): raise RuntimeError("OPEN GPU legacy identity failed")
                old, a = evaluate(case, "qualification_reference_A3", taps, 1, 17, "SMOKE", "reference")
                new, b = evaluate(case, "qualification_successor_A3", taps, 1, 17, "SMOKE")
                if old["token_ids"] != new["token_ids"] or not torch.equal(a, b): raise RuntimeError("OPEN original A3 identity failed")
            with torch.inference_mode():
                intact = maximum_forward()
                with BlockRecurrence(model, extra_loops=0) as w:
                    install_gpu_timing(w, synchronize); zero = maximum_forward()
                with FixedBlock(model, [12, 13], applications=2): legacy = maximum_forward()
                with BlockRecurrence(model, config=unit) as w:
                    install_gpu_timing(w, synchronize); current = maximum_forward()
                if not torch.equal(intact, zero) or not torch.equal(legacy, current): raise RuntimeError("OPEN maximum identity failed")
                with ReferenceBlock(model, config=taps) as w:
                    install_gpu_timing(w, synchronize); old = maximum_forward()
                with BlockRecurrence(model, config=taps) as w:
                    install_gpu_timing(w, synchronize); new = maximum_forward()
                if not torch.equal(old, new): raise RuntimeError("OPEN maximum A3 equivalence failed")
                def forced(_event): raise RuntimeError("EXPECTED_BULK_FORCED_EXCEPTION")
                try:
                    with BlockRecurrence(model, sink=forced) as w:
                        install_gpu_timing(w, synchronize); maximum_forward()
                except RuntimeError as error:
                    if str(error) != "EXPECTED_BULK_FORCED_EXCEPTION": raise
                else: raise RuntimeError("OPEN forced exception did not execute")
                if not torch.equal(intact, maximum_forward()): raise RuntimeError("OPEN exception restoration failed")
                for kwargs in ({"use_cache": True}, {"use_cache": False, "past_key_values": object()}):
                    forwards.append({"event": "CACHE_REJECTION_STARTED", "use_cache": kwargs["use_cache"], "cache_object_present": "past_key_values" in kwargs})
                    with BlockRecurrence(model) as w:
                        begin = time.perf_counter()
                        try: model(input_ids=maximum_ids[:, :1], **kwargs)
                        except RuntimeError as error:
                            if "cache" not in str(error): raise
                            forwards.append({"event": "CACHE_REJECTED", "model_compute_started": False, "wrapper_forward_count": w.forward_count,
                                             "completed": False, "forward_seconds": time.perf_counter() - begin})
                        else: raise RuntimeError("OPEN cache rejection failed")
            baseline_smoke = []
            for arm, cfg, extra in arms:
                current = []
                for case in smoke:
                    row, logits = evaluate(case, arm, cfg, extra, 17, "SMOKE"); current.append(row)
                    if arm == "A0": bases[("SMOKE", 17, case["case_id"])] = logits
                if arm == "A0": baseline_smoke = current
                elif not smoke_gate(current, baseline_smoke, old_config)["go_fixed_smoke"]: raise RuntimeError("OPEN required inherited smoke arm failed")
                with torch.inference_mode(), BlockRecurrence(model, extra_loops=extra, config=cfg) as w:
                    install_gpu_timing(w, synchronize); maximum_forward()
            if state_hash(model) != before_state or checkpoint_files(source) != before_files:
                raise RuntimeError("OPEN qualification changed model/source before task measurements")
            write_json(out / "qualification.json", {"status": "PASS_BULK_NEW_CAP_SAME_OPERATOR", "maximum_prefix_tokens": 357,
                       "noextra_unit_originalA3_identity": True, "forced_exception_restoration": True, "cache_rejections": 2,
                       "L0_L2_one_shot": "ALL_COMPLETED_FORWARDS", "source_unchanged": True,
                       "before_state_sha256": before_state, "after_state_sha256": before_state, "maximum_attempts": maximum_count,
                       "strict_scorer_parity": "reviewed prompt-only solver/strict scorer; model outputs not solver evidence",
                       "mask_positions": "native CUDA arange and same replay conditioning checked by frozen GPU helper",
                       "local_timing": "GPU synchronized diagnostic timing; first-pass coreL13 only; includes sync overhead",
                       "resources": resource_guard()})
            remaining_calls = 288 * (len(GRID) + 5) * 8 + 8 * 24 + 1
            warm_times = times[1:]  # first cold native forward is charged in actual elapsed
            if not warm_times: raise RuntimeError("bulk qualification lacks warm timing evidence")
            forecast = {"remaining_calls_cap": remaining_calls, "max_observed_warm_forward_seconds": max(warm_times),
                        "cold_first_forward_seconds": times[0],
                        "forecast_factor": 1.5, "remaining_forecast_seconds": remaining_calls * max(warm_times) * 1.5,
                        "timing_shape_status": "fresh max357 and all executed mechanical/SMOKE calls; cold/setup charged in elapsed",
                        "elapsed_seconds": time.perf_counter() - started, "finalization_reserve_seconds": admission.get("native_finalization_reserve_seconds", 1800),
                        "parent_deadline_seconds": admission["hard_deadline_seconds"]}
            forecast["admitted"] = forecast["remaining_forecast_seconds"] + forecast["elapsed_seconds"] + forecast["finalization_reserve_seconds"] <= forecast["parent_deadline_seconds"]
            write_json(out / "matrix_forecast.json", forecast)
            if not forecast["admitted"]: raise RuntimeError("complete OPEN matrix forecast does not fit registered deadline/reserve")
            arm_lookup = {(f"A2:{cfg.fixed_alpha:g}" if arm == "A2" else arm): (arm, cfg, extra) for arm, cfg, extra in arms}
            def deterministic_case(raw, split):
                order = deterministic_order(raw, parent_indices, variant_indices)
                first_logits = {}
                for position, label in enumerate(order):
                    arm, cfg, extra = arm_lookup[label]
                    row, logits = evaluate(adapt_bulk_case(raw), arm, cfg, extra, 17, split, raw=raw,
                                           policy_order=order, order_index=position)
                    rows.append(row); first_logits[label] = logits
                    if arm == "A0": bases[(split, 17, raw["id"])] = logits
                    if split == "OPEN_VALIDATION" and arm == "A2" and cfg.fixed_alpha == freeze["alpha"]:
                        outcomes.append({**row, "arm": "A2_frozen", "physical_run": False, "note": "pre-frozen DEV comparator reuses this exact grid rollout"})
                for label, logits in first_logits.items():
                    readout_pairs.append({"case_id": raw["id"], "split": split, "policy": label,
                                          "physical_run": False, "extra_forward_calls": 0,
                                          **logit_metrics(first_logits["A0"], logits, config["top_k"], None)})
            for raw in dev:
                deterministic_case(raw, "DEVELOPMENT")
            freeze = freeze_development(rows, pool_traces, admission["bulk_bank_sha256"], sha(token_path), sha(__file__), [raw["id"] for raw in dev])
            freeze["generation_trajectory_sha256_at_freeze"] = sha(out / "trajectories.jsonl")
            # Rebind the complete immutable freeze, including raw trajectory digest.
            freeze["freeze_sha256"] = digest({key: value for key, value in freeze.items() if key != "freeze_sha256"})
            write_json(out / "frozen_development.json", freeze)
            random_cfg = replace(taps, scheduler_mode="random", random_pool=tuple(freeze["pool"]))
            a5_smoke = [evaluate(case, "A5", random_cfg, 1, 17, "SMOKE")[0] for case in smoke]
            if not smoke_gate(a5_smoke, baseline_smoke, old_config)["go_fixed_smoke"]: raise RuntimeError("OPEN frozen pool A5 inherited smoke failed")
            with torch.inference_mode(), BlockRecurrence(model, config=random_cfg, arm_id="maximum_A5") as w:
                install_gpu_timing(w, synchronize); maximum_forward()
            write_json(out / "A5_qualification.json", {"status": "PASS_FROZEN_BULK_DEV_POOL", "actual_seed_binding": True})
            for seed in SEEDS:
                for raw in dev:
                    row, logits = evaluate(adapt_bulk_case(raw), "A5", random_cfg, 1, seed, "DEVELOPMENT", raw=raw); rows.append(row)
                    readout_pairs.append({"case_id": raw["id"], "split": "DEVELOPMENT", "policy": "A5", "seed": seed,
                                          "physical_run": False, "extra_forward_calls": 0,
                                          **logit_metrics(bases[("DEVELOPMENT", 17, raw["id"])], logits, config["top_k"], None)})
            for raw in validation:
                admit_validation(freeze, raw, admission["bulk_bank_sha256"], sha(token_path), sha(__file__))
                deterministic_case(raw, "OPEN_VALIDATION")
            for seed in SEEDS:
                for raw in validation:
                    admit_validation(freeze, raw, admission["bulk_bank_sha256"], sha(token_path), sha(__file__))
                    row, logits = evaluate(adapt_bulk_case(raw), "A5", random_cfg, 1, seed, "OPEN_VALIDATION", raw=raw)
                    rows.append(row)
                    readout_pairs.append({"case_id": raw["id"], "split": "OPEN_VALIDATION", "policy": "A5", "seed": seed,
                                          "physical_run": False, "extra_forward_calls": 0,
                                          **logit_metrics(bases[("OPEN_VALIDATION", 17, raw["id"])], logits, config["top_k"], None)})
            for split, cases in (("DEVELOPMENT", dev), ("OPEN_VALIDATION", validation)):
                for raw in cases:
                    seed = 17
                    fixed = [row for row in rows if row["split"] == split and row["seed"] == seed and row["case_id"] == raw["id"] and row["arm"] == "A2"]
                    selected = max(fixed, key=lambda row: (row["score"], -row["fixed_alpha"]))
                    outcomes.append({**selected, "arm": "A4_example", "physical_run": False, "deployable": False,
                                     "note": "retrospective per-case constant finite-grid diagnostic; no validation reselection"})
        synchronize()
        if state_hash(model) != before_state or checkpoint_files(source) != before_files or sha(code) != historical["checkpoint_files_before"]["modeling_supa_a11.py"]:
            raise RuntimeError("OPEN final model/source/custom code changed")
        receipt.update(status="BULK_MATRIX_COMPLETE_NOT_CONFIRMATORY", source_unchanged=True, checkpoint_files_unchanged=True,
                       trainable_parameters=0, bank_sha256=admission["bulk_bank_sha256"], frozen_alpha=freeze["alpha"],
                       DEVELOPMENT_cases=192, OPEN_VALIDATION_cases=96, physical_case_evaluations=3296, final_resources=resource_guard())
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt.update(wall_seconds=time.perf_counter() - started, ended_utc=datetime.now(timezone.utc).isoformat(),
                       final_resources_on_stop=final_resource_snapshot())
        write_json(out / "receipt.json", receipt)
    return 0 if receipt["status"] == "BULK_MATRIX_COMPLETE_NOT_CONFIRMATORY" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.out, args.admission))
