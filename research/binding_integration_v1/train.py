"""One supervised BINDING-INTEGRATION-001 training attempt using the pinned native primitives.

The public controller is torch-free. Production has no partial-group, rank, LR,
resume, or guard-override switches. Checkpoints describe this new LoRA experiment;
the mandatory native source qualification is separately preserved without claiming
that it qualifies the adapter. No in-run memorization generations are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
PILOT = Path("C:/Users/clclc/BusinessProject/ChatBubble-binding-integration-pilot-20260920")
PILOT_HEAD = "5b4c22dae78f587c41575269f419087b0c7997f6"
NATIVE_INPUT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/doseint001-20260919/prepared-dual899/microfit-train.jsonl")
BINDING_INPUT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-signal-prep-20260920/candidate-v2/compiled-v1/TRAIN_COMPILED.jsonl")
CONTROL_ROOT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/highdose-lora001-resume-20260920")
NATIVE_PLAN = CONTROL_ROOT / "PLAN.json"
PRIOR_MIXTURE_MEMBERSHIP = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-mixture001-20260920/prepared/SOURCE_MEMBERSHIP.json")
REFERENCE_CHECKPOINT = CONTROL_ROOT / "journal/runs/train-001/checkpoint-00002"
REFERENCE_MANIFEST_SHA256 = "afc09301260af14ea033f42e4ccb006982f8a36f7b31db4807d521e7399c5376"
DATA_FILES = ("microfit-train.jsonl", "INTEGRATION_GROUPS.json", "SOURCE_MEMBERSHIP.json", "DATA_BUILD.json")
sys.path.insert(0, str(PILOT))
from experiments.model_quality_gpu_v1.config import Design, rate_at
from experiments.model_quality_gpu_v1.integrity import (canonical, code_identity, digest, file_hash,
                                                       new_dir, read_json, seal, write_new)
from experiments.model_quality_gpu_v1.resources import GIB, PRELOAD_MINIMA, guard, lease
from experiments.model_quality_gpu_v1.supervisor import bounded_child, claim_worker
from experiments.reasoning_posttraining_v1.budget import Journal, register, validate_grant
from experiments.reasoning_posttraining_v1.controller import checkout, package_identity
from experiments.model_quality_gpu_v1.data import group_updates

EXPERIMENT = "BINDING-INTEGRATION-001"
PEAK_LR = 1e-4
TIMEOUT = 2400
DISK_FLOOR = 14 * GIB
TRAINABLE_PARAMETERS = 7_255_040
TRAINABLE_TENSORS = 360
NATIVE_PROCESSED_TOKENS = 380921
ZERO_PREFIX_UPDATES = 2
ZERO_PREFIX_TOKENS = 4032
ZERO_PREFIX_ROWS = 28
EXPECTED_CHARGED_TOKENS = 590245
EXPECTED_QUALIFICATION_CALLS = 76
CHECKPOINT_BYTES = TRAINABLE_PARAMETERS * 12 + 2**20
PARAMETERIZATION = {"rank": 16, "alpha": 32, "dropout": 0.0, "bias": "none",
                    "layers": list(range(8, 28)), "train_norms": True,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
LIMITS = {"total_owned_work_wall_seconds": 2700,
          "model_worker_wall_seconds_and_conservative_gpu_seconds": 2500,
          "processed_training_tokens_all_attempts": 600000,
          "evaluation_and_qualification_model_calls": 128,
          "generated_tokens_all_candidates": 0, "on_policy_rl_rollout_tokens": 0,
          "new_disk_bytes": 256 * 2**20, "adaptive_model_selection_rounds": 1,
          "model_worker_launches": 1, "retries": 0, "cloud_or_teacher_spend": 0}
EXCLUDES = ("RL", "larger_teacher", "cloud", "release", "architectural_change")
register(EXPERIMENT, ceiling=LIMITS, phases=("train",), excludes=EXCLUDES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def execution_identity():
    for name in ("experiments.model_quality_gpu_v1.config", "experiments.model_quality_gpu_v1.integrity",
                 "experiments.model_quality_gpu_v1.supervisor", "experiments.reasoning_posttraining_v1.controller"):
        require(Path(sys.modules[name].__file__).resolve().is_relative_to(PILOT.resolve()), "native primitive imported from another checkout")
    require(checkout() == PILOT_HEAD, "pinned native pilot checkout changed")
    files = {name: file_hash(HERE / name) for name in ("train.py", "adapters.py")}
    return {"runner_files": files, "runner_code_sha256": digest(files),
            "native_package_sha256": package_identity()["sha256"],
            "mq_code_sha256": code_identity()["sha256"], "pilot_checkout": PILOT_HEAD}


def expected_membership(native_groups, prior_membership):
    """Native groups stay intact; floor-partition the already frozen binding order."""
    prior_native = {row["source_index"]: row["mixture_index"] for row in prior_membership if row["source"] == "native_dual899"}
    prior_binding = [(row["source_index"], row["mixture_index"]) for row in prior_membership if row["source"] == "binding_v2"]
    require(native_groups and prior_binding, "both integration components are required")
    result = []
    for group, native in enumerate(native_groups):
        result.extend(("native_dual899", index, prior_native[index], group) for index in native)
        lo = group * len(prior_binding) // len(native_groups)
        hi = (group + 1) * len(prior_binding) // len(native_groups)
        result.extend(("binding_v2", index, previous, group) for index, previous in prior_binding[lo:hi])
    return result


def validate_reference_metadata(plan):
    """Pin the previously observed two-step control; no tensor/model loading."""
    value = plan["zero_addend_reference"]
    require(set(value) == {"checkpoint_path", "manifest_sha256", "tensor_files_sha256"}, "zero-addend reference fields differ")
    directory = Path(value["checkpoint_path"]).resolve()
    require(directory == REFERENCE_CHECKPOINT.resolve() and value["manifest_sha256"] == REFERENCE_MANIFEST_SHA256 and
            file_hash(directory / "manifest.json") == value["manifest_sha256"], "fixed two-update native reference changed")
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("sha256") == digest({key: item for key, item in manifest.items() if key != "sha256"}) and
            manifest["record"] == "DOSELORA_TRAINING_CHECKPOINT" and manifest["experiment"] == "DOSELORA-001" and
            manifest["status"] == "COMPLETE" and manifest["progress"] ==
            {"updates": 2, "next_group": 2, "processed_tokens": 4032, "supervised_tokens": 388}, "reference is not the complete native two-update checkpoint")
    entries = manifest["entries"]
    require(len(entries) == TRAINABLE_TENSORS and len({row["name"] for row in entries}) == TRAINABLE_TENSORS and
            [row["file"] for row in entries] == [f"tensor-{index:04d}.safetensors" for index in range(TRAINABLE_TENSORS)] and
            value["tensor_files_sha256"] == {row["file"]: row["sha256"] for row in entries}, "reference trainable file/name inventory differs")
    require(manifest["binding"]["plan_sha256"] == digest(read_json(NATIVE_PLAN)), "reference control plan identity changed")
    return manifest


def validate_prepared(prepared, plan):
    """Validate unchanged views and native-first execution at each original update."""
    from collections import Counter
    prepared = Path(prepared).absolute()
    contract = read_json(prepared / "DATA_CONTRACT.json")
    require(plan["experiment"] == contract["experiment"] == EXPERIMENT, "fresh integration plan/contract required")
    require(set(contract["files"]) == set(DATA_FILES), "integration contract must bind all four preparation artifacts")
    for name, expected in contract["files"].items():
        require(file_hash(prepared / name) == expected, "prepared mixture artifact changed: " + name)
    build = read_json(prepared / "DATA_BUILD.json")
    require(build["record"] == "BINDING_INTEGRATION_DATA_BUILD" and build["status"] == "PASS_DATA_PREPARATION_ONLY",
            "mixture was not admitted by its independent data builder")
    require(build["rows"] == build["presentations"] == 3465 and build["unique_worlds"] == build["parent_worlds"] == 2761,
            "views and parent worlds must remain distinct inventories")
    require(build["exposures_per_view"] == 1 and build["processed_tokens"] == 577417
            and build["supervised_tokens"] == 39502 and build["update_groups_count"] == 193,
            "frozen integration token/update inventory changed")
    require(build["rows_copied_byte_identically"] is True and build["source_rows_copied_byte_identically"] is True and
            build["labels_modified"] is False and build["source_rows_repeated_or_dropped"] is False and
            build["native_grouping_denominators_and_lr_clock_preserved"] is True and build["prior_binding_subsequence_preserved"] is True,
            "integration construction changed source examples or native coefficients")
    require(set(build["output_files_sha256"]) == set(DATA_FILES) - {"DATA_BUILD.json"}, "integration build output inventory differs")
    for name, expected in build["output_files_sha256"].items():
        require(name in DATA_FILES and file_hash(prepared / name) == expected, "data-build output identity differs")
    for path, expected in build["input_files_sha256"].items():
        require(file_hash(Path(path)) == expected, "data-build source/admission dependency changed")
    raw_sources = {}
    for label, path, expected_count in (("native_dual899", NATIVE_INPUT, 2697), ("binding_v2", BINDING_INPUT, 768)):
        require(build["input_files_sha256"].get(str(path)) == file_hash(path), "fixed source corpus is not bound by DATA_BUILD")
        lines = path.read_bytes().splitlines(keepends=True)
        require(len(lines) == expected_count and all(line.strip() and line.endswith(b"\n") for line in lines), "source JSONL rows differ")
        raw_sources[label] = lines
    for path in (NATIVE_PLAN, PRIOR_MIXTURE_MEMBERSHIP):
        require(build["input_files_sha256"].get(str(path)) == file_hash(path), "native grouping/binding-order lineage missing")
    original_plan = read_json(NATIVE_PLAN)
    original_rows = [json.loads(line) for line in raw_sources["native_dual899"]]
    original_groups = group_updates(original_rows, 2048)
    require(original_groups == original_plan["update_groups"] and len(original_groups) == 193 and
            [row["id"] for row in original_rows] == original_plan["row_order_per_exposure"], "original native groups/order changed")
    prior_membership = read_json(PRIOR_MIXTURE_MEMBERSHIP)
    require(len(prior_membership) == 3465 and sum(row["source"] == "binding_v2" for row in prior_membership) == 768,
            "prior frozen binding subsequence inventory differs")
    membership_order = expected_membership(original_groups, prior_membership)
    lines = (prepared / "microfit-train.jsonl").read_bytes().splitlines(keepends=True)
    membership = read_json(prepared / "SOURCE_MEMBERSHIP.json")
    require(len(lines) == len(membership) == 3465, "mixture line/membership count differs")
    rows, ids, parent_counts, binding_parents = [], [], Counter(), Counter()
    for index, (line, record, expected) in enumerate(zip(lines, membership, membership_order)):
        label, source_index, previous, group_index = expected
        require(line == raw_sources[label][source_index], "mixture row bytes or deterministic insertion changed")
        row = json.loads(line)
        require(record == {"execution_index": index, "id": row["id"], "parent_id": row["parent_id"],
                           "source": label, "source_index": source_index,
                           "previous_mixture_index": previous, "optimizer_group_index": group_index,
                           "source_row_sha256_including_newline": hashlib.sha256(line).hexdigest()},
                "source membership does not describe actual row bytes")
        require(row["source_split"] == "train" and row["mode"] == "direct" and row["truncated"] is False,
                "mixture must contain only unchanged untruncated TRAIN views")
        tokens, labels, prompt = row["ids"], row["labels"], row["prompt_ids"]
        end = row["answer_end"]
        require(len(tokens) == len(labels) == row["processed_tokens"] <= 2048 and labels[0] == -100
                and tokens[:len(prompt)] == prompt and row["answer_start"] == len(prompt)
                and len(prompt) <= end < len(tokens) and tokens[end] == 151645
                and labels == [token if len(prompt) <= i <= end else -100 for i, token in enumerate(tokens)]
                and sum(token != -100 for token in labels[1:]) == row["supervised_tokens"],
                "response-plus-EOS masking differs from source rows")
        rows.append(row); ids.append(row["id"]); parent_counts[row["parent_id"]] += 1
        if label == "binding_v2":
            binding_parents[row["parent_id"]] += 1
    require(len(set(ids)) == 3465 and len(parent_counts) == 2761 and len(binding_parents) == 64
            and set(binding_parents.values()) == {12} and Counter(parent_counts.values()) == {1: 2697, 12: 64},
            "mixture views/parent grouping differs from the registered 2697 singleton plus 64 twelve-view worlds")
    native_groups = [[index for index, value in enumerate(membership_order) if value[3] == group and value[0] == "native_dual899"] for group in range(193)]
    binding_groups = [[index for index, value in enumerate(membership_order) if value[3] == group and value[0] == "binding_v2"] for group in range(193)]
    groups = [native + added for native, added in zip(native_groups, binding_groups)]
    group_record = read_json(prepared / "INTEGRATION_GROUPS.json")
    require(groups == group_record["combined_update_groups"] == plan["update_groups"]
            and native_groups == group_record["native_update_groups"] == plan["native_update_groups"]
            and binding_groups == group_record["binding_update_groups"] == plan["binding_update_groups"]
            and original_groups == group_record["original_native_update_groups"]
            and ids == group_record["row_order_per_exposure"] == plan["row_order_per_exposure"]
            and digest(ids) == group_record["row_order_sha256"] == build["row_order_sha256"], "integration order/update groups differ")
    require([index for group in groups for index in group] == list(range(3465)), "integration groups must partition actual execution order")
    denominators, cumulative, seen = [], [], 0
    design = Design(processed_budget=NATIVE_PROCESSED_TOKENS)
    for group in native_groups:
        tokens = sum(len(rows[index]["ids"]) for index in group)
        require(0 < tokens <= 2048, "an original native group exceeded 2048 tokens")
        denominators.append(sum(sum(token != -100 for token in rows[index]["labels"][1:]) for index in group))
        seen += tokens; cumulative.append(seen)
    rates = [rate_at(PEAK_LR, count, NATIVE_PROCESSED_TOKENS, design) for count in cumulative]
    require(seen == NATIVE_PROCESSED_TOKENS and sum(denominators) == 37090 and all(value > 0 for value in denominators) and
            denominators == group_record["native_denominators"] == plan["native_denominators"] and
            cumulative == group_record["native_processed_cumulative"] == plan["native_processed_cumulative"] and
            rates == group_record["applied_native_learning_rates"] == plan["applied_native_learning_rates"], "native coefficient/LR clock changed")
    require(sum(len(row["ids"]) for row in rows) == 577417
            and sum(sum(token != -100 for token in row["labels"][1:]) for row in rows) == 39502
            and len(rows[0]["ids"]) == 116, "mixed training or qualification token geometry changed")
    for exclusion in build["acquisition_parent_exclusions"]:
        path = Path(exclusion["path"])
        require(file_hash(path) == exclusion["sha256"], "acquisition exclusion bank changed")
        parents = {json.loads(line)["parent_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        require(len(parents) == 384 and not set(parent_counts) & parents and exclusion["mixture_parent_overlap"] == 0,
                "mixed corpus overlaps a declared acquisition parent")
    require(build["zero_addend_reference"] == plan["zero_addend_reference"], "plan reference differs from prepared qualification lineage")
    validate_reference_metadata(plan)
    return rows


def inputs(args, *, retain=False):
    prepared = Path(args.prepared)
    plan, grant = read_json(Path(args.plan)), read_json(Path(args.grant))
    require(plan["objective"] == "assistant_response_plus_eos" and plan["precision"] == "fp32"
            and plan["trainable"] == "blocks8-27_lora16_projections_and_norms", "unregistered mixture training scope")
    require(plan["processed_tokens"] == 577417 and plan["supervised_tokens"] == 39502
            and plan["presentations"] == 3465 and plan["unique_worlds"] == 2761
            and plan["exposures_per_view"] == 1 and len(plan["update_groups"]) == 193 and
            plan["native_processed_tokens"] == NATIVE_PROCESSED_TOKENS,
            "mixture plan must separately count views, worlds and one exposure per view")
    require(plan["selection_seed"] == plan["order_seed"] == 17 and plan["max_sequence"] == 2048,
            "frozen mixture seed/window changed")
    require(all(plan.get("parameterization", {}).get(key) == value for key, value in PARAMETERIZATION.items()),
            "mixture parameterization differs from the qualified rank16 tail20 recipe")
    design = Design(processed_budget=NATIVE_PROCESSED_TOKENS)
    require(plan["optimizer"] == {"betas": [design.beta1, design.beta2], "epsilon": design.epsilon,
            "gradient_clip": design.clip_norm, "name": "MQ CpuAdamW", "peak_lr": PEAK_LR,
            "schedule": "WSD20/60/20_to_0.1_peak", "weight_decay": design.weight_decay}, "mixture optimizer recipe changed")
    validate_grant(grant, file_hash(prepared / "DATA_CONTRACT.json"), digest(plan), EXPERIMENT)
    require(grant["limits"]["model_worker_launches"] == 1 and grant["limits"]["retries"] == 0,
            "one fixed mixture launch and no optimizer retry are registered")
    rows = validate_prepared(prepared, plan)
    # There is one pass over 3465 views, not one presentation per parent world.
    return plan, grant, design, rows if retain else [], rows if retain else []


def load_adapters():
    spec = importlib.util.spec_from_file_location("doselora_adapters", HERE / "adapters.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def progress_at(rows, groups, count):
    indices = [index for group in groups[:count] for index in group]
    return {"updates": count, "next_group": count,
            "processed_tokens": sum(len(rows[index]["ids"]) for index in indices),
            "supervised_tokens": sum(sum(token != -100 for token in rows[index]["labels"][1:]) for index in indices)}


def save_state(directory, optimizer, binding, progress, adapter_spec, zero_addend_sha256):
    """Same safe storage primitive, explicitly a binding-integration checkpoint."""
    import torch
    from safetensors.torch import save_file
    from experiments.model_quality_gpu_v1.checkpoints import rng_state
    require(progress["updates"] == optimizer.step_count and all(p.grad is None for _, p in optimizer.named),
            "checkpoint requires a cleared complete optimizer boundary")
    require(isinstance(zero_addend_sha256, str) and len(zero_addend_sha256) == 64, "active checkpoint requires zero-addend qualification identity")
    directory = new_dir(directory)
    entries = []
    for index, (name, parameter) in enumerate(optimizer.named):
        first, second = optimizer.state[name]
        state = {"parameter": parameter.detach().cpu().contiguous(), "exp_avg": first.contiguous(), "exp_avg_sq": second.contiguous()}
        require(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in state.values())
                and not bool((second < 0).any()), "invalid checkpoint parameter or moments")
        filename = f"tensor-{index:04d}.safetensors"
        path = directory / filename
        save_file(state, str(path))
        with path.open("rb+") as stream:
            os.fsync(stream.fileno())
        entries.append({"name": name, "file": filename, "shape": list(parameter.shape), "sha256": file_hash(path)})
    manifest = seal({"schema": 1, "record": "BINDING_INTEGRATION_TRAINING_CHECKPOINT", "experiment": EXPERIMENT,
                     "status": "COMPLETE", "binding": binding, "progress": progress,
                     "zero_addend_qualification_sha256": zero_addend_sha256,
                     "adapter_spec": adapter_spec.as_dict(), "rng": rng_state(), "entries": entries})
    write_new(directory / "manifest.json", manifest)
    return file_hash(directory / "manifest.json")


def qualify_and_inject(model, tokenizer, reserve, resource_guard, adapter_module):
    """Actual source parity at zero B, then separate adapter gradient qualification."""
    import torch
    from experiments.model_quality_gpu_v1.checkpoints import rng_state, restore_rng
    from experiments.model_quality_gpu_v1.losses import chunked_sum
    from experiments.model_quality_gpu_v1.native import hidden_states, parameter_digest
    from experiments.reasoning_posttraining_v1.serialization import compile_row
    prompt = [{"role": "system", "content": "Answer with the requested number."},
              {"role": "user", "content": "Reply with 7."}]
    probe = compile_row(prompt, "7", tokenizer)
    tokens = len(probe["ids"])
    require(tokens == 31, "registered source qualification token count changed")
    charges = {"processed_training_tokens_all_attempts": 5 * tokens + 2048,
               "evaluation_and_qualification_model_calls": 2 * ((tokens + 2) // 3) + 4}
    reserve(charges)
    device = next(model.parameters()).device
    ids = torch.tensor([probe["ids"]], device=device)
    labels = torch.tensor([probe["labels"]], device=device)
    model.eval(); resource_guard()
    source_digest = parameter_digest(model)
    with torch.no_grad():
        source_logits = model(input_ids=ids, use_cache=False, return_dict=True).logits.detach()
        source_pieces, source_past = [], None
        for start in range(0, tokens, 3):
            resource_guard()
            source_part = model(input_ids=ids[:, start:start + 3], past_key_values=source_past, use_cache=True, return_dict=True)
            source_pieces.append(source_part.logits); source_past = source_part.past_key_values
        source_cached = torch.cat(source_pieces, dim=1)
        del source_pieces, source_past, source_part
    named, spec = adapter_module.inject_lora(model)
    require(spec.production and spec.trainable_parameters == TRAINABLE_PARAMETERS and len(named) == TRAINABLE_TENSORS,
            "actual adapter scope differs from registered inventory")
    require(all(parameter.dtype == torch.float32 for _, parameter in named), "adapter/norm state must be FP32")
    frozen = adapter_module.frozen_base_digest(model, spec)
    from experiments.model_quality_gpu_v1.integrity import canonical as encode
    # Preserve the post-initialization RNG; qualification may not choose the training trajectory.
    preserved_rng = rng_state()
    gradient_records = []
    try:
        model.eval(); resource_guard()
        with torch.no_grad():
            direct = model(input_ids=ids, use_cache=False, return_dict=True).logits
            torch.testing.assert_close(direct, source_logits, atol=0, rtol=0)
            pieces, past = [], None
            for start in range(0, tokens, 3):
                resource_guard()
                part = model(input_ids=ids[:, start:start + 3], past_key_values=past, use_cache=True, return_dict=True)
                pieces.append(part.logits); past = part.past_key_values
            cached = torch.cat(pieces, dim=1)
            torch.testing.assert_close(cached, source_cached, atol=0, rtol=0)
            torch.testing.assert_close(direct, cached, atol=2e-4, rtol=2e-4)
            cache_error = float((direct - cached).abs().max())
        del direct, source_logits, source_cached, cached, pieces, past, part
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Hash every trainable value as well as frozen values to prove zero optimizer updates.
        def trainable_digest():
            import hashlib
            hashed = hashlib.sha256()
            for name, parameter in named:
                hashed.update(encode([name, list(parameter.shape), str(parameter.dtype)]))
                flat = parameter.detach().reshape(-1)
                for start in range(0, flat.numel(), 262144):
                    hashed.update(flat[start:start + 262144].cpu().contiguous().view(torch.uint8).numpy())
            return hashed.hexdigest()
        before_trainable = trainable_digest()
        for label, input_ids, targets in (("native_chat", ids, labels), ("synthetic_2048_shape", None, None)):
            if input_ids is None:
                ordinary = tokenizer.encode("x y", add_special_tokens=False)
                require(bool(ordinary) and tokenizer.eos_token_id not in ordinary, "invalid shape-fixture tokenization")
                sequence = (ordinary * (2048 // len(ordinary) + 1))[:2048]
                sequence[-1] = tokenizer.eos_token_id
                input_ids = torch.tensor([sequence], device=device)
                targets = torch.full_like(input_ids, -100); targets[0, -2:] = input_ids[0, -2:]
            model.train(); model.zero_grad(set_to_none=True); resource_guard()
            loss, denominator = chunked_sum(hidden_states(model, input_ids), model.lm_head.weight, targets, 64)
            require(bool(torch.isfinite(loss)) and denominator > 0, "invalid adapter qualification loss")
            (loss / denominator).backward()
            gradients = adapter_module.gradient_summary(model, spec, require_active=label == "native_chat")
            require(gradients["finite_tensors"] == TRAINABLE_TENSORS, "adapter qualification has missing/nonfinite gradients")
            gradient_records.append({"probe": label, "tokens": input_ids.shape[1], **gradients})
            model.zero_grad(set_to_none=True)
            del loss
        require(trainable_digest() == before_trainable, "zero-update qualification changed adapter/norm values")
        require(adapter_module.frozen_base_digest(model, spec) == frozen, "adapter qualification changed frozen source values")
    finally:
        model.zero_grad(set_to_none=True); model.eval(); restore_rng(preserved_rng)
    return named, spec, {"record": "BINDING_INTEGRATION_ZERO_UPDATE_QUALIFICATION", "experiment": EXPERIMENT,
            "status": "PASS", "optimizer_steps": 0, "source_state_before_injection": source_digest,
            "frozen_base_sha256": frozen, "trainable_sha256_before_after": before_trainable,
            "zero_B_source_logits_exact": True, "zero_B_source_cached_logits_exact": True, "cached_max_abs": cache_error,
            "A_zero_gradient_expected_at_zero_B": True, "gradient_probes": gradient_records,
            "budget_charges": charges, "rng_restored_after_qualification": True,
            "adapter_spec": spec.as_dict(), "quality_claim": False}


def _tensor_sha(name, component, tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "qualification state must be finite FP32")
    hashed = hashlib.sha256(canonical([name, component, list(value.shape), str(value.dtype)]))
    hashed.update(value.reshape(-1).view(torch.uint8).numpy())
    return hashed.hexdigest()


def _same_bits(left, right):
    import torch
    return (left.dtype == right.dtype and left.shape == right.shape and
            torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                        right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def _trainable_sha(named):
    hashed = hashlib.sha256()
    for name, parameter in named:
        hashed.update(bytes.fromhex(_tensor_sha(name, "parameter", parameter)))
    return hashed.hexdigest()


def _apply_update(model, rows, native_group, binding_group, native_denominator, learning_rate,
                  optimizer, design, loss_sum, *, binding_enabled):
    """Private numerical core: disabled means no binding callback or RNG use.

    Production active training always passes True. Only the mandatory zero-addend
    prefix qualification and tiny CPU fixtures may pass False.
    """
    import torch
    require(type(binding_enabled) is bool and native_group and type(native_denominator) is int and native_denominator > 0,
            "invalid native-normalized update")
    model.train(); optimizer.zero_grad()
    objectives = {"native": 0.0, "binding": 0.0}; counts = {"native": 0, "binding": 0}
    components = (("native", native_group), ("binding", binding_group)) if binding_enabled else (("native", native_group),)
    for component, group in components:
        for index in group:
            loss = loss_sum(model, rows[index])
            require(bool(torch.isfinite(loss)), "nonfinite integration loss")
            (loss / native_denominator).backward()
            objectives[component] += float(loss.detach()) / native_denominator
            counts[component] += 1
    require(all(parameter.grad is None for _, parameter in model.named_parameters() if not parameter.requires_grad),
            "a frozen parameter acquired a gradient")
    norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in optimizer.named], design.clip_norm, error_if_nonfinite=True)
    optimizer.step(learning_rate); optimizer.zero_grad()
    return {"objective_ce": objectives["native"] + objectives["binding"],
            "native_objective_ce": objectives["native"], "binding_objective_native_normalized": objectives["binding"],
            "native_forward_calls": counts["native"], "binding_forward_calls": counts["binding"], "grad_norm": float(norm)}


def _disposable_prefix(model, named, optimizer_factory, step, verify, capture_rng, restore_rng):
    """Run exactly two disposable updates, always restore initial values and RNG."""
    import torch
    named = list(named)
    require(named and len({name for name, _ in named}) == len(named), "unique trainable names required")
    initial = {name: parameter.detach().cpu().clone() for name, parameter in named}
    rng_before = capture_rng(); before_sha = _trainable_sha(named); was_training = model.training
    optimizer = None; failure = None; compared = {}; steps = []; completed = 0
    try:
        optimizer = optimizer_factory()
        require(optimizer.step_count == 0 and all(not bool(t.any()) for state in optimizer.state.values() for t in state),
                "qualification Adam must begin with zero moments")
        for index in range(ZERO_PREFIX_UPDATES):
            steps.append(step(optimizer, index)); completed = optimizer.step_count
        compared = verify(optimizer)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if optimizer is not None:
            optimizer.zero_grad()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(initial[name])
        model.train(was_training)
        restore_rng(rng_before)
        restored = all(_same_bits(parameter, initial[name]) for name, parameter in named)
        after_sha = _trainable_sha(named); rng_after = capture_rng()
        # No optimizer or moment tensors escape this scope. Active training
        # constructs a fresh CpuAdamW only after this function has returned.
        del optimizer
    return {"failure": failure, "optimizer_updates": completed, "optimizer_step_count": completed,
            "steps": steps, "comparison": compared, "initial_trainable_sha256": before_sha,
            "restored_trainable_sha256": after_sha, "trainable_state_restored_exactly": restored and before_sha == after_sha,
            "rng_before_sha256": digest(rng_before), "rng_after_restore_sha256": digest(rng_after),
            "rng_restored_exactly": canonical(rng_before) == canonical(rng_after),
            "qualification_optimizer_discarded": True, "initial_model_training_mode_restored": model.training == was_training}


def _compare_reference_state(optimizer, manifest, reference_dir):
    """Stream the reference adapter state; never deserialize pickle or copy it in."""
    import torch
    from safetensors.torch import load_file
    require([entry["name"] for entry in manifest["entries"]] == [name for name, _ in optimizer.named],
            "reference trainable-name order differs")
    checks = []
    for entry, (name, parameter) in zip(manifest["entries"], optimizer.named, strict=True):
        path = reference_dir / entry["file"]
        require(path.parent == reference_dir and file_hash(path) == entry["sha256"] and entry["shape"] == list(parameter.shape),
                "reference tensor file/shape changed")
        expected = load_file(str(path), device="cpu")
        require(set(expected) == {"parameter", "exp_avg", "exp_avg_sq"} and
                all(t.dtype == torch.float32 and t.shape == parameter.shape and bool(torch.isfinite(t).all()) for t in expected.values()) and
                not bool((expected["exp_avg_sq"] < 0).any()), "invalid reference FP32 optimizer state")
        first, second = optimizer.state[name]
        actual = {"parameter": parameter, "exp_avg": first, "exp_avg_sq": second}
        record = {"name": name, "shape": list(parameter.shape), "reference_file": entry["file"], "reference_file_sha256": entry["sha256"]}
        for component in actual:
            record[component] = {"equal": _same_bits(actual[component], expected[component]),
                                 "actual_sha256": _tensor_sha(name, component, actual[component]),
                                 "expected_sha256": _tensor_sha(name, component, expected[component])}
        require(file_hash(path) == entry["sha256"], "reference tensor changed during comparison")
        checks.append(record)
        del expected, actual, first, second
    return checks


def _native_loss_sum(model, row, design):
    import torch
    from experiments.model_quality_gpu_v1.losses import chunked_sum
    from experiments.model_quality_gpu_v1.native import hidden_states
    ids = torch.tensor([row["ids"]], device="cuda:0", dtype=torch.long)
    labels = torch.tensor([row["labels"]], device="cuda:0", dtype=torch.long)
    loss, _ = chunked_sum(hidden_states(model, ids), model.lm_head.weight, labels, design.ce_chunk)
    return loss


def qualify_zero_addend(model, rows, plan, design, named, spec, binding, reserve, adapter_module):
    """Measured exactness covers only the original first two native updates."""
    from experiments.model_quality_gpu_v1.checkpoints import rng_state, restore_rng
    from experiments.model_quality_gpu_v1.optimizer import CpuAdamW
    manifest = validate_reference_metadata(plan)
    reference = Path(plan["zero_addend_reference"]["checkpoint_path"]).resolve()
    require(spec.production and len(named) == TRAINABLE_TENSORS and manifest["adapter_spec"] == spec.as_dict(), "reference adapter scope differs")
    require(manifest["binding"]["source"] == binding["source"] and manifest["binding"]["runtime"] == binding["runtime"] and
            manifest["binding"]["design"] == design.as_dict() and manifest["binding"]["parameterization"] == plan["parameterization"],
            "reference source/runtime/design/parameterization differs")
    require({path.name for path in reference.iterdir()} == {"manifest.json", *plan["zero_addend_reference"]["tensor_files_sha256"]},
            "reference checkpoint has extra or missing files")
    for name, expected in plan["zero_addend_reference"]["tensor_files_sha256"].items():
        require(Path(name).name == name and file_hash(reference / name) == expected, "reference tensor bytes changed before qualification")
    groups = plan["native_update_groups"][:ZERO_PREFIX_UPDATES]
    indices = [index for group in groups for index in group]
    charges = {"processed_training_tokens_all_attempts": sum(len(rows[index]["ids"]) for index in indices),
               "evaluation_and_qualification_model_calls": len(indices)}
    require(charges == {"processed_training_tokens_all_attempts": ZERO_PREFIX_TOKENS,
                        "evaluation_and_qualification_model_calls": ZERO_PREFIX_ROWS} and
            plan["native_denominators"][:2] == [213, 175] and
            plan["applied_native_learning_rates"][:2] == [2.672470144728172e-06, 5.2924359644125685e-06],
            "two-update qualification rows/denominators/learning rates differ")
    # Charge all prefix tokens and primary loss forward calls before any update.
    reserve(charges)
    frozen_before = adapter_module.frozen_base_digest(model, spec)
    def step(optimizer, index):
        guard()
        return _apply_update(model, rows, groups[index], plan["binding_update_groups"][index],
            plan["native_denominators"][index], plan["applied_native_learning_rates"][index], optimizer, design,
            lambda current, row: _native_loss_sum(current, row, design), binding_enabled=False)
    def compare(optimizer):
        checks = _compare_reference_state(optimizer, manifest, reference)
        return {"checks": checks, "reference_rng_bitwise_equal": canonical(rng_state()) == canonical(manifest["rng"]),
                "reference_rng_sha256": digest(manifest["rng"]), "actual_prefix_rng_sha256": digest(rng_state())}
    recovered = _disposable_prefix(model, named, lambda: CpuAdamW(named, design), step, compare, rng_state, restore_rng)
    frozen_after = adapter_module.frozen_base_digest(model, spec)
    comparison = recovered.pop("comparison")
    checks = comparison.get("checks", [])
    equality = {"all_trainable_parameters_bitwise_equal": len(checks) == TRAINABLE_TENSORS and all(row["parameter"]["equal"] for row in checks),
                "all_first_moments_bitwise_equal": len(checks) == TRAINABLE_TENSORS and all(row["exp_avg"]["equal"] for row in checks),
                "all_second_moments_bitwise_equal": len(checks) == TRAINABLE_TENSORS and all(row["exp_avg_sq"]["equal"] for row in checks)}
    binding_calls = sum(row["binding_forward_calls"] for row in recovered["steps"])
    complete = recovered["failure"] is None and len(checks) == TRAINABLE_TENSORS and recovered["optimizer_step_count"] == 2
    passed = (complete and all(equality.values()) and binding_calls == 0 and comparison.get("reference_rng_bitwise_equal") is True and
              recovered["trainable_state_restored_exactly"] and recovered["rng_restored_exactly"] and frozen_before == frozen_after)
    return {"schema": 1, "record": "BINDING_INTEGRATION_ZERO_ADDEND_QUALIFICATION", "experiment": EXPERIMENT,
            "status": "PASS" if passed else "FAILED", "complete": complete, "scope": "EXACT_TWO_UPDATE_PREFIX_ONLY",
            "binding": binding, "reference_checkpoint_path": str(reference),
            "reference_manifest_sha256": plan["zero_addend_reference"]["manifest_sha256"],
            "reference_tensor_files_sha256": plan["zero_addend_reference"]["tensor_files_sha256"],
            "reference_progress": manifest["progress"], "native_row_ids": [rows[index]["id"] for index in indices],
            "native_denominators": plan["native_denominators"][:2], "applied_native_learning_rates": plan["applied_native_learning_rates"][:2],
            "binding_forward_calls": binding_calls, "checked_trainable_tensors": len(checks), "checked_component_tensors": 3 * len(checks),
            "checks": checks, **{key: value for key, value in comparison.items() if key != "checks"}, **equality, **recovered,
            "component_hash_formula": "sha256(canonical([name,component,shape,'torch.float32']) + contiguous CPU uint8 bytes)",
            "frozen_base_unchanged": frozen_before == frozen_after, "frozen_base_sha256_before": frozen_before,
            "frozen_base_sha256_after": frozen_after, "budget_charges": charges, "full_193_update_trajectory_equality_measured": False,
            "generation_calls": 0, "quality_claim": False, "release_permitted": False}


def train_groups(model, rows, plan, design, named, spec, binding, directory, journal, adapter_module, zero_addend_sha256):
    import torch
    from experiments.model_quality_gpu_v1.checkpoints import rng_state
    from experiments.model_quality_gpu_v1.optimizer import CpuAdamW
    groups = plan["update_groups"]
    require(len(groups) == 193 and groups == [native + added for native, added in zip(plan["native_update_groups"], plan["binding_update_groups"], strict=True)],
            "runtime integration grouping differs from frozen plan")
    zero_path = directory / "zero_addend_qualification.json"
    zero = read_json(zero_path)
    require(file_hash(zero_path) == zero_addend_sha256 and zero["status"] == "PASS" and zero["complete"] is True and
            zero["binding"] == binding and zero["restored_trainable_sha256"] == _trainable_sha(named), "active training requires successful recovered zero-addend qualification")
    initial_rng_sha = digest(rng_state())
    require(initial_rng_sha == zero["rng_after_restore_sha256"] == zero["rng_before_sha256"],
            "active trajectory no longer starts at the initial qualified RNG state")
    optimizer = CpuAdamW(named, design)
    require(optimizer.step_count == 0 and all(not bool(t.any()) for state in optimizer.state.values() for t in state), "active Adam must start fresh with zero moments")
    frozen_before = adapter_module.frozen_base_digest(model, spec)
    progress = progress_at(rows, groups, 0)
    checkpoint_hashes = {}
    with (directory / "train.csv").open("x", encoding="utf-8", newline="") as csv_stream, \
            (directory / "steps.jsonl").open("xb") as event_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=("updates", "processed_tokens", "supervised_tokens", "objective_ce", "lr", "grad_norm"))
        writer.writeheader(); csv_stream.flush()
        for index, group in enumerate(groups):
            processed = sum(len(rows[i]["ids"]) for i in group)
            denominator = plan["native_denominators"][index]
            require(denominator > 0, "update lacks supervised assistant/EOS targets")
            journal.reserve({"processed_training_tokens_all_attempts": processed})
            resources = guard()
            progress = progress_at(rows, groups, index + 1)
            native_clock = plan["native_processed_cumulative"][index]
            learning_rate = rate_at(PEAK_LR, native_clock, NATIVE_PROCESSED_TOKENS, design)
            require(learning_rate == plan["applied_native_learning_rates"][index], "active native-token LR clock differs")
            measured = _apply_update(model, rows, plan["native_update_groups"][index], plan["binding_update_groups"][index],
                denominator, learning_rate, optimizer, design, lambda current, row: _native_loss_sum(current, row, design), binding_enabled=True)
            record = {**progress, **measured, "lr": learning_rate, "native_clock_tokens": native_clock,
                      "native_supervised_denominator": denominator, "binding_loss_divisor": denominator,
                      "row_ids": [rows[i]["id"] for i in group], "resources": resources}
            writer.writerow({key: record[key] for key in writer.fieldnames}); csv_stream.flush(); os.fsync(csv_stream.fileno())
            event_stream.write(canonical(record) + b"\n"); event_stream.flush(); os.fsync(event_stream.fileno())
            if progress["updates"] in (2, len(groups)):
                journal.reserve({"new_disk_bytes": CHECKPOINT_BYTES})
                require(shutil.disk_usage(directory).free - CHECKPOINT_BYTES >= DISK_FLOOR, "checkpoint would violate native disk floor")
                guard()
                path = directory / f"checkpoint-{progress['updates']:05d}"
                checkpoint_hashes[path.name] = save_state(path, optimizer, binding, progress, spec, zero_addend_sha256)
            print(json.dumps({"update": progress["updates"], "total_updates": len(groups), "processed_tokens": progress["processed_tokens"]}), flush=True)
    require(progress == {"updates": 193, "next_group": 193, "processed_tokens": 577417, "supervised_tokens": 39502}, "final trajectory incomplete")
    require(adapter_module.frozen_base_digest(model, spec) == frozen_before, "frozen source changed during LoRA training")
    return {"record": "BINDING_INTEGRATION_TRAINING_RESULT", "experiment": EXPERIMENT, "complete": True,
            "progress": progress, "binding": binding, "adapter_spec": spec.as_dict(),
            "native_clock_tokens": NATIVE_PROCESSED_TOKENS, "active_optimizer_started_fresh": True,
            "active_initial_rng_sha256": initial_rng_sha, "active_initial_trainable_sha256": zero["restored_trainable_sha256"],
            "zero_addend_qualification_path": str(zero_path), "zero_addend_qualification_sha256": zero_addend_sha256,
            "normalization": "native and binding summed losses each divided by the original native supervised denominator",
            "frozen_base_sha256_before_after": frozen_before,
            "checkpoint": str(directory / "checkpoint-00193"),
            "checkpoint_manifest_sha256": checkpoint_hashes["checkpoint-00193"],
            "checkpoint_manifests_sha256": checkpoint_hashes,
            "steps_sha256": file_hash(directory / "steps.jsonl"), "train_csv_sha256": file_hash(directory / "train.csv"),
            "memorization_probes": "OMITTED; no generation; optimizer RNG is unaffected by omitted restored-RNG probes",
            "quality_claim": False, "release_permitted": False}


def qualify_final_merge(model, tokenizer, rows, spec, binding, checkpoint, reserve, adapter_module):
    """Compare actual CPU-merged bytes against adapter forwards on fixed inputs."""
    import torch
    from torch.nn import functional as F
    from experiments.model_quality_gpu_v1.native import hidden_states
    from experiments.reasoning_posttraining_v1.serialization import compile_row
    prompt = [{"role": "system", "content": "Answer with the requested number."},
              {"role": "user", "content": "Reply with 7."}]
    short = compile_row(prompt, "7", tokenizer)["ids"]
    require(len(short) == 31 and 1 <= len(rows[0]["ids"]) <= 2048, "merge-qualification input geometry changed")
    vocabulary = min(model.config.vocab_size, 2048)
    sequences = (("reply_7", short), ("first_training_row", rows[0]["ids"]),
                 ("synthetic_2048_arange", [index % vocabulary for index in range(2048)]))
    charges = {"processed_training_tokens_all_attempts": 2 * sum(len(ids) for _, ids in sequences),
               "evaluation_and_qualification_model_calls": 6}
    reserve(charges)
    frozen_before = adapter_module.frozen_base_digest(model, spec)
    was_training = model.training
    model.eval()
    baseline, records = {}, []
    tensor_hashes = None
    def logits(token_ids):
        guard()
        ids = torch.tensor([token_ids], device="cuda:0", dtype=torch.long)
        hidden = hidden_states(model, ids)
        value = F.linear(hidden[:, -16:, :], model.lm_head.weight).detach().cpu().contiguous()
        require(bool(torch.isfinite(value).all()), "nonfinite merge-qualification logits")
        del hidden, ids
        return value
    try:
        with torch.no_grad():
            for label, token_ids in sequences:
                baseline[label] = logits(token_ids)
            # Only this worker's unused allocator blocks are released; no other process is touched.
            torch.cuda.empty_cache()
            with adapter_module.temporary_dense_merge(model, spec) as tensor_hashes:
                require(isinstance(tensor_hashes, dict) and len(tensor_hashes) == 220, "merged tensor identity inventory differs")
                for label, token_ids in sequences:
                    merged = logits(token_ids)
                    original = baseline[label]
                    difference = (merged - original).abs()
                    threshold = 2e-4 + 2e-4 * original.abs()
                    violations = int((difference > threshold).sum())
                    argmax_mismatches = int((merged.argmax(dim=-1) != original.argmax(dim=-1)).sum())
                    records.append({"input": label, "input_token_count": len(token_ids),
                                    "input_ids_sha256": digest(token_ids), "checked_logits_shape": list(merged.shape),
                                    "checked_last_positions": min(16, len(token_ids)),
                                    "max_abs_difference": float(difference.max()),
                                    "tolerance_violations": violations, "argmax_mismatches": argmax_mismatches,
                                    "passed": violations == 0 and argmax_mismatches == 0})
                    del merged, difference, threshold
    finally:
        model.train(was_training)
    require(adapter_module.frozen_base_digest(model, spec) == frozen_before,
            "restoring temporary merged modules changed frozen source parameters")
    passed = len(records) == 3 and all(record["passed"] for record in records)
    return {"schema": 1, "record": "BINDING_INTEGRATION_FINAL_MERGE_QUALIFICATION", "experiment": EXPERIMENT,
            "status": "PASS" if passed else "FAILED", "complete": True, "binding": binding,
            "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": file_hash(checkpoint / "manifest.json"),
            "dense_tensor_sha256": tensor_hashes,
            "tensor_hash_formula": "sha256(canonical([name,shape,'torch.float32']) + contiguous CPU uint8 bytes)",
            "merge_formula": "CPU FP32 weight + (CPU FP32 B @ CPU FP32 A) * 2; trained norms unchanged",
            "atol": 2e-4, "rtol": 2e-4, "argmax_equality_required": True,
            "inputs": records, "budget_charges": charges,
            "frozen_base_sha256_before_after": frozen_before, "original_modules_restored": True,
            "quality_claim": False, "release_permitted": False}


def worker(args):
    directory = Path(args.out)
    try:
        claim_worker(directory)
    except Exception as exc:
        print(f"Private LoRA worker refused: {exc}", file=sys.stderr)
        return 2
    try:
        with lease(Path(args.resource_lock)):
            plan, grant, design, unique, rows = inputs(args, retain=True)
            code = execution_identity()
            require(digest(code) == args.expected_code_digest, "execution code changed after dispatch")
            from experiments.model_quality_gpu_v1.native import audit_assets, configure_runtime, load_model, load_tokenizer, runtime
            from experiments.reasoning_posttraining_v1.native_runtime import qualify
            configure_runtime(17)
            before = guard(before_load=True, phase="train")
            require(shutil.disk_usage(directory).free - 2 * CHECKPOINT_BYTES >= DISK_FLOOR, "two LoRA checkpoints would violate free-space floor")
            journal = Journal(Path(args.journal), grant)
            require(not journal.breached, "approved budget is already breached")
            binding = {"experiment": EXPERIMENT, "plan_sha256": digest(plan),
                       "data_contract_sha256": file_hash(Path(args.prepared) / "DATA_CONTRACT.json"),
                       "data_build_sha256": file_hash(Path(args.prepared) / "DATA_BUILD.json"),
                       "training_inventory": {"views": 3465, "parent_worlds": 2761, "exposures_per_view": 1},
                       "plan_path": str(Path(args.plan).absolute()), "prepared": str(Path(args.prepared).absolute()),
                       "training_files_sha256": {name: file_hash(Path(args.prepared) / name)
                                                 for name in DATA_FILES},
                       "grant_sha256": digest(grant), "code_sha256": digest(code), "execution_code": code,
                       "checkout": PILOT_HEAD, "runtime": runtime(require=True), "source": audit_assets(Path(args.source)),
                       "design": design.as_dict(), "parameterization": plan["parameterization"],
                       "optimizer_peak_lr": PEAK_LR, "comparator_peak_lr": PEAK_LR,
                       "native_clock_tokens": NATIVE_PROCESSED_TOKENS,
                       "integration_rule": "native groups first, then floor-partitioned binding; both loss sums use original native denominator; LR uses native-only cumulative tokens",
                       "interpretation": "integration-regime test with original native coefficients; binding coefficient exposure differs from the previous mixture (weighted budget ratio0.6771691964091732); scalar coefficients are not gradient influence"}
            write_new(directory / "binding.json", {"binding": binding, "resources_before_load": before, "budget_before": journal.read()})
            tokenizer = load_tokenizer(Path(args.source))
            model = load_model(Path(args.source), "cuda:0")
            guard()
            base_qualification = qualify(model, tokenizer, unique, journal.reserve, guard)
            require(base_qualification["status"] == "PASS", "native base qualification failed")
            write_new(directory / "native_base_qualification.json", {**base_qualification, "binding": binding,
                      "scope": "Unmodified source only, before adapter injection; does not qualify LoRA"})
            adapter_module = load_adapters()
            named, spec, adapter_qualification = qualify_and_inject(model, tokenizer, journal.reserve, guard, adapter_module)
            write_new(directory / "adapter_qualification.json", {**adapter_qualification, "binding": binding})
            require(execution_identity() == code, "execution code changed during qualification")
            zero_qualification = qualify_zero_addend(model, rows, plan, design, named, spec, binding, journal.reserve, adapter_module)
            zero_path = directory / "zero_addend_qualification.json"
            write_new(zero_path, zero_qualification)
            require(zero_qualification["status"] == "PASS" and zero_qualification["complete"] is True,
                    "exact two-update native recovery or initial state/RNG restoration failed; active updates refused")
            require(execution_identity() == code, "execution code changed during zero-addend qualification")
            result = train_groups(model, rows, plan, design, named, spec, binding, directory, journal, adapter_module, file_hash(zero_path))
            merge_qualification = qualify_final_merge(model, tokenizer, rows, spec, binding,
                                   directory / "checkpoint-00193", journal.reserve, adapter_module)
            merge_path = directory / "merge_qualification.json"
            write_new(merge_path, merge_qualification)
            require(merge_qualification["status"] == "PASS", "final merged deployment failed fixed numerical/argmax qualification")
            require(execution_identity() == code, "execution code changed during training")
            require(audit_assets(Path(args.source)) == binding["source"], "source asset bytes changed during training")
            final_plan, final_grant, _, _, _ = inputs(args, retain=False)
            require(final_plan == plan and final_grant == grant, "frozen data plan or grant changed during training")
            charged = journal.read()["used_or_reserved"]
            require(charged["processed_training_tokens_all_attempts"] == EXPECTED_CHARGED_TOKENS and
                    charged["evaluation_and_qualification_model_calls"] == EXPECTED_QUALIFICATION_CALLS,
                    "active plus all mandatory qualification accounting differs")
            result.update(status="COMPLETE", merge_qualification_path=str(merge_path),
                          merge_qualification_sha256=file_hash(merge_path),
                          charged_processed_tokens=EXPECTED_CHARGED_TOKENS, qualification_model_calls=EXPECTED_QUALIFICATION_CALLS)
            write_new(directory / "result.json", result)
        return 0
    except Exception as exc:
        write_new(directory / "failure.json", {"record": "BINDING_INTEGRATION_WORKER_FAILURE", "status": "FAILED_OR_BLOCKED",
                  "type": type(exc).__name__, "message": str(exc), "quality_claim": False, "release_permitted": False})
        traceback.print_exc()
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prepared", "plan", "grant", "journal"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", default="train-001")
    parser.add_argument("--confirm-model-execution", action="store_true")
    parser.add_argument("--resource-lock", type=Path, default=Path(tempfile.gettempdir()) / "rineya-gpu-0.lock")
    parser.add_argument("--_worker", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-code-digest")
    args = parser.parse_args()
    require(args.confirm_model_execution, "model execution requires --confirm-model-execution")
    require(args.resource_lock.absolute() == (Path(tempfile.gettempdir()) / "rineya-gpu-0.lock").absolute(),
            "the shared rineya-gpu-0 lease path may not be overridden")
    if args._worker:
        return worker(args)
    require(args.out is None and args.expected_code_digest is None, "private worker arguments are not public overrides")
    inputs(args, retain=False)
    code = execution_identity()
    import psutil
    require(psutil.virtual_memory().available >= PRELOAD_MINIMA["train"][0] * GIB, "unchanged native 6 GiB pre-load RAM requirement not met")
    require(not args.resource_lock.exists() and not args.resource_lock.is_symlink(), "GPU resource lease is already present")
    grant = read_json(args.grant)
    journal_root = args.journal.absolute()
    journal = Journal(journal_root, grant, create=not journal_root.exists())
    out = journal_root / "runs" / args.run_id
    command = [sys.executable, str(Path(__file__).resolve())]
    for name in ("source", "prepared", "plan", "grant", "journal", "resource_lock"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name).absolute())]
    command += ["--run-id", args.run_id, "--out", str(out), "--expected-code-digest", digest(code),
                "--confirm-model-execution", "--_worker"]
    started = time.monotonic()
    journal.start(args.run_id, "train", TIMEOUT)
    returncode = bounded_child(command, out, TIMEOUT)
    supervisor = read_json(out / "supervisor.json")
    journal.settle(args.run_id, supervisor, file_hash(out / "supervisor.json"), time.monotonic() - started)
    write_new(out / "run_summary.json", {"experiment": EXPERIMENT, "run_id": args.run_id,
              "returncode": returncode, "supervisor": supervisor, "budget_after": journal.read(),
              "worker_failed": (out / "failure.json").exists(), "quality_claim": False, "release_permitted": False})
    print(json.dumps({"experiment": EXPERIMENT, "returncode": returncode, "out": str(out)}))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
