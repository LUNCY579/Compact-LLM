"""One supervised BINDING-MIXTURE-001 training attempt using the pinned native primitives.

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
PILOT = Path("C:/Users/clclc/BusinessProject/ChatBubble-binding-mixture-pilot-20260920")
PILOT_HEAD = "5b4c22dae78f587c41575269f419087b0c7997f6"
NATIVE_INPUT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/doseint001-20260919/prepared-dual899/microfit-train.jsonl")
BINDING_INPUT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-signal-prep-20260920/candidate-v2/compiled-v1/TRAIN_COMPILED.jsonl")
DATA_FILES = ("microfit-train.jsonl", "UPDATE_GROUPS.json", "SOURCE_MEMBERSHIP.json", "DATA_BUILD.json")
sys.path.insert(0, str(PILOT))
from experiments.model_quality_gpu_v1.config import Design, rate_at
from experiments.model_quality_gpu_v1.integrity import (canonical, code_identity, digest, file_hash,
                                                       new_dir, read_json, seal, write_new)
from experiments.model_quality_gpu_v1.resources import GIB, PRELOAD_MINIMA, guard, lease
from experiments.model_quality_gpu_v1.supervisor import bounded_child, claim_worker
from experiments.reasoning_posttraining_v1.budget import Journal, register, validate_grant
from experiments.reasoning_posttraining_v1.controller import checkout, package_identity
from experiments.model_quality_gpu_v1.data import group_updates

EXPERIMENT = "BINDING-MIXTURE-001"
PEAK_LR = 1e-4
TIMEOUT = 2400
DISK_FLOOR = 14 * GIB
TRAINABLE_PARAMETERS = 7_255_040
TRAINABLE_TENSORS = 360
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
    require(checkout() == PILOT_HEAD, "pinned native pilot checkout changed")
    files = {name: file_hash(HERE / name) for name in ("train.py", "adapters.py")}
    return {"runner_files": files, "runner_code_sha256": digest(files),
            "native_package_sha256": package_identity()["sha256"],
            "mq_code_sha256": code_identity()["sha256"], "pilot_checkout": PILOT_HEAD}


def expected_membership(native_count=2697, binding_count=768):
    """Private pure ordering primitive; production counts remain fixed in inputs."""
    import random
    require(native_count > 0 and binding_count > 0, "both mixture components are required")
    shuffled = list(range(binding_count)); random.Random(17).shuffle(shuffled)
    total = native_count + binding_count
    output, native_index, binding_index = [], 0, 0
    for position in range(total):
        if (position + 1) * binding_count // total > position * binding_count // total:
            output.append(("binding_v2", shuffled[binding_index])); binding_index += 1
        else:
            output.append(("native_dual899", native_index)); native_index += 1
    return output


def validate_prepared(prepared, plan):
    """Validate views, true parent worlds, unchanged row bytes and fixed insertion."""
    from collections import Counter
    prepared = Path(prepared).absolute()
    contract = read_json(prepared / "DATA_CONTRACT.json")
    require(plan["experiment"] == contract["experiment"] == EXPERIMENT, "fresh mixture plan/contract required")
    require(set(contract["files"]) == set(DATA_FILES), "mixture contract must bind all four preparation artifacts")
    for name, expected in contract["files"].items():
        require(file_hash(prepared / name) == expected, "prepared mixture artifact changed: " + name)
    build = read_json(prepared / "DATA_BUILD.json")
    require(build["record"] == "BINDING_MIXTURE_DATA_BUILD" and build["status"] == "PASS_DATA_PREPARATION_ONLY",
            "mixture was not admitted by its independent data builder")
    require(build["rows"] == build["presentations"] == 3465 and build["unique_worlds"] == build["parent_worlds"] == 2761,
            "views and parent worlds must remain distinct inventories")
    require(build["exposures_per_view"] == 1 and build["processed_tokens"] == 577417
            and build["supervised_tokens"] == 39502 and build["update_groups_count"] == 296,
            "frozen mixed-data token/update inventory changed")
    require(build["rows_copied_byte_identically"] is True and build["labels_or_teacher_inputs_modified"] is False
            and build["source_rows_repeated_or_dropped"] is False, "mixture construction changed source examples")
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
    membership_order = expected_membership()
    lines = (prepared / "microfit-train.jsonl").read_bytes().splitlines(keepends=True)
    membership = read_json(prepared / "SOURCE_MEMBERSHIP.json")
    require(len(lines) == len(membership) == 3465, "mixture line/membership count differs")
    rows, ids, parent_counts, binding_parents = [], [], Counter(), Counter()
    for index, (line, record, expected) in enumerate(zip(lines, membership, membership_order)):
        label, source_index = expected
        require(line == raw_sources[label][source_index], "mixture row bytes or deterministic insertion changed")
        row = json.loads(line)
        require(record == {"mixture_index": index, "id": row["id"], "parent_id": row["parent_id"],
                           "source": label, "source_index": source_index,
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
    groups = group_updates(rows, 2048)
    group_record = read_json(prepared / "UPDATE_GROUPS.json")
    require(len(groups) == 296 and groups == group_record["update_groups"] == plan["update_groups"]
            and ids == group_record["row_order_per_exposure"] == plan["row_order_per_exposure"]
            and digest(ids) == group_record["row_order_sha256"] == build["ordering"]["row_order_sha256"],
            "actual mixture order/update groups disagree with their frozen records")
    require(sum(len(row["ids"]) for row in rows) == 577417
            and sum(sum(token != -100 for token in row["labels"][1:]) for row in rows) == 39502
            and len(rows[0]["ids"]) == 116, "mixed training or qualification token geometry changed")
    for exclusion in build["acquisition_parent_exclusions"]:
        path = Path(exclusion["path"])
        require(file_hash(path) == exclusion["sha256"], "acquisition exclusion bank changed")
        parents = {json.loads(line)["parent_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        require(len(parents) == 384 and not set(parent_counts) & parents and exclusion["mixture_parent_overlap"] == 0,
                "mixed corpus overlaps a declared acquisition parent")
    return rows


def inputs(args, *, retain=False):
    prepared = Path(args.prepared)
    plan, grant = read_json(Path(args.plan)), read_json(Path(args.grant))
    require(plan["objective"] == "assistant_response_plus_eos" and plan["precision"] == "fp32"
            and plan["trainable"] == "blocks8-27_lora16_projections_and_norms", "unregistered mixture training scope")
    require(plan["processed_tokens"] == 577417 and plan["supervised_tokens"] == 39502
            and plan["presentations"] == 3465 and plan["unique_worlds"] == 2761
            and plan["exposures_per_view"] == 1 and len(plan["update_groups"]) == 296,
            "mixture plan must separately count views, worlds and one exposure per view")
    require(plan["selection_seed"] == plan["order_seed"] == 17 and plan["max_sequence"] == 2048,
            "frozen mixture seed/window changed")
    require(all(plan.get("parameterization", {}).get(key) == value for key, value in PARAMETERIZATION.items()),
            "mixture parameterization differs from the qualified rank16 tail20 recipe")
    design = Design(processed_budget=plan["processed_tokens"])
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


def save_state(directory, optimizer, binding, progress, adapter_spec):
    """Same safe storage primitive, explicitly a binding-mixture checkpoint."""
    import torch
    from safetensors.torch import save_file
    from experiments.model_quality_gpu_v1.checkpoints import rng_state
    require(progress["updates"] == optimizer.step_count and all(p.grad is None for _, p in optimizer.named),
            "checkpoint requires a cleared complete optimizer boundary")
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
    manifest = seal({"schema": 1, "record": "BINDING_MIXTURE_TRAINING_CHECKPOINT", "experiment": EXPERIMENT,
                     "status": "COMPLETE", "binding": binding, "progress": progress,
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
    return named, spec, {"record": "BINDING_MIXTURE_ZERO_UPDATE_QUALIFICATION", "experiment": EXPERIMENT,
            "status": "PASS", "optimizer_steps": 0, "source_state_before_injection": source_digest,
            "frozen_base_sha256": frozen, "trainable_sha256_before_after": before_trainable,
            "zero_B_source_logits_exact": True, "zero_B_source_cached_logits_exact": True, "cached_max_abs": cache_error,
            "A_zero_gradient_expected_at_zero_B": True, "gradient_probes": gradient_records,
            "budget_charges": charges, "rng_restored_after_qualification": True,
            "adapter_spec": spec.as_dict(), "quality_claim": False}


def train_groups(model, rows, plan, design, named, spec, binding, directory, journal, adapter_module):
    import torch
    from experiments.model_quality_gpu_v1.data import group_updates
    from experiments.model_quality_gpu_v1.losses import chunked_sum
    from experiments.model_quality_gpu_v1.native import hidden_states
    from experiments.model_quality_gpu_v1.optimizer import CpuAdamW
    groups = group_updates(rows, design.update_tokens)
    require(groups == plan["update_groups"] and len(groups) == 296, "runtime update grouping differs from frozen plan")
    optimizer = CpuAdamW(named, design)
    frozen_before = adapter_module.frozen_base_digest(model, spec)
    progress = progress_at(rows, groups, 0)
    checkpoint_hashes = {}
    with (directory / "train.csv").open("x", encoding="utf-8", newline="") as csv_stream, \
            (directory / "steps.jsonl").open("xb") as event_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=("updates", "processed_tokens", "supervised_tokens", "objective_ce", "lr", "grad_norm"))
        writer.writeheader(); csv_stream.flush()
        for index, group in enumerate(groups):
            processed = sum(len(rows[i]["ids"]) for i in group)
            denominator = sum(sum(token != -100 for token in rows[i]["labels"][1:]) for i in group)
            require(denominator > 0, "update lacks supervised assistant/EOS targets")
            journal.reserve({"processed_training_tokens_all_attempts": processed})
            resources = guard(); model.train(); optimizer.zero_grad()
            value = 0.0
            for row_index in group:
                row = rows[row_index]
                ids = torch.tensor([row["ids"]], device="cuda:0", dtype=torch.long)
                labels = torch.tensor([row["labels"]], device="cuda:0", dtype=torch.long)
                loss, _ = chunked_sum(hidden_states(model, ids), model.lm_head.weight, labels, design.ce_chunk)
                require(bool(torch.isfinite(loss)), "nonfinite LoRA objective")
                (loss / denominator).backward()
                value += float(loss.detach()) / denominator
            require(all(parameter.grad is None for _, parameter in model.named_parameters() if not parameter.requires_grad),
                    "a frozen parameter acquired a gradient")
            norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named], design.clip_norm, error_if_nonfinite=True)
            progress = progress_at(rows, groups, index + 1)
            learning_rate = rate_at(PEAK_LR, progress["processed_tokens"], plan["processed_tokens"], design)
            optimizer.step(learning_rate); optimizer.zero_grad()
            record = {**progress, "objective_ce": value, "lr": learning_rate, "grad_norm": float(norm),
                      "row_ids": [rows[i]["id"] for i in group], "resources": resources}
            writer.writerow({key: record[key] for key in writer.fieldnames}); csv_stream.flush(); os.fsync(csv_stream.fileno())
            event_stream.write(canonical(record) + b"\n"); event_stream.flush(); os.fsync(event_stream.fileno())
            if progress["updates"] in (2, len(groups)):
                journal.reserve({"new_disk_bytes": CHECKPOINT_BYTES})
                require(shutil.disk_usage(directory).free - CHECKPOINT_BYTES >= DISK_FLOOR, "checkpoint would violate native disk floor")
                guard()
                path = directory / f"checkpoint-{progress['updates']:05d}"
                checkpoint_hashes[path.name] = save_state(path, optimizer, binding, progress, spec)
            print(json.dumps({"update": progress["updates"], "total_updates": len(groups), "processed_tokens": progress["processed_tokens"]}), flush=True)
    require(progress == {"updates": 296, "next_group": 296, "processed_tokens": 577417, "supervised_tokens": 39502}, "final trajectory incomplete")
    require(adapter_module.frozen_base_digest(model, spec) == frozen_before, "frozen source changed during LoRA training")
    return {"record": "BINDING_MIXTURE_TRAINING_RESULT", "experiment": EXPERIMENT, "complete": True,
            "progress": progress, "binding": binding, "adapter_spec": spec.as_dict(),
            "frozen_base_sha256_before_after": frozen_before,
            "checkpoint": str(directory / "checkpoint-00296"),
            "checkpoint_manifest_sha256": checkpoint_hashes["checkpoint-00296"],
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
    return {"schema": 1, "record": "BINDING_MIXTURE_FINAL_MERGE_QUALIFICATION", "experiment": EXPERIMENT,
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
                       "interpretation": "added binding signal plus extra data/compute and changed token schedule/grouping; not a pure signal-isolation causal contrast"}
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
            result = train_groups(model, rows, plan, design, named, spec, binding, directory, journal, adapter_module)
            merge_qualification = qualify_final_merge(model, tokenizer, rows, spec, binding,
                                   directory / "checkpoint-00296", journal.reserve, adapter_module)
            merge_path = directory / "merge_qualification.json"
            write_new(merge_path, merge_qualification)
            require(merge_qualification["status"] == "PASS", "final merged deployment failed fixed numerical/argmax qualification")
            require(execution_identity() == code, "execution code changed during training")
            require(audit_assets(Path(args.source)) == binding["source"], "source asset bytes changed during training")
            final_plan, final_grant, _, _, _ = inputs(args, retain=False)
            require(final_plan == plan and final_grant == grant, "frozen data plan or grant changed during training")
            result.update(status="COMPLETE", merge_qualification_path=str(merge_path),
                          merge_qualification_sha256=file_hash(merge_path))
            write_new(directory / "result.json", result)
        return 0
    except Exception as exc:
        write_new(directory / "failure.json", {"record": "BINDING_MIXTURE_WORKER_FAILURE", "status": "FAILED_OR_BLOCKED",
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
