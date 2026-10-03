"""Bounded FP32 LoRA pilot with independent old-task KL. All outputs create-only.

Run prepare, baseline, train-control, train-anchor, eval-control, eval-anchor,
then summarize. Every GPU command is a synchronous supervised child holding the
shared GPU lease. Historical probes, checkouts and weights are read-only inputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from .core import bank, digest, exact, paired, rate_at, read_json, read_jsonl, validate_banks, write_new

HERE = Path(__file__).resolve().parent
BUSINESS = Path("C:/Users/clclc/BusinessProject")
RECORDS = BUSINESS / "rineya-dataset-retention-records"
SOURCE = BUSINESS / "Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze"
SOURCE_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
STAGES = ("baseline", "train-control", "train-anchor", "eval-control", "eval-anchor")


def prepare(out: Path):
    if out.exists():
        raise FileExistsError(out)
    anchor, dev = bank(301, 28, "anchor"), bank(303, 28, "dev")
    audit = validate_banks(anchor, dev)
    train = RECORDS / "rrp015-20260917/control-prepared/microfit-train.jsonl"
    groups = RECORDS / "rrp015-20260917/control-PLAN.json"
    acquisition = RECORDS / "select-views-001/SELECT_VIEWS.jsonl"
    training, selection = read_jsonl(train), read_jsonl(acquisition)
    if {r["parent_id"] for r in training} & {r["parent_id"] for r in selection}:
        raise ValueError("acquisition train/eval parent overlap")
    if len(training) != 384 or len(selection) != 384:
        raise ValueError("unexpected historical bank sizes")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True)
    for row in training:
        if prompt_ids(tokenizer, row) != row["prompt_ids"]:
            raise ValueError("current tokenizer differs from prepared historical prompt")
    lengths = [len(prompt_ids(tokenizer, row)) for row in [*anchor, *dev, *selection]]
    if max(lengths) + 96 > 2048:
        raise ValueError("planned prompt exceeds pilot context envelope")
    bindings = {str(p): digest(p) for p in [train, groups, acquisition, *sorted(SOURCE.glob("*.json")), SOURCE / "model.safetensors", SOURCE / "vocab.json", SOURCE / "merges.txt"]}
    if bindings[str(SOURCE / "model.safetensors")] != SOURCE_SHA:
        raise ValueError("source weight hash mismatch")
    plan = {"experiment": "CS-PILOT-001", "created_unix": time.time(),
            "class": "EXPLORATORY_MATCHED_HF_LANE", "quality_improvement_established": False,
            "source": str(SOURCE), "train": str(train), "groups": str(groups), "acquisition": str(acquisition),
            "bindings": bindings, "code": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
            "runtime": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "safetensors", "scipy", "psutil")},
            "seed": 17, "base_dtype": "float32", "adapter_dtype": "float32", "attention": "eager", "tf32": False,
            "rank": 16, "lora_alpha": 32, "layers": list(range(8, 28)), "peak_lr": 1e-4,
            "schedule": "processed-token WSD20/60/20 square-root decay to 0.1 peak",
            "optimizer": {"name": "torch.AdamW", "betas": [.9, .95], "eps": 1e-8, "weight_decay": 0, "clip": 1},
            "lambda": .25, "old_rows_per_group": 4, "acquisition_groups": 28,
            "objective": "CE(new) + lambda * mean_full_vocabulary_KL(base || student on independent old inputs)",
            "old_target_positions": "source-correct anchor gold answer plus EOS; teacher forced",
            "control": "same old forward evaluations with zero KL weight; full CE weight in both arms",
            "old_baseline_gate": {"minimum_correct_anchor_rows": 32, "minimum_correct_per_family": 8,
                                  "minimum_binding_answerable": 8, "minimum_binding_unknown": 8},
            "selection_rule": "report both fixed arms; no winner, hyperparameter search or confirmation in this pilot",
            "budget": {"total_wall_seconds": 2700, "baseline": 900, "train-control": 420, "train-anchor": 600,
                       "eval-control": 780, "eval-anchor": 780, "max_disk_bytes": 2 * 2**30,
                       "min_start_host_gib": 6, "min_running_host_gib": 1.5, "max_retries": 0,
                       "max_training_steps": 56, "cloud_jobs": 0},
            "data_audit": audit, "release_permitted": False}
    plan["tokenizer_preflight"] = {"matched_training_prompts": len(training), "max_prompt_tokens": max(lengths),
                                    "explicit_return_dict": False, "eos_token_id": tokenizer.eos_token_id}
    out.mkdir(parents=True)
    (out / "source_snapshot").mkdir()
    for p in [*sorted(HERE.glob("*.py")), HERE / "PROTOCOL.md"]:
        shutil.copy2(p, out / "source_snapshot" / p.name)
    write_new(out / "anchor.json", anchor)
    write_new(out / "dev.json", dev)
    plan["local_data"] = {p.name: digest(p) for p in (out / "anchor.json", out / "dev.json")}
    write_new(out / "PLAN.json", plan)
    print(json.dumps({"run": str(out), "plan_sha256": digest(out / "PLAN.json"), "audit": audit}))


def validate_plan(out: Path):
    plan = read_json(out / "PLAN.json")
    for name, expected in plan["code"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"source changed after preparation: {name}; use a new run")
    for name, expected in plan["local_data"].items():
        if digest(out / name) != expected:
            raise ValueError(f"local bank hash mismatch: {name}")
    for path, expected in plan["bindings"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"input hash mismatch: {path}")
    for name, expected in plan["runtime"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"runtime drift: {name}")
    return plan


def guard(plan):
    import psutil
    if psutil.virtual_memory().available / 2**30 < plan["budget"]["min_running_host_gib"]:
        raise MemoryError("running host-memory guard")


def require_complete(out: Path, stage: str):
    receipt = read_json(out / stage / "COMPLETE.json")
    supervisor = read_json(out / stage / "supervisor.json")
    if receipt.get("stage") != stage or supervisor.get("status") != "COMPLETE" or supervisor.get("child_returncode") != 0:
        raise ValueError(f"stage not successfully supervised: {stage}")
    for name, expected in receipt["artifacts"].items():
        if digest(out / stage / name) != expected:
            raise ValueError(f"completed output changed: {stage}/{name}")


def correctness(rows):
    result = {r["id"]: r["correct"] for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate prediction IDs")
    return result


def supervise(stage, out):
    import psutil
    plan = validate_plan(out)
    if stage in {"train-control", "train-anchor"}:
        require_complete(out, "baseline")
        qualification = read_json(out / "baseline/qualification.json")
        if not qualification["passed"]:
            raise ValueError("old-bank baseline competence gate failed")
    if stage.startswith("eval-"):
        require_complete(out, "train-" + stage.split("-", 1)[1])
    consumed = sum(read_json(p)["wall_seconds"] for p in out.glob("*/supervisor.json"))
    remaining = plan["budget"]["total_wall_seconds"] - consumed
    if remaining <= 0:
        raise ValueError("pilot total wall budget exhausted")
    if psutil.virtual_memory().available / 2**30 < plan["budget"]["min_start_host_gib"]:
        raise MemoryError("need 6 GiB free host RAM before model loading")
    if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) >= plan["budget"]["max_disk_bytes"]:
        raise ValueError("pilot disk budget exhausted")
    dest = out / stage
    dest.mkdir()  # Refuses every automatic retry or overwrite.
    lock = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
    lease = {"pid": os.getpid(), "create_time": psutil.Process().create_time(),
             "token": uuid.uuid4().hex, "owner": "CS-PILOT-001", "stage": stage}
    with lock.open("x", encoding="utf-8") as f:
        json.dump(lease, f)
    start = time.monotonic()
    child, status, failure = None, "FAILED", None
    try:
        command = [sys.executable, "-m", "research.composable_skills_v1.pilot", stage, "--run", str(out), "--worker"]
        write_new(dest / "launch.json", {"command": command, "plan_sha256": digest(out / "PLAN.json"),
                                        "lease": lease, "remaining_seconds": remaining})
        with (dest / "worker.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            limit = min(remaining, plan["budget"][stage])
            while child.poll() is None:
                if time.monotonic() - start > limit:
                    raise TimeoutError("supervised stage or cumulative wall budget exhausted")
                guard(plan)
                disk = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
                if disk > plan["budget"]["max_disk_bytes"]:
                    raise ValueError("pilot disk budget exceeded")
                time.sleep(1)
            code = child.returncode
        if code != 0:
            raise RuntimeError(f"worker exited {code}; see {dest / 'worker.log'}")
        if not (dest / "COMPLETE.json").is_file():
            raise RuntimeError("worker exited without completion receipt")
        status = "COMPLETE"
    except BaseException as e:
        failure = f"{type(e).__name__}: {e}"
        if child is not None and child.poll() is None:
            proc = psutil.Process(child.pid)
            descendants = proc.children(recursive=True)
            for descendant in descendants:
                try:
                    descendant.kill()
                except psutil.NoSuchProcess:
                    pass
            child.kill()
            child.wait(timeout=15)
        raise
    finally:
        write_new(dest / "supervisor.json", {"status": status, "failure": failure,
                                            "wall_seconds": time.monotonic() - start,
                                            "child_returncode": child.poll() if child else None,
                                            "quality_claim": False})
        if lock.is_file() and read_json(lock) == lease:
            lock.unlink()  # Only this parent's exact owned lease.
    print(json.dumps({"stage": stage, "status": status, "wall_seconds": time.monotonic() - start}))


def native(plan):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(plan["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * 2**30:
        raise MemoryError("need 6 GiB free GPU memory")
    tokenizer = AutoTokenizer.from_pretrained(plan["source"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(plan["source"], dtype=torch.float32,
                                              attn_implementation="eager", local_files_only=True).to("cuda").eval()
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise ValueError("base precision is not FP32")
    return model, tokenizer


def prompt_ids(tokenizer, row):
    result = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
    if not isinstance(result, list) or not result or any(type(v) is not int for v in result):
        raise TypeError("chat rendering must return a nonempty integer token list")
    return result


def generate(model, tokenizer, rows, plan, dest: Path, max_new: int):
    import torch
    outputs = []
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    model.eval()
    started = time.monotonic()
    with dest.open("x", encoding="utf-8") as log, torch.inference_mode():
        for i, row in enumerate(rows):
            guard(plan)
            ids = torch.tensor([prompt_ids(tokenizer, row)], device="cuda")
            if ids.shape[1] + max_new > 2048:
                raise ValueError("no truncation allowed; row exceeds pilot envelope")
            answer = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    do_sample=False, max_new_tokens=max_new, use_cache=True,
                                    pad_token_id=tokenizer.eos_token_id)
            new = answer[0, ids.shape[1]:].tolist()
            terminated = bool(new and new[-1] in eos)
            text = tokenizer.decode(new, skip_special_tokens=True).strip()
            result = {"id": row["id"], "family": row["family"], "world_id": row.get("world_id", row.get("parent_id")),
                      "reference": row["reference"], "prediction": text, "terminated": terminated,
                      "correct": exact(text, row["reference"], row["family"], terminated),
                      "input_tokens": ids.shape[1], "generated_tokens": len(new)}
            outputs.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()
            if (i + 1) % 28 == 0:
                print(f"{dest.name}: {i+1}/{len(rows)} {time.monotonic()-started:.0f}s", flush=True)
    return outputs


def compile_anchor(row, tokenizer):
    ids = prompt_ids(tokenizer, row)
    completion = tokenizer.encode(row["reference"], add_special_tokens=False) + [tokenizer.eos_token_id]
    # Qwen's chat end is <|im_end|>, which is the tokenizer EOS for this source.
    return {**row, "ids": ids + completion, "positions": list(range(len(ids) - 1, len(ids) + len(completion) - 1)),
            "targets": completion}


def body(model):
    return model.get_base_model().model if hasattr(model, "get_base_model") else model.model


def selected_hidden(model, row):
    import torch
    ids = torch.tensor([row["ids"]], device="cuda")
    return body(model)(input_ids=ids, use_cache=False).last_hidden_state[0, row["positions"]]


def kl_loss(logits, teacher_logits):
    import torch.nn.functional as F
    return F.kl_div(F.log_softmax(logits.float(), -1), F.log_softmax(teacher_logits.detach().float(), -1),
                    reduction="batchmean", log_target=True)


def baseline(out, plan):
    import torch
    from safetensors.torch import save_file
    dest = out / "baseline"
    model, tokenizer = native(plan)
    anchors = read_json(out / "anchor.json")
    predictions = generate(model, tokenizer, anchors, plan, dest / "anchor_predictions.jsonl", 12)
    qualified = {r["id"] for r in predictions if r["correct"]}
    compiled = [compile_anchor(r, tokenizer) for r in anchors if r["id"] in qualified]
    counts = {f: sum(r["family"] == f for r in compiled) for f in ("binding", "arithmetic", "time")}
    gate = plan["old_baseline_gate"]
    binding_answerable = sum(r["family"] == "binding" and r["answerable"] for r in compiled)
    binding_unknown = sum(r["family"] == "binding" and not r["answerable"] for r in compiled)
    passed = (len(compiled) >= gate["minimum_correct_anchor_rows"] and min(counts.values()) >= gate["minimum_correct_per_family"]
              and binding_answerable >= gate["minimum_binding_answerable"] and binding_unknown >= gate["minimum_binding_unknown"])
    write_new(dest / "qualification.json", {"passed": passed, "source_correct_anchor_rows": len(compiled), "by_family": counts,
                                            "correct_unknown": binding_unknown, "correct_binding_answerable": binding_answerable,
                                            "scope": "old-bank competence only, not native/DenseReference serving qualification"})
    if not passed:
        raise ValueError(f"insufficient source competence on old bank: {counts}")
    cache = {}
    with torch.inference_mode():
        for i, row in enumerate(compiled):
            cache[row["id"]] = selected_hidden(model, row).detach().cpu().contiguous()
    save_file(cache, str(dest / "teacher_hidden.safetensors"))
    write_new(dest / "compiled_anchors.json", compiled)
    # Zero-update head reconstruction check on an actual source sequence.
    r = compiled[0]
    with torch.inference_mode():
        ids = torch.tensor([r["ids"]], device="cuda")
        actual = model(input_ids=ids, use_cache=False,
                       logits_to_keep=torch.tensor(r["positions"], device="cuda")).logits[0]
        reconstructed = torch.nn.functional.linear(cache[r["id"]].to("cuda"), model.get_output_embeddings().weight)
        error = (actual - reconstructed).abs().max().item()
    if error > 1e-5:
        raise ValueError(f"teacher head reconstruction mismatch: {error}")
    write_new(dest / "cache.json", {"sha256": digest(dest / "teacher_hidden.safetensors"),
                                   "rows_sha256": digest(dest / "compiled_anchors.json"), "head_reconstruction_max_abs": error})
    generate(model, tokenizer, read_json(out / "dev.json"), plan, dest / "old_dev.jsonl", 12)
    generate(model, tokenizer, read_jsonl(Path(plan["acquisition"])), plan, dest / "acquisition.jsonl", 96)


def training(out, plan, arm):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    dest = out / f"train-{arm}"
    base, tokenizer = native(plan)
    model = get_peft_model(base, LoraConfig(r=plan["rank"], lora_alpha=plan["lora_alpha"], lora_dropout=0,
                                         bias="none", task_type="CAUSAL_LM", layers_to_transform=plan["layers"],
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not params or any(p.dtype != torch.float32 for p in params.values()):
        raise ValueError("adapter trainability/precision mismatch")
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    head = model.get_output_embeddings().weight
    if head.requires_grad:
        raise ValueError("teacher projection head must remain frozen")
    raw = read_jsonl(Path(plan["train"]))
    train = {}
    for row in raw:
        if prompt_ids(tokenizer, row) != row["prompt_ids"]:
            raise ValueError("current tokenizer rendering differs from prepared training prompt")
        positions = [i - 1 for i, label in enumerate(row["labels"]) if label != -100]
        if not positions or min(positions) < 0:
            raise ValueError("invalid answer label positions")
        train[row["id"]] = {**row, "positions": positions, "targets": [row["ids"][i+1] for i in positions]}
    order_plan = read_json(Path(plan["groups"]))
    order = order_plan["row_order_per_exposure"]
    if isinstance(order[0], list):
        order = [rid for exposure in order for rid in exposure]
    groups = [[order[i] for i in g] for g in order_plan["update_groups"]]
    if len(groups) != 28 or len([rid for g in groups for rid in g]) != 384:
        raise ValueError("training budget mismatch")
    anchors = read_json(out / "baseline/compiled_anchors.json")
    cache_meta = read_json(out / "baseline/cache.json")
    if digest(out / "baseline/teacher_hidden.safetensors") != cache_meta["sha256"] or digest(out / "baseline/compiled_anchors.json") != cache_meta["rows_sha256"]:
        raise ValueError("teacher cache binding mismatch")
    cache = load_file(str(out / "baseline/teacher_hidden.safetensors"), device="cpu")
    rng = random.Random(plan["seed"])
    rng.shuffle(anchors)
    total_tokens = sum(len(train[rid]["ids"]) for g in groups for rid in g)
    opt = torch.optim.AdamW(params.values(), lr=plan["peak_lr"], betas=(.9, .95), eps=1e-8, weight_decay=0)
    processed = supervised = old_tokens = 0
    old_family_counts, old_family_tokens = {}, {}
    records = []
    started = time.monotonic()
    with (dest / "steps.jsonl").open("x", encoding="utf-8") as stream:
        for gi, group in enumerate(groups):
            guard(plan)
            opt.zero_grad(set_to_none=True)
            count = sum(len(train[rid]["targets"]) for rid in group)
            ce_value = 0.
            for rid in group:
                row = train[rid]
                logits = F.linear(selected_hidden(model, row), head)
                target = torch.tensor(row["targets"], device="cuda")
                loss = F.cross_entropy(logits, target, reduction="sum") / count
                loss.backward()
                ce_value += float(loss.detach())
                processed += len(row["ids"])
                supervised += len(row["targets"])
                del loss, logits
            old_rows = [anchors[(gi * plan["old_rows_per_group"] + j) % len(anchors)] for j in range(plan["old_rows_per_group"])]
            old_count = sum(len(row["targets"]) for row in old_rows)
            kl_value = 0.
            for row in old_rows:
                with torch.set_grad_enabled(arm == "anchor"):
                    logits = F.linear(selected_hidden(model, row), head)
                    with torch.no_grad():
                        teacher = F.linear(cache[row["id"]].to("cuda"), head)
                    loss = kl_loss(logits, teacher) * len(row["targets"]) / old_count
                    if arm == "anchor":
                        (plan["lambda"] * loss).backward()
                    kl_value += float(loss.detach())
                old_tokens += len(row["ids"])
                old_family_counts[row["family"]] = old_family_counts.get(row["family"], 0) + 1
                old_family_tokens[row["family"]] = old_family_tokens.get(row["family"], 0) + len(row["targets"])
                del logits, teacher, loss
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0))
            if not math_isfinite(grad_norm):
                raise ValueError("nonfinite gradient")
            lr = rate_at(plan["peak_lr"], processed, total_tokens)
            for group_opt in opt.param_groups:
                group_opt["lr"] = lr
            before = {n: p.detach().clone() for n, p in params.items()}
            opt.step()
            step_norm = sum(float((p.detach() - before[n]).double().square().sum()) for n, p in params.items()) ** .5
            del before
            record = {"step": gi + 1, "processed_tokens": processed, "supervised_tokens": supervised,
                      "old_forward_tokens": old_tokens, "new_ce": ce_value, "old_kl": kl_value,
                      "kl_denominator_prediction_positions": old_count,
                      "lambda": plan["lambda"] if arm == "anchor" else 0,
                      "actual_lr": lr, "grad_norm_before_clip": grad_norm, "adapter_coordinate_step_l2": step_norm,
                      "old_ids": [r["id"] for r in old_rows], "wall_seconds": time.monotonic() - started}
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print(f"step {gi+1}/28 CE={ce_value:.4f} oldKL={kl_value:.5f} lr={lr:.3g}", flush=True)
    if any(p._version != frozen_versions[n] for n, p in model.named_parameters() if n in frozen_versions):
        raise ValueError("a frozen base parameter changed in place")
    opt.zero_grad(set_to_none=True)
    # Native weight-space delta norms, not incomparable LoRA A/B coordinate norms.
    native_norm_sq, native_elements = 0., 0
    for module in model.modules():
        if hasattr(module, "get_delta_weight") and hasattr(module, "lora_A"):
            delta = module.get_delta_weight("default").detach()
            native_norm_sq += float(delta.double().square().sum())
            native_elements += delta.numel()
            del delta
    model.save_pretrained(str(dest / "adapter"))
    write_new(dest / "TRAIN.json", {"arm": arm, "steps": len(records), "trainable_params": sum(p.numel() for p in params.values()),
                                   "processed_tokens": processed, "supervised_tokens": supervised, "old_forward_tokens": old_tokens,
                                   "old_family_presentations": old_family_counts, "old_family_target_tokens": old_family_tokens,
                                   "kl_reduction": "target-position mean over group, not family-balanced",
                                   "native_delta_l2": native_norm_sq ** .5, "native_delta_elements": native_elements,
                                   "base_frozen_versions_unchanged": True, "base_dtype": "float32", "adapter_dtype": "float32",
                                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                                   "adapter_sha256": digest(dest / "adapter/adapter_model.safetensors"),
                                   "lambda": plan["lambda"] if arm == "anchor" else 0, "quality_claim": False})


def math_isfinite(x):
    import math
    return math.isfinite(x)


def evaluate(out, plan, arm):
    from peft import PeftModel
    model, tokenizer = native(plan)
    training_receipt = read_json(out / f"train-{arm}/TRAIN.json")
    adapter = out / f"train-{arm}/adapter"
    if digest(adapter / "adapter_model.safetensors") != training_receipt["adapter_sha256"]:
        raise ValueError("adapter hash mismatch")
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).eval()
    dest = out / f"eval-{arm}"
    generate(model, tokenizer, read_json(out / "dev.json"), plan, dest / "old_dev.jsonl", 12)
    generate(model, tokenizer, read_jsonl(Path(plan["acquisition"])), plan, dest / "acquisition.jsonl", 96)


def summarize(out):
    from scipy.stats import beta
    plan = validate_plan(out)
    require_complete(out, "baseline")
    old_rows = {r["id"]: r for r in read_json(out / "dev.json")}
    results = {"experiment": plan["experiment"], "quality_improvement_established": False, "release_permitted": False,
               "inference": "matched local HF FP32; no DenseReference/native parity claim", "arms": {}}
    base_old = read_jsonl(out / "baseline/old_dev.jsonl")
    base_new = read_jsonl(out / "baseline/acquisition.jsonl")
    for arm, directory in (("base", "baseline"), ("control", "eval-control"), ("anchor", "eval-anchor")):
        if not (out / directory / "COMPLETE.json").exists():
            continue
        require_complete(out, directory)
        old = read_jsonl(out / directory / "old_dev.jsonl")
        new = read_jsonl(out / directory / "acquisition.jsonl")
        if correctness(old).keys() != old_rows.keys() or correctness(new).keys() != {r["id"] for r in read_jsonl(Path(plan["acquisition"]))}:
            raise ValueError("predictions do not exactly cover declared instruments")
        score = {"acquisition": paired(correctness(base_new), correctness(new)),
                 "old_all": paired(correctness(base_old), correctness(old)),
                 "old_by_family": {}, "new_by_family": {}}
        for family in sorted({r["family"] for r in old}):
            score["old_by_family"][family] = paired({r["id"]: r["correct"] for r in base_old if r["family"] == family},
                                                         {r["id"]: r["correct"] for r in old if r["family"] == family})
        for family, floor in (("ledger_replay", 30/128), ("prerequisite_plan", 10/128), ("register_machine", 9/128)):
            family_rows = [r for r in new if r["family"] == family]
            n, k = len(family_rows), sum(r["correct"] for r in family_rows)
            lower = float(beta.ppf(.025, k, n-k+1)) if k else 0.
            score["new_by_family"][family] = {"n": n, "correct": k, "cp_lower_95": lower, "floor": floor, "clears": lower > floor}
        binding = [r for r in old if r["family"] == "binding"]
        missing = [r for r in binding if not old_rows[r["id"]]["answerable"]]
        available = [r for r in binding if old_rows[r["id"]]["answerable"]]
        score["abstention"] = {"missing": len(missing), "correct_unknown": sum(r["correct"] for r in missing),
                               "answerable": len(available), "false_unknown": sum(r["prediction"] == "UNKNOWN" for r in available)}
        worlds = {}
        for r in binding:
            worlds.setdefault(r["world_id"], []).append(r["correct"])
        score["binding_all_four_correct_worlds"] = sum(all(v) for v in worlds.values())
        score["binding_worlds"] = len(worlds)
        results["arms"][arm] = score
    write_new(out / "SUMMARY.json", results)
    print(json.dumps({k: {"acquisition": v["acquisition"]["candidate_correct"], "old": v["old_all"]["candidate_correct"],
                          "old_lost": v["old_all"]["lost"], "old_gained": v["old_all"]["gained"],
                          "abstention": v["abstention"]} for k, v in results["arms"].items()}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "summarize", *STAGES))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()
    out = args.run.resolve()
    if args.stage == "prepare":
        prepare(out)
    elif args.stage == "summarize":
        summarize(out)
    elif not args.worker:
        supervise(args.stage, out)
    else:
        plan = validate_plan(out)
        # Workers must belong to the running supervisor recorded at dispatch.
        launch = read_json(out / args.stage / "launch.json")
        lock = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
        import psutil
        # Windows venv python.exe is a redirector; the real interpreter can be a
        # grandchild. Verify the live ancestor's identity, not just immediate PPID.
        owner = launch["lease"]
        ancestors = {p.pid: p.create_time() for p in psutil.Process().parents()}
        if ancestors.get(owner["pid"]) != owner["create_time"] or read_json(lock) != owner:
            raise RuntimeError("worker missing its owned supervisor lease")
        if args.stage == "baseline":
            baseline(out, plan)
        elif args.stage.startswith("train-"):
            training(out, plan, args.stage.split("-", 1)[1])
        else:
            evaluate(out, plan, args.stage.split("-", 1)[1])
        dest = out / args.stage
        artifacts = {str(p.relative_to(dest)): digest(p) for p in dest.rglob("*")
                     if p.is_file() and p.suffix in {".json", ".jsonl", ".safetensors"}}
        write_new(dest / "COMPLETE.json", {"stage": args.stage, "quality_claim": False,
                                           "completed_unix": time.time(), "artifacts": artifacts})


if __name__ == "__main__":
    main()
