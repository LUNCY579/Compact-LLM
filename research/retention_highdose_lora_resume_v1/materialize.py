"""Merge one sealed tail-LoRA checkpoint, then export its matched-norm scalar.

No model is instantiated. CPU tensor work holds the shared worker lease. Every
output is create-only; failed attempts and checkpoints are never removed. The
norm is the FP64 Frobenius norm of the actual FP32 dense update over all 220
changed base tensors, including the 80 trained norm vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.model_quality_gpu_v1.integrity import (  # noqa: E402
    canonical, decode, digest, file_hash, new_dir, read_json, regular, seal, unseal, write_new,
)
from experiments.model_quality_gpu_v1.resources import lease  # noqa: E402
from research.retention_dose_parameterization_v1.build_arm import (  # noqa: E402
    SIDECARS, tensor_header, require_ram, resource_lock_path,
)

GIB = 2**30
NATIVE_MANIFEST_SHA256 = "4142bb888b32e57e71d1225546ecd55179d0591af9ba3e88a0b3de5eb3834044"
FINAL_PROGRESS = {"updates": 193, "next_group": 193,
                  "processed_tokens": 380921, "supervised_tokens": 37090}
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight",
         "self_attn.q_norm.weight", "self_attn.k_norm.weight")
PRODUCTION_SHAPES = {"self_attn.q_proj": (1024, 2048), "self_attn.k_proj": (1024, 1024),
                     "self_attn.v_proj": (1024, 1024), "self_attn.o_proj": (2048, 1024),
                     "mlp.gate_proj": (1024, 3072), "mlp.up_proj": (1024, 3072),
                     "mlp.down_proj": (3072, 1024)}
NORM_WIDTHS = {name: 128 if name.startswith("self_attn.") else 1024 for name in NORMS}
BASE_PARAMETERS, ADAPTER_PARAMETERS, NORM_PARAMETERS = 596049920, 7208960, 46080
RANK, LORA_ALPHA, SCALE = 16, 32.0, 2.0
BETA_MIN, BETA_MAX, MATCH_TOLERANCE = 0.0, 2.0, 0.005
DENSE_NAMES = {f"model.layers.{i}.{s}.weight" for i in range(8, 28) for s in PROJECTIONS}
DENSE_NAMES |= {f"model.layers.{i}.{s}" for i in range(8, 28) for s in NORMS}
NORM_NAMES = {f"model.layers.{i}.{s}" for i in range(8, 28) for s in NORMS}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(value, name: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), f"invalid SHA256: {name}")
    return value


def _fresh(path: Path) -> None:
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "unsafe output path")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path.name}")


def _disk(out: Path, required: int) -> None:
    ancestor = out.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < required + 14 * GIB:
        raise RuntimeError("materialization would violate the 14 GiB disk floor")


def expected_lora_spec(header: dict) -> dict:
    """Independently reconstruct the immutable adapter spec without importing torch."""
    projections, norms, adapters, shapes = [], [], [], {}
    for i in range(8, 28):
        for suffix in PROJECTIONS:
            name = f"model.layers.{i}.{suffix}"
            nin, nout = PRODUCTION_SHAPES[suffix]
            require(header[name + ".weight"]["shape"] == [nout, nin], "source projection shape differs")
            projections.append({"name": name, "in_features": nin, "out_features": nout})
            for end, shape in ((".lora_A", [RANK, nin]), (".lora_B", [nout, RANK])):
                adapters.append(name + end)
                shapes[name + end] = shape
        for suffix in NORMS:
            name = f"model.layers.{i}.{suffix}"
            require(header[name]["shape"] == [NORM_WIDTHS[suffix]], "source norm shape differs")
            norms.append(name)
            shapes[name] = header[name]["shape"]
    base = sum(math.prod(v["shape"]) for v in header.values())
    adapter_count = sum(math.prod(shapes[n]) for n in adapters)
    norm_count = sum(math.prod(shapes[n]) for n in norms)
    require((base, adapter_count, norm_count) == (BASE_PARAMETERS, ADAPTER_PARAMETERS, NORM_PARAMETERS),
            "source/adapter/norm parameter counts differ from registered geometry")
    require(len(projections) == 140 and len(norms) == 80 and len(shapes) == 360,
            "registered 140 projection / 80 norm scope differs")
    return {"rank": RANK, "alpha": LORA_ALPHA, "scale": SCALE,
            "layer_start": 8, "layer_stop": 28,
            "projections": sorted(projections, key=lambda p: p["name"]),
            "norm_names": sorted(norms), "adapter_names": sorted(adapters),
            "trainable_names": sorted(shapes), "frozen_base_names": sorted(set(header) - set(norms)),
            "base_parameters": base, "adapter_parameters": adapter_count,
            "norm_parameters": norm_count, "trainable_parameters": adapter_count + norm_count,
            "augmented_parameters": base + adapter_count, "production": True}


def _training_files(binding: dict) -> dict:
    """Fresh plan/contract identities may differ; the actual training bytes must not."""
    plan = read_json(Path(binding["plan_path"]))
    require(digest(plan) == binding["plan_sha256"], "fresh training plan identity differs")
    prepared = Path(binding["prepared"]).absolute()
    contract_path = prepared / "DATA_CONTRACT.json"
    require(file_hash(contract_path) == binding["data_contract_sha256"], "fresh data contract identity differs")
    contract = read_json(contract_path)
    parent_path = Path(binding["parent_data_contract_path"])
    require(file_hash(parent_path) == _sha(binding.get("parent_data_contract_sha256"), "parent data contract"),
            "parent data contract identity differs")
    parent = read_json(parent_path)
    files = binding.get("training_files_sha256")
    require(isinstance(files, dict) and set(files) == {"microfit-train.jsonl", "ARM_BUILD.json"} and
            files == contract.get("files") == parent.get("files"), "training files differ from native control")
    for name, expected in files.items():
        require(file_hash(prepared / name) == _sha(expected, name), "actual prepared training bytes differ")
    return files


def inspect_checkpoint(source: Path, checkpoint: Path, kind: str) -> dict:
    """Verify the seal, exact final progress, source assets, scope and all tensor bytes."""
    require(kind in ("lora", "native"), "invalid checkpoint kind")
    source, checkpoint = Path(source).absolute(), Path(checkpoint).absolute()
    manifest_path = checkpoint / "manifest.json"
    manifest_sha = file_hash(manifest_path)
    manifest = unseal(read_json(manifest_path))
    expected_keys = {"schema", "experiment", "status", "binding", "progress", "rng", "entries"}
    if kind == "lora":
        expected_keys |= {"record", "adapter_spec"}
        require(manifest.get("record") == "DOSELORA_TRAINING_CHECKPOINT" and
                manifest.get("experiment") == "DOSELORA-001", "not a DOSELORA training checkpoint")
    else:
        require(manifest_sha == NATIVE_MANIFEST_SHA256, "native control manifest hash mismatch")
        require(manifest.get("experiment") == "MQ-001", "not the native control checkpoint")
    require(set(manifest) == expected_keys and type(manifest.get("schema")) is int and
            manifest["schema"] == 1 and manifest.get("status") == "COMPLETE",
            "checkpoint schema/status differs")
    require(manifest.get("progress") == FINAL_PROGRESS and
            all(type(v) is int for v in manifest["progress"].values()), "checkpoint is not at the exact final progress")
    require(isinstance(manifest["rng"], dict) and bool(manifest["rng"]), "checkpoint RNG custody is missing")
    binding = manifest["binding"]
    require(isinstance(binding, dict), "checkpoint training binding missing")
    for key in ("plan_sha256", "data_contract_sha256", "grant_sha256"):
        _sha(binding.get(key), key)
    require(isinstance(binding.get("runtime"), dict) and isinstance(binding.get("source"), dict),
            "checkpoint runtime/source binding missing")
    assets = binding["source"].get("assets")
    require(isinstance(assets, dict) and set(assets) == {"model.safetensors", *SIDECARS} and
            binding["source"].get("sha256") == digest(assets), "checkpoint source assets binding differs")
    for name, expected in assets.items():
        require(file_hash(source / name) == _sha(expected, name), f"source asset hash mismatch: {name}")
    config = read_json(source / "config.json")
    require(config.get("num_hidden_layers") == 28 and config.get("tie_word_embeddings") is True,
            "source must have registered 28-layer tied vocabulary geometry")
    header = tensor_header(source / "model.safetensors")
    expected_source = {f"model.layers.{i}.{s}.weight" for i in range(28) for s in PROJECTIONS}
    expected_source |= {f"model.layers.{i}.{s}" for i in range(28) for s in NORMS}
    expected_source |= {"model.embed_tokens.weight", "model.norm.weight"}
    require(set(header) == expected_source, "source tensor inventory differs from full 596M model")
    spec = expected_lora_spec(header)
    if kind == "lora":
        require(manifest["adapter_spec"] == spec, "checkpoint adapter spec differs from registered scope")
        recipe = binding.get("parameterization")
        required_recipe = {"rank": RANK, "alpha": LORA_ALPHA, "dropout": 0, "bias": "none",
                           "layers": list(range(8, 28)), "train_norms": True,
                           "target_modules": [p.rsplit(".", 1)[1] for p in PROJECTIONS]}
        require(isinstance(recipe, dict) and all(recipe.get(k) == v for k, v in required_recipe.items()),
                "training binding LoRA recipe differs")
        _training_files(binding)
        shapes = {n: header[n]["shape"] for n in NORM_NAMES}
        for p in spec["projections"]:
            shapes[p["name"] + ".lora_A"] = [RANK, p["in_features"]]
            shapes[p["name"] + ".lora_B"] = [p["out_features"], RANK]
    else:
        shapes = {n: header[n]["shape"] for n in DENSE_NAMES}
    entries = manifest["entries"]
    require(isinstance(entries, list) and len(entries) == len(shapes) and
            {e.get("name") for e in entries} == set(shapes), "checkpoint full tensor scope differs")
    expected_files = {"manifest.json"} | {f"tensor-{i:04d}.safetensors" for i in range(len(entries))}
    require({p.name for p in checkpoint.iterdir()} == expected_files, "checkpoint has missing or unexpected files")
    for i, entry in enumerate(entries):
        require(set(entry) == {"name", "file", "shape", "sha256"} and
                entry["file"] == f"tensor-{i:04d}.safetensors" and entry["shape"] == shapes[entry["name"]],
                "checkpoint entry filename/name/shape mismatch")
        path = checkpoint / entry["file"]
        require(file_hash(path) == _sha(entry["sha256"], "checkpoint tensor"), "checkpoint tensor hash mismatch")
        state = tensor_header(path)
        require(set(state) == {"parameter", "exp_avg", "exp_avg_sq"} and
                all(v["dtype"] == "F32" and v["shape"] == entry["shape"] for v in state.values()),
                "checkpoint parameter and optimizer states must match FP32 shape")
    return {"source": str(source), "checkpoint": str(checkpoint), "kind": kind,
            "source_assets_sha256": assets, "source_header": header, "manifest": manifest,
            "checkpoint_manifest_sha256": manifest_sha, "checkpoint_binding_sha256": digest(binding)}


def _finite_checkpoint(inspected: dict) -> None:
    import torch
    from safetensors import safe_open
    for entry in inspected["manifest"]["entries"]:
        path = Path(inspected["checkpoint"]) / entry["file"]
        require(file_hash(path) == entry["sha256"], "checkpoint tensor changed before reading")
        with safe_open(str(path), framework="pt", device="cpu") as state:
            for name in ("parameter", "exp_avg", "exp_avg_sq"):
                value = state.get_tensor(name)
                require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                        "checkpoint contains nonfinite parameter/optimizer values")
                if name == "exp_avg_sq":
                    require(bool((value >= 0).all()), "checkpoint has negative second moment")


def _parameter(inspected: dict, entries: dict, name: str):
    from safetensors import safe_open
    entry = entries[name]
    path = Path(inspected["checkpoint"]) / entry["file"]
    require(file_hash(path) == entry["sha256"], "checkpoint tensor changed before consumption")
    with safe_open(str(path), framework="pt", device="cpu") as state:
        return state.get_tensor("parameter").clone()


def merge_projection(source, a, b):
    """Registered dense FP32 merge; this computes 2BA, not factor norms."""
    import torch
    require(all(t.dtype == torch.float32 for t in (source, a, b)), "merge requires FP32 tensors")
    require(source.ndim == a.ndim == b.ndim == 2 and a.shape[0] == b.shape[1] == RANK and
            tuple(source.shape) == (b.shape[0], a.shape[1]), "LoRA merge shape mismatch")
    value = source + (b @ a) * SCALE
    require(bool(torch.isfinite(value).all()), "nonfinite merged projection")
    return value.contiguous()


def delta_square_sum(value, source) -> float:
    import torch
    require(value.shape == source.shape and value.dtype == source.dtype == torch.float32,
            "dense norm requires same-shape FP32 parameters")
    delta = value.to(torch.float64) - source.to(torch.float64)
    result = float(torch.sum(delta * delta).item())
    require(math.isfinite(result) and result >= 0, "nonfinite dense update norm")
    return result


def dense_tensor_sha256(name: str, value) -> str:
    """Canonical CPU merge identity shared with the pre-export qualification."""
    import torch
    require(value.dtype == torch.float32 and value.device.type == "cpu", "merge identity requires CPU FP32")
    hasher = hashlib.sha256(canonical([name, list(value.shape), "torch.float32"]))
    flat = value.detach().contiguous().view(torch.uint8).reshape(-1)
    for start in range(0, flat.numel(), 1024 * 1024):
        hasher.update(memoryview(flat[start:start + 1024 * 1024].numpy()))
    return hasher.hexdigest()


def scalar_value(source, trained, beta: float):
    import torch
    require(math.isfinite(beta) and BETA_MIN <= beta <= BETA_MAX, "scalar coefficient outside registered [0,2]")
    require(source.dtype == trained.dtype == torch.float32 and source.shape == trained.shape,
            "scalar update requires matching FP32 tensors")
    return source.clone() if beta == 0 else (trained.clone() if beta == 1 else source + beta * (trained - source))


def _identity(inspected: dict) -> dict:
    return {"checkpoint": inspected["checkpoint"],
            "checkpoint_manifest_sha256": inspected["checkpoint_manifest_sha256"],
            "checkpoint_binding_sha256": inspected["checkpoint_binding_sha256"],
            "checkpoint_progress": inspected["manifest"]["progress"],
            "tensor_sha256": {e["name"]: e["sha256"] for e in inspected["manifest"]["entries"]}}


def _paired_inputs(source: Path, lora_checkpoint: Path, native_checkpoint: Path) -> tuple[dict, dict]:
    lora = inspect_checkpoint(source, lora_checkpoint, "lora")
    native = inspect_checkpoint(source, native_checkpoint, "native")
    require(lora["source_assets_sha256"] == native["source_assets_sha256"], "LoRA/native sources differ")
    require(lora["manifest"]["binding"]["parent_data_contract_sha256"] ==
            native["manifest"]["binding"]["data_contract_sha256"], "LoRA data parent is not the native control contract")
    return lora, native


def _supervised_success(run: Path, binding: dict) -> dict:
    """Require observed successful termination and replay the settled budget chain."""
    require(not any((run / name).exists() or (run / name).is_symlink()
                    for name in ("failure.json", "supervisor_failure.json")),
            "training attempt has a worker/supervisor failure receipt")
    names = {"run_summary": "run_summary.json", "supervisor": "supervisor.json",
             "launch": "launch.json", "spawn": "spawn.json", "worker_claim": "worker_claim.json"}
    records = {key: read_json(run / name) for key, name in names.items()}
    summary, supervisor, launch, spawn, claim = (records[k] for k in names)
    require(summary.get("experiment") == "DOSELORA-001" and summary.get("run_id") == run.name and
            type(summary.get("returncode")) is int and summary["returncode"] == 0 and
            summary.get("worker_failed") is False and summary.get("supervisor") == supervisor,
            "training run summary is failed or disagrees with its supervisor")
    require(supervisor.get("status") == "EXITED" and type(supervisor.get("returncode")) is int and
            supervisor["returncode"] == 0 and supervisor.get("cleanup") is None and
            type(supervisor.get("child_pid")) is int and supervisor["child_pid"] > 0,
            "training supervisor did not observe a clean successful exit")
    require(launch.get("schema") == 1 and Path(launch["out"]).absolute() == run,
            "training supervisor launch points to a different output directory")
    argv = launch.get("argv")
    require(isinstance(argv, list) and all(isinstance(v, str) and v for v in argv), "invalid supervised argv")

    def option(name):
        require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), f"missing/duplicate launch option: {name}")
        return argv[argv.index(name) + 1]

    require(Path(option("--out")).absolute() == run and option("--run-id") == run.name and
            argv.count("--_worker") == 1, "training worker launch refers to a different run")
    journal = Path(option("--journal")).absolute()
    require(run.parent.name == "runs" and run.parent.parent == journal,
            "training run is outside the recorded budget journal")
    require(Path(option("--plan")).absolute() == Path(binding["plan_path"]).absolute() and
            Path(option("--prepared")).absolute() == Path(binding["prepared"]).absolute(),
            "supervised training data/plan differs from the checkpoint binding")
    launch_sha = digest(launch)
    require(spawn.get("launch_sha256") == claim.get("launch_sha256") == launch_sha and
            spawn.get("supervisor_pid") == claim.get("supervisor_pid") == launch.get("supervisor_pid") and
            spawn.get("child_pid") == supervisor["child_pid"], "training supervisor/worker identities disagree")
    require(type(claim.get("pid")) is int and claim["pid"] > 0 and
            (claim["pid"] == supervisor["child_pid"] or
             (claim.get("launcher_redirected") is True and claim.get("parent_pid") == supervisor["child_pid"])),
            "claimed training worker is not the recorded supervisor child")
    grant_path = journal / "grant.json"
    grant = read_json(grant_path)
    require(digest(grant) == binding["grant_sha256"] and grant.get("experiment") == "DOSELORA-001",
            "training budget grant differs from checkpoint binding")
    require(read_json(Path(option("--grant"))) == grant, "supervised grant differs from journal grant")
    limits = grant.get("limits")
    require(isinstance(limits, dict) and bool(limits) and
            all(type(v) is int and v >= 0 for v in limits.values()), "invalid training budget limits")
    paths = sorted((journal / "events").iterdir())
    require(0 < len(paths) <= 10000, "invalid training budget event count")
    used, active, settled, started = {key: 0 for key in limits}, {}, set(), set()
    previous, event_hashes = None, {}
    wall_keys = ("total_owned_work_wall_seconds", "model_worker_wall_seconds_and_conservative_gpu_seconds")
    for index, path in enumerate(paths):
        require(path.name == f"{index:06d}.json", "training budget event inventory is incomplete")
        raw = read_json(path); event = unseal(raw)
        require(event.get("index") == index and event.get("previous") == previous,
                "training budget event chain differs")
        charges = event.get("charges", {})
        require(isinstance(charges, dict) and not (set(charges) - set(limits)) and
                all(type(v) is int and v >= 0 for v in charges.values()), "invalid training budget charge")
        for key, value in charges.items():
            used[key] += value
        kind = event.get("kind")
        if kind == "start":
            run_id = event.get("run_id")
            require(isinstance(run_id, str) and bool(run_id) and not active and run_id not in started,
                    "training budget has overlapping or reused launches")
            active[run_id] = event; started.add(run_id)
        elif kind == "settle":
            run_id = event.get("run_id")
            require(run_id in active and run_id not in settled, "training budget settlement has no active launch")
            reservation = active.pop(run_id)
            for key in wall_keys:
                actual = event["actual_wall"][key]
                require(type(actual) is int and actual >= 0, "invalid settled training wall time")
                used[key] += actual - reservation["charges"][key]
            if run_id == run.name:
                require(event.get("receipt_sha256") == file_hash(run / "supervisor.json"),
                        "training settlement refers to a different supervisor receipt")
                wall = supervisor.get("wall_seconds")
                require(type(wall) in (float, int) and math.isfinite(wall) and wall >= 0 and
                        event["actual_wall"][wall_keys[1]] == math.ceil(wall) and
                        event["actual_wall"][wall_keys[0]] >= math.ceil(wall),
                        "settled training wall time differs from supervisor")
            settled.add(run_id)
        else:
            require(kind == "charge", "unknown training budget event")
        previous = raw["sha256"]
        event_hashes[path.name] = file_hash(path)
    require(not active and run.name in settled and not any(used[k] > limit for k, limit in limits.items()),
            "training budget is breached or has an unsettled launch")
    require({p.name for p in (journal / "runs").iterdir()} <= started,
            "training run has no budget reservation")
    state = {"used_or_reserved": used, "active_reservations": [], "budget_breached": False,
             "remaining": {key: limit - used[key] for key, limit in limits.items()}}
    require(summary.get("budget_after") == state, "training run summary budget differs from actual settled journal")
    identity = {"training_journal_path": str(journal), "training_journal_grant_sha256": file_hash(grant_path),
                "training_journal_event_sha256": event_hashes}
    for key, filename in names.items():
        identity[f"training_{key}_path"] = str(run / filename)
        identity[f"training_{key}_sha256"] = file_hash(run / filename)
    return identity


def training_qualification(lora: dict, training_result: Path | None = None) -> dict:
    checkpoint = Path(lora["checkpoint"])
    result_path = Path(training_result).absolute() if training_result else checkpoint.parent / "result.json"
    require(result_path == checkpoint.parent / "result.json", "training result is outside its supervised run")
    supervision = _supervised_success(checkpoint.parent, lora["manifest"]["binding"])
    result = read_json(result_path)
    require(result.get("record") == "DOSELORA_TRAINING_RESULT" and result.get("experiment") == "DOSELORA-001"
            and result.get("status") == "COMPLETE" and result.get("complete") is True,
            "LoRA training result has not completed successfully")
    require(Path(result["checkpoint"]).absolute() == checkpoint and
            result.get("checkpoint_manifest_sha256") == lora["checkpoint_manifest_sha256"] and
            result.get("progress") == FINAL_PROGRESS, "training result checkpoint/progress differs")
    require(result.get("binding") == lora["manifest"]["binding"] and
            result.get("adapter_spec") == lora["manifest"]["adapter_spec"],
            "training result source/adapter binding differs")
    qualification_path = Path(result["merge_qualification_path"]).absolute()
    require(qualification_path == checkpoint.parent / "merge_qualification.json", "merge qualification path differs")
    qualification_sha = file_hash(qualification_path)
    require(qualification_sha == _sha(result.get("merge_qualification_sha256"), "merge qualification"),
            "merge qualification receipt hash differs")
    qualified = read_json(qualification_path)
    require(qualified.get("schema") == 1 and qualified.get("record") == "DOSELORA_FINAL_MERGE_QUALIFICATION" and
            qualified.get("experiment") == "DOSELORA-001" and qualified.get("status") == "PASS" and
            qualified.get("complete") is True and
            qualified.get("checkpoint_manifest_sha256") == lora["checkpoint_manifest_sha256"],
            "CPU-merged native function was not qualified for this checkpoint")
    require(qualified.get("binding") == lora["manifest"]["binding"] and
            Path(qualified["checkpoint"]).absolute() == checkpoint and
            qualified.get("atol") == qualified.get("rtol") == 2e-4 and
            qualified.get("argmax_equality_required") is True and
            qualified.get("original_modules_restored") is True,
            "merge qualification source/scope/tolerances differ")
    frozen_sha = _sha(qualified.get("frozen_base_sha256_before_after"), "qualified frozen base")
    require(result.get("frozen_base_sha256_before_after") == frozen_sha, "training/qualification frozen base identity differs")
    records = qualified.get("inputs")
    expected_labels = {"reply_7", "first_training_row", "synthetic_2048_arange"}
    require(isinstance(records, list) and len(records) == 3 and
            {r.get("input") for r in records} == expected_labels, "merge qualification input set differs")
    training_path = Path(lora["manifest"]["binding"]["prepared"]) / "microfit-train.jsonl"
    with regular(training_path).open(encoding="utf-8") as stream:
        first_ids = decode(stream.readline())["ids"]
    lengths = {"reply_7": 31, "first_training_row": len(first_ids), "synthetic_2048_arange": 2048}
    expected_ids = {"first_training_row": digest(first_ids), "synthetic_2048_arange": digest(list(range(2048)))}
    require(1 <= len(first_ids) <= 2048, "first training qualification input length differs")
    for row in records:
        n = lengths[row["input"]]
        require(row.get("input_token_count") == n and row.get("checked_last_positions") == min(16, n) and
                row.get("checked_logits_shape") == [1, min(16, n), 151936] and row.get("passed") is True and
                type(row.get("tolerance_violations")) is int and row["tolerance_violations"] == 0 and
                type(row.get("argmax_mismatches")) is int and row["argmax_mismatches"] == 0,
                "merge qualification input did not pass the fixed comparisons")
        _sha(row.get("input_ids_sha256"), "qualification input")
        if row["input"] in expected_ids:
            require(row["input_ids_sha256"] == expected_ids[row["input"]], "merge qualification input tokens differ")
        difference = row.get("max_abs_difference")
        require(type(difference) in (float, int) and math.isfinite(difference) and difference >= 0,
                "merge qualification numerical evidence is invalid")
    require(qualified.get("budget_charges") == {
        "processed_training_tokens_all_attempts": 2 * sum(lengths.values()),
        "evaluation_and_qualification_model_calls": 6}, "merge qualification accounting differs")
    hashes = qualified.get("dense_tensor_sha256")
    require(isinstance(hashes, dict) and set(hashes) == DENSE_NAMES, "merge qualification omits dense tensor scope")
    for name, value in hashes.items():
        _sha(value, name)
    return {"training_result_path": str(result_path), "training_result_sha256": file_hash(result_path),
            "merge_qualification_path": str(qualification_path), "merge_qualification_sha256": qualification_sha,
            "merged_tensor_sha256": hashes, **supervision}


def _output_size(inspected: dict) -> int:
    return sum(math.prod(t["shape"]) * 4 for t in inspected["source_header"].values()) + 4 * 1024**2


def _export(out: Path, tensors: dict, inspected: dict, alpha: float, extras: dict) -> dict:
    """Publish an arm only after all tensors, norms and provenance pass."""
    from safetensors.torch import save_file
    _fresh(out)
    _disk(out, _output_size(inspected))
    new_dir(out)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    for name in SIDECARS:
        with regular(Path(inspected["source"]) / name).open("rb") as src, (out / name).open("xb") as dst:
            shutil.copyfileobj(src, dst)
        require(file_hash(out / name) == inspected["source_assets_sha256"][name], "copied sidecar identity changed")
    weight_sha = file_hash(out / "model.safetensors")
    manifest = inspected["manifest"]
    applied = 0 if extras.get("parameterization") == "native_scalar_norm_match" and alpha == 0 else len(DENSE_NAMES)
    receipt = {"schema": 1, "record": "CHECKPOINT_MATERIALIZATION", "layout": "huggingface",
               "experiment": "DOSELORA-001", "alpha": alpha,
               "source": inspected["source"], "source_assets_sha256": inspected["source_assets_sha256"],
               "source_weight_sha256": inspected["source_assets_sha256"]["model.safetensors"],
               "checkpoint": inspected["checkpoint"], "checkpoint_experiment": manifest["experiment"],
               "checkpoint_manifest_sha256": inspected["checkpoint_manifest_sha256"],
               "checkpoint_binding": manifest["binding"], "checkpoint_binding_sha256": inspected["checkpoint_binding_sha256"],
               "checkpoint_progress": manifest["progress"], "checkpoint_entries": manifest["entries"],
               "out_weight_sha256": weight_sha, "arm_weight_sha256": weight_sha,
               "arm_weight_bytes": (out / "model.safetensors").stat().st_size,
               "stored_dtype": "float32", "interpolation_dtype": "float32", "bf16_interpolation_used": False,
               "tensors_in_checkpoint": len(manifest["entries"]), "tensors_total": len(tensors),
               "tensors_applied": applied, "source_tensors_untouched_count": len(tensors) - applied,
               "trainable_layers": list(range(8, 28)), "sidecars_copied": list(SIDECARS),
               "builder_sha256": file_hash(Path(__file__)), "resource_lock": str(resource_lock_path()),
               "quality_claim": False, "release_permitted": False, **extras}
    write_new(out / "MATERIALIZATION.json", receipt)
    write_new(out / "ARM.json", {**receipt, "record": "DOSELORA_ARM"})
    return receipt


def materialize_lora(source: Path, checkpoint: Path, native_checkpoint: Path,
                     out: Path, norm_record: Path, training_result: Path | None = None) -> dict:
    out, norm_record = Path(out).absolute(), Path(norm_record).absolute()
    _fresh(out); _fresh(norm_record)
    require(not norm_record.is_relative_to(out), "norm record must live outside the arm directory")
    with lease(resource_lock_path()):
        available = require_ram()
        lora, native = _paired_inputs(source, checkpoint, native_checkpoint)
        qualification = training_qualification(lora, training_result)
        _disk(out, _output_size(lora))
        import torch
        from safetensors import safe_open
        torch.set_num_threads(1)
        _finite_checkpoint(lora); _finite_checkpoint(native)
        le = {e["name"]: e for e in lora["manifest"]["entries"]}
        ne = {e["name"]: e for e in native["manifest"]["entries"]}
        tensors, lsq, nsq = {}, [], []
        with safe_open(str(Path(source) / "model.safetensors"), framework="pt", device="cpu") as source_file:
            for name in source_file.keys():
                src = source_file.get_tensor(name).to(torch.float32).contiguous()
                require(bool(torch.isfinite(src).all()), "source contains nonfinite values")
                if name not in DENSE_NAMES:
                    tensors[name] = src.clone()
                    continue
                if name in NORM_NAMES:
                    merged = _parameter(lora, le, name)
                else:
                    prefix = name.removesuffix(".weight")
                    merged = merge_projection(src, _parameter(lora, le, prefix + ".lora_A"),
                                              _parameter(lora, le, prefix + ".lora_B"))
                trained = _parameter(native, ne, name)
                require(dense_tensor_sha256(name, merged) == qualification["merged_tensor_sha256"][name],
                        "materialized dense tensor differs from qualified CPU merge")
                lsq.append(delta_square_sum(merged, src)); nsq.append(delta_square_sum(trained, src))
                tensors[name] = merged.contiguous()
                del trained
        require(len(lsq) == len(nsq) == 220, "dense norm omitted part of the 220-tensor scope")
        lnorm, nnorm = math.sqrt(math.fsum(lsq)), math.sqrt(math.fsum(nsq))
        require(nnorm > 0, "native control has zero update norm")
        beta = lnorm / nnorm
        require(math.isfinite(beta), "nonfinite norm ratio")
        require(_paired_inputs(source, checkpoint, native_checkpoint) == (lora, native),
                "source/checkpoint changed during merge")
        require(training_qualification(lora, training_result) == qualification,
                "training/merge qualification changed during materialization")
        norm = {"schema": 1, "record": "DOSELORA_NORM_MATCH", "experiment": "DOSELORA-001",
                "status": "PASS" if BETA_MIN <= beta <= BETA_MAX else "BLOCKED_NORM_RATIO_OUT_OF_RANGE",
                "source": lora["source"], "source_assets_sha256": lora["source_assets_sha256"],
                "source_weight_sha256": lora["source_assets_sha256"]["model.safetensors"],
                "lora_checkpoint": _identity(lora), "native_checkpoint": _identity(native),
                "dense_updated_tensor_names": sorted(DENSE_NAMES), "dense_updated_tensor_count": 220,
                "projection_count": 140, "trained_norm_count": 80, "lora_rank": RANK, "lora_alpha": LORA_ALPHA,
                "lora_dense_delta_norm": lnorm, "native_dense_delta_norm": nnorm, "beta": beta,
                "beta_admissible_interval": [BETA_MIN, BETA_MAX], "relative_match_tolerance": MATCH_TOLERANCE,
                "norm_definition": "FP64 sum of squared actual FP32 merged-parameter minus FP32-upcast source, over all 220 dense updated tensors",
                "available_host_bytes_before": available, "builder_sha256": file_hash(Path(__file__)),
                "quality_claim": False, "release_permitted": False, **qualification}
        if norm["status"] != "PASS":
            del tensors
            norm_record.parent.mkdir(parents=True, exist_ok=True)
            write_new(norm_record, seal(norm))
            return norm
        receipt = _export(out, tensors, lora, 1.0,
                          {"parameterization": "lora", "lora_rank": RANK, "lora_alpha": LORA_ALPHA,
                           "formula": "projection = FP32(source + 2 * B @ A); trained norms copied directly",
                           "dense_delta_norm": lnorm, "norm_matched_scalar_beta": beta, **qualification})
        del tensors
        norm.update(lora_arm_dir=str(out), lora_arm_receipt_sha256=file_hash(out / "ARM.json"),
                    lora_materialization_sha256=file_hash(out / "MATERIALIZATION.json"),
                    lora_weight_sha256=receipt["out_weight_sha256"])
        norm_record.parent.mkdir(parents=True, exist_ok=True)
        write_new(norm_record, seal(norm))
        return norm


def validate_norm_record(source: Path, native_checkpoint: Path, norm_record: Path) -> tuple[dict, dict, dict]:
    norm = unseal(read_json(norm_record))
    require(norm.get("schema") == 1 and norm.get("record") == "DOSELORA_NORM_MATCH" and
            norm.get("experiment") == "DOSELORA-001" and norm.get("status") == "PASS", "norm record is not admissible")
    beta = norm.get("beta")
    require(type(beta) in (float, int) and math.isfinite(beta) and BETA_MIN <= beta <= BETA_MAX,
            "norm record coefficient outside registered [0,2]")
    require(norm.get("relative_match_tolerance") == MATCH_TOLERANCE and
            norm.get("beta_admissible_interval") == [BETA_MIN, BETA_MAX], "norm record alters prospective limits")
    lora, native = _paired_inputs(source, Path(norm["lora_checkpoint"]["checkpoint"]), native_checkpoint)
    qualification = training_qualification(lora, Path(norm["training_result_path"]))
    require(all(norm.get(k) == v for k, v in qualification.items()), "norm record merge qualification differs")
    require(norm["lora_checkpoint"] == _identity(lora) and norm["native_checkpoint"] == _identity(native),
            "norm record checkpoint identities differ")
    require(norm.get("source") == lora["source"] and norm.get("source_assets_sha256") == lora["source_assets_sha256"] and
            norm.get("source_weight_sha256") == lora["source_assets_sha256"]["model.safetensors"],
            "norm record source identity differs")
    require(norm.get("dense_updated_tensor_names") == sorted(DENSE_NAMES) and
            norm.get("dense_updated_tensor_count") == 220 and norm.get("projection_count") == 140 and
            norm.get("trained_norm_count") == 80, "norm record scope differs")
    ln, nn = norm.get("lora_dense_delta_norm"), norm.get("native_dense_delta_norm")
    require(type(ln) in (float, int) and type(nn) in (float, int) and math.isfinite(ln) and math.isfinite(nn)
            and ln >= 0 and nn > 0 and beta == ln / nn, "norm record ratio disagrees with norms")
    arm = Path(norm["lora_arm_dir"])
    for name, key in (("ARM.json", "lora_arm_receipt_sha256"), ("MATERIALIZATION.json", "lora_materialization_sha256"),
                      ("model.safetensors", "lora_weight_sha256")):
        require(file_hash(arm / name) == _sha(norm.get(key), key), "norm record LoRA arm identity differs")
    for name in ("ARM.json", "MATERIALIZATION.json"):
        receipt = read_json(arm / name)
        require(receipt.get("checkpoint_manifest_sha256") == lora["checkpoint_manifest_sha256"] and
                receipt.get("checkpoint_binding_sha256") == lora["checkpoint_binding_sha256"] and
                receipt.get("source_assets_sha256") == lora["source_assets_sha256"] and
                receipt.get("arm_weight_sha256") == norm["lora_weight_sha256"] and
                receipt.get("dense_delta_norm") == ln, "LoRA receipt disagrees with norm/checkpoint identity")
    return norm, lora, native


def materialize_scalar(source: Path, native_checkpoint: Path, out: Path, norm_record: Path) -> dict:
    out, norm_record = Path(out).absolute(), Path(norm_record).absolute()
    _fresh(out)
    with lease(resource_lock_path()):
        available = require_ram()
        norm_sha = file_hash(norm_record)
        norm, lora, native = validate_norm_record(source, native_checkpoint, norm_record)
        _disk(out, _output_size(native))
        import torch
        from safetensors import safe_open
        torch.set_num_threads(1)
        _finite_checkpoint(native)
        entries = {e["name"]: e for e in native["manifest"]["entries"]}
        tensors, sums, native_sums = {}, [], []
        beta = norm["beta"]
        with safe_open(str(Path(source) / "model.safetensors"), framework="pt", device="cpu") as source_file:
            for name in source_file.keys():
                src = source_file.get_tensor(name).to(torch.float32).contiguous()
                require(bool(torch.isfinite(src).all()), "source contains nonfinite values")
                if name not in DENSE_NAMES:
                    tensors[name] = src.clone()
                    continue
                trained = _parameter(native, entries, name)
                value = scalar_value(src, trained, beta).contiguous()
                require(bool(torch.isfinite(value).all()), "nonfinite scalar parameter")
                sums.append(delta_square_sum(value, src)); native_sums.append(delta_square_sum(trained, src))
                tensors[name] = value
                del trained
        measured = math.sqrt(math.fsum(sums))
        require(len(sums) == 220 and math.sqrt(math.fsum(native_sums)) == norm["native_dense_delta_norm"],
                "native dense norm differs from recorded norm")
        target = norm["lora_dense_delta_norm"]
        relative = abs(measured - target) / target if target else (0.0 if measured == 0 else math.inf)
        if not math.isfinite(relative) or relative > MATCH_TOLERANCE:
            blocked = {"schema": 1, "record": "DOSELORA_SCALAR_NORM_CHECK", "status": "BLOCKED_NORM_MATCH",
                       "norm_record_sha256": norm_sha, "beta": beta, "target_norm": target,
                       "measured_norm": measured, "relative_discrepancy": relative if math.isfinite(relative) else None,
                       "tolerance": MATCH_TOLERANCE, "coefficient_search_performed": False, "quality_claim": False}
            path = out.parent / (out.name + "-NORM_MATCH_BLOCKED.json")
            _fresh(path); path.parent.mkdir(parents=True, exist_ok=True); write_new(path, seal(blocked))
            return blocked
        require(file_hash(norm_record) == norm_sha and
                validate_norm_record(source, native_checkpoint, norm_record) == (norm, lora, native),
                "norm/source/checkpoint identity changed during scalar construction")
        verification = {"norm_record_path": str(norm_record), "norm_record_sha256": norm_sha,
                        "target_lora_dense_delta_norm": target, "dense_delta_norm": measured,
                        "relative_norm_discrepancy": relative, "relative_norm_tolerance": MATCH_TOLERANCE,
                        "coefficient_search_performed": False, "available_host_bytes_before": available}
        receipt = _export(out, tensors, native, beta,
                          {"parameterization": "native_scalar_norm_match",
                           "formula": "FP32(source + beta * (native_FP32_parameter - source)); endpoints copied",
                           **verification})
        del tensors
        write_new(out / "NORM_MATCH_VERIFICATION.json", seal({"schema": 1, "status": "PASS", **verification}))
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("lora", "scalar"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="sealed LoRA checkpoint (lora mode only)")
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--norm-record", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, help="LoRA completed result (default checkpoint.parent/result.json)")
    args = parser.parse_args()
    if args.mode == "lora":
        if args.checkpoint is None:
            parser.error("--checkpoint is required in lora mode")
        result = materialize_lora(args.source, args.checkpoint, args.native_checkpoint, args.out, args.norm_record,
                                  args.training_result)
    else:
        if args.checkpoint is not None:
            parser.error("scalar mode takes its LoRA checkpoint identity from --norm-record")
        if args.training_result is not None:
            parser.error("scalar mode takes its training-result identity from --norm-record")
        result = materialize_scalar(args.source, args.native_checkpoint, args.out, args.norm_record)
    status = result.get("status", "COMPLETE")
    print(json.dumps({"status": status, "out": str(args.out), "norm_record": str(args.norm_record),
                      **{k: result[k] for k in ("beta", "alpha", "dense_delta_norm", "relative_norm_discrepancy") if k in result}},
                     allow_nan=False))
    return 2 if status.startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
