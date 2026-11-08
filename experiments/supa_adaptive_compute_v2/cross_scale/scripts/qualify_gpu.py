"""Qualify streamed FP32 CUDA capture, actual donor deltas and bridge gradients."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROGRAMME = ROOT.parent
sys.path.insert(0, str(PROGRAMME))

from shared.hashing import canonical_sha256, sha256_file, source_files, write_json
from shared.guards import guard_non_sealed_path
from shared.manifests import new_run, utc_now, verify_registration
from cross_scale.scripts.run_alignment import calibration_rows, configure
from cross_scale.src.gpu_qualification import bridge_gradients, fixture_manifest, qualification_fixtures, tensor_agreement


def execute(args):
    import psutil
    import torch
    from transformers import AutoTokenizer
    from shared.hidden_capture import capture_native
    from shared.integrity import checkpoint_files, state_hash
    from shared.model_loading import require_available_memory, resolve_model
    from shared.runtime import configure_runtime, cuda_memory, runtime_identity
    from cross_scale.src.bridge_fit import fit_initial
    from cross_scale.src.delta_alignment import deadline, donor_delta, frozen, token_masks
    from cross_scale.src.donor_layers import donor_identity, load_donor_layer, strip_calibration_rows

    config = configure(args.config)
    if config["device"] != "cuda:0":
        raise ValueError("this qualifier requires the separately frozen CUDA configuration")
    registration = verify_registration("cross_scale", args.config)
    dataset = guard_non_sealed_path(args.dataset_manifest)
    relative = dataset.relative_to(PROGRAMME.parents[1]).as_posix()
    if registration["dataset_manifests"].get(relative) != sha256_file(dataset):
        raise ValueError("qualification requires this track's registered calibration manifest")
    configure_runtime(config)
    details = {"runtime_qualification": config["runtime_qualification"],
        "config_sha256": canonical_sha256(config), "registration_sha256": registration["registration_sha256"],
        "dataset_manifest_sha256": sha256_file(dataset), "sources": source_files("cross_scale"),
        "seed": config["seed"], "threads": config["threads"], "device": config["device"],
        "dtype": config["dtype"], "cache_mode": "none", "runtime_identity": runtime_identity(config),
        "pid": __import__("os").getpid(), "qualification_pass": False, "generation_count": 0}
    output, receipt = new_run("cross_scale", "00_runtime_qualification", details)
    started = time.perf_counter()
    try:
        receipt["admission"] = require_available_memory(config)
        selected = qualification_fixtures(calibration_rows(dataset), config)
        prompt_rows = strip_calibration_rows(selected)
        rows_receipt = fixture_manifest([{**row, "alignment_partition": selected[index]["alignment_partition"]}
                                       for index, row in enumerate(prompt_rows)])
        supa_path = resolve_model()
        supa_before = checkpoint_files(supa_path)
        donor_path, donor_before = donor_identity(config)
        tokenizer = AutoTokenizer.from_pretrained(supa_path, local_files_only=True, trust_remote_code=True)
        donor_tokenizer = AutoTokenizer.from_pretrained(donor_path, local_files_only=True)
        if tokenizer.get_vocab() != donor_tokenizer.get_vocab():
            raise ValueError("qualification token vocabularies differ")
        encoded = []
        for row in prompt_rows:
            text = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            if ids.numel() > config["calibration_max_tokens"]:
                raise ValueError("qualification prompt exceeds token budget; truncation forbidden")
            if not torch.equal(ids, donor_tokenizer(text, return_tensors="pt")["input_ids"]):
                raise ValueError("qualification token IDs differ")
            encoded.append(ids)
        special = sorted(set(tokenizer.all_special_ids)|set(tokenizer.get_added_vocab().values()))
        del tokenizer, donor_tokenizer
        gc.collect()

        def progress(index, maximum):
            deadline(started, config["max_capture_seconds"])
            print(f"Qualification capture layer {index}/{maximum}", flush=True)

        captures = {}
        donor_depths = tuple(sorted({depth for index in config["donor_layers"] for depth in (index-1, index)}))
        for device in ("cpu", config["device"]):
            captures[device] = {}
            for name, path, depths in (("supa", supa_path, tuple(config["supa_layers"])),
                                       ("donor", donor_path, donor_depths)):
                print(f"Qualification {name} streamed capture on {device}", flush=True)
                options = {"progress": progress}
                if device != "cpu":
                    options["device"] = device
                captures[device][name] = capture_native(path, encoded, depths, config["max_rss_gib"], **options)
                if device != "cpu":
                    repeated = capture_native(path, encoded, depths, config["max_rss_gib"], **options)
                    if (set(repeated) != set(captures[device][name]) or any(
                            len(repeated[depth]) != len(captures[device][name][depth]) or any(
                                not torch.equal(left, right) for left, right in
                                zip(repeated[depth], captures[device][name][depth])) for depth in repeated)):
                        raise RuntimeError("same-GPU streamed capture is not bit-identical")
                    del repeated
        comparisons = []
        thresholds = config["qualification"]

        def compare(label, reference, actual, ids=None):
            masks = token_masks([ids], special) if ids is not None else {"all": None}
            for stratum, mask in masks.items():
                if mask is not None and not mask.any():
                    comparisons.append({"label": label, "stratum": stratum, "tokens": 0, "pass": True})
                    continue
                metrics = tensor_agreement(reference if mask is None else reference[mask],
                                           actual if mask is None else actual[mask], thresholds)
                comparisons.append({"label": label, "stratum": stratum, **metrics})

        for name in ("supa", "donor"):
            for depth, states in captures["cpu"][name].items():
                for index, reference in enumerate(states):
                    compare(f"{name}_L{depth}_{prompt_rows[index]['example_id']}", reference,
                            captures[config["device"]][name][depth][index], encoded[index])
        fit_indices = [index for index, row in enumerate(selected) if row["alignment_partition"] == "fit"]
        held_indices = [index for index, row in enumerate(selected) if row["alignment_partition"] == "holdout"]
        cpu_config = {**config, "device": "cpu"}
        layer_integrity = []
        for donor_index in config["donor_layers"]:
            cpu_layer, cpu_rope, _ = load_donor_layer(cpu_config, donor_index, donor_path)
            gpu_layer, gpu_rope, _ = load_donor_layer(config, donor_index, donor_path)
            cpu_before, gpu_before = state_hash(cpu_layer), state_hash(gpu_layer)
            cpu_rope_before, gpu_rope_before = state_hash(cpu_rope), state_hash(gpu_rope)
            if cpu_before != gpu_before:
                raise RuntimeError("CPU and GPU donor parameter bytes differ before qualification")
            for supa_index in config["supa_layers"]:
                train_small = torch.cat([captures["cpu"]["supa"][supa_index][index] for index in fit_indices])
                train_big = torch.cat([captures["cpu"]["donor"][donor_index-1][index] for index in fit_indices])
                scenarios = [prompt_rows[index]["scenario_id"] for index in fit_indices for _ in range(encoded[index].numel())]
                bridge = fit_initial(train_small, train_big, "B0", config, scenarios)
                gpu_bridge = __import__("copy").deepcopy(bridge).to(config["device"])
                for index in held_indices:
                    with torch.inference_mode():
                        cpu_z = bridge.project(captures["cpu"]["supa"][supa_index][index].unsqueeze(0))
                        gpu_z = gpu_bridge.project(captures[config["device"]]["supa"][supa_index][index].to(config["device"]).unsqueeze(0))
                        cpu_delta = donor_delta(cpu_layer, cpu_rope, cpu_z, config=config)[0].cpu()
                        gpu_delta = donor_delta(gpu_layer, gpu_rope, gpu_z, config=config)[0].cpu()
                        repeat = donor_delta(gpu_layer, gpu_rope, gpu_z, config=config)[0].cpu()
                        if not torch.equal(gpu_delta, repeat):
                            raise RuntimeError("same-GPU donor forward is not bit-identical")
                    compare(f"S{supa_index}_D{donor_index}_B0_delta_{prompt_rows[index]['example_id']}",
                            cpu_delta, gpu_delta, encoded[index])
                if supa_index == thresholds["gradient_supa_layer"]:
                    index = fit_indices[0]
                    prompt = {"small": captures["cpu"]["supa"][supa_index][index],
                              "big": captures["cpu"]["donor"][donor_index-1][index],
                              "delta": captures["cpu"]["donor"][donor_index][index]-captures["cpu"]["donor"][donor_index-1][index]}
                    for objective in thresholds["bridge_gradient_objectives"]:
                        normalized = objective == "B3"
                        initial = fit_initial(train_small, train_big, "B1" if normalized else "B0", config, scenarios)
                        cpu_gradient = bridge_gradients(initial, cpu_layer, cpu_rope, prompt, config, normalized)
                        gpu_gradient = bridge_gradients(initial, gpu_layer, gpu_rope, prompt, config, normalized)
                        repeat_gradient = bridge_gradients(initial, gpu_layer, gpu_rope, prompt, config, normalized)
                        if (gpu_gradient["loss"] != repeat_gradient["loss"] or any(
                                not torch.equal(gpu_gradient[key], repeat_gradient[key]) for key in ("up", "bias"))):
                            raise RuntimeError("same-GPU bridge gradients are not bit-identical")
                        for key in ("up", "bias"):
                            # Gradients are one parameter vector: tiny nearly-zero rows
                            # cannot dilute or dominate a token-stratified diagnostic.
                            compare(f"S{supa_index}_D{donor_index}_{objective}_gradient_{key}",
                                    cpu_gradient[key].reshape(1, -1), gpu_gradient[key].reshape(1, -1))
                del bridge, gpu_bridge
                deadline(started, config["max_capture_seconds"])
            if (state_hash(cpu_layer) != cpu_before or state_hash(gpu_layer) != gpu_before or
                    state_hash(cpu_rope) != cpu_rope_before or state_hash(gpu_rope) != gpu_rope_before):
                raise RuntimeError("donor weights changed during runtime qualification")
            frozen(cpu_layer); frozen(gpu_layer); frozen(cpu_rope); frozen(gpu_rope)
            layer_integrity.append({"donor_layer": donor_index, "state_sha256": cpu_before,
                                    "CPU_GPU_parameter_bytes_equal": True, "frozen_unchanged": True,
                                    "cpu_rope_sha256": cpu_rope_before, "gpu_rope_sha256": gpu_rope_before,
                                    "backbone_gradients": 0, "trainable_backbone_parameters": 0})
            del cpu_layer, gpu_layer, cpu_rope, gpu_rope
            gc.collect()
        if donor_identity(config)[1] != donor_before or checkpoint_files(supa_path) != supa_before:
            raise RuntimeError("model checkpoints changed during qualification")
        if source_files("cross_scale") != details["sources"]:
            raise RuntimeError("qualification sources changed during execution")
        passed = bool(comparisons) and all(row["pass"] for row in comparisons)
        report = {"fixtures": rows_receipt, "thresholds": thresholds, "comparisons": comparisons,
                  "layer_integrity": layer_integrity, "supa_files": supa_before,
                  "donor_identity": donor_before, "qualification_pass": passed,
                  "same_gpu_donor_delta_repeat_bit_identical": True, "generation_count": 0,
                  "same_gpu_streamed_capture_repeat_bit_identical": True,
                  "same_gpu_bridge_gradient_repeat_bit_identical": True,
                  "scope": "Track B Stage 1 streamed capture, donor delta and bridge gradient runtime only"}
        from cross_scale.src.gpu_qualification import validate_qualification_report
        if passed:
            validate_qualification_report(report, config, calibration_rows(dataset))
        write_json(output/"report.json", report)
        receipt.update(report_path="report.json", report_sha256=sha256_file(output/"report.json"),
                       qualification_pass=passed, canonical_unchanged=True)
        if not passed:
            raise RuntimeError("prospective CPU/GPU Track B numerical qualification failed")
        receipt["status"] = "COMPLETE"
    except BaseException as error:
        receipt["status"] = "FAILED"
        write_json(output/"failure.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        receipt["failure_receipt"] = "failure.json"
        raise
    finally:
        memory = psutil.Process().memory_info()
        receipt.update(end_utc=utc_now(), wall_seconds=time.perf_counter()-started,
                       peak_rss_bytes=getattr(memory, "peak_wset", memory.rss))
        try:
            receipt["cuda_memory"] = cuda_memory(config)
        except BaseException as memory_error:
            receipt["cuda_memory_error"] = {"type": type(memory_error).__name__, "error": str(memory_error)}
            if receipt["status"] != "FAILED":
                receipt["status"] = "FAILED"
                receipt["qualification_pass"] = False
                write_json(output/"failure.json", {"type": type(memory_error).__name__,
                           "error": str(memory_error), "traceback": traceback.format_exc()})
                receipt["failure_receipt"] = "failure.json"
        write_json(output/"run_manifest.json", receipt)
    if receipt["status"] != "COMPLETE":
        raise RuntimeError("qualification terminal resource check failed; failure receipt preserved")
    print(f"COMPLETE {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/alignment_gpu_v2_1.json")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    args = parser.parse_args()
    from shared.timing import machine_lease
    with machine_lease():
        execute(args)


if __name__ == "__main__":
    main()
