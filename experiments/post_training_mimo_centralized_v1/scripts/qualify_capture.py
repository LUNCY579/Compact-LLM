"""One contained, bare-Titan short greedy capture qualification; never DEV/E1."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
from experiments.post_training_mimo_centralized_v1.scripts import greedy_capture as control


def run(args):
    started = time.monotonic()
    cfg = control.load_config(args.config)
    while not args.go_path.exists() and time.monotonic() - started < 30:
        time.sleep(.05)
    control.require(args.go_path.exists(), "capture supervisor GO missing")
    control.require(control.parse(args.go_path.read_bytes()) == {"config_sha256": control.CONFIG_SHA256}, "capture GO mismatch")
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    control.require(not output.exists(), "capture output must be new")
    snapshot = control.source_snapshot(root, cfg)
    control.require(control.file_identity(root / cfg["qualified_core_config"])["sha256"] == cfg["qualified_core_config_sha256"], "CE512 config drift")
    reference_path = control.PACKAGE / cfg["qualified_CE512_reference"]
    control.require(control.file_identity(reference_path)["sha256"] == cfg["qualified_CE512_reference_sha256"], "CE512 reference drift")
    reference = control.parse(reference_path.read_bytes())
    control.require(reference["status"] == "PASS_NATIVE_UNPADDED_CE512_QUALIFICATION_ONLY", "CE512 reference not passed")
    output.mkdir(parents=True)
    result = {"schema_version": 1, "experiment_id": cfg["experiment_id"], "status": "RUNNING",
              "runtime_scope": cfg["runtime_scope"], "resource_profile": cfg["resource_profile"], "config_sha256": control.CONFIG_SHA256,
              "code_sha": snapshot["code_sha"], "dirty_at_launch": False, "evidence_level": "MEASURED / REPRODUCED",
              "seed": cfg["seed"], "qualified_CE512_reference_sha256": cfg["qualified_CE512_reference_sha256"],
              "pretrained_loads": 0, "optimizer_steps": 0, "training_tokens": 0, "generation_tokens": 0,
              "teacher_calls": 0, "checks": {}, "resources": [], "fixtures": [], "accounting_version": 2,
              "native_evaluation_bank_access": False, "language_grading_authorized": False, "E1_authorized": False,
              "CHECK_FINAL_access": False}
    control.write(output / "CONFIG.json", cfg)
    control.write(output / "SOURCE_RECEIPT.json", snapshot)
    def event(kind, **values):
        record = control.safe_numbers({"elapsed_seconds": time.monotonic() - started, "event": kind, **values})
        with (output / "EVENTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(control.canonical(record).decode() + "\n"); stream.flush(); os.fsync(stream.fileno())
    # Reused sources were byte/commit checked before these imports.
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting, tensor_digest
    from experiments.post_training_mimo_centralized_v1.scripts import stream_checkpoint
    accounting = CallAccounting(event, maximum_optimizer_calls=0)
    calls = control.BudgetedCalls(accounting, cfg)
    result["call_accounting"] = accounting.snapshot()
    try:
        cpu = control.run_cpu_controls()
        cpu.update(helper_sha256=snapshot["source_files"][control.NEW_SOURCE_FILES[0]]["sha256"],
                   test_source_sha256=snapshot["source_files"][control.CPU_TEST_SOURCE]["sha256"])
        control.write(output / "CPU_CONTROL_RECEIPT.json", cpu)
        event("runtime_import_start")
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        control.require(Path(sys.executable).resolve() == Path(cfg["runtime"]["executable"]).resolve()
                        and sys.version.startswith(cfg["runtime"]["python"] + " "), "capture Python runtime drift")
        packages = {name: importlib.metadata.version(name) for name in cfg["runtime"]["packages"]}
        control.require(packages == cfg["runtime"]["packages"], "capture package versions drift")
        import psutil
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        event("runtime_import_complete")
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        control.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.version.cuda == cfg["runtime"]["cuda"], "BF16 CUDA runtime unavailable/drifted")
        runtime = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": packages,
                   "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                   "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "attention": "eager", "base_dtype": "bfloat16", "adapter": None, "use_cache": False,
                   "inference_context": "torch.no_grad()", "torch_threads": 2, "allow_tf32": False, "deterministic_algorithms": True}
        control.write(output / "RUNTIME_RECEIPT.json", runtime)
        process = psutil.Process()
        def guard(stage, admission=False):
            torch.cuda.synchronize(); free, total = torch.cuda.mem_get_info()
            sample = {"stage": stage, "elapsed_seconds": time.monotonic() - started,
                      "host_available_gib": psutil.virtual_memory().available / 2**30,
                      "rss_gib": process.memory_info().rss / 2**30, "gpu_free_gib": free / 2**30,
                      "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                      "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "disk_free_gib": psutil.disk_usage(str(output)).free / 2**30}
            sample["admission_required_gib"] = control.admission_required(cfg, sample["rss_gib"])
            result["resources"].append(sample); event("resource", **sample)
            control.require(sample["rss_gib"] <= cfg["worker_rss_limit_gib"], "CAPTURE_RSS_GUARD")
            control.require(sample["host_available_gib"] >= (sample["admission_required_gib"] if admission else cfg["worker_free_gib"]), "HOST_RAM_GUARD")
            control.require(sample["gpu_free_gib"] >= (cfg["gpu_free_gib"] if admission else cfg["gpu_worker_free_gib"]), "GPU_WORKER_GUARD")
            control.require(sample["disk_free_gib"] >= cfg["disk_free_gib"], "DISK_GUARD")
            control.require(time.monotonic() - started <= cfg["wall_seconds"], "CAPTURE_WALL_CEILING")
        guard("preload", True)
        model_dir = Path(cfg["model_dir"])
        asset_pins = {**cfg["asset_sha256"], "model.safetensors": cfg["weight_sha256"]}
        for name, sha in asset_pins.items():
            control.require(control.file_identity(model_dir / name)["sha256"] == sha, "model asset drift: " + name)
        primary_path = root / cfg["primary_receipt"].replace("\\", "/")
        control.require(control.file_identity(primary_path)["sha256"] == cfg["primary_receipt_sha256"], "primary receipt drift")
        primary = control.parse(primary_path.read_bytes())
        control.require(primary["weight_sha256"] == cfg["weight_sha256"] and primary["parameter_count"] == 596049920
                        and Path(primary["checkpoint_path"]).resolve() == model_dir.resolve(), "primary checkpoint identity")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        template = control.parse((model_dir / "tokenizer_config.json").read_bytes())["chat_template"]
        control.require(tokenizer.chat_template == template and tokenizer.eos_token_id == 151645, "tokenizer/template identity")
        roundtrip = "Qualification: 7 + 5 = 12."
        raw_ids = tokenizer.encode(roundtrip, add_special_tokens=False)
        control.require(tokenizer.decode(raw_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == roundtrip, "raw tokenizer roundtrip")
        control.write(output / "TOKENIZER_RECEIPT.json", {"class_name": type(tokenizer).__name__, "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "eos_id": tokenizer.eos_token_id, "original_pad_id": tokenizer.pad_token_id, "raw_ids": raw_ids, "raw_roundtrip": True,
            "enable_thinking": False, "add_generation_prompt": True, "skip_special_tokens": False, "clean_up_tokenization_spaces": False})
        model_cfg = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True); model_cfg._attn_implementation = "eager"
        control.require(model_cfg.rope_theta == 1000000 and model_cfg.num_hidden_layers == 28, "architecture drift")
        torch.cuda.reset_peak_memory_stats(); previous = torch.get_default_dtype(); load_started = time.monotonic()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda:0"):
                model = calls.call("model_construct", lambda: Qwen3ForCausalLM(model_cfg), stage="model_construct")
        finally:
            torch.set_default_dtype(previous)
        model.eval().requires_grad_(False)
        def load_checkpoint():
            hashes, diagnostics = stream_checkpoint.load_streamed(torch, model, model_dir / "model.safetensors", calls, tensor_digest)
            control.require(model.lm_head.weight is model.model.embed_tokens.weight and sum(parameter.numel() for parameter in model.parameters()) == 596049920,
                            "native tie/parameter count")
            return hashes, diagnostics
        base_hash, diagnostics = calls.call("checkpoint_load", load_checkpoint, stage="checkpoint_load")
        result.update(pretrained_loads=1, load_seconds=time.monotonic() - load_started, loader_diagnostics=diagnostics)
        result["checks"]["exact_native_load"] = True
        control.require(not model.training and all(not parameter.requires_grad for parameter in model.parameters()), "bare frozen eval contract")
        result["checks"]["bare_frozen_eval"] = True
        guard("loaded")
        control.write(output / "MODEL_RECEIPT.json", {"weight_sha256": cfg["weight_sha256"], "primary_receipt_sha256": cfg["primary_receipt_sha256"],
            "unique_parameters": 596049920, "serialized_tensors": 310, "tied": True, "trainable_parameters": 0, "adapter": None,
            "tensor_sha256": base_hash, "loader": "stock Qwen3 allocated BF16 on CUDA; strict safetensors header and 2MiB bounded read buffer copy",
            "loader_mode": cfg["loader_mode"], "loader_diagnostics": diagnostics, "config": model_cfg.to_dict()})
        def prompt(length):
            seq = tokenizer.apply_chat_template([{"role": "user", "content": cfg["fixtures"]["user_message"]}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            filler = tokenizer.encode(cfg["fixtures"]["filler"], add_special_tokens=False)
            control.require(bool(filler) and len(seq) <= length, "mechanical fixture token length")
            ids = (filler * (length // len(filler) + 1))[:length - len(seq)] + seq
            control.require(len(ids) == length, "fixture exact length")
            return ids
        def forward(ids, stage, keep):
            control.require(type(ids) is list and 1 <= len(ids) <= 512 and type(keep) is int and 1 <= keep <= len(ids), "unqualified capture forward shape")
            control.token_hash(ids)
            tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
            return calls.call("forward", lambda: model(tensor, use_cache=False, logits_to_keep=keep).logits,
                              stage=stage, input_positions=len(ids))
        with torch.no_grad():
            for length in cfg["fixtures"]["finite_prefix_probe_lengths"]:
                logits = forward(prompt(length), f"finite_prefix_probe_{length}", 1)
                control.require(tuple(logits.shape) == (1, 1, 151936) and bool(torch.isfinite(logits).all()), "nonfinite prefix probe")
                del logits; guard(f"finite_prefix_probe_{length}")
            result["checks"]["finite_prefix_probes"] = {"contexts": [256, 512], "logits_to_keep": 1, "finite": True}
            for prompt_length in cfg["fixtures"]["prompt_lengths"]:
                ids = prompt(prompt_length); fixture_id = "fixture_" + str(prompt_length)
                record = {"fixture_id": fixture_id, "prompt_tokens": prompt_length, "generation_cap": 128, "prompt_ids": ids,
                    "prompt_token_ids_sha256": control.token_hash(ids), "generated_ids": [], "prefix_token_ids_sha256": [],
                    "raw_argmax_evidence": [], "online_model_logprobs": [], "offline_model_logprobs": [], "behavior_logprobs": []}
                result["fixtures"].append(record)
                torch.cuda.synchronize(); generation_started = time.monotonic()
                event("fixture_start", fixture_id=fixture_id, prompt_tokens=prompt_length, generation_cap=128,
                      prompt_token_ids_sha256=record["prompt_token_ids_sha256"])
                while len(record["generated_ids"]) < 128:
                    ordinal = len(record["generated_ids"]); prefix = ids + record["generated_ids"]
                    prefix_hash = control.token_hash(prefix)
                    logits = forward(prefix, f"greedy_{prompt_length}_{ordinal}", 1)
                    control.require(tuple(logits.shape) == (1, 1, 151936) and bool(torch.isfinite(logits).all()), "nonfinite greedy forward")
                    scores = logits[0, -1].float()
                    token = int(scores.argmax(-1).item()); maximum = scores.max()
                    first_maximum = int((scores == maximum).nonzero()[0, 0].item())
                    selected_value, maximum_value = float(scores[token].item()), float(maximum.item())
                    behavior = control.validate_choice(token, selected_value, maximum_value, first_maximum)
                    logprob = float(scores.log_softmax(-1)[token].item())
                    control.require(math.isfinite(logprob) and logprob <= 0, "nonfinite selected model logprob")
                    control.require(result["generation_tokens"] < cfg["ceilings"]["generated_tokens"], "CAPTURE_GENERATION_CEILING")
                    evidence = {"selected_raw_logit": selected_value, "raw_max_logit": maximum_value, "first_maximum_token_id": first_maximum, "raw_argmax_verified": True}
                    record["generated_ids"].append(token); record["online_model_logprobs"].append(logprob); record["behavior_logprobs"].append(behavior)
                    record["prefix_token_ids_sha256"].append(prefix_hash); record["raw_argmax_evidence"].append(evidence); result["generation_tokens"] += 1
                    event("generated_token", fixture_id=fixture_id, prompt_tokens=prompt_length, ordinal=ordinal,
                          prefix_token_ids_sha256=prefix_hash, token_id=token, model_logprob=logprob, behavior_logprob=behavior, **evidence)
                    del logits, scores, maximum
                    guard(f"greedy_{prompt_length}_{ordinal}")
                    if control.termination_after_emit(ids, record["generated_ids"], 128, cfg["decoding"]["eos_ids"]) is not None: break
                torch.cuda.synchronize()
                record["generation_seconds"] = time.monotonic() - generation_started
                record["generation_tokens_per_second"] = len(record["generated_ids"]) / record["generation_seconds"]
                record["generation_timing"] = "instrumented generation wall including durable token logging and resource guards"
                record.update(control.render_response(tokenizer, record["generated_ids"], 128, cfg["decoding"]["eos_ids"]))
                record["generated_token_ids_sha256"] = control.token_hash(record["generated_ids"])
                record["visible_token_ids_sha256"] = control.token_hash(record["visible_ids"])
                length = len(record["generated_ids"]); full = ids + record["generated_ids"]
                record.update(offline_input_token_ids_sha256=control.token_hash(full), offline_logits_to_keep=length + 1,
                              offline_prediction_indices=control.offline_indices(length, length + 1))
                torch.cuda.synchronize(); offline_started = time.monotonic()
                logits = forward(full, f"teacher_force_{prompt_length}", length + 1)
                control.require(tuple(logits.shape) == (1, length + 1, 151936) and bool(torch.isfinite(logits).all()), "nonfinite teacher-forced forward")
                target = torch.tensor(record["generated_ids"], device="cuda", dtype=torch.long)
                selected = logits[0, :length].float().log_softmax(-1).gather(1, target[:, None]).squeeze(1)
                record["offline_model_logprobs"] = selected.cpu().tolist()
                del logits, target, selected
                guard(f"teacher_force_{prompt_length}")
                torch.cuda.synchronize()
                record["offline_teacher_seconds"] = time.monotonic() - offline_started
                record["offline_timing"] = "instrumented teacher-forcing wall including selected model logprobs and resource guard"
                record["likelihood"] = control.likelihood_measurement(record["online_model_logprobs"], record["offline_model_logprobs"],
                    cfg["teacher_forcing"]["logprob_max_atol"], cfg["teacher_forcing"]["logprob_mean_atol"])
                event("fixture_complete", fixture_id=fixture_id, generated_tokens=length, stop_reason=record["stop_reason"],
                      terminal_eos_id=record["terminal_eos_id"], likelihood=record["likelihood"], generated_token_ids_sha256=record["generated_token_ids_sha256"],
                      raw_response_sha256=record["raw_response_sha256"], visible_response_sha256=record["visible_response_sha256"])
                control.require(record["likelihood"]["passed"], "selected raw-model logprob qualification failed")
        result["checks"].update(fixture_input_boundaries=True, greedy_selected_raw_argmax=True, eos_stop_accounting=True, response_rendering=True)
        errors = [value for fixture in result["fixtures"] for value in fixture["likelihood"]["absolute_errors"]]
        result["checks"]["causal_shift_model_logprobs"] = {"fixtures": 2, "finite": True, "max_abs": max(errors), "mean_abs": sum(errors) / len(errors)}
        terminal = sorted({fixture["terminal_eos_id"] for fixture in result["fixtures"] if fixture["terminal_eos_id"] is not None})
        result["checks"]["native_eos_coverage"] = {"observed": bool(terminal), "terminal_eos_ids": terminal}
        control.require(bool(terminal), "native natural EOS coverage missing")
        observed = control.expected_counts(*(len(fixture["generated_ids"]) for fixture in result["fixtures"]))
        snap = accounting.snapshot()
        control.require(snap["last_call_id"] == observed["logical_calls"] and snap["operations"]["forward"]["returned"] == observed["forward_calls"]
                        and snap["forward_input_positions_returned"] == observed["forward_input_positions"], "capture dynamic accounting differs")
        result["actual_counts"] = observed
        for name, parameter in model.named_parameters():
            control.require(not parameter.requires_grad and parameter.grad is None and tensor_digest(torch, parameter) == base_hash[name], "frozen bare base drift")
        result["checks"]["frozen_base_unchanged"] = True
        for name, sha in asset_pins.items(): control.require(control.file_identity(model_dir / name)["sha256"] == sha, "source asset mutated")
        control.require(control.file_identity(root / cfg["qualified_core_config"])["sha256"] == cfg["qualified_core_config_sha256"]
                        and control.file_identity(reference_path)["sha256"] == cfg["qualified_CE512_reference_sha256"]
                        and control.file_identity(primary_path)["sha256"] == cfg["primary_receipt_sha256"], "qualified source metadata mutated")
        control.recheck_sources(root, snapshot); result["checks"]["source_assets_unchanged"] = True
        guard("complete")
        result["status"] = control.PASS_STATUS
        result["unqualified"] = ["DEV/source enrollment", "long context beyond512", "adapter capture", "E1/training",
                                 "human language grading", "stochastic/probability-based RL", "published-default generation"]
    except Exception as error:
        result["status"] = "BLOCKED_CAPTURE_QUALIFICATION"
        result["error"] = str(error); result["traceback"] = traceback.format_exc()
        event("failure", error=str(error))
        result["call_accounting"] = accounting.snapshot()
        control.write(output / "FAILURE_RECEIPT.json", control.safe_numbers(result))
    finally:
        result["wall_seconds"] = time.monotonic() - started; result["call_accounting"] = accounting.snapshot()
        result = control.safe_numbers(result)
        control.write(output / "CAPTURE_RESULT.json", result)
        control.write(output / "RESOURCE_BASELINE.json", {"status": result["status"], "samples": result["resources"]})
        control.write(output / "COST_LEDGER.json", {name: result[name] for name in ("wall_seconds", "pretrained_loads", "optimizer_steps",
            "training_tokens", "generation_tokens", "teacher_calls", "accounting_version", "call_accounting")})
        print(json.dumps({name: result.get(name) for name in ("status", "error", "wall_seconds", "generation_tokens", "optimizer_steps")}), flush=True)
    return 0 if result["status"] == control.PASS_STATUS else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--go-path", required=True, type=Path)
    sys.exit(run(parser.parse_args()))
