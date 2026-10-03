"""Execute the prospectively fixed single-layer graft and preserve all receipts."""
from __future__ import annotations

import gc
import inspect
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ARMR = REPO / "experiments/anchor_routed_middle_recurrence_v1"
LAB = REPO / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1"
sys.path[:0] = [str(ROOT), str(ARMR / "src"), str(ARMR / "scripts"), str(LAB / "src")]

import psutil
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from armr.evaluation import encode, evaluate_case, forward_logits
from armr.hooks import Reapplication, SafetyError, frozen_inference
from armr.provenance import cases_from, checkpoint_files, git, sha256, state_hash, write_json
from armr.reporting import mean, summarize
from run_experiment import resolve_model
from run_fixed_recurrence import development_result
from rineya_cross_scale.leakage import guard_non_sealed_path
from rineya_cross_scale.streaming_forward import StreamingWeights
from graft import (Bridges, Graft, ResidualExpert, alignment, capture_native,
                   extra_flops, finite, load_layer, memory_guard, normalize,
                   positions_mask, scenario_split)


def read_calibration():
    path = guard_non_sealed_path(LAB / "data/evaluation/calibration.jsonl")
    if sha256(path) != "d2f1f415e3cf34a8c7bb7a8cdef4240a8ce993c41d9b25a5d1471f19b8f453a1":
        raise SafetyError("calibration input changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Only these prompt/identity fields are retained. No target label is used.
    return [{k: r[k] for k in ("case_id", "scenario_id", "prompt")} for r in rows]


def donor_files(path, record):
    files = record["weights"]
    entries = files if isinstance(files, list) else files["files"]
    result = {entry["name"]: sha256(path / entry["name"]) for entry in entries}
    if result != {entry["name"]: entry["sha256"] for entry in entries}:
        raise SafetyError("donor shard hash mismatch")
    if sha256(path / "config.json") != record["config"]["sha256"]:
        raise SafetyError("donor config hash mismatch")
    return result


def calibrate(supa_path, donor_path, config, out, local):
    tokenizer = AutoTokenizer.from_pretrained(supa_path, local_files_only=True, trust_remote_code=True)
    donor_tokenizer = AutoTokenizer.from_pretrained(donor_path, local_files_only=True)
    if tokenizer.get_vocab() != donor_tokenizer.get_vocab():
        raise SafetyError("donor vocabulary/token IDs differ")
    rows, encoded, lengths = read_calibration(), [], []
    for r in rows:
        text = tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if not torch.equal(ids, donor_tokenizer(text, return_tensors="pt")["input_ids"]):
            raise SafetyError("calibration token alignment mismatch")
        encoded.append(ids[:, :config["calibration_max_tokens"]])
        lengths.append({"case_id": r["case_id"], "scenario_id": r["scenario_id"],
                        "full_tokens": ids.shape[1], "retained_tokens": encoded[-1].shape[1]})
    del donor_tokenizer
    train_prompts, heldout = scenario_split(rows, config["seed"], config["holdout_scenarios"])
    token_train = torch.cat([torch.full((ids.shape[1],), train, dtype=torch.bool)
                             for ids, train in zip(encoded, train_prompts)])
    write_json(out / "calibration_inputs.json", {"lengths": lengths, "holdout_scenarios": heldout,
                                                "train_prompt_mask": train_prompts,
                                                "train_tokens": int(token_train.sum()),
                                                "heldout_tokens": int((~token_train).sum())})
    sl, dl = config["supa_layer"], config["donor_layer"]
    started = time.perf_counter()
    def report(label):
        return lambda i, total: print(f"capture {label} L{i}/{total} elapsed={time.perf_counter()-started:.1f}s rss={psutil.Process().memory_info().rss/2**30:.2f}GiB", flush=True)
    small = capture_native(supa_path, encoded, (sl,), config["max_rss_gib"], report("Supa"))[sl]
    big = capture_native(donor_path, encoded, (dl-1, dl), config["max_rss_gib"], report("Qwen4B"))
    x, y = torch.cat(small), torch.cat(big[dl-1])
    delta_native = [b-a for a, b in zip(big[dl-1], big[dl])]
    print(f"fitting bridges train={int(token_train.sum())} heldout={int((~token_train).sum())}", flush=True)
    bridges = {"aligned": Bridges.fit(x[token_train], y[token_train], config["ridge_relative"])}
    perm = torch.randperm(int(token_train.sum()), generator=torch.Generator().manual_seed(config["seed"]))
    bridges["shuffled"] = Bridges.fit(x[token_train], y[token_train][perm], config["ridge_relative"])
    bridges["random"] = Bridges.random(x[token_train], y[token_train], config["seed"])
    diagnostic = {"fit": {}, "tokens": len(x), "train_tokens": int(token_train.sum()),
                  "heldout_tokens": int((~token_train).sum()), "method": "affine ridge on calibration states; fixed coefficients; delta has no bias"}
    for name, bridge in bridges.items():
        diagnostic["fit"][name] = {}
        for split, mask in (("train", token_train), ("heldout", ~token_train)):
            diagnostic["fit"][name][split] = {
                "up": alignment(bridge.project(x[mask]), y[mask]),
                "down": alignment(y[mask] @ bridge.down + bridge.down_bias, x[mask])}
    torch.save({"small": small, "big_input": big[dl-1], "big_delta": delta_native,
                "input_ids": encoded, "train_prompt_mask": train_prompts}, local / "activations.pt")
    torch.save({k: b.state_dict() for k, b in bridges.items()}, local / "bridges.pt")
    donor_config = AutoConfig.from_pretrained(donor_path, local_files_only=True)
    donor_config._attn_implementation = "eager"
    with StreamingWeights(donor_path) as weights:
        donor = load_layer(weights, donor_config, dl)
    expert = ResidualExpert(donor, donor_config, bridges["aligned"], 1.)
    predicted, native = [], []
    with torch.inference_mode():
        for h, dz, train in zip(small, delta_native, train_prompts):
            if not train:
                h = h.unsqueeze(0)
                pos, mask = positions_mask(h)
                z = bridges["aligned"].project(h)
                actual = donor(z, position_ids=pos, attention_mask=mask,
                               position_embeddings=expert.rope(z, pos), use_cache=False)-z
                predicted.append(actual[0])
                native.append(dz)
    diagnostic["heldout_donor_delta_from_bridged_vs_native_input"] = alignment(torch.cat(predicted), torch.cat(native))
    diagnostic["wall_seconds"] = time.perf_counter()-started
    diagnostic["artifacts"] = {p.name: sha256(p) for p in local.glob("*.pt")}
    write_json(out / "bridge_fit.json", diagnostic)
    # Retain a single calibration fixture for native Supa capture validation.
    fixture = (encoded[0], small[0].clone())
    return tokenizer, donor, donor_config, bridges, fixture


def run_arm(model, tokenizer, cases, config, out, name, base=None, expert=None, recurrence=False):
    if expert is None:
        wrapper = Reapplication(model, layer=config["supa_layer"] if recurrence else None,
                                applications=2 if recurrence else 1)
    else:
        wrapper = Graft(model, expert, layer=config["supa_layer"],
                        emergency_bounds=tuple(config["emergency_rms_ratio_bounds"]))
    rows, logits = [], {}
    with wrapper, frozen_inference(model):
        for case in cases:
            memory_guard(config["max_rss_gib"])
            row, first = evaluate_case(model, tokenizer, case, config, wrapper,
                                      None if base is None else base[case["case_id"]])
            if expert is not None:
                row["generation_dense_matmul_flops_estimate"] += row["processed_token_positions"] * extra_flops(expert)
            row["arm"] = name
            rows.append(row)
            logits[case["case_id"]] = first
            with (out / "raw_results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            print(f"{name} {case['case_id']} cos={row['logit_cosine']:.6f} KL={row['kl_base_to_arm']:.5f} score={row['score']} {row['termination']}", flush=True)
    write_json(out / f"{name}.json", rows)
    return rows, logits


def exact_identity(rows, logits, base_rows, base_logits):
    return (len(rows) == len(base_rows) and
            all(a["case_id"] == b["case_id"] and a["token_ids"] == b["token_ids"] for a, b in zip(rows, base_rows)) and
            set(logits) == set(base_logits) and all(torch.equal(logits[k], base_logits[k]) for k in logits))


def report(out, result, bridge_fit):
    lines = ["# Single-layer graft result", "", f"Status: {result['status']}.", "",
             "Supa post-L12 + one frozen Qwen3-4B L16 residual correction. CPU FP32/eager, no cache.", "",
             "| Smoke arm | Correct | Mean cosine | Mean KL | RMS range | Mechanical gate |",
             "|---|---:|---:|---:|---|---|"]
    for name, s in result.get("smoke", {}).items():
        lines.append(f"| {name} | {s['score']*s['n']:.0f}/{s['n']} | {s['mean_logit_cosine']:.6f} | {s['mean_kl']:.6f} | {s['min_rms_ratio']:.5f}–{s['max_rms_ratio']:.5f} | {s['go_fixed_smoke']} |")
    lines += ["", "Z: exact zero; I: identity donor; N: near-zero; R: random; S: shuffled pairs; G: aligned; A1: native recurrence.",
              "", f"Primary mechanical gate: **{result.get('primary_mechanical_pass')}**.",
              "A passed tiny-gate or zero-gate control does not demonstrate useful transfer.", ""]
    if bridge_fit:
        d = bridge_fit["fit"]["aligned"]["heldout"]
        lines += [f"Scenario-held-out bridge cosine: up {d['up']['token_cosine']:.4f}, down {d['down']['token_cosine']:.4f}.",
                  f"Calibration: {bridge_fit['train_tokens']} training tokens, {bridge_fit['heldout_tokens']} held-out tokens. No answer labels in fitting.", ""]
    if result.get("development"):
        lines += ["| Development arm | Correct | Wins / losses | Preservation |", "|---|---:|---:|---|"]
        for name, d in result["development"].items():
            lines.append(f"| {name} | {d['correct']:.0f}/{d['items']} | {len(d['wins'])} / {len(d['losses'])} | {d['preservation_pass']} |")
        lines += ["", "A0/G/R/S/A1 use the same 16 previously inspected development cases. Results are exploratory; no independent efficacy claim."]
    else:
        lines += ["Development was not licensed because the primary mechanical/baseline prerequisite failed."]
    lines += ["", "This tests a single prespecified donor layer, insertion point, map fit, and gate. Failure does not rule out other bridges or layers.",
              "All backbone weights stay frozen. No router, MLA, MTP, repeated graft, distilled model, sealed evaluation or published checkpoint.",
              "", "See run_manifest.json for integrity, hashes and resource receipts; bridge_fit.json for alignment; raw_results.jsonl for individual outcomes."]
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    config = json.loads((ROOT / "protocol.json").read_text())
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out, local = ROOT / "evidence" / stamp, ROOT / "local" / stamp
    out.mkdir(parents=True, exist_ok=False)
    local.mkdir(parents=True, exist_ok=False)
    print(f"RUN_DIR={out}", flush=True)
    donor_record_path = LAB / "cross_scale_evidence/02_architecture/qwen4b.json"
    donor_record = json.loads(donor_record_path.read_text())
    donor_path, supa_path = Path(donor_record["resolved_path"]), resolve_model(None)
    if donor_record["revision"] != config["donor_revision"]:
        raise SafetyError("pinned donor revision mismatch")
    source = [ROOT / "README.md", ROOT / "protocol.json", *ROOT.glob("*.py"), *ROOT.glob("tests/*.py"),
              *ARMR.glob("src/**/*.py"), *ARMR.glob("scripts/*.py"), *LAB.glob("src/**/*.py")]
    manifest = {"status": "STARTED", "started_utc": stamp, "config": config,
                "git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
                "git_status": git("status", "--short"), "supa_path": str(supa_path), "donor_path": str(donor_path),
                "donor_architecture_sha256": sha256(donor_record_path),
                "sources": {str(p.relative_to(REPO)): sha256(p) for p in source},
                "datasets": {n: sha256(LAB / "data/evaluation" / n) for n in
                             ("calibration.jsonl", "supa_layer_ablation_smoke.jsonl", "mergedev_fast.jsonl")},
                "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                "qwen_implementation_sha256": sha256(Path(inspect.getfile(Qwen3DecoderLayer))),
                "bridge_training": "closed-form affine ridge, calibration prompts only", "backbone_training": False,
                "device": "cpu", "dtype": "float32", "kv_cache_bytes": 0, "gpu_memory_bytes": None}
    write_json(out / "started_manifest.json", manifest)
    write_json(out / "protocol_snapshot.json", config)
    model, result, bridge_fit = None, {"status": "STARTED"}, None
    started = time.perf_counter()
    try:
        before = checkpoint_files(supa_path)
        manifest["supa_files_before"] = before
        if before["model.safetensors"] != config["checkpoint_sha256"]:
            raise SafetyError("Supa checkpoint mismatch")
        manifest["donor_files_before"] = donor_files(donor_path, donor_record)
        print("checkpoint hashes verified; calibration begins", flush=True)
        tokenizer, donor, donor_config, bridges, fixture = calibrate(supa_path, donor_path, config, out, local)
        bridge_fit = json.loads((out / "bridge_fit.json").read_text())
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(supa_path, local_files_only=True, trust_remote_code=True,
            dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True).eval().requires_grad_(False)
        if type(model).__name__ != "Qwen3AbstainForCausalLM" or len(model.model.layers) != 28:
            raise SafetyError("unexpected Supa class or depth")
        manifest["model_class"] = type(model).__name__
        manifest["supa_state_before"] = state_hash(model)
        manifest["donor_state_before"] = state_hash(donor)
        manifest["donor_layer_parameters"] = sum(p.numel() for p in donor.parameters())
        manifest["bridge_coefficients"] = sum(b.numel() for b in bridges["aligned"].buffers())
        captured = []
        handle = model.model.layers[config["supa_layer"]].register_forward_hook(lambda m, a, o: captured.append(o[0].clone()))
        try:
            with frozen_inference(model):
                forward_logits(model, fixture[0])
        finally:
            handle.remove()
        manifest["native_supa_capture_max_delta"] = float((captured[0]-fixture[1]).abs().max())
        if not torch.equal(captured[0], fixture[1]):
            raise SafetyError("native streamed Supa fixture differs from loaded Supa")
        del fixture, captured
        experts = {name: ResidualExpert(donor, donor_config, bridges[bridge], gate, identity)
                   for name, bridge, gate, identity in (
                       ("Z", "aligned", 0., False), ("I", "aligned", config["gate"], True),
                       ("N", "aligned", config["near_zero_gate"], False),
                       ("R", "random", config["gate"], False), ("S", "shuffled", config["gate"], False),
                       ("G", "aligned", config["gate"], False))}
        smoke = cases_from(LAB / "data/evaluation/supa_layer_ablation_smoke.jsonl")
        base_rows, base_logits = run_arm(model, tokenizer, smoke, config, out, "A0_smoke")
        result["smoke"] = {"A0": summarize(base_rows, base_rows, config)}
        result["identity"] = {}
        for name in ("Z", "I", "N", "R", "S", "G", "A1"):
            rows, logits = run_arm(model, tokenizer, smoke, config, out, name+"_smoke", base_logits,
                                   experts.get(name), recurrence=name=="A1")
            result["smoke"][name] = summarize(rows, base_rows, config)
            if name in ("Z", "I"):
                result["identity"][name] = exact_identity(rows, logits, base_rows, base_logits)
                if not result["identity"][name]:
                    raise SafetyError("exact identity control failed")
        result["primary_mechanical_pass"] = result["smoke"]["G"]["go_fixed_smoke"]
        result["development"] = {}
        write_json(out / "smoke_decision.json", result)
        if result["primary_mechanical_pass"] and not any(r["safety_stopped"] for r in base_rows):
            dev = cases_from(LAB / "data/evaluation/mergedev_fast.jsonl")
            dev_base, dev_logits = run_arm(model, tokenizer, dev, config, out, "A0_dev")
            result["development"]["A0"] = development_result(dev_base, dev_base)
            for name in ("G", "R", "S", "A1"):
                rows, _ = run_arm(model, tokenizer, dev, config, out, name+"_dev", dev_logits,
                                  experts.get(name), recurrence=name=="A1")
                d = development_result(rows, dev_base)
                d.pop("router_scaffold_licensed")
                result["development"][name] = d
            with Reapplication(model), frozen_inference(model):
                for case in dev:
                    if not torch.equal(dev_logits[case["case_id"]], forward_logits(model, encode(tokenizer, case))):
                        raise SafetyError("development restoration failed")
        with Reapplication(model), frozen_inference(model):
            for case in smoke:
                if not torch.equal(base_logits[case["case_id"]], forward_logits(model, encode(tokenizer, case))):
                    raise SafetyError("smoke restoration failed")
        manifest["restored_logits_bit_identical"] = True
        manifest["supa_state_after"], manifest["donor_state_after"] = state_hash(model), state_hash(donor)
        manifest["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in donor.parameters() if p.requires_grad)
        manifest["gradient_tensors"] = sum(p.grad is not None for p in model.parameters()) + sum(p.grad is not None for p in donor.parameters())
        if manifest["supa_state_after"] != manifest["supa_state_before"] or manifest["donor_state_after"] != manifest["donor_state_before"] or manifest["trainable_parameters"] or manifest["gradient_tensors"]:
            raise SafetyError("backbone mutation/training detected")
        manifest["supa_files_after"] = checkpoint_files(supa_path)
        manifest["donor_files_after"] = donor_files(donor_path, donor_record)
        manifest["source_hashes_unchanged"] = all(sha256(REPO / p) == h for p, h in manifest["sources"].items())
        manifest["dataset_hashes_unchanged"] = all(sha256(LAB / "data/evaluation" / p) == h for p, h in manifest["datasets"].items())
        if before != manifest["supa_files_after"] or not manifest["source_hashes_unchanged"] or not manifest["dataset_hashes_unchanged"]:
            raise SafetyError("checkpoint/source/input changed during run")
        result["status"] = "COMPLETE"
    except Exception as error:
        result["status"] = "FAILED"
        write_json(out / "failure.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
    finally:
        memory = psutil.Process().memory_info()
        manifest.update(status=result["status"], wall_seconds=time.perf_counter()-started,
                        peak_rss_bytes=getattr(memory, "peak_wset", memory.rss), result=result)
        write_json(out / "run_manifest.json", manifest)
        write_json(out / "decision.json", result)
        report(out, result, bridge_fit)
    print(f"FINAL_STATUS={result['status']} RUN_DIR={out}", flush=True)
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
