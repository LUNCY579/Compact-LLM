"""Run a fresh immutable baseline reproduction or frozen middle screen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(REPO / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from armr.evaluation import encode, evaluate_case, forward_logits
from armr.hooks import Reapplication, SafetyError, frozen_inference
from armr.provenance import (DATA, HISTORY, cases_from, checkpoint_files, new_run,
                             sha256, state_hash, write_json)
from armr.reporting import csv_file, mean, screen_report, summarize


def resolve_model(explicit: Path | None) -> Path:
    candidates = [explicit, Path(os.environ["RINEYA_SUPA_PATH"]) if os.environ.get("RINEYA_SUPA_PATH") else None,
                  REPO / "Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze",
                  REPO.parent / "Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze"]
    for path in candidates:
        if path is not None and (path / "model.safetensors").is_file():
            return path.resolve()
    raise FileNotFoundError("set RINEYA_SUPA_PATH to the local canonical checkpoint")


def run_arm(model, tokenizer, cases, config, out, name, bases=None, **kwargs):
    rows, logits = [], {}
    with Reapplication(model, **kwargs) as wrapper, frozen_inference(model):
        for case in cases:
            row, first = evaluate_case(model, tokenizer, case, config, wrapper,
                                       None if bases is None else bases[case["case_id"]])
            row.update(arm=name, layer=kwargs.get("layer"), applications=kwargs.get("applications", 1))
            rows.append(row)
            logits[case["case_id"]] = first
            with (out / "raw_results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
            print(f"{name} {case['case_id']} cos={row['logit_cosine']:.4f} stop={row['termination']} score={row['score']}", flush=True)
    (out / "outputs").mkdir(exist_ok=True)
    write_json(out / "outputs" / f"{name}.json", rows)
    return rows, logits


def cache_audit(model, tokenizer, case):
    ids = encode(tokenizer, case)
    with frozen_inference(model):
        full = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        cached = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        kv = cached.past_key_values
        lengths = [kv.get_seq_length(i) for i in range(28)]
        token = full.logits[:, -1].argmax(-1, keepdim=True)
        no_kv = model(input_ids=torch.cat((ids, token), 1), use_cache=False, logits_to_keep=1)
        with_kv = model(input_ids=token, past_key_values=kv, use_cache=True, logits_to_keep=1)
        return {"strategy": "intact baseline only; no recurrence", "cache_class": type(kv).__name__,
                "prefill_lengths": lengths, "decode_lengths": [kv.get_seq_length(i) for i in range(28)],
                "prefill_max_delta": float((full.logits-cached.logits).abs().max()),
                "decode_max_delta": float((no_kv.logits-with_kv.logits).abs().max()),
                "decode_close_at_1e-4": torch.allclose(no_kv.logits, with_kv.logits, atol=1e-4, rtol=1e-4),
                "layer_indices": [l.self_attn.layer_idx for l in model.model.layers]}


def baseline_stage(model, tokenizer, cases, config, out):
    baseline, bases = run_arm(model, tokenizer, cases, config, out, "intact")
    inactive, inactive_logits = run_arm(model, tokenizer, cases, config, out, "inactive", bases,
                                       layer=8, applications=3, enabled=False)
    checks = {"inactive_logits_bit_identical": all(torch.equal(bases[k], inactive_logits[k]) for k in bases),
              "inactive_outputs_identical": all(a["token_ids"] == b["token_ids"] for a,b in zip(baseline,inactive))}
    old = json.loads((HISTORY / "ranking.json").read_text())
    hist = {r["layer"]: r for r in old["ranked"]}
    skips = []
    for layer in config["baseline_skip_layers"]:
        rows, _ = run_arm(model, tokenizer, cases, config, out, f"skip_{layer:02d}", bases, layer=layer, skip=True)
        value = mean([r["logit_cosine"] for r in rows])
        expected = hist[layer]["mean_logit_cosine_vs_intact"]
        skips.append({"layer": layer, "mean_cosine": value, "historical_cosine": expected,
                      "absolute_difference": abs(value-expected), "output_safety_stops": sum(r["safety_stopped"] for r in rows)})
        checks[f"skip_{layer}_reproduced"] = abs(value-expected) <= config["baseline_cosine_tolerance"]
        checks[f"skip_{layer}_effect"] = value < .5 if layer < 3 else value > .95 if layer == 8 else value < .9
    restored, restored_logits = run_arm(model, tokenizer, cases, config, out, "restored", bases)
    checks["restored_logits_bit_identical"] = all(torch.equal(bases[k], restored_logits[k]) for k in bases)
    checks["restored_outputs_identical"] = all(a["token_ids"] == b["token_ids"] for a,b in zip(baseline,restored))
    old_base = json.loads((HISTORY / "smoke/baseline_intact.json").read_text())
    old_ids = {r["case_id"]: r["token_ids"] for r in old_base["predictions"]}
    checks["historical_baseline_output_identity"] = all(r["token_ids"] == old_ids[r["case_id"]] for r in baseline)
    checks["baseline_no_safety_stop"] = not any(r["safety_stopped"] for r in baseline)
    cache = cache_audit(model, tokenizer, cases[0])
    checks["intact_cache_equivalence"] = cache["decode_close_at_1e-4"] and cache["prefill_max_delta"] == 0
    checks["cache_layer_ownership"] = cache["layer_indices"] == list(range(28)) and all(x==cache["prefill_lengths"][0]+1 for x in cache["decode_lengths"])
    receipt = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "skip_results": skips, "cache_audit": cache,
               "history_sha256": sha256(HISTORY / "ranking.json")}
    write_json(out / "qualification.json", receipt)
    table = "\n".join(f"| {r['layer']} | {r['historical_cosine']:.6f} | {r['mean_cosine']:.6f} | {r['absolute_difference']:.6f} |" for r in skips)
    (out / "report.md").write_text(f"""# Baseline requalification

Status: {receipt['status']}.

Question: does the original anchor-layer hierarchy reproduce under the new
cache-free harness? Setup: same eight smoke prompts, seed 17, CPU FP32/eager,
four threads, canonical Supa custom class and abstention offset. Exact model,
source, runtime, dataset hashes and timings are in run_manifest.json.

Mechanical qualification: qualification.json records inactive/restored bit
identity, historical baseline greedy-ID identity, cache prefill/decode audit,
skip effects and tolerance checks. Final manifest records before/after state and
checkpoint hashes, zero trainable parameters and gradients, and peak RSS.

| Layer | Historical skip cosine | Reproduced | Absolute difference |
|---:|---:|---:|---:|
{table}

Interpretation: early anchor necessity and L8 removal tolerance reproduce only
if all checks pass. Expected negative-control corruptions end the affected case
early; unlike the historical decoder, the new decoder does not finish a known
loop. First-token skip comparisons precede those generation stopping decisions.

Non-claims: skip tolerance is not repeatability or efficacy; intact cache
equivalence does not establish cache semantics for recurrent passes.

Next gate: {'middle-layer frozen reapplication screen' if receipt['status']=='PASS' else 'STOP and diagnose harness/baseline drift'}.
""", encoding="utf-8")
    return receipt


def screen_stage(model, tokenizer, cases, config, out):
    baseline, bases = run_arm(model, tokenizer, cases, config, out, "intact")
    results, hidden_rows, logit_rows = [], [], []
    for layer in config["candidate_layers"]:
        for apps in config["applications"]:
            rows, _ = run_arm(model, tokenizer, cases, config, out, f"layer_{layer:02d}_apps_{apps}", bases,
                              layer=layer, applications=apps, emergency_bounds=tuple(config["emergency_rms_ratio_bounds"]))
            result = {"layer": layer, "applications": apps, **summarize(rows, baseline, config)}
            results.append(result)
            write_json(out / f"layer_{layer:02d}_apps_{apps}_summary.json", result)
            for row in rows:
                key = {"layer": layer, "applications": apps, "case_id": row["case_id"]}
                hidden_rows.extend({**key, **h} for h in row["hidden"])
                logit_rows.append({**key, **{k: row[k] for k in ("logit_cosine", "kl_base_to_arm", "top1_agreement", "topk_overlap", "margin_change", "gold_rank", "baseline_gold_rank", "gold_nll")}})
    with Reapplication(model), frozen_inference(model):
        for case in cases:
            if not torch.equal(bases[case["case_id"]], forward_logits(model, encode(tokenizer, case))):
                raise SafetyError("post-screen restoration failed")
    csv_file(out / "hidden_state_metrics.csv", hidden_rows)
    csv_file(out / "logit_metrics.csv", logit_rows)
    return screen_report(out, results, json.loads((HISTORY / "ranking.json").read_text()), config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["baseline", "screen"], required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/reapplication_screen.json").read_text())
    dataset = DATA / config["smoke_file"]
    cases = cases_from(dataset)
    model_path = resolve_model(args.model)
    if args.stage == "screen":
        if args.baseline is None:
            parser.error("--baseline passing immutable receipt required")
        receipt = json.loads((args.baseline / "run_manifest.json").read_text())
        if receipt["status"] != "PASS" or receipt["config"] != config or receipt["dataset_sha256"] != sha256(dataset):
            raise SafetyError("baseline receipt/config mismatch")
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    stage = "00_baseline_requalification" if args.stage == "baseline" else "01_middle_reapplication_screen"
    out, manifest = new_run(stage, config, model_path, dataset)
    print(f"RUN_DIR={out}", flush=True)
    start = time.perf_counter()
    model = None
    try:
        before_files = checkpoint_files(model_path)
        manifest["checkpoint_files_before"] = before_files
        if before_files["model.safetensors"] != config["checkpoint_sha256"]:
            raise SafetyError("canonical checkpoint hash mismatch")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True,
                    dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True).eval().requires_grad_(False)
        if type(model).__name__ != "Qwen3AbstainForCausalLM" or len(model.model.layers) != 28:
            raise SafetyError("unexpected model class/shape")
        manifest["model_class"] = type(model).__name__
        manifest["state_hash_before"] = state_hash(model)
        if args.stage == "screen":
            if receipt["checkpoint_files_before"] != before_files or receipt["state_hash_before"] != manifest["state_hash_before"]:
                raise SafetyError("baseline checkpoint/state mismatch")
            if receipt["torch"] != manifest["torch"] or receipt["transformers"] != manifest["transformers"]:
                raise SafetyError("baseline runtime mismatch")
            manifest["baseline_receipt"] = str(args.baseline.resolve())
            manifest["baseline_receipt_sha256"] = sha256(args.baseline / "run_manifest.json")
        decision = baseline_stage(model, tokenizer, cases, config, out) if args.stage == "baseline" else screen_stage(model, tokenizer, cases, config, out)
        manifest["status"] = decision["status"]
        manifest["decision"] = decision
    except BaseException as exc:
        manifest["status"] = "FAIL"
        manifest["error"] = repr(exc)
        write_json(out / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        if not (out / "report.md").exists():
            (out / "report.md").write_text(f"# Failed run\n\nStatus: FAIL.\n\n{exc!r}\n\nNo downstream gate is licensed. See failure.json.\n", encoding="utf-8")
        traceback.print_exc()
    finally:
        if model is not None:
            manifest["state_hash_after"] = state_hash(model)
            manifest["trainable_parameter_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            manifest["parameters_with_gradients"] = sum(p.grad is not None for p in model.parameters())
            manifest["checkpoint_files_after"] = checkpoint_files(model_path)
            unchanged = manifest["state_hash_after"] == manifest.get("state_hash_before") and manifest["checkpoint_files_after"] == manifest.get("checkpoint_files_before")
            manifest["canonical_unchanged"] = unchanged
            if not unchanged or manifest["trainable_parameter_count"] or manifest["parameters_with_gradients"]:
                manifest["status"] = "FAIL"
                manifest["integrity_error"] = "canonical mutation or unexpected trainable/gradient parameters"
        manifest["wall_seconds"] = time.perf_counter() - start
        memory = psutil.Process().memory_info()
        manifest["peak_rss_bytes"] = getattr(memory, "peak_wset", memory.rss)
        manifest["end_rss_bytes"] = memory.rss
        manifest["end_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        write_json(out / "run_manifest.json", manifest)
    print(f"STATUS={manifest['status']} RUN_DIR={out}", flush=True)
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
