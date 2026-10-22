"""SBHM-001 existing-evidence audit, using only Python standard library.

Reads existing DEV/qualification JSON and source text. Never imports the package,
Torch, transformers, native weights, bank contents or the historical audit scripts.
Writes a NEW JSON file outside tracked source (exclusive creation).
"""
import argparse
import ast
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def distribution(values):
    valid = sorted(v for v in values if v is not None)
    assert all(math.isfinite(v) for v in valid)
    def q(p):
        x = (len(valid) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return valid[lo] * (hi - x) + valid[hi] * (x - lo) if lo != hi else valid[lo]
    return {"count": len(valid), "null_count": len(values) - len(valid),
            "min": min(valid), "q25": q(.25), "median": statistics.median(valid),
            "q75": q(.75), "p95": q(.95), "max": max(valid), "mean": statistics.mean(valid)} if valid else {"count": 0, "null_count": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default="C:/Users/clclc/BusinessProject/ChatBubble-rineya-spikingbrain-hybrid-memory-v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repository).resolve()
    package = repo / "experiments/rineya_spikingbrain_hybrid_memory_v1"
    run = package / "runs/titan_transition_v2_20261004_memory_efficient"
    auditroot = Path("C:/Users/clclc/BusinessProject/sbhm-001-audits/20261005-closeout-01a10d69").resolve()
    out = Path(args.output).resolve()
    assert out.is_relative_to(auditroot) and not out.is_relative_to(repo) and not out.exists()
    inputs = {}
    assertions = []
    def read(path):
        raw = path.read_bytes()
        inputs[str(path.resolve())] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return raw.decode("utf-8-sig")
    def readj(path):
        return json.loads(read(path))
    def check(condition, label):
        assert condition, label
        assertions.append(label)
    config = readj(package / "configs/titan_transition.v2.memory_efficient.json")
    source_config_path = Path(config["source"]["path"]) / "config.json"
    geometry = readj(source_config_path)
    check(inputs[str(source_config_path.resolve())]["sha256"] == config["source"]["files"]["config.json"], "Source config bytes match locked SHA256")
    check([geometry[k] for k in ("num_hidden_layers", "num_key_value_heads", "head_dim", "num_attention_heads")] == [28, 8, 128, 16], "Explicit source head geometry is 28 layers, 8 KV groups, head_dim 128, 16 query heads")
    check(config["dtype"] == "float32" and config["window"] == 128, "Successful run remains FP32 with local window 128")
    check(config["reset_recurrent_at"] is None, "Main run is not state-reset diagnostic")
    per_layer_token = 2 * geometry["num_key_value_heads"] * geometry["head_dim"] * 4
    recurrent_bytes = geometry["num_key_value_heads"] * geometry["head_dim"] ** 2 * 4
    def expected_cache(arm, length):
        if arm in "AB":
            return geometry["num_hidden_layers"] * length * per_layer_token
        return (geometry["num_hidden_layers"] - 1) * length * per_layer_token + min(length, config["window"]) * per_layer_token + (recurrent_bytes if arm == "D" else 0)
    texts = {name: read(package / "sbhm" / name) for name in ("pipeline.py", "protocol.py", "native.py", "core.py")}
    code_files = {str(p.relative_to(package)): hashlib.sha256(read(p).encode()).hexdigest() for p in sorted((package / "sbhm").glob("*.py"))}
    code_sha256 = digest(code_files)
    protocol_ast = ast.parse(texts["protocol.py"])
    resource_function = next(node for node in protocol_ast.body if isinstance(node, ast.FunctionDef) and node.name == "resources")
    resource_text = ast.get_source_segment(texts["protocol.py"], resource_function)
    for token, label in [
        ("(6.0 if runtime else 7.4)*2**30", "Runtime/admission free RAM thresholds remain 6.0/7.4 GiB"),
        ("rss > 4*2**30", "Worker RSS ceiling remains 4 GiB"),
        ("device != 'cuda:0'", "Explicit CUDA:0 requirement remains"),
        ("free < 1*2**30", "VRAM reserve remains 1 GiB"),
        ("torch.cuda.empty_cache()", "Pressure relief releases only unused CUDA allocator cache then reapplies reserve guard"),
    ]:
        check(token in resource_text, label)
    check("written_bytes + size > 512*2**20" in texts["pipeline.py"] and "size + 2**30" in texts["pipeline.py"], "Teacher cache cap remains 512 MiB with 1 GiB disk reserve")
    check("storages[(str(x.device), storage.data_ptr())] = storage.nbytes()" in texts["core.py"], "Cache reporting deduplicates actual tensor storage pointers")
    stages = {}
    for path in sorted(run.glob("*/receipt.json")):
        name = path.parent.name
        receipt, request = readj(path), readj(path.parent / "request.json")
        check(receipt["binding"] == request["binding"], name + " request/receipt binding equality")
        check(receipt["binding"]["config_sha256"] == digest(config), name + " canonical config hash equality")
        check(receipt["binding"]["code_sha256"] == code_sha256, name + " current package source equals bound code digest")
        check(receipt.get("allocator_cache_release_events") == [], name + " records no allocator release event")
        duration = (datetime.fromisoformat(receipt["created_utc"]) - datetime.fromisoformat(request["created_utc"])).total_seconds()
        check(duration > 0, name + " request/receipt UTC ordering")
        stages[name] = {"receipt": receipt, "request": request, "elapsed_seconds": duration}
    check(len(stages) == 21, "21 successful model stages have requests/receipts; separate compare is command 22")
    quals = {k: v for k, v in stages.items() if "qualification" in k}
    check(len(quals) == 10, "Ten native inference qualification receipts exist")
    qualifications = {}
    for name, item in quals.items():
        r = item["receipt"]
        arm = r["binding"]["arm"]
        check(r["status"] == "PASS_NATIVE_MECHANICS_ONLY" and r["max_prompt_tokens_checked"] == 2048, name + " status is 2048-token mechanics PASS")
        check(r["identity_wrapper_max_abs_error"] == 0 and r["capability_claim"] is False and r["native_training_qualified"] is False, name + " preserves zero identity error and no capability/training claim")
        check(set(r["parity"]) == {"64:2048", "1:17"}, name + " has maximum-shape chunked and short tokenwise checks")
        snapshots = {}
        for key, p in r["parity"].items():
            length = int(key.split(":")[1])
            check(p["cache_storage_bytes"] == expected_cache(arm, length), name + " " + key + " actual storage matches explicit geometry")
            check(0 <= p["max_abs_logit_error"] < 3e-4, name + " " + key + " absolute logit discrepancy below FP32 tolerance")
            snapshots[key] = p
        qualifications[name] = {"arm": arm, "elapsed_seconds_request_to_created": item["elapsed_seconds"], "snapshots": snapshots,
                                "total_parameters": r["total_parameters"], "trainable_parameters": r["trainable_parameters"]}
    evaluations = {arm: stages[arm + "-evaluation"]["receipt"] for arm in "ABCD"}
    rowkeys = None
    cache_rows = {}
    costs = {}
    metrics = ("model_ttft_seconds", "decode_seconds_excluding_first", "decode_tokens_per_second", "peak_vram_allocated_bytes", "peak_vram_reserved_bytes", "rss_sample_max_bytes", "cache_storage_bytes", "prompt_tokens", "generated_tokens")
    for arm, r in evaluations.items():
        predictions = r["predictions"]
        check(r["status"] == "DEV_MEASUREMENT_ONLY" and len(predictions) == 512, arm + " has all 512 DEV measurements")
        check(r["energy_measured"] is False and r["speedup_claim"] is False and r["semantic_accuracy"] is None, arm + " keeps instrumented/no-energy/no-semantic claim limits")
        keys = [(p["id"], p["family_id"], p["target"], p["prompt_tokens"]) for p in predictions]
        if rowkeys is None:
            rowkeys = keys
        check(keys == rowkeys and len({p["id"] for p in predictions}) == 512, arm + " row identity/order/prompt token counts pair with A")
        cache_rows[arm] = {}
        for p in predictions:
            check(p["generated_tokens"] == len(p["generated_ids"]) and p["generated_tokens"] >= 1, arm + " " + p["id"] + " generated token accounting")
            cached_length = p["prompt_tokens"] + p["generated_tokens"] - 1
            check(p["cache_storage_bytes"] == expected_cache(arm, cached_length), arm + " " + p["id"] + " cache formula")
            check(p["peak_vram_reserved_bytes"] >= p["peak_vram_allocated_bytes"] > 0, arm + " " + p["id"] + " CUDA peak order")
            check(0 < p["rss_sample_max_bytes"] <= 4 * 2**30, arm + " " + p["id"] + " sampled RSS below guard")
            check(p["model_ttft_seconds"] > 0 and p["decode_seconds_excluding_first"] >= 0, arm + " " + p["id"] + " timing finite and positive")
            if p["generated_tokens"] == 1:
                check(p["decode_tokens_per_second"] is None, arm + " " + p["id"] + " one-token throughput is null")
            else:
                expected_tps = (p["generated_tokens"] - 1) / p["decode_seconds_excluding_first"]
                check(math.isclose(p["decode_tokens_per_second"], expected_tps, rel_tol=1e-12), arm + " " + p["id"] + " decode throughput excludes first token")
            cache_rows[arm][p["id"]] = {"cached_length": cached_length, "cache_storage_bytes": p["cache_storage_bytes"]}
        costs[arm] = {}
        selections = {"all_dev": predictions}
        selections.update({g: [p for p in predictions if p["family_id"] in families] for g, families in r["evaluation_groups"].items()})
        for group, selected in selections.items():
            costs[arm][group] = {m: distribution([p[m] for p in selected]) for m in metrics}
            costs[arm][group]["instrumented_ttft_plus_decode_sum_seconds"] = sum(p["model_ttft_seconds"] + p["decode_seconds_excluding_first"] for p in selected)
            costs[arm][group]["generated_tokens_total"] = sum(p["generated_tokens"] for p in selected)
            costs[arm][group]["decode_tokens_excluding_first_total"] = sum(p["generated_tokens"] - 1 for p in selected)
            costs[arm][group]["stop_reason_counts"] = {s: sum(p["stop_reason"] == s for p in selected) for s in sorted({p["stop_reason"] for p in selected})}
    matched = []
    unmatched = []
    for rid, c in cache_rows["C"].items():
        d = cache_rows["D"][rid]
        if c["cached_length"] == d["cached_length"]:
            delta = d["cache_storage_bytes"] - c["cache_storage_bytes"]
            check(delta == recurrent_bytes, rid + " equal-prefix D/C delta is 0.5 MiB")
            matched.append(rid)
        else:
            unmatched.append({"id": rid, "C": c, "D": d, "delta_bytes": d["cache_storage_bytes"] - c["cache_storage_bytes"]})
    check(len(matched) == 511 and len(unmatched) == 1, "511 equal-prefix C/D rows and one unequal-completion exclusion")
    elapsed = {name: {"stage": item["request"]["stage"], "arm": item["receipt"]["binding"]["arm"],
                      "request_utc": item["request"]["created_utc"], "receipt_created_utc": item["receipt"]["created_utc"],
                      "seconds": item["elapsed_seconds"]} for name, item in stages.items()}
    training = {}
    for arm in "BCD":
        training[arm] = {"trainable_parameters": stages[arm + "-recover"]["receipt"]["trainable_parameters"], "stages": {}}
        for stage in ("distill", "recover"):
            r = stages[arm + "-" + stage]["receipt"]
            check(r["status"] == "TRAINING_COMPLETED_NOT_EFFICACY" and len(r["records"]) == 512, arm + " " + stage + " completed 512 steps")
            training[arm]["stages"][stage] = {"elapsed_request_to_created_seconds": stages[arm + "-" + stage]["elapsed_seconds"],
                                             "input_tokens": distribution([x["input_tokens"] for x in r["records"]]),
                                             "input_tokens_total": sum(x["input_tokens"] for x in r["records"]),
                                             "supervised_positions_total": sum(x["supervised_positions"] for x in r["records"])}
    teacher = stages["teacher"]["receipt"]
    check(teacher["teacher_cache_file_bytes"] == 502904576 and teacher["teacher_cache_file_bytes"] <= 512 * 2**20, "Teacher recorded 502904576 serialized target bytes below unchanged cap")
    behavior = {}
    for group, families in evaluations["A"]["evaluation_groups"].items():
        rows = {arm: {p["id"]: p for p in r["predictions"] if p["family_id"] in families} for arm, r in evaluations.items()}
        behavior[group] = {}
        for arm in "ABCD":
            for rid, p in rows[arm].items():
                check(p["strict_exact_match"] == (p["prediction"] == p["target"]), arm + " " + rid + " raw strict score")
                check(p["normalized_exact_match"] == (p["prediction"].strip().casefold() == p["target"].strip().casefold()), arm + " " + rid + " raw normalized score")
            acorrect = {rid for rid, p in rows["A"].items() if p["strict_exact_match"]}
            correct = {rid for rid, p in rows[arm].items() if p["strict_exact_match"]}
            behavior[group][arm] = {"rows": len(rows[arm]), "correct": len(correct), "source_correct": len(acorrect),
                                    "retained_source_correct": len(correct & acorrect), "damaged_source_correct": len(acorrect - correct),
                                    "repaired_source_incorrect": len(correct - acorrect)}
        ccorrect = {rid for rid, p in rows["C"].items() if p["strict_exact_match"]}
        dcorrect = {rid for rid, p in rows["D"].items() if p["strict_exact_match"]}
        behavior[group]["D_minus_C"] = {"D_only_correct": sorted(dcorrect - ccorrect), "C_only_correct": sorted(ccorrect - dcorrect),
                                         "percentage_points": (len(dcorrect) - len(ccorrect)) / len(rows["C"]) * 100}
    check(behavior["distant_heldout"]["C"]["correct"] == 172 and behavior["distant_heldout"]["D"]["correct"] == 171 and not behavior["distant_heldout"]["D_minus_C"]["D_only_correct"], "Raw distant DEV supports no recurrent-addition gain")
    check(behavior["near_control"]["D_minus_C"]["percentage_points"] == behavior["distant_heldout"]["D_minus_C"]["percentage_points"], "Raw near contrast equals distant contrast")
    check(behavior["retention"]["C"]["damaged_source_correct"] == behavior["retention"]["D"]["damaged_source_correct"] == 4, "C/D each damage four source-correct retention rows")
    parameters_added_D_vs_C = training["D"]["trainable_parameters"] - training["C"]["trainable_parameters"]
    report = {
        "schema": "SBHM-001/offline-resource-audit-v1", "status": "PASS_EXISTING_EVIDENCE_RESOURCE_ACCOUNTING", "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "input_files": inputs,
        "execution": {"mode": "offline stdlib inspection only", "native_model_execution": False, "training": False, "torch_import": False, "bank_contents_read": False, "tracked_source_mutated": False},
        "assertion_count": len(assertions), "assertions": assertions,
        "geometry": {"source_config": geometry, "per_layer_token_kv_bytes": per_layer_token, "recurrent_state_bytes": recurrent_bytes,
                     "cache_formula": {"A_B": "28*T*8192", "C": "27*T*8192+min(T,128)*8192", "D": "27*T*8192+min(T,128)*8192+524288", "T": "prompt_tokens+generated_tokens-1"},
                     "at_2048_bytes": {arm: expected_cache(arm, 2048) for arm in "ABCD"},
                     "D_savings_vs_A_at_2048_bytes": expected_cache("A", 2048) - expected_cache("D", 2048),
                     "D_savings_vs_A_at_2048_fraction": 1 - expected_cache("D", 2048) / expected_cache("A", 2048),
                     "prediction_cache_snapshots_verified": 2048, "qualification_cache_snapshots_verified": 20,
                     "equal_prefix_C_D_rows": len(matched), "equal_prefix_D_minus_C_bytes": recurrent_bytes, "unequal_prefix_C_D_rows": unmatched},
        "qualification": qualifications, "instrumented_cost_distributions": costs, "stage_request_to_receipt_elapsed": elapsed,
        "training": training, "teacher_recorded_cache_bytes": teacher["teacher_cache_file_bytes"], "raw_behavior_crosscheck": behavior,
        "parameters_added_D_vs_C": parameters_added_D_vs_C,
        "D_vs_C_cost_contrast": {
            "distant_TTFT_median_relative_change": costs["D"]["distant_heldout"]["model_ttft_seconds"]["median"] / costs["C"]["distant_heldout"]["model_ttft_seconds"]["median"] - 1,
            "distant_decode_tps_median_relative_change": costs["D"]["distant_heldout"]["decode_tokens_per_second"]["median"] / costs["C"]["distant_heldout"]["decode_tokens_per_second"]["median"] - 1,
            "training_request_to_receipt_sum_seconds": {arm: sum(training[arm]["stages"][stage]["elapsed_request_to_created_seconds"] for stage in ("distill", "recover")) for arm in "BCD"}},
        "allocator_cache_release_events": {name: len(item["receipt"]["allocator_cache_release_events"]) for name, item in stages.items()},
        "timing_semantics": {"TTFT": "Synchronized instrumented chunked prefill, finite checks and guards through first greedy argmax; excludes model loading/tokenization. Synthetic 8-token warmup occurred before DEV.",
                             "decode": "From first-token timestamp through decode loop and final synchronization. Includes initial post-prefill resource query, finite checks and per-token guards. Throughput numerator excludes first generated token; one-token completions null.",
                             "CUDA_peaks": "reset_peak_memory_stats before each row; loaded model included; peak counters read after check_cache; post-timed check_cache inspection may contribute allocations.",
                             "RSS": "Maximum sampled only after first-token timestamp and during decoding. Prefill resource calls enforce RSS ceiling but their numeric RSS is not added to per-row rss_sample_max_bytes.",
                             "stage_elapsed": "Derived from request.created_utc before load_host through receipt.created_utc; includes model load/install and stage work, excludes binding/admission/startup before request and final serialization afterward. Wall-clock timestamps, no GPU-only or optimizer-only timer."},
        "guard_evidence_limits": ["Completion and unchanged guarded source establish checkpoints passed, not continuous free-RAM/free-VRAM traces; receipts do not persist numeric free-memory samples.", "Qualification checks 2048 chunked-prefix inference and 17-token tokenwise inference only; not worst-shape training, padded/batched serving or full-length tokenwise qualification.", "Six training records only exercise 42-259 input tokens; max_train_tokens=512 is an enforced upper limit, not empirical maximum-shape fit.", "No training/qualification CUDA-peak/RSS peak record is stored in these completed receipts.", "RSS per-row statistic excludes prefill samples and is not a transient allocator-level process peak.", "Runtime guards are sampled before/after updates or chunks and after token generation; unobserved transient peaks remain possible.", "Energy, optimized-kernel cost, broad serving cost and money expenditure are not measured.", "Equal exposure does not equal parameter count or elapsed compute. D adds " + str(parameters_added_D_vs_C) + " parameters relative to C."],
        "decision": {"recommendation": "JUSTIFIED_HOLD", "exact_native_run_proposal": None,
                     "reason": "One source/site/seed synthetic DEV pilot gives no D-only distant repair, one C-only row, the same near-position contrast, 4/22 damaged source-correct retention cases in C and D, and additional state/parameters/reference cost in D. Mechanics alone does not justify spending on a successor.",
                     "successor_admission_requirement": "A specific mechanistic new hypothesis plus minimal discriminating control, separately frozen configuration/data/analysis budget and exact reviewed run proposal. No automatic site/context/recovery/recurrence expansion."}
    }
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "assertions": len(assertions),
                      "distant_medians": {arm: {metric: costs[arm]["distant_heldout"][metric]["median"] for metric in ("model_ttft_seconds", "decode_tokens_per_second")} for arm in "ABCD"},
                      "unmatched_cache": unmatched, "training": training, "elapsed_seconds": {k: v["seconds"] for k, v in elapsed.items()}}, indent=2))


if __name__ == "__main__":
    main()
