"""Verified training-input patterns, never a measurement of model improvement."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path

from . import custody, packet, registry, schema, token_audit
from .opposite_analysis import analyze_opposite_directions


def summarize_input_rows(rows, row_audits):
    if len(rows) != 2000 or len(row_audits) != 2000:
        raise ValueError("STOP_INPUT_PATTERN_EXACT_TRAINING_DOSE")
    groups = defaultdict(lambda: {"examples": 0, "loss_tokens": 0, "prompt_tokens": 0})
    keys = set()
    for row, audit in zip(rows, row_audits, strict=True):
        schema.validate_row(row)
        if row["row_id"] != audit["row_id"] or row["row_id"] in keys:
            raise ValueError("STOP_INPUT_PATTERN_ROW_AUDIT_BINDING")
        keys.add(row["row_id"])
        key = (row["family"], row["variant_id"], row["answerable"], bool(row.get("is_replay")), row["support_expansion"])
        item = groups[key]
        item["examples"] += 1
        item["loss_tokens"] += audit["loss_tokens"]
        item["prompt_tokens"] += audit["prompt_tokens"]
    loss_tokens = sum(item["loss_tokens"] for item in groups.values())
    unanswerable = sum(not row["answerable"] for row in rows)
    replay = sum(bool(row.get("is_replay")) for row in rows)
    return {"examples": len(rows), "unanswerable_examples": unanswerable,
        "unanswerable_example_fraction": unanswerable / len(rows), "replay_examples": replay,
        "family_example_counts": dict(Counter(row["family"] for row in rows)),
        "groups": [{"family": key[0], "style": key[1], "answerable": key[2], "replay": key[3],
                    "support_expansion": key[4], **value,
                    "example_fraction": value["examples"] / 2000,
                    "supervised_token_fraction": value["loss_tokens"] / loss_tokens}
                   for key, value in sorted(groups.items())],
        "loss_reduction_interpretation": "Each microbatch's positive-target-token mean is divided by eight before accumulation: equal example coefficients, not an automatic global-token weighting claim.",
        "gradient_magnitude_inferred": False, "model_gain_verified": False}


def audit_current_inputs(packet_dir, output_dir):
    """Recheck old artifact hashes; don't open new/old final banks or any outcomes."""
    packet_dir, output_dir = custody.external(packet_dir), custody.external(output_dir)
    validation_data = (packet_dir / "CPU_VALIDATION.json").read_bytes()
    validation = json.loads(validation_data)
    fixed = registry.build_registry()
    if (validation.get("registry_sha256") != fixed["registry_sha256"]
            or validation.get("offline_validation_passed") is not True
            or validation.get("current_model_loads") != 0 or validation.get("current_training_updates") != 0):
        raise ValueError("STOP_INPUT_PATTERN_SOURCE_PACKET_IDENTITY")
    source = custody.source_identity()
    tokenizer = token_audit.load_tokenizer(custody.CORE / "tokenizer.json", source["sidecar_sha256"]["tokenizer.json"])
    findings = {}
    for trial_id, descriptor in validation["datasets"].items():
        rows = packet.read_training_dataset(descriptor)
        audit = token_audit.audit_rows(rows, tokenizer)
        summary = summarize_input_rows(rows, audit["row_audits"])
        summary.update(recipe_id=descriptor["recipe_id"], seed=descriptor["seed"],
                       artifact_sha256=descriptor["sha256"], payload_sha256=descriptor["payload_sha256"],
                       exact_token_audit_sha256=audit["audit_sha256"])
        findings[trial_id] = summary
    if len(findings) != 16:
        raise ValueError("STOP_INPUT_PATTERN_INCOMPLETE_TRIAL_GRID")
    comparisons = []
    recipes = fixed["recipes"]
    for seed in schema.SEEDS:
        for low in (r for r in recipes if r["replay_fraction"] == 0):
            high = next(r for r in recipes if r["replay_fraction"] == .2
                        and r["prompt_coverage"] == low["prompt_coverage"] and r["difficulty"] == low["difficulty"])
            a, b = findings[f"TQR-{low['recipe_id']}-S{seed}"], findings[f"TQR-{high['recipe_id']}-S{seed}"]
            comparisons.append({"seed": seed, "held_prompt_coverage": low["prompt_coverage"],
                "held_difficulty": low["difficulty"], "low_recipe": low["recipe_id"], "high_recipe": high["recipe_id"],
                "unanswerable_fraction_low": a["unanswerable_example_fraction"],
                "unanswerable_fraction_high": b["unanswerable_example_fraction"],
                "unanswerable_example_change": b["unanswerable_examples"] - a["unanswerable_examples"],
                "interpretation": "Replay replaces fresh examples and changes answerability/class prevalence; it is not isolated retention at equal repair dose."})
    report = {"experiment": schema.EXPERIMENT, "status": "VERIFIED_INPUT_PATTERNS_NO_MODEL_GAIN",
        "source_packet_validation_sha256": custody.sha_bytes(validation_data), "trial_inputs": findings,
        "fixed_budget_replay_contrasts": comparisons,
        "opposite_model_directions": analyze_opposite_directions(),
        "current_model_loads": 0, "current_training_updates": 0, "model_gain_verified": False,
        "final_banks_opened": False, "source_manifest": custody.source_manifest(),
        "next_questions": ["Does replacing 400 repair examples with historical transition replay retain T while reducing answerability acquisition?",
            "Does contract diversity improve off-canonical robustness while damaging canonical performance?",
            "Do matched counterfactuals improve evidence dependence or over-specialize a small model?"],
        "limitations": ["These are exact training-input observations, not gradient magnitude or capability measurements.",
            "No native model result is available; opposite-direction effects remain UNMEASURED.",
            "The existing final can confirm a selected recipe's gain, not all factorial causal claims."]}
    custody.write_json(output_dir / "INPUT_PATTERN_AUDIT.json", report)
    print(json.dumps({"status": report["status"], "trials": 16,
                      "replay_missing_example_changes": sorted({c["unanswerable_example_change"] for c in comparisons}),
                      "native_model_gain": False}, indent=2))
    return report
