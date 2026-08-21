#!/usr/bin/env python3
"""Score late_rineya on chat DEV v4.5 with the Halo soup-lab protocol (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Rineya_Layerwise_Soup_Lab_v1.0" / "src"))

from rineya_soup_lab.evaluation.native_v45 import evaluate_model_bundled_v45
from rineya_soup_lab.evaluation.objective import score_summary
from rineya_soup_lab.utils import write_json

MODEL = REPO / "Rineya_Qwen3_CrossDepth_Soup_Mini_v1" / "artifacts" / "alpha_recipes" / "late_rineya"
TOKENIZER = REPO / "Qwen3-0.6B"
DEV = REPO / "short-response-model-phase-0-1-2" / "data" / "dev" / "dev.jsonl"
QUALITY = (
    REPO
    / "Rineya_Layerwise_Soup_Lab_v1.0"
    / "configs"
    / "official_v45"
    / "quality_evaluation.json"
)
OUT = (
    REPO
    / "Rineya_Qwen3_CrossDepth_Soup_Mini_v1"
    / "outputs"
    / "alpha_recipes"
    / "late_rineya_chat_dev_v45"
)

# Same selection contract as experiment.qwen3_halo_uncensored.yaml official_no_think.
PROTOCOL = {
    "enable_thinking": False,
    "extract_final_after_thinking": False,
    "response_control": "eos_only",
    "max_new_tokens": 32,
    "min_new_tokens_before_boundary": 1,
    "do_sample": False,
    "temperature": 0.6,
    "top_p": 0.9,
    "capture_token_probabilities": False,
    "require_cuda": False,
    "wait_for_gpu": False,
    "device_map": "cpu",
    "seed": 20260820,
    "system_prompt": (
        "Reply like a casual text message. Use only a few words or one short sentence. "
        "Be accurate, use the conversation context, ask one brief clarification when "
        "information is missing, and keep safety boundaries concise."
    ),
}

# Soup-lab objective (same weights as Halo-pair config).
OBJECTIVE = {
    "metrics": [
        {"path": "mean_quality_score", "weight": 0.30, "transform": "maximize_01"},
        {"path": "worst_decile_mean_quality_score", "weight": 0.20, "transform": "maximize_01"},
        {"path": "complete_utterance_rate", "weight": 0.10, "transform": "maximize_01"},
        {"path": "observable_quality_pass_rate", "weight": 0.10, "transform": "maximize_01"},
        {"path": "mean_quality_dimensions.task_adequacy", "weight": 0.10, "transform": "maximize_01"},
        {"path": "mean_quality_dimensions.factual_correctness", "weight": 0.08, "transform": "maximize_01"},
        {"path": "mean_quality_dimensions.context_consistency", "weight": 0.05, "transform": "maximize_01"},
        {
            "path": "median_reply_tokens",
            "weight": 0.04,
            "transform": "target_range",
            "minimum": 1,
            "maximum": 12,
            "soft_upper": 24,
        },
        {"path": "duplicate_reply_rate", "weight": 0.03, "transform": "minimize_01"},
    ]
}

# Recorded soup-lab chat DEV rows for comparison.
COMPARATORS = {
    "qwen3_06b (α=0)": {
        "mean_quality_score": 0.8334060191551368,
        "worst_decile_mean_quality_score": 0.6986897546897546,
        "complete_utterance_rate": 0.984313725490196,
        "observable_quality_pass_rate": 0.9941176470588236,
        "median_reply_tokens": 15.0,
        "duplicate_reply_rate": 0.27058823529411763,
        "objective": 0.8108460311942959,
        "showcase_hits": "14/64",
        "note": "soup-lab anchor",
    },
    "Halo uniform α=0.5": {
        "mean_quality_score": 0.8618833426704015,
        "worst_decile_mean_quality_score": 0.7063720821661997,
        "complete_utterance_rate": 1.0,
        "observable_quality_pass_rate": 0.996078431372549,
        "median_reply_tokens": 11.0,
        "duplicate_reply_rate": 0.3627450980392157,
        "objective": 0.839735497665733,
        "showcase_hits": "30/64",
        "note": "soup-lab uniform",
    },
    "global_best α=0.40": {
        "mean_quality_score": 0.8642626957247546,
        "worst_decile_mean_quality_score": 0.7128263305322128,
        "complete_utterance_rate": 0.9921568627450981,
        "observable_quality_pass_rate": 0.9980392156862745,
        "median_reply_tokens": 12.0,
        "duplicate_reply_rate": 0.34705882352941175,
        "objective": 0.8422793689415161,
        "showcase_hits": "n/a yet",
        "note": "soup-lab global_best",
    },
    "Halo Uncensored α=1": {
        "mean_quality_score": 0.8600535294117647,
        "worst_decile_mean_quality_score": 0.713343137254902,
        "complete_utterance_rate": 1.0,
        "observable_quality_pass_rate": 0.9980392156862745,
        "median_reply_tokens": 9.0,
        "duplicate_reply_rate": 0.43333333333333335,
        "objective": 0.8377454705882353,
        "showcase_hits": "31/64",
        "note": "soup-lab greedy",
    },
}


def main() -> None:
    if not (MODEL / "model.safetensors").is_file():
        raise SystemExit(f"Missing late_rineya weights: {MODEL}")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating late_rineya on CPU -> {OUT}", flush=True)
    summary, _rows = evaluate_model_bundled_v45(
        model_path=MODEL,
        tokenizer_path=TOKENIZER,
        dataset_path=DEV,
        output_dir=OUT,
        quality_config_path=QUALITY,
        protocol_name="official_no_think",
        protocol=PROTOCOL,
        trust_remote_code=True,
        resume=True,
    )
    objective = score_summary(summary, OBJECTIVE)
    late = {
        "mean_quality_score": summary.get("mean_quality_score"),
        "worst_decile_mean_quality_score": summary.get("worst_decile_mean_quality_score"),
        "complete_utterance_rate": summary.get("complete_utterance_rate"),
        "observable_quality_pass_rate": summary.get("observable_quality_pass_rate"),
        "median_reply_tokens": summary.get("median_reply_tokens"),
        "duplicate_reply_rate": summary.get("duplicate_reply_rate"),
        "objective": objective.score,
        "showcase_hits": "31/64",
        "note": "CrossDepth hand recipe; CPU chat DEV v4.5",
    }
    rows = dict(COMPARATORS)
    rows["late_rineya"] = late
    payload = {
        "protocol": "bundled_v45 official_no_think; 32 tok; eos_only; seed 20260820; CPU",
        "dataset": str(DEV),
        "late_rineya_model": str(MODEL),
        "rows": rows,
        "late_rineya_summary": summary,
        "late_rineya_objective": objective.to_dict(),
    }
    write_json(OUT / "comparison.json", payload)

    headers = [
        "name",
        "mean_quality",
        "worst_decile",
        "complete",
        "obs_pass",
        "median_tok",
        "dup_rate",
        "objective",
        "showcase",
    ]
    lines = [
        "# late_rineya vs Halo family — chat DEV v4.5",
        "",
        "Same Quality Evaluator v4.5 / `official_no_think` contract as the Halo-pair soup lab.",
        "`late_rineya` was scored on **CPU**; Halo rows are from the soup-lab GPU run.",
        "",
        "| Name | Mean quality | Worst decile | Complete | Obs pass | Median tok | Dup rate | Objective | Showcase |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    order = [
        "qwen3_06b (α=0)",
        "Halo uniform α=0.5",
        "global_best α=0.40",
        "Halo Uncensored α=1",
        "late_rineya",
    ]
    for name in order:
        r = rows[name]
        lines.append(
            "| {name} | {mq:.4f} | {wd:.4f} | {cu:.4f} | {op:.4f} | {mt:.1f} | {dr:.4f} | {obj:.4f} | {sh} |".format(
                name=name,
                mq=float(r["mean_quality_score"]),
                wd=float(r["worst_decile_mean_quality_score"]),
                cu=float(r["complete_utterance_rate"]),
                op=float(r["observable_quality_pass_rate"]),
                mt=float(r["median_reply_tokens"]),
                dr=float(r["duplicate_reply_rate"]),
                obj=float(r["objective"]),
                sh=r["showcase_hits"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Showcase: `late_rineya` 31/64 is +1 vs Halo 30/64 (noise band); not a vault core.",
            "- Chat DEV objective is the soup-lab weighted score, not showcase hits.",
            "",
        ]
    )
    (OUT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "comparison.md").read_text(encoding="utf-8"), flush=True)
    print(json.dumps({"late_rineya_objective": objective.score, "mean_quality": late["mean_quality_score"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
