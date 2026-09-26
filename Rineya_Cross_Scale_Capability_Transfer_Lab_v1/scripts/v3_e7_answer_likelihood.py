"""v3 E7 — is there any capability gap to transfer in the first place?

The whole research line assumes Qwen3-4B and Qwen3-8B know things the 0.6B does
not. That premise has never been tested on this evaluation bank. E6 showed the
donors' residual streams are linearly redundant with the target's; if the donors
are also no better at the tasks, then there is nothing to transfer and the
negative result is not about transport at all.

This measures the teacher-forced log-probability that each model assigns to the
reference answer, which needs one forward pass per model and no generation.
Scores are per-answer-token so answers of different lengths are comparable.

Scope note: this reads calibration *answers*, where E4-E6 read prompts only.
Calibration is development data under the parent preregistration; the leakage
firewall covers the sealed holdout and Invisible-Limit TEST2, neither of which is
touched. Recorded in the artifact as a deliberate scope extension.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402
from rineya_cross_scale.streaming_forward import StreamingWeights, streaming_forward_many  # noqa: E402

TOKENIZER = Path("C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
                 "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")


def reference_answer(scorer: dict[str, Any]) -> str | None:
    """The single reference string a scorer accepts, where one exists."""
    kind = scorer.get("type")
    if kind == "keywords":
        return str(scorer["phrase"])
    if kind in ("choice", "numeric"):
        return str(scorer["value"])
    if kind == "json_exact":
        return json.dumps(scorer["value"], separators=(",", ":"))
    if kind == "exact":
        answers = scorer.get("answers") or []
        return str(answers[0]) if answers else None
    return None  # `constraints` has no single reference answer


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 E7 answer likelihood.")
    parser.add_argument("--models", nargs="+", default=["qwen4b", "qwen8b", "qwen06", "supa"])
    parser.add_argument("--dataset", type=Path,
                        default=LAB_ROOT / "data" / "evaluation" / "calibration.jsonl")
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
                        / "e7_answer_likelihood")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER))
    rows = [json.loads(line) for line in args.dataset.read_text("utf-8").splitlines() if line.strip()]

    cases, sequences, answer_slices = [], [], []
    for row in rows:
        answer = reference_answer(row["scorer"])
        if answer is None:
            continue
        prefix = tokenizer(render_prompt(tokenizer, row["prompt"]),
                           return_tensors="pt")["input_ids"][0]
        answer_ids = tokenizer(answer, return_tensors="pt",
                               add_special_tokens=False)["input_ids"][0]
        if answer_ids.numel() == 0:
            continue
        sequences.append(torch.cat([prefix, answer_ids]))
        # Predicting token at position p uses the state at p-1.
        answer_slices.append((int(prefix.shape[0]) - 1,
                              int(prefix.shape[0] + answer_ids.shape[0]) - 1,
                              answer_ids))
        cases.append({"case_id": row["case_id"], "family": row["family"],
                      "scorer_type": row["scorer"]["type"], "answer": answer,
                      "answer_tokens": int(answer_ids.shape[0])})

    positions = [torch.arange(start, stop) for start, stop, _ in answer_slices]
    print(f"scoring {len(cases)} of {len(rows)} calibration cases "
          f"({sum(c['answer_tokens'] for c in cases)} answer tokens)", flush=True)

    sources = {
        "qwen4b": ("stream", json.loads((LAB_ROOT / "cross_scale_evidence/02_architecture/qwen4b.json")
                                        .read_text("utf-8"))["resolved_path"]),
        "qwen8b": ("stream", json.loads((LAB_ROOT / "cross_scale_evidence/02_architecture/qwen8b.json")
                                        .read_text("utf-8"))["resolved_path"]),
        "qwen06": ("stream", str(TOKENIZER)),
        "supa": ("stream", "C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                           "Rineya-Supa-A1.1-1.1.0-devfreeze"),
    }

    scores: dict[str, np.ndarray] = {}
    resources: dict[str, Any] = {}
    for name in args.models:
        _, path = sources[name]
        started = time.perf_counter()
        print(f"[run ] {name}", flush=True)
        with StreamingWeights(Path(path)) as weights:
            with torch.inference_mode():
                produced = streaming_forward_many(
                    weights, sequences, capture_layers=(), want_logits=True,
                    logit_positions=positions,
                    progress=lambda d, t: (print(f"  [{name}] layer {d}/{t} "
                                                 f"({time.perf_counter() - started:.0f}s)", flush=True)
                                           if d % 9 == 0 else None),
                )
        per_case = np.zeros(len(cases), dtype=np.float64)
        for index, (got, (_, _, answer_ids)) in enumerate(zip(produced, answer_slices)):
            logits = got["logits"].double()
            log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            chosen = log_probs[torch.arange(answer_ids.shape[0]), answer_ids]
            per_case[index] = float(chosen.mean())  # mean log-prob per answer token
            del logits, log_probs, chosen
        scores[name] = per_case
        resources[name] = {"wall_seconds": time.perf_counter() - started}
        print(f"[done] {name} mean log-prob/token {per_case.mean():+.4f} "
              f"({resources[name]['wall_seconds']:.0f}s)", flush=True)
        del produced
        gc.collect()

    reference_model = "supa" if "supa" in scores else args.models[0]
    comparisons = {}
    for name in scores:
        if name == reference_model:
            continue
        delta = scores[name] - scores[reference_model]
        wins = int((delta > 0).sum())
        comparisons[f"{name}__vs__{reference_model}"] = {
            "mean_delta_log_prob_per_token": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "cases_donor_higher": wins,
            "cases_total": len(delta),
            "win_rate": wins / len(delta),
        }

    payload = {
        "schema_version": "1.0",
        "experiment": "v3_e7_answer_likelihood",
        "protocol": "amendment_v3_1_donor_side.md 3.1.0-diagnostic",
        "scope_extension": ("reads calibration reference answers; E4-E6 read prompts only. "
                            "Calibration is development data under the parent preregistration. "
                            "The sealed holdout and Invisible-Limit TEST2 are untouched."),
        "dataset_sha256": sha256_file(args.dataset),
        "cases_scored": len(cases),
        "cases_skipped": len(rows) - len(cases),
        "skipped_reason": "the `constraints` scorer has no single reference answer",
        "metric": "mean teacher-forced log-probability per answer token",
        "per_model_mean": {n: float(v.mean()) for n, v in scores.items()},
        "per_model_median": {n: float(np.median(v)) for n, v in scores.items()},
        "comparisons": comparisons,
        "per_case": [
            {**case, **{f"logprob_{n}": float(scores[n][i]) for n in scores}}
            for i, case in enumerate(cases)
        ],
        "resources": resources,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "answer_likelihood.json", payload)

    lines = [
        "# v3 E7 — Is There a Capability Gap to Transfer?",
        "",
        f"Teacher-forced log-probability of the reference answer, {len(cases)} of {len(rows)}",
        "calibration cases, mean per answer token. Higher is better. One forward pass per",
        "model, no generation.",
        "",
        "| model | mean log-prob / answer token | median |",
        "|---|---:|---:|",
    ]
    for name in args.models:
        lines.append(f"| {name} | {scores[name].mean():+.4f} | {np.median(scores[name]):+.4f} |")
    lines += ["", f"## Against {reference_model}", "",
              "| comparison | mean delta | median delta | cases higher | win rate |",
              "|---|---:|---:|---:|---:|"]
    for pair, value in comparisons.items():
        lines.append(f"| {pair} | {value['mean_delta_log_prob_per_token']:+.4f} | "
                     f"{value['median_delta']:+.4f} | "
                     f"{value['cases_donor_higher']}/{value['cases_total']} | "
                     f"{value['win_rate']:.3f} |")
    atomic_write_text(args.output / "answer_likelihood.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
