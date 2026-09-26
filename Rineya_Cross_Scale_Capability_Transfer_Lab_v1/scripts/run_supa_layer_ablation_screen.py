"""Fast Supa-A1.1 leave-one-layer-out catastrophic screen (identity skip)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, sha256_file  # noqa: E402
from rineya_cross_scale.layer_ablation import (  # noqa: E402
    decoder_layers,
    final_token_logits,
    logit_cosine,
    skip_layer,
    verify_skip_hook,
)
from rineya_cross_scale.leakage import guard_non_sealed_path  # noqa: E402
from rineya_cross_scale.resource_trace import resource_trace  # noqa: E402
from rineya_cross_scale.validation import short_period_repetition  # noqa: E402

SEED = 17
MAX_NEW_TOKENS = 24
PRINTABLE = set(
    "\t\n\r "
    + "".join(chr(c) for c in range(32, 127))
)

DEFAULT_SUPA_CANDIDATES = (
    LAB_ROOT.parent
    / "Rineya_Published_Cores"
    / "Rineya-Supa-A1.1-1.1.0-devfreeze",
    LAB_ROOT.parent.parent
    / "Chat-Bubble"
    / "Rineya_Published_Cores"
    / "Rineya-Supa-A1.1-1.1.0-devfreeze",
)
EVIDENCE = LAB_ROOT / "cross_scale_evidence" / "15_supa_layer_ablation_screen"
SMOKE_PATH = LAB_ROOT / "data" / "evaluation" / "supa_layer_ablation_smoke.jsonl"


def resolve_supa(explicit: Path | None = None) -> Path:
    env = os.environ.get("RINEYA_SUPA_PATH")
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    if env:
        candidates.append(Path(env))
    candidates.extend(DEFAULT_SUPA_CANDIDATES)
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if (path / "model.safetensors").is_file() or (
            path / "model.safetensors.index.json"
        ).is_file():
            return path
    raise FileNotFoundError(
        "could not resolve Supa-A1.1 weights; set RINEYA_SUPA_PATH"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _render(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def weird_char_rate(text: str) -> float:
    if not text:
        return 1.0
    weird = sum(1 for ch in text if ch not in PRINTABLE)
    return weird / len(text)


def invalid_generation(response: str, token_ids: list[int]) -> tuple[bool, str | None]:
    if not response.strip():
        return True, "empty"
    if len(token_ids) >= 8:
        most_common = max(token_ids.count(value) for value in set(token_ids))
        if most_common / len(token_ids) >= 0.8:
            return True, "token_repetition"
    return False, None


def mean_gen_nll(
    model: Any,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> float | None:
    """Length-normalized NLL of generated tokens teacher-forced after the prompt."""
    if generated_ids.numel() == 0:
        return None
    full = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)
    with torch.no_grad():
        logits = model(input_ids=full).logits
    # Predict tokens at positions corresponding to generated_ids
    prompt_len = input_ids.shape[1]
    step_logits = logits[0, prompt_len - 1 : -1, :]
    log_probs = torch.log_softmax(step_logits.float(), dim=-1)
    token_lp = log_probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
    return float((-token_lp.mean()).item())


def run_condition(
    *,
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    intact_logits: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, Any]:
    predictions = []
    cosines = []
    for case in cases:
        rendered = _render(tokenizer, case["prompt"])
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]

        prompt_logits = final_token_logits(
            model, input_ids, encoded.get("attention_mask")
        )
        cosine = None
        if intact_logits is not None:
            base = intact_logits[case["case_id"]].float()
            cosine = logit_cosine(prompt_logits, base)
            cosines.append(cosine)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(case.get("max_new_tokens", MAX_NEW_TOKENS)),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_ids = generated[0, input_ids.shape[1] :]
        token_ids = new_ids.tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        invalid, reason = invalid_generation(text, token_ids)
        nll = mean_gen_nll(model, input_ids, new_ids.cpu())
        predictions.append(
            {
                "case_id": case["case_id"],
                "family": case.get("family"),
                "response": text,
                "token_ids": token_ids,
                "weird_char_rate": weird_char_rate(text),
                "invalid_generation": invalid,
                "invalid_reason": reason,
                "short_period_repetition": short_period_repetition(token_ids),
                "mean_nll": nll,
                "logit_cosine_vs_intact": cosine,
                "top1_vs_intact": (
                    None
                    if intact_logits is None
                    else int(
                        intact_logits[case["case_id"]].argmax().item()
                        == int(prompt_logits.argmax().item())
                    )
                ),
            }
        )
    weird_hits = sum(1 for row in predictions if row["weird_char_rate"] >= 0.25)
    invalid_rate = sum(1 for row in predictions if row["invalid_generation"]) / max(
        1, len(predictions)
    )
    mean_cosine = sum(cosines) / len(cosines) if cosines else None
    critical = (
        weird_hits >= 2
        or invalid_rate >= 0.5
        or (mean_cosine is not None and mean_cosine < 0.5)
    )
    return {
        "n_prompts": len(predictions),
        "weird_hits_ge_0_25": weird_hits,
        "invalid_rate": invalid_rate,
        "mean_logit_cosine_vs_intact": mean_cosine,
        "mean_nll": (
            sum(row["mean_nll"] for row in predictions if row["mean_nll"] is not None)
            / max(
                1,
                sum(1 for row in predictions if row["mean_nll"] is not None),
            )
        ),
        "critical": critical,
        "predictions": predictions,
    }


def catastrophe_score(summary: dict[str, Any]) -> float:
    """Higher = worse. Combines invalid rate, weird hits, and cosine collapse."""
    cosine = summary.get("mean_logit_cosine_vs_intact")
    cosine_term = 0.0 if cosine is None else max(0.0, 1.0 - float(cosine))
    return (
        3.0 * float(summary.get("invalid_rate") or 0.0)
        + 1.5 * float(summary.get("weird_hits_ge_0_25") or 0) / 8.0
        + 2.0 * cosine_term
    )


def write_ranking(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: -row["catastrophe_score"])
    critical = [row for row in ranked if row["critical"]]
    soft = [
        row
        for row in ranked
        if not row["critical"]
        and (
            (row.get("mean_logit_cosine_vs_intact") or 1.0) < 0.9
            or row.get("invalid_rate", 0) > 0
            or row.get("weird_hits_ge_0_25", 0) > 0
        )
    ]
    null = [row for row in ranked if row not in critical and row not in soft]
    payload = {
        "schema_version": "15.0",
        "ranked": ranked,
        "critical_layers": [row["layer"] for row in critical],
        "soft_layers": [row["layer"] for row in soft],
        "null_layers": [row["layer"] for row in null],
        "mergedev_fast_followup_justified": bool(critical),
    }
    atomic_write_json(output_dir / "ranking.json", payload)
    lines = [
        "# Supa layer ablation ranking",
        "",
        f"- Critical layers: `{payload['critical_layers']}`",
        f"- Soft layers: `{payload['soft_layers']}`",
        f"- Null layers: `{payload['null_layers']}`",
        f"- MERGEDEV-fast follow-up justified: `{payload['mergedev_fast_followup_justified']}`",
        "",
        "| layer | critical | score | invalid_rate | weird_hits | mean_logit_cos | mean_nll |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            "| {layer} | {critical} | {score:.4f} | {invalid:.3f} | {weird} | {cos} | {nll} |".format(
                layer=row["layer"],
                critical=row["critical"],
                score=row["catastrophe_score"],
                invalid=row["invalid_rate"],
                weird=row["weird_hits_ge_0_25"],
                cos=(
                    "n/a"
                    if row.get("mean_logit_cosine_vs_intact") is None
                    else f"{row['mean_logit_cosine_vs_intact']:.4f}"
                ),
                nll=f"{row['mean_nll']:.4f}" if row.get("mean_nll") is not None else "n/a",
            )
        )
    (output_dir / "ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--smoke", type=Path, default=SMOKE_PATH)
    parser.add_argument("--output", type=Path, default=EVIDENCE)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)

    smoke_path = guard_non_sealed_path(args.smoke)
    model_path = resolve_supa(args.model)
    output = args.output.expanduser().resolve()
    smoke_dir = output / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    cases = _read_jsonl(smoke_path)
    with resource_trace("supa_layer_ablation_screen", output / "resource.json") as trace:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        model.to(args.device)
        model.eval()
        trace.record_model_load()

        n_layers = len(decoder_layers(model))
        hook_check = verify_skip_hook(model, tokenizer, probe_layer=min(14, n_layers - 1))
        atomic_write_json(output / "hook_verification.json", hook_check)
        if not hook_check["ok"]:
            print("hook verification FAILED", hook_check)
            return 1

        torch.manual_seed(SEED)
        baseline_path = smoke_dir / "baseline_intact.json"
        logits_path = smoke_dir / "intact_logits.pt"
        device = next(model.parameters()).device

        if logits_path.is_file():
            intact_logits_tensors: dict[str, torch.Tensor] = torch.load(
                logits_path, map_location="cpu", weights_only=True
            )
        else:
            intact_logits_tensors = {}
            for case in cases:
                rendered = _render(tokenizer, case["prompt"])
                encoded = tokenizer(rendered, return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                logits = final_token_logits(
                    model, encoded["input_ids"], encoded.get("attention_mask")
                )
                intact_logits_tensors[case["case_id"]] = logits
            torch.save(intact_logits_tensors, logits_path)

        # Keep tensors for cosine; do not explode JSON with full vocab vectors.
        intact_logits = intact_logits_tensors

        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            baseline = run_condition(
                model=model,
                tokenizer=tokenizer,
                cases=cases,
                intact_logits=None,
                device=device,
            )
            baseline["layer"] = None
            baseline["condition"] = "intact"
            baseline["catastrophe_score"] = 0.0
            baseline["critical"] = False
            atomic_write_json(baseline_path, baseline)

        ranking_rows: list[dict[str, Any]] = []

        for layer in range(n_layers):
            layer_path = smoke_dir / f"layer_{layer:02d}.json"
            if layer_path.is_file():
                summary = json.loads(layer_path.read_text(encoding="utf-8"))
            else:
                started = time.monotonic()
                with skip_layer(model, layer):
                    summary = run_condition(
                        model=model,
                        tokenizer=tokenizer,
                        cases=cases,
                        intact_logits=intact_logits,
                        device=device,
                    )
                summary["layer"] = layer
                summary["condition"] = f"skip_layer_{layer:02d}"
                summary["wall_seconds"] = time.monotonic() - started
                summary["catastrophe_score"] = catastrophe_score(summary)
                atomic_write_json(layer_path, summary)
                trace.sample_rss()
                print(
                    f"layer {layer:02d} critical={summary['critical']} "
                    f"score={summary['catastrophe_score']:.3f} "
                    f"cos={summary['mean_logit_cosine_vs_intact']}"
                )
            ranking_rows.append(
                {
                    "layer": summary["layer"],
                    "critical": summary["critical"],
                    "catastrophe_score": summary["catastrophe_score"],
                    "invalid_rate": summary["invalid_rate"],
                    "weird_hits_ge_0_25": summary["weird_hits_ge_0_25"],
                    "mean_logit_cosine_vs_intact": summary["mean_logit_cosine_vs_intact"],
                    "mean_nll": summary["mean_nll"],
                }
            )

        ranking = write_ranking(ranking_rows, output)
        atomic_write_json(
            output / "run_manifest.json",
            {
                "schema_version": "15.0",
                "model_path": str(model_path),
                "smoke_path": str(smoke_path),
                "smoke_sha256": sha256_file(smoke_path),
                "seed": SEED,
                "max_new_tokens": MAX_NEW_TOKENS,
                "n_layers": n_layers,
                "device": args.device,
                "threads": args.threads,
                "hook_verification": hook_check,
                "critical_layers": ranking["critical_layers"],
                "mergedev_fast_followup_justified": ranking[
                    "mergedev_fast_followup_justified"
                ],
            },
        )
        print(json.dumps({
            "critical_layers": ranking["critical_layers"],
            "soft_layers": ranking["soft_layers"],
            "n_layers": n_layers,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
