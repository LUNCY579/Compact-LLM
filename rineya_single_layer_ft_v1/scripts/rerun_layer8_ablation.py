#!/usr/bin/env python3
"""Re-run layer-8 identity-skip smoke for intact Supa vs layer-8 LoRA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
REPO_ROOT = LAB_ROOT.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (  # noqa: E402
    abstention_overrides,
    assert_gate_disabled,
    atomic_write_json,
    load_yaml,
    read_jsonl,
    resolve_core_path,
    resolve_repo_path,
)
from layer_ablation import (  # noqa: E402
    final_token_logits,
    logit_cosine,
    skip_layer,
    verify_skip_hook,
)

PRINTABLE = set("\t\n\r " + "".join(chr(c) for c in range(32, 127)))
MAX_NEW_TOKENS = 24


def weird_char_rate(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for ch in text if ch not in PRINTABLE) / len(text)


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
    if generated_ids.numel() == 0:
        return None
    gen = generated_ids.to(input_ids.device)
    if gen.dim() == 1:
        gen = gen.unsqueeze(0)
    full = torch.cat([input_ids, gen], dim=1)
    with torch.no_grad():
        logits = model(input_ids=full).logits
    prompt_len = input_ids.shape[1]
    step_logits = logits[0, prompt_len - 1 : -1, :]
    log_probs = torch.log_softmax(step_logits.float(), dim=-1)
    token_lp = log_probs.gather(1, gen[0].unsqueeze(1)).squeeze(1)
    return float((-token_lp.mean()).item())


def render(tokenizer: Any, prompt: str) -> str:
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


def catastrophe_score(summary: dict[str, Any]) -> float:
    cosine = summary.get("mean_logit_cosine_vs_intact")
    cosine_term = 0.0 if cosine is None else max(0.0, 1.0 - float(cosine))
    return (
        3.0 * float(summary.get("invalid_rate") or 0.0)
        + 1.5 * float(summary.get("weird_hits_ge_0_25") or 0) / 8.0
        + 2.0 * cosine_term
    )


def bucket_of(summary: dict[str, Any]) -> str:
    if summary.get("critical"):
        return "critical"
    cos = summary.get("mean_logit_cosine_vs_intact")
    if (
        (cos is not None and cos < 0.9)
        or summary.get("invalid_rate", 0) > 0
        or summary.get("weird_hits_ge_0_25", 0) > 0
    ):
        return "soft"
    return "null"


def run_condition(
    *,
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    intact_logits: dict[str, torch.Tensor],
    layer: int,
    device: torch.device,
) -> dict[str, Any]:
    predictions = []
    cosines = []
    with skip_layer(model, layer):
        for case in cases:
            rendered = render(tokenizer, case["prompt"])
            encoded = tokenizer(rendered, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            input_ids = encoded["input_ids"]
            prompt_logits = final_token_logits(
                model, input_ids, encoded.get("attention_mask")
            )
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
            nll = mean_gen_nll(model, input_ids, new_ids)
            predictions.append(
                {
                    "case_id": case["case_id"],
                    "family": case.get("family"),
                    "response": text,
                    "weird_char_rate": weird_char_rate(text),
                    "invalid_generation": invalid,
                    "invalid_reason": reason,
                    "mean_nll": nll,
                    "logit_cosine_vs_intact": cosine,
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
    summary = {
        "layer": layer,
        "n_prompts": len(predictions),
        "weird_hits_ge_0_25": weird_hits,
        "invalid_rate": invalid_rate,
        "mean_logit_cosine_vs_intact": mean_cosine,
        "mean_nll": (
            sum(row["mean_nll"] for row in predictions if row["mean_nll"] is not None)
            / max(1, sum(1 for row in predictions if row["mean_nll"] is not None))
        ),
        "critical": critical,
        "predictions": predictions,
    }
    summary["catastrophe_score"] = catastrophe_score(summary)
    summary["bucket"] = bucket_of(summary)
    return summary


def collect_intact_logits(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    intact: dict[str, torch.Tensor] = {}
    for case in cases:
        rendered = render(tokenizer, case["prompt"])
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        intact[case["case_id"]] = final_token_logits(
            model, encoded["input_ids"], encoded.get("attention_mask")
        )
    return intact


def load_model(model_path: Path, adapter_dir: Path | None, dtype: torch.dtype) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        **abstention_overrides(),
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    assert_gate_disabled(model)
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "layer8_lora_puzzle_smoke.yaml",
    )
    parser.add_argument(
        "--condition",
        choices=("baseline", "layer8_lora", "layer8_qlora"),
        required=True,
    )
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_root = args.output_dir or resolve_repo_path(cfg["train"]["output_dir"])
    smoke_path = LAB_ROOT / cfg["eval"]["smoke_bank"]
    layer = int(cfg["eval"]["ablation_layer"])
    cases = read_jsonl(smoke_path)

    adapter_dir = None
    if args.condition in {"layer8_lora", "layer8_qlora"}:
        adapter_dir = args.adapter_dir or (output_root / "adapter")
        if not adapter_dir.exists():
            raise FileNotFoundError(adapter_dir)

    model_path = resolve_core_path(cfg["model"]["path"])
    model, tokenizer = load_model(model_path, adapter_dir, torch.float16)
    device = next(model.parameters()).device

    hook = verify_skip_hook(model, tokenizer, probe_layer=layer)
    if not hook["ok"]:
        raise RuntimeError(f"skip hook verification failed: {hook}")

    intact = collect_intact_logits(model, tokenizer, cases, device)
    summary = run_condition(
        model=model,
        tokenizer=tokenizer,
        cases=cases,
        intact_logits=intact,
        layer=layer,
        device=device,
    )
    summary["condition"] = args.condition
    summary["adapter_dir"] = str(adapter_dir) if adapter_dir else None
    summary["hook_verification"] = hook
    # Drop bulky prediction bodies from the ranking-facing copy; keep full under smoke/
    out_dir = output_root / "ablation" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "layer8_skip.json", summary)
    slim = {k: v for k, v in summary.items() if k != "predictions"}
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
