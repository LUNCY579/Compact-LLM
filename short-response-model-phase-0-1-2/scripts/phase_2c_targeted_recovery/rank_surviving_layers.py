from __future__ import annotations

"""Rank surviving Phase 2B layers for targeted Phase 2C recovery.

Signals:
1. first-order Taylor sensitivity of assistant-target loss to each layer output;
2. target-NLL increase when that layer is temporarily bypassed;
3. optional final-stage structural-deletion damage from the Phase 2B registry.

The script does not train adapters and does not use sealed data.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, read_jsonl, write_json
from scripts.common.model_introspection import find_decoder_stack
from scripts.phase_2c_targeted_recovery.phase2c_common import combine_importance, relative_to_root


def apply_template(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def tokenized_example(tokenizer: Any, row: dict[str, Any], max_length: int, device: torch.device) -> tuple[torch.Tensor, int]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"Record {row.get('id')} must end with an assistant target.")
    prompt_text = apply_template(tokenizer, messages[:-1], generation=True)
    full_text = apply_template(tokenizer, messages, generation=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(f"Record {row.get('id')} has no target tokens after tokenization.")
    return torch.tensor([full_ids], dtype=torch.long, device=device), len(prompt_ids)


def target_loss(logits: torch.Tensor, input_ids: torch.Tensor, prompt_length: int) -> tuple[torch.Tensor, int]:
    shifted_logits = logits[:, :-1, :].float()
    shifted_targets = input_ids[:, 1:]
    start = max(0, prompt_length - 1)
    target_logits = shifted_logits[:, start:, :]
    target_ids = shifted_targets[:, start:]
    losses = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    )
    return losses.sum(), int(losses.numel())


def replace_hidden_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    if torch.is_tensor(output):
        return hidden
    raise TypeError(f"Unsupported decoder-layer output type: {type(output).__name__}")


def skip_hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
    if not inputs or not torch.is_tensor(inputs[0]):
        raise RuntimeError("Decoder layer did not receive hidden states as its first input.")
    return replace_hidden_output(output, inputs[0])


def hidden_tensor(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unsupported decoder-layer hidden output: {type(value).__name__}")
    return value


def lineage_for_model(model_path: Path, layer_count: int) -> list[int]:
    candidates = [
        model_path / "phase_2b_stage_manifest.json",
        model_path / "depth_reduction_manifest.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = read_json(path)
        lineage = payload.get("retained_original_layer_ids")
        if isinstance(lineage, list) and len(lineage) == layer_count:
            return [int(value) for value in lineage]
    return list(range(layer_count))


def structural_signals(registry_path: Path | None, original_ids: list[int]) -> dict[int, float]:
    if registry_path is None or not registry_path.exists():
        return {}
    payload = read_json(registry_path)
    rows = payload.get("candidates", payload if isinstance(payload, list) else [])
    raw_rows = [row for row in rows if row.get("variant_kind") == "raw deletion"]
    if not raw_rows:
        return {}
    latest_stage = max(int(row.get("stage", 0)) for row in raw_rows)
    latest = [row for row in raw_rows if int(row.get("stage", 0)) == latest_stage]
    scores: dict[int, float] = {}
    for row in latest:
        original = int(row.get("dropped_original_layer_id", -1))
        if original not in original_ids:
            continue
        assessment = row.get("assessment") or {}
        # All terms are damage-like and therefore larger means more important.
        score = (
            0.50 * float(assessment.get(
                "quality_score_loss_fraction",
                assessment.get("heuristic_score_loss_fraction", 0.0),
            ))
            + 0.25 * float(assessment.get("semantic_pass_loss", 0.0))
            + 0.15 * float(assessment.get(
                "maximum_category_quality_loss",
                assessment.get("maximum_category_score_loss", assessment.get("max_category_loss", 0.0)),
            ))
            + 0.10 * float(assessment.get(
                "target_nll_increase_fraction",
                assessment.get("relative_nll_increase", 0.0),
            ))
        )
        scores[original] = score
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank surviving layers for Phase 2C targeted recovery.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase2b-registry", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--gradient-weight", type=float, default=0.35)
    parser.add_argument("--ablation-weight", type=float, default=0.45)
    parser.add_argument("--structural-weight", type=float, default=0.20)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable and --require-cuda was requested.")
    rows = read_jsonl(args.calibration_file)[: args.limit]
    if not rows:
        raise ValueError("Calibration file contains no rows.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    model.config.use_cache = False
    model.eval()
    stack = find_decoder_stack(model)
    layer_count = len(stack.layers)
    model_path = Path(args.model_id)
    original_ids = lineage_for_model(model_path, layer_count) if model_path.exists() else list(range(layer_count))
    structural_by_original = structural_signals(args.phase2b_registry, original_ids)

    # Freeze weights and request gradients only through hidden activations.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    taylor_sums = [0.0] * layer_count
    taylor_counts = [0] * layer_count
    baseline_loss_sum = 0.0
    baseline_tokens = 0

    for row_index, row in enumerate(rows, 1):
        captured: dict[int, torch.Tensor] = {}
        hooks = []
        for layer_id, layer in enumerate(stack.layers):
            def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any, *, index: int = layer_id) -> Any:
                hidden = hidden_tensor(output)
                hidden.retain_grad()
                captured[index] = hidden
                return output
            hooks.append(layer.register_forward_hook(capture))
        try:
            input_ids, prompt_length = tokenized_example(tokenizer, row, args.max_sequence_length, model.device)
            model.zero_grad(set_to_none=True)
            output = model(input_ids=input_ids, use_cache=False)
            loss_sum, token_count = target_loss(output.logits, input_ids, prompt_length)
            (loss_sum / max(token_count, 1)).backward()
            baseline_loss_sum += float(loss_sum.detach().item())
            baseline_tokens += token_count
            for layer_id, hidden in captured.items():
                if hidden.grad is None:
                    continue
                taylor_sums[layer_id] += float((hidden.detach().float() * hidden.grad.detach().float()).abs().mean().item())
                taylor_counts[layer_id] += 1
        finally:
            for hook in hooks:
                hook.remove()
        print(f"[gradient {row_index:03d}/{len(rows):03d}] {row.get('id')}")

    baseline_nll = baseline_loss_sum / max(baseline_tokens, 1)
    ablated_nll: list[float] = []
    for layer_id, layer in enumerate(stack.layers):
        handle = layer.register_forward_hook(skip_hook)
        loss_sum_total = 0.0
        token_total = 0
        try:
            with torch.inference_mode():
                for row in rows:
                    input_ids, prompt_length = tokenized_example(tokenizer, row, args.max_sequence_length, model.device)
                    output = model(input_ids=input_ids, use_cache=False)
                    loss_sum, token_count = target_loss(output.logits, input_ids, prompt_length)
                    loss_sum_total += float(loss_sum.item())
                    token_total += token_count
        finally:
            handle.remove()
        value = loss_sum_total / max(token_total, 1)
        ablated_nll.append(value)
        print(f"[ablation {layer_id + 1:02d}/{layer_count:02d}] layer={layer_id} nll={value:.6f}")

    signal_rows: list[dict[str, Any]] = []
    for current_id, original_id in enumerate(original_ids):
        gradient = taylor_sums[current_id] / max(taylor_counts[current_id], 1)
        structural = structural_by_original.get(original_id)
        signal_rows.append(
            {
                "current_layer_id": current_id,
                "original_layer_id": original_id,
                "gradient_taylor_score": gradient,
                "ablated_target_nll": ablated_nll[current_id],
                "ablation_nll_increase": ablated_nll[current_id] - baseline_nll,
                "structural_sensitivity": structural,
            }
        )
    ranking = combine_importance(
        signal_rows,
        gradient_weight=args.gradient_weight,
        ablation_weight=args.ablation_weight,
        structural_weight=args.structural_weight,
    )
    write_json(
        args.output,
        {
            "manifest_version": 1,
            "model_id": args.model_id,
            "calibration_file": relative_to_root(args.calibration_file, ROOT),
            "calibration_case_count": len(rows),
            "sealed_data_used": False,
            "layer_count": layer_count,
            "original_layer_lineage": original_ids,
            "baseline_target_nll": baseline_nll,
            "weights": {
                "gradient_taylor": args.gradient_weight,
                "ablation_nll": args.ablation_weight,
                "phase2b_structural": args.structural_weight,
            },
            "signal_definitions": {
                "gradient_taylor": "mean absolute hidden-state times target-loss gradient",
                "ablation_nll": "target NLL increase when the layer output is replaced by its input",
                "phase2b_structural": "optional final-stage raw-deletion damage from Phase 2B",
            },
            "ranking": ranking,
            "interpretation_boundary": (
                "This ranks where limited recovery updates may be useful. It does not imply that "
                "high-ranked layers can be removed or that low-ranked layers are universally unimportant."
            ),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
