from __future__ import annotations

"""Profile residual, MLP-channel, and attention-head usage on Phase 9 data."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import environment, read_jsonl, sha256, write_json
from scripts.common.model_introspection import find_decoder_stack, get_gated_mlp, get_self_attention
from scripts.phase_9_student_distillation.train_student import apply_chat_template
from scripts.phase_9_student_distillation.phase9_common import model_parameter_breakdown, tree_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a Phase 9 student for structured reduction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--include-assistant-target", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def extract_messages(row: dict[str, Any], include_assistant_target: bool) -> list[dict[str, Any]]:
    messages = [dict(item) for item in (row.get("messages") or [])]
    if not messages:
        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError(f"{row.get('id')} has no messages or question")
        messages = [{"role": "user", "content": question}]
    if not include_assistant_target and messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return messages


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    rows = read_jsonl(args.data)
    if args.max_records > 0:
        rows = rows[: args.max_records]
    if not rows:
        raise ValueError("Structural profile data is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
        model = model.merge_and_unload(safe_merge=True)
    model.eval()
    stack = find_decoder_stack(model)
    layer_count = len(stack.layers)
    first_attention = get_self_attention(stack.layers[0])
    q_heads = int(getattr(first_attention, "num_heads", getattr(model.config, "num_attention_heads")))
    head_dim = int(first_attention.q_proj.out_features // q_heads)
    hidden_size = int(model.get_input_embeddings().embedding_dim)

    hidden_sum = torch.zeros(hidden_size, dtype=torch.float64)
    hidden_count = 0
    mlp_sum: dict[int, torch.Tensor] = {}
    mlp_count: dict[int, int] = {}
    attention_sum: dict[int, torch.Tensor] = {}
    attention_count: dict[int, int] = {}
    pending_gate: dict[int, torch.Tensor] = {}
    hooks: list[Any] = []

    def layer_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        nonlocal hidden_count
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        value = inputs[0].detach().float()
        reduced = value.pow(2).sum(dim=tuple(range(value.ndim - 1))).cpu().double()
        hidden_sum.add_(reduced)
        hidden_count += int(value.numel() // value.shape[-1])

    for layer_index, layer in enumerate(stack.layers):
        mlp = get_gated_mlp(layer)
        attention = get_self_attention(layer)
        mlp_width = int(mlp.gate_proj.out_features)
        mlp_sum[layer_index] = torch.zeros(mlp_width, dtype=torch.float64)
        mlp_count[layer_index] = 0
        attention_sum[layer_index] = torch.zeros(q_heads, dtype=torch.float64)
        attention_count[layer_index] = 0

        # Aggregate residual-stream coordinate importance across every retained block,
        # not only the first block. This produces a global hidden-dimension ranking
        # that is less sensitive to one layer's activation scale.
        hooks.append(layer.register_forward_pre_hook(layer_pre_hook))

        def gate_hook(_module: Any, _inputs: tuple[Any, ...], output: Any, *, idx: int = layer_index) -> None:
            if isinstance(output, torch.Tensor):
                pending_gate[idx] = output.detach()

        def up_hook(_module: Any, _inputs: tuple[Any, ...], output: Any, *, idx: int = layer_index, act_fn: Any = mlp.act_fn) -> None:
            gate = pending_gate.pop(idx, None)
            if gate is None or not isinstance(output, torch.Tensor):
                return
            product = act_fn(gate.float()) * output.detach().float()
            reduced = product.abs().sum(dim=tuple(range(product.ndim - 1))).cpu().double()
            mlp_sum[idx].add_(reduced)
            mlp_count[idx] += int(product.numel() // product.shape[-1])

        def oproj_pre_hook(_module: Any, inputs: tuple[Any, ...], *, idx: int = layer_index) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            value = inputs[0].detach().float()
            if value.shape[-1] != q_heads * head_dim:
                raise RuntimeError(
                    f"Layer {idx} attention o_proj input has {value.shape[-1]} channels; "
                    f"expected {q_heads * head_dim}."
                )
            grouped = value.reshape(*value.shape[:-1], q_heads, head_dim)
            reduced = grouped.pow(2).sum(dim=tuple(range(grouped.ndim - 2)) + (-1,)).cpu().double()
            attention_sum[idx].add_(reduced)
            attention_count[idx] += int(grouped.numel() // (q_heads * head_dim))

        hooks.append(mlp.gate_proj.register_forward_hook(gate_hook))
        hooks.append(mlp.up_proj.register_forward_hook(up_hook))
        hooks.append(attention.o_proj.register_forward_pre_hook(oproj_pre_hook))

    device = next(model.parameters()).device
    processed = 0
    token_count = 0
    try:
        with torch.no_grad():
            for row in rows:
                messages = extract_messages(row, args.include_assistant_target)
                text = apply_chat_template(tokenizer, messages, generation=False)
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_sequence_length,
                    add_special_tokens=False,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, use_cache=False)
                processed += 1
                token_count += int(encoded["attention_mask"].sum().item())
                print(f"Profiled {processed}/{len(rows)}: {row.get('id')}")
    finally:
        for hook in hooks:
            hook.remove()

    if processed == 0 or hidden_count == 0:
        raise RuntimeError("No profile activations were collected")
    profile = {
        "schema_version": "1.0",
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "records": processed,
        "tokens": token_count,
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "num_attention_heads": q_heads,
        "head_dim": head_dim,
        "hidden_importance": (hidden_sum / max(1, hidden_count)).float(),
        "mlp_importance": {
            str(index): (values / max(1, mlp_count[index])).float()
            for index, values in mlp_sum.items()
        },
        "attention_head_importance": {
            str(index): (values / max(1, attention_count[index])).float()
            for index, values in attention_sum.items()
        },
        "parameter_breakdown": model_parameter_breakdown(model),
        "model_checkpoint_bytes": tree_bytes(Path(args.model)) if Path(args.model).exists() else None,
        "environment": environment(),
        "interpretation": (
            "Activation magnitudes are candidate-selection evidence only. They do not prove that a coordinate, "
            "MLP channel, head group, or layer is semantically unnecessary. Every structural candidate requires "
            "fresh reload, RAG evaluation, recovery training, and human review."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output)
    summary_path = args.summary or args.output.with_suffix(".json")
    write_json(
        summary_path,
        {
            key: value
            for key, value in profile.items()
            if key not in {"hidden_importance", "mlp_importance", "attention_head_importance"}
        }
        | {
            "profile_path": str(args.output),
            "profile_bytes": args.output.stat().st_size,
            "hidden_importance_min": float(profile["hidden_importance"].min().item()),
            "hidden_importance_max": float(profile["hidden_importance"].max().item()),
        },
    )
    print(f"Profile: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
