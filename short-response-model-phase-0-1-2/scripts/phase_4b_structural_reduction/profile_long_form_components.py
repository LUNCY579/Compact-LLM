# Purpose: Measures attention-head and MLP-channel contribution across short and late continuation positions.
from __future__ import annotations

"""Profile components that are disproportionately important late in responses.

This is an attribution experiment, not proof that a component has a single
human-interpretable purpose. It teacher-forces complete assistant targets and
compares contribution proxies in early answer positions with later positions.
The output is used to build conservative structured-pruning plans.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, write_json
from scripts.common.model_introspection import (
    find_decoder_stack,
    first_parameter_device,
    get_gated_mlp,
    get_self_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile MLP channels and attention heads used early versus late in assistant continuations."
    )
    parser.add_argument("--model", required=True, help="Frozen dense reference model or merged checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="JSONL with conversations ending in assistant targets.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/phase_4b_structural_reduction/long_form_profile.pt")
    parser.add_argument("--short-cutoff", type=int, default=24, help="Assistant token positions below this are the short-answer bucket.")
    parser.add_argument("--late-start", type=int, default=48, help="Assistant token positions at/after this are the long-continuation bucket.")
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Render Qwen and generic chat templates without assuming a thinking flag."""

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_record(tokenizer: Any, row: dict[str, Any], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Each profiling record must contain messages ending in an assistant target.")
    prompt_text = apply_chat_template(
        tokenizer, messages[:-1], add_generation_prompt=True
    )
    full_text = apply_chat_template(
        tokenizer, messages, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Assistant target was lost during tokenization/truncation.")
    ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
    assistant_mask = torch.zeros(len(full_ids), dtype=torch.bool)
    assistant_mask[len(prompt_ids) :] = True
    return ids, assistant_mask


class Accumulator:
    def __init__(self) -> None:
        self.sums: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, layer: str, bucket: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        vector = values.detach().float().sum(dim=0).cpu()
        if bucket not in self.sums[layer]:
            self.sums[layer][bucket] = torch.zeros_like(vector)
        self.sums[layer][bucket] += vector
        self.counts[layer][bucket] += int(values.shape[0])

    def mean(self, layer: str, bucket: str, width: int) -> torch.Tensor:
        count = self.counts[layer].get(bucket, 0)
        if count == 0:
            return torch.zeros(width)
        return self.sums[layer][bucket] / count


def main() -> None:
    args = parse_args()
    if args.late_start <= args.short_cutoff:
        raise ValueError("--late-start must be greater than --short-cutoff.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=args.device_map,
    )
    model.eval()
    stack = find_decoder_stack(model)
    device = first_parameter_device(model)

    mlp_acc = Accumulator()
    head_acc = Accumulator()
    current_positions: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    # Hooks operate on teacher-forced hidden states. The down-projection input
    # is the gated MLP intermediate z; the o-projection input is concatenated
    # per-head attention output before it is mixed back into hidden size.
    for layer_index, layer in enumerate(stack.layers):
        mlp = get_gated_mlp(layer)
        attention = get_self_attention(layer)
        mlp_name = str(layer_index)
        down_norm = mlp.down_proj.weight.detach().float().norm(dim=0).cpu()
        o_weight = attention.o_proj.weight.detach().float().cpu()
        num_heads = int(getattr(attention, "num_heads", getattr(model.config, "num_attention_heads")))
        head_dim = attention.q_proj.out_features // num_heads
        o_head_norm = o_weight.reshape(o_weight.shape[0], num_heads, head_dim).norm(dim=(0, 2))

        def down_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], *, key: str = mlp_name, norm: torch.Tensor = down_norm) -> None:
            z = inputs[0].detach()[0].float().cpu()
            positions = current_positions["assistant_positions"]
            if positions.numel() == 0:
                return
            selected = z.index_select(0, positions)
            score = selected.abs() * norm
            relative = current_positions["relative_positions"]
            mlp_acc.add(key, "short", score[relative < args.short_cutoff])
            mlp_acc.add(key, "late", score[relative >= args.late_start])

        def o_hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            *,
            key: str = mlp_name,
            heads: int = num_heads,
            dim: int = head_dim,
            norms: torch.Tensor = o_head_norm,
        ) -> None:
            value = inputs[0].detach()[0].float().cpu()
            positions = current_positions["assistant_positions"]
            if positions.numel() == 0:
                return
            selected = value.index_select(0, positions).reshape(-1, heads, dim)
            score = selected.norm(dim=-1) * norms
            relative = current_positions["relative_positions"]
            head_acc.add(key, "short", score[relative < args.short_cutoff])
            head_acc.add(key, "late", score[relative >= args.late_start])

        handles.append(mlp.down_proj.register_forward_pre_hook(down_hook))
        handles.append(attention.o_proj.register_forward_pre_hook(o_hook))

    records = read_jsonl(args.data)[: args.max_records]
    used = 0
    short_tokens = 0
    late_tokens = 0
    try:
        for row in records:
            try:
                input_ids, assistant_mask = render_record(tokenizer, row, args.max_sequence_length)
            except ValueError:
                continue
            assistant_positions = torch.nonzero(assistant_mask, as_tuple=False).flatten()
            if assistant_positions.numel() == 0:
                continue
            relative = torch.arange(assistant_positions.numel(), dtype=torch.long)
            current_positions["assistant_positions"] = assistant_positions
            current_positions["relative_positions"] = relative
            short_tokens += int((relative < args.short_cutoff).sum().item())
            late_tokens += int((relative >= args.late_start).sum().item())
            with torch.no_grad():
                model(input_ids=input_ids.to(device), use_cache=False)
            used += 1
    finally:
        for handle in handles:
            handle.remove()

    if short_tokens == 0 or late_tokens == 0:
        raise RuntimeError(
            "The profiling corpus must contain both early assistant tokens and tokens at/after --late-start. "
            f"Observed short={short_tokens}, late={late_tokens}."
        )

    epsilon = 1e-8
    layers_payload: dict[str, Any] = {}
    for layer_index, layer in enumerate(stack.layers):
        mlp = get_gated_mlp(layer)
        attention = get_self_attention(layer)
        key = str(layer_index)
        mlp_width = mlp.gate_proj.out_features
        num_heads = int(getattr(attention, "num_heads", getattr(model.config, "num_attention_heads")))
        mlp_short = mlp_acc.mean(key, "short", mlp_width)
        mlp_late = mlp_acc.mean(key, "late", mlp_width)
        head_short = head_acc.mean(key, "short", num_heads)
        head_late = head_acc.mean(key, "late", num_heads)
        layers_payload[key] = {
            "mlp": {
                "short_mean": mlp_short,
                "late_mean": mlp_late,
                "late_to_short": (mlp_late + epsilon) / (mlp_short + epsilon),
                "long_specific_score": torch.relu(mlp_late - mlp_short) / (mlp_late + mlp_short + epsilon),
            },
            "attention": {
                "short_mean": head_short,
                "late_mean": head_late,
                "late_to_short": (head_late + epsilon) / (head_short + epsilon),
                "long_specific_score": torch.relu(head_late - head_short) / (head_late + head_short + epsilon),
            },
        }

    payload = {
        "version": 1,
        "model": args.model,
        "short_cutoff": args.short_cutoff,
        "late_start": args.late_start,
        "records_used": used,
        "short_token_observations": short_tokens,
        "late_token_observations": late_tokens,
        "num_attention_heads": int(getattr(model.config, "num_attention_heads")),
        "num_key_value_heads": int(getattr(model.config, "num_key_value_heads", getattr(model.config, "num_attention_heads"))),
        "layers": layers_payload,
        "warning": (
            "Attribution proxy only. High long-specific score does not prove a component exclusively stores long-form ability. "
            "Every pruning plan requires dense controls and post-pruning evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    summary = {
        key: {
            "top_mlp_long_specific_channels": torch.topk(value["mlp"]["long_specific_score"], k=min(16, value["mlp"]["long_specific_score"].numel())).indices.tolist(),
            "top_attention_long_specific_heads": torch.topk(value["attention"]["long_specific_score"], k=min(8, value["attention"]["long_specific_score"].numel())).indices.tolist(),
        }
        for key, value in layers_payload.items()
    }
    write_json(args.output.with_suffix(".json"), {**{k: payload[k] for k in payload if k != "layers"}, "layer_summary": summary})
    print(args.output)


if __name__ == "__main__":
    main()
