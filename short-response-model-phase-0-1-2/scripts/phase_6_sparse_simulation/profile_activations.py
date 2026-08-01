# Purpose: Collects per-layer decode hidden states, block contribution scores, and oracle masks.
from __future__ import annotations

"""Collect router-training examples and oracle block contribution labels.

The profiler freezes a dense reference, hooks each gated MLP, and records:
- the MLP input hidden state available to a future router;
- block contribution scores derived from |z_j| * ||W_down[:,j]||;
- an oracle mask retaining either a contribution fraction or a fixed ratio.

No sparse speed claim is allowed from this script. The dense model still runs in
full to produce the labels.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256_tree, write_json
from scripts.common.model_introspection import find_decoder_stack, first_parameter_device, get_gated_mlp
from scripts.common.sparse_mlp import block_scores_from_intermediate, top_block_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dense activation traces for Phase 6.")
    parser.add_argument("--model", required=True, help="Exact frozen dense reference checkpoint.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_6_sparse/router_dataset.pt")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--fixed-active-ratio", type=float, default=0.65,
        help="Experimental oracle compute budget used to create router labels; not a quality gate.",
    )
    parser.add_argument("--retain-contribution", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--max-samples-per-layer", type=int, default=20000)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument(
        "--positions",
        choices=("last", "assistant", "all"),
        default="last",
        help="Which token positions become router examples. 'last' best matches autoregressive decode.",
    )
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


def render_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token IDs and a boolean mask of positions eligible for tracing."""

    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        if messages[-1].get("role") == "assistant":
            prompt = apply_chat_template(
                tokenizer, messages[:-1], add_generation_prompt=True
            )
            full = apply_chat_template(
                tokenizer, messages, add_generation_prompt=False
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(
                full,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            mask = torch.zeros(len(full_ids), dtype=torch.bool)
            mask[min(len(prompt_ids), len(full_ids)) :] = True
            return torch.tensor(full_ids).unsqueeze(0), mask
        text = apply_chat_template(
            tokenizer, messages, add_generation_prompt=True
        )
    else:
        text = row.get("text") or row.get("prompt")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Calibration rows need messages, text, or prompt.")
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    return torch.tensor(encoded).unsqueeze(0), torch.ones(len(encoded), dtype=torch.bool)


def smallest_prefix_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0 < fraction <= 1:
        raise ValueError("retain_contribution must be in (0, 1].")
    sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
    cumulative = sorted_scores.cumsum(dim=-1)
    target = scores.sum(dim=-1, keepdim=True).clamp_min(1e-12) * fraction
    keep_counts = (cumulative < target).sum(dim=-1) + 1
    rank = torch.arange(scores.shape[-1], device=scores.device)
    sorted_mask = rank.view(*([1] * (scores.ndim - 1)), -1) < keep_counts.unsqueeze(-1)
    mask = torch.zeros_like(sorted_mask)
    return mask.scatter(-1, sorted_indices, sorted_mask)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    stack = find_decoder_stack(model)
    input_device = first_parameter_device(model)

    datasets: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"hidden": [], "scores": [], "oracle_mask": [], "record_id": [], "token_position": []}
    )
    current: dict[str, Any] = {}
    pending_hidden: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    for layer_index, layer in enumerate(stack.layers):
        mlp = get_gated_mlp(layer)
        key = str(layer_index)

        def mlp_pre_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], *, layer_key: str = key) -> None:
            hidden = inputs[0].detach()[0]
            positions = current["positions_cpu"].to(hidden.device)
            pending_hidden[layer_key] = hidden.index_select(0, positions).float().cpu()

        def down_pre_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], *, layer_key: str = key) -> None:
            if len(datasets[layer_key]["hidden"]) >= args.max_samples_per_layer:
                return
            z = inputs[0].detach()[0]
            positions = current["positions_cpu"].to(z.device)
            selected_z = z.index_select(0, positions)
            scores = block_scores_from_intermediate(selected_z, module.weight, args.block_size).cpu()
            if args.retain_contribution is not None:
                # Deprecated compatibility path for historical datasets only.
                oracle = smallest_prefix_mask(scores, args.retain_contribution).cpu()
            else:
                oracle = top_block_mask(scores, args.fixed_active_ratio).cpu()
            hidden = pending_hidden.pop(layer_key).cpu()
            remaining = args.max_samples_per_layer - len(datasets[layer_key]["hidden"])
            for row_index in range(min(remaining, hidden.shape[0])):
                datasets[layer_key]["hidden"].append(hidden[row_index])
                datasets[layer_key]["scores"].append(scores[row_index])
                datasets[layer_key]["oracle_mask"].append(oracle[row_index])
                datasets[layer_key]["record_id"].append(current["record_id"])
                datasets[layer_key]["token_position"].append(int(current["positions_cpu"][row_index]))

        handles.append(mlp.register_forward_pre_hook(mlp_pre_hook))
        handles.append(mlp.down_proj.register_forward_pre_hook(down_pre_hook))

    used = 0
    records = read_jsonl(args.calibration)[: args.max_records]
    try:
        for record_index, row in enumerate(records):
            try:
                ids, eligible = render_row(tokenizer, row, args.max_sequence_length)
            except ValueError:
                continue
            candidates = torch.nonzero(eligible, as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            if args.positions == "last":
                candidates = candidates[-1:]
            elif args.positions == "assistant" and isinstance(row.get("messages"), list) and row["messages"][-1].get("role") != "assistant":
                candidates = candidates[-1:]
            current["record_id"] = str(row.get("id", record_index))
            current["positions_cpu"] = candidates
            with torch.no_grad():
                model(input_ids=ids.to(input_device), use_cache=False)
            used += 1
    finally:
        for handle in handles:
            handle.remove()

    layer_payload: dict[str, Any] = {}
    summary_layers: dict[str, Any] = {}
    for layer_index in range(len(stack.layers)):
        key = str(layer_index)
        rows = datasets[key]
        if not rows["hidden"]:
            raise RuntimeError(f"Layer {key} produced no router samples.")
        hidden = torch.stack(rows["hidden"])
        scores = torch.stack(rows["scores"])
        masks = torch.stack(rows["oracle_mask"]).bool()
        layer_payload[key] = {
            "hidden": hidden,
            "block_scores": scores,
            "oracle_mask": masks,
            "record_id": rows["record_id"],
            "token_position": rows["token_position"],
        }
        normalized = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        retained = (normalized * masks).sum(dim=-1)
        summary_layers[key] = {
            "samples": int(hidden.shape[0]),
            "hidden_size": int(hidden.shape[-1]),
            "num_blocks": int(scores.shape[-1]),
            "mean_active_ratio": float(masks.float().mean().item()),
            "mean_contribution_retained": float(retained.mean().item()),
        }

    payload = {
        "version": 2,
        "model": args.model,
        "model_fingerprint": sha256_tree(Path(args.model)),
        "calibration": str(args.calibration),
        "records_used": used,
        "block_size": args.block_size,
        "retain_contribution": args.retain_contribution,
        "fixed_active_ratio": args.fixed_active_ratio,
        "positions": args.positions,
        "layers": layer_payload,
        "warning": "Dense profiling artifact. It demonstrates oracle labels, not runtime speed or memory savings.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    write_json(
        args.out.with_suffix(".json"),
        {
            "version": payload["version"],
            "model": args.model,
            "model_fingerprint": payload["model_fingerprint"],
            "calibration": str(args.calibration),
            "records_used": used,
            "block_size": args.block_size,
            "layers": summary_layers,
            "warning": payload["warning"],
        },
    )
    print(args.out)


if __name__ == "__main__":
    main()
