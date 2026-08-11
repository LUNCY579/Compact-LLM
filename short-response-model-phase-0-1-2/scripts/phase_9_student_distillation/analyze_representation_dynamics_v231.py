from __future__ import annotations

"""Cross-format representation diagnostics across selected v2.3.1 checkpoints."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import (
    apply_chat_template,
    discover_adapters,
    nearest_steps,
    read_json,
    read_jsonl,
    write_json,
)


def layer_indices(num_layers: int, positions: list[float]) -> list[int]:
    # output_hidden_states includes embedding output at index 0, then layer outputs 1..L.
    values = set()
    for position in positions:
        position = min(1.0, max(0.0, float(position)))
        layer = int(round(position * max(0, num_layers - 1)))
        values.add(layer)
    return sorted(values)


def build(base_model: str, adapter: str | None) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True)
    if adapter:
        if PeftModel is None:
            raise RuntimeError("peft is required")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tok


def encode_rows(model: Any, tok: Any, rows: list[dict[str, Any]], layers: list[int], max_seq: int) -> dict[int, torch.Tensor]:
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    device = next(model.parameters()).device
    for row in rows:
        messages = [dict(message) for message in (row.get("messages") or [])]
        prompt = apply_chat_template(tok, messages[:-1], generation=True)
        encoded = tok(prompt, return_tensors="pt", add_special_tokens=False)
        if int(encoded["input_ids"].shape[1]) > max_seq:
            raise RuntimeError(f"prompt too long: {row.get('id')}")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states
        for layer in layers:
            # layer 0 is transformer layer 0 => hidden_states[1].
            vector = hidden[layer + 1][0, -1].detach().float().cpu()
            collected[layer].append(vector)
    return {layer: torch.stack(vectors) for layer, vectors in collected.items()}


def centered_gram(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    return x @ x.T


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    k = centered_gram(x)
    l = centered_gram(y)
    n = k.shape[0]
    if n <= 1:
        return float("nan")
    h = torch.eye(n) - torch.full((n, n), 1.0 / n)
    kc = h @ k @ h
    lc = h @ l @ h
    numerator = float(torch.sum(kc * lc).item())
    denominator = math.sqrt(max(1e-30, float(torch.sum(kc * kc).item()) * float(torch.sum(lc * lc).item())))
    return numerator / denominator


def effective_rank(x: torch.Tensor) -> float:
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    total = float(singular.sum().item())
    if total <= 0:
        return 0.0
    p = singular / total
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()).item())


def centroid_accuracy(train: torch.Tensor, train_labels: list[str], test: torch.Tensor, test_labels: list[str]) -> float | None:
    labels = sorted(set(train_labels))
    if len(labels) < 2 or not test_labels:
        return None
    centroids = {}
    train_n = F.normalize(train.float(), dim=-1)
    test_n = F.normalize(test.float(), dim=-1)
    for label in labels:
        ids = [i for i, value in enumerate(train_labels) if value == label]
        if ids:
            centroids[label] = F.normalize(train_n[ids].mean(dim=0), dim=0)
    correct = 0
    total = 0
    for vector, gold in zip(test_n, test_labels):
        if gold not in centroids:
            continue
        pred = max(centroids, key=lambda label: float(torch.dot(vector, centroids[label]).item()))
        correct += pred == gold
        total += 1
    return correct / total if total else None


def separation_gap(x: torch.Tensor, labels: list[str]) -> float | None:
    x = F.normalize(x.float(), dim=-1)
    within: list[float] = []
    between: list[float] = []
    # Deterministic capped pair scan avoids O(n^2) blowup at larger probe sizes.
    cap = min(len(labels), 180)
    for i in range(cap):
        for j in range(i + 1, cap):
            score = float(torch.dot(x[i], x[j]).item())
            (within if labels[i] == labels[j] else between).append(score)
    if not within or not between:
        return None
    return statistics.mean(within) - statistics.mean(between)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--training-root", type=Path, required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-probe", type=Path, required=True)
    ap.add_argument("--ood-probe", type=Path, required=True)
    ap.add_argument("--time-curve", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--profile", choices=("pilot", "standard", "extended"), default="standard")
    ap.add_argument("--max-sequence-length", type=int, default=768)
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    for path in (args.train_probe, args.ood_probe):
        if "sealed" in str(path).casefold():
            raise ValueError("representation diagnostics refuse sealed data")
    cfg = read_json(args.config)
    train_rows = read_jsonl(args.train_probe)
    ood_rows = read_jsonl(args.ood_probe)
    train_labels = [str(row.get("reasoning_family") or "") for row in train_rows]
    ood_labels = [str(row.get("reasoning_family") or "") for row in ood_rows]

    checkpoints = discover_adapters(args.training_root)
    available = sorted(int(item["step"]) for item in checkpoints)
    by_step = {int(item["step"]): item for item in checkpoints}
    max_step = max(available)
    count = int(cfg["profiles"][args.profile]["representation_checkpoint_count"])
    fractions = [i / max(1, count - 1) for i in range(count)]
    chosen = set(nearest_steps(available, fractions, max_step))
    time_rows = read_json(args.time_curve).get("rows") or []
    ranked = []
    for left, right in zip(time_rows, time_rows[1:]):
        gain = float(right.get("ood_answerable_correctness") or 0) - float(left.get("ood_answerable_correctness") or 0)
        ranked.append((gain, int(right["step"])))
    if ranked:
        chosen.add(max(ranked)[1])
    steps = sorted(chosen)

    # Determine selected transformer layers from the anchor config.
    from transformers import AutoConfig
    model_cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    num_layers = int(getattr(model_cfg, "num_hidden_layers", getattr(model_cfg, "n_layer", 0)))
    layers = layer_indices(num_layers, [float(v) for v in cfg["representation"]["layer_positions"]])

    anchor_model, anchor_tok = build(args.base_model, None)
    anchor_train = encode_rows(anchor_model, anchor_tok, train_rows, layers, args.max_sequence_length)
    anchor_ood = encode_rows(anchor_model, anchor_tok, ood_rows, layers, args.max_sequence_length)
    del anchor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records: list[dict[str, Any]] = []
    for step in steps:
        item = by_step[step]
        if step == 0:
            train_repr = anchor_train
            ood_repr = anchor_ood
        else:
            model, tok = build(args.base_model, str(item.get("adapter")))
            train_repr = encode_rows(model, tok, train_rows, layers, args.max_sequence_length)
            ood_repr = encode_rows(model, tok, ood_rows, layers, args.max_sequence_length)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for layer in layers:
            records.append(
                {
                    "step": step,
                    "layer": layer,
                    "relative_layer_position": layer / max(1, num_layers - 1),
                    "cross_format_nearest_centroid_reasoning_accuracy": centroid_accuracy(train_repr[layer], train_labels, ood_repr[layer], ood_labels),
                    "train_reasoning_cosine_separation_gap": separation_gap(train_repr[layer], train_labels),
                    "ood_reasoning_cosine_separation_gap": separation_gap(ood_repr[layer], ood_labels),
                    "train_linear_cka_to_anchor": linear_cka(train_repr[layer], anchor_train[layer]),
                    "ood_linear_cka_to_anchor": linear_cka(ood_repr[layer], anchor_ood[layer]),
                    "train_activation_effective_rank": effective_rank(train_repr[layer]),
                    "ood_activation_effective_rank": effective_rank(ood_repr[layer]),
                }
            )
    payload = {
        "schema_version": "2.3.1",
        "sealed_data_used": False,
        "steps": steps,
        "layers": layers,
        "num_hidden_layers": num_layers,
        "records": records,
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
