from __future__ import annotations

import gc
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..errors import EvaluationError
from ..evaluation.dataset import load_dataset
from ..run_context import RunContext
from ..utils import sha256_file, write_json, write_jsonl


def _render_ids(tokenizer: Any, messages: list[dict[str, Any]], target: str) -> tuple[list[int], int]:
    prompt_kwargs = {"tokenize": True, "add_generation_prompt": True}
    full_kwargs = {"tokenize": True, "add_generation_prompt": False}
    try:
        prompt_ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **prompt_kwargs)
        full_ids = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": target}],
            enable_thinking=False,
            **full_kwargs,
        )
    except TypeError:
        prompt_ids = tokenizer.apply_chat_template(messages, **prompt_kwargs)
        full_ids = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": target}], **full_kwargs
        )
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    if isinstance(full_ids, torch.Tensor):
        full_ids = full_ids.tolist()
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if int(a) != int(b):
            break
        common += 1
    # Most chat templates make the generation prompt an exact prefix. The fallback
    # to longest-common-prefix is explicit in evidence if a template behaves differently.
    target_start = len(prompt_ids) if full_ids[: len(prompt_ids)] == prompt_ids else common
    if target_start >= len(full_ids):
        raise EvaluationError("Could not isolate assistant target tokens for probability profiling.")
    return [int(value) for value in full_ids], target_start


def _profile_model(
    *,
    model_path: Path,
    tokenizer_path: Path,
    records: Sequence[Mapping[str, Any]],
    targets: Mapping[str, str],
    trust_remote_code: bool,
    top_k: int,
    require_cuda: bool,
) -> dict[str, list[dict[str, Any]]]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise EvaluationError("Install transformers to run probability profiling.") from exc
    if require_cuda and not torch.cuda.is_available():
        raise EvaluationError("Probability profile requires CUDA under the selected protocol.")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    eos_value = tokenizer.eos_token_id
    eos_ids = [eos_value] if isinstance(eos_value, int) else list(eos_value or [])
    output: dict[str, list[dict[str, Any]]] = {}
    try:
        for record in records:
            stable_id = str(record["id"])
            target = str(targets[stable_id])
            messages = [dict(value) for value in record["messages"]]
            ids, target_start = _render_ids(tokenizer, messages, target)
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(input_ids=tensor, use_cache=False).logits[0]
            steps: list[dict[str, Any]] = []
            # Token at position p is predicted by logits at p-1.
            for position in range(target_start, len(ids)):
                token_id = int(ids[position])
                distribution = torch.softmax(logits[position - 1].float(), dim=-1)
                k = min(max(1, int(top_k)), distribution.numel())
                values, indices = torch.topk(distribution, k=k)
                retained_mass = float(values.sum().item())
                steps.append(
                    {
                        "position": position - target_start,
                        "target_token_id": token_id,
                        "target_probability": float(distribution[token_id].item()),
                        "eos_probability": (
                            float(distribution[eos_ids].sum().item()) if eos_ids else None
                        ),
                        "entropy": float(
                            -(distribution * torch.log(distribution.clamp_min(1e-30))).sum().item()
                        ),
                        "top_ids": [int(value) for value in indices.tolist()],
                        "top_probabilities": [float(value) for value in values.tolist()],
                        "residual_mass": max(0.0, 1.0 - retained_mass),
                    }
                )
            output[stable_id] = steps
            del logits, tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def _coarse_js_lower_bound(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """JS divergence on a shared coarse partition.

    We retain only token IDs that appear in *both* endpoint top-k sets and collapse
    every other vocabulary item into one residual bucket. This is a valid common
    partition, so the result is a conservative lower bound on full-vocabulary JS.
    Using the union without cross-endpoint probabilities would double-count mass.
    """

    first_map = dict(zip(first["top_ids"], first["top_probabilities"]))
    second_map = dict(zip(second["top_ids"], second["top_probabilities"]))
    ids = sorted(set(first_map) & set(second_map))
    p_known = [float(first_map[token]) for token in ids]
    q_known = [float(second_map[token]) for token in ids]
    p = p_known + [max(0.0, 1.0 - sum(p_known))]
    q = q_known + [max(0.0, 1.0 - sum(q_known))]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_probability_profile(
    context: RunContext,
    *,
    anchor_path: Path,
    greedy_path: Path,
    tokenizer_path: Path,
    anchor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    settings = context.config.get("experiments.probability_profile")
    if not settings.get("enabled", False):
        return None
    maximum = max(1, int(settings.get("max_cases", 150)))
    records = load_dataset(Path(context.config.get("datasets.dev")))[:maximum]
    anchor_by_id = {str(row["id"]): row for row in anchor_rows}
    include_reference = bool(settings.get("include_reference_trajectory", True))
    targets: dict[str, str] = {}
    for record in records:
        stable_id = str(record["id"])
        reference = str(record.get("reference") or "").strip()
        if include_reference and reference:
            targets[stable_id] = reference
        else:
            targets[stable_id] = str(anchor_by_id[stable_id].get("reply", ""))
        if not targets[stable_id]:
            raise EvaluationError(f"No common target trajectory is available for {stable_id}")
    tokenizer_path = tokenizer_path.expanduser().resolve()
    protocol = context.config.get(
        f"evaluation.protocols.{context.config.get('evaluation.selection_protocol')}"
    )
    top_k = int(settings.get("top_k", 128))
    common = {
        "tokenizer_path": tokenizer_path,
        "records": records,
        "targets": targets,
        "trust_remote_code": bool(context.config.get("models.trust_remote_code", False)),
        "top_k": top_k,
        "require_cuda": bool(protocol.get("require_cuda", False)),
    }
    anchor = _profile_model(model_path=anchor_path, **common)
    greedy = _profile_model(model_path=greedy_path, **common)
    rows: list[dict[str, Any]] = []
    for record in records:
        stable_id = str(record["id"])
        first_steps = anchor[stable_id]
        second_steps = greedy[stable_id]
        for first, second in zip(first_steps, second_steps):
            rows.append(
                {
                    "id": stable_id,
                    "category": record.get("category"),
                    "position": first["position"],
                    "target_token_id": first["target_token_id"],
                    "coarse_js_lower_bound": _coarse_js_lower_bound(first, second),
                    "top_k_overlap_count": len(
                        set(first["top_ids"]) & set(second["top_ids"])
                    ),
                    "top_k_jaccard": len(set(first["top_ids"]) & set(second["top_ids"]))
                    / max(1, len(set(first["top_ids"]) | set(second["top_ids"]))),
                    "anchor_target_probability": first["target_probability"],
                    "greedy_target_probability": second["target_probability"],
                    "anchor_eos_probability": first["eos_probability"],
                    "greedy_eos_probability": second["eos_probability"],
                    "anchor_entropy": first["entropy"],
                    "greedy_entropy": second["entropy"],
                }
            )
    anchor_eos_values = [
        float(row["anchor_eos_probability"])
        for row in rows
        if row["anchor_eos_probability"] is not None
    ]
    greedy_eos_values = [
        float(row["greedy_eos_probability"])
        for row in rows
        if row["greedy_eos_probability"] is not None
    ]
    summary = {
        "schema_version": "1.0",
        "case_count": len(records),
        "token_position_count": len(rows),
        "top_k": top_k,
        "development_dataset_sha256": sha256_file(
            Path(context.config.get("datasets.dev"))
        ),
        "divergence_is_lower_bound": True,
        "approximation": (
            "shared intersection of endpoint top-k token IDs plus one residual-mass "
            "bucket; conservative under coarse-graining"
        ),
        "mean_coarse_js_lower_bound": statistics.mean(
            float(row["coarse_js_lower_bound"]) for row in rows
        ),
        "mean_top_k_jaccard": statistics.mean(float(row["top_k_jaccard"]) for row in rows),
        "mean_anchor_target_probability": statistics.mean(
            float(row["anchor_target_probability"]) for row in rows
        ),
        "mean_greedy_target_probability": statistics.mean(
            float(row["greedy_target_probability"]) for row in rows
        ),
        "mean_anchor_eos_probability": (
            statistics.mean(anchor_eos_values) if anchor_eos_values else None
        ),
        "mean_greedy_eos_probability": (
            statistics.mean(greedy_eos_values) if greedy_eos_values else None
        ),
    }
    write_jsonl(context.path("analysis", "probability", "token_profile.jsonl"), rows)
    write_json(context.path("analysis", "probability", "summary.json"), summary)
    return summary
