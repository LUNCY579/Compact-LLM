from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Any, Literal

import psutil
import torch

from consensus_lab.models.committee import ExpertCommittee
from consensus_lab.voting.rules import AggregationDecision, aggregate_logits


EOSPolicy = Literal["ordinary", "strict_majority", "confidence_threshold", "minimum_tokens"]


@dataclass(frozen=True)
class GenerationResult:
    output_text: str
    output_token_ids: list[int]
    steps: list[dict[str, Any]]
    stop_reason: str
    latency_seconds: float
    time_to_first_final_token_seconds: float | None
    forward_passes: int
    adapter_evaluations: int
    peak_gpu_vram_bytes: int | None
    peak_cpu_rss_bytes: int
    cache_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output_text,
            "output_token_ids": self.output_token_ids,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
            "latency_seconds": self.latency_seconds,
            "time_to_first_final_token_seconds": self.time_to_first_final_token_seconds,
            "forward_passes": self.forward_passes,
            "adapter_evaluations": self.adapter_evaluations,
            "generated_tokens": len(self.output_token_ids),
            "peak_gpu_vram_bytes": self.peak_gpu_vram_bytes,
            "peak_cpu_rss_bytes": self.peak_cpu_rss_bytes,
            "cache_mode": self.cache_mode,
        }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _apply_eos_policy(
    decision: AggregationDecision,
    logits: dict[str, torch.Tensor],
    *,
    committee: ExpertCommittee,
    method: str,
    eos_policy: EOSPolicy,
    eos_confidence_threshold: float,
    minimum_tokens: int,
    generated_count: int,
) -> tuple[AggregationDecision, dict[str, Any]]:
    eos = committee.tokenizer.eos_token_id
    policy = {"name": eos_policy, "overrode": False, "original_winner": decision.token_id}
    if eos is None or decision.token_id != eos or eos_policy == "ordinary":
        return decision, policy
    allow = True
    if eos_policy == "strict_majority":
        allow = bool(decision.diagnostics["eos_vote_fraction"] > 0.5)
    elif eos_policy == "confidence_threshold":
        eos_probs = []
        for expert in committee.experts:
            temperature = expert.temperature
            eos_probs.append(float(torch.softmax(logits[expert.name] / temperature, dim=-1)[eos].item()))
        policy["mean_eos_probability"] = sum(eos_probs) / len(eos_probs)
        policy["threshold"] = eos_confidence_threshold
        allow = bool(policy["mean_eos_probability"] >= eos_confidence_threshold)
    elif eos_policy == "minimum_tokens":
        allow = generated_count >= minimum_tokens
        policy["minimum_tokens"] = minimum_tokens
    if allow:
        return decision, policy
    masked = {name: values.clone() for name, values in logits.items()}
    for values in masked.values():
        values[eos] = -torch.inf
    replacement = aggregate_logits(
        masked,
        method=method,
        eos_token_id=eos,
        validation_weights={expert.name: expert.validation_weight for expert in committee.experts},
        temperatures={expert.name: expert.temperature for expert in committee.experts},
    )
    policy["overrode"] = True
    policy["replacement_winner"] = replacement.token_id
    return replacement, policy


def _trace_step(
    *,
    t: int,
    decision: AggregationDecision,
    policy: dict[str, Any],
    committee: ExpertCommittee,
) -> dict[str, Any]:
    token_id = decision.token_id
    return {
        "t": t,
        "winner": token_id,
        "winner_text": committee.tokenizer.decode([token_id], skip_special_tokens=False),
        **decision.diagnostics,
        "eos_policy": policy,
    }


def generate_reference(
    committee: ExpertCommittee,
    input_ids: torch.Tensor,
    *,
    method: str = "hard_plurality",
    max_new_tokens: int = 32,
    eos_policy: EOSPolicy = "ordinary",
    eos_confidence_threshold: float = 0.5,
    minimum_tokens: int = 1,
) -> GenerationResult:
    """Mathematical reference: recompute the full shared prefix for every expert."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Reference decoder currently requires a single unpadded sequence")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if committee.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(committee.device)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    prefix = input_ids
    generated: list[int] = []
    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    first_token_time: float | None = None
    stop_reason = "max_new_tokens"
    eos = committee.tokenizer.eos_token_id
    for t in range(max_new_tokens):
        logits: dict[str, torch.Tensor] = {}
        attention_mask = torch.ones_like(prefix)
        for expert in committee.experts:
            values, _ = committee.forward(expert, prefix, attention_mask=attention_mask, use_cache=False)
            logits[expert.name] = values
        decision = aggregate_logits(
            logits,
            method=method,
            eos_token_id=eos,
            validation_weights={expert.name: expert.validation_weight for expert in committee.experts},
            temperatures={expert.name: expert.temperature for expert in committee.experts},
        )
        decision, policy = _apply_eos_policy(
            decision,
            logits,
            committee=committee,
            method=method,
            eos_policy=eos_policy,
            eos_confidence_threshold=eos_confidence_threshold,
            minimum_tokens=minimum_tokens,
            generated_count=len(generated),
        )
        generated.append(decision.token_id)
        traces.append(_trace_step(t=t, decision=decision, policy=policy, committee=committee))
        prefix = torch.cat(
            [prefix, torch.tensor([[decision.token_id]], device=prefix.device, dtype=prefix.dtype)], dim=1
        )
        _synchronize(committee.device)
        if first_token_time is None:
            first_token_time = time.perf_counter() - started
        peak_rss = max(peak_rss, process.memory_info().rss)
        if eos is not None and decision.token_id == eos:
            stop_reason = "eos"
            break
    _synchronize(committee.device)
    latency = time.perf_counter() - started
    return GenerationResult(
        output_text=committee.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        output_token_ids=generated,
        steps=traces,
        stop_reason=stop_reason,
        latency_seconds=latency,
        time_to_first_final_token_seconds=first_token_time,
        forward_passes=len(traces) * len(committee.experts),
        adapter_evaluations=len(traces) * len(committee.experts),
        peak_gpu_vram_bytes=torch.cuda.max_memory_allocated(committee.device) if committee.device.type == "cuda" else None,
        peak_cpu_rss_bytes=peak_rss,
        cache_mode="no_cache_full_prefix_recompute",
    )


def generate_separate_cache(
    committee: ExpertCommittee,
    input_ids: torch.Tensor,
    *,
    method: str = "hard_plurality",
    max_new_tokens: int = 32,
    eos_policy: EOSPolicy = "ordinary",
    eos_confidence_threshold: float = 0.5,
    minimum_tokens: int = 1,
) -> GenerationResult:
    """Exact optimized decoder with one independent KV cache per expert."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Cached decoder currently requires a single unpadded sequence")
    if committee.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(committee.device)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    logits: dict[str, torch.Tensor] = {}
    caches: dict[str, Any] = {}
    attention_mask = torch.ones_like(input_ids)
    for expert in committee.experts:
        values, cache = committee.forward(expert, input_ids, attention_mask=attention_mask, use_cache=True)
        logits[expert.name] = values
        caches[expert.name] = cache
    generated: list[int] = []
    traces: list[dict[str, Any]] = []
    first_token_time: float | None = None
    stop_reason = "max_new_tokens"
    eos = committee.tokenizer.eos_token_id
    for t in range(max_new_tokens):
        decision = aggregate_logits(
            logits,
            method=method,
            eos_token_id=eos,
            validation_weights={expert.name: expert.validation_weight for expert in committee.experts},
            temperatures={expert.name: expert.temperature for expert in committee.experts},
        )
        decision, policy = _apply_eos_policy(
            decision,
            logits,
            committee=committee,
            method=method,
            eos_policy=eos_policy,
            eos_confidence_threshold=eos_confidence_threshold,
            minimum_tokens=minimum_tokens,
            generated_count=len(generated),
        )
        generated.append(decision.token_id)
        traces.append(_trace_step(t=t, decision=decision, policy=policy, committee=committee))
        _synchronize(committee.device)
        if first_token_time is None:
            first_token_time = time.perf_counter() - started
        peak_rss = max(peak_rss, process.memory_info().rss)
        if eos is not None and decision.token_id == eos:
            stop_reason = "eos"
            break
        token = torch.tensor([[decision.token_id]], device=input_ids.device, dtype=input_ids.dtype)
        attention_mask = torch.cat([attention_mask, torch.ones_like(token)], dim=1)
        next_logits: dict[str, torch.Tensor] = {}
        for expert in committee.experts:
            values, cache = committee.forward(
                expert,
                token,
                attention_mask=attention_mask,
                past_key_values=caches[expert.name],
                use_cache=True,
            )
            next_logits[expert.name] = values
            caches[expert.name] = cache
        logits = next_logits
    _synchronize(committee.device)
    latency = time.perf_counter() - started
    return GenerationResult(
        output_text=committee.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        output_token_ids=generated,
        steps=traces,
        stop_reason=stop_reason,
        latency_seconds=latency,
        time_to_first_final_token_seconds=first_token_time,
        forward_passes=len(traces) * len(committee.experts),
        adapter_evaluations=len(traces) * len(committee.experts),
        peak_gpu_vram_bytes=torch.cuda.max_memory_allocated(committee.device) if committee.device.type == "cuda" else None,
        peak_cpu_rss_bytes=peak_rss,
        cache_mode="separate_cache_per_expert",
    )


def _levenshtein(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def generate_sequence_medoid(
    committee: ExpertCommittee,
    input_ids: torch.Tensor,
    *,
    method: str = "sequence_medoid",
    max_new_tokens: int = 32,
    eos_policy: EOSPolicy = "ordinary",
    eos_confidence_threshold: float = 0.5,
    minimum_tokens: int = 1,
) -> GenerationResult:
    """Generate one independent candidate per expert, then choose the token-edit medoid.

    This is deliberately a whole-candidate comparator: experts do not share an
    accepted prefix. Ties use normalized self log-confidence, then the stable
    token tuple and expert name. It must not be interpreted as token voting.
    """
    if method != "sequence_medoid":
        raise ValueError(f"Sequence decoder only supports sequence_medoid, got {method!r}")
    started = time.perf_counter()
    candidates: list[tuple[str, GenerationResult, float]] = []
    for expert in committee.experts:
        result = generate_reference(
            committee.subset([expert.name]),
            input_ids,
            method="hard_plurality",
            max_new_tokens=max_new_tokens,
            eos_policy=eos_policy,
            eos_confidence_threshold=eos_confidence_threshold,
            minimum_tokens=minimum_tokens,
        )
        log_confidence = sum(
            math.log(max(float(step["top_probabilities"][expert.name]), 1e-45))
            for step in result.steps
        ) / max(len(result.steps), 1)
        candidates.append((expert.name, result, log_confidence))
    distance_sums = [
        sum(
            _levenshtein(candidate[1].output_token_ids, other[1].output_token_ids)
            for other in candidates
        )
        for candidate in candidates
    ]
    winner_index = min(
        range(len(candidates)),
        key=lambda index: (
            distance_sums[index],
            -candidates[index][2],
            tuple(candidates[index][1].output_token_ids),
            candidates[index][0],
        ),
    )
    winner_name, winner, winner_confidence = candidates[winner_index]
    identical_count = sum(
        candidate.output_token_ids == winner.output_token_ids for _, candidate, _ in candidates
    )
    elapsed = time.perf_counter() - started
    selection_trace = {
        "t": None,
        "comparison_level": "whole_sequence",
        "method": method,
        "candidate_outputs": {name: result.output_text for name, result, _ in candidates},
        "candidate_token_ids": {name: result.output_token_ids for name, result, _ in candidates},
        "distance_sums": {name: distance for (name, _, _), distance in zip(candidates, distance_sums, strict=True)},
        "normalized_self_log_confidence": {
            name: score for name, _, score in candidates
        },
        "winning_expert": winner_name,
        "winner": winner.output_token_ids,
        "agreement": identical_count / len(candidates),
        "vote_entropy": None,
        "mean_pairwise_js": None,
        "tie": distance_sums.count(distance_sums[winner_index]) > 1,
        "eos_vote_fraction": None,
        "winner_is_eos": winner.stop_reason == "eos",
        "selection_tie_break_rule": "minimum_edit_distance_then_self_log_confidence_then_token_tuple",
        "winner_normalized_self_log_confidence": winner_confidence,
    }
    return GenerationResult(
        output_text=winner.output_text,
        output_token_ids=winner.output_token_ids,
        steps=[selection_trace],
        stop_reason=winner.stop_reason,
        latency_seconds=elapsed,
        time_to_first_final_token_seconds=elapsed,
        forward_passes=sum(result.forward_passes for _, result, _ in candidates),
        adapter_evaluations=sum(result.adapter_evaluations for _, result, _ in candidates),
        peak_gpu_vram_bytes=max(
            (result.peak_gpu_vram_bytes for _, result, _ in candidates if result.peak_gpu_vram_bytes is not None),
            default=None,
        ),
        peak_cpu_rss_bytes=max(result.peak_cpu_rss_bytes for _, result, _ in candidates),
        cache_mode="independent_no_cache_candidates",
    )


def generate_adaptive_consultation(
    committee: ExpertCommittee,
    input_ids: torch.Tensor,
    *,
    core_expert_names: list[str],
    agreement_threshold: float,
    method: str = "adaptive_hard_plurality",
    max_new_tokens: int = 32,
    eos_policy: EOSPolicy = "ordinary",
    eos_confidence_threshold: float = 0.5,
    minimum_tokens: int = 1,
) -> GenerationResult:
    """Consult extra experts only when the core hard vote has low agreement."""
    if method != "adaptive_hard_plurality":
        raise ValueError(f"Adaptive decoder only supports adaptive_hard_plurality, got {method!r}")
    if not 0 <= agreement_threshold <= 1:
        raise ValueError("agreement_threshold must be in [0, 1]")
    if not core_expert_names:
        raise ValueError("At least one core expert is required")
    core = committee.subset(core_expert_names)
    extra_names = [expert.name for expert in committee.experts if expert.name not in set(core_expert_names)]
    extras = committee.subset(extra_names) if extra_names else None
    prefix = input_ids
    generated: list[int] = []
    traces: list[dict[str, Any]] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    if committee.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(committee.device)
    started = time.perf_counter()
    first_token_time: float | None = None
    stop_reason = "max_new_tokens"
    forward_passes = 0
    eos = committee.tokenizer.eos_token_id
    for t in range(max_new_tokens):
        attention_mask = torch.ones_like(prefix)
        logits: dict[str, torch.Tensor] = {}
        for expert in core.experts:
            logits[expert.name], _ = core.forward(
                expert, prefix, attention_mask=attention_mask, use_cache=False
            )
        forward_passes += len(core.experts)
        core_decision = aggregate_logits(
            logits,
            method="hard_plurality",
            eos_token_id=eos,
            validation_weights={expert.name: expert.validation_weight for expert in core.experts},
            temperatures={expert.name: expert.temperature for expert in core.experts},
        )
        triggered = bool(
            extras is not None and float(core_decision.diagnostics["agreement"]) < agreement_threshold
        )
        active_view = core
        decision = core_decision
        if triggered and extras is not None:
            for expert in extras.experts:
                logits[expert.name], _ = extras.forward(
                    expert, prefix, attention_mask=attention_mask, use_cache=False
                )
            forward_passes += len(extras.experts)
            active_view = committee
            decision = aggregate_logits(
                logits,
                method="hard_plurality",
                eos_token_id=eos,
                validation_weights={expert.name: expert.validation_weight for expert in committee.experts},
                temperatures={expert.name: expert.temperature for expert in committee.experts},
            )
        decision, policy = _apply_eos_policy(
            decision,
            logits,
            committee=active_view,
            method="hard_plurality",
            eos_policy=eos_policy,
            eos_confidence_threshold=eos_confidence_threshold,
            minimum_tokens=minimum_tokens,
            generated_count=len(generated),
        )
        trace = _trace_step(t=t, decision=decision, policy=policy, committee=committee)
        trace["adaptive_consultation"] = {
            "triggered": triggered,
            "threshold": agreement_threshold,
            "core_experts": core_expert_names,
            "additional_experts": extra_names if triggered else [],
            "core_winner": core_decision.token_id,
            "core_agreement": core_decision.diagnostics["agreement"],
            "final_winner_changed": core_decision.token_id != decision.token_id,
        }
        traces.append(trace)
        generated.append(decision.token_id)
        prefix = torch.cat(
            [prefix, torch.tensor([[decision.token_id]], device=prefix.device, dtype=prefix.dtype)], dim=1
        )
        _synchronize(committee.device)
        if first_token_time is None:
            first_token_time = time.perf_counter() - started
        peak_rss = max(peak_rss, process.memory_info().rss)
        if eos is not None and decision.token_id == eos:
            stop_reason = "eos"
            break
    _synchronize(committee.device)
    return GenerationResult(
        output_text=committee.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        output_token_ids=generated,
        steps=traces,
        stop_reason=stop_reason,
        latency_seconds=time.perf_counter() - started,
        time_to_first_final_token_seconds=first_token_time,
        forward_passes=forward_passes,
        adapter_evaluations=forward_passes,
        peak_gpu_vram_bytes=torch.cuda.max_memory_allocated(committee.device)
        if committee.device.type == "cuda"
        else None,
        peak_cpu_rss_bytes=peak_rss,
        cache_mode="adaptive_no_cache_full_prefix_recompute",
    )
