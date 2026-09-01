"""A Qwen3 forward pass that never holds the model in memory.

Qwen3-8B is 15.26 GiB in bfloat16 on a host with 15.2 GiB of RAM, and the
installed Torch build is CPU-only. Loading a donor conventionally is impossible,
so this module streams: it memory-maps the safetensors shards, materialises one
weight tensor at a time, upcasts that tensor to FP32 for the matmul, and frees it
before reading the next. Peak residency is one weight tensor plus the activation
stream.

It computes only what the atlas needs — residual states at selected depths and
the final-token logit vector — so nothing else is ever built.

The implementation is hand-written and therefore must be validated against
`transformers`' own Qwen3 before any donor measurement is trusted. See
`scripts/v3_e5_validate_streaming_forward.py`; amendment v3.1 makes that a gate.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open

from .checkpoint import CheckpointError, discover_weight_map


@dataclass(frozen=True)
class Qwen3Geometry:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Qwen3Geometry":
        rope = config.get("rope_parameters") or {}
        theta = rope.get("rope_theta", config.get("rope_theta", 1000000.0))
        head_dim = config.get(
            "head_dim", config["hidden_size"] // config["num_attention_heads"]
        )
        return cls(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(head_dim),
            vocab_size=int(config["vocab_size"]),
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(theta),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        )


class StreamingWeights:
    """Lazy tensor access across a sharded safetensors checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.weight_map = discover_weight_map(self.path)
        self._handles: dict[str, Any] = {}
        self.config = json.loads((self.path / "config.json").read_text("utf-8"))

    def __enter__(self) -> "StreamingWeights":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()

    def _handle(self, key: str) -> Any:
        filename = self.weight_map.get(key)
        if filename is None:
            raise CheckpointError(f"missing tensor: {key}")
        if filename not in self._handles:
            handle = safe_open(str(self.path / filename), framework="pt", device="cpu")
            self._handles[filename] = handle.__enter__()
        return self._handles[filename]

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def tensor(self, key: str) -> torch.Tensor:
        """Materialise one tensor as FP32. The caller must free it."""
        return self._handle(key).get_tensor(key).float()

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        """Gather selected rows without materialising the whole matrix.

        The embedding matrix is up to 151936 x 4096; reading only the prompt's
        rows keeps a 1.2 GiB tensor off the heap.
        """
        handle = self._handle(key)
        try:
            view = handle.get_slice(key)
            return torch.stack([view[int(i): int(i) + 1, :][0] for i in indices]).float()
        except Exception:
            whole = handle.get_tensor(key)
            gathered = whole[indices].float()
            del whole
            return gathered

    def row_blocks(self, key: str, block: int) -> Iterator[torch.Tensor]:
        """Yield row blocks of a large matrix, for streamed output projection."""
        handle = self._handle(key)
        view = handle.get_slice(key)
        total = view.get_shape()[0]
        for start in range(0, total, block):
            yield view[start: min(start + block, total), :].float()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def rotary_tables(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inverse_frequency = 1.0 / (theta ** exponent)
    angles = positions.float().unsqueeze(-1) * inverse_frequency.unsqueeze(0)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x is [tokens, heads, head_dim]; cos/sin are [tokens, head_dim]."""
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    head_block: int = 4,
) -> torch.Tensor:
    """Causal attention, computed a few heads at a time.

    A full [heads, tokens, tokens] score tensor is 216 MiB at 32 heads and 1300
    tokens; blocking over heads keeps that under 30 MiB.
    """
    tokens, heads, head_dim = query.shape
    scale = 1.0 / (head_dim ** 0.5)
    causal = torch.full((tokens, tokens), float("-inf")).triu(1)
    output = torch.empty_like(query)
    for start in range(0, heads, head_block):
        stop = min(start + head_block, heads)
        q = query[:, start:stop, :].transpose(0, 1)
        k = key[:, start:stop, :].transpose(0, 1)
        v = value[:, start:stop, :].transpose(0, 1)
        scores = torch.baddbmm(causal.unsqueeze(0), q, k.transpose(1, 2), alpha=scale)
        weights = torch.softmax(scores, dim=-1)
        output[:, start:stop, :] = torch.bmm(weights, v).transpose(0, 1)
        del q, k, v, scores, weights
    del causal
    return output


def streaming_forward_many(
    weights: StreamingWeights,
    prompts: list[torch.Tensor],
    *,
    capture_layers: tuple[int, ...],
    want_logits: bool = True,
    logit_block: int = 16384,
    pooling: str = "last",
    progress: Any = None,
    logit_positions: list[torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Run every prompt through the checkpoint with ONE pass over the weights.

    `streaming_forward` re-reads the whole model per prompt, which for 59 prompts
    against Qwen3-8B would be 900 GiB of disk traffic. Inverting the loops — layer
    outermost, prompts innermost — reads each weight exactly once.

    The cost is holding every prompt's residual stream at once, which is small:
    9,013 tokens x 4096 features x 4 bytes is 148 MiB for the 8B. Peak residency
    is therefore one layer's weights plus that activation set, still far below
    one model.
    """
    if pooling not in ("last", "mean", "all"):
        raise ValueError(f"unsupported pooling: {pooling}")
    geometry = Qwen3Geometry.from_config(weights.config)
    groups = geometry.num_attention_heads // geometry.num_key_value_heads

    def reduce(state: torch.Tensor) -> torch.Tensor:
        # "all" keeps every token position, which E6's translator needs: 59
        # prompts cannot fit a map into a 1024-dimensional target space.
        if pooling == "all":
            return state
        return state.mean(dim=0) if pooling == "mean" else state[-1]

    states = [weights.rows("model.embed_tokens.weight", ids) for ids in prompts]
    tables = [rotary_tables(torch.arange(s.shape[0]), geometry.head_dim, geometry.rope_theta)
              for s in states]
    captured: list[dict[str, torch.Tensor]] = [
        {"embedding": reduce(s).clone()} for s in states
    ]

    for layer in range(geometry.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        block = {
            "input_layernorm": weights.tensor(f"{prefix}.input_layernorm.weight"),
            "q_proj": weights.tensor(f"{prefix}.self_attn.q_proj.weight"),
            "k_proj": weights.tensor(f"{prefix}.self_attn.k_proj.weight"),
            "v_proj": weights.tensor(f"{prefix}.self_attn.v_proj.weight"),
            "q_norm": weights.tensor(f"{prefix}.self_attn.q_norm.weight"),
            "k_norm": weights.tensor(f"{prefix}.self_attn.k_norm.weight"),
            "o_proj": weights.tensor(f"{prefix}.self_attn.o_proj.weight"),
            "post_attention_layernorm": weights.tensor(f"{prefix}.post_attention_layernorm.weight"),
            "gate_proj": weights.tensor(f"{prefix}.mlp.gate_proj.weight"),
            "up_proj": weights.tensor(f"{prefix}.mlp.up_proj.weight"),
            "down_proj": weights.tensor(f"{prefix}.mlp.down_proj.weight"),
        }
        for index, hidden in enumerate(states):
            cos, sin = tables[index]
            normed = rms_norm(hidden, block["input_layernorm"], geometry.rms_norm_eps)
            query = (normed @ block["q_proj"].T).view(
                -1, geometry.num_attention_heads, geometry.head_dim)
            key = (normed @ block["k_proj"].T).view(
                -1, geometry.num_key_value_heads, geometry.head_dim)
            value = (normed @ block["v_proj"].T).view(
                -1, geometry.num_key_value_heads, geometry.head_dim)
            del normed
            query = apply_rotary(rms_norm(query, block["q_norm"], geometry.rms_norm_eps), cos, sin)
            key = apply_rotary(rms_norm(key, block["k_norm"], geometry.rms_norm_eps), cos, sin)
            if groups > 1:
                key = key.repeat_interleave(groups, dim=1)
                value = value.repeat_interleave(groups, dim=1)
            attended = _attention(query, key, value)
            del query, key, value
            hidden = hidden + (attended.reshape(hidden.shape[0], -1) @ block["o_proj"].T)
            del attended

            normed = rms_norm(hidden, block["post_attention_layernorm"], geometry.rms_norm_eps)
            gate = torch.nn.functional.silu(normed @ block["gate_proj"].T)
            gate = gate * (normed @ block["up_proj"].T)
            del normed
            hidden = hidden + (gate @ block["down_proj"].T)
            del gate
            states[index] = hidden
            if layer in capture_layers:
                captured[index][f"layer_{layer}"] = reduce(hidden).clone()
        del block
        gc.collect()
        if progress is not None:
            progress(layer + 1, geometry.num_hidden_layers)

    norm_weight = weights.tensor("model.norm.weight")
    last = geometry.num_hidden_layers - 1
    finals = []
    counts = []
    for index, hidden in enumerate(states):
        normalised = rms_norm(hidden, norm_weight, geometry.rms_norm_eps)
        if last in capture_layers:
            captured[index][f"layer_{last}"] = reduce(normalised).clone()
        # `logit_positions` lets a caller score specific token positions -- E7
        # needs the log-probability of a reference answer, not just the next
        # token -- without ever materialising logits for the whole sequence.
        if logit_positions is not None:
            selected = normalised[logit_positions[index]].clone()
        else:
            selected = normalised[-1:].clone()
        counts.append(selected.shape[0])
        finals.append(selected)
        states[index] = None
    del norm_weight, states
    gc.collect()

    logits: list[torch.Tensor | None] = [None] * len(prompts)
    if want_logits:
        head_key = ("lm_head.weight" if weights.has("lm_head.weight")
                    else "model.embed_tokens.weight")
        stacked = torch.cat(finals, dim=0)
        pieces = []
        for chunk in weights.row_blocks(head_key, logit_block):
            pieces.append(stacked @ chunk.T)
            del chunk
        combined = torch.cat(pieces, dim=1)[:, : geometry.vocab_size]
        offset = 0
        logits = []
        for count in counts:
            block = combined[offset: offset + count]
            logits.append(block.squeeze(0).clone() if logit_positions is None else block.clone())
            offset += count
        del pieces, combined, stacked
    del finals
    gc.collect()
    return [{"captured": captured[i], "logits": logits[i], "geometry": geometry}
            for i in range(len(prompts))]


def streaming_forward(
    weights: StreamingWeights,
    input_ids: torch.Tensor,
    *,
    capture_layers: tuple[int, ...],
    want_logits: bool = True,
    logit_block: int = 16384,
    pooling: str = "last",
) -> dict[str, Any]:
    """Run one prompt through a Qwen3 checkpoint without loading the model.

    Returns the residual state at the embedding output and after each requested
    layer, reduced by `pooling`, plus the final-token logits.
    """
    if pooling not in ("last", "mean"):
        raise ValueError(f"unsupported pooling: {pooling}")
    geometry = Qwen3Geometry.from_config(weights.config)
    groups = geometry.num_attention_heads // geometry.num_key_value_heads

    hidden = weights.rows("model.embed_tokens.weight", input_ids)
    positions = torch.arange(hidden.shape[0])
    cos, sin = rotary_tables(positions, geometry.head_dim, geometry.rope_theta)

    def reduce(state: torch.Tensor) -> torch.Tensor:
        return state.mean(dim=0) if pooling == "mean" else state[-1]

    captured = {"embedding": reduce(hidden).clone()}

    for layer in range(geometry.num_hidden_layers):
        prefix = f"model.layers.{layer}"

        weight = weights.tensor(f"{prefix}.input_layernorm.weight")
        normed = rms_norm(hidden, weight, geometry.rms_norm_eps)
        del weight

        projection = weights.tensor(f"{prefix}.self_attn.q_proj.weight")
        query = (normed @ projection.T).view(-1, geometry.num_attention_heads, geometry.head_dim)
        del projection
        projection = weights.tensor(f"{prefix}.self_attn.k_proj.weight")
        key = (normed @ projection.T).view(-1, geometry.num_key_value_heads, geometry.head_dim)
        del projection
        projection = weights.tensor(f"{prefix}.self_attn.v_proj.weight")
        value = (normed @ projection.T).view(-1, geometry.num_key_value_heads, geometry.head_dim)
        del projection, normed

        # Qwen3 normalises each head's query and key vectors before RoPE.
        weight = weights.tensor(f"{prefix}.self_attn.q_norm.weight")
        query = rms_norm(query, weight, geometry.rms_norm_eps)
        del weight
        weight = weights.tensor(f"{prefix}.self_attn.k_norm.weight")
        key = rms_norm(key, weight, geometry.rms_norm_eps)
        del weight

        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)

        attended = _attention(query, key, value)
        del query, key, value

        projection = weights.tensor(f"{prefix}.self_attn.o_proj.weight")
        hidden = hidden + (attended.reshape(hidden.shape[0], -1) @ projection.T)
        del projection, attended

        weight = weights.tensor(f"{prefix}.post_attention_layernorm.weight")
        normed = rms_norm(hidden, weight, geometry.rms_norm_eps)
        del weight

        projection = weights.tensor(f"{prefix}.mlp.gate_proj.weight")
        gate = torch.nn.functional.silu(normed @ projection.T)
        del projection
        projection = weights.tensor(f"{prefix}.mlp.up_proj.weight")
        gate = gate * (normed @ projection.T)
        del projection, normed
        projection = weights.tensor(f"{prefix}.mlp.down_proj.weight")
        hidden = hidden + (gate @ projection.T)
        del projection, gate

        if layer in capture_layers:
            captured[f"layer_{layer}"] = reduce(hidden).clone()
        if layer % 8 == 7:
            gc.collect()

    weight = weights.tensor("model.norm.weight")
    normalised = rms_norm(hidden, weight, geometry.rms_norm_eps)
    del weight, hidden

    # `transformers` appends the post-norm state as its final `hidden_states`
    # entry, so `hidden_states[num_layers]` is norm(output of the last layer)
    # while every earlier entry is a raw layer output. Match that convention
    # exactly, or the last depth slot would not be comparable across the
    # streaming donors and the transformers-captured target.
    last = geometry.num_hidden_layers - 1
    if last in capture_layers:
        captured[f"layer_{last}"] = reduce(normalised).clone()

    logits = None
    if want_logits:
        final = normalised[-1]
        head_key = ("lm_head.weight" if weights.has("lm_head.weight")
                    else "model.embed_tokens.weight")
        pieces = []
        for block in weights.row_blocks(head_key, logit_block):
            pieces.append(block @ final)
            del block
        logits = torch.cat(pieces)[: geometry.vocab_size]
        del pieces, final
    del normalised
    gc.collect()
    return {"captured": captured, "logits": logits, "geometry": geometry}
