"""Analytic budgets for the deliberately simplified dense reference in this bundle.

This does not estimate DeepSeek-V4.1's MoE/MLA/SWA/CSA implementation.
Counts are unique parameters (tied vocabulary embedding counted once).
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math

@dataclass(frozen=True)
class Config:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 2688
    num_attention_heads: int = 14
    num_key_value_heads: int = 7
    head_dim: int = 128
    encoder_layers: int = 20
    decoder_layers: int = 20
    architecture: str = "ced"  # 'plain': encoder_layers counts all ordinary blocks
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 40960

    def __post_init__(self) -> None:
        for key in ("vocab_size", "hidden_size", "intermediate_size",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                    "encoder_layers", "max_position_embeddings"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if type(self.decoder_layers) is not int or self.decoder_layers < 0:
            raise ValueError("decoder_layers must be a nonnegative integer")
        if self.architecture not in {"plain", "ced"}:
            raise ValueError("architecture must be plain or ced")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for this RoPE implementation")
        if self.architecture == "plain" and self.decoder_layers != 0:
            raise ValueError("plain uses encoder_layers for all blocks; decoder_layers=0")
        if self.architecture == "ced" and self.decoder_layers <= 0:
            raise ValueError("ced requires a positive decoder depth")
        if any(not math.isfinite(v) or v <= 0 for v in (self.rope_theta, self.rms_norm_eps)):
            raise ValueError("RoPE base and RMS epsilon must be positive")

    @property
    def total_layers(self) -> int:
        return self.encoder_layers + self.decoder_layers


def budget(c: Config, bytes_per_scalar: int = 2) -> dict:
    if type(bytes_per_scalar) is not int or bytes_per_scalar <= 0:
        raise ValueError("bytes_per_scalar must be positive")
    d, m, h = c.hidden_size, c.intermediate_size, c.head_dim
    q, k = c.num_attention_heads * h, c.num_key_value_heads * h
    embed = c.vocab_size * d
    ff = 3 * d * m
    # Two residual RMS norms and one Q and one K per-head RMS norm.
    regular = 2 * d * (q + k) + ff + 2 * d + 2 * h
    # Decoder has its own Q/O and FFN; the shared bank supplies K/V.
    read_block = 2 * d * q + ff + 2 * d + h
    bank = (2 * d * k + d + h) if c.architecture == "ced" else 0
    parameters = embed + c.encoder_layers * regular + c.decoder_layers * read_block + bank + d
    cache_banks = c.encoder_layers + int(c.architecture == "ced")
    cache_bytes = 2 * cache_banks * k * bytes_per_scalar
    return {
        "parameters": parameters,
        "embedding_parameters": embed,
        "body_and_final_norm_parameters": parameters - embed,
        "shared_bank_parameters": bank,
        "kv_banks": cache_banks,
        "kv_bytes_per_token_batch1": cache_bytes,
        "kv_mib_at_8192_tokens_batch1": cache_bytes * 8192 / 2**20,
        "assumptions": "dense; tied embeddings; no biases; no decoder local SWA; no stored hidden-state cache; KV only, excluding temporary activations/allocator overhead",
    }


def presets() -> dict[str, Config]:
    def plain(l: int, d: int, m: int, nq: int, nk: int) -> Config:
        return Config(hidden_size=d, intermediate_size=m, num_attention_heads=nq,
                      num_key_value_heads=nk, encoder_layers=l, decoder_layers=0,
                      architecture="plain")
    return {
        "B28_1024_official_shape": plain(28, 1024, 3072, 16, 8),
        "D40_1024_expansion_diagnostic": plain(40, 1024, 3072, 16, 8),
        "D40_896_proportional": plain(40, 896, 2688, 14, 7),
        "W28_896_width_only_diagnostic": plain(28, 896, 2496, 14, 7),
        "D40_896_near_baseline_budget": plain(40, 896, 2496, 14, 7),
        "D40_768_compact": plain(40, 768, 2304, 12, 6),
        "C28_1024_topology_diagnostic": Config(hidden_size=1024, intermediate_size=3072,
            num_attention_heads=16, num_key_value_heads=8, encoder_layers=14, decoder_layers=14),
        "C28_1024_near_baseline_budget": Config(hidden_size=1024, intermediate_size=3392,
            num_attention_heads=16, num_key_value_heads=8, encoder_layers=14, decoder_layers=14),
        "C40_896_proportional": Config(),
        "C40_896_near_baseline_budget": Config(intermediate_size=2752),
        "C40_768_compact": Config(hidden_size=768, intermediate_size=2304,
            num_attention_heads=12, num_key_value_heads=6),
    }

if __name__ == "__main__":
    output = {name: {"config": asdict(c), "budget": budget(c)} for name, c in presets().items()}
    print(json.dumps(output, indent=2))
