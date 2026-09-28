"""Separate backbone geometry; attention width is never inferred as hidden/heads."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from experiments.architecture_benchmark_001.budget import Config, budget, presets


@dataclass(frozen=True)
class CrossConfig:
    encoder: Config
    decoder: Config
    mode: str = "H"

    def __post_init__(self) -> None:
        if not isinstance(self.encoder, Config) or not isinstance(self.decoder, Config):
            raise ValueError("two explicit Config objects required")
        if any(c.architecture != "plain" for c in (self.encoder, self.decoder)):
            raise ValueError("both backbones must remain ordinary causal stacks")
        if self.encoder.vocab_size != self.decoder.vocab_size:
            raise ValueError("v1 requires identical token IDs; verify tokenizer assets separately")
        if self.mode not in ("H", "P"):
            raise ValueError("only hybrid H and contextual tokenwise P; no pure-CED switch")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> CrossConfig:
        if set(value) != {"encoder", "decoder", "mode"}:
            raise ValueError("unexpected configuration fields")
        return cls(Config(**value["encoder"]), Config(**value["decoder"]), value["mode"])


def official_config(mode: str = "H") -> CrossConfig:
    e = presets()["B28_1024_official_shape"]
    d = replace(e, hidden_size=2560, intermediate_size=9728,
                num_attention_heads=32, encoder_layers=36)
    return CrossConfig(e, d, mode)


def tiny_config(mode: str = "H", *, full_depth: bool = False) -> CrossConfig:
    # Deliberately different residual widths, Q/KV counts AND head dimensions.
    # Both residual widths differ from query projection width.
    e = Config(vocab_size=41, hidden_size=12, intermediate_size=24,
               num_attention_heads=4, num_key_value_heads=2, head_dim=4,
               encoder_layers=28 if full_depth else 2, decoder_layers=0,
               architecture="plain", max_position_embeddings=40960)
    d = replace(e, hidden_size=20, intermediate_size=48, num_attention_heads=6,
                num_key_value_heads=3, head_dim=6, encoder_layers=36 if full_depth else 4)
    return CrossConfig(e, d, mode)


def cross_budget(c: CrossConfig, bytes_per_scalar: int = 2) -> dict:
    if type(bytes_per_scalar) is not int or bytes_per_scalar <= 0:
        raise ValueError("positive integer scalar size required")
    e, d = c.encoder, c.decoder
    q, k = d.num_attention_heads*d.head_dim, d.num_key_value_heads*d.head_dim
    writer = 2*e.hidden_size*k + e.hidden_size + d.head_dim
    reader = 2*d.hidden_size*q + d.hidden_size + d.head_dim
    interface = writer + d.encoder_layers*reader
    eb, db = budget(e, bytes_per_scalar), budget(d, bytes_per_scalar)
    total = eb["parameters"] + db["parameters"] + interface
    kv = eb["kv_bytes_per_token_batch1"] + db["kv_bytes_per_token_batch1"] + 2*k*bytes_per_scalar
    return dict(parameters=total, encoder_parameters=eb["parameters"],
                decoder_parameters=db["parameters"], trainable_interface_parameters=interface,
                writer_parameters=writer, reader_parameters_each=reader,
                kv_banks=e.encoder_layers+d.encoder_layers+1,
                kv_bytes_per_token_batch1=kv,
                kv_mib_at_8192_tokens_batch1=kv*8192/2**20,
                weight_bytes=total*bytes_per_scalar,
                assumptions="analytic only; both full backbones, separate tied vocabulary tables; shared writer; all decoder blocks get full-width reads; no fit/speed claim")
