"""Configuration for the pretrained Rineya/Qwen TGSA retrofit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

CONFIG_SCHEMA_VERSION = "rineya.tgsa-retrofit-config.v1"


@dataclass(frozen=True, slots=True)
class RineyaTGSAConfig:
    """Architecture of one retrofitted attention module.

    Every field except the ``trajectory_*`` and ``condition_*`` group is copied
    verbatim from the source model's configuration. A retrofit that silently
    changes a native field would break weight reuse, so the conversion path
    reads these from the source config rather than accepting defaults.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0
    attention_bias: bool = False
    output_bias: bool = False
    max_position_embeddings: int = 32768

    # Trajectory pathway. `trajectory_rank` is the bottleneck of the
    # residual-around-identity encoder.
    trajectory_rank: int = 32
    condition_q: bool = True
    condition_k: bool = False
    condition_v: bool = False

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) for GQA"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.trajectory_rank <= 0:
            raise ValueError("trajectory_rank must be positive")
        if not any((self.condition_q, self.condition_k, self.condition_v)):
            raise ValueError(
                "at least one conditioning target must be enabled, otherwise the "
                "trajectory pathway can never influence the output"
            )

    @property
    def group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def conditioning_targets(self) -> tuple[str, ...]:
        return tuple(
            target for target in ("q", "k", "v") if getattr(self, f"condition_{target}")
        )

    @classmethod
    def _read_rope(cls, source: Mapping[str, Any]) -> tuple[float, float]:
        """Extract RoPE theta and scaling, failing closed on an unknown layout.

        Transformers has moved this field: older Qwen configs use a top-level
        ``rope_theta``, newer ones (including Rineya-Supa-A1, transformers 5.x)
        nest it under ``rope_parameters``. Silently defaulting to 10000 when the
        real model uses 1000000 would corrupt every positional frequency while
        leaving an exactness test *green*, because a reference built from the
        same wrong config would be wrong in the same way. So a config that
        declares RoPE in neither place is an error, not a default.
        """

        parameters = source.get("rope_parameters")
        if isinstance(parameters, Mapping) and "rope_theta" in parameters:
            theta = float(parameters["rope_theta"])
            scaling_source: Any = parameters
        elif "rope_theta" in source:
            theta = float(source["rope_theta"])
            scaling_source = source.get("rope_scaling")
        else:
            raise KeyError(
                "source config declares no rope_theta, at the top level or under "
                "rope_parameters; refusing to guess a positional convention"
            )

        scaling = 1.0
        if isinstance(scaling_source, Mapping):
            rope_type = str(scaling_source.get("rope_type", "default"))
            if rope_type not in {"default", "linear"}:
                raise ValueError(
                    f"unsupported rope_type {rope_type!r}; the retrofit reproduces only the "
                    f"default and linear conventions, and a mismatched one would silently "
                    f"change every position"
                )
            scaling = float(scaling_source.get("factor", 1.0))
        return theta, scaling

    @classmethod
    def from_source_config(
        cls,
        source: Mapping[str, Any],
        **trajectory_overrides: Any,
    ) -> "RineyaTGSAConfig":
        """Derive the retrofit config from a Hugging Face-style model config.

        ``head_dim`` is read explicitly when present. Qwen3 sets it independently
        of ``hidden_size`` — Rineya-Supa-A1 has hidden 1024 with 16 heads of 128,
        so ``hidden_size // num_attention_heads`` would give 64 and every
        projection shape would be wrong. The fallback exists only for configs
        that genuinely omit the field.
        """

        heads = int(source["num_attention_heads"])
        hidden = int(source["hidden_size"])
        head_dim = int(source.get("head_dim") or hidden // heads)
        theta, scaling = cls._read_rope(source)
        return cls(
            hidden_size=hidden,
            num_attention_heads=heads,
            num_key_value_heads=int(source.get("num_key_value_heads", heads)),
            head_dim=head_dim,
            rope_theta=theta,
            rope_scaling=scaling,
            attention_bias=bool(source.get("attention_bias", False)),
            output_bias=bool(source.get("output_bias", False)),
            max_position_embeddings=int(source.get("max_position_embeddings", 32768)),
            **trajectory_overrides,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": CONFIG_SCHEMA_VERSION, **asdict(self)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "RineyaTGSAConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported retrofit config schema: {value.get('schema_version')}")
        value.pop("schema_version")
        return cls(**value)
