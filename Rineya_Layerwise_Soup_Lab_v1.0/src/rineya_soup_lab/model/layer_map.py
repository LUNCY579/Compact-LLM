from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..errors import ConfigurationError
from ..utils import stable_id


@dataclass(frozen=True)
class ParameterLocation:
    key: str
    group: str
    layer_index: int | None = None


@dataclass(frozen=True)
class MergeRecipe:
    """A deterministic anchor→greedy interpolation recipe.

    Alpha always means the greedy checkpoint's coefficient:

        merged = (1 - alpha) * anchor + alpha * greedy
    """

    global_alpha: float
    layer_alphas: Mapping[int, float] = field(default_factory=dict)
    group_alphas: Mapping[str, float] = field(default_factory=dict)
    label: str = "merge"
    kind: str = "merge"
    endpoint: str | None = None
    schema_version: str = "1.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"merge", "endpoint"}:
            raise ConfigurationError(f"Unsupported recipe kind: {self.kind}")
        if self.kind == "endpoint" and self.endpoint not in {"anchor", "greedy"}:
            raise ConfigurationError("Endpoint recipes require endpoint='anchor' or 'greedy'.")
        _validate_alpha(self.global_alpha, "global_alpha")
        if self.kind == "endpoint":
            expected_alpha = 0.0 if self.endpoint == "anchor" else 1.0
            if not math.isclose(float(self.global_alpha), expected_alpha, abs_tol=1e-12):
                raise ConfigurationError(
                    f"Endpoint {self.endpoint!r} requires global_alpha={expected_alpha}."
                )
            if self.layer_alphas or self.group_alphas:
                raise ConfigurationError("Endpoint recipes cannot contain layer/group overrides.")
        for layer, alpha in self.layer_alphas.items():
            if int(layer) < 0:
                raise ConfigurationError(f"Negative layer index in recipe: {layer}")
            _validate_alpha(alpha, f"layer_alphas[{layer}]")
        for group, alpha in self.group_alphas.items():
            _validate_alpha(alpha, f"group_alphas[{group!r}]")

    def identity_dict(self) -> dict[str, Any]:
        """Numerical identity, excluding human labels and provenance notes."""
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "endpoint": self.endpoint,
            "global_alpha": float(self.global_alpha),
            "layer_alphas": {str(int(key)): float(value) for key, value in sorted(self.layer_alphas.items())},
            "group_alphas": {str(key): float(value) for key, value in sorted(self.group_alphas.items())},
        }

    @property
    def id(self) -> str:
        # Labels and explanatory metadata are aliases, not numerical model identity.
        return stable_id(self.identity_dict(), prefix="cand_")

    def alpha_for(self, location: ParameterLocation) -> float:
        if location.layer_index is not None and location.layer_index in self.layer_alphas:
            return float(self.layer_alphas[location.layer_index])
        if location.group in self.group_alphas:
            return float(self.group_alphas[location.group])
        return float(self.global_alpha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "endpoint": self.endpoint,
            "global_alpha": float(self.global_alpha),
            "layer_alphas": {str(int(key)): float(value) for key, value in sorted(self.layer_alphas.items())},
            "group_alphas": {str(key): float(value) for key, value in sorted(self.group_alphas.items())},
            "label": self.label,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MergeRecipe":
        return cls(
            schema_version=str(value.get("schema_version", "1.0")),
            kind=str(value.get("kind", "merge")),
            endpoint=value.get("endpoint"),
            global_alpha=float(value.get("global_alpha", 0.0)),
            layer_alphas={int(key): float(alpha) for key, alpha in (value.get("layer_alphas") or {}).items()},
            group_alphas={str(key): float(alpha) for key, alpha in (value.get("group_alphas") or {}).items()},
            label=str(value.get("label", "merge")),
            metadata=dict(value.get("metadata") or {}),
        )

    @classmethod
    def endpoint_recipe(cls, endpoint: str) -> "MergeRecipe":
        alpha = 0.0 if endpoint == "anchor" else 1.0
        return cls(global_alpha=alpha, label=endpoint, kind="endpoint", endpoint=endpoint)


def _validate_alpha(value: float, label: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ConfigurationError(f"{label} must lie in [0, 1], got {value}")


class ParameterClassifier:
    def __init__(self, layer_key_regex: str) -> None:
        self.pattern = re.compile(layer_key_regex)

    def classify(self, key: str) -> ParameterLocation:
        match = self.pattern.search(key)
        if match:
            return ParameterLocation(key=key, group="transformer_layer", layer_index=int(match.group(1)))
        lowered = key.lower()
        if "embed_tokens" in lowered or ".wte." in lowered or lowered.endswith("wte.weight"):
            return ParameterLocation(key=key, group="embedding")
        if lowered.startswith("lm_head.") or ".lm_head." in lowered:
            return ParameterLocation(key=key, group="lm_head")
        if lowered.endswith("model.norm.weight") or lowered.endswith("transformer.ln_f.weight"):
            return ParameterLocation(key=key, group="final_norm")
        if "rotary_emb" in lowered or "inv_freq" in lowered:
            return ParameterLocation(key=key, group="rotary_or_buffer")
        return ParameterLocation(key=key, group="other")

    def layer_indices(self, keys: list[str]) -> list[int]:
        return sorted({location.layer_index for key in keys if (location := self.classify(key)).layer_index is not None})
