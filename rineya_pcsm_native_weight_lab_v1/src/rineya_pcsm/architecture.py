"""Architecture-driven discovery of same-role peer parameters.

PCSM treats layers as peers only when their parameter occupies the same semantic
slot in a repeated architecture block.  The discovery code deliberately uses
the module tree rather than tensor order and always removes the target itself.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torch import nn

_REPEATED_CONTAINER_NAMES = frozenset(
    {"blocks", "block", "h", "layer", "layers", "stages"}
)


@dataclass(frozen=True, slots=True)
class ParameterSite:
    """A parameter and its architecture-derived semantic role."""

    name: str
    role: str
    module_path: str
    module_type: str
    parameter_name: str
    layer_path: str
    layer_index: int | None
    shape: tuple[int, ...]


def _architecture_slot(module_path: str) -> tuple[str, str, int | None]:
    """Return ``(role_path, layer_path, layer_index)`` for a module path.

    The last numeric component is used because nested expert/module lists are
    common.  The component immediately before it is retained as the repeated
    container.  This makes ``model.layers.7.mlp.down_proj`` and
    ``model.layers.8.mlp.down_proj`` share the role ``mlp.down_proj``.
    """

    parts = module_path.split(".") if module_path else []
    numeric_positions = [index for index, part in enumerate(parts) if part.isdigit()]
    if not numeric_positions:
        return module_path, module_path, None

    position = numeric_positions[-1]
    index = int(parts[position])
    layer_path = ".".join(parts[: position + 1])
    suffix = parts[position + 1 :]
    if suffix:
        return ".".join(suffix), layer_path, index

    # A parameter directly on a repeated block still needs a stable role.
    container = parts[position - 1] if position else "block"
    if container in _REPEATED_CONTAINER_NAMES:
        container = "block"
    return container, layer_path, index


def discover_parameter_sites(
    model: nn.Module,
    *,
    include_bias: bool = False,
    minimum_rank: int = 2,
) -> tuple[ParameterSite, ...]:
    """Discover eligible parameter sites from ``model.named_modules()``.

    Only direct parameters of each module are considered, preventing duplicate
    discovery.  Matrix weights are the default because the PCSM property maps
    are defined over row/column structure.
    """

    sites: list[ParameterSite] = []
    for module_path, module in model.named_modules():
        role_path, layer_path, layer_index = _architecture_slot(module_path)
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name == "bias" and not include_bias:
                continue
            if parameter.ndim < minimum_rank:
                continue
            full_name = ".".join(part for part in (module_path, parameter_name) if part)
            role = role_path or module.__class__.__name__.lower()
            if parameter_name != "weight":
                role = f"{role}.{parameter_name}"
            sites.append(
                ParameterSite(
                    name=full_name,
                    role=role,
                    module_path=module_path,
                    module_type=module.__class__.__qualname__,
                    parameter_name=parameter_name,
                    layer_path=layer_path,
                    layer_index=layer_index,
                    shape=tuple(parameter.shape),
                )
            )
    return tuple(sorted(sites, key=lambda site: site.name))


class ArchitectureIndex:
    """Immutable index for architecture roles and target-safe peer lookup."""

    def __init__(self, sites: tuple[ParameterSite, ...]) -> None:
        by_name: dict[str, ParameterSite] = {}
        by_role_mutable: dict[str, list[ParameterSite]] = defaultdict(list)
        for site in sites:
            if site.name in by_name:
                raise ValueError(f"duplicate parameter site: {site.name}")
            by_name[site.name] = site
            by_role_mutable[site.role].append(site)
        by_role = {
            role: tuple(sorted(role_sites, key=lambda item: item.name))
            for role, role_sites in by_role_mutable.items()
        }
        self._sites = tuple(sorted(sites, key=lambda item: item.name))
        self._by_name: Mapping[str, ParameterSite] = MappingProxyType(by_name)
        self._by_role: Mapping[str, tuple[ParameterSite, ...]] = MappingProxyType(
            by_role
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        include_bias: bool = False,
        minimum_rank: int = 2,
    ) -> ArchitectureIndex:
        return cls(
            discover_parameter_sites(
                model, include_bias=include_bias, minimum_rank=minimum_rank
            )
        )

    @property
    def sites(self) -> tuple[ParameterSite, ...]:
        return self._sites

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_role))

    def site(self, name: str) -> ParameterSite:
        try:
            return self._by_name[name]
        except KeyError as error:
            raise KeyError(f"unknown parameter site: {name}") from error

    def sites_for_role(self, role: str) -> tuple[ParameterSite, ...]:
        return self._by_role.get(role, ())

    def peers_for(
        self, target_name: str, *, allow_empty: bool = False
    ) -> tuple[ParameterSite, ...]:
        """Return same-role peers, failing closed when none are available.

        ``allow_empty`` is an explicit escape hatch for experimental controls.
        Production peer-conditioned selection must never silently broaden the
        semantic role or continue without peer evidence.
        """

        target = self.site(target_name)
        peers = tuple(
            site
            for site in self.sites_for_role(target.role)
            if site.name != target.name
        )
        if any(peer.name == target.name for peer in peers):  # defensive invariant
            raise AssertionError("self advice escaped peer filtering")
        if not peers and not allow_empty:
            raise ValueError(
                f"no compatible same-role peers for target parameter: {target_name}"
            )
        return peers


__all__ = ["ArchitectureIndex", "ParameterSite", "discover_parameter_sites"]
