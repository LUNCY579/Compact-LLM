from __future__ import annotations

import pytest
from torch import nn

from rineya_pcsm.architecture import ArchitectureIndex, discover_parameter_sites


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = ToyMLP()
        self.self_attn = ToyAttention()


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(), ToyBlock(), ToyBlock()])
        self.output = nn.Linear(4, 2)


class LoneRoleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock()])


def test_discovery_derives_roles_from_repeated_architecture_slots() -> None:
    sites = discover_parameter_sites(ToyModel())
    by_name = {site.name: site for site in sites}
    assert by_name["layers.0.mlp.down_proj.weight"].role == "mlp.down_proj"
    assert by_name["layers.2.self_attn.q_proj.weight"].role == "self_attn.q_proj"
    assert by_name["layers.1.mlp.down_proj.weight"].layer_index == 1
    assert "output.bias" not in by_name


def test_peer_lookup_is_same_role_and_never_returns_self() -> None:
    index = ArchitectureIndex.from_model(ToyModel())
    target = "layers.1.mlp.down_proj.weight"
    peers = index.peers_for(target)
    assert [peer.name for peer in peers] == [
        "layers.0.mlp.down_proj.weight",
        "layers.2.mlp.down_proj.weight",
    ]
    assert all(peer.role == "mlp.down_proj" for peer in peers)
    assert all(peer.name != target for peer in peers)


def test_unknown_target_fails_closed() -> None:
    index = ArchitectureIndex.from_model(ToyModel())
    with pytest.raises(KeyError, match="unknown parameter site"):
        index.peers_for("missing.weight")


def test_known_target_with_zero_compatible_peers_fails_closed() -> None:
    index = ArchitectureIndex.from_model(LoneRoleModel())
    target = "layers.0.mlp.down_proj.weight"
    with pytest.raises(ValueError, match="no compatible same-role peers"):
        index.peers_for(target)
    assert index.peers_for(target, allow_empty=True) == ()
