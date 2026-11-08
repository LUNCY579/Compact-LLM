"""Tests for ``rineya_psa.merging``: blockwise arithmetic (section 34) and soups (35).

The arithmetic is checked two ways against each other. The in-memory path is checked
against torch doing the same blend in one shot, and the on-disk path is round-tripped
through the real ``safetensors`` reader and compared to the in-memory result -- so the
hand-rolled safetensors writer and the numpy bfloat16 encoder are validated against
independent implementations rather than against themselves.

The soup searches are driven by synthetic ``evaluate`` callables with a known optimum, so
"did the search find it" is a fact rather than a hope. No checkpoint is read; nothing
touches the GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from rineya_psa.merging import arithmetic as A
from rineya_psa.merging import soup as S
from rineya_psa.paths import EXPERIMENTS, MODELS

POSITIONS = (0.05, 0.50, 0.95)


# --------------------------------------------------------------------------------------
# Fixtures: a tiny Qwen3-shaped module and a pair of tiny safetensors checkpoints
# --------------------------------------------------------------------------------------


class _Norm(nn.Module):
    """RMSNorm-shaped leaf: one ``weight`` parameter, named ``norm`` in the state dict."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class _Attn(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class _Layer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = _Attn(dim)
        self.input_layernorm = _Norm(dim)


class _Stack(nn.Module):
    def __init__(self, n_layers: int, dim: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, dim)
        self.layers = nn.ModuleList(_Layer(dim) for _ in range(n_layers))
        self.norm = _Norm(dim)


class TinyModel(nn.Module):
    """State-dict keys identical in shape to Qwen3's, so ``parse_tensor_name`` applies."""

    def __init__(self, n_layers: int = 4, dim: int = 8, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.model = _Stack(n_layers, dim)


@pytest.fixture
def tiny_model() -> TinyModel:
    return TinyModel(seed=0).eval()


@pytest.fixture
def blocks() -> tuple[A.LayerBlock, ...]:
    return A.default_blocks(4, n_layer_blocks=2)


def _random_like(state: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        name: tensor + torch.randn(tensor.shape, generator=gen, dtype=torch.float32)
        for name, tensor in state.items()
    }


def _write_checkpoint(
    directory: Path, tensors: dict[str, torch.Tensor], *, dtype: torch.dtype
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {name: value.to(dtype).contiguous() for name, value in tensors.items()}
    payload["model.counter"] = torch.arange(4, dtype=torch.int64)
    save_file(payload, directory / "model.safetensors")
    (directory / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}\n')
    return directory


@pytest.fixture
def checkpoints(tmp_path: Path, tiny_model: TinyModel) -> tuple[Path, Path]:
    base_state = {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}
    other_state = _random_like(base_state, seed=99)
    base_dir = _write_checkpoint(tmp_path / "base", base_state, dtype=torch.bfloat16)
    other_dir = _write_checkpoint(tmp_path / "other", other_state, dtype=torch.bfloat16)
    return base_dir, other_dir


# --------------------------------------------------------------------------------------
# Layer blocks and coefficients
# --------------------------------------------------------------------------------------


def test_default_blocks_partition_the_qwen3_stack():
    early, middle, late, glob = A.default_blocks(28)
    assert (early.name, middle.name, late.name) == ("early", "middle", "late")
    assert early.layers == tuple(range(0, 9))
    assert middle.layers == tuple(range(9, 19))
    assert late.layers == tuple(range(19, 28))
    assert glob.is_global and glob.layers == ()
    covered = sorted(early.layers + middle.layers + late.layers)
    assert covered == list(range(28))


def test_layer_block_rejects_a_global_block_that_owns_layers():
    with pytest.raises(ValueError, match="owns no decoder layers"):
        A.LayerBlock(name="global", layers=(0,), is_global=True)
    with pytest.raises(ValueError, match="at least one layer"):
        A.LayerBlock(name="early", layers=())


def test_block_alphas_requires_exactly_one_alpha_per_block(blocks):
    names = [b.name for b in blocks]
    with pytest.raises(ValueError, match="no alpha given"):
        A.BlockAlphas(blocks=blocks, alphas={names[0]: 1.0})
    with pytest.raises(ValueError, match="unknown blocks"):
        A.BlockAlphas(blocks=blocks, alphas=dict.fromkeys([*names, "nope"], 0.0))
    with pytest.raises(ValueError, match="not finite"):
        A.BlockAlphas(blocks=blocks, alphas=dict.fromkeys(names, float("nan")))


def test_block_alphas_rejects_a_layer_claimed_twice():
    with pytest.raises(ValueError, match="claimed by both"):
        A.BlockAlphas(
            blocks=(A.LayerBlock("a", (0, 1)), A.LayerBlock("b", (1, 2))),
            alphas={"a": 1.0, "b": 1.0},
        )


def test_block_for_tensor_routes_stack_level_tensors_to_the_global_block():
    blocks = A.default_blocks(28)
    alphas = A.BlockAlphas(
        blocks=blocks, alphas={"early": 0.0, "middle": 0.5, "late": 1.0, "global": 1.2}
    )
    assert alphas.for_tensor("model.layers.3.self_attn.q_proj.weight") == 0.0
    assert alphas.for_tensor("model.layers.12.mlp.up_proj.weight") == 0.5
    assert alphas.for_tensor("model.layers.27.self_attn.o_proj.weight") == 1.0
    assert alphas.for_tensor("model.embed_tokens.weight") == 1.2
    assert alphas.for_tensor("model.norm.weight") == 1.2
    assert alphas.for_tensor("lm_head.weight") == 1.2


def test_block_alphas_slug_identity_and_derivation(blocks):
    zero = A.BlockAlphas.uniform(blocks, 0.0)
    assert zero.is_identity()
    assert "0p00" in zero.slug()
    moved = zero.with_alpha("block1", 1.2)
    assert not moved.is_identity()
    assert moved.alphas["block1"] == 1.2 and moved.alphas["block0"] == 0.0
    assert zero.alphas["block1"] == 0.0  # the original is untouched
    with pytest.raises(KeyError, match="unknown block"):
        zero.with_alpha("nope", 1.0)


def test_alpha_grid_is_the_briefs_grid():
    assert A.ALPHA_GRID == (0.0, 0.5, 1.0, 1.2)


def test_sweep_shapes(blocks):
    assert len(A.uniform_alpha_sweep(blocks)) == len(A.ALPHA_GRID)
    # background + one move per (block, non-background alpha)
    assert len(A.single_block_sweep(blocks)) == 1 + len(blocks) * (len(A.ALPHA_GRID) - 1)
    assert len(A.full_block_grid(blocks)) == len(A.ALPHA_GRID) ** len(blocks)
    with pytest.raises(ValueError, match="exceeds max_combinations"):
        A.full_block_grid(blocks, max_combinations=8)


def test_block_alphas_grid_selects_the_design():
    assert len(A.block_alphas_grid(28, design="uniform")) == 4
    assert len(A.block_alphas_grid(28, design="single_block")) == 1 + 4 * 3
    with pytest.raises(ValueError, match="unknown design"):
        A.block_alphas_grid(28, design="wishful")


# --------------------------------------------------------------------------------------
# In-memory blockwise delta (the preferred path)
# --------------------------------------------------------------------------------------


def _states(model: TinyModel, seed: int = 7):
    base = A.snapshot_state(model)
    other = {k: v + torch.randn(v.shape, generator=torch.Generator().manual_seed(seed)) for
             k, v in base.items()}
    return base, A.deltas_from_states(base, other), other


def test_snapshot_state_is_a_detached_float32_cpu_copy(tiny_model):
    snapshot = A.snapshot_state(tiny_model)
    live = dict(tiny_model.state_dict(keep_vars=True))
    for name, tensor in snapshot.items():
        assert tensor.dtype is torch.float32
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
        assert tensor.data_ptr() != live[name].data_ptr()


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 1.2])
def test_apply_blockwise_delta_matches_the_closed_form(tiny_model, blocks, alpha):
    base, deltas, _ = _states(tiny_model)
    alphas = A.BlockAlphas.uniform(blocks, alpha)

    applied = A.apply_blockwise_delta(tiny_model, base, deltas, alphas)
    after = tiny_model.state_dict()
    for name, base_tensor in base.items():
        expected = base_tensor + alpha * deltas.get(name, torch.zeros_like(base_tensor))
        assert torch.allclose(after[name].float(), expected, atol=1e-6), name
    assert applied.alphas is alphas
    assert set(applied.touched) == set(base)


def test_alpha_one_reproduces_the_other_checkpoint(tiny_model, blocks):
    base, deltas, other = _states(tiny_model)
    A.apply_blockwise_delta(tiny_model, base, deltas, A.BlockAlphas.uniform(blocks, 1.0))
    after = tiny_model.state_dict()
    for name in base:
        assert torch.allclose(after[name].float(), other[name], atol=1e-6), name


def test_restore_is_exact_and_idempotent(tiny_model, blocks):
    base, deltas, _ = _states(tiny_model)
    before = {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}

    alphas = A.BlockAlphas.uniform(blocks, 1.2)
    applied = A.apply_blockwise_delta(tiny_model, base, deltas, alphas)
    norm = tiny_model.state_dict()["model.norm.weight"]
    assert not torch.allclose(norm, before["model.norm.weight"])
    applied.restore()
    applied.restore()  # idempotent
    for name, tensor in tiny_model.state_dict().items():
        assert torch.equal(tensor, before[name]), name
    assert applied.restored


def test_applied_merge_restores_on_context_exit(tiny_model, blocks):
    base, deltas, _ = _states(tiny_model)
    before = {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}
    with A.apply_blockwise_delta(tiny_model, base, deltas, A.BlockAlphas.uniform(blocks, 1.0)):
        norm = tiny_model.state_dict()["model.norm.weight"]
        assert not torch.equal(norm, before["model.norm.weight"])
    for name, tensor in tiny_model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_only_the_named_block_moves(tiny_model, blocks):
    base, deltas, _ = _states(tiny_model)
    # blocks = block0 (layers 0-1), block1 (layers 2-3), global
    alphas = A.BlockAlphas(blocks=blocks, alphas={"block0": 0.0, "block1": 1.0, "global": 0.0})
    A.apply_blockwise_delta(tiny_model, base, deltas, alphas)
    after = tiny_model.state_dict()
    assert torch.allclose(after["model.layers.1.self_attn.q_proj.weight"].float(),
                          base["model.layers.1.self_attn.q_proj.weight"], atol=1e-6)
    assert not torch.allclose(after["model.layers.2.self_attn.q_proj.weight"].float(),
                              base["model.layers.2.self_attn.q_proj.weight"], atol=1e-6)
    assert torch.allclose(after["model.embed_tokens.weight"].float(),
                          base["model.embed_tokens.weight"], atol=1e-6)


def test_unknown_delta_names_are_skipped_not_fatal(tiny_model, blocks):
    base, deltas, _ = _states(tiny_model)
    deltas = dict(deltas)
    deltas["lm_head.weight"] = torch.ones(4, 4)  # a tied head the model does not expose
    applied = A.apply_blockwise_delta(tiny_model, base, deltas, A.BlockAlphas.uniform(blocks, 1.0))
    assert "lm_head.weight" in applied.skipped
    assert "lm_head.weight" not in applied.touched


def test_chunking_does_not_change_the_blend(tiny_model, blocks):
    base, deltas, _ = _states(tiny_model)
    alphas = A.BlockAlphas.uniform(blocks, 0.5)
    A.apply_blockwise_delta(tiny_model, base, deltas, alphas, chunk_bytes=1 << 20)
    big = {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}
    A.apply_blockwise_delta(tiny_model, base, deltas, alphas, chunk_bytes=8)
    for name, tensor in tiny_model.state_dict().items():
        assert torch.equal(tensor, big[name]), name


def test_deltas_from_states_drops_unshared_and_unchanged_tensors(tiny_model):
    base = A.snapshot_state(tiny_model)
    other = {k: v.clone() for k, v in base.items()}
    other["model.norm.weight"] = other["model.norm.weight"] + 1.0
    other["lm_head.weight"] = torch.ones(4, 4)  # present on one side only
    deltas = A.deltas_from_states(base, other)
    assert set(deltas) == {"model.norm.weight"}
    assert torch.allclose(deltas["model.norm.weight"], torch.ones_like(deltas["model.norm.weight"]))


def test_deltas_from_states_rejects_shape_mismatch(tiny_model):
    base = A.snapshot_state(tiny_model)
    other = {k: v.clone() for k, v in base.items()}
    other["model.norm.weight"] = torch.zeros(3)
    with pytest.raises(ValueError, match="not arithmetic-compatible"):
        A.deltas_from_states(base, other)


# --------------------------------------------------------------------------------------
# Float encoding
# --------------------------------------------------------------------------------------


def test_bf16_encoding_is_bit_identical_to_torch():
    rng = np.random.default_rng(3)
    values = np.concatenate(
        [
            rng.normal(0, 3, 4096).astype("<f4"),
            np.array([0.0, -0.0, 1.0, -1.0, 1e-45, 3.4e38, -3.4e38], dtype="<f4"),
            np.array([np.inf, -np.inf], dtype="<f4"),
        ]
    )
    ours = np.frombuffer(A.encode_from_float(values, "BF16"), dtype="<u2")
    theirs = (
        torch.from_numpy(values.copy()).to(torch.bfloat16).view(torch.int16).numpy().astype("<u2")
    )
    assert np.array_equal(ours, theirs)


def test_bf16_encoding_keeps_nan_a_nan():
    raw = A.encode_from_float(np.array([np.nan], dtype="<f4"), "BF16")
    wide = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
    assert np.isnan(wide.view("<f4"))[0]


@pytest.mark.parametrize("tag,dtype", [("F32", torch.float32), ("F16", torch.float16)])
def test_other_float_encodings_match_torch(tag, dtype):
    values = np.random.default_rng(4).normal(0, 2, 256).astype("<f4")
    ours = A.encode_from_float(values, tag)
    theirs = torch.from_numpy(values.copy()).to(dtype).numpy().tobytes()
    assert ours == theirs


def test_encode_rejects_a_non_float_dtype():
    with pytest.raises(ValueError, match="cannot encode"):
        A.encode_from_float(np.zeros(4, dtype="<f4"), "I64")


# --------------------------------------------------------------------------------------
# On-disk materialisation
# --------------------------------------------------------------------------------------


def test_materialise_merge_matches_the_in_memory_formula(tmp_path, checkpoints, blocks):
    base_dir, other_dir = checkpoints
    alphas = A.BlockAlphas(blocks=blocks, alphas={"block0": 1.0, "block1": 0.5, "global": 0.0})
    artifact = A.materialise_merge(base_dir, other_dir, alphas, out_dir=tmp_path / "merged")

    base = load_file(base_dir / "model.safetensors")
    other = load_file(other_dir / "model.safetensors")
    merged = load_file(artifact.weights_file)
    assert set(merged) == set(base)
    for name, tensor in base.items():
        if not tensor.is_floating_point():
            assert torch.equal(merged[name], tensor), f"{name} is integral and must pass through"
            continue
        alpha = alphas.for_tensor(name)
        blended = tensor.float() + alpha * (other[name].float() - tensor.float())
        expected = blended.to(tensor.dtype)
        assert torch.equal(merged[name], expected), name
        assert merged[name].dtype is tensor.dtype  # stored in the base dtype, never upcast


def test_materialise_merge_writes_a_recipe_with_verifiable_hashes(tmp_path, checkpoints, blocks):
    base_dir, other_dir = checkpoints
    alphas = A.BlockAlphas.uniform(blocks, 1.0)
    artifact = A.materialise_merge(base_dir, other_dir, alphas, out_dir=tmp_path / "m")

    recipe = json.loads(artifact.recipe_file.read_text(encoding="utf-8"))
    assert recipe["formula"].startswith("theta' = theta_base")
    assert recipe["alphas"]["alphas"] == dict(alphas.alphas)
    assert recipe["output"]["sha256"] == artifact.weights_sha256
    assert set(recipe["output"]["tensor_sha256"]) == set(artifact.tensor_sha256)

    import hashlib

    assert (
        hashlib.sha256(artifact.weights_file.read_bytes()).hexdigest() == artifact.weights_sha256
    )
    sidecar = artifact.recipe_file.with_name(artifact.recipe_file.name + ".provenance.json")
    assert sidecar.exists()
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance["config"]["merge_id"] == artifact.merge_id


def test_materialise_merge_copies_the_loadable_aux_files(tmp_path, checkpoints, blocks):
    base_dir, other_dir = checkpoints
    artifact = A.materialise_merge(
        base_dir, other_dir, A.BlockAlphas.uniform(blocks, 0.5), out_dir=tmp_path / "m"
    )
    assert "config.json" in artifact.aux_files
    assert (artifact.directory / "config.json").exists()


def test_materialise_merge_is_chunk_size_independent(tmp_path, checkpoints, blocks):
    base_dir, other_dir = checkpoints
    alphas = A.BlockAlphas.uniform(blocks, 1.2)
    big = A.materialise_merge(base_dir, other_dir, alphas, out_dir=tmp_path / "big")
    small = A.materialise_merge(
        base_dir, other_dir, alphas, out_dir=tmp_path / "small", chunk_bytes=16
    )
    assert big.weights_sha256 == small.weights_sha256


def test_materialise_merge_refuses_to_overwrite_without_permission(tmp_path, checkpoints, blocks):
    base_dir, other_dir = checkpoints
    alphas = A.BlockAlphas.uniform(blocks, 1.0)
    A.materialise_merge(base_dir, other_dir, alphas, out_dir=tmp_path / "m")
    with pytest.raises(FileExistsError):
        A.materialise_merge(base_dir, other_dir, alphas, out_dir=tmp_path / "m")
    A.materialise_merge(base_dir, other_dir, alphas, out_dir=tmp_path / "m", overwrite=True)


def test_materialise_merge_rejects_shape_incompatible_checkpoints(tmp_path, checkpoints, blocks):
    base_dir, other_dir = checkpoints
    tensors = load_file(other_dir / "model.safetensors")
    tensors["model.norm.weight"] = torch.ones(3, dtype=torch.bfloat16)
    save_file(tensors, other_dir / "model.safetensors")
    with pytest.raises(ValueError, match="not arithmetic-compatible"):
        A.materialise_merge(
            base_dir, other_dir, A.BlockAlphas.uniform(blocks, 1.0), out_dir=tmp_path / "m"
        )


def test_merge_workspace_stays_under_experiments(tmp_path):
    target = A.merge_workspace("qwen3+halo-late1p0")
    assert target.parent == EXPERIMENTS / "merges"
    for bad in ("", "a/b", "..", "..\\escape"):
        with pytest.raises(ValueError, match="single safe path component"):
            A.merge_workspace(bad)


def test_materialise_merge_refuses_to_write_inside_a_checkpoint(checkpoints, blocks):
    base_dir, other_dir = checkpoints
    inside = MODELS["qwen3"].path / "merged_variant"
    with pytest.raises(PermissionError, match="refusing to write inside source checkpoint"):
        A.materialise_merge(
            base_dir, other_dir, A.BlockAlphas.uniform(blocks, 1.0), out_dir=inside
        )
    assert not inside.exists()


def test_merging_stays_importable_without_torch():
    """SPEC section 2: only the whitelisted packages may import torch at module scope."""
    code = (
        "import sys; import rineya_psa.merging.arithmetic, rineya_psa.merging.soup; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------------------
# Soup: fingerprints
# --------------------------------------------------------------------------------------


def _fp(label: str, values, *, metric="accuracy", higher=True) -> S.PositionalFingerprint:
    return S.PositionalFingerprint(
        label=label, positions=POSITIONS, values=tuple(values), metric=metric,
        higher_is_better=higher,
    )


def test_fingerprint_summaries():
    fp = _fp("m", [0.6, 0.2, 0.5])
    assert fp.worst == pytest.approx(0.2)
    assert fp.worst_at == pytest.approx(0.50)
    assert fp.overall == pytest.approx((0.6 + 0.2 + 0.5) / 3)
    assert fp.edge_middle_gap() == pytest.approx((0.6 + 0.5) / 2 - 0.2)
    assert fp.robustness() == pytest.approx(0.2 / 0.6, abs=1e-6)
    assert fp.ushape().n == 3


def test_fingerprint_orientation_flips_for_nll():
    nll = _fp("m", [0.2, 0.9, 0.3], metric="gold_nll", higher=False)
    assert nll.worst == pytest.approx(0.9)  # the *raw* value at the weakest position
    assert nll.worst_at == pytest.approx(0.50)
    assert nll.oriented_worst == pytest.approx(-0.9)
    assert nll.robustness() is None  # a ratio of NLLs is not a robustness index


def test_metric_orientation_is_inferred_from_the_name():
    curve = {0.0: 0.4, 1.0: 0.6}
    assert S.fingerprint_from_curve("m", curve).higher_is_better
    assert not S.fingerprint_from_curve("m", curve, metric="gold_nll").higher_is_better
    assert S.fingerprint_from_curve(
        "m", {0.0: 0.4, 1.0: 0.6}, metric="gold_nll", higher_is_better=True
    ).higher_is_better


def test_fingerprint_validation():
    with pytest.raises(ValueError, match="sorted ascending"):
        S.PositionalFingerprint(label="m", positions=(0.9, 0.1), values=(1.0, 0.0))
    with pytest.raises(ValueError, match="positions but"):
        S.PositionalFingerprint(label="m", positions=(0.1, 0.9), values=(1.0,))
    with pytest.raises(ValueError, match="non-finite"):
        S.PositionalFingerprint(label="m", positions=(0.1,), values=(float("nan"),))
    with pytest.raises(ValueError, match="one count per position"):
        S.PositionalFingerprint(label="m", positions=(0.1, 0.9), values=(1.0, 0.0), support=(3,))


def test_fingerprint_from_records_aggregates_and_counts_support():
    records = [
        {"target_positions": (0.05,), "correct": True},
        {"target_positions": (0.05,), "correct": False},
        {"target_positions": (0.50,), "correct": False},
        {"target_positions": (0.50,), "correct": None},  # not scored: dropped, not zeroed
        {"target_positions": (0.95,), "correct": True},
    ]
    fp = S.fingerprint_from_records(records, label="qwen3")
    assert fp.positions == (0.05, 0.50, 0.95)
    assert fp.values == pytest.approx((0.5, 0.0, 1.0))
    assert fp.support == (2, 1, 1)


def test_fingerprint_from_records_honours_min_support_and_a_custom_reducer():
    records = [
        {"target_positions": (0.1, 0.9), "gold_nll": 1.0},
        {"target_positions": (0.1, 0.9), "gold_nll": 2.0},
        {"target_positions": (0.4, 0.6), "gold_nll": 3.0},
    ]
    fp = S.fingerprint_from_records(
        records,
        label="m",
        metric="gold_nll",
        position_reducer=lambda p: float(np.mean(p)),
        min_support=2,
    )
    assert fp.positions == (0.5,)  # both rows reduce to 0.5; the singleton 0.5 group merges
    assert fp.values == pytest.approx((2.0,))
    assert not fp.higher_is_better


def test_fingerprint_from_records_refuses_an_empty_result():
    with pytest.raises(ValueError, match="no position had at least"):
        S.fingerprint_from_records([{"target_positions": (0.1,), "correct": None}], label="m")


def test_fingerprints_on_different_grids_cannot_be_compared():
    a = _fp("a", [0.5, 0.5, 0.5])
    b = S.PositionalFingerprint(label="b", positions=(0.1, 0.5), values=(0.4, 0.4))
    with pytest.raises(ValueError, match="different position grids"):
        a.oriented_delta(b)
    nll = _fp("c", [0.5, 0.5, 0.5], metric="gold_nll", higher=False)
    with pytest.raises(ValueError, match="orientation"):
        a.oriented_delta(nll)


# --------------------------------------------------------------------------------------
# Soup: complementarity
# --------------------------------------------------------------------------------------


def test_complementarity_finds_the_crossover_and_prices_it():
    edges = _fp("edges", [0.9, 0.2, 0.9])
    middle = _fp("middle", [0.4, 0.7, 0.4])
    report = S.complementarity(edges, middle)
    assert report.complementary
    assert report.crossovers == 2
    assert report.best_by_position == ("edges", "middle", "edges")
    # oracle = [0.9, 0.7, 0.9] -> 0.8333 vs the best single mean 0.6667
    assert report.oracle_gain == pytest.approx(0.8333333 - 0.6666667, abs=1e-6)
    assert report.worst_gain == pytest.approx(0.7 - 0.4, abs=1e-9)
    assert report.correlation < 0


def test_a_dominant_model_is_not_complementary():
    strong = _fp("strong", [0.9, 0.8, 0.9])
    weak = _fp("weak", [0.4, 0.3, 0.4])
    report = S.complementarity(strong, weak)
    assert not report.complementary
    assert report.crossovers == 0
    assert report.oracle_gain == pytest.approx(0.0)
    assert set(report.best_by_position) == {"strong"}


def test_complementarity_matrix_is_symmetric_with_a_zero_diagonal():
    fps = [_fp("a", [0.9, 0.2, 0.9]), _fp("b", [0.4, 0.7, 0.4]), _fp("c", [0.5, 0.5, 0.5])]
    labels, matrix = S.complementarity_matrix(fps)
    assert labels == ("a", "b", "c")
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] > 0
    rows = S.complementarity_table(fps)
    assert len(rows) == 3
    assert {r["label_a"] for r in rows} == {"a", "b"}


def test_complementarity_of_two_flat_curves_reports_an_undefined_correlation():
    report = S.complementarity(_fp("a", [0.5, 0.5, 0.5]), _fp("b", [0.4, 0.4, 0.4]))
    assert np.isnan(report.correlation)
    assert not report.complementary


# --------------------------------------------------------------------------------------
# Soup: the acceptance rule
# --------------------------------------------------------------------------------------


def test_rule_accepts_a_worst_position_improvement_with_no_regression():
    rule = S.AcceptanceRule()
    base = _fp("base", [0.6, 0.2, 0.6])
    better = _fp("cand", [0.6, 0.35, 0.6])
    decision = rule.evaluate(base, better)
    assert decision.accepted
    assert decision.improvements_met == ("worst_position_gain", "overall_gain")
    assert decision.guards_failed == ()
    assert decision.criterion("worst_position_gain").measured == pytest.approx(0.15)


def test_rule_accepts_an_overall_improvement_alone():
    rule = S.AcceptanceRule()
    base = _fp("base", [0.6, 0.2, 0.6])
    better = _fp("cand", [0.8, 0.2, 0.8])  # worst unchanged, mean up
    decision = rule.evaluate(base, better)
    assert decision.accepted
    assert decision.improvements_met == ("overall_gain",)
    assert not decision.criterion("worst_position_gain").passed


def test_rule_blocks_an_unacceptable_regression_elsewhere():
    rule = S.AcceptanceRule(max_position_regression=0.02)
    base = _fp("base", [0.6, 0.2, 0.6])
    # The mean *rises*, so the overall route is satisfied -- and the merge is still refused
    # because one position paid 0.4 for it. That trade is the one section 35 forbids.
    lopsided = _fp("cand", [0.2, 0.7, 0.7])
    decision = rule.evaluate(base, lopsided)
    assert not decision.accepted
    assert decision.improvements_met == ("overall_gain",)
    assert decision.guards_failed == ("position_regression",)
    assert decision.criterion("position_regression").measured == pytest.approx(0.4)
    assert decision.criterion("overall_regression").passed
    assert "position_regression" in decision.reason


def test_rule_rejects_a_candidate_that_improves_nothing():
    decision = S.AcceptanceRule().evaluate(_fp("base", [0.6, 0.2, 0.6]), _fp("c", [0.6, 0.2, 0.6]))
    assert not decision.accepted
    assert decision.improvements_met == ()
    assert "neither" in decision.reason


def test_require_all_needs_both_routes():
    base = _fp("base", [0.6, 0.2, 0.6])
    overall_only = _fp("cand", [0.8, 0.2, 0.8])
    assert S.AcceptanceRule(require="any").evaluate(base, overall_only).accepted
    assert not S.AcceptanceRule(require="all").evaluate(base, overall_only).accepted
    with pytest.raises(ValueError, match="require must be"):
        S.AcceptanceRule(require="sometimes")


def test_preservation_guard_fails_when_the_measurement_is_missing():
    rule = S.AcceptanceRule(min_preservation=0.9)
    base, cand = _fp("base", [0.6, 0.2, 0.6]), _fp("cand", [0.6, 0.4, 0.6])
    assert not rule.evaluate(base, cand).accepted
    assert rule.evaluate(base, cand, preservation=0.95).accepted
    assert not rule.evaluate(base, cand, preservation=0.5).accepted


def test_rule_works_in_oriented_units_for_nll():
    rule = S.AcceptanceRule()
    base = _fp("base", [0.3, 0.9, 0.3], metric="gold_nll", higher=False)
    better = _fp("cand", [0.3, 0.6, 0.3], metric="gold_nll", higher=False)  # lower NLL is better
    decision = rule.evaluate(base, better)
    assert decision.accepted
    assert decision.criterion("worst_position_gain").measured == pytest.approx(0.3)


def test_decision_and_rule_are_serialisable_and_inspectable():
    decision = S.AcceptanceRule().evaluate(_fp("b", [0.5, 0.5, 0.5]), _fp("c", [0.6, 0.6, 0.6]))
    payload = decision.to_json()
    assert json.loads(json.dumps(payload))["accepted"] is True
    assert [c["name"] for c in payload["criteria"]][:2] == ["worst_position_gain", "overall_gain"]
    fields = {"measured", "threshold", "comparison", "passed"}
    assert all(fields <= set(c) for c in payload["criteria"])
    with pytest.raises(KeyError, match="no criterion named"):
        decision.criterion("wishful")
    assert json.loads(json.dumps(S.AcceptanceRule().describe()))["require"] == "any"


def test_default_rule_widens_tolerances_for_a_nats_metric():
    assert S.default_acceptance_rule("accuracy").max_position_regression == pytest.approx(0.02)
    assert S.default_acceptance_rule("gold_nll").max_position_regression == pytest.approx(0.05)


# --------------------------------------------------------------------------------------
# Soup: recipes
# --------------------------------------------------------------------------------------


def test_recipe_is_immutable_and_deterministically_named(blocks):
    base = S.SoupRecipe(base_key="qwen3")
    assert base.is_base() and base.ingredients == () and base.slug() == "qwen3-base"

    one = base.with_ingredient("halo", A.BlockAlphas.uniform(blocks, 1.0))
    assert base.ingredients == ()  # unchanged
    assert one.ingredients == ("halo",)
    assert one.slug() == base.with_ingredient("halo", A.BlockAlphas.uniform(blocks, 1.0)).slug()
    assert one.without_ingredient("halo").is_base()


def test_recipe_normalisation_divides_by_the_ingredient_count(blocks):
    alphas = A.BlockAlphas.uniform(blocks, 1.0)
    recipe = S.SoupRecipe(
        base_key="qwen3", alphas={"halo": alphas, "nimbus": alphas}, normalize=True
    )
    effective = recipe.effective_alphas()
    assert all(v == pytest.approx(0.5) for v in effective["halo"].alphas.values())
    assert recipe.alphas["halo"].alphas["block0"] == 1.0  # the recipe itself is untouched
    assert recipe.slug().endswith("-norm")


def test_recipe_rejects_the_base_as_its_own_ingredient(blocks):
    with pytest.raises(ValueError, match="cannot also be an ingredient"):
        S.SoupRecipe(base_key="qwen3", alphas={"qwen3": A.BlockAlphas.uniform(blocks, 1.0)})
    with pytest.raises(TypeError, match="expected BlockAlphas"):
        S.SoupRecipe(base_key="qwen3", alphas={"halo": 1.0})


# --------------------------------------------------------------------------------------
# Soup: the searches
# --------------------------------------------------------------------------------------


def _quadratic_evaluate(optimum, blocks, *, counter=None):
    """Score peaks when every block's alpha matches ``optimum``; a known answer to find."""

    def evaluate(recipe: S.SoupRecipe) -> S.PositionalFingerprint:
        if counter is not None:
            counter.append(recipe.slug())
        effective = recipe.effective_alphas()
        loss = 0.0
        for key, targets in optimum.items():
            # An absent ingredient is alpha = 0 everywhere, not "no contribution to loss":
            # otherwise the base would score as if it sat on the optimum.
            block_alphas = effective.get(key)
            for name, target in targets.items():
                alpha = block_alphas.alphas[name] if block_alphas is not None else 0.0
                loss += (alpha - target) ** 2
        lift = 0.4 * float(np.exp(-2.0 * loss))
        values = [0.30 + lift, 0.10 + lift, 0.30 + lift]
        return S.PositionalFingerprint(label=recipe.slug(), positions=POSITIONS, values=values)

    return evaluate


def test_global_alpha_search_finds_the_best_uniform_alpha(blocks):
    optimum = {"halo": {b.name: 1.0 for b in blocks}}
    result = S.global_alpha_search(
        _quadratic_evaluate(optimum, blocks), "halo", blocks, base_key="qwen3"
    )
    assert result.best.recipe.ingredients == ("halo",)
    assert all(a == 1.0 for a in result.best.recipe.alphas["halo"].alphas.values())
    assert result.n_evaluations == 4  # base plus the three non-zero alphas
    assert sum(1 for c in result.history if c.selected) == 1
    assert result.best.decision is not None and result.best.decision.accepted


def test_global_alpha_search_keeps_the_base_when_nothing_helps(blocks):
    optimum = {"halo": {b.name: 0.0 for b in blocks}}
    result = S.global_alpha_search(
        _quadratic_evaluate(optimum, blocks), "halo", blocks, base_key="qwen3"
    )
    assert result.best.recipe.is_base()
    assert all(not c.accepted for c in result.history if c.decision is not None)


def test_layerwise_alpha_search_beats_a_uniform_alpha(blocks):
    optimum = {"halo": {"block0": 1.0, "block1": 0.0, "global": 0.0}}
    evaluate = _quadratic_evaluate(optimum, blocks)
    uniform = S.global_alpha_search(evaluate, "halo", blocks, base_key="qwen3")
    layered = S.layerwise_alpha_search(
        evaluate,
        "halo",
        blocks,
        start=uniform.best.recipe.alphas.get("halo"),
        base_key="qwen3",
    )
    assert layered.best.score > uniform.best.score
    found = layered.best.recipe.alphas["halo"].alphas
    assert found["block0"] == 1.0
    assert found["block1"] == 0.0 and found["global"] == 0.0


def test_layerwise_search_rejects_a_zero_round_budget(blocks):
    with pytest.raises(ValueError, match="rounds must be at least 1"):
        S.layerwise_alpha_search(
            _quadratic_evaluate({"halo": {b.name: 0.0 for b in blocks}}, blocks),
            "halo",
            blocks,
            rounds=0,
        )


def _additive_evaluate(effects, blocks, *, base=(0.60, 0.20, 0.60)):
    """Each donor adds its own per-position vector, scaled by its mean alpha."""

    def evaluate(recipe: S.SoupRecipe) -> S.PositionalFingerprint:
        values = np.asarray(base, dtype=float)
        for key, block_alphas in recipe.effective_alphas().items():
            scale = float(np.mean(list(block_alphas.alphas.values())))
            values = values + scale * np.asarray(effects[key], dtype=float)
        return S.PositionalFingerprint(
            label=recipe.slug(), positions=POSITIONS, values=tuple(np.clip(values, 0.0, 1.0))
        )

    return evaluate


def test_greedy_soup_prefers_the_donor_that_lifts_the_worst_position(blocks):
    effects = {
        "mid": (0.0, 0.25, 0.0),  # fixes the middle: worst-position route
        "edge": (0.10, 0.0, 0.10),  # only lifts the edges: overall route
        "harmful": (0.0, -0.15, 0.0),
    }
    uniform = A.BlockAlphas.uniform(blocks, 1.0)
    result = S.greedy_position_aware_soup(
        _additive_evaluate(effects, blocks),
        {k: uniform for k in effects},
        base_key="qwen3",
    )
    assert set(result.best.recipe.ingredients) == {"mid", "edge"}
    assert "harmful" not in result.best.recipe.ingredients
    # the first offer is the worst-position donor, not the one with the better mean
    offers = [c for c in result.history if c.stage == "greedy_soup" and not c.recipe.is_base()]
    assert offers[0].recipe.ingredients == ("mid",)
    rejected = [c for c in offers if not c.accepted]
    assert rejected and all("harmful" in c.recipe.ingredients for c in rejected)


def test_greedy_soup_records_both_verdicts_for_each_addition(blocks):
    effects = {"mid": (0.0, 0.25, 0.0), "edge": (0.10, 0.0, 0.10)}
    uniform = A.BlockAlphas.uniform(blocks, 1.0)
    result = S.greedy_position_aware_soup(
        _additive_evaluate(effects, blocks), {k: uniform for k in effects}, base_key="qwen3"
    )
    second = [c for c in result.history if len(c.recipe.ingredients) == 2]
    assert second and second[0].decision is not None and second[0].decision_vs_base is not None
    # against the base the two-donor soup improves both routes; against the incumbent only one
    assert second[0].decision_vs_base.improvements_met == ("worst_position_gain", "overall_gain")
    assert second[0].decision.improvements_met == ("overall_gain",)


def test_greedy_soup_respects_the_ingredient_budget(blocks):
    effects = {"a": (0.05, 0.05, 0.05), "b": (0.04, 0.04, 0.04), "c": (0.03, 0.03, 0.03)}
    uniform = A.BlockAlphas.uniform(blocks, 1.0)
    result = S.greedy_position_aware_soup(
        _additive_evaluate(effects, blocks),
        {k: uniform for k in effects},
        base_key="qwen3",
        max_ingredients=2,
    )
    assert result.best.recipe.n_ingredients == 2


def test_greedy_soup_honours_a_preservation_floor(blocks):
    effects = {"mid": (0.0, 0.25, 0.0)}
    uniform = A.BlockAlphas.uniform(blocks, 1.0)
    evaluate = _additive_evaluate(effects, blocks)
    rule = S.AcceptanceRule(min_preservation=0.9)
    result = S.greedy_position_aware_soup(
        evaluate, {"mid": uniform}, base_key="qwen3", rule=rule
    )
    assert result.best.recipe.is_base()  # no preservation measurement -> guard fails

    slug = S.SoupRecipe(base_key="qwen3").with_ingredient("mid", uniform).slug()
    passed = S.greedy_position_aware_soup(
        evaluate, {"mid": uniform}, base_key="qwen3", rule=rule, preservation={slug: 0.97}
    )
    assert passed.best.recipe.ingredients == ("mid",)


def test_evaluate_is_memoised_per_distinct_recipe(blocks):
    calls: list[str] = []
    optimum = {"halo": {b.name: 1.0 for b in blocks}}
    S.layerwise_alpha_search(
        _quadratic_evaluate(optimum, blocks, counter=calls), "halo", blocks, base_key="qwen3"
    )
    assert len(calls) == len(set(calls))


def test_soup_pipeline_runs_all_three_stages_and_serialises(blocks):
    optimum = {
        "halo": {"block0": 1.0, "block1": 0.0, "global": 0.0},
        "nimbus": {"block0": 0.0, "block1": 0.5, "global": 0.0},
    }
    fps = [_fp("halo", [0.9, 0.2, 0.9]), _fp("nimbus", [0.4, 0.7, 0.4])]
    result = S.run_soup_pipeline(
        _quadratic_evaluate(optimum, blocks),
        ("halo", "nimbus"),
        blocks,
        base_key="qwen3",
        donor_fingerprints=fps,
    )
    assert len(result.global_stage) == 2 and len(result.layerwise_stage) == 2
    assert result.greedy_stage.stage == "greedy_soup"
    assert result.best_fingerprint.overall >= result.baseline.overall
    assert len(result.complementarity) == 1
    assert result.complementarity[0]["complementary"] is True
    assert json.loads(json.dumps(result.to_json()))["best_recipe"]["base_key"] == "qwen3"


def test_search_rejects_an_evaluate_that_returns_the_wrong_type(blocks):
    with pytest.raises(TypeError, match="must return a PositionalFingerprint"):
        S.global_alpha_search(lambda recipe: 0.5, "halo", blocks, base_key="qwen3")


# --------------------------------------------------------------------------------------
# Soup: applying a recipe to a live model
# --------------------------------------------------------------------------------------


def test_apply_soup_recipe_sums_the_donor_deltas(tiny_model, blocks):
    base = A.snapshot_state(tiny_model)
    d1 = {k: torch.full_like(v, 0.1) for k, v in base.items()}
    d2 = {k: torch.full_like(v, 0.2) for k, v in base.items()}
    recipe = S.SoupRecipe(
        base_key="qwen3",
        alphas={"a": A.BlockAlphas.uniform(blocks, 1.0), "b": A.BlockAlphas.uniform(blocks, 0.5)},
    )
    applied = S.apply_soup_recipe(tiny_model, base, {"a": d1, "b": d2}, recipe)
    after = tiny_model.state_dict()
    for name, tensor in base.items():
        assert torch.allclose(after[name].float(), tensor + 0.1 + 0.5 * 0.2, atol=1e-6), name
    applied.restore()
    for name, tensor in tiny_model.state_dict().items():
        assert torch.allclose(tensor.float(), base[name], atol=1e-6), name


def test_apply_soup_recipe_honours_normalisation_and_block_routing(tiny_model, blocks):
    base = A.snapshot_state(tiny_model)
    delta = {k: torch.full_like(v, 1.0) for k, v in base.items()}
    only_late = A.BlockAlphas(blocks=blocks, alphas={"block0": 0.0, "block1": 1.0, "global": 0.0})
    recipe = S.SoupRecipe(
        base_key="qwen3", alphas={"a": only_late, "b": only_late}, normalize=True
    )
    with S.apply_soup_recipe(tiny_model, base, {"a": delta, "b": delta}, recipe):
        after = tiny_model.state_dict()
        early = "model.layers.0.self_attn.q_proj.weight"
        late = "model.layers.3.self_attn.q_proj.weight"
        assert torch.allclose(after[early].float(), base[early], atol=1e-6)
        # 0.5 + 0.5 after normalisation, i.e. the full delta once
        assert torch.allclose(after[late].float(), base[late] + 1.0, atol=1e-6)
    assert torch.allclose(tiny_model.state_dict()[late].float(), base[late], atol=1e-6)


def test_apply_soup_recipe_rejects_an_ingredient_with_no_delta(tiny_model, blocks):
    base = A.snapshot_state(tiny_model)
    recipe = S.SoupRecipe(base_key="qwen3", alphas={"ghost": A.BlockAlphas.uniform(blocks, 1.0)})
    with pytest.raises(KeyError, match="no delta dict"):
        S.apply_soup_recipe(tiny_model, base, {}, recipe)
