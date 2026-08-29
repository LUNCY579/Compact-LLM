"""SPEC section 1's hard rule: the four source checkpoints are read-only.

The test is a before/after digest of everything under the four checkpoint roots with a
representative slice of the lab running in between. Two deliberate asymmetries:

* **Safetensors payloads are not hashed.** 6.5 GB of SHA-256 is a minute of wall clock
  per snapshot and this file runs on every commit. Their size and mtime *are* recorded,
  which catches a truncation, an append, or a rewrite-in-place — every plausible way
  the lab could damage one — without paying for the bytes. Everything else in the tree
  (config, tokenizer, JSON metadata, remote-code ``.py``, README, evidence files) is
  hashed in full; that is only ~65 MB and takes about a third of a second.
* **No real model is instantiated.** Another process owns the GPU (task rule 4), and
  loading a checkpoint is not what would corrupt one anyway: the risks are a memmap
  opened ``r+``, a writer resolving its output path relative to a source directory, and
  ``trust_remote_code`` dropping a ``__pycache__`` next to ``modeling_supa_a11.py``.
  The slice below exercises the first two directly and leaves the third alone by not
  importing the remote module at all.

The slice covers: the Phase-1 audit's read paths (safetensors header parse, tensor
location table, config normalisation, tokenizer identity, and a memory-mapped tensor
read), a ``psa_session`` forward on the toy model, and a real
``merging.materialise_merge`` that reads two source checkpoints and writes the blend
into a scratch experiment workspace.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rineya_psa.model_audit import distances as distances_mod
from rineya_psa.model_audit import hashing as hashing_mod
from rineya_psa.model_audit import inspect_ckpt as inspect_mod
from rineya_psa.paths import PANEL, assert_readonly_target, model_spec

torch = pytest.importorskip("torch")

WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth"})


def _present_keys() -> list[str]:
    return [key for key in PANEL if model_spec(key).path.is_dir()]


PRESENT = _present_keys()
requires_checkpoints = pytest.mark.skipif(
    not PRESENT, reason="no panel checkpoint present on this machine"
)


# --------------------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """Digest of one checkpoint tree, split by whether the bytes were read."""

    hashed: dict[str, str]
    stat_only: dict[str, tuple[int, int]]

    def describe(self) -> dict[str, Any]:
        return {"hashed": dict(self.hashed), "stat_only": dict(self.stat_only)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_tree(root: Path) -> Snapshot:
    hashed: dict[str, str] = {}
    stat_only: dict[str, tuple[int, int]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            st = path.stat()
            stat_only[rel] = (int(st.st_size), int(st.st_mtime_ns))
        else:
            hashed[rel] = _sha256(path)
    return Snapshot(hashed=hashed, stat_only=stat_only)


def snapshot_panel() -> dict[str, Snapshot]:
    return {key: snapshot_tree(model_spec(key).path) for key in PRESENT}


# --------------------------------------------------------------------------------------
# The slice of the lab that runs between the two snapshots
# --------------------------------------------------------------------------------------


def _audit_reads(key: str) -> dict[str, Any]:
    """Every read the Phase-1 audit performs, minus the full-file hashing."""
    root = model_spec(key).path
    files = hashing_mod.discover_model_files(root)
    assert files, f"{key}: no weight shard discovered"

    entries, metadata, data_start = hashing_mod.read_safetensors_header(files[0])
    locations = hashing_mod.collect_tensor_locations(files)
    assert locations, f"{key}: empty tensor location table"

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    normalized = inspect_mod.normalize_config(config)
    identity = hashing_mod.tokenizer_identity(root)

    # The memmap path is the one that could plausibly open a checkpoint for writing.
    name = next(n for n in sorted(locations) if locations[n].n_params <= 4_000_000)
    with distances_mod.MemmapPool() as pool:
        block = pool.get(locations[name].file)
        assert block.mode == "r", f"{key}: memmap opened {block.mode!r}, expected 'r'"
        values = hashing_mod.decode_to_float64(
            block[locations[name].begin : locations[name].end], locations[name].st_dtype
        )
    assert values.size == locations[name].n_params

    return {
        "n_header_entries": len(entries),
        "metadata_keys": sorted(metadata),
        "data_start": int(data_start),
        "n_tensors": len(locations),
        "architectures": normalized["architectures"],
        "tokenizer_files": sorted(identity["files_present"]),
        "probe_tensor": name,
    }


def _psa_forward(model: Any) -> float:
    """A PSA session over the toy model: registers the kernel, runs, and restores."""
    import numpy as np

    from rineya_psa.hooks import AbsoluteBinning, PSAContext, ValueScaleSpec, psa_session

    n_layers = int(model.config.num_hidden_layers)
    n_kv = int(model.config.num_key_value_heads)
    binning = AbsoluteBinning(n_bins=4)
    table = torch.from_numpy(
        np.full((n_layers, n_kv, binning.n_bins), 0.9, dtype=np.float32)
    )
    ctx = PSAContext(
        enabled=True,
        value_scale=ValueScaleSpec(table=table, binning=binning),
        position_bins=binning,
    )
    input_ids = torch.randint(0, int(model.config.vocab_size), (1, 64))
    with psa_session(model, ctx), torch.no_grad():
        logits = model(input_ids).logits
    return float(logits.float().abs().sum())


def _materialise_merge(out_dir: Path) -> Any:
    """A real two-checkpoint blend written outside the checkpoint trees."""
    from rineya_psa.merging.arithmetic import BlockAlphas, default_blocks, materialise_merge

    keys = [k for k in ("qwen3", "nimbus", "halo", "supa_a11") if k in PRESENT]
    if len(keys) < 2:
        pytest.skip("need two panel checkpoints to exercise the merge writer")
    base_key, other_key = keys[0], keys[1]

    blocks = default_blocks(28)
    alphas = BlockAlphas(blocks=blocks, alphas={b.name: 0.5 for b in blocks})
    # Two small tensors is enough to prove the reader/writer pair works; blending the
    # whole 0.6B would add a minute of I/O and test nothing extra.
    wanted = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
    }
    return materialise_merge(
        model_spec(base_key).path,
        model_spec(other_key).path,
        alphas,
        base_key=base_key,
        other_key=other_key,
        merge_id="immutability_smoke",
        out_dir=out_dir,
        copy_aux=False,
        include=lambda n: n in wanted,
    )


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


@requires_checkpoints
def test_representative_slice_leaves_every_checkpoint_byte_identical(
    tiny_qwen3_model: Any, lab_dirs: dict[str, Path]
) -> None:
    before = snapshot_panel()
    assert before, "nothing was snapshotted"
    for key, snap in before.items():
        assert snap.hashed, f"{key}: no hashable files found — snapshot is vacuous"
        assert snap.stat_only, f"{key}: no weight shard found — snapshot is vacuous"

    audits = {key: _audit_reads(key) for key in PRESENT}
    for key, record in audits.items():
        assert record["n_tensors"] > 0
        assert record["architectures"], f"{key}: config declares no architecture"

    assert _psa_forward(tiny_qwen3_model) > 0.0

    artifact = _materialise_merge(lab_dirs["experiments/merges"] / "immutability_smoke")
    assert artifact.weights_file.is_file()
    assert artifact.n_tensors == 2

    after = snapshot_panel()

    assert sorted(after) == sorted(before)
    for key in before:
        b, a = before[key], after[key]
        added = sorted(set(a.hashed) - set(b.hashed))
        removed = sorted(set(b.hashed) - set(a.hashed))
        assert not added, f"{key}: files appeared inside the checkpoint: {added}"
        assert not removed, f"{key}: files vanished from the checkpoint: {removed}"
        changed = sorted(p for p in b.hashed if b.hashed[p] != a.hashed[p])
        assert not changed, f"{key}: content changed: {changed}"
        moved = {
            p: (b.stat_only[p], a.stat_only.get(p))
            for p in b.stat_only
            if b.stat_only[p] != a.stat_only.get(p)
        }
        assert not moved, f"{key}: a weight shard's size or mtime moved: {moved}"


@requires_checkpoints
def test_weight_shards_are_recorded_by_size_and_mtime_not_hashed() -> None:
    """The cost trade-off above is only sound if the shards really are covered."""
    for key in PRESENT:
        snap = snapshot_tree(model_spec(key).path)
        shards = [p for p in snap.stat_only if p.endswith(".safetensors")]
        assert shards, f"{key}: no safetensors shard recorded"
        for shard in shards:
            size, mtime = snap.stat_only[shard]
            assert size > 100_000_000, f"{key}/{shard}: implausibly small weight shard"
            assert mtime > 0


@requires_checkpoints
@pytest.mark.parametrize("key", PRESENT)
def test_assert_readonly_target_refuses_every_checkpoint(key: str) -> None:
    root = model_spec(key).path
    for candidate in (
        root,
        root / "config.json",
        root / "model.safetensors",
        root / "nested" / "deeper" / "new_file.bin",
    ):
        with pytest.raises(PermissionError):
            assert_readonly_target(candidate)


def test_assert_readonly_target_allows_the_lab_output_tree(lab_dirs: dict[str, Path]) -> None:
    """The guard has to be a filter, not a blanket refusal, or writers would be untestable."""
    from rineya_psa.paths import EXPERIMENTS, RESULTS

    for path in (
        lab_dirs["results"] / "run.parquet",
        lab_dirs["experiments/merges"] / "m" / "model.safetensors",
        RESULTS / "phase3.json",
        EXPERIMENTS / "merges" / "m" / "model.safetensors",
    ):
        assert_readonly_target(path)


@requires_checkpoints
def test_merge_workspace_is_outside_every_checkpoint() -> None:
    from rineya_psa.merging.arithmetic import merge_workspace

    workspace = merge_workspace("probe").resolve()
    for key in PRESENT:
        root = model_spec(key).path.resolve()
        assert root != workspace and root not in workspace.parents


@requires_checkpoints
def test_no_module_opens_a_checkpoint_path_for_writing() -> None:
    """Static backstop: nothing may hand a checkpoint path to a writing open mode.

    The before/after digest only catches what the slice happens to execute. This
    catches the shape of the mistake in code that no test calls yet.
    """
    from rineya_psa.paths import LAB_ROOT

    write_mode = re.compile(r"""["'][waxr]\+?b?["']""")
    offenders: list[str] = []
    for path in sorted((LAB_ROOT / "src").rglob("*.py")) + sorted(
        (LAB_ROOT / "scripts").rglob("*.py")
    ):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "spec.path" not in line and "model_spec(" not in line:
                continue
            match = write_mode.search(line)
            if match and match.group(0).strip("\"'") not in ("r", "rb"):
                rel = path.relative_to(LAB_ROOT).as_posix()
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "checkpoint path passed to a writing open():\n" + "\n".join(offenders)
