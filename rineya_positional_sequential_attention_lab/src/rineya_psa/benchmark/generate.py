"""Bank construction: config in, ``position_examples.jsonl`` plus a split manifest out.

Determinism is a hard requirement -- two runs with the same config must produce
byte-identical files -- so nothing here reads the clock, the environment, or an
unordered container. Every seed is derived from ``types.stable_id`` over the config
name and the loop coordinates, iteration order is a fixed nest of lists, and the
manifest stores repo-relative POSIX paths.

Split assignment comes from :func:`rineya_psa.types.split_for_scenario` and nowhere
else. One family example is drawn per ``(grid, family, source_index)`` and reused
across every length, format, local position and global position, so the whole family
of variants shares one ``scenario_id`` and therefore one split, which is what makes
paired statistics across formats and lengths legitimate.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rineya_psa.benchmark import assemble, families
from rineya_psa.benchmark.assemble import SYSTEM_PROMPT
from rineya_psa.benchmark.tokenize import count_prompt_tokens, get_tokenizer, span_method
from rineya_psa.paths import CONFIGS, DATA, LAB_ROOT
from rineya_psa.types import BenchmarkItem, split_for_scenario, stable_id

JSONL_NAME = "position_examples.jsonl"
MANIFEST_NAME = "position_split_manifest.json"
SPLIT_DIRS: dict[str, str] = {
    "calibration": "calibration",
    "validation": "validation",
    "sealed": "sealed",
}

DEFAULT_CONFIG = CONFIGS / "benchmark_v1.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a mapping")
    return cfg


def config_sha256(config: Mapping[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _seed_from(*parts: object) -> int:
    return int(stable_id(*parts)[:12], 16)


def position_sets_for_example(
    family: str,
    example: families.FamilyExample,
    positions: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """The list of target tuples to sweep for one example.

    Single-evidence families sweep the grid directly. ``multi_hop`` sweeps the six
    start/middle/end combinations of the brief. ``state_constraint`` pins its
    non-decisive updates to evenly spaced anchors and sweeps the correction, which is
    the update whose position the experiment is actually about.
    """
    m = len(example.evidence_sentences)
    if m == 1:
        return tuple((float(p),) for p in positions)
    if family == "multi_hop":
        return tuple(
            families.multi_hop_targets(pair) for pair in families.MULTI_HOP_POSITION_PAIRS
        )
    decisive = int(example.meta["decisive_index"])
    anchors = [(j + 1) / m for j in range(m - 1)]
    out: list[tuple[float, ...]] = []
    for p in positions:
        targets: list[float] = []
        ai = 0
        for k in range(m):
            if k == decisive:
                targets.append(float(p))
            else:
                targets.append(round(anchors[ai], 4))
                ai += 1
        out.append(tuple(targets))
    return tuple(out)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


@dataclass
class BuildReport:
    jsonl_path: Path
    manifest_path: Path
    n_items: int
    n_scenarios: int
    jsonl_sha256: str
    counts: list[dict[str, Any]] = field(default_factory=list)
    split_paths: dict[str, str] = field(default_factory=dict)
    token_out_of_tolerance: int = 0
    token_abs_error_max: float = 0.0


def _rel(path: Path, root: Path) -> str:
    """Path relative to the build root, so a temp-dir build hashes like a real one."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _root_label(root: Path) -> str:
    """Lab-relative label for the build root; falls back to the bare directory name."""
    try:
        return root.resolve().relative_to(LAB_ROOT).as_posix()
    except ValueError:
        return root.resolve().name


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _grid_positions(cell: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float, ...]:
    raw = cell.get("positions", "full")
    if isinstance(raw, str):
        if raw != "full":
            raise ValueError(f"grid {cell.get('name')!r}: unknown positions alias {raw!r}")
        raw = config["positions"]
    return tuple(float(p) for p in raw)


def build(
    config: Mapping[str, Any] | str | Path = DEFAULT_CONFIG,
    *,
    out_root: str | Path | None = None,
) -> BuildReport:
    """Generate the benchmark bank described by ``config``.

    ``out_root`` replaces ``data/`` so tests can build into a temp dir; the layout
    underneath it (``generated/``, ``calibration/``, ...) is unchanged.
    """
    if isinstance(config, (str, Path)):
        config = load_config(config)
    cfg = dict(config)
    cfg_hash = config_sha256(cfg)
    name = str(cfg.get("name", "benchmark"))
    seed = int(cfg.get("seed", 0))
    scale = float(cfg.get("scale", 1.0))
    spc = int(cfg.get("sentences_per_chunk", 3))
    tolerance = float(cfg.get("token_tolerance", 0.02))
    tokenizer_key = str(cfg.get("tokenizer", "qwen3"))
    length_tokens = {str(k): int(v) for k, v in cfg["length_tokens"].items()}

    root = Path(out_root) if out_root is not None else DATA
    generated = root / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    for sub in SPLIT_DIRS.values():
        (root / sub).mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(tokenizer_key)

    def counter(system: str, user: str) -> int:
        return count_prompt_tokens(system, user, tokenizer)

    model_cache: dict[tuple[str, int, int], assemble.TokenModel] = {}
    jsonl_path = generated / JSONL_NAME
    manifest_path = root / MANIFEST_NAME

    counts: dict[tuple[str, str, str, str], int] = {}
    scenario_split: dict[str, str] = {}
    out_of_tolerance = 0
    max_abs_error = 0.0
    n_items = 0

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as sink:
        for cell in cfg["grids"]:
            cell_name = str(cell["name"])
            positions = _grid_positions(cell, cfg)
            n_scenarios = max(1, int(round(int(cell["scenarios"]) * scale)))
            # A grid may pin builder options (e.g. multi_hop's `guided`). They enter the
            # scenario id so a guided and a hard scenario at the same index are distinct
            # examples with independent splits, rather than colliding.
            family_kwargs = dict(cell.get("family_kwargs", {}))
            kwargs_tag = ",".join(f"{k}={family_kwargs[k]}" for k in sorted(family_kwargs))
            for family in cell["families"]:
                for source_index in range(n_scenarios):
                    ex_seed = _seed_from(name, seed, cell_name, family, source_index, kwargs_tag)
                    example = families.build_family(
                        family, random.Random(ex_seed), **family_kwargs
                    )
                    scenario_id = stable_id(name, cell_name, family, source_index, kwargs_tag)
                    split = split_for_scenario(scenario_id)
                    scenario_split[scenario_id] = split
                    target_sets = position_sets_for_example(family, example, positions)
                    decisive = int(example.meta["decisive_index"])

                    for length_bucket in cell["lengths"]:
                        token_target = length_tokens[str(length_bucket)]
                        for fmt in cell["formats"]:
                            for local in cell["local_positions"]:
                                bank_seed = _seed_from(
                                    name, seed, scenario_id, length_bucket, fmt, local
                                )
                                bank = assemble.fit_bank(
                                    example,
                                    token_target=token_target,
                                    tolerance=tolerance,
                                    fmt=str(fmt),
                                    local_position=str(local),
                                    seed=bank_seed,
                                    count_prompt_tokens=counter,
                                    sentences_per_chunk=spc,
                                    canonical_targets=target_sets[0],
                                    model_cache=model_cache,
                                )
                                for targets in target_sets:
                                    layout = assemble.render_variant(bank, targets)
                                    n_tok = counter(SYSTEM_PROMPT, layout.user)
                                    err = (n_tok - token_target) / float(token_target)
                                    max_abs_error = max(max_abs_error, abs(err))
                                    if abs(err) > tolerance:
                                        out_of_tolerance += 1
                                    item = _make_item(
                                        config_name=name,
                                        config_hash=cfg_hash,
                                        cell_name=cell_name,
                                        family=family,
                                        example=example,
                                        scenario_id=scenario_id,
                                        source_index=source_index,
                                        split=split,
                                        fmt=str(fmt),
                                        local=str(local),
                                        length_bucket=str(length_bucket),
                                        targets=targets,
                                        decisive=decisive,
                                        bank=bank,
                                        layout=layout,
                                        seed=bank_seed,
                                        token_target=token_target,
                                        n_tok=n_tok,
                                        tolerance=tolerance,
                                        tokenizer_key=tokenizer_key,
                                        span_mode=span_method(tokenizer),
                                    )
                                    sink.write(
                                        json.dumps(item.to_json(), ensure_ascii=False) + "\n"
                                    )
                                    n_items += 1
                                    key = (
                                        family,
                                        f"{targets[decisive]:.2f}",
                                        str(length_bucket),
                                        split,
                                    )
                                    counts[key] = counts.get(key, 0) + 1

    split_paths = _write_splits(jsonl_path, root)
    jsonl_hash = _sha256_file(jsonl_path)

    manifest = {
        "config_name": name,
        "config_sha256": cfg_hash,
        "seed": seed,
        "scale": scale,
        "tokenizer": tokenizer_key,
        "span_localisation": span_method(tokenizer),
        "sentences_per_chunk": spc,
        "token_tolerance": tolerance,
        "token_out_of_tolerance": out_of_tolerance,
        "token_abs_error_max": round(max_abs_error, 6),
        "n_items": n_items,
        "n_scenarios": len(scenario_split),
        "data_root": _root_label(root),
        "generated": {"path": _rel(jsonl_path, root), "sha256": jsonl_hash},
        "splits": {
            split: {
                "path": _rel(Path(p), root),
                "sha256": _sha256_file(Path(p)),
                "n_items": _count_lines(Path(p)),
                "n_scenarios": sum(1 for v in scenario_split.values() if v == split),
            }
            for split, p in sorted(split_paths.items())
        },
        "counts_by_family_position_length_split": [
            {
                "family": f,
                "position": pos,
                "length_bucket": length,
                "split": sp,
                "n_items": counts[(f, pos, length, sp)],
            }
            for (f, pos, length, sp) in sorted(counts)
        ],
        "scenario_split": dict(sorted(scenario_split.items())),
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    return BuildReport(
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        n_items=n_items,
        n_scenarios=len(scenario_split),
        jsonl_sha256=jsonl_hash,
        counts=manifest["counts_by_family_position_length_split"],
        split_paths={k: _rel(Path(v), root) for k, v in sorted(split_paths.items())},
        token_out_of_tolerance=out_of_tolerance,
        token_abs_error_max=max_abs_error,
    )


def _write_splits(jsonl_path: Path, root: Path) -> dict[str, str]:
    """Copy each split's subset into its own directory, preserving line order."""
    handles = {}
    paths: dict[str, str] = {}
    try:
        for split, sub in sorted(SPLIT_DIRS.items()):
            target = root / sub / JSONL_NAME
            paths[split] = str(target)
            handles[split] = target.open("w", encoding="utf-8", newline="\n")
        with jsonl_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                handles[json.loads(line)["split"]].write(line)
    finally:
        for fh in handles.values():
            fh.close()
    return paths


def _make_item(
    *,
    config_name: str,
    config_hash: str,
    cell_name: str,
    family: str,
    example: families.FamilyExample,
    scenario_id: str,
    source_index: int,
    split: str,
    fmt: str,
    local: str,
    length_bucket: str,
    targets: Sequence[float],
    decisive: int,
    bank: assemble.ScenarioBank,
    layout: assemble.Layout,
    seed: int,
    token_target: int,
    n_tok: int,
    tolerance: float,
    tokenizer_key: str,
    span_mode: str,
) -> BenchmarkItem:
    err = (n_tok - token_target) / float(token_target)
    meta: dict[str, Any] = {
        "config_name": config_name,
        "config_sha256": config_hash,
        "grid": cell_name,
        "source_index": source_index,
        "example": dict(example.meta),
        "question": example.question,
        "evidence_char_spans": [list(s) for s in layout.evidence_char_spans],
        "achieved_char_positions": [round(p, 6) for p in layout.achieved_char_positions],
        "distractor_char_positions": [
            round(p, 6) for p in layout.distractor_char_positions
        ],
        # `distractors` lists what the family drew; this is how many the bank had room
        # to place. They agree at every configured length, but the audit is cheap.
        "n_distractors_placed": len(layout.distractor_char_positions),
        "decisive_index": decisive,
        "primary_target": float(targets[decisive]),
        "primary_achieved_char": round(layout.achieved_char_positions[decisive], 6),
        "n_evidence": len(example.evidence_sentences),
        "n_chunks": bank.n_chunks,
        # Coarsest global position step this document can express: one paragraph.
        "position_resolution": round(1.0 / max(bank.n_chunks, 1), 6),
        "sentences_per_chunk": bank.sentences_per_chunk,
        "ballast_sentences": bank.ballast_sentences,
        "separator": assemble.SEPARATOR,
        "shuffle_scope": (
            "filler_sentences_within_paragraph" if fmt == "shuffled" else "none"
        ),
        "facts_invariant_across_formats": True,
        "filler_fingerprint": bank.filler_fingerprint,
        "order_signature": layout.order_signature,
        "token_target": token_target,
        "n_prompt_tokens": n_tok,
        "token_error": round(err, 6),
        "token_tolerance": tolerance,
        "token_in_tolerance": abs(err) <= tolerance,
        "tokenizer": tokenizer_key,
        "span_localisation": span_mode,
    }
    item_id = stable_id(
        scenario_id, length_bucket, fmt, local, ",".join(f"{t:.4f}" for t in targets)
    )
    return BenchmarkItem(
        item_id=item_id,
        scenario_id=scenario_id,
        family=family,
        fmt=fmt,
        length_bucket=length_bucket,
        target_positions=tuple(float(t) for t in targets),
        local_position=local,
        split=split,
        system=SYSTEM_PROMPT,
        user=layout.user,
        answer=example.answer,
        answer_aliases=tuple(example.aliases),
        distractors=tuple(example.distractor_sentences),
        evidence_sentences=tuple(example.evidence_sentences),
        seed=seed,
        meta=meta,
    )


def copy_into(source: Path, destination: Path) -> None:
    """Copy a produced artifact, used by scripts that stage a build elsewhere."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


__all__ = [
    "BuildReport",
    "JSONL_NAME",
    "MANIFEST_NAME",
    "build",
    "config_sha256",
    "copy_into",
    "load_config",
    "position_sets_for_example",
]
