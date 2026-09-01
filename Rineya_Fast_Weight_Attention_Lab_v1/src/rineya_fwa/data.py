"""Unsealed TRAIN/DEV builder.

Prefers official CC0 unsealed shards when they exist in this worktree. The CC0
tree is gitignored and was absent at worktree creation, so the default path is a
lab-owned synthetic retrieval mix with CC0-like metadata. Never touches sealed
evaluation.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from rineya_fwa.paths import DATA, LAB_ROOT, REPO_ROOT, assert_readonly_target, assert_unsealed_source

SYSTEM = (
    "You are answering a closed-context reading test. Use only the supplied context. "
    "Reply with the shortest exact answer and no explanation. If the context does not "
    "determine the answer, reply UNKNOWN."
)

COLOURS = (
    "amber",
    "cobalt",
    "ivory",
    "jade",
    "maroon",
    "ochre",
    "pearl",
    "slate",
    "teal",
    "violet",
)
FILLER_SENTENCE = (
    "The harbour ledger notes routine weather, ordinary cargo, and no change to the "
    "standing watch roster for this interval."
)

STRATUM_PROXY = {"2k": 400, "4k": 800}  # proxy words; tokenizer length is measured later


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_cc0_train() -> Path | None:
    root = REPO_ROOT / "rineya_long_context_cc0_v1_main" / "data" / "sft" / "train"
    if root.is_dir() and any(root.glob("*.jsonl")):
        return root
    sibling = REPO_ROOT.parent / "Chat-Bubble" / "rineya_long_context_cc0_v1_main" / "data" / "sft" / "train"
    # Read-only peek at the originating worktree; we never write there.
    if sibling.is_dir() and any(sibling.glob("*.jsonl")):
        return sibling
    return None


def _place(fact: str, filler_n: int, position: float, rng: random.Random) -> str:
    sentences = [FILLER_SENTENCE] * max(filler_n, 8)
    idx = int(round(position * (len(sentences) - 1)))
    idx = min(max(idx, 0), len(sentences) - 1)
    sentences[idx] = fact
    # light jitter so rows are not byte-identical
    if rng.random() < 0.3:
        sentences[(idx + 3) % len(sentences)] = FILLER_SENTENCE.replace("harbour", "quay")
    return " ".join(sentences)


def make_row(idx: int, *, stratum: str, seed: int, split: str) -> dict[str, Any]:
    rng = random.Random(seed + idx * 17)
    position = rng.choice([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    key = f"NX-{rng.randint(10000, 99999)}"
    colour = rng.choice(COLOURS)
    fact = f"The registry key {key} is assigned colour {colour}."
    ctx = _place(fact, STRATUM_PROXY[stratum], position, rng)
    question = f"What colour is registry key {key}?"
    return {
        "id": f"FWA-{split}-{stratum}-{idx:05d}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"CONTEXT\n{ctx}\n\nQUESTION\n{question}\n\nANSWER",
            },
            {"role": "assistant", "content": colour},
        ],
        "metadata": {
            "context_stratum": stratum,
            "evidence_position": position,
            "source": "lab_unsealed_synthetic",
            "split": split,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    assert_readonly_target(path)
    assert_unsealed_source(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return _sha256_file(path)


def build_lab_mix(
    *,
    train_rows: int,
    dev_rows: int,
    strata: list[str],
    seed: int,
    dest: Path | None = None,
) -> dict[str, Any]:
    dest = dest or (DATA / "unsealed")
    dest.mkdir(parents=True, exist_ok=True)
    official = official_cc0_train()
    source = "official_cc0_unsealed" if official is not None else "lab_unsealed_synthetic"
    if official is not None:
        # Copy a deterministic subset out of CC0 train into the lab tree.
        rows: list[dict[str, Any]] = []
        for shard in sorted(official.glob("*.jsonl")):
            assert_unsealed_source(shard)
            with shard.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
                    if len(rows) >= train_rows + dev_rows:
                        break
            if len(rows) >= train_rows + dev_rows:
                break
        rng = random.Random(seed)
        rng.shuffle(rows)
        train, dev = rows[:train_rows], rows[train_rows : train_rows + dev_rows]
        if not train:
            official = None
            source = "lab_unsealed_synthetic"
    if official is None:
        train = []
        n_per = max(1, train_rows // max(len(strata), 1))
        i = 0
        for stratum in strata:
            for _ in range(n_per):
                train.append(make_row(i, stratum=stratum, seed=seed, split="train"))
                i += 1
        train = train[:train_rows]
        dev = [
            make_row(10_000 + j, stratum=strata[j % len(strata)], seed=seed + 1, split="dev")
            for j in range(dev_rows)
        ]

    train_path = dest / "train.jsonl"
    dev_path = dest / "dev.jsonl"
    manifest = {
        "source": source,
        "official_cc0_present": official is not None,
        "train_path": str(train_path.relative_to(LAB_ROOT) if train_path.is_relative_to(LAB_ROOT) else train_path),
        "dev_path": str(dev_path.relative_to(LAB_ROOT) if dev_path.is_relative_to(LAB_ROOT) else dev_path),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_sha256": write_jsonl(train_path, train),
        "dev_sha256": write_jsonl(dev_path, dev),
        "seed": seed,
        "strata": strata,
        "note": (
            "CC0 tree is gitignored and was absent from this worktree. "
            "Synthetic rows are unsealed lab TRAIN/DEV, not the official CC0 release."
            if official is None
            else "Subset copied from official CC0 unsealed train shards."
        ),
    }
    man_path = dest / "mix_manifest.json"
    assert_readonly_target(man_path)
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
