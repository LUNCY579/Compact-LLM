"""Evaluation hooks: exact-verification accuracy by generation, and the teacher-forced 2x2
functional-interaction measurement.

Generation scoring is resumable per ROW: each scored row is appended (fsync'd JSONL) as soon as
it is produced, keyed by (config label, row id). A driver reset (TDR) loses at most one row, and a
restart skips rows already on disk. Greedy decoding is deterministic, so re-scoring a row cannot
change its result; completeness is a row COUNT, never a process exit code.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .functional import CONFIGS, interaction_rows, kappa, summarise_vocab, vocab_interaction


def _append(path: Path, rec: dict) -> None:
    line = (json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    with open(path, "ab") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_scored(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line.decode("ascii")))
    return out


def score_generation(*, rows: Sequence, prompt_ids: Mapping[str, Sequence[int]], generate: Callable,
                     decode: Callable[[Sequence[int]], str], score: Callable, out_path: Path,
                     label: Mapping, max_new_tokens: int, guard: Callable[[], None] | None = None) -> list[dict]:
    """Score ``rows`` under the currently installed model configuration.

    ``label`` must contain arm/seed/order/config/stage; it is copied into every record and forms
    the resume key together with the row id.
    """
    for k in ("arm", "seed", "order", "config", "stage"):
        if k not in label:
            raise ValueError(f"label missing {k!r}")
    done = {(r["config"], r["row_id"]) for r in read_scored(out_path)
            if all(r.get(k) == label[k] for k in ("arm", "seed", "order", "config", "stage"))}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if (label["config"], row.id) in done:
            continue
        t0 = time.time()
        g = generate(prompt_ids[row.id], max_new_tokens)
        text = decode(g["ids"])
        rec = {**label, "row_id": row.id, "parent_id": row.parent_id, "signature": row.signature,
               "correct": int(score(row, text)), "terminated": bool(g["terminated"]),
               "generated_tokens": len(g["ids"]), "response": text, "seconds": round(time.time() - t0, 4)}
        _append(out_path, rec)
        if guard is not None:
            guard()
    return [r for r in read_scored(out_path)
            if all(r.get(k) == label[k] for k in ("arm", "seed", "order", "config", "stage"))]


def pair_interaction(*, host, delta_a, delta_b, rows: Sequence[dict], teacher_forced: Callable,
                     topk: int = 32, seed: int = 20260927) -> dict:
    """The 2x2 design on ``rows`` (compiled rows with ids/labels): f(0), f(A), f(B), f(A+B).

    Both deltas are placed on the device once; A+B is their FP32 sum per slot (the same
    arithmetic as ``deltas.compose``). Configuration switches are pointer swaps.
    """
    import torch

    tA = host.delta_to_device(delta_a)
    tB = host.delta_to_device(delta_b)
    tAB = {n: tA[n] + tB[n] for n in tA}
    configs = {"0": None, "A": tA, "B": tB, "AB": tAB}
    sums = {c: [] for c in CONFIGS}
    per_pos: dict[str, list] = {}
    try:
        for r in rows:
            vecs = {}
            for c in CONFIGS:
                host.set_frozen_tensors(configs[c])
                tf = teacher_forced(r, keep_vectors=True)
                sums[c].append(tf["sum_logprob"])
                vecs[c] = tf["logsm"]
            vi = vocab_interaction(vecs, topk=topk)
            for k, v in vi.items():
                if k != "topk":
                    per_pos.setdefault(k, []).append(np.asarray(v))
            del vecs
    finally:
        host.set_frozen_tensors(None)
        del tA, tB, tAB
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    I = interaction_rows(sums)
    vocab = summarise_vocab({**{k: np.concatenate(v) for k, v in per_pos.items()}, "topk": topk})
    return {"rows": len(rows), "sum_logprob": {c: [float(x) for x in sums[c]] for c in CONFIGS},
            "interaction_rows": [float(x) for x in I],
            "kappa_A": kappa(sums, seed=seed, normaliser="A"),
            "kappa_sum": kappa(sums, seed=seed, normaliser="AB_sum"),
            "vocab": vocab,
            "what_this_measures": ("second difference of teacher-forced log-probabilities of ONE reference "
                                   "answer per row (sum over answer tokens) and of full log-softmax vectors at "
                                   "answer positions; not a measure of composition accuracy")}
