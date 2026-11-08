"""Capability-preservation suites, assembled from corpora that already exist in this repo.

Nothing here invents a benchmark. Every record is read verbatim out of a frozen project
corpus and re-labelled into one of the seven preservation axes the brief asks for:

===========================  ==========================================================
suite                        source
===========================  ==========================================================
``short_context_chat``       ``Rineya_General_...Corpus_v3.0`` communication classes
``normal_qa``                ``Rineya_General_...Corpus_v3.0`` grounded / commonsense QA
``basic_reasoning``          ``Rineya_General_...Corpus_v3.0`` inference classes
``arithmetic``               ``Rineya_General_...Corpus_v3.0`` numeric classes
``instruction_following``    ``Rineya_General_...Corpus_v3.0`` constraint / revision classes
``rineya_short_response``    ``Rineya_Short_Response_Behavior_Corpus_v1.0`` preservation slice
``generation_stability``     ``Rineya_Long_Context_Demo_v1`` ``stability_*`` split
===========================  ==========================================================

The reasoning/communication class -> axis mapping follows the corpus's own
``manifests/class_distribution.csv`` capability groups; it is not a fresh taxonomy.

Sources are opened read-only. The generation-stability probe reuses the project's
canonical degeneracy detector,
``short-response-model-phase-0-1-2/scripts/common/observable_quality.py``, loaded by path
because that tree is a script directory rather than an installable package. When that file
is absent, :func:`detect_degenerate_loop` falls back to a local implementation of the same
labels and the same thresholds (``longest run >= 4``, ``repeated-bigram fraction >= 0.55``
over >= 8 tokens, ``repeated-trigram fraction >= 0.30`` over >= 10 tokens) and says so in
``report["detector"]``.

DEVIATION: the brief names a "normal QA" axis, but no corpus in this repo ships a
short-context factual-QA split. ``normal_qa`` is therefore the v3 corpus's grounded and
commonsense classes (``evidence_grounded_dialogue``, ``evidence_inference``,
``physical_commonsense``, ``estimation_sanity``, ``troubleshooting_diagnosis``), which is
the nearest existing thing, rather than a newly written QA set.

DEVIATION: the suites are deterministic subsamples (``n_per_suite``, default 96) of corpora
holding 2 000-110 000 rows. Sampling is by ``sha256(f"{seed}|{suite}|{record_id}")`` sort
order, the same construction the corpora use for their own audit pools, so a suite is
reproducible from ``(seed, n_per_suite)`` alone.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import unicodedata
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rineya_psa.paths import REPO_ROOT
from rineya_psa.types import normalize_answer

__all__ = [
    "PRESERVATION_SUITES",
    "SUITE_SOURCES",
    "DegeneracyReport",
    "PreservationRecord",
    "detect_degenerate_loop",
    "generation_stability_report",
    "load_preservation_suites",
    "nan_guard",
    "preservation_source_status",
    "repetition_rate",
    "run_preservation",
    "score_preservation",
]

DEFAULT_SEED = 20260828
DEFAULT_N_PER_SUITE = 96

_GEN3 = "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0"
_SHORT = "Rineya_Short_Response_Behavior_Corpus_v1.0"
_DEMO = "Rineya_Long_Context_Demo_v1"
_OBSERVABLE_QUALITY = (
    "short-response-model-phase-0-1-2/scripts/common/observable_quality.py"
)


@dataclass(frozen=True)
class _SuiteSource:
    """Where one suite's rows come from and how many tokens its replies deserve."""

    kind: str  # "gen3_classes" | "sft_jsonl" | "demo_split"
    paths: tuple[str, ...]  # repo-root-relative
    max_new_tokens: int
    answer_key: str | None = None


SUITE_SOURCES: dict[str, _SuiteSource] = {
    "short_context_chat": _SuiteSource(
        kind="gen3_classes",
        paths=tuple(
            f"{_GEN3}/data/sft/by_class/{c}.jsonl"
            for c in (
                "informal_texting",
                "natural_follow_up",
                "conversation_closure",
                "clarification_request",
                "empathy_validation",
            )
        ),
        max_new_tokens=32,
    ),
    "normal_qa": _SuiteSource(
        kind="gen3_classes",
        paths=tuple(
            f"{_GEN3}/data/sft/by_class/{c}.jsonl"
            for c in (
                "evidence_grounded_dialogue",
                "evidence_inference",
                "physical_commonsense",
                "estimation_sanity",
                "troubleshooting_diagnosis",
            )
        ),
        max_new_tokens=48,
    ),
    "basic_reasoning": _SuiteSource(
        kind="gen3_classes",
        paths=tuple(
            f"{_GEN3}/data/sft/by_class/{c}.jsonl"
            for c in (
                "logical_implication",
                "causal_reasoning",
                "sequence_pattern",
                "spatial_relations",
                "comparison_ranking",
                "conditional_rules",
                "contradiction_detection",
            )
        ),
        max_new_tokens=48,
    ),
    "arithmetic": _SuiteSource(
        kind="gen3_classes",
        paths=tuple(
            f"{_GEN3}/data/sft/by_class/{c}.jsonl"
            for c in (
                "arithmetic_chain",
                "percentage_reasoning",
                "ratio_proportion",
                "unit_conversion",
                "counting_sets",
                "time_duration",
            )
        ),
        max_new_tokens=32,
    ),
    "instruction_following": _SuiteSource(
        kind="gen3_classes",
        paths=tuple(
            f"{_GEN3}/data/sft/by_class/{c}.jsonl"
            for c in (
                "constraint_selection",
                "insufficient_information",
                "plan_revision",
                "acknowledgment_update",
                "multi_turn_context_update",
                "communication_rewrite",
                "schedule_ordering",
            )
        ),
        max_new_tokens=48,
    ),
    "rineya_short_response": _SuiteSource(
        kind="sft_jsonl",
        paths=(
            f"{_SHORT}/data/sft/rineya_short_response_preservation_slice.jsonl",
            f"{_SHORT}/data/sft/rineya_short_response_combined_train.jsonl",
        ),
        max_new_tokens=32,
    ),
    "generation_stability": _SuiteSource(
        kind="demo_split",
        paths=(f"{_DEMO}/data/stability_prompts.jsonl",),
        # 24 new tokens is the frozen Rineya_Long_Context_Demo_v1 protocol.
        max_new_tokens=24,
        answer_key=f"{_DEMO}/data/stability_answer_key.jsonl",
    ),
}

PRESERVATION_SUITES: tuple[str, ...] = tuple(SUITE_SOURCES)


@dataclass(frozen=True)
class PreservationRecord:
    """One preservation prompt plus its reference reply, ready for ``runner.run_items``."""

    record_id: str
    suite: str
    source: str
    messages: tuple[Mapping[str, str], ...]
    reference: str
    accepted_answers: tuple[str, ...]
    max_new_tokens: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def system(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "system"), "")

    @property
    def user(self) -> str:
        """The last user turn; earlier turns stay inside ``messages`` for templating."""
        users = [m["content"] for m in self.messages if m["role"] == "user"]
        return users[-1] if users else ""

    def to_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "suite": self.suite,
            "source": self.source,
            "messages": [dict(m) for m in self.messages],
            "reference": self.reference,
            "accepted_answers": list(self.accepted_answers),
            "max_new_tokens": self.max_new_tokens,
            "meta": dict(self.meta),
        }


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - corpora are validated
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _rank(suite: str, seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}|{suite}|{record_id}".encode()).hexdigest()


def _clean_messages(raw: Any) -> tuple[tuple[Mapping[str, str], ...], str]:
    """Split a chat transcript into (prompt turns, final assistant reply)."""
    if not isinstance(raw, list):
        return (), ""
    turns = [
        {"role": str(m["role"]), "content": str(m["content"])}
        for m in raw
        if isinstance(m, dict) and "role" in m and "content" in m
    ]
    if turns and turns[-1]["role"] == "assistant":
        return tuple(turns[:-1]), turns[-1]["content"]
    return tuple(turns), ""


def _gen3_records(suite: str, source: _SuiteSource, root: Path) -> list[PreservationRecord]:
    records: list[PreservationRecord] = []
    for rel in source.paths:
        path = root / rel
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            messages, reference = _clean_messages(row.get("messages"))
            if not messages or not reference:
                continue
            meta = dict(row.get("metadata") or {})
            budget = int(meta.get("target_token_max") or source.max_new_tokens)
            records.append(
                PreservationRecord(
                    record_id=str(row.get("id", "")),
                    suite=suite,
                    source=rel,
                    messages=messages,
                    reference=reference,
                    accepted_answers=(reference,),
                    max_new_tokens=max(8, min(64, budget)),
                    meta={
                        "reply_class": meta.get("reply_class"),
                        "capability_group": meta.get("capability_group"),
                        "difficulty": meta.get("difficulty"),
                        "multi_turn": meta.get("multi_turn"),
                    },
                )
            )
    return records


def _sft_records(suite: str, source: _SuiteSource, root: Path) -> list[PreservationRecord]:
    """First existing path wins; the later paths are documented fallbacks, not a merge."""
    for rel in source.paths:
        path = root / rel
        if not path.exists():
            continue
        records: list[PreservationRecord] = []
        for row in _iter_jsonl(path):
            messages, reference = _clean_messages(row.get("messages"))
            if not messages or not reference:
                continue
            meta = dict(row.get("metadata") or {})
            budget = int(meta.get("target_token_max") or source.max_new_tokens)
            records.append(
                PreservationRecord(
                    record_id=str(row.get("id", "")),
                    suite=suite,
                    source=rel,
                    messages=messages,
                    reference=reference,
                    accepted_answers=(reference,),
                    max_new_tokens=max(8, min(64, budget)),
                    meta={
                        "reply_class": meta.get("reply_class"),
                        "length_mode": meta.get("length_mode"),
                        "safety_tier": meta.get("safety_tier"),
                    },
                )
            )
        if records:
            return records
    return []


def _demo_records(suite: str, source: _SuiteSource, root: Path) -> list[PreservationRecord]:
    prompts = root / source.paths[0]
    if not prompts.exists():
        return []
    key: dict[str, dict[str, Any]] = {}
    if source.answer_key:
        key_path = root / source.answer_key
        if key_path.exists():
            key = {str(r["id"]): r for r in _iter_jsonl(key_path) if "id" in r}

    records: list[PreservationRecord] = []
    for row in _iter_jsonl(prompts):
        messages, _ = _clean_messages(row.get("messages"))
        if not messages:
            continue
        row_id = str(row.get("id", ""))
        gold = key.get(row_id, {})
        reference = str(gold.get("answer", ""))
        accepted = tuple(str(a) for a in gold.get("accepted_answers", ()) or ())
        records.append(
            PreservationRecord(
                record_id=row_id,
                suite=suite,
                source=source.paths[0],
                messages=messages,
                reference=reference,
                accepted_answers=accepted or ((reference,) if reference else ()),
                max_new_tokens=source.max_new_tokens,
                meta={
                    "scenario_id": row.get("scenario_id"),
                    "variant": row.get("variant"),
                    "task_type": row.get("task_type"),
                    "needle_profile": row.get("needle_profile"),
                    "length_bucket": row.get("length_bucket"),
                    "answerable": row.get("answerable"),
                    "context_words": row.get("context_words"),
                },
            )
        )
    return records


_LOADERS = {
    "gen3_classes": _gen3_records,
    "sft_jsonl": _sft_records,
    "demo_split": _demo_records,
}


def preservation_source_status(repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Which suite sources are present on this machine. Cheap: stats only, no parsing."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    status: dict[str, dict[str, Any]] = {}
    for suite, source in SUITE_SOURCES.items():
        present = [rel for rel in source.paths if (root / rel).exists()]
        missing = [rel for rel in source.paths if rel not in present]
        if source.kind == "sft_jsonl":
            available = bool(present)  # first-existing-wins
        else:
            available = bool(present) and not (source.kind == "gen3_classes" and missing)
        status[suite] = {
            "available": available,
            "kind": source.kind,
            "present": present,
            "missing": missing,
            "answer_key_present": (
                None if source.answer_key is None else (root / source.answer_key).exists()
            ),
        }
    return status


@functools.lru_cache(maxsize=8)
def _load_cached(
    suites: tuple[str, ...],
    n_per_suite: int,
    seed: int,
    root_str: str,
    strict: bool,
) -> tuple[tuple[str, tuple[PreservationRecord, ...]], ...]:
    root = Path(root_str)
    out: list[tuple[str, tuple[PreservationRecord, ...]]] = []
    for suite in suites:
        source = SUITE_SOURCES[suite]
        records = _LOADERS[source.kind](suite, source, root)
        if not records:
            message = (
                f"preservation suite {suite!r} could not be loaded; expected one of "
                f"{list(source.paths)} under {root}"
            )
            if strict:
                raise FileNotFoundError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            continue
        records.sort(key=lambda r: _rank(suite, seed, r.record_id))
        if n_per_suite > 0:
            records = records[:n_per_suite]
        # Restore a stable, human-readable order inside the sampled set.
        records.sort(key=lambda r: r.record_id)
        out.append((suite, tuple(records)))
    return tuple(out)


def load_preservation_suites(
    *,
    suites: Sequence[str] | None = None,
    n_per_suite: int = DEFAULT_N_PER_SUITE,
    seed: int = DEFAULT_SEED,
    repo_root: Path | None = None,
    strict: bool = False,
) -> dict[str, list[PreservationRecord]]:
    """Load the seven preservation axes from the frozen corpora in this repo.

    ``n_per_suite <= 0`` returns every row of every source (110 000+ for the reasoning
    axes); the default of 96 keeps a full preservation sweep inside a few GPU-minutes.
    A suite whose source is missing is omitted with a ``RuntimeWarning`` unless
    ``strict=True``, so the caller can always tell the difference between "scored zero"
    and "never ran". Never substitutes one corpus for another silently.
    """
    names = tuple(suites) if suites is not None else PRESERVATION_SUITES
    unknown = [s for s in names if s not in SUITE_SOURCES]
    if unknown:
        raise KeyError(f"unknown preservation suite(s) {unknown}; known: {list(SUITE_SOURCES)}")
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    loaded = _load_cached(names, int(n_per_suite), int(seed), str(root), bool(strict))
    return {suite: list(records) for suite, records in loaded}


# --------------------------------------------------------------------------------------
# Generation-stability probe
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DegeneracyReport:
    """One generation's stability verdict."""

    degenerate: bool
    labels: tuple[str, ...]
    repetition_rate: float
    bigram_repeat: float
    longest_run: int
    n_words: int
    detector: str

    def to_json(self) -> dict[str, Any]:
        return {
            "degenerate": self.degenerate,
            "labels": list(self.labels),
            "repetition_rate": self.repetition_rate,
            "bigram_repeat": self.bigram_repeat,
            "longest_run": self.longest_run,
            "n_words": self.n_words,
            "detector": self.detector,
        }


_WORD_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*|\d+(?:[._-]\d+)*", re.UNICODE)


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.replace("’", "'").replace("–", "-").replace("—", "-")
    return _WORD_RE.findall(re.sub(r"\s+", " ", normalized).strip())


def _repeat_fraction(tokens: Sequence[str], n: int) -> float:
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))


def _longest_run(tokens: Sequence[str]) -> int:
    longest = current = 0
    previous: str | None = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        previous = token
        longest = max(longest, current)
    return longest


def repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of repeated ``n``-grams; 0.0 for text with no repeated n-gram."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return _repeat_fraction(_words(text), n)


@functools.lru_cache(maxsize=1)
def _observable_quality(root_str: str) -> Any | None:
    """The project's canonical detector, loaded by path (that tree is not a package)."""
    path = Path(root_str) / _OBSERVABLE_QUALITY
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("rineya_psa_observable_quality", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - a broken vendor script must not kill a run
        warnings.warn(f"could not load {path}: {exc}", RuntimeWarning, stacklevel=2)
        return None
    return module


_CRITICAL_LABELS = frozenset(
    {
        "empty_output",
        "encoding_replacement_character",
        "invalid_control_character",
        "repeated_token_loop",
        "repeated_phrase_loop",
        "repeated_clause_loop",
        "single_token_loop",
    }
)


def detect_degenerate_loop(text: str, *, repo_root: Path | None = None) -> DegeneracyReport:
    """Degenerate-loop verdict using the project's thresholds.

    Delegates to ``observable_quality.detect_observable_defects`` when that script is on
    disk; otherwise applies the same labels and thresholds locally. ``report.detector``
    records which path ran, because the vendored detector also fires on defects this
    fallback does not model (script islands, tokenizer fragmentation, delimiter damage).
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    tokens = _words(text)
    bigram = _repeat_fraction(tokens, 2)
    trigram = _repeat_fraction(tokens, 3)
    run = _longest_run(tokens)

    module = _observable_quality(str(root))
    if module is not None:
        result = module.detect_observable_defects(str(text))
        labels = tuple(
            str(d.get("label", "")) for d in result.get("defects", []) if isinstance(d, dict)
        )
        return DegeneracyReport(
            degenerate=any(label in _CRITICAL_LABELS for label in labels),
            labels=labels,
            repetition_rate=trigram,
            bigram_repeat=bigram,
            longest_run=run,
            n_words=len(tokens),
            detector="observable_quality_v4.5",
        )

    labels_list: list[str] = []
    if not tokens:
        labels_list.append("empty_output")
    if "�" in str(text):
        labels_list.append("encoding_replacement_character")
    if any(
        unicodedata.category(c) in {"Cc", "Cs", "Co"} and c not in "\n\r\t" for c in str(text)
    ):
        labels_list.append("invalid_control_character")
    if run >= 4:
        labels_list.append("repeated_token_loop")
    if len(tokens) >= 8 and bigram >= 0.55:
        labels_list.append("repeated_phrase_loop")
    elif len(tokens) >= 10 and trigram >= 0.30:
        labels_list.append("repeated_clause_loop")
    return DegeneracyReport(
        degenerate=bool(labels_list),
        labels=tuple(labels_list),
        repetition_rate=trigram,
        bigram_repeat=bigram,
        longest_run=run,
        n_words=len(tokens),
        detector="psa_fallback",
    )


def nan_guard(values: Iterable[float]) -> dict[str, Any]:
    """NaN / inf census over model scores. ``clean`` is the thing to assert on."""
    array = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(array)
    bad = np.flatnonzero(~finite)
    return {
        "n": int(array.size),
        "n_nonfinite": int(bad.size),
        "n_nan": int(np.isnan(array).sum()),
        "first_bad_index": int(bad[0]) if bad.size else None,
        "clean": bool(bad.size == 0),
    }


def generation_stability_report(
    texts: Sequence[str],
    *,
    scores: Sequence[float] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Repetition rate, degenerate-loop rate, empty rate and a NaN guard, in one dict."""
    reports = [detect_degenerate_loop(t, repo_root=repo_root) for t in texts]
    n = len(reports)
    label_counts = Counter(label for r in reports for label in r.labels)
    return {
        "n": n,
        "detector": reports[0].detector if reports else "none",
        "repetition_rate_mean": float(np.mean([r.repetition_rate for r in reports])) if n else 0.0,
        "bigram_repeat_mean": float(np.mean([r.bigram_repeat for r in reports])) if n else 0.0,
        "degenerate_rate": float(np.mean([r.degenerate for r in reports])) if n else 0.0,
        "empty_rate": float(np.mean([r.n_words == 0 for r in reports])) if n else 0.0,
        "max_longest_run": max((r.longest_run for r in reports), default=0),
        "labels": dict(sorted(label_counts.items())),
        "nan_guard": nan_guard(scores if scores is not None else ()),
    }


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def run_preservation(
    model: Any,
    tok: Any,
    records: Sequence[PreservationRecord],
    *,
    ctx: Any | None = None,
    batch_tokens: int | None = None,
    max_batch_rows: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Generate the preservation replies and score them, reusing ``runner``'s batching.

    Preservation prompts are multi-turn chats rather than positional benchmark items, so
    they cannot go through ``run_items`` (which owns ``BenchmarkItem``); this composes the
    same length-bucketed, token-capped batching primitives instead of re-implementing them.
    Returns ``{"predictions": {...}, "scores": {...}}``.
    """
    from rineya_psa.evaluation import runner as _runner

    if not records:
        return {"predictions": {}, "scores": {}}

    from rineya_psa.types import TokenizedItem

    budget = batch_tokens if batch_tokens is not None else _runner.DEFAULT_BATCH_TOKENS
    rows = max_batch_rows if max_batch_rows is not None else _runner.DEFAULT_MAX_BATCH_ROWS

    tok_items: list[TokenizedItem] = []
    for record in records:
        rendered = _runner.render_messages(tok, record.messages)
        tok_items.append(
            TokenizedItem(
                item_id=record.record_id,
                input_ids=tuple(int(i) for i in tok.encode(rendered, add_special_tokens=False)),
                answer_ids=tuple(int(i) for i in tok.encode(record.reference)),
                evidence_spans=(),
                actual_positions=(),
            )
        )

    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None) or 0
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0]
    device = next(model.parameters()).device

    lengths = [t.n_tokens + r.max_new_tokens for t, r in zip(tok_items, records, strict=True)]
    predictions: dict[str, str] = {}
    ctx_manager = _runner._psa_context(model, ctx)
    with ctx_manager:
        for batch in _runner.plan_batches(lengths, batch_tokens=budget, max_rows=rows):
            texts = _runner.generate_batch(
                model,
                tok,
                [tok_items[i] for i in batch],
                pad_id=pad_id,
                device=device,
                # A shared budget per batch; the newline stop ends short replies early.
                max_new_tokens=max(records[i].max_new_tokens for i in batch),
            )
            for slot, index in enumerate(batch):
                predictions[records[index].record_id] = texts[slot]

    return {
        "predictions": predictions,
        "scores": score_preservation(records, predictions, repo_root=repo_root),
    }


def score_preservation(
    records: Sequence[PreservationRecord],
    predictions: Mapping[str, str],
    *,
    scores: Mapping[str, float] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Exact-match rate plus the stability probe, per suite.

    Exact match here is the same normalisation the positional benchmark uses
    (``types.normalize_answer``). For open-ended communication classes it is a floor, not a
    quality measure; the stability block is what the preservation gate actually reads.
    """
    by_suite: dict[str, list[PreservationRecord]] = {}
    for record in records:
        by_suite.setdefault(record.suite, []).append(record)

    out: dict[str, Any] = {}
    for suite, rows in sorted(by_suite.items()):
        texts = [str(predictions.get(r.record_id, "")) for r in rows]
        covered = [r.record_id in predictions for r in rows]
        hits = []
        for record, text in zip(rows, texts, strict=True):
            targets = {normalize_answer(a) for a in record.accepted_answers}
            targets.discard("")
            hits.append(bool(targets) and normalize_answer(text) in targets)
        suite_scores = (
            [scores[r.record_id] for r in rows if r.record_id in scores]
            if scores is not None
            else None
        )
        out[suite] = {
            "n": len(rows),
            "coverage": float(np.mean(covered)) if rows else 0.0,
            "exact_match": float(np.mean(hits)) if rows else 0.0,
            "stability": generation_stability_report(
                texts, scores=suite_scores, repo_root=repo_root
            ),
        }
    return out
