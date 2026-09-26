"""Mechanical checks for the pieces that decide what the numbers mean.

Run from the repository root:

    ../Chat-Bubble/.venv/Scripts/python.exe -m pytest \
        experiments/supa_long_context_minimal_intervention_v1/tests -q

No model and no GPU. Token counting is stubbed with a whitespace counter so the
selector's logic is tested independently of the tokenizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from slcmi import context_selection as cs  # noqa: E402
from slcmi.stats import mcnemar_exact, paired_bootstrap_delta, relative_error_reduction  # noqa: E402


def words(text: str) -> int:
    return len(text.split())


CONTEXT = " ".join(f"Sentence {i} about topic {i} with filler words here." for i in range(40))


def test_chunking_never_splits_a_sentence_and_covers_everything():
    chunks = cs.chunk_context(CONTEXT, words, target_tokens=20)
    assert chunks
    rejoined = " ".join(c.text for c in chunks)
    assert rejoined == CONTEXT
    spans = [c.sentence_span for c in chunks]
    assert spans[0][0] == 0
    for earlier, later in zip(spans, spans[1:]):
        assert earlier[1] == later[0], "chunk spans must tile without gap or overlap"


def test_oversized_sentence_becomes_its_own_chunk_rather_than_being_truncated():
    long_sentence = "word " * 200 + "end."
    chunks = cs.chunk_context(long_sentence + " Short tail.", words, target_tokens=20)
    assert any("end." in c.text for c in chunks), "evidence must survive chunking intact"


def test_assemble_restores_document_order():
    chunks = cs.chunk_context(CONTEXT, words, target_tokens=20)
    out_of_order = [4, 0, 2]
    text = cs.assemble(chunks, out_of_order)
    positions = [text.index(chunks[i].text) for i in sorted(out_of_order)]
    assert positions == sorted(positions), "chunks must appear in document order"


def test_query_selection_prefers_chunks_that_share_query_terms():
    context = (
        "The archive opened on Monday. "
        "Unrelated filler about weather patterns and clouds. "
        "More filler concerning gardening tools and soil. "
        "Talia Frost carried the ivory seal to the vault. "
        "Further filler about distant mountains and rivers."
    )
    chunks = cs.chunk_context(context, words, target_tokens=8)
    selection = cs.select_query(chunks, "Who carried the ivory seal?", budget=40)
    chosen = " ".join(chunks[i].text for i in selection.chunk_ids)
    assert "ivory seal" in chosen


def test_random_control_matches_the_query_chunk_count():
    chunks = cs.chunk_context(CONTEXT, words, target_tokens=20)
    query = cs.select_query(chunks, "topic 7 filler", budget=60)
    control = cs.select_random(chunks, query.n_chunks, seed=1)
    assert control.n_chunks == query.n_chunks, (
        "B - C must isolate relevance, which requires a matched budget"
    )


def test_random_control_is_deterministic_under_seed():
    chunks = cs.chunk_context(CONTEXT, words, target_tokens=20)
    a = cs.select_random(chunks, 3, seed=99)
    b = cs.select_random(chunks, 3, seed=99)
    assert a.chunk_ids == b.chunk_ids


def test_oracle_is_unavailable_for_unanswerable_rows():
    chunks = cs.chunk_context(CONTEXT, words, target_tokens=20)
    assert cs.select_oracle(chunks, ["anything at all"], answerable=False) is None


def test_oracle_requires_every_fact_to_localise():
    chunks = cs.chunk_context(CONTEXT, words, target_tokens=20)
    good = "Sentence 3 about topic 3"
    unlocalisable = "zzz qqq xxx unrelated vocabulary nowhere in the document"
    assert cs.select_oracle(chunks, [good], answerable=True) is not None
    assert cs.select_oracle(chunks, [good, unlocalisable], answerable=True) is None, (
        "a two-hop oracle missing one hop is not an oracle"
    )


def test_exact_oracle_localisation_finds_the_containing_chunk():
    chunks = cs.chunk_context(CONTEXT, words, target_tokens=20)
    fact = chunks[3].text.split(".")[0] + "."
    chunk_id, confidence = cs.localise_fact_exact(chunks, fact, CONTEXT)
    assert chunk_id == 3
    assert confidence == 1.0


def test_exact_oracle_returns_miss_for_non_substring():
    chunks = cs.chunk_context(CONTEXT, words, target_tokens=20)
    assert cs.localise_fact_exact(chunks, "not present verbatim", CONTEXT) == (-1, 0.0)


def test_anchor_is_the_opening_sentence_and_is_suppressed_when_chunk_zero_is_kept():
    chunks = cs.chunk_context(CONTEXT, words, target_tokens=20)
    anchor = cs.anchor_text(CONTEXT)
    assert anchor.startswith("Sentence 0")
    with_anchor = cs.assemble(chunks, [3], anchor)
    assert with_anchor.startswith(anchor)
    without = cs.assemble(chunks, [0, 1], anchor)
    assert without.count(anchor) == 1, "anchor must not be duplicated into chunk 0"


def test_paired_bootstrap_recovers_a_known_delta_and_brackets_it():
    a = [True] * 40 + [False] * 60
    b = [True] * 60 + [False] * 40
    result = paired_bootstrap_delta(a, b, resamples=2000, seed=1)
    assert result["delta_points"] == pytest.approx(20.0, abs=1e-6)
    assert result["ci_low_points"] <= 20.0 <= result["ci_high_points"]


def test_paired_bootstrap_on_identical_inputs_has_zero_width():
    scores = [True, False] * 25
    result = paired_bootstrap_delta(scores, scores, resamples=500, seed=1)
    assert result["delta_points"] == 0.0
    assert result["ci_low_points"] == result["ci_high_points"] == 0.0
    assert result["ci_excludes_zero"] is False


def test_paired_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap_delta([True], [True, False])


def test_mcnemar_counts_only_discordant_pairs():
    a = [True, True, False, False]
    b = [True, False, True, True]
    result = mcnemar_exact(a, b)
    assert result["a_only_correct"] == 1
    assert result["b_only_correct"] == 2
    assert result["discordant"] == 3


def test_relative_error_reduction_matches_the_gate_definition():
    assert relative_error_reduction(0.60, 0.62) == pytest.approx(0.05)
    assert relative_error_reduction(1.0, 1.0) == 0.0


# --- provenance ------------------------------------------------------------


def _tiny_model():
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(64, 32)
            self.b = torch.nn.Embedding(100, 16)

    return Tiny().half()


def test_state_hash_is_deterministic_and_chunk_size_invariant():
    from slcmi.provenance import state_hash

    model = _tiny_model()
    baseline = state_hash(model)
    assert state_hash(model) == baseline
    assert state_hash(model, chunk_elements=7) == baseline, (
        "the digest must not depend on how the stream was chunked"
    )


def test_state_hash_detects_a_small_half_precision_change():
    import torch

    from slcmi.provenance import state_hash

    model = _tiny_model()
    baseline = state_hash(model)
    with torch.no_grad():
        model.a.weight[0, 0] += torch.tensor(0.01, dtype=torch.float16)
    assert state_hash(model) != baseline
