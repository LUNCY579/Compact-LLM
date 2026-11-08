"""Phase 2 benchmark: construction invariants, matched positional variants, determinism.

The tests that need the Qwen3 tokenizer are marked ``model`` and skip cleanly when the
checkpoint is not on disk; everything structural runs without it.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from rineya_psa.benchmark import assemble, families, filler
from rineya_psa.benchmark import generate as generate_mod
from rineya_psa.benchmark import tokenize as tokenize_mod
from rineya_psa.paths import REPO_ROOT, model_spec
from rineya_psa.types import (
    FAMILIES,
    FORMATS,
    BenchmarkItem,
    matches_gold,
    normalize_answer,
    split_for_scenario,
)

SHOWCASE = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_prompts.jsonl"

SMOKE_CONFIG: dict[str, Any] = {
    "name": "benchmark_test",
    "seed": 4242,
    "scale": 1.0,
    "tokenizer": "qwen3",
    "sentences_per_chunk": 3,
    "token_tolerance": 0.02,
    "length_tokens": {"2k": 2048, "4k": 4096},
    "positions": [0.02, 0.50, 0.98],
    "grids": [
        {
            "name": "core",
            "families": ["exact_retrieval", "semantic_single_hop", "state_constraint"],
            "lengths": ["2k"],
            "formats": ["natural"],
            "local_positions": ["middle"],
            "positions": "full",
            "scenarios": 2,
        },
        {
            "name": "hops",
            "families": ["multi_hop"],
            "lengths": ["2k"],
            "formats": ["natural"],
            "local_positions": ["middle"],
            "positions": "full",
            "scenarios": 2,
        },
        {
            "name": "fmt",
            "families": ["exact_retrieval"],
            "lengths": ["2k"],
            "formats": ["shuffled", "list", "json"],
            "local_positions": ["last"],
            "positions": [0.02, 0.98],
            "scenarios": 1,
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    if not model_spec("qwen3").path.exists():
        pytest.skip("Qwen3-0.6B checkpoint not present")
    return tokenize_mod.get_tokenizer("qwen3")


@pytest.fixture(scope="module")
def counter(tokenizer):
    return tokenize_mod.prompt_token_counter(tokenizer)


@pytest.fixture(scope="module")
def smoke_builds(tmp_path_factory, tokenizer) -> tuple[Path, Path]:
    """Build twice into the same root, snapshotting the first run in between.

    Comparing two *different* output roots would not be a fair determinism test: the
    manifest legitimately records where it was written. Re-running into the same root
    is what the brief means by "running twice must byte-match".
    """
    root = tmp_path_factory.mktemp("bench_run")
    snapshot = tmp_path_factory.mktemp("bench_snapshot") / "first"
    generate_mod.build(SMOKE_CONFIG, out_root=root)
    shutil.copytree(root, snapshot)
    generate_mod.build(SMOKE_CONFIG, out_root=root)
    return root, snapshot


def _items(root: Path) -> list[BenchmarkItem]:
    path = root / "generated" / generate_mod.JSONL_NAME
    with path.open("r", encoding="utf-8") as fh:
        return [BenchmarkItem.from_json(json.loads(line)) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Filler
# ---------------------------------------------------------------------------


def test_filler_is_deterministic():
    a = filler.generate_chunks(20, seed=99)
    b = filler.generate_chunks(20, seed=99)
    c = filler.generate_chunks(20, seed=100)
    assert [x.sentences for x in a] == [x.sentences for x in b]
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert [x.sentences for x in a] != [x.sentences for x in c]


def test_filler_contains_no_digits_and_no_proper_nouns():
    for chunk in filler.generate_chunks(60, seed=7):
        for sentence in chunk.sentences:
            assert not filler.has_digits(sentence), sentence
            # Only the leading word may be capitalised; a mid-sentence capital would
            # mean an invented proper noun that could shadow an evidence entity.
            for word in sentence.split()[1:]:
                assert not word[:1].isupper(), sentence


def test_filler_sentence_count_is_stable_except_for_the_ballast_chunk():
    chunks = filler.generate_chunks(15, seed=3, sentences_per_chunk=4, ballast_sentences=6)
    assert [len(c.sentences) for c in chunks[:-1]] == [4] * 14
    assert len(chunks[-1].sentences) == 6
    plain = filler.generate_chunks(15, seed=3, sentences_per_chunk=4)
    assert {len(c.sentences) for c in plain} == {4}


@pytest.mark.parametrize("bad", [2, 7])
def test_filler_rejects_out_of_range_paragraph_sizes(bad):
    with pytest.raises(ValueError):
        filler.generate_chunks(4, seed=1, sentences_per_chunk=bad)


def test_filler_vocabulary_is_disjoint_from_family_entities():
    overlap = filler.filler_vocabulary() & families.entity_tokens()
    assert overlap == frozenset(), f"filler can shadow evidence entities: {sorted(overlap)}"


@pytest.mark.parametrize("family", FAMILIES)
def test_filler_is_inert_against_every_family_draw(family):
    for seed in range(12):
        example = families.build_family(family, random.Random(seed))
        reserved = filler.reserved_tokens(example.meta["reserved_terms"])
        assert reserved, "an example with no reserved terms would make the check vacuous"
        for chunk in filler.generate_chunks(25, seed=seed, reserved=reserved):
            assert filler.chunk_is_inert(chunk, reserved), filler.collisions(chunk, reserved)


def test_reserved_token_check_actually_bites():
    reserved = filler.reserved_tokens(["gradient reading"])
    poisoned = filler.Chunk(
        chunk_id="x", domain="survey", sentences=("The gradient reading was fine.",),
        kinds=(filler.KIND_FILLER,),
    )
    assert not filler.chunk_is_inert(poisoned, reserved)
    assert filler.collisions(poisoned, reserved) == frozenset({"gradient", "reading"})


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_answers_are_short_exact_single_lines(family):
    for seed in range(25):
        ex = families.build_family(family, random.Random(seed))
        assert "\n" not in ex.answer and ex.answer.strip()
        assert len(ex.answer) <= 64
        assert matches_gold(ex.answer, ex.answer, ex.aliases)
        for alias in ex.aliases:
            assert matches_gold(alias, ex.answer, ex.aliases)
            assert normalize_answer(alias) != normalize_answer("")
        assert not matches_gold("UNKNOWN", ex.answer, ex.aliases)


@pytest.mark.parametrize("family", FAMILIES)
def test_evidence_count_matches_the_declared_arity(family):
    for seed in range(10):
        ex = families.build_family(family, random.Random(seed))
        assert len(ex.evidence_sentences) == families.n_evidence(family)
        assert len(ex.distractor_sentences) >= 6
        assert ex.meta["reserved_terms"]


def test_exact_retrieval_answer_appears_only_in_its_evidence():
    for seed in range(30):
        ex = families.build_family("exact_retrieval", random.Random(seed))
        assert ex.answer in ex.evidence_sentences[0]
        assert not any(ex.answer in d for d in ex.distractor_sentences)


def test_multi_hop_needs_both_facts():
    for seed in range(30):
        ex = families.build_family("multi_hop", random.Random(seed))
        hop_custody, hop_item = ex.evidence_sentences
        role, obj = ex.meta["role"], ex.meta["object"]
        assert role in hop_custody and obj not in hop_custody
        assert obj in hop_item and role not in hop_item
        # The shared reference is the only bridge between them.
        assert ex.meta["reference"] in hop_custody and ex.meta["reference"] in hop_item


def test_state_constraint_has_three_plus_updates_and_a_live_correction():
    seen = set()
    for seed in range(40):
        ex = families.build_family("state_constraint", random.Random(seed))
        seen.add(ex.meta["subtype"])
        assert ex.meta["n_updates"] >= 3
        decisive = ex.meta["decisive_index"]
        correction = ex.evidence_sentences[decisive]
        overridden = ex.meta["correction_overrides"]
        assert overridden in correction, correction
        # The overridden stamp belongs to an earlier update, so the correction is a
        # genuine later-overrides-earlier event and not a restatement.
        assert any(overridden in s for i, s in enumerate(ex.evidence_sentences) if i != decisive)
        if ex.meta["subtype"] == "inventory_count":
            assert ex.answer != ex.meta["naive_answer"]
    assert seen == {"custody_chain", "inventory_count"}


def test_unknown_family_is_rejected():
    with pytest.raises(KeyError):
        families.build_family("not_a_family", random.Random(0))


# ---------------------------------------------------------------------------
# Assembly, position control, formats
# ---------------------------------------------------------------------------


def _bank(family: str = "semantic_single_hop", *, fmt="natural", local="middle", n=40):
    example = families.build_family(family, random.Random(21))
    m = len(example.evidence_sentences)
    canonical = tuple((k + 0.5) / m for k in range(m))
    return example, assemble.build_bank(
        example,
        n_chunks=n,
        fmt=fmt,
        local_position=local,
        seed=1234,
        canonical_targets=canonical,
    )


@pytest.mark.parametrize("target", [0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98])
def test_place_evidence_lands_near_its_target(target):
    chunks = filler.generate_chunks(60, seed=5)
    _, placement = assemble.place_evidence(chunks, "The marker was set.", target)
    assert abs(placement.achieved_char_p - target) < 0.03
    assert placement.target_p == target


def test_place_evidence_will_not_double_book_a_paragraph():
    chunks = filler.generate_chunks(8, seed=5)
    chunks, first = assemble.place_evidence(chunks, "Fact one.", 0.5, slot=0)
    chunks, second = assemble.place_evidence(chunks, "Fact two.", 0.5, slot=1)
    assert first.host_id != second.host_id


@pytest.mark.parametrize("fmt", FORMATS)
def test_reordering_preserves_the_chunk_multiset(fmt):
    _, bank = _bank(fmt=fmt)
    baseline = Counter(assemble.render_chunk(c, fmt) for c in bank.chunks)
    for target in (0.02, 0.33, 0.5, 0.83, 0.98):
        ordered = assemble.reorder_for_targets(bank.chunks, bank.host_ids, (target,), fmt=fmt)
        assert Counter(assemble.render_chunk(c, fmt) for c in ordered) == baseline


@pytest.mark.parametrize("family", FAMILIES)
def test_matched_construction_filler_multiset_is_identical_across_positions(family):
    """The brief's strict-matching rule, stated as an assertion.

    Same scenario, every positional variant: identical filler multiset, identical
    distractor set, identical answer, identical separators and sentence budget. Only
    the order of the chunk list may differ.
    """
    example, bank = _bank(family)
    m = len(example.evidence_sentences)
    grid = (0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98)
    target_sets = generate_mod.position_sets_for_example(family, example, grid)

    reference_filler: Counter[str] | None = None
    reference_all: Counter[str] | None = None
    fingerprints: set[str] = set()
    orders: set[str] = set()
    for targets in target_sets:
        layout = assemble.render_variant(bank, targets)
        assert len(layout.chunks) == bank.n_chunks
        filler_multiset = Counter(s for c in layout.chunks for s in c.filler_sentences)
        all_multiset = Counter(s for c in layout.chunks for s in c.sentences)
        if reference_filler is None:
            reference_filler, reference_all = filler_multiset, all_multiset
        assert filler_multiset == reference_filler
        assert all_multiset == reference_all
        fingerprints.add(filler.filler_fingerprint(layout.chunks))
        orders.add(layout.order_signature)
        assert len(layout.evidence_char_spans) == m
        for (start, end), sentence in zip(
            layout.evidence_char_spans, example.evidence_sentences, strict=True
        ):
            offset = len(assemble.CONTEXT_HEADER)
            assert layout.user[start:end] == sentence
            assert start >= offset
        for distractor in example.distractor_sentences[: len(layout.distractor_char_positions)]:
            assert distractor in layout.context

    assert len(fingerprints) == 1, "filler was regenerated between positional variants"
    assert len(orders) > 1, "re-ordering produced no positional variation at all"


def test_positional_variants_track_their_targets():
    _, bank = _bank("exact_retrieval", n=80)
    achieved = []
    for target in (0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98):
        layout = assemble.render_variant(bank, (target,))
        achieved.append(layout.achieved_char_positions[0])
    assert achieved == sorted(achieved)
    assert achieved[0] < 0.05 and achieved[-1] > 0.93


def test_multi_hop_supports_all_six_position_pairs():
    example, bank = _bank("multi_hop", n=80)
    assert len(families.MULTI_HOP_POSITION_PAIRS) == 6
    seen = set()
    for pair in families.MULTI_HOP_POSITION_PAIRS:
        targets = families.multi_hop_targets(pair)
        layout = assemble.render_variant(bank, targets)
        a, b = layout.achieved_char_positions
        assert a <= b + 0.06, (pair, a, b)
        for target, got in zip(targets, (a, b), strict=True):
            assert abs(got - target) < 0.09, (pair, target, got)
        seen.add(pair)
    assert seen == set(families.MULTI_HOP_POSITION_PAIRS)
    assert len(example.evidence_sentences) == 2


@pytest.mark.parametrize("local", ["first", "middle", "last"])
def test_local_position_is_orthogonal_to_global_position(local):
    _, bank = _bank("exact_retrieval", local=local, n=60)
    layout = assemble.render_variant(bank, (0.50,))
    host = next(c for c in layout.chunks if c.carries_evidence)
    index = host.sentence_index_of_kind("evidence:0")
    if local == "first":
        assert index == 0
    elif local == "last":
        assert index == len(host.sentences) - 1
    else:
        assert 0 < index < len(host.sentences) - 1
    assert abs(layout.achieved_char_positions[0] - 0.50) < 0.05


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_carries_the_same_facts(fmt):
    example, bank = _bank("multi_hop", fmt=fmt)
    layout = assemble.render_variant(bank, (0.25, 0.75))
    for sentence in example.evidence_sentences:
        assert sentence in layout.context
    for distractor in example.distractor_sentences[: len(layout.distractor_char_positions)]:
        assert distractor in layout.context
    assert example.question in layout.user
    assert layout.user.startswith(assemble.CONTEXT_HEADER)


def test_list_format_is_neutral_bullets():
    _, bank = _bank(fmt="list")
    layout = assemble.render_variant(bank, (0.5,))
    for block in layout.context.split(assemble.SEPARATOR):
        for line in block.split("\n"):
            assert line.startswith("- ")


def test_json_format_is_one_record_per_paragraph():
    _, bank = _bank(fmt="json")
    layout = assemble.render_variant(bank, (0.5,))
    blocks = layout.context.split(assemble.SEPARATOR)
    assert len(blocks) == bank.n_chunks
    for block in blocks:
        record = json.loads(block)
        assert set(record) == {"record", "source", "entries"}
        assert isinstance(record["entries"], list) and record["entries"]


def test_shuffled_format_moves_filler_but_pins_the_evidence():
    example = families.build_family("exact_retrieval", random.Random(21))
    natural = assemble.build_bank(
        example, n_chunks=40, fmt="natural", local_position="middle", seed=1234
    )
    shuffled = assemble.build_bank(
        example, n_chunks=40, fmt="shuffled", local_position="middle", seed=1234
    )
    nat = assemble.render_variant(natural, (0.5,))
    shu = assemble.render_variant(shuffled, (0.5,))
    assert nat.context != shu.context
    assert Counter(c.sentences for c in nat.chunks) == Counter(c.sentences for c in shu.chunks)
    host = next(c for c in shu.chunks if c.carries_evidence)
    text, spans = assemble.render_chunk_with_offsets(host, "shuffled")
    index = host.sentence_index_of_kind("evidence:0")
    start, end = spans[index]
    assert text[start:end] == example.evidence_sentences[0]
    # "middle" splits the paragraph as it was before insertion, so a 3-sentence
    # paragraph puts the evidence second of four.
    assert index == (len(host.sentences) - 1) // 2


def test_system_prompt_matches_the_project_convention():
    if not SHOWCASE.exists():
        pytest.skip("showcase prompt file not present")
    with SHOWCASE.open("r", encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    assert row["messages"][0]["content"] == assemble.SYSTEM_PROMPT
    user = assemble.build_user_prompt("BODY", "Q?")
    assert user == "CONTEXT\nBODY\n\nQUESTION\nQ?\n\nANSWER"
    assert row["prompt"].startswith("CONTEXT\n") and row["prompt"].endswith("\n\nANSWER")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


@pytest.mark.model
def test_qwen3_tokenizer_exposes_offsets(tokenizer):
    assert tokenizer.is_fast
    assert tokenize_mod.offsets_available(tokenizer)
    assert tokenize_mod.span_method(tokenizer) == "offsets"


@pytest.mark.model
def test_chat_template_embeds_the_user_turn_verbatim(tokenizer):
    user = assemble.build_user_prompt("alpha beta", "who?")
    prompt = tokenize_mod.build_chat_prompt(assemble.SYSTEM_PROMPT, user, tokenizer)
    assert user in prompt
    assert assemble.SYSTEM_PROMPT in prompt
    assert prompt.rstrip().endswith("</think>") or prompt.endswith("\n")


@pytest.mark.model
@pytest.mark.parametrize("family", FAMILIES)
def test_tokenize_item_measures_the_evidence_span(family, tokenizer, counter):
    example = families.build_family(family, random.Random(8))
    m = len(example.evidence_sentences)
    canonical = tuple((k + 0.5) / m for k in range(m))
    bank = assemble.fit_bank(
        example,
        token_target=2048,
        tolerance=0.02,
        fmt="natural",
        local_position="middle",
        seed=77,
        count_prompt_tokens=counter,
        canonical_targets=canonical,
    )
    layout = assemble.render_variant(bank, canonical)
    item = BenchmarkItem(
        item_id="probe",
        scenario_id="probe-scenario",
        family=family,
        fmt="natural",
        length_bucket="2k",
        target_positions=canonical,
        local_position="middle",
        split="calibration",
        system=assemble.SYSTEM_PROMPT,
        user=layout.user,
        answer=example.answer,
        answer_aliases=example.aliases,
        distractors=example.distractor_sentences,
        evidence_sentences=example.evidence_sentences,
        seed=77,
        meta={"evidence_char_spans": [list(s) for s in layout.evidence_char_spans]},
    )
    tok_item = tokenize_mod.tokenize_item(item, tokenizer)
    assert tok_item.n_tokens > 1500
    assert len(tok_item.evidence_spans) == m
    for (start, end), sentence in zip(
        tok_item.evidence_spans, example.evidence_sentences, strict=True
    ):
        decoded = tokenizer.decode(tok_item.input_ids[start:end])
        assert sentence.strip(".") in decoded
    for measured, char_p in zip(
        tok_item.actual_positions, layout.achieved_char_positions, strict=True
    ):
        assert 0.0 <= measured <= 1.0
        assert abs(measured - char_p) < 0.06
    assert tok_item.answer_ids


@pytest.mark.model
def test_measured_token_position_is_monotone_in_the_target(tokenizer, counter):
    example = families.build_family("exact_retrieval", random.Random(2))
    bank = assemble.fit_bank(
        example,
        token_target=2048,
        tolerance=0.02,
        fmt="natural",
        local_position="middle",
        seed=5,
        count_prompt_tokens=counter,
        canonical_targets=(0.02,),
    )
    measured = []
    for target in (0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98):
        layout = assemble.render_variant(bank, (target,))
        item = BenchmarkItem(
            item_id=f"probe-{target}",
            scenario_id="probe",
            family="exact_retrieval",
            fmt="natural",
            length_bucket="2k",
            target_positions=(target,),
            local_position="middle",
            split="calibration",
            system=assemble.SYSTEM_PROMPT,
            user=layout.user,
            answer=example.answer,
            answer_aliases=(),
            distractors=example.distractor_sentences,
            evidence_sentences=example.evidence_sentences,
            seed=5,
            meta={"evidence_char_spans": [list(s) for s in layout.evidence_char_spans]},
        )
        measured.append(tokenize_mod.tokenize_item(item, tokenizer).actual_positions[0])
    assert measured == sorted(measured)
    assert measured[0] < 0.10 < 0.90 < measured[-1]


@pytest.mark.model
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("bucket,target", [("2k", 2048), ("4k", 4096)])
def test_token_budget_lands_inside_two_percent(fmt, bucket, target, counter):
    example = families.build_family("state_constraint", random.Random(13))
    m = len(example.evidence_sentences)
    canonical = tuple((k + 0.5) / m for k in range(m))
    bank = assemble.fit_bank(
        example,
        token_target=target,
        tolerance=0.02,
        fmt=fmt,
        local_position="middle",
        seed=31,
        count_prompt_tokens=counter,
        canonical_targets=canonical,
    )
    assert abs(bank.token_error) <= 0.02, (bucket, fmt, bank.n_prompt_tokens)
    # Every positional variant must stay in budget too, not just the fitted one.
    for target_p in (0.02, 0.5, 0.98):
        targets = tuple(target_p if k == m - 1 else canonical[k] for k in range(m))
        layout = assemble.render_variant(bank, targets)
        n = counter(assemble.SYSTEM_PROMPT, layout.user)
        assert abs(n - target) / target <= 0.02


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------


@pytest.mark.model
def test_smoke_build_writes_the_expected_artifacts(smoke_builds):
    root, _ = smoke_builds
    jsonl = root / "generated" / generate_mod.JSONL_NAME
    manifest_path = root / generate_mod.MANIFEST_NAME
    assert jsonl.exists() and manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    items = _items(root)
    assert len(items) == manifest["n_items"] == 36
    assert manifest["token_out_of_tolerance"] == 0
    assert manifest["span_localisation"] == "offsets"
    assert {i.family for i in items} == set(FAMILIES)
    assert {i.fmt for i in items} == {"natural", "shuffled", "list", "json"}

    for split, sub in generate_mod.SPLIT_DIRS.items():
        path = root / sub / generate_mod.JSONL_NAME
        assert path.exists()
        subset = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert all(row["split"] == split for row in subset)
        assert len(subset) == manifest["splits"][split]["n_items"]
    total = sum(manifest["splits"][s]["n_items"] for s in generate_mod.SPLIT_DIRS)
    assert total == manifest["n_items"]

    counted = sum(row["n_items"] for row in manifest["counts_by_family_position_length_split"])
    assert counted == manifest["n_items"]
    for row in manifest["counts_by_family_position_length_split"]:
        assert set(row) == {"family", "position", "length_bucket", "split", "n_items"}


@pytest.mark.model
def test_splits_come_only_from_split_for_scenario(smoke_builds):
    root, _ = smoke_builds
    manifest = json.loads((root / generate_mod.MANIFEST_NAME).read_text(encoding="utf-8"))
    for scenario_id, split in manifest["scenario_split"].items():
        assert split == split_for_scenario(scenario_id)
    for item in _items(root):
        assert item.split == split_for_scenario(item.scenario_id)


@pytest.mark.model
def test_build_is_byte_identical_on_a_second_run(smoke_builds):
    root, snapshot = smoke_builds
    produced = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    assert produced, "build produced no files"
    assert produced == sorted(p.relative_to(snapshot) for p in snapshot.rglob("*") if p.is_file())
    for rel in produced:
        assert (root / rel).read_bytes() == (snapshot / rel).read_bytes(), rel


@pytest.mark.model
def test_generated_items_keep_the_matched_construction(smoke_builds):
    """Grouped by scenario and condition, every position shares one filler bank."""
    root, _ = smoke_builds
    groups: dict[tuple[str, str, str, str], set[str]] = {}
    order_sigs: dict[tuple[str, str, str, str], set[str]] = {}
    for item in _items(root):
        key = (item.scenario_id, item.fmt, item.local_position, item.length_bucket)
        groups.setdefault(key, set()).add(item.meta["filler_fingerprint"])
        order_sigs.setdefault(key, set()).add(item.meta["order_signature"])
        assert item.system == assemble.SYSTEM_PROMPT
        assert item.meta["separator"] == "\n\n"
        assert item.meta["token_in_tolerance"]
    assert groups
    for key, fingerprints in groups.items():
        assert len(fingerprints) == 1, key
    assert any(len(sigs) > 1 for sigs in order_sigs.values())


@pytest.mark.model
def test_generated_items_round_trip_and_locate_their_evidence(smoke_builds, tokenizer):
    root, _ = smoke_builds
    for item in _items(root)[:12]:
        restored = BenchmarkItem.from_json(item.to_json())
        assert restored == item
        tok_item = tokenize_mod.tokenize_item(item, tokenizer)
        assert tok_item.n_tokens == item.meta["n_prompt_tokens"]
        assert len(tok_item.actual_positions) == len(item.target_positions)
        primary = item.meta["decisive_index"]
        assert abs(tok_item.actual_positions[primary] - item.meta["primary_target"]) < 0.12


@pytest.mark.model
def test_shipped_config_parses_and_describes_the_screening_grid():
    cfg = generate_mod.load_config(generate_mod.DEFAULT_CONFIG)
    assert cfg["positions"] == [0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98]
    assert cfg["screening_lengths"] == ["2k", "4k", "8k"]
    assert set(cfg["optional_lengths"]) == {"16k", "32k"}
    assert cfg["token_tolerance"] == 0.02

    total = 0
    for cell in cfg["grids"]:
        positions = (
            cfg["positions"] if cell["positions"] == "full" else cell["positions"]
        )
        for family in cell["families"]:
            n_pos = 6 if family == "multi_hop" else len(positions)
            total += (
                cell["scenarios"]
                * len(cell["lengths"])
                * len(cell["formats"])
                * len(cell["local_positions"])
                * n_pos
            )
    assert 4000 <= total <= 6000, total


@pytest.mark.model
@pytest.mark.parametrize("bucket,target", [("16k", 16384), ("32k", 32768)])
def test_long_buckets_are_supported_even_though_they_are_not_swept(bucket, target, counter):
    """16k and 32k are emitted only on request, but the machinery must already work."""
    cfg = generate_mod.load_config(generate_mod.DEFAULT_CONFIG)
    assert target == cfg["length_tokens"][bucket]
    assert all(bucket not in cell["lengths"] for cell in cfg["grids"])

    example = families.build_family("multi_hop", random.Random(6))
    bank = assemble.fit_bank(
        example,
        token_target=target,
        tolerance=0.02,
        fmt="natural",
        local_position="middle",
        seed=17,
        count_prompt_tokens=counter,
        canonical_targets=(0.02, 0.98),
    )
    assert abs(bank.token_error) <= 0.02, (bucket, bank.n_prompt_tokens)
    layout = assemble.render_variant(bank, (0.02, 0.98))
    assert layout.achieved_char_positions[0] < 0.05
    assert layout.achieved_char_positions[1] > 0.95


@pytest.mark.model
def test_scale_dials_the_bank_down(tmp_path, tokenizer):
    small = dict(SMOKE_CONFIG, name="benchmark_scaled", scale=0.5)
    report = generate_mod.build(small, out_root=tmp_path)
    assert 0 < report.n_items < 36
    assert report.n_scenarios < 6
    assert report.token_out_of_tolerance == 0
