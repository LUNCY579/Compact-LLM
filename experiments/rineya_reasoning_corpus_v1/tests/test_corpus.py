"""Validation for RTG-1. Software checks on synthetic and built artifacts.

Passing these is not evidence that the corpus teaches anything. It is evidence that the
corpus is what it claims to be: correctly solved, disjoint from the evaluation bank,
isolated across splits, balanced, within the token budget, and reproducible.
"""
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from experiments.multi_reasoning_scout_v1 import bank as scout
from experiments.multi_reasoning_scout_v1.core import RECORDS as SCOUT_RECORDS, decode
from experiments.multi_reasoning_scout_v2 import prompts as mrs2_prompts
from experiments.rineya_reasoning_corpus_v1 import corpus, families as F, sampling
from experiments.rineya_reasoning_corpus_v1.build import OUT, TRAINED_ROLE

SPLITS = ("train", "dev", "holdout")


@pytest.fixture(scope="module")
def scout_worlds():
    return [decode(line) for line
            in (SCOUT_RECORDS / "bank/worlds.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def scout_gold():
    return {decode(line)["id"]: decode(line) for line
            in (SCOUT_RECORDS / "bank/gold.jsonl").read_text(encoding="utf-8").splitlines()}


@pytest.fixture(scope="module")
def built():
    if not (OUT / "MANIFEST.json").exists():
        pytest.skip("corpus not built")
    data, gold, sft, worlds = [], [], [], []
    for split in SPLITS:
        data += [json.loads(l) for l in (OUT / f"data/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        gold += [json.loads(l) for l in (OUT / f"gold/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        sft += [json.loads(l) for l in (OUT / f"sft/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        worlds += [json.loads(l) for l in (OUT / f"worlds/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    return data, gold, sft, worlds, manifest


# ------------------------------------------- cross-validation against the eval bank

def test_solvers_reproduce_the_scout_solvers_and_gold(scout_worlds, scout_gold):
    """The strongest correctness evidence available: four independent implementations.

    RTG-1's forward and backward solvers, the scout's own forward and exhaustive solvers,
    and the frozen gold labels must all agree on all 120 evaluation worlds.
    """
    for entry in scout_worlds:
        world = entry["world"]
        expected = scout_gold[entry["id"]]["answer"]
        assert scout.solve_forward(world) == expected
        assert scout.solve_exhaustive(world) == expected
        assert F.solve_forward(world) == expected
        assert F.solve_backward(world) == expected


def test_renderer_is_byte_identical_to_the_scout(scout_worlds):
    """The `match` tier must present the eval surface exactly, not an approximation."""
    for entry in scout_worlds:
        world = entry["world"]
        assert F.render(world) == scout.render(world)


def test_own_solvers_agree_on_generated_worlds():
    rng = random.Random(99)
    checked = 0
    for family in F.FAMILIES:
        for tier in sampling.TIERS[family]:
            for _ in range(200):
                world = sampling.sample(rng, family, tier)
                forward, backward = F.solve_forward(world), F.solve_backward(world)
                assert forward == backward, (family, tier, world)
                checked += 1
    assert checked > 2000


# --------------------------------------------------------------- fingerprint hygiene

def test_fingerprint_ignores_entity_names():
    rng = random.Random(3)
    for family in F.FAMILIES:
        world = sampling.sample(rng, family, "match")
        renamed = dict(world, names=[n[::-1] for n in world["names"]])
        assert F.fingerprint(world) == F.fingerprint(renamed)


def test_evidence_fingerprint_is_invariant_to_node_relabelling():
    """Node integers are arbitrary labels. Without this, an isomorphic structure could
    receive a different fingerprint and slip across a train/holdout boundary."""
    rng = random.Random(11)
    for _ in range(200):
        world = sampling.sample(rng, "evidence_chain", "match")
        nodes = sorted({a for a, _ in world["links"]} | {b for _, b in world["links"]}
                       | {f[0] for f in world["facts"]} | {world["query"]})
        shuffled = list(nodes)
        rng.shuffle(shuffled)
        mapping = dict(zip(nodes, shuffled))
        relabelled = dict(
            world,
            links=[[mapping[a], mapping[b]] for a, b in world["links"]],
            facts=[[mapping[n], t, v] for n, t, v in world["facts"]],
            query=mapping[world["query"]])
        assert F.fingerprint(world) == F.fingerprint(relabelled)
        assert F.solve(world) == F.solve(relabelled)


def test_fingerprint_separates_different_structures():
    rng = random.Random(5)
    seen = {}
    for _ in range(3000):
        world = sampling.sample(rng, "register_machine", "match")
        fingerprint = F.fingerprint(world)
        answer = F.solve(world)
        # A fingerprint must never carry two different answers.
        assert seen.setdefault(fingerprint, answer) == answer


def test_schedule_fingerprint_erases_an_unrecorded_row():
    base = {"family": "prerequisite_schedule", "deps": [[], [0], [1], [2]], "query": 3,
            "missing": 2, "names": ["q" + c * 7 for c in "abcd"]}
    other = dict(base, deps=[[], [0], [0, 1], [2]])
    # deps[2] is unrecorded, so it cannot affect the rendered instance or the answer.
    assert F.fingerprint(base) == F.fingerprint(other)
    assert F.render(base) == F.render(other)
    assert F.solve(base) == F.solve(other)


# ----------------------------------------------------------------- namespace disjoint

def test_name_namespace_cannot_collide_with_the_bank():
    scout_pattern = re.compile(r"\An[abcdefghjkmnpqrstuvwxyz]{6}\Z")
    rng = random.Random(13)
    for name in F.names(rng, 200):
        assert name.startswith(F.NAME_PREFIX) and len(name) == 1 + F.NAME_BODY
        assert not scout_pattern.match(name)


# ------------------------------------------------------ measured capacity is honest

def test_four_job_schedule_space_is_exactly_480():
    subsets = {i: [list(c) for r in range(i + 1) for c in itertools.combinations(range(i), r)]
               for i in range(4)}
    seen = set()
    for combo in itertools.product(*[subsets[i] for i in range(4)]):
        for query in range(4):
            for missing in (None, 1, 2, 3):
                world = {"family": "prerequisite_schedule", "deps": [list(c) for c in combo],
                         "query": query, "missing": missing,
                         "names": ["q" + c * 7 for c in "abcd"]}
                seen.add(F.fingerprint(world))
    assert len(seen) == 480


def test_only_eight_four_job_schedules_answer_three_and_the_bank_owns_six(scout_worlds, scout_gold):
    subsets = {i: [list(c) for r in range(i + 1) for c in itertools.combinations(range(i), r)]
               for i in range(4)}
    threes = set()
    for combo in itertools.product(*[subsets[i] for i in range(4)]):
        for query in range(4):
            for missing in (None, 1, 2, 3):
                world = {"family": "prerequisite_schedule", "deps": [list(c) for c in combo],
                         "query": query, "missing": missing,
                         "names": ["q" + c * 7 for c in "abcd"]}
                if F.solve(world) == "3":
                    threes.add(F.fingerprint(world))
    assert len(threes) == 8
    used = {F.fingerprint(e["world"]) for e in scout_worlds
            if e["world"]["family"] == "prerequisite_schedule"
            and scout_gold[e["id"]]["answer"] == "3"}
    assert len(used) == 6
    assert len(threes - used) == 2, "exactly two remain available to a disjoint corpus"


def test_add_xor_sequences_collapse_to_eight_maps():
    """Disclosed limitation, asserted so it cannot be quietly forgotten."""
    generators = [tuple((x + k) % 4 for x in range(4)) for k in range(4)]
    generators += [tuple(x ^ k for x in range(4)) for k in range(4)]
    seen, frontier = {tuple(range(4))}, {tuple(range(4))}
    while frontier:
        nxt = set()
        for current in frontier:
            for gen in generators:
                composed = tuple(gen[current[x]] for x in range(4))
                if composed not in seen:
                    seen.add(composed)
                    nxt.add(composed)
        frontier = nxt
    assert len(seen) == 8


def test_three_job_schedule_cannot_answer_three():
    subsets = {i: [list(c) for r in range(i + 1) for c in itertools.combinations(range(i), r)]
               for i in range(3)}
    answers = set()
    for combo in itertools.product(*[subsets[i] for i in range(3)]):
        for query in range(3):
            for missing in (None, 1, 2):
                answers.add(F.solve({"family": "prerequisite_schedule",
                                     "deps": [list(c) for c in combo], "query": query,
                                     "missing": missing,
                                     "names": ["q" + c * 7 for c in "abc"]}))
    assert "3" not in answers, "the easy schedule tier is correctly excluded"


# ------------------------------------------------------------- answerability design

def test_information_gap_is_necessary_for_an_undetermined_answer():
    rng = random.Random(23)
    for family in F.FAMILIES:
        for tier in sampling.TIERS[family]:
            for _ in range(300):
                world = sampling.sample(rng, family, tier)
                if F.solve(world) == "UNKNOWN":
                    assert F.answerability_cue(world), (family, tier, world)


def test_gap_is_not_sufficient_for_an_undetermined_answer():
    """The scout bank's defect is exactly that a gap IS sufficient in three families."""
    rng = random.Random(29)
    for family in F.FAMILIES:
        answerable_with_gap = 0
        for _ in range(4000):
            world = sampling.sample(rng, family, "match" if "match" in sampling.TIERS[family] else "wide5")
            answer = F.solve(world)
            if answer in F.ANSWERS and answer != "UNKNOWN" and F.answerability_cue(world):
                answerable_with_gap += 1
        assert answerable_with_gap > 50, family


# ----------------------------------------------------------- built-artifact invariants

def test_built_counts_and_uniqueness(built):
    data, gold, sft, worlds, manifest = built
    assert len(data) == 24000 == len(gold) == len(sft) == len(worlds)
    assert len({r["structure_id"] for r in data}) == 24000, "variant cap of 1 violated"
    assert len({r["id"] for r in data}) == 24000
    assert dict(Counter(r["family"] for r in data)) == {f: 6000 for f in F.FAMILIES}


def test_built_answers_are_uniform(built):
    _, gold, _, _, _ = built
    assert dict(Counter(g["answer"] for g in gold)) == {a: 4800 for a in F.ANSWERS}


def test_built_splits_are_structurally_isolated(built):
    data, _, _, _, _ = built
    by_structure = {}
    for row in data:
        by_structure.setdefault(row["structure_id"], set()).add(row["split"])
    assert not [k for k, v in by_structure.items() if len(v) > 1]


def test_built_is_disjoint_from_the_evaluation_bank(built, scout_worlds):
    data, _, _, worlds, _ = built
    forbidden = {F.fingerprint(e["world"]) for e in scout_worlds}
    assert not {w["fingerprint"] for w in worlds} & forbidden
    scout_names = re.compile(r"\bn[abcdefghjkmnpqrstuvwxyz]{6}\b")
    assert not [r["id"] for r in data if scout_names.search(r["problem"])]
    blinded = {corpus._blind(F.render(e["world"]), e["world"]["names"]) for e in scout_worlds}
    assert not [r for r, w in zip(data, worlds)
                if corpus._blind(r["problem"], w["world"]["names"]) in blinded]


def test_built_answers_are_recomputable_from_the_stored_world(built):
    _, gold, _, worlds, _ = built
    labels = {g["id"]: g["answer"] for g in gold}
    for entry in worlds[::7]:
        assert F.solve(entry["world"]) == labels[entry["id"]]


def test_gold_is_not_present_in_the_model_visible_files(built):
    data, gold, _, _, _ = built
    assert not any("answer" in row for row in data)
    assert not any("answerable" in row for row in data)
    assert all("answer" in row for row in gold)


def test_sft_matches_the_frozen_mrs2_serving_contract(built):
    data, gold, sft, _, manifest = built
    labels = {g["id"]: g for g in gold}
    problems = {r["id"]: r["problem"] for r in data}
    assert manifest["serving_contract"]["system_sha256"] == mrs2_prompts.manifest()["sha256"]
    for entry in sft[::11]:
        assert entry["role"] == TRAINED_ROLE
        assert [m["role"] for m in entry["messages"]] == ["system", "user"]
        assert entry["messages"][0]["content"] == mrs2_prompts.SYSTEM
        assert entry["messages"][1]["content"] == (
            mrs2_prompts.ROLES[TRAINED_ROLE] + "\n\nPROBLEM\n" + problems[entry["id"]])
        assert entry["completion"] == labels[entry["id"]]["answer"]
        assert entry["completion"] in F.ANSWERS


def test_only_the_direct_role_is_trained(built):
    """Training every role would collapse them and manufacture a 2x2 outcome."""
    _, _, sft, _, _ = built
    assert {entry["role"] for entry in sft} == {"R0"}


def test_built_respects_the_token_budget(built):
    census = json.loads((OUT / "evidence/TOKEN_CENSUS.json").read_text(encoding="utf-8"))
    assert census["violation_count"] == 0
    assert census["truncated_rows"] == 0 and census["dropped_rows"] == 0
    assert census["total_tokens"]["max"] <= census["caps"]["rendered_total"]
    assert census["prompt_tokens"]["max"] <= census["caps"]["generation_prefix_prompt"]


def test_built_shortcut_floors_are_recorded_and_bounded(built):
    audit = json.loads((OUT / "evidence/SHORTCUT_AUDIT.json").read_text(encoding="utf-8"))
    floor = audit["constant_answer_floor_dev"]
    assert floor < 0.21, "answer prior should be near uniform"
    for name, row in audit["answerability_attacks_balanced"].items():
        if row["balanced_accuracy"] is not None:
            assert row["balanced_accuracy"] <= 0.55, (name, row)
    for family, row in audit["per_family"].items():
        assert row["structural_gap_balanced_accuracy"]["balanced_accuracy"] <= 0.55, family


def test_gap_conditional_is_far_from_the_banks_perfect_leak(built):
    audit = json.loads((OUT / "evidence/SHORTCUT_AUDIT.json").read_text(encoding="utf-8"))
    for family, row in audit["gap_conditional"].items():
        # The evaluation bank sits at 1.000 for three of its four families.
        assert row["P_unknown_given_gap"] < 0.45, family
        assert row["P_unknown_given_no_gap"] == 0.0, family


def test_determinism_receipt_is_clean():
    path = OUT / "evidence/DETERMINISM.json"
    if not path.exists():
        pytest.skip("determinism not yet verified")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["deterministic"] is True and not receipt["mismatches"]


def test_semantic_equivalence_limitation_is_disclosed():
    path = OUT / "evidence/SEMANTIC_EQUIVALENCE.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    for family in ("register_machine", "counterfactual_state"):
        row = report[family]
        assert row["collapses"] is True
        assert row["semantic_isolation"] is False
        assert not row["class_answer_inconsistencies"]


def test_manifest_hashes_match_the_files_on_disk(built):
    _, _, _, _, manifest = built
    from experiments.model_quality_gpu_v1.integrity import file_hash
    for name, digest in manifest["files"].items():
        assert file_hash(OUT / name) == digest, name
