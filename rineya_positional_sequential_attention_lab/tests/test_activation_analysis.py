"""Phase 6 tests: collection, probes, patching, head statistics (brief sections 16-18).

Everything runs on a hand-made 2-layer Qwen3 with random weights, CPU, float32 -- no
checkpoint is opened and nothing touches the GPU, per the task brief.

The two tests worth pointing at:

``test_final_query_patch_at_the_last_layer_reproduces_the_clean_score``
    With a one-token answer, the first answer logit is read from the final prompt
    position, and the residual stream after the last decoder block is the only thing
    downstream of it. So patching *that* vector from the clean run must reproduce the
    clean log-probability exactly. If the patch plumbing, the alignment arithmetic or the
    teacher-forcing off-by-one is wrong anywhere, this equality breaks.

``test_capture_conserves_attention_mass``
    ``edge_fraction = 0.5`` selects every bin, so ``edge_attn`` must be 1.0. That is the
    end-to-end check that the bin grid, the capture reduction and the window masks agree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from rineya_psa.activation_analysis import collect, head_stats, patching, probes
from rineya_psa.hooks.binning import AbsoluteBinning
from rineya_psa.significance.matrices import SignificanceMatrix
from rineya_psa.types import Arch, BenchmarkItem, TokenizedItem

TINY = dict(
    vocab_size=256,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    rope_theta=1e6,
    max_position_embeddings=512,
    tie_word_embeddings=True,
)
N_TOKENS = 96
EVIDENCE_LEN = 6


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(20260828)
    net = Qwen3ForCausalLM(Qwen3Config(**TINY)).to(torch.float32).eval()
    net.config.use_cache = False
    return net


@pytest.fixture(scope="module")
def arch():
    return Arch(
        num_layers=TINY["num_hidden_layers"],
        hidden_size=TINY["hidden_size"],
        num_q_heads=TINY["num_attention_heads"],
        num_kv_heads=TINY["num_key_value_heads"],
        head_dim=TINY["head_dim"],
        vocab_size=TINY["vocab_size"],
    )


# --------------------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------------------


def _item(
    item_id: str,
    scenario_id: str,
    *,
    position: float,
    answer: str = "alpha",
    split: str = "calibration",
) -> BenchmarkItem:
    return BenchmarkItem(
        item_id=item_id,
        scenario_id=scenario_id,
        family="exact_retrieval",
        fmt="natural",
        length_bucket="2k",
        target_positions=(position,),
        local_position="middle",
        split=split,
        system="sys",
        user="user",
        answer=answer,
        answer_aliases=(),
        distractors=(),
        evidence_sentences=(),
        seed=7,
    )


def _tok(
    item_id: str,
    ids: list[int],
    span: tuple[int, int],
    position: float,
    *,
    answer_ids: tuple[int, ...] = (11,),
) -> TokenizedItem:
    return TokenizedItem(
        item_id=item_id,
        input_ids=tuple(int(i) for i in ids),
        answer_ids=answer_ids,
        evidence_spans=(span,),
        actual_positions=(position,),
    )


def _variant(
    scenario_id: str,
    offset: int,
    *,
    seed: int = 0,
    n_tokens: int = N_TOKENS,
    answer_ids: tuple[int, ...] = (11,),
) -> tuple[BenchmarkItem, TokenizedItem]:
    """One positional variant: fixed filler, a fixed evidence block moved to ``offset``.

    Both variants of a scenario therefore have the same length and the same tokens
    everywhere except where the evidence moved, which is what makes them a matched pair.
    """
    rng = np.random.default_rng(1000 + seed)
    ids = list(rng.integers(20, TINY["vocab_size"], size=n_tokens))
    evidence = list(range(200, 200 + EVIDENCE_LEN))
    ids[offset : offset + EVIDENCE_LEN] = evidence
    span = (offset, offset + EVIDENCE_LEN)
    position = (offset + EVIDENCE_LEN / 2) / n_tokens
    item_id = f"{scenario_id}-p{offset:03d}"
    return (
        _item(item_id, scenario_id, position=position),
        _tok(item_id, ids, span, position, answer_ids=answer_ids),
    )


@pytest.fixture(scope="module")
def pair_edge_middle():
    clean_item, clean_tok = _variant("sc-A", 4)
    corrupt_item, corrupt_tok = _variant("sc-A", N_TOKENS // 2)
    return patching.PatchPair(
        scenario_id="sc-A",
        clean_item=clean_item,
        clean_tok=clean_tok,
        corrupted_item=corrupt_item,
        corrupted_tok=corrupt_tok,
        clean_position=clean_tok.actual_positions[0],
        corrupted_position=corrupt_tok.actual_positions[0],
    )


# ======================================================================================
# collect.py
# ======================================================================================


def test_position_policy_budget_arithmetic():
    policy = collect.PositionPolicy(max_evidence_tokens=8, n_final_tokens=2, n_control=8)
    assert policy.max_positions == 19
    # The module docstring's 300-item figure, restated as a test so it cannot rot.
    assert collect.estimate_bytes(300, 28, 3, 19, 1024) == 300 * 28 * 3 * 19 * 1024 * 2


def test_select_positions_respects_role_priority():
    selection = collect.select_positions(
        64, [(10, 18)], collect.PositionPolicy(n_control=8), item_key="x"
    )
    by_index = dict(zip(selection.indices, selection.roles, strict=True))
    assert by_index[0] == "bos"
    assert by_index[63] == "final"
    evidence = set(selection.of_role("evidence"))
    assert evidence and evidence <= set(range(10, 18))
    # A control draw may never land on an evidence token or shadow a named landmark.
    assert not set(selection.of_role("control")) & (evidence | {0, 62, 63})
    assert selection.normalized[0] == 0.0
    assert selection.offsets_from_end[-1] == 0


def test_select_positions_is_deterministic_in_the_item_key():
    a = collect.select_positions(64, [(10, 18)], item_key="alpha")
    b = collect.select_positions(64, [(10, 18)], item_key="alpha")
    c = collect.select_positions(64, [(10, 18)], item_key="beta")
    assert a == b
    assert a.of_role("control") != c.of_role("control")


def test_collector_recovers_the_residual_identity(model):
    """``resid[l] - resid[l-1] == attn_out[l] + mlp_out[l]`` -- proves the site mapping."""
    ids = list(np.random.default_rng(0).integers(1, TINY["vocab_size"], size=40))
    indices = [0, 7, 39]
    with collect.ActivationCollector(model) as collector:
        vectors = collector.run(ids, indices)

    assert vectors[(0, "resid")].shape == (len(indices), TINY["hidden_size"])
    assert vectors[(0, "resid")].dtype == np.float16
    for layer in range(1, TINY["num_hidden_layers"]):
        delta = (
            vectors[(layer, "resid")].astype(np.float64)
            - vectors[(layer - 1, "resid")].astype(np.float64)
        )
        parts = vectors[(layer, "attn_out")].astype(np.float64) + vectors[
            (layer, "mlp_out")
        ].astype(np.float64)
        assert np.allclose(delta, parts, atol=2e-2)


def test_collector_store_dtype_is_configurable(model):
    ids = list(range(1, 33))
    with collect.ActivationCollector(
        model, layers=[0], sites=["resid"], store_dtype=np.float32
    ) as collector:
        vectors = collector.run(ids, [0, 31])
    assert vectors[(0, "resid")].dtype == np.float32


def test_collect_activations_builds_a_dataset_and_round_trips(model, tmp_path):
    pairs = [_variant("sc-D", offset, seed=offset, n_tokens=48) for offset in (4, 20)]
    policy = collect.PositionPolicy(max_evidence_tokens=3, n_final_tokens=1, n_control=2)
    dataset = collect.collect_activations(
        model, pairs, layers=[0, 2], sites=["resid", "attn_out"], policy=policy,
        model_key="tiny",
    )
    assert len(dataset) == len(dataset.rows) > 0
    assert dataset.matrix(0, "resid").shape == (len(dataset), TINY["hidden_size"])
    assert dataset.nbytes == 2 * 2 * len(dataset) * TINY["hidden_size"] * 2
    assert set(dataset.scenario_ids) == {"sc-D"}

    target = dataset.save_npz(tmp_path / "acts.npz")
    reloaded = collect.load_dataset_npz(target)
    assert reloaded.rows == dataset.rows
    assert np.array_equal(reloaded.matrix(2, "attn_out"), dataset.matrix(2, "attn_out"))

    finals = dataset.subset(dataset.role_mask(["final"]))
    assert {r.role for r in finals.rows} == {"final"}
    assert finals.describe()["n_items"] == 2


# ======================================================================================
# probes.py
# ======================================================================================


def _planted_dataset(
    n_scenarios: int = 24, n_positions: int = 5, hidden: int = 16, noise: float = 0.05
):
    """A dataset where ``normalized_position`` is linearly decodable and nothing else is."""
    rng = np.random.default_rng(11)
    rows: list[collect.ActivationRow] = []
    features: list[np.ndarray] = []
    direction = rng.normal(size=hidden)
    for s in range(n_scenarios):
        answer = f"ans{s % 4}"
        for p in np.linspace(0.05, 0.95, n_positions):
            rows.append(
                collect.ActivationRow(
                    item_id=f"i{s}-{p:.2f}",
                    scenario_id=f"sc{s}",
                    family="exact_retrieval",
                    fmt="natural",
                    length_bucket="2k",
                    split="calibration",
                    token_index=int(p * 100),
                    n_tokens=100,
                    normalized_position=float(p),
                    offset_from_end=int(100 - p * 100),
                    role="final",
                    span_index=-1,
                    evidence_position=float(p),
                    target_position=float(p),
                    answer=answer,
                )
            )
            features.append(direction * float(p) + noise * rng.normal(size=hidden))
    matrix = np.asarray(features, dtype=np.float16)
    return collect.dataset_from_arrays(rows, {(0, "resid"): matrix}, model_key="synthetic")


def test_scenario_split_never_splits_a_scenario():
    scenarios = [f"sc{i // 5}" for i in range(100)]
    train, test = probes.scenario_split(scenarios, test_fraction=0.3, seed=3)
    assert train.size and test.size
    train_ids = {scenarios[i] for i in train}
    test_ids = {scenarios[i] for i in test}
    assert not train_ids & test_ids
    assert train_ids | test_ids == set(scenarios)


def test_ridge_probe_beats_its_permutation_baseline():
    dataset = _planted_dataset()
    results = probes.run_probe_suite(
        dataset,
        tasks=(probes.ProbeTask("normalized_position", "ridge", "normalized_position"),),
        n_permutations=24,
        seed=5,
    )
    assert len(results) == 1
    result = results[0]
    assert result.status == "ok"
    assert result.score > 0.9
    assert result.permutation_mean < 0.2
    assert result.beats_chance
    assert result.n_train_scenarios and result.n_test_scenarios
    assert result.permutation_scheme == "row_shuffle_of_labels"


def test_beats_chance_is_unreachable_below_twenty_permutations():
    """The empirical p floor is 1 / (n + 1); the result carries the count that sets it."""
    dataset = _planted_dataset()
    task = probes.ProbeTask("normalized_position", "ridge", "normalized_position")
    weak = probes.run_probe_suite(dataset, tasks=(task,), n_permutations=8, seed=5)[0]
    assert weak.score > 0.9
    assert weak.permutation_resolution == pytest.approx(1 / 9)
    assert not weak.beats_chance  # the signal is fine; the resolution is not


def test_numpy_and_sklearn_probe_backends_agree(monkeypatch):
    dataset = _planted_dataset(n_scenarios=12, hidden=8)
    features = np.asarray(dataset.matrix(0, "resid"), dtype=np.float64)
    targets = dataset.column("normalized_position").astype(np.float64)
    train, test = probes.scenario_split(dataset.scenario_ids, seed=1)

    def score() -> float:
        return probes.ridge_probe(
            features[train], targets[train], features[test], targets[test], alpha=10.0
        )

    if not probes.SKLEARN_AVAILABLE:  # pragma: no cover - sklearn is installed here
        pytest.skip("sklearn is unavailable; only one backend exists on this interpreter")
    with_sklearn = score()
    monkeypatch.setattr(probes, "SKLEARN_AVAILABLE", False)
    with_numpy = score()
    assert with_sklearn == pytest.approx(with_numpy, abs=5e-3)


def test_probe_suite_skips_rather_than_reporting_a_meaningless_zero():
    dataset = _planted_dataset(n_scenarios=6, n_positions=2)
    task = probes.ProbeTask(
        "answer_identity", "logistic", "answer", roles=("final",), min_class_scenarios=99
    )
    results = probes.run_probe_suite(dataset, tasks=(task,), n_permutations=2)
    assert [r.status.startswith("skipped:") for r in results] == [True]
    assert np.isnan(results[0].score)


def test_logistic_probe_reports_a_majority_chance_level():
    dataset = _planted_dataset(n_scenarios=20, n_positions=4)
    results = probes.run_probe_suite(
        dataset, tasks=(probes.DEFAULT_TASKS[1],), n_permutations=4, seed=2
    )
    result = results[0]
    assert result.status == "ok"
    assert result.n_classes is not None and result.n_classes >= 2
    assert 0.0 <= result.chance <= 1.0
    assert 0.0 <= result.score <= 1.0


# ======================================================================================
# patching.py -- alignments
# ======================================================================================


def test_final_query_alignment_is_the_same_offset_from_the_end():
    clean_item, clean_tok = _variant("sc-B", 4, n_tokens=80)
    corrupt_item, corrupt_tok = _variant("sc-B", 30, n_tokens=64, seed=1)
    pair = patching.PatchPair(
        scenario_id="sc-B",
        clean_item=clean_item,
        clean_tok=clean_tok,
        corrupted_item=corrupt_item,
        corrupted_tok=corrupt_tok,
    )
    alignment = patching.build_alignment("final_query", pair, n_final_tokens=3)
    assert alignment.exact
    assert alignment.donor_indices == (77, 78, 79)
    assert alignment.receiver_indices == (61, 62, 63)
    offsets_donor = [80 - i for i in alignment.donor_indices]
    offsets_recv = [64 - i for i in alignment.receiver_indices]
    assert offsets_donor == offsets_recv


def test_span_aligned_maps_the_clean_span_onto_the_corrupted_span(pair_edge_middle):
    alignment = patching.build_alignment("span_aligned", pair_edge_middle)
    assert not alignment.exact
    assert "RoPE" in alignment.note
    donor_span = pair_edge_middle.clean_tok.evidence_spans[0]
    recv_span = pair_edge_middle.corrupted_tok.evidence_spans[0]
    assert set(alignment.receiver_indices) == set(range(*recv_span))
    assert all(donor_span[0] <= i < donor_span[1] for i in alignment.donor_indices)
    assert list(alignment.donor_indices) == sorted(alignment.donor_indices)


def test_span_aligned_resamples_and_says_so():
    _, short = _variant("sc-C", 4, n_tokens=40)
    long_ids = list(np.random.default_rng(2).integers(20, 200, size=40))
    long_tok = _tok("sc-C-long", long_ids, (10, 22), 0.4)
    pair = patching.PatchPair(
        scenario_id="sc-C",
        clean_item=_item("sc-C-long", "sc-C", position=0.4),
        clean_tok=long_tok,
        corrupted_item=_item(short.item_id, "sc-C", position=0.1),
        corrupted_tok=short,
    )
    alignment = patching.build_alignment("span_aligned", pair)
    assert alignment.n_positions == EVIDENCE_LEN
    assert "resampled" in alignment.note
    assert min(alignment.donor_indices) >= 10 and max(alignment.donor_indices) < 22


def test_all_positions_is_skipped_when_the_lengths_differ():
    clean_item, clean_tok = _variant("sc-E", 4, n_tokens=80)
    corrupt_item, corrupt_tok = _variant("sc-E", 30, n_tokens=64, seed=1)
    pair = patching.PatchPair(
        scenario_id="sc-E",
        clean_item=clean_item,
        clean_tok=clean_tok,
        corrupted_item=corrupt_item,
        corrupted_tok=corrupt_tok,
    )
    alignment = patching.build_alignment("all_positions", pair)
    assert alignment.status.startswith("skipped:length_mismatch")
    assert not alignment.ok


def test_all_positions_covers_the_prompt_and_flags_itself(pair_edge_middle):
    alignment = patching.build_alignment("all_positions", pair_edge_middle)
    assert alignment.ok and not alignment.exact
    assert alignment.n_positions == N_TOKENS  # prompt only; answer positions excluded
    assert "localises nothing" in alignment.note


def test_random_control_avoids_the_evidence_and_the_final_window(pair_edge_middle):
    alignment = patching.build_alignment(
        "random_control", pair_edge_middle, n_final_tokens=2
    )
    span = pair_edge_middle.corrupted_tok.evidence_spans[0]
    assert alignment.n_positions == EVIDENCE_LEN
    assert not set(alignment.receiver_indices) & set(range(*span))
    assert not set(alignment.receiver_indices) & {N_TOKENS - 1, N_TOKENS - 2}
    assert alignment.donor_indices == alignment.receiver_indices
    # Seeded: two builds of the same pair agree.
    again = patching.build_alignment("random_control", pair_edge_middle, n_final_tokens=2)
    assert again.receiver_indices == alignment.receiver_indices


def test_site_label_names_the_briefs_five_sites():
    assert patching.site_label("resid", "final_query") == "final_query_representation"
    assert patching.site_label("resid", "span_aligned") == "evidence_span_representation"
    assert patching.site_label("resid", "all_positions") == "whole_residual_stream"
    assert patching.site_label("attn_out", "final_query") == "attention_output_final_query"
    assert patching.site_label("mlp_out", "span_aligned") == "mlp_output_span_aligned"


# ======================================================================================
# patching.py -- the measurement
# ======================================================================================


def test_gold_logprob_is_negative_and_matches_a_full_logit_forward(model):
    _, tok_item = _variant("sc-F", 10, n_tokens=48)
    measured = patching.gold_logprob(model, tok_item)
    ids = torch.tensor(
        list(tok_item.input_ids) + list(tok_item.answer_ids), dtype=torch.long
    ).reshape(1, -1)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    lp = torch.log_softmax(logits[0, -2].float(), dim=-1)
    assert measured == pytest.approx(float(lp[tok_item.answer_ids[0]]), abs=1e-4)
    assert measured < 0.0


def test_final_query_patch_at_the_last_layer_reproduces_the_clean_score(
    model, pair_edge_middle
):
    """See the module docstring: with a one-token answer this must be an equality."""
    last = TINY["num_hidden_layers"] - 1
    records = patching.run_patch_pair(
        model,
        pair_edge_middle,
        layers=[last],
        sites=["resid"],
        alignments=("final_query",),
        model_key="tiny",
    )
    assert len(records) == 1
    record = records[0]
    assert record.status == "ok"
    assert record.alignment_exact
    assert record.patched_gold_logprob == pytest.approx(record.clean_gold_logprob, abs=1e-5)
    assert record.delta_gold_logprob == pytest.approx(
        record.clean_gold_logprob - record.corrupted_gold_logprob, abs=1e-5
    )


def test_patching_a_run_into_itself_changes_nothing(model):
    """Donor == receiver: every site, every layer, every alignment must move zero nats."""
    item, tok_item = _variant("sc-G", 20, n_tokens=64)
    pair = patching.PatchPair(
        scenario_id="sc-G",
        clean_item=item,
        clean_tok=tok_item,
        corrupted_item=item,
        corrupted_tok=tok_item,
    )
    records = patching.run_patch_pair(
        model, pair, alignments=patching.ALIGNMENTS, model_key="tiny"
    )
    live = [r for r in records if r.status == "ok"]
    assert live, "no alignment produced a record"
    assert max(abs(r.delta_gold_logprob) for r in live) < 1e-4
    assert {r.alignment for r in live} == set(patching.ALIGNMENTS)


def test_run_patch_pair_covers_the_grid_and_flags_skipped_alignments(model):
    clean_item, clean_tok = _variant("sc-H", 4, n_tokens=64)
    corrupt_item, corrupt_tok = _variant("sc-H", 30, n_tokens=48, seed=3)
    pair = patching.PatchPair(
        scenario_id="sc-H",
        clean_item=clean_item,
        clean_tok=clean_tok,
        corrupted_item=corrupt_item,
        corrupted_tok=corrupt_tok,
    )
    alignments = ("final_query", "span_aligned", "all_positions")
    records = patching.run_patch_pair(
        model, pair, layers=[0, 1], sites=["resid", "attn_out"], alignments=alignments
    )
    assert len(records) == 2 * 2 * len(alignments)
    assert {(r.layer, r.site, r.alignment) for r in records} == {
        (layer, site, name)
        for layer in (0, 1)
        for site in ("resid", "attn_out")
        for name in alignments
    }
    skipped = [r for r in records if r.alignment == "all_positions"]
    assert all(r.status.startswith("skipped:length_mismatch") for r in skipped)
    assert all(np.isnan(r.patched_gold_logprob) for r in skipped)
    live = [r for r in records if r.status == "ok"]
    assert all(r.n_patched_positions > 0 for r in live)
    assert {r.n_answer_tokens for r in records} == {1}


def test_deeper_layers_are_not_all_identical(model, pair_edge_middle):
    """A sanity floor: the patch has to actually do something at some layer."""
    records = patching.run_patch_pair(
        model, pair_edge_middle, sites=["resid"], alignments=("span_aligned",)
    )
    deltas = [r.delta_gold_logprob for r in records if r.status == "ok"]
    assert len(deltas) == TINY["num_hidden_layers"]
    assert np.ptp(deltas) > 0.0


def test_recovery_fraction_is_nan_when_the_gap_is_too_small_to_divide_by():
    assert np.isnan(patching._recovery(0.5, 0.4, 0.401))
    assert patching._recovery(-1.0, -2.0, 0.0) == pytest.approx(0.5)


def test_find_matched_pairs_picks_the_edge_and_the_middle():
    lookup = {}
    records = []
    for scenario, gap in (("sc-1", 1.0), ("sc-2", 0.01)):
        for offset, position in ((2, 0.05), (48, 0.5), (92, 0.95)):
            item, tok_item = _variant(scenario, offset, seed=offset)
            lookup[item.item_id] = (item, tok_item)
            records.append(
                {
                    "item_id": item.item_id,
                    "scenario_id": scenario,
                    "actual_positions": [position],
                    "gold_nll": 1.0 + (gap if position == 0.5 else 0.0),
                    "correct": position != 0.5,
                }
            )
    pairs = patching.find_matched_pairs(records, lookup, min_nll_gap=0.25)
    assert [p.scenario_id for p in pairs] == ["sc-1"]  # sc-2's gap is below threshold
    pair = pairs[0]
    assert pair.corrupted_position == pytest.approx(0.5)
    assert pair.clean_position in (pytest.approx(0.05), pytest.approx(0.95))
    assert pair.selection == "gold_nll"

    by_accuracy = patching.find_matched_pairs(records, lookup, criterion="correct")
    assert {p.scenario_id for p in by_accuracy} == {"sc-1", "sc-2"}


def test_patch_pair_rejects_mismatched_answers():
    clean_item, clean_tok = _variant("sc-I", 4, n_tokens=48)
    corrupt_item, corrupt_tok = _variant("sc-I", 20, n_tokens=48, answer_ids=(11, 12))
    with pytest.raises(ValueError, match="different answer tokens"):
        patching.PatchPair(
            scenario_id="sc-I",
            clean_item=clean_item,
            clean_tok=clean_tok,
            corrupted_item=corrupt_item,
            corrupted_tok=corrupt_tok,
        )


def test_patch_cost_estimate_is_linear_in_the_grid():
    small = patching.patch_cost_estimate(n_pairs=1, n_layers=28, seq_len=4096)
    assert small["forwards_per_pair"] == 2 + 28 * 3 * 3
    big = patching.patch_cost_estimate(n_pairs=10, n_layers=28, seq_len=4096)
    assert big["n_forwards"] == 10 * small["forwards_per_pair"]
    assert big["estimated_seconds"] > small["estimated_seconds"]
    assert big["exceeds_budget"] is (big["estimated_gpu_hours"] > big["gpu_hour_budget"])


def test_patch_records_write_a_parquet_with_provenance(model, pair_edge_middle, tmp_path):
    records = patching.run_patch_pair(
        model, pair_edge_middle, layers=[0], sites=["resid"], alignments=("final_query",)
    )
    frame = patching.patch_records_to_dataframe(records)
    assert len(frame) == 1
    assert {"delta_gold_logprob", "alignment_note", "site_label"} <= set(frame.columns)

    target = patching.write_patch_records(records, tmp_path / "patch.parquet")
    assert target.exists()
    assert target.with_name(target.name + ".provenance.json").exists()
    import pandas as pd

    assert len(pd.read_parquet(target)) == 1


# ======================================================================================
# head_stats.py
# ======================================================================================


@pytest.fixture(scope="module")
def head_table(model, arch):
    items = [_variant("sc-H1", 8, seed=1)[1], _variant("sc-H2", 40, seed=2)[1]]
    config = head_stats.HeadStatsConfig(n_bins=8, final_rows=4, chunk=32)
    return head_stats.capture_head_stats(
        model, items, config=config, arch=arch, model_key="tiny"
    )


def test_capture_head_stats_shapes_and_ranges(head_table, arch):
    assert head_table.layers == tuple(range(arch.num_layers))
    assert head_table.n_q_heads == arch.num_q_heads
    assert head_table.n_kv_heads == arch.num_kv_heads
    assert head_table.n_items == 2
    for metric in head_stats.HEAD_METRICS:
        block = head_table.means[metric]
        assert block.shape == (arch.num_layers, arch.num_q_heads)
        assert np.isfinite(block).all(), metric
    assert (head_table.means["gold_span_attn"] >= 0.0).all()
    assert (head_table.means["gold_span_attn"] <= 1.0).all()
    assert (head_table.means["entropy"] >= 0.0).all()
    assert (head_table.means["entropy"] <= np.log(N_TOKENS) + 1e-6).all()
    assert (head_table.means["position_variance"] >= -1e-12).all()
    assert (head_table.means["head_out_norm"] > 0.0).all()


def test_capture_conserves_attention_mass(model, arch):
    """``edge_fraction = 0.5`` selects every bin, so the edge column must be 1.0."""
    items = [_variant("sc-H3", 12, seed=4)[1]]
    config = head_stats.HeadStatsConfig(n_bins=8, edge_fraction=0.5, final_rows=0)
    table = head_stats.capture_head_stats(model, items, config=config, arch=arch)
    assert table.means["edge_attn"] == pytest.approx(
        np.ones_like(table.means["edge_attn"]), abs=1e-3
    )
    assert table.recent_scope == "all_query_rows"


def test_gold_span_attention_saturates_when_the_span_is_everything(model, arch):
    _, tok_item = _variant("sc-H4", 12, seed=5)
    everywhere = TokenizedItem(
        item_id=tok_item.item_id,
        input_ids=tok_item.input_ids,
        answer_ids=tok_item.answer_ids,
        evidence_spans=((0, tok_item.n_tokens),),
        actual_positions=tok_item.actual_positions,
    )
    table = head_stats.capture_head_stats(
        model,
        [everywhere],
        layers=[0],
        config=head_stats.HeadStatsConfig(n_bins=8, final_rows=0),
        arch=arch,
    )
    assert table.means["gold_span_attn"] == pytest.approx(
        np.ones_like(table.means["gold_span_attn"]), abs=1e-3
    )


def test_recency_pass_is_restricted_to_the_final_rows(model, arch):
    items = [_variant("sc-H5", 16, seed=6)[1]]
    common = dict(n_bins=8, chunk=32)
    all_rows = head_stats.capture_head_stats(
        model, items, arch=arch,
        config=head_stats.HeadStatsConfig(final_rows=0, **common),
    )
    final_rows = head_stats.capture_head_stats(
        model, items, arch=arch,
        config=head_stats.HeadStatsConfig(final_rows=1, **common),
    )
    # The last query row is the only row that can put mass in the last absolute bin, so
    # restricting to it must raise the recency figure well above the all-rows average.
    assert final_rows.means["recent_attn"].mean() > all_rows.means["recent_attn"].mean()
    assert final_rows.recent_scope == "final_1_query_rows"
    assert all_rows.recent_scope == "all_query_rows"
    # Everything else is unaffected by the second pass.
    assert np.allclose(final_rows.means["entropy"], all_rows.means["entropy"], atol=1e-6)


def test_head_stats_rejects_a_window_that_selects_no_bin():
    with pytest.raises(ValueError, match="selects none of"):
        head_stats.HeadStatsConfig(n_bins=2, middle_halfwidth=0.05).masks(
            np.asarray([0.25, 0.75])
        )


def test_head_stats_rejects_a_query_dependent_binning(model, arch):
    from rineya_psa.hooks.binning import DistanceBinning

    with pytest.raises(ValueError, match="query-dependent"):
        head_stats.capture_head_stats(
            model, [_variant("sc-H6", 8)[1]], binning=DistanceBinning.uniform(4), arch=arch
        )


def test_kv_rollup_is_the_group_mean(head_table, arch):
    rows = {(r.layer, r.q_head): r for r in head_table.rows()}
    assert len(rows) == arch.num_layers * arch.num_q_heads
    for kv_row in head_table.kv_rows():
        members = arch.q_heads_for_kv(kv_row.kv_head)
        assert kv_row.q_heads == members
        expected = np.mean([rows[(kv_row.layer, g)].entropy for g in members])
        assert kv_row.entropy == pytest.approx(expected)
    # And the q-head rows agree with the array they came from.
    row = rows[(1, 3)]
    assert row.kv_head == arch.kv_head_for_q(3)
    assert row.gold_span_attn == pytest.approx(head_table.value("gold_span_attn", 1, 3))


def test_ablation_effect_is_attached_at_kv_head_resolution(head_table, arch):
    n_bins = 4
    values = np.zeros((arch.num_layers, arch.num_kv_heads, n_bins))
    values[2, 1, 3] = -0.75  # gold_nll: a negative Delta is an important cell
    matrix = SignificanceMatrix(
        values=values,
        layers=tuple(range(arch.num_layers)),
        kv_heads=tuple(range(arch.num_kv_heads)),
        bins=tuple(range(n_bins)),
        binning_desc=AbsoluteBinning(n_bins).describe(),
        objective="gold_nll",
        model_key="tiny",
        provenance={},
    )
    table = head_table.with_ablation_effect(matrix, reducer="max")
    assert table.ablation_scope == "kv_head:gold_nll:max"
    effect = table.means["ablation_effect"]
    for q_head in arch.q_heads_for_kv(1):
        assert effect[2, q_head] == pytest.approx(0.75)
    for q_head in arch.q_heads_for_kv(0):
        assert effect[2, q_head] == pytest.approx(0.0)
    assert all(r.ablation_scope.startswith("kv_head:") for r in table.rows())
    assert table.kv_rows()[0].ablation_scope == table.ablation_scope


def test_ablation_effect_keeps_unvisited_cells_nan(head_table, arch, recwarn):
    """An unmeasured cell is not a zero, and reducing over one must not warn."""
    n_bins = 4
    values = np.full((arch.num_layers, arch.num_kv_heads, n_bins), np.nan)
    values[0, 0, :] = -0.5  # only this row was ever evaluated
    matrix = SignificanceMatrix(
        values=values,
        layers=tuple(range(arch.num_layers)),
        kv_heads=tuple(range(arch.num_kv_heads)),
        bins=tuple(range(n_bins)),
        binning_desc=AbsoluteBinning(n_bins).describe(),
        objective="gold_nll",
        model_key="tiny",
        provenance={},
    )
    for reducer in head_stats.ABLATION_REDUCERS:
        effect = head_table.with_ablation_effect(matrix, reducer=reducer).means[
            "ablation_effect"
        ]
        assert np.isfinite(effect[0, arch.q_heads_for_kv(0)[0]])
        assert np.isnan(effect[0, arch.q_heads_for_kv(1)[0]])
        assert np.isnan(effect[1]).all()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_ablation_effect_rejects_a_head_count_mismatch(head_table, arch):
    matrix = SignificanceMatrix(
        values=np.zeros((arch.num_layers, arch.num_kv_heads + 1, 2)),
        layers=tuple(range(arch.num_layers)),
        kv_heads=tuple(range(arch.num_kv_heads + 1)),
        bins=(0, 1),
        binning_desc=AbsoluteBinning(2).describe(),
        objective="gold_nll",
        model_key="tiny",
        provenance={},
    )
    with pytest.raises(ValueError, match="KV heads"):
        head_table.with_ablation_effect(matrix)


def test_patch_effect_is_attached_at_layer_resolution(head_table):
    records = [
        {"layer": 1, "site": "attn_out", "alignment": "final_query",
         "delta_gold_logprob": 0.4, "status": "ok"},
        {"layer": 1, "site": "attn_out", "alignment": "final_query",
         "delta_gold_logprob": 0.6, "status": "ok"},
        {"layer": 0, "site": "resid", "alignment": "final_query",
         "delta_gold_logprob": 9.9, "status": "ok"},
        {"layer": 2, "site": "attn_out", "alignment": "final_query",
         "delta_gold_logprob": 1.0, "status": "skipped:length_mismatch"},
    ]
    table = head_table.with_patch_effect(records)
    effect = table.means["patch_effect"]
    assert np.allclose(effect[1], 0.5)
    assert np.isnan(effect[0]).all()  # different site
    assert np.isnan(effect[2]).all()  # skipped record
    assert table.patch_effect_scope == "layer:attn_out:final_query:delta_gold_logprob"


def test_head_stats_frames_and_writer(head_table, arch, tmp_path):
    wide = head_stats.head_stats_to_dataframe(head_table.rows())
    assert len(wide) == arch.num_layers * arch.num_q_heads
    assert {"q_head", "kv_head", "recent_scope"} <= set(wide.columns)

    long = head_table.long_frame()
    expected = (len(head_stats.HEAD_METRICS) + 2) * arch.num_layers * arch.num_q_heads
    assert len(long) == expected
    assert set(long["metric"]) >= set(head_stats.HEAD_METRICS)

    target = head_stats.write_head_stats(head_table, tmp_path / "heads.parquet", kv=True)
    assert target.exists()
    assert target.with_name(target.name + ".provenance.json").exists()
    import pandas as pd

    assert len(pd.read_parquet(target)) == arch.num_layers * arch.num_kv_heads


def test_capture_only_guard_rejects_an_intervention(arch):
    from rineya_psa.hooks.specs import HeadScaleSpec, PSAContext

    gamma = torch.ones((arch.num_layers, arch.num_q_heads))
    gamma[0, 0] = 0.0
    ctx = PSAContext(enabled=True, head_scale=HeadScaleSpec(gamma=gamma))
    with pytest.raises(RuntimeError, match="capture-only"):
        head_stats._assert_capture_only(ctx)
