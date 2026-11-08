"""End to end on the toy model: do the nine modules actually connect?

This file makes no scientific claim. A randomly initialised 2-layer Qwen3 has no
positional structure to find, and every number it produces is noise. What it proves is
that the *pipes* line up — that the benchmark builder's ``BenchmarkItem`` is the one
``benchmark/tokenize.py`` accepts, that its ``TokenizedItem`` is the one
``evaluation/runner.py`` scores, that the ``SignificanceMatrix`` the estimator returns
is the one ``interventions/psa.py`` can turn into a ``PSAContext``, that the context is
the one the hook registry installs, and so on down to the figure.

The pipeline runs on a stub tokenizer rather than the real Qwen3 one because the toy
model's vocabulary is 512 tokens wide; real ids in the 151 k range would index out of
its embedding table. ``conftest.StubTokenizer`` reproduces the parts of the tokenizer
contract the lab depends on — verbatim user turn, pre-closed think block, real offset
mappings — and nothing else.

Budget: about 15 s on this machine, dominated by the 208-forward Tier-2 sweep.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rineya_psa.types import BenchmarkItem, TokenizedItem

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

N_BINS = 4
LENGTH_BUCKET = "tiny"
#: Small enough to stay inside the toy model's 1024-position window once the chat
#: template and the answer are added.
TOKEN_TARGET = 700

SMOKE_CONFIG: dict[str, Any] = {
    "name": "integration_smoke",
    "seed": 20260828,
    "scale": 1.0,
    "tokenizer": "stub",
    "sentences_per_chunk": 3,
    "token_tolerance": 0.05,
    "length_tokens": {LENGTH_BUCKET: TOKEN_TARGET},
    "positions": [0.05, 0.50, 0.95],
    "grids": [
        {
            "name": "core",
            "families": ["exact_retrieval", "semantic_single_hop"],
            "lengths": [LENGTH_BUCKET],
            "formats": ["natural"],
            "local_positions": ["middle"],
            "positions": "full",
            "scenarios": 2,
        }
    ],
}
EXPECTED_ITEMS = 2 * 2 * 3  # families x scenarios x positions


# --------------------------------------------------------------------------------------
# Stage 1-2: build a bank and tokenize it
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bank(tmp_path_factory: pytest.TempPathFactory, stub_tokenizer):
    """Twelve items built through the real Phase-2 generator, tokenized for real.

    The stub is injected into ``benchmark.tokenize``'s cache rather than through a
    parameter because ``generate.build`` resolves its tokenizer from the config's
    ``tokenizer`` key by design — the bank has to record *which* vocabulary its token
    budget was measured against.
    """
    from rineya_psa.benchmark import generate as generate_mod
    from rineya_psa.benchmark import tokenize as tokenize_mod

    stub = stub_tokenizer
    tokenize_mod._TOKENIZERS["stub"] = stub
    try:
        root = tmp_path_factory.mktemp("smoke_bank")
        report = generate_mod.build(SMOKE_CONFIG, out_root=root)
        jsonl = root / "generated" / generate_mod.JSONL_NAME
        items = [
            BenchmarkItem.from_json(json.loads(line))
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tokenized = [tokenize_mod.tokenize_item(item, stub) for item in items]
        yield stub, report, items, tokenized, root
    finally:
        tokenize_mod._TOKENIZERS.pop("stub", None)


def test_stage1_generator_produces_a_split_bank_with_measured_positions(bank) -> None:
    stub, report, items, tokenized, root = bank

    assert report.n_items == EXPECTED_ITEMS
    assert report.n_scenarios == 4
    assert report.token_out_of_tolerance == 0, (
        f"the budget fitter missed tolerance on {report.token_out_of_tolerance} items "
        f"(max relative error {report.token_abs_error_max:.3f})"
    )
    assert (root / "position_split_manifest.json").is_file()

    # Every positional variant of a scenario shares its split (SPEC section 8).
    by_scenario: dict[str, set[str]] = defaultdict(set)
    for item in items:
        by_scenario[item.scenario_id].add(item.split)
    assert all(len(v) == 1 for v in by_scenario.values())

    # And the measured token position tracks the requested one, which is the whole
    # point of measuring rather than assuming.
    for item, tok_item in zip(items, tokenized, strict=True):
        assert isinstance(tok_item, TokenizedItem)
        assert tok_item.n_tokens < 1024, "prompt must fit the toy model's position window"
        assert tok_item.answer_ids
        assert len(tok_item.evidence_spans) == len(item.evidence_sentences)
        requested = item.target_positions[-1]
        measured = tok_item.actual_positions[-1]
        assert abs(measured - requested) < 0.12, (item.item_id, requested, measured)


def test_stage2_rendered_prompt_carries_the_pre_closed_think_block(bank) -> None:
    """SPEC 13.5: the ``<think>`` trap is the difference between 0/27 and a real score."""
    from rineya_psa.evaluation import runner as runner_mod

    stub, _report, items, _tokenized, _root = bank
    rendered = runner_mod.render_prompt(stub, items[0])
    assert "<think>\n\n</think>" in rendered
    assert items[0].user in rendered


# --------------------------------------------------------------------------------------
# Stage 3-4: score, and reduce to positional metrics
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_records(bank, tiny_qwen3_model):
    from rineya_psa.evaluation import runner as runner_mod

    stub, _report, items, tokenized, _root = bank
    pairs = list(zip(items, tokenized, strict=True))
    return pairs, runner_mod.run_items(tiny_qwen3_model, stub, pairs, mode="teacher_forced")


def _curve(records, field: str = "gold_nll") -> dict[float, float]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for record in records:
        grouped[round(record.target_positions[-1], 2)].append(float(getattr(record, field)))
    return {k: float(np.mean(v)) for k, v in sorted(grouped.items())}


def test_stage3_teacher_forced_run_produces_one_scored_record_per_item(
    baseline_records,
) -> None:
    pairs, records = baseline_records
    assert len(records) == EXPECTED_ITEMS
    assert {r.item_id for r in records} == {i.item_id for i, _ in pairs}
    for record in records:
        assert record.mode == "teacher_forced"
        assert record.gold_nll is not None and np.isfinite(record.gold_nll)
        assert record.answer_margin is not None and np.isfinite(record.answer_margin)
        assert record.gold_nll > 0.0  # a 512-token untrained vocabulary cannot do better
        assert record.n_tokens > 0


def test_stage4_positional_metrics_reduce_the_run_to_the_headline_numbers(
    baseline_records,
) -> None:
    from rineya_psa.evaluation import metrics as metrics_mod

    _pairs, records = baseline_records
    nll_curve = _curve(records)
    assert set(nll_curve) == {0.05, 0.50, 0.95}

    gap = metrics_mod.edge_middle_gap(nll_curve)
    assert np.isfinite(gap)
    fit = metrics_mod.ushape_fit(list(nll_curve), list(nll_curve.values()))
    assert fit.n == 3 and np.isfinite(fit.beta2)

    grouped: dict[float, list[float]] = defaultdict(list)
    for record in records:
        grouped[round(record.target_positions[-1], 2)].append(float(bool(record.correct)))
    acc_curve = {k: float(np.mean(v)) for k, v in sorted(grouped.items())}
    assert 0.0 <= metrics_mod.position_robustness_index(acc_curve) <= 1.0

    pytest.importorskip("pandas")
    from rineya_psa.evaluation.runner import to_dataframe

    frame = to_dataframe(records)
    assert len(frame) == EXPECTED_ITEMS
    assert {"item_id", "scenario_id", "gold_nll"} <= set(frame.columns)


# --------------------------------------------------------------------------------------
# Stage 5: the significance matrix
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def significance(bank, tiny_qwen3_model):
    """A full 2 x 2 x 4 Tier-2 sweep. 16 cells x 12 items plus 12 baselines."""
    from rineya_psa.significance import causal as causal_mod

    stub, _report, items, tokenized, _root = bank
    pairs = list(zip(items, tokenized, strict=True))
    return causal_mod.estimate_causal_significance_ablation(
        tiny_qwen3_model,
        stub,
        pairs,
        layers=[0, 1],
        kv_heads=[0, 1],
        n_bins=N_BINS,
        mode="full",
        model_key="tiny",
        force_backend=False,  # CPU has only the MATH backend; pinning would raise
    )


def test_stage5_significance_matrix_covers_the_whole_declared_grid(
    significance, tiny_arch
) -> None:
    result = significance
    assert result.matrix.shape == (tiny_arch.num_layers, tiny_arch.num_kv_heads, N_BINS)
    assert result.coverage() == 1.0
    assert result.tier == "ablation"
    assert result.n_items == EXPECTED_ITEMS
    assert np.isfinite(result.matrix.values).all()
    assert len(result.matrix.bin_centers) == N_BINS

    # The per-item evidence must actually be per item, not a broadcast of the mean.
    deltas = result.deltas_for((0, 0, 0))
    assert deltas.shape == (EXPECTED_ITEMS,)
    assert np.isfinite(deltas).all()


def test_stage5b_matrix_round_trips_through_disk_with_provenance(
    significance, lab_dirs: dict[str, Path]
) -> None:
    from rineya_psa.significance.matrices import SignificanceMatrix

    path = lab_dirs["results"] / "smoke_significance.npz"
    significance.matrix.save_npz(path)
    assert path.is_file()
    reloaded = SignificanceMatrix.load(path)
    np.testing.assert_allclose(reloaded.values, significance.matrix.values)
    assert reloaded.content_hash() == significance.matrix.content_hash()


# --------------------------------------------------------------------------------------
# Stage 6-7: the matrix drives an intervention, and a sequential arm runs
# --------------------------------------------------------------------------------------


def test_stage6_psa_context_built_from_the_matrix_changes_the_score(
    significance, bank, baseline_records, tiny_qwen3_model, tiny_arch
) -> None:
    from rineya_psa.evaluation import runner as runner_mod
    from rineya_psa.interventions.psa import build_psa_context

    stub, _report, _items, _tokenized, _root = bank
    pairs, base = baseline_records

    zero = build_psa_context(significance.matrix, mode="value", lam=0.0, arch=tiny_arch)
    zero_records = runner_mod.run_items(
        tiny_qwen3_model, stub, pairs, mode="teacher_forced", ctx=zero
    )
    # SPEC 5.2's identity invariant, seen from the far end of the pipeline: a
    # coefficient of zero must not move a single score.
    for a, b in zip(base, zero_records, strict=True):
        assert b.gold_nll == pytest.approx(a.gold_nll, abs=1e-5)

    live = build_psa_context(significance.matrix, mode="value", lam=0.5, arch=tiny_arch)
    assert live.enabled and live.value_scale is not None
    live_records = runner_mod.run_items(
        tiny_qwen3_model, stub, pairs, mode="teacher_forced", ctx=live
    )
    moved = max(abs(a.gold_nll - b.gold_nll) for a, b in zip(base, live_records, strict=True))
    assert moved > 1e-6, "a live PSA context left every score untouched"

    # And the hook must have uninstalled itself on the way out.
    after = runner_mod.run_items(tiny_qwen3_model, stub, pairs, mode="teacher_forced")
    for a, b in zip(base, after, strict=True):
        assert b.gold_nll == pytest.approx(a.gold_nll, abs=1e-5)


def test_stage7_two_stage_sequential_arm_runs_and_restores(
    bank, baseline_records, tiny_qwen3_model, tiny_arch
) -> None:
    from rineya_psa.evaluation import runner as runner_mod
    from rineya_psa.sequential_attention import arm_two_stage, sequential_session

    stub, _report, _items, _tokenized, _root = bank
    pairs, base = baseline_records

    arm = arm_two_stage(tiny_arch)
    assert arm.n_stages == 2
    assert sorted(h for group in arm.groups for h in group) == list(range(tiny_arch.num_q_heads))

    with sequential_session(tiny_qwen3_model, "all", arm):
        seq_records = runner_mod.run_items(tiny_qwen3_model, stub, pairs, mode="teacher_forced")
    assert len(seq_records) == EXPECTED_ITEMS
    assert all(np.isfinite(r.gold_nll) for r in seq_records)

    restored = runner_mod.run_items(tiny_qwen3_model, stub, pairs, mode="teacher_forced")
    for a, b in zip(base, restored, strict=True):
        assert b.gold_nll == pytest.approx(a.gold_nll, abs=1e-5)


# --------------------------------------------------------------------------------------
# Stage 8-9: paired inference and a figure
# --------------------------------------------------------------------------------------


def test_stage8_paired_bootstrap_clusters_on_scenario_not_item(
    bank, baseline_records, tiny_qwen3_model, tiny_arch, lab_dirs: dict[str, Path]
) -> None:
    from rineya_psa.evaluation import runner as runner_mod
    from rineya_psa.sequential_attention import arm_two_stage, sequential_session
    from rineya_psa.statistics import paired_bootstrap, write_result

    stub, _report, _items, _tokenized, _root = bank
    pairs, base = baseline_records
    arm = arm_two_stage(tiny_arch)
    with sequential_session(tiny_qwen3_model, "all", arm):
        seq = runner_mod.run_items(tiny_qwen3_model, stub, pairs, mode="teacher_forced")

    result = paired_bootstrap(
        [r.gold_nll for r in seq],
        [r.gold_nll for r in base],
        [r.scenario_id for r in base],
        n=400,
        seed=20260828,
    )
    assert result.n_items == EXPECTED_ITEMS
    assert result.n_scenarios == 4, "resampling must be over scenarios, not items"
    assert result.ci_low <= result.estimate <= result.ci_high
    assert np.isfinite(result.cohens_dz)

    # And the artifact carries its provenance sibling (SPEC section 9).
    path = write_result(result, lab_dirs["results"] / "smoke_paired_bootstrap.json")
    assert path.is_file()
    sibling = path.with_name(path.name + ".provenance.json")
    assert sibling.is_file()
    stamp_json = json.loads(sibling.read_text(encoding="utf-8"))
    assert {"versions", "git", "generated_at_utc"} <= set(stamp_json)
    assert "python" in stamp_json["versions"]


def test_stage9_a_figure_renders_from_the_run_records(
    baseline_records, lab_dirs: dict[str, Path]
) -> None:
    pytest.importorskip("matplotlib")
    pandas = pytest.importorskip("pandas")
    from rineya_psa.figures import plots as plots_mod

    _pairs, records = baseline_records
    curve = _curve(records)
    frame = pandas.DataFrame(
        [
            {
                "model": "tiny",
                "position": position,
                "gold_nll": value,
                "ci_low": value - 0.05,
                "ci_high": value + 0.05,
            }
            for position, value in curve.items()
        ]
    )
    written = plots_mod.nll_vs_position(
        frame, lab_dirs["figures"] / "fig02_nll_vs_position", title="smoke"
    )
    assert written
    for path in written:
        assert path.is_file() and path.stat().st_size > 0
    assert {p.suffix for p in written} == {".png", ".pdf"}


# --------------------------------------------------------------------------------------
# Cross-tier agreement: the one place Tier 1 and Tier 2 have to meet
# --------------------------------------------------------------------------------------


def test_attribution_and_ablation_tiers_are_comparable(
    bank, significance, tiny_qwen3_model
) -> None:
    """SPEC 11.2's gate. On noise the verdict is meaningless; the plumbing is not."""
    from rineya_psa.significance import causal as causal_mod

    stub, _report, items, tokenized, _root = bank
    pairs = list(zip(items, tokenized, strict=True))
    grads_before = [bool(p.requires_grad) for p in tiny_qwen3_model.parameters()]

    attribution = causal_mod.estimate_causal_significance_attribution(
        tiny_qwen3_model,
        stub,
        pairs,
        layers=[0, 1],
        kv_heads=[0, 1],
        n_bins=N_BINS,
        model_key="tiny",
        force_backend=False,
    )
    assert attribution.tier == "attribution"
    assert attribution.matrix.shape == significance.matrix.shape

    report = causal_mod.validate_attribution_against_ablation(
        attribution, significance, n_resamples=200
    )
    assert set(report) >= {"spearman", "spearman_ci", "verdict", "n_cells"}
    assert report["n_cells"] == significance.matrix.values.size
    assert -1.0 <= float(report["spearman"]) <= 1.0
    assert report["verdict"] in (
        "attribution_usable",
        "fallback_to_coarse_to_fine_ablation",
    )

    # SPEC 12.4's mitigation is to freeze the parameters for the backward pass. It has
    # to be reversed on the way out, or every later forward in the same process would
    # silently inherit a different autograd graph.
    assert [bool(p.requires_grad) for p in tiny_qwen3_model.parameters()] == grads_before
    assert not tiny_qwen3_model.training


# --------------------------------------------------------------------------------------
# Cross-package name collisions
# --------------------------------------------------------------------------------------


def test_the_two_tokenize_item_helpers_reject_a_swapped_call(bank) -> None:
    """``benchmark`` and ``evaluation`` both export ``tokenize_item``, in opposite orders.

    Both are legitimate — one measures spans from the benchmark's recorded character
    offsets, the other re-finds the evidence sentences in the rendered prompt — so the
    reconciliation is a loud failure rather than a rename that would break every caller.
    """
    from rineya_psa.benchmark import tokenize as bench_tokenize
    from rineya_psa.evaluation import runner as runner_mod

    stub, _report, items, _tokenized, _root = bank
    item = items[0]

    assert isinstance(bench_tokenize.tokenize_item(item, stub), TokenizedItem)
    assert isinstance(runner_mod.tokenize_item(stub, item), TokenizedItem)

    with pytest.raises(TypeError, match="evaluation.runner.tokenize_item"):
        bench_tokenize.tokenize_item(stub, item)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="benchmark.tokenize.tokenize_item"):
        runner_mod.tokenize_item(item, stub)  # type: ignore[arg-type]


def test_the_duplicated_helpers_that_were_left_alone_actually_agree(
    tiny_qwen3_model,
) -> None:
    """Three names are exported twice. Two of them are the same function, twice."""
    from rineya_psa.activation_analysis.collect import decoder_layers as collect_layers
    from rineya_psa.model_audit.hashing import canonical_json as audit_json
    from rineya_psa.sequential_attention.wrapper import decoder_layers as wrapper_layers
    from rineya_psa.statistics.provenance import canonical_json as provenance_json

    payload = {"b": 1, "a": [1, 2, {"z": None}], "c": 2.5, "d": True}
    assert audit_json(payload) == provenance_json(payload)
    assert list(collect_layers(tiny_qwen3_model)) == list(wrapper_layers(tiny_qwen3_model))
