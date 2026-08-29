"""Tests for ``rineya_psa.figures``: all thirteen of brief section 47, plus the axis guard.

The contract for a figure builder is narrow and checkable without eyes on the output: it
must accept synthetic data of the right shape, write both a 200 dpi PNG and a PDF, and put
something on the canvas (a blank axes still produces a ~10 kB PNG, so the size floors here
are set above what an empty figure costs). Everything the reader would otherwise have to
trust -- that zero is on every value axis, that a transposed matrix is rejected, that the
palette is stable -- is asserted.

The no-truncated-axis guard is tested twice over: directly, and by spying on the call each
figure function makes, so a future figure that forgets to call it fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rineya_psa.figures import plots as P
from rineya_psa.figures import style as ST
from rineya_psa.paths import MODELS

MODELS_4 = ("qwen3", "halo", "nimbus", "supa_a11")
POSITIONS = (0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98)
N_LAYERS = 28
N_KV = 8

# A blank 7x4.2in axes at 200 dpi is ~12 kB of PNG; anything above 20 kB has content.
MIN_PNG_BYTES = 20_000
MIN_PDF_BYTES = 4_000


# --------------------------------------------------------------------------------------
# Synthetic inputs, one per figure
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260828)


@pytest.fixture(scope="module")
def curve_rows(rng) -> list[dict[str, float | str]]:
    """A U-shaped accuracy/NLL table: the shape Phase 3 actually produces."""
    rows: list[dict[str, float | str]] = []
    for offset, model in enumerate(MODELS_4):
        for p in POSITIONS:
            acc = 0.34 + 0.28 * abs(p - 0.5) + 0.01 * offset
            nll = 0.48 - 0.26 * abs(p - 0.5) + 0.02 * offset
            rows.append(
                {
                    "model": model,
                    "position": p,
                    "accuracy": acc,
                    "ci_low": acc - 0.06,
                    "ci_high": acc + 0.06,
                    "gold_nll": nll,
                }
            )
    return rows


@pytest.fixture(scope="module")
def gap_rows() -> list[dict[str, float | str]]:
    return [
        {"model": m, "edge_middle_gap": g, "ci_low": g - 0.05, "ci_high": g + 0.05}
        for m, g in zip(MODELS_4, (0.19, 0.13, -0.04, 0.08), strict=True)
    ]


@pytest.fixture(scope="module")
def effect_rows(rng) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for binning, centers in (
        ("absolute", np.linspace(0.06, 0.94, 8)),
        ("distance", np.linspace(0.06, 0.94, 8)),
    ):
        for model in MODELS_4[:2]:
            for c in centers:
                effect = 0.03 * np.cos(np.pi * c) + 0.004
                rows.append(
                    {
                        "binning": binning,
                        "bin_center": float(c),
                        "model": model,
                        "effect": float(effect),
                        "ci_low": float(effect - 0.008),
                        "ci_high": float(effect + 0.008),
                    }
                )
    return rows


@pytest.fixture(scope="module")
def weight_delta_rows(rng) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for model in MODELS_4[1:]:
        for layer in range(N_LAYERS):
            for family in ("q_proj", "k_proj", "v_proj"):
                rows.append(
                    {
                        "model_key": model,
                        "layer": layer,
                        "family": family,
                        "rel_frobenius": float(abs(rng.normal(0.011, 0.003))),
                    }
                )
        rows.append(
            {"model_key": model, "layer": -1, "family": "embed_tokens", "rel_frobenius": 0.004}
        )
    return rows


@pytest.fixture(scope="module")
def arm_rows(rng) -> list[dict[str, float | str]]:
    return [
        {
            "arm": arm,
            "metric": metric,
            "score": float(score),
            "ci_low": float(score - 0.05),
            "ci_high": float(score + 0.05),
        }
        for arm in ("A baseline", "B sequential", "C recurrent", "D hybrid", "E control")
        for metric, score in (
            ("accuracy", rng.uniform(0.25, 0.55)),
            ("robustness", rng.uniform(0.35, 0.85)),
        )
    ]


@pytest.fixture(scope="module")
def interaction_rows() -> list[dict[str, float | str]]:
    return [
        {"sequentiality": s, "condition": c, "score": v, "ci_low": v - 0.05, "ci_high": v + 0.05}
        for s, c, v in (
            ("sequential", "baseline", 0.32),
            ("sequential", "seq_memory", 0.51),
            ("parallel", "baseline", 0.40),
            ("parallel", "seq_memory", 0.44),
        )
    ]


@pytest.fixture(scope="module")
def latency_rows() -> list[dict[str, float | str]]:
    return [
        {"model": m, "latency_s": 0.29 + 0.07 * i, "accuracy": 0.38 + 0.04 * i}
        for i, m in enumerate(MODELS_4)
    ]


@pytest.fixture(scope="module")
def robustness_rows() -> list[dict[str, float | str]]:
    return [
        {"model": m, "mean_accuracy": 0.40 + 0.05 * i, "worst_accuracy": 0.18 + 0.06 * i}
        for i, m in enumerate(MODELS_4)
    ]


@pytest.fixture(scope="module")
def matrices(rng) -> dict[str, np.ndarray]:
    return {
        "layer_position": rng.normal(0.0, 0.05, (N_LAYERS, len(POSITIONS))),
        "kv_position": rng.gamma(2.0, 0.02, (N_KV, len(POSITIONS))),
        "layer_kv": rng.normal(0.0, 0.04, (N_LAYERS, N_KV)),
    }


@pytest.fixture(scope="module")
def explained_variance(rng) -> dict[str, np.ndarray]:
    return {
        m: np.sort(rng.dirichlet(np.ones(10) * 0.4))[::-1].astype(float) for m in MODELS_4[:3]
    }


def _all_figure_calls(
    out: Path,
    curve_rows,
    gap_rows,
    matrices,
    effect_rows,
    explained_variance,
    weight_delta_rows,
    arm_rows,
    interaction_rows,
    latency_rows,
    robustness_rows,
) -> dict[str, tuple]:
    """``stem -> (callable, args, kwargs)`` for every one of the thirteen."""
    return {
        "fig01_accuracy_vs_position": (P.accuracy_vs_position, (curve_rows, out / "fig01"), {}),
        "fig02_nll_vs_position": (P.nll_vs_position, (curve_rows, out / "fig02"), {}),
        "fig03_edge_middle_gap": (P.edge_middle_gap_by_model, (gap_rows, out / "fig03"), {}),
        "fig04_layer_position_heatmap": (
            P.layer_position_heatmap,
            (matrices["layer_position"], out / "fig04"),
            {"positions": POSITIONS},
        ),
        "fig05_kv_head_position_heatmap": (
            P.kv_head_position_heatmap,
            (matrices["kv_position"], out / "fig05"),
            {"positions": POSITIONS},
        ),
        "fig06_layer_kv_head_significance": (
            P.layer_kv_head_significance,
            (matrices["layer_kv"], out / "fig06"),
            {},
        ),
        "fig07_absolute_vs_distance": (
            P.absolute_vs_distance_effects,
            (effect_rows, out / "fig07"),
            {},
        ),
        "fig08_low_rank_explained_variance": (
            P.low_rank_explained_variance,
            (explained_variance, out / "fig08"),
            {},
        ),
        "fig09_weight_delta_by_layer": (
            P.weight_delta_by_layer,
            (weight_delta_rows, out / "fig09"),
            {},
        ),
        "fig10_architecture_comparison": (
            P.architecture_comparison,
            (arm_rows, out / "fig10"),
            {},
        ),
        "fig11_significance_sequentiality": (
            P.significance_sequentiality_interaction,
            (interaction_rows, out / "fig11"),
            {"contrast": 0.148, "contrast_ci": (0.02, 0.27)},
        ),
        "fig12_accuracy_vs_latency": (P.accuracy_vs_latency, (latency_rows, out / "fig12"), {}),
        "fig13_worst_vs_mean_accuracy": (
            P.worst_vs_mean_accuracy,
            (robustness_rows, out / "fig13"),
            {},
        ),
    }


@pytest.fixture
def figure_calls(
    tmp_path,
    curve_rows,
    gap_rows,
    matrices,
    effect_rows,
    explained_variance,
    weight_delta_rows,
    arm_rows,
    interaction_rows,
    latency_rows,
    robustness_rows,
):
    return _all_figure_calls(
        tmp_path,
        curve_rows,
        gap_rows,
        matrices,
        effect_rows,
        explained_variance,
        weight_delta_rows,
        arm_rows,
        interaction_rows,
        latency_rows,
        robustness_rows,
    )


# --------------------------------------------------------------------------------------
# All thirteen render
# --------------------------------------------------------------------------------------


def test_all_thirteen_figures_render_to_png_and_pdf(figure_calls, tmp_path):
    assert len(figure_calls) == 13
    for stem, (fn, args, kwargs) in figure_calls.items():
        written = fn(*args, **kwargs)
        assert [p.suffix for p in written] == [".png", ".pdf"], stem
        png, pdf = written
        assert png.exists() and pdf.exists(), stem
        assert png.stat().st_size > MIN_PNG_BYTES, f"{stem}: {png.stat().st_size} B png"
        assert pdf.stat().st_size > MIN_PDF_BYTES, f"{stem}: {pdf.stat().st_size} B pdf"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", stem
        assert pdf.read_bytes()[:5] == b"%PDF-", stem
    assert len(list(tmp_path.glob("*.png"))) == 13


def test_registry_names_the_thirteen_in_order(figure_calls):
    assert list(P.FIGURES) == [stem for stem, _ in P.FIGURE_ORDER]
    assert len(P.FIGURE_ORDER) == 13
    assert list(P.FIGURES) == list(figure_calls)
    numbers = [int(stem[3:5]) for stem in P.FIGURES]
    assert numbers == list(range(1, 14))


def test_png_is_written_at_200_dpi(figure_calls):
    fn, args, kwargs = figure_calls["fig01_accuracy_vs_position"]
    png, _pdf = fn(*args, **kwargs)
    data = png.read_bytes()
    marker = data.index(b"pHYs")
    pixels_per_metre = int.from_bytes(data[marker + 4 : marker + 8], "big")
    assert round(pixels_per_metre * 0.0254) == ST.DPI == 200


def test_figures_do_not_leave_open_canvases(figure_calls):
    plt.close("all")
    for fn, args, kwargs in figure_calls.values():
        fn(*args, **kwargs)
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------------------
# The no-truncated-axis guard
# --------------------------------------------------------------------------------------


def _axes(ylim=None, xlim=None):
    fig, ax = plt.subplots()
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    return fig, ax


def test_guard_fires_on_a_ylim_that_hides_zero():
    fig, ax = _axes(ylim=(0.40, 0.62))
    try:
        with pytest.raises(ST.TruncatedAxisError, match="exclude zero"):
            ST.assert_no_truncated_axis(ax)
    finally:
        plt.close(fig)


def test_guard_passes_on_a_zero_based_axis():
    fig, ax = _axes(ylim=(0.0, 1.0))
    try:
        ST.assert_no_truncated_axis(ax)  # must not raise
    finally:
        plt.close(fig)


def test_guard_accepts_a_negative_axis_that_spans_zero():
    fig, ax = _axes(ylim=(-0.3, 0.2))
    try:
        ST.assert_no_truncated_axis(ax)
    finally:
        plt.close(fig)


def test_guard_fires_when_the_limits_clip_drawn_data():
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0.2, 1.8])
        ax.set_ylim(0.0, 1.0)
        with pytest.raises(ST.TruncatedAxisError, match="clip drawn data"):
            ST.assert_no_truncated_axis(ax)
        ST.assert_no_truncated_axis(ax, check_data=False)  # opt out is explicit
    finally:
        plt.close(fig)


def test_guard_sees_bar_heights_and_scatter_points():
    fig, ax = plt.subplots()
    try:
        ax.bar([0, 1], [0.5, 2.5])
        ax.set_ylim(0.0, 1.0)
        with pytest.raises(ST.TruncatedAxisError, match="clip drawn data"):
            ST.assert_no_truncated_axis(ax)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        ax.scatter([0.1, 0.2], [0.5, 3.0])
        ax.set_ylim(0.0, 1.0)
        with pytest.raises(ST.TruncatedAxisError, match="clip drawn data"):
            ST.assert_no_truncated_axis(ax)
    finally:
        plt.close(fig)


def test_guard_can_check_the_x_axis_too():
    fig, ax = _axes(xlim=(2.0, 5.0))
    try:
        with pytest.raises(ST.TruncatedAxisError, match="x-axis"):
            ST.assert_no_truncated_axis(ax, axis="x")
        ST.assert_no_truncated_axis(ax, axis="x", require_zero=False)
    finally:
        plt.close(fig)


def test_guard_rejects_an_unknown_axis_name():
    fig, ax = _axes()
    try:
        with pytest.raises(ValueError, match="axis must be"):
            ST.assert_no_truncated_axis(ax, axis="z")
    finally:
        plt.close(fig)


def test_guard_is_an_assertion_error_so_it_survives_a_broad_except():
    assert issubclass(ST.TruncatedAxisError, AssertionError)


# --------------------------------------------------------------------------------------
# Every figure calls the guard, and every accuracy axis starts at zero
# --------------------------------------------------------------------------------------


@pytest.fixture
def guard_spy(monkeypatch):
    """Record ``(axis, limits)`` for every guard call the figure builders make."""
    calls: list[tuple[str, tuple[float, float]]] = []
    real = ST.assert_no_truncated_axis

    def spy(ax, **kwargs):
        axis = kwargs.get("axis", "y")
        limits = ax.get_ylim() if axis == "y" else ax.get_xlim()
        calls.append((axis, (float(limits[0]), float(limits[1]))))
        return real(ax, **kwargs)

    monkeypatch.setattr(P, "assert_no_truncated_axis", spy)
    return calls


@pytest.mark.parametrize(
    "stem",
    [
        "fig01_accuracy_vs_position",
        "fig02_nll_vs_position",
        "fig03_edge_middle_gap",
        "fig07_absolute_vs_distance",
        "fig08_low_rank_explained_variance",
        "fig09_weight_delta_by_layer",
        "fig10_architecture_comparison",
        "fig11_significance_sequentiality",
        "fig12_accuracy_vs_latency",
        "fig13_worst_vs_mean_accuracy",
    ],
)
def test_every_value_figure_checks_its_axes(figure_calls, guard_spy, stem):
    fn, args, kwargs = figure_calls[stem]
    fn(*args, **kwargs)
    assert guard_spy, f"{stem} never called assert_no_truncated_axis"
    for axis, (lo, hi) in guard_spy:
        assert lo <= 0.0 <= hi, f"{stem}: {axis}-limits {(lo, hi)} exclude zero"


def test_a_narrow_accuracy_band_still_gets_a_full_zero_to_one_axis(tmp_path, guard_spy):
    """The failure mode section 47 exists to prevent: a 0.45-0.50 band magnified to fill."""
    rows = [
        {"model": m, "position": p, "accuracy": 0.45 + 0.01 * i}
        for m in ("qwen3", "halo")
        for i, p in enumerate((0.05, 0.5, 0.95))
    ]
    P.accuracy_vs_position(rows, tmp_path / "narrow", ci_low_col=None, ci_high_col=None)
    assert guard_spy
    (_axis, (lo, hi)) = guard_spy[0]
    assert lo == pytest.approx(0.0)
    assert hi >= 1.0


def test_nll_axis_is_zero_based_not_data_based(tmp_path, guard_spy, curve_rows):
    P.nll_vs_position(curve_rows, tmp_path / "nll", ci_low_col=None, ci_high_col=None)
    (_axis, (lo, hi)) = guard_spy[0]
    assert lo == pytest.approx(0.0)
    assert hi > 0.5


# --------------------------------------------------------------------------------------
# Limit helpers
# --------------------------------------------------------------------------------------


def test_set_zero_based_ylim_covers_zero_and_the_data():
    fig, ax = plt.subplots()
    try:
        lo, hi = ST.set_zero_based_ylim(ax, [0.4, 0.9, 0.6])
        assert lo == pytest.approx(0.0) and hi > 0.9
        lo, hi = ST.set_zero_based_ylim(ax, [-0.3, 0.2])
        assert lo < -0.3 and hi > 0.2
        lo, hi = ST.set_zero_based_ylim(ax, [0.0, 0.0])
        assert lo <= 0.0 < hi  # a degenerate series still yields a usable axis
    finally:
        plt.close(fig)


def test_set_accuracy_ylim_is_zero_to_one_and_widens_for_overflow():
    fig, ax = plt.subplots()
    try:
        lo, hi = ST.set_accuracy_ylim(ax, [0.1, 0.9])
        assert (lo, hi) == pytest.approx((0.0, 1.02))
        lo, hi = ST.set_accuracy_ylim(ax, [0.1, 1.4])
        assert hi > 1.4  # a CI band above 1 must not be clipped away
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------------
# Palette and style
# --------------------------------------------------------------------------------------


def test_panel_models_have_fixed_distinct_colours():
    assert set(ST.MODEL_COLORS) == set(MODELS)
    assert len(set(ST.MODEL_COLORS.values())) == len(ST.MODEL_COLORS)
    assert set(ST.MODEL_COLORS.values()) <= set(ST.PALETTE)
    assert len(set(ST.MODEL_MARKERS.values())) == len(ST.MODEL_MARKERS)


def test_assign_colors_pins_models_and_keeps_the_rest_distinct():
    assigned = ST.assign_colors(["halo", "arm_b", "arm_c", "qwen3"])
    assert assigned["halo"] == ST.MODEL_COLORS["halo"]
    assert assigned["qwen3"] == ST.MODEL_COLORS["qwen3"]
    assert len(set(assigned.values())) == 4
    assert assigned["arm_b"] not in {ST.MODEL_COLORS["halo"], ST.MODEL_COLORS["qwen3"]}


def test_colour_for_an_unknown_label_is_stable_across_calls():
    first = ST.color_for("psa_mode2_lambda0p5")
    assert first == ST.color_for("psa_mode2_lambda0p5")
    assert first in ST.PALETTE
    assert ST.marker_for("psa_mode2_lambda0p5") == ST.marker_for("psa_mode2_lambda0p5")


def test_yellow_is_excluded_from_the_line_palette():
    """Okabe-Ito's #F0E442 is colourblind-safe but unreadable as a line on white."""
    assert "#F0E442" not in ST.PALETTE


def test_panel_order_puts_the_registry_models_first():
    assert ST.panel_order(["arm_x", "nimbus", "qwen3"]) == ["qwen3", "nimbus", "arm_x"]


def test_display_name_falls_back_to_the_key():
    assert ST.display_name("halo") == MODELS["halo"].display_name
    assert ST.display_name("arm_c") == "arm_c"


def test_figure_style_does_not_leak_into_global_rcparams():
    before = plt.rcParams["axes.grid"]
    with ST.figure_style(**{"axes.grid": not before}):
        pass
    assert plt.rcParams["axes.grid"] == before


def test_style_manifest_is_json_serialisable():
    payload = json.loads(json.dumps(dict(ST.style_manifest())))
    assert payload["dpi"] == 200
    assert payload["formats"] == ["png", "pdf"]
    assert payload["no_truncated_axes"] is True


# --------------------------------------------------------------------------------------
# save_figure
# --------------------------------------------------------------------------------------


def test_save_figure_replaces_the_suffix_and_makes_the_directory(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = ST.save_figure(fig, tmp_path / "nested" / "deeper" / "plot.png")
    assert [p.name for p in written] == ["plot.png", "plot.pdf"]
    assert all(p.exists() for p in written)
    assert plt.get_fignums() == []


def test_save_figure_honours_a_single_format(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = ST.save_figure(fig, tmp_path / "one", formats=("png",))
    assert len(written) == 1 and written[0].suffix == ".png"


def test_save_figure_requires_at_least_one_format(tmp_path):
    fig, _ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="at least one output format"):
            ST.save_figure(fig, tmp_path / "none", formats=())
    finally:
        plt.close(fig)


def test_save_figure_refuses_to_write_inside_a_checkpoint():
    fig, _ax = plt.subplots()
    nested = MODELS["qwen3"].path / "psa_figures" / "figure"
    try:
        with pytest.raises(PermissionError, match="refusing to write inside source checkpoint"):
            ST.save_figure(fig, nested)
    finally:
        plt.close(fig)
    assert not nested.with_suffix(".png").exists()
    # the guard runs before mkdir, so nothing is left behind inside the read-only tree
    assert not nested.parent.exists()


# --------------------------------------------------------------------------------------
# Tidy-data adapters
# --------------------------------------------------------------------------------------


def test_columns_accepts_rows_mappings_and_dataframes(curve_rows, tmp_path):
    from_rows = P._columns(curve_rows)
    as_mapping = {name: list(values) for name, values in from_rows.items()}
    from_mapping = P._columns(as_mapping)
    for name, values in from_rows.items():
        assert np.array_equal(values.astype(str), from_mapping[name].astype(str))

    pd = pytest.importorskip("pandas")
    from_frame = P._columns(pd.DataFrame(curve_rows))
    assert set(from_frame) == set(from_rows)
    assert np.allclose(from_frame["accuracy"].astype(float), from_rows["accuracy"].astype(float))

    # and all three drive the same figure
    for source in (curve_rows, as_mapping, pd.DataFrame(curve_rows)):
        written = P.accuracy_vs_position(source, tmp_path / f"f{id(source)}")
        assert written[0].stat().st_size > MIN_PNG_BYTES


def test_columns_rejects_an_unsupported_container():
    with pytest.raises(TypeError, match="expected a DataFrame"):
        P._columns([1, 2, 3])


def test_a_missing_column_names_the_available_ones(gap_rows, tmp_path):
    with pytest.raises(KeyError, match="available"):
        P.edge_middle_gap_by_model(gap_rows, tmp_path / "f", value_col="not_a_column")


def test_curves_render_without_confidence_columns(curve_rows, tmp_path):
    written = P.accuracy_vs_position(
        curve_rows, tmp_path / "no_ci", ci_low_col=None, ci_high_col=None
    )
    assert written[0].stat().st_size > MIN_PNG_BYTES


# --------------------------------------------------------------------------------------
# Matrix figures
# --------------------------------------------------------------------------------------


def test_heatmap_rejects_a_transposed_matrix(tmp_path, matrices):
    transposed = matrices["layer_position"].T
    with pytest.raises(ValueError, match="transposed"):
        P.layer_position_heatmap(transposed, tmp_path / "f", positions=POSITIONS)


def test_heatmap_rejects_a_label_count_mismatch(tmp_path, matrices):
    with pytest.raises(ValueError, match="layer labels"):
        P.layer_position_heatmap(
            matrices["layer_position"], tmp_path / "f", positions=POSITIONS, layers=[0, 1]
        )
    with pytest.raises(ValueError, match="KV-head labels"):
        P.kv_head_position_heatmap(
            matrices["kv_position"], tmp_path / "f", positions=POSITIONS, kv_heads=[0]
        )
    with pytest.raises(ValueError, match="do not match matrix"):
        P.layer_kv_head_significance(matrices["layer_kv"], tmp_path / "f", kv_heads=[0, 1])


def test_heatmap_rejects_a_one_dimensional_input(tmp_path):
    with pytest.raises(ValueError, match="expected a 2-D matrix"):
        P.layer_position_heatmap(np.zeros(7), tmp_path / "f", positions=POSITIONS)


def test_signed_matrices_get_a_symmetric_scale(matrices):
    lo, hi = ST.symmetric_norm(matrices["layer_position"])
    assert lo == pytest.approx(-hi)
    assert hi >= float(np.abs(matrices["layer_position"]).max())
    assert ST.symmetric_norm(np.zeros((2, 2))) == (-1.0, 1.0)  # degenerate but usable


def test_kv_head_heatmap_accepts_the_gqa_head_count(tmp_path, matrices):
    written = P.kv_head_position_heatmap(
        matrices["kv_position"], tmp_path / "kv", positions=POSITIONS, kv_heads=list(range(N_KV))
    )
    assert len(written) == 2


# --------------------------------------------------------------------------------------
# Figure-specific input flexibility
# --------------------------------------------------------------------------------------


def test_low_rank_accepts_a_mapping_a_bare_array_and_a_tidy_frame(tmp_path, explained_variance):
    assert P.low_rank_explained_variance(explained_variance, tmp_path / "a")[0].exists()
    assert P.low_rank_explained_variance(
        np.array([0.5, 0.25, 0.15, 0.1]), tmp_path / "b"
    )[0].exists()
    rows = [
        {"model": model, "component": i + 1, "explained_variance_ratio": float(v)}
        for model, values in explained_variance.items()
        for i, v in enumerate(values)
    ]
    assert P.low_rank_explained_variance(rows, tmp_path / "c")[0].exists()


def test_low_rank_rejects_an_empty_input(tmp_path):
    with pytest.raises(ValueError, match="no explained-variance series"):
        P.low_rank_explained_variance({}, tmp_path / "empty")


def test_weight_delta_ignores_stack_level_rows(tmp_path, weight_delta_rows, guard_spy):
    """``layer == -1`` marks embed/norm tensors, which have no place on a per-layer x axis."""
    P.weight_delta_by_layer(weight_delta_rows, tmp_path / "wd")
    assert guard_spy and all(lo <= 0.0 for _axis, (lo, _hi) in guard_spy)


def test_weight_delta_requires_at_least_one_named_family(tmp_path, weight_delta_rows):
    with pytest.raises(ValueError, match="none of"):
        P.weight_delta_by_layer(weight_delta_rows, tmp_path / "f", families=("mlp_up",))


def test_architecture_comparison_handles_a_single_metric(tmp_path, arm_rows):
    rows = [dict(r) for r in arm_rows if r["metric"] == "accuracy"]
    for row in rows:
        row.pop("metric")
    written = P.architecture_comparison(rows, tmp_path / "arch", group_col=None)
    assert written[0].stat().st_size > MIN_PNG_BYTES


def test_interaction_annotates_the_contrast(tmp_path, interaction_rows):
    plain = P.significance_sequentiality_interaction(interaction_rows, tmp_path / "plain")
    annotated = P.significance_sequentiality_interaction(
        interaction_rows, tmp_path / "annotated", contrast=0.148, contrast_ci=(0.02, 0.27)
    )
    assert annotated[0].stat().st_size != plain[0].stat().st_size


def test_scatters_label_each_point(tmp_path, latency_rows, robustness_rows):
    assert P.accuracy_vs_latency(latency_rows, tmp_path / "lat")[0].exists()
    assert P.worst_vs_mean_accuracy(robustness_rows, tmp_path / "rob")[0].exists()


def test_latency_axis_starts_at_zero(tmp_path, latency_rows, guard_spy):
    P.accuracy_vs_latency(latency_rows, tmp_path / "lat")
    axes_checked = {axis for axis, _lim in guard_spy}
    assert axes_checked == {"x", "y"}
    for _axis, (lo, _hi) in guard_spy:
        assert lo == pytest.approx(0.0)
