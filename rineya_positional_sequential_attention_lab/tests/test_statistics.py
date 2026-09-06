"""Tests for ``rineya_psa.statistics``: clustered BCa bootstrap, McNemar, provenance.

The statistical claims are checked against independent implementations where one exists
(``scipy.stats.bootstrap`` for BCa, ``scipy.stats.binomtest`` for the exact McNemar tail)
and by simulation where one does not (interval coverage, cluster inflation).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from rineya_psa.evaluation.metrics import edge_middle_gap
from rineya_psa.statistics import bootstrap as B
from rineya_psa.statistics import provenance as PV

# --------------------------------------------------------------------------------------
# paired_bootstrap
# --------------------------------------------------------------------------------------


def test_paired_bootstrap_point_estimate_and_bookkeeping():
    a = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    b = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    scenarios = ["s0", "s0", "s1", "s1", "s2", "s2"]

    result = B.paired_bootstrap(a, b, scenarios, n=2000, seed=11)
    assert result.estimate == pytest.approx(np.mean(np.array(b) - np.array(a)))
    assert result.mean_a == pytest.approx(0.5)
    assert result.mean_b == pytest.approx(5 / 6)
    assert result.n_items == 6
    assert result.n_scenarios == 3
    assert result.seed == 11
    assert result.n_resamples == 2000
    assert result.ci_level == 0.95
    assert result.ci_low <= result.estimate <= result.ci_high
    assert result.sd > 0
    assert result.method.startswith(("bca", "percentile"))


def test_paired_bootstrap_is_reproducible_and_seed_sensitive():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 120)
    b = a + rng.normal(0.3, 1, 120)
    scenarios = [f"s{i // 4}" for i in range(120)]

    first = B.paired_bootstrap(a, b, scenarios, n=3000, seed=5)
    same = B.paired_bootstrap(a, b, scenarios, n=3000, seed=5)
    other = B.paired_bootstrap(a, b, scenarios, n=3000, seed=6)

    assert first.to_json() == same.to_json()
    assert (first.ci_low, first.ci_high) != (other.ci_low, other.ci_high)
    assert first.estimate == pytest.approx(other.estimate)  # the point estimate is not random


def test_clustered_resampling_widens_the_interval_when_variants_are_correlated():
    """Twelve positional variants of one scenario are not twelve independent items."""
    rng = np.random.default_rng(3)
    n_scen, n_var = 24, 12
    scenario_effect = rng.normal(0.4, 0.9, n_scen)
    diffs = np.concatenate(
        [effect + rng.normal(0, 0.05, n_var) for effect in scenario_effect]
    )
    scenarios = [f"s{i}" for i in range(n_scen) for _ in range(n_var)]
    a = np.zeros_like(diffs)

    clustered = B.paired_bootstrap(a, diffs, scenarios, n=4000, seed=2)
    naive = B.paired_bootstrap(a, diffs, None, n=4000, seed=2)

    assert clustered.n_scenarios == n_scen
    assert naive.n_scenarios == n_scen * n_var
    clustered_width = clustered.ci_high - clustered.ci_low
    naive_width = naive.ci_high - naive.ci_low
    assert clustered_width > 3 * naive_width
    # Cluster-level dz is the honest one; the item-level version is inflated here.
    assert abs(clustered.cohens_dz) < abs(clustered.cohens_dz_item)


def test_cohens_dz_matches_the_hand_computation():
    a = [0.0, 0.0, 0.0, 0.0]
    b = [1.0, 2.0, 3.0, 4.0]
    scenarios = ["s0", "s0", "s1", "s1"]
    result = B.paired_bootstrap(a, b, scenarios, n=500, seed=1)

    cluster_means = np.array([1.5, 3.5])
    assert result.cohens_dz == pytest.approx(
        cluster_means.mean() / cluster_means.std(ddof=1)
    )
    diff = np.array(b)
    assert result.cohens_dz_item == pytest.approx(diff.mean() / diff.std(ddof=1))
    assert result.sd_diff == pytest.approx(diff.std(ddof=1))


def test_paired_bootstrap_handles_a_constant_difference_without_crashing():
    result = B.paired_bootstrap([1.0] * 8, [2.0] * 8, [f"s{i // 2}" for i in range(8)], n=500)
    assert result.estimate == pytest.approx(1.0)
    assert result.ci_low == pytest.approx(1.0)
    assert result.ci_high == pytest.approx(1.0)
    # The mid-p bias correction keeps BCa defined on a point mass (z0 = 0, a = 0), so the
    # interval collapses onto the estimate instead of raising or falling back.
    assert result.method == "bca"
    assert result.sd == 0.0
    assert result.cohens_dz == 0.0


def test_paired_bootstrap_input_validation():
    with pytest.raises(ValueError):
        B.paired_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        B.paired_bootstrap([], [])
    with pytest.raises(ValueError):
        B.paired_bootstrap([1.0], [float("nan")])
    with pytest.raises(ValueError):
        B.paired_bootstrap([1.0, 2.0], [1.0, 2.0], ci=1.5)
    with pytest.raises(ValueError):
        B.paired_bootstrap([1.0, 2.0], [1.0, 2.0], ["only-one"])


def test_bca_agrees_with_scipy_on_the_iid_mean():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(17)
    # Skewed data, where BCa and the percentile interval genuinely differ.
    diff = rng.lognormal(0.0, 0.6, 150) - 1.0

    ours = B.paired_bootstrap(np.zeros_like(diff), diff, None, n=20000, seed=42)
    theirs = scipy_stats.bootstrap(
        (diff,),
        np.mean,
        n_resamples=20000,
        method="BCa",
        confidence_level=0.95,
        random_state=np.random.default_rng(43),
    )

    assert ours.method == "bca"
    tolerance = 4.0 * float(theirs.standard_error) / math.sqrt(20000) + 5e-3
    assert ours.ci_low == pytest.approx(float(theirs.confidence_interval.low), abs=tolerance)
    assert ours.ci_high == pytest.approx(float(theirs.confidence_interval.high), abs=tolerance)
    assert ours.sd == pytest.approx(float(theirs.standard_error), rel=0.05)


def test_bca_reduces_to_the_percentile_interval_when_unbiased_and_unaccelerated():
    boot = np.linspace(-1.0, 1.0, 20001)  # symmetric about 0
    jack = np.array([0.5, 0.5, 0.5, 0.5])  # zero spread -> zero acceleration
    lo, hi, method = B._bca_interval(0.0, boot, jack, 0.95)
    assert method == "bca"
    assert lo == pytest.approx(np.quantile(boot, 0.025), abs=1e-3)
    assert hi == pytest.approx(np.quantile(boot, 0.975), abs=1e-3)


def test_bca_falls_back_and_labels_the_fallback():
    boot = np.linspace(1.0, 2.0, 1001)
    # theta_hat below every resample: the bias correction is undefined.
    lo, hi, method = B._bca_interval(0.0, boot, np.array([1.0, 2.0, 3.0]), 0.95)
    assert method == "percentile_degenerate_bias"
    assert lo == pytest.approx(np.quantile(boot, 0.025))
    assert hi == pytest.approx(np.quantile(boot, 0.975))

    # An extreme acceleration drives the BCa denominator through zero.
    skewed = np.concatenate([np.zeros(500), np.linspace(0.0, 1.0, 500)])
    jack = np.array([0.0, 0.0, 0.0, 1000.0])
    _, _, method = B._bca_interval(0.2, skewed, jack, 0.95)
    assert method in {"bca", "percentile_degenerate_acceleration"}


def test_paired_bootstrap_interval_coverage_is_near_nominal():
    """A slow-ish simulation, but it is the only way to know the CI means what it says."""
    rng = np.random.default_rng(1234)
    truth, n_scen, n_var, trials = 0.25, 30, 4, 200
    covered = 0
    scenarios = [f"s{i}" for i in range(n_scen) for _ in range(n_var)]
    for trial in range(trials):
        effects = rng.normal(truth, 0.5, n_scen)
        diff = np.concatenate([e + rng.normal(0, 0.3, n_var) for e in effects])
        result = B.paired_bootstrap(
            np.zeros_like(diff), diff, scenarios, n=600, seed=1000 + trial
        )
        covered += int(result.ci_low <= truth <= result.ci_high)
    assert 0.87 <= covered / trials <= 1.0, f"coverage {covered / trials:.3f}"


# --------------------------------------------------------------------------------------
# mcnemar_exact
# --------------------------------------------------------------------------------------


def test_mcnemar_exact_matches_scipy_binomtest():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(9)
    for _ in range(20):
        a = rng.random(60) < 0.55
        b = rng.random(60) < 0.70
        result = B.mcnemar_exact(a, b)
        if result.discordant == 0:
            assert result.p_value == 1.0
            continue
        reference = scipy_stats.binomtest(
            result.b_only_correct, result.discordant, 0.5, alternative="two-sided"
        ).pvalue
        assert result.p_value == pytest.approx(float(reference), rel=1e-9, abs=1e-12)


def test_mcnemar_exact_counts_and_edge_cases():
    result = B.mcnemar_exact([True, False, True, False], [True, True, False, False])
    assert result.n_pairs == 4
    assert result.a_only_correct == 1
    assert result.b_only_correct == 1
    assert result.discordant == 2
    assert result.p_value == pytest.approx(1.0)

    identical = B.mcnemar_exact([True, False], [True, False])
    assert identical.discordant == 0
    assert identical.p_value == 1.0

    lopsided = B.mcnemar_exact([False] * 12, [True] * 12)
    assert lopsided.b_only_correct == 12
    assert lopsided.p_value == pytest.approx(2 * 0.5**12)

    with pytest.raises(ValueError):
        B.mcnemar_exact([], [])
    with pytest.raises(ValueError):
        B.mcnemar_exact([True], [True, False])


def test_summarize_paired_binary_carries_both_views():
    a = [True] * 10 + [False] * 10
    b = [True] * 16 + [False] * 4
    summary = B.summarize_paired_binary(a, b, [f"s{i // 2}" for i in range(20)], n=1000, seed=4)
    assert summary["bootstrap"]["estimate"] == pytest.approx(0.3)
    assert summary["mcnemar"]["discordant"] == 6
    assert 0.0 <= summary["mcnemar"]["p_value"] <= 1.0


# --------------------------------------------------------------------------------------
# interaction_contrast
# --------------------------------------------------------------------------------------


def test_interaction_contrast_point_estimate_and_components():
    rng = np.random.default_rng(21)
    n = 80
    seq = rng.normal(0.5, 0.1, n)
    par = rng.normal(0.5, 0.1, n)
    seq_sm = seq + 0.20 + rng.normal(0, 0.02, n)
    par_sm = par + 0.05 + rng.normal(0, 0.02, n)
    scenarios = [f"s{i // 4}" for i in range(n)]

    result = B.interaction_contrast(seq_sm, seq, par_sm, par, scenarios, n=4000, seed=8)
    assert result.seq_effect == pytest.approx(float(np.mean(seq_sm - seq)))
    assert result.par_effect == pytest.approx(float(np.mean(par_sm - par)))
    assert result.estimate == pytest.approx(result.seq_effect - result.par_effect)
    assert result.estimate == pytest.approx(0.15, abs=0.02)
    assert result.n_scenarios == 20
    assert result.n_items == n
    assert result.significant  # the interval excludes zero for a 0.15 interaction


def test_interaction_contrast_null_interval_covers_zero():
    rng = np.random.default_rng(22)
    n = 120
    seq = rng.normal(0.5, 0.2, n)
    par = rng.normal(0.5, 0.2, n)
    # The same +0.1 boost on both arms: an effect, but no interaction.
    result = B.interaction_contrast(
        seq + 0.1, seq, par + 0.1, par, [f"s{i // 3}" for i in range(n)], n=3000, seed=13
    )
    assert result.estimate == pytest.approx(0.0, abs=1e-9)
    assert result.ci_low <= 0.0 <= result.ci_high
    assert not result.significant


def test_interaction_contrast_rejects_ragged_arms():
    with pytest.raises(ValueError, match="same length"):
        B.interaction_contrast([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0])


# --------------------------------------------------------------------------------------
# bootstrap_ci_of_statistic
# --------------------------------------------------------------------------------------


def _gap_of(rows):
    """edge_middle_gap over a resampled record list -- a statistic of the whole set."""
    by_pos: dict[float, list[float]] = {}
    for row in rows:
        by_pos.setdefault(row["position"], []).append(float(row["correct"]))
    return edge_middle_gap({p: float(np.mean(v)) for p, v in by_pos.items()})


def test_bootstrap_ci_of_statistic_gives_a_derived_metric_an_interval():
    rng = np.random.default_rng(31)
    positions = (0.02, 0.5, 0.98)
    rates = {0.02: 0.9, 0.5: 0.4, 0.98: 0.85}
    rows = [
        {"scenario": f"s{i}", "position": p, "correct": bool(rng.random() < rates[p])}
        for i in range(60)
        for p in positions
    ]

    result = B.bootstrap_ci_of_statistic(
        _gap_of, rows, cluster_ids=[r["scenario"] for r in rows], n=800, seed=77
    )
    assert result.estimate == pytest.approx(_gap_of(rows))
    assert result.n_items == 180
    assert result.n_scenarios == 60
    assert result.ci_low < result.estimate < result.ci_high
    assert result.ci_low > 0  # the U shape is real in this synthetic curve
    assert result.seed == 77

    repeat = B.bootstrap_ci_of_statistic(
        _gap_of, rows, cluster_ids=[r["scenario"] for r in rows], n=800, seed=77
    )
    assert repeat.to_json() == result.to_json()


def test_bootstrap_ci_of_statistic_validates_and_reports_undefined_resamples():
    with pytest.raises(ValueError, match="non-empty"):
        B.bootstrap_ci_of_statistic(lambda rows: 0.0, [])
    with pytest.raises(ValueError, match="undefined on the full data"):
        B.bootstrap_ci_of_statistic(lambda rows: float("nan"), [1, 2, 3], n=10)

    def sometimes_undefined(rows):
        if len(set(rows)) < 2:
            raise ValueError("needs two distinct values")
        return float(np.mean(rows))

    result = B.bootstrap_ci_of_statistic(sometimes_undefined, [1.0, 2.0], n=200, seed=3)
    assert "partial" in result.method
    assert result.n_resamples < 200


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------


def test_stamp_contains_the_required_fields():
    payload = PV.stamp({"alpha": 1, "beta": [2, 3]})
    assert set(payload) >= {
        "generated_at_utc",
        "git",
        "versions",
        "gpu",
        "platform",
        "config",
        "config_sha256",
    }
    assert payload["generated_at_utc"].endswith("Z")
    assert payload["git"]["commit"] is None or len(payload["git"]["commit"]) == 40
    assert isinstance(payload["git"]["dirty"], (bool, type(None)))
    assert payload["versions"]["python"].startswith("3.")
    assert "numpy" in payload["versions"] and payload["versions"]["numpy"]
    assert set(payload["gpu"]) >= {"cuda_available", "name", "gpu_source"}
    assert payload["platform"]["system"]
    # The stamp must be JSON serialisable as written.
    json.loads(json.dumps(payload, default=PV.json_default))


def test_config_hash_is_order_insensitive_and_value_sensitive():
    assert PV.config_hash({"a": 1, "b": 2}) == PV.config_hash({"b": 2, "a": 1})
    assert PV.config_hash({"a": 1}) != PV.config_hash({"a": 2})
    assert PV.config_hash(None) == PV.config_hash({})
    assert len(PV.config_hash({"a": 1})) == 64


def test_write_result_emits_a_sidecar(tmp_path: Path):
    target = PV.write_result(
        {"estimate": 0.42, "seed": 7}, tmp_path / "runs" / "result.json", config={"arm": "A"}
    )
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["estimate"] == 0.42

    sidecar = target.with_name(target.name + ".provenance.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["config"] == {"arm": "A"}
    assert payload["config_sha256"] == PV.config_hash({"arm": "A"})
    assert payload["artifact"]["bytes"] == target.stat().st_size
    assert len(payload["artifact"]["sha256"]) == 64


def test_write_result_handles_jsonl_and_numpy_scalars(tmp_path: Path):
    rows = [{"i": i, "value": np.float64(i) / 4} for i in range(3)]
    target = PV.write_result(rows, tmp_path / "rows.jsonl")
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[2])["value"] == pytest.approx(0.5)


def test_write_result_writes_a_dataclass_result(tmp_path: Path):
    result = B.paired_bootstrap([0.0, 1.0], [1.0, 1.0], ["s0", "s1"], n=200, seed=1)
    target = PV.write_result(result, tmp_path / "paired.json", config={"n": 200})
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_scenarios"] == 2
    assert payload["seed"] == 1


def test_write_result_refuses_to_touch_a_checkpoint():
    from rineya_psa.paths import MODELS

    for key in ("qwen3", "supa_a11"):
        with pytest.raises(PermissionError):
            PV.write_result({"x": 1}, MODELS[key].path / "leak.json")
        with pytest.raises(PermissionError):
            PV.write_result({"x": 1}, MODELS[key].path / "nested" / "leak.json")


def test_write_result_rejects_an_unsupported_suffix(tmp_path: Path):
    with pytest.raises(ValueError, match="write_result handles"):
        PV.write_result({"x": 1}, tmp_path / "result.txt")


# --------------------------------------------------------------------------------------
# provenance: the config a sidecar carries, and whether it came from the caller
#
# Audit finding (a): the Phase-3 sidecars in results/ carry "config": {} and there is no
# field that says whether that means "this run had no config" or "the caller forgot".
# These tests pin the distinction.
# --------------------------------------------------------------------------------------


def test_write_result_without_a_config_warns_and_records_the_gap(tmp_path: Path):
    """Omitting ``config=`` must be visible *in the artifact*, not only in the terminal."""
    with pytest.warns(PV.ProvenanceWarning, match="no config= was passed"):
        target = PV.write_result({"x": 1}, tmp_path / "nocfg.json")

    payload = json.loads(PV.sidecar_path(target).read_text(encoding="utf-8"))
    assert payload["config"] == {}
    assert payload["config_source"] == "missing"


def test_an_explicitly_empty_config_is_not_the_same_as_a_forgotten_one(tmp_path: Path):
    """``config={}`` is an assertion that there was no config; it must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PV.ProvenanceWarning)
        empty = PV.write_result({"x": 1}, tmp_path / "empty.json", config={})
        none = PV.write_result({"x": 1}, tmp_path / "none.json", config=None)

    for target in (empty, none):
        payload = json.loads(PV.sidecar_path(target).read_text(encoding="utf-8"))
        assert payload["config"] == {}
        assert payload["config_source"] == "caller", target.name


def test_a_supplied_config_reaches_the_sidecar_verbatim(tmp_path: Path):
    cfg = {"model": "halo", "positions": [0.02, 0.5, 0.98], "seed": 20260828}
    target = PV.write_result({"x": 1}, tmp_path / "cfg.json", config=cfg)
    payload = json.loads(PV.sidecar_path(target).read_text(encoding="utf-8"))
    assert payload["config"] == cfg
    assert payload["config_sha256"] == PV.config_hash(cfg)
    assert payload["config_source"] == "caller"


def test_sidecar_path_appends_rather_than_replacing_the_suffix():
    """``a.parquet`` -> ``a.parquet.provenance.json``, never ``a.provenance.json``."""
    assert PV.sidecar_path("a/b.parquet").name == "b.parquet.provenance.json"
    assert PV.sidecar_path(Path("a/b.npz")).name == "b.npz.provenance.json"


# --------------------------------------------------------------------------------------
# provenance: write_sidecar, the helper for writers that serialise their own payload
#
# Audit finding (a): the primary Phase-3 artifacts
# (results/baseline_position_results_{calibration,validation}_tf.parquet) have no sidecar
# at all, because evaluation.runner.write_parquet hands the frame to pandas and never
# stamped the result. write_result cannot help those writers; write_sidecar can.
# --------------------------------------------------------------------------------------


def test_write_sidecar_stamps_a_file_written_by_someone_else(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    target = tmp_path / "rows.parquet"
    pd.DataFrame({"position": [0.02, 0.98], "score": [1.0, 0.0]}).to_parquet(
        target, index=False
    )

    sidecar = PV.write_sidecar(
        target, config={"split": "validation"}, artifact_extra={"rows": 2}
    )
    assert sidecar == PV.sidecar_path(target)

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["config"] == {"split": "validation"}
    assert payload["config_source"] == "caller"
    assert payload["artifact"]["bytes"] == target.stat().st_size
    assert payload["artifact"]["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert payload["artifact"]["rows"] == 2


def test_write_sidecar_refuses_to_stamp_a_file_that_was_never_written(tmp_path: Path):
    """A sidecar for a nonexistent artifact would be a provenance claim about nothing."""
    with pytest.raises(FileNotFoundError, match="was not written"):
        PV.write_sidecar(tmp_path / "ghost.parquet", config={})
    assert not PV.sidecar_path(tmp_path / "ghost.parquet").exists()


def test_write_sidecar_refuses_to_touch_a_checkpoint():
    from rineya_psa.paths import MODELS

    with pytest.raises(PermissionError):
        PV.write_sidecar(MODELS["qwen3"].path / "config.json", config={})


def test_write_sidecar_without_a_config_warns_like_write_result(tmp_path: Path):
    target = tmp_path / "plain.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.warns(PV.ProvenanceWarning, match="plain.json"):
        sidecar = PV.write_sidecar(target)
    assert json.loads(sidecar.read_text(encoding="utf-8"))["config_source"] == "missing"


# --------------------------------------------------------------------------------------
# provenance: git_state must let a reader date an artifact
#
# Audit finding (c): the published Phase-3 sidecar records commit 384f515 with
# "dirty": true and a flat dirty_paths list in which an untracked sibling lab
# (rineya_halo_long_context_sft_v1/) is indistinguishable from an edited source file
# (scripts/phase3_analyze.py). "dirty: true" was therefore true of every sidecar in
# results/ and carried no information.
# --------------------------------------------------------------------------------------

_GIT = shutil.which("git")
requires_git = pytest.mark.skipif(_GIT is None, reason="git is not on PATH")


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for args in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "t@example.invalid"),
        ("config", "user.name", "test"),
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _commit_all(root: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", message], cwd=root, check=True, capture_output=True
    )


@requires_git
def test_git_state_on_a_clean_tree_says_clean(tmp_path: Path):
    root = tmp_path / "repo"
    _init_repo(root)
    (root / "lab").mkdir()
    (root / "lab" / "src.py").write_text("x = 1\n", encoding="utf-8")
    _commit_all(root, "init")

    state = PV.git_state(root, scope="lab")
    assert state["dirty"] is False
    assert state["code_state"] == "clean"
    assert state["tracked_dirty_paths"] == []
    assert state["scope_dirty"] is False
    assert len(state["commit"]) == 40


@requires_git
def test_git_state_separates_edited_source_from_untracked_scratch(tmp_path: Path):
    """An untracked sibling directory must not read as "the code that ran was modified"."""
    root = tmp_path / "repo"
    _init_repo(root)
    (root / "lab").mkdir()
    (root / "lab" / "src.py").write_text("x = 1\n", encoding="utf-8")
    _commit_all(root, "init")

    (root / "lab" / "src.py").write_text("x = 2\n", encoding="utf-8")  # tracked edit
    (root / "scratch").mkdir()
    (root / "scratch" / "notes.md").write_text("hi\n", encoding="utf-8")  # untracked

    state = PV.git_state(root, scope="lab")
    assert state["dirty"] is True
    assert state["code_state"] == "dirty-tracked"
    assert state["tracked_dirty_paths"] == ["lab/src.py"]
    assert state["tracked_dirty_count"] == 1
    assert [p for p in state["untracked_paths"] if p.startswith("scratch")]
    assert "lab/src.py" not in state["untracked_paths"]
    assert state["scope"] == "lab"
    assert state["scope_dirty"] is True
    assert state["scope_dirty_paths"] == ["lab/src.py"]
    assert {e["path"] for e in state["dirty_entries"]} >= {"lab/src.py"}
    assert next(e for e in state["dirty_entries"] if e["path"] == "lab/src.py")["status"] == " M"


@requires_git
def test_untracked_only_tree_is_not_reported_as_modified_code(tmp_path: Path):
    root = tmp_path / "repo"
    _init_repo(root)
    (root / "lab").mkdir()
    (root / "lab" / "src.py").write_text("x = 1\n", encoding="utf-8")
    _commit_all(root, "init")
    (root / "other_lab").mkdir()
    (root / "other_lab" / "junk.txt").write_text("junk\n", encoding="utf-8")

    state = PV.git_state(root, scope="lab")
    assert state["dirty"] is True  # git status is non-empty ...
    assert state["code_state"] == "clean-tracked-untracked-present"  # ... but no code moved
    assert state["tracked_dirty_paths"] == []
    assert state["scope_dirty"] is False


@requires_git
def test_git_state_scope_ignores_a_dirty_file_in_another_lab(tmp_path: Path):
    root = tmp_path / "repo"
    _init_repo(root)
    for lab in ("lab", "other"):
        (root / lab).mkdir()
        (root / lab / "src.py").write_text("x = 1\n", encoding="utf-8")
    _commit_all(root, "init")
    (root / "other" / "src.py").write_text("x = 99\n", encoding="utf-8")

    state = PV.git_state(root, scope="lab")
    assert state["tracked_dirty_paths"] == ["other/src.py"]
    assert state["scope_dirty"] is False
    assert state["scope_dirty_paths"] == []


@requires_git
def test_diff_sha256_distinguishes_two_artifacts_stamped_at_the_same_commit(tmp_path: Path):
    """The field that makes a stale artifact detectable: same commit, different source."""
    root = tmp_path / "repo"
    _init_repo(root)
    (root / "lab").mkdir()
    (root / "lab" / "src.py").write_text("x = 1\n", encoding="utf-8")
    _commit_all(root, "init")

    clean = PV.git_state(root, scope="lab")
    (root / "lab" / "src.py").write_text("x = 2\n", encoding="utf-8")
    edited = PV.git_state(root, scope="lab")
    (root / "lab" / "src.py").write_text("x = 3\n", encoding="utf-8")
    edited_again = PV.git_state(root, scope="lab")

    assert clean["commit"] == edited["commit"] == edited_again["commit"]
    assert len(edited["diff_sha256"]) == 64
    assert clean["diff_sha256"] != edited["diff_sha256"]
    assert edited["diff_sha256"] != edited_again["diff_sha256"]
    assert clean["diff_bytes"] == 0
    assert edited["diff_bytes"] > 0

    (root / "lab" / "src.py").write_text("x = 1\n", encoding="utf-8")  # back to committed
    assert PV.git_state(root, scope="lab")["diff_sha256"] == clean["diff_sha256"]


@requires_git
def test_git_state_records_a_renamed_file_by_its_destination(tmp_path: Path):
    root = tmp_path / "repo"
    _init_repo(root)
    (root / "lab").mkdir()
    (root / "lab" / "old.py").write_text("x = 1\n", encoding="utf-8")
    _commit_all(root, "init")
    subprocess.run(
        ["git", "mv", "lab/old.py", "lab/new.py"], cwd=root, check=True, capture_output=True
    )

    state = PV.git_state(root, scope="lab")
    assert "lab/new.py" in state["tracked_dirty_paths"]
    assert not any(" -> " in p for p in state["tracked_dirty_paths"])


def test_stamp_embeds_the_working_tree_description_and_determinism_record():
    """SPEC 9 + SPEC 10: a sidecar answers "what code ran" and "what was seeded"."""
    payload = PV.stamp({"a": 1})
    git = payload["git"]
    assert set(git) >= {
        "code_state",
        "diff_sha256",
        "scope",
        "scope_dirty",
        "tracked_dirty_paths",
        "untracked_paths",
    }
    assert git["code_state"] in {
        "clean",
        "clean-tracked-untracked-present",
        "dirty-tracked",
        "unknown",
    }
    assert payload["determinism"]["spec_section"] == "10"
    json.loads(json.dumps(payload, default=PV.json_default))
