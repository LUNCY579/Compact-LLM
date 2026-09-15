"""Gates for the cluster-robust paired layer.

Hermetic and fully deterministic. The load-bearing tests are
`test_the_design_refuses_significance_when_it_cannot_reach_alpha` and
`test_clustering_changes_the_answer_when_rows_are_not_independent`: the first is the guard that
stops an underpowered design reporting a p-value as if it meant something, the second proves the
whole module is not decoration.
"""
import pytest

from experiments.reasoning_posttraining_v1 import statistics as st


def rows_for(clusters_spec, *, key="cluster_id"):
    """clusters_spec: {cluster: n_rows}. Returns rows plus their index ranges."""
    rows = []
    for cluster, n in clusters_spec.items():
        for i in range(n):
            rows.append({"id": f"{cluster}:{i}", key: cluster})
    return rows


# ------------------------------------------------------------------ resolution floor

def test_the_design_refuses_significance_when_it_cannot_reach_alpha():
    """With 4 clusters the smallest two-sided p is 0.125, so 0.05 is unreachable."""
    got = st.resolution_floor(4)
    assert got["smallest_two_sided_p"] == pytest.approx(0.125)
    assert got["can_reject_at_alpha"] is False
    assert "REFUSED" in got["verdict"]

    # A perfect, maximal effect on all 4 clusters still must not be called significant.
    rows = rows_for({"a": 10, "b": 10, "c": 10, "d": 10})
    baseline = [0] * 40
    arm = [1] * 40
    out = st.compare(rows, baseline, arm, cluster_key="cluster_id", margin=0.0)
    assert out["estimates"]["cluster_weighted_difference"] == 1.0
    assert out["exceeds_practical_margin"] is True
    assert out["permutation"]["significance_refused"] is True
    assert out["claim_supported"] is False
    assert "cannot reach alpha" in out["why_not_supported"]


def test_eight_clusters_can_just_reach_alpha():
    """The Forge banks' family-level unit: 2/2^8 = 0.0078."""
    got = st.resolution_floor(8)
    assert got["smallest_two_sided_p"] == pytest.approx(2 / 256)
    assert got["can_reject_at_alpha"] is True


def test_sixteen_clusters_is_the_preservation_sub_bank_b_case():
    got = st.resolution_floor(16)
    assert got["can_reject_at_alpha"] is True
    assert got["smallest_two_sided_p"] < 1e-4


# ------------------------------------------------------------------ the point of clustering

def test_clustering_changes_the_answer_when_rows_are_not_independent():
    """192 rows in 16 clusters must not behave like 192 independent rows.

    One cluster flips entirely and the rest do nothing. Row-level thinking sees 12 of 192 rows
    change; cluster-level thinking sees 1 of 16 units change, which is not distinguishable from
    noise. The second reading is the correct one.
    """
    spec = {f"c{i}": 12 for i in range(16)}
    rows = rows_for(spec)
    baseline = [0] * len(rows)
    arm = [1 if r["cluster_id"] == "c0" else 0 for r in rows]
    out = st.compare(rows, baseline, arm, cluster_key="cluster_id", margin=0.0)
    assert out["estimates"]["rows"] == 192
    assert out["estimates"]["clusters"] == 16
    assert out["estimates"]["arm_correct"] == 12
    assert out["estimates"]["clusters_with_no_change"] == 15
    # 1/16 clusters moved: the exact test must not reject.
    assert out["permutation"]["p_value"] > st.ALPHA
    assert out["claim_supported"] is False


def test_a_consistent_effect_across_clusters_is_detected():
    spec = {f"c{i}": 10 for i in range(16)}
    rows = rows_for(spec)
    baseline = [0] * len(rows)
    # Every cluster improves by 3/10: consistent, so the sign-flip test should reject.
    arm = []
    for r in rows:
        arm.append(1 if int(r["id"].split(":")[1]) < 3 else 0)
    out = st.compare(rows, baseline, arm, cluster_key="cluster_id", margin=0.0)
    assert out["estimates"]["cluster_weighted_difference"] == pytest.approx(0.3)
    assert out["permutation"]["p_value"] == pytest.approx(2 / 2 ** 16)
    assert out["permutation"]["significant_at_alpha"] is True
    assert out["claim_supported"] is True


def test_both_weightings_are_reported_and_can_disagree():
    """Unbalanced clusters: the row-weighted and cluster-weighted answers differ in sign."""
    rows = rows_for({"big": 18, "s1": 1, "s2": 1, "s3": 1})
    baseline = [0] * 21
    # The big cluster improves a lot; the three singletons each get worse.
    arm = [1 if r["cluster_id"] == "big" else 0 for r in rows]
    baseline = [0 if r["cluster_id"] == "big" else 1 for r in rows]
    est = st.point_estimates(st.paired_clusters(rows, baseline, arm,
                                                cluster_key="cluster_id"))
    assert est["row_weighted_difference"] > 0
    assert est["cluster_weighted_difference"] < 0
    assert est["largest_cluster_rows"] == 18


# ------------------------------------------------------------------ exactness and determinism

def test_exact_enumeration_is_used_below_the_threshold_and_is_reproducible():
    rows = rows_for({f"c{i}": 4 for i in range(10)})
    baseline = [0] * 40
    arm = [1 if int(r["id"].split(":")[1]) < 1 else 0 for r in rows]
    first = st.compare(rows, baseline, arm, cluster_key="cluster_id")
    second = st.compare(rows, baseline, arm, cluster_key="cluster_id")
    assert first["permutation"]["method"] == "exact_enumeration"
    assert first["permutation"]["draws"] == 2 ** 10
    assert first["permutation"]["p_value"] == second["permutation"]["p_value"]
    assert first["permutation"]["seed"] is None


def test_monte_carlo_is_seeded_reproducible_and_never_reports_p_zero():
    spec = {f"c{i}": 2 for i in range(30)}
    rows = rows_for(spec)
    baseline = [0] * 60
    arm = [1] * 60
    first = st.cluster_permutation_test(
        st.paired_clusters(rows, baseline, arm, cluster_key="cluster_id"), draws=2000)
    second = st.cluster_permutation_test(
        st.paired_clusters(rows, baseline, arm, cluster_key="cluster_id"), draws=2000)
    assert first["method"] == "monte_carlo_lcg"
    assert first["p_value"] == second["p_value"]
    assert first["p_value"] > 0, "add-one smoothing must prevent a reported p of exactly 0"
    assert first["seed"] == 20260915


def test_no_difference_gives_a_p_value_of_one():
    rows = rows_for({f"c{i}": 5 for i in range(8)})
    outcome = [1, 0, 1, 0, 0] * 8
    out = st.compare(rows, outcome, outcome, cluster_key="cluster_id")
    assert out["estimates"]["cluster_weighted_difference"] == 0.0
    assert out["permutation"]["p_value"] == 1.0
    assert out["claim_supported"] is False


# ------------------------------------------------------------------ guards

def test_mismatched_lengths_are_refused():
    rows = rows_for({"a": 3})
    with pytest.raises(ValueError, match="same length"):
        st.paired_clusters(rows, [1, 1, 1], [1, 1], cluster_key="cluster_id")


def test_a_missing_cluster_key_is_refused_rather_than_defaulted():
    """The cluster unit must live on the data, not be invented at analysis time."""
    rows = [{"id": "x"}]
    with pytest.raises(ValueError, match="carries no 'cluster_id'"):
        st.paired_clusters(rows, [1], [1], cluster_key="cluster_id")


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        st.paired_clusters([], [], [], cluster_key="cluster_id")


def test_every_registered_bank_declares_its_cluster_unit_and_rationale():
    for bank, spec in st.CLUSTER_UNITS.items():
        assert spec["primary"] and spec["secondary"], bank
        assert spec["note"].strip(), bank
    assert st.CLUSTER_UNITS["RRP2-PRES-DEV"]["primary"] == "cluster_id"
    assert st.CLUSTER_UNITS["RRP2-DEV-V2"]["primary"] == "family"


# ------------------------------------------------------------------ non-inferiority

def test_non_inferiority_needs_more_than_a_point_estimate_above_the_margin():
    """A tiny loss on few clusters must not be called preserved."""
    rows = rows_for({f"c{i}": 10 for i in range(6)})
    baseline = [1 if int(r["id"].split(":")[1]) < 5 else 0 for r in rows]
    arm = list(baseline)
    out = st.non_inferiority(rows, baseline, arm, cluster_key="cluster_id", margin=0.03)
    assert out["estimates"]["cluster_weighted_difference"] == 0.0
    assert "not evidence of preservation" in out["note"]


def test_non_inferiority_rejects_when_the_arm_clearly_holds_up():
    rows = rows_for({f"c{i}": 10 for i in range(16)})
    baseline = [1 if int(r["id"].split(":")[1]) < 5 else 0 for r in rows]
    arm = list(baseline)
    out = st.non_inferiority(rows, baseline, arm, cluster_key="cluster_id", margin=0.05)
    # Zero loss against a 0.05 margin, consistently across 16 clusters.
    assert out["permutation"]["one_sided"] is True
    assert out["permutation"]["shift"] == -0.05
    assert out["non_inferiority_supported"] is True


def test_a_real_regression_fails_non_inferiority():
    rows = rows_for({f"c{i}": 10 for i in range(16)})
    baseline = [1] * len(rows)
    arm = [0 if int(r["id"].split(":")[1]) < 4 else 1 for r in rows]   # -0.40 everywhere
    out = st.non_inferiority(rows, baseline, arm, cluster_key="cluster_id", margin=0.03)
    assert out["estimates"]["cluster_weighted_difference"] == pytest.approx(-0.4)
    assert out["non_inferiority_supported"] is False


# ------------------------------------------------------------------ minimal pairs

def pair_rows():
    rows = []
    for i in range(5):
        rows.append({"id": f"a{i}", "pair_id": f"p{i}", "answerable": False})
        rows.append({"id": f"b{i}", "pair_id": f"p{i}", "answerable": True})
    return rows


def test_minimal_pair_contrast_names_the_four_outcomes():
    rows = pair_rows()
    sensitive = [True, False] * 5          # abstains on unanswerable, answers answerable
    got = st.minimal_pair_contrast(rows, [], abstained=sensitive)
    assert got["complete_pairs"] == 5
    assert got["evidence_sensitive"] == 5
    assert got["evidence_sensitivity_rate"] == 1.0

    always = [True, True] * 5
    got = st.minimal_pair_contrast(rows, [], abstained=always)
    assert got["abstains_regardless"] == 5 and got.get("evidence_sensitive", 0) == 0

    never = [False, False] * 5
    got = st.minimal_pair_contrast(rows, [], abstained=never)
    assert got["answers_regardless"] == 5

    inverted = [False, True] * 5
    got = st.minimal_pair_contrast(rows, [], abstained=inverted)
    assert got["inverted"] == 5


def test_incomplete_pairs_are_excluded_not_guessed():
    rows = pair_rows() + [{"id": "lone", "pair_id": "p99", "answerable": True}]
    got = st.minimal_pair_contrast(rows, [], abstained=[True, False] * 5 + [False])
    assert got["complete_pairs"] == 5
    assert got["incomplete_pairs_excluded"] == 1


def test_rows_without_a_pair_id_are_ignored_by_the_pair_contrast():
    rows = pair_rows() + [{"id": "solo", "pair_id": None, "answerable": True}]
    got = st.minimal_pair_contrast(rows, [], abstained=[True, False] * 5 + [False])
    assert got["complete_pairs"] == 5 and got["incomplete_pairs_excluded"] == 0


# ------------------------------------------------------------------ multiplicity

def test_holm_is_step_down_and_stops_at_the_first_failure():
    got = st.holm({"a": 0.001, "b": 0.02, "c": 0.9})
    assert got["results"]["a"]["rejected"] is True      # 0.001 <= 0.05/3
    assert got["results"]["c"]["rejected"] is False
    assert "a" in got["rejected"] and "c" not in got["rejected"]


def test_holm_does_not_reject_a_borderline_p_that_survives_alone():
    """0.04 clears alpha on its own but not under a family of three."""
    alone = st.holm({"a": 0.04})
    assert alone["results"]["a"]["rejected"] is True
    family = st.holm({"a": 0.04, "b": 0.5, "c": 0.6})
    assert family["results"]["a"]["rejected"] is False


# ------------------------------------------------------------------ hygiene

def test_nothing_here_permits_a_quality_claim():
    rows = rows_for({f"c{i}": 4 for i in range(8)})
    out = st.compare(rows, [0] * 32, [1] * 32, cluster_key="cluster_id")
    assert out["quality_claim"] is False
    ni = st.non_inferiority(rows, [0] * 32, [1] * 32, cluster_key="cluster_id", margin=0.03)
    assert ni["quality_claim"] is False


def test_module_does_not_import_torch_at_module_level():
    import pathlib
    import subprocess
    import sys
    code = ("import experiments.reasoning_posttraining_v1.statistics, sys; "
            "print('torch' in sys.modules)")
    root = pathlib.Path(st.__file__).parents[2]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(root))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_zero_difference_clusters_do_not_buy_power():
    """589 clusters with 6 moving has the resolution of k=6, and must say so."""
    spec = {f"c{i}": 4 for i in range(40)}
    rows = rows_for(spec)
    movers = {f"c{i}" for i in range(6)}
    baseline = [0] * len(rows)
    arm = [1 if r["cluster_id"] in movers else 0 for r in rows]
    out = st.compare(rows, baseline, arm, cluster_key="cluster_id")
    perm = out["permutation"]
    assert perm["clusters"] == 40
    assert perm["effective_clusters_nonzero"] == 6
    # The floor is computed on 6, not 40.
    assert perm["resolution"]["clusters"] == 6
    assert perm["resolution"]["smallest_two_sided_p"] == pytest.approx(2 / 64)


def test_a_design_with_many_clusters_but_few_movers_can_still_be_refused():
    """Three movers out of 100: the honest verdict is that alpha is unreachable."""
    spec = {f"c{i}": 2 for i in range(100)}
    rows = rows_for(spec)
    movers = {"c0", "c1", "c2"}
    baseline = [0] * len(rows)
    arm = [1 if r["cluster_id"] in movers else 0 for r in rows]
    out = st.compare(rows, baseline, arm, cluster_key="cluster_id")
    assert out["permutation"]["effective_clusters_nonzero"] == 3
    assert out["permutation"]["resolution"]["can_reject_at_alpha"] is False
    assert out["permutation"]["significance_refused"] is True
    assert out["claim_supported"] is False
