"""Paired accuracy/retention reporting. A mechanical pass is never promotion."""
import numpy as np
import torch


def memory_diagnostics(state, addresses=None):
    report = dict(state_bytes=state.tensor.numel()*state.tensor.element_size(),
                  history_bytes=state.history.numel()*state.history.element_size(),
                  total_buffer_bytes=state.nbytes,
                  state_frobenius_norm=float(state.tensor.detach().double().norm()))
    if addresses is not None:
        if addresses.ndim != 2 or not addresses.numel():
            raise ValueError("addresses must be a nonempty [samples,E] matrix")
        singular = torch.linalg.svdvals(addresses.detach().double())
        power = singular.square()
        if power.sum() == 0:
            effective_rank = 0.0
        else:
            probability = power / power.sum()
            effective_rank = float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum()))
        report["second_key_effective_rank"] = effective_rank
    return report


def paired_report(baseline, candidate, bootstrap: int = 2000, seed: int = 731, families=None):
    """Cluster resampling at group_id level; one training seed per report.

    Do not pool independent seeds as independent question rows. Register/run at
    least three model seeds and review their individual reports before promotion.
    """
    if families is not None:
        selected = set(families)
        if not selected or not selected.issubset({r["family"] for r in baseline}):
            raise ValueError("requested metric families are absent")
        baseline = [r for r in baseline if r["family"] in selected]
        candidate = [r for r in candidate if r["family"] in selected]
    if bootstrap < 100:
        raise ValueError("at least 100 bootstrap replicates required")
    def index(rows):
        result = {}
        for row in rows:
            if row["id"] in result or row["correct"] not in (0, 1, False, True):
                raise ValueError("duplicate ID or nonbinary score")
            result[row["id"]] = row
        if not result:
            raise ValueError("empty predictions")
        return result
    a, b = index(baseline), index(candidate)
    if a.keys() != b.keys():
        raise ValueError("paired predictions require identical example IDs")
    clusters = {}
    repairs = damages = 0
    for key in sorted(a):
        x, y = a[key], b[key]
        for field in ("group_id", "family", "answer", "comparison_id", "seed", "split"):
            if field not in x or field not in y or x[field] != y[field]:
                raise ValueError(f"unpaired metadata: {field}")
        delta = int(y["correct"]) - int(x["correct"])
        repairs += delta == 1
        damages += delta == -1
        clusters.setdefault(x["group_id"], []).append(delta)
    sums = np.asarray([sum(v) for v in clusters.values()], dtype=float)
    counts = np.asarray([len(v) for v in clusters.values()], dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    # Bounded memory instead of [bootstrap, groups] allocation.
    for _ in range(bootstrap):
        selected = rng.integers(0, len(sums), size=len(sums))
        draws.append(sums[selected].sum() / counts[selected].sum())
    lower, upper = np.quantile(draws, [0.025, 0.975]).tolist()
    n = len(a)
    return dict(n_examples=n, n_groups=len(clusters),
                baseline_accuracy=sum(int(x["correct"]) for x in a.values())/n,
                candidate_accuracy=sum(int(x["correct"]) for x in b.values())/n,
                accuracy_delta=(repairs-damages)/n, delta_ci95=[lower, upper],
                repairs=int(repairs), damages=int(damages), bootstrap_replicates=bootstrap,
                evidence_class="single_seed_paired_diagnostic",
                promotion="NOT_EVALUATED_REQUIRES_MULTISEED_CONTROLS_RETENTION_AND_COST")
