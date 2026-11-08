"""Linear probes over captured activations (brief section 16).

**What a probe result means, and what it does not.** A probe says a linear readout of
layer ``l``'s residual stream carries information about some variable. It says nothing
about whether the model *uses* that information. High probe accuracy at layer 14 is a
nomination for the causal experiments in ``patching.py``, not a finding about
mechanism; a probe can succeed on information the forward pass ignores entirely, and
it can fail on information the model encodes non-linearly. Every number this module
produces is a *correlational screen*. Section 17's patching is where causation is
tested.

Three questions, per layer and per site:

1. ``normalized_position`` -- is a token's own normalized position ``i / T`` decodable
   from the residual stream? (ridge, R^2). Expected to be easy: RoPE puts position in
   the stream by construction. The interesting part is the *layer profile* and whether
   it survives to the final query token.
2. ``answer_identity`` -- is the gold answer decodable at the final query token?
   (logistic, accuracy).
3. ``evidence_location`` -- is the *evidence's* normalized position decodable at the
   final query token? (ridge, R^2). This is the lost-in-the-middle question in probe
   form: if the query representation does not know where the evidence was, no
   downstream circuit can condition on it.

Split discipline
----------------
Train/test is **by scenario**, never by row. Every positional variant of one source
example shares a ``scenario_id``, so a row split would put position 0.05 of a scenario
in train and position 0.50 of the same scenario in test, and a probe could pass by
memorising the scenario's filler text. :func:`scenario_split` is the only splitter here
and the alpha search uses a second, nested by-scenario split of the training half.

Chance level is always reported
-------------------------------
Every result carries a permutation baseline: the same fit repeated on shuffled labels,
``n_permutations`` times, reported as mean, sd and a one-sided empirical p-value. For
classification the majority-class rate on the test set is reported as ``chance`` too.
An R^2 of 0.4 is not interesting if shuffled labels also reach 0.35. The empirical p
cannot go below ``1 / (n_permutations + 1)``, so fewer than 20 permutations makes
``beats_chance`` unreachable by construction; the count is recorded on every result and
surfaced as ``ProbeResult.permutation_resolution``.

DEVIATION: the permutation shuffles the label vector across all rows, which destroys
the within-scenario correlation as well as the signal. That makes the baseline
*conservative* for clustered data (a permutation that preserved clusters would score
slightly higher), so a probe that beats it is not beating a strawman. It is recorded
in ``permutation_scheme`` rather than left implicit.

DEVIATION (answer_identity): scenarios in this benchmark mostly have unique answers, so
a by-scenario split would leave every test class unseen and force accuracy to zero by
construction rather than by fact. Labels appearing in fewer than
``ProbeTask.min_class_scenarios`` distinct scenarios are therefore dropped, and if
fewer than two labels survive the probe reports ``status="skipped"`` with the reason
instead of a meaningless zero.

scikit-learn is used when importable (checked at import time, see ``BACKEND``);
otherwise the closed-form ridge and the Nesterov softmax solver in this file take over.
Both paths are exercised by ``tests/test_activation_analysis.py`` and agree to a few
parts in a thousand, so a machine without sklearn produces comparable numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rineya_psa.activation_analysis.collect import ActivationDataset
from rineya_psa.types import stable_id

try:  # sklearn is an optional extra; the numpy fallbacks below are shipped, not stubs.
    from sklearn.linear_model import LogisticRegression, Ridge

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on a sklearn-free interpreter
    LogisticRegression = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False

BACKEND: str = "sklearn" if SKLEARN_AVAILABLE else "numpy"

__all__ = [
    "BACKEND",
    "DEFAULT_ALPHAS",
    "DEFAULT_TASKS",
    "SKLEARN_AVAILABLE",
    "ProbeResult",
    "ProbeTask",
    "logistic_probe",
    "probe_results_to_dataframe",
    "ridge_probe",
    "run_probe_suite",
    "scenario_split",
]

#: Ridge / inverse-C grid searched on a nested by-scenario split of the training half.
DEFAULT_ALPHAS: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeTask:
    """One probing question. ``target`` names a field on ``ActivationRow``."""

    name: str
    kind: str  # "ridge" | "logistic"
    target: str
    roles: tuple[str, ...] | None = None  # None means every role
    min_class_scenarios: int = 2  # logistic only; see the module DEVIATION note

    def __post_init__(self) -> None:
        if self.kind not in {"ridge", "logistic"}:
            raise ValueError(f"unknown probe kind {self.kind!r}")

    @property
    def metric(self) -> str:
        return "r2" if self.kind == "ridge" else "accuracy"


DEFAULT_TASKS: tuple[ProbeTask, ...] = (
    ProbeTask("normalized_position", "ridge", "normalized_position"),
    ProbeTask("answer_identity", "logistic", "answer", roles=("final",)),
    ProbeTask("evidence_location", "ridge", "evidence_position", roles=("final",)),
)


@dataclass(frozen=True)
class ProbeResult:
    """One (task, layer, site) outcome, with its chance level beside it."""

    task: str
    layer: int
    site: str
    kind: str
    metric: str
    score: float
    chance: float
    permutation_mean: float
    permutation_sd: float
    permutation_p: float
    permutation_scheme: str
    n_train: int
    n_test: int
    n_train_scenarios: int
    n_test_scenarios: int
    n_features: int
    n_classes: int | None
    alpha: float
    backend: str
    seed: int
    n_permutations: int = 0
    status: str = "ok"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def permutation_resolution(self) -> float:
        """Smallest p this many permutations can express, ``1 / (n_permutations + 1)``.

        Below 20 permutations that floor sits above 0.05 and :attr:`beats_chance` can
        never be True however clean the signal is. ``run_probe_suite`` defaults to 20 for
        exactly that reason.
        """
        return 1.0 / (int(self.n_permutations) + 1.0)

    @property
    def beats_chance(self) -> bool:
        """Score above the permutation mean *and* an empirical p below 0.05.

        Uninformative when fewer than 20 permutations were run; see
        :attr:`permutation_resolution`.
        """
        return bool(self.score > self.permutation_mean and self.permutation_p < 0.05)


def _skipped(task: ProbeTask, layer: int, site: str, reason: str, seed: int) -> ProbeResult:
    nan = float("nan")
    return ProbeResult(
        task=task.name,
        layer=int(layer),
        site=str(site),
        kind=task.kind,
        metric=task.metric,
        score=nan,
        chance=nan,
        permutation_mean=nan,
        permutation_sd=nan,
        permutation_p=nan,
        permutation_scheme="none",
        n_train=0,
        n_test=0,
        n_train_scenarios=0,
        n_test_scenarios=0,
        n_features=0,
        n_classes=None,
        alpha=nan,
        backend=BACKEND,
        seed=int(seed),
        n_permutations=0,
        status=f"skipped:{reason}",
    )


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------


def scenario_split(
    scenario_ids: Sequence[Any],
    *,
    test_fraction: float = 0.3,
    seed: int = 20260828,
    salt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for a train/test split that never breaks a scenario apart.

    Assignment is by a seeded hash of the scenario id, not by shuffling, so the split
    is identical whatever order the rows arrive in and whichever subset of a scenario's
    variants happens to be present.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1), got {test_fraction}")
    ids = [str(s) for s in scenario_ids]
    if not ids:
        raise ValueError("scenario_ids is empty")
    cut = int(round(test_fraction * 10_000))
    is_test = {}
    for scenario in set(ids):
        bucket = int(stable_id(seed, salt, scenario), 16) % 10_000
        is_test[scenario] = bucket < cut
    test = np.asarray([i for i, s in enumerate(ids) if is_test[s]], dtype=np.int64)
    train = np.asarray([i for i, s in enumerate(ids) if not is_test[s]], dtype=np.int64)
    return train, test


# --------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------


def _standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Centre and scale by *training* statistics only. Constant features are dropped."""
    mean = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < _EPS, 1.0, sd)
    return (train - mean) / sd, (test - mean) / sd


def _ridge_closed_form(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """``argmin ||Xw - y||^2 + alpha ||w||^2`` with an unpenalised intercept.

    Solved in whichever space is smaller: the primal ``[d, d]`` normal equations, or
    the dual ``[n, n]`` kernel form. With ``d = 1024`` and a few hundred rows the dual
    is the cheap one, and it is also the numerically better conditioned of the two.
    """
    y_mean = float(y.mean())
    yc = y - y_mean
    n, d = x.shape
    if d <= n:
        gram = x.T @ x + alpha * np.eye(d)
        w = np.linalg.solve(gram, x.T @ yc)
    else:
        kernel = x @ x.T + alpha * np.eye(n)
        w = x.T @ np.linalg.solve(kernel, yc)
    return w, y_mean


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination against the *test* mean; can go negative."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= _EPS:
        # A constant target has no variance to explain. Perfect prediction is the only
        # non-degenerate answer; anything else is a failure, not an R^2 of 0/0.
        return 1.0 if ss_res <= _EPS else 0.0
    return 1.0 - ss_res / ss_tot


def ridge_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float = 100.0,
) -> float:
    """Fit ridge on standardized features and return test R^2."""
    xtr, xte = _standardize(np.asarray(x_train, dtype=np.float64), np.asarray(x_test, np.float64))
    ytr = np.asarray(y_train, dtype=np.float64)
    yte = np.asarray(y_test, dtype=np.float64)
    if SKLEARN_AVAILABLE:
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(xtr, ytr)
        return _r2(yte, model.predict(xte))
    w, intercept = _ridge_closed_form(xtr, ytr, float(alpha))
    return _r2(yte, xte @ w + intercept)


def _power_iteration_norm(x: np.ndarray, iterations: int = 32) -> float:
    """Largest singular value of ``x``, deterministically. Cheaper than a full SVD."""
    v = np.ones(x.shape[1], dtype=np.float64) / np.sqrt(x.shape[1])
    value = 0.0
    for _ in range(iterations):
        u = x @ v
        norm_u = float(np.linalg.norm(u))
        if norm_u < _EPS:
            return 0.0
        u /= norm_u
        v = x.T @ u
        value = float(np.linalg.norm(v))
        if value < _EPS:
            return 0.0
        v /= value
    return value


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logistic_numpy(
    x: np.ndarray, y: np.ndarray, n_classes: int, alpha: float, max_iter: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic with L2, by Nesterov-accelerated gradient descent.

    The step size is ``1 / L`` with ``L`` the exact smoothness bound of the objective
    ``(1/n) sum CE + (alpha / 2n) ||W||^2``, namely ``0.5 * sigma_max(X)^2 / n +
    alpha / n``. A fixed provably-safe step means no line search, no tuning knob, and
    the same answer on every machine -- which matters because this is the fallback path
    a reproducer without sklearn will hit.
    """
    n, d = x.shape
    sigma = _power_iteration_norm(x)
    lipschitz = 0.5 * sigma * sigma / n + alpha / n
    step = 1.0 / max(lipschitz, _EPS)

    weights = np.zeros((d, n_classes), dtype=np.float64)
    bias = np.zeros(n_classes, dtype=np.float64)
    momentum_w, momentum_b = weights.copy(), bias.copy()
    t = 1.0
    onehot = np.zeros((n, n_classes), dtype=np.float64)
    onehot[np.arange(n), y] = 1.0

    for _ in range(int(max_iter)):
        probs = _softmax_rows(x @ momentum_w + momentum_b)
        residual = probs - onehot
        grad_w = x.T @ residual / n + alpha * momentum_w / n
        grad_b = residual.sum(axis=0) / n
        new_w = momentum_w - step * grad_w
        new_b = momentum_b - step * grad_b
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        factor = (t - 1.0) / t_next
        momentum_w = new_w + factor * (new_w - weights)
        momentum_b = new_b + factor * (new_b - bias)
        weights, bias, t = new_w, new_b, t_next
    return weights, bias


def logistic_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float = 100.0,
    n_classes: int | None = None,
) -> float:
    """Fit a multinomial logistic probe on standardized features; return test accuracy."""
    xtr, xte = _standardize(np.asarray(x_train, dtype=np.float64), np.asarray(x_test, np.float64))
    ytr = np.asarray(y_train, dtype=np.int64)
    yte = np.asarray(y_test, dtype=np.int64)
    k = int(n_classes) if n_classes is not None else int(max(ytr.max(), yte.max()) + 1)

    if len(np.unique(ytr)) < 2:
        # One class in train: every estimator degenerates to the constant predictor.
        return float((yte == ytr[0]).mean())

    if SKLEARN_AVAILABLE:
        model = LogisticRegression(
            C=1.0 / max(float(alpha), _EPS), max_iter=1000, solver="lbfgs"
        )
        model.fit(xtr, ytr)
        return float((model.predict(xte) == yte).mean())

    weights, bias = _logistic_numpy(xtr, ytr, k, float(alpha))
    return float((np.argmax(xte @ weights + bias, axis=1) == yte).mean())


# --------------------------------------------------------------------------------------
# Suite driver
# --------------------------------------------------------------------------------------


def _encode_labels(values: Sequence[Any]) -> tuple[np.ndarray, list[Any]]:
    classes = sorted({str(v) for v in values})
    lookup = {c: i for i, c in enumerate(classes)}
    return np.asarray([lookup[str(v)] for v in values], dtype=np.int64), classes


def _select_alpha(
    fit: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], float],
    x: np.ndarray,
    y: np.ndarray,
    scenarios: Sequence[Any],
    alphas: Sequence[float],
    seed: int,
) -> float:
    """Pick alpha on a nested by-scenario split of the *training* rows only."""
    alpha_list = [float(a) for a in alphas]
    if len(alpha_list) == 1:
        return alpha_list[0]
    inner_train, inner_test = scenario_split(
        scenarios, test_fraction=0.3, seed=seed, salt="alpha-search"
    )
    if inner_train.size < 2 or inner_test.size < 1:
        return alpha_list[len(alpha_list) // 2]
    best, best_score = alpha_list[0], -np.inf
    for alpha in alpha_list:
        score = fit(x[inner_train], y[inner_train], x[inner_test], y[inner_test], alpha)
        if np.isfinite(score) and score > best_score:
            best, best_score = alpha, score
    return best


def _run_one(
    task: ProbeTask,
    layer: int,
    site: str,
    features: np.ndarray,
    targets: np.ndarray,
    scenarios: Sequence[Any],
    train: np.ndarray,
    test: np.ndarray,
    *,
    alphas: Sequence[float],
    n_permutations: int,
    seed: int,
    n_classes: int | None,
    chance: float,
) -> ProbeResult:
    x = np.asarray(features, dtype=np.float64)

    if task.kind == "ridge":

        def fit(xa, ya, xb, yb, alpha):  # noqa: ANN001, ANN202 - local closure
            return ridge_probe(xa, ya, xb, yb, alpha=alpha)

    else:

        def fit(xa, ya, xb, yb, alpha):  # noqa: ANN001, ANN202 - local closure
            return logistic_probe(xa, ya, xb, yb, alpha=alpha, n_classes=n_classes)

    alpha = _select_alpha(
        fit, x[train], targets[train], [scenarios[i] for i in train], alphas, seed
    )
    score = fit(x[train], targets[train], x[test], targets[test], alpha)

    rng = np.random.default_rng(int(stable_id(seed, task.name, layer, site), 16) % (2**63))
    permuted = np.empty(int(n_permutations), dtype=np.float64)
    for i in range(int(n_permutations)):
        shuffled = targets[rng.permutation(targets.size)]
        permuted[i] = fit(x[train], shuffled[train], x[test], shuffled[test], alpha)
    finite = permuted[np.isfinite(permuted)]
    if finite.size:
        perm_mean = float(finite.mean())
        perm_sd = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        p_value = float((1 + int((finite >= score).sum())) / (1 + finite.size))
    else:  # pragma: no cover - every fit failing means a degenerate design matrix
        perm_mean = perm_sd = p_value = float("nan")

    return ProbeResult(
        task=task.name,
        layer=int(layer),
        site=str(site),
        kind=task.kind,
        metric=task.metric,
        score=float(score),
        chance=float(chance),
        permutation_mean=perm_mean,
        permutation_sd=perm_sd,
        permutation_p=p_value,
        permutation_scheme="row_shuffle_of_labels",
        n_train=int(train.size),
        n_test=int(test.size),
        n_train_scenarios=len({scenarios[i] for i in train}),
        n_test_scenarios=len({scenarios[i] for i in test}),
        n_features=int(x.shape[1]),
        n_classes=n_classes,
        alpha=float(alpha),
        backend=BACKEND,
        seed=int(seed),
        n_permutations=int(finite.size),
    )


def run_probe_suite(
    dataset: ActivationDataset,
    *,
    tasks: Sequence[ProbeTask] = DEFAULT_TASKS,
    layers: Sequence[int] | None = None,
    sites: Sequence[str] | None = None,
    test_fraction: float = 0.3,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    n_permutations: int = 20,
    seed: int = 20260828,
) -> list[ProbeResult]:
    """Every ``(task, layer, site)`` probe, with its permutation baseline.

    A task whose rows are too few, whose target is constant, or whose classes do not
    survive the ``min_class_scenarios`` filter yields one ``status="skipped:..."``
    record per (layer, site) rather than a number that looks like a measurement.
    """
    use_layers = tuple(dataset.layers if layers is None else sorted({int(i) for i in layers}))
    use_sites = tuple(dataset.sites if sites is None else [str(s) for s in sites])

    results: list[ProbeResult] = []
    for task in tasks:
        mask = dataset.role_mask(task.roles)
        rows = np.flatnonzero(mask)
        subset = dataset.subset(mask)
        targets_raw = [getattr(r, task.target) for r in subset.rows]
        scenarios = list(subset.scenario_ids)

        skip: str | None = None
        n_classes: int | None = None
        chance = float("nan")
        targets = np.zeros(0)
        keep = np.ones(len(subset.rows), dtype=bool)

        if rows.size < 8:
            skip = f"only_{rows.size}_rows"
        elif task.kind == "ridge":
            targets = np.asarray([float(v) for v in targets_raw], dtype=np.float64)
            finite = np.isfinite(targets)
            if not finite.all():
                keep = finite
            if int(keep.sum()) < 8:
                skip = "target_not_finite"
            elif float(np.std(targets[keep])) < _EPS:
                skip = "target_is_constant"
        else:
            counts: dict[str, set[Any]] = {}
            for value, scenario in zip(targets_raw, scenarios, strict=True):
                counts.setdefault(str(value), set()).add(scenario)
            usable = {v for v, s in counts.items() if len(s) >= int(task.min_class_scenarios)}
            keep = np.asarray([str(v) in usable for v in targets_raw], dtype=bool)
            if len(usable) < 2:
                skip = (
                    f"only_{len(usable)}_labels_appear_in_"
                    f"{task.min_class_scenarios}_or_more_scenarios"
                )
            elif int(keep.sum()) < 8:
                skip = "too_few_rows_after_class_filter"

        if skip is None:
            if task.kind == "logistic":
                encoded, classes = _encode_labels(
                    [v for v, k in zip(targets_raw, keep, strict=True) if k]
                )
                targets = encoded
                n_classes = len(classes)
            else:
                targets = targets[keep]
            scenarios = [s for s, k in zip(scenarios, keep, strict=True) if k]
            train, test = scenario_split(
                scenarios, test_fraction=test_fraction, seed=seed, salt=task.name
            )
            if train.size < 4 or test.size < 2:
                skip = f"split_left_{train.size}_train_{test.size}_test"
            elif task.kind == "logistic":
                values, tallies = np.unique(targets[test], return_counts=True)
                chance = float(tallies.max() / tallies.sum()) if values.size else float("nan")
            else:
                chance = 0.0  # R^2 of the test-mean predictor, by definition

        for layer in use_layers:
            for site in use_sites:
                if skip is not None:
                    results.append(_skipped(task, layer, site, skip, seed))
                    continue
                features = subset.matrix(layer, site)[keep]
                results.append(
                    _run_one(
                        task,
                        layer,
                        site,
                        features,
                        targets,
                        scenarios,
                        train,
                        test,
                        alphas=alphas,
                        n_permutations=n_permutations,
                        seed=seed,
                        n_classes=n_classes,
                        chance=chance,
                    )
                )
    return results


def probe_results_to_dataframe(results: Sequence[ProbeResult]) -> Any:
    """Tidy long format, one row per (task, layer, site). Parquet-friendly."""
    import pandas as pd

    if not results:
        return pd.DataFrame(columns=list(ProbeResult.__dataclass_fields__))
    return pd.DataFrame([r.to_json() for r in results])
