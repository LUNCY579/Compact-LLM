"""Matched ordinary FT / post-hoc linearization / tangent FT study engine.

All paths operate on the same frozen base and direct delta coordinates. Causal
response+EOS labels are supplied by the caller. No label/rendering guessing.
The first native port scores TEACHER-FORCED response sequences, not free generation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch.nn import functional as F
from .data import Row, require_design
from .metrics import disentanglement_error, normalized_accuracy, paired
from .tangent import Delta, FrozenAnchor, combine, tensor_digest

PAPER_ALPHA_GRID = tuple(i / 20 for i in range(21))


@dataclass(frozen=True)
class TrainSpec:
    steps: int = 28
    batch_size: int = 1
    lr: float = 1e-5
    schedule: str = "wsd"
    warmup_fraction: float = 0.2
    decay_fraction: float = 0.2
    end_fraction: float = 0.1
    weight_decay: float = 0.0
    sampling: str = "cycle"
    max_processed_tokens: int = 100000
    seed: int = 17

    def validate(self) -> None:
        if any(type(x) is not int or x <= 0 for x in
               (self.steps, self.batch_size, self.max_processed_tokens)):
            raise ValueError("positive integer training limits required")
        if not math.isfinite(self.lr) or self.lr <= 0 or self.schedule not in {"wsd", "cosine"}:
            raise ValueError("invalid learning-rate recipe")
        if self.sampling not in {"cycle", "one_pass"}:
            raise ValueError("sampling must be cycle or one_pass")
        if self.weight_decay != 0:
            raise ValueError("nonzero AdamW decay on delta != decay on theta; not qualified")
        if not (0 <= self.warmup_fraction < 1 and 0 < self.decay_fraction <= 1
                and self.warmup_fraction + self.decay_fraction <= 1
                and 0 <= self.end_fraction <= 1):
            raise ValueError("invalid schedule fractions")


def lr_at(step: int, spec: TrainSpec) -> float:
    warm = int(spec.steps * spec.warmup_fraction)
    if warm and step < warm:
        return spec.lr * (step + 1) / warm
    if spec.schedule == "cosine":
        progress = (step - warm) / max(1, spec.steps - warm - 1)
        factor = spec.end_fraction + (1 - spec.end_fraction) * (1 + math.cos(math.pi * progress)) / 2
    else:
        decay = max(1, int(spec.steps * spec.decay_fraction))
        start = spec.steps - decay
        progress = max(0.0, (step - start + 1) / decay)
        factor = 1 - (1 - spec.end_fraction) * progress
    return spec.lr * factor


def train(anchor: FrozenAnchor, rows: list[Row], mode: str, spec: TrainSpec) -> tuple[Delta, dict]:
    spec.validate()
    if mode not in {"nonlinear", "linear"} or not rows or any(r.split != "train" for r in rows):
        raise ValueError("only explicit train rows can reach optimizer")
    rng = torch.Generator().manual_seed(spec.seed)
    order = torch.randperm(len(rows), generator=rng).tolist()
    plan = [[rows[order[(step * spec.batch_size + j) % len(order)]]
             for j in range(spec.batch_size)] for step in range(spec.steps)]
    if spec.sampling == "one_pass":
        if spec.steps != math.ceil(len(rows) / spec.batch_size):
            raise ValueError("one_pass steps must equal ceil(rows/batch_size)")
        plan = [[rows[i] for i in order[start:start + spec.batch_size]]
                for start in range(0, len(order), spec.batch_size)]
    processed = sum(len(r.input_ids) for group in plan for r in group)
    supervised = sum(r.supervised_tokens for group in plan for r in group)
    if processed > spec.max_processed_tokens:
        raise ValueError("training token budget exceeded before first update")
    d = anchor.zero(trainable=True)
    optimizer = torch.optim.AdamW(list(d.values()), lr=spec.lr, betas=(0.9, 0.95),
                                 eps=1e-8, weight_decay=0)
    history = []
    device = next(iter(d.values())).device
    for step, group in enumerate(plan):
        optimizer.zero_grad(set_to_none=True)
        tokens = sum(r.supervised_tokens for r in group)
        total = 0.0
        for row in group:
            kwargs, labels = row.batch(device)
            logits = anchor.evaluate(d, mode=mode, **kwargs)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / tokens
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward(); total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(d.values()), 1.0, error_if_nonfinite=True)
        for group_ in optimizer.param_groups:
            group_["lr"] = lr_at(step, spec)
        optimizer.step()
        history.append({"step": step + 1, "loss": total, "lr": lr_at(step, spec),
                        "gradient_norm": float(grad_norm)})
    anchor.assert_unchanged()
    result = {k: v.detach().clone() for k, v in d.items()}
    return result, {"mode": mode, "recipe": asdict(spec), "processed_tokens": processed,
                    "supervised_tokens": supervised, "updates": len(history),
                    "delta_value_sha256": tensor_digest(result), "history": history}


def score(anchor: FrozenAnchor, delta: Delta, rows: list[Row], mode: str) -> dict:
    if not rows:
        raise ValueError("cannot score zero rows")
    predictions, flags = [], []
    device = next(iter(delta.values())).device
    with torch.no_grad():
        for row in rows:
            kwargs, labels = row.batch(device)
            logits = anchor.evaluate(delta, mode=mode, **kwargs)
            keep = labels[:, 1:] != -100
            pred = logits[:, :-1].argmax(-1)[keep].tolist()
            target = labels[:, 1:][keep].tolist()
            predictions.append(tuple(pred)); flags.append(pred == target)
    return {"accuracy": sum(flags) / len(flags), "predictions": predictions,
            "correct": flags, "ids": [r.id for r in rows],
            "metric": "teacher_forced_response_exact_match"}


def addition(anchor: FrozenAnchor, deltas: dict[str, Delta], rows: list[Row], mode: str,
             alphas=PAPER_ALPHA_GRID) -> dict:
    tasks = tuple(sorted(deltas))
    if len(tasks) < 2 or not alphas or any(not math.isfinite(a) or not 0 <= a <= 1 for a in alphas):
        raise ValueError("invalid task-addition design")
    subsets = {split: {t: [r for r in rows if r.split == split and r.task == t]
                       for t in tasks} for split in ("dev", "test")}
    # Denominators are the FULL single-task model in each mode, separately.
    singles = {split: {t: score(anchor, deltas[t], subsets[split][t], mode)
                       for t in tasks} for split in ("dev",)}
    candidates = []
    for alpha in sorted(set(alphas)):
        merged_delta = combine(*[(alpha, deltas[t]) for t in tasks])
        scores = {t: score(anchor, merged_delta, subsets["dev"][t], mode) for t in tasks}
        m = normalized_accuracy({t: scores[t]["accuracy"] for t in tasks},
                                {t: singles["dev"][t]["accuracy"] for t in tasks})
        candidates.append({"alpha": alpha, **m})
    eligible = [x for x in candidates if x["normalized_macro_accuracy"] is not None]
    if not eligible:
        return {"status": "UNDEFINED_NORMALIZATION", "dev_candidates": candidates,
                "selected_alpha": None, "test_read": False}
    chosen = max(eligible, key=lambda x: (x["normalized_macro_accuracy"], -x["alpha"]))
    alpha = chosen["alpha"]
    merged_delta = combine(*[(alpha, deltas[t]) for t in tasks])
    # First access to TEST is after coefficient selection, including denominators.
    singles["test"] = {t: score(anchor, deltas[t], subsets["test"][t], mode) for t in tasks}
    test = {t: score(anchor, merged_delta, subsets["test"][t], mode) for t in tasks}
    isolated = {t: score(anchor, combine((alpha, deltas[t])), subsets["test"][t], mode)
                for t in tasks}
    return {"status": "COMPLETE", "selected_alpha": alpha, "dev_candidates": candidates,
            "test_read": True,
            "test": normalized_accuracy({t: test[t]["accuracy"] for t in tasks},
                                         {t: singles["test"][t]["accuracy"] for t in tasks}),
            "xi_sum": disentanglement_error({t: isolated[t]["predictions"] for t in tasks},
                                             {t: test[t]["predictions"] for t in tasks}),
            "per_task": {t: {"merged": test[t], "single": singles["test"][t],
                             "isolated_at_alpha": isolated[t],
                             "paired_vs_isolated": paired(isolated[t]["correct"], test[t]["correct"])}
                         for t in tasks}}


def negation(anchor: FrozenAnchor, delta: Delta, target_dev: list[Row], control_dev: list[Row],
             mode: str, alphas=PAPER_ALPHA_GRID) -> dict:
    """Paper's theta0 - alpha*tau; NOT removing tau from theta0+tau."""
    if any(r.split != "dev" for r in target_dev + control_dev):
        raise ValueError("negation coefficient selection is DEV-only")
    if not alphas or any(not math.isfinite(a) or not 0 <= a <= 1 for a in alphas):
        raise ValueError("invalid negation alpha grid")
    baseline = score(anchor, anchor.zero(), control_dev, mode)["accuracy"]
    if baseline == 0:
        return {"status": "UNINFORMATIVE_ZERO_CONTROL", "selected_alpha": None}
    candidates = []
    for a in sorted(set(alphas)):
        d = combine((-a, delta))
        target = score(anchor, d, target_dev, mode)["accuracy"]
        control = score(anchor, d, control_dev, mode)["accuracy"]
        candidates.append({"alpha": a, "target": target, "control": control,
                           "eligible": control >= 0.95 * baseline})
    eligible = [x for x in candidates if x["eligible"]]
    return {"status": "COMPLETE" if eligible else "NO_ELIGIBLE_ALPHA",
            "selected_alpha": (min(eligible, key=lambda x: (x["target"], x["alpha"]))["alpha"]
                               if eligible else None), "dev_candidates": candidates,
            "control_baseline": baseline, "unlearning_claim": False}


def run_study(anchor: FrozenAnchor, rows: list[Row], tasks: tuple[str, str], spec: TrainSpec,
              alphas=PAPER_ALPHA_GRID, on_delta=None) -> dict:
    require_design(rows, tasks)
    trained, receipts = {}, {}
    for mode in ("nonlinear", "linear"):
        trained[mode], receipts[mode] = {}, {}
        for task in tasks:
            d, receipt = train(anchor, [r for r in rows if r.task == task and r.split == "train"], mode, spec)
            trained[mode][task], receipts[mode][task] = d, receipt
            if on_delta is not None:
                on_delta(mode, task, d)
    results = {}
    for name, source, mode in (("ordinary", "nonlinear", "nonlinear"),
                               ("posthoc", "nonlinear", "linear"),
                               ("tangent_trained", "linear", "linear")):
        results[name] = addition(anchor, trained[source], rows, mode, alphas)
    controls = {split: [r for r in rows if r.task == "control" and r.split == split]
                for split in ("dev", "test")}
    if controls["dev"] and controls["test"]:
        for name, source, mode in (("ordinary", "nonlinear", "nonlinear"),
                                   ("posthoc", "nonlinear", "linear"),
                                   ("tangent_trained", "linear", "linear")):
            neg = {}
            for task in tasks:
                d = trained[source][task]
                result = negation(anchor, d, [r for r in rows if r.task == task and r.split == "dev"],
                                  controls["dev"], mode, alphas)
                if result.get("selected_alpha") is not None:
                    nd = combine((-result["selected_alpha"], d))
                    result["test_target"] = score(anchor, nd, [r for r in rows if r.task == task and r.split == "test"], mode)
                    result["test_control"] = score(anchor, nd, controls["test"], mode)
                    result["test_control_baseline"] = score(anchor, anchor.zero(), controls["test"], mode)
                neg[task] = result
            results[name]["negation"] = neg
    else:
        for result in results.values():
            result["negation"] = {"status": "NOT_RUN_NO_CONTROL_BANK"}
    return {"training": receipts, "methods": results, "anchor_sha256": anchor.initial_digest,
            "nonlinear_calls": anchor.budget.nonlinear, "jvp_calls": anchor.budget.jvp,
            "quality_claim": False, "paper_benchmark_replication": False,
            "interpretation": "method transfer to causal teacher-forced task outputs"}


def pair_surface(anchor: FrozenAnchor, a: Delta, b: Delta, task_a: list[Row], task_b: list[Row],
                 mode: str, grid=None) -> list[dict]:
    """Eq. (6) surface; default paper 20x20 [-3,3]. Explicit diagnostic rows only.

    This can be expensive on a language model. It never selects an arm and must
    be separately budgeted. Predictions are teacher-forced response sequences.
    """
    if not task_a or not task_b or any(r.split != 'diagnostic' for r in task_a+task_b):
        raise ValueError('surface requires two developmental diagnostic supports')
    if {r.id for r in task_a} & {r.id for r in task_b}:
        raise ValueError('task supports overlap')
    values=tuple(-3+6*i/19 for i in range(20)) if grid is None else tuple(grid)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('invalid diagnostic grid')
    iso_a={x:score(anchor,combine((x,a)),task_a,mode)['predictions'] for x in values}
    iso_b={y:score(anchor,combine((y,b)),task_b,mode)['predictions'] for y in values}
    result=[]
    for x in values:
        for y in values:
            d=combine((x,a),(y,b))
            both={'A':score(anchor,d,task_a,mode)['predictions'],
                  'B':score(anchor,d,task_b,mode)['predictions']}
            result.append({'alpha_a':x,'alpha_b':y,
                           'xi_sum':disentanglement_error({'A':iso_a[x],'B':iso_b[y]},both)})
    return result
