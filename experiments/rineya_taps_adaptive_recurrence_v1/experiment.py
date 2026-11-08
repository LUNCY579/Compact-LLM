"""Small live-rollout factorial and split firewall. Fixtures are NOT model evidence."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
import csv
import json
import math
import statistics
import time
import torch
from .scheduler import TAPSConfig, TAPSScheduler, apply_update, scalar_events
from .safety import BASE, Journal, digest, file_hash, resources, write_json

GRID = (.8, .9, 1., 1.1, 1.2)
BUDGETS = (0, 1, 2, 4)
SEEDS = (17, 29)
ORACLE_CAP = 25


@dataclass(frozen=True)
class Case:
    case_id: str
    parent_id: str
    split: str
    task: str
    payload: dict
    label: tuple[float, ...]


def validate_cases(cases, split, protected_parents=()):
    if split not in ("DEV", "TEST") or not cases:
        raise ValueError("nonempty named split required")
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    for c in cases:
        if c.split != split or not c.parent_id:
            raise PermissionError("split mismatch or missing parent")
        if split == "DEV" and c.parent_id in protected_parents:
            raise PermissionError("protected TEST parent in calibration")


@dataclass(frozen=True)
class FrozenPlan:
    config_json: str
    dev_digest: str
    fixed: tuple[tuple[int, float], ...]
    pools: tuple[tuple[int, tuple[float, ...]], ...]
    protected_parents: tuple[str, ...]
    dev_parents: tuple[str, ...]

    @property
    def sha256(self):
        return digest(asdict(self))

    def fixed_for(self, budget):
        return dict(self.fixed)[budget]

    def pool_for(self, budget):
        return dict(self.pools)[budget]


def select_on_dev(cases, rows, taps_traces, config, protected_parents):
    # This check deliberately precedes reading any score/label-bearing object.
    validate_cases(cases, "DEV", protected_parents)
    if not rows or any(r["split"] != "DEV" for r in rows):
        raise PermissionError("calibration accepts DEV result rows only")
    allowed = {c.case_id for c in cases}
    parents = {c.case_id: c.parent_id for c in cases}
    if any(r["case_id"] not in allowed or r.get("parent_id") != parents.get(r["case_id"]) for r in rows):
        raise PermissionError("unknown calibration case")
    if any(e.get("split") != "DEV" or e.get("case_id") not in allowed or e.get("arm") != "A3"
           for e in taps_traces):
        raise PermissionError("calibration alpha pool accepts DEV TAPS traces only")
    fixed, pools = [], []
    for budget in sorted({r["extra_loops"] for r in rows}):
        candidates = [r for r in rows if r["arm"] == "A2" and r["extra_loops"] == budget]
        means = {}
        for alpha in GRID:
            rr = [r for r in candidates if r["fixed_alpha"] == alpha]
            if len(rr) != len(cases)*len(SEEDS) or {(r["case_id"], r["seed"]) for r in rr} != {(c.case_id, s) for c in cases for s in SEEDS}:
                raise ValueError("incomplete DEV fixed-alpha matrix")
            means[alpha] = statistics.mean(r["score"] for r in rr)
        best = max(GRID, key=lambda a: (means[a], -a))
        pool = tuple(min(config.alpha_max, max(config.alpha_min, e["alpha"]))
                     for e in taps_traces if e["extra_loops"] == budget and e["iteration"] > 0)
        fixed.append((budget, best))
        pools.append((budget, pool or (1.,)))
    return FrozenPlan(json.dumps(asdict(config), sort_keys=True), digest(rows),
                      tuple(fixed), tuple(pools), tuple(sorted(protected_parents)),
                      tuple(sorted({c.parent_id for c in cases})))


def fixtures(split):
    # Public mathematical apparatus only; names do not imply reasoning benchmarks.
    specs = [
        ("monotone", [.25, .25], [1., .4], [0., 0.]),
        ("alternating", [1.75, 1.75], [1., .4], [0., 0.]),
        ("two_mode", [1/1.2, 1/.8], [1., 1.], [0., 0.]),
        ("stable_wrong", [.25, .25], [.8, .8], [1., 1.]),
        ("zero_gate", [0., 0.], [1., .4], [0., 0.]),
    ]
    scale = 1. if split == "DEV" else 1.15
    return [Case(f"{split}-{name}", f"{split}-parent-{name}", split, name,
                 {"curvature": lam, "initial": [v*scale for v in initial], "attractor": attractor},
                 (0., 0.)) for name, lam, initial, attractor in specs]


@torch.inference_mode()
def rollout(case, config, extra_loops, seed, *, schedule=None, sink=None, arm="A3"):
    if type(extra_loops) is not int or extra_loops < 0 or extra_loops > 16:
        raise ValueError("invalid extra-loop budget")
    if schedule is not None and len(schedule) != extra_loops:
        raise ValueError("schedule horizon mismatch")
    h = torch.tensor([case.payload["initial"]], dtype=torch.float32)
    lam = torch.tensor(case.payload["curvature"])
    target = torch.tensor(case.payload["attractor"])
    controller = TAPSScheduler(replace(config, seed=seed))
    start = time.perf_counter()
    controller_seconds = core_seconds = logging_seconds = 0.
    traces = []
    previous_readout = None
    for k in range(extra_loops+1):
        t = time.perf_counter()
        proposal = h-lam*(h-target)  # no access to case.label
        loop_core_seconds = time.perf_counter()-t
        core_seconds += loop_core_seconds
        t = time.perf_counter()
        alpha, event = controller.observe(proposal.float()-h.float(), (case.case_id,),
                                         force_unit=k == 0,
                                         alpha_override=None if schedule is None or k == 0 else schedule[k-1])
        new_h = apply_update(h, proposal, alpha)
        loop_controller_seconds = time.perf_counter()-t
        controller_seconds += loop_controller_seconds
        e = scalar_events(event)[0]
        readout = new_h.clone()  # fixture readout, NOT language-model logits
        e.update(case_id=case.case_id, parent_id=case.parent_id, split=case.split,
                 task=case.task, seed=seed, arm=arm, extra_loops=extra_loops,
                 fixed_alpha=config.fixed_alpha if config.scheduler_mode == "fixed" else None,
                 state_norm=float(h.norm()), movement=float((new_h-h).norm()),
                 readout_movement=None if previous_readout is None else float((readout-previous_readout).norm()),
                 logit_movement=None, proxy_loss=float((new_h-torch.tensor(case.label)).square().mean()),
                 loop_core_seconds=loop_core_seconds, loop_controller_seconds=loop_controller_seconds,
                 cumulative_core_seconds=core_seconds, cumulative_controller_seconds=controller_seconds,
                 stopping_iteration=extra_loops, evidence_kind="synthetic_apparatus")
        traces.append(e)
        if sink:
            t = time.perf_counter()
            sink(e)
            logging_seconds += time.perf_counter()-t
        previous_readout, h = readout, new_h
    loss = float((h-torch.tensor(case.label)).square().mean())
    return {"case_id": case.case_id, "parent_id": case.parent_id, "split": case.split,
            "task": case.task, "seed": seed, "arm": arm, "extra_loops": extra_loops,
            "fixed_alpha": config.fixed_alpha if config.scheduler_mode == "fixed" else None,
            "score": float(loss <= .02), "proxy_loss": loss,
            "core_calls": extra_loops+1, "readout_calls": extra_loops+1,
            "controller_seconds": controller_seconds, "core_seconds": core_seconds,
            "logging_seconds": logging_seconds, "wall_seconds": time.perf_counter()-start,
            "physical_run": True, "evidence_kind": "synthetic_apparatus"}, traces


def _run_regular(cases, config, budgets, sink, row_sink, *, plan=None):
    rows, taps = [], []
    for seed in SEEDS:
        for c in cases:
            for budget in budgets:
                arms = [("A0" if budget == 0 else "A1", replace(config, scheduler_mode="unit"))]
                arms += [("A2", replace(config, scheduler_mode="fixed", fixed_alpha=a)) for a in GRID]
                arms += [("A3", config)]
                if plan is not None:
                    arms += [("A5", replace(config, scheduler_mode="random", random_pool=plan.pool_for(budget)))]
                # Reverse alternate seed order to expose apparatus-order sensitivity.
                for arm, cfg in (arms if seed == SEEDS[0] else list(reversed(arms))):
                    row, events = rollout(c, cfg, budget, seed, arm=arm, sink=sink)
                    rows.append(row)
                    row_sink(row)
                    if arm == "A3":
                        taps.extend(events)
    return rows, taps


def evaluate_test(cases, config, budgets, plan, sink, row_sink):
    validate_cases(cases, "TEST")
    if any(c.parent_id in plan.dev_parents or c.parent_id not in plan.protected_parents for c in cases):
        raise PermissionError("TEST parents do not match frozen binding")
    if json.dumps(asdict(config), sort_keys=True) != plan.config_json:
        raise PermissionError("configuration changed after calibration")
    if set(budgets) != set(dict(plan.fixed)):
        raise PermissionError("budget changed after calibration")
    return _run_regular(cases, config, budgets, sink, row_sink, plan=plan)


def add_oracles(cases, rows, config, budgets, row_sink, cap=ORACLE_CAP, *, plan=None):
    # Protected oracle analysis requires the immutable DEV policy, never calibration.
    for c in cases:
        if c.split == "TEST":
            if plan is None or c.parent_id not in plan.protected_parents or c.parent_id in plan.dev_parents:
                raise PermissionError("TEST oracle requires a frozen DEV plan")
            if json.dumps(asdict(config), sort_keys=True) != plan.config_json:
                raise PermissionError("oracle config changed after calibration")
        elif c.split != "DEV":
            raise PermissionError("unknown oracle split")
    if type(cap) is not int or cap < 1 or cap > ORACLE_CAP:
        raise ValueError("oracle cap cannot exceed preregistered cap")
    results = []
    for c in cases:
        for seed in SEEDS:
            for budget in budgets:
                rr = [r for r in rows if r["case_id"] == c.case_id and r["seed"] == seed
                      and r["extra_loops"] == budget and r["arm"] == "A2"]
                if len(rr) != len(GRID):
                    raise ValueError("oracle requires full fixed grid")
                best = max(rr, key=lambda r: (r["score"], -r["proxy_loss"], -r["fixed_alpha"]))
                result = {**best, "arm": "A4_example", "physical_run": False,
                          "deployable": False, "oracle_search_core_calls": 0,
                          "selection": "retrospective per-example constant on fixed-grid records"}
                results.append(result)
                row_sink(result)
                if len(GRID)**budget > cap:
                    continue  # explicitly NOT_ENUMERATED in report; never pretend greedy is an oracle
                if budget <= 1:
                    step = {**result, "arm": "A4_step", "oracle_schedule": [best["fixed_alpha"]]*budget,
                            "selection": "exhaustive finite-grid schedule; reuses fixed runs"}
                else:
                    candidates = []
                    for schedule in product(GRID, repeat=budget):
                        if len(set(schedule)) == 1:
                            r = next(r for r in rr if r["fixed_alpha"] == schedule[0])
                        else:
                            r, _ = rollout(c, config, budget, seed, schedule=schedule, arm="oracle_search")
                            r.update(oracle_schedule=list(schedule), deployable=False)
                            row_sink(r)
                        candidates.append((r, schedule))
                    best_step, schedule = max(candidates, key=lambda x: (x[0]["score"], -x[0]["proxy_loss"], tuple(-a for a in x[1])))
                    step = {**best_step, "arm": "A4_step", "physical_run": False, "deployable": False,
                            "oracle_schedule": list(schedule), "selection": "exhaustive finite-grid schedule",
                            "oracle_search_core_calls": (len(GRID)**budget-len(GRID))*(budget+1)}
                results.append(step)
                row_sink(step)
    return results


def csv_write(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, traces, plan):
    curves, headroom, fixed_curve, per_task, trajectories, targets = [], [], [], [], [], []
    mean = lambda rr, key: statistics.mean(r[key] for r in rr)
    for split in ("DEV", "TEST"):
        for budget, fixed_alpha in plan.fixed:
            selected = [r for r in rows if r["split"] == split and r["extra_loops"] == budget]
            fixed = {}
            for a in GRID:
                rr = [r for r in selected if r["arm"] == "A2" and r["fixed_alpha"] == a]
                fixed[a] = mean(rr, "score")
                fixed_curve.append(dict(split=split, extra_loops=budget, alpha=a, accuracy=fixed[a], proxy_loss=mean(rr, "proxy_loss")))
            envelope = max(fixed.values())
            taps = mean([r for r in selected if r["arm"] == "A3"], "score")
            for oracle in ("A4_example", "A4_step"):
                rr = [r for r in selected if r["arm"] == oracle]
                oracle_acc = mean(rr, "score") if rr else None
                room = None if oracle_acc is None else oracle_acc-envelope
                headroom.append(dict(split=split, extra_loops=budget, oracle=oracle,
                                     status="MEASURED_FIXTURE_ONLY" if rr else "NOT_ENUMERATED_CAP",
                                     dev_selected_alpha=fixed_alpha, dev_selected_fixed_accuracy=fixed[fixed_alpha],
                                     retrospective_global_fixed_accuracy=envelope, taps_accuracy=taps,
                                     oracle_accuracy=oracle_acc, adaptive_headroom=room,
                                     scheduler_capture_ratio=None if room is None or room <= 1e-12 else (taps-envelope)/room))
            arms = sorted({r["arm"] for r in selected})+ ["A2_DEV_selected"]
            for arm in arms:
                rr = [r for r in selected if r["arm"] == arm] if arm != "A2_DEV_selected" else [r for r in selected if r["arm"] == "A2" and r["fixed_alpha"] == fixed_alpha]
                if arm == "A2":
                    continue  # averaging different fixed alphas is not a baseline
                row = dict(split=split, extra_loops=budget, arm=arm, accuracy=mean(rr, "score"),
                           mean_core_calls=mean(rr, "core_calls"), mean_readout_calls=mean(rr, "readout_calls"),
                           instrumented_wall_seconds=mean(rr, "wall_seconds"), controller_seconds=mean(rr, "controller_seconds"),
                           proxy_loss=mean(rr, "proxy_loss"), deployable=not arm.startswith("A4"))
                curves.append(row)
                for task in sorted({r["task"] for r in rr}):
                    tr = [r for r in rr if r["task"] == task]
                    per_task.append(dict(split=split, extra_loops=budget, arm=arm, task=task, accuracy=mean(tr, "score"), proxy_loss=mean(tr, "proxy_loss")))
    for split in ("DEV", "TEST"):
        for budget in dict(plan.fixed):
            for k in range(budget+1):
                rr = [e for e in traces if e["split"] == split and e["extra_loops"] == budget and e["iteration"] == k and e["arm"] == "A3"]
                e = dict(split=split, extra_loops=budget, iteration=k)
                for key in ("alpha", "progress", "fluctuation", "delta_norm", "delta_cosine", "ema_cosine"):
                    values = [r[key] for r in rr if r[key] is not None]
                    e[key+"_mean"] = statistics.mean(values) if values else None
                    e[key+"_stdev"] = statistics.pstdev(values) if values else None
                trajectories.append(e)
        max_budget = max(dict(plan.fixed))
        target = next(r["accuracy"] for r in curves if r["split"] == split and r["extra_loops"] == max_budget and r["arm"] == "A1")
        for arm in ("A1", "A2_DEV_selected", "A3", "A5"):
            rr = [r for r in curves if r["split"] == split and (r["arm"] == arm or arm == "A1" and r["arm"] == "A0") and r["accuracy"] >= target]
            best = min(rr, key=lambda r: r["extra_loops"]) if rr else None
            targets.append(dict(split=split, arm=arm, accuracy_target=target, extra_loops=None if best is None else best["extra_loops"],
                                instrumented_wall_seconds=None if best is None else best["instrumented_wall_seconds"],
                                interpretation="retrospective fixed-budget target, not online stopping"))
    return {"accuracy_loops_latency": curves, "fixed_alpha": fixed_curve, "headroom": headroom,
            "per_task": per_task, "trajectory_by_loop": trajectories, "accuracy_target": targets}


def run_fixture_experiment(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = TAPSConfig()
    dev, test = fixtures("DEV"), fixtures("TEST")
    protected = {c.parent_id for c in test}
    validate_cases(dev, "DEV", protected)
    validate_cases(test, "TEST")
    source_root = Path(__file__).resolve().parent
    write_json(out/"source_manifest.json", {p.relative_to(source_root).as_posix(): file_hash(p)
               for p in sorted(source_root.rglob("*.py")) if "__pycache__" not in p.parts})
    write_json(out/"runtime.json", {"resources": resources(), "publication_base": BASE,
               "execution_scope": "scoped package, not a complete inherited checkout",
               "seed_semantics": "two technical RNG repetitions of five analytic systems per split"})
    write_json(out/"registration.json", {"config": asdict(config), "grid": GRID, "extra_loops": BUDGETS,
               "seeds": SEEDS, "oracle_cap": ORACLE_CAP, "success_mse_threshold": .02,
               "dev_cases": [asdict(c) for c in dev],
               "test_public_metadata": [{"case_id": c.case_id, "parent_id": c.parent_id, "task": c.task} for c in test],
               "evidence_kind": "synthetic_apparatus", "claim_ceiling": "NO_NATIVE_EFFICACY"})
    all_rows, all_traces = [], []
    with Journal(out/"trajectories.jsonl") as trace_log, Journal(out/"outcomes.jsonl") as row_log:
        def sink(e):
            trace_log.append(e)
            all_traces.append(e)
        def row_sink(r):
            row_log.append(r)
        rows, taps = _run_regular(dev, config, BUDGETS, sink, row_sink)
        plan = select_on_dev(dev, rows, taps, config, protected)
        write_json(out/"frozen_plan.json", {**asdict(plan), "sha256": plan.sha256})
        all_rows.extend(rows)
        # A5 DEV uses the now-frozen DEV pool; it cannot change the selected policy.
        for c in dev:
            for seed in SEEDS:
                for budget in BUDGETS:
                    r, _ = rollout(c, replace(config, scheduler_mode="random", random_pool=plan.pool_for(budget)), budget, seed, sink=sink, arm="A5")
                    all_rows.append(r)
                    row_sink(r)
        rows, _ = evaluate_test(test, config, BUDGETS, plan, sink, row_sink)
        all_rows.extend(rows)
        # TEST oracle is retrospective reporting AFTER the immutable DEV freeze.
        all_rows.extend(add_oracles(dev+test, all_rows, config, BUDGETS, row_sink, plan=plan))
    physical = [json.loads(line) for line in (out/"outcomes.jsonl").read_text().splitlines()
                if json.loads(line)["physical_run"]]
    write_json(out/"compute_ledger.json", {
        "physical_rollouts": len(physical),
        "core_calls_all_including_oracle_search": sum(r["core_calls"] for r in physical),
        "readout_calls_all_including_oracle_search": sum(r["readout_calls"] for r in physical),
        "oracle_search_rollouts": sum(r["arm"] == "oracle_search" for r in physical),
        "oracle_search_core_calls": sum(r["core_calls"] for r in physical if r["arm"] == "oracle_search"),
        "measured_controller_seconds_all": sum(r["controller_seconds"] for r in physical),
        "measured_core_seconds_all": sum(r["core_seconds"] for r in physical),
        "measured_logging_seconds_all": sum(r["logging_seconds"] for r in physical),
        "measured_instrumented_rollout_seconds_all": sum(r["wall_seconds"] for r in physical),
        "timing_scope": "CPU apparatus only; controller includes moments and scale application; scalar/readout overhead remains in wall time"
    })
    tables = summarize(all_rows, all_traces, plan)
    for name, table in tables.items():
        csv_write(out/(name+".csv"), table)
    write_json(out/"summary.json", {"recommendation": "BLOCKED", "native_runs": 0,
               "training_attempted": False, "fixture_only": True, "frozen_plan_sha256": plan.sha256,
               "headroom": tables["headroom"], "result_rows": len(all_rows), "trajectory_rows": len(all_traces),
               "oracle_note": "Finite grid only; extra=4 step oracle not enumerated; continuous TAPS can exceed discrete oracle."})
    return out
