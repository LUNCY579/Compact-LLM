"""CLARE-RINEYA-001 command line.

    python cli.py plan            [--config C] [--out FILE]
    python cli.py native-qualify  --out DIR [--config C]
    python cli.py train           --arm ARM --seed S --order canonical|secondary --out RUN_DIR [--config C]
    python cli.py evaluate        --run RUN_DIR [--config C]
    python cli.py interact        --run RUN_DIR [--config C]
    python cli.py analyse         --runs RUN_DIR... --out DIR [--config C]

Every native command runs a resource preflight first (host RAM via the vendored hostbudget,
free VRAM, disk, the shared GPU lease). A failed preflight writes a BLOCKED record and exits 3;
it is not a research result. Outputs are create-only. Use the registered interpreter
(configs/clare_rineya_001.json runtime.python_executable). For long runs launch through
experiments/interference_mechanism_v1/wmi_runner.py (see NATIVE_SMOKE_AND_HANDOFF.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from clare import EXPERIMENT_ID  # noqa: E402
from clare.config import DEFAULT_CONFIG, StudyConfig  # noqa: E402
from clare.io import read_json, write_json  # noqa: E402

EXIT_BLOCKED = 3
MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def row_order(rows, seed: int, skill: str, n: int):
    ranked = sorted(rows, key=lambda r: hashlib.sha256(f"{EXPERIMENT_ID}|order|{seed}|{skill}|{r.id}".encode()).hexdigest())
    return ranked[:n]


def eval_schedule(cfg: StudyConfig, order: tuple[str, ...]) -> dict:
    """stage k -> signatures scored after the k-th skill (registered schedule)."""
    sigs = cfg.raw["evaluation"]["signatures"]
    atomic = [s for s in sigs if len(s) == 1]
    out = {0: list(sigs)}
    for k in range(1, len(order) + 1):
        learned = set(order[:k])
        if k == len(order):
            out[k] = list(sigs)
        else:
            out[k] = atomic + [s for s in sigs if len(s) > 1 and set(s) <= learned]
    return out


def artifact_estimate_gib(cfg: StudyConfig, arm: str) -> float:
    fp_total = cfg.raw["footprint"]["expected_total_coordinates"]
    dense = fp_total * 4 / 1024 ** 3
    n = len(cfg.raw["skills"])
    if arm in ("A_dense_sequential", "B_independent_sum"):
        return n * dense
    frac = 1 - float(cfg.raw["rho"]) ** n
    return frac * dense + n * fp_total / 8 / 1024 ** 3


def _load_stack(cfg: StudyConfig, device: str):
    from clare import native
    from clare.training import DeltaHost

    native.require_registered_runtime()
    runtime = native.configure_runtime(0, device)
    tok = native.load_tokenizer(Path(cfg.raw["model"]["path"]))
    model = native.load_model(Path(cfg.raw["model"]["path"]), device,
                              attn_implementation=cfg.raw["model"]["attn_implementation"])
    fp = cfg.footprint()
    patch = native.make_fp32_compute(model, skip=[s.name[: -len(".weight")] for s in fp])
    host = DeltaHost(model, fp)
    return native, runtime, tok, model, fp, patch, host


def _blocked(out_json: Path, record: dict) -> int:
    record = {**record, "status": "BLOCKED", "is_research_result": False}
    write_json(out_json, record)
    print(json.dumps({"status": "BLOCKED", "blockers": record.get("blockers")}, indent=1))
    return EXIT_BLOCKED


# ---------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------

def cmd_plan(args) -> int:
    from clare.schedule import budgets, schedule_table

    cfg = StudyConfig.load(args.config)
    fp = cfg.footprint()
    plan = {"experiment": EXPERIMENT_ID, "config_sha256": cfg.sha256, "footprint": fp.summary(),
            "rho": cfg.raw["rho"], "orders": cfg.raw["orders"],
            "budgets": {name: [b.as_dict(fp.total) for b in budgets(fp.total, cfg.order(name), cfg.raw["rho"])]
                        for name in cfg.raw["orders"]},
            "rho_sweep_table_not_registered": schedule_table(fp.total, cfg.order("canonical"),
                                                              cfg.raw["rho_sweep_supported_not_registered"]),
            "eval_schedule": {name: eval_schedule(cfg, cfg.order(name)) for name in cfg.raw["orders"]},
            "artifact_gib": {arm: round(artifact_estimate_gib(cfg, arm), 3)
                             for arm in cfg.raw["arms"]["primary"] + cfg.raw["arms"]["secondary"]}}
    if args.out:
        write_json(Path(args.out), plan)
    print(json.dumps(plan, indent=1))
    return 0


# ---------------------------------------------------------------------------------------
# native qualification (Phase 3)
# ---------------------------------------------------------------------------------------

def cmd_native_qualify(args) -> int:
    from clare import resources

    cfg = StudyConfig.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "NATIVE_QUALIFICATION.json"
    pre = resources.preflight("train", out, artifact_gib=0.3, vram_need_gib=5.5)
    from clare.manifests import code_digest, git_identity

    rec = {"experiment": EXPERIMENT_ID, "phase": "3-native-qualification", "config_sha256": cfg.sha256,
           "code_digest": code_digest(), "git": git_identity(),
           "preflight": pre, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if pre["status"] != "READY":
        return _blocked(report_path, {**rec, "blockers": pre["blockers"]})
    gl = resources.gpu_lease()
    hb = resources.hostbudget()
    try:
        with gl.lease(label="CLARE-RINEYA-001", job="native-qualify", timeout_seconds=600) as held:
            rec["gpu_lease_token_prefix"] = held.token[:8]
            try:
                rec.update(_native_qualify_body(cfg, hb, out))
                rec["status"] = "QUALIFIED" if rec.get("gates_passed") else "FAILED_GATES"
            except Exception as exc:  # recorded, never swallowed silently
                rec["status"] = "ERROR"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()
    except Exception as exc:
        return _blocked(report_path, {**rec, "blockers": [f"GPU lease: {type(exc).__name__}: {exc}"]})
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(report_path, rec)
    print(json.dumps({k: rec.get(k) for k in ("status", "gates", "timing", "memory_gib")}, indent=1, default=str))
    return 0 if rec["status"] == "QUALIFIED" else 1


def _native_qualify_body(cfg: StudyConfig, hb, out: Path) -> dict:
    import numpy as np
    import torch

    from clare.data import Corpus
    from clare.masks import SkillMask
    from clare.recipe import group_updates
    from clare.schedule import budgets
    from clare.select import seed_key, select_random, select_top_importance
    from clare.training import base_checksum, make_optimizer, run_stage

    device = "cuda"
    gib = lambda b: round(b / 1024 ** 3, 3)
    res: dict = {"gates": {}, "memory_gib": {}, "timing": {}}
    hb.require(3.6, "clare-native-qualify")
    t0 = time.time()
    native, runtime, tok, model, fp, patch, host = _load_stack(cfg, device)
    res["runtime"] = runtime
    res["patch_inventory"] = patch
    res["timing"]["load_s"] = round(time.time() - t0, 2)
    res["memory_gib"]["after_load_allocated"] = gib(torch.cuda.memory_allocated())
    # --- inventory ------------------------------------------------------------------------
    n_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    res["inventory"] = {"unique_parameters": n_params, "footprint": fp.summary(),
                        "footprint_modules_verified": len(host.slots),
                        "trainable_parameters_frozen_model": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    res["gates"]["param_count_596049920"] = n_params == 596_049_920
    res["gates"]["footprint_188743680"] = fp.total == 188_743_680 and len(host.slots) == 60
    res["gates"]["base_frozen"] = res["inventory"]["trainable_parameters_frozen_model"] == 0
    base_sum_before = base_checksum(model)
    # --- data -------------------------------------------------------------------------------
    corpus = Corpus(cfg.raw["dataset"]["root"])
    r_rows = row_order(corpus.atomic_bank("R"), 0, "R", 16)
    compiled = native.compile_train_rows(r_rows, tok)
    res["gates"]["train_contract"] = all(c["supervised_tokens"] > 0 and not c["truncated"] for c in compiled)
    lens = [c["processed_tokens"] for c in compiled]
    res["train_rows_R_tokens"] = {"min": min(lens), "max": max(lens), "mean": float(np.mean(lens))}
    # --- parity gate: patched BF16-storage vs plain FP32 load (last-position logits) ----------
    from transformers import AutoModelForCausalLM

    model.eval()
    probe = [native.prompt_ids(r, tok) for r in r_rows[:2]]
    with torch.no_grad():
        mine = [model(input_ids=torch.tensor([p], device=device), logits_to_keep=1).logits[0, -1].double().cpu() for p in probe]
    native.set_eval_head_cache(model, True)
    with torch.no_grad():
        mine_cached = [model(input_ids=torch.tensor([p], device=device), logits_to_keep=1).logits[0, -1].double().cpu() for p in probe]
    native.set_eval_head_cache(model, False)
    ref = AutoModelForCausalLM.from_pretrained(cfg.raw["model"]["path"], dtype=torch.float32, local_files_only=True,
                                               attn_implementation=cfg.raw["model"]["attn_implementation"],
                                               low_cpu_mem_usage=True, device_map={"": device}).eval()
    with torch.no_grad():
        theirs = [ref(input_ids=torch.tensor([p], device=device), logits_to_keep=1).logits[0, -1].double().cpu() for p in probe]
    del ref
    torch.cuda.empty_cache()
    d_chunk = max(float((a - b).abs().max()) for a, b in zip(mine, theirs))
    d_cached = max(float((a - b).abs().max()) for a, b in zip(mine_cached, theirs))
    res["parity"] = {"max_abs_logit_diff_chunked_head": d_chunk, "max_abs_logit_diff_cached_head": d_cached,
                     "argmax_equal": all(int(a.argmax()) == int(b.argmax()) for a, b in zip(mine, theirs)),
                     "tolerance": 1e-4}
    res["gates"]["parity"] = d_cached <= 1e-4 and d_chunk <= 1e-4 and res["parity"]["argmax_equal"]
    # --- dense training step (arm A / discovery memory) --------------------------------------
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    recipe = cfg.recipe()
    loss_fn = native.make_loss_fn(model, device)
    groups = group_updates(compiled, recipe.update_tokens)[:2]
    torch.cuda.reset_peak_memory_stats()
    params = host.start_dense(None)
    res["memory_gib"]["dense_delta_allocated"] = gib(torch.cuda.memory_allocated())
    t1 = time.time()
    log, opt = run_stage(host, compiled, groups, loss_fn, recipe, guard=lambda: hb.guard(1.5, "clare-nq"))
    torch.cuda.synchronize()
    res["timing"]["dense_s_per_update"] = round((time.time() - t1) / max(1, log.updates), 3)
    res["timing"]["dense_tokens_per_s"] = round(log.processed_tokens / (time.time() - t1), 1)
    res["memory_gib"]["dense_peak_allocated"] = gib(torch.cuda.max_memory_allocated())
    res["memory_gib"]["adam_state_bytes_dense"] = gib(sum(v.numel() * v.element_size() for s in opt.state.values()
                                                         for v in s.values() if torch.is_tensor(v)))
    res["dense_log"] = log.as_dict()
    res["gates"]["dense_trains_finite"] = all(np.isfinite(log.losses)) and log.updates == len(groups)
    pre_eff = None
    host.stop_trainable()
    del opt, params
    torch.cuda.empty_cache()
    # --- discovery probe on free coordinates + reset ----------------------------------------
    b = budgets(fp.total, cfg.order("canonical"), cfg.raw["rho"])
    full = {s.name: np.ones(s.numel, dtype=bool) for s in fp}
    pretend_claimed, _ = select_random(fp, full, b[0].budget, seed_key("qualification-claim"))
    free = SkillMask.from_bool(fp, pretend_claimed).invert()
    pre_eff = host.effective_checksum()
    host.start_dense(free)
    torch.cuda.reset_peak_memory_stats()
    dlog, _ = run_stage(host, compiled, groups, loss_fn, recipe, l1_lambda=cfg.raw["discovery"]["l1_lambda"])
    res["memory_gib"]["discovery_peak_allocated"] = gib(torch.cuda.max_memory_allocated())
    leaked = host.outside_mask_nonzero()
    nz = sum(int(np.count_nonzero(host.importance(s.name)[free.bool(s.name)])) for s in fp)
    t2 = time.time()
    sel, rep = select_top_importance(fp, lambda sl: host.importance(sl.name), {s.name: free.bool(s.name) for s in fp},
                                     b[1].budget, seed_key("qualification-tiebreak"))
    res["timing"]["selection_s"] = round(time.time() - t2, 2)
    host.stop_trainable()
    post_eff = host.effective_checksum()
    res["discovery_probe"] = {"updates": dlog.updates, "free": free.count(), "nonzero_free_after_probe": nz,
                              "budget_skill2": b[1].budget, "ratio_nonzero_to_budget": nz / b[1].budget,
                              "outside_free_nonzero": leaked, "selection": rep.as_dict(),
                              "note": "2-update probe on 16 TRAIN R rows: mechanics and the lambda gate only"}
    res["gates"]["discovery_mask_respected"] = leaked == 0
    res["gates"]["discovery_reset_exact"] = post_eff == pre_eff
    res["gates"]["lambda_gate_2x_budget"] = nz >= 2 * b[1].budget
    # --- sparse (masked) training step --------------------------------------------------------
    mask = SkillMask.from_bool(fp, sel)
    host.set_frozen(None)
    params = host.start_sparse(mask)
    torch.cuda.reset_peak_memory_stats()
    t3 = time.time()
    slog, sopt = run_stage(host, compiled, groups, loss_fn, recipe)
    torch.cuda.synchronize()
    res["timing"]["sparse_s_per_update"] = round((time.time() - t3) / max(1, slog.updates), 3)
    res["memory_gib"]["sparse_peak_allocated"] = gib(torch.cuda.max_memory_allocated())
    delta = host.trainable_delta()
    res["sparse_step"] = {"allocated": mask.count(), "active_nonzero": delta.nonzero(), "l2_norm": delta.norm()}
    res["gates"]["sparse_active_within_mask"] = delta.nonzero() <= mask.count()
    host.stop_trainable()
    del sopt, params
    torch.cuda.empty_cache()
    model.gradient_checkpointing_disable()
    res["gates"]["base_unchanged"] = base_checksum(model) == base_sum_before
    # --- base competence on CALIBRATION (train signatures only) -----------------------------
    model.eval()
    native.set_eval_head_cache(model, True)
    cal = []
    for sig in ("R", "U", "S", "P", "RU", "US", "SP"):
        rows = sorted(corpus.rows("calibration", "calibration", [sig]), key=lambda r: hashlib.sha256(r.id.encode()).hexdigest())[:6]
        cal.extend(rows)
    per = {}
    t4 = time.time()
    examples = []
    for r in cal:
        g = native.greedy_generate(model, native.prompt_ids(r, tok), device, MAX_NEW_TOKENS)
        text = tok.decode(g["ids"], skip_special_tokens=False)
        ok = corpus.score(r, text)
        d = per.setdefault(r.signature, {"n": 0, "correct": 0, "terminated": 0})
        d["n"] += 1; d["correct"] += ok; d["terminated"] += int(g["terminated"])
        if len(examples) < 14:
            examples.append({"signature": r.signature, "response": text[:160], "correct": ok})
    res["timing"]["generate_s_per_row"] = round((time.time() - t4) / len(cal), 3)
    res["base_competence_calibration"] = {"per_signature": per, "examples": examples,
                                          "note": "6 CALIBRATION rows per train signature; a smoke reading, not an estimate"}
    native.set_eval_head_cache(model, False)
    res["memory_gib"]["final_allocated"] = gib(torch.cuda.memory_allocated())
    res["host_after"] = hb.snapshot()
    # --- projections ------------------------------------------------------------------------
    tok_per_row = res["train_rows_R_tokens"]["mean"]
    rows_per_skill = cfg.raw["train"]["rows_per_skill"]
    dense_stage_s = rows_per_skill * tok_per_row / max(1.0, res["timing"]["dense_tokens_per_s"])
    res["projection"] = {"dense_stage_minutes_R_rows": round(dense_stage_s / 60, 1),
                         "eval_minutes_per_960_rows": round(960 * res["timing"]["generate_s_per_row"] / 60, 1),
                         "note": "R rows are the longest atomic prompts; projections are upper-bound-ish for U/S/P"}
    res["gates_passed"] = all(v for k, v in res["gates"].items() if k != "lambda_gate_2x_budget")
    return res


# ---------------------------------------------------------------------------------------
# LR build gate (registered, outcome-blind: CALIBRATION R only, never DEV/TEST, never D/E)
# ---------------------------------------------------------------------------------------

def _resolved_recipe(cfg: StudyConfig, lr_from: str | None):
    from dataclasses import replace

    base = cfg.recipe()
    if not lr_from:
        return base, "config"
    pilot = read_json(Path(lr_from))
    if pilot.get("config_sha256") != cfg.sha256 or pilot.get("status") != "LR_CHOSEN":
        raise SystemExit(f"LR pilot {lr_from} is not a completed pilot of this registration")
    digest = hashlib.sha256(Path(lr_from).read_bytes()).hexdigest()
    return replace(base, peak_lr=float(pilot["chosen_peak_lr"])), f"{lr_from} sha256={digest}"


def cmd_lr_pilot(args) -> int:
    from dataclasses import replace

    from clare import resources
    from clare.data import Corpus
    from clare.recipe import group_updates
    from clare.training import run_stage

    cfg = StudyConfig.load(args.config)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    pilot_cfg = cfg.raw["lr_pilot"]
    pre = resources.preflight("train", out.parent, artifact_gib=0.05, vram_need_gib=5.5)
    if pre["status"] != "READY":
        return _blocked(out.with_name(f"LR_PILOT_BLOCKED_{int(time.time())}.json"),
                        {"preflight": pre, "blockers": pre["blockers"]})
    gl, hb = resources.gpu_lease(), resources.hostbudget()
    rec = {"experiment": EXPERIMENT_ID, "config_sha256": cfg.sha256, "rule": pilot_cfg["rule"], "ladder": []}
    with gl.lease(label="CLARE-RINEYA-001", job="lr-pilot", timeout_seconds=3600):
        hb.require(3.6, "clare-lr-pilot")
        native, runtime, tok, model, fp, patch, host = _load_stack(cfg, "cuda")
        rec["runtime"] = runtime
        corpus = Corpus(cfg.raw["dataset"]["root"])
        chosen_rows = row_order(corpus.atomic_bank("R"), 0, "R", cfg.raw["train"]["rows_per_skill"])
        rows = native.compile_train_rows(chosen_rows, tok)
        groups = group_updates(rows, cfg.recipe().update_tokens)
        cal = corpus.rows("calibration", "calibration", ["R"])
        pids = {r.id: native.prompt_ids(r, tok) for r in cal}

        def cal_acc():
            model.eval()
            native.set_eval_head_cache(model, True)
            ok = 0
            for r in cal:
                g = native.greedy_generate(model, pids[r.id], "cuda", MAX_NEW_TOKENS)
                ok += corpus.score(r, tok.decode(g["ids"], skip_special_tokens=False))
            native.set_eval_head_cache(model, False)
            return ok / len(cal)

        rec["base_calibration_R_acc"] = base = cal_acc()
        rec["calibration_rows"] = len(cal)
        chosen = None
        for lr in pilot_cfg["ladder"]:
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            host.start_dense(None)
            log, _ = run_stage(host, rows, groups, native.make_loss_fn(model, "cuda"),
                               replace(cfg.recipe(), peak_lr=lr), guard=lambda: hb.guard(1.5, "clare-lr-pilot"))
            model.gradient_checkpointing_disable()
            acc = cal_acc()
            host.stop_trainable()
            rec["ladder"].append({"peak_lr": lr, "calibration_R_acc": acc, "updates": log.updates,
                                  "seconds": log.seconds, "final_loss": log.losses[-1]})
            if acc > base + pilot_cfg["acquisition_bar_over_base"]:
                chosen = lr
                break
    rec["chosen_peak_lr"] = chosen
    rec["status"] = "LR_CHOSEN" if chosen is not None else "BLOCKED_ON_ACQUISITION"
    write_json(out, rec)
    print(json.dumps({k: rec[k] for k in ("status", "chosen_peak_lr", "base_calibration_R_acc", "ladder")}, indent=1))
    return 0 if chosen is not None else EXIT_BLOCKED


# ---------------------------------------------------------------------------------------
# train (Phase 4, one arm)
# ---------------------------------------------------------------------------------------

def _skill_rows(cfg: StudyConfig, corpus, native, tok, seed: int):
    rows, groups = {}, {}
    from clare.recipe import group_updates

    recipe = cfg.recipe()
    for skill in cfg.raw["skills"]:
        chosen = row_order(corpus.atomic_bank(skill), seed, skill, cfg.raw["train"]["rows_per_skill"])
        compiled = native.compile_train_rows(chosen, tok, recipe.max_sequence, recipe.completion_reserve)
        rows[skill] = compiled
        groups[skill] = group_updates(compiled, recipe.update_tokens)
    return rows, groups


def cmd_train(args) -> int:
    import torch

    from clare import resources
    from clare.allocation import ArmSpec, SkillData, SkillRecord, run_arm
    from clare.data import Corpus
    from clare.deltas import DiskDelta
    from clare.manifests import code_digest, git_identity, skill_manifest, utc_now
    from clare.masks import SkillMask, layer_table

    cfg = StudyConfig.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "RUN_SPEC.json"
    spec = ArmSpec(args.arm, cfg.order(args.order), rho=cfg.raw["rho"], seed=args.seed,
                   discovery_fraction=cfg.raw["discovery"]["fraction_of_update_groups"],
                   l1_lambda=cfg.raw["discovery"]["l1_lambda"],
                   lambda_fallback=tuple(cfg.raw["discovery"]["lambda_fallback"]))
    recipe, lr_source = _resolved_recipe(cfg, args.lr_from)
    run_spec = {"experiment": EXPERIMENT_ID, "config_sha256": cfg.sha256, "arm": spec.arm, "seed": spec.seed,
                "order_name": args.order, "order": "".join(spec.order), "rho": spec.rho,
                "l1_lambda": spec.l1_lambda, "lambda_fallback": list(spec.lambda_fallback),
                "discovery_fraction": spec.discovery_fraction, "peak_lr": recipe.peak_lr, "lr_source": lr_source}
    if spec_path.exists():
        if read_json(spec_path) != run_spec:
            raise SystemExit("RUN_SPEC.json differs: refusing to resume a different run in this directory")
    else:
        write_json(spec_path, run_spec)
    pre = resources.preflight("train", out, artifact_gib=artifact_estimate_gib(cfg, spec.arm), vram_need_gib=5.5)
    if pre["status"] != "READY":
        return _blocked(out / f"PREFLIGHT_BLOCKED_{int(time.time())}.json", {**run_spec, "preflight": pre,
                                                                            "blockers": pre["blockers"]})
    gl, hb = resources.gpu_lease(), resources.hostbudget()
    with gl.lease(label="CLARE-RINEYA-001", job=f"train {spec.arm} s{spec.seed} {''.join(spec.order)}",
                  timeout_seconds=3600):
        hb.require(3.6, "clare-train")
        native, runtime, tok, model, fp, patch, host = _load_stack(cfg, "cuda")
        git = git_identity()
        code = code_digest()
        corpus = Corpus(cfg.raw["dataset"]["root"])
        rows, groups = _skill_rows(cfg, corpus, native, tok, spec.seed)
        data = SkillData(rows, groups)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        loss_fn = native.make_loss_fn(model, "cuda")
        mdir, ddir = out / "manifests", out / "deltas"
        mdir.mkdir(exist_ok=True)
        ddir.mkdir(exist_ok=True)
        # resume: skills whose manifest AND delta verify are skipped
        resume = {}
        for k, skill in enumerate(spec.order):
            mp = mdir / f"{k:02d}_{skill}.json"
            if not mp.exists():
                break
            man = read_json(mp)
            d = DiskDelta(ddir / f"{k:02d}_{skill}", fp)
            if d.checksum() != man["delta_checksum"]:
                raise SystemExit(f"resume: {skill} delta does not match its manifest")
            mask = SkillMask.load(out / "masks" / f"{k:02d}_{skill}", fp) if (out / "masks" / f"{k:02d}_{skill}.json").exists() else None
            resume[skill] = SkillRecord(skill, k, d, mask, None, None)
        persist = lambda h, skill, k: h.save_trainable(ddir / f"{k:02d}_{skill}", meta={"arm": spec.arm, "skill": skill})
        started = {"t": utc_now()}

        def on_skill(rec, res):
            if rec.skill in resume:
                return
            if rec.mask is not None:
                (out / "masks").mkdir(exist_ok=True)
                rec.mask.save(out / "masks" / f"{rec.position:02d}_{rec.skill}", meta={"arm": spec.arm, "skill": rec.skill})
            side = rec.delta.side if hasattr(rec.delta, "side") else {"nonzero": rec.delta.nonzero(),
                                                                       "l2_norm": rec.delta.norm(),
                                                                       "delta_checksum": rec.delta.checksum()}
            man = skill_manifest(arm=spec.arm, skill=rec.skill, order="".join(spec.order), position=rec.position,
                                 seed=spec.seed, rho=spec.rho, footprint_summary=fp.summary(),
                                 eligible_before=rec.budget.free_before if rec.budget else fp.total,
                                 selected=rec.mask.count() if rec.mask is not None else None,
                                 budget=rec.budget.budget if rec.budget else None, delta_side=side,
                                 mask_checksum=rec.mask.checksum() if rec.mask is not None else None,
                                 recipe=recipe.as_dict(), train_log=rec.train_log.as_dict() if rec.train_log else None,
                                 discovery=rec.discovery, model_identity=cfg.raw["model"],
                                 dataset={**corpus.receipt(), "train_bank": f"atomic_{rec.skill}",
                                          "rows_used": cfg.raw["train"]["rows_per_skill"],
                                          "row_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in rows[rec.skill]).encode()).hexdigest()},
                                 runtime=runtime, git=git, frozen_checksum_before=rec.frozen_checksum_before,
                                 frozen_checksum_after=rec.frozen_checksum_after, code=code,
                                 started_utc=started["t"])
            write_json(mdir / f"{rec.position:02d}_{rec.skill}.json", man)
            started["t"] = utc_now()

        independent = None
        if spec.arm == "C_magnitude_sparse":
            bdir = Path(args.independent_from)
            independent = {}
            for k, skill in enumerate(spec.order):
                independent[skill] = DiskDelta(bdir / "deltas" / f"{k:02d}_{skill}", fp)
        res = run_arm(spec, host, data, loss_fn, recipe, independent=independent, on_skill=on_skill,
                      persist=persist, resume=resume or None, guard=lambda: hb.guard(1.5, "clare-train"))
        if res.allocation is not None:
            write_json(out / "allocation_by_layer.json", layer_table(res.allocation))
        write_json(out / "TRAIN_COMPLETE.json", {**run_spec, "skills": [r.skill for r in res.records],
                                                 "finished_utc": utc_now(), "git": git, "code_digest": code})
    return 0


# ---------------------------------------------------------------------------------------
# evaluate (generation accuracy per registered stage)
# ---------------------------------------------------------------------------------------

def cmd_evaluate(args) -> int:
    from clare import resources
    from clare.data import Corpus
    from clare.deltas import DiskDelta
    from clare.evaluator import score_generation

    cfg = StudyConfig.load(args.config)
    run = Path(args.run)
    rs = read_json(run / "RUN_SPEC.json")
    if rs["config_sha256"] != cfg.sha256 or not (run / "TRAIN_COMPLETE.json").exists():
        raise SystemExit("run is incomplete or was trained under a different registration")
    pre = resources.preflight("probe", run, artifact_gib=0.05, vram_need_gib=3.5)
    if pre["status"] != "READY":
        return _blocked(run / f"EVAL_BLOCKED_{int(time.time())}.json", {"preflight": pre, "blockers": pre["blockers"]})
    ev = cfg.raw["evaluation"]
    order = tuple(rs["order"])
    gl, hb = resources.gpu_lease(), resources.hostbudget()
    with gl.lease(label="CLARE-RINEYA-001", job=f"evaluate {rs['arm']} s{rs['seed']} {rs['order']}", timeout_seconds=3600):
        hb.require(3.0, "clare-evaluate")
        native, runtime, tok, model, fp, patch, host = _load_stack(cfg, "cuda")
        model.eval()
        native.set_eval_weight_cache(model, True, skip=[s.name[: -len(".weight")] for s in fp])
        corpus = Corpus(cfg.raw["dataset"]["root"])
        sample = corpus.sample("dev", "evaluation", ev["signatures"], ev["n_per_signature"], ev["selection_key"])
        pids = {r.id: native.prompt_ids(r, tok) for r in sample}
        gen = lambda ids, mnt: native.greedy_generate(model, ids, "cuda", mnt)
        decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
        deltas = [DiskDelta(run / "deltas" / f"{k:02d}_{s}", fp) for k, s in enumerate(order)]
        sched = eval_schedule(cfg, order)
        out = run / "scores.jsonl"
        for k in range(0, len(order) + 1):
            if k == 0 and not args.include_base:
                continue
            host.set_frozen(deltas[:k] if k else None)
            host.cache_effective(True)
            rows = [r for r in sample if r.signature in sched[k]]
            label = {"arm": rs["arm"], "seed": rs["seed"], "order": rs["order"], "config": "base" if k == 0 else f"prefix:{k}",
                     "stage": k, "config_sha256": cfg.sha256}
            got = score_generation(rows=rows, prompt_ids=pids, generate=gen, decode=decode, score=corpus.score,
                                   out_path=out, label=label, max_new_tokens=MAX_NEW_TOKENS,
                                   guard=lambda: hb.guard(1.2, "clare-evaluate"))
            if len(got) != len(rows):
                raise SystemExit(f"stage {k}: {len(got)} scored rows != {len(rows)} registered")
        if rs["arm"] == "B_independent_sum":
            for k, s in enumerate(order):
                host.set_frozen(deltas[k])
                host.cache_effective(True)
                rows = [r for r in sample if r.signature == s]
                label = {"arm": rs["arm"], "seed": rs["seed"], "order": rs["order"], "config": f"single:{s}",
                         "stage": -1, "config_sha256": cfg.sha256}
                score_generation(rows=rows, prompt_ids=pids, generate=gen, decode=decode, score=corpus.score,
                                 out_path=out, label=label, max_new_tokens=MAX_NEW_TOKENS)
        host.set_frozen(None)
    write_json(run / f"EVAL_COMPLETE_{int(time.time())}.json", {"runtime": runtime, "rows_file": out.name})
    return 0


# ---------------------------------------------------------------------------------------
# interact (functional 2x2 per skill pair)
# ---------------------------------------------------------------------------------------

def cmd_interact(args) -> int:
    from itertools import combinations

    from clare import resources
    from clare.data import Corpus
    from clare.deltas import DiskDelta
    from clare.evaluator import pair_interaction

    cfg = StudyConfig.load(args.config)
    run = Path(args.run)
    rs = read_json(run / "RUN_SPEC.json")
    order = tuple(rs["order"])
    pre = resources.preflight("probe", run, artifact_gib=0.05, vram_need_gib=4.0)
    if pre["status"] != "READY":
        return _blocked(run / f"INTERACT_BLOCKED_{int(time.time())}.json", {"preflight": pre, "blockers": pre["blockers"]})
    gl, hb = resources.gpu_lease(), resources.hostbudget()
    ic = cfg.raw["interaction"]
    with gl.lease(label="CLARE-RINEYA-001", job=f"interact {rs['arm']} s{rs['seed']}", timeout_seconds=3600):
        hb.require(3.0, "clare-interact")
        native, runtime, tok, model, fp, patch, host = _load_stack(cfg, "cuda")
        model.eval()
        native.set_eval_head_cache(model, True)
        corpus = Corpus(cfg.raw["dataset"]["root"])
        ev = cfg.raw["evaluation"]
        deltas = {s: DiskDelta(run / "deltas" / f"{k:02d}_{s}", fp) for k, s in enumerate(order)}
        results = {}
        for a, b in combinations(cfg.raw["skills"], 2):
            sig = "".join(x for x in cfg.raw["skills"] if x in (a, b))
            rows = corpus.sample("dev", "evaluation", [sig], ev["n_per_signature"], ev["selection_key"],
                                 with_targets=True)[: ic["rows_per_pair"]]
            compiled = native.compile_train_rows(rows, tok, 4096, 128)
            tf = lambda r, keep_vectors=False: native.teacher_forced(model, r, "cuda", keep_vectors)
            results[f"{a}+{b}"] = {"signature": sig, **pair_interaction(host=host, delta_a=deltas[a], delta_b=deltas[b],
                                                                        rows=compiled, teacher_forced=tf,
                                                                        topk=ic["vocab_topk"])}
            hb.guard(1.2, "clare-interact")
    write_json(run / "INTERACTION.json", {"run": rs, "runtime": runtime, "pairs": results})
    return 0


# ---------------------------------------------------------------------------------------
# analyse (offline)
# ---------------------------------------------------------------------------------------

def cmd_analyse(args) -> int:
    from clare import metrics
    from clare.deltas import DiskDelta
    from clare.evaluator import read_scored
    from clare.geometry import by_layer, layer_energy, pairwise

    cfg = StudyConfig.load(args.config)
    fp = cfg.footprint()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records, geometry, layers, runs = [], [], [], []
    for rd in map(Path, args.runs):
        rs = read_json(rd / "RUN_SPEC.json")
        runs.append(rs)
        records.extend(r for r in read_scored(rd / "scores.jsonl") if r.get("config_sha256") == cfg.sha256)
        order = tuple(rs["order"])
        deltas = {s: DiskDelta(rd / "deltas" / f"{k:02d}_{s}", fp) for k, s in enumerate(order)}
        for row in pairwise(deltas):
            geometry.append({"arm": rs["arm"], "seed": rs["seed"], "order": rs["order"], **row})
        for s, d in deltas.items():
            for row in by_layer(layer_energy(d)):
                layers.append({"arm": rs["arm"], "seed": rs["seed"], "order": rs["order"], "skill": s, **row})
    result = {"experiment": EXPERIMENT_ID, "config_sha256": cfg.sha256, "runs": runs,
              "labels": {"MEASURED": "accuracies, geometry, interaction values read from artifacts",
                         "DERIVED": "retention/forgetting/paired differences computed from MEASURED",
                         "HYPOTHESIZED": "none in this file"}}
    try:
        result["retention"] = {f"{r['arm']}|{r['seed']}|{r['order']}": metrics.retention_matrix(records, r["arm"], r["seed"], r["order"])
                               for r in runs if r["arm"] != "C_magnitude_sparse"}
    except Exception as exc:
        result["retention_error"] = str(exc)
    for r in runs:
        key = f"{r['arm']}|{r['seed']}|{r['order']}"
        if key in result.get("retention", {}):
            metrics.write_retention_csv(out / f"retention_matrix__{r['arm']}__s{r['seed']}__{r['order']}.csv",
                                        result["retention"][key])
            comp = metrics.composition_matrix(records, r["arm"], r["seed"], r["order"],
                                              cfg.raw["evaluation"]["heldout_signatures"] + cfg.raw["evaluation"]["seen_pair_signatures"])
            metrics.write_composition_csv(out / f"composition_matrix__{r['arm']}__s{r['seed']}__{r['order']}.csv", comp)
        alloc = Path(args.runs[runs.index(r)]) / "allocation_by_layer.json"
        if alloc.exists():
            metrics.write_table_csv(out / f"allocation_by_layer__{r['arm']}__s{r['seed']}__{r['order']}.csv", read_json(alloc))
    write_json(out / "results.json", result)
    if records:
        # responses (which may contain line breaks) stay in scores.jsonl; the CSV carries scalars only
        metrics.write_rows_csv(out / "scored_rows.csv", [{k: v for k, v in r.items() if k != "response"} for r in records],
                               extra_columns=["config_sha256", "generated_tokens", "seconds"])
    metrics.write_table_csv(out / "delta_geometry.csv", geometry)
    metrics.write_table_csv(out / "delta_energy_by_layer.csv", layers)
    return 0


# ---------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out")
    p = sub.add_parser("native-qualify"); p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--arm", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--order", default="canonical"); p.add_argument("--out", required=True)
    p.add_argument("--independent-from", help="C only: the B run directory")
    p.add_argument("--lr-from", help="LR_PILOT.json of this registration (the registered LR gate)")
    p = sub.add_parser("lr-pilot"); p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate"); p.add_argument("--run", required=True); p.add_argument("--include-base", action="store_true")
    p = sub.add_parser("interact"); p.add_argument("--run", required=True)
    p = sub.add_parser("analyse"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "native-qualify": cmd_native_qualify, "lr-pilot": cmd_lr_pilot,
            "train": cmd_train, "evaluate": cmd_evaluate,
            "interact": cmd_interact, "analyse": cmd_analyse}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
