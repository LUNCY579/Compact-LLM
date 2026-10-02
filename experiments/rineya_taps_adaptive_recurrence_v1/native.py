"""Guarded native DEV runner; no native TEST, training, CUDA or download lane."""
from __future__ import annotations
from dataclasses import asdict, replace
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from .experiment import GRID, SEEDS, csv_write
from .integration import BlockRecurrence
from .scheduler import TAPSConfig
from .safety import BASE, Journal, file_hash, native_preflight, runtime_guard, write_json

REPO = Path(__file__).resolve().parents[2]
ARMR = REPO/"experiments/anchor_routed_middle_recurrence_v1"
HISTORY = ARMR/"evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z"


def committed_bytes(path):
    rel = path.relative_to(REPO).as_posix()
    expected = subprocess.check_output(["git", "-C", str(REPO), "show", f"{BASE}:{rel}"], stderr=subprocess.PIPE)
    if path.read_bytes() != expected:
        raise RuntimeError(f"inherited source or receipt changed: {rel}")
    return expected


def preflight(source):
    result = native_preflight(source)
    try:
        # All inherited Python dependencies must be the exact integrated base.
        roots = [ARMR/"src", ARMR/"scripts", REPO/"Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"]
        for root in roots:
            if not root.is_dir(): raise RuntimeError(f"missing inherited source tree: {root.name}")
            for path in root.rglob("*.py"): committed_bytes(path)
        historical = json.loads(committed_bytes(HISTORY/"run_manifest.json"))
        if historical["status"] != "PASS": raise RuntimeError("historical qualification is not PASS")
        committed_bytes(ARMR/"configs/reapplication_screen.json")
        if source is None: raise RuntimeError("canonical source directory not provided")
        source = Path(source)
        expected = historical["checkpoint_files_before"]
        if any(not (source/n).is_file() or file_hash(source/n) != h for n,h in expected.items()):
            raise RuntimeError("checkpoint/tokenizer/custom model bytes differ from qualified ARMR source")
        for path in source.rglob("*.py"):
            if path.relative_to(source).as_posix() not in expected:
                raise RuntimeError("unverified Python inside model source directory")
        import transformers
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        if transformers.__version__ != historical["transformers"]:
            raise RuntimeError("historical Transformers version mismatch")
        if file_hash(Path(inspect.getfile(Qwen3DecoderLayer))) != historical["qwen_implementation_sha256"]:
            raise RuntimeError("native Qwen implementation hash mismatch")
        result["historical_receipt_sha256"] = file_hash(HISTORY/"run_manifest.json")
    except Exception as exc:
        result["reasons"].append(f"{type(exc).__name__}: {exc}")
    result["status"] = "BLOCKED" if result["reasons"] else "METADATA_READY_NOT_QUALIFIED"
    return result


def run(out, source):
    out, source = Path(out), None if source is None else Path(source)
    if source is not None and (out.resolve() == source.resolve() or source.resolve() in out.resolve().parents):
        raise ValueError("output must not overlap canonical model directory")
    out.mkdir(parents=True, exist_ok=False)
    check = preflight(source)
    write_json(out/"preflight.json", check)
    if check["status"] == "BLOCKED": return 2
    sys.path[:0] = [str(ARMR/"src"), str(REPO/"Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from armr.evaluation import evaluate_case
    from armr.provenance import DATA, cases_from, checkpoint_files, state_hash
    from armr.hooks import Reapplication
    from armr.recurrence import FixedBlock
    from armr.reporting import summarize as smoke_gate
    historical = json.loads((HISTORY/"run_manifest.json").read_text())
    config = dict(historical["config"])
    config["max_rss_gib"] = 6  # tighter new memory ceiling, never an old-gate change
    dev_path = DATA/config["development_file"]
    if file_hash(dev_path) != historical["dataset_sha256"]:
        raise RuntimeError("MERGEDEV-fast bytes changed")
    smoke, dev = cases_from(DATA/config["smoke_file"]), cases_from(dev_path)
    taps = TAPSConfig()
    write_json(out/"registration.json", {"base": BASE, "taps": asdict(taps), "grid": GRID,
               "seeds": SEEDS, "native_layers": [12,13], "extra_loops": [0,1],
               "config": config, "dataset_sha256": file_hash(dev_path),
               "dataset_status": "previously inspected DEV; NOT protected TEST",
               "training": False, "gate_interaction": "ungated ARMR host; no learned gate changed"})
    started = time.perf_counter(); model = None
    receipt = {"status": "FAIL", "native_model_loads": 0, "training": False, "sealed_read": False}
    before_files = checkpoint_files(source)
    try:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.manual_seed(17)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=True,
                    dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True).eval().requires_grad_(False)
        receipt["native_model_loads"] = 1
        if type(model).__name__ != "Qwen3AbstainForCausalLM": raise RuntimeError("custom abstention-aware Supa required")
        before_state = state_hash(model)
        if before_state != historical["state_hash_before"]: raise RuntimeError("loaded model state hash mismatch")
        rows, traces, bases = [], [], {}
        with Journal(out/"outcomes.jsonl") as outcomes, Journal(out/"trajectories.jsonl") as trajectory:
            def evaluate(case, arm, cfg, extra, seed, split, legacy=False):
                torch.manual_seed(seed)
                case_start = time.perf_counter()
                def sink(e):
                    e = {**e, "case_id": case["case_id"], "split": split, "seed": seed, "arm": arm,
                         "extra_loops": extra}
                    trajectory.append(e)
                    if arm == "A3" and split == "DEV" and e["iteration"] > 0: traces.append(e)
                if legacy:
                    wrapper = FixedBlock(model, [12,13], applications=2) if extra else Reapplication(model)
                else:
                    wrapper = BlockRecurrence(model, extra_loops=extra, config=cfg, sample_id=case["case_id"], sink=sink,
                        guard=lambda: runtime_guard(case_start, 900.))
                with wrapper, torch.inference_mode():
                    row, logits = evaluate_case(model, tokenizer, case, config, wrapper, bases.get((split,seed,case["case_id"])))
                    row.update(arm=arm, split=split, seed=seed, extra_loops=extra,
                               fixed_alpha=cfg.fixed_alpha if cfg.scheduler_mode == "fixed" else None,
                               controller_seconds=getattr(wrapper,"controller_seconds",None),
                               controller_logging_seconds=getattr(wrapper,"logging_seconds",None),
                               full_model_calls_including_gold_nll=getattr(wrapper,"forward_count",None),
                               total_token_positions_including_gold_nll=getattr(wrapper,"token_positions",None),
                               physical_run=True, deployable=True)
                    outcomes.append(row)
                    if row["safety_stopped"]: raise RuntimeError("native safety stop; matrix incomplete, do not select survivors")
                return row, logits
            unit = replace(taps, scheduler_mode="unit")
            # Fresh native passthrough and unit-step equivalence, before scaling.
            for c in smoke:
                for extra in (0,1):
                    old, old_logits = evaluate(c, "qualification_legacy", unit, extra, 17, "SMOKE", legacy=True)
                    new, new_logits = evaluate(c, "qualification_new", unit, extra, 17, "SMOKE")
                    if old["token_ids"] != new["token_ids"] or not torch.equal(old_logits, new_logits):
                        raise RuntimeError("new wrapper failed exact legacy equivalence")
            arms = [("A0",unit,0), ("A1",unit,1)] + [("A2",replace(taps,scheduler_mode="fixed",fixed_alpha=a),1) for a in GRID] + [("A3",taps,1)]
            baseline_smoke = []
            for arm,cfg,extra in arms:
                current = []
                for c in smoke:
                    r,l = evaluate(c,arm,cfg,extra,17,"SMOKE"); current.append(r)
                    if arm == "A0": bases[("SMOKE",17,c["case_id"])] = l
                if arm == "A0": baseline_smoke = current
                elif not smoke_gate(current,baseline_smoke,config)["go_fixed_smoke"]:
                    raise RuntimeError(f"new-arm smoke gate failed: {arm}, alpha={cfg.fixed_alpha}; retain all evidence")
            for seed in SEEDS:
                # Baseline must precede pairs; rotate only the treatment order.
                ordered = arms[:1] + (arms[1:] if seed == 17 else list(reversed(arms[1:])))
                for arm,cfg,extra in ordered:
                    for c in dev:
                        r,l = evaluate(c,arm,cfg,extra,seed,"DEV"); rows.append(r)
                        if arm == "A0": bases[("DEV",seed,c["case_id"])] = l
            scores = {a: statistics.mean(r["score"] for r in rows if r["arm"]=="A2" and r["fixed_alpha"]==a) for a in GRID}
            best = max(GRID,key=lambda a:(scores[a],-a))
            pool = tuple(min(taps.alpha_max,max(taps.alpha_min,e["alpha"])) for e in traces) or (1.,)
            write_json(out/"frozen_dev_selection.json", {"alpha":best,"pool":pool,"config":asdict(taps),
                       "dataset_sha256":file_hash(dev_path),"native_TEST_authorized":False})
            for seed in SEEDS:
                for c in dev:
                    r,_ = evaluate(c,"A5",replace(taps,scheduler_mode="random",random_pool=pool),1,seed,"DEV"); rows.append(r)
            oracle_rows = []
            for seed in SEEDS:
                for c in dev:
                    rr=[r for r in rows if r["arm"]=="A2" and r["seed"]==seed and r["case_id"]==c["case_id"]]
                    selected=max(rr,key=lambda r:(r["score"],-r["fixed_alpha"]))
                    selected={**selected,"arm":"A4_example","physical_run":False,"deployable":False,
                              "note":"retrospective constant alpha per problem, not an exhaustive token-prefix schedule oracle"}
                    oracle_rows.append(selected); outcomes.append(selected)
            rows.extend(oracle_rows)
            restored = state_hash(model)==before_state and checkpoint_files(source)==before_files
            if not restored: raise RuntimeError("model or source mutated")
            table=[]
            for arm in ("A0","A1","A2_DEV_selected","A3","A4_example","A5"):
                rr=[r for r in rows if (r["arm"]==arm or arm=="A2_DEV_selected" and r["arm"]=="A2" and r["fixed_alpha"]==best)]
                table.append({"arm":arm,"accuracy":statistics.mean(r["score"] for r in rr),
                              "wall_seconds":statistics.mean(r["wall_seconds"] for r in rr),
                              "extra_loops":0 if arm=="A0" else 1,"deployable":not arm.startswith("A4")})
            csv_write(out/"native_dev_matrix.csv",table)
            csv_write(out/"fixed_alpha.csv",[{"alpha":a,"accuracy":v} for a,v in scores.items()])
            accuracy={r["arm"]:r["accuracy"] for r in table}
            headroom=accuracy["A4_example"]-scores[best]
            receipt.update(status="DEV_COMPLETE_NOT_TESTED", source_unchanged=restored,
                fixed_alpha=best, adaptive_headroom=headroom,
                taps_minus_best_fixed=accuracy["A3"]-scores[best],
                capture_ratio=None if headroom<=1e-12 else (accuracy["A3"]-scores[best])/headroom,
                oracle_scope="per-example constant only; not upper bound on unrestricted per-token schedules",
                native_test=False, fixed_loop_comparison="one extra block application", training_gate_passed=False)
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        receipt["wall_seconds"] = time.perf_counter()-started
        receipt["checkpoint_files_unchanged"] = checkpoint_files(source)==before_files
        if model is not None:
            receipt["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        write_json(out/"receipt.json",receipt)
    return 0 if receipt["status"]=="DEV_COMPLETE_NOT_TESTED" else 2
