"""One explicitly confirmed, supervised gold-free controlled diagnostic worker.

No retries or scoring. Uses the frozen RAB DenseReference loader and its existing
greedy cache mechanics. Only prompt token IDs enter the model; row/world IDs are
opaque join keys and do not change tokenization or decoding.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if __package__:
    from . import common as c
else:
    import common as c


def setup_pilot(plan):
    c.code_snapshot(plan)
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    head = subprocess.check_output(["git", "-C", plan["pilot_root"], "rev-parse", "HEAD"],
                                   text=True, stderr=subprocess.PIPE, timeout=30, **options).strip()
    c.require(head == plan["pilot_head"], "frozen pilot checkout changed")
    sys.path.insert(0, plan["pilot_root"])


def worker_command(plan_path, grant_path, role, plan, out, attempt=1):
    c.require(role in c.MODEL_SHAS, "unknown diagnostic model")
    return [plan["python_path"], str(Path(__file__).resolve()), "--plan", str(c.absolute(plan_path)),
            "--grant", str(c.absolute(grant_path)), "--model", role, "--attempt", str(attempt), "--out", str(out),
            "--confirm-model-execution", "--_worker"]


def generate(model, tokenizer, rows, eos, torch):
    """Frozen process_probe.py greedy/cache mechanics; never reads gold fields."""
    predictions = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            # IDs/parents are used only when writing the result after generation.
            rendered = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=False)
            c.require(rendered == row["prompt_ids"], "native tokenizer no longer reproduces compiled prompt IDs")
            ids = torch.tensor([rendered], dtype=torch.long, device="cuda:0")
            generated, past, current, terminated, processed_tokens = [], None, ids, False, 0
            for _ in range(24):
                processed_tokens += int(current.shape[-1])
                step = model(current, past=past, use_cache=True, last_position_only=True)
                past = step.cache
                token = int(step.logits[0, -1].argmax())
                generated.append(token)
                if token in eos:
                    terminated = True
                    break
                current = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
            del past, step, ids, current
            predictions.append({"id": row["id"], "parent_id": row["parent_id"],
                                "prediction": tokenizer.decode(generated, skip_special_tokens=True),
                                "terminated": terminated, "input_tokens": len(rendered),
                                "generated_tokens": len(generated), "generated_ids_including_eos": generated,
                                "forward_calls": len(generated), "processed_input_tokens": processed_tokens})
            if index % 48 == 0:
                print(f"Generated {index}/{c.NROWS} diagnostic rows; no scoring", flush=True)
    return predictions


def worker(args, plan):
    setup_pilot(plan)
    from experiments.model_quality_gpu_v1.supervisor import claim_worker
    out = c.absolute(args.out, directory=True)
    claim_worker(out)
    before = after = None
    started = time.monotonic()
    terminal = {"schema": 1, "status": "FAILED", "complete": False, "failure": "worker did not finish",
                "model_role": args.model, "plan_sha256": c.sha(args.plan), "grant_sha256": c.sha(args.grant)}
    try:
        grant = c.read(args.grant); c.validate_grant(args.plan, grant)
        c.require(out == c.absolute(plan["output_root"]) / c.slot_name(args.model, args.attempt), "worker output is outside its fixed model slot")
        c.require(Path(sys.executable).resolve() == c.absolute(plan["python_path"]), "worker interpreter differs from pinned Python")
        rows = c.validate_inputs(c.read_jsonl(plan["input_path"]))
        c.verify_worker_reservation(plan, args.plan, args.grant, args.model, args.attempt, rows)
        from experiments.model_quality_gpu_v1.resources import guard, lease
        lock = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
        with lease(lock):
            c.phase_guard(plan, grant, seconds_needed=900)
            import psutil
            c.require(psutil.virtual_memory().available >= 5 * 2**30, "unchanged 5 GiB RAB probe host guard refused")
            resources_before = guard(before_load=True, phase="qualify")
            before = c.snapshot(plan, args.model)
            import torch
            from experiments.architecture_benchmark_001 import run
            torch.set_num_threads(1); torch.manual_seed(17)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            c.require(torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False and
                      torch.get_float32_matmul_precision() == "highest", "registered FP32 numerical controls did not apply")
            numerics = dict(c.NUMERICS)
            c.write_new(out / "PRECHECK.json", {"status": "PRECHECK_PASSED", "before": before,
                        "model_role": args.model, "resources_before": resources_before, "numerics": numerics,
                        "model_input_fields": ["prompt_ids"], "worker_input_fields": sorted(c.INPUT_KEYS),
                        "gold_file_read": False, "plan_sha256": terminal["plan_sha256"], "grant_sha256": terminal["grant_sha256"]})
            model, tokenizer, unused_native = run._load(Path(plan["models"][args.model]["path"]), "cuda:0")
            c.require(unused_native is None and all(p.dtype == torch.float32 for p in model.parameters()),
                      "diagnostic loader must create exactly one FP32 reference model")
            eos = run._eos_ids(Path(plan["source_dir"]), tokenizer)
            predictions = generate(model, tokenizer, rows, eos, torch)
            del model, tokenizer
            torch.cuda.empty_cache()
            phase_after = c.phase_guard(plan, grant); resources_after = guard(phase="qualify")
            after = c.snapshot(plan, args.model)
            c.require(before == after and c.sha(args.plan) == terminal["plan_sha256"] and
                      c.sha(args.grant) == terminal["grant_sha256"], "diagnostic model/input/code/runtime custody changed")
            result = {"schema": 1, "record": "BINDING_DIAGNOSTIC_GENERATIONS", "model_role": args.model,
                      "plan_sha256": terminal["plan_sha256"], "grant_sha256": terminal["grant_sha256"],
                      "rows": len(predictions), "parent_worlds": 128, "predictions": predictions, "eos_ids": sorted(eos),
                      "numerics": numerics, "gold_file_read": False, "evaluation_scope": "exploratory controlled diagnostic",
                      "before": before, "after": after, "quality_claim": False, "training_permission": False,
                      "actual_budget_charges": {"rows": len(predictions),
                           "generated_tokens_including_eos": sum(p["generated_tokens"] for p in predictions),
                           "forward_calls": sum(p["forward_calls"] for p in predictions),
                           "processed_input_tokens": sum(p["processed_input_tokens"] for p in predictions)},
                      "phase_after": phase_after, "extra_warmup_or_qualification_calls": 0}
            c.write_new(out / "GENERATIONS.json", result)
            terminal.update(status="COMPLETE", complete=True, failure=None,
                            generations_sha256=c.sha(out / "GENERATIONS.json"), resources_after=resources_after)
        # Lease cleanup must succeed before a successful worker terminal is written.
    except BaseException as exc:
        terminal.update(status="FAILED", complete=False, failure={"type": type(exc).__name__, "message": str(exc)})
        c.write_new(out / "FAILURE.json", terminal)
        raise
    finally:
        terminal.update(before=before, after=after, wall_seconds=time.monotonic() - started)
        c.write_new(out / "worker_terminal.json", terminal)
    return 0


def controller_admitted(args, plan):
    grant = c.read(args.grant); c.validate_grant(args.plan, grant)
    c.require(Path(sys.executable).resolve() == c.absolute(plan["python_path"]), "launch requires the exact pinned probe Python")
    setup_pilot(plan)
    c.require(not (Path(tempfile.gettempdir()) / "rineya-gpu-0.lock").exists(), "shared worker lease is already present")
    rows = c.validate_inputs(c.read_jsonl(plan["input_path"]))
    c.require(c.sha(plan["input_path"]) == plan["input_sha256"], "DEV input hash differs")
    root = c.absolute(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    c.phase_guard(plan, grant, seconds_needed=900)
    before_costs = c.attempt_costs(plan, args.grant)['totals']
    slot = c.slot_name(args.model, args.attempt)
    out = root / slot
    c.require(not out.exists(), "attempt output already exists; no overwrite or automatic retry")
    if args.attempt == 2:
        original = root / args.model
        prior = c.read(original / 'DIAGNOSTIC_BINDING.json')
        prior_supervisor = c.read(original / 'supervisor.json')
        c.require(prior['status'] == 'FAILED' and prior['complete'] is False and
                  prior['model_role'] == args.model and prior['attempt'] == 1 and
                  prior['plan_sha256'] == c.sha(args.plan) and prior['grant_sha256'] == c.sha(args.grant) and
                  prior['supervisor_sha256'] == c.sha(original / 'supervisor.json') and
                  not (original / 'GENERATIONS.json').exists() and
                  prior_supervisor.get('status') in ('EXITED', 'TIMEOUT_OR_INTERRUPTED') and
                  type(prior_supervisor.get('returncode')) is int and prior_supervisor['returncode'] != 0,
                  'one explicit retry requires a verified original worker failure')
        c.require(before_costs['retries'] == 0, 'the one global inference retry is already consumed')
    elif (root / (args.model + '-retry1')).exists():
        raise ValueError('retry cannot precede first attempt')
    reservation = {'schema': 1, 'model_role': args.model, 'attempt': args.attempt,
        'plan_sha256': c.sha(args.plan), 'grant_sha256': c.sha(args.grant),
        'reserved_worker_seconds': 900, 'rows': c.NROWS,
        'controller_lease_sha256': c.sha(root / 'CONTROLLER_LEASE.json'),
        'controller_owner': c.read(root / 'CONTROLLER_LEASE.json'),
        'reserved_generated_tokens': c.NROWS * 24, 'forward_calls': c.NROWS * 24,
        'processed_input_tokens': sum(len(row['prompt_ids']) for row in rows) + c.NROWS * 24}
    added = {'workers': 1, 'total_worker_seconds': 900, 'retries': args.attempt - 1,
             **{key: reservation[key] for key in ('rows', 'reserved_generated_tokens', 'forward_calls', 'processed_input_tokens')}}
    c.require(all(before_costs[key] + value <= c.LIMITS[key] for key, value in added.items()),
              'remaining all-attempt diagnostic budget cannot fit this worker')
    c.write_new(root / ('RESERVATION_' + slot + '.json'), reservation)
    from experiments.model_quality_gpu_v1.supervisor import bounded_child
    code = bounded_child(worker_command(args.plan, args.grant, args.model, plan, out, args.attempt), out, 900)
    supervisor = c.read(out / "supervisor.json")
    success = False; reason = None; launch_identity = {}
    try:
        c.require(code == 0 and not (out / "FAILURE.json").exists(), "supervised diagnostic failed")
        terminal = c.read(out / "worker_terminal.json"); c.verify_terminal(supervisor, terminal)
        launch_identity = c.verify_launch(out, args.plan, args.grant, plan, args.model, supervisor, args.attempt)
        c.require(terminal["plan_sha256"] == c.sha(args.plan) and terminal["grant_sha256"] == c.sha(args.grant) and
                  terminal["generations_sha256"] == c.sha(out / "GENERATIONS.json"), "terminal artifact identity differs")
        c.phase_guard(plan, grant)
        c.attempt_costs(plan, args.grant)
        success = True
    except (OSError, ValueError, TypeError, KeyError) as exc:
        reason = str(exc)
    receipt = {"schema": 1, "status": "COMPLETE" if success else "FAILED",
                "complete": success, "model_role": args.model, "attempt": args.attempt,
                "controller_returncode": 0 if success else 2, "worker_returncode": code,
                "plan_sha256": c.sha(args.plan), "grant_sha256": c.sha(args.grant),
                "reservation_sha256": c.sha(root / ("RESERVATION_" + slot + ".json")),
                "supervisor_sha256": c.sha(out / "supervisor.json"),
                "worker_terminal_sha256": c.sha(out / "worker_terminal.json") if (out / "worker_terminal.json").is_file() else None,
                "generations_sha256": c.sha(out / "GENERATIONS.json") if (out / "GENERATIONS.json").is_file() else None,
                "failure": reason, "quality_claim": False, "training_permission": False, **launch_identity}
    return out, receipt


def controller(args, plan):
    # This is separate from the GPU lease acquired only inside the model worker.
    c.validate_grant(args.plan, c.read(args.grant))
    with c.controller_lease(plan, args.plan, args.grant):
        out, receipt = controller_admitted(args, plan)
    # A successful individual receipt is published only after controller cleanup.
    c.phase_guard(plan, c.read(args.grant))
    c.write_new(out / 'DIAGNOSTIC_BINDING.json', receipt)
    print({'status': receipt['status'], 'model_role': args.model, 'output': str(out)})
    return receipt['controller_returncode']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--grant", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(c.MODEL_SHAS), required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--confirm-model-execution", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    c.require(args.confirm_model_execution, "model execution requires explicit --confirm-model-execution")
    plan = c.validate_plan(c.read(args.plan))
    if args._worker:
        c.require(args.out is not None, "private worker requires the supervisor output")
        return worker(args, plan)
    c.require(args.out is None, "output is fixed in the plan; private worker arguments are not public overrides")
    return controller(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
