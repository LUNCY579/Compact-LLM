"""Fresh-candidate BIND custody; frozen baseline inputs/code are read-only."""
from __future__ import annotations
from pathlib import Path
import math
import sys
import time

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.binding_signal_eval_v1 import common as baseline
from research.binding_mixture_v1 import run_probe

HERE = Path(__file__).resolve().parent
EXPERIMENT = "BINDING-MIXTURE-001"
LIMITS = {"workers": 2, "worker_seconds": 900, "total_worker_seconds": 1800, "rows": 768,
          "reserved_generated_tokens": 18432, "forward_calls": 19200, "processed_input_tokens": 225540,
          "retries": 1, "cloud_spend": 0}
require, read, sha, write_new, absolute, digest = (getattr(baseline, name) for name in
    ("require", "read", "sha", "write_new", "absolute", "digest"))


def context(args):
    grant = read(args.grant)
    fields = {"schema", "record", "experiment", "decision", "authorization_reference", "build_gate_path", "build_gate_sha256",
              "baseline_plan_path", "baseline_plan_sha256", "arm_path", "arm_weight_sha256", "limits", "deadline_unix", "code_sha256"}
    require(set(grant) == fields and grant["schema"] == 1 and grant["record"] == "BINDING_MIXTURE_DEV_GRANT" and
            grant["experiment"] == EXPERIMENT and grant["decision"] == "APPROVED" and
            isinstance(grant["authorization_reference"], str) and grant["authorization_reference"].strip(), "fresh mixed BIND grant required")
    require(grant["limits"] == LIMITS and all(type(v) is int for v in grant["limits"].values()), "mixed BIND grant changes frozen limits")
    require(absolute(grant["build_gate_path"]) == absolute(args.build_gate) and sha(args.build_gate) == grant["build_gate_sha256"], "candidate build gate changed")
    require(absolute(grant["baseline_plan_path"]) == absolute(args.baseline_plan) and
            sha(args.baseline_plan) == grant["baseline_plan_sha256"] == args.baseline_plan_sha256, "frozen gold-free baseline plan changed")
    gate = read(args.build_gate)
    require(str(gate.get("status", "")).startswith("PASS"), "candidate gate is not PASS")
    require(sha(Path(args.build_gate).parent / "PREREG.md") == gate["registration_sha256"], "mixed registration changed")
    envelope_path = Path(args.build_gate).parent / "SESSION_ENVELOPE.json"
    require(sha(envelope_path) == gate["session_envelope_sha256"], "mixed session envelope changed")
    envelope = read(envelope_path)
    require(grant["deadline_unix"] == envelope["deadline_unix"], "BIND local grant changed the global session deadline")
    require(absolute(grant["arm_path"]) == absolute(args.arm) == Path(args.build_gate).resolve().parent / "arms/mixed", "mixed candidate arm path differs")
    base = baseline.validate_plan(read(args.baseline_plan))
    require(set(grant["code_sha256"]) == {"binding_common.py", "binding_probe.py", "run_probe.py"}, "mixed BIND code pin inventory differs")
    require({name: sha(HERE / name) for name in grant["code_sha256"]} == grant["code_sha256"], "mixed BIND wrapper code changed")
    require(baseline.pilot_code_files(run_probe.PILOT) == base["pilot_code_sha256"], "clone B scientific loader/supervisor code differs from baseline")
    require({name: sha(baseline.HERE / name) for name in base["code_sha256"]} == base["code_sha256"], "frozen BIND generation code changed")
    require(sha(base["input_path"]) == base["input_sha256"], "frozen BIND DEV input changed")
    baseline.validate_inputs(baseline.read_jsonl(base["input_path"]))
    return grant, gate, base


def model_snapshot(args, grant, gate, base):
    """Worker scope excludes DATA_BUILD/DEV_GOLD; parent checks training provenance."""
    folder = absolute(args.arm)
    weight = sha(folder / "model.safetensors")
    require(weight == grant["arm_weight_sha256"], "mixed candidate weight identity changed")
    assets = {name: sha(folder / name) for name in baseline.SIDECARS}
    require(assets == base["sidecars_sha256"] == {name: sha(Path(base["source_dir"]) / name) for name in baseline.SIDECARS},
            "mixed/original tokenizer sidecars differ")
    for name in ("ARM.json", "MATERIALIZATION.json"):
        receipt = read(folder / name)
        require(receipt["out_weight_sha256"] == receipt["arm_weight_sha256"] == weight and
                receipt["source_weight_sha256"] == baseline.MODEL_SHAS["source"] and receipt["stored_dtype"] == "float32",
                "mixed candidate receipt/source/precision differs")
    require(baseline.runtime_versions() == base["runtime_versions"], "mixed BIND runtime differs from baseline")
    return {"weight_sha256": weight, "sidecars_sha256": assets,
            "arm_receipt_sha256": sha(folder / "ARM.json"), "materialization_sha256": sha(folder / "MATERIALIZATION.json"),
            "input_sha256": base["input_sha256"], "runtime_versions": baseline.runtime_versions(),
            "frozen_generation_code_sha256": base["code_sha256"], "pilot_code_sha256": baseline.pilot_code_files(run_probe.PILOT),
            "wrapper_code_sha256": {name: sha(HERE / name) for name in grant["code_sha256"]}}


def slot(args):
    root = Path(args.build_gate).resolve().parent
    out = absolute(args.out)
    require(out.parent == root / "eval" and out.name in ("mixed-binding", "mixed-binding-retry1"), "BIND attempt is not a registered output slot")
    return root, out


def time_guard(grant, seconds=0):
    require(type(grant["deadline_unix"]) in (int, float) and math.isfinite(grant["deadline_unix"]) and
            time.time() + seconds <= grant["deadline_unix"], "BIND attempt does not fit remaining session deadline")


def require_retry_failure(root, before):
    first = root / "eval/mixed-binding"
    binding = read(first / "BINDING.json")
    supervisor = read(first / "supervisor.json")
    terminal = read(first / "worker_terminal.json")
    require(binding["status"] == "FAILED" and binding["complete"] is False and
            supervisor["status"] == "EXITED" and type(supervisor["returncode"]) is int and supervisor["returncode"] != 0 and
            terminal["status"] == "FAILED" and terminal["complete"] is False and binding["before"] == before and
            terminal["before"] in (None, before),
            "BIND retry lacks a verified identical-input failed first attempt")
    require(not (first / "GENERATIONS.json").exists(), "a BIND attempt with surviving generations cannot be replaced")


def execution_identity(directory, args, base, supervisor):
    launch, spawn, claim = (read(directory / name) for name in ("launch.json", "spawn.json", "worker_claim.json"))
    argv = launch["argv"]
    require(isinstance(argv, list) and len(argv) >= 2 and all(isinstance(value, str) for value in argv), "invalid BIND launch argv")
    def option(name):
        require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "BIND launch argument is missing/duplicated")
        return argv[argv.index(name) + 1]
    require(absolute(launch["out"]) == directory and absolute(launch["cwd"]) == run_probe.PILOT.resolve() and
            launch["timeout_seconds"] == 900 and absolute(argv[0]) == absolute(base["python_path"]) and
            absolute(argv[1]) == HERE / "binding_probe.py" and argv.count("--_worker") == argv.count("--confirm-model-execution") == 1,
            "BIND launch changed worker/interpreter/cwd/output")
    for name in ("arm", "out", "build_gate", "baseline_plan", "grant"):
        require(absolute(option("--" + name.replace("_", "-"))) == absolute(getattr(args, name)), "BIND launch inputs differ")
    require(option("--baseline-plan-sha256") == args.baseline_plan_sha256 and
            spawn["launch_sha256"] == claim["launch_sha256"] == digest(launch) and
            spawn["supervisor_pid"] == claim["supervisor_pid"] == launch["supervisor_pid"] and
            supervisor["child_pid"] == spawn["child_pid"] and
            (claim["pid"] == spawn["child_pid"] or (claim.get("launcher_redirected") is True and claim["parent_pid"] == spawn["child_pid"])),
            "BIND supervisor and claimed worker do not belong to this launch")
    return {stem + "_sha256": sha(directory / (stem + ".json")) for stem in ("launch", "spawn", "worker_claim")}


def verify_binding(directory, *, check_live=True):
    """No scoring occurs until this successful foreground custody gate passes."""
    from types import SimpleNamespace
    directory = absolute(directory, directory=True)
    binding = read(directory / "BINDING.json")
    require(binding["status"] == "COMPLETE" and binding["complete"] is True and
            type(binding["controller_returncode"]) is int and binding["controller_returncode"] == 0,
            "mixed BIND foreground execution did not succeed")
    args = SimpleNamespace(**binding["arguments"])
    grant, gate, base = context(args)
    require(sha(args.grant) == binding["grant_sha256"], "mixed BIND grant changed")
    require(slot(args)[1] == directory, "mixed BIND binding points to another output slot")
    require(not (directory / "FAILURE.json").exists() and not (directory / "supervisor_failure.json").exists(), "mixed BIND failure receipt exists")
    for stem, name in (("supervisor", "supervisor.json"), ("worker_terminal", "worker_terminal.json"), ("generations", "GENERATIONS.json")):
        require(sha(directory / name) == binding[stem + "_sha256"], "mixed BIND artifact hash changed")
    supervisor, terminal, generations = (read(directory / name) for name in ("supervisor.json", "worker_terminal.json", "GENERATIONS.json"))
    baseline.verify_terminal(supervisor, terminal)
    require(all(binding.get(k) == v for k, v in execution_identity(directory, args, base, supervisor).items()), "mixed BIND launch identity changed")
    require(generations["before"] == generations["after"] == terminal["before"] == terminal["after"] == binding["before"], "mixed BIND before/after custody differs")
    before = generations["before"]
    require(before["weight_sha256"] == grant["arm_weight_sha256"] and before["sidecars_sha256"] == base["sidecars_sha256"] and
            before["runtime_versions"] == base["runtime_versions"] and before["input_sha256"] == base["input_sha256"] and
            before["frozen_generation_code_sha256"] == base["code_sha256"] and before["pilot_code_sha256"] == base["pilot_code_sha256"] and
            before["wrapper_code_sha256"] == grant["code_sha256"], "mixed BIND recorded runtime/code/input identity differs from registration")
    require(generations["grant_sha256"] == terminal["grant_sha256"] == binding["grant_sha256"] and
            generations["build_gate_sha256"] == terminal["build_gate_sha256"] == sha(args.build_gate) and
            terminal["baseline_plan_sha256"] == generations["baseline_plan_sha256"] and
            terminal["generations_sha256"] == binding["generations_sha256"], "mixed BIND terminal/plan/grant artifact identity differs")
    if check_live:
        # Runtime versions in an offline interpreter may differ, so verify bytes here.
        live = run_probe.arm_snapshot(Path(args.arm), gate)
        require(live == binding["training_before"] == binding["training_after"] and
                live["weight_sha256"] == before["weight_sha256"] and
                live["arm_receipt_sha256"] == before["arm_receipt_sha256"] and
                live["materialization_sha256"] == before["materialization_sha256"], "mixed training/model custody changed")
    require(generations["numerics"] == baseline.NUMERICS and generations["gold_file_read"] is False and
            generations["baseline_plan_sha256"] == sha(args.baseline_plan) and generations["rows"] == 384 and
            generations["worlds"] == 32 and generations["model_role"] == "mixed" and
            generations["record"] == "BINDING_MIXTURE_DEV_GENERATIONS",
            "mixed BIND measurement settings/input provenance differ")
    rows = baseline.validate_inputs(baseline.read_jsonl(base["input_path"])); by_id = {r["id"]: r for r in generations["predictions"]}
    require(len(by_id) == len(generations["predictions"]) == 384 and by_id.keys() == {r["id"] for r in rows}, "mixed BIND row coverage differs")
    eos = generations["eos_ids"]
    require(isinstance(eos, list) and bool(eos) and len(eos) == len(set(eos)) and
            all(type(t) is int and 0 <= t < 151936 for t in eos), "mixed BIND EOS IDs missing or invalid")
    for row in rows:
        pred = by_id[row["id"]]; tokens = pred["generated_ids_including_eos"]
        require(isinstance(tokens, list) and type(pred["terminated"]) is bool and
                pred["parent_id"] == row["parent_id"] and pred["input_tokens"] == len(row["prompt_ids"]) and
                1 <= len(tokens) == pred["generated_tokens"] == pred["forward_calls"] <= 24 and
                all(type(t) is int and 0 <= t < 151936 for t in tokens) and pred["terminated"] == (tokens[-1] in eos) and
                not any(t in eos for t in tokens[:-1]) and pred["processed_input_tokens"] == len(row["prompt_ids"]) + len(tokens) - 1,
                "mixed BIND token/EOS/forward accounting differs")
    actual = {"rows": len(by_id), "generated_tokens_including_eos": sum(p["generated_tokens"] for p in by_id.values()),
              "forward_calls": sum(p["forward_calls"] for p in by_id.values()),
              "processed_input_tokens": sum(p["processed_input_tokens"] for p in by_id.values())}
    require(generations["actual_budget_charges"] == actual and generations["extra_warmup_or_qualification_calls"] == 0 and
            actual["generated_tokens_including_eos"] <= 9216 and actual["forward_calls"] <= 9600 and
            actual["processed_input_tokens"] <= 112770, "mixed BIND actual budget accounting differs or exceeds local subcap")
    return generations, binding, base
