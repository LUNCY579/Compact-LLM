"""PARENTQ native worker; launched only by parentq_supervise.py.

It measures the GEO screen's parents (BASE, T, H) on fresh qualification rows, before the
one-shot GEO screen is spent. It runs no geometric arm, generates on no J or K row and
opens no protected bank (PREREG_PARENTQ.md).

Native internals come from the FROZEN GEO section H code, never from copies. The worker
binds GEO through GEO's own dependency binding and validates the whole GEO freeze
(FREEZE_SECTIONH.json, registered digest), including its CPU validation receipt, source
sidecars, weight digests and runtime packages. It then uses GEO's resource preflight, model
load call, block-14 hook (held at IDENTITY, as in the screen's parent units), padded input
layout, padding-parity procedure, in-memory arm loading and scorer. Its own code is the
schedule, the budget caps, the lease rule for this lane's label, and `decode_until_eos`.
That function is GEO's fixed-slot greedy decode stopped at the first EOS, and a CPU test
shows it returns GEO's tokens exactly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
RUNTIME_ROOT = HERE.parent.parent.parent / "triskill-geo001-parentq-runtime"
GEO_DIR = HERE.parent / "triskill_geometric_pilot_v1"
GEO_FREEZE_NAME = "FREEZE_SECTIONH.json"
GEO_FREEZE_SHA256 = "f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687"
LEASE_LABEL = "triskill-geo001-parentq"
FREEZE_NAME = "FREEZE_PARENTQ.json"
BANK_JSON, CANARY_JSON = "parentq_bank.json", "parentq_canary.json"
START_NAME = "SUPERVISOR_START_PARENTQ.json"
WORKER_PID_NAME = "WORKER_PID_PARENTQ.json"
LOAD_BEGUN_NAME = "LOAD_BEGUN_PARENTQ.json"
PROGRESS_NAME = "native_progress_parentq.jsonl"
BLOCKED_NAME = "blocked_result_parentq.json"
INCOMPLETE_NAME = "incomplete_result_parentq.json"
OUT_NAME = "native_parentq.json"
MIN_GRANT_CHARS = 12
WALL_SECONDS = 540                 # internal; parentq_supervise.py enforces the 600 s hard wall
PRELOAD_RAM_GIB = 6.5              # provisional fail-safe; NOT a measured native lane requirement
PARITY_SLOTS = 4
REQUIRED_FILES = ("parentq_worker.py", "parentq_rows.py", "parentq_decide.py", "parentq_supervise.py",
                  "parentq_launch_wmi.py", "parentq_safety.py", "READINESS_PARENTQ.json",
                  BANK_JSON, CANARY_JSON, "PREREG_PARENTQ.md")
_NO_LOCK = HERE / "__resources_only_no_lock__"   # GEO preflight measures resources; the lease is checked here
# The schedule: each arm's canary first, then its qualification blocks; weight loads BASE -> H -> T.
SCHEDULE = (("BASE", "CANARY"), ("BASE", "T"), ("BASE", "H_ON"), ("BASE", "H_OFF"),
            ("H", "CANARY"), ("H", "H_ON"), ("H", "H_OFF"),
            ("T", "CANARY"), ("T", "T"))
TERMINAL_PREFIXES = ("PARENTS_", "STOP_ARM_CANARY:", "STOP_NATIVE_MECHANICS_PADDING",
                     "NONDETERMINISM_OBSERVED", "UNVERIFIED_OVERLAP", "STOP_ATTEMPT_OVERLAP_ERROR")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _exec_pinned(path: Path, expected: str, name: str):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError(f"STOP_PINNED_SOURCE_MISMATCH:{path.name}")
    spec = importlib.util.spec_from_file_location(f"_parentq_{name}_{expected[:12]}", path)
    module = importlib.util.module_from_spec(spec)
    module._parentq_source_binding = (str(path), expected)
    sys.modules[spec.name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def bind_geo(geo_dir: Path = GEO_DIR, geo_freeze_sha: str = GEO_FREEZE_SHA256):
    """The frozen GEO worker, then GEO's own full freeze validation through it."""
    freeze_path = geo_dir / GEO_FREEZE_NAME
    source = freeze_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != geo_freeze_sha:
        raise ValueError("STOP_GEO_FREEZE_DIGEST_MISMATCH")
    geo_freeze = json.loads(source)
    entry = geo_freeze["dependency_binding"]["roles"]["binding"]
    binding = _exec_pinned(geo_dir / "dependency_binding.py", entry["sha256"], "geo_binding")
    geo = binding.validate_dependencies(geo_freeze["dependency_binding"]).load_worker()
    _, bound, geometry = geo._validate_freeze(freeze_path, geo_freeze_sha)
    return geo, bound, geometry


def allocation(n_generations: int, input_slots: int, output_slots: int) -> dict:
    """Upper bounds as if every generation used all output slots (decoding stops at EOS)."""
    parity_forwards, parity_positions = 2 * 2 * PARITY_SLOTS, 2 * 2 * (input_slots + PARITY_SLOTS - 1)
    forwards = parity_forwards + n_generations * output_slots
    positions = parity_positions + n_generations * (input_slots + output_slots - 1)
    return {"generations": n_generations, "forwards": forwards, "token_positions": positions,
            "parts": {"padding_parity": [parity_forwards, parity_positions],
                      "generation": [n_generations * output_slots, n_generations * (input_slots + output_slots - 1)]}}


def schedule_units(bank: dict, canary: list) -> list:
    blocks = {"CANARY": [row["row_id"] for row in canary], **{name: [row.row_id for row in rows]
                                                              for name, rows in bank.items()}}
    return [(arm, block, blocks[block]) for arm, block in SCHEDULE]


class Budget:
    def __init__(self, caps: dict, now=time.monotonic):
        self.caps, self.now = caps, now
        self.started = now()
        self.forwards = self.completed_forwards = self.token_positions = 0
        self.generations = self.completed_generations = 0

    def check(self):
        if self.now() - self.started >= WALL_SECONDS:
            raise RuntimeError("STOP_WALL_BUDGET")

    def forward(self, positions: int):
        self.check()
        if self.forwards + 1 > self.caps["forwards"] or self.token_positions + positions > self.caps["token_positions"]:
            raise RuntimeError("STOP_FORWARD_OR_TOKEN_BUDGET")
        self.forwards += 1
        self.token_positions += positions

    def generation(self):
        self.check()
        if self.generations >= self.caps["generations"]:
            raise RuntimeError("STOP_GENERATION_BUDGET")
        self.generations += 1

    def receipt(self):
        return {"model_forwards": self.forwards, "token_positions_including_padding": self.token_positions,
                "completed_model_forwards": self.completed_forwards, "generations_requested": self.generations,
                "completed_generations": self.completed_generations, "elapsed_seconds": self.now() - self.started}


def decode_until_eos(geo, model, torch, device, ids, *, pad_id, eos_ids, state, budget, reserve=None,
                     reference_tokens=None):
    """GEO's `decode` (padded, fixed input slots, greedy) stopped at the first EOS.

    Every forward up to and including the EOS step is the same call GEO's fixed-slot
    decode makes, so the text tokens are identical; GEO's compute-only suffix is skipped.
    """
    input_ids, mask, positions = geo._padded(torch, device, ids, pad_id, True)
    cache, generated, raw, margins, reference_margins = None, [], [], [], []
    for step in range(geo.OUTPUT_SLOTS):
        if reserve is not None:
            reserve()
        state.valid = mask if cache is None else torch.ones_like(input_ids)
        budget.forward(int(input_ids.numel()))
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                           past_key_values=cache, use_cache=True)
        budget.completed_forwards += 1
        cache = output.past_key_values
        logits = output.logits[0, -1]
        token = int(logits.argmax().item())
        top = torch.topk(logits.float(), 2).values
        raw.append(token)
        margins.append(float((top[0] - top[1]).item()))
        if reference_tokens is not None:
            reference_margins.append(float((logits[token].float() - logits[reference_tokens[step]].float()).item())
                                     if step < len(reference_tokens) else None)
        if token in eos_ids:
            break
        generated.append(token)
        input_ids = torch.tensor([[token]], device=device)
        mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype, device=device)), dim=1)
        positions = torch.tensor([[len(ids) + step]], device=device)
    return {"tokens": generated, "scored_tokens": raw, "margins": margins,
            "reference_token_margins": reference_margins, "ended": bool(raw) and raw[-1] in eos_ids}


def lease_owned(lease, token, ancestors) -> bool:
    return (isinstance(lease, dict) and token is not None and lease.get("token") == token
            and lease.get("pid") in ancestors and lease.get("label") == LEASE_LABEL)


def preflight(geo, token, *, lock=None, **kwargs) -> dict:
    """GEO preflight plus this lane's lease and provisional conservative RAM floor."""
    if _NO_LOCK.exists():
        raise RuntimeError("STOP_PREFLIGHT_SENTINEL_EXISTS")
    lock = geo.LOCK if lock is None else lock
    ancestors = set(kwargs.pop("ancestors", None) or geo.ancestor_pids())
    record = geo.preflight(None, lock=_NO_LOCK, ancestors=ancestors, **kwargs)
    record["parentq_preload_ram_floor"] = {"gib": PRELOAD_RAM_GIB, "provisional_fail_safe": True,
                                         "native_lane_measurement": False}
    if record.get("host_ram_free_gib", 0) < PRELOAD_RAM_GIB:
        record["blockers"].append("RAM below provisional 6.5 GiB PARENTQ preload fail-safe")
    lease = None
    if Path(lock).exists():
        try:
            lease = json.loads(Path(lock).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lease = "unreadable; fail closed"
    record["lease"] = lease
    record["lease_owned_by_supervisor"] = owned = lease_owned(lease, token, ancestors)
    if not owned:
        record["blockers"].append("shared GPU lease not held by this worker's supervisor")
    record["status"] = "BLOCKED" if record["blockers"] else "RESOURCES_CLEAR"
    return record


def verify_supervisor_start(attempt_dir: Path, lease_token: str, freeze_sha: str) -> dict:
    start_path = Path(attempt_dir) / START_NAME
    if not start_path.exists():
        raise RuntimeError("STOP_NO_SUPERVISOR_START_RECEIPT")
    start = json.loads(start_path.read_text(encoding="utf-8"))
    grant = start.get("operator_grant")
    if start.get("lease_token") != lease_token or not isinstance(grant, str) or len(grant.strip()) < MIN_GRANT_CHARS:
        raise RuntimeError("STOP_NO_SUPERVISOR_GRANT")
    if (start.get("experiment") != "TRISKILL-GEO-001-PARENTQ" or start.get("freeze_sha256") != freeze_sha
            or type(start.get("attempt_number")) is not int or start["attempt_number"] < 1):
        raise RuntimeError("STOP_SUPERVISOR_ATTEMPT_BINDING")
    try:
        value = json.loads(grant)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("STOP_NO_ATTEMPT_BOUND_SUPERVISOR_GRANT") from exc
    if (not isinstance(value, dict) or value.get("experiment") != "TRISKILL-GEO-001-PARENTQ"
            or value.get("freeze_sha256") != freeze_sha or value.get("attempt_number") != start["attempt_number"]
            or len(str(value.get("operator_grant", "")).strip()) < MIN_GRANT_CHARS
            or not str(value.get("hardware_slot", "")).strip()):
        raise RuntimeError("STOP_NO_ATTEMPT_BOUND_SUPERVISOR_GRANT")
    return start


def validate_freeze(freeze_path: Path, freeze_sha: str, here: Path = HERE) -> dict:
    data = freeze_path.read_bytes()
    if not freeze_sha or hashlib.sha256(data).hexdigest() != freeze_sha:
        raise ValueError("STOP_FREEZE_DIGEST_MISMATCH")
    freeze = json.loads(data)
    runtime_root = freeze.get("runtime_root")
    if (not isinstance(runtime_root, str) or not Path(runtime_root).is_absolute()
            or Path(runtime_root).resolve() != RUNTIME_ROOT.resolve()):
        raise ValueError("STOP_NONCANONICAL_NATIVE_RUNTIME_ROOT")
    files = freeze.get("files", {})
    if not set(REQUIRED_FILES) <= set(files):
        raise ValueError("STOP_INCOMPLETE_FREEZE_MEMBERSHIP")
    for name, expected in files.items():
        if Path(name).name != name or sha256(here / name) != expected:
            raise ValueError(f"STOP_FREEZE_FILE_MISMATCH:{name}")
    if freeze.get("geo_freeze", {}).get("sha256") != GEO_FREEZE_SHA256:
        raise ValueError("STOP_GEO_FREEZE_NOT_REGISTERED")
    validation = freeze.get("cpu_validation", {})
    if "path" in validation:
        receipt = Path(str(validation["path"]))
        repo = HERE.parent.parent.resolve()
        business = repo.parent
        resolved = receipt.resolve()
        if (not receipt.is_absolute() or not resolved.is_relative_to(business) or resolved.is_relative_to(repo)
                or ".codex" in [part.lower() for part in resolved.parts]
                or any((parent / ".git").exists() for parent in (resolved, *resolved.parents))):
            raise ValueError("STOP_CPU_VALIDATION_OUTPUT_LOCATION")
    else:  # Historical frozen packets used a local pinned receipt.
        receipt = here / str(validation.get("file", ""))
        if receipt.parent != here:
            raise ValueError("STOP_NO_PASSING_PINNED_CPU_VALIDATION")
    if validation.get("passed") is not True or not receipt.is_file():
        raise ValueError("STOP_NO_PASSING_PINNED_CPU_VALIDATION")
    receipt_bytes = receipt.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != validation.get("sha256"):
        raise ValueError("STOP_NO_PASSING_PINNED_CPU_VALIDATION")
    measured = json.loads(receipt_bytes)
    training = measured.get("delta_H_training_rows", {})
    native = training.get("native_prompt_counts", {})
    if (measured.get("passed") is not True or measured.get("validation_profile") != "FULL_CPU_REGISTRATION"
            or measured.get("mechanical_checks_passed") is not True or measured.get("design_ready") is not True
            or measured.get("counts") != {"T": 24, "H_ON": 32, "H_OFF": 32}
            or measured.get("canary_n") != 8 or measured.get("canary_eligible_rows") != 56
            or measured.get("canary_pairwise_different_before_length_filter") != 58
            or training.get("screen_contract_rows") != 10
            or training.get("screen_contract_rows_with_abstention_gold") != 0
            or set(native) != {"native_a", "native_b"}
            or any(v.get("rows") != 28 or v.get("exact_abstention_replies") != 28 for v in native.values())):
        raise ValueError("STOP_CPU_VALIDATION_FACTS_NOT_VERIFIED")
    geo_bytes = (GEO_DIR / GEO_FREEZE_NAME).read_bytes()
    if hashlib.sha256(geo_bytes).hexdigest() != GEO_FREEZE_SHA256:
        raise ValueError("STOP_GEO_FREEZE_DIGEST_MISMATCH")
    pinned_lease = json.loads(geo_bytes)["supervisor_dependencies"]["gpu_lease"]
    if freeze.get("supervisor_dependencies", {}).get("gpu_lease") != pinned_lease:
        raise ValueError("STOP_GPU_LEASE_PIN_DIFFERS_FROM_GEO")
    return freeze


def _write_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _check_output_location(path: Path) -> None:
    repo = HERE.parent.parent.resolve()
    resolved = Path(path).resolve()
    if (not resolved.is_relative_to(repo.parent) or resolved.is_relative_to(repo)
            or ".codex" in [part.lower() for part in resolved.parts]
            or any((parent / ".git").exists() for parent in (resolved, *resolved.parents))):
        raise ValueError("STOP_WORKER_OUTPUT_LOCATION")


def _refusal_receipt(attempt_dir: Path, freeze_sha: str, exc: Exception) -> dict:
    """Persist failures even when bootstrap/setup has not built the full result yet."""
    attempt_dir = Path(attempt_dir)
    begun = (attempt_dir / LOAD_BEGUN_NAME).exists()
    result = {"experiment": "TRISKILL-GEO-001-PARENTQ", "status": str(exc),
              "failure_type": type(exc).__name__, "confirmation": False, "model_loads": 0,
              "model_load_attempts": int(begun), "load_begun": begun, "attempt_dir": str(attempt_dir),
              "freeze_sha256": freeze_sha, "utc_end": utc_now(),
              "verdict": {"status": "STOPPED_BEFORE:padding"}, "preload_or_bootstrap_refusal": True}
    target = attempt_dir / (INCOMPLETE_NAME if begun else BLOCKED_NAME)
    try:
        _check_output_location(target)
        if not target.exists():
            _write_json(target, result)
    except Exception as write_error:
        result["receipt_write_error"] = {"type": type(write_error).__name__, "message": str(write_error)}
    return result


def _worker_identity(attempt_dir: Path, start: dict, token: str, freeze_sha: str) -> None:
    import psutil
    process = psutil.Process(os.getpid())
    record = {"pid": os.getpid(), "create_time": process.create_time(), "lease_token": token,
              "freeze_sha256": freeze_sha, "supervisor_pid": start["supervisor_pid"],
              "ancestors": sorted(p.pid for p in process.parents()), "utc": utc_now()}
    target = Path(attempt_dir) / WORKER_PID_NAME
    _check_output_location(target)
    if target.exists():
        old = json.loads(target.read_bytes())
        if any(old.get(k) != record[k] for k in ("pid", "create_time", "lease_token", "freeze_sha256", "supervisor_pid")):
            raise RuntimeError("STOP_WORKER_IDENTITY_RECORD_COLLISION")
    else:
        _write_json(target, record)


def safe_decide(decide, benchmark, bank, canary, outputs, mechanics, stopped_by):
    """A decision failure is a persisted error, never an exception escaping finally."""
    try:
        return decide.decide(benchmark, bank, canary, outputs, mechanics, stopped_by=stopped_by), None
    except Exception as exc:
        return {"status": "STOP_DECISION_ERROR", "decision_usable": False}, {
            "type": type(exc).__name__, "message": str(exc), "utc": utc_now()}


def canary_divergence(row, arm, text, decoded, tokenizer, eos_ids) -> dict:
    """Diagnostic only: stored text has no native token trace, so re-tokenize it explicitly."""
    source = {"BASE": "BASE", "T": "SPARSE", "H": "H25"}[arm]
    stored = row["stored"][source].strip()
    if text.strip() == stored:
        return {"text_matches": True}
    expected = tokenizer(stored, add_special_tokens=False)["input_ids"] + [eos_ids[0]]
    actual = decoded["scored_tokens"]
    index = next((i for i in range(max(len(expected), len(actual)))
                  if (expected[i] if i < len(expected) else None) != (actual[i] if i < len(actual) else None)), None)
    return {"text_matches": False, "reference_source": source,
            "reference_tokens_reconstructed_from_stored_text": True,
            "comparison_basis": "canonical stripped stored-text tokens plus first registered EOS; native source token ids unavailable",
            "first_divergence_index": index,
            "first_divergence_observed_token": actual[index] if index is not None and index < len(actual) else None,
            "first_divergence_reference_token": expected[index] if index is not None and index < len(expected) else None,
            "first_divergence_observed_margin": decoded["margins"][index] if index is not None and index < len(decoded["margins"]) else None,
            "first_divergence_chosen_minus_reference_logit": decoded.get("reference_token_margins", [])[index]
            if index is not None and index < len(decoded.get("reference_token_margins", [])) else None}


def compare_attempt_overlap(attempt_dir: Path, outputs: dict, records: dict, freeze_sha: str) -> dict:
    """Compare every prior same-freeze parent unit; never choose an attempt or row outcome."""
    comparisons, differences, errors = [], [], []
    for prior in sorted(Path(attempt_dir).parent.iterdir()):
        if prior == Path(attempt_dir) or not prior.is_dir() or not (prior / LOAD_BEGUN_NAME).is_file():
            continue
        try:
            start = json.loads((prior / START_NAME).read_bytes())
            if start.get("experiment") != "TRISKILL-GEO-001-PARENTQ" or start.get("freeze_sha256") != freeze_sha:
                continue
            overlap = []
            progress = prior / PROGRESS_NAME
            if not progress.is_file():
                errors.append({"attempt": str(prior), "diagnostic": "missing prior progress trace"})
            else:
                lines = progress.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines):
                    try:
                        event = json.loads(line)
                    except ValueError:
                        if i == len(lines) - 1:  # killed writers may leave a truncated last checkpoint
                            errors.append({"attempt": str(prior), "diagnostic": "truncated final progress event"})
                            continue
                        raise
                    if not isinstance(event, dict):
                        raise ValueError("non-object progress event")
                    if event.get("event") != "generation":
                        continue
                    if (event.get("arm") not in {"BASE", "H", "T"} or not isinstance(event.get("row_id"), str)
                            or not isinstance(event.get("text"), str) or not isinstance(event.get("scored_tokens"), list)
                            or any(type(token) is not int for token in event["scored_tokens"])):
                        raise ValueError("unverifiable generation progress event")
                    arm, row_id = event.get("arm"), event.get("row_id")
                    if row_id not in outputs.get(arm, {}):
                        continue
                    current = records.get(arm, {}).get(row_id, {})
                    same_text = event.get("text") == outputs[arm][row_id]
                    same_tokens = event.get("scored_tokens") == current.get("scored_tokens")
                    item = {"attempt": str(prior), "arm": arm, "row_id": row_id,
                            "text_identical": same_text, "tokens_identical": same_tokens}
                    overlap.append(item)
                    if not same_text or not same_tokens:
                        differences.append(item)
            comparisons.append({"attempt": str(prior), "overlapping_units": len(overlap), "units": overlap})
        except (OSError, ValueError, TypeError) as exc:
            errors.append({"attempt": str(prior), "diagnostic": str(exc)})
    status = ("NONDETERMINISM_OBSERVED" if differences else "UNVERIFIED_OVERLAP" if errors
              else "NO_DIFFERENCE_ON_OBSERVED_OVERLAP")
    return {"status": status,
            "comparisons": comparisons, "differences": differences, "diagnostic_errors": errors,
            "select_outcome": False, "retry_policy": "PENDING_OPERATOR_DECISION"}


def run(out: Path, freeze_path: Path, freeze_sha: str, lease_token: str, attempt_dir: Path):
    """Bootstrap: execute this worker's own pinned bytes, then run from that module."""
    started = time.monotonic()
    try:
        start = verify_supervisor_start(attempt_dir, lease_token, freeze_sha)
        _worker_identity(attempt_dir, start, lease_token, freeze_sha)
        freeze = validate_freeze(freeze_path, freeze_sha)
        module = _exec_pinned(HERE / "parentq_worker.py", freeze["files"]["parentq_worker.py"], "worker")
        return module._run_bound(out, freeze_path, freeze_sha, lease_token, attempt_dir, _started=started)
    except Exception as exc:
        return _refusal_receipt(attempt_dir, freeze_sha, exc)


def _run_bound(out: Path, freeze_path: Path, freeze_sha: str, lease_token: str, attempt_dir: Path, _started=None,
               _bind_geo=bind_geo):
    try:
        return _run_impl(out, freeze_path, freeze_sha, lease_token, attempt_dir, _started, _bind_geo)
    except Exception as exc:
        return _refusal_receipt(attempt_dir, freeze_sha, exc)


def _run_impl(out: Path, freeze_path: Path, freeze_sha: str, lease_token: str, attempt_dir: Path, _started=None,
              _bind_geo=bind_geo):
    """One model load; BASE, H and T in that weight order; per-generation checkpoints.

    Output policy (PREREG_PARENTQ.md section 7): an attempt that never reaches LOAD_BEGUN
    writes BLOCKED_NAME in its attempt directory. The marker spends an attempt even if
    from_pretrained raises before a resident model exists. A loaded attempt with a terminal verdict
    writes the create-only OUT_NAME; a loaded attempt that stops short writes
    INCOMPLETE_NAME in its attempt directory, and a later granted attempt may rerun it.
    """
    if getattr(sys.modules.get(__name__), "_parentq_source_binding", None) != (str(HERE / "parentq_worker.py"), sha256(HERE / "parentq_worker.py")):
        raise RuntimeError("STOP_UNBOUND_PARENTQ_WORKER")
    attempt_dir = Path(attempt_dir)
    freeze = validate_freeze(freeze_path, freeze_sha)
    rows_module = _exec_pinned(HERE / "parentq_rows.py", freeze["files"]["parentq_rows.py"], "rows")
    decide = _exec_pinned(HERE / "parentq_decide.py", freeze["files"]["parentq_decide.py"], "decide")
    safety = _exec_pinned(HERE / "parentq_safety.py", freeze["files"]["parentq_safety.py"], "safety")
    start_receipt = verify_supervisor_start(attempt_dir, lease_token, freeze_sha)
    if Path(freeze["runtime_root"]).resolve() != safety.RUNTIME_ROOT.resolve():
        raise ValueError("STOP_NONCANONICAL_NATIVE_RUNTIME_ROOT")
    runtime_root = safety.validate_runtime_root(Path(start_receipt["runtime_root"]))
    if runtime_root != Path(freeze["runtime_root"]).resolve():
        raise ValueError("STOP_NATIVE_RUNTIME_ROOT_PIN_MISMATCH")
    if not attempt_dir.resolve().is_relative_to(runtime_root) or not Path(out).resolve().is_relative_to(runtime_root):
        raise ValueError("STOP_WORKER_OUTPUT_OUTSIDE_RUNTIME_ROOT")
    _worker_identity(attempt_dir, start_receipt, lease_token, freeze_sha)
    safety.validate_readiness(HERE)
    safety.check_runtime_ownership(start_receipt, lease_token)
    geo, bound, geometry = _bind_geo()
    benchmark = bound.load_local("benchmark")
    bank = rows_module.build_bank(benchmark)
    if [row.to_dict() for row in rows_module.all_rows(bank)] != json.loads((HERE / BANK_JSON).read_text(encoding="utf-8")):
        raise ValueError("STOP_BANK_GENERATOR_DIFFERS_FROM_FROZEN_MEMBERSHIP")
    canary = json.loads((HERE / CANARY_JSON).read_text(encoding="utf-8"))["rows"]
    units = schedule_units(bank, canary)
    caps = allocation(sum(len(ids) for _, _, ids in units), geo.INPUT_SLOTS, geo.OUTPUT_SLOTS)
    if caps != freeze["native_allocation"]:
        raise ValueError("STOP_ALLOCATION_CHANGED")
    budget = Budget(caps)
    if _started is not None:
        budget.started = _started
    import numpy as np
    import psutil
    snapshot = preflight(geo, lease_token)
    if snapshot["blockers"]:
        raise RuntimeError(json.dumps(snapshot))
    supervisor_pid = snapshot["lease"]["pid"]
    if out.exists():
        raise FileExistsError("create-only PARENTQ output exists")
    geo_receipt = json.loads((GEO_DIR / geo.GEOMETRY_JSON).read_text(encoding="utf-8"))
    if geo_receipt["status"] != "PREPARED" or sha256(GEO_DIR / geo.GEOMETRY_NPZ) != geo_receipt["geometry_sha256"]:
        raise ValueError("STOP_GEO_GEOMETRY_UNVERIFIED")
    for path, expected in geometry.SOURCES.values():
        budget.check()
        if sha256(path) != expected:
            raise ValueError("STOP_SOURCE_WEIGHT_DIGEST_CHANGED")
    with np.load(GEO_DIR / geo.GEOMETRY_NPZ, allow_pickle=False) as archive:
        basis = archive["basis"]
    renderer = bound.load_renderer()
    renderer.template_selftest()
    rows_by_id = {row.row_id: row for row in rows_module.all_rows(bank)}
    canary_by_id = {row["row_id"]: row for row in canary}
    development = benchmark.build_development()

    def supervisor_alive():
        if not psutil.pid_exists(supervisor_pid):
            raise RuntimeError("STOP_SUPERVISOR_LOST")
        if psutil.Process(supervisor_pid).create_time() != start_receipt["supervisor_create_time"]:
            raise RuntimeError("STOP_SUPERVISOR_IDENTITY_CHANGED")

    def host_reserve(extra_bytes=0):
        supervisor_alive()
        if (psutil.virtual_memory().available - extra_bytes) / 2**30 < geo.ACTIVE_RAM_GIB:
            raise RuntimeError("STOP_HOST_ACTIVE_RESERVE")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    result = {"experiment": "TRISKILL-GEO-001-PARENTQ", "status": "STARTED", "confirmation": False,
              "geometric_arms_run": 0, "j_or_k_rows_generated": 0,
              "training_examples": 0, "model_loads": 0, "weight_loads": 0, "attempt_dir": str(attempt_dir),
              "operator_grant": start_receipt["operator_grant"], "worker_pid": os.getpid(),
              "supervisor_pid": supervisor_pid, "resources_before": snapshot, "allocation": caps,
              "freeze_sha256": freeze_sha, "geo_freeze_sha256": GEO_FREEZE_SHA256, "outputs": {}, "records": {},
              "dependency_binding": bound.receipt(),
              "runtime": {"torch": torch.__version__, "transformers": transformers.__version__, "dtype": "float32",
                          "attention": "eager", "fixed_input_slots": geo.INPUT_SLOTS,
                          "max_output_slots": geo.OUTPUT_SLOTS, "decoding": "greedy, stops at first EOS",
                          "eos_token_ids": list(geo.EOS_TOKEN_IDS), "tf32": False,
                          "gpu_memory_fraction": geo.GPU_MEMORY_FRACTION, "hook": "block 14 at IDENTITY"}}
    outputs, mechanics = result["outputs"], {}
    progress = (attempt_dir / PROGRESS_NAME).open("x", encoding="utf-8", newline="\n")

    def record_progress(entry):
        progress.write(json.dumps({**entry, "utc": utc_now(), "elapsed_seconds": budget.now() - budget.started}) + "\n")
        progress.flush()

    try:
        budget.check()
        resources = preflight(geo, lease_token)
        result["resources_immediately_before_load"] = resources
        if resources["blockers"]:
            raise RuntimeError(json.dumps(resources))
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(geo.GPU_MEMORY_FRACTION, 0)
        if torch.cuda.mem_get_info()[0] / 2**30 < geo.PRELOAD_GPU_GIB:
            raise RuntimeError("STOP_GPU_CHANGED_BEFORE_LOAD")

        def reserve():
            if torch.cuda.mem_get_info()[0] / 2**30 < geo.ACTIVE_GPU_GIB:
                raise RuntimeError("STOP_GPU_ACTIVE_RESERVE")
            host_reserve()

        safety.check_runtime_ownership(start_receipt, lease_token)
        bound.verify_current()
        _write_json(attempt_dir / LOAD_BEGUN_NAME, {"experiment": "TRISKILL-GEO-001-PARENTQ", "utc": utc_now(),
                    "worker_pid": os.getpid(), "attempt_number": start_receipt["attempt_number"],
                    "freeze_sha256": freeze_sha, "lease_token": lease_token,
                    "counts_as_loaded_attempt": True, "note": "Created before from_pretrained; survives crash/hard kill."})
        result["model_load_attempts"] = 1
        result["load_begun"] = True
        record_progress({"event": "load_begun"})
        model = AutoModelForCausalLM.from_pretrained(str(geometry.SOURCES["BASE"][0].parent),
                    torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True,
                    low_cpu_mem_usage=True, device_map={"": "cuda"}).eval()
        result["model_loads"] = 1
        record_progress({"event": "model_loaded"})
        tok = AutoTokenizer.from_pretrained(str(geometry.SOURCES["BASE"][0].parent), local_files_only=True)
        configured = model.generation_config.eos_token_id
        configured = set(configured if isinstance(configured, list) else [configured])
        if configured != set(geo.EOS_TOKEN_IDS) or tok.eos_token_id != geo.EOS_TOKEN_IDS[0]:
            raise RuntimeError("STOP_EOS_CONFIGURATION_CHANGED")
        pad_id = geo.EOS_TOKEN_IDS[0]
        device = next(model.parameters()).device
        b = torch.as_tensor(basis, dtype=torch.float32, device=device)
        state = geo.HookState(torch, b)   # change = 0: IDENTITY, as in the screen's parent units
        handle = model.model.layers[geometry.LAYER].register_forward_hook(geo.make_hook(state))
        loaded = {"target": "BASE"}

        def ids_for(system, user):
            return tok(renderer.render(system, user), add_special_tokens=False)["input_ids"]

        try:
            parity = []
            parity_traces = result["development_padding_parity_traces"] = []
            for row in development[:2]:
                ids = ids_for(row.system, row.user)
                trace = {"row_id": row.row_id, "input_token_count": len(ids)}
                parity_traces.append(trace)
                unpadded = geo.decode(model, torch, device, ids, pad_id=pad_id, eos_ids=geo.EOS_TOKEN_IDS, state=state,
                                      budget=budget, slots=PARITY_SLOTS, padded=False, reserve=reserve)
                trace["unpadded"] = unpadded
                record_progress({"event": "padding_parity_trace", "layout": "unpadded", **trace})
                padded = geo.decode(model, torch, device, ids, pad_id=pad_id, eos_ids=geo.EOS_TOKEN_IDS, state=state,
                                    budget=budget, slots=PARITY_SLOTS, padded=True, reserve=reserve)
                trace["padded"] = padded
                trace["tokens_identical"] = unpadded["raw_tokens"] == padded["raw_tokens"]
                parity.append(trace["tokens_identical"])
                record_progress({"event": "padding_parity_trace", "layout": "paired", **trace})
            result["development_padding_parity"] = parity
            mechanics["padding_parity_ok"] = all(parity)
            record_progress({"event": "padding_parity", "ok": all(parity)})
            if not all(parity):
                raise RuntimeError("STOP_NATIVE_MECHANICS_PADDING")
            for arm, block, row_ids in units:
                budget.check()
                supervisor_alive()
                safety.check_runtime_ownership(start_receipt, lease_token)
                if loaded["target"] != arm:
                    geo.load_arm_weights(model, arm, torch, geometry, budget, host_reserve)
                    loaded["target"] = arm
                    result["weight_loads"] += 1
                    record_progress({"event": "weights_loaded", "arm": arm})
                if state.change.abs().max().item() != 0:
                    raise RuntimeError("STOP_HOOK_NOT_IDENTITY")
                texts = outputs.setdefault(arm, {})
                records = result["records"].setdefault(arm, {})
                for row_id in row_ids:
                    safety.check_runtime_ownership(start_receipt, lease_token)
                    budget.generation()
                    if block == "CANARY":
                        system, user = canary_by_id[row_id]["system"], canary_by_id[row_id]["user"]
                        source = {"BASE": "BASE", "T": "SPARSE", "H": "H25"}[arm]
                        reference_tokens = tok(canary_by_id[row_id]["stored"][source].strip(),
                                               add_special_tokens=False)["input_ids"] + [geo.EOS_TOKEN_IDS[0]]
                    else:
                        system, user = rows_by_id[row_id].system, rows_by_id[row_id].user
                        reference_tokens = None
                    decoded = decode_until_eos(geo, model, torch, device, ids_for(system, user), pad_id=pad_id,
                                               eos_ids=geo.EOS_TOKEN_IDS, state=state, budget=budget, reserve=reserve,
                                               reference_tokens=reference_tokens)
                    text = tok.decode(decoded["tokens"], skip_special_tokens=True)
                    texts[row_id] = text
                    records[row_id] = {"scored_tokens": decoded["scored_tokens"], "margins": decoded["margins"],
                                       "ended": decoded["ended"]}
                    if block == "CANARY":
                        records[row_id]["canary_divergence"] = canary_divergence(
                            canary_by_id[row_id], arm, text, decoded, tok, geo.EOS_TOKEN_IDS)
                    budget.completed_generations += 1
                    record_progress({"event": "generation", "arm": arm, "block": block, "row_id": row_id,
                                     "text": text, "scored_tokens": decoded["scored_tokens"],
                                     "margins": decoded["margins"],
                                     **({"canary_divergence": records[row_id]["canary_divergence"]} if block == "CANARY" else {})})
                if block == "CANARY":
                    check = decide.canary_check(canary, {arm: texts})[arm]
                    record_progress({"event": "canary", "arm": arm, **check})
                    if not check["passed"]:
                        raise RuntimeError(f"STOP_ARM_CANARY:{arm}")
            result["completed_schedule"] = True
        except Exception as exc:
            result["status"] = str(exc)
        finally:
            handle.remove()
    except Exception as exc:
        result["status"] = str(exc)
    finally:
        gc.collect()
        try:
            if torch.cuda.is_initialized():
                result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as cleanup_error:
            result["cleanup_error"] = str(cleanup_error)
        completed = bool(result.get("completed_schedule"))
        verdict, decision_error = safe_decide(decide, benchmark, bank, canary, outputs, mechanics,
                                              None if completed else result["status"])
        if decision_error is not None:
            result["decision_error"] = decision_error
            result["status"] = "STOP_DECISION_ERROR"
        result["verdict"] = verdict
        if completed:
            result["status"] = verdict["status"]
        try:
            overlap = compare_attempt_overlap(attempt_dir, outputs, result["records"], freeze_sha)
        except Exception as exc:
            overlap = {"status": "STOP_ATTEMPT_OVERLAP_ERROR", "differences": [], "select_outcome": False,
                       "error": {"type": type(exc).__name__, "message": str(exc)},
                       "retry_policy": "PENDING_OPERATOR_DECISION"}
            result["status"] = "STOP_ATTEMPT_OVERLAP_ERROR"
        result["attempt_overlap"] = overlap
        if overlap["status"] in {"NONDETERMINISM_OBSERVED", "UNVERIFIED_OVERLAP", "STOP_ATTEMPT_OVERLAP_ERROR"}:
            result["status"] = overlap["status"]
            result["decision_usable"] = False
            result["verdict"] = {"status": overlap["status"], "diagnostic_verdict": verdict,
                                 "decision_usable": False, "retry_policy": "PENDING_OPERATOR_DECISION"}
            verdict = result["verdict"]
        result["budget"] = budget.receipt()
        result["utc_end"] = dt.datetime.now(dt.timezone.utc).isoformat()
        progress.close()
        if not (attempt_dir / LOAD_BEGUN_NAME).exists():
            target = attempt_dir / BLOCKED_NAME
        elif verdict["status"].startswith(TERMINAL_PREFIXES):
            target = out
        else:
            target = attempt_dir / INCOMPLETE_NAME
        _write_json(target, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--out", type=Path, default=RUNTIME_ROOT / OUT_NAME)
    parser.add_argument("--freeze", type=Path, default=HERE / FREEZE_NAME)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--lease-token", required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.freeze, args.freeze_sha256, args.lease_token, args.attempt_dir), indent=2))
