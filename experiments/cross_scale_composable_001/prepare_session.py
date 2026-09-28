"""Prepare one fresh Stage C registration after resource recovery; never execute it.

Only the full three-role scope is supported: at least 12 GiB host and 6 GiB free
VRAM, and an unoccupied shared lease. No lower-memory partial registration, wait,
retry, tokenizer encoding, framework import, or model operation is performed.
Bound preflights from an ancestor commit remain usable only if their producing
helpers are byte-identical in that commit and the current clean checkout.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from . import receipts as r
from .build_calibration import build as rebuild_calibration
from .native_capture import read_json, verify_assets
from .native_preflight import host_memory

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "experiments/cross_scale_composable_001/"
AUTHORIZATION = "Commit and push the branch as you change and progress. Let's proceed with the goal to test if the model's intelligence can be improved further."
ROLES = ["student_base", "donor_base", "donor_skilled"]
ANALYSIS_PROTOCOL = {"version": "C-mechanism-002", "require_beats_shuffle": True,
                     "rcond": 1e-12, "seed": 1729}
RENDERING = {"add_generation_prompt": True, "enable_thinking": False,
             "add_special_tokens": False, "truncation": False, "padding": False}
PINS = {
    "student_base": {"torch": "2.10.0+cu126", "transformers": "4.57.1", "safetensors": "0.7.0",
                     "tokenizers": "0.22.2", "numpy": "2.4.6", "psutil": "7.2.2"},
    "donor": {"torch": "2.12.1+cu126", "transformers": "5.14.1", "safetensors": "0.8.0",
              "tokenizers": "0.22.2", "numpy": "2.4.6", "psutil": "7.2.2",
              "bitsandbytes": "0.50.0", "peft": "0.19.1", "accelerate": "1.14.0"}}
DEPENDENCIES = ["experiments/model_quality_gpu_v1/" + name for name in
                ("__init__.py", "integrity.py", "native.py", "resources.py", "source_lock.json", "supervisor.py")]
DEPENDENCIES += ["experiments/rab001_exploration/__init__.py", "experiments/rab001_exploration/hostbudget.py"]
RECEIPT_FILES = {"calibration": "evidence/CALIBRATION_BINDING.v2.json",
                 "architecture": "evidence/ARCHITECTURE_MAP.json", "assets": "evidence/BINDING_ASSETS.json"}


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=30)
    r.require(result.returncode == 0, "Git source inspection failed: " + " ".join(args))
    return result.stdout


def resource_snapshot(nvidia_smi):
    """No Torch/NVML Python import and no GPU context allocation."""
    value = {**host_memory(), "observed_utc": utc_now().isoformat()}
    command = [str(nvidia_smi), "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    r.require(result.returncode == 0, "NVIDIA resource measurement unavailable")
    rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) == 2 and row[0].strip() == "0"]
    r.require(len(matches) == 1, "GPU 0 resource measurement ambiguous")
    value.update(free_gpu_bytes=int(matches[0][1].strip()) * 2**20,
                 gpu_command=command, gpu_stdout=result.stdout, gpu_stderr=result.stderr)
    return value


def resource_gate(snapshot, lease_path):
    r.require(snapshot.get("available_host_bytes", 0) >= 12 * 2**30,
              "full Stage C registration requires 12 GiB available host RAM; no registration or wait started")
    r.require(snapshot.get("free_gpu_bytes", 0) >= 6 * 2**30,
              "full Stage C registration requires 6 GiB free VRAM")
    r.require(not Path(lease_path).exists(), "shared GPU lease occupied; never steal/delete it")


def source_inventory(root, commit):
    r.verify_source(root, commit, clean=True)
    names = {path.relative_to(root).as_posix() for path in (root / PACKAGE).glob("*.py")}
    names.update(DEPENDENCIES)
    names.update(PACKAGE + value for value in RECEIPT_FILES.values())
    names.update(PACKAGE + name for name in ("PREREGISTRATION.md", "STAGE_C_MECHANISM_AMENDMENT.md"))
    names.add(".gitattributes")
    return r.implementation_digest(root, sorted(names))


def verify_preflight_provenance(root, receipt, current_commit):
    source = receipt["source"]
    r.require(source["commit"]["returncode"] == source["worktree"]["returncode"] == 0,
              "tokenizer receipt has failed source inspection")
    r.require(source["worktree"]["stdout"] == "", "tokenizer receipt was produced from a dirty worktree")
    previous = source["commit"]["stdout"].strip()
    r.require(bool(re.fullmatch(r"[0-9a-f]{40}", previous)), "tokenizer source commit missing")
    git(root, "merge-base", "--is-ancestor", previous, current_commit)
    for filename, field in (("native_preflight.py", "helper_sha256"),
                            ("build_calibration.py", "selection_helper_sha256")):
        relative = PACKAGE + filename
        recorded = source[field]
        r.valid_hash(recorded, field)
        r.require(hashlib.sha256(git(root, "show", previous + ":" + relative)).hexdigest() == recorded,
                  "tokenizer producer does not match its committed source")
        r.require(r.file_digest(root / relative) == recorded, "tokenizer producer changed; fresh qualification required")


def validate_token_pair(student, donor, calibration, calibration_file_sha256):
    """Validate every original row, exact mask/position/rendering, and cross-runtime parity."""
    r.validate_calibration(calibration)
    r.require(len(calibration["rows"]) == 32, "expected fixed 32-row calibration")
    rows = []
    for name, receipt in (("student_base", student), ("donor", donor)):
        r.require(receipt.get("status") == "TOKEN_ELIGIBILITY_PASSED", "token qualification did not pass")
        r.require(receipt.get("model_loads") == receipt.get("model_forwards") == receipt.get("optimizer_steps") == 0,
                  "token preflight performed model work")
        r.require(receipt.get("cuda_initialized", False) is False, "token preflight initialized CUDA")
        r.require(receipt["calibration_file_sha256"] == calibration_file_sha256 and
                  receipt["calibration_sha256"] == calibration["calibration_sha256"] and
                  receipt["dataset_sha256"] == calibration["dataset_sha256"], "wrong calibration qualification")
        r.require(receipt["rendering"] == RENDERING and receipt["registered_cap"] == 256 and
                  receipt["overflow_ids"] == [], "unregistered rendering/overflow policy")
        r.require(receipt["encoded_rows_sha256"] == r.digest(receipt["rows"]), "token rows digest differs")
        r.require(receipt["row_count"] == len(receipt["rows"]) == 32, "tokenizer row count differs")
        r.require({k: v for k, v in receipt["runtime"]["packages"].items() if v is not None} == PINS[name],
                  "tokenizer runtime pins changed")
    for field in ("rows", "encoded_rows_sha256", "tokenizer_assets", "template_sha256", "rendering"):
        r.require(student[field] == donor[field], "cross-runtime token parity failed: " + field)
    for encoded, original in zip(student["rows"], calibration["rows"]):
        for key in ("id", "parent_id", "split", "open", "prompt_sha256"):
            r.require(encoded[key] == original[key], "original calibration row changed")
        r.require(r.digest(encoded["prompt"]) == original["prompt_sha256"], "message digest differs")
        ids, mask, positions = encoded["input_ids"], encoded["attention_mask"], encoded["position_ids"]
        r.require(isinstance(ids, list) and 0 < len(ids) <= 256 and
                  all(type(value) is int and 0 <= value < 151936 for value in ids), "invalid/overflowing input tokens")
        r.require(mask == [1] * len(ids) and positions == list(range(len(ids))), "mask/positions differ")
        r.require(encoded["input_ids_sha256"] == r.digest(ids) and
                  encoded["attention_mask_sha256"] == r.digest(mask) and encoded["tokens"] == len(ids),
                  "token/mask identity differs")
        prompt = encoded["rendered_prompt"]
        r.require(isinstance(prompt, str) and prompt and hashlib.sha256(prompt.encode()).hexdigest() ==
                  encoded["rendered_prompt_sha256"], "rendered prompt differs")
        rows.append({key: original[key] for key in ("id", "parent_id", "split", "open")} |
                    {"messages_sha256": original["prompt_sha256"], "prompt": prompt,
                     "prompt_sha256": r.digest(prompt), "input_ids": ids, "input_ids_sha256": r.digest(ids)})
    r.require(sum(row["split"] == "fit" for row in rows) == 24, "fit/check allocation differs")
    return rows


def runtime_binding(receipt, name):
    python = Path(receipt["runtime"]["python"]).resolve()
    r.require(python.is_file(), "native interpreter missing")
    code = ("import importlib.metadata as m,json,sys;names=" + repr(list(PINS[name])) +
            ";print(json.dumps({'python':sys.executable,'packages':{n:m.version(n) for n in names},"
            "'torch_imported':'torch' in sys.modules}))")
    probe = subprocess.run([str(python), "-B", "-c", code], capture_output=True, text=True, timeout=30)
    r.require(probe.returncode == 0, "runtime metadata probe failed")
    actual = json.loads(probe.stdout)
    r.require(Path(actual["python"]).resolve() == python and actual["packages"] == PINS[name]
              and actual["torch_imported"] is False, "native runtime/executable drift")
    result = {"python_executable": str(python), "python_executable_sha256": r.file_digest(python),
              "packages": dict(PINS[name]), "device": "cuda:0", "seed": 1729,
              "attention": "eager" if name == "student_base" else "sdpa"}
    if name == "donor":
        result["quantization"] = {"type": "nf4", "double_quant": True,
                                  "compute_dtype": "bfloat16", "nonquantized_dtype_after_prepare": "float32"}
    return result


def assets_binding(inventory):
    result = {}
    for role, group in (("student", "student_base"), ("donor", "donor_base"),
                        ("adapter", "donor_existing_adapter")):
        items = inventory["groups"][group]
        if role == "adapter":
            items = [item for item in items if Path(item["path"]).name in
                     ("adapter_model.safetensors", "adapter_config.json")]
            r.require(len(items) == 2, "missing adapter pair")
        directories = {Path(item["path"]).resolve().parent for item in items}
        r.require(len(directories) == 1, "model asset inventory crosses directories")
        files = {Path(item["path"]).name: item["sha256"] for item in items}
        r.require(len(files) == len(items), "duplicate model asset")
        value = {"directory": str(directories.pop()), "files": files, "model_sha256": r.digest(files)}
        verify_assets(value)
        result[role] = value
    return result


def ensure_fresh(output, session_id, now):
    r.require(bool(re.fullmatch(r"CSC-001-STAGE-C-[A-Za-z0-9-]+", session_id)), "invalid fresh session id")
    r.require(not output.exists(), "create-only session root already exists")
    r.require(output.parent.is_dir(), "existing external evidence namespace required")
    for path in output.parent.glob("*/SESSION_SPEC.json"):
        previous = read_json(path)
        r.require(previous.get("session_id") != session_id, "session id already registered")
        end = dt.datetime.fromisoformat(previous["deadline_utc"].replace("Z", "+00:00"))
        r.require(end.tzinfo is not None, "previous session deadline has no timezone")
        closed = path.parent / "SESSION_CLOSED.json"
        if closed.is_file():
            r.require(read_json(closed).get("spec_sha256") == r.file_digest(path),
                      "previous closure is not bound to its registration")
        finished = all((path.parent / "attempts" / role / "SESSION_TERMINAL.json").is_file() and
                       read_json(path.parent / "attempts" / role / "SESSION_TERMINAL.json").get("completed") is True
                       for role in ROLES)
        r.require(end <= now or closed.is_file() or finished, "another registered Stage C session is still open")


def prepare(*, root, source_commit, student_path, student_sha256, donor_path, donor_sha256,
            output, session_id, nvidia_smi, minutes=60):
    root, output = Path(root).resolve(), Path(output).resolve()
    r.require(type(minutes) is int and 1 <= minutes <= 60, "session deadline must be 1..60 minutes")
    r.require(not output.is_relative_to(root), "session evidence must be external")
    lease_path = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
    first = resource_snapshot(nvidia_smi)
    resource_gate(first, lease_path)  # Refuse low RAM before expensive hashing and before any write.
    ensure_fresh(output, session_id, utc_now())
    inventory = source_inventory(root, source_commit)
    student_path, donor_path = Path(student_path), Path(donor_path)
    r.verify_file(student_path, student_sha256)
    r.verify_file(donor_path, donor_sha256)
    student, donor = read_json(student_path), read_json(donor_path)
    for receipt in (student, donor):
        verify_preflight_provenance(root, receipt, source_commit)
    files = {key: root / PACKAGE / value for key, value in RECEIPT_FILES.items()}
    calibration, architecture, assets = (read_json(files[key]) for key in ("calibration", "architecture", "assets"))
    rebuild_calibration(calibration)
    rows = validate_token_pair(student, donor, calibration, r.file_digest(files["calibration"]))
    mapping = architecture["layer_map"]
    nearest = [min(range(28), key=lambda j: (abs(i * 27 - j * 35), j)) for i in range(36)]
    r.require(mapping["donor_to_student"] == nearest and
              mapping["capture_pairs_student_donor"] == [[0, 0], [13, 17], [27, 35]], "wrong frozen layer map")
    families = {}
    with gzip.open(calibration["selection"]["source_path"], "rt", encoding="utf-8") as stream:
        selected = {row["id"] for row in rows}
        for line in stream:
            item = json.loads(line)
            if item["id"] in selected:
                r.require(item["id"] not in families and item["valid"] is True and item["source_split"] == "train",
                          "family metadata identity differs")
                families[item["id"]] = item["family"]
    r.require(set(families) == {row["id"] for row in rows}, "family metadata absent")
    for row in rows:
        row["family"] = families[row["id"]]
    models = assets_binding(assets)
    token_files = {name: value for name, value in models["student"]["files"].items() if name != "model.safetensors"}
    r.require(token_files == student["tokenizer_assets"], "qualified tokenizer bytes differ from source")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        r.require(models["donor"]["files"][name] == token_files[name], "donor tokenizer differs")
    tokenizer = {"directory": models["student"]["directory"], "files": token_files}
    runtimes = {"student_base": runtime_binding(student, "student_base"), "donor": runtime_binding(donor, "donor")}
    coordinates = [{"id": row["id"], "input_ids_sha256": row["input_ids_sha256"],
                    "position": len(row["input_ids"]) - 1, "token_id": row["input_ids"][-1]} for row in rows]
    amendment = root / PACKAGE / "STAGE_C_MECHANISM_AMENDMENT.md"
    final = resource_snapshot(nvidia_smi)
    resource_gate(final, lease_path)
    now = utc_now()
    ensure_fresh(output, session_id, now)
    r.verify_source(root, source_commit, clean=True)
    r.require(source_inventory(root, source_commit) == inventory, "source changed during preparation")
    r.verify_file(student_path, student_sha256)
    r.verify_file(donor_path, donor_sha256)
    require_no_framework_import()
    spec = {"schema": 1, "experiment": "CSC-001", "stage": "C", "session_id": session_id,
        "authorization_evidence": AUTHORIZATION, "native_capture_authorized": True,
        "registered_utc": now.isoformat(), "deadline_utc": (now + dt.timedelta(minutes=minutes)).isoformat(),
        "max_total_forwards": 96, "max_role_forwards": 32, "max_optimizer_steps": 0, "max_generated_tokens": 0,
        "role_scope": ROLES, "source_commit": source_commit, "source_files": inventory["files"],
        "implementation_digest": inventory["sha256"], "lease_path": str(lease_path),
        "attempt_root": str(output / "attempts"), "reservation_dir": str(output / "reservations"),
        "assets": models, "tokenizer": tokenizer, "runtime": runtimes,
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "stage_c_mechanism_amendment": {"path": amendment.relative_to(root).as_posix(),
                                         "sha256": r.file_digest(amendment)},
        "tokenizer_qualification_sha256": {"student": student_sha256, "donor": donor_sha256},
        "tokenizer_qualification_paths": {"student": str(student_path.resolve()), "donor": str(donor_path.resolve())},
        "resource_gates": {"registration_host_gib": 12, "registration_free_gpu_gib": 6,
                           "student_launcher_host_gib": 7, "student_child_host_gib": 6,
                           "donor_host_gib": 12},
        "calibration": {"rows": rows, "rows_sha256": r.digest(rows), "source_binding_rows": calibration["rows"],
            "calibration_sha256": calibration["calibration_sha256"], "dataset_sha256": calibration["dataset_sha256"],
            "source_binding_file_sha256": r.file_digest(files["calibration"]),
            "architecture_file_sha256": r.file_digest(files["architecture"]),
            "layer_map_sha256": r.digest(mapping), "tokenizer_sha256": r.digest(token_files),
            "prompt_positions_sha256": r.digest(coordinates)}}
    # First mutation: only after all resource/source/runtime/input checks passed.
    output.mkdir(exist_ok=False)
    (output / "reservations").mkdir()
    (output / "attempts").mkdir()
    spec_sha = r.write_once(output / "SESSION_SPEC.json", spec)
    r.write_once(output / "REGISTRATION_RECEIPT.json", {
        "schema": 1, "status": "REGISTERED_NOT_EXECUTED", "spec_sha256": spec_sha,
        "source_commit": source_commit, "expires_utc": spec["deadline_utc"],
        "tokenizer_parity": True, "resources_before": first, "resources_at_registration": final,
        "tokenizer_qualifications": spec["tokenizer_qualification_sha256"],
        "native_model_calls_during_preparation": 0, "native_tokenizer_encodings_during_preparation": 0,
        "optimizer_steps": 0, "automatic_retry": False, "partial_student_scope_supported": False,
        "analysis_protocol": ANALYSIS_PROTOCOL, "stage_c_mechanism_amendment": spec["stage_c_mechanism_amendment"]})
    return {"path": str(output / "SESSION_SPEC.json"), "sha256": spec_sha, "session_id": session_id}


def require_no_framework_import():
    r.require(not {"torch", "transformers", "peft", "bitsandbytes"} & set(sys.modules),
              "registration helper must remain framework-free")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-commit", "student-sha256", "donor-sha256", "session-id"):
        parser.add_argument("--" + name, required=True)
    for name in ("student", "donor", "output", "nvidia-smi"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--minutes", type=int, default=60)
    args = parser.parse_args()
    result = prepare(root=ROOT, source_commit=args.source_commit, student_path=args.student,
        student_sha256=args.student_sha256, donor_path=args.donor, donor_sha256=args.donor_sha256,
        output=args.output, session_id=args.session_id, nvidia_smi=args.nvidia_smi, minutes=args.minutes)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
