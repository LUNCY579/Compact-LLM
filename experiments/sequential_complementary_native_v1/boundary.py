"""Review-locked native boundary. Stdlib only; no model or approval side effects.

A request is not a grant. This module deliberately has no enable flag, approved
sample, lock-removal command, permit consumer, or native PASS producer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, NoReturn

PARENT = "b4f010619290a562c54cdaf3bb6edd4eb9112a91"
BRANCH = "research/sequential-complementary-native-boundary-v1"
SCHEMA = "SCP_NQ_FIRST_HOST_REQUEST_V1"
LOCK_REASON = "NATIVE_REVIEW_LOCKED: exact-host review and a separately approved one-launch grant are still required"
WEIGHTS = {
    "source": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
    "trained_partial": "07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d",
    "subject": "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284",
}
CLOSED = (
    "native_permission", "training_permission", "backward_permission",
    "optimizer_permission", "teacher_comparisons_permission", "protected_reads_permission",
    "scoring_permission", "confirmation_permission", "quality_permission", "release_permission",
)
# This is narrower than the inherited two-launch student/MR envelope. It is not
# automatically registered with, or accepted by, the inherited RRP Journal.
LIMITS = {
    "model_worker_launches": 1, "retries": 0,
    "worker_timeout_seconds": 600,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 620,
    "total_owned_work_wall_seconds": 900,
    "evaluation_and_qualification_model_calls": 64,
    "generation_requests": 0, "generated_tokens_all_candidates": 0,
    "processed_training_tokens_all_attempts": 0, "on_policy_rl_rollout_tokens": 0,
    "adaptive_model_selection_rounds": 0, "cloud_or_teacher_spend": 0,
    "new_disk_bytes": 64 * 2**20, "max_host_rss_bytes": 8 * 2**30,
    "min_free_disk_bytes": 14 * 2**30, "max_sequence_tokens": 2048,
}
HOST_TEXT = (
    "host_id", "os", "machine", "python_executable", "python_version", "gpu_uuid",
    "gpu_name", "driver_version", "cuda_runtime",
)
RUNTIME = {"torch": "2.10.0", "transformers": "4.57.1", "safetensors": "0.7.0"}


class NativeReviewLocked(RuntimeError):
    """Not a grant refusal that can be repaired by supplying another old permit."""


def require_native_review() -> NoReturn:
    """Unconditional publication lock; review requires a new committed change."""
    raise NativeReviewLocked(LOCK_REASON)


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def hex_digest(value: Any, length: int = 64) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def generation_eos_ids(value: Any, *, vocab_size: int, tokenizer_eos: Any) -> tuple[int, ...]:
    """Validate the complete bound generation-config EOS set; NEVER fall back.

    Normalization preserves every configured terminal ID. bool is not an integer
    token ID here, and duplicate/empty/unknown values are configuration failures.
    """
    require(type(vocab_size) is int and vocab_size > 0, "invalid vocabulary size")
    ids = [value] if type(value) is int else value
    require(type(ids) is list and len(ids) > 0, "generation config must declare EOS IDs")
    require(all(type(x) is int and 0 <= x < vocab_size for x in ids), "invalid generation EOS ID")
    require(len(ids) == len(set(ids)), "duplicate generation EOS ID")
    require(type(tokenizer_eos) is int and tokenizer_eos in ids,
            "tokenizer EOS is absent from the bound generation config")
    return tuple(sorted(ids))


def terminal_token(token: int, eos_ids: tuple[int, ...]) -> bool:
    require(type(token) is int and token >= 0, "invalid emitted token")
    require(type(eos_ids) is tuple and len(eos_ids) > 0
            and all(type(x) is int and x >= 0 for x in eos_ids), "unvalidated EOS set")
    return token in eos_ids


def host_template() -> dict:
    return {**{k: None for k in HOST_TEXT}, "python_executable_sha256": None,
            "gpu_total_bytes": None, "runtime": copy.deepcopy(RUNTIME),
            "device": "cuda:0", "compute_dtype": "float32", "attention": "eager"}


def validate_host(host: dict) -> None:
    require(type(host) is dict and set(host) == set(host_template()), "exact host fields required")
    for key in HOST_TEXT:
        require(type(host[key]) is str and 0 < len(host[key].strip()) <= 1024,
                f"missing host identity: {key}")
    require(host["gpu_uuid"].startswith("GPU-") and len(host["gpu_uuid"]) > 4,
            "physical GPU UUID required, not just cuda:0")
    require(hex_digest(host["python_executable_sha256"]), "Python executable hash required")
    require(type(host["gpu_total_bytes"]) is int and host["gpu_total_bytes"] > 0,
            "GPU capacity must be a positive integer")
    require(host["runtime"] == RUNTIME, "runtime pins changed")
    require(host["device"] == "cuda:0" and host["compute_dtype"] == "float32"
            and host["attention"] == "eager", "unqualified device/precision/backend")


def checkout_snapshot(root: Path) -> dict:
    """Inspect a real clean dedicated checkout without importing the model stack."""
    root = root.resolve(strict=True)
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                       stderr=subprocess.STDOUT, timeout=20).strip()
    require(Path(git("rev-parse", "--show-toplevel")).resolve() == root, "full checkout required")
    require(git("branch", "--show-current") == BRANCH, "wrong successor branch")
    require(not git("status", "--porcelain", "--untracked-files=all"), "dirty checkout refused")
    flags = git("ls-files", "-v").splitlines()
    require(bool(flags) and all(line.startswith("H ") for line in flags), "hidden index entries refused")
    subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", PARENT, "HEAD"],
                   check=True, capture_output=True, timeout=20)
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "branch": BRANCH, "parent": PARENT}


def request_for(host: dict, checkout: dict) -> dict:
    validate_host(host)
    require(type(checkout) is dict and set(checkout) == {"commit", "tree", "branch", "parent"},
            "exact checkout identity required")
    require(hex_digest(checkout["commit"], 40) and hex_digest(checkout["tree"], 40)
            and checkout["branch"] == BRANCH and checkout["parent"] == PARENT, "foreign checkout")
    value = {"schema": SCHEMA, "experiment": "SCP-NQ-001", "phase": "student",
             "status": "PENDING_REVIEW_NOT_A_GRANT", "approval_reference": None,
             "checkout": copy.deepcopy(checkout), "host": copy.deepcopy(host),
             "host_sha256": digest(host), "weight_sha256": copy.deepcopy(WEIGHTS),
             "limits": copy.deepcopy(LIMITS), "permissions": {key: False for key in CLOSED},
             "assets_verified": False, "host_observation_independently_verified": False,
             "native_review_lock": True,
             "purpose": "One inference-only loader/tokenizer/cache qualification; no generation, training, or scored bank"}
    validate_request(value)
    return value


def validate_request(value: dict) -> None:
    require(type(value) is dict and set(value) == {
        "schema", "experiment", "phase", "status", "approval_reference", "checkout", "host",
        "host_sha256", "weight_sha256", "limits", "permissions", "assets_verified",
        "host_observation_independently_verified", "native_review_lock", "purpose"}, "foreign request schema")
    require(value["schema"] == SCHEMA and value["experiment"] == "SCP-NQ-001"
            and value["phase"] == "student" and value["status"] == "PENDING_REVIEW_NOT_A_GRANT",
            "only the pending first student request is supported")
    validate_host(value["host"])
    require(value["host_sha256"] == digest(value["host"]), "host binding changed")
    checkout = value["checkout"]
    require(type(checkout) is dict and set(checkout) == {"commit", "tree", "branch", "parent"}
            and hex_digest(checkout["commit"], 40) and hex_digest(checkout["tree"], 40)
            and checkout["branch"] == BRANCH and checkout["parent"] == PARENT, "source binding changed")
    require(value["weight_sha256"] == WEIGHTS, "subject/source/partial conflation")
    limits = value["limits"]
    require(type(limits) is dict and set(limits) == set(LIMITS)
            and all(type(limits[k]) is int and limits[k] == LIMITS[k] for k in LIMITS),
            "first-launch envelope changed")
    permissions = value["permissions"]
    require(type(permissions) is dict and set(permissions) == set(CLOSED)
            and all(permissions[k] is False for k in CLOSED), "draft cannot grant permission")
    require(value["approval_reference"] is None and value["assets_verified"] is False
            and value["host_observation_independently_verified"] is False
            and value["native_review_lock"] is True, "draft cannot claim execution approval or verification")
    require(value["purpose"] == "One inference-only loader/tokenizer/cache qualification; no generation, training, or scored bank",
            "purpose changed")


def safe_external(path: Path, checkout: Path) -> Path:
    path = path.absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "symlink or traversal path refused")
    root = checkout.resolve(strict=True)
    require(path != root and root not in path.parents, "records must be outside checkout")
    require(path.parent.is_dir(), "record parent must already exist")
    return path


def write_new(path: Path, value: dict, checkout: Path) -> None:
    path = safe_external(path, checkout)
    data = canonical(value) + b"\n"
    require(len(data) <= 2**20, "oversized record")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def read_object(path: Path) -> dict:
    path = path.absolute()
    require(path.is_file() and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "regular JSON file required")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    with path.open("rb") as stream:
        data = stream.read(2**20 + 1)
    require(len(data) <= 2**20, "oversized JSON input")
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject_constant)
    require(type(value) is dict, "JSON object required")
    return value


def status() -> dict:
    return {"schema": "SCP_NATIVE_BOUNDARY_STATUS_V1", "status": "NATIVE_REVIEW_LOCKED",
            "native_launch_available": False, "approval_present": False,
            "native_qualification": "NOT_RUN", "full_checkout_regression": "NOT_RUN",
            "permissions": {key: False for key in CLOSED},
            "next_gate": "Full-checkout review, exact local asset/runtime observations, then separate one-launch approval"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("host-template")
    draft = commands.add_parser("draft")
    draft.add_argument("--host", type=Path, required=True)
    draft.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        print(canonical(status()).decode())
    elif args.command == "host-template":
        print(canonical(host_template()).decode())
    else:
        root = Path(__file__).resolve().parents[2]
        host = read_object(safe_external(args.host, root))
        value = request_for(host, checkout_snapshot(root))
        write_new(args.out, value, root)
        print(canonical({"status": value["status"], "request_sha256": digest(value),
                         "native_launch_available": False}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
