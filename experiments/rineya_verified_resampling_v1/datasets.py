"""Prepare public-only inputs and a separately stored, verified oracle catalog."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

from .contracts import (BRIDGE, BRIDGE_BLOB, ROOT, VERSION, matches_git_blob, digest, encoded,
                        file_sha, integer, parse_answer, read_json, read_jsonl, require,
                        validate_config, validate_public, write_jsonl, write_new)

OPEN_ROLES = {"DEV", "TRAIN_SELECTION"}


def save_dataset(out: Path, public: list[dict], private: list[dict], *, kind: str,
                 role: str, provenance: dict) -> dict:
    require(role in OPEN_ROLES, "CHECK/FINAL/TRAIN are not RVS evaluation roles")
    validate_public(public)
    require(len(private) == len(public), "public/private size mismatch")
    require([r["id"] for r in public] == [r["id"] for r in private], "public/private order mismatch")
    parents = set()
    for r in private:
        require(r["role"] == role and parse_answer(r["expected"]) == r["expected"], "invalid oracle target/role")
        parents.add(r["parent"])
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "public.jsonl", public)
    write_jsonl(out / "oracle.private.jsonl", private)
    manifest = {"schema": VERSION, "kind": kind, "role": role, "rows": len(public),
                "parents": len(parents), "public_sha256": file_sha(out / "public.jsonl"),
                "oracle_sha256": file_sha(out / "oracle.private.jsonl"),
                "family_counts": dict(Counter(r["family"] for r in private)),
                "provenance": provenance, "CHECK": "CLOSED", "FINAL": "CLOSED"}
    write_new(out / "manifest.json", manifest)
    return manifest


def prepare_centralized(out: Path, per_family: int = 32, role: str = "DEV",
                        selection_seed: int = 20260927) -> dict:
    """Call the existing generator, allocator, renderer, and independent verifier.

    Only open roles receive nonzero quotas. No CHECK/FINAL payload is rendered.
    New files preserve original parent and role identity; no E1 file is modified.
    """
    require(role in OPEN_ROLES, "only DEV/TRAIN_SELECTION preparation is allowed")
    integer(per_family, 1, 64, "per-family count")
    integer(selection_seed, 0, 2**63-1, "selection seed")
    require(matches_git_blob(ROOT / BRIDGE, BRIDGE_BLOB), "centralized generator source drift")
    from experiments.post_training_mimo_centralized_v1.scripts import bridge_data as bridge
    started = time.monotonic()
    counts = {key: 4 * per_family if key == role else 0 for key in bridge.ROLES}
    parents, allocation = bridge.prepare_parents(counts, selection_seed)
    public, private = [], []
    for parent in parents:
        require(parent.role == role, "allocator returned forbidden role")
        # Immutable original generator lineage, not a claim of a new model run.
        row = bridge.make_row(parent, "ba683ac0fcadbf04e4e179200d035b6a980e388f", 0)
        require(bridge.verify_answer(parent, row["target"]), "independent reference verification failed")
        messages = [{"role": "user", "content": row["prompt"]}]
        opaque = digest(["rvs-public-v1", messages])
        public.append({"id": opaque, "messages": messages})
        private.append({"id": opaque, "parent": parent.parent_id, "family": row["family"],
                        "role": role, "expected": parse_answer(row["target"]),
                        "messages_digest": digest(messages), "row_digest": digest(row)})
    return save_dataset(out, public, private, kind="centralized_bridge", role=role,
                        provenance={"generator_git_blob": BRIDGE_BLOB, "selection_seed": selection_seed,
                                    "allocation": allocation, "seconds": time.monotonic()-started,
                                    "reference_verification": "existing_independent_bridge_verifier",
                                    "unknown_pretraining_exclusion": "NOT_ESTABLISHED"})


def prepare_fixture(out: Path, count: int = 16) -> dict:
    """Hand-authored mechanics population, never a native Titan result."""
    integer(count, 1, 128, "fixture count")
    public, private = [], []
    for i in range(count):
        messages = [{"role": "user", "content": f'Fixture {i}: return {{"answer":{i+7}}}.'}]
        opaque = digest(["rvs-public-v1", messages])
        public.append({"id": opaque, "messages": messages})
        private.append({"id": opaque, "parent": f"fixture-parent-{i}", "family": "mechanical_fixture",
                        "role": "DEV", "expected": encoded({"answer": i+7}).decode(),
                        "messages_digest": digest(messages), "row_digest": digest(["fixture", i])})
    return save_dataset(out, public, private, kind="synthetic_fixture", role="DEV",
                        provenance={"warning": "scripted mechanics only; prompt states target"})


def freeze_study(data: Path, config_path: Path, out: Path) -> dict:
    manifest = read_json(data / "manifest.json")
    require(manifest["schema"] == VERSION and manifest["role"] in OPEN_ROLES,
            "invalid dataset schema/role")
    require(manifest["CHECK"] == manifest["FINAL"] == "CLOSED", "protected roles must remain closed")
    require(file_sha(data / "public.jsonl") == manifest["public_sha256"], "public data drift")
    public = read_jsonl(data / "public.jsonl")
    validate_public(public)
    require(manifest["rows"] == len(public), "row count drift")
    config = validate_config(read_json(config_path))
    slots = len(public) * (config["samples"] + 1)
    require(slots * config["max_new_tokens"] <= config["max_generated_tokens"],
            "token budget must cover every planned slot including failures")
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "public.jsonl", public)
    # Deliberately no oracle file or oracle data path in the sampling study.
    study = {"schema": VERSION, "config": config, "dataset": manifest,
             "public_sha256": file_sha(out / "public.jsonl"), "slots": slots,
             "sampling_policy": "independent_temperature_softmax_no_topk_no_topp",
             "vote_policy": "strict_JSON_answer_plurality_earliest_tie_ignore_non_EOS",
             "status": "FROZEN_EXPLORATORY_INPUTS_NOT_NATIVE_QUALIFICATION"}
    study["study_id"] = digest(study)
    write_new(out / "study.json", study)
    return study


def load_study(path: Path) -> tuple[dict, list[dict]]:
    study = read_json(path / "study.json")
    identity = dict(study)
    claimed = identity.pop("study_id")
    require(digest(identity) == claimed and study["schema"] == VERSION, "study drift")
    validate_config(study["config"])
    require(study["dataset"]["role"] in OPEN_ROLES, "forbidden role")
    require(file_sha(path / "public.jsonl") == study["public_sha256"], "public inputs changed after freeze")
    rows = read_jsonl(path / "public.jsonl")
    validate_public(rows)
    require(study["slots"] == len(rows) * (study["config"]["samples"] + 1), "slot inventory drift")
    return study, rows
