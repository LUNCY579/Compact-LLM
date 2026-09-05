"""Immutable resume contracts using tiny local artifacts and mocked identities."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from shared.hashing import canonical_sha256, sha256_file, write_json
from cross_scale.scripts import run_alignment as caller


@pytest.fixture
def protocol():
    return caller.configure(Path(caller.ROOT)/"configs/alignment_v2.json")


def start_receipt(config, registration, arm="pair"):
    return {"status": "STARTED", "track": "cross_scale", "arm": arm, "pid": 12345678,
            "config_sha256": canonical_sha256(config), "registration_sha256": canonical_sha256(registration),
            "source_sha256": canonical_sha256(registration["sources"]), "sources": registration["sources"]}


def create_attempt(root, local, config, registration, attempt, status="COMPLETE"):
    directory = root/"evidence/02_bridge_objectives/resume/S8_D12"/f"attempt_{attempt:03d}"
    directory.mkdir(parents=True)
    started = start_receipt(config, registration)
    write_json(directory/"started.json", started)
    if status == "STARTED": return directory
    final = {**started, "status": status}
    if status == "FAILED":
        write_json(directory/"failure.json", {"error": "intentional toy failure"})
    else:
        report = {"supa_layer": 8, "donor_layer": 12}
        bridge = local/f"bridge_{attempt}.pt"
        bridge.parent.mkdir(parents=True, exist_ok=True)
        caller.exclusive_tensor(bridge, {"tiny": torch.ones(1)})
        write_json(directory/"metrics.json", report)
        final.update(capture_sha256="capture", metrics_sha256=sha256_file(directory/"metrics.json"),
                     bridge_path=str(bridge), bridge_sha256=sha256_file(bridge))
    write_json(directory/"run_manifest.json", final)
    return directory


def test_failed_attempt_is_preserved_and_new_success_can_aggregate(tmp_path, monkeypatch, protocol):
    root, local = tmp_path/"cross_scale", tmp_path/"cross_scale/local/resume"
    monkeypatch.setattr(caller, "ROOT", root)
    protocol["supa_layers"], protocol["donor_layers"] = [8], [12]
    registration = {"sources": {"source.py": "frozen"}}
    failed = create_attempt(root, local, protocol, registration, 0, "FAILED")
    before = {p.name: p.read_bytes() for p in failed.iterdir()}
    create_attempt(root, local, protocol, registration, 1)
    records, sources = caller.collect_completed_pairs("resume", protocol, registration, "capture", local)
    assert records == [{"supa_layer": 8, "donor_layer": 12}]
    assert [s["status"] for s in sources] == ["FAILED", "COMPLETE"]
    assert before == {p.name: p.read_bytes() for p in failed.iterdir()}


@pytest.mark.parametrize("damage", ["unfinished", "duplicate", "bridge", "metrics", "source", "unknown_status"])
def test_aggregation_rejects_unsafe_resume(tmp_path, monkeypatch, protocol, damage):
    root, local = tmp_path/"cross_scale", tmp_path/"cross_scale/local/resume"
    monkeypatch.setattr(caller, "ROOT", root)
    protocol["supa_layers"], protocol["donor_layers"] = [8], [12]
    registration = {"sources": {"source.py": "frozen"}}
    directory = create_attempt(root, local, protocol, registration, 0)
    if damage == "unfinished": create_attempt(root, local, protocol, registration, 1, "STARTED")
    elif damage == "duplicate": create_attempt(root, local, protocol, registration, 1)
    elif damage == "bridge": (local/"bridge_0.pt").write_bytes(b"changed")
    elif damage == "metrics": (directory/"metrics.json").write_text("{}")
    else:
        result = json.loads((directory/"run_manifest.json").read_text())
        result["source_sha256" if damage == "source" else "status"] = "different"
        (directory/"run_manifest.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        caller.collect_completed_pairs("resume", protocol, registration, "capture", local)


def test_interrupted_attempt_requires_explicit_verified_abandonment(tmp_path, monkeypatch, protocol):
    import psutil
    root, local = tmp_path/"cross_scale", tmp_path/"cross_scale/local/resume"
    monkeypatch.setattr(caller, "ROOT", root)
    protocol["supa_layers"], protocol["donor_layers"] = [8], [12]
    registration = {"sources": {"source.py": "frozen"}}
    directory = create_attempt(root, local, protocol, registration, 0, "STARTED")
    before = (directory/"started.json").read_bytes()
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    with pytest.raises(RuntimeError, match="live"):
        caller.abandon_pair_attempt("resume", protocol, registration, 8, 12, 0)
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: False)
    caller.abandon_pair_attempt("resume", protocol, registration, 8, 12, 0)
    assert before == (directory/"started.json").read_bytes()
    assert not (directory/"run_manifest.json").exists()
    create_attempt(root, local, protocol, registration, 1)
    _, sources = caller.collect_completed_pairs("resume", protocol, registration, "capture", local)
    assert [s["status"] for s in sources] == ["ABANDONED", "COMPLETE"]
    with pytest.raises(ValueError, match="terminal"):
        caller.abandon_pair_attempt("resume", protocol, registration, 8, 12, 0)


def test_capture_cache_binds_source_data_and_registered_identity(tmp_path, monkeypatch, protocol):
    programme = tmp_path/"repo/experiments/programme"
    root = programme/"cross_scale"
    monkeypatch.setattr(caller, "ROOT", root)
    monkeypatch.setattr(caller, "PROGRAMME", programme)
    evidence, local = caller.run_directories("cache")
    local.mkdir(parents=True)
    dataset = programme/"data/manifest.json"
    write_json(dataset, {"toy": "manifest"})
    sources = {"source.py": "source hash"}
    registration = {"sources": sources, "dataset_manifests": {dataset.relative_to(programme.parents[1]).as_posix(): sha256_file(dataset)}}
    expected = [{"example_id": f"e{i}", "scenario_id": f"s{i}", "family": "toy", "prompt": f"toy {i}",
                 "track": "cross_scale", "split": "calibration", "alignment_partition": "fit" if i < 2 else "holdout"} for i in range(4)]
    monkeypatch.setattr(caller, "calibration_rows", lambda path: expected)
    monkeypatch.setattr(caller, "source_files", lambda track: sources)
    bundle = {"track": "cross_scale", "role": "calibration", "sources": sources,
              "source_sha256": canonical_sha256(sources), "config_sha256": canonical_sha256(protocol),
              "canonical_fixture_bit_identical": True, "supa_identity": {}, "donor_identity": {},
              "rows": [{k: row[k] for k in ("example_id", "scenario_id", "family", "prompt")} for row in expected],
              "input_ids": [torch.tensor([[1, 2]]) for _ in expected], "special_ids": [],
              "train_prompt_mask": [True, True, False, False], "heldout_scenarios": ["s2", "s3"],
              "small": {i: [torch.ones(2, 3) for _ in expected] for i in protocol["supa_layers"]},
              "donor": {i: [torch.ones(2, 4) for _ in expected] for j in protocol["donor_layers"] for i in (j-1, j)}}
    caller.exclusive_tensor(local/"activations.pt", bundle)
    start = start_receipt(protocol, registration, "capture")
    final = {**start, "status": "COMPLETE", "activation_sha256": sha256_file(local/"activations.pt"),
             "dataset_manifest": str(dataset), "dataset_manifest_sha256": sha256_file(dataset),
             "canonical_fixture_bit_identical": True, "supa_identity": {}, "donor_identity": {},
             "fit_prompts": 2, "holdout_prompts": 2, "heldout_scenarios": ["s2", "s3"]}
    write_json(evidence/"started.json", start)
    write_json(evidence/"run_manifest.json", final)
    write_json(evidence/"capture.json", final)
    _, loaded = caller.read_capture("cache", protocol, registration)
    assert len(loaded["rows"]) == 4
    with monkeypatch.context() as context:
        def forbidden_load(*args, **kwargs):
            raise AssertionError("aggregation must not load dense activations")
        context.setattr(torch, "load", forbidden_load)
        _, loaded = caller.read_capture("cache", protocol, registration, metadata_only=True)
        assert loaded is None
    monkeypatch.setattr(caller, "source_files", lambda track: {"source.py": "changed"})
    with pytest.raises(ValueError, match="source"):
        caller.read_capture("cache", protocol, registration)
    with pytest.raises(FileExistsError): caller.exclusive_tensor(local/"activations.pt", {})


@pytest.mark.parametrize("mutation", ["axis", "control", "deadline", "bounds", "gate", "generation"])
def test_cli_rejects_unregistered_or_nonfinite_contract(tmp_path, protocol, mutation):
    if mutation == "axis": protocol["donor_layers"].pop()
    elif mutation == "control": protocol["objectives"].remove("BS")
    elif mutation == "deadline": protocol["max_pair_seconds"] = float("nan")
    elif mutation == "bounds": protocol["hidden_rms_ratio_bounds"] = [10, .05]
    elif mutation == "gate": protocol["mechanism_gate"]["minimum_delta_cosine"] = float("nan")
    else: protocol["generation_enabled"] = True
    path = tmp_path/"config.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError): caller.configure(path)
