import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from experiments.rineya_triadic_working_memory_v1.data import generate_rows, write_jsonl, load_rows, FAMILIES
from experiments.rineya_triadic_working_memory_v1.provenance import (
    create_checkpoint_lock, verify_checkpoint_lock, canonical_hash, check_device)
from experiments.rineya_triadic_working_memory_v1.analysis import paired_report
from experiments.rineya_triadic_working_memory_v1.runner import read_job, answer_loss, smoke


@pytest.mark.parametrize("split", ["train", "dev", "holdout"])
def test_fresh_data_determinism_coverage_and_oracle(split):
    rows = generate_rows(split, 30)
    assert rows == generate_rows(split, 30)
    assert {r["family"] for r in rows} == set(FAMILIES)
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["row_sha256"] for r in rows}) == len(rows)
    for row in rows:
        assert row["template_partition"] == row["entity_partition"] == split
        assert "Return only the answer" in row["oracle_prompt"]
        assert "noise_" not in row["oracle_prompt"]
        if row["family"] in ("two_hop", "three_hop"):
            assert len(row["intermediate_answers"]) == 2
        if row["family"] in ("missing", "role_state"):
            assert row["answer"] == "UNKNOWN"


def test_disjoint_templates_entities_ids():
    a, b = generate_rows("train", 20), generate_rows("dev", 20)
    assert not {r["id"] for r in a} & {r["id"] for r in b}
    assert all("person_t" not in r["prompt"] for r in b)
    assert all("Record:" not in r["prompt"] for r in a)


def test_training_holdout_and_duplicate_guards(tmp_path):
    p = tmp_path / "rows.jsonl"
    write_jsonl(p, generate_rows("holdout", 4))
    with pytest.raises(ValueError):
        load_rows(p, "train")
    with pytest.raises(ValueError):
        load_rows(p, "eval")
    assert len(load_rows(p, "eval", allow_holdout=True)) == 4
    with pytest.raises(FileExistsError):
        write_jsonl(p, [])
    p = tmp_path / "duplicates.jsonl"
    row = generate_rows("dev", 1)[0]
    write_jsonl(p, [row, row])
    with pytest.raises(ValueError):
        load_rows(p, "eval")
    with pytest.raises(ValueError):
        load_rows(tmp_path / "sealed_TEST2.jsonl", "eval")


def fake_checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    (root / "model.safetensors").write_bytes(b"synthetic hashing fixture, NOT a checkpoint")
    return root


def test_checkpoint_lock_detects_weight_and_tokenizer_changes(tmp_path):
    root = fake_checkpoint(tmp_path)
    lock = create_checkpoint_lock(root, "test-fixture")
    assert verify_checkpoint_lock(root, lock) == root.resolve()
    (root / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(ValueError):
        verify_checkpoint_lock(root, lock)
    lock = create_checkpoint_lock(root, "test-fixture")
    (root / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_checkpoint_lock(root, lock)


def test_checkpoint_lfs_and_traversal_rejected(tmp_path):
    root = fake_checkpoint(tmp_path)
    (root / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError):
        create_checkpoint_lock(root, "lfs-not-loaded")
    (root / "model.safetensors").unlink()
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../escape.safetensors"}}))
    with pytest.raises(ValueError):
        create_checkpoint_lock(root, "escaped")


def prediction_rows(scores):
    return [dict(id=str(i), group_id=str(i//2), correct=score, family="binding", answer="x",
                 split="dev", comparison_id="same-checkpoint-data-and-decoder", seed=17)
            for i, score in enumerate(scores)]


def test_paired_cluster_report_and_repair_damage_accounting():
    a, b = prediction_rows([0, 1, 0, 1, 0, 1]), prediction_rows([1, 1, 1, 0, 0, 1])
    report = paired_report(a, b, bootstrap=200)
    assert report["repairs"] == 2 and report["damages"] == 1
    assert report["accuracy_delta"] == pytest.approx(1/6)
    assert report["n_groups"] == 3
    assert report["promotion"].startswith("NOT_EVALUATED")
    assert paired_report(a, list(reversed(b)), 200) == report


@pytest.mark.parametrize("field", ["comparison_id", "seed", "answer", "split", "family", "group_id"])
def test_unpaired_metadata_rejected(field):
    a, b = prediction_rows([0, 1]), prediction_rows([1, 1])
    b[0][field] = "mismatch"
    with pytest.raises(ValueError):
        paired_report(a, b, 100)


def test_answer_only_loss_masks_prompt_tokens():
    logits = torch.randn(1, 5, 8, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, 3, 4]])
    loss = answer_loss(logits, labels)
    loss.backward()
    assert logits.grad[:, :2].abs().sum() == 0
    assert logits.grad[:, 2:4].abs().sum() > 0
    with pytest.raises(ValueError):
        answer_loss(logits, torch.full_like(labels, -100))


def test_job_and_device_guards(tmp_path):
    job_path = Path(__file__).parents[1] / "configs" / "pilot.json"
    job = read_job(job_path)
    assert job["seeds"] == [17, 43, 91]
    job["memory"]["slices"] = 8
    altered = tmp_path / "job.json"
    altered.write_text(json.dumps(job))
    with pytest.raises(ValueError):
        read_job(altered)
    with pytest.raises(ValueError):
        check_device("meta")


def test_native_command_requires_explicit_execution_flag(tmp_path):
    out = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, "-m", "experiments.rineya_triadic_working_memory_v1.runner",
        "qualify", "--checkpoint", "missing", "--lock", "missing", "--out", str(out)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "--execute-native" in result.stderr
    assert not out.exists()


def test_smoke_runs_all_arms_without_native_claim(tmp_path):
    result = smoke(tmp_path / "smoke.json")
    assert result["real_checkpoint_loaded"] is False
    assert result["efficacy_claim"] is False
    assert len(result["arms"]) == 5
    assert all(r["last_preupdate_loss"] >= 0 for r in result["arms"])


def test_retrieval_probes_share_group_not_answer_prompt():
    from experiments.rineya_triadic_working_memory_v1.data import with_retrieval_probes
    rows = generate_rows("dev", 10)
    expanded = with_retrieval_probes(rows)
    assert len(expanded) == 14
    edges = [r for r in expanded if r["family"] == "hop_edge_retrieval"]
    assert len(edges) == 4
    assert all(r["id"] != r["group_id"] and r["group_id"] in {x["id"] for x in rows} for r in edges)
    assert all("Follow container" not in r["prompt"] for r in edges)


def test_qualification_binding_and_version_drift(tmp_path):
    from experiments.rineya_triadic_working_memory_v1.runner import verify_qualification
    from experiments.rineya_triadic_working_memory_v1.provenance import source_hash, environment
    job = read_job(Path(__file__).parents[1] / "configs" / "pilot.json")
    lock = {"lock_sha256": "synthetic-not-a-real-checkpoint"}
    receipt = dict(status="NATIVE_MECHANICAL_PASS", checkpoint_lock_sha256=lock["lock_sha256"],
                   source_sha256=source_hash(), job_sha256=canonical_hash(job), device="cpu", environment=environment(),
                   results={k: True for k in ("zero_gate_exact_identity", "active_cache_equivalence",
                                             "greedy_token_equality", "disabled_after_activation_exact_identity")})
    p = tmp_path / "synthetic-qualification.json"
    p.write_text(json.dumps(receipt))
    verify_qualification(p, job, lock, "cpu")
    with pytest.raises(ValueError):
        verify_qualification(p, job, lock, "cuda:0")
    receipt["results"]["active_cache_equivalence"] = False
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_qualification(p, job, lock, "cpu")


def test_family_metric_filtering():
    a, b = prediction_rows([0, 1, 0, 1]), prediction_rows([1, 1, 0, 1])
    for rows in (a, b):
        for row in rows[2:]:
            row["family"] = "two_hop"
    result = paired_report(a, b, 100, families=["binding"])
    assert result["n_examples"] == 2 and result["accuracy_delta"] == 0.5
    with pytest.raises(ValueError):
        paired_report(a, b, 100, families=["nonexistent"])
