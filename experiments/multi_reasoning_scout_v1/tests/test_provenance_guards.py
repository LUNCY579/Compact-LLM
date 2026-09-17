from pathlib import Path
import pytest
from experiments.multi_reasoning_scout_v1 import core, runner, budget


def test_peak_allocation_cannot_be_hidden_by_freeing_memory():
    observed = dict(host_available_bytes=10*budget.GIB, gpu_free_bytes=10*budget.GIB,
                    disk_free_bytes=10*budget.GIB, host_rss_bytes=budget.GIB,
                    gpu_allocated_bytes=budget.GIB, gpu_peak_bytes=7*budget.GIB)
    with pytest.raises(RuntimeError, match="gpu_peak_bytes"):
        budget.check(observed)


def test_prelaunch_source_rejects_non_descendant(monkeypatch, tmp_path):
    frozen = {"repository":"fixture", "branch":"fixture", "sha":"old", "files":{}}
    freeze = {"files":{}, "source_files":{}, "source":frozen}
    monkeypatch.setattr(core, "read_json", lambda path: freeze)
    monkeypatch.setattr(core, "source_files", lambda: {})
    monkeypatch.setattr(core, "source_identity", lambda: {**frozen,"sha":"current"})
    monkeypatch.setattr(core, "git", lambda *args: "unrelated")
    with pytest.raises(ValueError, match="source lineage"):
        core.load_frozen()
    monkeypatch.setattr(core, "git", lambda *args: "old")
    assert core.load_frozen() == freeze


def registration(tmp_path, monkeypatch):
    old = {"id":"d1-seed17-bf16-export", "path":str(tmp_path/"old"),
           "assets":{"model.safetensors":"a"*64, "config.json":"b"*64},
           "policy":"same", "runtime_dtype":"float32", "storage_dtype":["BF16"],
           "attention_backend":"eager", "decoding":{}, "source_repository":"repo", "source_branch":"old", "source_sha":"old"}
    new = {**old, "id":"future", "path":str(tmp_path/"new"), "source_branch":"new", "source_sha":"new",
           "assets":{**old["assets"],"model.safetensors":"c"*64}}
    provenance = {k:new[k] for k in ("source_repository","source_branch","source_sha")}
    provenance.update(source_checkpoint_sha256="a"*64, dataset_manifest_sha256="d"*64)
    for name, stage in (("dataset_acceptance","DATASET_ACCEPTANCE"),("training_contract","TRAINING_CONTRACT"),("training_execution","TRAINING_EXECUTION")):
        path = tmp_path/(name+".json")
        core.write_new(path, {"status":"PASS", "stage":stage, "dataset_manifest_sha256":"d"*64,
                              "source_checkpoint_sha256":"a"*64, "checkpoint_sha256":"c"*64})
        provenance[name]={"status":"PASS", "path":str(path), "sha256":core.file_hash(path)}
    reg={"frozen_protocol_sha256":"f"*64, "model":new, "training_provenance":provenance}
    original_read, original_hash = core.read_json, core.file_hash
    monkeypatch.setattr(runner,"read_json",lambda p: reg if str(p)=="future" else {"selected":[old]} if Path(p).name=="MODEL_MANIFEST.json" else original_read(p))
    monkeypatch.setattr(runner,"file_hash",lambda p: "f"*64 if Path(p).name=="FREEZE.json" else original_hash(p))
    return reg


def test_future_receipts_bound_to_real_bytes_and_checkpoint(tmp_path,monkeypatch):
    reg=registration(tmp_path,monkeypatch)
    assert runner.candidate("future","future")==reg["model"]
    receipt=reg["training_provenance"]["training_execution"]
    path=Path(receipt["path"])
    path.write_text('{"status":"PASS"}')
    with pytest.raises(ValueError,match="changed future training receipt"):
        runner.candidate("future","future")


def test_future_model_source_cannot_disagree_with_training(tmp_path,monkeypatch):
    reg=registration(tmp_path,monkeypatch)
    reg["model"]["source_sha"]="inconsistent"
    with pytest.raises(ValueError,match="source identities disagree"):
        runner.candidate("future","future")
