from copy import deepcopy
import random
import torch
import pytest

from experiments.model_quality_gpu_v1.config import Design
from experiments.model_quality_gpu_v1.integrity import read_json
from experiments.model_quality_gpu_v1.tests.fixtures import TinyLM
from experiments.reasoning_posttraining_v1.engine import microfit


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(17); random.seed(17)
    model = TinyLM()
    rows = [{"id": str(i), "source_split": "train", "ids": [12, 13, 14, 15 + i, 16, 2],
             "labels": [-100, -100, -100, 15 + i, 16, 2]} for i in range(8)]
    d = Design(processed_budget=48, max_sequence=16, update_tokens=16, trainable_tail=2, ce_chunk=2)
    return model, rows, d


def evaluate(model, label, reserve):
    # Even evaluator randomness cannot perturb the training random stream.
    random.random(); torch.rand(3)
    reserve({"evaluation_and_qualification_model_calls": 1})
    return {"label": label, "fixture_parameter_sum": sum(float(p.sum()) for p in model.parameters()), "synthetic": True}


def test_fresh_instance_resume_matches_complete_model(tmp_path):
    a, rows, d = fixture(); charges = []
    first = microfit(a, rows, d, {"fixture": True, "experiment": "FIXTURE"}, tmp_path / "full", evaluate=evaluate,
                     reserve=charges.append, resource_guard=lambda: {}, synthetic=True)
    b, _, _ = fixture(); resumed_charges = []
    second = microfit(b, rows, d, {"fixture": True, "experiment": "FIXTURE"}, tmp_path / "resumed", evaluate=evaluate,
                      reserve=resumed_charges.append, resource_guard=lambda: {}, synthetic=True,
                      resume=tmp_path / "full/checkpoint-00002")
    assert first["final_state_sha256"] == second["final_state_sha256"]
    assert first["final_evaluation_sha256"] == second["final_evaluation_sha256"]
    assert first["attempt_updates"] == 4 and second["attempt_updates"] == 2
    assert sum(x.get("processed_training_tokens_all_attempts", 0) for x in charges) == 48
    assert sum(x.get("processed_training_tokens_all_attempts", 0) for x in resumed_charges) == 24
    assert (tmp_path / "full/checkpoint-00002/manifest.json").is_file()
    assert (tmp_path / "full/checkpoint-00004/manifest.json").is_file()
    assert (tmp_path / "resumed/checkpoint-00004/manifest.json").is_file()
    assert not second["quality_improvement_established"]


def test_budget_refusal_precedes_optimizer_update(tmp_path):
    model, rows, d = fixture()
    before = deepcopy(model.state_dict())
    def refuse(charge):
        if "processed_training_tokens_all_attempts" in charge:
            raise RuntimeError("budget exhausted")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        microfit(model, rows, d, {"fixture": True, "experiment": "FIXTURE"}, tmp_path / "refused", evaluate=evaluate,
                 reserve=refuse, resource_guard=lambda: {}, synthetic=True)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert (tmp_path / "refused/baseline.json").is_file()
    assert (tmp_path / "refused/steps.jsonl").read_bytes() == b""


def test_output_create_only_and_test_split_rejected(tmp_path):
    model, rows, d = fixture()
    binding = {"experiment": "FIXTURE"}
    out = tmp_path / "existing"; out.mkdir(); (out / "original").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        microfit(model, rows, d, binding, out, evaluate=evaluate, reserve=lambda x: None,
                 resource_guard=lambda: {}, synthetic=True)
    assert (out / "original").read_bytes() == b"keep"
    rows[0]["source_split"] = "test_iid"
    with pytest.raises(ValueError, match="TRAIN"):
        microfit(model, rows, d, binding, tmp_path / "bad-split", evaluate=evaluate,
                 reserve=lambda x: None, resource_guard=lambda: {}, synthetic=True)
    assert not (tmp_path / "bad-split").exists()


def test_the_result_is_attributed_to_the_bindings_experiment_not_a_literal(tmp_path):
    """A successor arm must not appear in its own result file as RRP-001-MICROFIT."""
    model, rows, d = fixture()
    got = microfit(model, rows, d, {"fixture": True, "experiment": "RRP-003-ARM-A1"},
                   tmp_path / "arm", evaluate=evaluate, reserve=lambda x: None,
                   resource_guard=lambda: {}, synthetic=True)
    assert got["experiment"] == "RRP-003-ARM-A1"


def test_an_unattributed_binding_is_refused(tmp_path):
    model, rows, d = fixture()
    with pytest.raises(ValueError, match="binding must name its experiment"):
        microfit(model, rows, d, {"fixture": True}, tmp_path / "x", evaluate=evaluate,
                 reserve=lambda v: None, resource_guard=lambda: {}, synthetic=True)
