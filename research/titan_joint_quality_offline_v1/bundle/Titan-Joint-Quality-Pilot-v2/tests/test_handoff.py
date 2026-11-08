from copy import deepcopy
import json
import pytest
from titan_joint_quality_v1.common import CONTRACT, digest, file_sha
from titan_joint_quality_v1.__main__ import verify_handoff_coupling
from titan_joint_quality_v1 import exposure, common

def fixture(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    profile = tmp_path / "profile"; profile.mkdir()
    f = data / "DATA_MANIFEST.json"; f.write_text('{"kind":"fixture"}', encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"status": "PASS_METADATA_ONLY", "native_qualified": False,
        "models": {"sparse": {"sidecars": {"tokenizer.json": "t"}}}}), encoding="utf-8")
    tokenizer = {"parent_sha256": CONTRACT["parent_sha256"], "model_loads": 0, "sidecars": {"tokenizer.json": "t"}}
    expected = {"parent_sha256": CONTRACT["parent_sha256"], "data_sha256": file_sha(f),
        "tokenizer_sha256": digest(tokenizer), "source_receipt_sha256": file_sha(source)}
    plan = {"status": "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN", "source_bindings": expected,
            "source_bindings_sha256": digest(expected), "exposure_source_sha256": file_sha(exposure.__file__),
            "common_source_sha256": file_sha(common.__file__)}
    plan["plan_sha256"] = digest(plan)
    counts = {"status": "PASS_OFFLINE_TOKEN_SHAPES", "data_binding": expected, "errors": []}
    for name, d in (("EXPOSURE_PLAN.json", plan), ("TOKENIZER_BINDING.json", tokenizer), ("TOKEN_COUNT_CONTRACT.json", counts)):
        (profile / name).write_text(json.dumps(d), encoding="utf-8")
    return data, source, profile

def test_exact_handoff_files_are_coupled(tmp_path):
    paths = fixture(tmp_path); assert verify_handoff_coupling(*paths)

@pytest.mark.parametrize("defect", ["data", "source", "tokenizer", "plan", "rehashed_foreign_plan", "counts"])
def test_handoff_rejects_changed_or_foreign_bindings_before_output(defect, tmp_path):
    data, source, profile = fixture(tmp_path)
    if defect == "data":
        (data / "DATA_MANIFEST.json").write_text('{"changed":true}', encoding="utf-8")
    elif defect == "source":
        source.write_text('{"status":"BLOCKED"}', encoding="utf-8")
    else:
        name = {"tokenizer": "TOKENIZER_BINDING.json", "counts": "TOKEN_COUNT_CONTRACT.json"}.get(defect, "EXPOSURE_PLAN.json")
        f = profile / name; d = json.loads(f.read_text(encoding="utf-8"))
        if defect == "tokenizer": d["sidecars"]["tokenizer.json"] = "wrong"
        elif defect == "counts": d["data_binding"]["parent_sha256"] = "wrong"
        elif defect == "plan": d["changed"] = True
        else:
            d["source_bindings"]["data_sha256"] = "wrong"
            d["source_bindings_sha256"] = digest(d["source_bindings"])
            d["plan_sha256"] = digest({k: v for k, v in d.items() if k != "plan_sha256"})
        f.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError): verify_handoff_coupling(data, source, profile)
