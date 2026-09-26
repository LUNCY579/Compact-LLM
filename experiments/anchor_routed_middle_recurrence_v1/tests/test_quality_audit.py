"""The derived quality audit must preserve source evidence and censor timeouts."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import audit_fixed_results


def test_quality_audit_preserves_source_and_blocks_censored_gain(tmp_path,monkeypatch):
    run=tmp_path/"evidence/02_fixed_recurrence/source"
    outputs=run/"outputs"
    outputs.mkdir(parents=True)
    base=[{"case_id":"a","family":"qa","score":0.,"invalid":False,
           "safety_stopped":True,"termination":"timeout","wall_seconds":91.,
           "logit_cosine":1.,"kl_base_to_arm":0.}]
    candidate=[{**base[0],"score":1.,"safety_stopped":False,"termination":"eos"}]
    source_files={run/"run_manifest.json":{"canonical_unchanged":True,"status":"PASS"},
                  outputs/"A0_dev.json":base,outputs/"A1_dev.json":candidate}
    for path,value in source_files.items():
        path.write_text(json.dumps(value),encoding="utf-8")
    before={p:p.read_bytes() for p in source_files}
    monkeypatch.setattr(audit_fixed_results,"ROOT",tmp_path)
    monkeypatch.setattr(sys,"argv",["audit_fixed_results.py",str(run)])
    assert audit_fixed_results.main()==0
    result_path=next((tmp_path/"evidence/02b_fixed_quality_audit").glob("*/decision.json"))
    result=json.loads(result_path.read_text())
    assert result["status"]=="INCONCLUSIVE" and not result["adaptive_routing"]
    assert result["coverage"]["A0_dev"]["completed_cases"]==0
    assert before=={p:p.read_bytes() for p in source_files}
