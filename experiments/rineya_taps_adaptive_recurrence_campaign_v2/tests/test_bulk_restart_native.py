"""Synthetic metadata/model mocks only; no CUDA query or Rineya loader."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_restart_native as restart


def rows(count=192, random=False, split="DEVELOPMENT"):
    result=[]
    for index in range(count):
        case=f"synthetic:{index}"
        policies=[("A0",None,17),("A1",None,17),("A3",None,17)]
        policies += [("A2",alpha,17) for alpha in restart.GRID]
        if random:policies += [("A5",None,seed) for seed in (17,29)]
        result += [{"case_id":case,"arm":arm,"seed":seed,"fixed_alpha":alpha,
                    "split":split,"physical_run":True,"score":0}
                   for arm,alpha,seed in policies]
    return result


def checkpoint(tmp_path, *, state="original", mutation=None, sync_failure=False):
    calls=[]
    model=SimpleNamespace(training=False,forward_calls=42,
        parameters=lambda:[SimpleNamespace(requires_grad=False,grad=None,numel=lambda:1)])
    files={"modeling_supa_a11.py":"code","model.safetensors":"weights"}
    admission={"checkpoint_files":files,"dispatch_hashes":{"synthetic_source.py":"source"}}
    if mutation:mutation(model,admission)
    def sync():
        calls.append("sync")
        if sync_failure:raise RuntimeError("SYNTHETIC_CUDA_FAULT_NO_RECOVERY")
    def state_hash(value):calls.append("state");assert value is model;return state
    def asset_hash(_source):calls.append("assets");return files
    def hash_file(path):return "code" if Path(path).name=="modeling_supa_a11.py" else "source"
    def guard():calls.append("resources");return {"device":"fabricated-no-native"}
    kwargs=dict(phase="SYNTHETIC_BOUNDARY",counts={"scientific_physical_rows":1728},
        path=tmp_path/"proof.json",state_hash=state_hash,checkpoint_files=asset_hash,
        synchronize=sync,guard=guard,started=restart.time.perf_counter(),sha_fn=hash_file)
    return model,admission,files,calls,kwargs


def test_complete_unfavorable_DEV_counts_are_not_a_quality_selection_gate():
    deterministic=rows();ids={x["case_id"] for x in deterministic}
    assert restart.scientific_phase_counts(deterministic,ids,"DEVELOPMENT_DETERMINISTIC_BEFORE_FREEZE",1848)["scientific_physical_rows"]==1728
    all_dev=rows(random=True)
    assert restart.scientific_phase_counts(all_dev,ids,"DEVELOPMENT_COMPLETE_BEFORE_VALIDATION",2240)["scientific_physical_rows"]==2112
    final=rows(count=288,random=True)
    assert restart.scientific_phase_counts(final,{x["case_id"] for x in final},"FINAL_MATRIX_COMPLETE",3296)["scientific_physical_rows"]==3168
    bad=deepcopy(deterministic);bad[-1]=deepcopy(bad[0])
    with pytest.raises(RuntimeError,match="scientific cells"):
        restart.scientific_phase_counts(bad,ids,"DEVELOPMENT_DETERMINISTIC_BEFORE_FREEZE",1848)
    bad=deepcopy(deterministic);bad[0]["split"]="OPEN_VALIDATION"
    with pytest.raises(PermissionError,match="validation rows"):
        restart.scientific_phase_counts(bad,ids,"DEVELOPMENT_DETERMINISTIC_BEFORE_FREEZE",1848)


def test_integrity_records_actual_hash_and_zero_forward_or_RNG_work(tmp_path):
    model,admission,files,calls,kwargs=checkpoint(tmp_path)
    before_rng=torch.get_rng_state().clone()
    proof=restart.integrity_checkpoint(model,tmp_path,tmp_path/"modeling_supa_a11.py","original",files,admission,**kwargs)
    assert calls==["sync","state","assets","resources"] and model.forward_calls==42
    assert proof["after_state_sha256"]=="original" and proof["model_forward_calls_added"]==0
    assert proof["status"]=="PASS_MODEL_ASSETS_SOURCE_INTEGRITY"
    assert torch.equal(torch.get_rng_state(),before_rng)
    with pytest.raises(FileExistsError):
        restart.integrity_checkpoint(model,tmp_path,tmp_path/"modeling_supa_a11.py","original",files,admission,**kwargs)


@pytest.mark.parametrize("failure",["state","source","assets","code","grad","requires_grad"])
def test_boundary_failure_preserves_failed_proof_and_prevents_next_phase(tmp_path,failure):
    def mutation(model,admission):
        if failure=="source":admission["dispatch_hashes"]["synthetic_source.py"]="changed"
        elif failure=="grad":model.parameters=lambda:[SimpleNamespace(requires_grad=False,grad=object(),numel=lambda:1)]
        elif failure=="requires_grad":model.parameters=lambda:[SimpleNamespace(requires_grad=True,grad=None,numel=lambda:1)]
    model,admission,files,calls,kwargs=checkpoint(tmp_path,state="changed" if failure=="state" else "original",mutation=mutation)
    if failure=="assets":kwargs["checkpoint_files"]=lambda _source:{**files,"model.safetensors":"changed"}
    elif failure=="code":kwargs["sha_fn"]=lambda _path:"changed"
    progression=[]
    with pytest.raises(RuntimeError,match="integrity differs"):
        restart.integrity_checkpoint(model,tmp_path,tmp_path/"modeling_supa_a11.py","original",files,admission,**kwargs)
        progression.append("validation")
    proof=json.loads(kwargs["path"].read_text())
    assert not progression and proof["status"]=="FAIL_MODEL_ASSETS_SOURCE_INTEGRITY"
    assert proof["resources"] is None and model.forward_calls==42


def test_checkpoint_callback_binds_completed_proof_and_exact_journal_prefix(tmp_path):
    model,admission,files,calls,kwargs=checkpoint(tmp_path)
    stream=tmp_path/"outcomes.jsonl";stream.write_bytes(b'{"physical_run":true}\n{"physical_run":true}\n')
    original_hash=kwargs["sha_fn"]
    kwargs["sha_fn"]=lambda path:restart.sha(path) if Path(path).name in ("proof.json","outcomes.jsonl") else original_hash(path)
    events=[]
    def sink(event):
        if event["event"]=="INTEGRITY_CHECK_PASSED":assert kwargs["path"].is_file()
        events.append(event)
    kwargs.update(event_sink=sink,checkpoint_index=2,journal_paths=(stream,))
    proof=restart.integrity_checkpoint(model,tmp_path,tmp_path/"modeling_supa_a11.py","original",files,admission,**kwargs)
    assert [event["sequence"] for event in events]==[3,4]
    assert [event["event"] for event in events]==["INTEGRITY_CHECK_STARTED","INTEGRITY_CHECK_PASSED"]
    assert events[-1]["proof_sha256"]==restart.sha(kwargs["path"])
    assert proof["journal_commitments"]["outcomes.jsonl"]=={"bytes":stream.stat().st_size,"lines":2,"sha256":restart.sha(stream)}


def test_cuda_fault_has_no_state_resource_recovery_or_next_forward(tmp_path):
    model,admission,files,calls,kwargs=checkpoint(tmp_path,sync_failure=True)
    with pytest.raises(RuntimeError,match="NO_RECOVERY"):
        restart.integrity_checkpoint(model,tmp_path,tmp_path/"modeling_supa_a11.py","original",files,admission,**kwargs)
    proof=json.loads(kwargs["path"].read_text())
    assert calls==["sync"] and model.forward_calls==42
    assert proof["after_state_sha256"] is None and proof["checkpoint_files"] is None and proof["resources"] is None


def test_parent_clock_exact_scope_no_import_time_fallback(monkeypatch):
    monkeypatch.setattr(restart.time,"perf_counter",lambda:1000.)
    admission={"hard_deadline_seconds":16000,"native_finalization_reserve_seconds":1800}
    monkeypatch.setenv(restart.PARENT_CLOCK_ENV,"900")
    assert restart.validated_parent_clock(admission)==900
    for value in ("1001","nan","inf","notnumeric","-16000"):
        monkeypatch.setenv(restart.PARENT_CLOCK_ENV,value)
        with pytest.raises(RuntimeError):restart.validated_parent_clock(admission)
    monkeypatch.delenv(restart.PARENT_CLOCK_ENV)
    with pytest.raises(RuntimeError,match="missing"):restart.validated_parent_clock(admission)


def test_forward_clock_stamps_are_parent_bound_and_cannot_retain_untrusted_fields(monkeypatch):
    monkeypatch.setattr(restart.time,"perf_counter",lambda:1000.)
    event=restart.timed_event({"event":"FORWARD_STARTED","parent_elapsed_seconds":99999,"timestamp_utc":"wrong"},900.)
    assert event["parent_elapsed_seconds"]==100 and event["timestamp_utc"]!="wrong"
    assert event["event"]=="FORWARD_STARTED"


def test_missing_actual_grant_rejects_before_any_source_model_load(monkeypatch,tmp_path):
    admission=tmp_path/"admission.json";admission.write_text("{}")
    def missing(_admission):raise PermissionError("EXPLICIT_ADDITIONAL_ATTEMPT_REQUIRED")
    def forbidden(*_args,**_kwargs):raise AssertionError("source/model loader reached without grant")
    monkeypatch.setattr(restart,"validate_human_scope",missing)
    monkeypatch.setattr(restart,"admitted_source",forbidden)
    monkeypatch.setattr(restart,"final_resource_snapshot",lambda:{"device":"no-model-fixture"})
    out=tmp_path/"result";assert restart.run(tmp_path/"no-model",out,admission)==2
    receipt=json.loads((out/"receipt.json").read_text())
    assert receipt["model_load_attempts"]==receipt["native_model_loads"]==0
    assert "EXPLICIT_ADDITIONAL_ATTEMPT_REQUIRED" in receipt["error"] and receipt["wall_seconds"] is None


def test_frozen_scientific_helpers_caps_and_phase_call_order_remain_exact():
    package=Path(restart.__file__).resolve().parent
    old=ast.parse((package/"bulk_native.py").read_text());new=ast.parse(Path(restart.__file__).read_text())
    definitions=lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.FunctionDef)}
    before,after=definitions(old),definitions(new)
    for name in ("bank_module","count_forecast","deterministic_order","read_bulk_packet","adapt_bulk_case",
                 "strict_score","bulk_diagnostics","freeze_development","admit_validation"):
        assert before[name]==after[name],name
    forecast=restart.count_forecast()
    assert forecast["physical_evaluations"]==3296 and forecast["model_API_attempt_cap"]==28436
    assert restart.adapt_bulk_case({"id":"fake","category":"fake","parent_id":"fake","prompt":"fake"})["max_new_tokens"]==8
    run=next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name=="run")
    calls=sorted((node.lineno,ast.unparse(node.func),ast.unparse(node)) for node in ast.walk(run) if isinstance(node,ast.Call))
    det=next(line for line,fn,text in calls if fn=="scientific_phase_counts" and "det_phase" in text)
    freeze=next(line for line,fn,text in calls if fn=="freeze_development")
    dev=next(line for line,fn,text in calls if fn=="scientific_phase_counts" and "dev_phase" in text)
    validation=next(line for line,fn,text in calls if fn=="admit_validation")
    assert det<freeze<dev<validation
