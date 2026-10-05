"""Source/JSON specification fidelity only; never imports experiment modules."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
SOURCES=("score_import.py","study_native_chunk.py","study_torch_backend.py","full27_chunk.py",
    "full27_contract.py","final_adapter_inference.py","inference_component.py",
    "short_collection_worker.py","study_schedule.py","rs_contract.py",
    "DEV_FINISH_POLICY_001.json","DEV_STATISTICS_POLICY.json","native_loader.py")

def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(",",":"),allow_nan=False)
def descriptor(path):
    raw=path.read_bytes()
    return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
def tree(name):return ast.parse((PACKAGE/name).read_text(encoding="utf-8"))
def function(module,name):
    values=[node for node in ast.walk(module) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name]
    assert len(values)==1,name
    return values[0]
def literal_return_keys(node):
    returns=[value.value for value in node.body if isinstance(value,ast.Return)]
    assert len(returns)==1 and isinstance(returns[0],ast.Dict),node.name
    return {value.value for value in returns[0].keys if isinstance(value,ast.Constant) and type(value.value) is str}
def literal_constant(module,name):
    values=[node.value for node in module.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)]
    assert len(values)==1,name
    return ast.literal_eval(values[0])

def main():
    target=HERE/"SOURCE_SPECIFICATION_VALIDATION_002.json"
    if target.exists():raise ValueError("write-once source-specification review already exists")
    before={name:descriptor(PACKAGE/name) for name in SOURCES}
    proposal=json.loads((HERE/"PROPOSED_SCHEMAS.json").read_text(encoding="utf-8"))
    findings=[]
    def check(name,condition):
        assert condition,name
        findings.append(name)
    schemas=proposal["schemas"]
    check("proposed specification is not native implementation or authority",proposal["native_api_implemented"] is False and proposal["real_native_authority_materialized"] is False)
    check("all outer field sets closed",all(item["additional_fields_allowed"] is False for item in schemas.values()))
    component=tree("inference_component.py")
    raw=literal_return_keys(function(component,"_receipt"))|literal_return_keys(function(component,"_identity"))|{"decoded_response","parameter_preservation"}
    check("raw receipt exact source field set",raw==set(schemas["InferenceRawReceipt"]["fields"]))
    metadata=literal_return_keys(function(tree("study_torch_backend.py"),"_metadata"))
    check("study metadata exact source field set",metadata==set(schemas["StudyCheckpointMetadata"]["fields"]))
    chunk=tree("study_native_chunk.py");qualification=tree("full27_chunk.py");backend=tree("study_torch_backend.py")
    check("binding exact seven source fields",set(schemas["StudyBinding"]["fields"])==literal_constant(chunk,"BINDING_FIELDS"))
    check("study and qualification kinds differ",literal_constant(chunk,"KIND")=="SYNTH_STUDY_CHUNK_PROTOCOL_V1" and literal_constant(qualification,"KIND")=="SYNTH_FULL27_QUALIFICATION_CHUNK_PROTOCOL_V1")
    check("shared backend archive kind does not authenticate study",literal_constant(chunk,"BACKEND_KIND")==literal_constant(qualification,"BACKEND_KIND")==literal_constant(backend,"KIND"))
    check("no actual receipt paths or hashes materialized",proposal["native_NULL_paths_hashes_or_stub_receipts_allowed"] is False and len(proposal["missing_native_slots"])==7)
    finish=json.loads((PACKAGE/"DEV_FINISH_POLICY_001.json").read_text(encoding="utf-8"))
    check("exact original row order",schemas["NativeCandidateSelection"]["fields"]["row_order_sha256"]==finish["population"]["row_order_sha256"])
    check("exact original panel metadata",schemas["NativeCandidateSelection"]["fields"]["panel_metadata_sha256"]==finish["population"]["panel_metadata_sha256"])
    check("finish scoring authority still absent",finish["native_scoring_authorized"] is False and finish["all_three_arm_checkpoint_registration_complete"] is False)
    check("plain source anchors have no invented JSON body",set(schemas["PlainFileAnchor"]["fields"])=={"path","bytes","file_sha256"}
        and set(schemas["JsonDocumentAnchor"]["fields"])=={"path","bytes","file_sha256","body_sha256","body_hash_convention"}
        and schemas["NativeCandidateSelection"]["fields"]["collector_auditor_source"].startswith("PlainFileAnchor"))
    check("postcapture six audits externally selected without identity cycle",set(schemas["CandidateBatchAnchors"]["fields"])=={"batch_ordinal","audit","config"}
        and "batch_anchors" in schemas["CandidateImportSelection"]["fields"]
        and "import_selection_file_sha256" in schemas["CandidateImportPacket"]["fields"]
        and not any("import_selection" in key or "batch_anchors" in key for name in ("NativeCandidateSelection","CandidateIdentity","CandidateRowEnvelope","CandidateBatchAudit") for key in schemas[name]["fields"]))
    restore=function(tree("final_adapter_inference.py"),"restore_final");success_keys=set()
    for node in ast.walk(restore):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="result" for t in node.targets) and isinstance(node.value,ast.Dict):
            success_keys.update(k.value for k in node.value.keys)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="result" and node.func.attr=="update" and node.args and isinstance(node.args[0],ast.Dict):
            keys={k.value for k in node.args[0].keys}
            if "error" not in keys:success_keys.update(keys)
    success_keys.add("operation_costs")
    check("unsealed final restore success fields exact and full-result digest explicit",success_keys==set(schemas["FactorRestoreReceipt"]["fields"])
        and "ASCII_UNSEALED_FULL_DOCUMENT" in schemas["CandidateBatchAudit"]["fields"]["restore_receipt_body_sha256"]
        and "exactly112" in schemas["FactorRestoreReceipt"]["fields"]["operation_costs"])
    trace=function(component,"_trace");trace_text=ast.unparse(trace)
    readme=(HERE/"README.md").read_text(encoding="utf-8")
    check("trace prefix uses source zero-based pre-generation position", "'prefix_length': len(prefix)" in trace_text
        and "prompt_tokens+step0" in readme and "prompt_tokens+step1-1" in readme
        and "step0" in schemas["AuditFindings"]["fields"]["actual_capture_input_positions"])
    runner=function(chunk,"run_chunk")
    admissions=[node for node in ast.walk(runner) if isinstance(node,ast.Set) and all(isinstance(e,ast.Constant) for e in node.elts)
        and {e.value for e in node.elts}=={"binding","authorized","owned_supervision_verified","model_load_and_constructor_in_scope","observed_monotonic","expires_monotonic"}]
    check("actual study factory proof retains exact six source fields",len(admissions)==1 and set(schemas["StudyNativeAdmission"]["fields"])=={e.value for e in admissions[0].elts}
        and "native_admission" in schemas["OwnedStudyChunkAudit"]["fields"] and "fresh_start_attestation" in schemas["OwnedStudyChunkAudit"]["fields"])
    shape_text=ast.unparse(function(tree("native_loader.py"),"_adapter_shapes"))
    check("source rank8 FP32 factor byte convention/order and three observations fixed", "range(28)" in shape_text and "[2048, 8]" in shape_text
        and "[8, 1024]" in shape_text and "[1024, 8]" in shape_text
        and set(schemas["FactorByteObservations"]["fields"])=={"archive","resident_before_capture","resident_after_capture"}
        and schemas["FactorByteObservation"]["fields"]["total_factor_bytes"]=="integer4587520 under unchanged rank8/source inventory")
    after={name:descriptor(PACKAGE/name) for name in SOURCES}
    check("all existing source and policy bytes unchanged",before==after)
    artifacts={name:descriptor(HERE/name) for name in ("README.md","PROPOSED_SCHEMAS.json","ADVERSARIAL_CONTROLS.md","validate_source_specification.py")}
    control_lines=[line for line in (HERE/"ADVERSARIAL_CONTROLS.md").read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    receipt={"schema_version":1,"recorded_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS_SOURCE_SPECIFICATION_FIDELITY_ONLY_NO_IMPLEMENTATION_OR_NATIVE_AUTHORITY",
        "checks":findings,"check_count":len(findings),"planned_adversarial_control_count":len(control_lines)-1,
        "source_before":before,"source_after":after,"source_unchanged":before==after,"artifact_files":artifacts,
        "experiment_modules_imported_or_executed":False,"native_model_asset_runtime_binary_GPU_process_probes_or_execution":False,
        "real_study_or_capture_receipts_invented":False,"native_scoring_or_grader_called":False,"tracked_files_or_HEAD_changed_by_specification":False,
        "native_contract_implemented":False,"native_or_scoring_authority_issued":False,"scientific_goal_complete":False}
    receipt["sha256"]=hashlib.sha256(canonical(receipt).encode("ascii")).hexdigest()
    with target.open("x",encoding="utf-8",newline="\n") as stream:stream.write(canonical(receipt)+"\n")
    print(canonical({key:receipt[key] for key in ("status","check_count","planned_adversarial_control_count","source_unchanged","sha256")}))

if __name__=="__main__":main()
