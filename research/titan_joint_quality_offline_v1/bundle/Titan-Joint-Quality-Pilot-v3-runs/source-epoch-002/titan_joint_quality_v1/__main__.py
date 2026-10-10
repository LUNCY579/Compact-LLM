"""Offline CLI. No model loading, training dispatch or confirmation entrypoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .common import CONTRACT, OWNER_CHAT, SOURCE_ROOT, digest, file_sha, new_directory, write_json, write_jsonl, utc

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def dataset(path):
    path = Path(path).resolve()
    if any("confirm" in p.lower() or "sealed" in p.lower() for p in path.parts):
        raise PermissionError("confirmation/sealed inputs are closed")
    manifest = read_json(path / "DATA_MANIFEST.json")
    if manifest.get("schema") != "TITAN_OFFLINE_DATA_V1" or manifest.get("confirmation_materialized") is not False:
        raise ValueError("only this open TRAIN/PILOT artifact schema is accepted")
    from . import data as teacher_source
    if manifest.get("teacher", {}).get("source_digest") != digest({"data.py": Path(teacher_source.__file__).read_text(encoding="utf-8")}):
        raise ValueError("data generator/teacher source changed; preserve old artifacts and build a new exclusive epoch")
    names = ("train.jsonl", "pilot.prompts.jsonl", "pilot.references.jsonl")
    if set(manifest.get("artifact_sha256", {})) != set(names):
        raise ValueError("unexpected artifact membership")
    for n in names:
        if file_sha(path / n) != manifest["artifact_sha256"][n]:
            raise ValueError("changed data artifact:" + n)
    rows = {n: [json.loads(line) for line in (path / n).read_text(encoding="utf-8").splitlines()] for n in names}
    from .data import validate_bank
    validate_bank(rows["train.jsonl"], "train")
    prompts = {r["case_id"]: r for r in rows["pilot.prompts.jsonl"]}
    if len(prompts) != 384 or any(set(r) != {"case_id", "messages"} for r in prompts.values()):
        raise ValueError("pilot projection/coverage differs")
    pilot = [{**r, "messages": prompts[r["case_id"]]["messages"]} for r in rows["pilot.references.jsonl"]]
    validate_bank(pilot, "pilot")
    return manifest, rows["train.jsonl"], pilot

def plan_exposure(data_path, source_path, out):
    from .tokenizer_adapter import LocalTokenizerAdapter
    from .exposure import encode_rows, build_exposure_plan
    manifest, train, pilot = dataset(data_path)
    source = read_json(source_path)
    adapter = LocalTokenizerAdapter(source)
    encoded = encode_rows(train, adapter)
    binding = {"parent_sha256": CONTRACT["parent_sha256"],
        "data_sha256": file_sha(Path(data_path) / "DATA_MANIFEST.json"),
        "tokenizer_sha256": digest(adapter.manifest), "source_receipt_sha256": file_sha(source_path)}
    plan = build_exposure_plan(encoded, binding)
    pilot_shapes, pilot_errors = [], []
    for row in pilot:
        try:
            e = adapter.encode_case(row["messages"], row["target"])
            if e["prompt_tokens"] + CONTRACT["max_new_tokens"] > CONTRACT["max_eval_total_tokens"] or e["response_tokens"] > CONTRACT["max_new_tokens"]:
                raise ValueError("pilot input or canonical output exceeds frozen bounds")
            pilot_shapes.append({"case_id": row["case_id"], "prompt_tokens": e["prompt_tokens"],
                                 "canonical_response_plus_EOS_tokens": e["response_tokens"]})
        except ValueError as error:
            pilot_errors.append({"case_id": row["case_id"], "error": str(error)})
    path = new_directory(out)
    write_json(path / "TOKENIZER_BINDING.json", adapter.manifest)
    write_json(path / "EXPOSURE_PLAN.json", plan)
    summary = {k: v for k, v in encoded.items() if k != "encoded_rows"}
    write_json(path / "ENCODING_RECEIPT.json", summary)
    write_jsonl(path / "encoded_train.jsonl", encoded.get("encoded_rows", []))
    counts = {"complete_training_inputs_per_arm": len(plan.get("selected_case_ids", [])),
        "training_processed_positions_pair_cap": 2 * plan.get("processed_tokens", 0),
        "training_full_row_forward_cap_pair": 2 * len(plan.get("selected_case_ids", [])),
        "training_work_scope": "input exposure only; native backward/checkpoint recomputation, optimizer and hashing costs must also be measured and charged",
        "evaluation_model_configs": 5, "evaluation_case_generations": 5 * 384,
        "evaluation_generation_API_cap": 5 * 384 * CONTRACT["max_new_tokens"],
        "evaluation_full_prefix_positions_cap": 5 * sum(48 * r["prompt_tokens"] + 48 * 47 // 2 for r in pilot_shapes),
        "qualification_forward_cap": None, "qualification_backward_cap": None,
        "limit": "Qualification counts require the owner's exact reviewed native adapter; no complete forecast yet."}
    token_receipt = {"status": "PASS_OFFLINE_TOKEN_SHAPES" if not pilot_errors and len(pilot_shapes) == 384 else "BLOCKED_OFFLINE_TOKEN_SHAPES",
        "pilot_shapes": pilot_shapes, "errors": pilot_errors, "count_contract": counts,
        "data_binding": binding, "native_qualified": False, "native_model_calls": 0,
        "trained_model_quality_evidence": False}
    write_json(path / "TOKEN_COUNT_CONTRACT.json", token_receipt)
    return {"exposure_status": plan.get("status"), "processed_tokens_per_arm": plan.get("processed_tokens"),
            "selected_rows": len(plan.get("selected_case_ids", [])), "token_shapes_status": token_receipt["status"],
            "output": str(path), "native_authorized": False}

def handoff(data_path, source_path, plan_path, out):
    data_manifest, _, _ = dataset(data_path)
    source, plan = read_json(source_path), read_json(Path(plan_path) / "EXPOSURE_PLAN.json")
    verify_handoff_coupling(data_path, source_path, plan_path)
    path = new_directory(out)
    sources = {str(f.relative_to(SOURCE_ROOT)): file_sha(f) for f in SOURCE_ROOT.rglob("*.py") if "__pycache__" not in f.parts}
    packet = {"created_utc": utc(), "status": "HELD_OWNER_REVIEW_AND_NATIVE_INTEGRATION_REQUIRED",
        "existing_owner_chat": OWNER_CHAT, "owner_messages_by_this_implementation": 0,
        "contract": CONTRACT, "data_manifest_sha256": file_sha(Path(data_path) / "DATA_MANIFEST.json"),
        "source_receipt_sha256": file_sha(source_path), "exposure_plan_sha256": file_sha(Path(plan_path) / "EXPOSURE_PLAN.json"),
        "source_closure": sources, "observed_data_counts": {"train": 2000, "pilot": 384},
        "prospective_confirmation": {"rows": 768, "materialized": False, "read_authorized": False},
        "metadata_source_status": source.get("status"), "exposure_status": plan.get("status"),
        "complete_exposure_mechanics_proof": False, "executable_native_command": None,
        "blockers": ["Teacher independence, family/shortcut/overlap and instrument reviews are not signed.",
            "The new Sparse source, shared-denominator orchestration, tokenizer and runtime require native qualification.",
            "Owner adapter/contained process supervision, exact qualification counts and complete measured forecast are absent.",
            "No new direct user native-run grant; the six-hour ceiling is a prospective budget only.",
            "Native baseline capture/freeze and raw result review are not implemented by the offline fixture analyser."],
        "data_review_status": data_manifest.get("status"), "serving_policy_changes": False,
        "native_model_calls": 0, "quality_improvement_established": False,
        "next_decision": "Existing Titan owner reviews the additive package, chooses whether to integrate its masked-loss plan into the qualified contained lane, then prepares one exact reviewed native-run request. No competing worker or historical permit reuse."}
    write_json(path / "OWNER_HANDOFF.json", packet)
    return {"status": packet["status"], "output": str(path), "native_authorized": False}

def verify_handoff_coupling(data_path, source_path, plan_path):
    source = read_json(source_path)
    plan = read_json(Path(plan_path) / "EXPOSURE_PLAN.json")
    tokenizer = read_json(Path(plan_path) / "TOKENIZER_BINDING.json")
    counts = read_json(Path(plan_path) / "TOKEN_COUNT_CONTRACT.json")
    if source.get("status") != "PASS_METADATA_ONLY" or source.get("native_qualified") is not False:
        raise ValueError("exact offline source audit required; native qualification cannot be inferred")
    original = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("status") != "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN" or plan.get("plan_sha256") != digest(original):
        raise ValueError("missing/changed/blocked exposure plan")
    from . import exposure as producer
    from . import common as scientific_contract
    if plan.get("exposure_source_sha256") != file_sha(producer.__file__) or plan.get("common_source_sha256") != file_sha(scientific_contract.__file__):
        raise ValueError("exposure/contract producer source changed")
    expected = {"parent_sha256": CONTRACT["parent_sha256"],
        "data_sha256": file_sha(Path(data_path) / "DATA_MANIFEST.json"),
        "tokenizer_sha256": digest(tokenizer), "source_receipt_sha256": file_sha(source_path)}
    if plan.get("source_bindings") != expected or plan.get("source_bindings_sha256") != digest(expected):
        raise ValueError("plan/data/source/tokenizer coupling differs")
    if counts.get("data_binding") != expected or counts.get("status") != "PASS_OFFLINE_TOKEN_SHAPES" or counts.get("errors"):
        raise ValueError("token-shape receipt coupling differs")
    if tokenizer.get("parent_sha256") != CONTRACT["parent_sha256"] or tokenizer.get("model_loads") != 0:
        raise ValueError("wrong tokenizer binding")
    if tokenizer.get("sidecars") != source["models"]["sparse"]["sidecars"]:
        raise ValueError("tokenizer assets differ from source audit")
    return expected

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("build-data"); d.add_argument("--out", required=True); d.add_argument("--seed", type=int, default=17)
    s = sub.add_parser("source-audit"); s.add_argument("--out", required=True)
    p = sub.add_parser("plan-exposure"); p.add_argument("--data", required=True); p.add_argument("--source", required=True); p.add_argument("--out", required=True)
    h = sub.add_parser("handoff"); h.add_argument("--data", required=True); h.add_argument("--source", required=True); h.add_argument("--plan", required=True); h.add_argument("--out", required=True)
    f = sub.add_parser("fixture-pair"); f.add_argument("--out", required=True)
    sub.add_parser("native-run"); sub.add_parser("confirmation")
    args = parser.parse_args()
    if args.command in ("native-run", "confirmation"):
        parser.exit(2, "REFUSED: native execution and confirmation are closed; continue through the existing owner.\n")
    if args.command == "build-data":
        from .data import build_dataset
        value = build_dataset(args.out, args.seed)
        value = {"status": value["status"], "counts": {"train": 2000, "pilot": 384}, "confirmation_materialized": False, "out": args.out}
    elif args.command == "source-audit":
        from .source import audit_sources
        value = audit_sources(); write_json(args.out, value)
        value = {"status": value["status"], "errors": value["errors"], "native_qualified": False, "out": args.out}
    elif args.command == "plan-exposure":
        value = plan_exposure(args.data, args.source, args.out)
    elif args.command == "fixture-pair":
        from .fixtures import run_fixture_pair
        value = run_fixture_pair(args.out)
    else:
        value = handoff(args.data, args.source, args.plan, args.out)
    print(json.dumps(value, indent=2))
    if str(value.get("status", "")).startswith("BLOCKED") or str(value.get("exposure_status", "")).startswith("BLOCKED") or str(value.get("token_shapes_status", "")).startswith("BLOCKED"):
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
