"""CPU-only pre-score join. No model/decoder/solver import and no native grader.

Native M0 imports require independently byte-pinned successful audit receipts.
The audit, rather than this adapter, establishes model/trace/resource closure.
Generation-cap correctness policy and native M1/M2 audit schemas are unregistered.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

VERSION = "synth-pre-score-import-staged-1"
FIXTURE = "CPU_SYNTHETIC_SCORE_IMPORT_FIXTURE"
NATIVE = "PRESERVED_AUDITED_M0_RAW_UNSCORED"
M0_ROLE = "fresh_dense_M0_zero_B_only_no_STATE1_or_study_checkpoint"
AUDIT_PASS = "PASS_SCOPED_SHORT_M0_RAW_COLLECTION_MECHANICS_UNSCORED"
SCORER_FILES = {"evaluation_panel.py", "rs_contract.py", "rs_solver.py", "rs_verifier.py",
                "task_compiler.py", "balanced_corpus.py", "EVALUATION_PANEL_POLICY.json",
                "EVALUATION_PANEL_POLICY_002.json"}
FROZEN_FILES = {
    "panel.json": "495cadc57d97d232acc0f093257b1dc0c4ae29897d17bd319e158e0cc7af9acb",
    "policy/prompts.json": "a7baf928b38ebccd86173340e27d3cae8467ef71e49df71c48785b2e682a2957",
    "prompt_token_inputs_cpu.json": "4664c1770a88a6345c6bfa02cad3c7bdf667c3037f8811364049b503a6b90a4f",
}
STATISTICS_FILES = {
    "dev_statistics.py": "2851ee7f7f2aa2399b7030a05b2bda14e85a8eff98ac23f71e7d918014d3059c",
    "DEV_STATISTICS_POLICY.json": "9db137bfa6c0e0e07ec8161e4127ec0a2ec20bb2c224529face447986e876d11",
}
HERE = Path(__file__).resolve().parent
STAGED = HERE.name == "score_import_work" and HERE.parent.name == "artifacts"
PACKAGE = HERE.parents[1] if STAGED else HERE
_LOADED_TOKEN = object()
_FIXTURE_TOKEN = object()


class _FixtureRegistration(dict):
    """Synthetic factory handle; a scope label or resealed dictionary is insufficient."""
    def __init__(self, value, *, token):
        require(token is _FIXTURE_TOKEN, "fixture registration requires synthetic factory")
        super().__init__(value)
        self._factory_body_sha256 = value["sha256"]


class _LoadedM0Batch(dict):
    """Nonserializable native admission handle; factory and reread are mandatory."""
    def __init__(self, value, anchor, *, token):
        require(token is _LOADED_TOKEN, "native record requires pinned loader factory")
        super().__init__(value)
        self._anchor = deepcopy(anchor)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def _hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError("nonfinite JSON constant")


def read_pinned(path, expected_sha256):
    """Bounded reads of selected JSON/source bytes only; never model bytes."""
    path = Path(path)
    require(_hash(expected_sha256) and path.is_file() and not path.is_symlink(), "pinned regular file required")
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024**2 + 1)
    require(len(raw) <= 16 * 1024**2, "bounded pre-score file")
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "pre-score file byte drift: " + path.name)
    return raw, {"path": str(path.resolve()), "bytes": len(raw), "sha256": expected_sha256}


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_bad_constant)


def _body(value):
    require(type(value) in (dict, _FixtureRegistration) and value.get("sha256") == digest({k: v for k, v in value.items() if k != "sha256"}),
            "canonical body checksum")


def _seal(value):
    return {**value, "sha256": digest(value)}


def _panel(panel, prompts, tokens, refs, source_refs, statistics_refs, scope):
    _body(panel); _body(tokens)
    require(prompts.get("panel_sha256") == panel["sha256"], "prompt/panel body join")
    require(not panel.get("shortages") and len(panel["selected_parents"]) == 96, "whole96 parent population")
    require(set(source_refs) == SCORER_FILES and all(_hash(v["sha256"]) for v in source_refs.values()), "scorer source closure")
    require(set(statistics_refs) == set(STATISTICS_FILES), "statistics source/policy closure")
    parents = panel["selected_parents"]
    if scope == FIXTURE:
        require(all(type(p["parent_id"]) is str and p["parent_id"].startswith("fixture_")
                    and type(p["component_id"]) is str and p["component_id"].startswith("component_") for p in parents),
                "synthetic parent/component namespace required; native identities cannot be rebranded")
        require(all(v.get("path") == "CPU_SYNTHETIC_NOT_NATIVE_AUTHORITY"
                    for group in (refs, source_refs, statistics_refs) for v in group.values()),
                "synthetic source namespace required; native anchors cannot be rebranded")
    require(len({p["parent_id"] for p in parents}) == len({p["component_id"] for p in parents}) == 96,
            "distinct joined parents/components")
    require(Counter(p["skill"] for p in parents) == {"R": 32, "S": 32, "RS": 32}, "32 components per skill")
    require(len(prompts["rows"]) == len(tokens["rows"]) == 384, "whole384 denominator")
    rows = []
    for parent in parents:
        require(_hash(parent["source_hash"]) and type(parent["stratum"]) is str, "family identity")
        for condition in (0, 1):
            for rendering in ("familiar", "heldout_ledger"):
                rows.append({"row_id": digest({"panel": panel["sha256"], "parent": parent["parent_id"],
                    "condition": condition, "rendering": rendering}), "component_id": parent["component_id"],
                    "skill": parent["skill"], "condition": condition, "rendering": rendering,
                    "parent_id": parent["parent_id"], "source_hash": parent["source_hash"], "stratum": parent["stratum"]})
    for metadata, prompt, token in zip(rows, prompts["rows"], tokens["rows"]):
        require(set(prompt) == {"row_id", "messages", "prompt_sha256"} and metadata["row_id"] == prompt["row_id"] == token["row_id"],
                "frozen panel/prompt/token order")
        require(prompt["prompt_sha256"] == digest(prompt["messages"]) == token["prompt_sha256"], "whole visible prompt binding")
        require(set(token) == {"row_id", "prompt_sha256", "prompt_ids", "prompt_tokens", "attention_mask"}
                and type(token["prompt_tokens"]) is int and 0 < token["prompt_tokens"] <= 480
                and len(token["prompt_ids"]) == token["prompt_tokens"]
                and all(type(t) is int and 0 <= t < 151936 for t in token["prompt_ids"])
                and token["attention_mask"] == [1] * token["prompt_tokens"], "untruncated prompt-only tokens")
        metadata.update({"prompt": deepcopy(prompt), "input": deepcopy(token)})
    require(len({r["row_id"] for r in rows}) == 384 and tokens["source_packet_sha256"] == digest(prompts), "full ordered public source packet")
    require(tokens["context_cap"] == 512 and tokens["maximum_generated_tokens"] == 32
            and tokens["rows_dropped"] == tokens["rows_truncated"] == 0 and tokens["overlong_row_ids"] == [], "frozen context/dose")
    metadata = [{k: r[k] for k in ("row_id", "component_id", "skill", "condition", "rendering")} for r in rows]
    value = _seal({"version": VERSION, "scope": scope, "panel_body_sha256": panel["sha256"],
        "rows": rows, "row_order_sha256": digest([r["row_id"] for r in rows]),
        "panel_metadata_sha256": digest(sorted(metadata, key=lambda r: r["row_id"])),
        "panel_files": deepcopy(refs), "scorer_sources": deepcopy(source_refs), "statistics_sources": deepcopy(statistics_refs),
        "scoring_policy_status": "UNREGISTERED_GENERATION_CAP32_CORRECTNESS_TREATMENT",
        "scoring_authorized": False, "native_launch_authorized": False, "confirmation_eligible": False})
    return _FixtureRegistration(value, token=_FIXTURE_TOKEN) if scope == FIXTURE else value


def load_frozen_panel(package):
    """Bind existing compact002 public inputs without reading oracle labels."""
    package = Path(package); directory = package / "evidence/fresh_dev_compact002_20261003"
    documents, refs = {}, {}
    for name, pin in FROZEN_FILES.items():
        raw, refs[name] = read_pinned(directory / name, pin); documents[name] = _json(raw)
    panel = documents["panel.json"]; sources = {}
    for original, pin in panel["generator_and_solver_source_pins"].items():
        name = Path(original).name
        _, sources[name] = read_pinned(directory / "code_snapshot" / name, pin)
    statistics = {}
    for name, pin in STATISTICS_FILES.items():
        _, statistics[name] = read_pinned(package / name, pin)
    return _panel(panel, documents["policy/prompts.json"], documents["prompt_token_inputs_cpu.json"],
                  refs, sources, statistics, NATIVE)


def load_m0_batch(preserved_directory, audit_path, *, expected_audit_sha256, expected_config_sha256):
    """Use audit/config pins selected by a trusted external authority.

    Hash equality authenticates the supplied bytes against those supplied pins;
    this loader does not establish auditor identity or make arbitrary self-signed
    audits authoritative. Independent selection/approval of anchors is required.

    Only consumed JSON and its preserved final-row bytes are reread. The prior
    independent audit remains the authority for model/runtime/resource closure.
    """
    saved = Path(preserved_directory)
    raw, audit_ref = read_pinned(audit_path, expected_audit_sha256); audit = _json(raw); _body(audit)
    require(audit.get("status") == AUDIT_PASS and audit.get("raw_collection_mechanics_verified") is True
            and audit.get("all_exact_row_prompt_token_trace_prefix_finish_decode_and_cost_joins") is True,
            "successful independent raw collection audit required")
    require(audit.get("correctness_grade_performed") is False and audit.get("scoring_authorized") is False
            and audit.get("confirmation_eligible") is False and audit.get("rows") == 64, "unscored64 audit scope")
    require(audit["preserved_path"] == str(saved.resolve()) and audit["config_file_sha256"] == expected_config_sha256,
            "independent audit preserved/config identity")
    closure = audit["hash_closure"]
    require(closure["all_equal"] is True and closure["before"] == closure["after"], "independent stable hash closure")
    raw, config_ref = read_pinned(saved / "config_snapshot.json", expected_config_sha256); config = _json(raw); _body(config)
    require(config["sha256"] == audit["config_body_sha256"] and config["model_role"] == M0_ROLE
            and config["batch_selector"] == audit["batch_selector"], "M0 frozen config/audit scope")
    raw, preservation_ref = read_pinned(saved / "preservation_receipt.json", closure["preservation_receipt_file_sha256"])
    preservation = _json(raw); _body(preservation)
    require(preservation["sha256"] == closure["preservation_receipt_body_sha256"]
            and preservation["source_commit"] == audit["source_commit"], "preservation body/source binding")
    require(digest(preservation["raw_files"]) == closure["after"]["preserved_raw"]["manifest_sha256"], "preserved raw inventory")
    raw, worker_ref = read_pinned(saved / "worker_result.json", preservation["raw_files"]["worker_result.json"]["sha256"])
    worker = _json(raw)
    require(worker["status"] == "COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED" and worker["error"] is None
            and worker["config_sha256"] == expected_config_sha256 and worker["source_commit"] == audit["source_commit"],
            "successful whole worker receipt")
    outputs = worker["collection"]["rows"]; row_refs = []
    require(len(outputs) == 64 and worker["batch_selector"] == config["batch_selector"], "complete audited64 worker")
    for ordinal, output in enumerate(outputs, 1):
        name = "stage_inference_row_%04d.json" % ordinal
        raw, ref = read_pinned(saved / name, preservation["raw_files"][name]["sha256"])
        require(_json(raw) == output, "terminal/final raw row byte join"); row_refs.append(ref)
    manifest = [{"ordinal": i, "row_id": row["row_id"], "input_row_sha256": digest(row),
        "raw_output_sha256": digest(output), "generated_tokens": len(output["raw_generated_ids"]),
        "finish_reason": output["finish_reason"]} for i, (row, output) in enumerate(zip(config["rows"], outputs), 1)]
    require(digest(manifest) == audit["ordered_raw_row_receipts_sha256"], "independent audited row manifest")
    value = {"scope": NATIVE, "arm": "M0", "ordinal": config["batch_selector"]["ordinal"],
        "source_kind": config["batch_selector"]["source_kind"], "inputs": config["rows"], "outputs": outputs,
        "source_packet_sha256": config["source_packet_sha256"], "audit_passed": True,
        "references": {"audit": audit_ref, "config": config_ref, "preservation": preservation_ref,
            "worker": worker_ref, "raw_rows": row_refs, "source_commit": audit["source_commit"],
            "source_pins": config["source_pins"], "model_pins": config["model_pins"],
            "model_role": config["model_role"], "audit_hash_closure": closure}}
    anchor = {"preserved_directory": str(saved.resolve()), "audit_path": str(Path(audit_path).resolve()),
              "expected_audit_sha256": expected_audit_sha256, "expected_config_sha256": expected_config_sha256}
    return _LoadedM0Batch(value, anchor, token=_LOADED_TOKEN)


def prepare_import(registration, batches, *, arm):
    """Require the full roster before producing any grader-facing records."""
    _body(registration)
    require(arm in ("M0", "M1", "M2"), "declared arm")
    scope = registration["scope"]
    require(scope in (NATIVE, FIXTURE), "declared import scope")
    if scope == FIXTURE:
        require(type(registration) is _FixtureRegistration
                and registration["sha256"] == registration._factory_body_sha256,
                "fixture registration requires unmodified synthetic factory identity")
    require(scope == FIXTURE or arm == "M0", "native M1/M2 collector/audit schema unregistered")
    if scope == NATIVE:
        require(registration == load_frozen_panel(PACKAGE), "native registration differs from frozen files/source closure")
    require(type(batches) is list and len(batches) == 6, "whole384 requires all six audited64 batches")
    joined, references, model_pins = [], [], None
    for ordinal, batch in enumerate(batches, 1):
        if scope == NATIVE:
            require(type(batch) is _LoadedM0Batch, "native batch requires independently pinned loader record")
            anchor = batch._anchor
            reread = load_m0_batch(anchor["preserved_directory"], anchor["audit_path"],
                expected_audit_sha256=anchor["expected_audit_sha256"], expected_config_sha256=anchor["expected_config_sha256"])
            require(batch == reread, "native batch differs from independently pinned file joins")
        require(batch["scope"] == scope and batch["arm"] == arm and batch["source_kind"] == "atomic_RS_DEV"
                and batch["ordinal"] == ordinal and batch["audit_passed"] is True, "audited arm/batch/order/scope")
        expected = registration["rows"][(ordinal - 1) * 64:ordinal * 64]
        require(batch["inputs"] == [r["input"] for r in expected] and len(batch["outputs"]) == 64
                and digest(batch["inputs"]) == batch["source_packet_sha256"], "whole frozen batch source/order")
        refs = batch["references"]
        require(all(_hash(refs[k]["sha256"]) for k in ("audit", "config", "preservation", "worker")), "required receipt references")
        require(type(refs["source_commit"]) is str and refs["source_pins"] and refs["model_pins"], "source/model provenance")
        require(model_pins is None or model_pins == refs["model_pins"], "same frozen model assets across batches")
        model_pins = refs["model_pins"]
        for row, output in zip(expected, batch["outputs"]):
            require(output["row_id"] == row["row_id"] and output["prompt_sha256"] == row["input"]["prompt_sha256"]
                    and output["prompt_ids_sha256"] == digest(row["input"]["prompt_ids"])
                    and output["input_row_sha256"] == digest(row["input"])
                    and output["source_packet_sha256"] == batch["source_packet_sha256"], "raw row/prompt/input provenance")
            require(output["status"] == "RAW_RESPONSE_COLLECTED_UNSCORED" and output["capability_correctness_scored"] is False
                    and output["oracle_inputs_read"] is False and output["prompt_truncated"] is False,
                    "unscored complete raw row; no qualification/partial substitution")
            require(output.get("scope") == (FIXTURE if scope == FIXTURE else "UNQUALIFIED_NATIVE_IMPLEMENTATION"),
                    "explicit synthetic/native raw scope; fixtures cannot import native captures")
            ids = output["raw_generated_ids"]
            require(type(ids) is list and 1 <= len(ids) <= 32 and all(type(t) is int and 0 <= t < 151936 for t in ids)
                    and not any(t in (151643, 151645) for t in ids[:-1]), "first EOS/typed generated IDs")
            eos = ids[-1] in (151643, 151645)
            require(output["finish_reason"] == ("eos" if eos else "generation_cap") and output["truncated"] is (not eos)
                    and output["eos_token_id"] == (ids[-1] if eos else None) and (eos or len(ids) == 32), "preserved EOS/cap/truncation semantics")
            require(output["context_cap"] == 512 and output["maximum_generated_tokens"] == 32
                    and type(output["decoded_response"]) is str, "frozen generation/decode metadata")
            joined.append({**deepcopy(row), "arm": arm, "batch_ordinal": ordinal, "raw": deepcopy(output)})
        references.append(deepcopy(refs))
    require(len(joined) == 384 and digest([r["row_id"] for r in joined]) == registration["row_order_sha256"], "complete384 frozen order")
    return _seal({"version": VERSION, "scope": scope, "arm": arm, "rows": joined,
        "registration_sha256": registration["sha256"], "panel_metadata_sha256": registration["panel_metadata_sha256"],
        "row_order_sha256": registration["row_order_sha256"], "scorer_sources": registration["scorer_sources"],
        "statistics_sources": registration["statistics_sources"], "batch_references": references,
        "status": "PRE_SCORE_JOIN_COMPLETE_POLICY_GATE_CLOSED", "scoring_authorized": False,
        "native_launch_authorized": False, "confirmation_eligible": False, "retention_certified": False})


def score_cpu_fixture(registration, batches, *, arm, grader):
    """Injected callback control only. This function cannot grade native imports."""
    packet = prepare_import(registration, batches, arm=arm)
    require(packet["scope"] == FIXTURE, "generation_cap32 correctness policy unregistered; native grader unavailable")
    scores = {}
    for row in packet["rows"]:
        # Pass the entire decoded response and finish metadata, never last-JSON extraction.
        value = grader(deepcopy(row["prompt"]["messages"]), row["raw"]["decoded_response"], deepcopy(row["raw"]))
        require(type(value) is bool, "strict binary fixture callback")
        scores[row["row_id"]] = value
    return {"scope": FIXTURE, "scores": scores, "scoring_authorized": False, "quality_gain_established": False}
