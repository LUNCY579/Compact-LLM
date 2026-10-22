"""Offline producer attachment v1; never load, train, evaluate, or authorize a model.

The contract is a JSON object with schema ``gd001_attachment_v1``, evidence_kind
``producer_metadata`` (or ``mechanical_fixture``), trajectory_id, exact
expected_base_parameters, producer {owner, study_id, independently_approved:true,
artifact_role:"original_training_trajectory"}, approval_plan and completion_receipt
file references {path, sha256}, base/tokenizer {path, content_sha256}, 2-19 ordered
checkpoints {step, kind:"hf"|"peft", path, content_sha256, snapshot_receipt:{path,
sha256}}, and provenance {events:{path,sha256}, training_manifest:{path,sha256}}.
Every path is absolute and retained. The last checkpoint is the completed final.

The prospective owner plan has status APPROVED_PROSPECTIVE, purpose
research_training, approved_at_utc, retention all_original_snapshots_and_provenance,
and identical trajectory_id/producer/expected_base_parameters/base/tokenizer plus
checkpoint_schedule [{step,kind,path}]. The completed producer receipt repeats these
bindings, status COMPLETED, purpose research_training, training_started_at_utc,
completed_at_utc, final_step, checkpoints (the exact full contract checkpoint list),
final_checkpoint (the last checkpoint including content hash and snapshot receipt),
provenance references, source_count, rejected_count,
consumed_count, unused_accepted_count, and consumed_per_step {"1":count,...,
"final_step":count}. Dates are
timezone-aware ISO 8601. These are audited owner declarations, not signatures.

Events are JSONL, with globally unique event_id and event_type source/filter/consume/unused:
* source: parent_id, source_row_id, source_row_number (one-based JSONL record ordinal),
  source_ref {path,sha256}, raw_record (the exact original JSON object). Every row of
  each retained source JSONL must appear once, including subsequently rejected rows.
* filter: source_event_id, decision accepted/rejected, reason, filtering_history
  (nonempty objects with operation, decision keep/reject, reason; last decision agrees).
* consume: source_event_id, filter_event_id, matching parent_id/source_row_id, step,
  prompt, raw_rendered_prompt, consumed_prompt, input_ids, tokenizer_content_sha256,
  rendering_history (nonempty objects with operation), and truncation {policy:
  none/left/right, original_input_ids, max_tokens:null|int, history:[]|[truncate record]}.
  A truncate record is exactly {operation:"truncate",side,removed_tokens,max_tokens}.
  Nontruncated text and IDs must agree; truncated IDs must be the declared token slice.
* unused: source_event_id, filter_event_id, reason; an accepted source not consumed
  before completion needs this explicit sampling/selection disposition. Accepted rows
  must be consumed or unused, never both; rejected rows must not be consumed/unused.
  Consumption steps are ordered
  and cover every optimizer step 1..final_step. Repeated occurrences remain separate.
The overlap manifest must be exactly [{prompt:consumed_prompt,parent_id},...] in
consumption-event order, preserving repeats. No tokenizer execution is performed;
prompt-to-token parity, actual parameter count, native loading/scoring and efficacy
remain unqualified. Reconciled counts cannot prove an upstream owner omitted no data.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .common import Blocked, digest, file_hash, require, signed, verify
from .protected_paths import reject_protected_path
from .registry import freeze_registry, snapshot

SCHEMA = "gd001_attachment_v1"
_HEX = re.compile(r"[0-9a-f]{64}")


def _exact(actual, expected) -> bool:
    """JSON identity distinguishes booleans, integers, and floating point values."""
    return digest(actual) == digest(expected)


def _text(value, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"Missing {label}")
    return value


def _sha(value, label: str) -> str:
    require(isinstance(value, str) and _HEX.fullmatch(value) is not None,
            f"Invalid {label} SHA256")
    return value


def _path(value, label: str) -> Path:
    candidate = Path(_text(value, label)).expanduser()
    require(candidate.is_absolute(), f"{label} needs an absolute stable path")
    require(not candidate.is_symlink(), f"Symlink {label} is not retained original payload")
    return reject_protected_path(candidate, label)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _json(text: str):
    def reject(value):
        raise Blocked(f"Non-finite JSON number: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject)


def _object(path: Path) -> dict:
    value = _json(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _ref(value: dict, label: str) -> dict:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"},
            f"{label} needs a retained path and SHA256")
    path = _path(value["path"], label)
    require(path.is_file(), f"Missing retained {label} payload: {path}")
    require(file_hash(path) == _sha(value["sha256"], label), f"{label} payload changed")
    return {"path": str(path), "sha256": value["sha256"]}


def _rows(path: Path) -> list[dict]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = _json(line)
                require(isinstance(item, dict), f"Expected JSONL object: {path}")
                result.append(item)
    require(bool(result), f"Empty retained JSONL payload: {path}")
    return result


def _date(value, label: str) -> datetime:
    stamp = datetime.fromisoformat(_text(value, label).replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp.utcoffset() is not None,
            f"{label} must include timezone")
    return stamp


def _tokens(value, label: str) -> list[int]:
    require(isinstance(value, list) and bool(value)
            and all(type(t) is int and t >= 0 for t in value), f"Invalid {label}")
    return value


def _asset(value: dict, kind: str) -> dict:
    require(isinstance(value, dict) and set(value) == {"path", "content_sha256"},
            f"{kind} needs exact path/content_sha256")
    path = _path(value["path"], kind)
    asset = snapshot(str(path), kind)
    require(asset["content_sha256"] == _sha(value["content_sha256"], kind),
            f"{kind} content identity differs")
    return asset


def _provenance(events: list[dict], manifest: list[dict], tokenizer_hash: str,
                completion: dict) -> dict:
    sources, filters, ids, payloads, source_rows = {}, {}, set(), {}, set()
    occurrences, counts, used_sources, unused = [], Counter(), Counter(), set()
    last_step = 0
    for event in events:
        event_id = _text(event["event_id"], "event ID")
        require(event_id not in ids, "Duplicate provenance event ID")
        ids.add(event_id)
        kind = event["event_type"]
        if kind == "source":
            for field in ("parent_id", "source_row_id"):
                _text(event[field], field)
            declared = event["source_ref"]
            require(isinstance(declared, dict) and set(declared) == {"path", "sha256"},
                    "Source inventory needs retained path and SHA256")
            path = str(_path(declared["path"], "source inventory"))
            ref = {"path": path, "sha256": _sha(declared["sha256"], "source inventory")}
            if path not in payloads:
                ref = _ref(declared, "source inventory")
                payloads[path] = {"ref": ref, "rows": _rows(Path(path))}
            require(_exact(payloads[path]["ref"], ref), "Conflicting source inventory identity")
            ordinal = event["source_row_number"]
            require(type(ordinal) is int and 1 <= ordinal <= len(payloads[path]["rows"]),
                    "Invalid source row number")
            require((path, ordinal) not in source_rows, "Source row represented twice")
            require(isinstance(event["raw_record"], dict)
                    and _exact(event["raw_record"], payloads[path]["rows"][ordinal - 1]),
                    "Original source payload was not retained exactly")
            source_rows.add((path, ordinal))
            sources[event_id] = event
        elif kind == "filter":
            source_id = event["source_event_id"]
            require(source_id in sources and source_id not in filters,
                    "Filter needs one prior, previously unfiltered source")
            require(event["decision"] in {"accepted", "rejected"}, "Missing filter decision")
            _text(event["reason"], "filter reason")
            history = event["filtering_history"]
            require(isinstance(history, list) and bool(history), "Missing actual filtering history")
            for entry in history:
                require(isinstance(entry, dict), "Invalid filter history")
                _text(entry["operation"], "filter operation")
                _text(entry["reason"], "filter operation reason")
                require(entry["decision"] in {"keep", "reject"}, "Invalid filter history decision")
            require(history[-1]["decision"] == ("keep" if event["decision"] == "accepted" else "reject"),
                    "Filter history disagrees with disposition")
            filters[source_id] = event
        elif kind == "unused":
            source_id = event["source_event_id"]
            require(source_id in filters and source_id not in unused and not used_sources[source_id],
                    "Unused disposition needs one previously unconsumed filtered source")
            require(filters[source_id]["event_id"] == event["filter_event_id"]
                    and filters[source_id]["decision"] == "accepted", "Rejected/foreign filter unused")
            _text(event["reason"], "unused accepted source reason")
            unused.add(source_id)
        elif kind == "consume":
            source_id = event["source_event_id"]
            require(source_id in filters and source_id not in unused,
                    "Consumed occurrence needs prior source/filter history and cannot be unused")
            source, filtered = sources[source_id], filters[source_id]
            require(filtered["event_id"] == event["filter_event_id"]
                    and filtered["decision"] == "accepted", "Rejected/foreign filter consumed")
            require(all(event[k] == source[k] for k in ("parent_id", "source_row_id")),
                    "Consumed occurrence parent/source differs")
            step = event["step"]
            require(type(step) is int and 1 <= step <= completion["final_step"] and step >= last_step,
                    "Consumption steps missing, unordered, or outside completed training")
            last_step = step
            for field in ("prompt", "raw_rendered_prompt", "consumed_prompt"):
                _text(event[field], field)
            require(event["tokenizer_content_sha256"] == tokenizer_hash, "Consumed tokenizer differs")
            rendering = event["rendering_history"]
            require(isinstance(rendering, list) and bool(rendering), "Missing actual rendering history")
            for entry in rendering:
                require(isinstance(entry, dict), "Invalid rendering history")
                _text(entry["operation"], "render operation")
            tokens = _tokens(event["input_ids"], "consumed input_ids")
            truncation = event["truncation"]
            require(isinstance(truncation, dict)
                    and set(truncation) == {"policy", "original_input_ids", "max_tokens", "history"},
                    "Missing actual truncation history")
            original = _tokens(truncation["original_input_ids"], "original input_ids")
            side, limit = truncation["policy"], truncation["max_tokens"]
            if side == "none":
                require(limit is None and truncation["history"] == [] and tokens == original
                        and event["raw_rendered_prompt"] == event["consumed_prompt"],
                        "Silent truncation or inconsistent no-truncation history")
            else:
                require(side in {"left", "right"} and type(limit) is int and 0 < limit < len(original),
                        "Invalid actual truncation policy")
                expected = original[-limit:] if side == "left" else original[:limit]
                require(_exact(tokens, expected) and _exact(truncation["history"], [{"operation": "truncate",
                        "side": side, "removed_tokens": len(original) - limit, "max_tokens": limit}]),
                        "Truncation history does not match consumed tokens")
            occurrences.append({"prompt": event["consumed_prompt"], "parent_id": event["parent_id"]})
            counts[step] += 1
            used_sources[source_id] += 1
        else:
            raise Blocked("Unknown provenance event type")
    expected_rows = {(path, n) for path, item in payloads.items()
                     for n in range(1, len(item["rows"]) + 1)}
    require(source_rows == expected_rows, "Source inventory rows omitted, including rejected inputs")
    require(set(filters) == set(sources), "Source filtering disposition omitted")
    require(all((filtered["decision"] == "accepted") == (bool(used_sources[source_id]) or source_id in unused)
                for source_id, filtered in filters.items()), "Accepted/rejected consumption omitted")
    require(bool(occurrences) and _exact(manifest, occurrences),
            "Training overlap manifest must exactly project consumed occurrences, including repeats")
    require(len(counts) == completion["final_step"]
            and all(step == index for index, step in enumerate(sorted(counts), 1)),
            "Incomplete optimizer-step consumption coverage")
    observed = {"source_count": len(sources), "rejected_count": sum(
                f["decision"] == "rejected" for f in filters.values()),
                "consumed_count": len(occurrences), "unused_accepted_count": len(unused)}
    for field, count in observed.items():
        require(type(completion[field]) is int and completion[field] == count,
                f"Completed training {field} differs from retained provenance")
    per_step = {str(k): counts[k] for k in sorted(counts)}
    declared_counts = completion["consumed_per_step"]
    require(isinstance(declared_counts, dict)
            and all(type(value) is int and value > 0 for value in declared_counts.values())
            and _exact(declared_counts, per_step), "Completed per-step consumption differs")
    return {**observed, "consumed_per_step": per_step,
            "source_payloads": [v["ref"] for v in payloads.values()]}


def _validate(contract_path: str) -> dict:
    path = _path(contract_path, "attachment contract")
    require(path.is_file(), "Missing attachment contract")
    contract_hash = file_hash(path)
    contract = _object(path)
    require(contract["schema"] == SCHEMA, "Unsupported attachment schema")
    require(contract["evidence_kind"] in {"producer_metadata", "mechanical_fixture"},
            "Unknown attachment evidence kind")
    trajectory = _text(contract["trajectory_id"], "trajectory ID")
    producer = contract["producer"]
    require(isinstance(producer, dict) and set(producer) == {
        "owner", "study_id", "independently_approved", "artifact_role"}, "Incomplete producer identity")
    _text(producer["owner"], "producer owner")
    _text(producer["study_id"], "producer study ID")
    require(producer["independently_approved"] is True
            and producer["artifact_role"] == "original_training_trajectory",
            "Only independently approved original research trajectories qualify")
    count = contract["expected_base_parameters"]
    require(type(count) is int and 500_000_000 <= count <= 700_000_000,
            "Exact source must be 500M-700M parameters")
    base = _asset(contract["base"], "hf")
    tokenizer = _asset(contract["tokenizer"], "tokenizer")
    checkpoints = contract["checkpoints"]
    require(isinstance(checkpoints, list) and 2 <= len(checkpoints) <= 19,
            "Attachment needs 2-19 original saved snapshots")
    schedule, entries, assets, snapshot_refs, seen_paths, last_step = [], [], [], [], {Path(base["path"])}, 0
    for checkpoint in checkpoints:
        step, kind = checkpoint["step"], checkpoint["kind"]
        require(type(step) is int and step > last_step and kind in {"hf", "peft"},
                "Saved checkpoint schedule must strictly increase with original HF/PEFT exports")
        last_step = step
        asset_path = _path(checkpoint["path"], "saved checkpoint")
        require(asset_path not in seen_paths, "Snapshot directory reused or aliases original base")
        seen_paths.add(asset_path)
        asset = snapshot(str(asset_path), kind)
        require(asset["content_sha256"] == _sha(checkpoint["content_sha256"], "snapshot"),
                "Saved checkpoint content differs")
        ref = _ref(checkpoint["snapshot_receipt"], "post-save snapshot receipt")
        receipt = verify(_object(Path(ref["path"])))
        entry = {"id": f"step_{step:08d}", "step": step, "kind": kind, "path": str(asset_path)}
        require(_exact(receipt.get("entry"), entry) and _exact(receipt.get("asset"), asset)
                and receipt.get("evaluated") is False,
                "Post-save receipt must match exact step, stable path, kind and retained payload")
        schedule.append({"step": step, "kind": kind, "path": str(asset_path)})
        entries.append(entry)
        assets.append(asset)
        snapshot_refs.append(ref)
    approval_ref = _ref(contract["approval_plan"], "owner prospective approval plan")
    completed_ref = _ref(contract["completion_receipt"], "training completion receipt")
    approval, completion = _object(Path(approval_ref["path"])), _object(Path(completed_ref["path"]))
    bindings = {"trajectory_id": trajectory, "producer": producer,
                "expected_base_parameters": count, "base": contract["base"],
                "tokenizer": contract["tokenizer"], "checkpoint_schedule": schedule}
    require(all(_exact(approval.get(k), v) and _exact(completion.get(k), v) for k, v in bindings.items()),
            "Owner plan/completed training identity or preregistered schedule differs")
    require(approval.get("status") == "APPROVED_PROSPECTIVE"
            and approval.get("purpose") == "research_training"
            and approval.get("retention") == "all_original_snapshots_and_provenance",
            "Missing authorized prospective research plan and payload retention policy")
    require(completion.get("status") == "COMPLETED" and completion.get("purpose") == "research_training"
            and type(completion.get("final_step")) is int and completion["final_step"] == last_step,
            "Incomplete training or discarded qualification updates are not a trajectory")
    require(_exact(completion.get("checkpoints"), checkpoints)
            and _exact(completion.get("final_checkpoint"), checkpoints[-1]),
            "Completed training must bind exact saved payloads and final checkpoint receipt")
    require(_date(approval["approved_at_utc"], "approval time") <=
            _date(completion["training_started_at_utc"], "training start") <=
            _date(completion["completed_at_utc"], "completion time"), "Approval was not prospective")
    provenance = contract["provenance"]
    require(isinstance(provenance, dict) and set(provenance) == {"events", "training_manifest"},
            "Complete consumed-prompt provenance and overlap projection required")
    event_ref = _ref(provenance["events"], "consumed/rendered prompt events")
    manifest_ref = _ref(provenance["training_manifest"], "training manifest")
    require(_exact(completion.get("provenance"), provenance), "Completed provenance identities differ")
    inventory = _provenance(_rows(Path(event_ref["path"])), _rows(Path(manifest_ref["path"])),
                            tokenizer["content_sha256"], completion)
    spec = {"trajectory_id": trajectory, "expected_base_parameters": count,
            "base_model": base["path"], "tokenizer": tokenizer["path"],
            "training_manifest": manifest_ref["path"],
            "checkpoints": [{"id": "reference", "step": 0, "kind": "base"}] + entries}
    registry = freeze_registry(spec)
    require(_exact(registry["base"], base) and _exact(registry["tokenizer"], tokenizer)
            and _exact([c["asset"] for c in registry["checkpoints"][1:]], assets),
            "Original assets changed during attachment validation")
    for reference in [approval_ref, completed_ref, event_ref, manifest_ref,
                      *snapshot_refs, *inventory["source_payloads"]]:
        require(file_hash(Path(reference["path"])) == reference["sha256"],
                "Retained receipt/provenance changed during attachment validation")
    require(file_hash(path) == contract_hash, "Contract changed during attachment validation")
    return signed({"schema": SCHEMA, "status": "ATTACHMENT_METADATA_VALIDATED",
                   "trajectory_id": trajectory, "expected_base_parameters": count,
                   "evidence_kind": contract["evidence_kind"],
                   "attachment": {"path": str(path), "sha256": contract_hash},
                   "producer": producer, "approval_plan": approval_ref,
                   "completion_receipt": completed_ref, "checkpoint_schedule": schedule,
                   "provenance": {"events": event_ref, "training_manifest": manifest_ref, **inventory},
                   "registry": registry, "native_loading_qualified": False,
                   "export_parity_qualified": False, "checkpoint_selection_efficacy_established": False,
                   "authorizes_training": False, "authorizes_native_run": False,
                   "authorizes_confirmation": False, "permits_model_promotion": False})


def validate_attachment(contract_path: str) -> dict:
    """Read retained producer artifacts and return an explicitly metadata-only receipt."""
    try:
        return _validate(contract_path)
    except (KeyError, TypeError, OSError, json.JSONDecodeError, OverflowError) as exc:
        raise Blocked(f"Invalid/incomplete producer attachment: {exc}") from exc
    except ValueError as exc:
        if isinstance(exc, Blocked):
            raise
        raise Blocked(f"Invalid producer attachment value: {exc}") from exc


def native_spec_from_attachment(contract_path: str, runtime: dict) -> dict:
    """Prepare a reviewed native specification; grants no execution authorization.

    Keep the returned attachment reference and validation receipt beside registration:
    pipeline.initialize currently hashes the registry, not extra specification keys.
    Revalidate the attachment immediately before registration and qualification.
    """
    from .pipeline import DEFAULT_POLICY, validate_config
    validated = validate_attachment(contract_path)
    require(validated["evidence_kind"] == "producer_metadata",
            "Mechanical attachment fixtures cannot become native registrations")
    require(isinstance(runtime, dict), "Explicit complete native runtime required")
    validate_config(DEFAULT_POLICY, runtime)
    registry = validated["registry"]
    return {"trajectory_id": registry["trajectory_id"],
            "expected_base_parameters": registry["expected_base_parameters"],
            "base_model": registry["base"]["path"], "tokenizer": registry["tokenizer"]["path"],
            "training_manifest": registry["training"]["path"],
            "checkpoints": [{"id": c["id"], "step": c["step"], "kind": c["kind"],
                             **({} if c["kind"] == "base" else {"path": c["asset"]["path"]})}
                            for c in registry["checkpoints"]],
            "runtime": dict(runtime), "attachment": validated["attachment"]}
