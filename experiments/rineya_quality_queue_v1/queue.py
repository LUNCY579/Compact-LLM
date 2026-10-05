"""Build and verify an offline evidence queue without loading models or banks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .contracts import REQUIRED_GATES, preservation_decision, validate_metric

ROOT = Path(__file__).resolve().parents[2]
BUSINESS = ROOT.parent
IDS = ("ce8", "pruning", "matched_mlp", "data_sft", "rsd")
CLASSES = {"historical", "native", "fixture", "exploratory", "confirmed", "preparation"}
ROLES = {"receipt", "source", "protocol", "proposal"}
NO_ACTIONS = {"native_execution_authorized": False, "protected_bank_access_authorized": False,
              "spending_authorized": False, "promotion_permitted": False, "release_permitted": False,
              "serving_change_authorized": False}
PROTECTED = {"sealed", "sealed_banks", "sealed-bank", "sealed_bank", "confirmation_banks",
             "confirmation-bank", "confirmation_bank", "protected", "check", "final"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)


def pointer(value, path):
    require(type(path) is str and path.startswith("/"), "absolute JSON pointer required")
    for part in path[1:].split("/"):
        require(re.search(r"~(?![01])", part) is None, "invalid JSON pointer escape")
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            require(re.fullmatch(r"0|[1-9][0-9]*", key) is not None, "array pointer index")
            value = value[int(key)]
        else:
            require(type(value) is dict and key in value, "missing JSON pointer: " + path)
            value = value[key]
    return value


def source_path(path, business=BUSINESS):
    requested = Path(path).absolute()
    resolved, business = requested.resolve(), Path(business).resolve()
    worktrees = business.parent / ".codex/worktrees"
    require(resolved.is_relative_to(business) or resolved.is_relative_to(worktrees),
            "evidence outside known BusinessProject/source worktrees")
    require(not any(part.casefold() in PROTECTED for part in (*requested.parts, *resolved.parts)),
            "protected bank path refused before reading")
    return requested, resolved


def read_evidence(record, business=BUSINESS):
    require(record.get("classification") in CLASSES and record.get("role") in ROLES,
            "invalid evidence classification/role")
    require(record.get("format") in ("json", "text"), "metadata JSON/text only")
    requested, path = source_path(record["path"], business)
    require(path.suffix.casefold() in (".json", ".md", ".txt", ".py", ".ps1", ".yml", ".yaml"),
            "weights, token arrays, raw banks and binary artifacts cannot be queue inputs")
    require(path.stem.casefold() not in {"check", "final", "sealed", "raw_responses", "responses", "catalog"},
            "bank/output payload refused before reading")
    require(path.is_file() and path.stat().st_size <= 8 * 1024**2, "missing/oversized evidence")
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    value = strict_json(raw) if record["format"] == "json" else None
    require(value is None or type(value) is dict, "receipt must be object metadata, never a task array")
    binding = dict(record, path=str(path), requested_path=str(requested), sha256=sha_bytes(raw), bytes=len(raw))
    return {"binding": binding, "raw": raw, "text": text, "json": value}


def check_reference(item, snapshots):
    ids = item.get("evidence_ids", [])
    if "evidence_id" in item:
        ids = [*ids, item["evidence_id"]]
    require(bool(ids) and all(key in snapshots for key in ids), "unknown/missing source reference")
    if item.get("text_anchor"):
        require(any(item["text_anchor"].replace("\r\n", "\n") in snapshots[key]["text"] for key in ids),
                "text assertion does not occur in bound evidence")
    if item.get("json_pointer"):
        pointer(snapshots[item["evidence_id"]]["json"], item["json_pointer"])


def same_number(actual, expected):
    return type(actual) in (int, float) and type(expected) in (int, float) and actual == expected


def has_number(text, value):
    numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)", text)
    return any(float(number) == value for number in numbers)


def bind_metric(metric, snapshots):
    classifications = {key: snap["binding"]["classification"] for key, snap in snapshots.items()}
    validate_metric(metric, classifications)
    result = dict(metric)
    if metric["status"] == "NOT_MEASURED":
        if metric.get("evidence_id") or metric.get("evidence_ids"):
            check_reference(metric, snapshots)
        result["source_classification"] = "NOT_MEASURED"
        return result
    snap = snapshots[metric["evidence_id"]]
    if metric.get("json_pointer"):
        require(same_number(pointer(snap["json"], metric["json_pointer"]), metric["numerator"]),
                "metric numerator differs from source pointer")
    if metric.get("text_anchor"):
        require(metric["text_anchor"].replace("\r\n", "\n") in snap["text"], "metric source anchor missing")
        require(has_number(metric["text_anchor"], metric["numerator"]),
                "metric numerator absent from source anchor")
    denominator_id = metric.get("denominator_evidence_id", metric["evidence_id"])
    denominator_snap = snapshots[denominator_id]
    if metric.get("denominator_json_pointer"):
        require(same_number(pointer(denominator_snap["json"], metric["denominator_json_pointer"]), metric["denominator"]),
                "metric denominator differs from source pointer")
    elif metric.get("denominator_text_anchor"):
        require(metric["denominator_text_anchor"].replace("\r\n", "\n") in denominator_snap["text"],
                "metric denominator anchor missing")
        require(has_number(metric["denominator_text_anchor"], metric["denominator"]),
                "metric denominator absent from source anchor")
    elif metric.get("text_anchor"):
        require(has_number(metric["text_anchor"], metric["denominator"]), "denominator needs source support")
    else:
        require(metric.get("denominator") == 1 and metric.get("aggregation") == "source_mean",
                "denominator needs a bound pointer/anchor; source means must be explicit")
    result["source_classification"] = snap["binding"]["classification"]
    result["assertion_verification"] = "JSON_VALUE" if metric.get("json_pointer") else "BOUND_TEXT_EXCERPT"
    result["interpretation"] = "Reported source result; not a new model run or candidate qualification."
    return result


def git_state(checkout, expected_branch):
    path = Path(checkout)
    require(path.is_dir(), "missing source checkout/directory")
    if expected_branch is None:
        return {"path": str(path.resolve()), "status": "DIRECTORY_METADATA_NOT_GIT", "branch": None, "head": None}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    actual = git("branch", "--show-current")
    require(actual == expected_branch, "source branch mismatch: " + actual)
    return {"path": str(path.resolve()), "status": "SOURCE_SNAPSHOT", "branch": actual,
            "head": git("rev-parse", "HEAD"), "local_changes": git("status", "--porcelain", "--untracked-files=all")}


def build_card(spec, *, business=BUSINESS, inspect_git=True, supplied_snapshots=None):
    require(type(spec) is dict and spec.get("id") in IDS, "unknown quality candidate")
    require(type(spec.get("priority")) is int and spec["priority"] == IDS.index(spec["id"]) + 1,
            "fixed programme priorities required")
    owner = spec.get("owner", {})
    require(all(type(owner.get(key)) is str and owner[key].strip() for key in ("thread_id", "title", "host_id")),
            "existing owner identity required")
    require(spec.get("advancement") is False, "offline queue cannot advance candidates")
    snapshots = {}
    for record in spec["evidence"]:
        key = record["id"]
        require(type(key) is str and re.fullmatch(r"[a-z][a-z0-9_]*", key), "safe evidence identity required")
        require(key not in snapshots, "duplicate evidence identity")
        if supplied_snapshots is None:
            snapshots[key] = read_evidence(record, business)
        else:
            require(key in supplied_snapshots, "missing bound evidence capsule")
            snapshots[key] = supplied_snapshots[key]
            bound = snapshots[key]["binding"]
            require(all(bound.get(field) == record.get(field) for field in ("id", "classification", "role", "format")),
                    "evidence classification/role differs from specification")
            require(bound.get("requested_path") == str(Path(record["path"]).absolute()), "evidence request path differs from specification")
    if supplied_snapshots is not None:
        require(set(supplied_snapshots) == set(snapshots), "evidence capsule inventory differs from specification")
    require(bool(snapshots), "source evidence required")
    metrics = [bind_metric(metric, snapshots) for metric in spec["metrics"]]
    require(len({metric["name"] for metric in metrics}) == len(metrics), "duplicate metric identity")
    for asset in spec.get("source_assets", []):
        check_reference(asset, snapshots)
        require(type(asset.get("sha256")) is str and re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]), "invalid recorded asset hash")
        require(pointer(snapshots[asset["evidence_id"]]["json"], asset["json_pointer"]) == asset["sha256"],
                "asset hash differs from custody metadata")
    requirements = spec["requirements"]
    require(len({item["id"] for item in requirements}) == len(requirements), "duplicate requirement identity")
    for item in requirements:
        require(item["status"] in ("PASS", "BLOCKED", "NOT_MEASURED"), "requirement status")
        check_reference(item, snapshots)
    for item in spec.get("observed_patterns", []):
        check_reference(item, snapshots)
    for item in spec.get("quality_gates", []):
        check_reference(item, snapshots)
    for item in spec.get("completed_failed_attempts", []):
        check_reference(item, snapshots)
    proposal = spec["next_proposal"]
    require(proposal["proposal_evidence_id"] in snapshots and proposal.get("native_execution_authorized") is False,
            "source-bound proposal, never native authority")
    require(bool(proposal.get("limits")), "bounded next proposal required")
    requirement_map = {item["id"]: item for item in requirements}
    require(bool(proposal.get("requires")) and all(key in requirement_map for key in proposal["requires"]),
            "proposal needs declared requirements")
    for name, value in spec["candidate"].items():
        if name.endswith("sha256"):
            require(value is None or value == "NOT_MEASURED" or (type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)),
                    "invalid candidate hash")
    gates = [{"id": key, "status": "NOT_MEASURED", "reason": "Current proposed candidate lacks qualified balanced outcome evidence."}
             for key in REQUIRED_GATES]
    gates += [{"id": "registered:" + item["id"], "status": "NOT_MEASURED",
               "reason": "Registered rule preserved; current candidate outcome unmeasured."}
              for item in spec.get("quality_gates", [])]
    state = git_state(spec["checkout"], spec.get("expected_branch")) if inspect_git else {"status": "FIXTURE_ONLY"}
    card = dict(spec, schema=1, status="OFFLINE_SOURCE_BOUND_CARD", source_checkout=state,
                evidence=[snap["binding"] for snap in snapshots.values()], metrics=metrics,
                candidate_outcome_status="NOT_MEASURED", runtime={"status": "NOT_MEASURED",
                   "registered_recipe_reference": proposal["proposal_evidence_id"], "native_runtime_qualified": False},
                data_ancestry={"status": "NOT_MEASURED", "review_required": True,
                   "limits": "Metadata cards do not establish family/exclusion/reference closure."},
                preservation_gates=gates, decision=preservation_decision(gates), **NO_ACTIONS)
    card["next_proposal"] = dict(proposal, ready_for_native_authorization=False,
        unresolved_requirements=[key for key in proposal["requires"] if requirement_map[key]["status"] != "PASS"], **NO_ACTIONS)
    card["next_proposal"]["limits_status"] = "BOUND_PROPOSAL_METADATA_NOT_NATIVE_ADMISSION_OR_MEASURED_COST"
    card["limitations"] = ["Source hashes and excerpts verify reported metadata, not its truth or a new model execution.",
        "Prose carrying only evidence IDs is indexed interpretation; it is not mechanically or semantically certified.",
        "Historical outcomes do not fill current candidate gates; fixture results are software evidence only.",
        "Asset hashes are imported from custody receipts; this tool never reads weight payloads.",
        "Different source checkpoints, banks and decoding conditions cannot be pooled into one quality score."]
    return card, snapshots


def external_output(path, business=BUSINESS):
    path, business = Path(path).resolve(), Path(business).resolve()
    require(path.is_relative_to(business) and ".codex" not in {part.casefold() for part in path.parts},
            "output must be under BusinessProject outside .codex")
    require(not any((ancestor / ".git").exists() for ancestor in (path, *path.parents)),
            "output must be outside every tracked checkout")
    require(not path.exists(), "output exists; preserve it and choose a new packet")
    return path


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_packet(output, spec_dir=None):
    output = external_output(output)
    spec_dir = Path(spec_dir) if spec_dir else Path(__file__).parent / "specs"
    cards, snapshots, specs, spec_bytes = [], {}, [], {}
    for key in IDS:
        path = spec_dir / (key + ".json")
        raw = path.read_bytes()
        spec = strict_json(raw)
        require(spec["id"] == key, "spec filename/identity mismatch")
        card, evidence = build_card(spec)
        specs.append({"id": key, "path": str(path.resolve()), "sha256": sha_bytes(raw), "capsule": "specs/" + key + ".json"})
        spec_bytes[key] = raw
        cards.append(card)
        snapshots[key] = evidence
    package = Path(__file__).parent
    code = {str(path.relative_to(ROOT).as_posix()): sha_file(path)
            for path in sorted(package.rglob("*")) if path.is_file()
            and path.suffix in (".py", ".json", ".md") and "__pycache__" not in path.parts}
    # Check all owned files before creating a packet, including concurrent owner edits.
    for values in snapshots.values():
        for snap in values.values():
            require(sha_file(snap["binding"]["path"]) == snap["binding"]["sha256"], "evidence changed during snapshot")
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    (output / "specs").mkdir()
    for key, raw in spec_bytes.items():
        with (output / "specs" / (key + ".json")).open("xb") as stream:
            stream.write(raw)
    for name, expected_hash in code.items():
        require(sha_file(ROOT / name) == expected_hash, "software source changed during snapshot")
        target = output / "software" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
    for card in cards:
        key = card["id"]
        for evidence_id, snap in snapshots[key].items():
            name = "evidence/" + key + "/" + evidence_id + Path(snap["binding"]["path"]).suffix
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(snap["raw"])
            snap["binding"]["capsule"] = name
        write_new(output / (key + ".quality-card.json"), card)
    queue = {"schema": 1, "status": "OFFLINE_QUEUE_ALL_CANDIDATES_HOLD",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_policy": "retention-first; every source-specific registered preservation gate mandatory",
        "compute": "current local 8GB GPU; owner-only serial native work after separate grants",
        "first_behavioral_study": "pruning versus scalar attenuation at matched acquisition, conditional on one recovered donor",
        "ordered_candidates": [{"id": c["id"], "priority": c["priority"], "owner": c["owner"],
           "card": c["id"] + ".quality-card.json", "decision": c["decision"]["status"],
           "next_phase": c["next_proposal"]["phase"], "ready_for_native_authorization": False,
           "blockers": [item["id"] for item in c["requirements"] if item["status"] != "PASS"]} for c in cards],
        "ready_for_native_authorization": [], "specs": specs, "software_source_sha256": code,
        "first_milestone": "source-bound cards, explicit blockers, existing owners and one bounded next proposal per priority",
        "next_decision": "Existing CE8 owner reviews its current accounting design; pruning owner completes and reviews Q0 runner. No model invocation is authorized by the queue.",
        "activity": {"model_loads": 0, "native_forwards": 0, "native_updates": 0,
                     "protected_bank_reads": 0, "owner_workers_launched": 0, "spend": 0}, **NO_ACTIONS}
    write_new(output / "QUEUE.json", queue)
    lines = ["Rineya retention-first quality queue", "", "Offline source-bound evidence. Every proposed candidate remains HOLD.",
             "Historical gains are reported separately; no native action, bank read or promotion is authorized.", ""]
    for card in cards:
        lines += [str(card["priority"]) + ". " + card["title"], "Owner: " + card["owner"]["title"],
                  "Next proposal: " + card["next_proposal"]["phase"],
                  "Blockers: " + ", ".join(item["id"] for item in card["requirements"] if item["status"] != "PASS"), ""]
    lines += ["First proposed quality study: one-donor pruning versus acquisition-matched scalar attenuation.",
              "Keep CE8/pruning/matched MLP ahead of data/SFT and RSD. Defer SRCL and twelve-draw expansion.",
              "Use existing owners. Finish source/data/runtime review and obtain separate phase authority before native work."]
    with (output / "REVIEW.txt").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    for path in sorted(output.rglob("*")):
        if path.is_file():
            inventory[path.relative_to(output).as_posix()] = sha_file(path)
    write_new(output / "MANIFEST.json", {"schema": 1, "files": inventory})
    return verify_packet(output)


def packet_file(output, name, inventory):
    require(type(name) is str and name in inventory, "packet path missing from inventory")
    path = (output / name).resolve()
    require(path.is_relative_to(output), "packet path escapes root")
    return path


def verify_packet(output, *, check_live_sources=True):
    output = Path(output).resolve()
    manifest = strict_json((output / "MANIFEST.json").read_bytes())
    expected = set(manifest["files"]) | {"MANIFEST.json"}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    require(actual == expected, "packet inventory mismatch")
    for name, expected_hash in manifest["files"].items():
        path = packet_file(output, name, manifest["files"])
        require(sha_file(path) == expected_hash, "packet file drift: " + name)
    queue = strict_json((output / "QUEUE.json").read_bytes())
    require([item["id"] for item in queue["ordered_candidates"]] == list(IDS), "queue priority drift")
    require(all(queue.get(key) is False for key in NO_ACTIONS), "queue permission drift")
    require(queue.get("ready_for_native_authorization") == [], "queue readiness must remain closed")
    require([spec["id"] for spec in queue["specs"]] == list(IDS), "specification roster drift")
    checked = 0
    for entry, spec_binding in zip(queue["ordered_candidates"], queue["specs"], strict=True):
        require(entry.get("ready_for_native_authorization") is False, "candidate queue readiness drift")
        require(entry["card"] == entry["id"] + ".quality-card.json", "card path/identity mismatch")
        card = strict_json(packet_file(output, entry["card"], manifest["files"]).read_bytes())
        spec_path = packet_file(output, spec_binding["capsule"], manifest["files"])
        require(sha_file(spec_path) == spec_binding["sha256"], "specification capsule drift")
        spec = strict_json(spec_path.read_bytes())
        require(card["id"] == entry["id"] and card["decision"] == preservation_decision(card["preservation_gates"]),
                "card decision/source mismatch")
        require(card["decision"]["status"] == "HOLD" and all(card.get(key) is False for key in NO_ACTIONS),
                "offline packet cannot authorize advancement")
        require(all(card["next_proposal"].get(key) is False for key in NO_ACTIONS)
                and card["next_proposal"].get("ready_for_native_authorization") is False, "nested proposal permission drift")
        snapshots = {}
        records = {record["id"]: record for record in spec["evidence"]}
        for evidence in card["evidence"]:
            require(evidence["id"] in records and evidence["id"] not in snapshots, "evidence identity drift")
            capsule = packet_file(output, evidence["capsule"], manifest["files"])
            require(sha_file(capsule) == evidence["sha256"], "evidence capsule drift")
            snap = read_evidence(dict(records[evidence["id"]], path=str(capsule)))
            require(snap["binding"]["sha256"] == evidence["sha256"], "capsule bytes differ from source binding")
            snap["binding"] = evidence
            snapshots[evidence["id"]] = snap
            if check_live_sources:
                _, resolved = source_path(evidence["requested_path"])
                require(str(resolved) == evidence["path"], "live source resolution drift; generate a new packet")
                require(sha_file(resolved) == evidence["sha256"], "live source drift; generate a new packet")
            checked += 1
        rebuilt, _ = build_card(spec, inspect_git=False, supplied_snapshots=snapshots)
        rebuilt["source_checkout"] = card["source_checkout"]  # Historical Git observation, not a new execution.
        require(rebuilt == card, "card assertions differ from specification/capsule replay")
        require(entry["owner"] == card["owner"] and entry["priority"] == card["priority"]
                and entry["decision"] == card["decision"]["status"]
                and entry["next_phase"] == card["next_proposal"]["phase"], "queue/card metadata drift")
        if check_live_sources:
            spec_requested, spec_resolved = source_path(spec_binding["path"])
            require(spec_resolved.suffix == ".json" and sha_file(spec_resolved) == spec_binding["sha256"], "live specification drift")
    for name, expected_hash in queue["software_source_sha256"].items():
        require(type(name) is str and "\\" not in name and ":" not in name
                and not Path(name).is_absolute() and ".." not in Path(name).parts, "invalid software source path")
        path = (ROOT / name).resolve()
        require(path.is_relative_to(Path(__file__).parent.resolve()), "software source path escapes package")
        capsule = packet_file(output, "software/" + name, manifest["files"])
        require(sha_file(capsule) == expected_hash, "software capsule drift")
        if check_live_sources:
            require(sha_file(path) == expected_hash, "queue software drift")
    return {"status": "PASS_OFFLINE_SOURCE_BINDINGS_ONLY", "cards_verified": len(IDS),
            "evidence_bindings_verified": checked, "files_verified": len(manifest["files"]),
            "live_sources_checked": check_live_sources, "manifest_sha256": sha_file(output / "MANIFEST.json"),
            "quality_improvement_established": False, **NO_ACTIONS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Read explicit metadata sources; create-only outside tracked source")
    build.add_argument("--output", required=True)
    verify = sub.add_parser("verify", help="Check immutable evidence capsules and current source drift")
    verify.add_argument("--packet", required=True)
    verify.add_argument("--capsules-only", action="store_true", help="Verify historical packet bytes without claiming current-source validity")
    args = parser.parse_args(argv)
    try:
        result = build_packet(args.output) if args.command == "build" else verify_packet(args.packet, check_live_sources=not args.capsules_only)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, IndexError, TypeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "BLOCKED_OFFLINE_QUEUE", "reason": str(exc), **NO_ACTIONS}))
        return 2
