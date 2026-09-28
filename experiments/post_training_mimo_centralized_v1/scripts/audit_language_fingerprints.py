"""Compare open language prompts with one pinned hash-only historical inventory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from zipfile import ZipFile
from . import audit_exclusions as history
from . import e1_data as data


def compare(rows, prompt_hashes):
    seen = set(); matches = []
    for row in rows:
        data.require(row["role"] in ("DEV", "CAL") and row["parent_id"] not in seen, "open unique parent required")
        seen.add(row["parent_id"])
        messages = row["messages"]
        data.require(type(messages) is list and messages and len(messages) % 2 == 1 and
            all(set(m) == {"role", "content"} and m["role"] == ("user" if i % 2 == 0 else "assistant")
                and type(m["content"]) is str and m["content"] for i, m in enumerate(messages)), "full policy context required")
        value = history.digest(messages)
        if value in prompt_hashes:
            matches.append({"parent_id": row["parent_id"], "role": row["role"], "exact_chat_sha256": value})
    return matches


def audit(bank, archive, output):
    started = time.perf_counter(); data.require(not output.exists(), "write-once output exists")
    package = Path(__file__).resolve().parents[1]; root = package.parents[1]
    bank = data.safe_path(root, bank.resolve().relative_to(root).as_posix())
    def path(name): return data.safe_path(root, (bank / name).relative_to(root).as_posix())
    receipt, receipt_id = data.read_json(path("DATA_RECEIPT.json"))
    data.require(receipt["status"] == "AUTHORED_LANGUAGE_CANDIDATES_UNQUALIFIED", "bound candidate bank required")
    quarantine_path = root / "research/reasoning_posttraining_v1/QUARANTINE.json"
    quarantine, quarantine_id = data.read_json(quarantine_path)
    data.require(quarantine["no_quarantine_is_lifted"] and all(not e["lifted"] for e in quarantine["entries"]), "quarantine unchanged")
    archive_id = data.identity(archive); data.require(archive_id["sha256"] == history.ARCHIVE_SHA, "archive drift")
    with ZipFile(archive) as stream:
        data.require(stream.getinfo(history.MEMBER).file_size < 8 * 1024 ** 2, "inventory size")
        raw = stream.read(history.MEMBER)
    data.require(data.hashlib.sha256(raw).hexdigest() == history.MEMBER_SHA, "hash inventory drift")
    inventory = history.inventory(raw)
    rows = []; files = {}
    for role, count in (("DEV", 64), ("CAL", 16)):
        name = role + ".jsonl"; candidate_path = path(name)
        history.readable(root, candidate_path, quarantine)
        raw = candidate_path.read_bytes()
        ident = {"sha256": data.hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        data.require(ident == receipt["files"][name], "candidate bytes drift")
        group = [data.parse(line) for line in raw.splitlines()]
        data.require(len(group) == count and all(r["role"] == role for r in group), "role count drift")
        rows.extend(group); files[name] = ident
    matches = compare(rows, set(inventory["prompt_hashes"]))
    data.require(data.identity(archive) == archive_id and all(data.identity(path(n)) == i for n, i in files.items()), "concurrent source drift")
    result = {"status": "SCOPED_HISTORICAL_EXACT_CHAT_COMPARISON_ONLY", "candidate_receipt": receipt_id,
        "candidate_files": files, "candidate_parents": len(rows), "historical_archive": archive_id,
        "hash_only_member": history.MEMBER, "member_sha256": history.MEMBER_SHA,
        "historical_prompt_hashes": len(inventory["prompt_hashes"]), "exact_chat_matches": matches,
        "historical_problem_hashes_not_comparable": len(inventory["problem_hashes"]),
        "historical_provenance": inventory["provenance"], "quarantine": quarantine_id,
        "protected_payloads_read": False, "full_historical_exclusion_qualified": False,
        "semantic_or_unknown_pretraining_exclusion": False, "native_model_calls": 0,
        "script": data.identity(Path(__file__)), "inventory_helper": data.identity(Path(history.__file__)),
        "wall_seconds": time.perf_counter() - started,
        "limits": "Full supplied chat equality only. No normalized historical or user-only hash inventory; different system wrappers or semantics can evade this comparison. Distinct generators have no qualified common problem fingerprint."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f: f.write(data.canonical(result) + b"\n")
    print(json.dumps({k: result[k] for k in ("candidate_parents", "historical_prompt_hashes", "exact_chat_matches")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); audit(args.bank, args.archive, args.output)
