"""Read the written corpus, parse visible records, and independently verify labels."""
from __future__ import annotations
import argparse
import copy
from itertools import zip_longest
from pathlib import Path
from typing import Any
from .common import FAMILIES, SPLITS, SYSTEM_REASON, digest, dumps, read_jsonl, sha_file, loads, write_json
from .text import parse_prompt, canonical_case
from .oracle import audit_answer, verify_solution, replay_transition, verify_trace
from .rewards import score_response
from .tasks import trace_text


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_manifest(root: Path, name: str = "DATA_MANIFEST.json") -> int:
    manifest = loads((root / name).read_text(encoding="utf-8"))
    for rel, entry in manifest["files"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escaped root")
        require(path.is_file(), f"Missing manifest file {rel}")
        require(path.stat().st_size == entry["bytes"] and sha_file(path) == entry["sha256"], f"Checksum mismatch: {rel}")
    return len(manifest["files"])


def validate(root: Path, *, emit_progress: bool = True) -> dict[str, Any]:
    checked_files = verify_manifest(root)
    build = loads((root / "reports" / "build_report.json").read_text(encoding="utf-8"))
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    seen_cases: set[str] = set()
    seen_prompts: set[str] = set()
    parent_split: dict[str, str] = {}
    parent_family: dict[str, str] = {}
    train_meta: dict[str, Any] = {}
    parent_references: dict[str, Any] = {}
    counts = {}; trace_steps = 0; sft_rows = 0
    for split in SPLITS:
        pairs = zip_longest(read_jsonl(root / "data" / "rl" / f"{split}.jsonl.gz"),
                            read_jsonl(root / "metadata" / f"{split}.jsonl.gz"),
                            read_jsonl(root / "data" / "rl_reason" / f"{split}.jsonl.gz"))
        n = 0
        for row, meta, reason_row in pairs:
            require(row is not None and meta is not None and reason_row is not None, "Unaligned canonical/view/metadata counts")
            require(set(row) == {"id", "group_id", "family", "split", "prompt"}, "Unexpected RL fields (possible target leak)")
            require(row["id"] == meta["id"] == row["group_id"] == meta["group_id"], "Bad ID join")
            require(row["id"] not in seen_ids and row["group_id"] not in seen_groups, "Duplicated independent ID/group")
            require(row["split"] == meta["split"] == split and row["family"] in FAMILIES, "Split/family mismatch")
            seen_ids.add(row["id"]); seen_groups.add(row["group_id"])
            parent_split[row["id"]] = split; parent_family[row["id"]] = row["family"]
            parsed = parse_prompt(row["prompt"])
            require(parsed["family"] == row["family"] == meta["family"], "Task family mismatch")
            case = digest(canonical_case(parsed))
            require(case not in seen_cases, "Exact logical-case overlap")
            require(case == meta["layout"]["case_sha256"] == digest(canonical_case(meta["world"])), "World/prompt identity mismatch")
            seen_cases.add(case)
            prompt_sha = digest(row["prompt"])
            require(prompt_sha not in seen_prompts and prompt_sha == meta["prompt_sha256"], "Duplicate or changed prompt")
            seen_prompts.add(prompt_sha)
            expected_reason = copy.deepcopy(row)
            expected_reason["prompt"][0]["content"] = SYSTEM_REASON
            require(expected_reason == reason_row and digest(reason_row["prompt"]) == meta["reason_prompt_sha256"], "Reason-mode view mismatch")
            certificate = audit_answer(parsed)
            require(certificate == meta["certificate"], "Certificate disagreement")
            require(verify_solution(parsed, meta["reference"], certificate)["correct"], "Invalid gold witness")
            require(meta["reference_text"] == dumps(meta["reference"]), "Gold serialization mismatch")
            require(score_response(parsed, meta["reference_text"], certificate=certificate)["reward"] == 1, "Gold scoring failure")
            user = row["prompt"][1]["content"]
            require(len(user.split()) == meta["layout"]["input_words"] and len(user) == meta["layout"]["input_characters"], "Length metadata mismatch")
            require(meta["layout"]["token_count"] is None, "Unqualified token claim")
            spans = meta["layout"]["target_character_spans"]
            require(len(spans) == len(parsed["records"]), "Evidence span coverage mismatch")
            for start, end in spans:
                require(0 <= start < end <= len(user) and user[start:end].startswith("Record ") and ('scope=' + dumps(meta["layout"]["scope"])) in user[start:end], "Invalid evidence span")
            if parsed["family"] in ("ledger_replay", "register_machine", "prerequisite_plan"):
                for step in meta["trace_steps"]:
                    require(replay_transition(parsed["family"], step["before"], step["event"]) == step["after"], "Trace transition error")
                    trace_steps += 1
                for previous, current in zip(meta["trace_steps"], meta["trace_steps"][1:]):
                    require(previous["after"] == current["before"], "Discontinuous trace state")
            require(verify_trace(parsed, meta["reference"], meta["trace_steps"]), "Structured trace disagrees with visible-record replay")
            require(trace_text(parsed, meta["reference"], meta["trace_steps"]) == meta["teacher_trace"], "Trace serialization mismatch")
            parent_references[row["id"]] = meta["reference"]
            if split == "train":
                train_meta[row["id"]] = meta
            n += 1
        require(n == build["splits"][split]["rows"], "Build count mismatch")
        counts[split] = n
        if split in ("train", "validation"):
            for row, m, sf, sr in zip_longest(read_jsonl(root / "data" / "rl" / f"{split}.jsonl.gz"), read_jsonl(root / "metadata" / f"{split}.jsonl.gz"), read_jsonl(root / "data" / "sft_final" / f"{split}.jsonl.gz"), read_jsonl(root / "data" / "sft_reason" / f"{split}.jsonl.gz")):
                require(all(x is not None for x in (row, m, sf, sr)), "SFT alignment mismatch")
                common = {k: row[k] for k in ("id", "group_id", "family", "split")}
                require(sf == {**common, "messages": row["prompt"] + [{"role": "assistant", "content": m["reference_text"]}]}, "Final SFT differs from reference")
                prompt = copy.deepcopy(row["prompt"]); prompt[0]["content"] = SYSTEM_REASON
                expected = "<think>\n" + m["teacher_trace"] + "\n</think>\n<final>" + m["reference_text"] + "</final>"
                require(sr == {**common, "messages": prompt + [{"role": "assistant", "content": expected}]}, "Reason SFT mismatch")
                require(score_response(m["world"], expected, mode="reason_then_final", certificate=m["certificate"])["reward"] == 1, "Reason-view gold scoring failure")
                sft_rows += 2
        if emit_progress:
            print(f"validated {split}: {n} independent worlds", flush=True)
    process_count = 0; process_ids = set()
    for p in read_jsonl(root / "process" / "train_transitions.jsonl.gz"):
        require(p["id"] not in process_ids and p["group_id"] == p["parent_id"] and parent_split.get(p["parent_id"]) == "train", "Process ID/split leak")
        process_ids.add(p["id"])
        m = train_meta[p["parent_id"]]
        s = m["trace_steps"][p["step_index"]]
        require(p["before"] == s["before"] and p["event"] == s["event"] and p["family"] == m["family"], "Process source mismatch")
        valid = replay_transition(p["family"], p["before"], p["event"]) == p["proposed_after"]
        require(type(p["valid"]) is bool and valid == p["valid"], "Incorrect process label")
        process_count += 1
    pref_count = 0; pref_ids = set()
    for p in read_jsonl(root / "preferences" / "train_pairs.jsonl.gz"):
        require(p["id"] not in pref_ids and p["group_id"] == p["parent_id"] and parent_split.get(p["parent_id"]) == "train", "Preference ID/split leak")
        pref_ids.add(p["id"])
        m = train_meta[p["parent_id"]]
        chosen = score_response(m["world"], p["chosen"], certificate=m["certificate"])
        rejected = score_response(m["world"], p["rejected"], certificate=m["certificate"])
        require(chosen["reward"] == 1 and rejected["reward"] == 0, "Preference direction not supported")
        pref_count += 1
    episode_count = 0
    for split in SPLITS:
        groups: dict[str, list[Any]] = {}
        for e in read_jsonl(root / "episodes" / f"{split}.jsonl.gz"):
            require(parent_split.get(e["parent_id"]) == split == e["split"] and e["group_id"] == e["parent_id"], "Episode split leak")
            parsed = parse_prompt(e["prompt"])
            cert = audit_answer(parsed)
            require(cert == e["certificate"] and verify_solution(parsed, e["reference"], cert)["correct"], "Episode label invalid")
            require(all(r["tick"] <= parsed["query"]["cutoff"] for r in parsed["records"]), "Future records leaked")
            groups.setdefault(e["parent_id"], []).append((e, parsed))
            episode_count += 1
        for parent, episodes in groups.items():
            require([e["stage"] for e, _ in episodes] == [0, 1, 2, 3], "Missing or duplicate episode stage")
            for (before, wb), (after, wa) in zip(episodes, episodes[1:]):
                old = {dumps(r) for r in wb["records"]}; new = {dumps(r) for r in wa["records"]}
                require(old <= new, "Historical records disappeared between snapshots")
                require(wa["query"]["cutoff"] > wb["query"]["cutoff"], "Noncausal snapshot order")
                if before["reference"].get("path") and len(before["reference"]["path"]) > 1:
                    require(wa["query"]["start"] == before["reference"]["path"][1], "Teacher movement mismatch")
            require(episodes[-1][0]["reference"] == parent_references[parent], "Final episode differs from canonical reference")
    require(process_count == build["derived_process_labels"] and pref_count == build["derived_preference_pairs"] and episode_count == build["derived_episode_decision_views"], "Derived-view count mismatch")
    for filename, split, maximum in (("train_256.jsonl", "train", 256), ("validation_128.jsonl", "validation", 128)):
        pilot = list(read_jsonl(root / "pilot" / filename))
        require(len(pilot) <= maximum and len({p["id"] for p in pilot}) == len(pilot), "Pilot count/duplicates")
        for p in pilot:
            require(parent_split.get(p["id"]) == split and digest(p["prompt"]) in seen_prompts, "Pilot split/content mismatch")
    return {"status": "PASS", "manifest_files_verified": checked_files, "independent_worlds": len(seen_ids),
            "splits": counts, "independent_solver_reference_checks": len(seen_ids),
            "sft_views_checked": sft_rows, "complete_structured_traces_checked": len(seen_ids), "independently_replayed_trace_steps": trace_steps,
            "process_labels_checked": process_count, "preference_pairs_checked": pref_count,
            "causal_episode_views_checked": episode_count, "exact_case_and_prompt_overlap": 0,
            "native_tokenizer_run": False, "model_calls": 0, "optimizer_steps": 0,
            "human_review": "NOT_PERFORMED", "model_quality": "NOT_MEASURED"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.root)
    if args.output:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite validation receipt")
        write_json(args.output, report)
    print(dumps(report))


if __name__ == "__main__":
    main()
