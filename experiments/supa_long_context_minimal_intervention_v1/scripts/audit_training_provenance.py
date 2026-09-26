"""Read-only reconstruction of the merged Stage A adapter's data provenance."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
PRIMARY = REPO.parent / "Chat-Bubble"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def signature(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    stage = PRIMARY / "artifacts/halo_long_context_sft_v1/stage_a"
    mix = stage / "mix/train_stage_a.jsonl"
    lcr = PRIMARY / "Rineya_LCR_SFT_v1/Rineya_LCR_SFT_v1/data/lcr_sft_v1.jsonl"
    demo = REPO / "Rineya_Long_Context_Demo_v1/data/rineya_long_context_demo_full.jsonl"
    corpus = PRIMARY / "rineya_long_context_cc0_v1_main/data/sft/train"
    replay = PRIMARY / "short-response-model-phase-0-1-2/data/train/train.jsonl"
    core = PRIMARY / "Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze/CORE.json"
    core_files = [p for name in ("Rineya-Supa-A1.1-1.1.0-devfreeze", "Rineya-Supa-A1-1.0.0-devfreeze")
                  for p in (PRIMARY / "Rineya_Published_Cores" / name).iterdir()
                  if p.is_file() and p.name in {"model.safetensors", "config.json", "generation_config.json",
                      "modeling_supa_a11.py", "modeling_supa_a1.py", "tokenizer_config.json", "tokenizer.json",
                      "vocab.json", "merges.txt", "CORE.json", "FILES.json"}]
    checkpoint_before = {str(p): sha(p) for p in sorted(core_files)}
    inputs = [mix, lcr, demo, replay, core, stage / "mix/mix_manifest.json",
              stage / "checkpoint-200/adapter_config.json", stage / "checkpoint-200/trainer_state.json",
              stage / "checkpoint-200/adapter_model.safetensors", stage / "train.log",
              PRIMARY / "rineya_halo_long_context_sft_v1/configs/stage_a_qlora.yaml",
              PRIMARY / "rineya_halo_long_context_sft_v1/scripts/build_stage_mix.py",
              PRIMARY / "rineya_halo_long_context_sft_v1/scripts/train_halo_engine.py"]
    hashes_before = {str(p): sha(p) for p in inputs}
    # Only hashes and identifiers from the large source corpus are held in memory.
    source_sigs = {}
    for p in sorted(corpus.rglob("*.jsonl")):
        for row in rows(p):
            source_sigs[row["id"]] = signature(row.get("messages"))
    replay_sigs = {r["id"]: signature(r.get("messages")) for r in rows(replay)}
    lcr_rows, demo_rows = list(rows(lcr)), list(rows(demo))
    lcr_ids = {r["id"] for r in lcr_rows}
    demo_ids = {r["id"] for r in demo_rows}
    lcr_users = {m["content"] for r in lcr_rows for m in r.get("messages", []) if m["role"] == "user"}
    # LCR and demo both expose structured context/question fields.
    reference_text = {"lcr": lcr_rows, "demo": demo_rows}
    anchors = {}
    for suite, data in reference_text.items():
        for field in ("context", "question"):
            values = {r[field] for r in data if r.get(field)}
            # Whole-string presence handles different chat wrapper conventions.
            anchors[(suite, field)] = (values, re.compile("|".join(re.escape(x) for x in sorted({x[:80] for x in values}))))
    counts, strata, matches = Counter(), Counter(), Counter()
    overlap = Counter()
    prefix_candidates = {}
    for key, (values, _) in anchors.items():
        d = {}
        for value in values:
            d.setdefault(value[:80], []).append(value)
        prefix_candidates[key] = d
    for row in rows(mix):
        metadata = row.get("metadata") or {}
        source = metadata.get("mix_source", "MISSING")
        counts[source] += 1
        strata[metadata.get("context_stratum", "MISSING")] += 1
        sig = signature(row.get("messages"))
        matches[source] += int((source_sigs if source == "long_context" else replay_sigs).get(row["id"]) == sig)
        overlap["lcr_id"] += row["id"] in lcr_ids
        overlap["demo_id"] += row["id"] in demo_ids
        users = [m["content"] for m in row.get("messages", []) if m["role"] == "user"]
        overlap["lcr_user_message"] += any(u in lcr_users for u in users)
        for key, (_, pattern) in anchors.items():
            found = False
            for u in users:
                for match in pattern.finditer(u):
                    if any(v in u for v in prefix_candidates[key][match.group()]):
                        found = True
                        break
                if found:
                    break
            overlap["_".join(key) + "_substring_rows"] += found
    log = (stage / "train.log").read_bytes().decode("utf-8", errors="replace").replace("\x00", "")
    log_evidence = [line.strip() for line in log.splitlines() if any(
        word in line for word in ("--train-file", "Loading train rows", "Loaded train_rows"))]
    state = json.loads((stage / "checkpoint-200/trainer_state.json").read_text())
    result = {
        "source": json.loads(core.read_text())["source"],
        "mix_rows_recomputed": sum(counts.values()), "mix_sources": dict(counts),
        "mix_strata": dict(strata), "exact_id_and_messages_matches_to_source": dict(matches),
        "mix_overlap_with_lcr_and_demo": dict(overlap), "log_evidence": log_evidence,
        "checkpoint_global_step": state["global_step"],
        "same_lcr_sft_corpus_hypothesis": "REFUTED" if all(v == 0 for v in overlap.values()) and matches == counts else "UNCERTAIN",
        "limits": "Current cached mix plus source identity and trainer launch log establish CC0 + replay, not LCR SFT. No immutable historical mix hash or consumed-row trace binds exact individual examples seen by checkpoint-200. No claim that all cached rows were trained on or that related tasks were never seen upstream.",
        "input_sha256": hashes_before,
        "inputs_unchanged": hashes_before == {str(p): sha(p) for p in inputs},
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": {str(p): sha(p) for p in sorted(core_files)},
    }
    result["canonical_checkpoint_unchanged"] = result["checkpoint_sha256_before"] == result["checkpoint_sha256_after"]
    out = LAB / "results/audit_20260905/training_provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_sha256", "log_evidence", "checkpoint_sha256_before", "checkpoint_sha256_after")}, indent=2))


if __name__ == "__main__":
    main()
