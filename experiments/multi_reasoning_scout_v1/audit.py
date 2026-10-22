"""Read repository refs, metadata and specifically allowlisted local model assets.

No dataset glob reader and no checkpoint restoration. Does not modify other repos.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess

from .core import ROOT, RECORDS, PARENT, file_hash, read_json, write_new, new_dir
from .budget import runtime_versions, snapshot

WORK = ROOT.parent
ORIGINAL = WORK / "ChatBubble-ced-native-loader-qualification"
ASSETS = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "vocab.json", "merges.txt")


def command(*args, cwd=None):
    return subprocess.check_output(list(args), cwd=cwd, text=True, encoding="utf-8", timeout=60).strip()


def local_repo(path):
    return {"path": str(path), "branch": command("git", "branch", "--show-current", cwd=path),
            "sha": command("git", "rev-parse", "HEAD", cwd=path),
            "remote": command("git", "remote", "get-url", "origin", cwd=path),
            "status_porcelain": command("git", "status", "--porcelain", cwd=path),
            "heads": command("git", "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads", cwd=path).splitlines()}


def storage(path):
    with path.open("rb") as stream:
        n = struct.unpack("<Q", stream.read(8))[0]
        if n > 16*2**20:
            raise ValueError("invalid tensor header")
        header = json.loads(stream.read(n))
    return sorted({v["dtype"] for k, v in header.items() if k != "__metadata__"})


def model_entry(name, path, sha, classification):
    assets = {n: file_hash(path/n) for n in ASSETS}
    entry = {"id": name, "path": str(path), "assets": assets,
             "storage_dtype": storage(path / "model.safetensors"), "runtime_dtype": "float32",
             "attention_backend": "eager", "policy": "plain_dense_greedy_no_bias",
             "source_repository": "LUNCY579/ChatBubble", "source_branch": "codex/dataset-retention-audit-v1",
             "source_sha": sha, "classification": classification,
             "checkpoint_bytes": (path/"model.safetensors").stat().st_size,
             "decoding": {"sample": False, "greedy": True, "tf32": False, "thinking": False,
                          "eos": 151645, "context_cap": 4096, "prompt_cap": 2048,
                          "role_output_cap": 64, "A1_output_cap": 320, "abstention_bias": 0}}
    if (path/"MATERIALIZATION.json").is_file():
        r = read_json(path/"MATERIALIZATION.json")
        if assets["model.safetensors"] != r["out_weight_sha256"]:
            raise ValueError("retained export differs from materialization receipt")
        entry["checkpoint_lineage"] = {k: r[k] for k in ("checkpoint", "checkpoint_manifest_sha256", "source_weight_sha256", "checkpoint_progress")}
        entry["checkpoint_lineage"]["receipt_sha256"] = file_hash(path/"MATERIALIZATION.json")
        entry["checkpoint_lineage"]["original_optimizer_checkpoint_present"] = (Path(r["checkpoint"])/"manifest.json").is_file()
    return entry


def main():
    RECORDS.mkdir(parents=True, exist_ok=True)
    out = new_dir(RECORDS / "evidence/audit")
    remotes = {}
    for repo in ("ChatBubble", "SupaExperiment", "RineyaBeta", "RineyaArchitecture"):
        remotes[repo] = {"repository": "LUNCY579/"+repo,
            "metadata": json.loads(command("gh", "api", f"repos/LUNCY579/{repo}", "--jq", "{full_name,default_branch,pushed_at,private}")),
            "heads": command("git", "ls-remote", "--heads", f"https://github.com/LUNCY579/{repo}.git").splitlines()}
    locals_ = {name: local_repo(WORK/name) for name in (
        "ChatBubble-ced-native-loader-qualification", "SupaExperiment", "SupaExperiment-nonrecurrent-adaptation-v1",
        "RineyaArchitecture", "RineyaBeta-supa-fwa-v1")}
    # Metadata documents only; explicitly never follow the dataset paths they mention.
    docs = ["research/reasoning_posttraining_v1/QUARANTINE.json", "CONFIRMATION_INSTRUMENT_GAP.md",
            "research/reasoning_posttraining_v1/RUN_LEDGER.jsonl", "research/reasoning_posttraining_v1/CLAIM_LEDGER.json",
            "research/reasoning_posttraining_v1/RRP002_INSTRUMENT_PREREG.md",
            "docs/model-quality/audit-2026-09-17/DATASET_CONSUMED_CONTACT_AUDIT.json",
            "docs/model-quality/audit-2026-09-17/PRESERVED_CANDIDATES.json"]
    metadata = {name: {"sha256": file_hash(ORIGINAL/name)} for name in docs}
    quarantine = read_json(ORIGINAL/docs[0])
    access = {"registry": str(ORIGINAL/docs[0]), "registry_sha256": file_hash(ORIGINAL/docs[0]),
              "metadata_documents": metadata, "protected_data_read": False,
              "quarantined_ids": sorted({i for e in quarantine["entries"] for i in e.get("ids", [])}),
              "entries": [{k: e.get(k) for k in ("key", "path", "bars", "reason")} for e in quarantine["entries"]],
              "data_allowlist": ["bank/public.jsonl", "bank/gold.jsonl", "bank/worlds.jsonl"],
              "policy": "No historical bank content is read, copied or hashed. Metadata is read; protected paths are never followed."}
    write_new(RECORDS/"ACCESS_MANIFEST.json", access)
    observed = snapshot()
    write_new(out/"resource_snapshot.json", observed)
    source = model_entry("supa-1.7-frozen", ORIGINAL/"Supa", PARENT, "BLOCKED ON RESOURCE LIMIT")
    lock = read_json(ORIGINAL/"experiments/model_quality_gpu_v1/source_lock.json")
    if source["assets"] != lock["assets"]:
        raise ValueError("source release hash mismatch")
    source["source_release"] = {k: lock[k] for k in ("source_id", "revision", "path", "parameters")}
    d1 = model_entry("d1-seed17-bf16-export", WORK/"rineya-dataset-retention-records/lc-D1-hf", PARENT, "BLOCKED ON RESOURCE LIMIT")
    for model in (source, d1):
        model["qualification_requirement"] = "Fresh no-update native/scout qualification required on this exact runner; historical DenseReference scores are not a receipt"
    opallo = model_entry("opallo-1.0-frozen", ORIGINAL/"Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze", PARENT,
                         "BLOCKED ON NATIVE QUALIFICATION")
    opallo["selection"] = "Not selected: D1 supplies the smaller and directly relevant atomic-transition comparison"
    supa_work = WORK/"SupaExperiment-nonrecurrent-adaptation-v1"
    recommendation = read_json(supa_work/"experiments/nonrecurrent_adaptation_v1/deploy/adapter_recommendation.json")
    rejected = [opallo,
        {"id": "RRP-asymmetric-A1-seeds17-23-29", "classification": "BLOCKED ON MISSING CHECKPOINT",
         "paths_checked": [], "weight_files_found": []},
        {"id": "Supa-A1.1-matched_mlp", "classification": "INCOMPARABLE MODEL FAMILY / POLICY",
         "source_repository": "LUNCY579/SupaExperiment", "source_branch": locals_[supa_work.name]["branch"],
         "source_sha": locals_[supa_work.name]["sha"], "source_reported_bundle": recommendation["bundle"],
         "policy": "Custom gated Supa-A1.1 + 156225-parameter shared adapter; deployed bias 3.75; no substitution into dense no-bias protocol",
         "evidence_class": "source-reported open development; not rerun"},
        {"id": "other-D1-seeds23-29", "classification": "NOT RELEVANT",
         "reason": "Retained exports exist; extra selection seeds would increase cost without adding a mechanism control"},
        {"id": "future-dataset-trained", "classification": "BLOCKED ON MISSING CHECKPOINT", "future": True}]
    for seed in (17, 23, 29):
        directory = WORK/f"rineya-dataset-retention-records/rel-a1s{seed}-journal"
        rejected[1]["paths_checked"].append(str(directory))
        rejected[1]["weight_files_found"] += [str(p) for suffix in ("*.safetensors", "*.pt", "*.bin") for p in directory.rglob(suffix)]
    if rejected[1]["weight_files_found"]:
        rejected[1]["classification"] = "BLOCKED ON NATIVE QUALIFICATION"
    write_new(RECORDS/"MODEL_MANIFEST.json", {"selected": [source, d1], "other_candidates": rejected,
              "primary_old_candidate_for_future": d1["id"], "secondary_source_reference": source["id"],
              "runtime_versions_observed_without_native_import": runtime_versions(), "no_weights_modified": True})
    pins = {"inspected_utc": datetime.now(timezone.utc).isoformat(), "remotes": remotes, "locals": locals_,
            "parent": PARENT, "new_worktree": str(ROOT), "evidence_class": "fresh read-only audit",
            "RineyaBeta_local_correction": "RineyaBeta-supa-fwa-v1 is actually a ChatBubble worktree, not a RineyaBeta checkout; RineyaBeta audited via remote refs and source documents",
            "parent_rationale": "Newest remote-matching ChatBubble research head containing dense native and RRP instruments; uncommitted newer audit is read-only evidence, not copied executable authority"}
    write_new(RECORDS/"SOURCE_PINS.json", pins)
    write_new(out/"audit_receipt.json", {"status": "PASS", "model_asset_bytes_hashed": sum(m["checkpoint_bytes"] for m in (source, d1, opallo)),
              "models_loaded": 0, "protected_data_reads": 0, "remote_writes": 0, "original_source_changes": 0})
    print(json.dumps({"selected": [m["id"] for m in (source, d1)], "observed": observed}))


if __name__ == "__main__":
    main()
