"""Replay-qualified export; private guidance never enters reconstructed SFT inputs."""
from __future__ import annotations

from collections import Counter

from .core import Environment, Task, digest, enroll, require, student_prompt
from .pipeline import validate_rewrite
from .runtime import audit


def compile_dataset(tasks: list[Task], ancestors: dict[str, list[str]], sources: list[dict],
                    rewrites: list[dict], *, arm: str, parent_cap=4,
                    source_ids: list[str] | None = None) -> dict:
    require(arm in {"direct", "one_pass", "recursive", "corrected"}, "dataset arm")
    require(type(parent_cap) is int and parent_cap > 0, "parent cap")
    catalog = enroll(tasks, ancestors)
    by_id, sandboxes = {}, set()
    for row in sources:
        require(row["id"] not in by_id and row["sandbox"] not in sandboxes, "duplicate source")
        require(row["source_id"] is None and row["arm"] in {"portfolio", "h0_resampling"}, "source role")
        require(row["task"] in catalog, "unknown task")
        audit(row, catalog[row["task"]])  # Failed sources are audited too.
        by_id[row["id"]] = row
        sandboxes.add(row["sandbox"])
    selected_ids = set(source_ids) if source_ids is not None else {
        r["id"] for r in sources if r["arm"] == "portfolio"}
    require(selected_ids <= set(by_id), "unknown selected source")
    require(source_ids is None or len(source_ids) == len(selected_ids), "duplicate source selection")
    candidates = []
    if arm == "direct":
        candidates = [(by_id[key], "source_context") for key in selected_ids]
    seen_rewrites = set()
    for rewrite in rewrites:
        require(rewrite["source_id"] in by_id, "missing source execution")
        source = by_id[rewrite["source_id"]]
        validate_rewrite(rewrite, source, catalog[source["task"]])
        for row in rewrite["records"]:
            require(row["id"] not in seen_rewrites and row["sandbox"] not in sandboxes, "duplicate rewrite")
            seen_rewrites.add(row["id"])
            sandboxes.add(row["sandbox"])
            if rewrite["arm"] == arm and rewrite["source_id"] in selected_ids:
                candidates.append((row, rewrite["track"]))
    counts, rows, retained, tracks = Counter(), [], [], set()
    rejected = Counter()
    for record, track in sorted(candidates, key=lambda pair: (pair[0]["task"], pair[0]["attempt"], pair[0]["harness"])):
        task = catalog[record["task"]]
        if not audit(record, task)["eligible"]:
            rejected["unsuccessful_or_unsupported"] += 1
            continue
        if counts[task.parent] >= parent_cap:
            rejected["parent_cap"] += 1
            continue
        require(task.split == "train", "nontraining export")
        tracks.add(track)
        require(len(tracks) == 1, "do not pool self/teacher-assisted datasets")
        counts[task.parent] += 1
        retained.append(record["id"])
        env = Environment(task)
        for index, step in enumerate(record["steps"]):
            # Direct control keeps the source-visible controller hint. It is labeled accordingly.
            # Reconstruction strips guidance from INPUTS, not merely from loss labels.
            prompt = step["student_prompt"] if arm == "direct" else student_prompt(env.view())
            rows.append(dict(prompt=prompt, completion=step["action"], arm=arm, track=track,
                             task=task.key, task_hash=task.fingerprint, parent=task.parent,
                             source_id=record["source_id"] or record["id"],
                             trajectory_id=record["id"], step=index, split="train"))
            env.execute(step["action"])
    require(bool(rows), "no eligible supervised actions")
    return dict(schema="rsr-sft-v1", arm=arm, track=next(iter(tracks)), rows=rows,
                rows_hash=digest(rows), selected_source_ids=sorted(selected_ids),
                retained_trajectories=retained, parent_counts=dict(counts),
                candidate_trajectories=len(candidates), rejected=dict(rejected),
                parent_cap=parent_cap, native_tokenization="NOT_RUN")


def encode_row(row: dict, tokenizer, max_context: int) -> dict:
    require(row.get("split") == "train", "nontraining tokenization")
    require(type(max_context) is int and max_context > 0, "context")
    prompt = list(tokenizer.encode(row["prompt"], add_special_tokens=False))
    completion = list(tokenizer.encode(row["completion"], add_special_tokens=False))
    eos = tokenizer.eos_token_id
    require(type(eos) is int and eos >= 0, "EOS required")
    require(prompt and completion and eos not in completion, "empty/EOS-contaminated completion")
    require(all(type(t) is int and t >= 0 for t in prompt + completion), "invalid token IDs")
    ids = prompt + completion + [eos]
    require(len(ids) <= max_context, "context overflow: truncation forbidden")
    return dict(input_ids=ids, attention_mask=[1] * len(ids),
                labels=[-100] * len(prompt) + completion + [eos], prompt_ids=prompt)


def match_controls(datasets: list[dict], training_receipts: list[dict]) -> dict:
    require(len(datasets) == len(training_receipts) and len(datasets) >= 2, "control count")
    signatures = []
    for data, receipt in zip(datasets, training_receipts):
        require(digest(data["rows"]) == data["rows_hash"] == receipt["rows_hash"], "training data drift")
        require(receipt["dataset_arm"] == data["arm"], "training arm mismatch")
        signatures.append({
            "parents": sorted(data["parent_counts"]),
            "sources": data["selected_source_ids"],
            **{k: receipt[k] for k in ("initial_checkpoint", "tokenizer_id", "optimizer_steps",
                                      "supervised_tokens", "trainable_names", "learning_rate",
                                      "seed", "retention_policy", "max_context")},
        })
    require(all(s == signatures[0] for s in signatures), "unmatched training controls")
    return dict(matched=True, controls=signatures[0],
                note="Matched supervision and updates, not equal forward FLOPs or generation costs.")
