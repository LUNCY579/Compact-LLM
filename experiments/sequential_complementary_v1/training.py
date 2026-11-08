"""Compile matched views into an auditable RRP handoff, not a new trainer.

The native engine and its supervisor remain unchanged. New subject qualification
is mandatory; this preparation file cannot be used as a native execution permit.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from .integrity import digest, new_directory, require, valid_hash, write_new, write_jsonl_new


def compile_handoff(views, tokenizer, out: Path, *, acceptance, recipe, compiler=None, fixture=False):
    require(set(views) in ({"F", "P"}, {"F", "P", "T"}), "prepare exactly F/P or F/P/T")
    require(acceptance.get("schema") == "SCP_DATA_AUDIT_V1", "data audit missing")
    require(acceptance.get("candidate_ready_for_review") is True, "data mechanics/shortcut/identity gates failed")
    require(valid_hash(recipe.get("student_sha256")) and recipe.get("student_dtype") == "float32", "exact FP32 incumbent missing")
    require(type(recipe.get("seed")) is int and type(recipe.get("epochs")) is int and recipe["epochs"] > 0,
            "seed/epochs must be frozen before compilation")
    require(recipe.get("comparison") == "MATCHED_WORLDS_AND_PRESENTATIONS", "comparison estimand missing")
    if compiler is None:
        # Reuse the actual Qwen template, EOS and post-EOS masking contract.
        from experiments.reasoning_posttraining_v1.serialization import compile_row
        compiler = compile_row
    else:
        require(fixture is True, "injected compiler is fixture-only")
    admitted = {r["id"]: r for r in acceptance.get("row_bindings", [])}
    require(len(admitted) == len(acceptance.get("row_bindings", [])) and admitted, "acceptance row identities missing/duplicate")
    ids = None
    preservation_hash = None
    compiled, summaries = {}, {}
    for arm, presentations in views.items():
        require(presentations and len({r["id"] for r in presentations}) == len(presentations), "duplicate/empty presentations")
        signature = [(r["id"], r["world_id"], r["public_sha256"], r["role"]) for r in presentations]
        require(ids is None or ids == signature, "arm populations or ordering differ")
        ids = signature
        anchors = [r for r in presentations if r["role"] == "preservation"]
        h = digest(anchors)
        require(preservation_hash is None or h == preservation_hash, "preservation exposure changed between arms")
        preservation_hash = h
        require(all(r["source_split"] == "train" for r in presentations), "non-TRAIN presentation")
        for row in presentations:
            a = admitted.get(row["id"], {})
            require(a.get("split") == "TRAIN" and a.get("world_id") == row["world_id"] and
                    a.get("public_sha256") == row["public_sha256"] and
                    a.get("reference_sha256") == digest(row["canonical_answer"]), "presentation not bound to audited TRAIN row")
        rows = []
        for epoch in range(recipe["epochs"]):
            for row in presentations:
                native = compiler(row["prompt"], row["reference"], tokenizer, max_sequence=2048, completion_reserve=128)
                require(native["post_eos_tokens_supervised"] is False and native["truncated"] is False,
                        "compiler violated EOS/truncation policy")
                require(len(native["ids"]) == len(native["labels"]) and native["labels"][0] == -100,
                        "invalid shifted loss labels")
                rows.append({**row, **native, "parent_id": row["world_id"], "presentation_epoch": epoch})
        compiled[arm] = rows
        summaries[arm] = {"presentations": len(rows), "unique_worlds": len({r["world_id"] for r in rows}),
            "processed_tokens": sum(len(r["ids"]) for r in rows),
            "supervised_tokens": sum(sum(t != -100 for t in r["labels"][1:]) for r in rows),
            "role_counts": dict(Counter(r["role"] for r in rows)), "preservation_sha256": h}
    # Complete all compilation BEFORE creating any success artifact. No silent row drops.
    out = new_directory(out)
    files = {arm: write_jsonl_new(out / f"{arm}.jsonl", rows) for arm, rows in compiled.items()}
    result = {"schema": "SCP_TRAINING_HANDOFF_V1", "experiment": "SCP-001", "recipe": recipe,
              "views_sha256": digest(views), "acceptance_sha256": digest(acceptance), "files": files,
              "arms": summaries, "preservation_sha256": preservation_hash,
              "equal_compute_claim": False, "classification": "FIXTURE_ONLY" if fixture else "PREPARATION_ONLY",
              "native_execution_permitted": False,
              "remaining": ["review exact compiled F/P/T rows and masks", "bind full native recipe and new session grant",
                            "qualify exact Titan FP32 subject, inherited engine and supervisor together",
                            "train all registered arms before scoring; write records outside checkout"],
              "optimizer_steps": 0, "quality_claim": False}
    write_new(out / "HANDOFF.json", result)
    return result
