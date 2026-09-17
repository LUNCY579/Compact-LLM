"""Build, audit and verify the RTG-1 corpus.

Layout mirrors the delivered LCR corpus so downstream tooling sees a familiar shape:

    data/{train,dev,holdout}.jsonl     model-visible text only, no answer
    gold/{train,dev,holdout}.jsonl     answers and metadata, stored separately
    sft/{train,dev,holdout}.jsonl      rendered chat plus the supervised completion
    worlds/{train,dev,holdout}.jsonl   generative state and fingerprint, for audit only
    MANIFEST.json                      seed, quotas, counts, SHA-256 of every file
    evidence/                          shortcut battery, token census, determinism receipt

The supervised target is the BARE ANSWER, because MRS-FP-1 measured that this model family
emits exactly one field and then stops. Training and serving therefore share one contract:
the MRS-2 system prompt and R0 role instruction, verbatim.

Only R0 is trained. That is the neutral choice: cell C is A0, which is R0, so C is
in-distribution, while A2's three rephrasings and A3's three distinct roles are all
out-of-distribution. Training every role would collapse the roles and manufacture the
"multi-reasoning became unnecessary" outcome; training the direct rephrasings only would
bias A2 upward against A3. Leaving all six unseen keeps the A2-vs-A3 contrast symmetric.
"""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import file_hash, write_new
from experiments.multi_reasoning_scout_v1.core import RECORDS as SCOUT_RECORDS, decode
from experiments.multi_reasoning_scout_v2 import prompts as mrs2_prompts

from . import audit, corpus
from . import families as F

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
OUT = ROOT / "research/rineya_reasoning_corpus_v1"
BRANCH = "research/rineya-reasoning-corpus-v1"
VERSION = "RTG-1.0.0"
TOKENIZER = Path(r"C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\Supa")
SPLIT_NAMES = ("train", "dev", "holdout")
TRAINED_ROLE = "R0"


def scout_worlds():
    return [decode(line)["world"] for line
            in (SCOUT_RECORDS / "bank/worlds.jsonl").read_text(encoding="utf-8").splitlines()]


def render_sft(rows, gold):
    """One rendered training pair per row, under the frozen MRS-2 serving contract."""
    labels = {g["id"]: g for g in gold}
    out = []
    for row in rows:
        # Assembled exactly as MRS-2 assembles a served prompt, so the training surface and
        # the evaluation surface cannot drift apart.
        messages = [{"role": "system", "content": mrs2_prompts.SYSTEM},
                    {"role": "user", "content": mrs2_prompts.ROLES[TRAINED_ROLE]
                     + "\n\nPROBLEM\n" + row["problem"]}]
        out.append({"id": row["id"], "structure_id": row["structure_id"],
                    "family": row["family"], "tier": row["tier"], "split": row["split"],
                    "role": TRAINED_ROLE, "messages": messages,
                    "completion": labels[row["id"]]["answer"],
                    "supervise": "completion_plus_eos"})
    return out


def _public(row):
    return {k: row[k] for k in ("id", "structure_id", "family", "tier", "split",
                                "difficulty", "problem")}


def dump(path, rows):
    with Path(path).open("xb") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8") + b"\n")


def runtime():
    import transformers
    import tokenizers
    return {"python": sys.version, "platform": platform.platform(),
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}


def source_identity():
    def git(*a):
        return subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("wrong isolated branch")
    own = sorted(PACKAGE.glob("*.py"))
    return {"repository": "LUNCY579/ChatBubble", "branch": BRANCH, "sha": git("rev-parse", "HEAD"),
            "files": {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in own}}


def build(out=OUT):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already populated; corpus directories are create-only")
    worlds_in = scout_worlds()
    rows, gold, worlds, report = corpus.generate(worlds_in)
    sft = render_sft(rows, gold)

    isolation = corpus.split_isolation(rows, worlds)
    if not isolation["isolated"]:
        raise ValueError("split isolation failed: a structure appears in two splits")

    # Disjointness from the evaluation bank, re-verified on the emitted bytes.
    forbidden_fp, forbidden_text = corpus.scout_exclusions(worlds_in)
    emitted_fp = {entry["fingerprint"] for entry in worlds}
    overlap = emitted_fp & forbidden_fp
    if overlap:
        raise ValueError(f"{len(overlap)} emitted structures collide with the scout bank")
    scout_texts = {corpus._blind(F.render(w), w["names"]) for w in worlds_in}
    text_overlap = {r["id"] for r, e in zip(rows, worlds)
                    if corpus._blind(r["problem"], e["world"]["names"]) in scout_texts}
    if text_overlap:
        raise ValueError(f"{len(text_overlap)} emitted problems collide with scout text")

    # No scout entity name may appear anywhere in the corpus.
    import re
    scout_name = re.compile(r"\bn[abcdefghjkmnpqrstuvwxyz]{6}\b")
    leaked = [r["id"] for r in rows if scout_name.search(r["problem"])]
    if leaked:
        raise ValueError(f"scout-style entity name present in {len(leaked)} rows")

    shortcut = audit.shortcuts(rows, gold)
    semantic = audit.semantic_equivalence(rows, gold, worlds)
    census = audit.token_census(rows, gold, sft, TOKENIZER)
    if census["violation_count"]:
        raise ValueError(f"{census['violation_count']} rows exceed the frozen token caps")

    for directory in ("data", "gold", "sft", "worlds", "evidence"):
        (out / directory).mkdir(parents=True)
    for split in SPLIT_NAMES:
        dump(out / f"data/{split}.jsonl", [_public(r) for r in rows if r["split"] == split])
        dump(out / f"gold/{split}.jsonl", [g for g in gold if g["split"] == split])
        dump(out / f"sft/{split}.jsonl", [s for s in sft if s["split"] == split])
        dump(out / f"worlds/{split}.jsonl", [w for w, r in zip(worlds, rows)
                                             if r["split"] == split])

    write_new(out / "evidence/SHORTCUT_AUDIT.json", shortcut)
    write_new(out / "evidence/SEMANTIC_EQUIVALENCE.json", semantic)
    write_new(out / "evidence/TOKEN_CENSUS.json", census)
    write_new(out / "evidence/GENERATION_REPORT.json", report)
    write_new(out / "evidence/SPLIT_ISOLATION.json",
              {k: v for k, v in isolation.items() if k != "crossing_splits"})

    files = {str(p.relative_to(out)).replace("\\", "/"): file_hash(p)
             for p in sorted(out.rglob("*.jsonl"))}
    manifest = {
        "version": VERSION, "classification": "TRAINING CORPUS / OPEN DEVELOPMENT",
        "seed": corpus.SEED, "quotas": corpus.QUOTAS,
        "rows": report["totals"], "families": list(F.FAMILIES), "answers": list(F.ANSWERS),
        "supervised_target": "bare answer token plus EOS",
        "trained_role": TRAINED_ROLE,
        "serving_contract": {"source": "MRS-2", "system_sha256": mrs2_prompts.manifest()["sha256"],
                             "output_contract": "C1_BARE"},
        "evaluation_bank_disjointness": {
            "bank": "MRS-bank-1", "structure_overlap": 0, "text_overlap": 0,
            "name_namespace": "q + 7 chars; the bank uses n + 6 chars",
            "collisions_rejected_during_generation":
                report["scout_fingerprint_collisions"]},
        "split_isolation": {"structures_crossing_splits": 0,
                            "unique_structures": report["totals"]["unique_structures"],
                            "semantic_equivalence_disclosure": {
                                family: {k: v for k, v in row.items() if k != "note"}
                                for family, row in semantic.items()}},
        "answer_priors": shortcut["answer_priors"],
        "constant_answer_floor_dev": shortcut["constant_answer_floor_dev"],
        "gap_conditional": shortcut["gap_conditional"],
        "token_caps": census["caps"], "token_violations": 0,
        "processed_tokens_one_epoch": census["processed_tokens_one_epoch"],
        "supervised_tokens_one_epoch": census["supervised_tokens_one_epoch"],
        "verification": "two independent solvers per family agree on every row",
        "tokenizer": {"path": str(TOKENIZER),
                      "assets": {name: file_hash(TOKENIZER / name) for name in
                                 ("tokenizer.json", "tokenizer_config.json",
                                  "merges.txt", "vocab.json")}},
        "runtime": runtime(), "source": source_identity(), "files": files,
    }
    write_new(out / "MANIFEST.json", manifest)
    return manifest


def verify(out=OUT):
    """Regenerate from the seed and confirm the emitted bytes are reproduced exactly."""
    out = Path(out)
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    rows, gold, worlds, _ = corpus.generate(scout_worlds(), seed=manifest["seed"])
    sft = render_sft(rows, gold)
    import hashlib
    rebuilt = {}
    for split in SPLIT_NAMES:
        for name, values in (("data", [_public(r) for r in rows if r["split"] == split]),
                             ("gold", [g for g in gold if g["split"] == split]),
                             ("sft", [s for s in sft if s["split"] == split]),
                             ("worlds", [w for w, r in zip(worlds, rows)
                                         if r["split"] == split])):
            digest = hashlib.sha256()
            for row in values:
                digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode("utf-8") + b"\n")
            rebuilt[f"{name}/{split}.jsonl"] = digest.hexdigest()
    mismatches = {k: {"manifest": manifest["files"][k], "regenerated": v}
                  for k, v in rebuilt.items() if manifest["files"][k] != v}
    receipt = {"deterministic": not mismatches, "files_compared": len(rebuilt),
               "mismatches": mismatches, "seed": manifest["seed"],
               "manifest_sha256": file_hash(out / "MANIFEST.json")}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "capacity"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "build":
        manifest = build(args.out)
        print(json.dumps({k: manifest[k] for k in
                          ("version", "rows", "constant_answer_floor_dev",
                           "processed_tokens_one_epoch", "supervised_tokens_one_epoch",
                           "token_violations")}, indent=1))
    elif args.command == "verify":
        receipt = verify(args.out)
        write_new(Path(args.out) / "evidence/DETERMINISM.json", receipt)
        print(json.dumps(receipt, indent=1))
    else:
        from .capacity import main as capacity_main
        capacity_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
