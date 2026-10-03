"""Deterministic full bundle data generation, with non-overwriting atomic publish."""
from __future__ import annotations
import argparse
import copy
import json
import math
import platform
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from .common import FAMILIES, SPLITS, SEED, SYSTEM_REASON, Writer, digest, dumps, write_json, sha_file
from .tasks import make_world, solve, trace_text
from .oracle import audit_answer, verify_solution, replay_transition
from .text import render, parse_prompt, canonical_case, LAYOUTS
from .rewards import score_response


def tier_for(i: int, split: str) -> int:
    if split == "test_length":
        return 4 + i % 2
    # Balanced crossing with layout (i%4); cycle 100 avoids short-tier dominance.
    j = (i // 4) % 25
    return 0 if j < 10 else 1 if j < 17 else 2 if j < 22 else 3


def negative_answer(answer: dict[str, Any]) -> str:
    bad = copy.deepcopy(answer)
    if "status" in bad:
        bad["status"] = "UNKNOWN" if bad["status"] == "IMPOSSIBLE" else "IMPOSSIBLE"
    elif "value" in bad:
        bad["value"] += 1
    elif "cost" in bad:
        bad["cost"] += 1
    elif "finish" in bad:
        bad["finish"] += 1
    else:
        bad["answer"] += "_wrong"
    return dumps(bad)


def build_data(root: Path, *, seed: int = SEED, scale: float = 1.0) -> dict[str, Any]:
    if not math.isfinite(scale) or not 0 < scale <= 10:
        raise ValueError("Scale must be in (0,10]")
    root = root.resolve()
    if root.exists():
        raise FileExistsError("Output must not exist; refusing to overwrite a release")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".forge-build-", dir=root.parent))
    try:
        report = _build(staging, seed, scale)
        staging.rename(root)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build(root: Path, seed: int, scale: float) -> dict[str, Any]:
    seen_cases: set[str] = set()
    seen_prompts: set[str] = set()
    counters: dict[str, Any] = {}
    process_n = preference_n = episode_n = 0
    samples = []
    with ExitStack() as stack:
        process = stack.enter_context(Writer(root / "process" / "train_transitions.jsonl.gz"))
        prefs = stack.enter_context(Writer(root / "preferences" / "train_pairs.jsonl.gz"))
        pilot_train = stack.enter_context(Writer(root / "pilot" / "train_256.jsonl"))
        pilot_dev = stack.enter_context(Writer(root / "pilot" / "validation_128.jsonl"))
        for split, base_count in SPLITS.items():
            count = max(1, int(round(base_count * scale)))
            stats = {"rows": 0, "families": Counter(), "statuses": Counter(), "layouts": Counter(), "word_tiers": Counter(),
                     "words_min": 10**20, "words_max": 0, "words_total": 0, "target_records_min": 10**20, "target_records_max": 0,
                     "trace_characters_total": 0, "trace_steps_total": 0, "case_rejections": 0}
            rl = stack.enter_context(Writer(root / "data" / "rl" / f"{split}.jsonl.gz"))
            rl_reason = stack.enter_context(Writer(root / "data" / "rl_reason" / f"{split}.jsonl.gz"))
            meta = stack.enter_context(Writer(root / "metadata" / f"{split}.jsonl.gz"))
            episode = stack.enter_context(Writer(root / "episodes" / f"{split}.jsonl.gz"))
            final_sft = reason_sft = None
            if split in ("train", "validation"):
                final_sft = stack.enter_context(Writer(root / "data" / "sft_final" / f"{split}.jsonl.gz"))
                reason_sft = stack.enter_context(Writer(root / "data" / "sft_reason" / f"{split}.jsonl.gz"))
            for family in FAMILIES:
                for i in range(count):
                    tier = tier_for(i, split)
                    salt = f"{seed}:{split}:{family}:{i}"
                    for attempt in range(1000):
                        rng = random.Random(digest([salt, attempt]))
                        world = make_world(family, rng, tier, split == "test_transfer")
                        answer, steps = solve(world)
                        case_sha = digest(canonical_case(world))
                        if case_sha not in seen_cases:
                            break
                        stats["case_rejections"] += 1
                    else:
                        raise RuntimeError("Could not sample a unique case")
                    seen_cases.add(case_sha)
                    prompt, layout = render(world, rng, tier, LAYOUTS[i % 4])
                    parsed = parse_prompt(prompt)
                    if digest(canonical_case(parsed)) != case_sha:
                        raise AssertionError("Prompt/world roundtrip mismatch")
                    certificate = audit_answer(parsed)
                    if not verify_solution(parsed, answer, certificate)["correct"]:
                        raise AssertionError("Independent solver disagreement")
                    for step in steps:
                        if family in ("ledger_replay", "register_machine", "prerequisite_plan"):
                            if replay_transition(family, step["before"], step["event"]) != step["after"]:
                                raise AssertionError("Trace transition disagreement")
                    row_id = "RRFR2-" + digest(salt)[:24]
                    prompt_sha = digest(prompt)
                    if prompt_sha in seen_prompts:
                        raise AssertionError("Duplicate prompt")
                    seen_prompts.add(prompt_sha)
                    row = {"id": row_id, "group_id": row_id, "family": family, "split": split, "prompt": prompt}
                    rl.write(row)
                    reason_prompt = copy.deepcopy(prompt)
                    reason_prompt[0]["content"] = SYSTEM_REASON
                    reason_row = {**row, "prompt": reason_prompt}
                    rl_reason.write(reason_row)
                    trace = trace_text(parsed, answer, steps)
                    text = dumps(answer)
                    reason_text = "<think>\n" + trace + "\n</think>\n<final>" + text + "</final>"
                    private = {"id": row_id, "group_id": row_id, "family": family, "split": split,
                               "world": parsed, "reference": answer, "reference_text": text, "certificate": certificate,
                               "teacher_trace": trace, "trace_steps": steps, "layout": layout,
                               "prompt_sha256": prompt_sha, "reason_prompt_sha256": digest(reason_prompt),
                               "generation": {"seed": seed, "index": i, "attempt": attempt, "tier": tier, "transfer": split == "test_transfer"}}
                    meta.write(private)
                    if final_sft is not None:
                        final_sft.write({"id": row_id, "group_id": row_id, "family": family, "split": split,
                                         "messages": prompt + [{"role": "assistant", "content": text}]})
                        reason_sft.write({"id": row_id, "group_id": row_id, "family": family, "split": split,
                                          "messages": reason_prompt + [{"role": "assistant", "content": reason_text}]})
                        if split == "train" and i < 32:
                            pilot_train.write(row)
                        if split == "validation" and i < 16:
                            pilot_dev.write(row)
                    if split == "train":
                        wrong = negative_answer(answer)
                        if score_response(parsed, wrong, certificate=certificate)["correct"]:
                            raise AssertionError("Negative answer accidentally correct")
                        for kind, rejected in (("wrong_concise", wrong), ("unnecessary_prose", text + "\nI have carefully solved the entire task and this is the answer.")):
                            prefs.write({"id": row_id + "-" + kind, "group_id": row_id, "parent_id": row_id,
                                         "chosen": text, "rejected": rejected, "category": kind,
                                         "source_split": "train", "synthetic_preference": True})
                            preference_n += 1
                        if family in ("ledger_replay", "register_machine", "prerequisite_plan") and steps:
                            indices = sorted(set(round(j * (len(steps) - 1) / min(3, len(steps) - 1)) if len(steps) > 1 else 0 for j in range(min(4, len(steps)))))
                            for index in indices:
                                step = steps[index]
                                good = {"id": row_id + f"-step{index}-valid", "group_id": row_id, "parent_id": row_id,
                                        "source_split": "train", "family": family, "step_index": index,
                                        "before": step["before"], "event": step["event"], "proposed_after": step["after"], "valid": True}
                                process.write(good)
                                bad = copy.deepcopy(good)
                                bad["id"] = row_id + f"-step{index}-invalid"
                                bad["valid"] = False
                                if isinstance(bad["proposed_after"], dict):
                                    key = sorted(bad["proposed_after"])[index % len(bad["proposed_after"])]
                                    bad["proposed_after"][key] += (-3, -1, 1, 3)[index % 4]
                                else:
                                    bad["proposed_after"] ^= 1 << (index % parsed["query"]["bits"])
                                process.write(bad)
                                process_n += 2
                    if family == "continuous_replan":
                        for stage in world["episode"]:
                            erng = random.Random(digest([salt, "episode", stage["stage"]]))
                            ep, em = render(stage["world"], erng, min(tier, 2), "distributed")
                            ep_parsed = parse_prompt(ep)
                            ec = audit_answer(ep_parsed)
                            if not verify_solution(ep_parsed, stage["reference"], ec)["correct"]:
                                raise AssertionError("Episode reference mismatch")
                            episode.write({"id": row_id + f"-stage{stage['stage']}", "group_id": row_id, "parent_id": row_id,
                                           "split": split, "stage": stage["stage"], "prompt": ep,
                                           "reference": stage["reference"], "certificate": ec,
                                           "observation": stage["observation"], "teacher_forced": True,
                                           "future_records_in_prompt": False})
                            episode_n += 1
                    if split == "train" and i == 0:
                        samples.append({"family": family, "id": row_id, "prompt": prompt, "reference": answer, "teacher_trace": trace})
                    stats["rows"] += 1
                    stats["families"][family] += 1
                    stats["statuses"][family + ":" + answer.get("status", "ANSWER")] += 1
                    stats["layouts"][layout["layout"]] += 1
                    stats["word_tiers"][str(layout["target_words"])] += 1
                    stats["words_min"] = min(stats["words_min"], layout["input_words"])
                    stats["words_max"] = max(stats["words_max"], layout["input_words"])
                    stats["words_total"] += layout["input_words"]
                    stats["target_records_min"] = min(stats["target_records_min"], layout["target_records"])
                    stats["target_records_max"] = max(stats["target_records_max"], layout["target_records"])
                    stats["trace_characters_total"] += len(trace)
                    stats["trace_steps_total"] += len(steps)
                print(f"{split}/{family}: {count} worlds", flush=True)
            counters[split] = stats
    write_json(root / "examples" / "train_samples.json", samples)
    report = {"version": "2.0.0", "seed": seed, "scale": scale, "independent_worlds": len(seen_cases),
              "splits": counters, "derived_process_labels": process_n, "derived_preference_pairs": preference_n,
              "derived_episode_decision_views": episode_n, "tokenizer_qualification": "NOT_RUN",
              "pretrained_model_calls": 0, "optimizer_steps": 0,
              "python": platform.python_version(), "synthetic_only": True,
              "derived_views_are_not_independent_worlds": True}
    write_json(root / "reports" / "build_report.json", report)
    # Data-only manifest, nonrecursive by construction; source release manifest is separate.
    entries = {str(p.relative_to(root)).replace("\\", "/"): {"sha256": sha_file(p), "bytes": p.stat().st_size}
               for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root / "DATA_MANIFEST.json", {"files": entries, "seed": seed, "scale": scale})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    report = build_data(args.output, seed=args.seed, scale=args.scale)
    print(dumps({"worlds": report["independent_worlds"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
