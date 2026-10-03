"""Export training-friendly views without verifier/reference leakage into prompts."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any
from .common import read_jsonl, Writer, dumps
from .text import RULES


def export(root: Path, output: Path, kind: str, split: str, allow_heldout: bool = False) -> int:
    if split not in ("train", "validation", "test_iid", "test_transfer", "test_length"):
        raise ValueError("Unknown split")
    if split.startswith("test") and not allow_heldout:
        raise ValueError("Held-out export requires explicit --allow-heldout for evaluation only")
    if kind in ("sft_final", "sft_reason", "preferences", "process") and split not in ("train", "validation"):
        raise ValueError("Test labels cannot be exported as training targets")
    if kind in ("preferences", "process") and split != "train":
        raise ValueError("These derived views are TRAIN-only")
    if output.exists():
        raise FileExistsError("Output exists")
    try:
        with Writer(output) as writer:
            if kind in ("rl", "rl_reason", "sft_final", "sft_reason"):
                for row in read_jsonl(root / "data" / kind / f"{split}.jsonl.gz"):
                    writer.write(row)
            elif kind == "episodes":
                for e in read_jsonl(root / "episodes" / f"{split}.jsonl.gz"):
                    writer.write({k: e[k] for k in ("id", "group_id", "split", "stage", "prompt")})
            elif kind == "preferences":
                pairs = defaultdict(list)
                for pair in read_jsonl(root / "preferences" / "train_pairs.jsonl.gz"):
                    pairs[pair["parent_id"]].append(pair)
                for row in read_jsonl(root / "data" / "rl" / "train.jsonl.gz"):
                    for pair in pairs.pop(row["id"], []):
                        writer.write({"id": pair["id"], "group_id": row["group_id"], "prompt": row["prompt"],
                                      "chosen": [{"role": "assistant", "content": pair["chosen"]}],
                                      "rejected": [{"role": "assistant", "content": pair["rejected"]}]})
                if pairs:
                    raise ValueError("Preference parent missing")
            elif kind == "process":
                for row in read_jsonl(root / "process" / "train_transitions.jsonl.gz"):
                    system = "Check the proposed single state transition under the given rules. Return only {\"valid\":true} or {\"valid\":false}."
                    user = RULES[row["family"]] + "\nFor this local check only, the supplied before state is authoritative and the event is the next event to apply. Ignore the original final-question output format.\n" + dumps({k: row[k] for k in ("before", "event", "proposed_after")})
                    writer.write({"id": row["id"], "group_id": row["group_id"], "prompt": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                  "completion": [{"role": "assistant", "content": dumps({"valid": row["valid"]})}]})
            else:
                raise ValueError("Unknown export kind")
            count = writer.count
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["rl", "rl_reason", "sft_final", "sft_reason", "preferences", "process", "episodes"], required=True)
    p.add_argument("--split", default="train"); p.add_argument("--allow-heldout", action="store_true")
    a = p.parse_args()
    count = export(a.root, a.output, a.kind, a.split, a.allow_heldout)
    print(dumps({"rows": count, "output": str(a.output)}))


if __name__ == "__main__":
    main()
