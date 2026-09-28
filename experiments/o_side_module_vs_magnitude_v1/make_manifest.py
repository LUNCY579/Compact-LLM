"""OSIDE-001 row manifest (PREREG.md section 2). Tokenizer only: no model, no forward, no score.

Rows, prompts and answers come from titan_opallo_skill_vector_v1/full_scale/o_reply_nll.py (imported, not
copied), so this is the same 6,741-row held-out pool that O_REPLY_NLL.json scored. For every row it records
the reply-token count, the final token id and whether that final token is TERMINAL punctuation (decoded text,
stripped, non-empty and made only of '.', '?', '!'). It also flags the rows that Opallo's three seed
replicates consumed in training; the seed gate (PREREG section 4, DG3) excludes them.

Output: ROWS_MANIFEST.json (create-only). No reply text is written.
"""
from __future__ import annotations

import hashlib, json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FS = EXP / "titan_opallo_skill_vector_v1" / "full_scale"
sys.path.insert(0, str(FS))
import o_reply_nll as O  # noqa: E402

EVID = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/"
            r"architecture_benchmark_001/evidence/retention-design-v1")
TOKENIZER_DIR = EXP / "titan_opallo_skill_vector_v1" / "arms" / "BASE"
SEEDS = (23, 29, 41)
OUT = HERE / "ROWS_MANIFEST.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arm_dose(seed: int) -> Path:
    return EVID / f"ARM_DOSE_B28last20_lr1e-05_b5200_seed{seed}_wsd0.2-0.2_b20.95.json"


def is_terminal(decoded: str) -> bool:
    t = decoded.strip()
    return bool(t) and all(c in ".?!" for c in t)


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    from transformers import AutoTokenizer
    rows = O.make_rows()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    consumed = {}
    for s in SEEDS:
        consumed[s] = set(json.loads(arm_dose(s).read_text(encoding="utf-8"))["train_row_ids_consumed"])
    union = set().union(*consumed.values())
    out_rows, finals = [], Counter()
    for r in rows:
        ids = tok(r["answer"], add_special_tokens=False)["input_ids"]
        last = tok.decode([ids[-1]])
        term = is_terminal(last)
        if term:
            finals[last.strip()] += 1
        out_rows.append({"id": r["id"], "reply_class": r["reply_class"], "n_tokens": len(ids),
                         "final_token_id": ids[-1], "terminal": term, "replicate_consumed": r["id"] in union})
    content_tokens = sum(x["n_tokens"] - x["terminal"] for x in out_rows)
    if min(x["n_tokens"] - x["terminal"] for x in out_rows) < 1:
        raise SystemExit("a row has no content token; the content readout is undefined for it")
    man = {
        "schema": 1, "experiment": "OSIDE-001",
        "source": "titan_opallo_skill_vector_v1/full_scale/o_reply_nll.py make_rows() and parse_reply_text()",
        "o_reply_nll_py_sha256": sha256(FS / "o_reply_nll.py"),
        "source_train_sha256": sha256(O.TRAIN), "consumed_arm_dose_sha256": sha256(O.ARM_DOSE),
        "tokenizer_json_sha256": sha256(TOKENIZER_DIR / "tokenizer.json"),
        "terminal_rule": "final reply token is TERMINAL iff tokenizer.decode([id]).strip() is non-empty and only '.', '?', '!'",
        "replicate_arm_dose_sha256": {str(s): sha256(arm_dose(s)) for s in SEEDS},
        "counts": {"rows": len(out_rows), "tokens": sum(x["n_tokens"] for x in out_rows),
                   "content_tokens": content_tokens, "terminal_rows": sum(x["terminal"] for x in out_rows),
                   "terminal_by_text": dict(sorted(finals.items())), "classes": len({x["reply_class"] for x in out_rows}),
                   "replicate_consumed_rows": sum(x["replicate_consumed"] for x in out_rows),
                   "replicate_consumed_by_seed": {str(s): len(consumed[s] & {x["id"] for x in out_rows}) for s in SEEDS}},
        "rows": out_rows,
    }
    OUT.write_bytes((json.dumps(man, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(man["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
