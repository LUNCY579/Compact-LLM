"""SMALLVD-001 Amendment 6 evidence -- REJECT rule 1 applied to the IDENTITY tokenizer (its missing
positive control), and the corrected rule 1': every candidate's round-trip equals the TEACHER's.

Registered rule 1 was "any HELD-OUT row fails decode(encode(s)) == NFC(s)". Its first application
rejected all four candidates on ONE row -- and the unpruned teacher fails that same row, so the rule
rejects the identity tokenizer and measures a normaliser property, not pruning. This script records,
create-only, beside the untouched VOCAB_CANDIDATES.json:
  - the teacher's own rule-1 failures (the positive control that was never run at registration)
  - per candidate: rows whose round-trip differs from the teacher's (rule 1'), and its rule-1 failures
usage: python vocab_rule1_teacher_equivalence.py <teacher_dir> <vocab_dir> <corpus.jsonl>
"""
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import tokenizers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import token_utility as tu  # noqa: E402


def main():
    teacher, vdir, corpus = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    held = tu.load_corpus(corpus, "heldout")
    texts = [r["text"] for r in held]
    old = tokenizers.Tokenizer.from_file(str(teacher / "tokenizer.json"))
    tdec = [old.decode(e.ids, skip_special_tokens=False) for e in old.encode_batch(texts, add_special_tokens=False)]
    nfc = [unicodedata.normalize("NFC", s) for s in texts]
    res = {"schema": 1, "record": "SMALLVD_VOCAB_RULE1_TEACHER_EQUIVALENCE", "experiment": "SMALLVD-001",
           "amendment": 6, "heldout_rows": len(held),
           "python_unicodedata_version": unicodedata.unidata_version, "tokenizers_version": tokenizers.__version__,
           "teacher_fails_registered_rule1": [r["id"] for r, t, n in zip(held, tdec, nfc) if t != n],
           "candidates": {}}
    for d in sorted(p for p in vdir.iterdir() if p.is_dir() and p.name[:1] == "V"):
        new = tokenizers.Tokenizer.from_file(str(d / "tokenizer.json"))
        ndec = [new.decode(e.ids, skip_special_tokens=False) for e in new.encode_batch(texts, add_special_tokens=False)]
        res["candidates"][d.name] = {
            "rule1_prime_rows_differing_from_teacher": [r["id"] for r, a, b in zip(held, ndec, tdec) if a != b],
            "registered_rule1_failures": [r["id"] for r, a, n in zip(held, ndec, nfc) if a != n],
            "tokenizer_sha256": hashlib.sha256((d / "tokenizer.json").read_bytes()).hexdigest()}
        c = res["candidates"][d.name]
        c["rule1_prime"] = "PASS" if not c["rule1_prime_rows_differing_from_teacher"] else "FAIL"
    out = vdir / "VOCAB_RULE1_TEACHER_EQUIVALENCE.json"
    with open(out, "x", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({"teacher_fails_rule1": res["teacher_fails_registered_rule1"],
                      **{k: (v["rule1_prime"], len(v["registered_rule1_failures"])) for k, v in res["candidates"].items()}}))


if __name__ == "__main__":
    main()
