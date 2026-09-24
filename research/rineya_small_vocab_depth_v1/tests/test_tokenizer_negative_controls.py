"""Negative controls: every tokenizer gate must be shown to FAIL on a deliberately broken candidate.

A passing invariant suite is only evidence if the same checks reject a candidate that violates
them. Each mutation below breaks exactly one property; the corresponding check must catch it.
"""
from __future__ import annotations

import copy
import json
import sys
import unicodedata
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vocab_prune as vp  # noqa: E402
from test_tokenizer import (FIXTURES, TEACHER, invariance_violations,  # noqa: E402
                            special_id_violations, teacher_tokenizer)

tokenizers = pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def base():
    files = vp.load_teacher_tokenizer_files(TEACHER)
    tj = files["tokenizer.json"]
    r, _ = vp.greedy_prune(tj["model"]["vocab"], vp.merges_of(tj), {}, set(), 65510)
    new_tj, id_map = vp.prune_tokenizer_json(tj, r)
    return files, new_tj, id_map


def roundtrip_failures(tj_obj) -> list[str]:
    tok = tokenizers.Tokenizer.from_str(json.dumps(tj_obj))
    bad = []
    for name, s in FIXTURES.items():
        try:
            got = tok.decode(tok.encode(s, add_special_tokens=False).ids, skip_special_tokens=False)
        except Exception:
            bad.append(name)
            continue
        if got != unicodedata.normalize("NFC", s):
            bad.append(name)
    return bad


def test_intact_candidate_passes(base):
    _, new_tj, _ = base
    assert roundtrip_failures(new_tj) == []


def test_removing_a_base_byte_breaks_roundtrip(base):
    """Drop one base byte symbol (and every merge/token built on it). A byte-level BPE with a
    missing base cannot represent that byte; the round-trip check must catch it."""
    _, new_tj, _ = base
    broken = copy.deepcopy(new_tj)
    victim = "Ġ"                       # byte-level symbol for the SPACE byte
    vocab = broken["model"]["vocab"]
    dead = {t for t in vocab if victim in t}
    broken["model"]["vocab"] = {t: i for t, i in vocab.items() if t not in dead}
    broken["model"]["merges"] = [m for m in broken["model"]["merges"]
                                 if victim not in m[0] and victim not in m[1]]
    assert roundtrip_failures(broken), "round-trip check failed to detect a missing base byte"


def test_unclosed_merge_set_is_refused(base):
    files, _, _ = base
    tj = files["tokenizer.json"]
    vocab = tj["model"]["vocab"]
    parents, _, _ = vp.defining_parents(vocab, vp.merges_of(tj))
    t = max(parents, key=lambda x: len(x) if x.isascii() else 0)
    with pytest.raises(ValueError):
        vp.prune_tokenizer_json(tj, set(vocab) - {parents[t][1]})


def test_special_id_corruption_is_detected(base):
    """Swap the LIST POSITIONS of <|im_start|> and <|im_end|>.

    Swapping their "id" FIELDS is not a usable control: tokenizers 0.21.4 ignores that field and
    assigns added-token ids by list position after the model vocab. The first version of this
    control swapped the fields and PASSED A CORRUPTED FILE -- the corruption never took effect.
    Measured: swap id fields -> (151644, 151645); swap list entries -> (151645, 151644)."""
    _, new_tj, id_map = base
    tok = tokenizers.Tokenizer.from_str(json.dumps(new_tj))
    assert special_id_violations(tok, id_map) == []
    broken = copy.deepcopy(new_tj)
    L = broken["added_tokens"]
    i = next(k for k, a in enumerate(L) if a["content"] == "<|im_start|>")
    j = next(k for k, a in enumerate(L) if a["content"] == "<|im_end|>")
    L[i], L[j] = L[j], L[i]
    bad = special_id_violations(tokenizers.Tokenizer.from_str(json.dumps(broken)), id_map)
    assert bad, "swapped special tokens were not detected"


def test_out_of_order_added_tokens_are_refused():
    files = vp.load_teacher_tokenizer_files(TEACHER)
    tj = copy.deepcopy(files["tokenizer.json"])
    L = tj["added_tokens"]
    L[1], L[2] = L[2], L[1]
    with pytest.raises(ValueError):
        vp.prune_tokenizer_json(tj, set(tj["model"]["vocab"]))


def test_non_monotone_map_is_refused(tmp_path, base):
    files, new_tj, id_map = base
    m = dict(id_map)
    a, b = sorted(m)[:2]
    m[a], m[b] = m[b], m[a]
    d = tmp_path / "bad"
    d.mkdir()
    (d / "token_id_map.json").write_text(json.dumps({"map": {str(k): v for k, v in m.items()}}),
                                         encoding="utf-8")
    import surgery  # noqa: E402  (imports torch; the check itself is pure)
    with pytest.raises(ValueError):
        surgery.load_id_map(d)


def test_a_non_invariant_tokenizer_is_caught():
    """The invariance check must be able to fail. Same token set, REVERSED merge priority: the
    identity id map is valid, so every piece is eligible, and segmentations must differ. (Swapping
    two early merges was tried first and changed NOTHING on the fixtures -- a control too weak to
    fail is not a control.)"""
    files = vp.load_teacher_tokenizer_files(TEACHER)
    tj = copy.deepcopy(files["tokenizer.json"])
    tj["model"]["merges"] = tj["model"]["merges"][::-1]
    tok = tokenizers.Tokenizer.from_str(json.dumps(tj))
    identity = {i: i for i in range(151669)}
    checked, bad = invariance_violations(tok, identity, list(FIXTURES.values()))
    assert checked >= 10 and bad > 0, (checked, bad)
