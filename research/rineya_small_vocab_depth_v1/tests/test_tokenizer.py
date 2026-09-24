"""Tokenizer invariants (brief Phase 19, "Tokenizer").

Runs against (a) a candidate built fresh inside the test from the teacher with a corpus-free
utility, so the suite needs no artifacts, and (b) every real candidate directory listed in the
SMALLVD_TOKENIZER_DIRS environment variable (os.pathsep-separated). The real candidates are the
ones that matter; (a) exists so the suite can fail on a clean checkout.

Run:  .venv-rab001-cu126/Scripts/python.exe -B -m pytest -p no:cacheprovider tests/test_tokenizer.py
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import protected_tokens as pt  # noqa: E402
import vocab_prune as vp  # noqa: E402

TEACHER = Path(os.environ.get(
    "SMALLVD_TEACHER",
    r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze"))
tokenizers = pytest.importorskip("tokenizers")

FIXTURES = {
    "english_long": ("The committee reconvened on Thursday afternoon to reconsider the proposal. "
                     "Nevertheless, several members expressed reservations about its feasibility, "
                     "arguing that the implementation timeline was unrealistic. ") * 12,
    "python": "def fib(n: int) -> int:\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\n"
              "print({k: v for k, v in zip('abc', range(3))})  # dict comp\n",
    "typescript": "export const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);\n"
                  "if (x !== null && y?.z ?? false) { console.log(`val=${x}`); }\n",
    "json": '{"name": "Rineya", "version": [1, 7, 0], "tags": ["a", "b"], "nested": {"ok": true, "n": null}}',
    "markdown": "# Title\n\n## Section\n\n- item **bold** and `code`\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nx = 1\n```\n",
    "math": "Let f(x) = 3x^2 + 2x - 5. Then f'(x) = 6x + 2, and ∫₀¹ f(x) dx = 1 + 1 - 5 = -3. "
            "For all ε > 0 there exists δ ≥ 0 with |x − a| < δ ⇒ |f(x) − f(a)| < ε. π ≈ 3.14159, √2 ≈ 1.414.",
    "numbers_dates": "On 2026-09-24 at 14:05:59 UTC, 1,234,567.89 units shipped (±0.5%); ID #00042, v3.10.12.",
    "urls_paths": "See https://github.com/LUNCY579/ChatBubble/blob/main/README.md?plain=1#L10 and "
                  "C:\\Users\\clclc\\BusinessProject\\x.py or ~/projects/rineya/src/main.ts",
    "accented": "José García, François Müller, Zoë Brontë, Ångström, São Paulo, naïve café résumé, "
                "Dvořák, Łódź, Øresund, Straße, crème brûlée, jalapeño.",
    "cjk_and_emoji": "中文测试。日本語のテキスト。한국어 문장. Emoji: 😀🚀👍🏽 and ZWJ 👩‍💻.",
    "whitespace": "a\tb\n\n\n    indented\r\n\u00a0nbsp  two  spaces\n\t\t\ttabs\n" + " " * 33 + "end",
    "control_and_odd": "\x00\x01\x7f\u200b\u2028\ufeffzero-width and BOM \U0001F9EA",
    "special_in_content": "user wrote <|im_start|> and <think> literally, then <|im_end|>",
    "punct_heavy": "!!!??? ... --- *** ((())) [[[ ]]] {{{ }}} <<< >>> ;;; ::: ~~~ ^^^ %%% $$$ @@@ ###",
}


def teacher_tokenizer():
    return tokenizers.Tokenizer.from_file(str(TEACHER / "tokenizer.json"))


@pytest.fixture(scope="session")
def fresh_candidate(tmp_path_factory):
    """A 65,536-row candidate built with ZERO corpus utility (teacher merge order as the only
    prior) and the structural protections. Exercises the machinery, not the selection."""
    files = vp.load_teacher_tokenizer_files(TEACHER)
    tj = files["tokenizer.json"]
    vocab, merges = tj["model"]["vocab"], vp.merges_of(tj)
    tok = teacher_tokenizer()
    chat_ids = set()
    for s in chat_render_samples():
        chat_ids |= set(tok.encode(s, add_special_tokens=False).ids)
    inv = {v: k for k, v in vocab.items()}
    chat_tokens = {inv[i] for i in chat_ids if i in inv}
    prot = pt.build_protected(vocab, chat_tokens, {})
    protected = set().union(*prot["sets"].values())
    target = vp.rows_for(65536, len(tj["added_tokens"]))
    retained, _ = vp.greedy_prune(vocab, merges, {}, protected, target)
    new_tj, id_map = vp.prune_tokenizer_json(tj, retained)
    out = tmp_path_factory.mktemp("cand") / "tok"
    vp.write_tokenizer_dir(out, files, new_tj, id_map, 65536)
    return out


def chat_render_samples():
    return ["<|im_start|>system\nYou are Rineya.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>\n",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call><tool_response>ok</tool_response>"]


def special_id_violations(tok, id_map) -> list[str]:
    """Every added token must map to its remapped id and encode to exactly that single id."""
    bad = []
    for a in json.loads((TEACHER / "tokenizer.json").read_text(encoding="utf-8"))["added_tokens"]:
        new_id = id_map[a["id"]]
        if tok.token_to_id(a["content"]) != new_id:
            bad.append(f"token_to_id({a['content']})")
        if tok.encode(a["content"], add_special_tokens=False).ids != [new_id]:
            bad.append(f"encode({a['content']})")
    return bad


def invariance_violations(tok, id_map, texts) -> tuple[int, int]:
    """(pieces checked, pieces whose retained-only teacher encoding was NOT reproduced)."""
    old = teacher_tokenizer()
    checked = bad = 0
    for s in texts:
        for piece in s.split("\n"):
            oi = old.encode(piece, add_special_tokens=False).ids
            if oi and all(i in id_map for i in oi):
                checked += 1
                bad += tok.encode(piece, add_special_tokens=False).ids != [id_map[i] for i in oi]
    return checked, bad


def candidate_dirs(fresh):
    extra = [Path(p) for p in os.environ.get("SMALLVD_TOKENIZER_DIRS", "").split(os.pathsep) if p]
    return [fresh] + extra


def load_candidate(d: Path):
    tok = tokenizers.Tokenizer.from_file(str(d / "tokenizer.json"))
    id_map = {int(k): v for k, v in json.loads((d / "token_id_map.json").read_text(encoding="utf-8"))["map"].items()}
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return tok, id_map, cfg


def test_roundtrip_exact(fresh_candidate):
    for d in candidate_dirs(fresh_candidate):
        tok, _, _ = load_candidate(d)
        old = teacher_tokenizer()
        for name, s in FIXTURES.items():
            ids = tok.encode(s, add_special_tokens=False).ids
            got = tok.decode(ids, skip_special_tokens=False)
            want = old.decode(old.encode(s, add_special_tokens=False).ids, skip_special_tokens=False)
            assert got == want, f"{d.name}:{name} round-trip differs from the teacher's"
            assert got == unicodedata.normalize("NFC", s), f"{d.name}:{name} not NFC-reversible"


def test_ids_in_range_and_specials(fresh_candidate):
    for d in candidate_dirs(fresh_candidate):
        tok, id_map, cfg = load_candidate(d)
        rows = cfg["vocab_size"]
        assert tok.get_vocab_size(with_added_tokens=True) == rows == len(id_map)
        for s in FIXTURES.values():
            assert max(tok.encode(s, add_special_tokens=False).ids) < rows
        assert special_id_violations(tok, id_map) == []
        # no REGULAR token decodes to a special string (accidental collision)
        specials = {a["content"] for a in json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))["added_tokens"]}
        for t in json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]:
            assert pt.token_bytes(t).decode("utf-8", "replace") not in specials


def test_special_ids_in_configs(fresh_candidate):
    for d in candidate_dirs(fresh_candidate):
        tok, id_map, cfg = load_candidate(d)
        gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8"))
        tcfg = json.loads((TEACHER / "config.json").read_text(encoding="utf-8"))
        tgen = json.loads((TEACHER / "generation_config.json").read_text(encoding="utf-8"))
        assert cfg["eos_token_id"] == id_map[tcfg["eos_token_id"]]
        assert cfg["bos_token_id"] == id_map[tcfg["bos_token_id"]]
        assert gen["eos_token_id"] == [id_map[x] for x in tgen["eos_token_id"]]
        assert gen["pad_token_id"] == id_map[tgen["pad_token_id"]]
        assert tok.id_to_token(cfg["eos_token_id"]) == "<|im_end|>"
        tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
        ttc = json.loads((TEACHER / "tokenizer_config.json").read_text(encoding="utf-8"))
        assert tc["chat_template"] == ttc["chat_template"], "chat template must be byte-identical"
        for k, v in tc["added_tokens_decoder"].items():
            assert tok.id_to_token(int(k)) == v["content"]


def test_retained_only_text_encodes_identically(fresh_candidate):
    """THE INVARIANCE THEOREM: if the teacher's encoding of a text uses only retained tokens, the
    pruned tokenizer reproduces it exactly (mapped). Because every merge used in the teacher's
    derivation is an ancestor of a retained token, all of them survive, and BPE chooses the
    lowest-rank applicable merge from a subset that contains the teacher's choice."""
    old = teacher_tokenizer()
    for d in candidate_dirs(fresh_candidate):
        tok, id_map, _ = load_candidate(d)
        n_checked = 0
        for s in list(FIXTURES.values()) + chat_render_samples():
            for piece in s.split("\n"):
                oi = old.encode(piece, add_special_tokens=False).ids
                if oi and all(i in id_map for i in oi):
                    assert tok.encode(piece, add_special_tokens=False).ids == [id_map[i] for i in oi]
                    n_checked += 1
        assert n_checked >= 10, "the invariance test checked too few pieces to mean anything"


def test_chat_template_tokens_all_retained(fresh_candidate):
    old = teacher_tokenizer()
    for d in candidate_dirs(fresh_candidate):
        tok, id_map, _ = load_candidate(d)
        for s in chat_render_samples():
            oi = old.encode(s, add_special_tokens=False).ids
            assert all(i in id_map for i in oi), "a chat-template token was pruned"
            assert tok.encode(s, add_special_tokens=False).ids == [id_map[i] for i in oi]


def test_mapping_monotone_and_deterministic(fresh_candidate, tmp_path):
    for d in candidate_dirs(fresh_candidate):
        _, id_map, _ = load_candidate(d)
        olds = sorted(id_map)
        assert [id_map[o] for o in olds] == list(range(len(olds)))
    files = vp.load_teacher_tokenizer_files(TEACHER)
    tj = files["tokenizer.json"]
    hashes = []
    for i in range(2):
        r, _ = vp.greedy_prune(tj["model"]["vocab"], vp.merges_of(tj), {"Ġthe": 5.0}, set(), 60000)
        nt, m = vp.prune_tokenizer_json(tj, r)
        hashes.append((vp.sha256_bytes(vp.canonical_json_bytes(nt)), tuple(sorted(m.items()))))
    assert hashes[0] == hashes[1]


def test_transformers_loads_candidate(fresh_candidate):
    tr = pytest.importorskip("transformers")
    for d in candidate_dirs(fresh_candidate):
        tok = tr.AutoTokenizer.from_pretrained(str(d), local_files_only=True, trust_remote_code=False)
        ttok = tr.AutoTokenizer.from_pretrained(str(TEACHER), local_files_only=True, trust_remote_code=False)
        msgs = [{"role": "system", "content": "You are Rineya."}, {"role": "user", "content": "Name a prime."}]
        a = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        b = ttok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert a == b and a.endswith("<think>\n\n</think>\n\n")
        _, id_map, cfg = load_candidate(d)
        ids_new = tok(a, add_special_tokens=False)["input_ids"]
        ids_old = ttok(a, add_special_tokens=False)["input_ids"]
        assert ids_new == [id_map[i] for i in ids_old]
        assert len(tok) == cfg["vocab_size"]


def test_closure_violation_is_refused():
    files = vp.load_teacher_tokenizer_files(TEACHER)
    tj = files["tokenizer.json"]
    vocab = tj["model"]["vocab"]
    parents, _, _ = vp.defining_parents(vocab, vp.merges_of(tj))
    t = next(t for t in parents if t.startswith("Ġthe") and len(t) > 5)
    bad = set(vocab) - {parents[t][0]}
    with pytest.raises(ValueError):
        vp.prune_tokenizer_json(tj, bad)


def test_protected_never_removed():
    files = vp.load_teacher_tokenizer_files(TEACHER)
    tj = files["tokenizer.json"]
    vocab = tj["model"]["vocab"]
    keep = {pt.text_to_token(" photosynthesis")} & set(vocab) or {pt.text_to_token(" hello")}
    r, _ = vp.greedy_prune(vocab, vp.merges_of(tj), {}, keep, 20000)
    assert keep <= r
    parents, _, bases = vp.defining_parents(vocab, vp.merges_of(tj))
    assert set(bases) <= r and vp.is_closed(r, parents)
