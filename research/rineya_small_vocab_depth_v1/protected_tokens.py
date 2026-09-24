"""SMALLVD-001 Phase 4 -- the protected token set, and the evidence-based Unicode policy.

A protected token can never be removed, and because pruning is closure-respecting, neither can any
token in its merge ancestry. Protection is therefore stronger than it looks. The categories
below are listed with their REASON, because a protection rule that cannot say what it protects
against is just a thumb on the scale.

STRUCTURAL (unconditional -- the tokenizer or the chat contract breaks without them)
  base_bytes       256 byte symbols. Byte coverage and exact reversibility of ANY input.
  added_tokens     all 26 added/special tokens (<|endoftext|>, <|im_start|>, <|im_end|>, <think>,
                   </think>, <tool_call>, fim/vision tokens ...). They live outside the BPE vocab
                   and are kept whole, with remapped ids.
  chat_template    every regular token emitted when the real chat template is rendered (roles,
                   the injected "<think>\\n\\n</think>\\n\\n" prefix, tool-call scaffolding).
  whitespace       every whitespace-only token (425). Indentation runs and newline clusters are
                   what code and Markdown are made of; losing "        " multiplies code length.

OUTPUT CONTRACT (unconditional -- a family behaviour depends on the exact token)
  contract         the literal abstention answer "UNKNOWN" (id 51378, also Supa-A1.1's
                   abstain_token_id) and " UNKNOWN" (61395). PRIOR ART TRAP: RineyaBeta's R2 32K
                   trim dropped exactly these ids (merge rank 51122, never observed in its corpus),
                   silently breaking the literal-abstention contract, and R2 hybrids then scored
                   0.0 on unanswerables. A frequency rule cannot see a contract token; a contract
                   list can.

LEXICAL (unconditional lists -- existence in the vocab is checked, absence is reported)
  code_operators   multi-character operators across Python/JS/TS/C-family/shell.
  json_markdown    JSON and Markdown syntax tokens.
  url_path         URL and path components.

EVIDENCE-BASED (conditional on the calibration corpus -- the brief forbids whole-range rules)
  unicode_observed a single-codepoint non-ASCII token is protected iff it OCCURS in the
                   calibration FIT split at all. The corpus deliberately contains accented Latin
                   names, Greek/maths/science notation, typographic punctuation and currency, so
                   the ones that are actually used are kept and the ones that are not compete on
                   utility like any other token.
  ascii_punct_observed an ASCII punctuation/symbol token is protected iff it occurs >= MIN_PUNCT
                   times in the code, structured-data or Markdown categories.

Everything else competes on mixture-weighted utility (vocab_prune.greedy_prune).
"""
from __future__ import annotations

from functools import lru_cache

CODE_OPERATORS = [
    "==", "!=", "<=", ">=", "->", "=>", "::", "**", "//", "&&", "||", "+=", "-=", "*=", "/=",
    "%=", "++", "--", "<<", ">>", "===", "!==", "?.", "??", ":=", "**=", "//=", "<<=", ">>=",
    "&=", "|=", "^=", "...", "..", "#!", "/*", "*/", "<!--", "-->", "</", "/>", "${", "@{",
    "()", "[]", "{}", "();", "):", "():", "):\n", "();\n", "});", "})", "]),", "])", "',", "',\n",
    '",', '",\n', "'),", '"),', ".__", "__", "->\n", "<>", "!!", "~/", "./", "../", "|>",
]
JSON_MARKDOWN = [
    '{"', '":', '":"', '","', '"}', '"},', '},', '[{', '}]', '":{', '":[', '"]', '],', ': "',
    "##", "###", "####", "```", "```\n", "**", "__", "---", "***", "- ", "* ", "> ", "|", "||",
    "|-", "-|", ":|", "|:", "![", "](", "[^", "`", "``",
]
URL_PATH = [
    "http", "https", "://", "www", ".com", ".org", ".net", ".io", ".edu", ".gov", ".html",
    ".py", ".js", ".ts", ".json", ".md", ".txt", ".csv", ".yaml", ".yml", ".toml", ".png",
    ".jpg", ".pdf", ".git", "github", "localhost", "C:", "\\\\", "src", "lib", "usr", "bin",
    "home", "tmp", "api", "?", "&", "=", "%20", "#",
]
CONTRACT = ["UNKNOWN", "Unknown", "unknown"]
CONTRACT_IDS = {51378: "UNKNOWN", 61395: "ĠUNKNOWN"}
MIN_PUNCT = 3


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict:
    """Inverse of the GPT-2 / Qwen2 byte-level alphabet."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict:
    return {b: c for c, b in unicode_to_bytes().items()}


def token_bytes(tok: str) -> bytes:
    m = unicode_to_bytes()
    return bytes(m[c] for c in tok)


def text_to_token(s: str) -> str:
    m = bytes_to_unicode()
    return "".join(m[b] for b in s.encode("utf-8"))


def byte_class(tok: str) -> str:
    """Descriptive class used for REPORTING only -- never as a deletion rule."""
    b = token_bytes(tok)
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return "partial_utf8"
    if s and all(c in " \t\n\r\x0b\x0c" for c in s):
        return "whitespace"
    if all(ord(c) < 128 for c in s):
        core = s.strip(" \t\n\r")
        if core.isalpha():
            return "ascii_alpha"
        if core and all(not c.isalnum() for c in core):
            return "ascii_punct"
        return "ascii_mixed"
    visible = [c for c in s if not c.isspace()]
    if len(visible) == 1:
        return "single_codepoint_nonascii"
    return "multi_codepoint_nonascii"


def lexical_candidates(strings: list[str]) -> list[str]:
    """Each listed string with and without a leading space, as byte-level token strings."""
    out = []
    for s in strings:
        out.append(text_to_token(s))
        out.append(text_to_token(" " + s))
    return out


def build_protected(vocab: dict, chat_template_tokens: set, fit_counts_by_cat: dict,
                    code_like_categories: tuple = ("python", "javascript", "json", "markdown")) -> dict:
    """Return {category: set(tokens)} plus a report of listed strings absent from the vocab.

    `fit_counts_by_cat[cat][token] = count` on the calibration FIT split (teacher tokenisation)."""
    prot = {}
    prot["whitespace"] = {t for t in vocab if byte_class(t) == "whitespace"}
    prot["chat_template"] = set(chat_template_tokens) & set(vocab)
    absent = {}
    for name, lst in (("code_operators", CODE_OPERATORS), ("json_markdown", JSON_MARKDOWN),
                      ("url_path", URL_PATH)):
        cands = lexical_candidates(lst)
        prot[name] = {t for t in cands if t in vocab}
        absent[name] = sorted(token_bytes(t).decode("utf-8", "replace") for t in cands if t not in vocab)
    prot["contract"] = {t for t in lexical_candidates(CONTRACT) if t in vocab}
    inv = {i: t for t, i in vocab.items()}
    for i, t in CONTRACT_IDS.items():
        if inv.get(i) != t:
            raise ValueError(f"contract token id {i} is {inv.get(i)!r}, expected {t!r}")
        prot["contract"].add(t)
    seen_any = set()
    for cat, cnt in fit_counts_by_cat.items():
        seen_any |= {t for t, c in cnt.items() if c > 0}
    prot["unicode_observed"] = {t for t in seen_any if t in vocab and byte_class(t) == "single_codepoint_nonascii"}
    punct = {}
    for cat in code_like_categories:
        for t, c in fit_counts_by_cat.get(cat, {}).items():
            if byte_class(t) == "ascii_punct":
                punct[t] = punct.get(t, 0) + c
    prot["ascii_punct_observed"] = {t for t, c in punct.items() if c >= MIN_PUNCT and t in vocab}
    return {"sets": prot, "absent_from_vocab": absent}
