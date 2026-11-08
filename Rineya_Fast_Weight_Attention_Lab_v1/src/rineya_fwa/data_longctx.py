"""Non-ceilinged, token-exact, position-conditioned retrieval suite (P2b).

Why this exists
---------------
The P1/P2 suite (``data.py`` + ``eval_position.py``) could not discriminate
between arms. Three properties made it a ceiling:

1. the filler was ONE sentence repeated 21-42 times, so the needle was the only
   non-duplicated sentence in the context -- "spot the odd sentence out", not
   retrieval;
2. the answer space was 10 colour words, so a lucky prior gets you a long way;
3. ``target_length`` counted filler *sentences*, not tokens, so a "512" item was
   really ~990 tokens and no length was ever what it claimed.

Every arm scored 14/14 and the preregistered C-B contrast was untestable.

What this module does differently
---------------------------------
* **Real filler.** Prose is drawn from ``data/unsealed/`` -- the official CC0
  unsealed shard that the lab already built, hashed and committed but never
  actually used. It is natural, varied text, not one repeated sentence.
* **Decoys of identical surface form.** Every context carries ``n_decoys``
  further ``Vault <KEY> uses code <CODE>.`` sentences with distinct keys and
  distinct codes. The target sentence is therefore *not* distinguishable by
  form, only by its key. This is what forces actual key->value binding.
* **Token-exact lengths.** Contexts are assembled against the Supa tokenizer and
  the realised prompt length is recorded per item, so "4096" means 4096.
* **Single-token answers.** Codes are two-letter callsigns that are exactly one
  token in the bare (answer-position) form, so an item is scored by ONE forward
  pass: argmax over the candidate token ids at the answer position. That removes
  the malformed-output confound entirely -- the P2 run's only arm-A "miss" was
  the string "color slate" instead of "slate", which is a formatting artifact
  scored as a retrieval failure.

Nothing here reads a sealed source.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from rineya_fwa.paths import DATA, assert_unsealed_source

SYSTEM = (
    "You are answering a closed-context reading test. Use only the supplied context. "
    "Reply with the shortest exact answer and no explanation. If the context does not "
    "determine the answer, reply UNKNOWN."
)

FACT_TEMPLATE = "Vault {key} uses code {code}."
QUESTION_TEMPLATE = "What code does vault {key} use?"

# Two-hop indirection. Calibration showed the one-hop form leaves the frozen
# backbone at 0.89-1.00 even with 48 confusable decoys, because a single
# key->code lookup is a surface match the attention can do directly. The two-hop
# form requires joining two separate sentences, which is the binding operation a
# fast-weight memory is supposed to help with.
LINK_TEMPLATE = "Vault {key} is administered by unit {unit}."
UNIT_TEMPLATE = "Unit {unit} uses code {code}."

UNITS = (
    "Kestrel", "Marlin", "Aurora", "Basalt", "Cinder", "Dovetail", "Ember", "Fathom",
    "Granite", "Halcyon", "Ironwood", "Juniper", "Kelvin", "Lantern", "Meridian",
    "Nimbus", "Obsidian", "Pallas", "Quarry", "Ridgeline", "Sable", "Tundra",
    "Umbra", "Verdigris", "Wharf", "Xenon", "Yarrow", "Zephyr", "Alcove", "Bramble",
    "Cascade", "Driftwood", "Estuary", "Fernwood", "Gable", "Harrow", "Inlet",
    "Jetty", "Kiln", "Larkspur", "Mistral", "Nocturne", "Orchard", "Pinnacle",
    "Quill", "Rampart", "Sextant", "Thicket", "Undertow", "Vellum",
)

# Interference battery categories, per the experiment brief section 14.
INTERFERENCE_CATEGORIES = ("A", "B", "C", "D", "E", "F", "G")


def _stable_seed(*parts: Any) -> int:
    """Process-stable seed from arbitrary parts, including strings.

    ``tuple.__hash__()`` over a tuple containing str is randomised per process by
    PYTHONHASHSEED, so the previous seeding produced a DIFFERENT suite in every
    process for identical arguments. Verified: two fresh interpreters returned
    different gold answers and different target keys for the same item id.

    Consequence for the completed P2b run: its internal validity is intact,
    because a single process built the suite once and scored every arm against
    those same items -- the paired contrasts are sound. But the run is not
    reproducible, and the recorded suite fingerprint cannot be regenerated. See
    PREREGISTRATION Amendment 10.
    """
    blob = "".join(repr(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "big") & 0xFFFFFFFF


@lru_cache(maxsize=1)
def _tok() -> Any:
    """Process-local tokenizer cache.

    ``loading.load_tokenizer`` re-reads an 11 MB tokenizer.json on every call;
    suite construction asks for token ids tens of times per item.
    """
    from rineya_fwa.loading import load_tokenizer

    return load_tokenizer()


@lru_cache(maxsize=1)
def single_token_codes(tokenizer_id: int = 0) -> tuple[str, ...]:
    """Two-letter uppercase codes that are exactly one token in bare form."""
    tok = _tok()
    out = []
    for a, b in itertools.product(string.ascii_uppercase, repeat=2):
        word = a + b
        ids = tok.encode(word, add_special_tokens=False)
        if len(ids) == 1 and tok.decode(ids) == word:
            out.append(word)
    return tuple(out)


@lru_cache(maxsize=4096)
def code_token_id(code: str) -> int:
    ids = _tok().encode(code, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"code {code!r} is not a single token: {ids}")
    return ids[0]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def cc0_filler_sentences() -> tuple[str, ...]:
    """Natural filler prose from the unsealed CC0 shard.

    Falls back to nothing -- if the shard is missing we raise, because silently
    reverting to a single repeated sentence is exactly the defect this module
    exists to remove.
    """
    path = DATA / "unsealed" / "train.jsonl"
    assert_unsealed_source(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"unsealed CC0 shard missing at {path}; refusing to fall back to degenerate filler"
        )
    seen: set[str] = set()
    out: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for message in row.get("messages") or []:
                if message.get("role") != "user":
                    continue
                for sentence in _SENT_SPLIT.split(str(message.get("content") or "")):
                    s = " ".join(sentence.split())
                    # keep mid-length declarative prose; drop headers and stubs
                    if not (60 <= len(s) <= 300):
                        continue
                    if s.isupper() or s.startswith(("CONTEXT", "QUESTION", "ANSWER")):
                        continue
                    if "vault" in s.lower() or "code" in s.lower():
                        continue  # never let filler collide with the probe vocabulary
                    if s in seen:
                        continue
                    seen.add(s)
                    out.append(s)
    if len(out) < 200:
        raise RuntimeError(f"only {len(out)} usable filler sentences; suite would repeat itself")
    return tuple(out)


@lru_cache(maxsize=100_000)
def _cached_tlen(text: str) -> int:
    return len(_tok().encode(text, add_special_tokens=False))


@dataclass(frozen=True)
class Assembled:
    context: str
    n_context_tokens: int
    fact_token_offsets: dict[str, int]


def assemble_context(
    tokenizer: Any,
    *,
    facts: list[tuple[str, float]],
    target_tokens: int,
    rng: random.Random,
) -> Assembled:
    """Build a context of exactly ``target_tokens`` tokens with facts at fixed
    relative token positions.

    ``facts`` is a list of ``(sentence, relative_position)``. Relative position
    is a fraction of the *context* token budget.
    """
    pool = cc0_filler_sentences()
    order = list(range(len(pool)))
    rng.shuffle(order)

    tlen = _cached_tlen

    wanted = sorted(
        ((sent, max(0.0, min(1.0, pos))) for sent, pos in facts), key=lambda x: x[1]
    )
    targets = [(sent, int(round(pos * target_tokens))) for sent, pos in wanted]

    pieces: list[str] = []
    used = 0
    cursor = 0
    fact_offsets: dict[str, int] = {}
    next_fact = 0

    while used < target_tokens:
        # insert any fact whose target offset we have reached
        if next_fact < len(targets) and used >= targets[next_fact][1]:
            sent = targets[next_fact][0]
            n = tlen(sent + " ")
            if used + n > target_tokens:
                break
            pieces.append(sent)
            fact_offsets[sent] = used
            used += n
            next_fact += 1
            continue
        # Take the next filler sentence that fits the remaining budget rather
        # than stopping at the first one that does not. CC0 sentences run to
        # ~70 tokens, so breaking on the first overflow leaves up to ~13% of a
        # 512-token target unfilled and "token-exact" stops being true.
        room = target_tokens - used
        sent = None
        for probe in range(len(order)):
            candidate = pool[order[(cursor + probe) % len(order)]]
            if tlen(candidate + " ") <= room:
                sent = candidate
                cursor += probe + 1
                break
        if sent is None:
            break  # nothing in the pool fits; we are as close as we can get
        n = tlen(sent + " ")
        pieces.append(sent)
        used += n

    # any facts we never reached (target beyond the fill) go at the end
    while next_fact < len(targets):
        sent = targets[next_fact][0]
        pieces.append(sent)
        fact_offsets[sent] = used
        used += tlen(sent + " ")
        next_fact += 1

    text = " ".join(pieces)
    # Per-sentence token counts are not additive under joining -- tokenizers
    # merge across the separator -- so `used` drifts a few percent above the
    # true length. Top up against the encoded count so the realised context is
    # actually close to the target rather than only close by estimate.
    ids = tokenizer.encode(text, add_special_tokens=False)
    shortest = min(_cached_tlen(pool[i] + " ") for i in order[:64])
    for _ in range(32):
        room = target_tokens - len(ids)
        if room < shortest:
            break
        pick = None
        for probe in range(len(order)):
            candidate = pool[order[(cursor + probe) % len(order)]]
            if _cached_tlen(candidate + " ") <= room:
                pick = candidate
                cursor += probe + 1
                break
        if pick is None:
            break
        pieces.append(pick)
        text = " ".join(pieces)
        ids = tokenizer.encode(text, add_special_tokens=False)

    # trim to exact length from the tail, never cutting a fact
    if len(ids) > target_tokens:
        last_fact_end = max(
            (off + tlen(s + " ") for s, off in fact_offsets.items()), default=0
        )
        keep = max(target_tokens, last_fact_end)
        ids = ids[:keep]
        text = tokenizer.decode(ids)
    return Assembled(context=text, n_context_tokens=len(ids), fact_token_offsets=fact_offsets)


def _keys(rng: random.Random, n: int, *, mode: str = "random") -> list[str]:
    """Vault identifiers.

    ``random`` draws unrelated 4-digit ids. ``confusable`` draws them from one
    narrow numeric band so every key shares a prefix with the target and differs
    only in the last digit or two. Calibration showed the ``random`` mode leaves
    the frozen backbone at ceiling because the query key can be matched as a
    literal string; ``confusable`` removes that shortcut and forces the model to
    discriminate between near-identical identifiers.
    """
    if mode == "confusable":
        base = rng.randint(100, 989) * 10
        span = max(n * 3, 40)
        pool = [f"RX-{base + d}" for d in range(span)]
        rng.shuffle(pool)
        return pool[:n]
    out: set[str] = set()
    while len(out) < n:
        out.add(f"RX-{rng.randint(1000, 9999)}")
    return sorted(out)


def make_position_item(
    tokenizer: Any,
    *,
    idx: int,
    target_tokens: int,
    position: float,
    n_decoys: int,
    n_candidates: int,
    seed: int,
    split: str,
    key_mode: str = "confusable",
    hops: int = 1,
) -> dict[str, Any]:
    rng = random.Random(
        _stable_seed(seed, idx, target_tokens, round(position * 1000), split, key_mode, hops)
    )
    codes = list(single_token_codes())
    rng.shuffle(codes)
    keys = _keys(rng, n_decoys + 1, mode=key_mode)
    rng.shuffle(keys)
    target_key = keys[0]
    gold = codes[0]
    decoy_codes = codes[1 : 1 + n_decoys]

    if hops == 2:
        units = list(UNITS)
        rng.shuffle(units)
        n_link = max(1, n_decoys // 2)
        target_unit = units[0]
        # both decisive sentences sit at the swept position, a fixed gap apart,
        # so `position` still means "where the evidence is"
        facts = [
            (LINK_TEMPLATE.format(key=target_key, unit=target_unit), position),
            (UNIT_TEMPLATE.format(unit=target_unit, code=gold), min(1.0, position + 0.03)),
        ]
        # decoy vault->unit links
        for j, k in enumerate(keys[1 : 1 + n_link]):
            p = (j + 0.5) / n_link
            if abs(p - position) < 0.02:
                p = min(1.0, p + 0.04)
            facts.append((LINK_TEMPLATE.format(key=k, unit=units[1 + (j % (len(units) - 1))]), p))
        # decoy unit->code rows, so no unit is uniquely "the one with a code"
        for j, u in enumerate(units[1 : 1 + n_link]):
            p = (j + 0.25) / n_link
            if abs(p - position) < 0.02:
                p = min(1.0, p + 0.05)
            facts.append((UNIT_TEMPLATE.format(unit=u, code=decoy_codes[j % len(decoy_codes)]), p))
    else:
        facts = [(FACT_TEMPLATE.format(key=target_key, code=gold), position)]
        # decoys spread evenly over the context, avoiding the target's slot
        for j, (k, c) in enumerate(zip(keys[1:], decoy_codes)):
            p = (j + 0.5) / max(n_decoys, 1)
            if abs(p - position) < 0.02:
                p = min(1.0, p + 0.04)
            facts.append((FACT_TEMPLATE.format(key=k, code=c), p))

    asm = assemble_context(tokenizer, facts=facts, target_tokens=target_tokens, rng=rng)
    question = QUESTION_TEMPLATE.format(key=target_key)

    distractor_pool = [c for c in codes[1 + n_decoys :]]
    candidates = [gold] + decoy_codes + distractor_pool[: max(0, n_candidates - 1 - n_decoys)]
    candidates = candidates[:n_candidates]
    rng.shuffle(candidates)

    return {
        "id": f"LC-{target_tokens}-{position:.2f}-{split}-{idx:04d}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"CONTEXT\n{asm.context}\n\nQUESTION\n{question}\n\nANSWER",
            },
        ],
        "answer": gold,
        "gold_token_id": code_token_id(gold),
        "candidates": candidates,
        "candidate_token_ids": [code_token_id(c) for c in candidates],
        "position": position,
        "target_length": target_tokens,
        "n_context_tokens": asm.n_context_tokens,
        "n_decoys": n_decoys,
        "target_key": target_key,
        "split": split,
        "key_mode": key_mode,
        "hops": hops,
    }


def build_position_suite_v2(
    tokenizer: Any,
    *,
    lengths: list[int],
    positions: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    n_per_cell: int = 8,
    n_decoys: int = 24,
    n_candidates: int = 32,
    seed: int = 101,
    split: str = "eval",
    key_mode: str = "confusable",
    hops: int = 1,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    i = 0
    for length in lengths:
        for pos in positions:
            for _ in range(n_per_cell):
                items.append(
                    make_position_item(
                        tokenizer,
                        idx=i,
                        target_tokens=length,
                        position=pos,
                        n_decoys=n_decoys,
                        n_candidates=n_candidates,
                        seed=seed,
                        split=split,
                        key_mode=key_mode,
                        hops=hops,
                    )
                )
                i += 1
    return items


# ---------------------------------------------------------------------------
# Interference battery (brief section 14), n items per category
# ---------------------------------------------------------------------------


def make_interference_item(
    tokenizer: Any,
    *,
    category: str,
    idx: int,
    target_tokens: int,
    seed: int,
    n_candidates: int = 32,
) -> dict[str, Any]:
    rng = random.Random(_stable_seed(seed, category, idx, target_tokens))
    codes = list(single_token_codes())
    rng.shuffle(codes)
    keys = _keys(rng, 40)
    rng.shuffle(keys)
    target_key = keys[0]
    facts: list[tuple[str, float]] = []
    note = ""

    if category == "A":  # one important fact + filler
        gold = codes[0]
        facts = [(FACT_TEMPLATE.format(key=target_key, code=gold), 0.5)]
        note = "single fact, pure filler"
    elif category == "B":  # multiple unrelated important facts
        gold = codes[0]
        facts = [(FACT_TEMPLATE.format(key=target_key, code=gold), 0.5)]
        for j in range(7):
            facts.append((FACT_TEMPLATE.format(key=keys[j + 1], code=codes[j + 1]), (j + 1) / 9))
        note = "8 distinct keys, query one"
    elif category == "C":  # semantically similar facts (adjacent keys)
        base = rng.randint(1000, 9000)
        target_key = f"RX-{base}"
        gold = codes[0]
        facts = [(FACT_TEMPLATE.format(key=target_key, code=gold), 0.5)]
        for j in range(7):
            facts.append((FACT_TEMPLATE.format(key=f"RX-{base + j + 1}", code=codes[j + 1]), (j + 1) / 9))
        note = "near-identical keys, high confusability"
    elif category == "D":  # contradictory updates, later wins
        old, gold = codes[0], codes[1]
        facts = [
            (FACT_TEMPLATE.format(key=target_key, code=old), 0.25),
            (f"Correction: vault {target_key} now uses code {gold}, replacing {old}.", 0.75),
        ]
        note = "contradiction; gold is the later value"
    elif category == "E":  # repeated evidence
        gold = codes[0]
        facts = [(FACT_TEMPLATE.format(key=target_key, code=gold), p) for p in (0.2, 0.5, 0.8)]
        note = "same fact repeated three times"
    elif category == "F":  # early fact, many later distractors
        gold = codes[0]
        facts = [(FACT_TEMPLATE.format(key=target_key, code=gold), 0.03)]
        for j in range(12):
            facts.append((FACT_TEMPLATE.format(key=keys[j + 1], code=codes[j + 1]), 0.2 + j * 0.06))
        note = "decisive fact first, 12 distractors after"
    elif category == "G":  # late correction of an earlier fact
        old, gold = codes[0], codes[1]
        facts = [(FACT_TEMPLATE.format(key=target_key, code=old), 0.10)]
        for j in range(6):
            facts.append((FACT_TEMPLATE.format(key=keys[j + 1], code=codes[j + 2]), 0.2 + j * 0.09))
        facts.append((f"Update: vault {target_key} has been reassigned to code {gold}.", 0.95))
        note = "late correction after intervening facts"
    else:
        raise KeyError(category)

    asm = assemble_context(tokenizer, facts=facts, target_tokens=target_tokens, rng=rng)
    question = QUESTION_TEMPLATE.format(key=target_key)
    used = {gold} | {c for c in codes[:16]}
    candidates = [gold] + [c for c in codes if c not in {gold}][: n_candidates - 1]
    rng.shuffle(candidates)
    item = {
        "id": f"INT-{category}-{target_tokens}-{idx:03d}",
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"CONTEXT\n{asm.context}\n\nQUESTION\n{question}\n\nANSWER",
            },
        ],
        "answer": gold,
        "gold_token_id": code_token_id(gold),
        "candidates": candidates,
        "candidate_token_ids": [code_token_id(c) for c in candidates],
        "target_length": target_tokens,
        "n_context_tokens": asm.n_context_tokens,
        "target_key": target_key,
        "note": note,
    }
    if category in ("D", "G"):
        item["superseded_answer"] = old
        item["superseded_token_id"] = code_token_id(old)
    return item


def build_interference_battery_v2(
    tokenizer: Any,
    *,
    target_tokens: int = 2048,
    n_per_category: int = 8,
    seed: int = 202,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for category in INTERFERENCE_CATEGORIES:
        for i in range(n_per_category):
            items.append(
                make_interference_item(
                    tokenizer, category=category, idx=i, target_tokens=target_tokens, seed=seed
                )
            )
    return items


def build_train_rows(
    tokenizer: Any,
    *,
    n_rows: int,
    lengths: list[int],
    n_decoys: int = 24,
    seed: int = 909,
    hops: int = 1,
) -> list[dict[str, Any]]:
    """TRAIN split for the FWA adapters.

    Uses split='train' so the RNG stream, keys and codes are disjoint from the
    eval suite by construction. The assistant turn is the gold code, so the
    supervised token is exactly the retrieval decision.
    """
    rows: list[dict[str, Any]] = []
    positions = (0.05, 0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95)
    for i in range(n_rows):
        length = lengths[i % len(lengths)]
        pos = positions[i % len(positions)]
        item = make_position_item(
            tokenizer,
            idx=i,
            target_tokens=length,
            position=pos,
            n_decoys=n_decoys,
            n_candidates=2,
            seed=seed,
            split="train",
            hops=hops,
        )
        rows.append(
            {
                "id": item["id"],
                "messages": item["messages"] + [{"role": "assistant", "content": item["answer"]}],
                "metadata": {
                    "context_stratum": f"{length}",
                    "evidence_position": pos,
                    "source": "fwa_longctx_v2",
                    "split": "train",
                    "n_context_tokens": item["n_context_tokens"],
                    "n_decoys": n_decoys,
                    "hops": hops,
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    from rineya_fwa.paths import assert_readonly_target

    assert_readonly_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
