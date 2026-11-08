"""Stdlib semantic mechanics for a proposed long-context DEV sentinel.

No tokenizer, model, production export or protected-bank reader lives here.
Role assignment precedes rendering/solving. Numeric mass and cycle facts define
sampled worlds; arbitrary names, answer symbols and presentation do not. These
worlds share two mechanism classes: their count is not proof of independence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import random
import re
from typing import Any

VERSION = "rineya-long-context-semantic-v1"
VERIFIER_VERSION = "rineya-long-context-visible-verifier-v1"
SCORER_VERSION = "rineya-long-context-exact-symbol-v1"
FAMILIES = ("three_fact_keyed_join", "latest_authorized_record_selection")
POSITIONS = ("early", "middle", "late", "distributed")
LENGTHS = (2024, 4072)
ROLES = ("TRAIN", "TRAIN_SELECTION", "DEV", "CHECK", "FINAL")
ROLE_BUCKETS = ((7000, "TRAIN"), (7800, "TRAIN_SELECTION"), (8600, "DEV"),
                (9400, "CHECK"), (10000, "FINAL"))
ROLE_DOMAIN = b"rineya-long-context-semantic-role-v1\0"
FILLER_WORDS = ("archive", "aside", "quiet", "routine", "record", "detail",
                "plain", "page", "text", "ink")
FILLER_TEXTS = ("NOTE.", "NOTE archive aside.", "NOTE quiet routine record.",
                "NOTE plain page text ink detail.")
SYSTEM_PROMPT = (
    "Use only CONTEXT records. MASS gives a consignment's numeric mass; ROUTES "
    "lists inclusive mass intervals and their depots; SEALS maps depots to symbols. "
    "For a routed seal, find the interval containing the mass, then its depot's seal. "
    "AUTH identifies the authorized writer. CAT entries are revision:cycle:writer; "
    "VAL maps revisions to symbols. For a latest authorized record, choose the "
    "authorized writer's greatest cycle, then that revision's value. Other keys "
    "and unauthorized writers do not count. ARCHIVE records are obsolete and "
    "UNSIGNED records are unauthorized; both are nonbinding and cannot supply "
    "any required evidence. Every dependency must be explicit. "
    "NOTE lines carry no evidence. Return only the case-sensitive symbol, or "
    "UNKNOWN if required evidence is missing or ambiguous."
)
_NAME = r"[A-Z][A-Z0-9]{0,15}"
_SYMBOL = r"S[A-Z]{5}"


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _int(value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside semantic contract")
    return value


def _name(value: Any) -> str:
    if type(value) is not str or re.fullmatch(_NAME, value) is None:
        raise ValueError("invalid literal name")
    return value


def _symbol(value: Any) -> str:
    if type(value) is not str or re.fullmatch(_SYMBOL, value) is None:
        raise ValueError("invalid answer symbol")
    return value


def canonicalize(spec: dict) -> dict:
    if type(spec) is not dict or spec.get("family") not in FAMILIES:
        raise ValueError("unknown family")
    fields = ({"family", "subject", "mass", "routes", "seals"}
              if spec["family"] == FAMILIES[0]
              else {"family", "subject", "authority", "catalog", "values"})
    if set(spec) != fields:
        raise ValueError("unexpected semantic world fields")
    _name(spec["subject"])
    out = {"family": spec["family"], "subject": spec["subject"]}
    if spec["family"] == FAMILIES[0]:
        _int(spec["mass"], 1, 99)
        if type(spec["routes"]) is not list or type(spec["seals"]) is not list:
            raise ValueError("route and seal tables required")
        routes, seals = [], []
        for row in spec["routes"]:
            if type(row) is not dict or set(row) != {"low", "high", "depot"}:
                raise ValueError("invalid route entry")
            low, high = _int(row["low"], 1, 99), _int(row["high"], 1, 99)
            if low > high:
                raise ValueError("reversed mass interval")
            routes.append({"low": low, "high": high, "depot": _name(row["depot"])})
        for row in spec["seals"]:
            if type(row) is not dict or set(row) != {"depot", "symbol"}:
                raise ValueError("invalid seal entry")
            seals.append({"depot": _name(row["depot"]), "symbol": _symbol(row["symbol"])})
        # Identical duplicate/order variants normalize to the same world.
        routes = sorted({json_bytes(row): row for row in routes}.values(), key=lambda r: (r["low"], r["high"]))
        seals = sorted({json_bytes(row): row for row in seals}.values(), key=lambda r: r["depot"])
        if len(routes) != 3 or len(seals) != 3:
            raise ValueError("three route/seal entries required")
        if (len({r["depot"] for r in routes}) != 3
                or {r["depot"] for r in routes} != {r["depot"] for r in seals}
                or len({r["symbol"] for r in seals}) != 3
                or any(a["high"] >= b["low"] for a, b in zip(routes, routes[1:]))
                or sum(r["low"] <= spec["mass"] <= r["high"] for r in routes) != 1):
            raise ValueError("route intervals and distinct symbols must determine one answer")
        out.update(mass=spec["mass"], routes=routes, seals=seals)
    else:
        _name(spec["authority"])
        if type(spec["catalog"]) is not list or type(spec["values"]) is not list:
            raise ValueError("catalog and value tables required")
        catalog, values = [], []
        for row in spec["catalog"]:
            if type(row) is not dict or set(row) != {"revision", "cycle", "writer"}:
                raise ValueError("invalid catalog entry")
            catalog.append({"revision": _name(row["revision"]), "cycle": _int(row["cycle"], 1, 99), "writer": _name(row["writer"])})
        for row in spec["values"]:
            if type(row) is not dict or set(row) != {"revision", "symbol"}:
                raise ValueError("invalid value entry")
            values.append({"revision": _name(row["revision"]), "symbol": _symbol(row["symbol"])})
        catalog = sorted({json_bytes(row): row for row in catalog}.values(), key=lambda r: r["cycle"])
        values = sorted({json_bytes(row): row for row in values}.values(), key=lambda r: r["revision"])
        if (len(catalog) != 4 or len(values) != 4 or len({r["cycle"] for r in catalog}) != 4
                or len({r["revision"] for r in catalog}) != 4
                or {r["revision"] for r in catalog} != {r["revision"] for r in values}
                or len({r["symbol"] for r in values}) != 4
                or sum(r["writer"] == spec["authority"] for r in catalog) != 2
                or sorted(Counter(r["writer"] for r in catalog).values()) != [2, 2]):
            raise ValueError("four distinct revisions with two per writer required")
        out.update(authority=spec["authority"], catalog=catalog, values=values)
    return out


def semantic_signature(spec: dict) -> dict:
    """Normalize all arbitrary aliases, output labels and surface metadata.

    Numeric measurements/timeline bounds remain substantive world facts. The
    family-level structural cluster is also exported; numeric differences do
    not establish different mechanisms or prove statistical independence.
    """
    spec = canonicalize(spec)
    if spec["family"] == FAMILIES[0]:
        return {"family": spec["family"], "mass": spec["mass"],
                "intervals": [[r["low"], r["high"]] for r in spec["routes"]]}
    return {"family": spec["family"], "catalog": [
        [r["cycle"], r["writer"] == spec["authority"]] for r in spec["catalog"]]}


@dataclass(frozen=True)
class Parent:
    parent_id: str
    role: str
    family: str
    placement: str
    missing_index: int
    canonical_json: str
    split_sha256: str
    structural_cluster_id: str


def allocate_parent(spec: dict, placement: str = "early", missing_index: int = 0) -> Parent:
    spec = canonicalize(spec)
    if placement not in POSITIONS:
        raise ValueError("invalid evidence placement")
    _int(missing_index, 0, 2)
    signature = json_bytes(semantic_signature(spec))
    parent_id = "lc-" + sha256(signature)
    split = sha256(ROLE_DOMAIN + signature)
    bucket = int(split[:16], 16) % 10000
    role = next(name for bound, name in ROLE_BUCKETS if bucket < bound)
    # No public Parent object for closed populations carries a world payload.
    payload = json_bytes(spec).decode() if role == "DEV" else ""
    return Parent(parent_id, role, spec["family"], placement, missing_index,
                  payload, split, spec["family"])


def _validated_dev(parent: Parent) -> dict:
    if not isinstance(parent, Parent) or parent.role != "DEV" or not parent.canonical_json:
        raise ValueError("only DEV payloads may be opened; CHECK/FINAL remain closed")
    spec = canonicalize(json.loads(parent.canonical_json))
    rebuilt = allocate_parent(spec, parent.placement, parent.missing_index)
    if rebuilt != parent:
        raise ValueError("parent identity/role metadata does not match semantic world")
    return spec


def _letters(number: int, width: int = 5) -> str:
    chars = []
    for _ in range(width):
        chars.append(chr(65 + number % 26))
        number //= 26
    return "".join(chars)


def generate_spec(family: str, seed: int, candidate_index: int) -> dict:
    if family not in FAMILIES:
        raise ValueError("unknown family")
    _int(seed, 0, 2 ** 63 - 1)
    _int(candidate_index, 0, 999999)
    digest = sha256(json_bytes([family, seed, candidate_index]))
    number = int(digest[:12], 16)
    rng = random.Random(int(digest, 16))
    common = {"family": family, "subject": "C" + _letters(number, 3)}
    symbols = ["S" + _letters(i) for i in rng.sample(range(26 ** 5), 3 if family == FAMILIES[0] else 4)]
    if family == FAMILIES[0]:
        cut1 = 10 + number % 31
        cut2 = 50 + int(digest[12:16], 16) % 31
        intervals = ((1, cut1), (cut1 + 1, cut2), (cut2 + 1, 99))
        depots = list(("D", "E", "F"))
        rng.shuffle(depots)
        return canonicalize({**common, "mass": 1 + int(digest[16:24], 16) % 99,
                             "routes": [{"low": low, "high": high, "depot": depot}
                                        for (low, high), depot in zip(intervals, depots)],
                             "seals": [{"depot": depot, "symbol": symbol}
                                       for depot, symbol in zip(depots, symbols)]})
    cycles = [1 + number % 20, 26 + int(digest[12:16], 16) % 20,
              51 + int(digest[16:20], 16) % 20, 76 + int(digest[20:24], 16) % 20]
    revisions = list(("R", "T", "U", "V"))
    rng.shuffle(revisions)
    writers = rng.sample(("A", "B", "C"), 2) * 2
    rng.shuffle(writers)
    # Independent draw: neither writer frequency nor newest-catalog position
    # substitutes for reading the explicit AUTH block.
    authority = rng.choice(sorted(set(writers)))
    return canonicalize({**common, "authority": authority,
                         "catalog": [{"revision": revision, "cycle": cycle, "writer": writer}
                                     for revision, cycle, writer in zip(revisions, cycles, writers)],
                         "values": [{"revision": revision, "symbol": symbol}
                                    for revision, symbol in zip(revisions, symbols)]})


def allocate_bank(seed: int = 20260928, candidate_limit: int = 100000) -> dict:
    """Allocate 32 DEV worlds and 64 opaque CHECK reservations before rendering.

    Four DEV/eight CHECK worlds per family/placement cell. No solve, render,
    token-fit or model outcome participates in selection. FINAL receives zero.
    """
    _int(candidate_limit, 1, 1_000_000)
    dev: list[Parent] = []
    check = []
    seen: set[str] = set()
    attempted = {}
    for family in FAMILIES:
        counts = Counter()
        for index in range(candidate_limit):
            p = allocate_parent(generate_spec(family, seed, index))
            if p.parent_id in seen:
                continue
            seen.add(p.parent_id)
            quota = 16 if p.role == "DEV" else 32 if p.role == "CHECK" else 0
            if not quota or counts[p.role] >= quota:
                continue
            placement = POSITIONS[counts[p.role] % 4]
            missing = len(dev) % 3 if p.role == "DEV" else counts[p.role] % 3
            p = replace(p, placement=placement, missing_index=missing)
            counts[p.role] += 1
            if p.role == "DEV":
                dev.append(p)
            else:
                check.append({"parent_id": p.parent_id, "role": "CHECK",
                              "family": family, "placement": placement,
                              "split_sha256": p.split_sha256,
                              "structural_cluster_id": p.structural_cluster_id})
            if counts["DEV"] == 16 and counts["CHECK"] == 32:
                attempted[family] = index + 1
                break
        else:
            raise ValueError("bounded candidate ceiling exhausted; no role borrowing")
    return {"dev": dev, "check_reservations": check,
            "allocation": {"seed": seed, "candidate_limit": candidate_limit,
                           "candidates_attempted": attempted,
                           "DEV_worlds": 32, "CHECK_hash_reservations": 64,
                           "FINAL_worlds": 0, "structural_clusters": 2,
                           "support_unit": "Three necessary record blocks, not three atomic facts",
                           "missing_index_counts": dict(Counter(p.missing_index for p in dev)),
                           "independence_qualified": False,
                           "independence_limit": "Sampled numeric semantic worlds share two mechanisms; world counts do not establish independent structural tasks."}}


def reference_answer(parent: Parent, answerable: bool = True) -> str:
    """Canonical-world oracle, independent of the visible text parser below."""
    spec = _validated_dev(parent)
    if type(answerable) is not bool:
        raise ValueError("answerable must be bool")
    if not answerable:
        return "UNKNOWN"
    if spec["family"] == FAMILIES[0]:
        route = next(r for r in spec["routes"] if r["low"] <= spec["mass"] <= r["high"])
        return next(r["symbol"] for r in spec["seals"] if r["depot"] == route["depot"])
    selected = sorted((r for r in spec["catalog"] if r["writer"] == spec["authority"]),
                      key=lambda r: r["cycle"], reverse=True)[0]
    return next(r["symbol"] for r in spec["values"] if r["revision"] == selected["revision"])


def filler_line(index: int) -> str:
    _int(index, 0, 2 ** 63 - 1)
    return FILLER_TEXTS[index % len(FILLER_TEXTS)]


def _unused_alias(used: set[str]) -> str:
    """An unrelated alias chosen from the full set, never from a selected path."""
    for index in range(len(used) + 1):
        candidate = "X" + _letters(index, 2)
        if candidate not in used:
            return candidate
    raise AssertionError("finite alias selection failed")


def render_variant(parent: Parent, answerable: bool, adversarial: bool) -> dict:
    spec = _validated_dev(parent)
    if type(answerable) is not bool or type(adversarial) is not bool:
        raise ValueError("variant switches must be bool")
    subject = spec["subject"]
    surface_rng = random.Random(int(parent.parent_id[3:19], 16))
    if parent.family == FAMILIES[0]:
        route_rows, seal_rows = list(spec["routes"]), list(spec["seals"])
        surface_rng.shuffle(route_rows)
        surface_rng.shuffle(seal_rows)
        routes = ",".join(f"{r['low']}-{r['high']}:{r['depot']}" for r in route_rows)
        seals = ",".join(f"{r['depot']}:{r['symbol']}" for r in seal_rows)
        support = [f"MASS {subject}={spec['mass']}.", f"ROUTES {subject}={routes}.",
                   f"SEALS {subject}={seals}."]
        query = f"Return the routed seal for {subject}."
        symbols = {r["symbol"] for r in spec["seals"]}
        other = next("S" + _letters(i) for i in range(4) if "S" + _letters(i) not in symbols)
        unrelated = _unused_alias({r["depot"] for r in spec["routes"]})
        decoy_kind, decoy_payload = (
            ("MASS", "0"),
            ("ROUTES", f"1-99:{unrelated}"),
            ("SEALS", f"{unrelated}:{other}"),
        )[parent.missing_index]
    else:
        catalog_rows, value_rows = list(spec["catalog"]), list(spec["values"])
        surface_rng.shuffle(catalog_rows)
        surface_rng.shuffle(value_rows)
        catalog = ",".join(f"{r['revision']}:{r['cycle']}:{r['writer']}" for r in catalog_rows)
        values = ",".join(f"{r['revision']}:{r['symbol']}" for r in value_rows)
        support = [f"AUTH {subject}={spec['authority']}.", f"CAT {subject}={catalog}.",
                   f"VAL {subject}={values}."]
        query = f"Return the latest authorized record for {subject}."
        symbols = {r["symbol"] for r in spec["values"]}
        other = next("S" + _letters(i) for i in range(5) if "S" + _letters(i) not in symbols)
        unrelated_revision = _unused_alias({r["revision"] for r in spec["catalog"]})
        unrelated_writer = _unused_alias({r["writer"] for r in spec["catalog"]})
        decoy_kind, decoy_payload = (
            ("AUTH", unrelated_writer),
            ("CAT", f"{unrelated_revision}:0:{unrelated_writer}"),
            ("VAL", f"{unrelated_revision}:{other}"),
        )[parent.missing_index]
    decoys = [f"{decoy_kind} OTHER={decoy_payload}.",
              f"ARCHIVE {decoy_kind} {subject}={decoy_payload}.",
              f"UNSIGNED {decoy_kind} {subject}={decoy_payload}."]
    slots = list(support)
    if not answerable:
        slots[parent.missing_index] = "NOTE."
    if not adversarial:
        decoys = [filler_line(i) for i in range(len(decoys))]
    return {"parent_id": parent.parent_id, "role": "DEV", "family": parent.family,
            "placement": parent.placement, "missing_index": parent.missing_index,
            "structural_cluster_id": parent.structural_cluster_id,
            "support_unit": "record_block",
            "distractor_modes": (["different_key", "obsolete_archive", "unauthorized_unsigned"]
                                 if adversarial else ["neutral"] * 3),
            "answerable": answerable, "adversarial": adversarial,
            "query": query, "support_facts": support, "support_slots": slots,
            "decoys": decoys, "target": reference_answer(parent, answerable)}


def _unique(values: list) -> Any:
    values = set(values)
    return next(iter(values)) if len(values) == 1 else None


def solve_visible(query: str, body: str) -> str:
    """Strict independently parsed visible evidence; no spec/label/metadata input.

    Blank lines and a closed NOTE vocabulary contain no evidence. ARCHIVE and
    UNSIGNED records are validated with the same grammar but are nonbinding.
    Conflicting required evidence yields UNKNOWN; malformed evidence raises.
    """
    if type(query) is not str or type(body) is not str:
        raise ValueError("visible query/body strings required")
    parsed: dict[str, list[tuple]] = {key: [] for key in ("MASS", "ROUTES", "SEALS", "AUTH", "CAT", "VAL")}
    entry_patterns = {"ROUTES": rf"([0-9]+)-([0-9]+):({_NAME})",
                      "SEALS": rf"({_NAME}):({_SYMBOL})",
                      "CAT": rf"({_NAME}):([0-9]+):({_NAME})",
                      "VAL": rf"({_NAME}):({_SYMBOL})"}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line == "NOTE.":
            continue
        if line.startswith("NOTE ") and line.endswith("."):
            words = line[5:-1].split(" ")
            if words and all(word in FILLER_WORDS for word in words):
                continue
            raise ValueError("unregistered NOTE filler")
        match = re.fullmatch(rf"(?:(ARCHIVE|UNSIGNED) )?(MASS|ROUTES|SEALS|AUTH|CAT|VAL) ({_NAME})=([^\s.]+)\.", line)
        if not match:
            raise ValueError("unparsed visible evidence line")
        shadow, kind, subject, payload = match.groups()
        if kind == "MASS":
            if not re.fullmatch(r"[0-9]+", payload):
                raise ValueError("invalid visible mass")
            if shadow is None:
                parsed[kind].append((subject, int(payload)))
        elif kind == "AUTH":
            if not re.fullmatch(_NAME, payload):
                raise ValueError("invalid visible authority")
            if shadow is None:
                parsed[kind].append((subject, payload))
        else:
            for entry in payload.split(","):
                item = re.fullmatch(entry_patterns[kind], entry)
                if item is None:
                    raise ValueError("invalid visible table entry")
                values = item.groups()
                if kind == "ROUTES" and int(values[0]) > int(values[1]):
                    raise ValueError("reversed visible route interval")
                if shadow is None:
                    parsed[kind].append((subject, *values))
    join = re.fullmatch(rf"Return the routed seal for ({_NAME})\.", query)
    latest = re.fullmatch(rf"Return the latest authorized record for ({_NAME})\.", query)
    if join:
        subject = join[1]
        mass = _unique([m for name, m in parsed["MASS"] if name == subject])
        if mass is None:
            return "UNKNOWN"
        depot = _unique([depot for key, low, high, depot in parsed["ROUTES"]
                         if key == subject and int(low) <= mass <= int(high)])
        if depot is None:
            return "UNKNOWN"
        return _unique([symbol for key, d, symbol in parsed["SEALS"]
                        if key == subject and d == depot]) or "UNKNOWN"
    if latest:
        subject = latest[1]
        authority = _unique([a for key, a in parsed["AUTH"] if key == subject])
        if authority is None:
            return "UNKNOWN"
        candidates = [(int(cycle), revision) for key, revision, cycle, writer in parsed["CAT"]
                      if key == subject and writer == authority]
        if not candidates:
            return "UNKNOWN"
        newest = max(cycle for cycle, _ in candidates)
        revision = _unique([r for cycle, r in candidates if cycle == newest])
        if revision is None:
            return "UNKNOWN"
        return _unique([symbol for key, r, symbol in parsed["VAL"]
                        if key == subject and r == revision]) or "UNKNOWN"
    raise ValueError("unregistered query grammar")


def verify_variant(parent: Parent, variant: dict) -> dict:
    """Check canonical oracle, visible output, pairing and all three ablations."""
    expected = render_variant(parent, variant.get("answerable"), variant.get("adversarial"))
    if variant != expected:
        raise ValueError("variant differs from registered canonical rendering")
    body = "\n".join(variant["support_slots"] + variant["decoys"])
    solved = solve_visible(variant["query"], body)
    if solved != variant["target"]:
        raise ValueError("visible solver disagrees with independent world oracle")
    ablations = []
    for missing in range(3):
        facts = list(variant["support_facts"])
        facts[missing] = "NOTE."
        result = solve_visible(variant["query"], "\n".join(facts + variant["decoys"]))
        if result != "UNKNOWN":
            raise ValueError("support ablation leaves an alternate answer path")
        ablations.append(result)
    if reference_answer(parent) in "\n".join([variant["query"]] + variant["decoys"]):
        raise ValueError("target leaks through query or distractors")
    return {"visible_solution": solved, "single_fact_ablations": ablations,
            "independent_oracle_agrees": True}


def parse_answer(text: str, diagnostic: bool = False) -> str | None:
    if type(text) is not str or type(diagnostic) is not bool:
        return None
    text = text.strip()
    pattern = rf"(?:{_SYMBOL}|UNKNOWN)"
    if diagnostic:
        pattern = rf"(?:Answer: )?{pattern}"
    if re.fullmatch(pattern, text) is None:
        return None
    return text[8:] if diagnostic and text.startswith("Answer: ") else text


def score_answer(text: str, target: str) -> bool:
    return type(target) is str and parse_answer(target) == target and parse_answer(text) == target


def diagnostic_answer(text: str, target: str) -> bool:
    return type(target) is str and parse_answer(target) == target and parse_answer(text, True) == target
