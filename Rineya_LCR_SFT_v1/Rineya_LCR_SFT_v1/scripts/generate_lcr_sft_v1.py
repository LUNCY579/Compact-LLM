#!/usr/bin/env python3
"""Generate Rineya closed-context long-document SFT corpus v1.

The generator is deterministic, uses no network access, writes streaming JSONL,
and creates four coordinated variants per scenario:
base, paraphrase, counterfactual, and unanswerable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SYSTEM_MESSAGE = (
    "Use only the supplied context. Answer correctly and completely. Do not guess. "
    "If the context does not determine the answer, reply UNKNOWN. Brevity is optional; "
    "correctness is required."
)
INTENDED_USE = "sft_train_disjoint_from_rineya_long_context_demo_v1"

BANNED_TERMS = {
    "RLC-", "Asterhall", "Bellweather", "Talia Frost", "Thalia Lark",
    "Record of the Hidden Key",
}

DOMAINS: List[Dict[str, object]] = [
    {
        "key": "academy_chronicle",
        "site_suffix": ["Academy", "Collegium", "Scholarium", "Learning Court"],
        "roles": ["provost", "bursar", "archive tutor", "gate steward", "lecture marshal", "observatory keeper", "registrar"],
        "rooms": ["north cloister", "prism classroom", "bell archive", "roof observatory", "map refectory", "west tutorial room", "ink vault"],
        "routes": ["lantern walk", "scholar stair", "copper arcade", "quiet cloister", "sundial passage"],
        "objects": ["examination ledger", "prism key", "atlas case", "bell token", "sealed rubric", "lecture cylinder", "star chart folio"],
        "actions": ["registered", "escorted", "certified", "cataloged", "sealed", "transferred", "audited"],
        "records": ["prefect journal", "attendance roll", "seal memorandum", "tutorial register", "bursar docket"],
        "events": ["term-opening audit", "observatory inspection", "archive rotation", "scholarship review", "night-bell rehearsal"],
        "title_nouns": ["Chronicle", "Register", "Bell Ledger", "Cloister Record", "Provost File"],
        "unit": "folios",
    },
    {
        "key": "detective_file",
        "site_suffix": ["Inquiry House", "Case Office", "Evidence Court", "Investigation Annex"],
        "roles": ["lead examiner", "evidence clerk", "route investigator", "interview recorder", "seal custodian", "night dispatcher", "case auditor"],
        "rooms": ["interview room", "evidence cage", "map office", "sealed corridor", "observation booth", "dispatch desk", "document vault"],
        "routes": ["witness loop", "evidence stair", "gray corridor", "lantern route", "sealed bypass"],
        "objects": ["evidence satchel", "coded notebook", "glass token", "sealed photograph case", "timeline reel", "trace envelope", "interview cylinder"],
        "actions": ["logged", "examined", "escorted", "sealed", "reconciled", "interviewed", "verified"],
        "records": ["case digest", "evidence chain", "interview abstract", "dispatch sheet", "closure warrant"],
        "events": ["night-route inquiry", "missing-token review", "sealed-room reconstruction", "witness-order audit", "chain-of-custody hearing"],
        "title_nouns": ["Case File", "Inquiry Ledger", "Evidence Docket", "Closure Record", "Investigator Digest"],
        "unit": "evidence units",
    },
    {
        "key": "station_log",
        "site_suffix": ["Orbital Station", "Relay Habitat", "Research Ring", "Sky Platform"],
        "roles": ["shift commander", "systems keeper", "cargo controller", "airlock marshal", "signal analyst", "maintenance recorder", "flight auditor"],
        "rooms": ["airlock bay", "signal cupola", "cargo spindle", "maintenance cell", "habitat ring", "thermal gallery", "command alcove"],
        "routes": ["inner ringway", "blue spoke", "service crawl", "cargo arc", "sunward bridge"],
        "objects": ["telemetry cartridge", "seal canister", "reaction gauge", "cargo lattice", "navigation wafer", "thermal probe", "signal cassette"],
        "actions": ["docked", "cycled", "calibrated", "transferred", "isolated", "logged", "verified"],
        "records": ["shift log", "airlock manifest", "telemetry digest", "maintenance closure", "cargo reconciliation"],
        "events": ["shadow-pass inspection", "relay alignment", "cargo-ring audit", "thermal drift review", "airlock timing exercise"],
        "title_nouns": ["Station Log", "Ring Manifest", "Relay Chronicle", "Shift Ledger", "Orbital Docket"],
        "unit": "modules",
    },
    {
        "key": "expedition_journal",
        "site_suffix": ["Field Camp", "Survey Base", "Ridge Outpost", "Marsh Station"],
        "roles": ["expedition lead", "survey cartographer", "sample keeper", "route scout", "weather recorder", "camp quartermaster", "journal auditor"],
        "rooms": ["sample tent", "map shelter", "ridge cache", "instrument hut", "supply awning", "signal post", "drying room"],
        "routes": ["reed traverse", "basalt track", "wind pass", "survey loop", "north cairn path"],
        "objects": ["sample chest", "survey compass", "soil cylinder", "weather drum", "specimen folio", "marker beacon", "altitude reel"],
        "actions": ["surveyed", "cached", "measured", "sealed", "carried", "charted", "verified"],
        "records": ["field journal", "route card", "sample register", "weather correction", "camp closure sheet"],
        "events": ["ridge transect", "marsh sampling run", "cairn verification", "storm-route revision", "dawn instrument census"],
        "title_nouns": ["Expedition Journal", "Field Chronicle", "Survey Record", "Camp Ledger", "Ridge Docket"],
        "unit": "samples",
    },
    {
        "key": "museum_conservation",
        "site_suffix": ["Conservation Museum", "Gallery Institute", "Collection Hall", "Restoration Court"],
        "roles": ["chief conservator", "collection registrar", "frame technician", "gallery marshal", "condition photographer", "crate custodian", "audit curator"],
        "rooms": ["varnish studio", "north gallery", "crate vault", "paper lab", "frame room", "loading vestibule", "condition booth"],
        "routes": ["gallery circuit", "crate corridor", "linen stair", "visitor bypass", "conservation passage"],
        "objects": ["painted panel", "ceramic vessel", "frame cassette", "textile roll", "condition folio", "sealed plinth", "pigment tray"],
        "actions": ["conditioned", "photographed", "crated", "mounted", "transferred", "cataloged", "verified"],
        "records": ["condition report", "accession ledger", "crate manifest", "gallery closure", "conservation amendment"],
        "events": ["gallery rehanging", "humidity audit", "crate-seal review", "loan-return inspection", "frame stabilization cycle"],
        "title_nouns": ["Conservation Record", "Gallery Ledger", "Accession Chronicle", "Crate Docket", "Curator File"],
        "unit": "collection units",
    },
    {
        "key": "lab_chronology",
        "site_suffix": ["Methods Laboratory", "Measurement Institute", "Chronology Lab", "Instrument Court"],
        "roles": ["lab director", "method recorder", "instrument custodian", "sample coordinator", "calibration analyst", "safety marshal", "quality auditor"],
        "rooms": ["calibration room", "sample cold room", "instrument bay", "analysis alcove", "wash station", "record vault", "control chamber"],
        "routes": ["clean corridor", "instrument loop", "sample stair", "control passage", "service bypass"],
        "objects": ["calibration block", "sample rack", "sensor capsule", "method binder", "control cartridge", "reference plate", "timing reel"],
        "actions": ["calibrated", "measured", "sealed", "incubated", "transferred", "reconciled", "verified"],
        "records": ["method chronology", "instrument log", "sample chain", "calibration correction", "quality closure"],
        "events": ["reference run", "sensor comparison", "method transfer", "control audit", "timing-sequence review"],
        "title_nouns": ["Lab Chronology", "Method Ledger", "Instrument Record", "Control Docket", "Quality File"],
        "unit": "test units",
    },
    {
        "key": "civic_record",
        "site_suffix": ["Civic Hall", "Municipal Court", "Public Works Annex", "Council House"],
        "roles": ["records commissioner", "permit clerk", "route inspector", "council recorder", "works marshal", "seal officer", "civic auditor"],
        "rooms": ["permit chamber", "map registry", "council gallery", "works desk", "seal archive", "public counter", "inspection room"],
        "routes": ["council arcade", "permit stair", "works passage", "public loop", "registry bridge"],
        "objects": ["permit folio", "survey plaque", "seal packet", "route map", "council token", "inspection cylinder", "notice bundle"],
        "actions": ["filed", "inspected", "certified", "posted", "transferred", "reconciled", "verified"],
        "records": ["council minute", "permit register", "works docket", "inspection amendment", "seal closure"],
        "events": ["bridge-permit review", "market-route hearing", "registry audit", "public notice reconciliation", "works-seal inspection"],
        "title_nouns": ["Civic Record", "Council Ledger", "Permit Chronicle", "Works Docket", "Registry File"],
        "unit": "registered units",
    },
    {
        "key": "estate_business_saga",
        "site_suffix": ["Trading Estate", "Family Works", "Mercantile Court", "Workshop Holding"],
        "roles": ["managing partner", "estate steward", "accounts keeper", "workshop foreman", "dispatch manager", "seal secretary", "independent auditor"],
        "rooms": ["accounts room", "sample parlor", "dispatch yard", "contract vault", "workshop loft", "receiving office", "estate archive"],
        "routes": ["merchant walk", "yard circuit", "contract stair", "delivery lane", "workshop passage"],
        "objects": ["contract chest", "sample case", "invoice reel", "seal box", "dispatch folio", "tool crate", "share ledger"],
        "actions": ["invoiced", "dispatched", "received", "sealed", "reconciled", "assigned", "verified"],
        "records": ["partner minute", "estate ledger", "dispatch receipt", "contract correction", "audit closure"],
        "events": ["quarterly settlement", "workshop transfer", "contract-renewal review", "inventory succession", "dispatch reconciliation"],
        "title_nouns": ["Estate Chronicle", "Partner Ledger", "Mercantile Record", "Workshop Docket", "Settlement File"],
        "unit": "accounted units",
    },
    {
        "key": "clinic_inventory",
        "site_suffix": ["Supply Clinic", "Care Inventory Center", "Community Stock Annex", "Clinic Storehouse"],
        "roles": ["inventory coordinator", "supply clerk", "storage inspector", "delivery marshal", "label recorder", "cabinet custodian", "stock auditor"],
        "rooms": ["linen store", "sealed supply room", "labeling desk", "receiving bay", "cabinet aisle", "records office", "cool storage room"],
        "routes": ["supply corridor", "receiving loop", "cabinet passage", "clean stair", "delivery bypass"],
        "objects": ["bandage carton", "empty specimen box", "label roll", "sealed linen pack", "inventory tablet", "storage gauge", "delivery tote"],
        "actions": ["counted", "labeled", "received", "stored", "transferred", "reconciled", "verified"],
        "records": ["stock ledger", "cabinet manifest", "delivery receipt", "label correction", "inventory closure"],
        "events": ["nonclinical stocktake", "cabinet-label audit", "linen delivery review", "storage-route inspection", "supply reconciliation"],
        "title_nouns": ["Inventory Record", "Supply Ledger", "Cabinet Chronicle", "Stock Docket", "Delivery File"],
        "unit": "inventory units",
    },
    {
        "key": "warehouse_logistics",
        "site_suffix": ["Freight Warehouse", "Distribution Yard", "Cargo Depot", "Logistics Court"],
        "roles": ["yard controller", "manifest clerk", "fork marshal", "seal inspector", "route dispatcher", "bay custodian", "freight auditor"],
        "rooms": ["receiving bay", "high-rack aisle", "seal cage", "dispatch office", "weigh room", "return dock", "manifest vault"],
        "routes": ["fork lane", "cargo loop", "west conveyor", "inspection bypass", "dispatch track"],
        "objects": ["freight crate", "pallet cage", "seal drum", "manifest tube", "return tote", "weight block", "dispatch canister"],
        "actions": ["received", "weighed", "sealed", "routed", "loaded", "reconciled", "verified"],
        "records": ["freight manifest", "bay ledger", "seal receipt", "routing correction", "dispatch closure"],
        "events": ["night dispatch", "pallet reconciliation", "seal-cage audit", "return-dock review", "conveyor reroute"],
        "title_nouns": ["Warehouse Log", "Freight Ledger", "Cargo Chronicle", "Bay Docket", "Dispatch File"],
        "unit": "cargo units",
    },
    {
        "key": "archive_catalog",
        "site_suffix": ["Archive House", "Catalog Institute", "Manuscript Court", "Records Repository"],
        "roles": ["chief archivist", "catalog editor", "shelf marshal", "document conservator", "reading-room clerk", "seal custodian", "catalog auditor"],
        "rooms": ["manuscript vault", "reading room", "index gallery", "seal cabinet", "map stack", "repair alcove", "catalog office"],
        "routes": ["shelf circuit", "reader stair", "map passage", "sealed aisle", "catalog bridge"],
        "objects": ["manuscript bundle", "index tablet", "map folio", "seal packet", "catalog reel", "repair case", "reading token"],
        "actions": ["accessioned", "shelved", "indexed", "sealed", "transferred", "reconciled", "verified"],
        "records": ["accession catalog", "shelf ledger", "reader receipt", "index correction", "archive closure"],
        "events": ["shelf-number audit", "manuscript accession", "map-stack reconciliation", "reader-token review", "seal-cabinet inspection"],
        "title_nouns": ["Archive Catalog", "Shelf Chronicle", "Manuscript Ledger", "Index Docket", "Archivist File"],
        "unit": "catalog units",
    },
    {
        "key": "ships_log",
        "site_suffix": ["Survey Vessel", "Harbor Cutter", "Research Ship", "Tide Barge"],
        "roles": ["master", "first mate", "cargo purser", "route navigator", "signal keeper", "deck marshal", "voyage auditor"],
        "rooms": ["chart room", "forward hold", "signal cabin", "aft locker", "engine gallery", "boat deck", "purser office"],
        "routes": ["port passage", "weather deck", "cargo companionway", "signal stair", "aft service route"],
        "objects": ["chart cylinder", "cargo chest", "signal lamp case", "tide gauge", "sealed logbook", "mooring token", "sample barrel"],
        "actions": ["stowed", "charted", "signaled", "transferred", "sealed", "reconciled", "verified"],
        "records": ["ship's log", "cargo manifest", "watch receipt", "course correction", "voyage closure"],
        "events": ["harbor departure", "tide-course review", "cargo-watch audit", "signal-lamp inspection", "anchorage reconciliation"],
        "title_nouns": ["Ship's Log", "Voyage Ledger", "Harbor Chronicle", "Watch Docket", "Master's File"],
        "unit": "stowed units",
    },
]

TASK_TYPES = [
    "alias_resolution",
    "object_state_tracking",
    "temporal_correction",
    "rule_exception",
    "count_exclusions_duplicates",
    "causal_disambiguation",
    "multi_condition_identity",
    "latest_verified_record_vs_rumor",
    "definition_binding",
    "address_route_tracking",
]

STARTS = [
    "Ar", "Bel", "Cae", "Dor", "Eri", "Fen", "Gal", "Hae", "Ily", "Jor", "Kel", "Lor", "Mer", "Niv", "Or", "Pha",
    "Quin", "Ral", "Sor", "Tav", "Ul", "Vey", "Wen", "Xel", "Yor", "Zan", "Bre", "Cyr", "Dov", "Esa", "Fri", "Glo",
]
MIDS = ["a", "e", "i", "o", "u", "ae", "ei", "io", "ai", "ar", "el", "en", "ir", "or", "ul", "ur", "av", "ev", "im", "on", "ra", "ri", "va", "yn"]
ENDS = ["n", "ra", "s", "el", "in", "or", "a", "en", "is", "um", "eth", "iel", "an", "os", "er", "yn", "ik", "ad", "ir", "us", "ia", "om", "ek", "un"]
TAILS = ["", "a", "en", "or", "is", "um", "eth", "al", "in", "os", "ar", "el"]

ADJECTIVES = [
    "Amber", "Ashen", "Azure", "Brass", "Cinder", "Copper", "Dappled", "Dusk", "Ember", "Frosted", "Glass", "Golden",
    "Harbor", "Indigo", "Ivory", "Jade", "Keen", "Lattice", "Misted", "Narrow", "Ochre", "Pale", "Quartz", "Reed",
    "Sable", "Tidal", "Umber", "Violet", "Woven", "Yellow", "Zinc", "Silent", "Hollow", "Cobalt", "Marble", "Verdant",
    "Silver", "Scarlet", "Clouded", "Lantern", "Oblique", "Quiet", "Riven", "Velvet", "Winter", "Lunar", "Solar", "Morrow",
]
NOUNS = [
    "Finch", "Heron", "Kestrel", "Moth", "Lark", "Otter", "Fox", "Badger", "Wren", "Rook", "Pike", "Marten",
    "Spindle", "Compass", "Lantern", "Anvil", "Quill", "Cairn", "Harbor", "Anchor", "Prism", "Loom", "Bell", "Reed",
    "Cedar", "Flint", "Basin", "Arc", "Gable", "Trellis", "Drum", "Crown", "Spar", "Helix", "Keel", "Gate",
    "Cipher", "Vane", "Needle", "Shuttle", "Thimble", "Beacon", "Mirror", "Pillar", "Wicket", "Tide", "Ember", "Orbit",
]
FICTIONAL_MONTHS = ["Auren", "Brindle", "Cevren", "Dovrel", "Esmar", "Fallow", "Galen", "Hesper", "Ivrin", "Jorrel", "Kelm", "Luneth"]

AUTHORITY_ORDERS = [
    ("signed closure memorandum", "time-stamped receipt", "working sheet", "oral recollection"),
    ("countersigned final docket", "sealed movement log", "provisional roster", "later interview"),
    ("auditor-certified amendment", "gate or room receipt", "planning card", "margin note"),
    ("final registry entry bearing two seals", "instrument or custody log", "unsigned draft", "witness rumor"),
]

LOCAL_TERMS = [
    ("cleared", "inspected and resealed; it does not mean removed from the site"),
    ("quiet", "processed without a public announcement; it does not mean inactive"),
    ("closed", "accepted into the current ledger; it does not mean physically locked"),
    ("returned", "placed under the receiving custodian's responsibility; it does not necessarily mean moved back to the starting room"),
    ("blue", "reviewed by the route clerk; it describes a record status rather than a color"),
    ("dry", "free of unresolved paperwork; it says nothing about moisture"),
]

# Each domain/task pair occurs 16 or 17 times in the full corpus. The occurrence
# index selects a distinct plot motif and evidence mechanism, preventing a pack
# from being merely the same story with renamed entities.
PLOT_MOTIFS = [
    "dawn resealing", "cross-shift exchange", "storm detour", "duplicate-label review",
    "night return", "inspection appeal", "two-stage transfer", "route reopening",
    "clock reconciliation", "custody reversal", "priority release", "shelf evacuation",
    "manifest dispute", "emergency diversion", "final accession", "seal restoration",
    "closing-day handover",
]
ACTION_LABELS = [
    "sealed handoff", "checkout relay", "return intake", "inspection transfer",
    "night deposit", "custody exchange", "route release", "catalog accession",
    "stock reconciliation", "signal transfer", "manifest closure", "permit filing",
    "sample delivery", "display relocation", "watch turnover", "ledger deposit",
    "emergency reroute",
]
EVIDENCE_RECORDS = [
    "signed completion slip", "two-seal gate receipt", "countersigned room docket",
    "auditor-certified movement line", "sealed return ticket", "route closure receipt",
    "custody relay card", "final accession entry", "inventory reconciliation sheet",
    "signal-room completion strip", "manifest closure page", "permit counter receipt",
    "sample-chain terminal line", "gallery movement certificate", "watch turnover page",
    "ledger deposit receipt", "emergency routing confirmation",
]
EXCEPTION_CONDITIONS = [
    "declared route closure", "after-hours lock period", "inventory freeze",
    "storm-warning interval", "seal-discrepancy review", "shift-overlap window",
    "public-access suspension", "instrument isolation", "loading-bay hold",
    "archive evacuation", "quorum interruption", "signal blackout",
    "temporary custody gap", "tide-delay window", "inspection quarantine",
    "return-processing hold", "final-day closure",
]
COUNT_MARKS = [
    "scarlet independence notch", "double-ring stamp", "brass split mark",
    "indigo unit slash", "hollow-square punch", "three-dot separation mark",
    "copper fork seal", "ivory twin bar", "black crescent cut",
    "green ledger star", "amber standalone tab", "blue crosshatch",
    "silver corner clip", "red second-unit loop", "violet forked mark",
    "white double seal", "ochre independence stripe",
]
ALERT_LABELS = [
    "alarm", "custody alert", "route halt", "checksum failure", "seal mismatch",
    "inventory discrepancy", "timing fault", "access warning", "loading interruption",
    "catalog conflict", "signal drop", "permit hold", "sample-chain break",
    "gallery warning", "watch interruption", "ledger imbalance", "dispatch fault",
]
VERIFIER_LABELS = [
    "final verifier", "closure courier", "authorized releaser", "audit carrier",
    "custody confirmer", "route closer", "seal witness", "terminal registrar",
    "inventory certifier", "signal confirmer", "manifest closer", "permit verifier",
    "sample-chain closer", "gallery transfer witness", "watch certifier",
    "ledger receiver", "emergency dispatch confirmer",
]
ROUTE_COMPLETION_LABELS = [
    "completed dispatch", "sealed delivery", "return run", "inspection route",
    "night transfer", "custody run", "release route", "accession route",
    "stock movement", "signal courier run", "manifest delivery", "permit conveyance",
    "sample route", "gallery transfer", "watch delivery", "ledger route",
    "emergency diversion",
]

@dataclass
class TaskData:
    decisive_base: str
    decisive_cf: str
    decisive_u: str
    evidence_sentences: List[str]
    task_rule_sentences: List[str]
    distractor_sentences: List[str]
    question_base: str
    question_para: str
    answer_base: str
    answer_cf: str
    answer_u: str
    accepted_base: List[str]
    accepted_cf: List[str]
    accepted_u: List[str]
    must_base: List[str]
    must_cf: List[str]
    must_u: List[str]
    must_not_base: List[str]
    must_not_cf: List[str]
    must_not_u: List[str]
    support_base: List[str]
    support_cf: List[str]
    support_u: List[str]


def words(text: str) -> int:
    return len(text.split())


def compact_join(parts: Sequence[str]) -> str:
    out = "".join(parts)
    out = re.sub(r"([aeiou])\1+", r"\1", out, flags=re.I)
    return out


def coined(index: int, salt: int = 0) -> str:
    # A bijective modular permutation makes neighboring indices vary across all
    # syllable positions while remaining deterministic and collision-free within
    # each salt for the index range used by this corpus.
    capacity = len(STARTS) * len(MIDS) * len(ENDS) * len(TAILS)
    n = ((index + salt * 19937) * 65537) % capacity
    a = STARTS[n % len(STARTS)]
    n //= len(STARTS)
    b = MIDS[n % len(MIDS)]
    n //= len(MIDS)
    c = ENDS[n % len(ENDS)]
    n //= len(ENDS)
    d = TAILS[n % len(TAILS)]
    return compact_join((a, b, c, d)).capitalize()


def full_name(global_index: int, shared_surname_index: int | None = None) -> str:
    given = coined(global_index, salt=1)
    surname_idx = shared_surname_index if shared_surname_index is not None else global_index
    surname = coined(surname_idx, salt=4)
    return f"{given} {surname}"


def unique_title(i: int, profile: Dict[str, object], site_root: str, rng: random.Random) -> str:
    adjective = ADJECTIVES[(i * 7 + rng.randrange(len(ADJECTIVES))) % len(ADJECTIVES)]
    noun = rng.choice(profile["title_nouns"])  # type: ignore[index]
    return f"The {adjective} {noun} of {site_root}"


def serial_for(i: int, slot: int, prefix: str) -> str:
    # Guaranteed-unique within each domain prefix. Consecutive slots in one
    # scenario intentionally look similar, creating distractors without risking
    # cross-pack collisions.
    code = i * 20 + slot
    return f"{prefix[:2].upper()}-{(code // 1000):03X}-{(code % 1000):03d}"


def time_for(i: int, slot: int) -> str:
    # Six strictly increasing same-day times. This prevents a later evidence
    # line from accidentally preceding an earlier plan or exceeding a stated
    # verifier deadline.
    base = 6 * 60 + ((i * 7) % 37)
    step = 92 + (i % 5) * 4
    total = base + slot * step
    hour, minute = divmod(total, 60)
    return f"{hour:02d}:{minute:02d}"


def date_for(i: int, offset: int) -> str:
    month = FICTIONAL_MONTHS[(i // 17) % len(FICTIONAL_MONTHS)]
    # Keep the four-day sequence inside one invented month and in order.
    day = 3 + ((i * 7) % 18) + offset
    year = 640 + (i % 180)
    return f"{day} {month} {year}"


def make_schedule(total: int, seed: int) -> List[Tuple[int, int]]:
    if total != 2000:
        # Generic balanced fallback for development-sized runs.
        rng = random.Random(seed)
        pairs = [(d, t) for d in range(len(DOMAINS)) for t in range(len(TASK_TYPES))]
        out: List[Tuple[int, int]] = []
        while len(out) < total:
            cycle = list(pairs)
            rng.shuffle(cycle)
            out.extend(cycle[: min(len(cycle), total - len(out))])
        return out

    # At the requested 2,000-pack scale, every one of the 120 domain/task
    # combinations appears either 16 or 17 times. Task totals are exactly 200;
    # domain totals differ by at most one.
    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []
    for d in range(len(DOMAINS)):
        for t in range(len(TASK_TYPES)):
            pairs.extend([(d, t)] * 16)

    remaining_by_domain = [7 if d < 8 else 6 for d in range(len(DOMAINS))]
    extras: List[Tuple[int, int]] = []
    for t in range(len(TASK_TYPES)):
        candidates = list(range(len(DOMAINS)))
        rng.shuffle(candidates)
        candidates.sort(key=lambda d: remaining_by_domain[d], reverse=True)
        chosen = [d for d in candidates if remaining_by_domain[d] > 0][:8]
        if len(chosen) != 8:
            raise RuntimeError("Could not construct balanced domain/task extras")
        for d in chosen:
            extras.append((d, t))
            remaining_by_domain[d] -= 1
    if any(remaining_by_domain):
        raise RuntimeError(f"Unfilled domain schedule: {remaining_by_domain}")
    pairs.extend(extras)
    rng.shuffle(pairs)

    # Reduce adjacent domain/task repetition without changing counts.
    for _ in range(12):
        for j in range(1, len(pairs)):
            if pairs[j][0] == pairs[j - 1][0] or pairs[j][1] == pairs[j - 1][1]:
                candidates = [k for k in range(j + 1, min(len(pairs), j + 200))
                              if pairs[k][0] != pairs[j - 1][0] and pairs[k][1] != pairs[j - 1][1]]
                if candidates:
                    k = rng.choice(candidates)
                    pairs[j], pairs[k] = pairs[k], pairs[j]
    return pairs


def make_entities(i: int, profile: Dict[str, object], rng: random.Random, plot_variant: int) -> Dict[str, object]:
    shared_family = 50000 + i
    people = [
        full_name(i * 9, shared_family),
        full_name(i * 9 + 1, shared_family),
    ] + [full_name(i * 9 + j) for j in range(2, 8)]

    site_root = coined(i, salt=7)
    site = f"{site_root} {rng.choice(profile['site_suffix'])}"  # type: ignore[index]
    secondary_root = coined(i, salt=8)
    secondary_site = f"{secondary_root} Registry Annex"

    room_labels = profile["rooms"]  # type: ignore[index]
    rng.shuffle(room_labels := list(room_labels))
    rooms = [f"{site_root} {room_labels[j]} {chr(65 + j)}-{10 + ((i * 7 + j * 3) % 80)}" for j in range(7)]
    address = f"Parcel Ring {3 + i % 17}, Door {chr(65 + (i % 20))}, {site_root} precinct"

    route_labels = list(profile["routes"])  # type: ignore[index]
    rng.shuffle(route_labels)
    routes = [f"{site_root} {route_labels[j % len(route_labels)]} Route-{chr(75 + j)}" for j in range(4)]

    object_labels = list(profile["objects"])  # type: ignore[index]
    rng.shuffle(object_labels)
    prefix = profile["key"].split("_")[0]  # type: ignore[index]
    serials = [serial_for(i, j, str(prefix)) for j in range(8)]
    objects = [f"{object_labels[j % len(object_labels)]} {serials[j]}" for j in range(8)]

    callsigns = [
        f"{ADJECTIVES[(i * 3 + j * 11) % len(ADJECTIVES)]}-{NOUNS[(i * 5 + j * 13) % len(NOUNS)]}-{coined(i * 4 + j, salt=10)}"
        for j in range(4)
    ]

    roles = list(profile["roles"])  # type: ignore[index]
    rng.shuffle(roles)

    event = f"{coined(i, salt=11)} {PLOT_MOTIFS[plot_variant]} {rng.choice(profile['events'])}"  # type: ignore[index]
    action = rng.choice(profile["actions"])  # type: ignore[index]
    record = rng.choice(profile["records"])  # type: ignore[index]
    term, term_definition = LOCAL_TERMS[i % len(LOCAL_TERMS)]

    return {
        "people": people,
        "site_root": site_root,
        "site": site,
        "secondary_site": secondary_site,
        "rooms": rooms,
        "address": address,
        "routes": routes,
        "objects": objects,
        "serials": serials,
        "callsigns": callsigns,
        "roles": roles,
        "event": event,
        "action": action,
        "record": record,
        "term": term,
        "term_definition": term_definition,
        "plot_variant": plot_variant,
        "dates": [date_for(i, j) for j in range(4)],
        "times": [time_for(i, j) for j in range(6)],
    }


def answer_variants_person(person: str, action_phrase: str) -> List[str]:
    return [
        f"{person} {action_phrase}.",
        f"The record identifies {person} as the person who {action_phrase}.",
        f"It was {person} who {action_phrase}.",
        f"The named person is {person}; they {action_phrase}.",
    ]


def answer_variants_value(value: str, subject: str) -> List[str]:
    return [
        f"{subject} was {value}.",
        f"The record determines {value} for {subject}.",
        f"The correct {subject} is {value}.",
        f"According to the controlling record, {subject} was {value}.",
    ]


def unknown_variants(canonical: str) -> List[str]:
    return [
        canonical,
        "UNKNOWN",
        "unknown",
        "cannot be determined",
        "The supplied record does not determine the answer.",
    ]


def task_data(task: str, i: int, e: Dict[str, object], profile: Dict[str, object], rng: random.Random, plot_variant: int) -> TaskData:
    p = e["people"]  # type: ignore[assignment]
    r = e["rooms"]  # type: ignore[assignment]
    q = e["routes"]  # type: ignore[assignment]
    o = e["objects"]  # type: ignore[assignment]
    s = e["serials"]  # type: ignore[assignment]
    c = e["callsigns"]  # type: ignore[assignment]
    d = e["dates"]  # type: ignore[assignment]
    t = e["times"]  # type: ignore[assignment]
    event = str(e["event"])
    record = str(e["record"])
    unit = str(profile["unit"])
    action_label = ACTION_LABELS[plot_variant]
    evidence_record = EVIDENCE_RECORDS[plot_variant]
    exception_condition = EXCEPTION_CONDITIONS[plot_variant]
    count_mark = COUNT_MARKS[plot_variant]
    alert_label = ALERT_LABELS[plot_variant]
    verifier_label = VERIFIER_LABELS[plot_variant]
    route_completion = ROUTE_COMPLETION_LABELS[plot_variant]

    if task == "alias_resolution":
        decisive_b = f'The sealed assignment line paired with the {evidence_record} states that the call-sign "{c[0]}" belonged to {p[0]} from {d[1]} through final closure.'
        decisive_c = f'The sealed assignment line paired with the {evidence_record} states that the call-sign "{c[0]}" belonged to {p[1]} from {d[1]} through final closure.'
        decisive_u = f'The assignment line paired with the {evidence_record} for call-sign "{c[0]}" is missing; {p[0]} and {p[1]} were both eligible, and no later document names its holder.'
        evidence = f'The {evidence_record} records that "{c[0]}" completed the {action_label} of {o[0]} during the {event} after presenting both route stamps.'
        rule = 'For this operation, a sealed call-sign assignment controls over any unsigned roster, duty list, or spoken nickname.'
        distractor = f'An unsigned planning card proposed {p[1]} for the {action_label}, while a witness later guessed that {p[2]} used "{c[0]}"; neither statement is an assignment record.'
        action_phrase = f"completed the {action_label} of {o[0]} during the {event}"
        ans_b = f"{p[0]} completed the {action_label} of {o[0]} during the {event}."
        ans_c = f"{p[1]} completed the {action_label} of {o[0]} during the {event}."
        ans_u = f'UNKNOWN — the {evidence_record} identifies call-sign "{c[0]}" but the surviving record does not identify its holder.'
        return TaskData(
            decisive_b, decisive_c, decisive_u, [evidence], [rule], [distractor],
            f'Which named person completed the {action_label} of {o[0]} during the {event}?',
            f'The {evidence_record} uses call-sign "{c[0]}" for the {action_label} of {o[0]}. Who was the actual person behind it?',
            ans_b, ans_c, ans_u,
            [ans_b] + answer_variants_person(str(p[0]), action_phrase)[1:],
            [ans_c] + answer_variants_person(str(p[1]), action_phrase)[1:],
            unknown_variants(ans_u),
            [str(p[0]), str(s[0])], [str(p[1]), str(s[0])], ["UNKNOWN"],
            [str(p[1]), str(p[2])], [str(p[0]), str(p[2])], [str(p[0]), str(p[1])],
            [rule, evidence, decisive_b], [rule, evidence, decisive_c], [rule, evidence, decisive_u],
        )

    if task == "object_state_tracking":
        shell_label = ["photography shell", "label mock-up", "reference housing", "training duplicate", "empty display copy"][plot_variant % 5]
        identify = f'The target item is the full serial {s[0]}; {s[1]} is a {shell_label} with a similar number and is not the same physical object.'
        decisive_b = f'The countersigned {evidence_record} places {o[0]} in {r[0]} after the last completed {action_label}.'
        decisive_c = f'The countersigned {evidence_record} places {o[0]} in {r[1]} after the last completed {action_label}.'
        decisive_u = f'The destination field on the countersigned {evidence_record} is abraded; the surviving marks fit either {r[0]} or {r[1]}, and no later receipt resolves the location.'
        rule = f'A countersigned {evidence_record} supersedes proposed destinations, intermediate scans, and unsigned shelf notes for this {event}.'
        evidence = f'The movement log shows that the full serial {s[0]} left {r[2]} for the {action_label} after its seal was checked; the similar {shell_label} {s[1]} remained behind.'
        distractor = f'The original plan named {r[1]} as the destination, and an intermediate scanner pinged {r[3]}, but neither entry was the final {evidence_record}.'
        ans_b = f"The final verified location of {o[0]} was {r[0]}."
        ans_c = f"The final verified location of {o[0]} was {r[1]}."
        ans_u = f"UNKNOWN — the final custody destination for {o[0]} is not recoverable from the surviving record."
        return TaskData(
            decisive_b, decisive_c, decisive_u, [identify, evidence], [rule], [distractor],
            f'Where was {o[0]} located after the final completed {action_label} in the {event}?',
            f'Using the controlling {evidence_record} rather than the plan or intermediate scan, what was the final location of full serial {s[0]}?',
            ans_b, ans_c, ans_u,
            [ans_b] + answer_variants_value(str(r[0]), f"the final location of {o[0]}")[1:],
            [ans_c] + answer_variants_value(str(r[1]), f"the final location of {o[0]}")[1:],
            unknown_variants(ans_u),
            [str(s[0]), str(r[0])], [str(s[0]), str(r[1])], ["UNKNOWN"],
            [str(r[1]), str(r[3]), str(s[1])], [str(r[0]), str(r[3]), str(s[1])], [str(r[0]), str(r[1])],
            [identify, rule, evidence, decisive_b], [identify, rule, evidence, decisive_c], [identify, rule, evidence, decisive_u],
        )

    if task == "temporal_correction":
        planned = f'The provisional schedule placed the {action_label} at {t[0]} on {d[1]}, but that sheet was never countersigned.'
        actual_time_b = t[3]
        actual_time_c = t[4] if t[4] != t[3] else t[5]
        decisive_b = f'After the {exception_condition}, the signed timing amendment fixes the completed {action_label} in the {event} at {actual_time_b} on {d[2]}.'
        decisive_c = f'After the {exception_condition}, the signed timing amendment fixes the completed {action_label} in the {event} at {actual_time_c} on {d[2]}.'
        decisive_u = f'The signed timing amendment confirms {d[2]} after the {exception_condition}, but its hour-and-minute field is torn away; two synchronized clocks preserve {actual_time_b} and {actual_time_c}, with no rule choosing between them.'
        rule = 'A signed timing amendment replaces every earlier planned slot, even when an earlier sheet carries more detailed notes.'
        evidence = f'The {evidence_record} says the {action_label} finished in the "amended slot" and deliberately omits a duplicate clock time.'
        distractor = f'A margin calculation used {t[1]} as an estimate, while a witness remembered {t[2]}; both were entered before the timing amendment.'
        ans_b = f"The {action_label} in the {event} occurred at {actual_time_b} on {d[2]}."
        ans_c = f"The {action_label} in the {event} occurred at {actual_time_c} on {d[2]}."
        ans_u = f"UNKNOWN — the date is fixed as {d[2]}, but the surviving record does not determine the completion time."
        return TaskData(
            decisive_b, decisive_c, decisive_u, [planned, evidence], [rule], [distractor],
            f'At what corrected time was the {action_label} in the {event} completed on {d[2]}?',
            f'Ignoring the provisional schedule and witness estimates, what time does the controlling amendment give for the {action_label} on {d[2]}?',
            ans_b, ans_c, ans_u,
            [ans_b, f"Completion occurred at {actual_time_b} on {d[2]}.", f"The corrected completion time was {actual_time_b} on {d[2]}.", f"According to the signed amendment, it finished at {actual_time_b} on {d[2]}."],
            [ans_c, f"Completion occurred at {actual_time_c} on {d[2]}.", f"The corrected completion time was {actual_time_c} on {d[2]}.", f"According to the signed amendment, it finished at {actual_time_c} on {d[2]}."],
            unknown_variants(ans_u),
            [actual_time_b, str(d[2])], [actual_time_c, str(d[2])], ["UNKNOWN"],
            [str(t[0]), str(t[1]), str(t[2]), actual_time_c], [str(t[0]), str(t[1]), str(t[2]), actual_time_b], [actual_time_b, actual_time_c],
            [planned, rule, evidence, decisive_b], [planned, rule, evidence, decisive_c], [rule, evidence, decisive_u],
        )

    if task == "rule_exception":
        token_kind = ["Exception Token", "Override Seal", "Substitute Warrant", "Closure Pass", "Authority Tab"][plot_variant % 5]
        pass_name = f"{ADJECTIVES[(i + 9) % len(ADJECTIVES)]} {token_kind}"
        rule = f'Ordinarily only {p[2]}, the section lead, may authorize the {action_label}; during the {exception_condition}, however, the named holder of the {pass_name} has that authority instead.'
        event_sentence = f'The controlling log declares the {exception_condition} before {o[0]} moved, and the authorization box cites the {pass_name} rather than the ordinary section-lead rule.'
        decisive_b = f'The sealed issue register names {p[0]} as the sole holder of the {pass_name} for {d[2]}.'
        decisive_c = f'The sealed issue register names {p[1]} as the sole holder of the {pass_name} for {d[2]}.'
        decisive_u = f'The issue register page for the {pass_name} lacks its required seal and contains competing names for {p[0]} and {p[1]}; no valid replacement page survives.'
        distractor = f'{p[2]} signed the original ordinary-rule plan, and {p[3]} witnessed the movement, but neither fact overrides the {exception_condition} exception.'
        ans_b = f"{p[0]} had authority to authorize the {action_label} as holder of the {pass_name}."
        ans_c = f"{p[1]} had authority to authorize the {action_label} as holder of the {pass_name}."
        ans_u = f"UNKNOWN — the valid record does not determine who held the {pass_name} during the {exception_condition}."
        return TaskData(
            decisive_b, decisive_c, decisive_u, [event_sentence], [rule], [distractor],
            f'Who had valid authority to authorize the {action_label} of {o[0]} during the {exception_condition}?',
            f'Under the {exception_condition} exception, which named person could validly approve the {action_label} of {o[0]}?',
            ans_b, ans_c, ans_u,
            [ans_b, f"{p[0]} was the authorized holder of the {pass_name}.", f"The valid authorizer was {p[0]} because the exception token was issued to them.", f"Authority belonged to {p[0]} under the {exception_condition} exception."],
            [ans_c, f"{p[1]} was the authorized holder of the {pass_name}.", f"The valid authorizer was {p[1]} because the exception token was issued to them.", f"Authority belonged to {p[1]} under the {exception_condition} exception."],
            unknown_variants(ans_u),
            [str(p[0])], [str(p[1])], ["UNKNOWN"],
            [str(p[1]), str(p[2]), str(p[3])], [str(p[0]), str(p[2]), str(p[3])], [str(p[0]), str(p[1]), str(p[2])],
            [rule, event_sentence, decisive_b], [rule, event_sentence, decisive_c], [rule, event_sentence, decisive_u],
        )

    if task == "count_exclusions_duplicates":
        regular_count = 3 + (i % 4)
        base_special_counts = (i % 2 == 0)
        count_label = ["settled", "closed", "verified", "accounted", "resolved"][plot_variant % 5]
        special_status_b = f"bears the {count_mark}, so it is a separate physical unit"
        special_status_c = f"does not bear the {count_mark}, so it is only a reference copy"
        if not base_special_counts:
            special_status_b, special_status_c = special_status_c, special_status_b
        count_b = regular_count + 1 + (1 if base_special_counts else 0)
        count_c = regular_count + 1 + (0 if base_special_counts else 1)
        rule1 = f'In this {record}, a "{count_label} {unit[:-1] if unit.endswith("s") else unit}" means one unique, non-provisional physical unit.'
        rule2 = f'A reference copy is excluded unless it carries the {count_mark}; an entry marked "cleared" still counts because cleared means inspected and resealed, not removed.'
        evidence = f'The reconciliation lists {regular_count} ordinary sealed entries, one cleared entry that remains present, one provisional entry, one repeated scan of an ordinary serial, and the special reference entry {s[1]}.'
        decisive_b = f'The physical label on reference entry {s[1]} {special_status_b}.'
        decisive_c = f'The physical label on reference entry {s[1]} {special_status_c}.'
        decisive_u = f'The part of the physical label that would show whether reference entry {s[1]} has the {count_mark} is unreadable, and no photograph or second inspection survives.'
        wrong_high = max(count_b, count_c) + 2
        wrong_low = max(1, min(count_b, count_c) - 2)
        distractor = f'A draft subtotal says {wrong_high}, but it includes the provisional entry and the repeated scan; a second margin note says {wrong_low}, but it incorrectly removes the cleared entry.'
        ans_b = f"The corrected count is {count_b} qualifying {unit}."
        ans_c = f"The corrected count is {count_c} qualifying {unit}."
        ans_u = f"UNKNOWN — the count depends on whether {s[1]} carries the {count_mark}, which the record does not determine."
        return TaskData(
            decisive_b, decisive_c, decisive_u, [evidence], [rule1, rule2], [distractor],
            f'How many qualifying {unit} are in the final corrected reconciliation?',
            f'After applying the file’s duplicate, provisional, cleared, and {count_mark} rules, what is the final count of {unit}?',
            ans_b, ans_c, ans_u,
            [ans_b, f"There are {count_b} qualifying {unit}.", f"The final corrected total is {count_b} {unit}.", f"Applying all exclusions gives a count of {count_b}."],
            [ans_c, f"There are {count_c} qualifying {unit}.", f"The final corrected total is {count_c} {unit}.", f"Applying all exclusions gives a count of {count_c}."],
            unknown_variants(ans_u),
            [str(count_b)], [str(count_c)], ["UNKNOWN"],
            [f"count is {count_c}", f"total is {wrong_high}", f"count of {wrong_low}"],
            [f"count is {count_b}", f"total is {wrong_high}", f"count of {wrong_low}"],
            [f"count is {count_b}", f"count is {count_c}"],
            [rule1, rule2, evidence, decisive_b], [rule1, rule2, evidence, decisive_c], [rule1, rule2, evidence, decisive_u],
        )

    if task == "causal_disambiguation":
        cause_pairs = [
            (f"a jam in {o[2]}", f"a loose coupling on {o[3]}"),
            (f"a blocked latch on {o[2]}", f"a stale route table in {o[3]}"),
            (f"a misaligned seal on {o[2]}", f"timing drift in {o[3]}"),
            (f"a duplicated label on {o[2]}", f"a reversed connector on {o[3]}"),
            (f"a missing gate stamp on {o[2]}", f"an offset reading from {o[3]}"),
        ]
        cause_a, cause_b = cause_pairs[plot_variant % len(cause_pairs)]
        rule = f'The file uses "initiating cause" to mean the earliest verified condition required for the {alert_label}, not the last symptom noticed and not the person who reported it.'
        chronology = f'The {alert_label} began after {o[0]} crossed {r[3]}; the flicker at {r[4]} and the delayed reading on {o[4]} occurred only afterward.'
        decisive_b = f'The sealed diagnostic comparison shows that the {alert_label} disappears when {cause_a} is removed while every other recorded condition is held unchanged.'
        decisive_c = f'The sealed diagnostic comparison shows that the {alert_label} disappears when {cause_b} is removed while every other recorded condition is held unchanged.'
        decisive_u = f'The diagnostic comparison strip is missing; the surviving setup notes permit both {cause_a} and {cause_b}, and no controlled rerun was completed.'
        distractor = f'{p[4]} blamed the later flicker, while an unsigned repair ticket blamed {p[5]}; both statements concern observations after the {alert_label} had already begun.'
        ans_b = f"The verified initiating cause of the {alert_label} was {cause_a}."
        ans_c = f"The verified initiating cause of the {alert_label} was {cause_b}."
        ans_u = f"UNKNOWN — the surviving diagnostic record does not determine which candidate condition initiated the {alert_label}."
        return TaskData(
            decisive_b, decisive_c, decisive_u, [chronology], [rule], [distractor],
            f'What was the verified initiating cause of the {alert_label} during the {event}?',
            f'Using the file’s special definition of initiating cause and the controlled comparison, which condition actually started the {alert_label}?',
            ans_b, ans_c, ans_u,
            [ans_b, f"The {alert_label} was initiated by {cause_a}.", f"The controlled record identifies {cause_a} as the initiating cause.", f"The cause was {cause_a}, not the later symptoms."],
            [ans_c, f"The {alert_label} was initiated by {cause_b}.", f"The controlled record identifies {cause_b} as the initiating cause.", f"The cause was {cause_b}, not the later symptoms."],
            unknown_variants(ans_u),
            [cause_a], [cause_b], ["UNKNOWN"],
            [cause_b, str(p[4]), str(p[5])], [cause_a, str(p[4]), str(p[5])], [cause_a, cause_b],
            [rule, chronology, decisive_b], [rule, chronology, decisive_c], [rule, chronology, decisive_u],
        )

    if task == "multi_condition_identity":
        credential_kind = ["Badge", "Gate Seal", "Route Tab", "Verifier Pin", "Custody Marker"][plot_variant % 5]
        badge = f"{credential_kind} {ADJECTIVES[(i + 5) % len(ADJECTIVES)]}-{chr(65 + i % 20)}"
        rule = f'A valid {verifier_label} had to enter by {q[0]}, carry {o[0]}, present {badge}, and sign before {t[4]}; satisfying only three conditions does not qualify.'
        evidence = f'The {evidence_record} records one entrant using {q[0]} with {o[0]} and {badge}, followed by a signature at {t[3]}.'
        decisive_b = f'The sealed credential ledger assigns {badge} to {p[0]} for the whole verification window.'
        decisive_c = f'The sealed credential ledger assigns {badge} to {p[1]} for the whole verification window.'
        decisive_u = f'The credential-ledger entry for {badge} is split between two uncountersigned copies, one naming {p[0]} and one naming {p[1]}; no valid master copy survives.'
        distractor = f'{p[2]} used {q[0]} without the credential, {p[3]} carried {o[0]} after {t[4]}, and {p[4]} displayed a similarly marked credential with a different letter.'
        ans_b = f"{p[0]} was the valid {verifier_label} who satisfied all four conditions."
        ans_c = f"{p[1]} was the valid {verifier_label} who satisfied all four conditions."
        ans_u = f"UNKNOWN — the surviving credential records do not identify the valid holder of {badge}."
        return TaskData(
            decisive_b, decisive_c, decisive_u, [evidence], [rule], [distractor],
            f'Which named person satisfied every condition for the valid {verifier_label}?',
            f'Who can be identified as the {verifier_label} after combining the required route, object, credential, and signing-time conditions?',
            ans_b, ans_c, ans_u,
            [ans_b, f"The valid {verifier_label} was {p[0]}.", f"{p[0]} met the route, object, badge, and time requirements.", f"All four conditions identify {p[0]}."],
            [ans_c, f"The valid {verifier_label} was {p[1]}.", f"{p[1]} met the route, object, badge, and time requirements.", f"All four conditions identify {p[1]}."],
            unknown_variants(ans_u),
            [str(p[0])], [str(p[1])], ["UNKNOWN"],
            [str(p[1]), str(p[2]), str(p[3]), str(p[4])], [str(p[0]), str(p[2]), str(p[3]), str(p[4])], [str(p[0]), str(p[1])],
            [rule, evidence, decisive_b], [rule, evidence, decisive_c], [rule, evidence, decisive_u],
        )

    if task == "latest_verified_record_vs_rumor":
        rule = f'When locations conflict after the {route_completion}, the latest record counts only if it bears both the custodian seal and the auditor countersignature; later rumors and unsigned drafts do not displace it.'
        chronology = f'An early movement receipt placed {o[0]} in {r[2]}, and a later unsigned draft proposed {r[1]} as the next stop in the {event}.'
        decisive_b = f'At {t[5]} on {d[3]}, the latest doubly signed {evidence_record} verifies {r[0]} as the location of {o[0]}.'
        decisive_c = f'At {t[5]} on {d[3]}, the latest doubly signed {evidence_record} verifies {r[1]} as the location of {o[0]}.'
        decisive_u = f'Two versions of the {evidence_record} at {t[5]} on {d[3]} conflict between {r[0]} and {r[1]}; each has only one of the two required signatures, and neither is a valid latest verified record.'
        distractor = f'A message posted after {t[5]} claims that {o[0]} was seen in {r[3]}, but it has neither required signature and identifies the near-duplicate serial {s[1]} in its attachment.'
        ans_b = f"The latest verified location of {o[0]} was {r[0]}."
        ans_c = f"The latest verified location of {o[0]} was {r[1]}."
        ans_u = f"UNKNOWN — no surviving record with both required signatures determines the latest verified location of {o[0]}."
        return TaskData(
            decisive_b, decisive_c, decisive_u, [chronology], [rule], [distractor],
            f'What was the latest verified location of {o[0]} after the {route_completion}?',
            f'After rejecting rumors and incomplete versions of the {evidence_record}, where does the newest valid record place {o[0]}?',
            ans_b, ans_c, ans_u,
            [ans_b] + answer_variants_value(str(r[0]), f"the latest verified location of {o[0]}")[1:],
            [ans_c] + answer_variants_value(str(r[1]), f"the latest verified location of {o[0]}")[1:],
            unknown_variants(ans_u),
            [str(s[0]), str(r[0])], [str(s[0]), str(r[1])], ["UNKNOWN"],
            [str(r[1]), str(r[2]), str(r[3]), str(s[1])], [str(r[0]), str(r[2]), str(r[3]), str(s[1])], [str(r[0]), str(r[1]), str(r[3])],
            [rule, chronology, decisive_b], [rule, chronology, decisive_c], [rule, chronology, decisive_u],
        )

    if task == "definition_binding":
        term = f"{ADJECTIVES[(i + 17) % len(ADJECTIVES)].lower()}-ready"
        property_sets = [
            (f"the {q[0]} stamp", f"the {q[1]} stamp", "route-stamp field"),
            (f"the {r[0]} seal", f"the {r[1]} seal", "room-seal field"),
            (f'the "{c[1]}" endorsement', f'the "{c[2]}" endorsement', "endorsement field"),
            (f"the {p[2]} custodian mark", f"the {p[3]} custodian mark", "custodian-mark field"),
            (f"the pre-{t[2]} time band", f"the post-{t[2]} time band", "time-band field"),
        ]
        property_a, property_b, property_field = property_sets[plot_variant % len(property_sets)]
        item_a = o[2]
        item_b = o[3]
        evidence_a = f'{item_a} is sealed, non-provisional, and bears {property_a}; it does not bear {property_b}.'
        evidence_b = f'{item_b} is sealed, non-provisional, and bears {property_b}; it does not bear {property_a}.'
        decisive_b = f'Within this document, "{term}" means a sealed, non-provisional item bearing {property_a}.'
        decisive_c = f'Within this document, "{term}" means a sealed, non-provisional item bearing {property_b}.'
        decisive_u = f'The local definition of "{term}" is damaged exactly at its {property_field}; the surviving text could refer to either {property_a} or {property_b}.'
        rule = 'Local defined terms in quotation marks override ordinary-language guesses and any informal label used by a witness.'
        distractor = f'A witness called {o[4]} "{term}" because its wrapping looked finished, but {o[4]} is still provisional and carries neither required property.'
        ans_b = f"{item_a} is the item that is {term} under the document’s local definition."
        ans_c = f"{item_b} is the item that is {term} under the document’s local definition."
        ans_u = f"UNKNOWN — the damaged definition does not determine whether {property_a} or {property_b} is required for {term}."
        return TaskData(
            decisive_b, decisive_c, decisive_u, [evidence_a, evidence_b], [rule], [distractor],
            f'Which item is {term} under this document’s local definition?',
            f'Apply the file-specific meaning of “{term}.” Which registered item satisfies it?',
            ans_b, ans_c, ans_u,
            [ans_b, f"The {term} item is {item_a}.", f"Under the local definition, {item_a} qualifies as {term}.", f"{item_a} satisfies the document’s {term} rule."],
            [ans_c, f"The {term} item is {item_b}.", f"Under the local definition, {item_b} qualifies as {term}.", f"{item_b} satisfies the document’s {term} rule."],
            unknown_variants(ans_u),
            [str(s[2]), term], [str(s[3]), term], ["UNKNOWN"],
            [str(s[3]), str(s[4])], [str(s[2]), str(s[4])], [str(s[2]), str(s[3])],
            [rule, evidence_a, evidence_b, decisive_b], [rule, evidence_a, evidence_b, decisive_c], [rule, evidence_a, evidence_b, decisive_u],
        )

    if task == "address_route_tracking":
        route_code = c[2]
        rule = f'In this route book, a movement recorded as the {route_completion} under code "{route_code}" ends at the destination assigned to that code, even if it passes other rooms or an earlier plan names another address.'
        evidence = f'The {evidence_record} for {o[0]} says that code "{route_code}" completed the {route_completion} after departure from {r[2]} by way of {q[2]}.'
        decisive_b = f'The sealed route key assigns code "{route_code}" to final address {r[0]}.'
        decisive_c = f'The sealed route key assigns code "{route_code}" to final address {r[1]}.'
        decisive_u = f'The line assigning a final address to code "{route_code}" is missing from the route key; surviving maps permit both {r[0]} and {r[1]}, and the endpoint stamp is shared by those two rooms.'
        distractor = f'The loading plan proposed {r[1]}, an intermediate scan occurred at {r[3]}, and a copied map labels {r[4]} with a similar code; none is the sealed route-key assignment.'
        ans_b = f"The {route_completion} delivered {o[0]} to {r[0]}."
        ans_c = f"The {route_completion} delivered {o[0]} to {r[1]}."
        ans_u = f"UNKNOWN — the surviving route key does not determine the final address assigned to code {route_code}."
        return TaskData(
            decisive_b, decisive_c, decisive_u, [evidence], [rule], [distractor],
            f'What was the final address of {o[0]} after code {route_code} completed the {route_completion}?',
            f'Follow the sealed route-key meaning of code {route_code}. Where did the {route_completion} of {o[0]} end?',
            ans_b, ans_c, ans_u,
            [ans_b, f"{o[0]} was delivered to {r[0]}.", f"The final address was {r[0]}.", f"Code {route_code} ended at {r[0]}, which was the final destination."],
            [ans_c, f"{o[0]} was delivered to {r[1]}.", f"The final address was {r[1]}.", f"Code {route_code} ended at {r[1]}, which was the final destination."],
            unknown_variants(ans_u),
            [str(s[0]), str(r[0])], [str(s[0]), str(r[1])], ["UNKNOWN"],
            [str(r[1]), str(r[3]), str(r[4])], [str(r[0]), str(r[3]), str(r[4])], [str(r[0]), str(r[1])],
            [rule, evidence, decisive_b], [rule, evidence, decisive_c], [rule, evidence, decisive_u],
        )

    raise ValueError(task)


def common_sections(i: int, title: str, e: Dict[str, object], profile: Dict[str, object], td: TaskData, rng: random.Random) -> List[Tuple[str, str]]:
    p = e["people"]  # type: ignore[assignment]
    r = e["rooms"]  # type: ignore[assignment]
    q = e["routes"]  # type: ignore[assignment]
    o = e["objects"]  # type: ignore[assignment]
    s = e["serials"]  # type: ignore[assignment]
    c = e["callsigns"]  # type: ignore[assignment]
    roles = e["roles"]  # type: ignore[assignment]
    d = e["dates"]  # type: ignore[assignment]
    t = e["times"]  # type: ignore[assignment]
    site = str(e["site"])
    site_root = str(e["site_root"])
    address = str(e["address"])
    secondary = str(e["secondary_site"])
    event = str(e["event"])
    record = str(e["record"])
    term = str(e["term"])
    term_definition = str(e["term_definition"])
    plot_variant = int(e["plot_variant"])
    authority = AUTHORITY_ORDERS[i % len(AUTHORITY_ORDERS)]
    exception_condition = EXCEPTION_CONDITIONS[plot_variant]
    margin_conditions = [
        "seal check", "route clearance", "duplicate-label review", "clock synchronization",
        "custody confirmation", "room availability", "second signature", "weather clearance",
        "inventory recount", "access-window confirmation", "manifest comparison",
        "permit seal", "sample-chain check", "gallery clearance", "watch relief",
        "ledger reconciliation", "emergency-route release",
    ]
    obstruction_types = [
        "temporary barrier", "unverified seal", "misplaced route placard", "clock-check hold",
        "custody dispute", "room-key mismatch", "missing second stamp", "weather latch",
        "count discrepancy", "access queue", "manifest mismatch", "permit hold",
        "sample-chain pause", "gallery screen", "watch turnover", "ledger lock",
        "emergency detour",
    ]

    sections: List[Tuple[str, str]] = []
    sections.append((
        "Scope and compilation",
        f'{title} consolidates the working papers for the {event} conducted at {site} from {d[0]} through {d[3]}. '
        f'The site uses the invented parcel address {address}; it is an internal locator, not a public street address. '
        f'The compilation draws from the {record}, gate stamps, room receipts, object labels, and signed amendments. '
        f'Plans are described because they explain later confusion, but a plan is not proof that an action occurred. '
        f'The compiler preserved conflicting notes rather than silently harmonizing them, so readers must apply the stated authority order and local definitions.'
    ))
    sections.append((
        "Authority and vocabulary",
        f'The record hierarchy is explicit, from highest to lowest: {authority[0]}; {authority[1]}; {authority[2]}; {authority[3]}. '
        f'A higher-ranked record controls over a lower-ranked one, and timestamps break ties only between records of equal authority. In this file, the word "{term}" means {term_definition}. '
        f'The word therefore cannot be interpreted by ordinary English. A separate rule says that a document labeled "copy" may preserve facts but cannot itself supply a missing signature. '
        f'An exception applies during the {exception_condition}: a properly issued exception token may substitute for the ordinary officeholder, but an unsigned token proposal has no force.'
    ))
    personnel = "; ".join(f"{p[j]}, {roles[j % len(roles)]}" for j in range(7))
    sections.append((
        "Personnel and names",
        f'The principal personnel were {personnel}. {p[0]} and {p[1]} share a family name and are often confused in hurried handwriting, but they are different people. '
        f'Temporary call-signs appearing in radio and gate notes include "{c[0]}", "{c[1]}", and "{c[2]}"; a call-sign is not a job title. '
        f'{p[2]} supervised the first shift, {p[3]} kept the room book, {p[4]} collected witness statements, {p[5]} inspected routes, and {p[6]} performed the final cross-check. '
        f'No person may be inferred to hold a call-sign, token, or badge merely because that person was on duty.'
    ))
    sections.append((
        "Site directory",
        f'The internal directory lists {r[0]}, {r[1]}, {r[2]}, {r[3]}, and {r[4]} as separate controlled locations. '
        f'{q[0]} links {r[2]} with the central checkpoint; {q[1]} passes the service desk; {q[2]} reaches the loading or receiving side; and {q[3]} is an inspection-only bypass. '
        f'A route scan records passage, not final delivery. {secondary} stores duplicate paperwork but no physical object from this operation. '
        f'Room suffixes are significant: a label ending A-{r[0].split("-")[-1]} is not interchangeable with one ending B-{r[1].split("-")[-1]}, even when the descriptive room words look similar.'
    ))
    sections.append((
        "Object register",
        f'The object register distinguishes {o[0]}, {o[1]}, {o[2]}, {o[3]}, {o[4]}, and {o[5]}. '
        f'The first two serials, {s[0]} and {s[1]}, differ only near the end and caused repeated transcription errors. '
        f'Quantities were recorded separately: {2 + i % 7} sealed packets accompanied {o[2]}, {4 + i % 9} labels accompanied {o[3]}, and {1 + i % 4} spare sleeves accompanied {o[4]}. '
        f'Accessory quantities do not turn one registered object into several objects. The register also marks {o[5]} as provisional, meaning that it cannot satisfy any rule requiring a completed or settled entry.'
    ))
    sections.append((
        "Initial plan",
        f'At {t[0]} on {d[0]}, an unsigned planning sheet proposed that {p[1]} move {o[0]} from {r[2]} to {r[1]} by {q[1]}. '
        f'The plan also reserved {r[3]} for a count of {6 + i % 8} {profile["unit"]} and listed {p[2]} as the expected approver. '
        f'Before work began, the route steward wrote "subject to {margin_conditions[plot_variant]}" across the page. The plan remained in the file after later changes, which is why its names, room, and time recur in witness recollections. '
        f'Nothing on the planning sheet is marked completed, received, or countersigned.'
    ))
    sections.append((
        "First operational interval",
        f'At {t[1]} on {d[1]}, {p[3]} opened the room book and recorded {o[2]} at {r[4]}. '
        f'{p[5]} then found a {obstruction_types[plot_variant]} on {q[1]} and marked that route "{term}" under the file-specific definition. '
        f'The obstruction did not affect {q[0]} or {q[2]}. At {t[2]}, {p[4]} saw a person carrying a case with a serial beginning {s[0].split("-")[0]} but could not read the final digits. '
        f'That observation is useful for chronology but cannot distinguish {s[0]} from {s[1]}.'
    ))
    sections.append((
        "Task-specific rules",
        " ".join(td.task_rule_sentences) + " " +
        f'In addition, any exception must be documented before the action it authorizes, and a later signature cannot retroactively validate an explicitly rejected proposal. '
        f'A room receipt proves custody only for the exact serial written on it. When a field says "same as plan," the phrase imports only the referenced field, not every name or time on the planning sheet.'
    ))
    sections.append((
        "Operational evidence",
        " ".join(td.evidence_sentences) + " " +
        f'A separate gate scan at {t[3]} records {p[6]} checking the seal of {o[3]}, which is unrelated to the question-bearing action. '
        f'The compiler retained this neighboring entry because its timestamp is close enough to be a tempting false lead. The scan does not mention {s[0]} and does not carry a completion box.'
    ))
    correction_modes = [
        f'corrected the quantity attached to {o[3]} from {8 + i % 6} to {7 + i % 6} and moved its routine inspection from {r[3]} to {r[4]}',
        f'replaced the proposed route {q[3]} for {o[3]} with {q[0]} while leaving its destination unchanged',
        f'replaced the provisional check time {t[2]} for {o[3]} with {t[3]} and voided the earlier clock box',
        f'corrected the room for {o[3]} from {r[3]} to {r[4]} after the first room receipt was found to carry the wrong suffix',
        f'replaced {p[4]} with {p[5]} as the inspector for {o[3]} after the original assignment was declined',
    ]
    correction_mode = correction_modes[plot_variant % len(correction_modes)]
    sections.append((
        "Signed correction on a neighboring matter",
        f'A two-seal amendment issued on {d[2]} {correction_mode}. '
        f'This amendment demonstrates that later signed corrections supersede earlier plans, but it concerns {o[3]}, not {o[0]}. '
        f'An older copy at {secondary} still shows the abandoned version. {p[3]} annotated the master index so that copied details would not be mistaken for completed facts.'
    ))
    sections.append((
        "Rumors and misleading similarities",
        " ".join(td.distractor_sentences) + " " +
        f'Another rumor used the short serial {s[0].split("-")[1]} without the final block, which could refer to more than one registered item. '
        f'{p[4]} also used the word "{term}" in ordinary conversation, but the compiler warns that only the defined record meaning applies. '
        f'No rumor carries both a custodian seal and an auditor countersignature, and none can fill a damaged field in a controlling document.'
    ))
    sections.append((
        "Final audit protocol",
        f'On {d[3]}, {p[6]} compared all surviving records line by line. The audit checked person names, call-signs, exact serials, route codes, room suffixes, times, quantities, and signatures as separate fields. '
        f'A match in one field was never used to infer a match in another. The audit rejected any subtotal that counted a repeated scan as a physical object and any location claim based only on passage through a route. '
        f'The closing note says that unresolved fields must remain unresolved rather than being completed from expectation, family resemblance, or majority rumor.'
    ))
    sections.append((
        "Closing note",
        f'The consolidated file therefore preserves both the authoritative chain and the discarded alternatives. It does not assume that the earliest note is correct, that the latest timestamp is automatically controlling, or that a familiar-looking serial is the target. '
        f'Readers are instructed to identify the exact question-bearing object or action, apply local definitions, check exceptions, and then use the highest-authority surviving fact. '
        f'The record ends after the {event}; no outside source, later legend, or general knowledge is relevant to the events at {site_root}.'
    ))
    # Six safe permutations of the directory/personnel/object-register block
    # change document organization without changing any facts. Later sections
    # also vary the placement of the neighboring correction and rumor digest.
    block = sections[2:5]
    permutations = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]
    order = permutations[plot_variant % len(permutations)]
    sections[2:5] = [block[j] for j in order]
    if plot_variant % 2 == 1:
        corr_idx = next(j for j, (h, _) in enumerate(sections) if h == "Signed correction on a neighboring matter")
        rumor_idx = next(j for j, (h, _) in enumerate(sections) if h == "Rumors and misleading similarities")
        sections[corr_idx], sections[rumor_idx] = sections[rumor_idx], sections[corr_idx]
    return sections


FILLER_TEMPLATES = [
    "At {time}, {person} {action} {object} in {room}; the entry carries a single clerk mark and is therefore descriptive rather than controlling.",
    "A quantity check in {room} found {qty} {unit}, but {exclude} belonged to a neighboring register and were copied only for reconciliation.",
    "The route book notes that {route} was open between {time} and {time2}, while {route2} required an inspection stamp at both ends.",
    "A copied label shortened {serial} to its middle block, creating a superficial match with {serial2}; the full labels remain distinct.",
    "{person} proposed moving {object} to {room}, yet the proposal box remained blank and the movement column was crossed through before the shift ended.",
    "The desk log calls {object} \"{term}\" in the local sense, so the word records a paperwork condition and does not establish location or ownership.",
    "At {time}, {person} inspected {route} and found no broken seal; this inspection did not certify any object's final destination.",
    "A witness placed {person} near {room}, but the witness could not identify the carried serial and signed only an interview summary.",
    "The duplicate-paper annex received {qty} pages at {time}; those pages were copies of records, not additional physical {unit}.",
    "A provisional tally listed {qty}, then a clerk removed {exclude} repeated scans and left the physical objects unchanged.",
    "The room key distinguishes {room} from {room2} even though both share the same descriptive noun in abbreviated notes.",
    "A late message claimed that {route} was closed, but the route steward's timed inspection shows that the closure applied only to {route2}.",
    "{person} and {person2} exchanged desk duties at {time}; the exchange changed who wrote notes but did not transfer badges, call-signs, or exception tokens.",
    "The seal count on {object} was corrected from {qty} to {qty2}; neither figure changes the object's serial identity.",
    "One margin arrow points toward {room}, but the compiler labels it a proposed direction rather than a receipt or route-key assignment.",
    "The final index preserves an unsigned sheet because it explains a rumor, not because the sheet has authority over a signed record.",
    "At {time}, the checkpoint scanned {object} and {object2} within one minute; each retained its own serial and custody line.",
    "The phrase \"completion window\" covers the interval from {time} through {time2}; it is not itself an exact completion time.",
    "A room stamp from {room} appears on the back of a copied form, but the front identifies {object2}, not the question-bearing object.",
    "The audit counted {qty} accessory pieces beside {object}; accessories were logged as contents and not as separate registered objects.",
]


def filler_paragraph(i: int, idx: int, e: Dict[str, object], profile: Dict[str, object], rng: random.Random, max_sentences: int = 4) -> Tuple[str, str]:
    p = e["people"]  # type: ignore[assignment]
    r = e["rooms"]  # type: ignore[assignment]
    q = e["routes"]  # type: ignore[assignment]
    o = e["objects"]  # type: ignore[assignment]
    s = e["serials"]  # type: ignore[assignment]
    t = e["times"]  # type: ignore[assignment]
    term = str(e["term"])
    templates = rng.sample(FILLER_TEMPLATES, k=max_sentences)
    sentences = []
    for j, template in enumerate(templates):
        sentences.append(template.format(
            time=t[(idx + j) % 4],
            time2=t[(idx + j) % 4 + 2],
            person=p[(idx + j + 2) % len(p)],
            person2=p[(idx + j + 4) % len(p)],
            action=profile["actions"][(idx + j) % len(profile["actions"])],  # type: ignore[index]
            object=o[(idx + j + 2) % len(o)],
            object2=o[(idx + j + 4) % len(o)],
            room=r[(idx + j + 2) % len(r)],
            room2=r[(idx + j + 4) % len(r)],
            route=q[(idx + j) % len(q)],
            route2=q[(idx + j + 1) % len(q)],
            qty=3 + ((i + idx * 3 + j) % 14),
            qty2=2 + ((i + idx * 5 + j) % 11),
            exclude=1 + ((i + idx + j) % 3),
            unit=profile["unit"],
            serial=s[(idx + j + 2) % len(s)],
            serial2=s[(idx + j + 4) % len(s)],
            term=term,
        ))
    return (f"Supplemental entry {idx + 1}", " ".join(sentences))


def render_sections(sections: Sequence[Tuple[str, str]]) -> str:
    return "\n\n".join(f"{heading}.\n{text}" for heading, text in sections)


def build_context_variants(i: int, title: str, e: Dict[str, object], profile: Dict[str, object], td: TaskData, needle: str, bucket: str, rng: random.Random) -> Tuple[str, str, str, int, int]:
    sections = common_sections(i, title, e, profile, td, rng)

    if bucket == "compact_long":
        # Keep all required reasoning ingredients but remove the optional full
        # audit recap so compact documents genuinely occupy the lower end of
        # the requested 900-2,200 word range.
        sections = [section for section in sections if section[0] != "Final audit protocol"]
        sections = [
            (heading,
             f'The file closes after the {e["event"]}. Readers must use the exact serial, local definitions, documented exceptions, and the highest-authority surviving record; outside knowledge is irrelevant.'
             if heading == "Closing note" else text)
            for heading, text in sections
        ]

    # Place operational evidence opposite the decisive entry for distributed cases.
    evidence_idx = next(j for j, (h, _) in enumerate(sections) if h == "Operational evidence")
    if needle == "distributed":
        evidence_section = sections.pop(evidence_idx)
        if i % 2 == 0:
            sections.insert(len(sections) - 1, evidence_section)
        else:
            sections.insert(2, evidence_section)

    lower, upper = {
        "compact_long": (1000, 1190),
        "standard_long": (1270, 1550),
        "extended_long": (1720, 2140),
    }[bucket]
    target = lower + ((i * 53) % (upper - lower + 1))

    # Add coherent supplemental entries before inserting the decisive paragraph.
    filler_idx = 0
    while words(render_sections(sections)) < target - 90:
        section = filler_paragraph(i, filler_idx, e, profile, rng, max_sentences=3 if bucket == "compact_long" else 4)
        sections.insert(len(sections) - 1, section)
        filler_idx += 1

    def insertion_index(n: int) -> int:
        if needle == "early":
            return max(2, int(n * 0.16))
        if needle == "middle":
            return int(n * 0.50)
        if needle == "late":
            return min(n - 1, int(n * 0.84))
        # distributed: decisive fact and operational evidence are far apart.
        return max(2, int(n * 0.18)) if i % 2 == 0 else min(n - 1, int(n * 0.82))

    insert_at = insertion_index(len(sections))
    heading = f"Restricted controlling entry {coined(i, salt=12)}"

    def render(decisive: str) -> str:
        local = list(sections)
        local.insert(insert_at, (heading, decisive))
        return render_sections(local)

    base = render(td.decisive_base)
    cf = render(td.decisive_cf)
    ua = render(td.decisive_u)

    # If a compact target was slightly missed, add one short neutral supplement equally to all variants.
    min_wc = min(words(base), words(cf), words(ua))
    if min_wc < 900:
        neutral = (
            "Neutral index note",
            f'The index includes {e["objects"][6]} and {e["objects"][7]} as unrelated neighboring entries. Neither shares the full serial, route code, or controlling signature of the question-bearing record.'
        )
        sections.insert(len(sections) - 1, neutral)
        base = render(td.decisive_base)
        cf = render(td.decisive_cf)
        ua = render(td.decisive_u)

    return base, cf, ua, insert_at, len(sections) + 1


def row_for(
    sid: str,
    suffix: str,
    split: str,
    variant: str,
    domain: str,
    title: str,
    task: str,
    needle: str,
    bucket: str,
    context: str,
    question: str,
    answer: str,
    accepted: List[str],
    must_include: List[str],
    must_not_include: List[str],
    supporting: List[str],
    answerable: bool,
) -> Dict[str, object]:
    prompt = f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"
    return {
        "id": f"{sid}-{suffix}",
        "scenario_id": sid,
        "split": split,
        "variant": variant,
        "domain": domain,
        "title": title,
        "task_type": task,
        "difficulty": "hard",
        "needle_profile": needle,
        "length_bucket": bucket,
        "answerable": answerable,
        "context": context,
        "question": question,
        "prompt": prompt,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "answer": answer,
        "accepted_answers": accepted,
        "must_include": must_include,
        "must_not_include": must_not_include,
        "supporting_facts": supporting,
        "context_words": words(context),
        "intended_use": INTENDED_USE,
    }


def generate(output: Path, packs: int, seed: int, log_path: Path) -> Dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    schedule = make_schedule(packs, seed)
    prompt_hashes = set()
    title_set = set()
    context_hashes = set()
    person_names = set()
    site_names = set()
    object_serials = set()
    combo_counts: Counter[Tuple[str, str]] = Counter()
    domain_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    needle_counts: Counter[str] = Counter()
    rows_written = 0
    progress_lines: List[str] = []
    combo_occurrences: Counter[Tuple[str, str]] = Counter()
    plot_signatures = set()

    with output.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(1, packs + 1):
            rng = random.Random(seed * 1000003 + i)
            domain_idx, task_idx = schedule[i - 1]
            profile = dict(DOMAINS[domain_idx])
            task = TASK_TYPES[task_idx]
            domain_key = str(profile["key"])
            plot_variant = combo_occurrences[(domain_key, task)]
            combo_occurrences[(domain_key, task)] += 1
            if plot_variant >= len(PLOT_MOTIFS):
                raise RuntimeError(f"Too many repetitions for {domain_key}/{task}: {plot_variant + 1}")
            entities = make_entities(i, profile, rng, plot_variant)
            plot_signature = (domain_key, task, plot_variant, str(entities["event"]))
            if plot_signature in plot_signatures:
                raise RuntimeError(f"Repeated plot signature: {plot_signature}")
            plot_signatures.add(plot_signature)

            for name in entities["people"]:  # type: ignore[index]
                if name in person_names:
                    raise RuntimeError(f"Repeated person name: {name}")
                person_names.add(name)
            if entities["site"] in site_names:
                raise RuntimeError(f"Repeated site: {entities['site']}")
            site_names.add(entities["site"])
            for serial in entities["serials"]:  # type: ignore[index]
                if serial in object_serials:
                    raise RuntimeError(f"Repeated object serial: {serial}")
                object_serials.add(serial)

            title = unique_title(i, profile, str(entities["site_root"]), rng)
            if title in title_set:
                raise RuntimeError(f"Repeated title: {title}")
            title_set.add(title)

            td = task_data(task, i, entities, profile, rng, plot_variant)
            needle = ["early", "middle", "late", "distributed"][(i - 1) % 4]
            bucket = ["compact_long", "standard_long", "compact_long", "extended_long", "standard_long"][(i - 1) % 5]
            base_ctx, cf_ctx, u_ctx, _, _ = build_context_variants(i, title, entities, profile, td, needle, bucket, rng)

            sid = f"LCR-{i:04d}"
            split = "dev" if i % 10 == 0 else "train"
            rows = [
                row_for(sid, "B", split, "base", str(profile["key"]), title, task, needle, bucket, base_ctx, td.question_base, td.answer_base, td.accepted_base, td.must_base, td.must_not_base, td.support_base, True),
                row_for(sid, "P", split, "paraphrase", str(profile["key"]), title, task, needle, bucket, base_ctx, td.question_para, td.answer_base, td.accepted_base, td.must_base, td.must_not_base, td.support_base, True),
                row_for(sid, "C", split, "counterfactual", str(profile["key"]), title, task, needle, bucket, cf_ctx, td.question_base, td.answer_cf, td.accepted_cf, td.must_cf, td.must_not_cf, td.support_cf, True),
                row_for(sid, "U", split, "unanswerable", str(profile["key"]), title, task, needle, bucket, u_ctx, td.question_base, td.answer_u, td.accepted_u, td.must_u, td.must_not_u, td.support_u, False),
            ]

            # Pack-level self-check before writing.
            assert rows[0]["context"] == rows[1]["context"]
            assert rows[0]["answer"] == rows[1]["answer"]
            assert rows[0]["answer"] != rows[2]["answer"]
            assert rows[0]["question"] != rows[1]["question"]
            for row in rows:
                wc = int(row["context_words"])
                if not 900 <= wc <= 2200:
                    raise RuntimeError(f"{row['id']} context length {wc} outside 900-2200")
                for token in BANNED_TERMS:
                    if token.lower() in json.dumps(row, ensure_ascii=False).lower():
                        raise RuntimeError(f"{row['id']} contains banned term {token!r}")
                answer_lower = str(row["answer"]).lower()
                for token in row["must_include"]:  # type: ignore[index]
                    if str(token).lower() not in answer_lower:
                        raise RuntimeError(f"{row['id']} answer misses must_include {token!r}")
                for token in row["must_not_include"]:  # type: ignore[index]
                    if str(token).lower() in answer_lower:
                        raise RuntimeError(f"{row['id']} answer contains must_not_include {token!r}")
                ph = hashlib.sha256(str(row["prompt"]).encode()).hexdigest()
                if ph in prompt_hashes:
                    raise RuntimeError(f"Duplicate prompt at {row['id']}")
                prompt_hashes.add(ph)
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows_written += 1

            b_hash = hashlib.sha256(base_ctx.encode()).hexdigest()
            if b_hash in context_hashes:
                raise RuntimeError(f"Duplicate base context at {sid}")
            context_hashes.add(b_hash)

            domain = str(profile["key"])
            combo_counts[(domain, task)] += 1
            domain_counts[domain] += 1
            task_counts[task] += 1
            split_counts[split] += 1
            bucket_counts[bucket] += 1
            needle_counts[needle] += 1

            if i % 50 == 0:
                line = f"packs={i}, rows={rows_written}, last_id={sid}-U"
                print(line, flush=True)
                progress_lines.append(line)

    log_path.write_text("\n".join(progress_lines) + "\n", encoding="utf-8")
    return {
        "packs": packs,
        "rows": rows_written,
        "last_id": f"LCR-{packs:04d}-U",
        "domain_counts": dict(domain_counts),
        "task_counts": dict(task_counts),
        "split_pack_counts": dict(split_counts),
        "split_row_counts": {k: v * 4 for k, v in split_counts.items()},
        "length_bucket_pack_counts": dict(bucket_counts),
        "needle_profile_pack_counts": dict(needle_counts),
        "domain_task_min": min(combo_counts.values()),
        "domain_task_max": max(combo_counts.values()),
        "unique_titles": len(title_set),
        "unique_people": len(person_names),
        "unique_object_serials": len(object_serials),
        "unique_base_contexts": len(context_hashes),
        "unique_prompts": len(prompt_hashes),
        "unique_plot_signatures": len(plot_signatures),
        "output_bytes": output.stat().st_size,
        "seed": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/lcr_sft_v1.jsonl"))
    ap.add_argument("--packs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=271828)
    ap.add_argument("--log", type=Path, default=Path("reports/generation_progress.log"))
    ap.add_argument("--summary", type=Path, default=Path("reports/generation_summary.json"))
    args = ap.parse_args()
    summary = generate(args.output, args.packs, args.seed, args.log)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
