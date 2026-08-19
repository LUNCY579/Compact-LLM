#!/usr/bin/env python3
"""Streaming validator for Rineya LCR SFT v1 JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SYSTEM_MESSAGE = (
    "Use only the supplied context. Answer correctly and completely. Do not guess. "
    "If the context does not determine the answer, reply UNKNOWN. Brevity is optional; "
    "correctness is required."
)
INTENDED_USE = "sft_train_disjoint_from_rineya_long_context_demo_v1"
REQUIRED_KEYS = {
    "id", "scenario_id", "split", "variant", "domain", "title", "task_type",
    "difficulty", "needle_profile", "length_bucket", "answerable", "context",
    "question", "prompt", "messages", "answer", "accepted_answers",
    "must_include", "must_not_include", "supporting_facts", "context_words",
    "intended_use",
}
VARIANT_ORDER = [("B", "base"), ("P", "paraphrase"), ("C", "counterfactual"), ("U", "unanswerable")]
DOMAINS = {
    "academy_chronicle", "detective_file", "station_log", "expedition_journal",
    "museum_conservation", "lab_chronology", "civic_record", "estate_business_saga",
    "clinic_inventory", "warehouse_logistics", "archive_catalog", "ships_log",
}
TASKS = {
    "alias_resolution", "object_state_tracking", "temporal_correction", "rule_exception",
    "count_exclusions_duplicates", "causal_disambiguation", "multi_condition_identity",
    "latest_verified_record_vs_rumor", "definition_binding", "address_route_tracking",
}
BANNED_TERMS = {
    "RLC-", "Asterhall", "Bellweather", "Talia Frost", "Thalia Lark",
    "Record of the Hidden Key",
}
SERIAL_RE = re.compile(r"\b[A-Z]{2}-[A-F0-9]{3}-\d{3}\b")
PERSONNEL_RE = re.compile(r"The principal personnel were (.*?)\. [A-Z][^.]+ share a family name", re.S)
SITE_RE = re.compile(r"conducted at (.*?) from \d+ [A-Za-z]+ \d+ through", re.S)


def word_count(text: str) -> int:
    return len(text.split())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shingle_hashes(text: str, n: int = 20) -> Iterable[int]:
    tokens = re.findall(r"\b[\w'-]+\b", text.lower())
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        phrase = " ".join(tokens[i:i+n]).encode("utf-8")
        yield int.from_bytes(hashlib.blake2b(phrase, digest_size=8).digest(), "big")


def load_v1(v1_path: Path | None) -> Tuple[set[str], set[str], set[int]]:
    ids: set[str] = set()
    titles: set[str] = set()
    shingles: set[int] = set()
    if v1_path is None or not v1_path.exists():
        return ids, titles, shingles
    seen_context_hash: set[str] = set()
    with v1_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.add(str(row.get("id", "")))
            titles.add(str(row.get("title", "")))
            ctx = str(row.get("context", ""))
            ctx_hash = hashlib.sha256(ctx.encode()).hexdigest()
            if ctx_hash not in seen_context_hash:
                seen_context_hash.add(ctx_hash)
                shingles.update(shingle_hashes(ctx, 20))
    return ids, titles, shingles


def parse_person_names(context: str) -> List[str]:
    m = PERSONNEL_RE.search(context)
    if not m:
        return []
    names: List[str] = []
    for entry in m.group(1).split("; "):
        if ", " in entry:
            names.append(entry.split(", ", 1)[0].strip())
    return names


def parse_site(context: str) -> str | None:
    m = SITE_RE.search(context)
    return m.group(1).strip() if m else None


def compare_one_paragraph(a: str, b: str) -> Tuple[bool, int | None, int, int]:
    pa = a.split("\n\n")
    pb = b.split("\n\n")
    if len(pa) != len(pb):
        return False, None, len(pa), len(pb)
    diffs = [i for i, (x, y) in enumerate(zip(pa, pb)) if x != y]
    return len(diffs) == 1, (diffs[0] if len(diffs) == 1 else None), len(pa), len(pb)


def validate_row(row: Dict[str, object], errors: List[str], warnings: List[str], prompt_hashes: set[str]) -> None:
    rid = str(row.get("id", "<missing-id>"))
    missing = REQUIRED_KEYS - set(row)
    if missing:
        errors.append(f"{rid}: missing keys {sorted(missing)}")
        return

    if row["split"] not in {"train", "dev"}:
        errors.append(f"{rid}: invalid split {row['split']!r}")
    if row["domain"] not in DOMAINS:
        errors.append(f"{rid}: invalid domain {row['domain']!r}")
    if row["task_type"] not in TASKS:
        errors.append(f"{rid}: invalid task_type {row['task_type']!r}")
    if row["difficulty"] != "hard":
        warnings.append(f"{rid}: difficulty is not hard")
    if row["needle_profile"] not in {"early", "middle", "late", "distributed"}:
        errors.append(f"{rid}: invalid needle_profile")
    if row["length_bucket"] not in {"compact_long", "standard_long", "extended_long"}:
        errors.append(f"{rid}: invalid length_bucket")
    if row["intended_use"] != INTENDED_USE:
        errors.append(f"{rid}: intended_use mismatch")

    context = str(row["context"])
    question = str(row["question"])
    prompt = str(row["prompt"])
    answer = str(row["answer"])
    expected_prompt = f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"
    if prompt != expected_prompt:
        errors.append(f"{rid}: prompt does not exactly match context/question template")
    ph = hashlib.sha256(prompt.encode()).hexdigest()
    if ph in prompt_hashes:
        errors.append(f"{rid}: duplicate prompt")
    prompt_hashes.add(ph)

    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(f"{rid}: messages must contain exactly three entries")
    else:
        expected = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        if messages != expected:
            errors.append(f"{rid}: messages do not exactly match fixed system/user/assistant content")

    wc = word_count(context)
    if row["context_words"] != wc:
        errors.append(f"{rid}: context_words={row['context_words']} but actual={wc}")
    if not 900 <= wc <= 2200:
        errors.append(f"{rid}: context word count {wc} outside 900-2200")

    bucket = str(row["length_bucket"])
    bucket_ranges = {
        "compact_long": (900, 1240),
        "standard_long": (1200, 1650),
        "extended_long": (1650, 2200),
    }
    lo, hi = bucket_ranges[bucket]
    if not lo <= wc <= hi:
        errors.append(f"{rid}: {wc} words inconsistent with {bucket} range {lo}-{hi}")

    accepted = row["accepted_answers"]
    if not isinstance(accepted, list) or not 3 <= len(accepted) <= 8:
        errors.append(f"{rid}: accepted_answers must contain 3-8 strings")
    elif answer not in accepted:
        errors.append(f"{rid}: canonical answer missing from accepted_answers")
    elif any(not isinstance(x, str) or not x.strip() for x in accepted):
        errors.append(f"{rid}: accepted_answers contains a non-string or empty string")

    must = row["must_include"]
    must_not = row["must_not_include"]
    support = row["supporting_facts"]
    if not isinstance(must, list) or not must:
        errors.append(f"{rid}: must_include must be a non-empty list")
    if not isinstance(must_not, list) or not must_not:
        errors.append(f"{rid}: must_not_include must be a non-empty list")
    if not isinstance(support, list) or not 2 <= len(support) <= 6:
        errors.append(f"{rid}: supporting_facts must contain 2-6 entries")

    al = answer.lower()
    if isinstance(must, list):
        for token in must:
            if str(token).lower() not in al:
                errors.append(f"{rid}: answer misses must_include {token!r}")
    if isinstance(must_not, list):
        for token in must_not:
            if str(token).lower() in al:
                errors.append(f"{rid}: answer contains must_not_include {token!r}")
    if isinstance(support, list):
        for fact in support:
            if str(fact) not in context:
                errors.append(f"{rid}: supporting fact is not an exact substring of its context: {fact!r}")

    variant = str(row["variant"])
    answerable = bool(row["answerable"])
    if variant == "unanswerable":
        if answerable:
            errors.append(f"{rid}: unanswerable row marked answerable")
        if not answer.upper().startswith("UNKNOWN"):
            errors.append(f"{rid}: unanswerable answer does not lead with UNKNOWN")
        lower_accepted = {str(x).lower() for x in accepted} if isinstance(accepted, list) else set()
        for required in {"unknown", "cannot be determined"}:
            if required not in lower_accepted:
                errors.append(f"{rid}: unanswerable accepted_answers missing {required!r}")
    elif not answerable:
        errors.append(f"{rid}: answerable variant marked false")

    blob = json.dumps(row, ensure_ascii=False).lower()
    for banned in BANNED_TERMS:
        if banned.lower() in blob:
            errors.append(f"{rid}: contains banned v1 term {banned!r}")


def validate(input_path: Path, expected_packs: int, v1_path: Path | None) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    v1_ids, v1_titles, v1_shingles = load_v1(v1_path)

    prompt_hashes: set[str] = set()
    title_owner: Dict[str, str] = {}
    person_owner: Dict[str, str] = {}
    serial_owner: Dict[str, str] = {}
    site_owner: Dict[str, str] = {}
    base_context_hashes: set[str] = set()

    counts = Counter()
    domain_counts = Counter()
    task_counts = Counter()
    combo_counts = Counter()
    split_pack_counts = Counter()
    bucket_counts = Counter()
    needle_counts = Counter()
    variant_counts = Counter()
    word_counts: List[int] = []
    words_by_bucket: Dict[str, List[int]] = defaultdict(list)
    pack_rows: List[Dict[str, object]] = []
    pack_count = 0
    row_count = 0
    v1_shingle_overlaps = 0

    def process_pack(rows: List[Dict[str, object]]) -> None:
        nonlocal pack_count, v1_shingle_overlaps
        if len(rows) != 4:
            errors.append(f"Incomplete pack near {rows[0].get('scenario_id') if rows else '<eof>'}: {len(rows)} rows")
            return
        pack_count += 1
        sid = f"LCR-{pack_count:04d}"
        by_variant = {str(r["variant"]): r for r in rows}
        if set(by_variant) != {"base", "paraphrase", "counterfactual", "unanswerable"}:
            errors.append(f"{sid}: variants are {sorted(by_variant)}")
            return
        b, p, c, u = (by_variant[x] for x in ("base", "paraphrase", "counterfactual", "unanswerable"))

        for suffix, variant in VARIANT_ORDER:
            r = by_variant[variant]
            expected_id = f"{sid}-{suffix}"
            if r["id"] != expected_id:
                errors.append(f"{sid}: expected {expected_id}, found {r['id']}")
            if r["scenario_id"] != sid:
                errors.append(f"{r['id']}: scenario_id should be {sid}")

        shared_fields = ["scenario_id", "split", "domain", "title", "task_type", "difficulty", "needle_profile", "length_bucket", "intended_use"]
        for field in shared_fields:
            vals = {json.dumps(r[field], sort_keys=True) for r in rows}
            if len(vals) != 1:
                errors.append(f"{sid}: pack-level field {field} differs across variants")

        if b["context"] != p["context"]:
            errors.append(f"{sid}: base and paraphrase contexts are not character-identical")
        if b["answer"] != p["answer"]:
            errors.append(f"{sid}: base and paraphrase answers differ")
        if b["accepted_answers"] != p["accepted_answers"] or b["must_include"] != p["must_include"] or b["must_not_include"] != p["must_not_include"]:
            errors.append(f"{sid}: base/paraphrase scoring fields differ")
        if b["question"] == p["question"]:
            errors.append(f"{sid}: base and paraphrase questions are identical")
        if b["answer"] == c["answer"]:
            errors.append(f"{sid}: base and counterfactual answers are identical")
        if b["question"] != c["question"] or b["question"] != u["question"]:
            errors.append(f"{sid}: counterfactual/unanswerable question differs from base")

        one_cf, diff_idx, para_n, _ = compare_one_paragraph(str(b["context"]), str(c["context"]))
        if not one_cf:
            errors.append(f"{sid}: counterfactual must change exactly one paragraph")
        one_u, diff_u, _, _ = compare_one_paragraph(str(b["context"]), str(u["context"]))
        if not one_u:
            errors.append(f"{sid}: unanswerable must replace exactly one decisive paragraph")
        if one_cf and one_u and diff_idx != diff_u:
            errors.append(f"{sid}: counterfactual and unanswerable alter different decisive paragraphs")

        needle = str(b["needle_profile"])
        if diff_idx is not None and para_n > 1:
            ratio = diff_idx / (para_n - 1)
            if needle == "early" and ratio > 0.32:
                errors.append(f"{sid}: early needle at paragraph ratio {ratio:.3f}")
            if needle == "middle" and not 0.30 <= ratio <= 0.70:
                errors.append(f"{sid}: middle needle at paragraph ratio {ratio:.3f}")
            if needle == "late" and ratio < 0.68:
                errors.append(f"{sid}: late needle at paragraph ratio {ratio:.3f}")
            if needle == "distributed":
                positions = []
                context = str(b["context"])
                for fact in b["supporting_facts"]:
                    pos = context.find(str(fact))
                    if pos >= 0:
                        positions.append(pos / max(1, len(context)))
                if len(positions) >= 2 and max(positions) - min(positions) < 0.35:
                    errors.append(f"{sid}: distributed supporting facts span only {max(positions)-min(positions):.3f}")

        title = str(b["title"])
        if title in title_owner:
            errors.append(f"{sid}: title repeats title from {title_owner[title]}: {title}")
        title_owner[title] = sid
        if title in v1_titles:
            errors.append(f"{sid}: title duplicates frozen v1 title {title!r}")

        context = str(b["context"])
        ctx_hash = hashlib.sha256(context.encode()).hexdigest()
        if ctx_hash in base_context_hashes:
            errors.append(f"{sid}: duplicate canonical context")
        base_context_hashes.add(ctx_hash)
        if v1_shingles:
            overlaps = sum(1 for h in shingle_hashes(context, 20) if h in v1_shingles)
            if overlaps:
                v1_shingle_overlaps += overlaps
                errors.append(f"{sid}: contains {overlaps} exact 20-word sequence(s) from frozen v1")

        people = parse_person_names(context)
        if len(people) != 7:
            errors.append(f"{sid}: could not parse exactly seven principal people (found {len(people)})")
        for name in people:
            prev = person_owner.get(name)
            if prev and prev != sid:
                errors.append(f"{sid}: person {name!r} already used by {prev}")
            person_owner[name] = sid

        site = parse_site(context)
        if not site:
            errors.append(f"{sid}: could not parse site name")
        else:
            prev = site_owner.get(site)
            if prev and prev != sid:
                errors.append(f"{sid}: site {site!r} already used by {prev}")
            site_owner[site] = sid

        for serial in set(SERIAL_RE.findall(context)):
            prev = serial_owner.get(serial)
            if prev and prev != sid:
                errors.append(f"{sid}: serial {serial} already used by {prev}")
            serial_owner[serial] = sid

        split = str(b["split"])
        domain = str(b["domain"])
        task = str(b["task_type"])
        split_pack_counts[split] += 1
        domain_counts[domain] += 1
        task_counts[task] += 1
        combo_counts[(domain, task)] += 1
        bucket_counts[str(b["length_bucket"])] += 1
        needle_counts[needle] += 1

    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                errors.append(f"line {line_no}: blank line")
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_no}: row is not an object")
                continue
            row_count += 1
            validate_row(row, errors, warnings, prompt_hashes)
            variant_counts[str(row.get("variant"))] += 1
            wc = int(row.get("context_words", 0))
            word_counts.append(wc)
            words_by_bucket[str(row.get("length_bucket"))].append(wc)

            rid = str(row.get("id", ""))
            if rid in v1_ids:
                errors.append(f"{rid}: ID collides with frozen v1")

            if pack_rows and row.get("scenario_id") != pack_rows[0].get("scenario_id"):
                process_pack(pack_rows)
                pack_rows = []
            pack_rows.append(row)
        if pack_rows:
            process_pack(pack_rows)

    if row_count != expected_packs * 4:
        errors.append(f"Expected {expected_packs*4} rows, found {row_count}")
    if pack_count != expected_packs:
        errors.append(f"Expected {expected_packs} packs, found {pack_count}")
    if len(prompt_hashes) != row_count:
        errors.append(f"Unique prompt count {len(prompt_hashes)} != row count {row_count}")

    expected_variants = {v: expected_packs for _, v in VARIANT_ORDER}
    if dict(variant_counts) != expected_variants:
        errors.append(f"Variant counts mismatch: {dict(variant_counts)}")

    if expected_packs == 2000:
        if split_pack_counts != Counter({"train": 1800, "dev": 200}):
            errors.append(f"Pack split counts mismatch: {dict(split_pack_counts)}")
        if set(task_counts.values()) != {200}:
            errors.append(f"Task counts are not all 200: {dict(task_counts)}")
        if max(domain_counts.values(), default=0) - min(domain_counts.values(), default=0) > 1:
            errors.append(f"Domain counts differ by more than one: {dict(domain_counts)}")
        if set(combo_counts.values()) - {16, 17}:
            errors.append(f"Domain/task combinations are not 16 or 17 each")
        if bucket_counts != Counter({"compact_long": 800, "standard_long": 800, "extended_long": 400}):
            errors.append(f"Length-bucket counts mismatch: {dict(bucket_counts)}")
        if needle_counts != Counter({"early": 500, "middle": 500, "late": 500, "distributed": 500}):
            errors.append(f"Needle counts mismatch: {dict(needle_counts)}")

    report = {
        "valid": not errors,
        "errors_count": len(errors),
        "warnings_count": len(warnings),
        "errors": errors[:500],
        "warnings": warnings[:500],
        "input": str(input_path),
        "file_bytes": input_path.stat().st_size,
        "sha256": sha256_file(input_path),
        "packs": pack_count,
        "rows": row_count,
        "unique_prompts": len(prompt_hashes),
        "unique_titles": len(title_owner),
        "unique_people": len(person_owner),
        "unique_sites": len(site_owner),
        "unique_serials": len(serial_owner),
        "unique_base_contexts": len(base_context_hashes),
        "variant_counts": dict(variant_counts),
        "split_pack_counts": dict(split_pack_counts),
        "split_row_counts": {k: v * 4 for k, v in split_pack_counts.items()},
        "domain_counts": dict(sorted(domain_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "domain_task_min": min(combo_counts.values()) if combo_counts else 0,
        "domain_task_max": max(combo_counts.values()) if combo_counts else 0,
        "length_bucket_pack_counts": dict(bucket_counts),
        "needle_profile_pack_counts": dict(needle_counts),
        "context_words": {
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "mean": round(statistics.mean(word_counts), 3) if word_counts else 0,
            "median": round(statistics.median(word_counts), 3) if word_counts else 0,
        },
        "context_words_by_bucket": {
            k: {
                "min": min(v), "max": max(v), "mean": round(statistics.mean(v), 3), "rows": len(v)
            } for k, v in sorted(words_by_bucket.items()) if v
        },
        "frozen_v1_checked": bool(v1_path and v1_path.exists()),
        "frozen_v1_20_word_shingle_overlaps": v1_shingle_overlaps,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--expected-packs", type=int, default=2000)
    ap.add_argument("--v1", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=Path("reports/validation_report.json"))
    args = ap.parse_args()
    report = validate(args.input, args.expected_packs, args.v1)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"errors", "warnings"}}, indent=2, ensure_ascii=False))
    if report["errors_count"]:
        print("\nFirst validation errors:")
        for err in report["errors"][:50]:
            print("-", err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
