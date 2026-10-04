"""Build only the frozen, open DEV long-context sentinel with a pinned tokenizer.

No model framework, weights, GPU, downloads, native execution, or sealed payloads
are used. A failed construction consumes its output directory and never replaces
a parent. The complete visible prompt is independently solved after fitting.
"""
from __future__ import annotations

# This import sets offline/framework guards before any tokenizer import.
from experiments.post_training_mimo_centralized_v1.scripts import tokenize_bridge as bridge_tokens
from experiments.post_training_mimo_centralized_v1.scripts import long_context_tasks as tasks

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import time

VERSION = "long-context-native-additive-note-dp-v1"
LENGTHS = (2024, 4072)
RESERVE = 24
TARGET_CAP = 16
INTERVALS = {"early": (0.10, 0.20), "middle": (0.45, 0.55), "late": (0.80, 0.90)}
FIT_POLICY = {"algorithm": "native_additive_note_dp_v1", "maximum_prompt_tokens": 4072,
              "maximum_slot_width_candidates": 128, "maximum_placement_offset_candidates": 32}
TEMPLATE_KWARGS = bridge_tokens.TEMPLATE_KWARGS
TOKENIZER_FILES = bridge_tokens.TOKENIZER_FILES
load_tokenizer = bridge_tokens.load_tokenizer
file_identity = bridge_tokens.file_identity
read_json = bridge_tokens.read_json
require = bridge_tokens.require
json_bytes = bridge_tokens.bridge.json_bytes
sha256 = bridge_tokens.bridge.sha256
FORBIDDEN_BODY_IDS = (151643, 151645)
CONSTRUCTION_WALL_SECONDS = 1800
VARIANTS = ((True, False), (True, True), (False, False), (False, True))


def _encode(tokenizer, text):
    return bridge_tokens._ids(tokenizer.encode(text, add_special_tokens=False))


def _hash_text(text):
    return sha256(text.encode("utf-8"))


def _variant_name(answerable, adversarial):
    return ("answerable" if answerable else "unanswerable") + ("_adversarial" if adversarial else "_neutral")


def _special_ids(tokenizer):
    return set(getattr(tokenizer, "all_special_ids", ())).union(FORBIDDEN_BODY_IDS)


class NoteFitter:
    """Bounded integer DP over complete, predeclared irrelevant prose lines.

    Each budget is measured in actual native tokens. Concatenation is checked,
    never assumed. This is token fitting, not truncation or masked padding.
    """
    def __init__(self, tokenizer, limit=FIT_POLICY["maximum_prompt_tokens"]):
        require(type(limit) is int and 0 < limit <= 4072, "invalid fit bound")
        self.tokenizer, self.limit = tokenizer, limit
        self.lines = tuple(line + "\n" for line in tasks.FILLER_TEXTS)
        self.ids = tuple(_encode(tokenizer, line) for line in self.lines)
        require(all(ids and not set(ids).intersection(_special_ids(tokenizer)) for ids in self.ids),
                "invalid irrelevant prose tokens")
        self.previous = [None] * (limit + 1)
        self.previous[0] = (-1, -1)
        for budget in range(1, limit + 1):
            for index, ids in enumerate(self.ids):
                if budget >= len(ids) and self.previous[budget - len(ids)] is not None:
                    self.previous[budget] = (budget - len(ids), index)
                    break
        self.cache = {0: ""}

    def possible(self, budget):
        return type(budget) is int and 0 <= budget <= self.limit and self.previous[budget] is not None

    def fill(self, budget):
        require(self.possible(budget), "bounded NOTE fitter cannot fill token budget")
        if budget not in self.cache:
            pieces, cursor = [], budget
            while cursor:
                cursor, index = self.previous[cursor]
                pieces.append(self.lines[index])
            text = "".join(pieces)
            require(len(_encode(self.tokenizer, text)) == budget,
                    "NOTE line token boundaries are not additive")
            self.cache[budget] = text
        return self.cache[budget]

    def common_slot(self, alternatives):
        raw = ["".join(line + "\n" for line in lines) for lines in alternatives]
        sizes = [len(_encode(self.tokenizer, text)) for text in raw]
        start = max(sizes)
        for width in range(start, start + FIT_POLICY["maximum_slot_width_candidates"]):
            if all(self.possible(width - size) for size in sizes):
                texts = [text + self.fill(width - size) for text, size in zip(raw, sizes)]
                require(all(len(_encode(self.tokenizer, text)) == width for text in texts),
                        "semantic slot token boundaries are not additive")
                return width, texts
        raise ValueError("bounded common-slot fit failed; parent cannot be replaced")


def _wrapper(tokenizer, query):
    """Derive native wrapper text without ever including private metadata."""
    before = tasks.SYSTEM_PROMPT + "\nCONTEXT\n"
    after = "QUERY\n" + query
    marker = "LONG_CONTEXT_BODY_BOUNDARY_7F4A9C"
    messages = [{"role": "user", "content": before + marker + after}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS)
    require(type(rendered) is str and rendered.count(marker) == 1, "template changed body marker")
    prefix, suffix = rendered.split(marker)
    require(rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "non-thinking assistant generation prefix mismatch")
    return before, after, prefix, suffix


def _native_span(offsets, start, end):
    require(type(start) is int and start < end, "empty text span")
    indices = [i for i, (left, right) in enumerate(offsets) if right > start and left < end]
    require(bool(indices), "visible substring has no native-token span")
    first, last = indices[0], indices[-1] + 1
    require(indices == list(range(first, last)), "noncontiguous native-token span")
    require(offsets[first][0] <= start and offsets[last - 1][1] >= end, "incomplete native-token span")
    return first, last


def _offset_encoding(tokenizer, text, expected_ids):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    require(encoded["input_ids"] == expected_ids, "offset tokenizer IDs differ from complete template")
    offsets = encoded["offset_mapping"]
    require(len(offsets) == len(expected_ids), "native offset count drift")
    require(all(isinstance(pair, (tuple, list)) and len(pair) == 2
                and all(type(x) is int for x in pair) and 0 <= pair[0] <= pair[1] <= len(text)
                for pair in offsets), "invalid native offset mapping")
    return offsets


def _layout(fitter, body_tokens, placement, slot_widths, decoy_width):
    regions = ("early", "middle", "late") if placement == "distributed" else (placement,) * 3
    cursor, starts, gaps = decoy_width, [], []
    for index, (region, width) in enumerate(zip(regions, slot_widths)):
        lower, upper = INTERVALS[region]
        minimum = max(cursor, int(lower * body_tokens + 0.999999))
        # Leave room for all remaining support slots in the same interval.
        remaining = sum(slot_widths[j] for j in range(index, 3) if regions[j] == region)
        ceiling = int(upper * body_tokens) - remaining
        options = range(minimum, min(ceiling + 1, minimum + FIT_POLICY["maximum_placement_offset_candidates"]))
        start = next((candidate for candidate in options if fitter.possible(candidate - cursor)), None)
        require(start is not None, "support slots cannot fit their frozen body interval")
        gaps.append(fitter.fill(start - cursor))
        starts.append(start)
        cursor = start + width
    require(fitter.possible(body_tokens - cursor), "bounded tail fit failed")
    return starts, gaps, fitter.fill(body_tokens - cursor), regions


def _span_receipt(text, offsets, span_text, char_start, body_start, body_token_start, ids):
    char_end = char_start + len(span_text)
    require(text[char_start:char_end] == span_text, "support substring integrity failed")
    first, last = _native_span(offsets, char_start, char_end)
    return {"text": span_text, "text_sha256": _hash_text(span_text),
            "whole_prompt_char_start": char_start, "whole_prompt_char_end": char_end,
            "body_char_start": char_start - body_start, "body_char_end": char_end - body_start,
            "whole_prompt_token_start": first, "whole_prompt_token_end": last,
            "body_token_start": first - body_token_start, "body_token_end": last - body_token_start,
            "token_ids": ids[first:last], "token_ids_sha256": sha256(json_bytes(ids[first:last])),
            "substring_integrity": True}


def render_parent(parent, tokenizer, fitter=None, deadline=None):
    require(parent.role == "DEV", "only DEV parents may be rendered")
    require(tokenizer.eos_token_id == 151645 and tokenizer.eos_token == "<|im_end|>", "Titan EOS drift")
    fitter = fitter or NoteFitter(tokenizer)
    variants = [tasks.render_variant(parent, *flags) for flags in VARIANTS]
    semantic_checks = [tasks.verify_variant(parent, variant) for variant in variants]
    query = variants[0]["query"]
    require(all(v["query"] == query and len(v["support_slots"]) == 3 for v in variants), "query or support slot drift")
    require(all(v["support_facts"] == variants[0]["support_facts"] for v in variants), "support graph drift")
    before, after, prefix, suffix = _wrapper(tokenizer, query)
    prefix_ids, suffix_ids = _encode(tokenizer, prefix), _encode(tokenizer, suffix)
    slots = [fitter.common_slot([[v["support_slots"][i]] for v in variants]) for i in range(3)]
    decoy_width, decoy_texts = fitter.common_slot([v["decoys"] for v in variants])
    rows = []
    for prompt_cap in LENGTHS:
        body_tokens = prompt_cap - len(prefix_ids) - len(suffix_ids)
        require(body_tokens > 0, "template leaves no context body")
        starts, gaps, tail, regions = _layout(fitter, body_tokens, parent.placement,
                                             [slot[0] for slot in slots], decoy_width)
        for variant_index, (variant, (answerable, adversarial)) in enumerate(zip(variants, VARIANTS)):
            require(deadline is None or time.perf_counter() < deadline, "construction wall ceiling exceeded")
            pieces, slot_chars = [decoy_texts[variant_index]], []
            for index in range(3):
                pieces.append(gaps[index])
                slot_chars.append(sum(len(piece) for piece in pieces))
                pieces.append(slots[index][1][variant_index])
            pieces.append(tail)
            body = "".join(pieces)
            body_ids = _encode(tokenizer, body)
            require(len(body_ids) == body_tokens, "body segment token boundaries are not additive")
            require(not set(body_ids).intersection(_special_ids(tokenizer)), "special/PAD/EOS filler is forbidden")
            prompt = before + body + after
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS)
            ids = bridge_tokens._ids(tokenizer.apply_chat_template(messages, tokenize=True, **TEMPLATE_KWARGS))
            require(rendered == prefix + body + suffix, "complete template rendering drift")
            require(ids == _encode(tokenizer, rendered) == prefix_ids + body_ids + suffix_ids,
                    "complete native template token boundaries are not additive")
            require(len(ids) == prompt_cap, "exact complete prompt length failed; truncation forbidden")
            offsets = _offset_encoding(tokenizer, rendered, ids)
            body_char_start, body_char_end = len(prefix), len(prefix) + len(body)
            body_token_start, body_token_end = _native_span(offsets, body_char_start, body_char_end)
            require((body_token_start, body_token_end) == (len(prefix_ids), len(prefix_ids) + body_tokens),
                    "body-relative native interval is not isolated")
            oracle = tasks.solve_visible(query, body)
            require(oracle == variant["target"], "final visible-text oracle disagrees with target")
            target_ids = _encode(tokenizer, variant["target"])
            require(0 < len(target_ids) <= TARGET_CAP and not set(target_ids).intersection(_special_ids(tokenizer)),
                    "target violates canonical-symbol token cap")
            require(tokenizer.decode(target_ids, skip_special_tokens=False) == variant["target"], "target token roundtrip drift")
            spans, slot_receipts = [], []
            for index in range(3):
                line = variant["support_slots"][index]
                char_start = body_char_start + slot_chars[index]
                span = _span_receipt(rendered, offsets, line, char_start, body_char_start, body_token_start, ids)
                lower, upper = INTERVALS[regions[index]]
                require(lower * body_tokens <= span["body_token_start"] < span["body_token_end"] <= upper * body_tokens,
                        "support span escapes body-relative interval")
                require(span["body_token_start"] == starts[index], "support slot moved during template tokenization")
                present = answerable or index != parent.missing_index
                span.update({"dependency_index": index, "required_fact_present": present, "region": regions[index],
                             "interval": [lower, upper], "body_token_count": body_tokens})
                slot_receipts.append({"dependency_index": index, "body_token_start": starts[index],
                                      "body_token_end": starts[index] + slots[index][0], "present": present,
                                      "replacement_sha256": None if present else _hash_text(line)})
                spans.append(span)
            facts = variant["support_facts"]
            for index, fact in enumerate(facts):
                present = answerable or index != parent.missing_index
                require(body.splitlines().count(fact) == int(present), "missing/duplicate required fact")
                if present:
                    ablated = body.replace(fact + "\n", tasks.FILLER_TEXTS[0] + "\n", 1)
                    require(tasks.solve_visible(query, ablated) == "UNKNOWN", "required fact is redundant in final body")
            source_target = variants[0]["target"]
            require(source_target not in query + tasks.SYSTEM_PROMPT + "\n".join(variant["decoys"]) + "\n".join(tasks.FILLER_TEXTS),
                    "query/filler/decoys leak source target")
            if not answerable:
                require(source_target not in variant["support_slots"][parent.missing_index], "missing-slot replacement leaks source target")
            variant_name = _variant_name(answerable, adversarial)
            row_id = f"{parent.parent_id}-{prompt_cap}-{variant_name}"
            row = {"schema_version": 1, "row_id": row_id, "parent_id": parent.parent_id, "role": "DEV",
                   "generator_version": VERSION, "semantic_generator_version": tasks.VERSION,
                   "verifier_version": tasks.VERIFIER_VERSION, "support_unit": "record_block",
                   "split_sha256": parent.split_sha256,
                   "family": parent.family, "evidence_position": parent.placement, "variant_id": variant_name,
                   "distractor_modes": variant["distractor_modes"],
                   "answerable": answerable, "adversarial": adversarial, "missing_index": parent.missing_index,
                   "canonical_parent_sha256": _hash_text(parent.canonical_json),
                   "structural_cluster_id": getattr(parent, "structural_cluster_id", parent.family),
                   "query": query, "context_body": body, "prompt": prompt, "rendered_prompt": rendered,
                   "input_ids": ids, "prompt_tokens": len(ids), "body_tokens": body_tokens,
                   "total_sequence_envelope": prompt_cap + RESERVE, "generation_reserve_tokens": RESERVE,
                   "target": variant["target"], "target_token_ids": target_ids, "target_tokens": len(target_ids),
                   "prompt_sha256": _hash_text(prompt), "rendered_prompt_sha256": _hash_text(rendered),
                   "context_body_sha256": _hash_text(body), "target_sha256": _hash_text(variant["target"]),
                   "input_token_ids_sha256": sha256(json_bytes(ids)),
                   "target_token_ids_sha256": sha256(json_bytes(target_ids)),
                   "template_kwargs": dict(TEMPLATE_KWARGS), "support_spans": spans, "controlled_slots": slot_receipts,
                   "body_whole_prompt_token_start": body_token_start, "body_whole_prompt_token_end": body_token_end,
                   "body_whole_prompt_char_start": body_char_start, "body_whole_prompt_char_end": body_char_end,
                   "native_offset_mapping": offsets, "visible_text_oracle": oracle,
                   "semantic_verification": semantic_checks[variant_index], "individual_ablations_unknown": True,
                   "policy_input_only": ["prompt", "input_ids"], "training_eligibility": False,
                   "source_correct_membership": None, "native_qualification_inferred": False,
                   "e1_launch_ready": False}
            row["row_sha256"] = sha256(json_bytes(row))
            validate_row(row, tokenizer)
            rows.append(row)
            require(deadline is None or time.perf_counter() < deadline, "construction wall ceiling exceeded")
    validate_parent_rows(rows)
    return rows


def validate_row(row, tokenizer):
    """Recheck a stored row from its actual policy input, including native spans."""
    require(row["row_sha256"] == sha256(json_bytes({k: v for k, v in row.items() if k != "row_sha256"})), "row hash drift")
    require(row["role"] == "DEV" and row["prompt_tokens"] in LENGTHS, "closed role or unsupported length")
    require(row["template_kwargs"] == TEMPLATE_KWARGS, "template kwargs drift")
    require(type(row["answerable"]) is bool and type(row["adversarial"]) is bool
            and type(row["missing_index"]) is int and 0 <= row["missing_index"] <= 2, "factor fields drift")
    require(row["variant_id"] == _variant_name(row["answerable"], row["adversarial"])
            and row["row_id"] == f"{row['parent_id']}-{row['prompt_tokens']}-{row['variant_id']}", "row/factor identity drift")
    require(row["generation_reserve_tokens"] == RESERVE and row["total_sequence_envelope"] == row["prompt_tokens"] + RESERVE,
            "generation reserve or envelope drift")
    require(row["evidence_position"] in tasks.POSITIONS and row["family"] in tasks.FAMILIES, "family/position drift")
    require(len(row["support_spans"]) == len(row["controlled_slots"]) == 3, "exactly three support slots required")
    expected_regions = ("early", "middle", "late") if row["evidence_position"] == "distributed" else (row["evidence_position"],) * 3
    require([s["dependency_index"] for s in row["support_spans"]] == [0, 1, 2]
            and [s["dependency_index"] for s in row["controlled_slots"]] == [0, 1, 2], "dependency indices drift")
    require(tuple(s["region"] for s in row["support_spans"]) == expected_regions, "placement interval drift")
    prompt = row["prompt"]
    before = tasks.SYSTEM_PROMPT + "\nCONTEXT\n"
    require(prompt.startswith(before), "visible instructions changed")
    content = prompt[len(before):]
    require(content.count("QUERY\n") == 1, "visible query boundary is ambiguous")
    body, query = content.split("QUERY\n")
    require(body == row["context_body"] and query == row["query"], "visible body/query metadata drift")
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, **TEMPLATE_KWARGS)
    ids = _encode(tokenizer, rendered)
    require(rendered == row["rendered_prompt"] and ids == row["input_ids"] and len(ids) == row["prompt_tokens"],
            "rendered policy input drift")
    for name, text in (("prompt", prompt), ("rendered_prompt", rendered), ("context_body", body), ("target", row["target"])):
        require(row[name + "_sha256"] == _hash_text(text), name + " hash drift")
    require(row["input_token_ids_sha256"] == sha256(json_bytes(ids)), "prompt token hash drift")
    target_ids = _encode(tokenizer, row["target"])
    require(target_ids == row["target_token_ids"] and 0 < len(target_ids) == row["target_tokens"] <= TARGET_CAP,
            "target native tokens or cap drift")
    require(row["target_token_ids_sha256"] == sha256(json_bytes(target_ids))
            and not set(target_ids).intersection(_special_ids(tokenizer)), "target token hash or special IDs drift")
    offsets = _offset_encoding(tokenizer, rendered, ids)
    require(json_bytes(offsets) == json_bytes(row["native_offset_mapping"]), "native offsets drift")
    body_start = rendered.index(before) + len(before)
    body_end = body_start + len(body)
    start, end = _native_span(offsets, body_start, body_end)
    require((body_start, body_end, start, end) == (row["body_whole_prompt_char_start"], row["body_whole_prompt_char_end"],
            row["body_whole_prompt_token_start"], row["body_whole_prompt_token_end"]), "body native span drift")
    require(end - start == row["body_tokens"] and not set(ids[start:end]).intersection(_special_ids(tokenizer)),
            "invalid context body token interval")
    require(tasks.solve_visible(query, body) == row["target"] == row["visible_text_oracle"], "visible policy-input oracle disagrees")
    for span, slot in zip(row["support_spans"], row["controlled_slots"]):
        line = span["text"]
        require(body.splitlines().count(line) >= 1, "recorded support line absent")
        actual = _span_receipt(rendered, offsets, line, body_start + span["body_char_start"], body_start, start, ids)
        require(all(span[key] == value for key, value in actual.items()), "support native span or substring drift")
        lower, upper = INTERVALS[span["region"]]
        require(span["interval"] == [lower, upper] and span["body_token_count"] == end - start, "body interval receipt drift")
        require(lower * (end - start) <= actual["body_token_start"] < actual["body_token_end"] <= upper * (end - start),
                "support native span outside frozen interval")
        require(slot["body_token_start"] == actual["body_token_start"] and actual["body_token_end"] <= slot["body_token_end"] <= upper * (end - start),
                "controlled slot footprint drift")
        expected_present = row["answerable"] or span["dependency_index"] != row["missing_index"]
        require(span["required_fact_present"] is expected_present and slot["present"] is expected_present, "missing support receipt drift")
        if span["required_fact_present"]:
            ablated = body[:span["body_char_start"]] + tasks.FILLER_TEXTS[0] + body[span["body_char_end"]:]
            require(tasks.solve_visible(query, ablated) == "UNKNOWN", "support ablation does not remove required evidence")
    return True


def validate_parent_rows(rows):
    require(len(rows) == 8 and len({r["row_id"] for r in rows}) == 8, "parent must have all eight distinct derivatives")
    require(len({r["parent_id"] for r in rows}) == len({r["canonical_parent_sha256"] for r in rows}) == 1,
            "semantic parent changed across derivatives")
    require(len({r["query"] for r in rows}) == 1, "query changed across derivatives")
    for field in ("family", "evidence_position", "missing_index", "split_sha256", "structural_cluster_id"):
        require(len({r[field] for r in rows}) == 1, field + " changed across derivatives")
    require(len({r["target"] for r in rows if r["answerable"]}) == 1
            and {r["target"] for r in rows if not r["answerable"]} == {"UNKNOWN"}, "targets changed across lengths or factorial")
    for index in range(3):
        present_texts = {r["support_spans"][index]["text"] for r in rows
                         if r["support_spans"][index]["required_fact_present"]}
        require(len(present_texts) == 1, "support graph changed across lengths")
    for length in LENGTHS:
        group = [r for r in rows if r["prompt_tokens"] == length]
        require({(r["answerable"], r["adversarial"]) for r in group} == set(VARIANTS), "factorial is incomplete")
        require(len({r["body_tokens"] for r in group}) == 1, "paired body lengths differ")
        coordinates = lambda row: [(s["body_token_start"], s["body_token_end"]) for s in row["controlled_slots"]]
        require(all(coordinates(row) == coordinates(group[0]) for row in group), "paired controlled positions differ")
        for index in range(3):
            present = [r["support_spans"][index] for r in group if r["support_spans"][index]["required_fact_present"]]
            require(len({(s["text"], s["body_token_start"], s["body_token_end"]) for s in present}) == 1,
                    "other facts or their native positions changed")


def validate_registration(registration):
    require(registration.get("status") == "FROZEN_FAST_LANE_LONG_CONTEXT_ONLY", "long-context registration is not frozen")
    require(registration.get("selection_seed") == 20260928, "selection seed drift")
    require(registration.get("counts") == {"DEV": 32, "CHECK": 64, "FINAL": 0}, "frozen role counts drift")
    require(registration.get("prompt_lengths") == list(LENGTHS), "prompt length drift")
    require(registration.get("generation_reserve_tokens") == RESERVE and registration.get("max_target_tokens") == TARGET_CAP,
            "generation/target cap drift")
    require(registration.get("families") == list(tasks.FAMILIES) and registration.get("positions") == list(tasks.POSITIONS),
            "family/position contract drift")
    require(registration.get("fit_policy") == FIT_POLICY, "bounded fit policy drift")
    require(registration.get("wall_seconds") == CONSTRUCTION_WALL_SECONDS, "construction wall ceiling drift")
    require(registration.get("template_kwargs") == TEMPLATE_KWARGS, "template kwargs drift")
    auth = registration.get("authorization", {})
    require(auth.get("dataset_artifact_generation") is True and all(auth.get(key) is False for key in
            ("native_execution", "training", "check_rendering", "final_rendering")), "artifact-only authorization drift")
    require(registration.get("model_outcome_based_selection") is False, "model outcome selection forbidden")
    require(all(isinstance(registration.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", registration[key]) for key in
                ("primary_student_weight_sha256", "tokenizer_sha256", "chat_template_sha256")), "frozen primary identity hashes required")


def _committed_source(registration_path, source_revision):
    require(re.fullmatch(r"[0-9a-f]{40}", source_revision or "") is not None, "full committed source revision required")
    root = Path(subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "--show-toplevel"], text=True).strip())
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    require(head == source_revision, "source revision is not current committed HEAD")
    files = [Path(__file__), Path(tasks.__file__), Path(bridge_tokens.__file__), Path(bridge_tokens.bridge.__file__), registration_path]
    paths = [str(path.resolve().relative_to(root.resolve())) for path in files]
    for path in paths:
        subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", "--", path], check=True, capture_output=True)
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", *paths], text=True)
    require(not dirty.strip(), "generator and registration must be committed before production")
    return {str(path.resolve().relative_to(root.resolve())).replace("\\", "/"): file_identity(path) for path in files}


def _write(path, value):
    with path.open("xb") as handle:
        handle.write(json_bytes(value) + b"\n")


def build_dataset(registration_path, tokenizer_receipt_path, output, source_revision):
    started = time.perf_counter()
    require(not output.exists(), "output is write-once and must not exist")
    require("torch" not in sys.modules, "tokenizer process must not already contain torch")
    registration = read_json(registration_path)
    validate_registration(registration)
    source_files = _committed_source(registration_path, source_revision)
    registration_identity, receipt_identity = file_identity(registration_path), file_identity(tokenizer_receipt_path)
    tokenizer_receipt = read_json(tokenizer_receipt_path)
    model_dir = Path(tokenizer_receipt["checkpoint_path"])
    bound = tokenizer_receipt.get("tokenizer_files") or {name: tokenizer_receipt["files"][name] for name in TOKENIZER_FILES}
    tokenizer_files = {name: file_identity(model_dir / name) for name in TOKENIZER_FILES}
    require(tokenizer_files == bound, "bound tokenizer file drift")
    template = read_json(model_dir / "tokenizer_config.json")["chat_template"]
    require(_hash_text(template) == tokenizer_receipt["chat_template_sha256"], "bound template drift")
    require(registration.get("primary_student_weight_sha256") == tokenizer_receipt.get("weight_sha256"), "primary source weight binding drift")
    require(registration.get("tokenizer_sha256") == tokenizer_files["tokenizer.json"]["sha256"], "registered tokenizer identity drift")
    require(registration.get("chat_template_sha256") == _hash_text(template), "registered template identity drift")
    output.mkdir(parents=True, exist_ok=False)
    completed_parents, completed_rows = 0, 0
    try:
        tokenizer = load_tokenizer(model_dir)
        require(time.perf_counter() - started < CONSTRUCTION_WALL_SECONDS, "construction wall ceiling exceeded")
        require(tokenizer.chat_template == template, "runtime tokenizer template drift")
        allocation = tasks.allocate_bank(seed=registration["selection_seed"])
        parents, reservations = allocation["dev"], allocation["check_reservations"]
        require(len(parents) == 32 and len(reservations) == 64, "semantic allocation count drift")
        require(Counter((p.family, p.placement) for p in parents) == Counter({(f, p): 4 for f in tasks.FAMILIES for p in tasks.POSITIONS}),
                "DEV family/placement allocation drift")
        require(len({p.parent_id for p in parents}) == 32, "duplicate semantic parent")
        reservation_keys = {"parent_id", "role", "family", "placement", "split_sha256", "structural_cluster_id"}
        require(all(set(r) == reservation_keys and r["role"] == "CHECK"
                    and re.fullmatch(r"lc-[0-9a-f]{64}", r["parent_id"])
                    and re.fullmatch(r"[0-9a-f]{64}", r["split_sha256"]) for r in reservations),
                "CHECK reservations must contain opaque commitments only")
        require(len({r["parent_id"] for r in reservations}) == 64
                and not {p.parent_id for p in parents}.intersection(r["parent_id"] for r in reservations), "role overlap or duplicate reservation")
        require(Counter((r["family"], r["placement"]) for r in reservations)
                == Counter({(f, p): 8 for f in tasks.FAMILIES for p in tasks.POSITIONS}), "CHECK family/placement reservation drift")
        fitter, rows = NoteFitter(tokenizer), []
        for parent in parents:
            require(time.perf_counter() - started < CONSTRUCTION_WALL_SECONDS, "construction wall ceiling exceeded")
            group = render_parent(parent, tokenizer, fitter, deadline=started + CONSTRUCTION_WALL_SECONDS)
            rows.extend(group)
            completed_parents += 1
            completed_rows += len(group)
        require(time.perf_counter() - started < CONSTRUCTION_WALL_SECONDS, "construction wall ceiling exceeded")
        require(len(rows) == len({r["row_id"] for r in rows}) == len({r["input_token_ids_sha256"] for r in rows}) == 256,
                "DEV row or token-prompt uniqueness drift")
        require("torch" not in sys.modules, "unexpected torch import in tokenizer-only process")
        require({name: file_identity(model_dir / name) for name in TOKENIZER_FILES} == tokenizer_files, "tokenizer files changed during build")
        require(file_identity(registration_path) == registration_identity and file_identity(tokenizer_receipt_path) == receipt_identity,
                "registration or tokenizer receipt changed during build")
        require(_committed_source(registration_path, source_revision) == source_files, "source changed during build")
        require(time.perf_counter() - started < CONSTRUCTION_WALL_SECONDS, "construction wall ceiling exceeded")
        overall_dirty_status = subprocess.check_output(["git", "-C", str(Path(__file__).parent), "status", "--porcelain"], text=True)
        _write(output / "REGISTRATION.json", registration)
        with (output / "DEV.jsonl").open("xb") as handle:
            for row in rows:
                handle.write(json_bytes(row) + b"\n")
        with (output / "SEALED_RESERVATIONS.jsonl").open("xb") as handle:
            for reservation in reservations:
                handle.write(json_bytes(reservation) + b"\n")
        files = {name: file_identity(output / name) for name in ("REGISTRATION.json", "DEV.jsonl", "SEALED_RESERVATIONS.jsonl")}
        receipt = {"schema_version": 1, "status": "MEASURED_LONG_CONTEXT_DEV_TOKENIZER_MECHANICS_ONLY",
                   "created_utc": datetime.now(timezone.utc).isoformat(), "generator_version": VERSION,
                   "source_revision": source_revision, "source_files": source_files, "registration_identity": registration_identity,
                   "overall_dirty_status": overall_dirty_status, "source_scope_committed": True,
                   "tokenizer_receipt_identity": receipt_identity, "tokenizer_files": tokenizer_files,
                   "tokenizer_class": type(tokenizer).__name__, "chat_template_sha256": _hash_text(template),
                   "template_kwargs": dict(TEMPLATE_KWARGS), "fit_policy": dict(FIT_POLICY),
                   "wall_seconds_ceiling": CONSTRUCTION_WALL_SECONDS,
                   "allocation": allocation["allocation"], "parent_counts": {"DEV": 32, "CHECK": 64, "FINAL": 0},
                   "open_row_counts": {"DEV": 256, "CHECK": 0, "FINAL": 0}, "check_reserved_rows": 512,
                   "factor_counts": {name: dict(Counter(str(r[name]) for r in rows)) for name in
                                     ("family", "evidence_position", "prompt_tokens", "answerable", "adversarial", "variant_id")},
                   "group_counts": {f + "/" + p: {"parents": 4, "rows": 32} for f in tasks.FAMILIES for p in tasks.POSITIONS},
                   "files": files, "sealed_access": {"payloads_opened": False, "check_rendered": False, "final_rendered": False,
                                                       "hash_only_check_reservations": 64},
                   "provenance": "Original deterministic authored synthetic facts and NOTE prose; no external or teacher text",
                   "policy_input_only": ["prompt", "input_ids"], "truncation": False, "pad_eos_filler": False,
                   "cost": {"wall_seconds": time.perf_counter() - started, "model_loads": 0, "model_calls": 0,
                            "teacher_calls": 0, "gpu_seconds": 0, "actual_training_tokens": 0},
                   "torch_imported": "torch" in sys.modules, "python_executable": sys.executable,
                   "packages": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
                   "historical_exclusion_qualified": False, "source_correct_enrollment_qualified": False,
                   "long_context_preservation_qualified": False, "broad_preservation_coverage_qualified": False,
                   "native_qualification_inferred": False, "training_eligibility": False, "e1_launch_ready": False}
        require(time.perf_counter() - started < CONSTRUCTION_WALL_SECONDS, "construction wall ceiling exceeded")
        _write(output / "DATA_RECEIPT.json", receipt)
        return receipt
    except BaseException as error:
        _write(output / "FAILED_CONSTRUCTION.json", {"schema_version": 1, "status": "FAILED_CLOSED_NO_BANK_ADMISSION",
               "exception_type": type(error).__name__, "error": str(error), "completed_parents": completed_parents,
               "completed_rows": completed_rows, "source_revision": source_revision, "source_files": source_files,
               "registration_identity": registration_identity, "tokenizer_receipt_identity": receipt_identity,
               "wall_seconds": time.perf_counter() - started, "automatic_retry": False, "parent_replacement": False,
               "model_calls": 0, "training_eligibility": False, "e1_launch_ready": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_dataset(args.registration, args.tokenizer_receipt, args.output, args.source_revision)
    print(json.dumps({"receipt": str((args.output / "DATA_RECEIPT.json").resolve()),
                      "status": result["status"], "open_row_counts": result["open_row_counts"],
                      "torch_imported": result["torch_imported"], "e1_launch_ready": False}, indent=2))


if __name__ == "__main__":
    main()
