"""New SYNTH public CPU task guard candidates, adapted with pinned attribution.

Central retention_data.py/dev_inputs.py/dev_scorer.py provide the task mechanisms,
strict response semantics and separation design. No central module or population
is imported. CHECK/FINAL payloads, models and frameworks are never read/loaded.
These exact-world CPU candidates are not M0 enrollment or retention qualification.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] if HERE.name == "guards_cpu_work" and HERE.parent.name == "artifacts" else HERE
ROOT = PACKAGE.parents[1]
POLICY_PATH = HERE / "TASK_RETENTION_GUARD_POLICY.json"
VERSION = "synth-task-retention-guards-1"
WORDS = ("amber", "birch", "cedar", "dawn", "ember", "fern", "glen", "harbor", "iris", "jade",
         "kelp", "linen", "moss", "north", "opal", "pine", "quartz", "river", "silver", "tulip",
         "umber", "violet", "willow", "zenith")
FORMATS = ("To: {recipient}\nMessage: {message}", "TO[{recipient}] MSG[{message}]",
           "Recipient={recipient}; Message={message}", "{message} (for {recipient})")
MECHANISMS = ("retrieval", "instruction_following", "answerability_abstention")
PRIORITIES = ("base", "task", "final")
SYSTEM = 'Use only the displayed facts and rules. Return one JSON object with exactly key "answer". No explanations or extra keys.'
RECORD_RULE = "Match entity and field exactly. Return both exact value strings in query order."
MISSING_RULE = "Match entity and field exactly. Return both exact values in query order, or null for the whole pair if either key lacks one unique value."
INSTRUCTION_RULE = "For each slot final overrides task overrides base, regardless of line order. Use recipient and message; ignore note. Preserve literal spaces and case."
FRAMES = {("retrieval", "json_lines"): ("RECORD QUERY", RECORD_RULE),
          ("retrieval", "typed_ledger"): ("RECORD LEDGER", RECORD_RULE),
          ("answerability_abstention", "json_lines"): ("EVIDENCE QUERY", MISSING_RULE),
          ("answerability_abstention", "typed_ledger"): ("EVIDENCE LEDGER", MISSING_RULE),
          ("instruction_following", "json_lines"): ("DIRECTIVE TASK", INSTRUCTION_RULE),
          ("instruction_following", "typed_ledger"): ("DIRECTIVE LEDGER", INSTRUCTION_RULE)}
MAX_JSON_BYTES = 16 * 1024 * 1024
CENTRAL_SOURCE_KEYS = frozenset(("scripts/retention_data.py", "scripts/dev_inputs.py", "scripts/dev_scorer.py",
    "receipts/RETENTION_DATA_RECEIPT.json", "receipts/RETENTION_GENERATION_SOURCE.json",
    "receipts/RETENTION_TOKENIZATION_RECEIPT.json", "receipts/DEV_INPUT_SCORER_BINDING.json",
    "receipts/DEV_BINDING_ARTIFACT_AUDIT.json"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON")


def strict_json(value, limit=4096):
    require(type(value) is str and len(value.encode("utf-8")) <= limit, "bounded JSON text")
    return json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)


def _read(path):
    with Path(path).open("rb") as f:
        raw = f.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, "public CPU packet byte bound")
    return strict_json(raw.decode("utf-8"), MAX_JSON_BYTES)


def no_frameworks():
    require(not any(name.split(".")[0] in {"torch", "transformers", "jax", "tensorflow", "flax"}
                    for name in sys.modules), "CPU guard preparation cannot contain frameworks")


def policy():
    value = _read(POLICY_PATH)
    require(value["seed"] == 2026100313 and value["components_per_mechanism"] == 64
            and value["mechanisms"] == list(MECHANISMS) and value["renderings"] == ["json_lines", "typed_ledger"]
            and value["records_per_complete_world"] == 4 and value["directives_per_instruction_world"] == 7
            and value["context_cap_tokens"] == 512 and value["generation_reserve_tokens"] == 32
            and value["maximum_prompt_tokens"] == 480, "task guard policy drift")
    require(value["variants"] == {"retrieval": ["standard", "reverse_queries"],
            "instruction_following": ["standard", "reverse_directives"],
            "answerability_abstention": ["answerable", "missing_first"]}, "registered guard variants")
    require(type(value.get("central_source_pins")) is dict and set(value["central_source_pins"]) == CENTRAL_SOURCE_KEYS,
            "exact eight-key public central source closure required")
    require(all(value[key] is False for key in ("native_launch_authorized", "training_authorized",
        "protected_bank_access", "M0_enrollment_performed", "source_correct_membership_enrolled",
        "retention_qualified", "language_qualification_established", "long_context_qualification_established",
        "confirmation_eligible")), "closed guard scope")
    return value


def central_attribution():
    frozen = policy()
    require(type(frozen.get("central_source_pins")) is dict and set(frozen["central_source_pins"]) == CENTRAL_SOURCE_KEYS,
            "exact eight-key public central source closure required")
    folder = ROOT / "experiments/post_training_mimo_centralized_v1"
    require(not folder.is_symlink() and not (folder / "scripts").is_symlink() and not (folder / "receipts").is_symlink(),
            "redirected central source directories")
    pins = {}
    for relative, expected in frozen["central_source_pins"].items():
        require(relative in CENTRAL_SOURCE_KEYS and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts and "\\" not in relative
                and relative.startswith(("scripts/", "receipts/")) and "CHECK" not in relative and "FINAL" not in relative,
                "public central source metadata only")
        path = folder / relative
        require(not path.is_symlink() and path.resolve() == folder.resolve() / relative,
                "redirected central source metadata file")
        require(filehash(path) == expected, "central attribution source changed")
        pins[str(path.resolve())] = expected
    return {"scope": "PUBLIC_SOURCE_AND_RECEIPT_METADATA_ONLY_NO_POPULATION_IMPORT",
            "source_notice": frozen["source_attribution"], "file_pins": pins,
            "central_sources_or_populations_modified": False, "CHECK_FINAL_payloads_read": False}


def _public_rs_files():
    paths = [PACKAGE / "data/crossed_source2_v1_20261003" / ("student_pool_" + arm + ".jsonl")
             for arm in ("M1", "M2")]
    folder = PACKAGE / "artifacts/evaluation_dev"
    require(folder.is_dir() and not folder.is_symlink(), "public fresh DEV packet directory")
    for leaf in sorted(folder.iterdir()):
        if leaf.is_dir():
            require(not leaf.is_symlink(), "redirected public DEV packet")
            paths += [leaf / "policy/prompts.json", leaf / "oracle_CPU_DEV_exposed/labels.json"]
    require(all(path.is_file() and not path.is_symlink() for path in paths), "complete public TRAIN/DEV packet inventory")
    return paths


def collect_public_bindings():
    """Read only two SYNTH TRAIN packets and explicitly CPU-exposed fresh DEV."""
    no_frameworks()
    pins, prompts, row_ids, components = {}, set(), set(), set()
    for path in _public_rs_files():
        require(PACKAGE.resolve() in path.resolve().parents and "CHECK" not in str(path)
                and "FINAL" not in str(path), "public SYNTH packet scope")
        pins[str(path.resolve())] = filehash(path)
        values = [_read(path)] if path.suffix == ".json" else [strict_json(line, 65536)
                 for line in path.read_text(encoding="utf-8").splitlines() if line]
        for value in values:
            rows = value.get("rows", [value])
            for row in rows:
                if "messages" in row:
                    prompts.add(digest(row["messages"]))
                if "row_id" in row:
                    row_ids.add(row["row_id"])
                if "component_id" in row:
                    components.add(row["component_id"])
    value = {"scope": "CURRENT_SYNTH_ATOMIC_TRAIN_AND_ALL_EXPOSED_FRESH_DEV_PACKETS",
             "file_pins": pins, "prompt_hashes": sorted(prompts), "row_ids": sorted(row_ids),
             "component_ids": sorted(components), "protected_payloads_read": False}
    value["sha256"] = digest(value)
    return value


def _text(value):
    require(type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}", value), "bounded literal text")
    return value


def validate_spec(spec, *, complete=True):
    require(type(spec) is dict and spec.get("domain") in MECHANISMS, "task guard domain")
    if spec["domain"] == "instruction_following":
        require(set(spec) == {"domain", "directives", "template"} and spec["template"] in FORMATS
                and type(spec["directives"]) is list and len(spec["directives"]) == 7, "instruction source schema")
        keys = set()
        for item in spec["directives"]:
            require(type(item) is dict and set(item) == {"priority", "slot", "text"}
                    and item["priority"] in PRIORITIES and item["slot"] in ("recipient", "message", "note"), "directive schema")
            require((item["priority"], item["slot"]) not in keys, "duplicate priority/slot")
            keys.add((item["priority"], item["slot"]))
            _text(item["text"])
        require(keys == {("base", "recipient"), ("base", "message"), ("base", "note"),
                        ("task", "recipient"), ("task", "message"), ("final", "recipient"), ("final", "message")},
                "registered priority overrides and irrelevant note required")
    else:
        require(set(spec) == {"domain", "records", "queries"} and type(spec["records"]) is list
                and len(spec["records"]) in ((4,) if complete else (3, 4))
                and type(spec["queries"]) is list and len(spec["queries"]) == 2, "record-query source schema")
        keys = set()
        for row in spec["records"]:
            require(type(row) is dict and set(row) == {"entity", "field", "value"}, "record fields")
            for value in row.values(): _text(value)
            key = (row["entity"], row["field"])
            require(key not in keys, "duplicate or contradictory record key")
            keys.add(key)
        queries = []
        for row in spec["queries"]:
            require(type(row) is dict and set(row) == {"entity", "field"}, "query fields")
            for value in row.values(): _text(value)
            queries.append((row["entity"], row["field"]))
        require(len(set(queries)) == 2 and (not complete or set(queries) <= keys), "distinct supported complete-world queries")
        if spec["domain"] == "retrieval": require(set(queries) <= keys, "retrieval prompt missing evidence")
    return spec


def world_key(spec):
    validate_spec(spec)
    if spec["domain"] == "instruction_following":
        return digest({"kind": "priority_text_world", "directives": sorted(spec["directives"], key=canonical)})
    return digest({"kind": "record_query_world", "records": sorted(spec["records"], key=canonical)})


def _generate():
    rng = random.Random(policy()["seed"])
    entity_pool = [a + " " + b for a in WORDS for b in WORDS]
    directive_pool = [a.title() + " " + b for a in WORDS for b in WORDS]
    rng.shuffle(entity_pool)
    rng.shuffle(directive_pool)
    parents, entity_cursor, text_cursor = [], 0, 0
    for mechanism in MECHANISMS:
        for index in range(64):
            if mechanism == "instruction_following":
                keys = [("base", "recipient"), ("base", "message"), ("base", "note"), ("task", "recipient"),
                        ("task", "message"), ("final", "recipient"), ("final", "message")]
                texts = directive_pool[text_cursor:text_cursor + 7]
                text_cursor += 7
                spec = {"domain": mechanism, "directives": [{"priority": p, "slot": s, "text": t}
                    for (p, s), t in zip(keys, texts)], "template": FORMATS[index % 4]}
            else:
                a, b, c = entity_pool[entity_cursor:entity_cursor + 3]
                entity_cursor += 3
                fields = ("badge", "drawer") if mechanism == "retrieval" else ("access key", "storage bay")
                values = rng.sample(directive_pool, 4)
                spec = {"domain": mechanism, "records": [
                    {"entity": a, "field": fields[0], "value": values[0]},
                    {"entity": a, "field": fields[1], "value": values[1]},
                    {"entity": b, "field": fields[index % 2], "value": values[2]},
                    {"entity": c, "field": fields[1 - index % 2], "value": values[3]}],
                    "queries": [{"entity": a, "field": fields[index % 2]},
                                {"entity": b, "field": fields[index % 2]}]}
            validate_spec(spec)
            key = world_key(spec)
            parents.append({"parent_id": "task_guard_" + key, "component_id": "task_component_" + key,
                            "mechanism": mechanism, "source_index": index, "spec": spec})
    require(len(parents) == 192 and len({p["component_id"] for p in parents}) == 192, "192 independent literal worlds")
    return parents


def visible_variant(spec, variant):
    value = copy.deepcopy(spec)
    require(variant in policy()["variants"][spec["domain"]], "registered guard variant")
    if variant == "reverse_queries": value["queries"].reverse()
    if variant == "reverse_directives": value["directives"].reverse()
    if variant == "missing_first":
        q = value["queries"][0]
        value["records"] = [row for row in value["records"] if any(row[k] != q[k] for k in ("entity", "field"))]
    validate_spec(value, complete=False)
    return value


def reference_answer(spec):
    """Primary: ascending-priority overwrite or direct ordered-record lookup."""
    validate_spec(spec, complete=False)
    if spec["domain"] == "instruction_following":
        slots = {}
        for priority in PRIORITIES:
            for row in spec["directives"]:
                if row["priority"] == priority: slots[row["slot"]] = row["text"]
        return spec["template"].format(**slots)
    result = []
    for query in spec["queries"]:
        values = [row["value"] for row in spec["records"] if all(row[k] == query[k] for k in query)]
        if len(values) != 1: return None
        result.append(values[0])
    return result


def render(spec, rendering):
    validate_spec(spec, complete=False)
    require(rendering in ("json_lines", "typed_ledger"), "guard rendering")
    heading, rule = FRAMES[(spec["domain"], rendering)]
    lines = [heading, rule]
    if spec["domain"] == "instruction_following":
        lines += [("D " + canonical(row)) if rendering == "json_lines" else
                  ("DIRECTIVE " + canonical([row["priority"], row["slot"], row["text"]])) for row in spec["directives"]]
        lines.append("FORMAT " + canonical(spec["template"]))
    else:
        lines += [("R " + canonical(row)) if rendering == "json_lines" else
                  ("FACT " + canonical([row["entity"], row["field"], row["value"]])) for row in spec["records"]]
        if rendering == "json_lines": lines.append("Q " + canonical(spec["queries"]))
        else: lines += ["ASK " + canonical([q["entity"], q["field"]]) for q in spec["queries"]]
    lines.append("END " + heading)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse_visible(messages):
    """Independent typed reconstruction; never calls the emitting renderer."""
    require(type(messages) is list and len(messages) == 2 and messages[0] == {"role": "system", "content": SYSTEM}
            and type(messages[1]) is dict and set(messages[1]) == {"role", "content"}
            and messages[1]["role"] == "user" and type(messages[1]["content"]) is str
            and len(messages[1]["content"].encode("utf-8")) <= 8192, "guard policy message boundary")
    lines = messages[1]["content"].splitlines()
    headings = {"RECORD QUERY": ("retrieval", "json_lines"), "RECORD LEDGER": ("retrieval", "typed_ledger"),
                "EVIDENCE QUERY": ("answerability_abstention", "json_lines"),
                "EVIDENCE LEDGER": ("answerability_abstention", "typed_ledger"),
                "DIRECTIVE TASK": ("instruction_following", "json_lines"),
                "DIRECTIVE LEDGER": ("instruction_following", "typed_ledger")}
    require(len(lines) >= 7 and lines[0] in headings and lines[-1] == "END " + lines[0], "guard prompt frame")
    mechanism, rendering = headings[lines[0]]
    expected_rule = (INSTRUCTION_RULE if mechanism == "instruction_following" else
                     MISSING_RULE if mechanism == "answerability_abstention" else RECORD_RULE)
    require(lines[1] == expected_rule, "guard visible rule drift")
    if mechanism == "instruction_following":
        require(lines[-2].startswith("FORMAT "), "instruction format boundary")
        items = []
        for line in lines[2:-2]:
            prefix = "D " if rendering == "json_lines" else "DIRECTIVE "
            require(line.startswith(prefix), "directive line boundary")
            raw = strict_json(line[len(prefix):])
            if rendering == "typed_ledger":
                require(type(raw) is list and len(raw) == 3, "typed directive arity")
                raw = dict(zip(("priority", "slot", "text"), raw))
            items.append(raw)
        spec = {"domain": mechanism, "directives": items, "template": strict_json(lines[-2][7:])}
    else:
        records, queries = [], []
        query_started = False
        for line in lines[2:-1]:
            prefix = "R " if rendering == "json_lines" else "FACT "
            if line.startswith(prefix):
                require(not query_started, "record after query")
                raw = strict_json(line[len(prefix):])
                if rendering == "typed_ledger":
                    require(type(raw) is list and len(raw) == 3, "typed record arity")
                    raw = dict(zip(("entity", "field", "value"), raw))
                records.append(raw)
            else:
                prefix = "Q " if rendering == "json_lines" else "ASK "
                require(line.startswith(prefix), "query line boundary")
                raw = strict_json(line[len(prefix):])
                if rendering == "json_lines":
                    require(not query_started and type(raw) is list, "one query list")
                    queries = raw
                else:
                    require(type(raw) is list and len(raw) == 2, "typed query arity")
                    queries.append(dict(zip(("entity", "field"), raw)))
                query_started = True
        spec = {"domain": mechanism, "records": records, "queries": queries}
    validate_spec(spec, complete=False)
    return spec


def solve_visible(messages):
    """Independent: max-ranked slot resolution and keyed set-valued evidence."""
    spec = parse_visible(messages)
    if spec["domain"] == "instruction_following":
        rank, slots = {"base": 0, "task": 1, "final": 2}, {}
        for slot in ("recipient", "message"):
            choices = [(rank[row["priority"]], row["text"]) for row in spec["directives"] if row["slot"] == slot]
            slots[slot] = max(choices, key=lambda item: item[0])[1]
        return spec["template"].replace("{recipient}", slots["recipient"]).replace("{message}", slots["message"])
    evidence = defaultdict(set)
    for row in spec["records"]: evidence[(row["entity"], row["field"])].add(row["value"])
    keys = [(query["entity"], query["field"]) for query in spec["queries"]]
    if any(len(evidence[key]) != 1 for key in keys): return None
    return [next(iter(evidence[key])) for key in keys]


def score_response(messages, response):
    spec = parse_visible(messages)  # Invalid source raises, rather than becoming model malformedness.
    expected = solve_visible(messages)
    try:
        raw = strict_json(response)
        require(type(raw) is dict and set(raw) == {"answer"}, "answer object fields")
        answer = raw["answer"]
        shape = (type(answer) is str if spec["domain"] == "instruction_following" else
                 answer is None and spec["domain"] == "answerability_abstention" or
                 type(answer) is list and len(answer) == 2 and all(type(value) is str for value in answer))
        if not shape: return {"correct": False, "malformed": True}
    except (ValueError, TypeError, RecursionError, UnicodeError):
        return {"correct": False, "malformed": True}
    return {"correct": answer == expected, "malformed": False}


def _check_bindings(bindings):
    require(type(bindings) is dict and bindings.get("sha256") == digest({k: v for k, v in bindings.items() if k != "sha256"}),
            "guard public input bindings")
    require(type(bindings.get("file_pins")) is dict and all(type(bindings.get(key)) is list
            and bindings[key] == sorted(set(bindings[key])) and all(type(value) is str for value in bindings[key])
            for key in ("row_ids", "prompt_hashes", "component_ids")), "guard exclusion inventory schema")
    if bindings["scope"] != "TEST_ONLY_NOT_REPOSITORY_ADMISSION":
        require(bindings["scope"] == "CURRENT_SYNTH_ATOMIC_TRAIN_AND_ALL_EXPOSED_FRESH_DEV_PACKETS"
                and set(bindings["file_pins"]) == {str(p.resolve()) for p in _public_rs_files()}, "guard public packet inventory changed")
        for path, expected in bindings["file_pins"].items(): require(filehash(path) == expected, "guard public packet byte drift")
        require(canonical(bindings) == canonical(collect_public_bindings()), "guard public exclusions differ from actual packet contents")


def compile_guards(bindings=None):
    no_frameworks()
    bindings = collect_public_bindings() if bindings is None else bindings
    _check_bindings(bindings)
    parents, rows, oracle = _generate(), [], []
    for parent in parents:
        require(parent["component_id"] not in bindings["component_ids"], "guard component overlaps current TRAIN/DEV")
        for variant in policy()["variants"][parent["mechanism"]]:
            spec = visible_variant(parent["spec"], variant)
            expected = reference_answer(spec)
            for rendering in ("json_lines", "typed_ledger"):
                messages = render(spec, rendering)
                require(parse_visible(messages) == spec and solve_visible(messages) == expected, "independent guard visible/reference disagreement")
                row_id = digest({"source_parent": parent["component_id"], "variant": variant, "rendering": rendering,
                                 "policy": digest(policy())})
                prompt_hash = digest(messages)
                require(row_id not in bindings["row_ids"] and prompt_hash not in bindings["prompt_hashes"], "guard row/prompt overlaps current TRAIN/DEV")
                require(parent["parent_id"] not in messages[1]["content"] and parent["component_id"] not in messages[1]["content"]
                        and row_id not in messages[1]["content"], "guard metadata leaked into prompt")
                rows.append({"row_id": row_id, "messages": messages, "prompt_sha256": prompt_hash})
                oracle.append({"row_id": row_id, "parent_id": parent["parent_id"], "component_id": parent["component_id"],
                    "mechanism": parent["mechanism"], "variant": variant, "rendering": rendering,
                    "visible_spec": spec, "target": canonical({"answer": expected}), "prompt_sha256": prompt_hash})
    packet = {"version": VERSION, "policy": policy(), "policy_sha256": digest(policy()), "source_attribution": central_attribution(),
              "local_source_pins": {str(Path(__file__).resolve()): filehash(Path(__file__)),
                                     str(POLICY_PATH.resolve()): filehash(POLICY_PATH)},
              "public_bindings": bindings, "parents": parents, "policy_rows": rows, "oracle_rows": oracle,
              "guard_domain": {"domain_id": "synth_task_retention001", "class": "task",
                  "row_ids": [row["row_id"] for row in oracle], "component_ids": [row["component_id"] for row in oracle]},
              "independent_literal_world_components": 192, "rows": 768,
              "protocol_spaces_disjoint": {"guards": "literal_record_priority_text", "TRAIN_fresh_DEV": "public_R_S_item_index_contract"},
              "status": "CPU_TASK_GUARD_CANDIDATES_NATIVE_UNADMITTED", "native_launch_authorized": False,
              "training_authorized": False, "M0_enrollment_performed": False, "retention_qualified": False,
              "language_qualification_established": False, "long_context_qualification_established": False,
              "confirmation_eligible": False}
    packet["sha256"] = digest(packet)
    return packet


def audit_packet(packet):
    require(type(packet) is dict and packet.get("sha256") == digest({k: v for k, v in packet.items() if k != "sha256"}), "guard packet checksum")
    expected = compile_guards(packet["public_bindings"])
    require(canonical(packet) == canonical(expected), "guard source/policy/parent/oracle drift")
    count, negatives = Counter(), 0
    for row, truth in zip(packet["policy_rows"], packet["oracle_rows"]):
        require(set(row) == {"row_id", "messages", "prompt_sha256"}, "oracle/parent fields in policy row")
        require(score_response(row["messages"], truth["target"]) == {"correct": True, "malformed": False}, "guard target scoring disagreement")
        answer = strict_json(truth["target"])["answer"]
        wrong = [canonical({"answer": "UNSUPPORTED"}), '{"answer":[],"oracle":true}', '{"answer":true}',
                 canonical({"answer": ["UNSUPPORTED", "UNSUPPORTED"] if answer is None else None})]
        for response in wrong:
            require(not score_response(row["messages"], response)["correct"], "deliberate negative guard accepted")
            negatives += 1
        count[truth["mechanism"]] += 1
    require(count == {kind: 256 for kind in MECHANISMS} and len({r["component_id"] for r in packet["oracle_rows"]}) == 192,
            "complete task guard parent/row quotas")
    return {"status": "CPU_MECHANICS_AUDIT_ONLY_NOT_RETENTION_QUALIFICATION", "parent_components": 192,
            "rows": 768, "independent_oracle_agreements": 768, "negative_responses_rejected": negatives,
            "parent_counts_by_mechanism": {kind: 64 for kind in MECHANISMS}, "M0_enrollment_performed": False,
            "retention_qualified": False, "native_launch_authorized": False,
            "language_qualification_established": False, "long_context_qualification_established": False}


def measure_token_fit(packet, tokenizer):
    """Use pinned CPU engine/template directly; never append gold to policy IDs."""
    audit_packet(packet)
    binding = tokenizer.verify_binding()
    rows, overflows, completion_counts = [], [], []
    for row, truth in zip(packet["policy_rows"], packet["oracle_rows"]):
        text = tokenizer._template.render(messages=copy.deepcopy(row["messages"]), tools=None,
                                          add_generation_prompt=True, enable_thinking=False)
        require(type(text) is str and text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "pinned nonthinking generation prefix")
        ids = tokenizer._engine.encode(text, add_special_tokens=False).ids
        target_ids = tokenizer._engine.encode(truth["target"], add_special_tokens=False).ids
        require(type(ids) is list and ids and all(type(token) is int and token >= 0 for token in ids)
                and type(target_ids) is list and target_ids
                and all(type(token) is int and token >= 0 for token in target_ids)
                and 151645 not in target_ids, "guard CPU token arrays")
        require(tokenizer._engine.decode(ids, skip_special_tokens=False) == text
                and tokenizer._engine.decode(target_ids, skip_special_tokens=False) == truth["target"],
                "guard prompt/target tokenizer roundtrip")
        full_messages = copy.deepcopy(row["messages"]) + [{"role": "assistant", "content": truth["target"]}]
        full = tokenizer._template.render(messages=full_messages, tools=None,
                                           add_generation_prompt=False, enable_thinking=False)
        require(full == text + truth["target"] + "<|im_end|>\n", "guard full chat rendering differs")
        joint_ids = tokenizer._engine.encode(full, add_special_tokens=False).ids
        end = len(ids) + len(target_ids) + 1
        require(joint_ids[:len(ids)] == ids and joint_ids[len(ids):end] == target_ids + [151645]
                and joint_ids[end:] == tokenizer._engine.encode("\n", add_special_tokens=False).ids,
                "guard joint prompt/target/EOS token boundaries differ")
        record = {**row, "input_ids": ids, "attention_mask": [1] * len(ids), "prompt_tokens": len(ids),
                  "prompt_token_sha256": digest(ids)}
        rows.append(record)
        completion_counts.append({"row_id": row["row_id"], "completion_including_EOS_tokens": len(target_ids) + 1,
                                  "completion_token_sha256": digest(target_ids + [151645])})
        if len(ids) > 480 or len(target_ids) + 1 > 32:
            overflows.append({"row_id": row["row_id"], "prompt_tokens": len(ids), "oracle_completion_including_EOS_tokens": len(target_ids) + 1})
    require(tokenizer.verify_binding() == binding, "guard tokenizer binding changed")
    return {"packet_sha256": packet["sha256"], "binding": binding, "rows": rows,
            "oracle_completion_counts_CPU_only": completion_counts,
            "maximum_prompt_tokens": max(row["prompt_tokens"] for row in rows),
            "minimum_prompt_tokens": min(row["prompt_tokens"] for row in rows),
            "overflows": overflows, "context_cap_tokens": 512, "generation_reserve_tokens": 32,
            "prompt_target_EOS_joint_boundaries_verified": True,
            "status": "BLOCKED_CONTEXT_FIT" if overflows else "CPU_CONTEXT_MEASURED_NATIVE_UNADMITTED",
            "native_launch_authorized": False, "M0_enrollment_performed": False,
            "retention_qualified": False, "language_qualification_established": False,
            "long_context_qualification_established": False}


def validate_measurement(packet, measurement, *, tokenizer=None):
    require(type(measurement) is dict and measurement.get("packet_sha256") == packet["sha256"]
            and type(measurement.get("rows")) is list and len(measurement["rows"]) == 768
            and type(measurement.get("oracle_completion_counts_CPU_only")) is list
            and len(measurement["oracle_completion_counts_CPU_only"]) == 768,
            "complete guard measurement binding")
    fields = {"row_id", "messages", "prompt_sha256", "input_ids", "attention_mask", "prompt_tokens", "prompt_token_sha256"}
    overflow = []
    for row, original, completion in zip(measurement["rows"], packet["policy_rows"],
                                         measurement["oracle_completion_counts_CPU_only"]):
        require(type(row) is dict and set(row) == fields and all(row[key] == original[key] for key in original),
                "guard oracle leak or policy measurement row drift")
        ids = row["input_ids"]
        require(type(ids) is list and ids and all(type(token) is int and 0 <= token < 151936 for token in ids)
                and row["attention_mask"] == [1] * len(ids) and type(row["prompt_tokens"]) is int
                and row["prompt_tokens"] == len(ids) and row["prompt_token_sha256"] == digest(ids),
                "guard prompt-only token arrays/mask/count/hash")
        require(type(completion) is dict and set(completion) == {"row_id", "completion_including_EOS_tokens", "completion_token_sha256"}
                and completion["row_id"] == row["row_id"] and type(completion["completion_including_EOS_tokens"]) is int
                and completion["completion_including_EOS_tokens"] > 0 and re.fullmatch(r"[0-9a-f]{64}", completion["completion_token_sha256"]),
                "guard exposed-oracle token count metadata")
        if len(ids) > 480 or completion["completion_including_EOS_tokens"] > 32:
            overflow.append({"row_id": row["row_id"], "prompt_tokens": len(ids),
                "oracle_completion_including_EOS_tokens": completion["completion_including_EOS_tokens"]})
    require(measurement["overflows"] == overflow and measurement["status"] == (
        "BLOCKED_CONTEXT_FIT" if overflow else "CPU_CONTEXT_MEASURED_NATIVE_UNADMITTED")
        and measurement["context_cap_tokens"] == 512 and measurement["generation_reserve_tokens"] == 32
        and measurement["prompt_target_EOS_joint_boundaries_verified"] is True
        and measurement["maximum_prompt_tokens"] == max(r["prompt_tokens"] for r in measurement["rows"])
        and measurement["minimum_prompt_tokens"] == min(r["prompt_tokens"] for r in measurement["rows"]),
        "guard token fit/overflow claim drift")
    require(all(measurement[key] is False for key in ("native_launch_authorized", "M0_enrollment_performed", "retention_qualified",
                "language_qualification_established", "long_context_qualification_established")), "guard measurement permission drift")
    if tokenizer is not None:
        require(packet["public_bindings"]["scope"] == "TEST_ONLY_NOT_REPOSITORY_ADMISSION",
                "injected tokenizer is test-only, not a production measurement verifier")
        checked = tokenizer
    else:
        if str(PACKAGE) not in sys.path: sys.path.insert(0, str(PACKAGE))
        from cpu_tokenizer import load_local_tokenizer
        checked = load_local_tokenizer()
    # Recompute every actual prompt, canonical target and joint EOS boundary.
    # Self-reported hashes/counts are not an independent byte witness and cannot
    # authorize a shorter vector containing different policy text.
    fresh = measure_token_fit(packet, checked)
    require(canonical(measurement) == canonical(fresh),
            "fresh pinned CPU retokenization disagrees with prompt/target/EOS measurement")


def write_bundle(packet, measurement, directory, *, tokenizer=None):
    no_frameworks()
    audit = audit_packet(packet)
    validate_measurement(packet, measurement, tokenizer=tokenizer)
    target = Path(directory).absolute()
    root = PACKAGE / "artifacts/guards_dev"
    require(target.parent.resolve() == root.resolve() and not target.exists()
            and not root.is_symlink()
            and not (PACKAGE / "artifacts").is_symlink(), "new owned task guard bundle leaf required")
    root.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / "policy").mkdir()
    (target / "oracle_CPU_DEV_exposed").mkdir()
    values = {"packet.json": packet, "audit.json": audit, "token_measurement.json": {k: v for k, v in measurement.items() if k != "rows"},
              "policy/prompts.json": {"rows": measurement["rows"], "native_launch_authorized": False},
              "oracle_CPU_DEV_exposed/labels.json": {"rows": packet["oracle_rows"], "oracle_labels_exposed_CPU_DEV": True}}
    manifest = {"status": measurement["status"], "files": {}, "native_launch_authorized": False,
                "M0_enrollment_performed": False, "retention_qualified": False, "confirmation_eligible": False,
                "language_qualification_established": False, "long_context_qualification_established": False}
    for name, value in values.items():
        raw = (canonical(value) + "\n").encode("ascii")
        require(len(raw) <= MAX_JSON_BYTES, "guard bundle byte bound")
        with (target / name).open("xb") as stream: require(stream.write(raw) == len(raw), "short guard write")
        manifest["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    with (target / "manifest.json").open("x", encoding="utf-8") as stream: stream.write(canonical(manifest) + "\n")
    return manifest
