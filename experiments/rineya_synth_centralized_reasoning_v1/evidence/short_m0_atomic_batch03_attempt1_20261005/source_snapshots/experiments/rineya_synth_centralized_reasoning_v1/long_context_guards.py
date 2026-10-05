"""CPU-only long-context guard construction; no model or protected-bank reader.

The canonical-world oracle and the public independent visible-text solver must
agree on the actual fitted prompt. Only policy messages/IDs leave the policy
packet; source facts, labels and factor metadata stay in a separate exposed DEV
sidecar. Longer-context native qualification and baseline enrollment stay closed.
"""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] if HERE.name == "long_context_cpu_work" else HERE
ROOT = PACKAGE.parents[1]
for location in (ROOT, PACKAGE):
    if str(location) not in sys.path: sys.path.insert(0, str(location))
import rs_contract as contract
from cpu_tokenizer import load_local_tokenizer, TEMPLATE_KWARGS, ASSISTANT_PREFIX

VERSION = "synth-long-context-cpu-guards-1"
POLICY_PATH = HERE / "LONG_CONTEXT_GUARD_POLICY.json"
PUBLIC_PACKAGE = ROOT / "experiments/post_training_mimo_centralized_v1"
PUBLIC_SCRIPTS = PUBLIC_PACKAGE / "scripts"
PUBLIC_METADATA = PUBLIC_PACKAGE / "receipts/LONG_CONTEXT_DEV_DATA_RECEIPT.json"
PUBLIC_SOURCE_NAMES = ("long_context_tasks.py", "build_long_context.py", "tokenize_bridge.py", "bridge_data.py")
FRAMEWORKS = {"torch", "transformers", "tensorflow", "jax", "flax"}
VARIANTS = ((True, False), (True, True), (False, False), (False, True))
MAX_LINE_BYTES = 1024 * 1024


def no_frameworks():
    contract.require(not any(name.split(".", 1)[0] in FRAMEWORKS for name in sys.modules), "CPU guards cannot import model frameworks")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("rb") as stream: raw = stream.read(4 * 1024 * 1024 + 1)
    contract.require(len(raw) <= 4 * 1024 * 1024, "bounded public metadata/policy")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=contract._pairs, parse_constant=contract._bad_constant)


def public_modules():
    no_frameworks()
    # Both modules are public source-only. Never call their model/tokenizer
    # loader, allocate_bank(), protected-role allocator, or dataset builder.
    tasks = importlib.import_module("experiments.post_training_mimo_centralized_v1.scripts.long_context_tasks")
    builder = importlib.import_module("experiments.post_training_mimo_centralized_v1.scripts.build_long_context")
    no_frameworks()
    return tasks, builder


def policy():
    value = read_json(POLICY_PATH)
    expected = {"source_pool_seed": 2026100309, "source_candidates_per_mechanism": 4096,
                "world_components": 192, "components_per_mechanism": 96,
                "components_per_mechanism_position": 24, "total_sequence_envelopes": [1024, 2048, 4096],
                "prompt_token_targets": [992, 2016, 4064], "generation_reserve_tokens": 32,
                "max_target_tokens": 16, "CPU_process_RSS_limit_bytes": 512 * 1024**2,
                "body_position_intervals": {"early": [.1, .2], "middle": [.45, .55], "late": [.8, .9]}}
    contract.require(all(contract.canonical(value.get(key)) == contract.canonical(want) for key, want in expected.items()), "long-context source/bounds policy drift")
    for field in ("native_execution_authorized", "training_authorized", "protected_bank_access",
                  "native_long_context_profile_qualified", "source_correct_membership_enrolled",
                  "statistical_preservation_qualified", "confirmation_eligible"):
        contract.require(value.get(field) is False, "closed long-context qualification flag")
    return value


def source_inventory():
    """Exact public and imported local executable-source/metadata closure."""
    return [Path(__file__).resolve(), POLICY_PATH.resolve(), PUBLIC_METADATA.resolve()] + \
           [(PUBLIC_SCRIPTS / name).resolve() for name in PUBLIC_SOURCE_NAMES] + \
           [(PACKAGE / name).resolve() for name in ("rs_contract.py", "cpu_tokenizer.py", "rs_verifier.py")]


def validate_catalog_sources(catalog):
    contract.require(type(catalog) is dict and type(catalog.get("sources")) is dict and
                     set(catalog["sources"]) == {str(path) for path in source_inventory()}, "complete CPU guard source inventory changed")
    for path, expected in catalog["sources"].items():
        contract.require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected) and sha(path) == expected,
                         "public/imported/owned source changed")
    return True


def _historical_signature(family, seed, index, tasks):
    """Reconstruct only normalized numerical commitments, never old payloads.

    No subject, depot/revision alias, answer symbol, role payload or label is
    generated. This mirrors the public generator's disclosed hash/RNG recipe.
    """
    digest = tasks.sha256(tasks.json_bytes([family, seed, index]))
    number = int(digest[:12], 16)
    if family == tasks.FAMILIES[0]:
        first = 10 + number % 31
        second = 50 + int(digest[12:16], 16) % 31
        return {"family": family, "mass": 1 + int(digest[16:24], 16) % 99,
                "intervals": [[1, first], [first + 1, second], [second + 1, 99]]}
    rng = random.Random(int(digest, 16))
    rng.sample(range(26**5), 4)  # Advance the source's symbol draw without constructing labels.
    revisions = list(range(4)); rng.shuffle(revisions)
    writers = rng.sample((0, 1, 2), 2) * 2; rng.shuffle(writers)
    authority = rng.choice(sorted(set(writers)))
    cycles = [1 + number % 20, 26 + int(digest[12:16], 16) % 20,
              51 + int(digest[16:20], 16) % 20, 76 + int(digest[20:24], 16) % 20]
    return {"family": family, "catalog": [[cycle, writer == authority] for cycle, writer in zip(cycles, writers)]}


def historical_exclusions(tasks):
    metadata = read_json(PUBLIC_METADATA)
    allocation = metadata["allocation"]
    seed, counts = allocation["seed"], allocation["candidates_attempted"]
    contract.require(type(seed) is int and seed == 20260928 and set(counts) == set(tasks.FAMILIES), "public source-prefix metadata")
    keys = set()
    for family, count in counts.items():
        contract.require(type(count) is int and 0 < count <= 100000, "bounded prior source prefix")
        keys.update(tasks.sha256(tasks.json_bytes(_historical_signature(family, seed, index, tasks))) for index in range(count))
    return {"scope": "PUBLIC_NUMERIC_SOURCE_PREFIX_COMMITMENTS_ONLY_NO_PAYLOADS", "source_metadata_sha256": sha(PUBLIC_METADATA),
            "seed": seed, "prefix_counts": counts, "world_commitments": sorted(keys),
            "protected_payloads_opened": False, "SPENT_demo_outputs_opened": False}


def compile_catalog():
    no_frameworks()
    frozen = policy()
    tasks, builder = public_modules()
    contract.require(list(tasks.FAMILIES) == frozen["mechanisms"] and list(tasks.POSITIONS) == frozen["positions"], "public mechanism/position drift")
    contract.require({key: list(value) for key, value in builder.INTERVALS.items()} == frozen["body_position_intervals"], "public evidence band drift")
    excluded = historical_exclusions(tasks)
    seen, worlds, attempts = set(excluded["world_commitments"]), [], []
    # Source generation is independent of rendering, token fit and M0 outcomes.
    for family in tasks.FAMILIES:
        family_candidates = []
        for index in range(frozen["source_candidates_per_mechanism"]):
            spec = tasks.generate_spec(family, frozen["source_pool_seed"], index)
            signature = tasks.semantic_signature(spec)
            component = tasks.sha256(tasks.json_bytes(signature))
            duplicate = component in seen
            attempts.append({"family": family, "candidate_index": index, "component_id": component,
                             "excluded_or_duplicate": duplicate})
            if duplicate: continue
            seen.add(component)
            family_candidates.append({"family": family, "candidate_index": index, "component_id": component, "spec": spec})
        contract.require(len(family_candidates) >= 96, "BLOCKED_FRESH_WORLD_COMPONENTS_NO_POOL_EXTENSION")
        for ordinal, candidate in enumerate(family_candidates[:96]):
            candidate.update(world_id="synth-lc-" + candidate["component_id"], population=frozen["population"],
                             placement=tasks.POSITIONS[ordinal % 4], missing_index=ordinal % 3,
                             structural_mechanism_id=family)
            worlds.append(candidate)
    contract.require(len(worlds) == len({world["component_id"] for world in worlds}) == 192, "192 complete world components")
    sources = {str(path): sha(path) for path in source_inventory()}
    catalog = {"version": VERSION, "policy": frozen, "exclusions": excluded, "sources": sources,
               "worlds": worlds, "source_attempts": attempts, "planned_rows": 192 * 3 * 4,
               "numeric_world_components": 192, "structural_mechanisms": 2,
               "statistical_independence_qualified": False, "native_execution_authorized": False,
               "source_correct_membership_enrolled": False, "confirmation_eligible": False}
    catalog["sha256"] = contract.digest(catalog)
    validate_catalog_sources(catalog)
    return catalog


def canonical_label(spec, answerable=True):
    tasks, _ = public_modules()
    spec = tasks.canonicalize(spec)
    contract.require(type(answerable) is bool, "answerability flag")
    if not answerable: return "UNKNOWN"
    if spec["family"] == tasks.FAMILIES[0]:
        matching = [row["depot"] for row in spec["routes"] if row["low"] <= spec["mass"] <= row["high"]]
        return next(row["symbol"] for row in spec["seals"] if row["depot"] == matching[0])
    authorized = [row for row in spec["catalog"] if row["writer"] == spec["authority"]]
    revision = max(authorized, key=lambda row: row["cycle"])["revision"]
    return next(row["symbol"] for row in spec["values"] if row["revision"] == revision)


def variant(world, answerable, adversarial):
    """Faithful public record contract on a separately registered SYNTH DEV world."""
    tasks, _ = public_modules()
    spec = tasks.canonicalize(world["spec"])
    subject, missing = spec["subject"], world["missing_index"]
    contract.require(type(answerable) is bool and type(adversarial) is bool and type(missing) is int and 0 <= missing <= 2, "variant flags")
    rng = random.Random(int(world["component_id"][:16], 16))
    if spec["family"] == tasks.FAMILIES[0]:
        routes, seals = list(spec["routes"]), list(spec["seals"]); rng.shuffle(routes); rng.shuffle(seals)
        facts = [f"MASS {subject}={spec['mass']}.", f"ROUTES {subject}=" + ",".join(f"{row['low']}-{row['high']}:{row['depot']}" for row in routes) + ".",
                 f"SEALS {subject}=" + ",".join(f"{row['depot']}:{row['symbol']}" for row in seals) + "."]
        query = f"Return the routed seal for {subject}."
        symbols = {row["symbol"] for row in seals}
        other = next("S" + tasks._letters(i) for i in range(4) if "S" + tasks._letters(i) not in symbols)
        alias = tasks._unused_alias({row["depot"] for row in routes})
        kind, payload = (("MASS", "0"), ("ROUTES", f"1-99:{alias}"), ("SEALS", f"{alias}:{other}"))[missing]
    else:
        catalog, values = list(spec["catalog"]), list(spec["values"]); rng.shuffle(catalog); rng.shuffle(values)
        facts = [f"AUTH {subject}={spec['authority']}.", f"CAT {subject}=" + ",".join(f"{row['revision']}:{row['cycle']}:{row['writer']}" for row in catalog) + ".",
                 f"VAL {subject}=" + ",".join(f"{row['revision']}:{row['symbol']}" for row in values) + "."]
        query = f"Return the latest authorized record for {subject}."
        symbols = {row["symbol"] for row in values}
        other = next("S" + tasks._letters(i) for i in range(5) if "S" + tasks._letters(i) not in symbols)
        revision = tasks._unused_alias({row["revision"] for row in catalog})
        writer = tasks._unused_alias({row["writer"] for row in catalog})
        kind, payload = (("AUTH", writer), ("CAT", f"{revision}:0:{writer}"), ("VAL", f"{revision}:{other}"))[missing]
    decoys = [f"{kind} OTHER={payload}.", f"ARCHIVE {kind} {subject}={payload}.", f"UNSIGNED {kind} {subject}={payload}."]
    if not adversarial: decoys = [tasks.filler_line(i) for i in range(3)]
    slots = list(facts)
    if not answerable: slots[missing] = "NOTE."
    return {"query": query, "facts": facts, "slots": slots, "decoys": decoys,
            "answerable": answerable, "adversarial": adversarial, "label": canonical_label(spec, answerable)}


def parse_prompt(messages):
    tasks, _ = public_modules()
    contract.require(type(messages) is list and len(messages) == 1 and type(messages[0]) is dict and
                     set(messages[0]) == {"role", "content"} and messages[0]["role"] == "user", "prompt-only public LC wrapper")
    text = messages[0]["content"]
    prefix = tasks.SYSTEM_PROMPT + "\nCONTEXT\n"
    contract.require(type(text) is str and text.startswith(prefix), "LC rule drift")
    remainder = text[len(prefix):]
    contract.require(remainder.count("QUERY\n") == 1, "LC query boundary")
    body, query = remainder.split("QUERY\n")
    return query, body


def solve_policy(messages):
    tasks, _ = public_modules()
    query, body = parse_prompt(messages)
    return tasks.solve_visible(query, body)


def validate_registered_records(world, messages, answerable, adversarial):
    """Validate the factorial's actual record multiset, including all decoys.

    A semantic answer alone cannot establish an adversarial exposure. Only the
    registered complete NOTE lines are irrelevant filler. No additional live,
    other-key, ARCHIVE or UNSIGNED records may silently enter the prompt.
    """
    tasks, _ = public_modules()
    query, body = parse_prompt(messages)
    expected = variant(world, answerable, adversarial)
    contract.require(query == expected["query"], "registered query changed")
    decoy_prefix = "".join(line + "\n" for line in expected["decoys"])
    contract.require(body.startswith(decoy_prefix), "registered decoy prefix missing or changed")
    actual_records = Counter(line for line in body.splitlines() if line and line not in tasks.FILLER_TEXTS)
    expected_records = Counter(line for line in expected["slots"] + expected["decoys"]
                               if line not in tasks.FILLER_TEXTS)
    contract.require(actual_records == expected_records, "registered record/decoy multiset changed or unexpected authoritative/shadow record")
    return expected


class CPUAdapter:
    """Only the pinned Rust tokenizer/Jinja engine, not an HF model framework."""
    eos_token_id, eos_token = 151645, "<|im_end|>"
    all_special_ids = (151643, 151644, 151645)
    def __init__(self, tokenizer): self.checked = tokenizer
    def encode(self, text, add_special_tokens=False):
        contract.require(add_special_tokens is False, "no extra special tokens")
        return self.checked._engine.encode(text, add_special_tokens=False).ids
    def decode(self, ids, skip_special_tokens=False):
        return self.checked._engine.decode(ids, skip_special_tokens=skip_special_tokens)
    def apply_chat_template(self, messages, tokenize, **kwargs):
        contract.require(kwargs == TEMPLATE_KWARGS, "exact non-thinking template kwargs")
        text = self.checked._template.render(messages=copy.deepcopy(messages), tools=None, **kwargs)
        contract.require(text.endswith(ASSISTANT_PREFIX), "assistant prefix drift")
        return self.encode(text) if tokenize else text
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = self.checked._engine.encode(text, add_special_tokens=add_special_tokens)
        result = {"input_ids": encoded.ids}
        if return_offsets_mapping: result["offset_mapping"] = encoded.offsets
        return result


def render_length(world, tokenizer, fitter, prompt_tokens):
    tasks, builder = public_modules()
    variants = [variant(world, *flags) for flags in VARIANTS]
    query = variants[0]["query"]
    before, after, prefix, suffix = builder._wrapper(tokenizer, query)
    prefix_ids, suffix_ids = tokenizer.encode(prefix), tokenizer.encode(suffix)
    slots = [fitter.common_slot([[item["slots"][i]] for item in variants]) for i in range(3)]
    decoy_width, decoy_texts = fitter.common_slot([item["decoys"] for item in variants])
    body_tokens = prompt_tokens - len(prefix_ids) - len(suffix_ids)
    starts, gaps, tail, regions = builder._layout(fitter, body_tokens, world["placement"], [slot[0] for slot in slots], decoy_width)
    rows = []
    for index, item in enumerate(variants):
        pieces, slot_chars = [decoy_texts[index]], []
        for i in range(3):
            pieces.append(gaps[i]); slot_chars.append(sum(map(len, pieces))); pieces.append(slots[i][1][index])
        pieces.append(tail)
        body = "".join(pieces)
        messages = [{"role": "user", "content": before + body + after}]
        validate_registered_records(world, messages, item["answerable"], item["adversarial"])
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS)
        ids = tokenizer.encode(rendered)
        contract.require(rendered == prefix + body + suffix and ids == prefix_ids + tokenizer.encode(body) + suffix_ids and
                         len(ids) == prompt_tokens and len(tokenizer.encode(body)) == body_tokens, "exact prompt length/additive boundary fit")
        contract.require(not set(tokenizer.encode(body)).intersection(tokenizer.all_special_ids), "PAD/EOS/special-token filler forbidden")
        observed = solve_policy(messages)
        contract.require(observed == item["label"], "visible prompt solver/canonical oracle disagreement")
        offsets = tokenizer(rendered, return_offsets_mapping=True)["offset_mapping"]
        spans = []
        for i, fact in enumerate(item["slots"]):
            span = builder._span_receipt(rendered, offsets, fact, len(prefix) + slot_chars[i], len(prefix), len(prefix_ids), ids)
            low, high = builder.INTERVALS[regions[i]]
            contract.require(low * body_tokens <= span["body_token_start"] < span["body_token_end"] <= high * body_tokens and
                             span["body_token_start"] == starts[i], "source evidence placement changed")
            spans.append(span)
        for i, fact in enumerate(item["facts"]):
            present = item["answerable"] or i != world["missing_index"]
            contract.require(body.splitlines().count(fact) == int(present), "required record duplication/deletion")
            if present: contract.require(tasks.solve_visible(query, body.replace(fact + "\n", "NOTE.\n", 1)) == "UNKNOWN", "record block is redundant")
        target_ids = tokenizer.encode(item["label"])
        contract.require(0 < len(target_ids) <= 16 and tokenizer.decode(target_ids) == item["label"], "symbol token bound/roundtrip")
        name = ("answerable" if item["answerable"] else "unanswerable") + ("_adversarial" if item["adversarial"] else "_neutral")
        row_id = contract.digest({"component": world["component_id"], "prompt_tokens": prompt_tokens, "variant": name})
        policy_row = {"row_id": row_id, "messages": messages, "input_ids": ids, "attention_mask": [1] * len(ids),
                      "prompt_tokens": len(ids), "prompt_token_sha256": contract.digest(ids)}
        oracle = {"row_id": row_id, "world_id": world["world_id"], "component_id": world["component_id"], "family": world["family"],
                  "placement": world["placement"], "missing_index": world["missing_index"], "variant": name,
                  "answerable": item["answerable"], "adversarial": item["adversarial"], "target": item["label"],
                  "target_tokens": len(target_ids), "prompt_tokens": len(ids), "body_tokens": body_tokens,
                  "total_sequence_envelope": prompt_tokens + 32, "generation_reserve_tokens": 32,
                  "support_spans": spans, "canonical_oracle_agrees": True, "record_block_ablations_UNKNOWN": True,
                  "native_profile_qualified": False, "source_correct_membership": None}
        rows.append((policy_row, oracle))
    return rows


def _write_line(stream, value):
    raw = (contract.canonical(value) + "\n").encode("ascii")
    contract.require(len(raw) <= MAX_LINE_BYTES and stream.write(raw) == len(raw), "bounded complete JSONL write")


def audit_packet(world, policy_row, oracle, tokenizer):
    """Recheck input, oracle and evidence spans from actual visible/token inputs."""
    tasks, builder = public_modules()
    contract.require(type(policy_row) is dict and set(policy_row) == {"row_id", "messages", "input_ids", "attention_mask", "prompt_tokens", "prompt_token_sha256"}, "strict policy-only field allowlist")
    fields = {"row_id", "world_id", "component_id", "family", "placement", "missing_index", "variant", "answerable", "adversarial", "target", "target_tokens", "prompt_tokens", "body_tokens", "total_sequence_envelope", "generation_reserve_tokens", "support_spans", "canonical_oracle_agrees", "record_block_ablations_UNKNOWN", "native_profile_qualified", "source_correct_membership"}
    contract.require(type(oracle) is dict and set(oracle) == fields, "strict oracle packet fields")
    contract.require(type(oracle["answerable"]) is bool and type(oracle["adversarial"]) is bool and
                     oracle["native_profile_qualified"] is False and oracle["source_correct_membership"] is None,
                     "unqualified native/membership or factor claim")
    for key in ("world_id", "component_id", "family", "placement", "missing_index"):
        contract.require(oracle[key] == world[key], "world component identity drift")
    length = policy_row["prompt_tokens"]
    contract.require(type(length) is int and length in (992, 2016, 4064) and oracle["prompt_tokens"] == length and
                     oracle["total_sequence_envelope"] == length + 32 and oracle["generation_reserve_tokens"] == 32,
                     "length/reserve contract drift")
    name = ("answerable" if oracle["answerable"] else "unanswerable") + ("_adversarial" if oracle["adversarial"] else "_neutral")
    row_id = contract.digest({"component": world["component_id"], "prompt_tokens": length, "variant": name})
    contract.require(oracle["variant"] == name and policy_row["row_id"] == oracle["row_id"] == row_id, "row factor identity drift")
    query, body = parse_prompt(policy_row["messages"])
    expected_variant = validate_registered_records(world, policy_row["messages"], oracle["answerable"], oracle["adversarial"])
    observed = tasks.solve_visible(query, body)
    contract.require(observed == canonical_label(world["spec"], oracle["answerable"]) == oracle["target"], "visible/canonical oracle or label drift")
    rendered = tokenizer.apply_chat_template(policy_row["messages"], tokenize=False, **TEMPLATE_KWARGS)
    ids = tokenizer.encode(rendered)
    contract.require(ids == policy_row["input_ids"] and len(ids) == length and
                     policy_row["attention_mask"] == [1] * length and policy_row["prompt_token_sha256"] == contract.digest(ids),
                     "stored native IDs/length/mask drift")
    _, _, prefix, suffix = builder._wrapper(tokenizer, query)
    prefix_ids = tokenizer.encode(prefix)
    body_tokens = length - len(prefix_ids) - len(tokenizer.encode(suffix))
    contract.require(oracle["body_tokens"] == body_tokens and not set(tokenizer.encode(body)).intersection(tokenizer.all_special_ids), "body/PAD/EOS boundary drift")
    target_ids = tokenizer.encode(oracle["target"])
    contract.require(oracle["target_tokens"] == len(target_ids) and 0 < len(target_ids) <= 16, "target token cap drift")
    offsets = tokenizer(rendered, return_offsets_mapping=True)["offset_mapping"]
    regions = ("early", "middle", "late") if world["placement"] == "distributed" else (world["placement"],) * 3
    contract.require(type(oracle["support_spans"]) is list and len(oracle["support_spans"]) == 3, "three dependency span receipts")
    for i, span in enumerate(oracle["support_spans"]):
        contract.require(span["text"] == expected_variant["slots"][i], "support text drift")
        actual = builder._span_receipt(rendered, offsets, span["text"], span["whole_prompt_char_start"], len(prefix), len(prefix_ids), ids)
        contract.require(actual == span, "native evidence span drift")
        low, high = builder.INTERVALS[regions[i]]
        contract.require(low * body_tokens <= actual["body_token_start"] < actual["body_token_end"] <= high * body_tokens, "evidence placement drift")
    for i, fact in enumerate(expected_variant["facts"]):
        present = oracle["answerable"] or i != world["missing_index"]
        contract.require(body.splitlines().count(fact) == int(present), "required record changed")
        if present: contract.require(tasks.solve_visible(query, body.replace(fact + "\n", "NOTE.\n", 1)) == "UNKNOWN", "fitted record ablation failed")
    return True


def build_bundle(directory):
    """Stream every planned derivative; retain exact blockers without replacement."""
    started = time.monotonic()  # Entire catalogue + tokenizer + fitting scope.
    no_frameworks()
    import psutil
    sampled_peak, os_peak = 0, None
    def resource_guard():
        nonlocal sampled_peak, os_peak
        info = psutil.Process().memory_info()
        sampled_peak = max(sampled_peak, info.rss)
        measured = getattr(info, "peak_wset", None)
        if measured is not None:
            contract.require(type(measured) is int and measured >= info.rss, "OS working-set high-water measurement")
            os_peak = max(os_peak or 0, measured)
        contract.require(info.rss <= 512 * 1024**2 and (os_peak is None or os_peak <= 512 * 1024**2)
                         and time.monotonic() - started < 1800, "CPU512MiB/full-scope wall bound")
    resource_guard()
    catalog = compile_catalog()
    resource_guard()
    target = Path(directory).absolute()
    root = PACKAGE / "artifacts" / "long_context_dev"
    contract.require(target.parent.resolve() == root.resolve() and not target.exists() and not target.is_symlink()
                     and not root.is_symlink() and not (PACKAGE / "artifacts").is_symlink(), "new owned CPU guard bundle leaf")
    root.mkdir(parents=True, exist_ok=True); target.mkdir()
    raw_catalog = (contract.canonical(catalog) + "\n").encode("ascii")
    contract.require(len(raw_catalog) <= 4 * 1024**2, "bounded source catalog")
    with (target / "catalog.json").open("xb") as stream:
        contract.require(stream.write(raw_catalog) == len(raw_catalog), "complete catalog write")
    checked = load_local_tokenizer()
    binding = checked.verify_binding()
    tokenizer = CPUAdapter(checked)
    tasks, builder = public_modules()
    fitter = builder.NoteFitter(tokenizer, limit=4064)
    resource_guard()
    constructed, blockers, counts = 0, [], Counter()
    with (target / "policy_inputs.jsonl").open("xb", buffering=0) as policy_stream, (target / "oracle_CPU_DEV_exposed.jsonl").open("xb", buffering=0) as oracle_stream:
        for world in catalog["worlds"]:
            resource_guard()
            for length in catalog["policy"]["prompt_token_targets"]:
                try:
                    rows = render_length(world, tokenizer, fitter, length)
                except ValueError as error:
                    blockers.append({"world_id": world["world_id"], "component_id": world["component_id"], "family": world["family"],
                                     "placement": world["placement"], "prompt_token_target": length, "planned_variant_rows": 4,
                                     "reason": str(error), "parent_replaced": False})
                    continue
                for policy_row, oracle in rows:
                    audit_packet(world, policy_row, oracle, tokenizer)
                    wrong = "UNKNOWN" if oracle["answerable"] else canonical_label(world["spec"])
                    contract.require(not tasks.score_answer(wrong, oracle["target"]), "deliberately wrong answer accepted")
                    for malformed in ('{"answer":"' + oracle["target"] + '"}', "Answer: " + oracle["target"]):
                        contract.require(not tasks.score_answer(malformed, oracle["target"]), "format negative control accepted")
                    _write_line(policy_stream, policy_row); _write_line(oracle_stream, oracle)
                    constructed += 1; counts[(world["family"], length)] += 1
    contract.require(checked.verify_binding() == binding, "pinned CPU tokenizer changed")
    validate_catalog_sources(catalog)
    resource_guard()
    receipt = {"status": "BLOCKED_EXACT_LENGTH_OR_PLACEMENT_FIT" if blockers else "CPU_LONG_CONTEXT_GUARDS_PREPARED_NATIVE_UNADMITTED",
               "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "catalog_sha256": catalog["sha256"],
               "planned_world_components": 192, "planned_rows": 2304, "constructed_rows": constructed,
               "blocked_variant_rows": sum(item["planned_variant_rows"] for item in blockers), "blockers": blockers,
               "actual_prompt_counts": [{"family": family, "prompt_tokens": length, "rows": count} for (family, length), count in counts.items()],
               "source_mechanisms": 2, "all_worlds_retained": True, "row_filtering_for_admission": False,
               "generation_reserve_tokens": 32, "CPU_sampled_peak_RSS_bytes": sampled_peak,
               "CPU_OS_peak_working_set_bytes": os_peak,
               "CPU_peak_measurement_scope": "Windows_process_lifetime_peak_wset" if os_peak is not None else "sampled_RSS_only_not_OS_HWM",
               "CPU_wall_seconds": time.monotonic() - started, "CPU_timer_scope": "entire_catalogue_tokenizer_fitter_construction_and_validation",
               "tokenizer_binding": binding, "policy_file_sha256": sha(target / "policy_inputs.jsonl"),
               "oracle_file_sha256": sha(target / "oracle_CPU_DEV_exposed.jsonl"),
               "model_or_GPU_calls": 0, "protected_payloads_opened": False, "SPENT_demo_outputs_opened": False,
               "native_long_context_profile_qualified": False, "source_correct_membership_enrolled": False,
               "statistical_preservation_qualified": False, "confirmation_eligible": False,
               "native_resource_qualification_inferred": False}
    contract.require(constructed + receipt["blocked_variant_rows"] == 2304, "planned derivative denominator changed")
    with (target / "receipt.json").open("xb") as stream: _write_line(stream, receipt)
    return receipt
