"""Excluded finite mechanical witness for query-dependent closed-world lookup.

build_witness(seed=20261005, family="minimal"|"all_regular") returns signed metadata:
schema gd001_query_dependence_witness_v1, evidence_kind finite_mechanical_witness,
fixture_seed, family, entities (4), fields (4), nuisance_entities (2), values (12),
generator_policy, contexts. Each context has context_id, 12 records {entity,field,
value}, context_text/context_sha256 and 16 queries {query_id,entity,field,
public_input:{context_text,request},oracle_answer:{v:string|null}}. Context/row IDs,
oracle values and generator metadata are excluded from public_input. Every question
uses the same public string|null rule. No absent-cell or queried-role markers occur
in records. Symbols and values are opaque, distinct and equal-width within domains.

The minimal core relations are 1010/0101/1010/0101, its complement,
0110/1001/1010/0101 (one 2x2 switch), and its complement. all_regular enumerates
all 90 binary 4x4 matrices with row/column degree 2; every one includes all 16
queries. Four nuisance records are identical between contexts. Record counts,
entity/field sets AND frequencies, value multisets/lengths are fixed between
contexts. Each fixed query is present in half the contexts. audit_witness compares
an independently built table with brute scans and validates conditional chance
bounds plus lookup-operator invariances. It performs no model, bank or file IO.

All outputs are EXCLUDED_MECHANICAL_FIXTURE: no actual TRAIN/DEV/native bank,
learning, retention, quality, admission, confirmation or execution claim. Conditional
chance certificates concern only the declared finite population and information
sets. They are not statistical results or proof of a model reading the query.
"""
from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict

from .common import Blocked, digest, require, signed, verify

SCHEMA = "gd001_query_dependence_witness_v1"
DEFAULT_SEED = 20261005
_POLICY = {"scope": "EXCLUDED_MECHANICAL_FIXTURE",
           "public_null_rule": "closed_world_exact_single_relation_lookup",
           "actual_train_admission": False, "actual_dev_admission": False,
           "native_bank": False, "actual_learning_established": False,
           "retention_established": False, "model_quality_claim": False,
           "confirmation_authorized": False, "authorizes_execution": False,
           "no_model_quality_claim": True}


def _fields(value, fields, label):
    require(isinstance(value, dict) and set(value) == set(fields), f"Invalid/extra {label} fields; metadata is not public input")


def _symbols(value, count, label):
    require(isinstance(value, list) and len(value) == count
            and all(isinstance(v, str) and v.isascii() and v.isalnum() and len(v) == 4 for v in value)
            and len(set(value)) == count, f"Distinct opaque equal-width {label} required")


def _render(records):
    return "\n".join(f'Record(entity={r["entity"]},field={r["field"]},value={r["value"]}).' for r in records)


def _request(entity, field):
    return (f'Query entity={entity}; field={field}. Use only exact matching records. '
            'Return compact JSON {"v":string|null}; use null exactly when no matching record exists.')


def _scan(records, entity, field):
    matches = [r["value"] for r in records if r["entity"] == entity and r["field"] == field]
    require(len(matches) <= 1, "Ambiguous duplicate entity-field fact")
    return {"v": matches[0] if matches else None}


def _table(records):
    result = {}
    for record in records:
        row = result.setdefault(record["entity"], {})
        require(record["field"] not in row, "Duplicate entity-field record in independent table")
        row[record["field"]] = record["value"]
    return result


def _table_answer(table, entity, field):
    return {"v": table.get(entity, {}).get(field)}


def _matrix(records, entities, fields):
    keys = {(r["entity"], r["field"]) for r in records}
    return tuple(tuple(int((e, f) in keys) for f in fields) for e in entities)


def _complement(matrix):
    return tuple(tuple(1 - bit for bit in row) for row in matrix)


def _matrices(family):
    if family == "minimal":
        base = ((1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1))
        switched = ((0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1))
        return [base, _complement(base), switched, _complement(switched)]
    require(family == "all_regular", "Only excluded minimal/all_regular fixture families are supported")
    rows = [tuple(int(j in pair) for j in range(4)) for pair in itertools.combinations(range(4), 2)]
    return [matrix for matrix in itertools.product(rows, repeat=4)
            if all(sum(row[column] for row in matrix) == 2 for column in range(4))]


def _profile(records):
    return {"record_count": len(records), "entity_frequencies": dict(sorted(Counter(r["entity"] for r in records).items())),
            "field_frequencies": dict(sorted(Counter(r["field"] for r in records).items())),
            "value_multiset": sorted(r["value"] for r in records),
            "value_lengths": sorted(len(r["value"]) for r in records)}


def build_witness(seed: int = DEFAULT_SEED, *, family: str = "minimal") -> dict:
    """Build only an excluded finite fixture in memory; no actual bank admission."""
    require(type(seed) is int and 0 <= seed < 2**63, "Fixture seed must be an exact bounded integer")
    require(isinstance(family, str) and family in {"minimal", "all_regular"}, "Unsupported fixture family")
    rng = random.Random(seed)
    entities, fields = ["Q7MX", "R2VA", "S9KP", "T4NZ"], ["K8DS", "L3FH", "M6JT", "N5WR"]
    nuisance_entities = ["U1BC", "W0YG"]
    values = ["A1QZ", "B2RT", "C3SU", "D4TV", "E5UW", "F6VX", "G7WY", "H8XZ", "J9YZ", "K0ZA", "L1AB", "M2BC"]
    for symbols in (entities, fields, nuisance_entities, values):
        rng.shuffle(symbols)
    order, question_order = list(range(12)), list(itertools.product(entities, fields))
    rng.shuffle(order)
    rng.shuffle(question_order)
    contexts = []
    for index, matrix in enumerate(_matrices(family)):
        records = []
        for row, entity in enumerate(entities):
            occupied = [column for column, bit in enumerate(matrix[row]) if bit]
            for slot, column in enumerate(occupied):
                records.append({"entity": entity, "field": fields[column], "value": values[row * 2 + slot]})
        records.extend({"entity": nuisance_entities[column // 2], "field": field, "value": values[8 + column]}
                       for column, field in enumerate(fields))
        records = [records[i] for i in order]
        text = _render(records)
        queries = [{"query_id": f"fixture_q_{q:02d}", "entity": entity, "field": field,
                    "public_input": {"context_text": text, "request": _request(entity, field)},
                    "oracle_answer": _scan(records, entity, field)}
                   for q, (entity, field) in enumerate(question_order)]
        contexts.append({"context_id": f"fixture_ctx_{index:03d}", "records": records,
                         "context_text": text, "context_sha256": digest(text), "queries": queries})
    return signed({"schema": SCHEMA, "evidence_kind": "finite_mechanical_witness", "fixture_seed": seed,
                   "family": family, "entities": entities, "fields": fields, "nuisance_entities": nuisance_entities,
                   "values": values, "generator_policy": dict(_POLICY), "contexts": contexts})


def _bounds(groups, name):
    label_numerator = typed_numerator = total = 0
    conditional_bounds = []
    for outcomes in groups.values():
        n = len(outcomes)
        labels = Counter(value is not None for value in outcomes)
        typed = Counter(outcomes)
        require(labels[True] == labels[False] and n > 0, f"Not conditionally balanced for {name}")
        label_best, typed_best = max(labels.values()), max(typed.values())
        require(typed_best * 2 == n, f"Typed-answer bound for {name} is not exactly one half")
        label_numerator += label_best
        typed_numerator += typed_best
        total += n
        conditional_bounds.append({"denominator": n, "max_label_correct": label_best, "max_typed_correct": typed_best})
    return {"information_set": name, "groups": len(groups), "total_queries": total,
            "maximum_label_correct": label_numerator, "maximum_typed_correct": typed_numerator,
            "maximum_label_accuracy": label_numerator / total, "maximum_typed_accuracy": typed_numerator / total,
            "exact_fraction": "1/2", "conditional_group_sizes": dict(sorted(Counter(x["denominator"] for x in conditional_bounds).items())),
            "bound_scope": "this complete finite fixture population only"}


def _switch(records, entities, fields):
    matrix = _matrix(records, entities, fields)
    for i, j in itertools.combinations(range(4), 2):
        for a, b in itertools.permutations(range(4), 2):
            if matrix[i][a] == matrix[j][b] == 1 and matrix[i][b] == matrix[j][a] == 0:
                changed = [dict(r) for r in records]
                for record in changed:
                    if (record["entity"], record["field"]) == (entities[i], fields[a]):
                        record["field"] = fields[b]
                    elif (record["entity"], record["field"]) == (entities[j], fields[b]):
                        record["field"] = fields[a]
                return changed, [(entities[i], fields[a]), (entities[i], fields[b]),
                                 (entities[j], fields[a]), (entities[j], fields[b])]
    raise Blocked("No degree-preserving counterfactual relation switch exists")


def _operators(records, table, entities, fields, nuisance_entities):
    questions = list(itertools.product(entities, fields))
    profile = _profile(records)
    counts = Counter()
    for permuted in (list(reversed(records)), records[1:] + records[:1]):
        independent = _table(permuted)
        for e, f in questions:
            require(_scan(permuted, e, f) == _table_answer(table, e, f) == _table_answer(independent, e, f),
                    "Record permutation changed the unique lookup oracle")
            counts["record_permutation_query_checks"] += 1
    irrelevant = [dict(r) for r in records]
    nuisance = [r for r in irrelevant if r["entity"] in nuisance_entities]
    old_values = [r["value"] for r in nuisance]
    for record, value in zip(nuisance, old_values[1:] + old_values[:1]):
        record["value"] = value
    require(_profile(irrelevant) == profile, "Irrelevant perturbation changed nuisance profile")
    independent = _table(irrelevant)
    for e, f in questions:
        require(_scan(irrelevant, e, f) == _table_answer(table, e, f) == _table_answer(independent, e, f),
                "Irrelevant-record values changed a queried answer")
        counts["irrelevant_value_query_checks"] += 1
    for e in entities:
        changed = [dict(r) for r in records]
        selected = [r for r in changed if r["entity"] == e]
        require(len(selected) == 2, "Selected-value swap requires two unique facts")
        original = [(r["field"], r["value"]) for r in selected]
        selected[0]["value"], selected[1]["value"] = selected[1]["value"], selected[0]["value"]
        require(_profile(changed) == profile, "Selected-value swap changed nuisance inventory")
        independent = _table(changed)
        for entity, field in questions:
            expected = _table_answer(table, entity, field)
            if entity == e and field in dict(original):
                expected = {"v": original[1][1] if field == original[0][0] else original[0][1]}
            require(_scan(changed, entity, field) == expected == _table_answer(independent, entity, field),
                    "Oracle failed to follow the selected value independently")
            counts["selected_value_query_checks"] += 1
    switched, affected = _switch(records, entities, fields)
    require(_profile(switched) == profile, "Relation switch changed matched nuisance profile")
    independent = _table(switched)
    for e, f in questions:
        before, after = _table_answer(table, e, f), _table_answer(independent, e, f)
        require(_scan(switched, e, f) == after, "Independent switched lookup mismatch")
        require(((before["v"] is None) != (after["v"] is None)) == ((e, f) in affected),
                "Fixed-query counterfactual switch did not flip exactly its four cells")
        if (e, f) not in affected:
            require(before == after, "Relation switch changed an unrelated query")
        counts["relation_switch_query_checks"] += 1
    return counts


def _audit(witness):
    require(isinstance(witness, dict) and "sha256" in witness, "Signed finite witness required")
    verify(witness)
    body = {k: v for k, v in witness.items() if k != "sha256"}
    _fields(body, {"schema", "evidence_kind", "fixture_seed", "family", "entities", "fields", "nuisance_entities",
                   "values", "generator_policy", "contexts"}, "finite witness")
    require(body["schema"] == SCHEMA and body["evidence_kind"] == "finite_mechanical_witness",
            "Only excluded finite mechanical evidence is accepted")
    require(type(body["fixture_seed"]) is int and 0 <= body["fixture_seed"] < 2**63, "Invalid exact fixture seed")
    require(digest(body["generator_policy"]) == digest(_POLICY), "Fixture cannot authorize learning, admission or execution")
    entities, fields, nuisance_entities, values = [body[k] for k in ("entities", "fields", "nuisance_entities", "values")]
    for symbols, count, label in ((entities, 4, "entities"), (fields, 4, "fields"),
                                  (nuisance_entities, 2, "nuisance entities"), (values, 12, "values")):
        _symbols(symbols, count, label)
    require(len(set(entities + fields + nuisance_entities + values)) == 22, "Fixture symbol namespaces overlap")
    expected_matrices = _matrices(body["family"])
    contexts = body["contexts"]
    require(isinstance(contexts, list) and len(contexts) == len(expected_matrices), "Fixture context family incomplete")
    matrix_inventory, context_ids, fixed_profile, fixed_nuisance = [], set(), None, None
    groups = {name: defaultdict(list) for name in ("context_only", "entity_only_with_context", "field_only_with_context",
                                                  "query_only_without_context", "old_nuisance_count_shortcut")}
    operator_counts = Counter()
    all_questions = set(itertools.product(entities, fields))
    for context in contexts:
        _fields(context, {"context_id", "records", "context_text", "context_sha256", "queries"}, "context")
        cid = context["context_id"]
        require(isinstance(cid, str) and bool(cid) and cid not in context_ids, "Duplicate/invalid private context ID")
        context_ids.add(cid)
        records = context["records"]
        require(isinstance(records, list) and len(records) == 12, "Exactly 12 matched context records required")
        for record in records:
            _fields(record, {"entity", "field", "value"}, "record")
            require(record["entity"] in entities + nuisance_entities and record["field"] in fields
                    and isinstance(record["value"], str) and record["value"] in values, "Record symbols/types invalid")
        table = _table(records)
        matrix = _matrix(records, entities, fields)
        require(all(sum(row) == 2 for row in matrix)
                and all(sum(row[j] for row in matrix) == 2 for j in range(4)),
                "Per-context conditional entity/field degrees must each be two; global balance is insufficient")
        matrix_inventory.append(matrix)
        profile = _profile(records)
        require(profile["entity_frequencies"] == {e: 2 for e in entities + nuisance_entities}
                and profile["field_frequencies"] == {f: 3 for f in fields},
                "Every context must retain exact entity and field frequency profiles")
        require(profile["value_multiset"] == sorted(values), "Distinct full value inventory was not preserved")
        nuisance = sorted((r["entity"], r["field"], r["value"]) for r in records if r["entity"] in nuisance_entities)
        if fixed_profile is None:
            fixed_profile, fixed_nuisance = profile, nuisance
        require(profile == fixed_profile and nuisance == fixed_nuisance, "Context nuisance profile or irrelevant facts changed between graphs")
        require(context["context_text"] == _render(records) and context["context_sha256"] == digest(context["context_text"]),
                "Public context bytes/hash do not match the uniform records")
        queries = context["queries"]
        require(isinstance(queries, list) and len(queries) == 16, "Every unchanged context requires all 16 equal-weight queries")
        seen_pairs, seen_ids = set(), set()
        for query in queries:
            _fields(query, {"query_id", "entity", "field", "public_input", "oracle_answer"}, "query")
            e, f, qid = query["entity"], query["field"], query["query_id"]
            require((e, f) in all_questions and (e, f) not in seen_pairs, "Duplicate/omitted or invalid query pair")
            require(isinstance(qid, str) and bool(qid) and qid not in seen_ids, "Duplicate/invalid private query ID")
            seen_pairs.add((e, f))
            seen_ids.add(qid)
            _fields(query["public_input"], {"context_text", "request"}, "public input projection")
            require(query["public_input"]["context_text"] == context["context_text"], "Changing context per query reintroduces a query-blind shortcut")
            require(query["public_input"]["request"] == _request(e, f), "Public request changed or leaked role/ID/label metadata")
            answer = query["oracle_answer"]
            _fields(answer, {"v"}, "single-field oracle")
            require(answer["v"] is None or type(answer["v"]) is str, "Oracle must be exact string or null; no numeric/bool coercion")
            expected = _table_answer(table, e, f)
            require(digest(answer) == digest(expected) == digest(_scan(records, e, f)), "Independent table/brute oracle or supplied answer mismatch")
            outcome = answer["v"]
            groups["context_only"][cid].append(outcome)
            groups["entity_only_with_context"][(cid, e)].append(outcome)
            groups["field_only_with_context"][(cid, f)].append(outcome)
            groups["query_only_without_context"][(e, f)].append(outcome)
            groups["old_nuisance_count_shortcut"][digest(profile)].append(outcome)
        require(seen_pairs == all_questions, "Complete conditional query coverage missing")
        operator_counts.update(_operators(records, table, entities, fields, nuisance_entities))
    require(Counter(matrix_inventory) == Counter(expected_matrices), "Exact regular graph family changed or duplicated")
    require(all(_complement(m) in matrix_inventory for m in matrix_inventory), "Counterfactual complement coverage missing")
    certificates = {name: _bounds(group, name) for name, group in groups.items()}
    return signed({"schema": SCHEMA, "status": "PASS_FINITE_MECHANICAL_QUERY_DEPENDENCE_WITNESS",
                   "witness_sha256": witness["sha256"], "evidence_kind": "finite_mechanical_witness",
                   "family": body["family"], "contexts": len(contexts), "queries": len(contexts) * 16,
                   "oracle_agreement": "exact independent table and brute record scan", "matched_nuisance_profile": fixed_profile,
                   "conditional_chance_certificates": certificates,
                   "operator_invariances": {"scope": "oracle properties only; no model invariance established", **dict(operator_counts)},
                   "token_length_invariance_established": False, **_POLICY,
                   "warnings": ["Private context/query IDs, oracle answers and grouping metadata must never enter predictor input.",
                                "Presentation order/index, carry-over state and role metadata are potential side channels; no native execution is admitted.",
                                "Repeated queries, permutations and complements are dependent finite witness cases, not independent experimental samples.",
                                "The minimal graphs allow structured pattern recognition; even the 90-graph family is not model-quality evidence."]})


def audit_witness(witness: dict) -> dict:
    """Certify only this signed finite fixture; never read/write a bank or model."""
    try:
        return _audit(witness)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, Blocked):
            raise
        raise Blocked(f"Invalid/incomplete finite witness: {exc}") from exc
