"""Controlled-English author renderer; contains no final-query solver.

Only a complete typed world can be rendered. Production requires immutable
catalogue review and a verified prospective registry reservation. Development
artifacts are explicitly mechanics-only and cannot be exported as PublicPackets.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import string

try:
    from . import schema
except ImportError:  # Direct file loading for the isolated mechanics suite.
    _spec = importlib.util.spec_from_file_location("dossier_author_schema", Path(__file__).with_name("schema.py"))
    schema = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(schema)


class RenderError(ValueError):
    pass


class _Rendered(dict):
    """Carries an in-process production gate result; ordinary dicts cannot export."""


class Text:
    def __init__(self, text="", spans=()):
        self.text, self.spans = text, list(spans)

    def append(self, other):
        if type(other) is str:
            other = Text(other)
        offset = len(self.text)
        self.text += other.text
        self.spans.extend((p, a + offset, b + offset, v) for p, a, b, v in other.spans)
        return self


def _atom(text, path, value):
    return Text(text, [(path, 0, len(text), value)])


def _format(template, values):
    result = Text()
    for literal, field, spec, conversion in string.Formatter().parse(template):
        result.append(literal)
        if field is not None:
            if spec or conversion or field not in values:
                raise RenderError("unsupported catalogue template placeholder: " + str(field))
            result.append(values[field])
    return result


def _joined(items, separator=", ", brackets=True):
    out = Text("[" if brackets else "")
    for n, item in enumerate(items):
        if n:
            out.append(separator)
        out.append(item)
    return out.append("]" if brackets else "")


class _Author:
    def __init__(self, world, cat):
        self.world, self.cat = world, cat
        self.entities = {e["entity_id"]: e["public_label"] for e in world["entities"]}

    def scalar(self, value, path, entity=False):
        if type(value) is dict:
            text = "{unreported " + value["var"] + "}"
            # The whole variable reference is the value of the weakened field.
            return _atom(text, path, value)
        text = json.dumps(self.entities[value], ensure_ascii=False) if entity else str(value)
        return _atom(text, path, value)

    def refs(self, values, path):
        return _joined([self.scalar(v, f"{path}[{n}]", True) for n, v in enumerate(values)])

    def state(self, value, path):
        return _format(self.cat["nested_productions"]["state"], {r: self.scalar(value[r], path + "." + r) for r in "mxy"})

    def arguments(self, predicate, args, path):
        types = self.cat["predicates"][predicate]["arguments"]
        return _joined([self.scalar(v, f"{path}[{n}]", not typ.startswith("slot ")) for n, (v, typ) in enumerate(zip(args, types))])

    def program(self, program, path):
        values = {"program_id": self.scalar(program["program_id"], path + ".program_id"),
                  "guard": _atom(self.cat["guard_realizations"][program["guard"]], path + ".guard", program["guard"]),
                  "action": _atom(self.cat["action_realizations"][program["action"]], path + ".action", program["action"])}
        key = "program_always" if program["guard"] == "always" else "program_conditional"
        result = _format(self.cat["nested_productions"][key], values)
        if program["guard"] == "always":
            start = result.text.index("always")
            result.spans.append((path + ".guard", start, start + 6, "always"))
        return result

    def policy(self, predicate, path):
        code = predicate["code"]
        template = self.cat["policy_predicate_realizations"][code].replace("<credential>", "{credential}")
        result = _format(template, {"credential": self.scalar(predicate.get("credential"), path + ".credential", True)} if "credential" in predicate else {})
        result.spans.append((path + ".code", 0, len(result.text), code))
        return result

    def selector(self, selector, path):
        code = selector["kind"]
        template = self.cat["selector_realizations"][code]
        for field in ("plan", "leg-id", "group", "bundle"):
            template = template.replace("<" + field + ">", "{" + field.replace("-", "_") + "}")
        values = {key: self.scalar(value, path + "." + key, key != "leg_id") for key, value in selector.items() if key != "kind"}
        result = _format(template, values)
        result.spans.append((path + ".kind", 0, len(result.text), code))
        return result

    def condition(self, condition, path):
        return _format(self.cat["condition"], {
            "predicate": self.scalar(condition["predicate"], path + ".predicate"),
            "arguments": self.arguments(condition["predicate"], condition["arguments"], path + ".arguments"),
            "comparison": _atom(self.cat["comparison_realizations"][condition["comparison"]], path + ".comparison", condition["comparison"]),
            "value": self.scalar(condition["value"], path + ".value")})

    def leg(self, leg, path):
        values = {key: self.scalar(value, path + "." + key, key in ("room", "kit", "leader", "group")) for key, value in leg.items() if key != "people"}
        values["people"] = self.refs(leg["people"], path + ".people")
        return _format(self.cat["nested_productions"]["leg"], values)

    def field(self, kind, key, value, record, table=False):
        path = key
        if key in ("input", "output", "state"):
            return self.state(value, path)
        if key == "candidates":
            return _joined([self.program(v, f"{path}[{n}]") for n, v in enumerate(value)], " / ")
        if key == "children":
            return _joined([_format(self.cat["nested_productions"]["child"], {
                "commitment": self.scalar(v["commitment"], f"{path}[{n}].commitment", True),
                "owner": self.scalar(v["owner"], f"{path}[{n}].owner", True),
                "quantity": self.scalar(v["quantity"], f"{path}[{n}].quantity")}) for n, v in enumerate(value)], " / ")
        if key == "legs":
            return _joined([self.leg(v, f"{path}[{n}]") for n, v in enumerate(value)], " / ")
        if key in ("plans", "destinations", "sources"):
            return self.refs(value, path)
        if key == "arguments":
            return self.arguments(record["predicate"], value, path)
        if key == "selector":
            return self.selector(value, path)
        if key == "condition":
            return self.condition(value, path)
        if key == "predicate" and kind == "base_policy":
            return self.policy(value, path)
        if key == "scope":
            return _atom(self.cat["scope_realizations"][value], path, value)
        if key == "attribute":
            return _atom(self.cat["attribute_realizations"][value], path, value)
        if key == "operation":
            return _atom(self.cat["operation_realizations"][value], path, value)
        descriptor = self.cat["productions"][kind]["fields"][key]
        entity = "EntityId" in descriptor and not descriptor.startswith("typed ")
        return self.scalar(value, path, entity)

    def record(self, record, variant, table=False):
        kind = record["kind"]
        fields = self.cat["productions"][kind]["fields"]
        values = {key: self.field(kind, key, record[key], record, table) for key in fields}
        if table:
            header = self.cat["productions"][kind]["table_headers"][variant]
            row = _joined([Text("[" + record["record_id"] + "]"), _atom(kind, "kind", kind), *values.values()], " | ", False)
            result = Text(" | ".join(header) + "\n" + " | ".join(self.cat["productions"][kind]["table_separator"]) + "\n")
            return result.append(row)
        if kind in ("fact_bool", "fact_count"):
            predicate = record["predicate"]
            args = values["arguments"]
            argument_values = {f"arguments[{n}]": self.scalar(v, f"arguments[{n}]", not typ.startswith("slot ")) for n, (v, typ) in enumerate(zip(record["arguments"], self.cat["predicates"][predicate]["arguments"]))}
            fact_value = values["value"]
            if kind == "fact_bool" and type(record["value"]) is int:
                fact_value = _atom(self.cat["fact_bool_realizations"][str(record["value"])], "value", record["value"])
            clause = _format(self.cat["fact_templates"][predicate], {**argument_values, "value": fact_value})
            clause.spans.append(("predicate", 0, len(clause.text), predicate))
            values = {"fact_clause": clause}
        clause = _format(self.cat["productions"][kind]["variants"][variant], values)
        clause.spans.append(("kind", 0, len(clause.text), kind))
        return Text("[" + record["record_id"] + "] ").append(clause)


def _verify_catalogue_pin(cat, catalogue_id, supplied=None):
    path = Path(__file__).with_name("catalogue_freeze_receipt.json")
    if not path.is_file():
        raise RenderError("complete catalogue has no independent frozen receipt")
    receipt = schema.loads_json(path.read_bytes())
    required = cat["gate_receipt_schema"]["catalogue_freeze_receipt.json"]["exact_fields"]
    schema._closed(receipt, required, "catalogue freeze receipt")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise RenderError("unknown catalogue freeze receipt schema")
    if supplied is not None and schema.canonical_bytes(supplied) != schema.canonical_bytes(receipt):
        raise RenderError("caller catalogue receipt differs from pinned bytes")
    expected = hashlib.sha256(schema.CATALOGUE_PATH.read_bytes()).hexdigest()
    if receipt.get("catalogue_id") != catalogue_id or receipt.get("catalogue_sha256") != expected or receipt.get("status") != "INDEPENDENTLY_REVIEWED_FROZEN":
        raise RenderError("catalogue identity/status/hash mismatch")
    review_hash = receipt.get("independent_review_sha256")
    if type(review_hash) is not str or not schema.HASH.fullmatch(review_hash):
        raise RenderError("missing independent catalogue review identity")
    for key, filename in (("design_sha256", "DESIGN.md"), ("contract_sha256", "CONTRACT.json")):
        actual = hashlib.sha256((schema.BLUEPRINT / filename).read_bytes()).hexdigest()
        if receipt.get(key) != actual:
            raise RenderError("catalogue blueprint binding mismatch")
    pin = schema.loads_json((schema.BLUEPRINT / "PIN_RECEIPT.json").read_bytes())
    if (type(pin.get("schema_version")) is not int or pin["schema_version"] != 1 or
        pin.get("status") != "HASH_PINNED_REVIEWED_DESIGN_ONLY" or
        pin.get("design_id") != cat["contract_id"] or
        type(pin.get("files")) is not dict or set(pin["files"]) != {"DESIGN.md", "CONTRACT.json"}):
        raise RenderError("reviewed blueprint pin identity/schema is invalid")
    if pin["files"]["DESIGN.md"] != receipt["design_sha256"] or pin["files"]["CONTRACT.json"] != receipt["contract_sha256"]:
        raise RenderError("reviewed blueprint pin drift")
    fixed_review = "cycles/003-dossier-implementation/catalogue_review/REVIEW.json"
    if receipt["review_path"] != fixed_review:
        raise RenderError("catalogue independent review path must be the fixed reviewed artifact")
    review_path = schema.ROOT / fixed_review
    if not review_path.is_file() or hashlib.sha256(review_path.read_bytes()).hexdigest() != review_hash:
        raise RenderError("independent review absent or hash changed")
    review = schema.loads_json(review_path.read_bytes())
    if type(review.get("schema_version")) is not int or review["schema_version"] != 1 or review.get("status") != "PASS_INDEPENDENT_CATALOGUE_AND_IMPLEMENTATION_REVIEW" or review.get("unresolved_critical_or_major_findings") != []:
        raise RenderError("independent review has unresolved findings or wrong disposition")
    if any(review.get(k) != receipt[k] for k in ("design_sha256", "contract_sha256")):
        raise RenderError("independent review blueprint binding mismatch")
    filenames = ["schema.py", "renderer.py", "public_parser.py", "independent_solver.py"]
    if type(receipt["code_identity"]) is not dict or set(receipt["code_identity"]) != set(filenames):
        raise RenderError("freeze receipt code identities incomplete")
    expected_producers = {}
    for filename in [*filenames, "catalogue.json"]:
        code_path = Path(__file__).with_name(filename)
        if not code_path.is_file(): raise RenderError("reviewed implementation file absent")
        expected_producers[filename] = hashlib.sha256(code_path.read_bytes()).hexdigest()
    if review.get("producer_identity") != expected_producers or any(receipt["code_identity"][f] != expected_producers[f] for f in filenames):
        raise RenderError("reviewed implementation identity drift")
    return receipt


def _verify_reservation(wrapper, world):
    schema._closed(wrapper, ["reservation", "adapter_receipt", "registry_path", "registry_snapshot_sha256"], "reservation wrapper")
    schema._str(wrapper["registry_path"], "registry_path")
    registry_path = Path(wrapper["registry_path"]).resolve()
    try:
        registry_path.relative_to(schema.ROOT.resolve())
    except ValueError as error:
        raise RenderError("dataset reservation registry is outside this foundry") from error
    reservation, adapter = wrapper["reservation"], wrapper["adapter_receipt"]
    schema.validate_reservation(reservation, world)
    contract = schema.loads_json((schema.BLUEPRINT / "CONTRACT.json").read_bytes())
    schema._closed(adapter, contract["types"]["RegistryAdapterReceipt"]["fields"], "adapter receipt")
    if schema.digest(adapter) != reservation["registry_adapter_receipt_sha256"]:
        raise RenderError("adapter receipt binding mismatch")
    mapping = {"author_parent_id": "parent_id", "semantic_parent_id": "semantic_parent_id", "author_structure_id": "structure_id", "structural_cluster_id": "structure_cluster_id", "content_family_id": "content_family_id", "source_domain_id": "source_domain_id", "role": "role", "author_alpha_semantic_sha256": "semantic_recipe_sha256", "author_alpha_structure_sha256": "normalized_structure_sha256"}
    if any(adapter[k] != reservation[v] for k, v in mapping.items()):
        raise RenderError("adapter/reservation identity mismatch")
    split = "DEV" if reservation["role"] == "EXPOSED_DEV" else "TRAIN"
    if adapter["registry_split"] != split or adapter["registration_mode"] != "PROSPECTIVE_PRE_GENERATION":
        raise RenderError("adapter role/mode mismatch")
    for key, value in adapter.items():
        if key.endswith("sha256"):
            schema._str(value, "adapter." + key, schema.HASH)
    for key in ("content_source_ids", "producer_source_ids"):
        schema._list(adapter[key], 1, 1000, key, unique=True, ordered=False)
        for item in adapter[key]: schema._str(item, key, schema.REGISTRY_ID)
    schema._integer(adapter["registry_snapshot_sequence"], 1, 2**63-1, "registry_snapshot_sequence")
    if wrapper["registry_snapshot_sha256"] != adapter["registry_snapshot_sha256"]:
        raise RenderError("named registry snapshot mismatch")
    # Use the generic registry's full append-only chain verifier. Read only.
    registry_spec = importlib.util.spec_from_file_location("dossier_generic_registry", schema.ROOT / "registry_tools/registry.py")
    registry = importlib.util.module_from_spec(registry_spec)
    registry_spec.loader.exec_module(registry)
    snapshot = registry.Registry(registry_path).snapshot()
    sequence = adapter["registry_snapshot_sequence"]
    filename = f"{sequence:08d}-{adapter['registry_snapshot_sha256']}.json"
    named_path = registry_path / "snapshots" / filename
    if not named_path.is_file() or hashlib.sha256(named_path.read_bytes()).hexdigest() != adapter["registry_snapshot_sha256"]:
        raise RenderError("reserved registry snapshot absent or changed")
    named = schema.loads_json(named_path.read_bytes())
    if named["sequence"] != reservation["reservation_sequence"] or named["previous_snapshot_sha256"] != reservation["registry_previous_sha256"]:
        raise RenderError("reservation snapshot sequence/chain mismatch")
    receipt_core = {"sequence": sequence, "snapshot_sha256": adapter["registry_snapshot_sha256"],
                    "snapshot_path": str(named_path.resolve()), "summary": named["summary"]}
    if reservation["registry_receipt_sha256"] != schema.digest(receipt_core):
        raise RenderError("canonical core registry registration receipt binding mismatch")
    parent = snapshot["state"]["parents"].get(adapter["semantic_parent_id"])
    if parent is None or parent["split"] != split or parent["registration_mode"] != "PROSPECTIVE_PRE_GENERATION" or parent["prospective"] is not True or parent["structural_cluster_id"] != adapter["structural_cluster_id"]:
        raise RenderError("prospective parent missing or role/structure drift")
    if world["parent_id"] not in parent["aliases"]:
        raise RenderError("author parent ID missing from registered aliases")
    if snapshot["state"]["parent_reserved_sequence"].get(adapter["semantic_parent_id"]) != sequence:
        raise RenderError("parent was not reserved in the named snapshot")
    spec = parent["specification"]
    if type(spec) is not dict or set(spec) != {"world", "source_cell_metadata"} or schema.canonical_bytes(spec["world"]) != schema.canonical_bytes(world):
        raise RenderError("registered unsolved world differs from render input")
    metadata = spec["source_cell_metadata"]
    metadata_fields = ["schema_version", "catalogue_id", "catalogue_sha256", "design_sha256", "contract_sha256", "allocated_cell", "content_family_id", "source_domain_id", "producer_family_id", "producer_version_sha256", "ancestry"]
    schema._closed(metadata, metadata_fields, "registered source/cell metadata")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise RenderError("unknown source/cell metadata schema")
    expected_binding = {"catalogue_id": schema.load_catalogue()["catalogue_version"],
                        "catalogue_sha256": hashlib.sha256(schema.CATALOGUE_PATH.read_bytes()).hexdigest(),
                        "design_sha256": hashlib.sha256((schema.BLUEPRINT / "DESIGN.md").read_bytes()).hexdigest(),
                        "contract_sha256": hashlib.sha256((schema.BLUEPRINT / "CONTRACT.json").read_bytes()).hexdigest(),
                        "content_family_id": world["content_family_id"], "source_domain_id": world["source_domain_id"],
                        "producer_family_id": world["producer_family_id"], "producer_version_sha256": reservation["producer_version_sha256"],
                        "ancestry": reservation["ancestry"]}
    if any(metadata[key] != value for key, value in expected_binding.items()):
        raise RenderError("registered source/cell catalogue/provenance binding mismatch")
    cell_fields = ["mechanism", "structure_id", "structure_index", "level", "case_index", "role", "format"]
    schema._closed(metadata["allocated_cell"], cell_fields, "allocated cell")
    if any(metadata["allocated_cell"][key] != reservation[key] for key in cell_fields):
        raise RenderError("registered allocation cell differs from reservation")
    # Equality does not coerce Boolean/float allocation numbers into integers.
    for key in ("structure_index", "level", "case_index"):
        if type(metadata["allocated_cell"][key]) is not int:
            raise RenderError("registered cell integer type changed")
    if parent["specification_sha256"] != adapter["registry_full_specification_sha256"] or parent["normalized_specification_sha256"] != adapter["registry_mechanical_normalized_sha256"]:
        raise RenderError("registry full/mechanical specification binding drift")
    if set(parent["source_ids"]) != set(adapter["content_source_ids"] + adapter["producer_source_ids"]):
        raise RenderError("registered source IDs mismatch")
    for source_id in adapter["content_source_ids"]:
        source = snapshot["state"]["sources"][source_id]
        if source["source_family_id"] != world["content_family_id"] or source["split_scope"] != "CONTENT_FAMILY" or source["content_sha256"] != reservation["content_family_sha256"]:
            raise RenderError("content source family/scope/hash mismatch")
    for source_id in adapter["producer_source_ids"]:
        source = snapshot["state"]["sources"][source_id]
        if source["source_family_id"] != world["producer_family_id"] or source["split_scope"] != "PRODUCER_ONLY" or source["content_sha256"] != reservation["producer_version_sha256"]:
            raise RenderError("producer source family/scope/hash mismatch")
    return reservation


def render_public(world, reservation, catalogue_id, *, mode="production", view_plan=None, catalogue_receipt=None):
    """Return public messages and UTF-8 source/record/leaf span maps, never target.

    view_plan only selects frozen variants and changes presentation order. It
    cannot alter entity labels, source metadata, semantic fields, or source IDs.
    """
    cat = schema.load_catalogue()
    if catalogue_id != cat["catalogue_version"]:
        raise RenderError("unknown catalogue identity")
    if mode not in ("production", "EXPOSED_MECHANICS_ONLY"):
        raise RenderError("unknown rendering mode")
    schema.validate_world(world, catalogue=cat)
    if mode == "production":
        _verify_catalogue_pin(cat, catalogue_id, catalogue_receipt)
        reserved = _verify_reservation(reservation, world)
    else:
        if reservation is not None or catalogue_receipt is not None:
            raise RenderError("mechanics fixtures cannot carry dataset reservation/export receipts")
        reserved = None
    plan = view_plan or {}
    allowed = {"document_order", "record_orders", "record_variants", "query_variant"}
    if type(plan) is not dict or set(plan) - allowed:
        raise RenderError("view plan changes unsupported fields")
    docs = {d["document_id"]: d for d in world["documents"]}
    records = {r["record_id"]: r for r in world["records"]}
    order = plan.get("document_order", list(docs))
    if type(order) is not list or len(order) != len(docs) or set(order) != set(docs):
        raise RenderError("document order is not a permutation")
    record_orders, variants = plan.get("record_orders", {}), plan.get("record_variants", {})
    if type(record_orders) is not dict or set(record_orders) - set(docs) or type(variants) is not dict or set(variants) - set(records):
        raise RenderError("unknown document or record in view plan")
    for variant in variants.values():
        if type(variant) is not int or variant not in (0, 1): raise RenderError("invalid frozen record variant")
    qv = plan.get("query_variant", 0)
    if type(qv) is not int or qv not in (0, 1): raise RenderError("invalid frozen query variant")
    author, envelope = _Author(world, cat), cat["envelope"]
    user = Text()
    framing, record_spans, atom_spans, source_spans = [], [], [], []
    def line(text, category=None):
        if type(text) is str: text = Text(text)
        if user.text: user.append("\n")
        start = len(user.text); user.append(text)
        if category:
            framing.extend((p, a + start, b + start, v) for p, a, b, v in text.spans)
        return start, len(user.text)
    line(envelope["title"]); line(envelope["guide_header"])
    for text in cat["reading_guide_lines"]: line(text)
    line("")
    line(_format(envelope["task"], {"mechanism": _atom(world["mechanism"], "mechanism", world["mechanism"]), "format": _atom(world["format"], "format", world["format"])}), True)
    line(""); line(envelope["vocabulary_header"])
    for n, entity in enumerate(world["entities"]):
        line(_format(envelope["entity"], {key: _atom(json.dumps(value, ensure_ascii=False) if key == "public_label" else value, f"entities[{n}].{key}", value) for key, value in entity.items()}), True)
    line(""); line(envelope["variable_header"])
    for n, variable in enumerate(world["variables"]):
        line(_format(envelope["variable"], {"variable_id": _atom(variable["variable_id"], f"variables[{n}].variable_id", variable["variable_id"]), "scalar_type": _atom(variable["scalar_type"], f"variables[{n}].scalar_type", variable["scalar_type"]), "choices": _joined([author.scalar(v, f"variables[{n}].choices[{i}]", variable["scalar_type"] == "EntityId") for i, v in enumerate(variable["choices"])])}), True)
    if not world["variables"]: line(envelope["no_variables"])
    line(""); line(envelope["framework_header"])
    for n, clause in enumerate(world["framework"]["rule_clauses"]):
        text = cat["framework_variants"][clause["code"]][clause["public_variant"]]
        rendered = _format(envelope["framework"], {"rule_id": _atom(clause["rule_id"], f"framework.rule_clauses[{n}].rule_id", clause["rule_id"]), "clause": _atom(text, f"framework.rule_clauses[{n}].code", clause["code"])})
        rendered.spans.append((f"framework.rule_clauses[{n}].public_variant", 0, len(rendered.text), clause["public_variant"]))
        line(rendered, True)
    line(""); line(envelope["documents_header"])
    incidental_bytes = 0
    for did in order:
        document = docs[did]
        index = world["documents"].index(document)
        header = _format(envelope["document"], {key: _atom(json.dumps(value, ensure_ascii=False) if key == "title" else str(value), f"documents[{index}].{key}", value) for key, value in document.items() if key != "record_ids"})
        doc_start, _ = line(header, True)
        record_order = record_orders.get(did, document["record_ids"])
        if type(record_order) is not list or len(record_order) != len(document["record_ids"]) or set(record_order) != set(document["record_ids"]):
            raise RenderError("record order changes source inventory")
        for rid in record_order:
            record = records[rid]
            rendered = author.record(record, variants.get(rid, 0), document["style"] == "small_table")
            start, end = line(rendered)
            record_spans.append((rid, start, end))
            atom_spans.extend((rid, path, a + start, b + start, value) for path, a, b, value in rendered.spans)
            if record["kind"] == "incidental": incidental_bytes += len(rendered.text.encode("utf-8"))
        _, doc_end = line(_format(envelope["end_document"], {"document_id": Text(did)}))
        source_spans.append((did, doc_start, doc_end)); line("")
    dossier_end = len(user.text)
    if incidental_bytes > len(user.text.encode("utf-8")) * cat["limits"]["incidental_dossier_bytes_fraction_max"]:
        raise RenderError("incidental dossier bytes exceed the frozen 15 percent limit")
    line(envelope["question_header"])
    query = world["query"]
    values = {}
    for key, value in query.items():
        path = "query." + key
        if key in ("candidate_ids", "final_stations"): values[key] = author.refs(value, path)
        elif key == "group_label": values[key] = author.scalar(value, path, True)
        elif key == "combine": values[key] = _atom(cat["combine_realizations"][value], path, value)
        else: values[key] = author.scalar(value, path)
    qtext = _format(cat["query_variants"][world["mechanism"]][qv], values)
    used = {p for p, _, _, _ in qtext.spans}
    for key in query:
        if "query." + key not in used and key not in ("candidate_ids", "final_stations"):
            qtext.spans.append(("query." + key, 0, len(qtext.text), query[key]))
    line(qtext, True); line(cat["answer_instructions"][world["format"]])
    def span(a, b):
        return {"message_index": 1, "utf8_start": len(user.text[:a].encode("utf-8")), "utf8_end": len(user.text[:b].encode("utf-8"))}
    messages = [{"role": "system", "content": cat["system_message"]}, {"role": "user", "content": user.text}]
    result = _Rendered({"messages": messages, "messages_sha256": schema.digest(messages), "catalogue_id": catalogue_id,
              "mode": mode, "export_allowed": mode == "production", "record_spans": [{"record_id": rid, "span": span(a, b)} for rid, a, b in record_spans],
              "atom_spans": [{"atom_id": f"A{n:04d}", "record_id": rid, "field_path": path, "span": span(a, b), "typed_value_sha256": schema.digest(value)} for n, (rid, path, a, b, value) in enumerate(atom_spans, 1)],
              "source_spans": [{"document_id": did, "span": span(a, b)} for did, a, b in source_spans],
              "framing_spans": [{"field_path": path, "span": span(a, b), "typed_value_sha256": schema.digest(value)} for path, a, b, value in framing],
              "dossier_span": span(0, dossier_end)})
    result._production_parent_id = reserved["parent_id"] if reserved is not None else None
    result._verified_messages_sha256 = result["messages_sha256"]
    result._verified_reservation_sha256 = schema.digest(reserved) if reserved is not None else None
    if reserved is not None: result["reservation_receipt_sha256"] = schema.digest(reserved)
    return result


def export_public_packet(rendered, *, row_id, parent_id, view_id, revision):
    """A development artifact cannot accidentally become a dataset packet."""
    if not isinstance(rendered, _Rendered) or rendered._production_parent_id is None or rendered.get("export_allowed") is not True or rendered.get("mode") != "production":
        raise RenderError("EXPOSED_MECHANICS_ONLY artifacts cannot be exported")
    if parent_id != rendered._production_parent_id or rendered._verified_messages_sha256 != schema.digest(rendered["messages"]) or rendered["messages_sha256"] != rendered._verified_messages_sha256:
        raise RenderError("export parent or public messages changed after production verification")
    if rendered["reservation_receipt_sha256"] != rendered._verified_reservation_sha256:
        raise RenderError("verified reservation receipt changed before export")
    for key, value in (("row_id", row_id), ("parent_id", parent_id), ("view_id", view_id)):
        schema._str(value, key, schema.ID)
    schema._integer(revision, 1, 2**63-1, "revision")
    return {"row_id": row_id, "parent_id": parent_id, "view_id": view_id, "revision": revision,
            "messages": rendered["messages"], "messages_sha256": rendered["messages_sha256"],
            "reservation_receipt_sha256": rendered["reservation_receipt_sha256"]}
