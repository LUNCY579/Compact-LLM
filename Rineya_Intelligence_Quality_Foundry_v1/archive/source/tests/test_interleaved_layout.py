"""EXPOSED_MECHANICS_ONLY: layout fixtures do not generate or admit dataset rows."""
import ast
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import threading

from mechanisms.distributed_dossier_v1.interleaved_layout import (
    LAYOUT_VERSION, LayoutError, parse_interleaved, render_interleaved, solve_interleaved,
    export_interleaved_public_packet,
)
from mechanisms.distributed_dossier_v1 import interleaved_layout as layout
from mechanisms.distributed_dossier_v1.independent_solver import SchemaError, solve_parsed_public
from mechanisms.distributed_dossier_v1.public_parser import PublicParseError, parse_public
from tests.test_public_parser import _a, _b, _c
from mechanisms.distributed_dossier_v1 import renderer
from tests.test_schema_renderer import mechanics_world

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "mechanisms/distributed_dossier_v1"


def handwritten_layout(messages, order=None):
    """Test-only text assembly, independent of the author layout implementation."""
    text = messages[1]["content"]
    title, remaining = text.split("\n", 1)
    guide, remaining = remaining.split("\n\n", 1)
    task, remaining = remaining.split("\n\n", 1)
    references, source = remaining.split("Documents\n", 1)
    source, question = source.rsplit("Question\n", 1)
    documents = []
    while source:
        header = source.split("\n", 1)[0]
        did = header.split(":", 1)[0].split()[1]
        ending = "End document " + did + ".\n\n"
        document, source = source.split(ending, 1)
        documents.append(document + ending)
    if order is not None:
        documents = [documents[i] for i in order]
    ref = (guide + "\n\n" + references).splitlines(keepends=True)
    total = sum(len(line.encode()) for line in ref)
    bands, cumulative = [], 0
    for index in range(len(documents) - 1):
        band = []
        while ref:
            line = ref.pop(0)
            band.append(line)
            cumulative += len(line.encode())
            if index < len(documents) - 2 and cumulative * (len(documents) - 1) >= total * (index + 1):
                break
        bands.append("".join(band))
    result = title + "\nPublic layout: " + LAYOUT_VERSION + ".\n" + task + "\n\nDocuments\n"
    for index, document in enumerate(documents):
        result += document
        if index < len(bands):
            result += f"Begin reference block {index+1}.\n" + bands[index] + f"End reference block {index+1}.\n"
    result += "Question\n" + question
    return [copy.deepcopy(messages[0]), dict(role="user", content=result)]


def reconstruct(messages, parsed):
    raw = messages[1]["content"].encode()
    transport = parsed["normalization"]["transport"]
    return [copy.deepcopy(messages[0]), dict(role="user", content=b"".join(
        raw[item["actual_span"]["utf8_start"]:item["actual_span"]["utf8_end"]]
        for item in transport).decode())]


def slice_span(messages, span):
    return messages[span["message_index"]]["content"].encode()[span["utf8_start"]:span["utf8_end"]]


@contextmanager
def simulated_review_pin():
    """Temporary pin/review fixtures; never write the actual freeze or a registry."""
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        review_path, pin_path = out / "REVIEW.json", out / "receipt.json"
        identities = dict(code_identity={"interleaved_layout.py": hashlib.sha256((MODULE / "interleaved_layout.py").read_bytes()).hexdigest()},
                          test_identity_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          design_sha256=hashlib.sha256(layout._DESIGN_PATH.read_bytes()).hexdigest(),
                          excluded_derivative_supplement_sha256=hashlib.sha256(layout._SUPPLEMENT_PATH.read_bytes()).hexdigest(),
                          base_freeze_receipt_sha256=hashlib.sha256((MODULE / "catalogue_freeze_receipt.json").read_bytes()).hexdigest(),
                          base_code_identity={name: hashlib.sha256((MODULE / name).read_bytes()).hexdigest() for name in layout._BASE_FILES})
        review = dict(schema_version=1, layout_version=LAYOUT_VERSION, status="PASS_INDEPENDENT_RESEARCH_LAYOUT_REVIEW",
                      unresolved_critical_or_major_findings=[], **identities)
        review_path.write_text(json.dumps(review), encoding="utf-8")
        receipt = dict(schema_version=1, layout_version=LAYOUT_VERSION, status="INDEPENDENTLY_REVIEWED_RESEARCH_LAYOUT",
                       review_path=layout._REVIEW_RELATIVE,
                       independent_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(), **identities)
        pin_path.write_text(json.dumps(receipt), encoding="utf-8")
        with patch.object(layout, "_PIN_PATH", pin_path), patch.object(layout, "_REVIEW_PATH", review_path):
            yield pin_path, review_path, receipt, review


def simulated_production_base():
    """Existing exposed helper bytes with an in-process simulated producer capability."""
    base = renderer.render_public(mechanics_world("commitment_ledger"), None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
    base.update(mode="production", export_allowed=True, reservation_receipt_sha256="1" * 64)
    base._production_parent_id = "ExposedSimulation"
    base._verified_reservation_sha256 = "1" * 64
    return base


class InterleavedLayoutTests(unittest.TestCase):
    def test_handwritten_all_mechanisms_all_prose_and_table_variants(self):
        for source in (_a, _b, _c):
            for variant in (0, 1):
                for table in (False, True):
                    with self.subTest(source=source.__name__, variant=variant, table=table):
                        canonical = source(variant=variant, table=table)
                        messages = handwritten_layout(canonical)
                        parsed = parse_interleaved(messages)
                        self.assertEqual(reconstruct(messages, parsed), canonical)
                        base = parse_public(canonical)
                        self.assertEqual(parsed["world"], base["world"])
                        for field in ("record_spans", "atom_spans", "framing_spans", "document_spans", "source_spans"):
                            self.assertEqual(len(parsed[field]), len(base[field]))
                            for observed, original in zip(parsed[field], base[field]):
                                self.assertEqual({k: v for k, v in observed.items() if k != "span"},
                                                 {k: v for k, v in original.items() if k != "span"})
                                self.assertEqual(slice_span(messages, observed["span"]), slice_span(canonical, original["span"]))
                        self.assertEqual(parsed["unparsed_assertions"], [])
                        self.assertFalse(parsed["admission_ready"])

    def test_full_dossier_geometry_is_unchanged_by_answers(self):
        for source in (_a, _b, _c):
            messages = handwritten_layout(source())
            parsed = parse_interleaved(messages)
            dossier = parsed["dossier_span"]
            self.assertEqual(dossier["utf8_start"], 0)
            self.assertEqual(slice_span(messages, dossier), messages[1]["content"].split("Question\n")[0].encode())
            midpoints = [(item["span"]["utf8_start"] + item["span"]["utf8_end"]) / 2 / dossier["utf8_end"]
                         for item in parsed["source_spans"]]
            self.assertGreaterEqual(len({min(3, int(point * 4)) for point in midpoints}), 3)
            self.assertGreaterEqual(max(midpoints) - min(midpoints), .6)
            mutated = copy.deepcopy(messages)
            mutated[1]["content"] = mutated[1]["content"].replace("with 4 units", "with 5 units")
            changed = parse_interleaved(mutated)
            self.assertEqual(parsed["layout_version"], changed["layout_version"])
            self.assertEqual(parsed["layout_metadata_spans"], changed["layout_metadata_spans"])

    def test_renderer_preserves_facts_source_membership_leg_tick_and_metadata(self):
        for mechanism in ("commitment_ledger", "guarded_transition_learning", "policy_feasibility"):
            world = mechanics_world(mechanism)
            base = renderer.render_public(world, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
            for order in (None, ["D2", "D0", "D1"]):
                rendered = render_interleaved(base, document_order=order)
                parsed = parse_interleaved(rendered["messages"])
                expected = parse_public(base["messages"])["world"]
                for key in ("entities", "variables", "framework", "query", "format", "mechanism"):
                    self.assertEqual(parsed["world"][key], expected[key])
                self.assertEqual({r["record_id"]: r for r in parsed["world"]["records"]},
                                 {r["record_id"]: r for r in expected["records"]})
                self.assertEqual({d["document_id"]: d for d in parsed["world"]["documents"]},
                                 {d["document_id"]: d for d in expected["documents"]})
                self.assertFalse(rendered["export_allowed"])
                with self.assertRaises(renderer.RenderError):
                    renderer.export_public_packet(rendered, row_id="R", parent_id="P", view_id="V", revision=1)

    def test_four_complete_documents_occupy_all_full_dossier_quartiles(self):
        world = mechanics_world("commitment_ledger")
        for index, record in enumerate(world["records"]):
            record["document_id"] = "D" + str(index % 4)
        world["documents"] = [dict(document_id="D" + str(index), purpose="maintenance_log", title="Workshop notes " + name,
                                    dated_day=index, style="controlled_prose", record_ids=[r["record_id"] for r in world["records"] if r["document_id"] == "D" + str(index)])
                              for index, name in enumerate(("One", "Two", "Three", "Four"))]
        base = renderer.render_public(world, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
        parsed = parse_interleaved(render_interleaved(base)["messages"])
        length = parsed["dossier_span"]["utf8_end"]
        midpoints = [(s["span"]["utf8_start"] + s["span"]["utf8_end"]) / 2 / length for s in parsed["source_spans"]]
        self.assertEqual({int(m * 4) for m in midpoints}, {0, 1, 2, 3})
        self.assertGreater(max(midpoints) - min(midpoints), .7)

    def test_independent_solver_bridge_reports_actual_extension_and_structural_core(self):
        for source in (_a, _b, _c):
            canonical = source()
            messages = handwritten_layout(canonical)
            parsed = parse_interleaved(messages)
            observed = solve_interleaved(parsed, messages=messages)
            expected = solve_parsed_public(parse_public(canonical), messages=canonical, require_source_bytes=True)
            for key in ("status", "possible_answers", "compatible_world_count", "enumeration"):
                self.assertEqual(observed[key], expected[key])
            self.assertFalse(observed["public_grounding"]["source_bytes_verified"])
            self.assertFalse(observed["public_grounding"]["admission_ready"])
            self.assertTrue(observed["layout_grounding"]["source_bytes_verified"])
            self.assertFalse(observed["layout_grounding"]["admission_ready"])
            self.assertEqual(observed["public_messages_sha256"], parsed["messages_sha256"])
            self.assertNotEqual(parsed["messages_sha256"], parsed["normalization"]["messages_sha256"])

    def test_every_actual_byte_covered_and_utf8_spans_preserved(self):
        canonical = _a(titles=["Café notes", "Résumé notes", "Workshop Ω records"])
        messages = handwritten_layout(canonical)
        parsed = parse_interleaved(messages)
        for mi in (0, 1):
            spans = [item["span"] for item in parsed["coverage_spans"] if item["span"]["message_index"] == mi]
            cursor = 0
            for span in spans:
                self.assertEqual(span["utf8_start"], cursor)
                slice_span(messages, span).decode("utf-8")
                cursor = span["utf8_end"]
            self.assertEqual(cursor, len(messages[mi]["content"].encode()))
        for field in ("atom_spans", "framing_spans", "source_spans"):
            for item in parsed[field]:
                slice_span(messages, item["span"]).decode("utf-8")

    def test_three_model_output_classes_for_all_three_mechanisms(self):
        # Positive byte fixtures use exposed helper clauses only. They are neither
        # prospective worlds nor generated datasets or model admission evidence.
        cases = []
        for factory in (_a, _b, _c):
            cases.append((factory.__name__, "DETERMINATE", factory()))
        unknown_a = _a(variables=['Unreported U, a Quantity, is one of [3, 4].'])
        unknown_a[1]["content"] = unknown_a[1]["content"].replace(
            '[R6] Event E3, effective day 4, closes commitment "Cora" after E2; signer "Welo".\n', "").replace(
            '"Ceri" for "Soma" at 3 units', '"Ceri" for "Soma" at {unreported U} units')
        unknown_b = _b()
        unknown_b[1]["content"] = unknown_b[1]["content"].replace(
            '[R4] At tick 2, copy register x of "Qora" into register m of "Qora".\n', "").replace(
            '[R5] At tick 3, duplicate station "Qora" into fresh stations ["Vela", "Zeri"].\n', "").replace(
            '[R6] At tick 4, merge stations ["Vela", "Zeri"] by registerwise XOR into fresh station "Juno".\n', "").replace(
            'for stations ["Juno"]', 'for stations ["Qora"]')
        unknown_c = _c(variables=['Unreported U, a SmallCount, is one of [1, 2].'])
        unknown_c[1]["content"] = unknown_c[1]["content"].replace(
            'holds at most 2 people', 'holds at most {unreported U} people')
        for factory, messages in zip((_a, _b, _c), (unknown_a, unknown_b, unknown_c)):
            cases.append((factory.__name__, "UNDERDETERMINED", messages))
        conflict_a = _a()
        conflict_a[1]["content"] = conflict_a[1]["content"].replace(
            "End document D0.", '[RX] Event E0, effective day 1, starts commitment "Cora" for "Sera" with 5 units; signer "Welo".\nEnd document D0.')
        conflict_b = _b()
        conflict_b[1]["content"] = conflict_b[1]["content"].replace(
            'to (m=0, x=1, y=0)', 'to (m=1, x=1, y=0)')
        conflict_c = _c()
        conflict_c[1]["content"] = conflict_c[1]["content"].replace(
            "End document D0.", '[RC] Room "Elm" holds at most 3 people.\nEnd document D0.')
        for factory, messages in zip((_a, _b, _c), (conflict_a, conflict_b, conflict_c)):
            cases.append((factory.__name__, "INCONSISTENT", messages))
        for mechanism, status, canonical in cases:
            with self.subTest(mechanism=mechanism, status=status):
                messages = handwritten_layout(canonical)
                parsed = parse_interleaved(messages)
                solution = solve_interleaved(parsed, messages=messages)
                self.assertEqual(solution["status"], status)
                self.assertTrue(solution["layout_grounding"]["source_bytes_verified"])
                self.assertFalse(solution["layout_grounding"]["admission_ready"])

    def test_unknown_assertions_unknown_blocks_drop_duplicate_and_bad_sequence_rejected(self):
        messages = handwritten_layout(_a())
        text = messages[1]["content"]
        mutations = [
            text.replace("Begin reference block 1.", "Begin reference block 9."),
            text.replace("Public layout: " + LAYOUT_VERSION, "Public layout: oracle_DETERMINATE"),
            text.replace("Reading guide\n", "Reading guide\nThe answer is five.\n"),
            text.replace("End reference block 1.\n", "End reference block 1.\nUnknown block.\n"),
            text.replace('[R0] For sponsor, label "Soma" refers to the same entity as label "Sera".\n', ""),
            text.replace('[R1] "Sera" belongs', '[R0] "Sera" belongs'),
            text.replace("End document D0.\n", "End document D0.\n\nEnd document D0.\n"),
            text + "\n",
        ]
        # Dropping a record is valid source-edit semantics, not detectable from
        # bytes alone. The frozen blueprint bounds may allow it; solve bridge
        # rejects stale parsed maps rather than treating it as an original source.
        drop = mutations.pop(4)
        stale = parse_interleaved(messages)
        changed = [copy.deepcopy(messages[0]), dict(role="user", content=drop)]
        try:
            parse_interleaved(changed)
        except PublicParseError:
            pass
        with self.assertRaises((PublicParseError, SchemaError)):
            solve_interleaved(stale, messages=changed)
        for mutated in mutations:
            with self.subTest(mutated=mutated[:80]):
                with self.assertRaises(PublicParseError):
                    parse_interleaved([copy.deepcopy(messages[0]), dict(role="user", content=mutated)])

    def test_forged_world_maps_hashes_and_metadata_rejected(self):
        messages = handwritten_layout(_a())
        parsed = parse_interleaved(messages)
        for mutation in (lambda p: p["world"]["records"][3].update(quantity=11),
                         lambda p: p.update(messages_sha256="0" * 64),
                         lambda p: p["atom_spans"][0]["span"].update(utf8_start=1),
                         lambda p: p["coverage_spans"].pop(),
                         lambda p: p["source_spans"].pop(),
                         lambda p: p.update(admission_ready=True),
                         lambda p: p.update(layout_version="answer_five"),
                         lambda p: p["normalization"].update(messages_sha256="0" * 64),
                         lambda p: p["layout_metadata_spans"].clear()):
            forged = copy.deepcopy(parsed)
            mutation(forged)
            with self.assertRaises(SchemaError):
                solve_interleaved(forged, messages=messages)
        edited = copy.deepcopy(messages)
        edited[1]["content"] = edited[1]["content"].replace("with 4 units", "with 5 units")
        forged = copy.deepcopy(parsed)
        forged["messages_sha256"] = hashlib.sha256(json.dumps(edited, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.assertRaises(SchemaError):
            solve_interleaved(forged, messages=edited)

    def test_permutation_bad_order_tampered_base_and_review_fail_closed(self):
        base = renderer.render_public(mechanics_world("commitment_ledger"), None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
        for order in (["D0", "D0", "D2"], ["D0", "D1"], ["D0", "D1", "DX"], ["D0", "D1", {}]):
            with self.assertRaises(LayoutError):
                render_interleaved(base, document_order=order)
        with self.assertRaises(LayoutError):
            render_interleaved(dict(base))
        class Forged(dict):
            pass
        forged = Forged(base)
        forged._verified_messages_sha256 = base["messages_sha256"]
        with self.assertRaises(LayoutError):
            render_interleaved(forged)
        changed = copy.deepcopy(base)
        changed["messages"][1]["content"] += "\n"
        with self.assertRaises(LayoutError):
            render_interleaved(changed)
        with tempfile.TemporaryDirectory() as directory, patch.object(layout, "_PIN_PATH", Path(directory) / "absent.json"):
            with self.assertRaises(LayoutError):
                parse_interleaved(handwritten_layout(_a()), require_reviewed=True)
        with self.assertRaises(SchemaError):
            solve_interleaved(parse_interleaved(handwritten_layout(_a())), messages=handwritten_layout(_a()), require_source_bytes=False)

    def test_public_parser_path_does_not_import_author_modules_or_call_renderer(self):
        tree = ast.parse((MODULE / "interleaved_layout.py").read_text(encoding="utf-8"))
        author_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "render_interleaved"]
        for node in tree.body:
            if node in author_functions:
                continue
            imports = [item.module or "" for item in ast.walk(node) if isinstance(item, ast.ImportFrom)]
            self.assertTrue(all(not any(author in module for author in ("schema", "renderer", "world_constructor", "generator")) for module in imports))
        parser = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_interleaved")
        calls = [node.func.id for node in ast.walk(parser) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn("render_interleaved", calls)

    def test_reviewed_research_layout_positive_and_export_binds_identity(self):
        with simulated_review_pin():
            base = simulated_production_base()
            rendered = render_interleaved(base, require_reviewed=True)
            parsed = parse_interleaved(rendered["messages"], require_reviewed=True)
            self.assertTrue(parsed["admission_ready"])
            self.assertFalse(parsed["model_admission"])
            solution = solve_interleaved(parsed, messages=rendered["messages"], require_reviewed=True)
            self.assertFalse(solution["public_grounding"]["source_bytes_verified"])
            self.assertFalse(solution["public_grounding"]["admission_ready"])
            self.assertTrue(solution["layout_grounding"]["source_bytes_verified"])
            self.assertTrue(solution["layout_grounding"]["admission_ready"])
            self.assertEqual(solution["layout_grounding"]["readiness_scope"], "RESEARCH_CANDIDATE_ROW_LAYOUT_ONLY")
            self.assertFalse(solution["layout_grounding"]["model_admission"])
            kwargs = dict(row_id="ExposedRow", parent_id="ExposedSimulation", view_id="ExposedView", revision=1)
            packet = export_interleaved_public_packet(rendered, **kwargs)
            self.assertEqual(export_interleaved_public_packet(rendered, **kwargs), packet)
            self.assertEqual(packet["messages"], rendered["messages"])
            self.assertEqual(packet["scope"], "RESEARCH_CANDIDATE")
            self.assertFalse(packet["model_admission"])
            self.assertFalse(packet["training_authorized"])
            for field, value in (("row_id", "OtherRow"), ("parent_id", "OtherParent"), ("view_id", "OtherView"), ("revision", 2)):
                with self.assertRaises(LayoutError):
                    export_interleaved_public_packet(rendered, **{**kwargs, field: value})

    def test_reviewed_export_rejects_copies_mutations_and_evidence_artifacts(self):
        with simulated_review_pin():
            kwargs = dict(row_id="ExposedRow", parent_id="ExposedSimulation", view_id="ExposedView", revision=1)
            rendered = render_interleaved(simulated_production_base(), require_reviewed=True)
            with self.assertRaises(LayoutError):
                export_interleaved_public_packet(dict(rendered), **kwargs)
            for cloned in (copy.copy(rendered), copy.deepcopy(rendered)):
                with self.assertRaises(LayoutError):
                    export_interleaved_public_packet(cloned, **kwargs)
            private_changed = render_interleaved(simulated_production_base(), require_reviewed=True)
            private_changed._production_parent_id = "ChangedParent"
            with self.assertRaises(LayoutError):
                export_interleaved_public_packet(private_changed, **{**kwargs, "parent_id": "ChangedParent"})
            for mutation in (lambda r: r.update(export_allowed=False), lambda r: r.update(messages_sha256="0" * 64),
                             lambda r: r.update(reservation_receipt_sha256="0" * 64), lambda r: r.update(model_admission=True),
                             lambda r: r.update(training_authorized=True), lambda r: r.update(unknown_metadata="oracle"),
                             lambda r: r["atom_spans"].pop(), lambda r: r["messages"][1].update(content=r["messages"][1]["content"] + "\n")):
                changed = render_interleaved(simulated_production_base(), require_reviewed=True)
                mutation(changed)
                with self.assertRaises(LayoutError):
                    export_interleaved_public_packet(changed, **kwargs)
            export_interleaved_public_packet(rendered, **kwargs)
            rendered._export_identity = None
            with self.assertRaises(LayoutError):
                export_interleaved_public_packet(rendered, **{**kwargs, "revision": 2})
            late_changed = render_interleaved(simulated_production_base(), require_reviewed=True)
            parse = layout.parse_interleaved
            def alter_after_reread(messages, *args, **options):
                fresh = parse(messages, *args, **options)
                messages[1]["content"] += "\n"
                return fresh
            with patch.object(layout, "parse_interleaved", alter_after_reread), self.assertRaises(LayoutError):
                export_interleaved_public_packet(late_changed, **kwargs)
            base = renderer.render_public(mechanics_world("commitment_ledger"), None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
            evidence = render_interleaved(base, require_reviewed=True)
            evidence.update(export_allowed=True, mode="production")
            with self.assertRaises(LayoutError):
                export_interleaved_public_packet(evidence, **kwargs)

    def test_runtime_pin_missing_drift_unknown_fields_and_review_failure_rejected(self):
        mutations = [lambda p: p.update(schema_version=True), lambda p: p.update(layout_version="answer_five"),
                     lambda p: p.update(status="SELF_REVIEWED"), lambda p: p.update(extra=True),
                     lambda p: p["code_identity"].update(interleaved_layout="0" * 64),
                     lambda p: p.update(test_identity_sha256="0" * 64), lambda p: p.update(design_sha256="0" * 64),
                     lambda p: p.update(base_freeze_receipt_sha256="0" * 64),
                     lambda p: p.update(excluded_derivative_supplement_sha256="0" * 64),
                     lambda p: p["base_code_identity"].update({"renderer.py": "0" * 64}),
                     lambda p: p.update(independent_review_sha256="0" * 64), lambda p: p.update(review_path="other.json")]
        for mutation in mutations:
            with simulated_review_pin() as (pin_path, _, receipt, _):
                mutation(receipt)
                pin_path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaises(LayoutError):
                    parse_interleaved(handwritten_layout(_a()), require_reviewed=True)
        for mutation in (lambda r: r.update(status="FAIL"), lambda r: r.update(unresolved_critical_or_major_findings=["finding"]),
                         lambda r: r.update(design_sha256="0" * 64), lambda r: r.update(schema_version=True)):
            with simulated_review_pin() as (pin_path, review_path, receipt, review):
                mutation(review)
                review_path.write_text(json.dumps(review), encoding="utf-8")
                receipt["independent_review_sha256"] = hashlib.sha256(review_path.read_bytes()).hexdigest()
                pin_path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaises(LayoutError):
                    parse_interleaved(handwritten_layout(_a()), require_reviewed=True)
        with simulated_review_pin() as (pin_path, _, _, _):
            pin_path.unlink()
            with self.assertRaises(LayoutError):
                render_interleaved(simulated_production_base(), require_reviewed=True)
        with simulated_review_pin() as (pin_path, _, receipt, _):
            rendered = render_interleaved(simulated_production_base(), require_reviewed=True)
            pin_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            with self.assertRaises(LayoutError):
                export_interleaved_public_packet(rendered, row_id="ExposedRow", parent_id="ExposedSimulation", view_id="ExposedView", revision=1)

    def test_production_render_rejects_unreviewed_or_changed_base_capability(self):
        base = simulated_production_base()
        with self.assertRaises(LayoutError):
            render_interleaved(base)
        with simulated_review_pin():
            for mutation in (lambda b: b.update(export_allowed=False), lambda b: b.update(reservation_receipt_sha256="0" * 64),
                             lambda b: b.update(messages_sha256="0" * 64), lambda b: setattr(b, "_production_parent_id", None),
                             lambda b: setattr(b, "_verified_reservation_sha256", None)):
                changed = copy.deepcopy(base)
                mutation(changed)
                with self.assertRaises(LayoutError):
                    render_interleaved(changed, require_reviewed=True)

    def test_concurrent_exports_bind_one_identity(self):
        with simulated_review_pin():
            rendered = render_interleaved(simulated_production_base(), require_reviewed=True)
            barrier, visited, visit_lock = threading.Barrier(2), set(), threading.Lock()
            verify = layout._verify_layout_pin
            def synchronize_first_pin_read():
                identifier = threading.get_ident()
                with visit_lock:
                    first = identifier not in visited
                    visited.add(identifier)
                if first:
                    barrier.wait(timeout=10)
                return verify()
            def attempt(index):
                try:
                    return export_interleaved_public_packet(rendered, row_id=f"ExposedRow{index}",
                        parent_id="ExposedSimulation", view_id=f"ExposedView{index}", revision=index + 1)
                except LayoutError as error:
                    return error
            with patch.object(layout, "_verify_layout_pin", synchronize_first_pin_read), ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(attempt, (0, 1)))
            self.assertEqual(sum(type(result) is dict for result in outcomes), 1)
            self.assertEqual(sum(isinstance(result, LayoutError) for result in outcomes), 1)


if __name__ == "__main__":
    unittest.main()
