"""Versioned interleaved public text, independent transport parser and solver bridge.

This extension moves complete source documents between bands of
existing reference clauses. It neither constructs worlds nor alters the frozen
catalogue or implementation. The public parser path imports no author implementation.
New layout review/admission requires a separate pin; the old freeze is insufficient.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import weakref
import threading

from .public_parser import PublicParseError, parse_public

LAYOUT_VERSION = "rineya.distributed_dossier_interleaved.v2"
_LAYOUT_LINE = "Public layout: " + LAYOUT_VERSION + ".\n"
_OPEN = "Begin reference block {index}.\n"
_CLOSE = "End reference block {index}.\n"
_CATALOGUE = Path(__file__).with_name("catalogue.json")
_ROOT = Path(__file__).resolve().parents[2]
_PIN_PATH = Path(__file__).with_name("interleaved_layout_freeze_receipt.json")
_REVIEW_RELATIVE = "cycles/003-dossier-implementation/layout_revision/REVIEW.json"
_REVIEW_PATH = _ROOT / _REVIEW_RELATIVE
_DESIGN_PATH = _ROOT / "cycles/003-dossier-implementation/layout_revision/DESIGN.md"
_TEST_PATH = _ROOT / "tests/test_interleaved_layout.py"
_SUPPLEMENT_PATH = _ROOT / "cycles/003-dossier-implementation/layout_revision/EXCLUDED_DERIVATIVE_SUPPLEMENT.json"
_BASE_FILES = ("catalogue.json", "schema.py", "renderer.py", "public_parser.py", "independent_solver.py")
_MINTED_RENDERS = {}
_EXPORT_LOCK = threading.Lock()


class LayoutError(ValueError):
    pass


class _InterleavedRender(dict):
    """Deliberately incompatible with the frozen renderer's export capability."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _catalogue(path):
    return json.loads((Path(path) if path is not None else _CATALOGUE).read_bytes())


def _identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _span(start, end, message_index=1):
    return {"message_index": message_index, "utf8_start": start, "utf8_end": end}


def _validate_messages(messages):
    if type(messages) is not list or len(messages) != 2:
        raise PublicParseError("interleaved layout requires exactly two messages")
    for item, role in zip(messages, ("system", "user")):
        if (type(item) is not dict or set(item) != {"role", "content"}
                or item["role"] != role or type(item["content"]) is not str):
            raise PublicParseError("invalid actual layout message")
        value = item["content"]
        if (not value or "\r" in value or "\x00" in value
                or any(0xD800 <= ord(c) <= 0xDFFF for c in value)):
            raise PublicParseError("unsupported control or Unicode in layout")
        if len(value.encode("utf-8")) > 200000:
            raise PublicParseError("layout message byte bound")


def _json_file(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise LayoutError("duplicate key in independent layout pin/review")
            out[key] = value
        return out
    try:
        return json.loads(path.read_bytes(), object_pairs_hook=pairs)
    except (OSError, ValueError) as exc:
        raise LayoutError("independent layout pin/review absent or malformed") from exc


def _file_hash(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise LayoutError("independent layout pinned source absent") from exc


def _verify_layout_pin():
    """Only an independent new receipt can grant research-layout readiness."""
    receipt = _json_file(_PIN_PATH)
    fields = {"schema_version", "layout_version", "status", "code_identity", "test_identity_sha256",
              "design_sha256", "base_freeze_receipt_sha256", "base_code_identity", "review_path",
              "independent_review_sha256", "excluded_derivative_supplement_sha256"}
    if type(receipt) is not dict or set(receipt) != fields:
        raise LayoutError("closed independent layout freeze receipt fields")
    if (type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or receipt["layout_version"] != LAYOUT_VERSION
            or receipt["status"] != "INDEPENDENTLY_REVIEWED_RESEARCH_LAYOUT"
            or receipt["review_path"] != _REVIEW_RELATIVE):
        raise LayoutError("layout review receipt version/status/path mismatch")
    base = {name: _file_hash(Path(__file__).with_name(name)) for name in _BASE_FILES}
    base_receipt = _file_hash(Path(__file__).with_name("catalogue_freeze_receipt.json"))
    expected = {"code_identity": {"interleaved_layout.py": _identity()},
                "test_identity_sha256": _file_hash(_TEST_PATH),
                "design_sha256": _file_hash(_DESIGN_PATH),
                "excluded_derivative_supplement_sha256": _file_hash(_SUPPLEMENT_PATH),
                "base_code_identity": base, "base_freeze_receipt_sha256": base_receipt}
    if any(receipt[field] != value for field, value in expected.items()):
        raise LayoutError("independent layout code/test/design/base freeze drift")
    if receipt["independent_review_sha256"] != _file_hash(_REVIEW_PATH):
        raise LayoutError("independent layout review hash changed")
    review = _json_file(_REVIEW_PATH)
    if (type(review) is not dict or type(review.get("schema_version")) is not int
            or review["schema_version"] != 1 or review.get("layout_version") != LAYOUT_VERSION
            or review.get("status") != "PASS_INDEPENDENT_RESEARCH_LAYOUT_REVIEW"
            or review.get("unresolved_critical_or_major_findings") != []
            or any(review.get(field) != value for field, value in expected.items())):
        raise LayoutError("independent layout review disposition or identity mismatch")
    return {"receipt": receipt, "receipt_sha256": _file_hash(_PIN_PATH)}


def _review_policy(require_reviewed, catalogue_path=None):
    if type(require_reviewed) is not bool:
        raise LayoutError("require_reviewed must be Boolean")
    if require_reviewed:
        if catalogue_path is not None and Path(catalogue_path).resolve() != _CATALOGUE.resolve():
            raise LayoutError("reviewed layout requires the official pinned base catalogue")
        return _verify_layout_pin()
    return None


def _remap_span(span, transport):
    if span["message_index"] == 0:
        return copy.deepcopy(span)
    start, end = span["utf8_start"], span["utf8_end"]
    pieces = []
    cursor = start
    for block in transport:
        old, new = block["canonical_span"], block["actual_span"]
        left, right = max(start, old["utf8_start"]), min(end, old["utf8_end"])
        if left < right:
            if left != cursor:
                raise PublicParseError("non-total canonical transport")
            a = new["utf8_start"] + left - old["utf8_start"]
            b = a + right - left
            if pieces and pieces[-1][1] != a:
                raise PublicParseError("semantic span crosses reordered blocks")
            pieces.append((a, b))
            cursor = right
    if cursor != end or not pieces:
        raise PublicParseError("source span missing from canonical transport")
    return _span(pieces[0][0], pieces[-1][1])


def render_interleaved(verified_base_render, *, catalogue_path=None, document_order=None,
                       require_reviewed=False):
    """Reflow an in-process, hash-verified base render; separately gate research export.

    Only complete document permutation is configurable. Clauses, titles, dates,
    source membership, record order and all leg/tick ordering remain byte-identical.
    Production requires the independent new layout pin plus the genuine base
    production capability. This grants research-row layout eligibility only.
    """
    # Capability check belongs solely to this author path. parse_interleaved and
    # solve_interleaved never execute it or import the author renderer/schema.
    from .renderer import _Rendered
    if type(verified_base_render) is not _Rendered:
        raise LayoutError("expected verified base render")
    messages = verified_base_render.get("messages")
    _validate_messages(messages)
    digest = _digest(messages)
    if (getattr(verified_base_render, "_verified_messages_sha256", None) != digest
            or verified_base_render.get("messages_sha256") != digest):
        raise LayoutError("base render has no intact in-process verification")
    pin = _review_policy(require_reviewed, catalogue_path)
    production = verified_base_render.get("mode") == "production"
    if production:
        parent = getattr(verified_base_render, "_production_parent_id", None)
        reservation_hash = getattr(verified_base_render, "_verified_reservation_sha256", None)
        if (pin is None or verified_base_render.get("export_allowed") is not True
                or type(parent) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", parent) is None
                or type(reservation_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", reservation_hash) is None
                or verified_base_render.get("reservation_receipt_sha256") != reservation_hash):
            raise LayoutError("production layout requires pinned review and intact base parent/reservation capability")
    elif verified_base_render.get("mode") != "EXPOSED_MECHANICS_ONLY" or verified_base_render.get("export_allowed") is not False:
        raise LayoutError("base render mode/export flag mismatch")
    base = parse_public(messages, catalogue_path, require_frozen=bool(pin))
    cat = _catalogue(catalogue_path)
    raw = messages[1]["content"].encode("utf-8")
    env = cat["envelope"]
    # Author-side extraction uses verified source maps. Parser-side recognition
    # below reads block grammar from actual text and does not call this routine.
    title_end = raw.index(b"\n") + 1
    task_item = next(item for item in base["coverage_spans"] if item["category"] == "task")
    task_start, task_end = task_item["span"]["utf8_start"], task_item["span"]["utf8_end"] + 1
    heading_item = next(item for item in base["coverage_spans"] if item["category"] == "documents_heading")
    heading_start, heading_end = heading_item["span"]["utf8_start"], heading_item["span"]["utf8_end"]
    question_start = base["dossier_span"]["utf8_end"]
    documents = {item["document_id"]: (item["span"]["utf8_start"], item["span"]["utf8_end"] + 2)
                 for item in base["document_spans"]}
    order = document_order if document_order is not None else list(documents)
    if (type(order) is not list or any(type(did) is not str for did in order) or len(order) != len(documents)
            or len(set(order)) != len(order) or set(order) != set(documents)):
        raise LayoutError("document_order must be a complete unique permutation")
    if len(order) < 3:
        raise LayoutError("interleaving requires at least three source documents")
    reference_intervals = [(title_end, task_start), (task_end, heading_start)]
    reference_lines = []
    for a, b in reference_intervals:
        cursor = a
        for line in raw[a:b].splitlines(keepends=True):
            reference_lines.append((cursor, cursor + len(line)))
            cursor += len(line)
    bands = [[] for _ in range(len(order) - 1)]
    total = sum(b - a for a, b in reference_lines)
    running, band = 0, 0
    for a, b in reference_lines:
        bands[band].append((a, b))
        running += b - a
        if band < len(bands) - 1 and running * len(bands) >= total * (band + 1):
            band += 1
    if any(not item for item in bands):
        raise LayoutError("insufficient complete reference lines for source interleaving")
    output = bytearray()
    def move(a, b):
        output.extend(raw[a:b])
    move(0, title_end)
    output.extend(_LAYOUT_LINE.encode())
    move(task_start, task_end)
    move(heading_start, heading_end)
    for index, did in enumerate(order):
        move(*documents[did])
        if index < len(bands):
            output.extend(_OPEN.format(index=index + 1).encode())
            for a, b in bands[index]:
                move(a, b)
            output.extend(_CLOSE.format(index=index + 1).encode())
    move(question_start, len(raw))
    actual = [copy.deepcopy(messages[0]), {"role": "user", "content": output.decode("utf-8")}]
    parsed = parse_interleaved(actual, catalogue_path, require_reviewed=require_reviewed)
    result = _InterleavedRender({key: copy.deepcopy(value) for key, value in parsed.items()
                               if key != "world"})
    result.update(messages=actual, mode="production" if production else "EXPOSED_MECHANICS_ONLY",
                  export_allowed=production, base_messages_sha256=digest,
                  model_admission=False, training_authorized=False)
    result._verified_messages_sha256 = parsed["messages_sha256"]
    result._production_parent_id = verified_base_render._production_parent_id if production else None
    result._verified_base_messages_sha256 = digest
    result._verified_world_sha256 = _digest(parsed["world"])
    result._verified_layout_receipt_sha256 = pin["receipt_sha256"] if pin else None
    result._verified_reservation_sha256 = verified_base_render._verified_reservation_sha256 if production else None
    result._export_identity = None
    if production:
        result["reservation_receipt_sha256"] = result._verified_reservation_sha256
    result._verified_render_sha256 = _digest(result)
    if production:
        key = id(result)
        bindings = tuple((field, getattr(result, field)) for field in
                         ("_production_parent_id", "_verified_messages_sha256", "_verified_base_messages_sha256",
                          "_verified_world_sha256", "_verified_layout_receipt_sha256",
                          "_verified_reservation_sha256", "_verified_render_sha256"))
        _MINTED_RENDERS[key] = {"reference": weakref.ref(result, lambda ref, minted_key=key: _MINTED_RENDERS.pop(minted_key, None)),
                                "bindings": bindings, "export_identity": None}
    return result


def parse_interleaved(messages, catalogue_path=None, *, require_reviewed=False):
    """Independently parse actual v2 text, then transport every public UTF8 map.

    The canonical parser is used only for its frozen clause grammar. Its input is
    explicitly a normalization, never described as the actual presented source.
    Every actual byte is covered by either that grammar or a fixed layout literal.
    No author renderer/schema, sidecar world, answer or oracle metadata is an input.
    """
    pin = _review_policy(require_reviewed, catalogue_path)
    _validate_messages(messages)
    cat = _catalogue(catalogue_path)
    env = cat["envelope"]
    raw = messages[1]["content"].encode("utf-8")
    lines = raw.splitlines(keepends=True)
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    cursor, literals = 0, []
    def take():
        nonlocal cursor
        if cursor == len(lines):
            raise PublicParseError("truncated interleaved layout")
        a, line = starts[cursor], lines[cursor]
        cursor += 1
        return a, a + len(line), line
    def expect(text, category=None):
        a, b, line = take()
        if line != text.encode("utf-8"):
            raise PublicParseError("missing, altered or extra interleaved layout delimiter", message_index=1, utf8_start=a)
        if category:
            literals.append({"category": category, "span": _span(a, b)})
        return a, b
    title = expect(env["title"] + "\n")
    layout = expect(_LAYOUT_LINE, "layout_version")
    ta, tb, task_line = take()
    if not task_line.startswith(b"Task: ") or not task_line.endswith(b".\n"):
        raise PublicParseError("invalid interleaved task block")
    _, tb = expect("\n")
    task = ta, tb
    heading = expect(env["documents_header"] + "\n")
    docs, references = [], []
    index = 1
    while cursor < len(lines) and lines[cursor] != (env["question_header"] + "\n").encode():
        a, _, line = take()
        # Header meaning, all scalar bounds and title restrictions are separately
        # established by the grammar parser after normalization.
        match = re.match(rb"Document ([A-Za-z][A-Za-z0-9_-]{0,31}): ", line)
        if match is None:
            raise PublicParseError("expected complete source document", message_index=1, utf8_start=a)
        ending = env["end_document"].replace("{document_id}", match[1].decode()) + "\n"
        while cursor < len(lines) and lines[cursor] != ending.encode():
            take()
        expect(ending)
        _, b = expect("\n")
        docs.append((a, b))
        if cursor < len(lines) and lines[cursor] == (env["question_header"] + "\n").encode():
            break
        expect(_OPEN.format(index=index), "reference_block_open")
        ref_start = starts[cursor] if cursor < len(lines) else len(raw)
        closing = _CLOSE.format(index=index).encode()
        while cursor < len(lines) and lines[cursor] != closing:
            take()
        ref_end = starts[cursor] if cursor < len(lines) else len(raw)
        if ref_end == ref_start:
            raise PublicParseError("empty reference block")
        references.append((ref_start, ref_end))
        expect(_CLOSE.format(index=index), "reference_block_close")
        index += 1
    if len(docs) < 3 or len(references) != len(docs) - 1:
        raise PublicParseError("layout needs three or more documents with alternating reference blocks")
    question_start = starts[cursor] if cursor < len(lines) else len(raw)
    question = question_start, len(raw)
    # Independently enforce equal-byte cuts at whole reference-line boundaries.
    # The invariant uses public bytes only, with no world/answer dependent tuning.
    ref_lines = [(a + delta, a + delta + len(line))
                 for a, b in references
                 for delta, line in _positioned_lines(raw[a:b])]
    total = sum(b - a for a, b in ref_lines)
    consumed, next_cut, observed = 0, 1, []
    for a, b in ref_lines:
        consumed += b - a
        if next_cut < len(references) and consumed * len(references) >= total * next_cut:
            observed.append(b)
            next_cut += 1
    if observed != [b for _, b in references[:-1]]:
        raise PublicParseError("noncanonical public reference band boundaries")
    # Reference stream consists of guide, vocabulary, finite domains and rules.
    # The task belongs between the first blank guide terminator and vocabulary.
    ref_payload = b"".join(raw[a:b] for a, b in references)
    guide_end = ref_payload.find(b"\n\n") + 2
    if guide_end < 2 or not ref_payload.startswith((env["guide_header"] + "\n").encode()):
        raise PublicParseError("missing ordered reference reading guide")
    first, rest, left = [], [], guide_end
    for a, b in references:
        if left:
            width = min(left, b - a)
            first.append((a, a + width))
            if a + width < b:
                rest.append((a + width, b))
            left -= width
        else:
            rest.append((a, b))
    intervals = [title, *first, task, *rest, heading, *docs, question]
    canonical, transport = bytearray(), []
    for a, b in intervals:
        if a == b:
            continue
        old_start = len(canonical)
        canonical.extend(raw[a:b])
        transport.append({"canonical_span": _span(old_start, len(canonical)), "actual_span": _span(a, b)})
    normalized = [copy.deepcopy(messages[0]), {"role": "user", "content": canonical.decode("utf-8")}]
    parsed = parse_public(normalized, catalogue_path, require_frozen=bool(pin))
    # Transport is bijective over every old byte; wrappers are the only new bytes.
    ranges = sorted([(item["actual_span"]["utf8_start"], item["actual_span"]["utf8_end"])
                     for item in transport] + [(item["span"]["utf8_start"], item["span"]["utf8_end"])
                                               for item in literals])
    cursor = 0
    for a, b in ranges:
        if a != cursor or b <= a:
            raise PublicParseError("transport fails total actual-byte bijection")
        cursor = b
    if cursor != len(raw):
        raise PublicParseError("uncovered actual layout bytes")
    for field in ("record_spans", "atom_spans", "framing_spans", "document_spans", "source_spans", "coverage_spans"):
        parsed[field] = [{**item, "span": _remap_span(item["span"], transport)} for item in parsed[field]]
    parsed["coverage_spans"] += literals
    parsed["coverage_spans"].sort(key=lambda item: (item["span"]["message_index"], item["span"]["utf8_start"]))
    for mi, message in enumerate(messages):
        cursor = 0
        for item in parsed["coverage_spans"]:
            span = item["span"]
            if span["message_index"] == mi:
                if span["utf8_start"] != cursor or span["utf8_end"] <= cursor:
                    raise PublicParseError("non-total actual layout grammar coverage")
                message["content"].encode()[span["utf8_start"]:span["utf8_end"]].decode("utf-8")
                cursor = span["utf8_end"]
        if cursor != len(message["content"].encode("utf-8")):
            raise PublicParseError("unconsumed actual layout bytes")
    canonical_parser = parsed["parser_identity_sha256"]
    normalized_hash = parsed["messages_sha256"]
    parsed.update(layout_version=LAYOUT_VERSION,
                  layout_review_status="INDEPENDENTLY_REVIEWED_RESEARCH_LAYOUT" if pin else "UNREVIEWED_MECHANICS_ONLY",
                  parser_identity_sha256=_identity(), messages_sha256=_digest(messages),
                  dossier_span=_span(0, question_start), admission_ready=bool(pin),
                  model_admission=False, training_authorized=False,
                  layout_freeze_receipt_sha256=pin["receipt_sha256"] if pin else None,
                  layout_metadata_spans=[{"field_path": "layout_version", "span": _span(layout[0] + len("Public layout: "), layout[1] - 2),
                                          "typed_value_sha256": _digest(LAYOUT_VERSION)}],
                  normalization={"messages_sha256": normalized_hash, "canonical_parser_identity_sha256": canonical_parser,
                                 "transport": transport, "byte_transport_complete": True})
    return parsed


def _positioned_lines(raw):
    cursor = 0
    for line in raw.splitlines(keepends=True):
        yield cursor, line
        cursor += len(line)


def solve_interleaved(parsed, *, messages, require_source_bytes=True, require_reviewed=False):
    """Reparse actual v2 bytes, then use the frozen solver's structural path.

    Core grounding flags remain truthful (no core UTF8 reread). The explicit v2
    extension records the independent actual-byte reread and transported maps.
    Reviewed layout readiness applies only to research-candidate row layout,
    never to model or training admission.
    """
    from .independent_solver import SchemaError, solve_parsed_public
    if type(require_source_bytes) is not bool or not require_source_bytes:
        raise SchemaError("interleaved solving requires actual source bytes")
    fresh = parse_interleaved(messages, require_reviewed=require_reviewed)
    if type(parsed) is not dict or _digest(parsed) != _digest(fresh):
        raise SchemaError("supplied v2 world/hash/maps/identity differ from independent actual-source reread")
    solution = solve_parsed_public(fresh, require_source_bytes=False)
    solution["layout_grounding"] = {
        "layout_version": LAYOUT_VERSION,
        "mode": "REPARSED_INTERLEAVED_PUBLIC_SOURCE_WITH_UTF8_TRANSPORT",
        "source_bytes_verified": True,
        "actual_messages_sha256": fresh["messages_sha256"],
        "normalization_messages_sha256": fresh["normalization"]["messages_sha256"],
        "actual_source_maps_verified": True,
        "byte_transport_complete": True,
        "layout_parser_identity_sha256": _identity(),
        "canonical_parser_identity_sha256": fresh["normalization"]["canonical_parser_identity_sha256"],
        "layout_review_status": fresh["layout_review_status"],
        "layout_freeze_receipt_sha256": fresh["layout_freeze_receipt_sha256"],
        "admission_grounding": fresh["admission_ready"],
        "admission_ready": fresh["admission_ready"],
        "readiness_scope": "RESEARCH_CANDIDATE_ROW_LAYOUT_ONLY",
        "model_admission": False,
        "training_authorized": False,
    }
    solution["model_admission"] = False
    solution["training_authorized"] = False
    solution["limitations"].append("Interleaved v2 byte grounding is a separate extension; reviewed readiness concerns research-row layout only.")
    return solution


def export_interleaved_public_packet(rendered, *, row_id, parent_id, view_id, revision):
    """Export only a genuine reviewed production capability as a research candidate.

    One capability binds one row/parent/view/revision. Mechanics/evidence artifacts,
    copies, changed messages/maps/world/reservation and drifted pins cannot export.
    """
    if (type(rendered) is not _InterleavedRender or rendered.get("mode") != "production"
            or rendered.get("export_allowed") is not True
            or getattr(rendered, "_production_parent_id", None) is None):
        raise LayoutError("mechanics/evidence or copied layout artifacts cannot export")
    mint = _MINTED_RENDERS.get(id(rendered))
    if mint is None or mint["reference"]() is not rendered:
        raise LayoutError("production layout capability was not minted for this exact object")
    if any(getattr(rendered, field, None) != value for field, value in mint["bindings"]):
        raise LayoutError("production layout private bindings changed after capability mint")
    for field, value in (("row_id", row_id), ("parent_id", parent_id), ("view_id", view_id)):
        if type(value) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", value) is None:
            raise LayoutError("invalid export identity: " + field)
    if type(revision) is not int or not 1 <= revision <= 2**63 - 1:
        raise LayoutError("invalid export revision")
    identity = row_id, parent_id, view_id, revision
    if parent_id != rendered._production_parent_id or (mint["export_identity"] is not None and identity != mint["export_identity"]):
        raise LayoutError("layout capability parent/row/view/revision mismatch")
    pin = _verify_layout_pin()
    if pin["receipt_sha256"] != rendered._verified_layout_receipt_sha256:
        raise LayoutError("layout receipt changed after production rendering")
    if _digest(rendered) != rendered._verified_render_sha256:
        raise LayoutError("production render artifact changed before export")
    messages = rendered.get("messages")
    if (_digest(messages) != rendered._verified_messages_sha256
            or rendered.get("messages_sha256") != rendered._verified_messages_sha256
            or rendered.get("base_messages_sha256") != rendered._verified_base_messages_sha256
            or rendered.get("reservation_receipt_sha256") != rendered._verified_reservation_sha256):
        raise LayoutError("production messages/base/reservation changed before export")
    fresh = parse_interleaved(messages, require_reviewed=True)
    if _digest(fresh["world"]) != rendered._verified_world_sha256:
        raise LayoutError("production public world changed before export")
    # Every visible correspondence and readiness identity must remain intact.
    if any(rendered.get(field) != value for field, value in fresh.items() if field != "world"):
        raise LayoutError("production layout metadata/maps/readiness changed before export")
    if rendered.get("model_admission") is not False or rendered.get("training_authorized") is not False:
        raise LayoutError("layout review cannot self-authorize model or training use")
    # Verification can run concurrently, but an exact capability can bind only
    # one identity. Recheck and commit together after all potentially slow reads.
    with _EXPORT_LOCK:
        current = _MINTED_RENDERS.get(id(rendered))
        if current is not mint or current["reference"]() is not rendered:
            raise LayoutError("production capability identity changed during export")
        if any(getattr(rendered, field, None) != value for field, value in current["bindings"]):
            raise LayoutError("production bindings changed during export")
        if _digest(rendered) != rendered._verified_render_sha256:
            raise LayoutError("production artifact changed during export verification")
        output_messages = copy.deepcopy(messages)
        immutable = dict(current["bindings"])
        if _digest(output_messages) != immutable["_verified_messages_sha256"]:
            raise LayoutError("production source changed while capturing export bytes")
        if current["export_identity"] is not None and current["export_identity"] != identity:
            raise LayoutError("concurrent export attempted a different row/view/revision")
        rendered._export_identity = identity
        current["export_identity"] = identity
    return {"row_id": row_id, "parent_id": parent_id, "view_id": view_id, "revision": revision,
            "messages": output_messages, "messages_sha256": immutable["_verified_messages_sha256"],
            "reservation_receipt_sha256": immutable["_verified_reservation_sha256"],
            "layout_version": LAYOUT_VERSION, "layout_freeze_receipt_sha256": pin["receipt_sha256"],
            "scope": "RESEARCH_CANDIDATE", "model_admission": False, "training_authorized": False}
