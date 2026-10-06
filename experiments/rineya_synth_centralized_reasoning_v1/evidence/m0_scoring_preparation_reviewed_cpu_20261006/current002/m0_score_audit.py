"""Independent consumer replay; native capture reads use the same closed admission."""
from copy import deepcopy
import m0_scoring as scoring


def audit_authored_fixture(fixture, packet):
    scoring.require(type(fixture) is scoring._Fixture
                    and fixture.sha256 == fixture._factory_sha256 == scoring.digest(fixture.rows),
                    "original unmodified authored fixture")
    scoring.require(packet.get("scope") == scoring.FIXTURE, "fixture cannot audit native scores")
    _audit_rows(fixture.rows, packet, scoring.FIXTURE, {"fixture_factory_sha256": fixture.sha256})
    return scoring.seal({"kind": "CPU_AUTHORED_M0_SCORE_REPLAY", "packet_sha256": packet["sha256"],
        "rows_replayed": 384, "native_scoring_authorized": False, "quality_gain_established": False})


def audit_complete_m0_score_packet(packet, registration_anchor, *, approval_anchor, review_anchor):
    # Public producer admission rechecks exact approved registration, all six raw
    # receipts and the full frozen verifier before any native replay is possible.
    rows, lineage = scoring.prepare_complete_m0(registration_anchor,
        approval_anchor=approval_anchor, review_anchor=review_anchor)
    _audit_rows(rows, packet, scoring.SCOPE, lineage)
    return scoring.seal({"kind": "INDEPENDENT_COMPLETE_M0_SCORE_REPLAY",
        "scope": scoring.SCOPE, "packet_sha256": packet["sha256"], "rows_replayed": 384,
        "lineage": deepcopy(lineage), "comparison_authorized": False,
        "retention_certified": False, "quality_gain_established": False})


def audit_published_authored_fixture(fixture, publication):
    """Read only a path issued by the synthetic publication capability."""
    scoring.require(type(fixture) is scoring._Fixture
        and fixture.sha256 == fixture._factory_sha256 == scoring.digest(fixture.rows),
        "unmodified authored fixture")
    scoring.require(type(publication) is scoring._PublishedFixture
        and publication.fixture_sha256 == fixture.sha256
        and publication._factory_anchor_sha256 == scoring.digest(publication.anchor),
        "original authored publication capability")
    packet = scoring.read_anchor(publication.anchor)
    result = audit_authored_fixture(fixture, packet)
    return scoring.seal({"kind": "CPU_PUBLISHED_AUTHORED_M0_SCORE_REPLAY",
        "publication_anchor": publication.anchor, "replay": result,
        "native_scoring_authorized": False, "quality_gain_established": False})


def audit_registered_m0_score_packet(publication_anchor, registration_anchor, *, approval_anchor, review_anchor):
    """Require approved exact output selection before reading saved native scores."""
    registration = scoring.validate_m0_scoring_registration(registration_anchor,
        approval_anchor=approval_anchor, review_anchor=review_anchor)
    scoring.require(type(publication_anchor) is dict
        and publication_anchor.get("path") == registration["output_path"],
        "saved score path differs from exact registration")
    packet = scoring.read_anchor(publication_anchor)
    result = audit_complete_m0_score_packet(packet, registration_anchor,
        approval_anchor=approval_anchor, review_anchor=review_anchor)
    return scoring.seal({"kind": "INDEPENDENT_PUBLISHED_COMPLETE_M0_SCORE_REPLAY",
        "publication_anchor": publication_anchor, "replay": result,
        "comparison_authorized": False, "retention_certified": False,
        "quality_gain_established": False})


def _audit_rows(rows, packet, scope, lineage):
    """Consumer constructs expected row fields independently of producer scoring loop."""
    scoring.require(type(packet) is dict and packet.get("sha256") == scoring.digest(
        {k: v for k, v in packet.items() if k != "sha256"}), "score packet checksum")
    scoring.require(len(rows) == len(packet.get("rows", [])) == 384
        and len({r["row_id"] for r in rows}) == 384
        and len({r["component_id"] for r in rows}) == 96, "whole384/96 audit population")
    verified = []
    with scoring.frozen_verifier() as verifier:
        for raw, scored in zip(rows, packet["rows"]):
            result = verifier.verify_response(raw["messages"], raw["raw"]["decoded_response"])
            scoring.require(type(result.get("accepted")) is bool, "replay exact accepted bool")
            expected = {name: raw[name] for name in ("row_id", "component_id", "skill", "condition", "rendering")}
            expected.update({"messages_sha256": scoring.digest(raw["messages"]),
                "raw_sha256": scoring.digest(raw["raw"]), "finish_reason": raw["raw"]["finish_reason"],
                "raw_generated_ids": raw["raw"]["raw_generated_ids"],
                "accepted": result["accepted"] is True, "reason": result["reason"]})
            scoring.require(scored == expected, "exact complete score row replay")
            verified.append(expected)
    expected_packet = scoring.seal({"kind": "COMPLETE_M0_WHOLE_RESPONSE_SCORE_PACKET",
        "scope": scope, "arm": "M0", "rows": verified,
        "row_order_sha256": scoring.digest([r["row_id"] for r in rows]),
        "input_rows_sha256": scoring.digest(rows), "lineage": lineage,
        "scorer_sources": scoring.SCORER_PINS, "verifier_calls": 384,
        "comparison_authorized": False, "retention_certified": False, "training_authorized": False,
        "promotion_authorized": False, "quality_gain_established": False})
    scoring.require(packet == expected_packet, "complete packet lineage and closed claims")
