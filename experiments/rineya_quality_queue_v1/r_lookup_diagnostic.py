"""Bounded synthetic atomic-R coverage; no files, models or bank discovery."""
from __future__ import annotations

import re

_ALIAS = re.compile(r"[a-z][a-z0-9_]{0,19}\Z")
PUBLIC_CONTRACT_SHA256 = "6f04b61a31e07c23b4621cfb850cbf4711d8cc95e2bcf4bad28a328bf4d587e9"


def _grammar(spec):
    # Shared grammar validation is explicit; it is not a third semantic solver.
    if type(spec) is not dict or len(spec) != 2 or set(spec) != {"initial", "lookup"}:
        raise ValueError("only explicit atomic R fixture metadata is supported")
    initial = spec["initial"]
    if (type(initial) is not list or not 6 <= len(initial) <= 15
            or any(type(v) is not int or not -100 <= v <= 100 for v in initial)):
        raise ValueError("initial fixture domain")
    lookup = spec["lookup"]
    if type(lookup) is not dict or len(lookup) != 2 or set(lookup) != {"links", "requested"}:
        raise ValueError("lookup fixture fields")
    links, requested = lookup["links"], lookup["requested"]
    if type(links) is not dict or not 1 <= len(links) <= 80:
        raise ValueError("link fixture bound")
    valid_alias = lambda a: type(a) is str and _ALIAS.fullmatch(a) is not None
    for alias, target in links.items():
        if not valid_alias(alias) or not (
                (type(target) is int and 0 <= target < len(initial))
                or valid_alias(target)):
            raise ValueError("alias or terminal fixture domain")
    if (type(requested) is not list or not 1 <= len(requested) <= len(initial)
            or not all(valid_alias(a) for a in requested)):
        raise ValueError("requested fixture domain")
    return dict(links), list(requested)


def _walk(links, start):
    seen = {}
    sequence = []
    node = start
    edges = 0
    for _ in range(len(links) + 1):
        if node in seen:
            witness = ("CYCLE", tuple(sorted(sequence[seen[node]:])))
            return {"endpoint": None, "cause": "CYCLE", "edges": edges, "failure_witness": witness}
        if node not in links:
            return {"endpoint": None, "cause": "DANGLING", "edges": edges,
                    "failure_witness": ("DANGLING", node)}
        seen[node] = len(sequence)
        sequence.append(node)
        target = links[node]
        edges += 1
        if type(target) is int:
            return {"endpoint": target, "cause": "RESOLVED", "edges": edges, "failure_witness": None}
        node = target
    raise RuntimeError("bounded traversal invariant failed")


def _closure_endpoints(links, requested):
    # This route consumes no traversal endpoints, traces or classifications.
    resolved = {alias: target for alias, target in links.items() if type(target) is int}
    rounds = 0
    for _ in range(len(links)):
        rounds += 1
        previous = resolved
        next_resolved = dict(previous)
        for alias, target in links.items():
            if type(target) is str and target in previous:
                next_resolved[alias] = previous[target]
        resolved = next_resolved
        if resolved == previous:
            break
    return [resolved.get(alias) for alias in requested], rounds


def analyze_atomic_r_fixture(spec):
    """Classify explicit synthetic metadata, refusing malformed/unsupported input.

    UNKNOWN means a valid requested lookup is unresolved. Refusal, operational
    failure or route disagreement raises instead. Sorted indices represent one
    allowed answer order; this helper does not score language-model responses.
    """
    links, requested = _grammar(spec)
    walks = [_walk(links, alias) for alias in requested]
    endpoints = [walk["endpoint"] for walk in walks]
    closure, rounds = _closure_endpoints(links, requested)
    if endpoints != closure:
        raise RuntimeError("independent fixture routes disagree")
    all_resolved = all(endpoint is not None for endpoint in endpoints)
    unique_requests = list(dict.fromkeys(requested))
    request_results = {alias: walks[i] for i, alias in enumerate(requested)}
    unique_successes = [request_results[a]["endpoint"] for a in unique_requests
                        if request_results[a]["endpoint"] is not None]
    requested_causes = sorted({walk["cause"] for walk in walks
                               if walk["cause"] != "RESOLVED"})
    reached = set()
    for start in unique_requests:
        node = start
        while node in links and node not in reached:
            reached.add(node)
            target = links[node]
            if type(target) is int:
                break
            node = target
    # Global diagnostic causes come from traversal only. Unreachable components
    # never change the public answer or closure route's resolvability result.
    irrelevant_walks = [_walk(links, a) for a in links if a not in reached]
    requested_witnesses = {walk["failure_witness"] for walk in walks if walk["failure_witness"] is not None}
    irrelevant_causes = sorted({walk["cause"] for walk in irrelevant_walks
                               if walk["failure_witness"] is not None
                               and walk["failure_witness"] not in requested_witnesses})
    answer = sorted(set(endpoints)) if all_resolved else "UNKNOWN"
    labels = []
    if len(unique_requests) < len(requested):
        labels.append("REPEATED_REQUEST_STRINGS")
    if len(unique_successes) > len(set(unique_successes)):
        labels.append("DISTINCT_ALIASES_SHARED_ENDPOINT")
    if all_resolved and len(set(endpoints)) == len(requested):
        labels.append("FULLY_RESOLVED_DISTINCT_ENDPOINTS")
    labels += ["REQUESTED_" + cause for cause in requested_causes]
    labels += ["IRRELEVANT_" + cause for cause in irrelevant_causes]
    return {
        "classification": "SYNTHETIC_UNSELECTED_ATOMIC_R_COVERAGE_ONLY",
        "public_contract_source_sha256": PUBLIC_CONTRACT_SHA256,
        "fixture_reference": {"answer": answer},
        "semantic_status": "RESOLVED" if all_resolved else "UNKNOWN",
        "request_count": len(requested),
        "distinct_requested_aliases": len(unique_requests),
        "resolved_request_count": sum(e is not None for e in endpoints),
        "resolved_endpoint_count": len({e for e in endpoints if e is not None}),
        "partial_endpoints_are_an_answer": False,
        "coverage_labels": labels,
        "requested_causes_from_traversal": requested_causes,
        "irrelevant_causes_from_traversal": irrelevant_causes,
        "routes_agree": True,
        "logical_work": {"request_traversal_edges": sum(w["edges"] for w in walks),
                         "irrelevant_traversal_edges": sum(w["edges"] for w in irrelevant_walks),
                         "closure_rounds": rounds,
                         "maximum_requested_classification_iterations": len(requested) * (len(links) + 1),
                         "maximum_closure_link_visits": len(links) ** 2,
                         "maximum_irrelevant_classification_iterations": len(links) * (len(links) + 1),
                         "partial_counts_not_inclusive_resource_prices": True},
        "native_runtime_or_resource_fit": False,
        "actual_bank_coverage_or_independence": False,
        "model_quality_evidence": False,
    }
