from rineya_harness.domain.enums import RouteAction, RunMode
from rineya_harness.orchestration.router import route_request


def test_explicit_direct_mode() -> None:
    assert route_request("Explain gravity", RunMode.DIRECT).action is RouteAction.DIRECT


def test_current_request_routes_to_search() -> None:
    decision = route_request("What is the latest release?", RunMode.AUTO)
    assert decision.action is RouteAction.SEARCH


def test_stable_auto_request_routes_direct() -> None:
    assert route_request("Why does ice float?", RunMode.AUTO).action is RouteAction.DIRECT


def test_empty_request_clarifies() -> None:
    assert route_request("   ", RunMode.AUTO).action is RouteAction.CLARIFY
