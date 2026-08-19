from __future__ import annotations

import re

from rineya_harness.domain.enums import RouteAction, RunMode
from rineya_harness.domain.models import RouteDecision

_CURRENT_MARKERS = re.compile(
    r"\b(current|currently|latest|today|tonight|this week|recent|now|live|price|weather|"
    r"schedule|status|version|release|news|who is|when is|available now)\b",
    re.IGNORECASE,
)
_SEARCH_MARKERS = re.compile(
    r"\b(search|browse|look up|find sources?|verify online|check the web|cite sources?)\b",
    re.IGNORECASE,
)


def route_request(question: str, mode: RunMode) -> RouteDecision:
    """Code-driven mode gate for known routing rules.

    The function intentionally avoids asking a small model to rediscover explicit product policy.
    """

    normalized = " ".join(question.split())
    if not normalized:
        return RouteDecision(
            action=RouteAction.CLARIFY,
            reason="empty_request",
            clarification="What would you like me to help with?",
        )
    if mode is RunMode.DIRECT:
        return RouteDecision(action=RouteAction.DIRECT, reason="explicit_direct_mode")
    if mode in {RunMode.GROUNDED, RunMode.RESEARCH}:
        return RouteDecision(
            action=RouteAction.SEARCH,
            reason=f"explicit_{mode.value}_mode",
            query=normalized,
        )
    if _SEARCH_MARKERS.search(normalized):
        return RouteDecision(
            action=RouteAction.SEARCH,
            reason="explicit_search_language",
            query=normalized,
        )
    if _CURRENT_MARKERS.search(normalized):
        return RouteDecision(
            action=RouteAction.SEARCH,
            reason="temporally_unstable_request",
            query=normalized,
        )
    return RouteDecision(action=RouteAction.DIRECT, reason="stable_request_default")
