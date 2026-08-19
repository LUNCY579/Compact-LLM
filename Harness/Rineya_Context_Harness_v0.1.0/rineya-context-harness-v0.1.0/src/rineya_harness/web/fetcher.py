from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from rineya_harness.domain.models import FetchedDocument
from rineya_harness.errors import SecurityViolationError
from rineya_harness.security.network import NetworkPolicy, validate_outbound_url
from rineya_harness.security.untrusted import scan_untrusted_text

from .extractor import ReadableTextExtractor


@dataclass(slots=True)
class SafeWebFetcher:
    """Bounded, read-only fetcher with redirect and SSRF checks.

    This is defense in depth, not a substitute for an isolated egress service. A production browser
    tier should additionally pin DNS/connect targets, restrict egress at the network layer, and run
    without credentials.
    """

    network_policy: NetworkPolicy = field(default_factory=NetworkPolicy)
    timeout_seconds: float = 15.0
    max_response_bytes: int = 2_000_000
    max_redirects: int = 3
    user_agent: str = "RineyaContextHarness/0.1 (+local-research-runtime)"
    extractor: ReadableTextExtractor = field(default_factory=ReadableTextExtractor)

    async def fetch(self, url: str) -> FetchedDocument:
        requested_url = url
        current_url = url
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html, text/plain;q=0.9, application/xhtml+xml;q=0.8",
        }
        async with httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(self.timeout_seconds),
            follow_redirects=False,
            headers=headers,
        ) as client:
            for redirect_count in range(self.max_redirects + 1):
                await validate_outbound_url(current_url, self.network_policy)
                async with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise SecurityViolationError(
                                "REDIRECT_LOCATION_MISSING",
                                f"Redirect from {current_url} did not include Location",
                            )
                        if redirect_count >= self.max_redirects:
                            raise SecurityViolationError(
                                "TOO_MANY_REDIRECTS",
                                f"Exceeded {self.max_redirects} redirects for {requested_url}",
                            )
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
                    if content_type not in {
                        "text/html",
                        "text/plain",
                        "application/xhtml+xml",
                    }:
                        raise SecurityViolationError(
                            "CONTENT_TYPE_BLOCKED",
                            f"Unsupported content type {content_type!r} for {current_url}",
                        )
                    chunks: list[bytes] = []
                    byte_count = 0
                    async for chunk in response.aiter_bytes():
                        byte_count += len(chunk)
                        if byte_count > self.max_response_bytes:
                            raise SecurityViolationError(
                                "RESPONSE_TOO_LARGE",
                                f"Response exceeded {self.max_response_bytes} bytes",
                            )
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    decoded = raw.decode(response.encoding or "utf-8", errors="replace")
                    if content_type in {"text/html", "application/xhtml+xml"}:
                        title, text = self.extractor.extract(decoded)
                    else:
                        title, text = None, decoded.strip()
                    if not text:
                        raise SecurityViolationError("EMPTY_DOCUMENT", f"No readable text in {current_url}")
                    scan = scan_untrusted_text(text)
                    return FetchedDocument(
                        requested_url=requested_url,
                        final_url=current_url,
                        title=title,
                        text=text,
                        content_type=content_type,
                        content_sha256=hashlib.sha256(raw).hexdigest(),
                        status_code=response.status_code,
                        byte_count=len(raw),
                        metadata={
                            "redirect_count": redirect_count,
                            "prompt_injection_suspicious": scan.suspicious,
                            "prompt_injection_flags": list(scan.flags),
                        },
                    )
        raise RuntimeError("unreachable fetch loop")
