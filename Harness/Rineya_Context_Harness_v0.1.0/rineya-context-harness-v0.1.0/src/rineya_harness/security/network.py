from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

from rineya_harness.errors import SecurityViolationError


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    allowed_schemes: frozenset[str] = frozenset({"http", "https"})
    allow_private_networks: bool = False
    allowed_ports: frozenset[int] = field(default_factory=lambda: frozenset({80, 443}))
    allow_nonstandard_ports: bool = False


async def validate_outbound_url(url: str, policy: NetworkPolicy) -> None:
    parsed = urlparse(url)
    if parsed.scheme.casefold() not in policy.allowed_schemes:
        raise SecurityViolationError("URL_SCHEME_BLOCKED", f"Blocked scheme: {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise SecurityViolationError(
            "URL_USERINFO_BLOCKED",
            "Outbound URLs must not contain embedded credentials",
        )
    if not parsed.hostname:
        raise SecurityViolationError("URL_HOST_MISSING", "Outbound URL has no hostname")
    hostname = parsed.hostname.casefold().rstrip(".")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        raise SecurityViolationError("LOCALHOST_BLOCKED", f"Blocked local hostname: {hostname}")
    try:
        port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
    except ValueError as exc:
        raise SecurityViolationError("URL_PORT_INVALID", str(exc)) from exc
    if not policy.allow_nonstandard_ports and port not in policy.allowed_ports:
        raise SecurityViolationError("URL_PORT_BLOCKED", f"Outbound port {port} is not allowed")

    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise SecurityViolationError("DNS_RESOLUTION_FAILED", f"Cannot resolve {hostname}: {exc}") from exc
    if not addresses:
        raise SecurityViolationError("DNS_EMPTY", f"No addresses resolved for {hostname}")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if policy.allow_private_networks:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise SecurityViolationError(
                "PRIVATE_ADDRESS_BLOCKED",
                f"Outbound URL {hostname} resolved to blocked address {ip}",
            )
