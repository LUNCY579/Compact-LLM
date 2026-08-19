from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rineya_harness.config import DatabaseConfig, HarnessConfig, TraceConfig
from rineya_harness.domain.models import FetchedDocument, SearchResult
from rineya_harness.factory import HarnessContainer, create_container
from rineya_harness.search.fixture import FixtureSearchProvider
from rineya_harness.web.fixture import FixtureDocumentFetcher


@pytest.fixture()
def config(tmp_path: Path) -> HarnessConfig:
    return HarnessConfig(
        database=DatabaseConfig(path=tmp_path / "state.sqlite3"),
        trace=TraceConfig(directory=tmp_path / "traces"),
    )


@pytest.fixture()
def container(config: HarnessConfig) -> HarnessContainer:
    return create_container(config)


@pytest.fixture()
def research_container(tmp_path: Path) -> HarnessContainer:
    question = "How does Rineya forget obsolete memory?"
    url = "https://example.org/rsf"
    text = (
        "Rineya keeps immutable evidence in an external ledger. It evicts obsolete records from "
        "active context only after replacement state and dependencies are validated."
    )
    document = FetchedDocument(
        requested_url=url,
        final_url=url,
        title="RSF fixture",
        text=text,
        content_type="text/plain",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        status_code=200,
        byte_count=len(text),
    )
    search = FixtureSearchProvider(
        fixtures={
            question: [
                SearchResult(title="RSF fixture", url=url, snippet=text, rank=1, provider="fixture")
            ]
        }
    )
    fetcher = FixtureDocumentFetcher(documents={url: document})
    config = HarnessConfig(
        database=DatabaseConfig(path=tmp_path / "research.sqlite3"),
        trace=TraceConfig(directory=tmp_path / "traces"),
    )
    return create_container(config, fixture_search=search, fixture_fetcher=fetcher)
