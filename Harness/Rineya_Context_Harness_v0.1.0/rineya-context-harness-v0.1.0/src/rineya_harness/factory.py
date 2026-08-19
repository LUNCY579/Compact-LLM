from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rineya_harness.config import HarnessConfig
from rineya_harness.context.assembler import ContextAssembler
from rineya_harness.context.compaction import DeterministicCompactor
from rineya_harness.evidence.ledger import EvidenceLedgerBuilder
from rineya_harness.evidence.ranker import LexicalPassageRanker
from rineya_harness.evidence.verifier import AnswerabilityGate
from rineya_harness.memory.service import MemoryService
from rineya_harness.models.base import ChatModel
from rineya_harness.models.deterministic import DeterministicChatModel
from rineya_harness.models.ollama import OllamaChatModel
from rineya_harness.models.openai_compatible import OpenAICompatibleChatModel
from rineya_harness.orchestration.evidence_pipeline import EvidencePipeline
from rineya_harness.orchestration.policy import DeterministicResearchPolicy, ModelResearchPolicy
from rineya_harness.orchestration.workflow import ContextHarnessWorkflow
from rineya_harness.search.base import SearchProvider
from rineya_harness.search.fixture import FixtureSearchProvider
from rineya_harness.search.searxng import SearxNGSearchProvider
from rineya_harness.search.service import SearchService
from rineya_harness.security.network import NetworkPolicy
from rineya_harness.state.store import SQLiteStateStore
from rineya_harness.tools import FinishToolSet, MemoryToolSet, ResearchToolSet, ToolRegistry
from rineya_harness.tracing.jsonl import JsonlTraceRecorder
from rineya_harness.web.base import DocumentFetcher
from rineya_harness.web.fetcher import SafeWebFetcher
from rineya_harness.web.fixture import FixtureDocumentFetcher


@dataclass(slots=True)
class HarnessContainer:
    config: HarnessConfig
    store: SQLiteStateStore
    memory: MemoryService
    model: ChatModel
    search_provider: SearchProvider
    search: SearchService
    fetcher: DocumentFetcher
    workflow: ContextHarnessWorkflow
    tools: ToolRegistry
    traces: JsonlTraceRecorder


def create_container(
    config: HarnessConfig,
    *,
    fixture_search: FixtureSearchProvider | None = None,
    fixture_fetcher: FixtureDocumentFetcher | None = None,
) -> HarnessContainer:
    store = SQLiteStateStore(config.database, config.memory)
    store.initialize()
    memory = MemoryService(store)
    model = _build_model(config)
    search_provider = fixture_search or _build_search(config)
    search = SearchService(search_provider, max_results=config.search.max_results)
    fetcher = fixture_fetcher or _build_fetcher(config)
    traces = JsonlTraceRecorder(
        config.trace.directory,
        max_payload_chars=config.trace.max_payload_chars,
        include_tool_payloads=config.trace.include_tool_payloads,
    )
    context = ContextAssembler(store, static_policy=config.static_policy)
    ledger = EvidenceLedgerBuilder()
    evidence_pipeline = EvidencePipeline(
        search=search,
        fetcher=fetcher,
        ranker=LexicalPassageRanker(
            passage_chars=config.search.passage_chars,
            overlap_chars=config.search.passage_overlap_chars,
        ),
        ledger=ledger,
        store=store,
        fetch_concurrency=config.search.fetch_concurrency,
        max_passages_per_source=config.search.max_passages_per_source,
    )
    policy = (
        DeterministicResearchPolicy()
        if config.model.provider == "deterministic"
        else ModelResearchPolicy(
            model=model,
            max_output_tokens=config.model.max_output_tokens,
            temperature=config.model.temperature,
        )
    )
    tools = ToolRegistry()
    for definition in MemoryToolSet(store, memory).definitions():
        tools.register(definition)
    for definition in ResearchToolSet(search, store).definitions():
        tools.register(definition)
    for definition in FinishToolSet().definitions():
        tools.register(definition)

    workflow = ContextHarnessWorkflow(
        config=config,
        store=store,
        context=context,
        evidence_pipeline=evidence_pipeline,
        policy=policy,
        verifier=AnswerabilityGate(minimum_independent_domains=1, require_required_claim=True),
        compactor=DeterministicCompactor(store, memory),
        traces=traces,
    )
    return HarnessContainer(
        config=config,
        store=store,
        memory=memory,
        model=model,
        search_provider=search_provider,
        search=search,
        fetcher=fetcher,
        workflow=workflow,
        tools=tools,
        traces=traces,
    )


def _build_model(config: HarnessConfig) -> ChatModel:
    model_config = config.model
    if model_config.provider == "openai_compatible":
        return OpenAICompatibleChatModel(
            model=model_config.model,
            base_url=model_config.base_url,
            api_key=(
                model_config.api_key.get_secret_value() if model_config.api_key is not None else None
            ),
            timeout_seconds=model_config.timeout_seconds,
            enable_thinking=model_config.enable_thinking,
        )
    if model_config.provider == "ollama":
        return OllamaChatModel(
            model=model_config.model,
            base_url=model_config.base_url,
            timeout_seconds=model_config.timeout_seconds,
        )
    return DeterministicChatModel()


def _build_search(config: HarnessConfig) -> SearchProvider:
    if config.search.provider == "searxng":
        return SearxNGSearchProvider(
            base_url=config.search.base_url,
            timeout_seconds=config.search.timeout_seconds,
            max_response_bytes=config.search.max_response_bytes,
        )
    return FixtureSearchProvider()


def _build_fetcher(config: HarnessConfig) -> DocumentFetcher:
    if not config.web.enabled:
        return FixtureDocumentFetcher()
    return SafeWebFetcher(
        network_policy=NetworkPolicy(
            allow_private_networks=config.web.allow_private_networks,
            allowed_ports=frozenset(config.web.allowed_ports),
            allow_nonstandard_ports=config.web.allow_nonstandard_ports,
        ),
        timeout_seconds=config.web.timeout_seconds,
        max_response_bytes=config.web.max_response_bytes,
        max_redirects=config.web.max_redirects,
        user_agent=config.web.user_agent,
    )


def resolve_config_relative_paths(config: HarnessConfig, config_path: str | Path | None) -> HarnessConfig:
    """Resolve local artifact paths relative to the selected configuration file.

    This helper is opt-in so programmatic callers can continue to pass already-resolved paths.
    """

    if config_path is None:
        return config
    base = Path(config_path).resolve().parent
    updated = config.model_copy(deep=True)
    if not updated.database.path.is_absolute():
        updated.database.path = base / updated.database.path
    if not updated.trace.directory.is_absolute():
        updated.trace.directory = base / updated.trace.directory
    return updated
