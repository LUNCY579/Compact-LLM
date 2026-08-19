from __future__ import annotations

from typing import Any

from pydantic import Field

from rineya_harness.domain.enums import ToolRisk
from rineya_harness.domain.models import StrictModel
from rineya_harness.search.service import SearchService
from rineya_harness.state.store import SQLiteStateStore

from .base import ToolDefinition


class SearchWebArgs(StrictModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=8, ge=1, le=20)


class InspectEvidenceArgs(StrictModel):
    run_id: str
    evidence_ids: list[str] = Field(default_factory=list, max_length=64)


class ResearchToolSet:
    def __init__(self, search: SearchService, store: SQLiteStateStore) -> None:
        self.search = search
        self.store = store

    async def search_web(self, args: SearchWebArgs) -> dict[str, Any]:
        results = await self.search.search(args.query, limit=args.limit)
        return {"results": [result.model_dump(mode="json") for result in results]}

    async def inspect_evidence(self, args: InspectEvidenceArgs) -> dict[str, Any]:
        records = self.store.list_evidence(args.run_id)
        if args.evidence_ids:
            selected = [record for record in records if record.id in set(args.evidence_ids)]
        else:
            selected = records
        return {"evidence": [record.model_dump(mode="json") for record in selected]}

    def definitions(self) -> list[ToolDefinition[Any]]:
        return [
            ToolDefinition(
                name="search_web",
                description="Search the configured read-only web provider for one information gap.",
                args_model=SearchWebArgs,
                risk=ToolRisk.READ_ONLY,
                handler=self.search_web,
            ),
            ToolDefinition(
                name="inspect_evidence",
                description="Read attributable evidence already stored for the current run.",
                args_model=InspectEvidenceArgs,
                risk=ToolRisk.READ_ONLY,
                handler=self.inspect_evidence,
            ),
        ]
