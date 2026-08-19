from __future__ import annotations

import time
from pathlib import Path

from ..utils import iter_jsonl
from .base import GenerationRequest, GenerationResponse


class ReplayBackend:
    name = "replay"

    def __init__(self, path: str | Path, *, id_field: str = "request_id", text_field: str = "text") -> None:
        self.path = Path(path).expanduser().resolve()
        self.outputs: dict[str, str] = {}
        for row in iter_jsonl(self.path, tolerate_truncated_tail=True):
            if id_field in row and text_field in row:
                self.outputs[str(row[id_field])] = str(row[text_field])
            elif id_field in row and "raw_output" in row:
                self.outputs[str(row[id_field])] = str(row["raw_output"])

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResponse]:
        responses: list[GenerationResponse] = []
        for request in requests:
            start = time.perf_counter()
            if request.request_id not in self.outputs:
                responses.append(
                    GenerationResponse(
                        request_id=request.request_id,
                        text="",
                        latency_ms=(time.perf_counter() - start) * 1000,
                        backend=self.name,
                        error=f"No replay output for {request.request_id}",
                    )
                )
            else:
                responses.append(
                    GenerationResponse(
                        request_id=request.request_id,
                        text=self.outputs[request.request_id],
                        latency_ms=(time.perf_counter() - start) * 1000,
                        backend=self.name,
                        metadata={"source": str(self.path)},
                    )
                )
        return responses

    def close(self) -> None:
        return None
