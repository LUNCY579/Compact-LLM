from __future__ import annotations

import time

from .base import GenerationRequest, GenerationResponse


class ReferenceBackend:
    name = "reference"

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResponse]:
        responses: list[GenerationResponse] = []
        for request in requests:
            start = time.perf_counter()
            if request.reference_output is None:
                responses.append(
                    GenerationResponse(
                        request_id=request.request_id,
                        text="",
                        latency_ms=(time.perf_counter() - start) * 1000,
                        backend=self.name,
                        error="ReferenceBackend requires reference_output",
                    )
                )
            else:
                responses.append(
                    GenerationResponse(
                        request_id=request.request_id,
                        text=request.reference_output,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        backend=self.name,
                        metadata={"oracle": True},
                    )
                )
        return responses

    def close(self) -> None:
        return None
