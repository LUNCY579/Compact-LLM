from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from rineya_harness.examples import fixture_container, run_fixture_research

with tempfile.TemporaryDirectory(prefix="rineya-research-") as directory:
    payload = asyncio.run(run_fixture_research(fixture_container(Path(directory))))
    print(json.dumps(payload, indent=2, default=str))
