from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rineya_harness.examples import build_restaurant_demo, fixture_container

with tempfile.TemporaryDirectory(prefix="rineya-route-") as directory:
    payload = build_restaurant_demo(fixture_container(Path(directory)))
    print(json.dumps(payload, indent=2, default=str))
