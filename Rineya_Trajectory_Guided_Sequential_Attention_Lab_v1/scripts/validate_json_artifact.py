#!/usr/bin/env python3
"""Validate one or more JSON artifacts against a Draft 2020-12 schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import jsonschema


def _load_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--document", type=Path, action="append", required=True)
    args = parser.parse_args()

    schema = _load_object(args.schema.resolve())
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    validated: list[str] = []
    for document_path in args.document:
        resolved = document_path.resolve()
        validator.validate(_load_object(resolved))
        validated.append(str(resolved))
    print(
        json.dumps(
            {
                "schema": str(args.schema.resolve()),
                "validated": validated,
                "count": len(validated),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
