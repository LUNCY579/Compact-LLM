.PHONY: install test lint typecheck check demo-route demo-hanoi schemas

install:
	python -m pip install -e ".[api,dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests
	python -m ruff format --check src tests

typecheck:
	python -m mypy src

check: lint typecheck test

demo-route:
	python examples/restaurant_route.py

demo-hanoi:
	python examples/tower_of_hanoi.py

schemas:
	python scripts/export_schemas.py
