.PHONY: test compile verify clean

compile:
	python -m compileall -q src tests scripts

test:
	PYTHONPATH=src python -m pytest -q

verify: compile test
	python scripts/validate_install.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info
