.PHONY: install prepare validate test smoke verify

install:
	python -m pip install -e ".[train,dev]"

prepare:
	python -m rineya_state_lab data prepare --archive assets/datasets/Rineya_Sequence_Independence_Dataset_v1.0_COMPLETE.zip --output data/prepared

validate:
	python -m rineya_state_lab data validate --data-root data/prepared --strict

test:
	python -m compileall -q src tests
	PYTHONPATH=src pytest -q

smoke:
	PYTHONPATH=src python -m rineya_state_lab smoke-test --work-dir outputs/smoke --force

verify:
	python tools/verify_release.py .
