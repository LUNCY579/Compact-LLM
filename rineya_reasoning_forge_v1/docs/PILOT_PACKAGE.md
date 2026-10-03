# Pilot-only package

This archive contains the 512 training and 128 validation pilot records, their audit metadata, source code, tests, preview, and documentation. The pilot is a subset of the full 10,000-example release, not new independent data.

The complete data splits and paired evaluation probes are omitted to keep this package small. Full-corpus reports describe the full release; obtain the full archive before running `verify.py --root .` against all canonical splits. The unit tests create temporary test data and run without the full release.

Start with README.md and docs/TRAINING_PLAN.md. Before optimization, run prepare_for_model.py with your exact local Rineya tokenizer and inspect the trainer's assistant-only loss masks. Real tokenizer lengths, hardware fit, independent human review, and model efficacy remain unverified.
