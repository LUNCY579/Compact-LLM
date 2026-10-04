# Staged CPU long-input adapter

All implementation, controls and output in this directory remain staging work.
Package root sources, tests, configs and existing evidence were not changed.
No model, framework, scorer, oracle sidecar, GPU or native worker ran here.

`streaming_inputs.py` validates the complete fixed 57,864,922-byte policy source
with bounded 64-KiB buffering and a 1-MiB line ceiling. It checks strict public
fields, UTF-8/JSON, duplicate/nonfinite values, hashes, row order, exact all-one
masks and visible-message retokenization before creating output. Every output
row reserves 32 tokens within its exact 1024/2048/4096 total envelope. Nothing is
truncated, padded, packed or dropped. Source/tokenizer drift or final resource
failure leaves no success receipt.

The fixed source contains 2304 prompts: 768 each at 992, 2016 and 4064 tokens,
or 5,431,296 prompt positions. The adapter reuses the pinned local
tokenizers/Jinja prompt profile, binds `cpu_tokenizer.py` bytes and original
receipt assets, and avoids importing its supervised verifier. The historical
512-token field in that tokenizer binding grants no long native qualification.

The buffered attempt in `inputs_root1_buffered_cpu_20261004` completed in
15.235 seconds, with measured process lifetime peak 155,512,832 bytes. Its
39,259,534-byte arrays are byte-for-byte identical to the preserved 253.359-second
unbuffered attempt. The 13 focused CPU controls passed without skips; they include
path/reparse rejection, late-row private data/envelope failure, bounded malformed
records, source/tokenizer/order drift and failure of the final resource guard.
Snapshots and parity hashes are in `buffered_attempt_preservation_20261004`.
The earlier completed attempt and initial buffering-control failure remain
preserved separately.

Run focused controls:

```powershell
python -B -m unittest discover -s experiments/rineya_synth_centralized_reasoning_v1/artifacts/long_input_work -p test_streaming_inputs.py -q
```

`validate_packet()` returns an `Admission` and tokenizer; `materialize()` accepts
them and a new direct staging output leaf. Injected tokenizers/source pins and
default constructed admissions are explicitly labeled `CPU_FIXTURE`. A future
production consumer must independently reconstruct fixed source/tokenizer pins
and qualify its own execution path; it must not treat the preparation flag or
an externally constructed admission as launch authority.

Native execution, long-context numerical qualification, source-correct cohort
enrollment, statistics and human-language qualification remain separate and
closed. These arrays establish CPU input preparation only.
