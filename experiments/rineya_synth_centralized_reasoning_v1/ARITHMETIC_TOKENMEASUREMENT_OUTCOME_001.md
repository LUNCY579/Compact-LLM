# Arithmetic CPU tokenizer measurement, original output preserved

The reviewed tokenizer-only attempt measured all512 frozen arithmetic rows with
the exact local Titan tokenizer/template. No row overflowed: maximum prompt225,
completion including EOS9 and prompt+completion234 tokens, within480/32/512.
Prompt-only policy arrays remain separate from exposed CPU oracle diagnostics.

[Original outcome](evidence/arithmetic_tokenmeasurement_attempt001_20261009/ROOT_PRESERVATION.json)
preserves all four output files, reviewed source, current passive inventory,
fixtures and failed preparation history. Before/after identity for1,392 selected
files was equal. The original cooperative ResourceWatch recorded28.895209s and
161,841,152-byte peak working set after result writes, satisfying600s/512MiB.
These are CPU process measurements, not hard containment or native model fit.
No weights/config, framework/model/GPU/optimizer/scoring/M0 operation occurred.

A late source review identified a serialization issue while the attempt was
already running: `measurement.json` omits policy/oracle arrays but keeps the
full-result SHA. The projected metadata is not self-sealed. Root verified the
original receipt's file hashes and reconstructed the full result from both
saved arrays; that full digest matches. Preserve this original point and create
an additive corrected projection with its own seal, explicit full-result digest
and array-file references. Do not silently rewrite original output or infer
native qualification from successful tokenizer mechanics. The corrected
projection and independent outcome audit remain pending in this record.
