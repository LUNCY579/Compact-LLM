# One-prefix CPU core: reviewed, native entry closed

The [preserved CPU pipeline](evidence/prefix_input_core_reviewed_cpu_20261005/pipeline.py)
uses injected tensor/resource APIs and the committed422-token diagnostic input.
Its source-faithful checks retain short-circuit truthiness, the second IDs-shape
read during mask comparison and `str(device)`. It records actual predicate
metadata and host hashes separately from its summary. Native dispatch refuses
before paths or APIs; no actual Torch tensor, model, GPU or grading ran.

The [builder16-control run](evidence/prefix_input_core_reviewed_cpu_20261005/CPU_VALIDATION_001.json)
passes on frozen source. Root independently
[ran all16](evidence/prefix_input_core_reviewed_cpu_20261005/ROOT_CPU_VALIDATION_001.json)
in1.437seconds and [reproduced16 from the preserved copy](evidence/prefix_input_core_reviewed_cpu_20261005/ROOT_PRESERVED_COPY_REPRODUCTION_001.json),
with no skips or framework imports. The
[independent six-control review](evidence/prefix_input_core_reviewed_cpu_20261005/prefix_cpu_focused_review_1791225134351873800.json)
checks source ordering, error fidelity, final acceptance and native refusal on
identical bytes. These overlapping counts are not independent study results.

Review corrected final-write durability/self-accounting, result/error/time
projection, original-error preservation during failed logging, counted reference
release, terminal value/floor checks, module-origin ambiguity, immutable initial
identity, actual operation order and separate secondary failures. The rejected
source variants remain in the [history](evidence/prefix_input_core_reviewed_cpu_20261005/ROOT_PRESERVATION.json).
Ordinary primary messages are exact only when representable within4096 characters;
truncated or unavailable messages are explicitly incomplete failed evidence.

The code derives35 success operations and eight guard sites, with caps of seven
initialization and seven memory queries. One optional failure-frame capture gives
a conservative58 phases/116 markers within128. The normal fake run observes
seven guard initialization and four guard memory calls, plus terminal/preload
queries. Resource floors remain active in fixtures. Durable final accounting
includes its own projected/actual write. Earlier worker JSON is provisional:
late publication/close/final-write failure prevents acceptance even if that
earlier file says COMPLETE. Missing terminal data after failed preflight is not
invented.

Reproduce only CPU controls from the repository root:

```powershell
python -I -B -m unittest discover -s experiments/rineya_synth_centralized_reasoning_v1/evidence/prefix_input_core_reviewed_cpu_20261005 -p test_prefix_input.py -v
```

The separate native worker/supervisor, exact source/runtime inventory, current
manifest/admission and independent terminal auditor remain absent. Existing
D001 and M0 authorization cannot be reused for new tensor work. This core proves
no native execution, prior CUDA/TDR cause, model recovery or fit, M0 membership,
training, correctness, retention or transfer. Full M0/M1/M2 evidence remains
required.
