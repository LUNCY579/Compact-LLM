# Stream validation before native execution

The old controller materialized all 726 TRAIN rows even in its source-only baseline
phase. A fresh process measured 25.1 MiB RSS after controller import and 132.8 MiB after
input validation. Both supervisor and worker retained this unused token payload. Merely
deleting the list left 102.1 MiB RSS in the measured process because of allocator retention.

The controller now validates the same rows sequentially. It retains only IDs, parent IDs,
sequence lengths and target totals long enough to recompute the frozen accounting and
update groups. The supervisor always uses this mode; the worker uses it only for baseline
evaluation. Training/qualification workers still receive the exact complete token rows.
Campaign-specific reference validation still traverses all rows; it does not get an empty
iterator as a substitute for a check.

Fresh measured RSS after streamed validation: **27.4 MiB**. See
`STREAMED_MEMORY_PROBE.json`. There is no torch import in the parent. Source/split/mode,
truncation, ordering, unique-world counts, exposures, processed/supervised totals and
update groups retain their checks. Drift tests run under both retention modes.

The native runtime itself was measured separately, without a model load or forward:
RSS 16.8→662.7 MiB; available host memory 5.134→4.632 GiB, in 3.06 seconds. This
explains why an OS-level free-memory reading close to the minimum can still refuse in
the worker. No guard was reduced: inference preload 4.6 GiB, training preload 6.0 GiB,
and the same post-load host/GPU minima.

Targeted regression run: **74 passed**, `tests-streamed-controller.xml`. The prior full
suite remains 379 passed / one platform skip. The run-summary experiment label is also
corrected to use its actual grant instead of always reporting RRP-001-MICROFIT.

Direct GPU placement is a further possible loader study. Transformers 4.57.1 documents
the `device_map` placement option ([official docs](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model)).
The current dedicated runtime lacks Accelerate and PEFT. Neither was installed, no loader
was replaced, and no parity or peak-memory saving from a different loader is claimed.
Such a study needs its own fixed runtime and native source/parity/resource measurements.
