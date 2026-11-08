# Amendment 018: bounded admission after framework imports

30 September 2026. The second D2 admission window reached consecutive3.797119 and
3.801788 GiB readings. The owned worker then imported the pinned runtime and
found1.986759 GiB available, below its2.581776 GiB remaining-growth budget.
No constructor, checkpoint load, tensor copy, forward or update ran. Preserve
that terminal failure and its raw source/runtime/resource receipts unchanged.

Register a distinct D2 successor with a resource-only wait after imports, before
any model call. Observe at most60 seconds, poll every2 seconds, and require two
consecutive samples satisfying1.25 + max(0,2.0-currentRSS) GiB available RAM and
4 GiB free GPU memory. While waiting, retain the2 GiB RSS cap,1.25 GiB available
system reserve,0.5 GiB runtime GPU reserve,4 GiB disk and whole-worker600-second
ceiling. The external100 ms supervisor and Windows containment remain active.
Any runtime-floor/source/journal failure stops the worker. Recheck source/HEAD
during observations and immediately before strict preload admission. Timeout is
not admission and never triggers a model retry.

This operational change avoids repeating imports when ambient headroom recovers.
It lowers no guard and changes no checkpoint, dtype, model math, fixture, mask,
position, numerical tolerance or call count. Keep the original four length131
forwards,524 positions and316 logical calls; zero generation/updates/teacher.
Every observed resource sample and partial wait is durable. The wait helper has
no model callbacks and cannot grant numerical qualification or E1 execution.

Preserve the original D2 config/source/CPU controls and all earlier raw failures.
Use a separately frozen config, producer and source review for the wait successor.
Even a completed causal diagnostic is trace-local evidence only; original bulk
likelihood failure remains unqualified. Capture, language/long-context preservation,
sustained native training and current launch validators remain necessary for E1.
The original full research objective and unchanged best are retained.

Correction to the preceding new ledger note: its worker-wall text24.437 seconds
was a transcription error. The raw D2 result, CSV and indexed receipt record
24.422 seconds; those unchanged raw measurements are authoritative.
