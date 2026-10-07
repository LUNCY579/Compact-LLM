# Owned import postrun preservation and identity-bound audit

The new postrun CLI now joins the actual scoped supervisor to preservation and
the packet-bound terminal audit. It validates the registered preservation target
before supervision, requires every selected future raw/lease/source-snapshot
exception, preserves first, then accepts only the terminal consumer's unqualified
measurement result. A supervisor zero exit is insufficient.

Expected measurement identity is derived independently from the qualified context,
exact measurement producer descriptor, owned receipt and ownership operation's
executing PID/birth. The terminal successor compares that expectation before raw
effect replay. Context, producer and birth copied from raw output cannot serve as
their own oracle. Admission denial without a resumed worker retains the original
frozen raw receipt and has no fabricated worker result or terminal audit.

Seven authored and Root controls pass. [Independent six controls](evidence/import_dispatch_postrun_reviewed_cpu_20261007/independent_review/INDEPENDENT_POSTRUN001.json)
pass in 3.945 seconds with forty selected pins unchanged. They execute the pure
terminal/Reader/raw replay and original CPU writer, with explicit inert context,
preserver and supervisory records. Identity, grant, target, preservation and
partial/failed-result cases refuse or remain partial; no native dispatch occurs.
The new helper is a separate public entry; the older supervisor-only dispatcher
remains historical code and was not promoted as a postrun audit.

Root applied [import_dispatch_postrun.py](import_dispatch_postrun.py) and the
identity-bound terminal successor, then loaded their actual public sources and
confirmed absent-grant refusal before launch.
[Preservation](evidence/import_dispatch_postrun_reviewed_cpu_20261007/ROOT_PRESERVATION_001.json)
binds 36 files and 168,944 bytes, including predecessors and failed CPU provenance
fixtures. Driver is `560cd7f6e17de962d9782e6e44731445b3c5411ab10984c3bac2c5df1ae8e4cb`;
terminal is `3bd489a1f42458a931d72620f023e3dcd0443c501fe57059378d7e855359e94e`.

Final source/runtime/config/registration construction, immutable dispatch anchors,
fresh complete byte/roster/resource checks and separate owner permission remain
pending. Actual main/preload/provider/baseline/terminal facts belong to that later
owned process. No native DLL/API, psutil, Torch, CUDA, model, training or scoring
ran. The original condition-diversity transfer and retention result is unmeasured.
