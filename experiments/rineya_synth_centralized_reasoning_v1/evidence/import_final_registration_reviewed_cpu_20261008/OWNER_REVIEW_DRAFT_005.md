# Proposed bounded Torch import observation

Proposed run: `prefix_input_owned_import_observation_001`.
Implementation: `8dccdb88870f85f98f6498d432a85c818803387b`.
The closed recipe is `EXACT_IMPORT_RUN_RECIPE005.json`; its SHA-256 is
`32314c147e92546d7459994eaf82ee0fc830144277ee627acaf7d47820872683`.

This step measures the installed framework's import and Windows library effects
inside one owned supervised worker. It records the framework version, resources,
import boundaries, cleanup and terminal audit. Torch import may reach CUDA driver
interfaces. The scope includes no explicit user CUDA context, tensor allocation,
model, forward/backward pass, optimizer, decoding, scoring or protected bank.

The unchanged limits are 600 seconds, 4 GiB free GPU memory at admission,
0.5 GiB during work, 3.3125 GiB available RAM before the worker, 1.25 GiB RAM
reserve, 2 GiB worker process-tree RSS and 4 GiB free disk. Original sampler
cadence, numerical checks, owned-process containment and failure preservation
remain binding. Admission failure ends the attempt; unrelated jobs are not stopped.

If approved, instantiate a separate run manifest from these reviewed maps and
rules, attach the completed current byte-review receipt, commit the dispatch
packets, and bind the exact nine packet/dispatch anchors to the actual direct
human approval receipt. Because the three run packet paths are under ignored
`artifacts`, force-add only those exact approved config/runtime/registration
files with `git add -f -- <exact paths>`; do not add all artifacts. The distinct
dispatch commit must contain those exact packet bytes. Source-roster metadata
and external byte-review/owner receipts retain their selected file/body anchors
outside that dispatch cycle. Revalidate current bytes, directories and admission
before one launch through `import_dispatch_postrun.py`. Preserve raw results on
every failure; for resumed workers preserve leases/source snapshots and run the
terminal audit. A no-resume denial retains its original receipt without claiming
a semantic terminal audit. The closed recipe stays historical. A new implementation, changed
resource rule or expanded operation requires a separately reviewed proposal.

This approval would cover one import-observation attempt only. Tensor/model
qualification and M0/M1/M2 training remain later separately registered steps.
No result from this observation can establish learning or transfer.

Preparation status: the full current byte window, independent source/roster/cache
review and full qualifier CPU replay passed. Permission is absent and launch
remains disabled. This draft is not an approval receipt.
