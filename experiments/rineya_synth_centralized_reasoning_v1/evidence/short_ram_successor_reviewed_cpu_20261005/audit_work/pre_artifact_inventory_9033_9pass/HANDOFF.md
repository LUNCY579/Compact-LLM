# New-profile-only auditor preparation

The original auditor18f6a48fdd3e365523bdf4e5d03ffebf5f2e4a19366676382b39d483b23795ab
remains unchanged and byte-preserved as `original_auditor_18f6.py.txt`. Its old
source83/runtime64 path remains the legacy authority for old receipts. This
separate auditor admits only the reviewed short RAM2 collector profile; it does
not dynamically accept an arbitrary inventory length or relabel an old scope.

Before any future terminal audit, a trusted root/reviewer must freeze a new
write-once registration under package `configs/`, after the reviewed source
commit and scope file are fixed. Do not create that actual registration from
these synthetic fixtures or while the native qualification source is live.
Select its file hash and profile body hash independently. Hash equality binds
the supplied bytes; it does not establish approval of self-signed audit authority.

Required JSON fields are `schema_version:1`,
`id:SYNTH_SHORT_RAM2_INDEPENDENT_AUDIT_INVENTORY_001`, the exact new `contract`,
`root_path`, `package_path`, `config_path`, `config_file_sha256`, `source_commit`,
`profile_body_sha256`, `qualified_runtime64_sha256`, `inventories`, `counts` and
canonical body `sha256`. `inventories` holds exact full path→SHA256 maps for
`source`, `runtime`, `decoder`, `data`, and `model`, copied from the reviewed root
contract's frozen scope. Source count is separately frozen to that complete map.
Runtime must be exactly65: the unchanged independently frozen64 core plus the
executed singleton AST pin. Decoder17/data5/model7 remain exact.

Early admission compares all maps and the entire expected profile, including
helper/callback/CPU policy pins, checks the reviewed successor/helper source
closure, verifies the runtime extension/core, and rejects success receipts that
claim any abort diagnostic. Unknown/old profile, changed pins, omitted runtime,
source substitution and diagnostic contradictions reject before the heavy maps.
The terminal path requires the actual reviewed integrated root contract, so
staged metadata cannot accidentally supply a different source roster.

The terminal/preservation signal and independently selected config/preservation
hashes remain mandatory. Full64 row/order/prompt/token, strict raw finish/EOS/cap,
all per-token durable prefixes, decoded generated-only response, logprob/numerics,
API/physical costs, resources, final310-base/zeroB/activation proof, source/runtime/
decoder/data/model byte closure, cleanup and immutable inventories are retained.
The original body from event joins through final closure is compared directly in
a CPU control. A success cannot contain `SHORT_RAM_ABORT_DIAGNOSTIC.json`.

This preparation runs tiny metadata controls only. No actual old full decoder
audit, model byte hashing, runtime binary hashing, framework/model/GPU execution,
or correctness grading was performed. DEV_FINISH_POLICY_001 remains design only;
raw cap32 finish checks establish collection metadata, not correctness treatment.

```powershell
python -B -m unittest discover -s experiments/rineya_synth_centralized_reasoning_v1/artifacts/short_ram_successor_work/audit_work -p test_auditor_successor_controls.py -v
```

Any future full audit needs explicit terminal/preservation authorization plus
the independent registration anchors. Keep old failed scopes and M0 receipts;
no partial source/data population or altered numeric/resource gate is accepted.
