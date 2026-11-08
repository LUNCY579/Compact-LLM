# Supa adaptive compute v2

One shared codebase, two scientific callers. Track A asks whether bounded native
recurrence generalizes. Track B first asks whether an actual frozen 4B layer
transformation can be aligned faithfully. Their calibration, efficacy banks,
registrations, selection rules and evidence are independent.

Start with [STATUS.md](STATUS.md), the [original goal](00_audit/master_goal.md),
the [component audit](00_audit/existing_work_inventory.md), the
[data design](data/DESIGN.md), and the two protocols:
[native](native_armr/protocol/preregistration.md) and
[cross-scale](cross_scale/protocol/transformation_alignment_v2.md).

## Shared engineering

`shared/legacy.py` imports the preserved ARMR wrappers, hash/scoring/statistics
primitives and streamed actual-HF donor extractor. It does not copy them.
`model_loading`, `hidden_capture`, `generation`, `integrity`, `guards`, `hashing`,
`manifests`, `datasets`, `scoring`, `statistics`, `timing`, `evaluation` and
`reporting` supply common infrastructure. Native operators and decisions live in
`native_armr/`; bridge objectives and matrix selection live in `cross_scale/`.

Historical `experiments/cross_scale_layer_graft_v1/` was already **untracked** at
task start. It remains an untouched local dependency, with source and artifact
digests in [historical_bytes.json](00_audit/historical_bytes.json). A clone of the
new branch alone will need that exact preserved dependency to execute Track B.
It is not falsely attributed to the starting Git commit. No historical folders
or canonical checkpoints are reorganized or saved.

The runtime is the existing `Chat-Bubble/.venv`: torch 2.12.1+cu126 and
transformers 5.14.1, CPU FP32/eager, four threads, no cache. The installed Qwen3
source hash and canonical checkpoint/config/tokenizer sidecars are checked against
the qualified historical replication. The PATH Python has different versions.
All expensive jobs share a machine lease and require at least 3 GiB available
physical memory at start. An individual native Torch forward is not preemptible;
deadline checks surround forwards, including prefill, EOS and answer NLL. Expired
work is failed evidence, never an accepted answer.

## Frozen inputs and scientific limits

Each track has 128 calibration scenarios (96 fitting / 32 alignment holdout),
128 architecture scenarios, 1,024 independent primary scenario realizations and
128 linked robustness diagnostics. The eight families contain 32 structures;
eight structures are excluded from engineering. Diagnostics never inflate primary
N. A separately implemented fact parser solves all rendered scenarios and rejects
even consistent corruption of the saved answer and scorer.

Both manifests are under `data/generated/v2/<track>/manifest.json`. They bind
exact JSONL bytes, semantic generator and strict scorer hashes. The banks are
English and synthetic. Grounded ordinary QA does not measure broad factual
knowledge; bounded retrieval does not establish long-context competence. Shared
templates limit external generalization. These limits remain even if results win.

Exact McNemar is checked against the
[SciPy two-sided binomial implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
Scenario bootstrap, family effects, NLL, identity, preservation and descriptive
accuracy/cost frontiers accompany aggregate accuracy. See
[power_planning.json](data/generated/v2/power_planning.json): at N=1,024 a two-point
gain has approximately 82%, 53% or 30% planning power under assumed discordance
of 5%, 10% or 20%. This is not observed power. The corresponding 80%-power
minimum effects are about 1.96, 2.77 and 3.92 points.

## Reproducible commands

Run from this repository in PowerShell:

```powershell
$researchRepo = 'C:/Users/clclc/BusinessProject/ChatBubble-cross-scale'
$researchPython = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
Set-Location -LiteralPath $researchRepo
$programme = 'experiments/supa_adaptive_compute_v2'
$nativeManifest = "$programme/data/generated/v2/native_armr/manifest.json"
$donorManifest = "$programme/data/generated/v2/cross_scale/manifest.json"

& $researchPython -m pytest -q "$programme/tests" "$programme/native_armr/tests" "$programme/cross_scale/tests"
& $researchPython -m pytest -q experiments/anchor_routed_middle_recurrence_v1/tests experiments/cross_scale_layer_graft_v1/tests
& $researchPython "$programme/scripts/audit_history.py" --snapshot "$programme/00_audit/historical_bytes.json" --verify
```

The data is already frozen. `scripts/build_banks.py` intentionally refuses to
overwrite it. To reproduce bank construction without changing the registered
files, use an empty temporary output directory, then compare hashes. A scientific
change requires a new version and freeze; do not edit failed questions.

Registrations are created once, **after** tests/input validation and **before**
model execution, then committed locally with the new code/data/protocols:

```powershell
& $researchPython "$programme/scripts/freeze_protocols.py" --track native_armr --manifest $nativeManifest
& $researchPython "$programme/scripts/freeze_protocols.py" --track cross_scale --manifest $donorManifest
```

Existing registrations must not be recreated. `verify_registration` requires
exact source, textual protocol, config and manifest hashes and committed files.
Changing a shared implementation invalidates both old source bindings; it never
silently reinterprets their evidence. New registrations are prospective.

Track A can be planned or resumed serially. Architecture qualification requires
all 128 cases and all five registered arms; accuracy cannot be used to delete a
control. Confirmation stays closed until complete mechanical admission:

```powershell
& $researchPython "$programme/scripts/run_native_campaign.py" --manifest $nativeManifest --role architecture_dev --plan
& $researchPython "$programme/scripts/run_native_campaign.py" --manifest $nativeManifest --role architecture_dev --max-wall-seconds 3600
# Repeat exactly the preceding command to resume missing shards.
# Once architecture qualification is complete:
& $researchPython "$programme/scripts/run_native_campaign.py" --manifest $nativeManifest --role confirmatory_dev --max-wall-seconds 3600
& $researchPython "$programme/scripts/run_native_campaign.py" --manifest $nativeManifest --role confirmatory_diagnostics --max-wall-seconds 3600
```

The campaign budget stops between shards; it never discards a running forward.
Native shard resumption creates new attempt directories for incomplete work and
skips only completed compatible shards. It rejects duplicate arm/shard evidence.
The complete final bank is independently rescored by the
[native auditor](native_armr/README.md), including A3 vs A0 and matched AM,
secondary Holm comparisons and the preservation bound. No partial-bank GO.

Track B's next causal experiment is the full calibration-only 16-pair,
six-objective matrix. The [Track B commands](cross_scale/README.md) include
capture, status-aware pair resumption and independent complete-matrix aggregation:

```powershell
& $researchPython "$programme/cross_scale/scripts/run_alignment.py" preflight
& $researchPython "$programme/cross_scale/scripts/run_alignment.py" capture --run-id alignment-v2-001 --dataset-manifest $donorManifest
# Then run the registered 16-pair loop and aggregate shown in cross_scale/README.md.
```

Mechanism selection needs every pair and B0/B1/B2/B3/BS/BR control. It uses only
held-out calibration alignment. No selected bridge, dose or capability result
may be inferred from partial matrix cells. A passing matrix licenses a separately
frozen small generation study with the registered gate grid; generation and
4B/8B escalation are not automatically enabled. The current caller implements
Stage 1; later generation is deliberately gated until its prerequisite is real.

## Interrupted workers and evidence

Every attempt owns its directory, with exclusive started/final manifests, raw
case files or matrix artifacts, and failure receipts. First logits/activation
tensors stay locally under ignored paths and are required by hash for resumption
and independent audit; Git records their hashes and compact raw metrics.

An interrupted process can leave `.runtime.lock`. Read its `pid`, verify that
exact process is absent with `Get-Process -Id`, then remove only that lock with
`Remove-Item -LiteralPath`. A timeout observing a live worker does not authorize
restarting it. Never kill an unrelated process or infer completion from a lock.
Track B has an explicit `abandon-pair` command that checks the original PID is
absent and adds an immutable abandonment receipt before a retry. Failure evidence
is retained. No worker mutates another worker's evidence directory.

Sealed evaluation, routing, MLA/MHA/MTP, PEFT, backbone training, donor
distillation and Qwen3-8B remain closed. Their future eligibility follows the
original goal, not an implementation convenience.
