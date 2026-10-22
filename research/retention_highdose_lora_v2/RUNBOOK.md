# DOSELORA-002: fresh bounded session for the high-dose LoRA run

> # DO NOT RUN THIS LANE. THE TRAJECTORY IS SPENT.
>
> **Superseded 2026-09-20T19:0xZ.** While this lane was being staged, another session
> executed the same experiment by a different route: an isolated clone
> `highdose-lora001-resume-20260920`, prepared 16:51Z, results 17:49Z, status
> `COMPLETE_COMPARABLE`. It transferred the one-launch permission out of
> `highdose-lora001-20260920` with a create-only
> `NATIVE_DISPATCH_RESERVATION.json` (`status: TRANSFERRED_WITHOUT_LAUNCH`,
> `allowed_training_trajectories_across_both_roots: 1`).
>
> **That single allowed trajectory has been spent.** Running `prepare.py` here and launching
> would be a SECOND training run against a one-run grant, and a second unregistered arm.
> Do not do it.
>
> This directory is preserved deliberately — the executing session's own renewal prereg says
> *"Preserve that checkout and the unfinished retention_highdose_lora_v2 directory."* Keep it
> as the record of the route not taken. `register_session.py` and the byte-level reasoning in
> `SESSION_REREGISTRATION.md` remain valid for any FUTURE experiment that needs to
> re-register a sealed root; nothing below should be executed for DOSELORA.
>
> Results of the run that did happen: acquisition 291 (dose) / 296 (select) of 384,
> `references_retained` 38/47, arithmetic panel 74/150, long context 154/256, beta 1.611,
> relative norm discrepancy 3.3e-08. Primary engineering conjunction FAILED_PREDICTION,
> solely on `references_retained 38 < 40`. Versus its exact norm control: dose net +13
> (p = 0.092), select net +19 (p = 0.016, Holm 0.033) — **both below the registered
> cross-training `|net| >= 20` floor, select by one view.** See `SCOREFIX001_PREREG.md`:
> 38/47 is UNCERTIFIABLE under the registered GATEFIX-001 rule, which `report.py` never
> implemented.

This is `research/retention_highdose_lora_v1/` copied, with `prepare.py`'s output root
repointed at `highdose-lora002-20260920`. Read `PREREG.md`, `AMENDMENT_STORAGE.md` and
`SESSION_REREGISTRATION.md` before any model work. The experiment, bands, predictions,
budgets and gates are unchanged from DOSELORA-001; only the session clock is new.

DOSELORA-001 (`highdose-lora001-20260920`) is preserved untouched and its
`PRELAUNCH_BINDING.json` still verifies. Do not write into it.

Records root: `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/highdose-lora002-20260920`
Code branch: `research/retention-dose-parameterization-v1` in `ChatBubble-dose-param`.
Pinned native pilot checkout: `5b4c22d`. Do not change the pilot, source assets, pinned
venvs, shared GPU lease or resource guards.

## STOP — two blockers, neither clearable from this session

Re-measure both before doing anything. As of 2026-09-20T18:48Z:

**1. The pinned pilot checkout is DIRTY. This refuses every native lane.**

```
C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917
 M experiments/rab001_exploration/process_probe.py
?? experiments/architecture_benchmark_001/evidence/retention-design-v1/PROCESS_probefix_verify.json
```

`experiments/reasoning_posttraining_v1/controller.py:112` `checkout()` raises on any
porcelain output, `--untracked-files=all` included. Verified empirically, not inferred:
importing `train.py` against the live pilot and calling `execution_identity()` raises
`ValueError: native execution requires a clean checkout`. Both files belong to the
`Fine-tuning model audit and improvement` session; `process_probe.py` was committed to a
different worktree, leaving this one modified. Not ours to revert or delete.

> **DO NOT RUN `prepare.py` WHILE THE PILOT IS DIRTY.** It reaches
> `execution_identity()` at L77, *after* `prepared.mkdir(exist_ok=False)` (L48),
> `copy_new` (L50, L63) and `write` (L55, L62, L74). A dirty pilot leaves a
> half-populated root that can never be completed, because create-only `mkdir` and `'x'`
> writes refuse every retry. **The root name is burned permanently and you must move to
> `-003`.** Verify `execution_identity()` succeeds BEFORE running anything.

**2. Host RAM is short.** 3.581 GiB against the unchanged 7.00 GiB launch gate. No model
worker is running and the lease is free; the memory is held by desktop applications
(claude.exe 3.10 GiB over 16 processes, ChatGPT/Codex 1.92 GiB). Never terminate another
process or lower a gate to pass this.

Free disk is **no longer** a blocker: 20.580 GiB against the 17.00 GiB gate, margin 3.58.
It recovered on its own between 14:42Z and 18:48Z with nothing compressed or deleted. The
664-file DOSEINT-001 compression described in `AMENDMENT_STORAGE.md` is already complete;
no further storage action is required or authorised.

## Preflight, in this order

```bash
git -C C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917 status --porcelain --untracked-files=all
```

Must print nothing. Then confirm the gate the trainer actually uses:

```bash
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -c "import sys,importlib.util; sys.path.insert(0,'C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917'); s=importlib.util.spec_from_file_location('t','C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_highdose_lora_v2/train.py'); m=importlib.util.module_from_spec(s); sys.modules['t']=m; s.loader.exec_module(m); print(m.execution_identity()['pilot_checkout'])"
```

Must print `5b4c22dae78f587c41575269f419087b0c7997f6`. Then check resources: >=7.00 GiB
available host RAM, >=17.00 GiB free disk, and `rineya-gpu-0.lock` absent.

Only when all three pass, and with enough of the 12600 s session left to fit the 2700 s
native reserve, proceed. The envelope clock starts at step 1, so do not start until the
gates are actually passing.

## Registration and launch

```bash
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_highdose_lora_v2/prepare.py
```

```bash
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_highdose_lora_v2/register_session.py
```

`register_session.py` copies `AMENDMENT_STORAGE.md` into the root (nothing else copies it,
and `finalize_prelaunch.py` hashes it), then writes `SESSION_ENVELOPE_V2.json` applying the
registered storage amendment to the reserve `prepare.py` wrote. It compares every other key
against DOSELORA-001's envelope and refuses on any difference outside the session clock, so
the fresh session cannot buy itself a larger budget. Copy `SESSION_REREGISTRATION.md` into
the root at this point too.

```bash
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_highdose_lora_v2/finalize_prelaunch.py
```

```bash
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_highdose_lora_v2/launch.py --wait-id initial
```

A fresh root has no prior terminal, so `--wait-id initial` is correct and writes its wait
log to the root. Wait IDs are create-only; do not burn one to demonstrate a refusal you can
compute from the envelope. The launcher checks the full 2700 s native reserve before
readiness and again immediately before dispatch, and distinguishes starting the controller
from proof of a model-worker launch. Never delete `NATIVE_DISPATCH_RESERVATION.json` to
retry; inspect its owner and the authoritative native receipts instead.

## After a genuinely successful qualified run

```bash
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_highdose_lora_v2/posttrain.py
```

LoRA merge, norm record, transparent compression, exact norm control, eight sequential
probes. LoRA uses `BUILD_GATE_LORA_FINAL`; the scalar uses `BUILD_GATE_NATIVE_NORM`. Beta is
frozen before inference, admissible `[0,2]`, never clipped or searched. One total inference
retry is allowed after a verified failure, with a fresh label and directory.

```bash
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_highdose_lora_v2/report.py --root C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/highdose-lora002-20260920
```

Always pass `--root` explicitly.

## Scoring: report.py does NOT implement GATEFIX-001

`report.py:307` still applies the two-valued rule
`retention["references_retained"] >= 40 and retention["panel_correct"] >= 62`, and
`report.py:315` bands `references_retained` against `(37, 45)`. There is no confidence
interval anywhere in the file: grepping for `CERTIF`, `UNCERTIF`, `clopper`, `interval` and
`confidence` returns nothing.

GATEFIX-001 (`ChatBubble-dmrf/experiments/delta_multi_reasoning_v1/probe/GATEFIX001_PREREG.md`,
commits `ffc8079d` + `0bce03e3`) is REGISTERED and makes the `references_retained` conjunct
three-valued by where the exact 95% Clopper-Pearson interval sits relative to the threshold.
**The code and the registered rule disagree.** Reconciling them is a scoring-rule change and
must be registered before any arm is scored — not applied silently while reporting.

Independently recomputed at n=47, T=37 (exact Clopper-Pearson, reproduces GATEFIX-001's
published table to the decimal):

| verdict | k | count |
|---|---|---:|
| CERTIFIED_FAIL | 0-30 | 31 |
| **UNCERTIFIABLE** | **31-42** | **12** |
| CERTIFIED_PASS | 43-47 | 5 |

The uncertifiable band is 12 of 48 possible outcomes (25.0%); the separately quoted "76%"
describes how many of the 84 arms on record actually land there, which is a different
quantity — do not conflate them.

Consequence for this experiment: of `PREREG.md`'s registered retention band 37-45, only
**k = 43, 44, 45** can yield CERTIFIED_PASS. The registered primary target `retained >= 40`
admits 40, 41 and 42, all of which are UNCERTIFIABLE. **An arm can therefore meet its own
registered primary target and still be UNCERTIFIABLE on the gate conjunct.** Expect to
report the decomposition rather than a verdict, and do not let UNCERTIFIABLE read as either
a pass or a failure.
