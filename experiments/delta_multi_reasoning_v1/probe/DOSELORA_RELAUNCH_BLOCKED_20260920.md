# DOSELORA-001 relaunch — what I did, and the four things that stop the rest

**2026-09-20T14:14Z.** Operator instruction: *"register the gate repair prereg before scoring
anything / register a fresh bounded session for DOSELORA and launch it."*

**Half is done. The other half is blocked by a seal I should not break, and by three resource
gates the runbook says never to lower.**

## Done: GATEFIX-001

`probe/GATEFIX001_PREREG.md`, committed `ffc8079d` **before** any launch, so no DOSELORA arm can
be the first scored under an unregistered rule. No threshold moved.

## Not done: the fresh session. Every route destroys another session's create-only artifacts.

DOSELORA-001 is sealed so that **a session cannot extend its own envelope.** It is working, and
I am the session it is stopping.

| route | what blocks it |
|---|---|
| Edit `SESSION_ENVELOPE_V2.json` | `launch.py:verify_frozen()` binds its **sha256** into `PRELAUNCH_BINDING.json`, alongside `PLAN.json`, `GRANT.json`, `PREREG.md`, `BUILD_GATE_LORA_FINAL.json` and all five executables. Editing it raises *"Frozen prelaunch input changed"* |
| Write a `SESSION_ENVELOPE_V3.json` | `launch.py` reads **V2 by name**. A V3 is never consulted |
| Edit `launch.py` to read V3 | `code_files_sha256` covers it — raises *"Executable changed after prelaunch freeze"* |
| Re-run `finalize_prelaunch.py` to re-freeze | `archive = OUT/'code_before_training'; archive.mkdir()` — no `exist_ok`, and **the directory already exists** → `FileExistsError` |
| Re-run `prepare.py` | `write()` opens with mode **`'x'`** and `copy_new()` with **`'xb'`** — create-only throughout |

> **This is not an obstacle to route around. It is the control that stops an agent extending its
> own deadline, which is exactly what the runbook forbids: *"do not silently extend its
> deadline"*, *"never delete it to retry automatically"*, *"Never lower these gates to
> manufacture a launch."***

## And three resource gates are short right now anyway

Measured, using the envelope's own thresholds — **none lowered**:

| gate | required | measured | |
|---|---:|---:|---|
| session admission | `now + 2700s <= deadline` | deadline passed **5.3 h** ago | `SESSION_CANNOT_FIT_NATIVE_OWNED_WORK` |
| host RAM | 7.00 GiB | **6.24 GiB** | SHORT 0.76 |
| free disk | 17.00 GiB | **16.18 GiB** | SHORT 0.82 |
| GPU lease | free | **HELD** — pid 44380, 2.00 GiB RSS, GPU 3113 MiB at 32% | BLOCKED |

**Something is actively training or probing right now.** Even with a fresh envelope the launcher
would refuse today, and contending for the lease would be wrong regardless.

## The three legitimate routes, and a recommendation

1. **The owning session re-registers** (`ChatBubble-dose-param`, branch
   `research/retention-dose-parameterization-v1`). They built the seal and know the intended
   path. **Recommended** — lowest risk, no artifact destroyed.
2. **A new experiment root** — `highdose-lora002-20260920`, its own envelope, freeze and grant,
   `prepare -> finalize -> launch` run clean. Fully prospective and destroys nothing, but `OUT`
   is hardcoded in `prepare.py` so it needs a code change in their worktree, and **`prepare.py`
   must reproduce the `exact2697` rows byte-identically** or it is a different experiment wearing
   this prereg. Real risk, and not mine to take unilaterally.
3. **Operator authorises superseding named create-only artifacts** — specifically
   `SESSION_ENVELOPE_V2.json`, `PRELAUNCH_BINDING.json`, `BUILD_GATE_LORA_FINAL.json` and
   `code_before_training/`. Each would have to be named, because a blanket "overwrite what you
   need" is how a seal stops meaning anything.

**In all three cases the resource gates must be met first**: the running job has to finish,
~0.8 GiB of host RAM and ~0.9 GiB of disk have to come back.

## What I did not do

I did not run `launch.py` with a fresh wait id. It would have refused with
`SESSION_CANNOT_FIT_NATIVE_OWNED_WORK` and **consumed a create-only wait id** to demonstrate
something already computable from the envelope. Wait ids are a finite, create-only resource and
spending one to produce a foregone refusal is waste, not evidence.
