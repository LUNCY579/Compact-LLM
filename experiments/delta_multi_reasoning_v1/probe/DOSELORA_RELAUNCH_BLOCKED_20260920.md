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

---

# AMENDMENT 1 — 2026-09-20T18:5xZ. Three of this note's rows are now wrong.

The DOSELORA-001 owning session corrected two of them and found a blocker this note missed.
All three verified here before acceptance.

## 1. NEW BLOCKER, and it outranks everything else in this note: the pinned pilot is DIRTY

    C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917   HEAD 5b4c22d
     M  experiments/rab001_exploration/process_probe.py                     (+52 lines, 14:08:55Z)
     ?? .../evidence/retention-design-v1/PROCESS_probefix_verify.json       (133611 B, 14:17:46Z)

`controller.py:123-124`:

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("native execution requires a clean checkout")

`--untracked-files=all` is in the invocation, so **the untracked JSON refuses the lane exactly as
hard as the modified `.py`.** The owning session verified it the better way — imported
DOSELORA's `train.py` against the live pilot and called `execution_identity()`, which raised.

> **This blocks EVERY native lane on the host, route-independently — in-place relaunch, a fresh
> experiment root, and the PREPARATION step, because `prepare.py` calls `execution_identity()`
> too.** It is `rrp-clean-checkout-trap` firing exactly as the memory describes.

**This note's line *"the pilot is clean at 5b4c22d"* was true when written at 14:14Z and was
false within minutes.** The files landed at 14:08Z and 14:17Z — the first one *before* this note
was written. **The note asserted a clean checkout it had verified at the branch level and never
re-checked at the working-tree level**, which is the level-mismatch defect again.

**Ownership: neither file is mine.** My six bridge artifacts are **committed** (`git ls-files …
| grep -c bridge001` → 6, at HEAD `5b4c22d`). Both dirty files belong to the
`Fine-tuning model audit and improvement` session — the diff adds `weight_sha256`,
`receipt_matches` and its own *"RECORDS, DOES NOT ENFORCE"* comment. That fix was reported as
committed at `7f188d5`; **that commit went to a different worktree and left this one modified.**

## 2. A TRAP this note did not know about — do NOT run prepare.py while the pilot is dirty

`prepare.py` reaches `execution_identity()` at **L77**. Before it: `prepared.mkdir(exist_ok=False)`
at **L48**, `copy_new()` at **L50**, `write()` at **L55, 62, 74**, `copy_new(PREREG)` at **L63**.

> Running it dirty creates the root, copies the rows, writes DATA_CONTRACT / PLAN / PREREG /
> GRANT, **then raises** — and `mkdir(exist_ok=False)` plus the `'x'` writes refuse every retry.
> **The root name is burned permanently.**

## 3. The route-2 byte-identity risk was OVERSTATED here

§"three legitimate routes" warns that `prepare.py` must reproduce `exact2697` byte-identically or
the new root is a different experiment. **It does not regenerate those rows — it `copy_new()`s
them** from `doseint001-20260919/prepared-dual899/` with a hash assert before the copy (L44-45)
and a sha compare after (L30-34). `microfit-train.jsonl` and `ARM_BUILD.json` are already
byte-identical across parent and `lora001` and match the parent DATA_CONTRACT.

The caveat had the right *shape* — a rebuild is not a byte-identical arm — but it was applied to
a path that **copies** rather than regenerates. `DATA_CONTRACT.json` is the only file in
`prepared/` that is written rather than copied, which narrows the residual risk to one artifact.

## 4. The disk row is stale; the binding blocker moved to host RAM

| gate | at 14:14Z | now | required | |
|---|---:|---:|---:|---|
| host RAM | 6.24 GiB | **3.862 GiB** | 7.00 | **SHORT 3.14 — now binding** |
| free disk | 16.18 GiB | **20.576 GiB** | 17.00 | **OK** — nothing was deleted or compressed |
| gpu lease | HELD | **free** | free | OK |

RAM is consumed by desktop applications, not a model worker. **So of the four blockers this note
recorded, two cleared on their own, one (disk) resolved without intervention, and two new ones
appeared that neither this session nor the owning session can clear alone** — the dirty pilot
needs its owner, and the RAM needs the operator.

## What this amendment does not change

The seal analysis in §"Every route destroys another session's create-only artifacts" stands
unchanged, and the owning session has taken route 2 with a new root
(`highdose-lora002-20260920`), superseding no create-only artifact of `lora001`.
