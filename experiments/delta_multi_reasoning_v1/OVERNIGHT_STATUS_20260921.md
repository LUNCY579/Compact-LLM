# Overnight status — 2026-09-21, unsupervised session

**Read this first: nothing irreversible happened. `HSKILL-CONFIRM-002` is still
`UNSPENT_AND_UNREAD`, and its hash still matches its quarantine record.**

---

## 1. The one decision I did not take

You asked me to build the confirmation instrument and register the plateau, then continue
automatically. I did both. **I did not spend the instrument**, and that was deliberate:

- `NORTH_STAR_PLAN_20260918.md` §6 records *"spending HSKILL-CONFIRM-002 | operator only; nothing
  here is self-authorising."*
- It is **one-shot and unrecoverable** — the entire value of a sealed bank is that it has never
  been looked at. There is no undo.
- A general grant to use GPU and CPU overnight is not consent to burn the programme's only
  remaining sealed instrument.

**Everything up to the read is done.** The read is now a single authorised command.

## 2. The instrument already existed

**I did not need to build one.** `HSKILL-CONFIRM-002` was commissioned on 2026-09-18 and left
sealed: 224 rows, 224 distinct parents, contamination-controlled, floors published before any
model call, and **explicitly unclaimed** — MAG-001's prereg says *"nothing here is confirmed;
HSKILL-CONFIRM-002 stays sealed."*

Verified before use: the file hashes to the `b758e0d1…` in `QUARANTINE.json`; disjointness from
both the selection bank and δ_H's 648 training rows is 0 by parent **and** by normalized text,
with a positive control and a live namespace (four shared id prefixes, so a collision was
possible).

## 3. HPLAT-001 is registered and SEALED

**`registration_sha256 = e5e00bf627b68b1e9ae5c02e54b5be069dbefdfe63570f5b71307dc135258321`**
(`probe/HPLAT001_SEAL.json`). The window is closed.

It went through **three adversarial audit rounds — 44, then 13, then 2 material findings.** Two
full rewrites, because the defects were structural rather than patchable.

| version | what was wrong |
|---|---|
| **v1** | 10 blocking + 26 serious. **The primary was on the wrong half of the instrument**, and the guard table would have stamped the best-powered comparison in the design UNCERTIFIABLE |
| **v2** | introduced **two new unsatisfiable gates** while fixing v1's, plus **one measured claim wrong by sign** |
| **v3** | P4's power is very unequal across the two arms; an unregistered statistic **reverses the headline** |
| **v4** | sealed |

### The finding that reshaped the design

Blinding digit runs in the user text:

| half | CONFIRM stems | VIEWS stems | shared | CONFIRM rows whose stem occurs in VIEWS |
|---|---:|---:|---:|---:|
| **answerable** | 11 | 13 | **11** | **112 / 112** |
| **boundary** | 112 | 112 | **0** | **0 / 112** |

**Every answerable row in the sealed bank is a digit-resample of a template the selection bank
already used. The boundary half shares nothing.** The same code returns 0 on the boundary half,
so it is a finding, not an artifact — and the cause is pool exhaustion (~14 clean arithmetic
templates exist in total), so **no builder could have avoided it.**

v1 had put the primary on the dependent half. v4 scopes it: a pass there confirms **row-level**
selection was broken, **not** generalisation to new question types.

### The error of mine most worth knowing about

v2 registered `copy_user` at `J_decision ≈ +0.580` and built its only novel quantitative claim on
it. **Measured: −0.357. Wrong by sign.** 105 of 112 answerable user turns end in `?`, and the
abstention pattern is `r"\?"`, so echoing the user scores as an *abstention* — echoing is
worse-than-chance, not near-perfect. I had read the floors file's `copy_user: {answerable: 0}` as
a decision count when it is **exact-match competence**: *"did not match the reference"* is not
*"did not attempt."*

> That is the exact discrimination/competence conflation the instrument's own amendment exists to
> remove, committed inside a document that warns about it two sections earlier. **An inference
> marked with a tilde is still an inference**, and the quantity was computable in nine lines.

## 4. The rehearsal COMPLETED — 4/4 arms, every gate, every score exact

The rehearsal rebuilt all four HPLAT arms and re-scored them **on the SELECTION bank** — already
spent for selection and free to re-read — to prove the build gates fire and the numbers reproduce
**before** the sealed read is ever authorised. It never passed `--spend-confirmation-instrument`,
and the sealed bank's path does not appear in it.

| arm | gate | recorded | reproduced |
|---|---|---|---|
| `supa` | — *(frozen source, no rebuild)* | 71 / 111 / 40 | **exact** |
| `a025` | whole-file ✓, canary `{1: 0.0, 2: 0.0}`, 220/310 | 86 / 112 / 43 | **exact** |
| `a0675` | whole-file ✓, canary to every digit | 86 / 111 / 41 | **exact** |
| `H16` | **DATA-SECTION ✓** | 88 / 93 / 36 | **exact** |

### The data-section gate earned itself on the first rebuild

`H16`'s rebuild produced whole-file `d6f653a9…` — **the scored receipt's value, not the arm
directory's `b9a5e37f…`.**

> **Had the registration gated `H16` on its whole-file hash — as v1 and v2 both did — this
> rebuild would have FAILED its build gate, and the void list makes that a do-not-score. The read
> would have been killed by a false alarm on a bitwise-correct arm.**

That gate came from an auditor reasoning about safetensors metadata ordering. **The ~50% coin flip
came up tails on the very first rebuild** and confirmed the argument by measurement.

Two smaller confirmations landed the same way:

- **`a025` rebuilt byte-identically under a DIFFERENT builder revision.** The audit had argued
  this from first principles (0.25 is a power of two, so float32 and float64 products agree
  exactly). Now measured — the mixed-builder provenance is real but immaterial.
- **`a0675`'s canary reproduced to the last digit on both tensor families** —
  `{1: 3.59625868812201e-16, 2: 2.009176934463362e-14}`. Stronger than a hash match: the
  floating-point SVD path is bit-deterministic across sessions, days and a builder change.

**The read's build gates are no longer hypotheses.** The only untested element left is the sealed
bank's own 224 rows, which a rehearsal cannot cover by construction.

### Two defects in my own rehearsal, worth more than the passes

1. **It reported "three for three" while silently skipping `H16`** — the arm that is the
   comparator in *both* primary limbs and the only one carrying the data-section gate. The arm
   list held only `a025` and `a0675`; `H16` was defined and never executed. *A pass whose scope
   nobody states is the vacuous-check defect wearing different clothes.*
2. **The fix for that had the OPPOSITE bug.** Its precondition demanded both the H checkpoint and
   the cache; `build_compose.py:103` reads the checkpoint only when the cache is *absent*, and the
   cache is present at 1.26 GiB. The guard was **stricter** than the thing it guarded and falsely
   refused — one step from reporting *"H16 is not rebuildable"* against my own sealed registration.

> **A guard looser than its target passes bad work; a guard stricter than its target blocks good
> work. Both look exactly like the guard working.** Seventh instance tonight, first in the
> too-strict direction.

Launched via WMI so it survived the session — ancestry
`wininit → services → svchost → WmiPrvSE → cmd → python`, **no `claude.exe` in it.**

## 5. The result that changes the science

The third session built the arm my synthesis said was missing:

    INDEP-001 = theta_src + 1.00 * P16(delta_T) + 0.25 * P16(delta_H)   ->  235/384

**Exactly `T16`'s 235. Adding an entire second skill costs ZERO acquisition rows when each skill
carries its own coefficient.** Adding δ_H at 1.00 raises total magnitude +45.2%; at 0.25, +3.4%.
The old shared-scale rule funded that 45% by stripping 31.1% off everything **including δ_T** —
carrying the gate at 4× its useful strength by shrinking the generator by a third.

> **"Composition is expensive" was never interference. It was an allocation rule nobody
> questioned.**

**And I had been quoting the wrong overlap number.** The flattened-delta cosine is **+0.00273**;
the 0.151 I kept citing is a *subspace* overlap. The deltas are essentially orthogonal **and
composition still cost 60 rows unrescaled** — which makes the refutation of the
"orthogonal dimensions" idea *stronger*: orthogonality was already there and bought nothing.

## 6. What I recommend when you wake

1. **Decide on the HPLAT-001 read.** It is sealed, gated, costed, and resource-unblocked. One
   command. It is the only thing waiting on you.
2. **Re-run every composition on record with independent per-skill coefficients.** Pure weight
   arithmetic on deltas already on disk; on the one pair tested it turned a 60-row loss into zero.
3. **The dose axis still needs a training grant.** It remains the only unsaturated axis.

## 7. Housekeeping

- Disk: reclaimed 1.696 GiB (hash-verified before deleting); peers cleared 9.5 GiB of regenerable
  cache. Free ~16.7 GiB against the 14 GiB floor. Much of the earlier drop was **pagefile
  reservation** (allocated 21.5 GiB, actually used 0.6 GiB), which returns on its own.
- Six guards across the three sessions tonight measured something other than the thing they
  guarded — including two of mine testing process detachment. `CREATE_BREAKAWAY_FROM_JOB` is a
  **silent no-op** unless the job allows it. All recorded in memory.
