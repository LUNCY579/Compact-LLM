# Safe-to-delete survey — BusinessProject and ChatBubble, 2026-09-22

> **NOTHING WAS DELETED, MOVED OR REWRITTEN BY THIS SURVEY.** Output is a recommendation list.
> Every rating is backed by a check that is stated, and anything without positive proof of
> safety is rated **NEEDS-OPERATOR**, not SAFE.

**Headline, and it is not the one I expected: there is almost nothing safe to delete.** The
large items are load-bearing — receipt-pinned trained weights, source assets, pinned venvs. Four
of my own candidate ratings **inverted** once I checked them properly, every time in the
direction of *"this looked disposable and is not."* The genuine reclaim route is **lossless
compression**, projected at **9.88 GiB**, and the genuine *action item* the survey produced is a
**backup gap**, not a deletion.

---

## 1. Scale

| | |
|---|---|
| `C:\Users\clclc\BusinessProject` | **471.87 GiB**, 1,006,797 files, 73 top-level directories |
| registered git worktrees of the ChatBubble repo | 27 |
| ChatBubble-family directories that are **not** worktrees | 5 |

## 2. DO-NOT-TOUCH register, fixed before any candidate was rated

| item | why |
|---|---|
| `Chat-Bubble` (88.32 GiB) | primary checkout; the official cores exist **only** here |
| `rineya-dataset-retention-records` (88.20 GiB) | the evidence tree — git + private remote, receipts pin sha256 |
| `ChatBubble-retention-pilot-20260917` | the pinned pilot checkout (5b4c22d) — and see §3.2, it anchors an alternates chain |
| `.venv-mq001`, `Chat-Bubble/.venv`, `.venv-rab001-cu126` | pinned environments; binding equality forbids changing them mid-lane |
| `~/.cache/huggingface/hub` | holds Qwen3-4B, in use by the running battery |
| `~/.cache/uv` | hardlinks *into* the venvs — deleting it corrupts them |
| `Rineya-Nimbus-Abstain-1.0` | the source checkpoint every delta is defined against |

## 3. The four inversions — candidates that looked deletable and are not

### 3.1 `arm-checkpoints` — 21.45 GiB — **NOT SAFE.** My first check was vacuous.

38 files of trained `trainable.safetensors` under
`ChatBubble-ced-native-loader-qualification/experiments/architecture_benchmark_001/outputs/`.
These are **training runs** (B28, B28last20 at seeds 17/23/29/41, WSD schedules, a 4-way SOUP) —
not tensor arithmetic. Re-creating them needs GPU training under a budget grant.

I first grepped `rineya-dataset-retention-records` for each arm name and got **0 of 19
referenced**, which reads as "delete 21 GiB".

**That was the wrong namespace, and the zero was vacuous.** Positive control on the same grep:

```
'compose4b002'       -> 5 files      grep works
'TH16s'              -> 75 files     grep works
'retention-a329eb7b' -> 92 files     grep works
'B28last20'          -> 0 files      <- a real absence, but from the wrong tree
```

These arms belong to the **CED programme**, whose evidence lives in its own lane. Re-run there:

> **19 of 19 arms are referenced by name in `architecture_benchmark_001/evidence`** (1–5 files
> each), and the checkpoint hash `e90a974c9f810a3a…` appears in **22** evidence files.

**A 0 from an absence check has no known number to collide with.** Had I reported the first
number, this survey would have recommended destroying 21.45 GiB of receipt-pinned evidence for
another programme.

### 3.2 Three "0.09 GiB scratch clones" — **NOT SAFE, and the reason is invisible from outside**

`ChatBubble-binding-integration-pilot-20260920`, `ChatBubble-binding-mixture-pilot-20260920` and
`ChatBubble-retention-doselora-resume-20260920` are 0.09 GiB each, idle, and their untracked
evidence is **byte-identical to copies already in the records tree** (16 of 16 files verified by
sha256). Every heuristic says disposable.

They are small because they **borrow their object stores in a chain**:

```
binding-integration-pilot --alternates--> binding-mixture-pilot
binding-mixture-pilot     --alternates--> retention-doselora-resume
retention-doselora-resume --alternates--> retention-pilot-20260917   (THE PINNED PILOT)
```

Verified rather than inferred — objects physically present in each clone's own store:

| clone | own packs | own loose objects | own object bytes |
|---|---:|---:|---:|
| binding-integration-pilot | **0** | **0** | 1 MB |
| binding-mixture-pilot | **0** | **0** | 1 MB |
| retention-doselora-resume | **0** | **0** | 1 MB |
| **retention-pilot-20260917** | 8 | 7,203 | **738 MB** |

**Deleting any link breaks every clone upstream of it.** Total reclaim if all three went: 0.27
GiB. **Not worth it at any risk.** Recommendation: leave them, or have the operator `git repack
-a` each into independence *first* — which would *increase* disk use, so the honest advice is
simply to leave them.

### 3.3 `tgsa-backups` — 0.73 GiB — **NOT SAFE as named.** "final" is not a superset.

Three git bundles of the same refs on the same day. The obvious move is "keep `final`, delete the
two intermediates" for 0.49 GiB.

| bundle | refs |
|---|---:|
| `tgsa-all-refs-2026-09-04.bundle` | 56 |
| `tgsa-all-refs-2026-09-04-post-merge.bundle` | 56 |
| `tgsa-all-refs-2026-09-04-**final**.bundle` | **54** |

The "final" bundle has **fewer** refs, and the two earlier ones carry **9** and **6** refs it
lacks — including distinct commits of `research/tgsa-programme-ledger` (`2e0e3feb`, `4581fc0f`)
that exist in **no** other bundle. **`final` is a later snapshot, not a cumulative one.** A
name-based rule would have destroyed unique ref history. **NEEDS-OPERATOR.**

### 3.4 `ChatBubble-compose-lane-20260918` — 2.96 GiB — **NOT SAFE, and it is a BACKUP GAP**

This is the survey's one genuine action item, and it points the opposite way from deletion.

Of its 32 untracked evidence JSONs, **32 of 32 exist nowhere else** — not in git, not in the
records tree, checked by sha256 against all 11,086 distinct JSON contents there:

```
LONGCTX_int001_a05_lc.json   LONGCTX_int001_a10_lc.json   LONGCTX_tt001_TT16s.json
LONGCTX_lwf_{l0,l025,l05,ord23,ownA05,ownA075}_lc.json
PROCESS_lwf001{,v2}_{l0,l0p25,l0p5}.json     PROCESS_tt001_TT16s{,_1729}.json
RETENTION_PROBE_lwf001{,v2}_{l0,l0p25,l0p5}.json   ... 32 files
```

**These are the source measurements behind recorded INT-001, LWF-001 and TT-001 findings** —
including the interpolation-frontier result that the knee is lane-dependent (native 0.75, LWF
0.50). They sit on a single untracked copy in a directory that is **not even a registered
worktree**.

> **RECOMMENDED ACTION — not deletion: copy these 32 files into the records tree** so they are
> covered by the evidence-backup policy. Until then this directory is one of the least safe
> things on the disk to remove, and it is the only place a real loss risk was found.

The other three clones' evidence *is* fully duplicated (§3.2), so this gap is specific.

## 4. Remaining candidates, rated

| item | GiB | rating | basis |
|---|---:|---|---|
| `cross-scale/…/local_models` | 25.36 | **NEEDS-OPERATOR** | another programme's lane. 4 of 9 subdirs cited in its own evidence (11–76 files each); the other 5 cited in its code. Not mine to clear. |
| `architecture_benchmark_001/arm-runs`, `refcache`, `device_parity` | 9.05 | **NOT ASSESSED** | same CED lane as §3.1, same owner |
| `.venv-rab001` (torch 2.10.0+**cpu**) | 0.53 | **NEEDS-OPERATOR** | superseded in function by `.venv-rab001-cu126` (torch 2.10.0+cu126), but pinned-venv class — the operator decides, not me |
| `mq-data-001` | 0.01 | **LEAVE** | its own `SUPERSEDED.txt` says *"Kept, not deleted, and never used"* — a deliberate prior decision |
| duplicated source corpora across 27 worktrees | ~59 | **LEAVE** | git-tracked checkouts of the branch tree, not junk; removal dirties every worktree |

**Total rated SAFE-to-delete: 0.00 GiB.** That is the honest result.

## 5. The reclaim that IS available — lossless compression, 9.88 GiB

Transparent NTFS LZX (`compact /C /EXE:LZX`): logical bytes, paths and content unchanged,
reversible with `compact /U`, sha256 verifiable before and after. Already exercised on 1,264
files with **0 removed and 0 hash mismatches**.

**13 records roots totalling 46.86 GiB are still entirely uncompressed** (physical ≥ 97 % of
logical): `lwf001` 13.66, `int001` 10.17, `sparse001` 4.59, `lora596` 4.55, `rep001` 2.41,
`rrp007-a1-research` 2.22, `indep001` 1.71, `scaledh001` 1.71, `compose001` 1.23, `rrp015` 1.22,
`d1r-s29-hf` / `d1r-s23-hf` / `lc-D1-hf` 1.13 each.

**Projection, built the way the last one should have been.** My previous estimate was 4× too high
(3.165 GiB projected, 0.744 actual) because I applied a **safetensors** ratio to **JSONL**
journals. So this projection is per content type, using ratios measured **on the same file types
in this same tree**:

| extension | measured ratio | measured on | share of the 46.86 GiB |
|---|---:|---:|---:|
| `.safetensors` | **0.794** (saves 20.6 %) | 32.32 GiB already compressed here | **46.17 GiB (98.5 %)** |
| `.json` | 0.405 | 0.12 GiB | 0.60 GiB |
| `.txt` | 0.511 | 0.01 GiB | 0.07 GiB |
| `.jsonl` | **0.998** (saves 0.2 %) | 0.04 GiB | 0.01 GiB |

The mix is overwhelmingly the one type whose ratio is best measured, which is why the ratio
transfers here and did not last time. Unmeasured types are credited **zero** saving.

> **Projected: 9.88 GiB, lossless and reversible. Requires operator approval, as the last
> compression did.** The `.jsonl` row is the record of the earlier error: journals compress
> essentially not at all.

## 6. What this survey did NOT assess

**Out of scope by the operator's framing** (ChatBubble / BusinessProject research assets). These
are separate products and I make **no** claim about them: `PrivateScan` 71.46, `desktop-pet`
10.69, `OddOnThePhone` 9.65, `smolvlm-storyforge` 7.55, `Short_Response_Memory_Governor_Lab`
6.14, `Spherical_Relational_Response_Networks` 5.44, `WayVoice` 4.87, `Bomi-Wallpaper` 3.96,
`RineyaWorkspace` 1.48, `Whereview` 1.24, `LocalBoost` 1.09, and the `SupaExperiment*` family
24.86 GiB.

**Also not assessed:** whether any CED or cross-scale lane is *closed* — that is its owners' call,
and a closed lane's evidence is still evidence.

## 7. Method, and the one thing that kept going wrong

Each candidate was checked by: (a) size and file count measured, not estimated; (b) git status
**including ignored files** — my first pass used plain `git status`, which hides ignored
artifacts, so weights were invisible to it; (c) content-hash comparison against the records tree;
(d) reference search in the **owning lane's** namespace; (e) a **positive control** on every
absence result.

> **Every error in this survey was an absence read without a control**, and every one of them
> pointed toward deleting something load-bearing. A check that has never found anything is not
> evidence that there is nothing to find.
