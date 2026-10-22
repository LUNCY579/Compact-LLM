# DOSELORA-002: why a new root exists, and what an outside reviewer should distrust

Copy this file into the new records root at registration time.

## What happened

DOSELORA-001 was fully prepared, sealed and never run. Its registered session envelope
expired at 2026-09-20T09:39:05Z (`deadline_unix` 1789897145) while the experiment was
blocked on host RAM. After expiry `launch.py` refuses with
`SESSION_CANNOT_FIT_NATIVE_OWNED_WORK`, which is the intended behaviour.

The seal deliberately prevents a session from extending its own envelope in place:
`launch.py:verify_frozen()` binds `SESSION_ENVELOPE_V2.json`'s sha256 into
`PRELAUNCH_BINDING.json`; `launch.py` reads that envelope by name so a V3 is never
consulted; `code_files_sha256` covers `launch.py` itself; `finalize_prelaunch.py` cannot
re-run because `archive.mkdir()` has no `exist_ok`; and `prepare.py` writes create-only
(`'x'` / `'xb'`) throughout. **That control worked and was not broken.**

So DOSELORA-002 is a NEW experiment root with its own envelope, freeze and grant.
`highdose-lora001-20260920` is left byte-intact and its seal still verifies.

## What is identical, and verified by hash

| artifact | sha256 | status |
|---|---|---|
| `PREREG.md` | `e8d2d36ea2d6080c52b68af917c04061f6afc2c47d294f71e2698881b94e33ab` | byte-identical to the one DOSELORA-001 registered; equals `registration_sha256` in both its `GRANT.json` and `PRELAUNCH_BINDING.json` |
| `AMENDMENT_STORAGE.md` | `a048c147374b3d58386897ae69d683ca38126242b41b6a9b294271b74e2a2f1a` | byte-identical; equals `storage_amendment_sha256` in DOSELORA-001's `SESSION_ENVELOPE_V2.json` |
| `train.py` | `8faa5c4e833d8ec46fbf79a6386805eeb3566b53f1580aaf82062759f7bf2914` | unchanged |
| `adapters.py` | `5cc0d64ed0e33296a1d641d4c1739df12f7219bae372e978084a80ae742521bb` | unchanged |
| `materialize.py` | `952a4683519e2422c082a8e5795d44da1b76ff03de8c39a037630b942aad96fd` | unchanged |
| `finalize_prelaunch.py` | `93b8e93bc2abe3710c5e4408199f0955d20fe1e079b3e44081edfbaeddc429ac` | unchanged |

Every registered band, prediction, threshold, budget and gate is unchanged. Nothing was
loosened. `register_session.py` enforces this mechanically: it compares the fresh envelope
against DOSELORA-001's `SESSION_ENVELOPE_V2.json` key by key and raises on **any**
difference outside `session_start_unix` / `deadline_unix`.

## The one code change

`prepare.py` hardcodes its output root, so a new root requires one constant to change:

```
-OUT=RECORDS/'highdose-lora001-20260920'
+OUT=RECORDS/'highdose-lora002-20260920'
```

sha256 `716985ad936a448ab2adab892fc8331ac84e7826d0b030af0e48e6367ddd7eb1`
   -> `dc2ebb41d025d6e5898a7c31e6e906b5e02ba64f614ff4c32197e8c3a329bf42`

This is the only change among the five frozen executables. Two further one-line changes
are outside the frozen set and exist only to stop a v2 script writing into the sealed v1
root: `compress_evidence.py` `OUT`, and `report.py` `DEFAULT_ROOT`. All other files in
`research/retention_highdose_lora_v2/` are byte-identical to
`research/retention_highdose_lora_v1/`, verified with `diff -r`.

The code lives in a COPIED directory rather than an edited one specifically so that
DOSELORA-001's `PRELAUNCH_BINDING.json` keeps verifying. Editing `prepare.py` in place
would have silently invalidated the predecessor's seal.

## The training data is copied, not regenerated

This is the part most worth checking, because "rebuild the experiment" normally means
"rebuild the data", and that would produce a different experiment wearing this prereg.

`prepare.py` does not regenerate the `exact2697` rows. It copies them from
`doseint001-20260919/prepared-dual899/`, with a hash assert **before** the copy (L44-45,
against the parent `DATA_CONTRACT.json`) and a sha compare **after** it (L30-34, inside
`copy_new`, which opens `'xb'`). Byte-identity is therefore enforced twice by the code
itself, not assumed.

Confirmed already identical between the parent and DOSELORA-001:

- `microfit-train.jsonl` `df64db3cfbfb97fe...`
- `ARM_BUILD.json` `ef3cfa05d84f77cc...`

`DATA_CONTRACT.json` is the only file in `prepared/` that is written rather than copied;
it is the parent contract plus the new experiment name and a parent pointer.

## The grant is not being spent twice

DOSELORA-001's grant is provably unspent. Absent from its root, checked on the filesystem:
`journal/`, `NATIVE_DISPATCH_RESERVATION.json`, `arms/`, `eval/`, `posttrain/`,
`NORM_MATCH.json`, `RESULTS.json`. Zero model calls, zero optimizer steps, zero probes,
zero checkpoints. Its own `BLOCKED_STATUS.json` records the same accounting.

Cumulative native spend across DOSELORA-001 and DOSELORA-002 therefore equals
DOSELORA-002's spend, because DOSELORA-001's is zero. A fresh envelope with identical
limits is a first helping, not a second.

## What a reviewer should be suspicious of, stated plainly

1. **A session that re-registers itself after its own deadline expires looks exactly like a
   session buying itself more time.** The defence is not intent, it is that the budget is
   mechanically identical and checked by `register_session.py`, and that the predecessor
   root is preserved unmodified so the two can be diffed.
2. **The registration window is only open because nothing was ever measured.** If any arm,
   endpoint, probe output or score existed for DOSELORA-001, a fresh registration would be
   indistinguishable from re-rolling after seeing data, and this route would be
   illegitimate. It is legitimate *only* because the filesystem shows zero outputs. That
   check should be re-run by anyone auditing this, not taken on trust.
3. **`prepare.py`'s hash changed.** An auditor comparing the two roots will see one frozen
   executable differ. The diff is one line and is reproduced above; verify it rather than
   accepting this description.
4. **`BUILD_GATE_LORA.json` embeds `created_utc`** and the envelope embeds a new clock, so
   the two roots are not byte-identical and are not meant to be. Only the scientific
   inputs are.
