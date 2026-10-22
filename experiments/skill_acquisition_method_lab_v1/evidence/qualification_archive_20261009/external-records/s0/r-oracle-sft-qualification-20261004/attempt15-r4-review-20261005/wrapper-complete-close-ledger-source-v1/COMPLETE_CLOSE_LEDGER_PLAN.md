# Offline complete close operation ledger

This is the single preserved successor for the saved close-guard/log-state
correction. NoAdmit remains. No Native/candidate/model/metadata producer, current
packet/ENTRY/grant/eligibility or core/protected source is created or executed.

The whole original close operation now shares the existing fixed secondary
ledger: operation admission, original before guard, actual stream.close and
original after guard. Primary operational exception identity remains separate.
Before refusal records secondary type with zero calls and retains stream; actual
close error records one attempted unknown close; successful return clears owned
stream before a late afterguard records its secondary error. First recorded
secondary type is retained and actual attempt/return/failure flags are not
overwritten. A second actual close is forbidden.

No additional error slots are introduced. perform() does not run an afterguard
when the action raises, so simultaneous actual-close-error plus later-afterguard
is not a reachable path. There is no new Native authority, renewed deadline or
forced close after expiry. Guard/admission failures are now visible rather than
lost when a primary error already exists.

Actual internal reads are1536 bytes: two blocks3072 plus5120 logical state remain
original8192. The state owns four128-byte capture-error names, two128-byte
secondary names, fixed arrays/identity/status fields and prior reader/thread/
FileIO/callback recipes. All registered2048 operation/8MiB read/8192 persisted
ceilings and byte order are unchanged. This replaces actual retained payload
storage; no fee/cap or data domain is reduced to assert a fit.

The changed close wrapper has one selected I/O attempt in the original bounded
log operation recipe. Added pure metadata/clock/error handling remains within
that prospective recipe and must be composed in the full source-unit contract.
PREP remains50000ops/128MiB read-hash/4MiB write/8MiB meta/2048SHA/1800s; CPU root
150000ops/256MiB read/1MiB write/2048SHA/32MiB hash/1MiB meta/60s; original tail
256ops/65536 read-write-hash-meta/16SHA/single5s. Bounds/resources/workers remain.

Ten actual-source synthetic controls cover before/admission zero calls, actual
error one unknown call, no retry/no typename overwrite, observed successful
return plus late guard, normal operation accounting and complete unchanged IO
caps. All four final static executable references are rehashed.

Remaining concrete proof: exact actual source graph22696 candidate versus22528
gate (168 conservative excess), probe/ledger typename identities, full5120 state
and FileIO/native ABI/callback/encoder/frame/error lifetimes; complete PREP
trusted/owned metadata window and fixed publisher/compile/source-loader accounting;
image8192/cleanup16/full logical all-seven and fresh source/domain metadata.
These are prospective source-proof requirements; a Native pass is separately
required for physical qualification. No logical fit or qualification credit is
claimed merely from passing fixtures. Updates0.
