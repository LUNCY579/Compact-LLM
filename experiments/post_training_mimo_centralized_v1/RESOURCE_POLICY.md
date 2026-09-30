# Resource policy

AMENDMENT_011 adds LAPTOP_STREAMED_V1: fixed 2 MiB checkpoint read buffer,
1.25 GiB system reserve, 1.75 GiB process-tree RSS ceiling and admission reserving
all remaining allowed worker growth. Before-worker budget is 3 GiB; the standalone
controller adds 64 MiB for its supervisor. The loader has CPU byte-identity controls,
but its complete GPU path and backward fit remain unqualified. See LAPTOP_RUNBOOK.


AMENDMENT_009 and the user's 29 September authorization introduce the named
LAPTOP_GPU_RESIDENT_V1 profile for this 16 GB laptop. Keep 2 GiB system RAM
available and bound the worker process tree to 2.5 GiB RSS. After imports, admission
requires available RAM >=2 + max(0, 2.5 - current worker RSS) GiB, budgeting all
remaining allowed worker growth. An over-cap worker fails. GPU admission/runtime
floors remain 4/0.5 GiB, disk 4 GiB. Independent containment/watchdog limits apply
throughout. These prospective pilot bounds require native measurement.

The load_forward stage lasts at most 600 seconds, performs strict exact loading
and one 256-token frozen last-logit forward, and has optimizer/generation caps zero.
The full stage retains every original E0 numerical fixture and the 1800-second/five-
optimizer-call bounds. Source, configs and outputs must be committed/write-once.
No new profile status or resource-only PASS grants E1 authorization. Register the
measured laptop queue/E1/long-context successors before their affected runs.

## Historical profile retained

One major GPU worker. The original config preserves inherited PRLC minima: host available >=7.4 GiB
before native load, worker available >=6.0 GiB throughout, device available >=4.0
GiB before load, disk free >=4.0 GiB. These are minima, not evidence of fit.
Monitor actual worker RSS, available host RAM, GPU allocated/reserved and free
memory; fail closed when unavailable. Its floors remain fixed. The named laptop
successor is an explicit authorized amendment, not a modification of old receipts.

E0: 30-minute wall ceiling; maximum 32 non-evaluation parity fixtures, 32 generated
tokens for throughput/logprob check, 256/512/1024 context profiles. Bounded
reversible optimizer fixture only, with adapter save/reload and deterministic
repeat; never overwrite the published source. Tensor hashing is streamed.
E1: <=24h wall/GPU per seed and actual token manifest required; seed101 only until
DEV gate. No remote rental or paid teacher calls. Resource fit at one context
does not qualify another. Failed larger context permits a separately documented
smaller-context amendment, never a silent change. CPU/reference serialization or
activation checkpointing also changes the registered runtime and must be qualified.

Fast work may accompany heavy work only if the same floors continue to hold. The
queue planner is a planning aid; receipt strings do not themselves authorize a run.
