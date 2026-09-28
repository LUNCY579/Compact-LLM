# Resource policy

One major GPU worker. Preserve inherited PRLC minima: host available >=7.4 GiB
before native load, worker available >=6.0 GiB throughout, device available >=4.0
GiB before load, disk free >=4.0 GiB. These are minima, not evidence of fit.
Monitor actual worker RSS, available host RAM, GPU allocated/reserved and free
memory; fail closed when unavailable. Do not lower floors to launch.

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
