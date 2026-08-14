# DEV cached committee diagnostics

Status: **DEV_COMMITTEE_DIAGNOSTICS_COMPLETE**

This supplement adds answer/abstain all-token accuracy, proposal vote entropy, and pre-tie-break tie rates to all 127 committees under all nine frozen cached aggregators (1,143 records). It exactly reproduces every frozen token and example count.

## Full-K=7 hard reference

- Answerable all-token example accuracy: 24.160%.
- Abstention all-token example accuracy: 32.407%.
- Mean proposal vote entropy: 0.7154 nats.
- Hard primary tie rate: 5.848%.

## Systems boundary

Latency and VRAM are intentionally null for cached counterfactual subsets. They are measured only for selected free-running conditions; no systems value is imputed from forward proxies.

Cache SHA-256: `4b192a736451e81fd9e3dcd212f3f3aa97cda0a67c440536802453187377ee13`.
