# MQ-001 operational successor and current evidence

**Current authority: [MODEL_QUALITY_STATUS.md](../../MODEL_QUALITY_STATUS.md).**

Branch `codex/model-quality-audit-v2` descends from immutable MQ subject `8c3ed88a821b83e531ab682b3201879ea4fc928b`. It reconciles evidence and repairs supervised execution; it does not change the scientific recipe or claim new model quality.

## What has actually run

Historical local records contain four zero-update official-Supa CUDA qualification envelopes and eight completed training cells, all at peak LR1e-5. The first pair used an obsolete DEV instrument. The corrected six cells vary only order seeds17/23/29 over the same238 selected training rows; whole scores7/7/8 and response13/13/13 on51 scoreable DEV rows. Every cell processes24,910 tokens in13 updates; whole supervises24,672 targets and response1,944. These are exploratory order repetitions, not independent training draws or confirmed superiority.

All8result sets retain valid hashes for baseline/final outputs and step logs. All8checkpoint directories are absent. The corrected6cells bypassed the watchdog and lack launch/supervisor receipts. The old implementation receipt is an accurate creation-time fixture record, not the latest execution status. [Raw records and archive](../../docs/model-quality/audit-2026-09-15/historical-mq-archive-manifest.json) preserve failures, revisions and review records.

## Fixed scientific design

| Objective | Peak LR |
|---|---|
| Whole-sequence next-token |3e-6 /1e-5 /3e-5|
| Assistant response plus EOS |3e-6 /1e-5 /3e-5|

Exact ungated Supa596,049,920 parameters; blocks8–27 train314,618,880 parameters across220tensors. Embeddings/output tying, final norm and blocks0–7 remain frozen. FP32 computation, CPU FP32Adam moments, betas(.9,.95), epsilon1e-8, decay0, clip1, WSD20%warmup/60%stable/20%1-sqrt decay to0.1×peak. One selected token stream, ordering and full-row update grouping per paired screen; response-only gets no extra examples. Native design retains25,000 processed-token ceiling,90%coverage floor,2,048maximumsequence/group tokens and64DEVrows.

## Current operational change

The public foreground supervisor avoids heavy model-library imports before launching its worker. Direct unsupervised internal-worker use is rejected; creation/start failures and watchdog outcomes retain records. The old shell wrappers now refuse and preserve their historical source in the documentation archive; no checkpoint cleanup is automatic. Source locks and scientific algorithms are unchanged. Exact branch/parent binding requires new qualification.

## Still required

A new quality bank and independent data/scorer review, representative/max-shape fit, zero-update native qualification on this exact commit/runtime and bounded official-source resume qualification. The current2,000-row v3pool is a development instrument; no explicit missing-evidence contracts/pairs exist and underlying generator overlap remains. The proposed QTB-001 bank is not built or approved. A larger DEV instrument needs a separately registered successor to the64row cap.

Use [RUNBOOK.md](RUNBOOK.md), [POOL_STATUS.md](POOL_STATUS.md) and the [decision memo](../../MODEL_QUALITY_STATUS.md). Historical README bytes are [preserved](../../docs/history/model-quality-audit-2026-09-15/experiments/model_quality_gpu_v1/README.md). Synthetic/random-native fixture tests, native qualification, training completion, publication and efficacy are distinct evidence classes.
