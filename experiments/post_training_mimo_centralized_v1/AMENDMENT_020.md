# Amendment 020: internal native adapter backend and engine publication V2

30 September 2026. Add concrete injected-Torch operations to the existing scheduled
TRAIN engine, without a public model-launch CLI or an approval callback. Framework
imports, verified GPU loading, source/runtime binding, resource containment and
scientific launch authority remain the responsibility of a future owned worker.
The internal backend is source-ready and physically unqualified.

The constructor factory requires the already verified and q/v-wrapped exact Titan:
310 frozen BF16 CUDA base tensors,596,049,920 base parameters,tiedembedding and112
FP32 adapter tensors totaling1,146,880. It keeps eager/eval/cache-off/CE512 math,
uses the qualified completion-loss helper and original AdamW implementation choices,
and preserves adapter-only trainability. The versioned engine supplies the exact
planned component instead of guessing from row IDs. Each sequence mean loss is
scaled1/20 once; reset,clip andoptimizerstep occur once per20-sequence group.
Final LR0 still calls AdamW. No saved target is truncated.

Save only adapterparameters,Adamstate andPython/TorchCPU/CUDA0 RNG,using bounded
CPU copies and write-once owned artifacts up to32 MiB. The backend returns small
artifact references,not resident tensor populations retained by five envelopes.
EngineV2 validates the POST-save cursor/cost envelope,then publishes metadata and
its journal event before exposing the boundary. Publication failures retain the
already returned save cost and refuse exposing that checkpoint. Restore requires
its own same-process published manifest,validates identities,reads boundedbytes,
and deserializes the SAME verifiedBytesIO with weights_only=True/mapCPU. Validate
physical schema/order/shapes/dtypes/moments/RNG before copying intoGPU state. No
cross-process recovery or rollback after later failed work is claimed.

Outer engine counters describe backend-method spans. Separate primitive counters
record each actual Torchforward/backward/clip/reset/step/save/load attempt and return
immediately, before later validation orresource checks. A laterfailed backend method
must not erase a returnedTorch operation. The factory attaches detached primitive
costs to the ORIGINAL constructor exception,so the ownedworker can retain costs
even whenconstructionneverreturns. Restore doesnotreset costcounters. A native
worker must publish both ledgers and qualified asset/source evidence on failures.

FakeTensor/optimizer dispatch and synthetic engine controls exercise scope/order,
bytebounds,publication/RNG interfaces and failure accounting without importing a
framework or running a model. They do NOT qualify Tensorphysics,realCUDA accumulation,
checkpointdurability,bitwise continuation orresourcefit. Every native/E1/closed-bank
grant remains false. Next connect a separately registered owned qualification
worker on non-evaluation mechanicalfixtures,then retain original capture/language/
long-context/sustained-training prerequisites before E1. Original engines/configs
and bestrecipe remain unchanged; the full capability/retention objective remainsopen.
