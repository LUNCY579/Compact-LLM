# Closed proposal resource/profile validation correction

Independent review showed the graph002 closed validator accepted a coordinated
reseal lowering GPU admission from4GiB to1GiB, because resources/profile were not
compared to the original selection. The source/packets/counterexample remain in
pre_resource_profile_fix002 and independent_review. The packet bodies themselves
were correct and are unchanged; this is a validator successor.

The validator now reads the exact independently pinned original D001 semantic
source and requires the complete original resources dictionary and recomputed
scope/permission/resource profile digest. It also checks exact kinds/version,
original helper/offline/runtime/Torch/tolerances/ordinal fields and all closed
unsupported-authority flags. No approved reader or synthetic grant is invoked.

Five final controls006 pass, including GPU admission1, CPU reserve1, deadline601,
added and removed resource fields after coordinated config/profile/body reseals.
All prior source/map/ref/grant/review negatives remain. New source is preserved
separately; graph002 closed packets keep their actual saved bytes and pending
implementation/dispatch/grant/review state. No native permission changed.
