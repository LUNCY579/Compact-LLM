# Amendment 019: native-backend metadata and checkpoint byte contracts

30 September 2026. Register a stdlib-only component needed by the future native
training backend. It does not import a framework, instantiate a model, deserialize
a checkpoint, restore RNG or approve training. Every authority flag remains false.

Validate exactly112 ordered FP32 q/v adapter metadata entries:28 layers,rank8,
q-A8x1024/q-B2048x8/v-A8x1024/v-B1024x8 and1,146,880 parameters. Frozen-base metadata
must describe596,049,920 BF16 parameters,310 unique serialized names and the tied
embedding alias. Only the installed q/v base-name aliases map back to original
names. These declarations are witnesses, not physical tensor evidence.

Checkpoint references must use one identical adapter-only artifact for parameters,
AdamW state and training RNG, bound to the frozen weight/data/plan and caller's
current source/inventory digests. Require exact optimizer metadata, order, moment
shapes/dtypes and completed-update scalar counters; initial empty state is valid.
RNG metadata covers Python MT19937 version3, Torch CPU and exactly CUDA0. Shapes
are supplied by a separately frozen caller; synthetic expected sizes do not
establish physical runtime values. Artifacts are limited to32 MiB and streamed
through1 MiB hash buffers inside the approved local run directory, with closed
bank/published-source/path escapes refused. Base and gradient storage is forbidden
by metadata contract, not proven by reading opaque checkpoint bytes.

The initial18-control draft passed, then the Windows drift enhancement exposed
path-stat/fstat ctime incompatibility. Those failed19-control runs and unavailable
original streams are recorded separately. The final implementation normalizes
CreationTime and ChangeTime through a cached Win32 handle query, compares opened
and reopened identities, and rejects same-size/mtime-restored mutation after the
hash consumed its original bytes. Final21 controls pass on system and pinned
Python; root's canonical-package streams are preserved.

Byte identity covers this validation only. Later file changes remain possible;
the eventual Torch restore must verify and deserialize the same bounded bytes or
opened handle, not blindly reopen a path after verification. Physical tensor
contents, finite values, optimizer devices, restored RNG, checkpoint durability,
bitwise continuation and resource fit still need native qualification. A native
backend/worker and engine publication hook remain absent; all capture/preservation
and original E1 scientific prerequisites remain unchanged. No recipe is promoted.
