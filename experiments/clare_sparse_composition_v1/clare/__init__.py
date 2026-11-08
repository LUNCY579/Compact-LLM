"""CLARE-RINEYA-001: capacity-aware sparse skill allocation for composable skill deltas.

The package is split so that every mask/allocation invariant is testable without torch or a
model (``footprint``, ``select``, ``masks``, ``schedule``), while the torch-dependent pieces
(``training``, ``native``) sit on top of it.

Support disjointness is a *mechanical* property of the allocation. It implies a zero parameter
dot product between skill deltas; it does NOT imply independent hidden states, independent
logits, retention or composition. Those are measured separately (``functional``, ``metrics``).
"""

EXPERIMENT_ID = "CLARE-RINEYA-001"
PACKAGE_VERSION = "1.0.0"
