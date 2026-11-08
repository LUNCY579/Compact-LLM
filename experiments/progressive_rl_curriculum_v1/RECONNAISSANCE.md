# Repository integration record

- Repository: `LUNCY579/ChatBubble` (private).
- New branch: `research/progressive-rl-curriculum-v1`.
- Exact base: `b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2` (`main`).
- Base tree: `571ed17d70bb7a5e9ea0f664ad999ac8be704212`.
- Read the root tree, `experiments/` tree and root `requirements.txt` at the base.
- A targeted branch-name search for `curriculum` returned no matches. This was not
  a claim of a full semantic audit of every historical branch.
- At this base, `experiments/` contains the anchor-routed recurrence and Supa
  long-context intervention studies, not the separate reasoning/RoLA branch tree.
- Inspected `research/rola-ngu-publication-reconciliation-v1` and its
  `experiments/rola_ngu_factorial_v1/objective.py`, blob
  `9ba6b17307bb84af9e0980ab9d9362a863e9f048`. Its sequence-mean clipped policy/KL
  convention informed `objectives.py`. No whole branch or launcher was imported.
- A request for that branch's reasoning-posttraining README returned 404; no
  inference about unread authorization or implementation was made from that path.
- All implementation paths are additive. No existing research refs, main content,
  checkpoints, production loaders, sealed banks, or native launch grants changed.
- Direct Git transport was unavailable in the working container. Files were tested
  locally and published via GitHub Git-tree/commit/ref operations on the new branch.

## Study boundary

PRLC-001 tests a data curriculum, not simultaneous capacity unfreezing, RoLA/NGU,
new reward shaping, architecture surgery, or a MiMo-scale training reproduction.
The complete runnable path is deterministic single-turn text; external agent
runtimes remain a separate qualification boundary. Tests do not authorize training.
