# Procedural Retrieval Collision Audit

- Answerable procedure cases: **150**
- Authored support recall @ top 3: **1.0**
- Equivalent trusted procedure retrieval recall: **1.0**
- Legacy prompts with same-topic sibling collisions: **150** / **150**
- Query-aware v2.1 prompts with same-topic sibling collisions: **0** / **150**
- Legacy sibling passages across those prompts: **300**
- Query-aware mean selected knowledge passages: **1**

Interpretation: when support recall and procedure-rule recall are both complete but every legacy prompt contains near-identical sibling records, a 1.0 Oracle / 0.0 Real procedural gap is consistent with evidence-binding interference rather than simple retrieval absence. Model-output diagnosis is still required to distinguish retrieval-policy gain from learned-weight gain.
