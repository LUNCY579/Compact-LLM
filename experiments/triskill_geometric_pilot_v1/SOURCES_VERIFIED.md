# Primary-source check, 2026-10-01

These sources motivate distinctions and future controls; none establishes the
effectiveness or mechanism of the Rineya construction. Fixed arXiv versions
prevent a later unversioned PDF from silently changing this review.

| Source | Verified content and limit |
|---|---|
| Geiger, Wu, Potts, Icard and Goodman, *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*, CLeaR 2024, PMLR 236:160-187, [primary PDF](https://proceedings.mlr.press/v236/geiger24a/geiger24a.pdf), sections 3.3-3.5 | DAS learns an alignment using an objective and tests counterfactual agreement. Distributed interchange rotates, patches selected coordinates, then unrotates. A coordinate change alone is not a behavioral intervention. This pilot neither learns that alignment nor tests causal-abstraction counterfactual agreement. |
| Makelov, Lange and Nanda, *Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching*, [arXiv v2, 2023-12-06](https://arxiv.org/pdf/2311.17030v2), sections 3.2-3.5, 5 and 8 | Successful patching may use a dormant pathway instead of the natural computation. The paper examines both faithful and illusory cases and additional circuit checks. It supports distinguishing output steering from a natural mechanism; it proves neither interpretation for this pilot. |
| Todd, Li, Sharma, Mueller, Wallace and Bau, *Function Vectors in Large Language Models*, ICLR 2024, [arXiv v2, 2024-02-25](https://arxiv.org/pdf/2310.15213v2), section 3.3, Table 7 and Appendix C | Function vectors are extracted using task demonstrations and causally ranked heads. Some composition examples use three vectors, including subtraction of a shared function; successes and resistant tasks both occur. This is no general guarantee that two weight-derived subspaces create a third capability without a task objective. |
| Opiełka, Rosenbusch and Stevenson, *Causality ≠ Invariance: Function and Concept Vectors in LLMs*, [arXiv v1, 2026-02-25](https://arxiv.org/html/2602.22424v1), sections 2.2, 3.2.2-3.2.3 and 5 | Format and language can change steering effects; a vector may encourage brackets or foreign-language output rather than the intended semantics. Reported function-vector/concept-vector contrasts are bounded by their tasks. Its invariance means stability across formats/languages, distinct from an exact compensated change of basis. It does not test this pilot's commutator or guarantee semantic transfer. |

The compensated-invariance identity, commuting-projector null, parent-swap
inverse and behavioral-inverse counterexample in this successor are explicit
mathematical deductions verified with NumPy. They are not empirical conclusions
borrowed from those papers. A later positive benchmark could support useful
steering or recombination of existing latent ability; demonstrating genuinely
new generalization and a natural causal mechanism requires stronger evidence.
