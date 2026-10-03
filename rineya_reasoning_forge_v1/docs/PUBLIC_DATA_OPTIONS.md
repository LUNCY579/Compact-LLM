# Public data options, checked 2026-09-11

These are candidates to inspect and sample, **not imported components of this bundle**. Hugging Face hosting is not itself a license or a quality guarantee. Pin the chosen revision, retain provenance, audit benchmark overlap, and inspect source-level terms before use.

## Best complementary roles

| Candidate | Complement to this corpus | Important qualification |
|---|---|---|
| HuggingFaceTB/smoltalk — original Smol-Magpie-Ultra, Smol-constraints, Smol-rewrite, and Smol-summarize components | More varied instructions, rewriting, summarization, and conversations | The card explicitly licenses the newly generated components under Apache 2.0; existing imported components require their own license review. Do not treat the full mixture as uniformly cleared. |
| HuggingFaceTB/smol-smoltalk | A compact-conversation selection is a useful candidate for small Rineya models | The SmolTalk authors say this subset was selected for 135M/360M models, with more concise conversations and without advanced math. This supports inspecting it, not a guarantee of benefit to Rineya. Check components and provenance. |
| open-r1/OpenR1-Math-220k | Supplement with checked mathematical examples | The card lists Apache 2.0 and correctness information for multiple traces. At least one correct trace per problem does not mean every trace is correct. Select verified traces, audit source overlap, and do not blindly truncate a long solution into a short answer. |
| HuggingFaceTB/cosmopedia | Broader educational and story-like prose for a separate general-language data stream | The release is labeled Apache 2.0, but prompts are seeded from web/educational/instruction sources. The card itself discusses contamination risk. Inspect provenance, factual quality, and duplication; synthetic text is not automatically factually reliable or rights-risk-free. |

## Correction to a permissive-only shortlist

Do not use **allenai/tulu-3-sft-mixture wholesale** as an unrestricted/commercially cleared mixture. Its official card says the collection is ODC-BY-1.0, different licenses apply to subsets, and some portions are non-commercial. It specifically lists No Robots under CC-BY-NC-4.0 and notes separate third-party model-output terms. A separately audited subset may be suitable; the unfiltered mixture is not a blanket permissive option.

## Primary sources

- SmolTalk card: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- Smol-SmolTalk card: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- OpenR1-Math-220k card: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- Cosmopedia card: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- Tulu 3 SFT mixture card: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- Qwen3-0.6B model card and non-thinking rendering guidance: https://huggingface.co/Qwen/Qwen3-0.6B

## Suggested selection process

Start with approved general/instruction examples. Keep short and medium natural conversations, faithful rewriting, straightforward reasoning, and grounded summaries. Reject malformed messages, unsupported answers, incorrect math traces, duplicated families that dominate the mixture, and private/evaluation overlaps. Keep source licenses attached. Evaluate the selected sample on the actual model rather than assuming that a larger or more fashionable dataset is better.

The generated bundle avoids copying any of these public rows; its independent data license therefore does not relicense them.
