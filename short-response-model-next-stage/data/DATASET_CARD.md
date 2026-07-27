# Short-Response SFT Pilot v1

This package contains 550 train, 110 development, and 110 sealed records across eleven response classes.

The data is synthetic, template-family curated, and marked `human_review_status: pending`. It is a pilot—not a production or publication-ready corpus.

Template families are disjoint across splits. Never train on `data/sealed/sealed.jsonl`, and do not copy the separate ShortChatBench benchmark into training.

Run `scripts/01_validate_data.ps1`, review the generated audit packet, evaluate the untouched base, train the all-layer QLoRA reference, and only then begin layer selection.
