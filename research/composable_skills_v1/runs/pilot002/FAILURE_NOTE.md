# Tokenizer API integration failure, zero generations

The model loaded, but the first token-tensor construction failed because installed
Transformers5.14.1 defaults `apply_chat_template(return_dict=True)`. No model forward,
training update or generation ran. The source snapshot, worker log and failed
supervisor receipt are preserved. Pilot003 requests return_dict=False explicitly,
checks integer token lists and moves a full384 historical-prompt equality check and
context-envelope census into CPU preparation. The scientific design is unchanged.
