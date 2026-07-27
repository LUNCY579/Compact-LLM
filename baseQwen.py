from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix PyTorch/GPU setup first.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "Reply like a casual text message. "
                "Use only a few words or one short sentence."
            ),
        },
        {
            "role": "user",
            "content": "I finally finished the project.",
        },
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        rendered,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0, inputs.input_ids.shape[1] :]
    reply = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    print(f"Reply: {reply}")
    print(
        "Allocated VRAM:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "GB",
    )


if __name__ == "__main__":
    main()