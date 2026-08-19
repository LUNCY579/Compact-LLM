from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..training.tokenization import render_chat
from .base import GenerationRequest, GenerationResponse


class LocalTransformersBackend:
    name = "local_transformers"

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError('Local model evaluation requires: pip install -e ".[train]"') from exc
        self.torch = torch
        self.config = config
        model_name = str(config["model_name_or_path"])
        tokenizer_name = str(config.get("tokenizer_name_or_path", model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            local_files_only=bool(config.get("local_files_only", False)),
            use_fast=bool(config.get("use_fast_tokenizer", True)),
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError("Tokenizer needs an EOS or pad token")
        self.tokenizer.padding_side = "left"

        dtype_name = str(config.get("dtype", "auto")).casefold()
        if dtype_name == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            dtype = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }[dtype_name]
        kwargs: dict[str, Any] = {
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
            "local_files_only": bool(config.get("local_files_only", False)),
            "low_cpu_mem_usage": True,
        }
        if config.get("attn_implementation"):
            kwargs["attn_implementation"] = config["attn_implementation"]
        quant = config.get("quantization", {}) or {}
        if quant.get("load_in_4bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(quant.get("quant_type", "nf4")),
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)),
            )
            kwargs["device_map"] = config.get("device_map", "auto")
        else:
            kwargs["torch_dtype"] = dtype
            kwargs["device_map"] = config.get("device_map", "auto" if torch.cuda.is_available() else None)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        adapter = config.get("adapter_path")
        if adapter:
            adapter_path = str(Path(str(adapter)).expanduser().resolve())
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model.eval()
        self.batch_size = max(1, int(config.get("batch_size", 1)))
        generation = config.get("generation", {}) or {}
        self.generation = {
            "max_new_tokens": int(generation.get("max_new_tokens", 256)),
            "do_sample": bool(generation.get("do_sample", False)),
            "temperature": float(generation.get("temperature", 1.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "top_k": int(generation.get("top_k", 0)),
            "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not self.generation["do_sample"]:
            self.generation.pop("temperature", None)
            self.generation.pop("top_p", None)
            self.generation.pop("top_k", None)
        self.max_input_tokens = int(config.get("max_input_tokens", 4096))

    def _device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except Exception:
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResponse]:
        responses: list[GenerationResponse] = []
        for start_index in range(0, len(requests), self.batch_size):
            batch = requests[start_index : start_index + self.batch_size]
            rendered = [render_chat(self.tokenizer, request.messages, add_generation_prompt=True) for request in batch]
            tokenized = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                add_special_tokens=False,
            )
            device = self._device()
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            begin = time.perf_counter()
            try:
                with self.torch.inference_mode():
                    generated = self.model.generate(**tokenized, **self.generation)
                elapsed = (time.perf_counter() - begin) * 1000
                input_width = tokenized["input_ids"].shape[1]
                for row_index, request in enumerate(batch):
                    output_ids = generated[row_index, input_width:]
                    text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    responses.append(
                        GenerationResponse(
                            request_id=request.request_id,
                            text=text,
                            latency_ms=elapsed / len(batch),
                            backend=self.name,
                            metadata={
                                "input_tokens_padded": int(input_width),
                                "generated_tokens": int(output_ids.numel()),
                            },
                        )
                    )
            except Exception as exc:
                elapsed = (time.perf_counter() - begin) * 1000
                for request in batch:
                    responses.append(
                        GenerationResponse(
                            request_id=request.request_id,
                            text="",
                            latency_ms=elapsed / len(batch),
                            backend=self.name,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        return responses

    def close(self) -> None:
        try:
            del self.model
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass
