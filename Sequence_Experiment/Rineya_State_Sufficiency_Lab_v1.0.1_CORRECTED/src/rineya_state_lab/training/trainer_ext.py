from __future__ import annotations

from typing import Any


def make_weighted_trainer_class(*, preserve_order: bool = False) -> type:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DistributedSampler, SequentialSampler
    from transformers import Trainer

    class WeightedCausalTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            inputs = dict(inputs)
            weights = inputs.pop("sample_weight", None)
            labels = inputs.get("labels")
            outputs = model(**inputs)
            if labels is None:
                loss = outputs.loss
            else:
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.shape)
                valid = shift_labels.ne(-100)
                per_example = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
                if weights is None:
                    loss = per_example.mean()
                else:
                    weights = weights.to(per_example.device, dtype=per_example.dtype)
                    loss = (per_example * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
            return (loss, outputs) if return_outputs else loss

        def _get_train_sampler(self, *args: Any, **kwargs: Any) -> Any:
            if not preserve_order:
                return super()._get_train_sampler(*args, **kwargs)
            dataset = kwargs.get("train_dataset") or (args[0] if args else self.train_dataset)
            world_size = int(getattr(self.args, "world_size", 1))
            if world_size <= 1:
                return SequentialSampler(dataset)
            return DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=int(getattr(self.args, "process_index", 0)),
                shuffle=False,
                drop_last=bool(getattr(self.args, "dataloader_drop_last", False)),
            )

    return WeightedCausalTrainer
