"""Fixed contiguous middle block; licensed only by completed reapplication gates."""
from __future__ import annotations

from typing import Any

from .hooks import Reapplication


class FixedBlock(Reapplication):
    def __init__(self, model: Any, layers: list[int], applications: int = 2, **kwargs):
        if len(layers) != 2 or any(type(i) is not int for i in layers) or layers[1] != layers[0] + 1:
            raise ValueError("initial block must contain two contiguous layers")
        if layers[0] < 3 or layers[-1] >= min(25, len(model.model.layers)):
            raise ValueError("block must stay inside the anchored middle region")
        if any(model.config.layer_types[i] != "full_attention" for i in layers):
            raise ValueError("mixed/sliding attention block semantics are unaudited")
        self.layers = tuple(layers)
        super().__init__(model, layer=layers[-1], applications=applications, **kwargs)

    @property
    def extra_layer_applications(self) -> int:
        return len(self.layers) * super().extra_layer_applications

    def __enter__(self):
        super().__enter__()
        if not self.enabled:
            return self
        last = self.original
        leading = [self.model.model.layers[i].forward for i in self.layers[:-1]]

        def operator(*args, **kwargs):
            # The ordinary stack already ran leading layers on the first pass.
            # Subsequent applications replay the WHOLE block in original order.
            if self.calls % self.applications == 0:
                return last(*args, **kwargs)
            hidden = args[0] if args else kwargs["hidden_states"]
            for forward in leading:
                if args:
                    hidden = forward(hidden, *args[1:], **kwargs)
                else:
                    hidden = forward(**{**kwargs, "hidden_states": hidden})
            if args:
                return last(hidden, *args[1:], **kwargs)
            return last(**{**kwargs, "hidden_states": hidden})

        self.original = operator
        return self
