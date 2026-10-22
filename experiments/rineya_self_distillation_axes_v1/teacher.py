"""Memory-bounded teacher: share immutable backbone, bank only changed parameters.

Equivalent to a separate teacher when all other parameters/buffers are fixed.
Never swap during a live student graph. Contexts are local and non-concurrent.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import threading
import torch
from .objectives import ema_update


def tensor_digest(named_tensors) -> str:
    h = hashlib.sha256()
    for name, t in sorted(named_tensors, key=lambda pair: pair[0]):
        h.update(json.dumps([name, list(t.shape), str(t.dtype)]).encode())
        flat = t.detach().reshape(-1)
        chunk = max(1, 1048576 // t.element_size())
        for start in range(0, flat.numel(), chunk):
            h.update(flat[start:start + chunk].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class TeacherBank:
    def __init__(self, model):
        self.model = model
        self.selected = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not self.selected or any(p.dtype != torch.float32 for p in self.selected.values()):
            raise ValueError("Select FP32 master parameters before creating the teacher")
        self.initial = self.snapshot()
        self.teacher = {n: p.clone() for n, p in self.initial.items()}
        self._busy = False
        self._owner = threading.get_ident()

    def snapshot(self):
        return {n: p.detach().cpu().clone() for n, p in self.selected.items()}

    def frozen_digest(self):
        tensors = [(n, p) for n, p in self.model.named_parameters() if n not in self.selected]
        tensors += [("buffer:" + n, b) for n, b in self.model.named_buffers()]
        return tensor_digest(tensors)

    def _enter(self):
        if self._busy or threading.get_ident() != self._owner:
            raise RuntimeError("Overlapping graphs/teacher swaps or concurrent use forbidden")
        self._busy = True

    @contextmanager
    def student_graph(self):
        """Keep this scope open through backward and the optimizer update."""
        self._enter()
        try:
            yield
        finally:
            self._busy = False

    @contextmanager
    def use(self, which="teacher"):
        if which not in {"teacher", "initial"}:
            raise ValueError("Unknown parameter bank")
        self._enter()
        saved = None
        try:
            saved = self.snapshot()
            target = self.teacher if which == "teacher" else self.initial
            with torch.no_grad():
                for n, p in self.selected.items():
                    p.copy_(target[n])
                yield
        finally:
            try:
                if saved is not None:
                    with torch.no_grad():
                        for n, p in self.selected.items():
                            p.copy_(saved[n])
            finally:
                self._busy = False

    def update(self, rate):
        self._enter()
        try:
            ema_update(self.teacher, self.selected, rate)
        finally:
            self._busy = False

    def reset(self):
        self._enter()
        try:
            with torch.no_grad():
                for n, p in self.selected.items():
                    p.copy_(self.initial[n])
            self.teacher = {n: p.clone() for n, p in self.initial.items()}
            self.model.zero_grad(set_to_none=True)
        finally:
            self._busy = False
