"""The actual installed donor decoder, streamed without loading a whole donor."""
from .legacy import legacy_graft


def capture_native(*args, **kwargs):
    return legacy_graft().capture_native(*args, **kwargs)


def load_layer(*args, **kwargs):
    return legacy_graft().load_layer(*args, **kwargs)


def positions_mask(*args, **kwargs):
    return legacy_graft().positions_mask(*args, **kwargs)
