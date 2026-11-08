"""Common operator adapter; only frozen, preregistered donors are eligible."""
from shared.legacy import legacy_graft
from .delta_alignment import frozen


class CrossScaleOperator:
    def __init__(self, expert, insertion_layer, emergency_bounds=(.05, 10.0)):
        self.expert = expert
        self.insertion_layer = insertion_layer
        self.emergency_bounds = emergency_bounds

    def context(self, model):
        frozen(model)
        frozen(self.expert)
        return legacy_graft().Graft(model, self.expert, layer=self.insertion_layer,
                                    emergency_bounds=self.emergency_bounds)

    def refine(self, hidden, context):
        frozen(self.expert)
        return self.expert(hidden, attention_mask=context["attention_mask"],
                           position_ids=context["position_ids"])
