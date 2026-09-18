"""Cache-based sampling with real behavior probabilities, no hints/gold in inputs."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .actor import Actor
from .sampling import Completion


@dataclass(frozen=True)
class Rollout:
    completion: Completion
    prompt: tuple[int, ...]
    tokens: tuple[int, ...]
    terminated: bool

    def validate(self, learner_version: int) -> None:
        self.completion.validate(learner_version)
        if not self.prompt or not self.tokens or len(self.tokens) != len(self.completion.behavior_logps):
            raise ValueError('rollout alignment invalid')
        if any(type(t) is not int or t < 0 for t in self.prompt+self.tokens) or type(self.terminated) is not bool:
            raise ValueError('rollout tokens/termination invalid')


@torch.no_grad()
def generate(actor: Actor, *, prompt: tuple[int,...], prompt_id: str, sample_id: str,
             generator: torch.Generator, verifier, eos: int, max_new: int = 128) -> Rollout:
    """CPU full-support temperature=1 policy; nothing is appended to the prompt.

    The caller reserves/settles resource cost. Exceptions are not refunded by this
    helper. No generated text is executed as code. Verifier is a trusted callable.
    """
    if next(actor.parameters()).device.type != 'cpu' or actor.training:
        raise ValueError('CPU eval actor required')
    if type(max_new) is not int or not 1 <= max_new <= 4096 or not callable(verifier):
        raise ValueError('invalid generation contract')
    if type(eos) is not int or not 0 <= eos < actor.backbone.config.vocab_size:
        raise ValueError('invalid EOS')
    if not prompt or any(type(t) is not int for t in prompt) or len(prompt)+max_new > actor.backbone.config.max_position_embeddings:
        raise ValueError('invalid prompt/context bound')
    versions,version=actor.parameter_versions(),actor.version
    ids=torch.tensor([prompt],dtype=torch.int64)
    tokens,logps=[],[]; cache=None; terminated=False
    for _ in range(max_new):
        logits,cache=actor(ids,cache=cache,use_cache=True,last_only=True,sequence_id=sample_id)
        logp=logits[0,-1].log_softmax(-1)
        token=int(torch.multinomial(logp.exp(),1,generator=generator))
        tokens.append(token); logps.append(float(logp[token]))
        if token == eos:
            terminated=True
            break
        ids=torch.tensor([[token]],dtype=torch.int64)
    if version != actor.version or versions != actor.parameter_versions():
        raise RuntimeError('policy changed during rollout')
    reward=verifier(tuple(tokens),terminated)
    completion=Completion(sample_id,prompt_id,version,reward,tuple(logps))
    result=Rollout(completion,prompt,tuple(tokens),terminated); result.validate(version)
    return result
