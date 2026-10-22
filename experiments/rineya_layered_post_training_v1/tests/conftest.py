from dataclasses import replace
import pytest
import torch
from experiments.rineya_layered_post_training_v1.config import Config
from experiments.rineya_layered_post_training_v1.runtime import tiny_runtime


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    return replace(Config(), actor_steps=28, critic_steps=6, max_prompt_tokens=8, max_new_tokens=2, max_context_tokens=16, learning_rate=.005, critic_lr=.01, min_eval_groups=2, required_suites=("acquisition", "retention"), bootstrap_samples=100)


@pytest.fixture
def rt(cfg):
    return tiny_runtime(cfg)
