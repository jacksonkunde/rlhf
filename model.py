import torch as t
import torch.nn as nn
from torch import Tensor
import wandb
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer
from typing import List, Optional, Tuple, Union, Dict, Any, Callable
import einops
from jaxtyping import Float, Int
import os
import sys
from pathlib import Path
from rich import print as rprint
from rich.table import Table
from dataclasses import dataclass
import numpy as np
import time
from functools import partial

class TransformerWithValueHead(nn.Module):
    '''
    Defines a GPT model with a value head (the latter taking the last hidden state as input,
    post-layernorm).

    The value head is a simple MLP with one hidden layer, and scalar output:

        Linear(d_model -> 4*d_model)
        ReLU
        Linear(4*d_model -> 1)

    All linear layers have biases.
    '''
    base_model: HookedTransformer
    value_head: nn.Sequential

    def __init__(self, base_model: str = "gpt2-medium"):
        super().__init__()
        self.base_model = HookedTransformer.from_pretrained(base_model)

        d_model = self.base_model.cfg.d_model

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, 1)
        )

        self.value_est = None

    def forward(self, input_ids: Int[Tensor, "batch seq"]) -> Tuple[
        Float[Tensor, "batch seq d_vocab"],
        Int[Tensor, "batch seq"]
    ]:
        def get_value_est(resid_post: Float[Tensor, "batch seq d_model"], hook: HookPoint):
            self.value_est = self.value_head(resid_post).squeeze(-1)

        logits = self.base_model.run_with_hooks(
            input_ids,
            return_type = "logits",
            fwd_hooks = [
                (utils.get_act_name("normalized"), get_value_est)
            ]
        )
        assert self.value_est is not None

        return logits, self.value_est