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

@dataclass
class RLHFTrainingArgs():

    # Basic / global
    seed: int = 1
    cuda: bool = t.cuda.is_available()

    # Wandb / logging
    exp_name: str = "RLHF_Implementation"
    wandb_project_name: Optional[str] = "rlhf"
    wandb_entity: Optional[str] = None
    use_wandb: bool = False

    # Duration of different phases
    total_phases: int = 200
    batch_size: int = 256
    num_minibatches: int = 4
    batches_per_learning_phase: int = 2

    # Optimization hyperparameters
    base_learning_rate: float = 2e-5
    head_learning_rate: float = 5e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 20
    final_scale: float = 0.1

    # Computing other PPO loss functions
    clip_coef: float = 0.2
    vf_coef: float = 0.15
    ent_coef: float = 0.001

    # Base model & sampling arguments
    base_model: str = "gpt2-medium"
    gen_len: int = 30
    temperature: float = 0.6
    prefix: str = "This is"

    # Extra stuff for RLHF
    kl_coef: float = 1.0
    reward_fn: Callable = None
    normalize_reward: bool = True

    def __post_init__(self):
        assert self.batch_size % self.num_minibatches == 0, "Batch size should be divisible by the number of minibatches."
        self.minibatch_size = self.batch_size // self.num_minibatches