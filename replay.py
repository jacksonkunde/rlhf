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

from utils import RLHFTrainingArgs

@dataclass
class ReplayMinibatch:
    '''
    Samples from the replay memory.
    '''
    sample_ids: Float[Tensor, "minibatch_size seq_len"]
    logprobs: Float[Tensor, "minibatch_size seq_len"]
    advantages: Float[Tensor, "minibatch_size gen_len"]
    returns: Float[Tensor, "minibatch_size gen_len"]
    ref_logits: Float[Tensor, "minibatch_size seq_len d_vocab"]


class ReplayMemory:
    def __init__(
        self,
        args: RLHFTrainingArgs,
        sample_ids: Float[Tensor, "batch_size seq_len"],
        logprobs: Float[Tensor, "batch_size seq_len"],
        advantages: Float[Tensor, "batch_size gen_len"],
        values: Float[Tensor, "batch_size seq_len"],
        ref_logits: Float[Tensor, "batch_size seq_len d_vocab"],
    ):
        '''
        Initializes the replay memory, with all the data generated from the rollout phase at once.

        The advantages are (batch_size, gen_len) because we only compute advantages for the generated
        tokens. The other tensors are (batch_size, seq_len) because they are computed for all tokens.
        '''
        self.args = args
        self.sample_ids = sample_ids
        self.logprobs = logprobs
        self.advantages = advantages
        self.values = values
        self.ref_logits = ref_logits


    def get_minibatches(self) -> List[ReplayMinibatch]:
        '''
        Generates a list of minibatches by randomly sampling from the replay memory. Each sequence appears
        exactly `batches_per_learning_phase` times in total.
        '''
        minibatches = []

        returns = self.advantages + self.values[:, -self.args.gen_len-1:-1]

        for _ in range(self.args.batches_per_learning_phase):

            idxs = t.randperm(self.args.batch_size).reshape(self.args.num_minibatches, self.args.minibatch_size)
            for idx in idxs:
                minibatches.append(
                    ReplayMinibatch(
                        sample_ids = self.sample_ids[idx],
                        logprobs = self.logprobs[idx],
                        advantages = self.advantages[idx],
                        returns = returns[idx],
                        ref_logits = self.ref_logits[idx],
                    )
                )

        return minibatches