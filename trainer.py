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

# import from other files
from utils import RLHFTrainingArgs
from model import TransformerWithValueHead
from replay import ReplayMemory, ReplayMinibatch

# util fuctions for RLHFTrainer
@t.no_grad()
def get_samples(base_model: HookedTransformer, prompt: str, batch_size: int, gen_len: int, temperature: float):
    '''
    Generates samples from the model, which will be fed into the reward model and evaluated.

    Inputs:
        base_model: the transformer to generate samples from (note we use gpt, not the model wrapper, cause we don't need value head)
        prompt: the initial prompt fed into the model
        batch_size: the number of samples to generate
        gen_len: the length of the generated samples (i.e. the number of *new* tokens to generate)

    Returns:
        sample_ids: the token ids of the generated samples (including initial prompt)
        samples: the generated samples (including initial prompt)
    '''
    # Make sure we've passed in the base model (the bit we use for sampling)
    assert not isinstance(base_model, TransformerWithValueHead), "Please pass in the base model, not the model wrapper."

    # Convert our prompt into tokens
    input_ids = base_model.to_tokens(prompt, prepend_bos=False).squeeze(0)

    # Generate samples (we repeat the input ids which is a bit wasteful but ¯\_(ツ)_/¯)
    input_ids = einops.repeat(input_ids, "seq -> batch seq", batch=batch_size)

    # Generate samples
    output_ids = base_model.generate(
        input_ids,
        max_new_tokens = gen_len,
        stop_at_eos = False,
        temperature = temperature, # higher means more random completions
        verbose = False,
    )
    samples = base_model.to_string(output_ids)

    return output_ids.clone(), samples


def normalize_reward(reward: Float[Tensor, "batch_size"], eps=1e-5) -> Float[Tensor, "batch_size"]:
    '''
    Normalizes the reward function values over the batch of sequences.
    '''
    return (reward - reward.mean())/(reward.std() + eps)

@t.no_grad()
def compute_advantages(
    values: Float[Tensor, "minibatch_size seq_len"],
    rewards: Float[Tensor, "minibatch_size"],
    prefix_len: int,
) -> Float[Tensor, "minibatch_size gen_len"]:
    '''
    Computes the advantages for the PPO loss function, i.e. A_pi(s, a) = Q_pi(s, a) - V_pi(s).

    In this formula we replace Q(s, a) with the 1-step Q estimates, and V(s) with the 0-step value estimates.

    Inputs:
        values:
            the value estimates for each token in the generated sequence
        rewards:
            the rewards for the entire generated sequence
        prefix_len:
            the length of the prefix (i.e. the length of the initial prompt)

    Returns:
        advantages:
            the advantages for each token in the generated sequence (not the entire sequence)
    '''
    one_step_q_est = t.cat([
        values[:, prefix_len:-1], # shape [minibatch_size, gen_len-1]
        rewards[:, None], # shape [minibatch_size, 1]
    ], dim=-1)

    zero_step_value_est = values[:, prefix_len-1:-1]  # shape [minibatch_size, gen_len]

    advantages = one_step_q_est - zero_step_value_est

    return advantages

# compute loss function helpers
def calc_kl_penalty(
    logits: Float[Tensor, "minibatch_size seq_len d_vocab"],
    ref_logits: Float[Tensor, "minibatch_size seq_len d_vocab"],
    kl_coef: float,
    prefix_len: int,
) -> Float[Tensor, ""]:
    '''
    Computes the KL divergence between the logits and the reference logits, scaled
    by the penalty function. This is used to stop the learned policy from diverging
    too much from the original reference model's policy.

    logits:
        The logits of the generated samples (under the new model).
    ref_logits:
        The logits of the generated samples (under the reference model).
    kl_coef:
        The coefficient of the KL penalty.
    prefix_len:
        The length of the prefix to ignore when computing the KL divergence.
    '''

    log_probs = logits.log_softmax(-1) # P
    ref_log_probs = ref_logits.log_softmax(-1) # Q
    probs = log_probs.exp()

    # compute the KL-div /sum_i
    kl_div = (probs * (log_probs - ref_log_probs))[:, prefix_len-1:-1, :].sum(-1)

    return (kl_coef * kl_div.mean())


def calc_entropy_bonus(
    logits: Float[Tensor, "minibatch_size seq_len d_vocab"],
    ent_coef: float,
    prefix_len: int
) -> Float[Tensor, ""]:
    '''
    Return the entropy bonus term, suitable for gradient ascent.

    logits:
        the logits of the tokens generated by the model.
    ent_coef:
        the coefficient for the entropy loss, which weights its contribution to the overall objective function.
    prefix_len:
        The length of the prefix to ignore when computing the KL divergence.
    '''
    log_probs = logits[:, prefix_len-1:-1, :].log_softmax(-1) # P
    probs = log_probs.exp()

    entropy = (-1 * probs * log_probs).sum(-1)
    return ent_coef * entropy.mean()


def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size gen_len"],
    mb_returns: Float[Tensor, "minibatch_size gen_len"],
    vf_coef: float
) -> Float[Tensor, ""]:
    '''Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    assert values.shape == mb_returns.shape,\
        f"Shape mismatch: {values.shape=}, {mb_returns.shape=}. Did you slice 'values' tokens correctly?"
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()


def calc_clipped_surrogate_objective(
    logprobs: Float[Tensor, "minibatch_size gen_len"],
    mb_logprobs: Float[Tensor, "minibatch_size gen_len"],
    mb_advantages: Float[Tensor, "minibatch_size gen_len"],
    clip_coef: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    logprobs:
        the logprobs of the action taken by the agent, according to the new policy
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    mb_advantages:
        advantages calculated from the sampled minibatch
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    '''
    assert logprobs.shape == mb_logprobs.shape == mb_advantages.shape,\
        f"Shape mismatch: {logprobs.shape=}, {mb_logprobs.shape=}, {mb_advantages.shape=}. Did you create logprobs using 'get_logprobs' correctly?"

    logits_diff = logprobs - mb_logprobs

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()


def get_logprobs(
    logits: Float[Tensor, "batch seq_len vocab"],
    tokens: Int[Tensor, "batch seq_len"],
    prefix_len: int = 1,
) -> Float[Tensor, "batch gen_len"]:
    '''
    Returns correct logprobs for the given logits and tokens, for all the tokens
    after the prefix tokens (which have length equal to `prefix_len`).

    If prefix_len = None then we return shape (batch, seq_len-1). If not, then
    we return shape (batch, seq_len-prefix_len) representing the predictions for
    all tokens after the prefix tokens.
    '''
    prefix_len = prefix_len or 1
    log_probs = logits.log_softmax(-1)
    log_probs = t.gather(log_probs[:, prefix_len-1:-1], dim=-1, index=tokens[:, prefix_len:, None]).squeeze()

    return log_probs


# Optimizer
def get_optimizer(args: RLHFTrainingArgs, model: TransformerWithValueHead) -> t.optim.Optimizer:
    '''
    Returns an Adam optimizer for the model, with the correct learning rates for the base and head.
    '''
    return t.optim.Adam([
        {"params": model.base_model.parameters(), "lr": args.base_learning_rate},
        {"params": model.value_head.parameters(), "lr": args.head_learning_rate},
    ], maximize=True)

def get_lr_scheduler(warmup_steps, total_steps, final_scale):
    '''
    Creates an LR scheduler that linearly warms up for `warmup_steps` steps,
    and then linearly decays to `final_scale` over the remaining steps.
    '''
    def lr_lambda(step):
        assert step <= total_steps, f"Step = {step} should be less than total_steps = {total_steps}."
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1 - (1 - final_scale) * (step - warmup_steps) / (total_steps - warmup_steps)

    return lr_lambda


def get_optimizer_and_scheduler(args: RLHFTrainingArgs, model: TransformerWithValueHead):
    optimizer = get_optimizer(args, model)
    lr_lambda = get_lr_scheduler(args.warmup_steps, args.total_phases, args.final_scale)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler



# main class definition, used to train the RLHF model
class RLHFTrainer:
    model: TransformerWithValueHead
    ref_model: HookedTransformer
    memory: ReplayMemory
    device: t.device

    def __init__(self, args: RLHFTrainingArgs):
        t.manual_seed(args.seed)
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.args = args
        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        self.model = TransformerWithValueHead(args.base_model).to(self.device).train()
        self.ref_model = HookedTransformer.from_pretrained(args.base_model).to(self.device).eval()
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.prefix_len = len(self.model.base_model.to_str_tokens(self.args.prefix, prepend_bos=False))


    def compute_rlhf_objective(self, mb: ReplayMinibatch):
        '''
        Computes the RLHF objective function to maximize, which equals the PPO objective function minus
        the KL penalty term.

        Steps of this function are:
            - Get logits & values for the samples in minibatch
            - Get the logprobs of the minibatch actions taken
            - Use this data to compute all 4 terms of the RLHF objective function, and create function
        '''
        logits, values = self.model(mb.sample_ids) # get the logits, values for the samples in the minibatch
        values = values[:, self.prefix_len-1:-1] # get the values only for the generated tokens

        logprobs = get_logprobs(logits, mb.sample_ids, self.prefix_len) # get the logprobs

        # compute the parts of the objective function
        clipped_surrogate_objective = calc_clipped_surrogate_objective(logprobs, mb.logprobs, mb.advantages, self.args.clip_coef)
        value_function_loss = calc_value_function_loss(values, mb.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus(logits, self.args.ent_coef, self.prefix_len)
        kl_penalty = calc_kl_penalty(logits, mb.ref_logits, self.args.kl_coef, self.prefix_len)

        # Compute the full objective function
        objective_function = clipped_surrogate_objective - value_function_loss + entropy_bonus
        objective_function = objective_function - kl_penalty

        # Logging
        with t.inference_mode():
            logratio = logprobs - mb.logprobs
            ratio = logratio.exp()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        if self.args.use_wandb: wandb.log(dict(
            total_steps = self.step,
            learning_rate = self.scheduler.get_last_lr()[0],
            clipped_surrogate_objective = clipped_surrogate_objective.item(),
            clipfrac = np.mean(clipfracs),
            value_loss = value_function_loss.item(),
            values = values.mean().item(),
            entropy_bonus = entropy_bonus.item(),
            kl_penalty = kl_penalty.item(),
        ), step=self.step)

        return objective_function


    def rollout_phase(self) -> ReplayMemory:
        '''
        Performs a single rollout phase, returning a ReplayMemory object containing the data generated
        during this phase. Note that all forward passes here should be done in inference mode.

        Steps of this function are:
            - Generate samples from our model
            - Get logits of those generated samples (from model & reference model)
            - Get other data for memory (logprobs, normalized rewards, advantages)
            - Return this data in a ReplayMemory object
        '''
        # Get samples
        sample_ids, samples_text = get_samples(
            self.model.base_model,
            prompt=self.args.prefix,
            batch_size=self.args.batch_size,
            gen_len=self.args.gen_len,
            temperature=self.args.temperature,
        )

        # pass the sample ids into the RLHF model and the ref model
        with t.inference_mode():
            logits, value_est = self.model(sample_ids)
            ref_logits = self.ref_model(sample_ids)

        # get the lob probs for the tokens we predicted.
        logprobs = get_logprobs(logits, sample_ids, self.prefix_len)


        # get the rewards (and normalize)
        rewards = self.args.reward_fn(samples_text)
        normalized_rewards = normalize_reward(rewards)

        # ensure on correct device
        if normalized_rewards.device != self.device:
            normalized_rewards = normalized_rewards.to(self.device)
        if value_est.device != device:
            value_est.to(self.device)

        advantages = compute_advantages(value_est, normalized_rewards, self.prefix_len)

        # Logging
        mean_reward = rewards.mean().item()
        if self.args.use_wandb: wandb.log({'mean_reward': mean_reward}, step=self.step)
        ref_logprobs = get_logprobs(ref_logits[:3], sample_ids[:3], self.prefix_len).sum(-1)
        table = Table(
            "Reward", "Ref logprobs", "Sample",
            title=f"Phase {self.phase:03}/{self.args.total_phases}, Mean reward: {mean_reward:.4f}",
            show_lines=True
        )
        for r, lp, s in zip(rewards.tolist(), ref_logprobs, samples_text):
            table.add_row(str(int(r)), f"{lp:.2f}", repr(s))
        rprint(table); print("")

        return ReplayMemory(
            args = self.args,
            sample_ids = sample_ids,
            logprobs = logprobs,
            advantages = advantages,
            values = value_est,
            ref_logits = ref_logits,
        )

    def learning_phase(self, memory: ReplayMemory) -> None:
        '''
        Performs a learning step on `self.memory`. This involves the standard gradient descent steps
        (i.e. zeroing gradient, computing objective function, doing backprop, stepping optimizer).

        You should also remember the following:
            - Clipping grad norm to the value given in `self.args.max_grad_norm`
            - Incrementing `self.step` by 1 for each phase (not minibatch!)
            - Stepping the scheduler (once per calling of this function)
        '''
        for mb in memory.get_minibatches():
            self.optimizer.zero_grad() # zero out grad
            objective_fcn = self.compute_rlhf_objective(mb) # compute value of obj func
            objective_fcn.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.args.max_grad_norm) # clip to prevent exploding grads
            self.optimizer.step()
            self.step += 1

        self.scheduler.step()



    def train(self) -> None:
        '''
        Performs a full training run.
        '''
        self.step = 0

        if self.args.use_wandb: wandb.init(
            project = self.args.wandb_project_name,
            entity = self.args.wandb_entity,
            name = self.run_name,
            config = self.args,
        )

        for self.phase in range(self.args.total_phases):
            memory = self.rollout_phase()
            self.learning_phase(memory)

        if self.args.use_wandb: wandb.finish()