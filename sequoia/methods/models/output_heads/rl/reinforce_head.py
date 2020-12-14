from abc import abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Union

import gym
import numpy as np
import torch
from gym import spaces
from torch import LongTensor, Tensor, nn
from torch.distributions import Categorical

from sequoia.common import Loss
from sequoia.common.layers import Lambda
from sequoia.methods.models.forward_pass import ForwardPass
from sequoia.settings.base.objects import Actions, Observations, Rewards
from sequoia.utils.utils import prod
from .policy_head import ClassificationOutput, PolicyHead

# TODO: Refactor this to use a RegressionHead for the predicted reward and a
# ClassificationHead for the choice of action?


class ReinforceHead(PolicyHead):

    @dataclass
    class HParams(PolicyHead.HParams):
        gamma: float = 0.9

    def __init__(self,
                 input_size: int,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: "OutputHead.HParams" = None,
                 name: str = "policy"):
        super().__init__(
            input_size=input_size,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        self.hparams: "ReinforceHead.HParams"

    def get_loss(self, forward_pass: ForwardPass, y: Tensor) -> Loss:
        # Extract our prediction from the forward pass.
        actions: ClassificationOutput = forward_pass.actions
        logits: Tensor = actions.logits
        y_pred: Tensor = actions.y_pred

    def policy_gradient_from_blogpost(self, rewards, log_probs):
        discounted_rewards = []
        gamma = 0.9
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
            
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        
        return torch.stack(policy_gradient).sum()


    def policy_gradient_from_blogpost_improved(self, rewards, log_probs):
        log_probs = torch.stack(log_probs)
        discounted_rewards = torch.empty_like(log_probs)
        gamma = 0.9
        for t in range(len(rewards)):
            Gt = 0 
            for pw, r in enumerate(rewards[t:]):
                Gt += gamma ** pw * r
            discounted_rewards[t] = Gt

        discounted_rewards = normalize(discounted_rewards) # normalize discounted rewards

        policy_gradient = - log_probs.dot(discounted_rewards)
        return policy_gradient

   
def policy_gradient_optimized(rewards: List[float], log_probs: List[Tensor], gamma: float):
    """Implementation of the REINFORCE algorithm.

    Adapted from https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

    Parameters
    ----------
    - episode_rewards : List[Tensor]

        The rewards at each step in an episode

    - episode_log_probs : List[Tensor]

        The log probabilities associated with the actions that were taken at
        each step.

    Returns
    -------
    Tensor
        The "policy gradient" resulting from that episode.
    """
    T = len(rewards)
    log_probs = torch.stack(log_probs)
    rewards = torch.as_tensor(rewards).type_as(log_probs)
    # Construct a reward matrix, with previous rewards masked out (with each
    # row as a step along the trajectory).
    reward_matrix = rewards.expand([T, T]).triu()
    # Get the gamma matrix (upper triangular), see make_gamma_matrix for
    # more info.
    gamma_matrix = make_gamma_matrix(gamma, T, device=log_probs.device)

    discounted_rewards = (reward_matrix * gamma_matrix).sum(dim=-1)
    # normalize discounted rewards
    discounted_rewards = normalize(discounted_rewards)
    policy_gradient = - log_probs.dot(discounted_rewards)
    return policy_gradient

# @torch.jit.script
# @lru_cache()
def make_gamma_matrix(gamma: float, T: int, device=None) -> Tensor:
    """
    Create an upper-triangular matrix [T, T] with the gamma factors,
    starting at 1.0 on the diagonal, and decreasing exponentially towards
    the right.
    """
    gamma_matrix = torch.empty([T, T]).triu_()
    # Neat indexing trick to fill up the upper triangle of the matrix:
    rows, cols = torch.triu_indices(T, T)
    # Precompute all the powers of gamma in range [0, T]
    all_gammas = gamma ** torch.arange(T)
    # Put the right value at each entry in the upper triangular matrix.
    gamma_matrix[rows, cols] = all_gammas[cols - rows]
    return gamma_matrix.to(device) if device else gamma_matrix

def normalize(x: Tensor):
    return (x - x.mean()) / (x.std() + 1e-9)
