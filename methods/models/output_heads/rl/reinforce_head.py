from dataclasses import dataclass
from typing import Dict, Tuple, Union, List
from abc import abstractmethod

import gym
import numpy as np
import torch
from gym import spaces
from torch import LongTensor, Tensor, nn

from common import Loss
from common.layers import Lambda
from settings.base.objects import Actions, Observations, Rewards
from utils.utils import prod

from torch.distributions import Categorical
from methods.models.forward_pass import ForwardPass 
from .policy_head import PolicyHead, ClassificationOutput

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
        

    def get_policy_gradient_for_episode(self, episode_rewards: List[Tensor],
                                              episode_log_probs: List[Tensor]) -> Tensor:
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
        rewards = torch.as_tensor(episode_rewards)
        log_probs = torch.as_tensor(episode_log_probs)

        discounted_rewards_list: List[Tensor] = []

        gamma = self.hparams.gamma
        assert rewards.dim() == 1, "not sure if we can support batches at the moment."
        
        T = len(rewards)

        all_indices = torch.arange(T)
        gammas = torch.pow(gamma, all_indices)

        # Create a 2D view of the rewards
        rewards_matrix = rewards.expand([T, T])
        # Only keep the 'future' rewards. Row `i` corresponds to step 'i'.
        future_rewards = rewards_matrix.triu()
        
        discounted_rewards = torch.empty_like(rewards, requires_grad=True)
        discounted_rewards_list: List[Tensor] = []
        for t in range(T):
            # Calculate the discounted sum of future rewards at step t.
            gammas = torch.pow(gamma, torch.arange(T - t))
            discounted_sum_of_future_rewards = (gammas * rewards[t:]).sum()
            discounted_rewards_list.append(discounted_sum_of_future_rewards)
        
        discounted_rewards = torch.as_tensor(discounted_rewards)
        # normalize discounted rewards
        discounted_rewards = normalize(discounted_rewards) 
        # Sum of the product of the log prob and the discounted rewards. 
        policy_gradient =  - log_probs.dot(discounted_rewards)
        return policy_gradient


def random_choice_prob_index(probabilities: Tensor, axis: int = 1, normalize: bool=False) -> LongTensor:
    """ vectorized, pytorch version of np.random.choice, for a 2D array of probabilites.
    
    Taken from https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
    """
    assert probabilities.dim() == 2
    if normalize:
        probabilities = probabilities.softmax(dim=axis)
    r = torch.unsqueeze(torch.rand(probabilities.shape[0]), dim=axis)
    return (probabilities.cumsum(dim=axis) > r).argmax(dim=axis)


def normalize(x: Tensor):
    return (x - x.mean()) / (x.std() + 1e-9)