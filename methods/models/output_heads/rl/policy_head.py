from dataclasses import dataclass
from typing import Dict, Tuple, Union, List
from abc import abstractmethod, ABC

import gym
import numpy as np
import torch
from gym import spaces
from torch import LongTensor, Tensor, nn

from common import Loss
from common.layers import Lambda
from settings.base.objects import Actions, Observations, Rewards
from utils.utils import prod

from methods.models.forward_pass import ForwardPass 
from ..classification_head import ClassificationOutput, ClassificationHead
from ..output_head import OutputHead
from torch.distributions import Categorical


@dataclass(frozen=True)
class PolicyHeadOutput(ClassificationOutput):
    """ WIP: Adds the action pdf to ClassificationOutput. """
    action_pdf: Categorical

    @property
    def y_pred_log_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.action_pdf.log_prob(self.y_pred)

    @property
    def y_pred_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.action_pdf.probs(self.y_pred)


class PolicyHead(ClassificationHead):
    def __init__(self,
                 input_size: int,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: "OutputHead.HParams" = None,
                 name: str = "policy"):
        assert isinstance(action_space, spaces.Discrete), f"Only support discrete action space for now (got {action_space})."
        assert isinstance(reward_space, spaces.Box), f"Reward space should be a Box (scalar rewards) (got {reward_space})."
        super().__init__(
            input_size=input_size,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        self.action_space: spaces.Discrete
        self.reward_spaces: spaces.Box
        self.softmax = nn.Softmax()
        self.density: Categorical


    def forward(self, observations: Observations, representations: Tensor) -> PolicyHeadOutput:
        """ Forward pass of a Policy head.
        
        NOTE: (@lebrice) This is identical to the forward pass of the
        ClassificationHead atm, but I'm also testing out sampling the actions
        from the probabilities described by the logits, rather than by always
        selecting the action with the highest probability.
        
        Very similar to the forward pass of
        the ClassificationHead, but samples the action from the probabilites
        rather than take the one with highest probability.
        """
        # Get the raw / unscaled logits for each action using the
        # ClassificationHead's forward method.

        # NOTE: Not sure if this is that useful.
        # Also, doesn't work on CUDA atm, for some reason.
        
        # Choose the actions according to their probabilities, rather than just
        # taking the action with highest probability, as is done in the
        # ClassificationHead.
        logits = self.dense(representations)

        density = Categorical(logits=logits)
        actions = density.sample()

        # NOTE: Since the action space is discrete(n), the index chosen also
        # corresponds to the chosen action.
        output = PolicyHeadOutput(
            logits=logits,
            y_pred=actions,
            action_pdf=density
        )
        return output

    def get_loss(self, forward_pass: ForwardPass, y: Tensor) -> Loss:
        """ Given the forward pass (including the actions produced by this
        output head), and the corresponding rewards, get a Loss to use for
        training.

        NOTE: The training procedure is fundamentally on-policy atm, i.e. the
        observation is a single state, not a rollout, and the reward is the
        immediate reward at the current step. Therefore, this should be taken
        into consideration when implementing an RL output head.
        """
        # Extract the outputs of this head from the forwardpass object.
        # TODO: Its a bit dumb to have to retrieve our own predictions from the forward pass dict this way.
        actions: PolicyHeadOutput = forward_pass.actions
        rewards = y
        m = actions.action_pdf
        loss =  - actions.y_pred_log_prob * rewards
        return Loss(self.name, loss)


def normalize(x: Tensor):
    return (x - x.mean()) / (x.std() + 1e-9)