from dataclasses import dataclass
from typing import Dict, List

import gym
import torch
from gym import spaces
from torch import Tensor, nn

from sequoia.common import Batch, Loss, RegressionMetrics
from sequoia.settings import Actions, Observations, Rewards
from sequoia.utils import prod

from ..forward_pass import ForwardPass
from .output_head import OutputHead
from ..fcnet import FCNet

class RegressionHead(OutputHead):
    """ Output head used for regression problems. """

    @dataclass
    class HParams(FCNet.HParams, OutputHead.HParams):
        """ Hyper-parameters of the regression output head. """


    def __init__(self,
                 input_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space = None,
                 hparams: OutputHead.HParams = None,
                 name: str = "regression"):
        assert isinstance(action_space, spaces.Box)
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                f"TODO: Regression head doesn't support output shapes that are "
                f"more than 1d for atm, (output space: {action_space})."
            )
            # TODO: Add support for something like a "decoder head" (maybe as a
            # subclass of RegressionHead)?
        super().__init__(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        assert isinstance(action_space, spaces.Box)
        output_size = prod(action_space.shape)
        
        hidden_layers: List[nn.Module] = []
        in_features = self.input_size
        for i, neurons in enumerate(self.hparams.hidden_neurons):
            out_features = neurons
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(nn.ReLU())
            in_features = out_features # next input size is output size of prev.

        self.dense = nn.Sequential(
            nn.Flatten(),
            *hidden_layers,
            nn.Linear(in_features, output_size)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, observations: Observations, representations: Tensor) -> Actions:
        y_pred = self.dense(representations)
        return Actions(y_pred)

    def get_loss(self, forward_pass: ForwardPass, actions: Actions, rewards: Rewards) -> Loss:
        actions: Actions = forward_pass.actions
        y_pred: Tensor = actions.y_pred
        y: Tensor = rewards.y

        loss = self.loss_fn(y_pred, y)
        metrics = RegressionMetrics(y_pred=y_pred, y=y)

        assert self.name, "Output Heads should have a name!"
        loss = Loss(
            name=self.name,
            loss=loss,
            # NOTE: we're passing the tensors to the Loss object because we let
            # it create the Metrics for us automatically.
            metrics={self.name: metrics},
        )
        return loss
