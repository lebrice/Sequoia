from dataclasses import dataclass
from typing import Dict

import gym
import torch
from common import Batch, RegressionMetrics, Loss
from gym import spaces
from settings import Actions, Observations, Rewards
from torch import Tensor, nn
from utils import prod

from ..forward_pass import ForwardPass
from .output_head import OutputHead


class RegressionHead(OutputHead):
    def __init__(self,
                 input_size: int,
                 output_space: gym.Space,
                 hparams: OutputHead.HParams = None,
                 name: str = "regression"):
        assert isinstance(output_space, spaces.Box)
        if len(output_space.shape) > 1:
            raise NotImplementedError(
                f"TODO: Regression head doesn't support output shapes that are "
                f"more than 1d for atm, (output space: {output_space})."
            )
            # TODO: Add support for something like a "decoder head"?
        output_size = prod(output_space.shape)
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hparams=hparams,
            name=name,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, observations: Observations, representations: Tensor) -> Actions:
        y_pred = self.dense(representations)
        return Actions(y_pred)

    def get_loss(self, forward_pass: ForwardPass, y: Tensor) -> Loss:
        actions: Actions = forward_pass.actions
        y_pred: Tensor = actions.y_pred

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
