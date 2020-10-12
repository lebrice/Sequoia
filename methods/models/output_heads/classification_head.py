from dataclasses import dataclass
from typing import Dict, List

import gym
import torch
from common import Batch, ClassificationMetrics, Loss
from gym import spaces
from torch import Tensor, nn

from .output_head import OutputHead
from ..forward_pass import ForwardPass
from settings import Observations, Actions, Rewards


@dataclass(frozen=True)
class ClassificationActions(Actions):
    y_pred: Tensor
    logits: Tensor


class ClassificationHead(OutputHead):
    def __init__(self,
                 input_size: int,
                 output_space: gym.Space,
                 hparams: "OutputHead.HParams" = None,
                 name: str = "classification"):
        super().__init__(
            input_size=input_size,
            output_space=output_space,
            hparams=hparams,
            name=name,
        )
        assert isinstance(output_space, spaces.Discrete)
        output_size = output_space.n
        
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
        # if output_size == 2:
        #     # TODO: Should we be using this loss instead?
        #     self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, observations: Observations, representations: Tensor) -> Actions:
        # TODO: This should probably take in a dict and return a dict, or something like that?
        # TODO: We should maybe convert this to also return a dict instead
        # of a Tensor, just to be consistent with everything else. This could
        # also maybe help with having multiple different output heads, each
        # having a different name and giving back a dictionary of their own
        # forward pass tensors (if needed) and predictions?
        logits = self.dense(representations)
        y_pred = logits.argmax(dim=-1)
        return ClassificationActions(
            logits=logits,
            y_pred=y_pred,
        )

    def get_loss(self, forward_pass: ForwardPass, y: Tensor) -> Loss:
        actions: ClassificationActions = forward_pass.actions
        logits: Tensor = actions.logits
        y_pred: Tensor = actions.y_pred

        n_classes = logits.shape[-1]
        # Could remove these: just used for debugging.
        assert len(y.shape) == 1, y.shape
        assert not torch.is_floating_point(y), y.dtype
        assert 0 <= y.min(), y
        assert y.max() < n_classes, y

        loss = self.loss_fn(logits, y)
        
        assert loss.shape == ()
        metrics = ClassificationMetrics(y_pred=logits, y=y)
        
        assert self.name, "Output Heads should have a name!"
        loss_object = Loss(
            name=self.name,
            loss=loss,
            # NOTE: we're passing the tensors to the Loss object because we let
            # it create the Metrics for us automatically.
            metrics={self.name: metrics},
        )
        return loss_object
