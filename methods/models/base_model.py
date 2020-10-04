"""Base class for the Model to be used as part of a Method.

This is meant

TODO: There is a bunch of work to be done here.
"""
import dataclasses
import itertools
import functools
from abc import ABC
from dataclasses import dataclass
from collections import abc as collections_abc
from typing import (Any, ClassVar, Dict, Generic, List, NamedTuple, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from simple_parsing.helpers.flatten import FlattenedAccess
from torch import Tensor

from common.config import Config
from common.loss import Loss
from methods.models.output_heads import OutputHead
from utils.logging_utils import get_logger

from .base_hparams import BaseHParams
logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=LightningDataModule)
from .split_batch import split_batch
from .batch import Batch, Observation, Action, Reward


@dataclass(frozen=True)
class ForwardPass(Batch, FlattenedAccess):
    """ Typed version of the result of a forward pass through a model.

    FlattenedAccess is really sweet. We can get/set any attributes in the
    children by getting/setting them directly on the parent. So if the
    `observation` has an `x` attribute, we can get on this object directly with
    `self.x`, and it will fetch the attribute from the observation. Same goes
    for setting the attribute.
    """  
    observations: Observation
    representations: Any
    actions: Action

    @property
    def h_x(self) -> Any:
        return self.representations


class BaseModel(LightningModule, Generic[SettingType]):
    """ Base model LightningModule (nn.Module extended by pytorch-lightning)
    
    WIP: (@lebrice): Trying to tidy up the hierarchy of the different kinds of
    models a little bit. 
    
    This model splits the learning task into a representation-learning problem
    and a downstream task (output head) applied on top of it.   

    The most important method to understand is the `get_loss` method, which
    is used by the [train/val/test]_step methods which are called by
    pytorch-lightning.
    """
    # NOTE: we put this here just so its easier to subclass these classes from
    # future subclasses of 'Model'.
    Observation = Observation
    Action = Action
    Reward = Reward

    @dataclass
    class HParams(BaseHParams):
        """ HParams of the Model. """

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__()
        self.setting: SettingType = setting
        self.hp = hparams
        self.config: Config = config
        # TODO: Decided to Not set this property, so the trainer doesn't
        # fallback to using it instead of the passed datamodules/dataloaders.
        # self.datamodule: LightningDataModule = setting

        # TODO: Debug this, seemed to be causing issues because it was trying to
        # save the 'setting' argument, which contained some things that weren't
        # pickleable.
        all_params_dict = {
            "hparams": hparams.to_dict(),
            "config": config.to_dict(),
        }
        self.save_hyperparameters(all_params_dict)

        self.input_shape  = self.setting.dims
        self.output_shape = self.setting.action_shape
        self.reward_shape = self.setting.reward_shape
        # (Testing) Setting this attribute is supposed to help with ddp/etc
        # training in pytorch-lightning. Not 100% sure.
        # self.example_input_array = torch.rand(self.batch_size, *self.input_shape)

        # Create the encoder and the output head. 
        self.encoder, self.hidden_size = self.hp.make_encoder()
        self.output_head = self.create_output_head()

    # @auto_move_data
    def forward(self, input_batch: Any) -> ForwardPass:
        """ Forward pass of the Model. Returns a dict.
        """
        observations = self.Observation.from_inputs(input_batch)
        
        # Encode the observation to get representations.
        representations = self.encode(observations)
        # Pass the observations and representations to the output head to get
        # the 'action' (prediction).
        actions = self.get_actions(observations, representations)

        forward_pass = ForwardPass(
            observations=observations,
            representations=representations,
            actions=actions,
        )
        return forward_pass

    def encode(self, observation: Union[Tensor, Observation]) -> Tensor:
        """Encodes a batch of samples `x` into a hidden vector.

        Args:
            observations (Union[Tensor, Observation]): Tensor of Observation
            containing a batch of samples (before preprocess_observations).

        Returns:
            Tensor: The hidden vector / embedding for that sample, with size
                [<batch_size>, `self.hp.hidden_size`].
        """
        preprocessed_observations = self.preprocess_observations(observation)
        # Here in this base model we only use the 'x' from the observations.
        x: Tensor
        if isinstance(observation, Tensor):
            x = observation
        else:
            x = observation.x

        h_x = self.encoder(x)
        if isinstance(h_x, list) and len(h_x) == 1:
            # Some pretrained encoders sometimes give back a list with one tensor. (?)
            h_x = h_x[0]
        return h_x

    def get_actions(self, observations: Union[Tensor, Observation], h_x: Tensor) -> Action:
        """ Pass the required inputs to the output head and get predictions.
        
        NOTE: This method is basically just here so we can customize what we
        pass to the output head, or what we take from it, similar to the
        `encode` method.
        """
        # Here in this base model we only use the 'x' from the observation.
        x: Tensor = observations if isinstance(observations, Tensor) else observations.x
        if self.hp.detach_output_head:
            h_x = h_x.detach()
        y_pred = self.output_head(x, h_x)
        # TODO: Actually change the return type of the output head maybe, so it
        # gives back an action?
        return self.Action(y_pred)

    def create_output_head(self) -> OutputHead:
        """ Create the output head for the task. """
        return OutputHead(self.hidden_size, self.output_shape, name="classification")

    def training_step(self, batch: Tuple[Tensor, Optional[Tensor]], batch_idx: int):
        self.train()
        return self.shared_step(batch, batch_idx, loss_name="train", training=True)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        loss_name = "val"
        return self.shared_step(batch, batch_idx, dataloader_idx=dataloader_idx, loss_name=loss_name, training=False)

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        loss_name = "test"
        return self.shared_step(batch, batch_idx, dataloader_idx=dataloader_idx, loss_name=loss_name, training=False)

    def shared_step(self,
                    batch: Tuple[Tensor, ...],
                    batch_idx: int,
                    dataloader_idx: int = None,
                    loss_name: str = "",
                    training: bool = True) -> Dict:
        """
        This is the shared step for this 'example' LightningModule. 
        Feel free to customize/change it if you want!
        """
        if dataloader_idx is not None:
            assert isinstance(dataloader_idx, int)
            loss_name += f"/{dataloader_idx}"

        # Split the batch into the observations and the rewards.
        observation, reward = self.split_batch(batch)

        # Get the results of the forward pass:
        # - "observation": the augmented/transformed/processed observation.
        # - "representations": the representations for the observations.
        # - "actions": The actions (predictions)
        forward_pass = self(observation)
        reward = self.preprocess_rewards(reward)
        loss_object: Loss = self.get_loss(forward_pass, reward, loss_name=loss_name)
        return {
            "loss": loss_object.loss,
            "log": loss_object.to_log_dict(),
            "progress_bar": loss_object.to_pbar_message(),
            "loss_object": loss_object,
        }
        # The above can also easily be done like this:
        result = loss_object.to_pl_dict()
        return result
    
    def split_batch(self, batch: Any) -> Tuple[Observation, Reward]:
        """ Splits the batch into the observations and the rewards. """
        return split_batch(
            batch,
            Observation=self.Observation,
            Reward=self.Reward
        )

    def get_loss(self, forward_pass: Dict[str, Batch], reward: Reward = None, loss_name: str = "") -> Loss:
        """Returns a Loss object containing the total loss and metrics. 

        Args:
            x (Tensor): The input examples.
            y (Tensor, optional): The associated labels. Defaults to None.
            name (str, optional): Name to give to the resulting loss object. Defaults to "".

        Returns:
            Loss: An object containing everything needed for logging/progressbar/metrics/etc.
        """
        assert loss_name
        # Create an 'empty' Loss object with the given name, so that we always
        # return a Loss object, even when `y` is None and we can't the loss from
        # the output_head.
        total_loss = Loss(name=loss_name)
        labels = None if reward is None else reward.y
        if labels is not None:
            # Here in this base model, we only use 'y' from the rewards.
            # TODO: change that, so that the output heads can be extended to
            # accept other things than just `y`, if needed (seems fine for now).
            supervised_loss = self.output_head.get_loss(forward_pass, y=labels)
            total_loss += supervised_loss
        return total_loss

    def preprocess_observations(self, observation) -> Observation:
        return observation
    
    def preprocess_rewards(self, reward: Reward) -> Reward:
        return reward
    
    def validation_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
        ) -> Dict[str, Dict[str, Tensor]]:
        return self._shared_epoch_end(outputs, loss_name="val")

    def test_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
        ) -> Dict[str, Dict[str, Tensor]]:
        # assert False, outputs
        return self._shared_epoch_end(outputs, loss_name="test")

    def _shared_epoch_end(
        self,
        outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]],
        loss_name: str="",
    ) -> Dict[str, Dict[str, Tensor]]:
        
        # Sum of the metrics acquired during the epoch.
        # NOTE: This is the 'online' metrics in the case of a training/val epoch
        # and the 'average' & 'online' in the case of a test epoch (as they are
        # the same in that case).

        epoch_loss: Loss = Loss(name=loss_name)

        if not isinstance(outputs[0], list):
            # We used only a single dataloader.
            for output in outputs:
                if isinstance(output, list):
                    # we had multiple test/val dataloaders (i.e. multiple tasks)
                    # We get the loss for each task at each step. The outputs are for each of the dataloaders.
                    for i, task_output in enumerate(output):
                        task_loss = task_output["loss_object"] 
                        epoch_loss += task_loss
                elif isinstance(output, dict):
                    # There was a single dataloader: `output` is the dict returned
                    # by (val/test)_step.
                    loss_info = output["loss_object"]
                    epoch_loss += loss_info
                else:
                    raise RuntimeError(f"Unexpected output: {output}")
        else:
            for i, dataloader_output in enumerate(outputs):
                loss_i: Loss = Loss(name=f"{i}")
                for output in dataloader_output:
                    if isinstance(output, dict) and "loss_object" in output:
                        loss_info = output["loss_object"]
                        loss_i += loss_info
                    else:
                        raise RuntimeError(f"Unexpected output: {output}")
                epoch_loss += loss_i
        # TODO: Log stuff here?
        for name, value in epoch_loss.to_log_dict().items():
            logger.info(f"{name}: {value}")
            if self.logger:
                self.logger.log(name, value)
        return epoch_loss.to_pl_dict()
        
    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters())

    @property
    def batch_size(self) -> int:
        return self.hp.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.hp.batch_size = value 
    
    @property
    def learning_rate(self) -> float:
        return self.hp.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.hp.learning_rate = value

    def on_task_switch(self, task_id: int, training: bool = False) -> None:
        """Called when switching between tasks.
        
        Args:
            task_id (int): the Id of the task.
            training (bool): Wether we are currently training or valid/testing.
        """

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
        model_summary = ModelSummary(self, mode=mode)
        log.debug('\n' + str(model_summary))
        return model_summary
