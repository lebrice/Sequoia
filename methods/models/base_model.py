"""Base class for the Model to be used as part of a Method.

This is meant

TODO: There is a bunch of work to be done here.
"""
import dataclasses
import itertools
from abc import ABC
from dataclasses import dataclass
from collections import abc as collections_abc
from typing import (Any, ClassVar, Dict, Generic, List, NamedTuple, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from torch import Tensor

from common.config import Config
from common.loss import Loss
from methods.models.output_heads import OutputHead
from utils.logging_utils import get_logger

from .base_hparams import BaseHParams

logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=LightningDataModule)

from .batch import Batch


@dataclass(frozen=True)
class BaseObservation(Batch):
    """ """
    x: Tensor


@dataclass(frozen=True)
class ClassIncrementalObservation(BaseObservation):
    """ """
    t: Union[Optional[Tensor], Sequence[Optional[Tensor]]] = None


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

    @dataclass
    class HParams(BaseHParams):
        """ HParams of the Model. """

    @dataclass(frozen=True)
    class Observation(Batch):
        __slots__ = ("x",)
        x: Tensor

    @dataclass(frozen=True)
    class Action(Batch):
        # Predictions from the model in a supervised setting, or chosen action
        # in an RL setting.
        __slots__ = ("y_pred",)
        y_pred: Tensor

    @dataclass(frozen=True)
    class Reward(Batch):
        y: Union[Optional[Tensor], Sequence[Optional[Tensor]]] = None

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

        logger.debug(f"setting: {self.setting}")
        logger.debug(f"Input shape: {self.input_shape}")
        logger.debug(f"Output shape: {self.output_shape}")
        logger.debug(f"Reward shape: {self.reward_shape}")
        
        # (Testing) Setting this attribute is supposed to help with ddp/etc
        # training in pytorch-lightning. Not 100% sure.
        self.example_input_array = torch.rand(self.batch_size, *self.input_shape)
        
        # Create an 'encoder' and an 'output head'.
        self.encoder, self.hidden_size = self.hp.make_encoder()
        self.output_head = self.create_output_head()

        if self.config.debug and self.config.verbose:
            logger.debug("Config:")
            logger.debug(self.config.dumps(indent="\t"))
            logger.debug("Hparams:")
            logger.debug(self.hp.dumps(indent="\t"))

    @auto_move_data
    def forward(self, observation: Tuple[Tensor, ...]) -> Dict[str, Tensor]:
        """ Forward pass of the Model. Returns a dict.
        """
        # TODO: Playing with the idea that we could use something like an object
        # or NamedTuple for the 'observation'.
        # TODO: If the default/non-default ordering required by NamedTuple ever
        # becomes a problem, then we could switch out to using dicts. 
        observation = self.Observation(*observation)
        x, *_ = self.preprocess_batch(x)
        h_x = self.encode(x)
        y_pred = self.output_task(x=x, h_x=h_x)

        return dict(
            x=x,
            h_x=h_x,
            y_pred=y_pred,
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encodes a batch of samples `x` into a hidden vector.

        Args:
            x (Tensor): Tensor for a batch of pre-processed samples.

        Returns:
            Tensor: The hidden vector / embedding for that sample, with size
                [<batch_size>, `self.hp.hidden_size`].
        """
        h_x = self.encoder(x)
        if isinstance(h_x, list) and len(h_x) == 1:
            # Some pretrained encoders sometimes give back a list with one tensor. (?)
            h_x = h_x[0]
        return h_x

    def output_task(self, x: Tensor, h_x: Tensor) -> Tensor:
        """ Pass the required inputs to the output head and get predictions.
        
        NOTE: This method is basically just here so we can customize what we
        pass to the output head, or what we take from it, similar to the
        `encode` method.
        """
        if self.hp.detach_output_head:
            h_x = h_x.detach()
        return self.output_head(x, h_x)

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
    
    def split_batch(self, batch: Tuple[Tensor, ...]) -> Tuple[Observation, Reward]:
        """ WIP: IDEA: Split the batch intelligently, depending on the
        number of 'required' fields in `Observation` and in `Reward` classes.
        
        To make this simpler, we're always going to return an Observation and a Reward
        object, even if the batch is unsupervised, however in that case the
        Reward object will have `y=None`.
        """
        assert isinstance(batch, (tuple, list))
        def n_required_fields(named_tuple_type: Type[NamedTuple]):
            # Need to figure out a way to get the number fields through the
            # class itself.
            named_tuple = named_tuple_type._make(itertools.count())
            return len(named_tuple._fields) - len(named_tuple._field_defaults)
        required_for_obs = n_required_fields(self.Observation)
        required_for_reward = n_required_fields(self.Reward)
        # TODO       
        
        observation = self.Observation(x=tensors.x, t=tensors.t)
        reward = Reward(y=tensors.y)
        return observation, reward
            
    def shared_step(self, batch: Tuple[Tensor, ...],
                          batch_idx: int,
                          dataloader_idx: int = None,
                          loss_name: str = "",
                          training: bool = True,
                    ) -> Dict:
        """
        This is the shared step for this 'example' LightningModule. 
        Feel free to customize/change it if you want!
        """
        if dataloader_idx is not None:
            assert isinstance(dataloader_idx, int)
            loss_name += f"/{dataloader_idx}"
        # TODO: This makes sense? What if the batch is unlabeled?
        observation: Observation
        reward: Reward
        observation, reward = self.split_batch(batch)
        forward_pass = self(observation)
        loss_object: Loss = self.get_loss(forward_pass, y=reward.y, loss_name=loss_name)
        return {
            "loss": loss_object.loss,
            "log": loss_object.to_log_dict(),
            "progress_bar": loss_object.to_pbar_message(),
            "loss_object": loss_object,
        }
        # The above can also easily be done like this:
        result = loss_object.to_pl_dict()
        return result

    def get_loss(self, forward_pass: Dict[str, Tensor], y: Tensor = None, loss_name: str="") -> Loss:
        """Returns a Loss object containing the total loss and metrics. 

        Args:
            x (Tensor): The input examples.
            y (Tensor, optional): The associated labels. Defaults to None.
            name (str, optional): Name to give to the resulting loss object. Defaults to "".

        Returns:
            Loss: An object containing everything needed for logging/progressbar/metrics/etc.
        """
        assert loss_name
        x = forward_pass["x"]
        h_x = forward_pass["h_x"]
        y_pred = forward_pass["y_pred"]
        # Create an 'empty' Loss object with the given name, so that we always
        # return a Loss object, even when `y` is None and we can't the loss from
        # the output_head.
        total_loss = Loss(name=loss_name)
        if y is not None:
            supervised_loss = self.output_head.get_loss(forward_pass, y=y)
            total_loss += supervised_loss
        return total_loss

    def preprocess_observation(self, observation) -> Observation:
        return observation
    
    def preprocess_batch(self,
                         *batch: Union[Tensor, Tuple[Tensor, ...]]
                         ) -> Tuple[Tensor, Optional[Tensor]]:
        """Preprocess the input batch before it is used for training.
                
        By default this just splits a (potentially unsupervised) batch into x
        and y's, and any batch which is a tuple of more than 2 items is left
        unchanged.
               
        When tackling a different problem or if additional preprocessing or data
        augmentations are needed, feel free to customize/change this to fit your
        needs.
        
        TODO: Re-add the task labels for each sample so we use the right output
        head at the 'example' level.

        Parameters
        ----------
        - batch : Tensor
        
            a batch of inputs.
        
        Returns
        -------
        Tensor
            The preprocessed inputs.
        Optional[Tensor]
            The processed labels, if there are any.
        """
        assert isinstance(batch, tuple)
        
        if len(batch) == 1:
            batch = batch[0]
        if isinstance(batch, Tensor):
            return batch, None

        if len(batch) == 2:
            return batch[0], batch[1]
        else:
            # Batch has more than 2 items, so we return it as-is..
            return batch
    
    # def validation_epoch_end(
    #         self,
    #         outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    #     ) -> Dict[str, Dict[str, Tensor]]:
    #     return self._shared_epoch_end(outputs, loss_name="val")

    # def test_epoch_end(
    #         self,
    #         outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    #     ) -> Dict[str, Dict[str, Tensor]]:
    #     # assert False, outputs
    #     return self._shared_epoch_end(outputs, loss_name="test")

    # def _shared_epoch_end(
    #     self,
    #     outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]],
    #     loss_name: str="",
    # ) -> Dict[str, Dict[str, Tensor]]:
        
    #     # Sum of the metrics acquired during the epoch.
    #     # NOTE: This is the 'online' metrics in the case of a training/val epoch
    #     # and the 'average' & 'online' in the case of a test epoch (as they are
    #     # the same in that case).

    #     total_loss: Loss = Loss(name=loss_name)
    #     assert len(outputs) == 1
    #     output = outputs[0]

    #     for output in outputs:
    #         if isinstance(output, list):
    #             # we had multiple test/val dataloaders (i.e. multiple tasks)
    #             # We get the loss for each task at each step. The outputs are for each of the dataloaders.
    #             for i, task_output in enumerate(output):
    #                 task_loss = task_output["loss_object"] 
    #                 total_loss += task_loss
    #         elif isinstance(output, dict):
    #             # There was a single dataloader: `output` is the dict returned
    #             # by (val/test)_step.
    #             loss_info = output["loss_object"]
    #             total_loss += loss_info
    #         else:
    #             raise RuntimeError(f"Unexpected output: {output}")

    #     return total_loss.to_pl_dict()

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
