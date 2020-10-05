"""Base class for the Model to be used as part of a Method.

This is meant

TODO: There is a bunch of work to be done here.
"""
from dataclasses import dataclass
from typing import *

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from torch import Tensor, nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from common.config import Config
from common.loss import Loss
from common.tasks.auxiliary_task import AuxiliaryTask
from methods.models.output_heads import OutputHead
from simple_parsing import Serializable, choice, mutable_field
from utils.logging_utils import get_logger

logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=LightningDataModule)


# WIP (@lebrice): Playing around with this idea, to try and maybe use the idea
# of creating typed objects for the 'Observation', the 'Action' and the 'Reward'
# for each kind of model.
from .base_model import ForwardPass
from settings import Observations, Actions, Rewards
from .model_addons import SemiSupervisedModel, ClassIncrementalModel, SelfSupervisedModel


class Model(SemiSupervisedModel,
            ClassIncrementalModel,
            SelfSupervisedModel,
            Generic[SettingType]):
    """ Base model LightningModule (nn.Module extended by pytorch-lightning)
    
    This model splits the learning task into a representation-learning problem
    and a downstream task (output head) applied on top of it.   

    The most important method to understand is the `get_loss` method, which
    is used by the [train/val/test]_step methods which are called by
    pytorch-lightning.
    """

    @dataclass
    class HParams(
        # SemiSupervisedModel.HParams,
        SelfSupervisedModel.HParams,
        ClassIncrementalModel.HParams,
    ):
        """ HParams of the Model. """


    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)

        logger.debug(f"setting of type {type(self.setting)}")
        logger.debug(f"Input shape: {self.input_shape}")
        logger.debug(f"Output shape: {self.output_shape}")
        logger.debug(f"Reward shape: {self.reward_shape}")
        
        if self.config.debug and self.config.verbose:
            logger.debug("Config:")
            logger.debug(self.config.dumps(indent="\t"))
            logger.debug("Hparams:")
            logger.debug(self.hp.dumps(indent="\t"))

    # @auto_move_data
    # def forward(self, observation: "Model.Observation") -> Dict[str, Tensor]:
    #     """ Forward pass of the Model. Returns a dict.
    #     """
    #     # TODO: Playing with the idea that we could use something like an object
    #     # or NamedTuple for the 'observation'. What would we return then? A
    #     # Prediction object? or a dict with all the forward pass tensors?
    #     x, *_ = self.preprocess_batch(observation.x)
    #     h_x = self.encode(x)
    #     y_pred = self.output_task(x=x, h_x=h_x)

    #     return dict(
    #         x=x,
    #         h_x=h_x,
    #         y_pred=y_pred,
    #     )

    # def training_step(self, batch: Tuple[Tensor, Optional[Tensor]], batch_idx: int):
    #     self.train()
    #     return self.shared_step(batch, batch_idx, loss_name="train", training=True)

    # def validation_step(self, batch, batch_idx: int, dataloader_idx: int = None):
    #     loss_name = "val"
    #     return self.shared_step(batch, batch_idx, dataloader_idx=dataloader_idx, loss_name=loss_name, training=False)

    # def test_step(self, batch, batch_idx: int, dataloader_idx: int = None):
    #     loss_name = "test"
    #     return self.shared_step(batch, batch_idx, dataloader_idx=dataloader_idx, loss_name=loss_name, training=False)
            
    # def shared_step(self, batch: Tuple[Tensor, ...],
    #                       batch_idx: int,
    #                       dataloader_idx: int = None,
    #                       loss_name: str = "",
    #                       training: bool = True,
    #                 ) -> Dict:
    #     """
    #     This is the shared step for this 'example' LightningModule. 
    #     Feel free to customize/change it if you want!
    #     """
    #     if dataloader_idx is not None:
    #         assert isinstance(dataloader_idx, int)
    #         loss_name += f"/{dataloader_idx}"
    #     # TODO: This makes sense? What if the batch is unlabeled?
    #     observation, reward = self.split_batch(batch)
    #     forward_pass = self(observation)
    #     loss_object: Loss = self.get_loss(forward_pass, y=reward.y, loss_name=loss_name)
    #     return {
    #         "loss": loss_object.loss,
    #         "log": loss_object.to_log_dict(),
    #         "progress_bar": loss_object.to_pbar_message(),
    #         "loss_object": loss_object,
    #     }
    #     # The above can also easily be done like this:
    #     result = loss_object.to_pl_dict()
    #     return result

    # def get_loss(self, forward_pass: Dict[str, Tensor], y: Tensor = None, loss_name: str="") -> Loss:
    #     """Returns a Loss object containing the total loss and metrics. 

    #     Args:
    #         x (Tensor): The input examples.
    #         y (Tensor, optional): The associated labels. Defaults to None.
    #         name (str, optional): Name to give to the resulting loss object. Defaults to "".

    #     Returns:
    #         Loss: An object containing everything needed for logging/progressbar/metrics/etc.
    #     """
    #     assert loss_name
    #     x = forward_pass["x"]
    #     h_x = forward_pass["h_x"]
    #     y_pred = forward_pass["y_pred"]
    #     # Create an 'empty' Loss object with the given name, so that we always
    #     # return a Loss object, even when `y` is None and we can't the loss from
    #     # the output_head.
    #     total_loss = Loss(name=loss_name)
    #     if y is not None:
    #         supervised_loss = self.output_head.get_loss(forward_pass, y=y)
    #         total_loss += supervised_loss
    #     return total_loss


    # def preprocess_batch(self,
    #                      *batch: Union[Tensor, Tuple[Tensor, ...]]
    #                      ) -> Tuple[Tensor, Optional[Tensor]]:
    #     """Preprocess the input batch before it is used for training.
                
    #     By default this just splits a (potentially unsupervised) batch into x
    #     and y's, and any batch which is a tuple of more than 2 items is left
    #     unchanged.
               
    #     When tackling a different problem or if additional preprocessing or data
    #     augmentations are needed, feel free to customize/change this to fit your
    #     needs.
        
    #     TODO: Re-add the task labels for each sample so we use the right output
    #     head at the 'example' level.

    #     Parameters
    #     ----------
    #     - batch : Tensor
        
    #         a batch of inputs.
        
    #     Returns
    #     -------
    #     Tensor
    #         The preprocessed inputs.
    #     Optional[Tensor]
    #         The processed labels, if there are any.
    #     """
    #     assert isinstance(batch, tuple)
        
    #     if len(batch) == 1:
    #         batch = batch[0]
    #     if isinstance(batch, Tensor):
    #         return batch, None

    #     if len(batch) == 2:
    #         return batch[0], batch[1]
    #     else:
    #         # Batch has more than 2 items, so we return it as-is..
    #         return batch
    
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

    # def configure_optimizers(self):
    #     return self.hp.make_optimizer(self.parameters())

    # @property
    # def batch_size(self) -> int:
    #     return self.hp.batch_size

    # @batch_size.setter
    # def batch_size(self, value: int) -> None:
    #     self.hp.batch_size = value 
    
    # @property
    # def learning_rate(self) -> float:
    #     return self.hp.learning_rate
    
    # @learning_rate.setter
    # def learning_rate(self, value: float) -> None:
    #     self.hp.learning_rate = value

    # def on_task_switch(self, task_id: int, training: bool = False) -> None:
    #     """Called when switching between tasks.
        
    #     Args:
    #         task_id (int): the Id of the task.
    #         training (bool): Wether we are currently training or valid/testing.
    #     """

    # def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
    #     model_summary = ModelSummary(self, mode=mode)
    #     log.debug('\n' + str(model_summary))
    #     return model_summary
