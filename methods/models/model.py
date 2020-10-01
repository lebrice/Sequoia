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

from .hparams import HParams as HParamsBase

logger = get_logger(__file__)
SettingType = TypeVar("SettingType", bound=LightningDataModule)


class Model(LightningModule, Generic[SettingType]):
    """ Base model LightningModule (nn.Module extended by pytorch-lightning)
    
    This model splits the learning task into a representation-learning problem
    and a downstream task (output head) applied on top of it.   

    The most important method to understand is the `get_loss` method, which
    is used by the [train/val/test]_step methods which are called by
    pytorch-lightning.
    """

    # NOTE: we put this here just so its easier to subclass the HParams from
    # within a subclass of Model.
    @dataclass
    class HParams(HParamsBase):
        """ HParams of the Model. """

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__()
        # super().__init__(setting=setting, hparams=hparams, config=config)
        self.setting: SettingType = setting
        # TODO: Not setting this property, so the trainer doesn't fallback to
        # using it.
        # self.datamodule: LightningDataModule = setting
        self.hp = hparams
        self.config: Config = config

        # self.save_hyperparameters()
        self.input_shape  = self.setting.dims
        self.output_shape = self.setting.action_shape
        self.reward_shape = self.setting.reward_shape

        logger.debug(f"setting: {self.setting}")
        logger.debug(f"Input shape: {self.input_shape}")
        logger.debug(f"Output shape: {self.output_shape}")
        logger.debug(f"Reward shape: {self.reward_shape}")

        # Here we assume that all methods have a form of 'encoder' and 'output head'.
        self.encoder, self.hidden_size = self.hp.make_encoder()
        self.output_head = self.create_output_head()

        if self.config.debug and self.config.verbose:
            logger.debug("Config:")
            logger.debug(self.config.dumps(indent="\t"))
            logger.debug("Hparams:")
            logger.debug(self.hp.dumps(indent="\t"))

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

    def output_task(self, h_x: Tensor) -> Tensor:
        if self.hp.detach_output_head:
            h_x = h_x.detach()
        return self.output_head(h_x)

    def create_output_head(self) -> OutputHead:
        """ Create the output head for the task. """
        return OutputHead(self.hidden_size, self.output_shape, name="classification")

    def training_step(self, batch: Tuple[Tensor, Optional[Tensor]], batch_idx: int):
        self.train()
        return self._shared_step(batch, batch_idx, loss_name="train", training=True)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        loss_name = "val"
        return self._shared_step(batch, batch_idx, dataloader_idx=dataloader_idx, loss_name=loss_name, training=False)

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        loss_name = "test"
        return self._shared_step(batch, batch_idx, dataloader_idx=dataloader_idx, loss_name=loss_name, training=False)
    
    def _shared_step(self, batch: Tuple[Tensor, Optional[Tensor]],
                           batch_idx: int,
                           dataloader_idx: int = None,
                           loss_name: str = "",
                           training: bool = True,
                    ) -> Dict:
        assert loss_name
        
        if dataloader_idx is not None:
            assert isinstance(dataloader_idx, int)
            loss_name += f"/{dataloader_idx}"
        x, y, t = batch
        # x, y = self.preprocess_batch(x, y, t)
        
        forward_pass = self(x, task_labels=t)
        loss: Loss = self.get_loss(forward_pass, y=y, loss_name=loss_name)
        # NOTE: loss is supposed to be a tensor, but I'm testing out giving a Loss object instead.
        result = loss.to_pl_dict()
        result["objective"] = loss.losses[self.output_head.name].metric
        return result

        # return {
        #     "loss": loss.loss,
        #     "log": loss.to_log_dict(),
        #     "progress_bar": loss.to_pbar_message(),
        #     "loss_object": loss,
        #     "objective": loss.losses[self.output_head.name].metric,
        # }

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

    @auto_move_data
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        x, *_ = self.preprocess_batch(x)
        h_x = self.encode(x)
        y_pred = self.output_task(h_x)
        return dict(
            x=x,
            h_x=h_x,
            y_pred=y_pred,
        )

    def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int) -> None:
        """ Customize the backward pass.
        Was thinking of using the Loss object as the loss, but it feels a bit hacky.
        """
        if isinstance(loss, Loss):
            loss.total_loss.backward()
        else:
            super().backward(trainer, loss, optimizer, optimizer_idx)

    def preprocess_batch(self,
                         *batch: Union[Tensor, Tuple[Tensor, ...]]
                         ) -> Tuple[Tensor, Optional[Tensor]]:
        """Preprocess the input batch before it is used for training.
                
        By default this just splits a (potentially unsupervised) batch into x
        and y's. 
        When tackling a different problem or if additional preprocessing or data
        augmentations are needed, you could just subclass Classifier and change
        this method's behaviour.

        NOTE: This also discards the task labels for each example, which are
        normally given by the dataloaders from Continuum.

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
