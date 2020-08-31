from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from common.config import Config
from methods.models.model import Model, OutputHead
from settings import ClassIncrementalSetting
from simple_parsing import mutable_field

from .self_supervised_model import SelfSupervisedModel
from .semi_supervised_model import SemiSupervisedModel

SettingType = TypeVar("SettingType", bound=ClassIncrementalSetting)

class ClassIncrementalModel(SelfSupervisedModel[SettingType], SemiSupervisedModel):
    """ Extension of the Classifier LightningModule aimed at CL settings.
    TODO: Add the stuff related to multihead/continual learning here?
    """

    @dataclass
    class HParams(SelfSupervisedModel.HParams):
        """ Hyperparameters specific to a Continual Learning classifier.
        TODO: Add any hyperparameters specific to CL here.
        """
        # Wether to create one output head per task.
        # TODO: Does it make no sense to have multihead=True when the model doesn't
        # have access to task labels. Need to figure out how to manage this between TaskIncremental and Classifier.
        multihead: bool = False

    def __init__(self, setting: ClassIncrementalSetting, hparams: HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.hp: ClassIncrementalModel.HParams
        self.setting: SettingType

        if self.hp.multihead:
            # TODO: Actually implement something that uses this setting property
            # (task_label_is_readable), as it is not used anywhere atm really.
            # Maybe when we implement something like task-free CL? 
            assert self.setting.task_label_is_readable, (
                "Using a multihead model in a setting where the task label "
                "can't be read?"
            )
            self.output_heads: Dict[str, OutputHead] = nn.ModuleDict()
            self.output_heads[str(self.setting.current_task_id)] = self.create_output_head()

    def create_output_head(self) -> OutputHead:
        """Creates a new output head for the current task.

        Returns:
            [type]: [description]
        """
        output_size = self.setting.action_dims
        if self.hp.multihead:
            output_size = self.setting.num_classes_in_current_task 
        return self.output_head_class(
            input_size=self.hidden_size,
            output_size=output_size,
        )

    @property
    def output_head_class(self) -> Type[OutputHead]:
        """Property which returns the type of output head to use.

        overwrite this if your model does something different than classification.

        Returns:
            Type[OutputHead]: A subclass of OutputHead.
        """
        return OutputHead

    def output_task(self, h_x: Tensor) -> Tensor:
        if self.hp.detach_output_head:
            h_x = h_x.detach()
        if self.hp.multihead:
            output_head = self.output_heads[str(self.setting.current_task_id)]
            return output_head(h_x)
        return super().output_task(h_x)

    def train_dataloaders(self, **kwargs) -> List[DataLoader]:
        """ Returns the dataloaders for all train tasks.
        See the `TaskIncrementalSetting` class for more info.
        """
        kwargs = self.dataloader_kwargs(**kwargs)
        return self.setting.train_dataloaders(**kwargs)
    
    def val_dataloaders(self, **kwargs) -> List[DataLoader]:
        """ Returns the dataloaders for all validation tasks.
        See the `TaskIncrementalSetting` class for more info.
        """
        kwargs = self.dataloader_kwargs(**kwargs)
        return self.setting.val_dataloaders(**kwargs)
    
    def test_dataloaders(self, **kwargs) -> List[DataLoader]:
        """ Returns the dataloaders for all test tasks.
        See the `TaskIncrementalSetting` class for more info.
        """
        kwargs = self.dataloader_kwargs(**kwargs)
        return self.setting.test_dataloaders(**kwargs)
    
    def _shared_step(self, batch: Tuple[Tensor, Optional[Tensor]],
                           batch_idx: int,
                           dataloader_idx: int = None,
                           loss_name: str = "",
                           training: bool = True,
                    ) -> Dict:
        assert loss_name
        if dataloader_idx is not None:
            self.on_task_switch(dataloader_idx, training=training)
        return super()._shared_step(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            loss_name=loss_name,
        )

    def on_task_switch(self, task_id: int, training: bool=False) -> None:
        """Called when switching between tasks.
        
        Args:
            task_id (int): the Id of the task.
            training (bool): Wether we are currently training or valid/testing.
        """
        super().on_task_switch(task_id=task_id, training=training)
        self.setting.current_task_id = task_id
        if self.hp.multihead and str(task_id) not in self.output_heads:
            self.output_heads[str(task_id)] = self.create_output_head()

    @property
    def current_task_classes(self) -> List[int]:
        # TODO: detect wether we are training or testing.
        train = True
        return self.setting.current_task_classes(train)

    def preprocess_batch(self, *batch) -> Tuple[Tensor, Optional[Tensor]]:
        x, y = super().preprocess_batch(*batch)
        
        if y is not None and self.hp.multihead:
            print(self.setting.current_task_classes)
            # y_unique are the (sorted) unique values found within the batch.
            # idx[i] holds the index of the value at y[i] in y_unique, s.t. for
            # all i in range(0, len(y)) --> y[i] == y_unique[idx[i]]
            y_unique, idx = y.unique(sorted=True, return_inverse=True)
            # TODO: Could maybe decide which output head to use depending on the labels
            # (perhaps like the "labels trick" from https://arxiv.org/abs/1803.10123)
            if not (set(y_unique.tolist()) <= set(self.current_task_classes)):
                raise RuntimeError(
                    f"There are labels in the batch that aren't part of the "
                    f"current task! (current task: {self.setting.current_task_id}) \n"
                    f"(Current task classes: {self.current_task_classes}, "
                    f"batch labels: {y_unique})"
                )

            # Re-label the given batch so the losses/metrics work correctly.
            # Example: if the current task classes is [2, 3] then relabel that
            # those examples as [0, 1].
            # TODO: Double-check that that this is what is usually done in CL.
            new_y = torch.empty_like(y)
            for i, label in enumerate(self.current_task_classes):
                new_y[y == label] = i
            y = new_y
        return x, y
