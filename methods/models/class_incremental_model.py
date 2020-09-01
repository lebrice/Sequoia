from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from common.config import Config
from methods.models.model import Model, OutputHead
from settings import ClassIncrementalSetting
from simple_parsing import mutable_field
from utils.logging_utils import get_logger

from .self_supervised_model import SelfSupervisedModel
from .semi_supervised_model import SemiSupervisedModel

logger = get_logger(__file__)


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
        self._output_head: OutputHead = None

        super().__init__(setting=setting, hparams=hparams, config=config)
        self.hp: ClassIncrementalModel.HParams
        self.setting: SettingType

        # TODO: Add an optional task inference mechanism for ClassIncremental
        # methods!
        self.task_inference_module: nn.Module = None

        self.output_heads: Dict[str, OutputHead] = nn.ModuleDict()
        if self.hp.multihead:
            # TODO: Actually implement something that uses this setting property
            # (task_label_is_readable), as it is not used anywhere atm really.
            # Maybe when we implement something like task-free CL?
            assert self.setting.task_label_is_readable, (
                "Using a multihead model in a setting where the task label "
                "can't be read?"
            )
            output_head = self.create_output_head()
            self.output_head = output_head
            self.output_heads[str(self.setting.current_task_id)] = output_head


    @property
    def output_head(self) -> OutputHead:
        """ Get the output head for the current task.

        FIXME: It's generally bad practice to do heavy computation on a property
        so we should probably add something like a get_output_head(task) method.
        """
        if self.hp.multihead:
            if ((self.training and self.setting.task_labels_at_train_time) or
                (not self.training and self.setting.task_labels_at_test_time)):
                current_task_id = self.setting.current_task_id

            elif self.task_inference_module is not None:
                # current_task_id = self.task_inference_module(...)
                raise NotImplementedError("TODO")
            elif self._output_head is not None:
                return self._output_head
            else:
                raise RuntimeError("No way of determining the task id and output head is None!")

            key = str(current_task_id)
            if key not in self.output_heads:
                # Create the output head, since it's not already in there.
                output_head = self.create_output_head()
                self.output_heads[key] = output_head
            else:
                output_head = self.output_heads[key]
            self._output_head = output_head
            # Return the output head for the current task.
            return output_head
        if self._output_head is None:
            self._output_head = self.create_output_head()
        return self._output_head

    @output_head.setter
    def output_head(self, value: OutputHead) -> None:
        # logger.debug(f"Setting output head to {value}")
        self._output_head = value

    def create_output_head(self) -> OutputHead:
        """Creates a new output head for the current task.
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
        return self.setting.current_task_classes(self.training)


    def preprocess_batch(self, *batch) -> Tuple[Tensor, Optional[Tensor]]:
        x, y = super().preprocess_batch(*batch)

        if y is not None and self.hp.multihead:
            logger.debug(f"Classes in current task: {self.current_task_classes}")
            logger.debug(f"Current task: {self.current_task_classes}")

            # y_unique are the (sorted) unique values found within the batch.
            # idx[i] holds the index of the value at y[i] in y_unique, s.t. for
            # all i in range(0, len(y)) --> y[i] == y_unique[idx[i]]
            y_unique, idx = y.unique(sorted=True, return_inverse=True)


            output_size = self.setting.num_classes_in_current_task
            
            if set(y_unique.tolist()) <= set(range(output_size)):
                # The images were already re-labeled by the continuum package.
                return x, y
            # TODO: This might not make sense anymore, given that I think the
            # Continuum package already re-labels these for us. We could however
            # just figure out which output head to use, if we stored the
            # 'classes' that were assigned to each output head during training.
            # (This might be similar to the "labels trick" from
            # https://arxiv.org/abs/1803.10123)
            elif not (set(y_unique.tolist()) <= set(self.current_task_classes)):
                import matplotlib.pyplot as plt
                plt.imshow(x[0].cpu().numpy().transpose([1, 2, 0]))
                plt.title(str(y[0]))
                plt.show()

                raise RuntimeError(
                    f"There are labels in the batch that aren't part of the "
                    f"current task! (current task: "
                    f"{self.setting.current_task_id}), Current task classes: "
                    f"{self.current_task_classes}, batch labels: {y_unique})"
                )
            else:
                y = self.relabel(y)
        return x, y

    def relabel(self, y: Tensor):
        # Re-label the given batch so the losses/metrics work correctly.
        # Example: if the current task classes is [2, 3] then relabel that
        # those examples as [0, 1].
        # TODO: Double-check that that this is what is usually done in CL.
        new_y = torch.empty_like(y)
        for i, label in enumerate(self.current_task_classes):
            new_y[y == label] = i
        return new_y


    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        if self.hp.multihead:
            # TODO: Figure out exactly where/when/how pytorch-lightning is
            # trying to load the model from, because there are some keys
            # missing (['output_heads.1.output.weight', 'output_heads.1.output.bias'])
            # For now, we're just gonna pretend it's not a problem, I guess?
            strict = False

        missing_keys, unexpected_keys = super().load_state_dict(state_dict=state_dict, strict=False)
        
        if self.hp.multihead and unexpected_keys:
            for i in range(self.setting.nb_tasks):
                output_head_i_keys = filter(
                    lambda s: s.startswith(f"output_heads.{i}"),
                    unexpected_keys,
                )
                new_output_head = self.create_output_head()
                new_output_head.load_state_dict(
                    {k: state_dict[k] for k in unexpected_keys},
                    strict=True,
                )
                key = str(i)
                self.output_heads[key] = new_output_head

        if missing_keys or unexpected_keys:
            logger.debug(f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
        
        return missing_keys, unexpected_keys
