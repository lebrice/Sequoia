""" Addon that allows training on batches that are partially labeled. """
from dataclasses import dataclass
from typing import Optional, Union, List

import torch
from torch import Tensor

from common.losses import LossInfo

from .addon import ExperimentAddon
from utils.logging_utils import get_logger

logger = get_logger(__file__)

@dataclass
class SemiSupervisedBatchesAddon(ExperimentAddon):
    """ Experiment capable of training on semi-supervised batches. """

    def train_batch(self, data: Tensor,
                          target: Union[Optional[Tensor], List[Optional[Tensor]]],
                          name: str="Train") -> LossInfo:
        # Fully labeled / unlabeled batch
        if target is None or isinstance(target, Tensor):
            return super().train_batch(data, target, name=name)

        self.model.train()
        self.model.optimizer.zero_grad()

        batch_loss_info = self.model.get_loss(data, target, name=name)
        
        unsupervised_x_list: List[Tensor] = []
        supervised_x_list: List[Tensor] = []
        supervised_y_list: List[Tensor] = [] 

        # TODO: Might have to somehow re-order the results based on the indices?
        # TODO: Join (merge) the metrics? or keep them separate?
        unlabeled_indices: List[int] = []
        labeled_indices: List[int] = []

        for i, (x, y) in enumerate(zip(data, target)):

            if y is None:
                unlabeled_indices.append(i)
                unsupervised_x_list.append(x)
            else:
                labeled_indices.append(i)
                supervised_x_list.append(x)
                supervised_y_list.append(y)
        # Create a batch of labeled / unlabeled data.
        unsupervised_x = torch.stack(unsupervised_x_list)
        supervised_x = torch.stack(supervised_x_list)
        supervised_y = torch.stack(supervised_y_list)
        
        supervised_ratio = len(labeled_indices) / len(unlabeled_indices + labeled_indices)
        logger.debug(f"supervised ratio: {supervised_ratio:%}")
        
        loss = LossInfo(name=name)
        
        supervised_loss = self.model.get_loss(supervised_x, supervised_y, name=name)
        loss += supervised_loss
        
        unsupervised_loss = self.model.get_loss(unsupervised_x, None, name=name)
        loss += unsupervised_loss
        
        total_loss = loss.total_loss
        total_loss.backward()

        self.model.optimizer_step(global_step=self.global_step)

        self.global_step += data.shape[0]
        return loss
