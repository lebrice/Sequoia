""" Addon that allows training on batches that are partially labeled. """
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor

from common.losses import LossInfo
from simple_parsing import mutable_field
from utils.logging_utils import get_logger

from .addon import ExperimentAddon

logger = get_logger(__file__)

@dataclass
class SemiSupervisedBatchesAddon(ExperimentAddon):
    """ Experiment capable of training on semi-supervised batches. 
    
    NOTE: For now, this addon is always enabled. Might be better placed
    somewhere else perhaps.
    """

    def train_batch(self, data: Tensor,
                          target: Union[Optional[Tensor], List[Optional[Tensor]]],
                          name: str="Train", **kwargs) -> LossInfo: 
        """Trains the model on a batch of (potentially partially labeled) data. 

        Args:
            data (Tensor): Examples
            target (Union[Optional[Tensor], List[Optional[Tensor]]]):
                Labels associated with the data. Can either be:
                - None: fully unlabeled batch
                - Tensor: fully labeled batch
                - List[Optional[Tensor]]: Partially labeled batch.
            name (str, optional): Name of the resulting loss object. Defaults to
                "Train".

        Returns:
            LossInfo: a loss object made from both the unsupervised and
                supervised losses. 
        """
        if target is None or isinstance(target, Tensor):
            # Fully labeled/unlabeled batch
            labeled_ratio = float(target is not None)
            self.log(dict(labeled_ratio=labeled_ratio))
            return super().train_batch(data, target, name=name, **kwargs)

        self.model.train()
        self.model.optimizer.zero_grad()
        
        # Batch is maybe a mix of labeled / unlabeled data.
        labeled_x_list: List[Tensor] = []
        labeled_y_list: List[Tensor] = [] 
        unlabeled_x_list: List[Tensor] = []

        # TODO: Might have to somehow re-order the results based on the indices?
        # TODO: Join (merge) the metrics? or keep them separate?
        labeled_indices: List[int] = []
        unlabeled_indices: List[int] = []

        for i, (x, y) in enumerate(zip(data, target)):
            if y is None:
                unlabeled_indices.append(i)
                unlabeled_x_list.append(x)
            else:
                labeled_indices.append(i)
                labeled_x_list.append(x)         
                labeled_y_list.append(torch.LongTensor([y]))
        
        labeled_ratio = len(labeled_indices) / len(unlabeled_indices + labeled_indices)
        self.log(dict(labeled_ratio=labeled_ratio))

        loss = LossInfo(name=name)
        
        if unlabeled_indices:
            unlabeled_x = torch.stack(unlabeled_x_list) 
            unsupervised_loss = self.model.get_loss(unlabeled_x, None, name=name, **kwargs)
            loss += unsupervised_loss

        if labeled_indices:
            labeled_x = torch.stack(labeled_x_list)
            labeled_y = torch.stack(labeled_y_list)
            supervised_loss = self.model.get_loss(labeled_x, labeled_y, name=name, **kwargs)
            loss += supervised_loss

        total_loss = loss.total_loss
        total_loss.backward()
        self.model.optimizer_step(global_step=self.global_step, **kwargs)

        self.global_step += data.shape[0]
        return loss
