
from dataclasses import dataclass
from typing import *

from torch import Tensor

from common.losses import LossInfo
from utils.logging_utils import get_logger

from .classifier import Classifier

logger = get_logger(__file__)

import torch

class SemiSupervisedClassifier(Classifier):
    def get_loss(self, x: Tensor, y: Union[Optional[Tensor], List[Optional[Tensor]]]=None, name: str="") -> LossInfo:
        """Trains the model on a batch of (potentially partially labeled) data. 

        Args:
            batch: Tuple[data, target]:
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
        if y is None or isinstance(y, Tensor):
            # Fully labeled/unlabeled batch
            labeled_ratio = float(y is not None)
            logger.debug(f"Labeled ratio: {labeled_ratio}")
            return super().get_loss(x, y, name=name)

        # Batch is maybe a mix of labeled / unlabeled data.
        labeled_x_list: List[Tensor] = []
        labeled_y_list: List[Tensor] = []
        unlabeled_x_list: List[Tensor] = []

        # TODO: Might have to somehow re-order the results based on the indices?
        # TODO: Join (merge) the metrics? or keep them separate?
        labeled_indices: List[int] = []
        unlabeled_indices: List[int] = []

        for i, (x, y) in enumerate(zip(x, y)):
            if y is None:
                unlabeled_indices.append(i)
                unlabeled_x_list.append(x)
            else:
                labeled_indices.append(i)
                labeled_x_list.append(x)
                labeled_y_list.append(y)
        
        labeled_ratio = len(labeled_indices) / len(unlabeled_indices + labeled_indices)
        logger.debug(f"Labeled ratio: {labeled_ratio}")

        loss = LossInfo(name=name)
        
        if unlabeled_indices:
            unlabeled_x = torch.stack(unlabeled_x_list)
            unsupervised_loss = super().get_loss(unlabeled_x, None, name="unsupervised")
            loss += unsupervised_loss

        if labeled_indices:
            labeled_x = torch.stack(labeled_x_list)
            labeled_y = torch.stack(labeled_y_list)
            supervised_loss = self.model.get_loss(labeled_x, labeled_y, name="supervised")
            loss += supervised_loss
        
        return loss
