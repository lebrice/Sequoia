"""
Addon that enables training on semi-supervised batches.

NOTE: Not used at the moment, but should work just fine.
"""
from typing import List, Optional, Union

import torch
from torch import Tensor

from common.loss import Loss
from utils.logging_utils import get_logger

from .model import Model

logger = get_logger(__file__)


class SemiSupervisedModel(Model):
    def get_loss(self,
                 x: Tensor,
                 y: Union[Optional[Tensor], List[Optional[Tensor]]]=None,
                 loss_name: str="") -> Loss:
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
            Loss: a loss object made from both the unsupervised and
                supervised losses. 
        """
        if y is None or isinstance(y, Tensor):
            # Fully labeled/unlabeled batch
            labeled_ratio = float(y is not None)
            logger.debug(f"Labeled ratio: {labeled_ratio}")
            return super().get_loss(x, y, loss_name=loss_name)

        # Batch is maybe a mix of labeled / unlabeled data.
        labeled_x_list: List[Tensor] = []
        labeled_y_list: List[Tensor] = []
        unlabeled_x_list: List[Tensor] = []

        # TODO: Might have to somehow re-order the results based on the indices?
        # TODO: Join (merge) the metrics? or keep them separate?
        labeled_indices: List[int] = []
        unlabeled_indices: List[int] = []

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            if y is None:
                unlabeled_indices.append(i)
                unlabeled_x_list.append(x_i)
            else:
                labeled_indices.append(i)
                labeled_x_list.append(x_i)
                labeled_y_list.append(y_i)
        
        labeled_ratio = len(labeled_indices) / len(unlabeled_indices + labeled_indices)
        logger.debug(f"Labeled ratio: {labeled_ratio}")

        # Create the 'total' loss for the batch, with the required name.
        # We will then create two 'sublosses', one named 'unsupervised' and one
        # named 'supervised', each containing the respective losses and metrics.
        # TODO: Make sure that this doesn't make it harder to get the metrics
        # from the Loss object. If it does, then we could maybe just fuse the
        # labeled and unlabeled losses and metrics, but that might also cause
        # issues.
        loss = Loss(loss_name=loss_name)
        if unlabeled_indices:
            unlabeled_x = torch.stack(unlabeled_x_list)
            unsupervised_loss = super().get_loss(
                x=unlabeled_x,
                y=None,
                loss_name="unsupervised"
            )
            loss += unsupervised_loss

        if labeled_indices:
            labeled_x = torch.stack(labeled_x_list)
            labeled_y = torch.stack(labeled_y_list)
            supervised_loss = self.model.get_loss(
                x=labeled_x,
                y=labeled_y,
                loss_name="supervised"
            )
            loss += supervised_loss

        return loss
