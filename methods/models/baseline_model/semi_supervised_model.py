"""
Addon that enables training on semi-supervised batches.

NOTE: Not used at the moment, but should work just fine.
"""
from typing import Dict, List, Optional, Union, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from common.loss import Loss
from common.batch import Batch
from settings import SettingType, Observations, Actions, Rewards
from utils.logging_utils import get_logger

from .base_model import BaseModel

logger = get_logger(__file__)


class SemiSupervisedModel(BaseModel[SettingType]):
    def get_loss(self,
                 forward_pass: Dict[str, Tensor],
                 reward: Optional[Rewards] = None,
                 loss_name: str="") -> Loss:
        """Trains the model on a batch of (potentially partially labeled) data. 

        Args:
            forward_pass (Dict[str, Tensor]): WIP: The results of the forward
                pass (processed input, predictions, etc.)
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
        
        # TODO: We could also just use '-1' instead as the 'no-label' val: this
        # would make it a bit simpler than having both lists and tensors in the
        # batch, but then we sometimes might have to check if everything in `y`
        # is equal to -1, which sounds a bit dumb.

        y: Union[Optional[Tensor], List[Optional[Tensor]]] = reward.y
        if y is None or isinstance(y, Tensor):
            # Fully labeled/unlabeled batch
            labeled_ratio = float(y is not None)
            return super().get_loss(forward_pass, reward, loss_name=loss_name)

        is_labeled: np.ndarray = np.asarray([y_i is None for y_i in y])

        # Batch is maybe a mix of labeled / unlabeled data.
        labeled_y = y[is_labeled]
        # TODO: Might have to somehow re-order the results based on the indices?
        # TODO: Join (merge) the metrics? or keep them separate?
        labeled_forward_pass = {k: v[is_labeled] for k, v in forward_pass}
        unlabeled_forward_pass = {k: v[~is_labeled] for k, v in forward_pass}
        
        labeled_ratio = len(labeled_y) / len(y)
        logger.debug(f"Labeled ratio: {labeled_ratio}")

        # Create the 'total' loss for the batch, with the required name.
        # We will then create two 'sublosses', one named 'unsupervised' and one
        # named 'supervised', each containing the respective losses and metrics.
        # TODO: Make sure that this doesn't make it harder to get the metrics
        # from the Loss object. If it does, then we could maybe just fuse the
        # labeled and unlabeled losses and metrics, but that might also cause
        # issues.
        loss = Loss(loss_name=loss_name)
        if unlabeled_forward_pass:
            # TODO: Setting a different loss name for the for this is definitely going to cause trouble! 
            unsupervised_loss = super().get_loss(
                unlabeled_forward_pass,
                y=None,
                loss_name="unsupervised",
            )
            loss += unsupervised_loss

        if labeled_forward_pass:
            supervised_loss = self.model.get_loss(
                labeled_forward_pass,
                y=labeled_y,
                loss_name="supervised",
            )
            loss += supervised_loss

        return loss
