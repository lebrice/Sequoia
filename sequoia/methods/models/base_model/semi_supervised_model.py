"""
Addon that enables training on semi-supervised batches.

NOTE: Not used at the moment, but should work just fine.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np
from torch import Tensor

# from sequoia.common.callbacks import KnnCallback
from sequoia.common.loss import Loss
from sequoia.settings import Rewards, SettingType
from sequoia.utils.logging_utils import get_logger

from .model import Model

logger = get_logger(__file__)


class SemiSupervisedModel(Model[SettingType]):
    @dataclass
    class HParams(Model.HParams):
        """Hyperparameters of a Self-Supervised method."""

        # Adds Options for a KNN classifier callback, which is used to evaluate
        # the quality of the representations on each task after each training
        # epoch.
        # TODO: Debug/test this callback to make sure it still works fine.
        # knn_callback: KnnCallback = mutable_field(KnnCallback)

    def get_loss(
        self,
        forward_pass: Dict[str, Tensor],
        rewards: Optional[Rewards] = None,
        loss_name: str = "",
    ) -> Loss:
        """Trains the model on a batch of (potentially partially labeled) data.

        Args:
            forward_pass (Dict[str, Tensor]): WIP: The results of the forward
                pass (processed input, predictions, etc.)
            rewards (Union[Optional[Tensor], List[Optional[Tensor]]]):
                Labels associated with the data. Can either be:
                - None: fully unlabeled batch
                - Tensor: fully labeled batch
                - List[Optional[Tensor]]: Partially labeled batch.
            loss_name (str, optional): Name of the resulting loss object. Defaults to
                "Train".

        Returns:
            Loss: a loss object made from both the unsupervised and
                supervised losses.
        """

        # TODO: We could also just use '-1' instead as the 'no-label' val: this
        # would make it a bit simpler than having both numpy arrays and tensors
        # in the batch

        y: Union[Optional[Tensor], Sequence[Optional[Tensor]]] = rewards.y
        if y is None or all(y_i is not None for y_i in y):
            # Fully labeled/unlabeled batch
            # NOTE: Tensors can't have None items, so if we get a Tensor that
            # means that we have all task labels.
            labeled_ratio = float(y is not None)
            return super().get_loss(forward_pass, rewards, loss_name=loss_name)

        is_labeled: np.ndarray = np.asarray([y_i is not None for y_i in y])

        # Batch is maybe a mix of labeled / unlabeled data.
        labeled_y = y[is_labeled]
        # TODO: Might have to somehow re-order the results based on the indices?
        # TODO: Join (merge) the metrics? or keep them separate?
        labeled_forward_pass = {k: v[is_labeled] for k, v in forward_pass.items()}
        unlabeled_forward_pass = {k: v[~is_labeled] for k, v in forward_pass.items()}

        labeled_ratio = len(labeled_y) / len(y)
        logger.debug(f"Labeled ratio: {labeled_ratio}")

        # Create the 'total' loss for the batch, with the required name.
        # We will then create two 'sublosses', one named 'unsupervised' and one
        # named 'supervised', each containing the respective losses and metrics.
        # TODO: Make sure that this doesn't make it harder to get the metrics
        # from the Loss object. If it does, then we could maybe just fuse the
        # labeled and unlabeled losses and metrics, but that might also cause
        # issues.
        loss = Loss(name=loss_name)
        if unlabeled_forward_pass:
            # TODO: Setting a different loss name for the for this is definitely going to cause trouble!
            unsupervised_loss = super().get_loss(
                unlabeled_forward_pass,
                rewards=None,
                loss_name="unsupervised",
            )
            loss += unsupervised_loss

        if labeled_forward_pass:
            supervised_loss = super().get_loss(
                labeled_forward_pass,
                rewards=labeled_y,
                loss_name="supervised",
            )
            loss += supervised_loss

        return loss
