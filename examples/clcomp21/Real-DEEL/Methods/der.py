from .base_method import BaseMethod


from typing import Dict, Optional, Tuple
from torch import Tensor, nn

from sequoia.settings.passive.cl.objects import (
    Actions,
    Environment,
    Observations,
    Rewards,
)
from dataclasses import dataclass
from Utils import BaseHParams
from sequoia.settings import ClassIncrementalSetting
from Utils import Buffer


class DER(BaseMethod):
    @dataclass
    class HParams(BaseHParams):
        dark_coefficient: float = 0.8
        buffer_size: int = 3000
        replay_minibatch_size: int = 32
        scale_loss: float = 0.01
        darkL1: bool = False

    def __init__(self, hparams: HParams = None) -> None:
        super().__init__(hparams=hparams or self.HParams())

    def _method_specific_configure(self, setting: ClassIncrementalSetting):
        """Method specific initialization used for vars and settings needed per method

        Args:
            setting (ClassIncrementalSetting): Setting used in the configuration
        """  
        self.buffer = Buffer(
            self.hparams.buffer_size,
            loss_priority=self.hparams.priority_reservoir,
            balanced=self.hparams.balanced_reservoir,
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        if self.hparams.darkL1:
            self.dark_criterion = nn.L1Loss()
        else:
            self.dark_criterion = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

    def _loss(self, preds, target, return_individual=False):
        """Computing loss along with individual losses

        Args:
            preds (tensor): model predictions
            target (tensor): true labels
            return_individual (bool, optional): return individual losses. Defaults to False.

        Returns:
            tensor: mean loss or a tuple of mean loss and indiviual losses
        """        
        individual_losses = self.criterion(preds, target)
        loss = individual_losses.mean()
        if return_individual:
            return loss, individual_losses.detach().cpu()
        return loss

    def _replay_der_loss(self):
        """Dark experience replay loss

        Returns:
            loss_dict: dark experience replay loss
        """        
        memory_dict = self.buffer.get_data(self.hparams.replay_minibatch_size)
        memory_inputs = memory_dict["examples"]
        memory_logits = memory_dict["logits"]
        memory_targets = memory_dict["labels"]
        #memory_losses = memory_dict["loss_scores"]

        loss_dict = {}
        new_logits = self.model(memory_inputs)
        dark_loss = self.dark_criterion(new_logits, memory_logits)
        loss_dict["dark"] = self.hparams.dark_coefficient * dark_loss
        return loss_dict

    def shared_step(
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        environment: Environment,
        validation: bool = False,
    ) -> Tuple[Tensor, Dict]:
        """Shared step used for both training and validation.

        Parameters
        ----------
        batch : Tuple[Observations, Optional[Rewards]]
            Batch containing Observations, and optional Rewards. When the Rewards are
            None, it means that we'll need to provide the Environment with actions
            before we can get the Rewards (e.g. image labels) back.

            This happens for example when being applied in a Setting which cares about
            sample efficiency or training performance, for example.

        environment : Environment
            The environment we're currently interacting with. Used to provide the
            rewards when they aren't already part of the batch (as mentioned above).

        validation : bool
            A flag to denote if this shared step is a validation

        Returns
        -------
        Tuple[Tensor, Dict]
            dict of losses name and tensor value, and a dict of metrics to be logged.
        """
        observations: Observations = batch[0]
        rewards: Optional[Rewards] = batch[1]

        # Get the predictions:

        logits = self.model(observations.x)
        y_pred = logits.argmax(-1).detach()

        if rewards is None:
            # If the rewards in the batch is None, it means we're expected to give
            # actions before we can get rewards back from the environment.
            rewards = environment.send(Actions(y_pred))

        assert rewards is not None
        image_labels = rewards.y.to(self.device)

        replay = not self.buffer.is_empty() and not validation
        loss_dict = {}

        if not self.hparams.priority_reservoir:
            loss_dict["CrossEntropyLoss"] = self._loss(logits, image_labels)
            individual_losses = None
        else:
            loss_dict["CrossEntropyLoss"], individual_losses = self._loss(
                logits, image_labels, return_individual=True
            )

        if replay:
            der_loss_dict = self._replay_der_loss()
            loss_dict.update(der_loss_dict)
        metrics_dict = self._create_metric_dict(
            loss_dict, y_pred, image_labels)

        # specify when to add examples to buffer
        if not validation:
            self._add_to_buffer(
                examples=observations.x,
                logits=logits,
                labels=image_labels,
                task_labels=observations.task_labels,
                loss_scores=individual_losses,
            )
        return loss_dict, metrics_dict

