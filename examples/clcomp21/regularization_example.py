""" Example: Defines a new Method based on the ExampleMethod, adding an EWC-like loss to
help prevent the weights from changing too much between tasks.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple, Type

import gym
import torch
from torch import Tensor

from sequoia.common.hparams import uniform
from sequoia.settings import DomainIncrementalSLSetting
from sequoia.settings.sl.incremental.objects import Observations, Rewards
from sequoia.utils.utils import dict_intersection
from sequoia.utils.logging_utils import get_logger

from .multihead_classifier import ExampleTaskInferenceMethod, MultiHeadClassifier

logger = get_logger(__name__)


class RegularizedClassifier(MultiHeadClassifier):
    """Adds an ewc-like penalty to the base classifier, to prevent its weights from
    shifting too much during training.
    """

    @dataclass
    class HParams(MultiHeadClassifier.HParams):
        """Hyperparameters of this improved method.

        Adds the hyper-parameters related the 'ewc-like' regularization to those of the
        ExampleMethod.

        NOTE: These `uniform()` and `log_uniform` and `HyperParameters` are just there
        to make it easier to run HPO sweeps for your Method, which isn't required for
        the competition.
        """

        # Coefficient of the ewc-like loss.
        reg_coefficient: float = uniform(0.0, 10.0, default=1.0)
        # Distance norm used in the regularization loss.
        reg_p_norm: int = 2

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        reward_space: gym.Space,
        hparams: "RegularizedClassifier.HParams" = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            reward_space,
            hparams=hparams,
        )
        self.reg_coefficient = self.hparams.reg_coefficient
        self.reg_p_norm = self.hparams.reg_p_norm

        self.previous_model_weights: Dict[str, Tensor] = {}

        self._previous_task: Optional[int] = None
        self._n_switches: int = 0

    def shared_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        base_loss, metrics = super().shared_step(batch, *args, **kwargs)
        ewc_loss = self.reg_coefficient * self.ewc_loss()
        metrics["ewc_loss"] = ewc_loss
        return base_loss + ewc_loss, metrics

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Executed when the task switches (to either a known or unknown task)."""
        super().on_task_switch(task_id)
        if self._previous_task is None and self._n_switches == 0:
            logger.debug("Starting the first task, no EWC update.")
        elif task_id is None or task_id != self._previous_task:
            # NOTE: We also switch between unknown tasks.
            logger.info(
                f"Switching tasks: {self._previous_task} -> {task_id}: "
                f"Updating the EWC 'anchor' weights."
            )
            self._previous_task = task_id
            self.previous_model_weights.clear()
            self.previous_model_weights.update(
                deepcopy({k: v.detach() for k, v in self.named_parameters()})
            )
        self._n_switches += 1

    def ewc_loss(self) -> Tensor:
        """Gets an 'ewc-like' regularization loss.

        NOTE: This is a simplified version of EWC where the loss is the P-norm
        between the current weights and the weights as they were on the begining
        of the task.
        """
        if self._previous_task is None:
            # We're in the first task: do nothing.
            return 0.0

        old_weights: Dict[str, Tensor] = self.previous_model_weights
        new_weights: Dict[str, Tensor] = dict(self.named_parameters())

        loss = 0.0
        for weight_name, (new_w, old_w) in dict_intersection(new_weights, old_weights):
            loss += torch.dist(new_w, old_w.type_as(new_w), p=self.reg_p_norm)
        return loss


class ExampleRegMethod(ExampleTaskInferenceMethod):
    """Improved version of the ExampleMethod that uses a `RegularizedClassifier`."""

    HParams: ClassVar[Type[HParams]] = RegularizedClassifier.HParams

    def __init__(self, hparams: HParams = None):
        super().__init__(hparams=hparams or self.HParams.from_args())

    def configure(self, setting: DomainIncrementalSLSetting):
        # Use the improved model, with the added EWC-like term.
        self.model = RegularizedClassifier(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
            hparams=self.hparams,
        )
        self.optimizer = self.model.configure_optimizers()

    def on_task_switch(self, task_id: Optional[int]):
        self.model.on_task_switch(task_id)


if __name__ == "__main__":
    # Create the Method:
    # - Manually:
    # method = ExampleRegMethod()
    # - From the command-line:
    from simple_parsing import ArgumentParser

    from sequoia.common import Config
    from sequoia.settings import ClassIncrementalSetting

    parser = ArgumentParser()
    ExampleRegMethod.add_argparse_args(parser)
    args = parser.parse_args()
    method = ExampleRegMethod.from_argparse_args(args)

    # Create the Setting:

    # - "Easy": Domain-Incremental MNIST Setting, useful for quick debugging, but
    #           beware that the action space is different than in class-incremental!
    #           (which is the type of Setting used in the SL track!)
    # from sequoia.settings.sl.class_incremental.domain_incremental import DomainIncrementalSLSetting
    # setting = DomainIncrementalSLSetting(
    #     dataset="mnist", nb_tasks=5, monitor_training_performance=True
    # )

    # - "Medium": Class-Incremental MNIST Setting, useful for quick debugging:
    # setting = ClassIncrementalSetting(
    #     dataset="mnist",
    #     nb_tasks=5,
    #     monitor_training_performance=True,
    #     known_task_boundaries_at_test_time=False,
    #     batch_size=32,
    #     num_workes=4,
    # )

    # - "HARD": Class-Incremental Synbols, more challenging.
    # NOTE: This Setting is very similar to the one used for the SL track of the
    # competition.
    setting = ClassIncrementalSetting(
        dataset="synbols",
        nb_tasks=12,
        known_task_boundaries_at_test_time=False,
        monitor_training_performance=True,
        batch_size=32,
        num_workers=4,
    )

    # Run the experiment:
    results = setting.apply(method, config=Config(debug=True, data_dir="./data"))
    print(results.summary())
