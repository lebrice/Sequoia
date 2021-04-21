################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta, Andrea Cossu                                       #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from typing import List, Optional, Sequence, Union
from sequoia.settings.passive import TaskIncrementalSetting
from avalanche.training import default_logger
from avalanche.training.plugins import (
    AGEMPlugin,
    CWRStarPlugin,
    EvaluationPlugin,
    EWCPlugin,
    GDumbPlugin,
    GEMPlugin,
    LwFPlugin,
    ReplayPlugin,
    StrategyPlugin,
    SynapticIntelligencePlugin,
)
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.training.strategies import Naive as Naive_
from torch.nn import Module
from torch.optim import Optimizer
from .base import AvalancheMethod
from .base_strategy import BaseStrategy
from avalanche.models import SimpleMLP


class Naive(Naive_, BaseStrategy):
    """
    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[StrategyPlugin]] = None,
        evaluator: EvaluationPlugin = default_logger,
        eval_every=-1,
    ):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop.
                if -1: no evaluation during training.
                if  0: calls `eval` after the final epoch of each training
                    experience.
                if >0: calls `eval` every `eval_every` epochs and at the end
                    of all the epochs for a single experience.
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )


class NaiveMethod(AvalancheMethod):
    def create_model(self, setting: TaskIncrementalSetting) -> SimpleMLP:
        return super().create_model(setting)

    def create_cl_strategy(self, setting: TaskIncrementalSetting) -> BaseStrategy:
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = CrossEntropyLoss()
        strategy = Naive(
            self.model,
            self.optimizer,
            self.criterion,
            train_mb_size=64,
            train_epochs=1,
            eval_mb_size=1,
            device=self.device,
            eval_every=0,
        )
        strategy.setting = setting
        return strategy


if __name__ == "__main__":
    from sequoia.settings.passive import TaskIncrementalSetting
    from .base import AvalancheMethod

    setting = TaskIncrementalSetting(dataset="mnist", nb_tasks=5, monitor_training_performance=True)
    method = NaiveMethod()
    results = setting.apply(method)

