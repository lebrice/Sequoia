from dataclasses import dataclass
from typing import List, Optional

import gym
import torch
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)
import tqdm
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import (
    AGEMPlugin,
    CWRStarPlugin,
    EvaluationPlugin,
    GDumbPlugin,
    GEMPlugin,
    LwFPlugin,
    ReplayPlugin,
    StrategyPlugin,
    SynapticIntelligencePlugin,
)
from avalanche.training.plugins.ewc import EWCPlugin as EWCPlugin_
from avalanche.training.plugins.ewc import avalanche_forward, copy_params_dict
from avalanche.training.strategies import EWC as EWC_
from avalanche.training.strategies.strategy_wrappers import default_logger
from avalanche.training.utils import zerolike_params_dict
from sequoia.settings.passive import PassiveEnvironment, TaskIncrementalSetting
from sequoia.settings.passive.cl.objects import Observations, Rewards
from simple_parsing.helpers.hparams import HyperParameters
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from .base import AvalancheMethod
from .base_strategy import BaseStrategy


class EWCPlugin(EWCPlugin_):
    def __init__(
        self, ewc_lambda, mode="separate", decay_factor=None, keep_importance_data=False
    ):
        super().__init__(
            ewc_lambda,
            mode=mode,
            decay_factor=decay_factor,
            keep_importance_data=keep_importance_data,
        )
        self.xs: List[Tensor] = []
        self.ys: List[Tensor] = []
        self.ts: List[Tensor] = []
        self.dataset: TensorDataset

    def after_training_exp(self, strategy: BaseStrategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        if strategy.setting:
            strategy.experience.dataset = strategy.dataset_plugin.train()

        importances = self.compute_importances(
            model=strategy.model,
            criterion=strategy.criterion,
            optimizer=strategy.optimizer,
            dataset=strategy.experience.dataset,
            device=strategy.device,
            batch_size=strategy.train_mb_size,
        )

        self.update_importances(importances, strategy.training_exp_counter)
        self.saved_params[strategy.training_exp_counter] = copy_params_dict(
            strategy.model
        )
        # clear previous parameter values
        if strategy.training_exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[strategy.training_exp_counter - 1]

    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size
    ):
        """
        Compute EWC importance matrix for each parameter
        """
        # return super().compute_importances(model, criterion, optimizer, dataset, device, batch_size)
        model.train()

        # list of list
        importances = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader = tqdm.tqdm(dataloader, desc="Computing EWC importances")
        for i, (x, y, task_labels) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), importances):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))

        return importances


class EWC(BaseStrategy):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        ewc_lambda: float,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[StrategyPlugin]] = None,
        evaluator: EvaluationPlugin = default_logger,
        eval_every=-1,
    ):
        """ Elastic Weight Consolidation (EWC) strategy.
            See EWC plugin for details.
            This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param decay_factor: used only if mode is `onlineweightedsum`.
               It specify the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
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
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

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


class EWCMethod(AvalancheMethod):
    @dataclass
    class HParams(HyperParameters):
        ewc_lambda: float = 1e-2
        ewc_mode: str = "separate"
        decay_factor: Optional[float] = None
        keep_importance_data: bool = False
        train_mb_size: int = 1
        train_epochs: int = 1
        eval_mb_size: int = None
        epochs: int = 1

    def __init__(self, hparams: HParams = None):
        super().__init__()
        self.hparams: EWCMethod.HParams = hparams or self.HParams()

    def create_model(self, setting: TaskIncrementalSetting) -> SimpleMLP:
        return super().create_model(setting)

    def create_cl_strategy(self, setting: TaskIncrementalSetting) -> BaseStrategy:
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = CrossEntropyLoss()

        # choose some metrics and evaluation method
        interactive_logger = InteractiveLogger()

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            # forgetting_metrics(experience=True), # TODO: Fix forgetting metric.
            loggers=[interactive_logger],
        )

        # create strategy
        strategy = EWC(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            ewc_lambda=self.hparams.ewc_lambda,
            mode=self.hparams.ewc_mode,
            decay_factor=self.hparams.decay_factor,
            train_epochs=self.hparams.epochs,
            device=self.device,
            train_mb_size=self.hparams.train_mb_size,
            evaluator=eval_plugin,
            eval_mb_size=self.hparams.eval_mb_size,
            eval_every=0,
        )
        strategy.setting = setting
        return strategy


if __name__ == "__main__":
    from sequoia.settings.passive import TaskIncrementalSetting

    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    method = EWCMethod()
    results = setting.apply(method)

