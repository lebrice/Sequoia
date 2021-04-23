""" WIP: @lebrice: Plugins that I was using while trying to get the BaseStrategy and
plugins from Avalanche to work directly with the Sequoia environments.
"""
from typing import List

import numpy as np
import torch
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins import StrategyPlugin
from torch import Tensor
from torch.utils.data import TensorDataset


class GatherDataset(StrategyPlugin):
    """ IDEA: A Plugin that accumulates the tensors from the env to create a "proper"
    Dataset to be used by the plugins.
    """
    def __init__(self):
        self.train_xs: List[Tensor] = []
        self.train_ys: List[Tensor] = []
        self.train_ts: List[Tensor] = []
        self.train_dataset: TensorDataset
        self.train_datasets: List[TensorDataset] = []
        self.eval_xs: List[Tensor] = []
        self.eval_ys: List[Tensor] = []
        self.eval_ts: List[Tensor] = []
        self.eval_dataset: TensorDataset
        self.eval_datasets: List[TensorDataset] = []

    def after_forward(self, strategy, **kwargs):
        x, y, t = strategy.mb_x, strategy.mb_task_id, strategy.mb_y
        self.train_xs.append(x)
        self.train_ys.append(y)
        self.train_ts.append(t)
        return super().after_forward(strategy, **kwargs)

    def after_training_epoch(self, strategy, **kwargs):
        self.train_dataset = TensorDataset(
            torch.cat(self.train_xs), torch.cat(self.train_ys), torch.cat(self.train_ts)
        )
        self.train_xs.clear()
        self.train_ys.clear()
        self.train_ts.clear()
        return super().after_training_epoch(strategy, **kwargs)
    
    def after_eval_forward(self, strategy, **kwargs):
        x, y, t = strategy.mb_x, strategy.mb_task_id, strategy.mb_y
        self.eval_xs.append(x)
        self.eval_ys.append(y)
        self.eval_ts.append(t)
        return super().after_eval_forward(strategy, **kwargs)

    def after_eval_exp(self, strategy, **kwargs):
        self.eval_dataset = TensorDataset(
            torch.cat(self.eval_xs), torch.cat(self.eval_ys), torch.cat(self.eval_ts)
        )
        self.eval_xs.clear()
        self.eval_ys.clear()
        self.eval_ts.clear()
        if strategy.setting:
            strategy.experience.dataset = self.eval_dataset
        self.eval_datasets.append(self.eval_dataset)
        return super().after_eval_exp(strategy, **kwargs)

    def train(self):
        return self.train_dataset

    def eval(self):
        return self.eval_dataset

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        if strategy.setting:
            strategy.experience.dataset = self.train_dataset
        self.train_datasets.append(self.train_dataset)
        return super().after_training_exp(strategy, **kwargs)

    # def after_eval_exp(self, strategy: "BaseStrategy", **kwargs):
    #     """
    #     Compute importances of parameters after each experience.
    #     """
    #     return super().after_eval_exp(strategy, **kwargs)


class OnlineAccuracyPlugin(StrategyPlugin):
    def __init__(self):
        self.current_task_accuracies: List[float] = []
        self.all_task_accuracies: List[List[float]] = []
        self.enabled: bool = True

    def _calc_accuracy(self, strategy: "BaseStrategy") -> float:
        y_pred = strategy.logits.argmax(-1)
        y = strategy.mb_y
        acc = ((y_pred == y).sum() / len(y_pred)).item()
        return acc

    def after_forward(self, strategy: "BaseStrategy", **kwargs):
        if not self.enabled:
            return
        acc = self._calc_accuracy(strategy)
        self.current_task_accuracies.append(acc)
        return super().after_forward(strategy, **kwargs)

    def after_training_epoch(self, strategy, **kwargs):
        # Turn off at the end of the first epoch.
        self.all_task_accuracies.append(np.mean(self.current_task_accuracies))
        self.current_task_accuracies.clear()
        self.enabled = False
        return super().after_training_epoch(strategy, **kwargs)
