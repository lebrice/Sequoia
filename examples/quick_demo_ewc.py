""" TODO: Same as the 'simple demo', but with addition of an EWC-like loss.
"""
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple, ClassVar, Optional, List, Dict, Type
from pathlib import Path

import gym
import pytorch_lightning as pl
import torch
import wandb
from gym import spaces
from torch import nn, Tensor
from simple_parsing import ArgumentParser, Serializable

from settings import Setting, PassiveEnvironment, PassiveSetting, ClassIncrementalSetting, Results
from common.config import Config
from methods import MethodABC as Method
from utils.logging_utils import get_logger
from utils import dict_intersection

from .quick_demo import MyModel, DemoMethod, demo, Observations, Actions, Rewards
logger = get_logger(__file__)


class MyImprovedModel(MyModel):
    """ Adds an ewc-like penalty to the demo model. """
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space,
                 ewc_coefficient: float = 1.0,
                 ewc_p_norm: int = 2,
                 ):
        super().__init__(
            observation_space,
            action_space,
            reward_space,
        )
        self.ewc_coefficient = ewc_coefficient
        self.ewc_p_norm = ewc_p_norm

        self.previous_model_weights: Dict[str, Tensor] = {}

        self._previous_task: Optional[int] = None
        self._n_switches: int = 0

    def shared_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        base_loss, metrics = super().shared_step(batch, *args, **kwargs)
        ewc_loss = self.ewc_coefficient * self.ewc_loss()
        metrics["ewc_loss"] = ewc_loss
        return base_loss + ewc_loss, metrics

    def on_task_switch(self, task_id: int)-> None:
        """ Executed when the task switches (to either a known or unknown task).
        """
        if self._previous_task is None and self._n_switches == 0:
            logger.debug("Starting the first task, no EWC update.")
        elif task_id is None or task_id != self._previous_task:
            # NOTE: We also switch between unknown tasks.
            logger.debug(f"Switching tasks: {self._previous_task} -> {task_id}: "
                         f"Updating the EWC 'anchor' weights.")
            self._previous_task = task_id
            self.previous_model_weights.clear()
            self.previous_model_weights.update(deepcopy({
                k: v.detach() for k, v in self.named_parameters()
            }))
        self._n_switches += 1

    def ewc_loss(self) -> Tensor:
        """Gets an 'ewc-like' regularization loss.

        NOTE: This is a simplified version of EWC where the loss is the P-norm
        between the current weights and the weights as they were on the begining
        of the task.
        """
        if self._previous_task is None:
            # We're in the first task: do nothing.
            return 0.

        old_weights: Dict[str, Tensor] = self.previous_model_weights
        new_weights: Dict[str, Tensor] = dict(self.named_parameters())

        loss = 0.
        for weight_name, (new_w, old_w) in dict_intersection(new_weights, old_weights):
            loss += torch.dist(new_w, old_w.type_as(new_w), p=self.ewc_p_norm)
        return loss


class ImprovedDemoMethod(DemoMethod):
    """ Improved version of the demo method, that adds an ewc-like regularizer.
    """
    # Name of this method:
    name: ClassVar[str] = "demo_ewc"
    
    @dataclass
    class HParams(DemoMethod.HParams):
        """ Hyperparameters of this new improved method. (Adds ewc params)."""
        # Coefficient of the ewc-like loss.
        ewc_coefficient: float = 1.0
        # Distance norm used in the ewc loss.
        ewc_p_norm: int = 2

    def __init__(self, hparams: HParams):
        super().__init__(hparams=hparams)
    
    def configure(self, setting: ClassIncrementalSetting):
        # Use the improved model, with the added EWC-like term.
        self.model = MyImprovedModel(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
            ewc_coefficient=self.hparams.ewc_coefficient,
            ewc_p_norm = self.hparams.ewc_p_norm,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def on_task_switch(self, task_id: Optional[int]):
        self.model.on_task_switch(task_id)


from .quick_demo import evaluate_on_all_settings, create_method


def demo():
    base_method = create_method(DemoMethod)
    base_results = evaluate_on_all_settings(base_method)
    
    improved_method = create_method(ImprovedDemoMethod)
    improved_results = evaluate_on_all_settings(improved_method)

    compare_results({
        DemoMethod: base_results,
        ImprovedDemoMethod: improved_results,
    })


def compare_results(all_results: Dict[Type[Method], Dict[Type[Setting], Dict[str, Results]]]):
    """Compare the results of the different methods, arranging them in a table.

    Parameters
    ----------
    methods : List[Method]
        [description]
    """
    # Make one huge dictionary that maps from:
    # <method, <setting, <dataset, result>>>
    from .demo_utils import make_comparison_dataframe
    comparison_df = make_comparison_dataframe(all_results)
    
    print("----- All Results -------")
    print(comparison_df)

    csv_path = Path("examples/results/comparison.csv")
    latex_path = Path("examples/results/table_comparison.tex")
    
    comparison_df.to_csv(csv_path)
    print(f"Saved dataframe with results to path {csv_path}")
    
    caption = f"Comparison of different methods on their applicable settings."
    comparison_df.to_latex(
        latex_path,
        caption=caption,
        multicolumn=False,
        multirow=False
    )
    print(f"Saved LaTeX table with results to path {latex_path}")


if __name__ == "__main__":
    demo()
