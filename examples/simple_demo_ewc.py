""" TODO: Same as the 'simple demo', but with addition of an EWC-like loss.
"""
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple, ClassVar, Optional, List, Dict, Type

import gym
import pytorch_lightning as pl
import torch
import wandb
from gym import spaces
from torch import nn, Tensor
from simple_parsing import ArgumentParser, Serializable

from settings import Setting, PassiveEnvironment, PassiveSetting, ClassIncrementalSetting, Results
from common.config import Config
from methods import Method
from utils.logging_utils import get_logger
from utils import dict_intersection

from .simple_demo import MyModel, MyNewMethod, demo, Observations, Actions, Rewards
logger = get_logger(__file__)


class MyNewImprovedMethod(MyNewMethod):
    """ Improved version of the demo method, that adds an ewc-like regularizer.
    """
    name: ClassVar[str] = "demo_ewc"
    @dataclass
    class HParams(Serializable):
        """ Example of HyperParameters of this method. """
        learning_rate: float = 0.01
        ewc_coefficient: float = 0.01

    def create_model(self, setting: ClassIncrementalSetting):
        return MyImprovedModel(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
            **self.hparams.to_dict()
        )

    def on_task_switch(self, task_id: Optional[int]):
        self.model.on_task_switch(task_id)


class MyImprovedModel(MyModel):
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space,
                 learning_rate: float = 0.0001,
                 ewc_coefficient: float = 1.0,
                 ewc_p_norm: int = 2
                 ):
        super().__init__(
            observation_space,
            action_space,
            reward_space,
            learning_rate=learning_rate,
        )
        self.save_hyperparameters()
        
        self.ewc_coefficient = ewc_coefficient
        self.ewc_p_norm = ewc_p_norm
        self._previous_task: Optional[int] = None
        self.previous_model_weights: Dict[str, Tensor] = {}
        self.n_switches: int = 0

    def shared_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        result = super().shared_step(batch, *args, **kwargs)
        # Here we just add the following loss to the result of the base model. 
        ewc_loss = self.ewc_coefficient * self.ewc_loss()
        result["loss"] += ewc_loss
        result["log"]["ewc_loss"] = ewc_loss
        result["progress_bar"]["ewc_loss"] = ewc_loss
        return result

    def on_task_switch(self, task_id: int)-> None:
        """ Executed when the task switches (to either a known or unknown task).
        """
        if self._previous_task is None and self.n_switches == 0:
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
        self.n_switches += 1

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



def compare_methods(methods: List[Type[Method]]):
    # Make one huge dictionary that maps from:
    # <method, <setting, <dataset, result>>>
    all_results: Dict[Type[Method], Dict[Type[Setting], Dict[str, Results]]] = {}
    
    for method_class in methods:
        all_results[method_class] = demo(method_class)

    print("----- All Results -------")
    all_methods: List[Type[Method]] = list(all_results.keys())
    all_method_names: List[str] = [m.get_name() for m in all_methods]
    
    all_settings: List[Type[Setting]] = []
    for method_class, setting_to_dataset_to_results in all_results.items():
        all_settings.extend(setting_to_dataset_to_results.keys())
    all_settings = list(set(all_settings))
    all_setting_names: List[str] = [s.get_name() for s in all_settings]

    all_datasets: List[str] = []
    for method_class, setting_to_dataset_to_results in all_results.items():
        for setting, dataset_to_results in setting_to_dataset_to_results.items():
            all_datasets.extend(dataset_to_results.keys())                
    all_datasets = list(set(all_datasets))
    
    import pandas as pd
    
    # Create the a multi-index, so we can later index df[setting, datset][method]
    # iterables = [all_setting_names, all_datasets]
    # columns = pd.MultiIndex.from_product(iterables, names=["setting", "dataset"])

    # Create the column index using the tuples that apply.
    tuples = []
    for method_class, setting_to_dataset_to_results in all_results.items():
        for setting, dataset_to_results in setting_to_dataset_to_results.items():
            setting_name = setting.get_name()
            tuples.extend((setting_name, dataset) for dataset in dataset_to_results.keys())
    tuples = list(set(tuples))          
    columns = pd.MultiIndex.from_tuples(tuples, names=["setting", "dataset"])
    rows = pd.Index(all_method_names, name="Method")
    df = pd.DataFrame(index=rows, columns=columns)
    # df.index.rename("Method", inplace=True)
    
    for method_class, setting_to_dataset_to_results in all_results.items():
        method_name = method_class.get_name()
        for setting, dataset_to_results in setting_to_dataset_to_results.items():
            setting_name = setting.get_name()
            for dataset, result in dataset_to_results.items():
                df[setting_name, dataset][method_name] = result.objective
    
    caption = f"Comparison of different methods on their applicable settings."
    print(df)

    from pathlib import Path
    results_csv_path = Path("examples/results/comparison.csv")
    latex_table_path = Path("examples/results/table_comparison.tex")
    
    with open(results_csv_path, "w") as f:
        df.to_csv(f)
    print(f"Saved dataframe with results to path {results_csv_path}")

    with open(latex_table_path, "w") as f:
        print(df.to_latex(caption=caption,
                          multicolumn=False,
                          multirow=False), file=f)
    print(f"Saved LaTeX table with results to path {latex_table_path}")


if __name__ == "__main__":
    compare_methods([MyNewMethod, MyNewImprovedMethod])
