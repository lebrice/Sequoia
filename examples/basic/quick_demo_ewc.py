""" Example script: Defines a new Method based on the DemoMethod from the
quick_demo.py script, adding an EWC-like loss to prevent the weights from
changing too much between tasks.
"""
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple

import gym
import torch
from torch import Tensor

# This "hack" is required so we can run `python examples/quick_demo_ewc.py`
sys.path.extend([".", ".."])
from sequoia.settings import DomainIncrementalSetting
from sequoia.settings.passive.cl.objects import Observations, Rewards
from sequoia.utils import dict_intersection
from sequoia.utils.logging_utils import get_logger

from quick_demo import DemoMethod, MyModel

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

    def __init__(self, hparams: HParams = None):
        super().__init__(hparams=hparams or self.HParams.from_args())
    
    def configure(self, setting: DomainIncrementalSetting):
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


def demo_ewc():
    """ Demo: Comparing two methods on the same setting: """
    from sequoia.settings import DomainIncrementalSetting

    ## 1. Create the Setting (same as in quick_demo.py)
    setting = DomainIncrementalSetting(dataset="fashionmnist", nb_tasks=5, batch_size=64)
    # setting = DomainIncrementalSetting.from_args()

    # 2.1: Get the results for the base method
    base_method = DemoMethod()
    base_results = setting.apply(base_method)
    
    # 2.2: Get the results for the 'improved' method:
    new_method = ImprovedDemoMethod()
    new_results = setting.apply(new_method)
    
    # Compare the two results:
    print(f"\n\nComparison: DemoMethod vs ImprovedDemoMethod - (DomainIncrementalSetting, dataset=fashionmnist):")
    print(base_results.summary())
    print(new_results.summary())
    
    exit()


if __name__ == "__main__":
    # Example: Comparing two methods on the same setting:
    from sequoia.settings import DomainIncrementalSetting
    
    ## 1. Create the Setting (same as in quick_demo.py)
    setting = DomainIncrementalSetting(dataset="fashionmnist", nb_tasks=5)
    # setting = DomainIncrementalSetting.from_args()
    
    # Get the results for the base method:
    base_method = DemoMethod()
    base_results = setting.apply(base_method)
    
    # Get the results for the 'improved' method:
    new_method = ImprovedDemoMethod()
    new_results = setting.apply(new_method)
    
    print(f"\n\nComparison: DemoMethod vs ImprovedDemoMethod - (DomainIncrementalSetting, dataset=fashionmnist):")
    print(base_results.summary())
    print(new_results.summary())
    
    # exit()
    
      
    ##
    ## As a little bonus: Evaluate *both* methods on *ALL* their applicable
    ## settings, and aggregate the results in a nice LaTeX-formatted table.
    ##
    from examples.demo_utils import compare_results, demo_all_settings
    
    base_results = demo_all_settings(DemoMethod, datasets=["mnist", "fashionmnist"])
    improved_results = demo_all_settings(ImprovedDemoMethod, datasets=["mnist", "fashionmnist"])

    compare_results({
        DemoMethod: base_results,
        ImprovedDemoMethod: improved_results,
    })
