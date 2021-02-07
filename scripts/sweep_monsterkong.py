"""Runs a hyper-parameter tuning sweep for the BaselineMethod on Multi-task MonsterKong
environment.
"""
import wandb
from sequoia.common import Config
from sequoia.methods.baseline_method import BaselineMethod
from sequoia.settings import IIDSetting, Results, Setting
from sequoia.utils.logging_utils import get_logger
from simple_parsing import ArgumentParser
from sequoia.settings import RLSetting
logger = get_logger(__file__)


if __name__ == "__main__":

    ## Create the Setting:
    setting = RLSetting(dataset="monsterkong", nb_tasks=10, max_steps=10_000)
    # from sequoia.settings import TaskIncrementalSetting
    # setting = TaskIncrementalSetting(dataset="cifar10")
    
    ## Create the BaselineMethod:
    # Option 1: Create the method manually:
    # method = BaselineMethod()

    # Option 2: From the command-line:
    method, unused_args = BaselineMethod.from_known_args()

    # Search space for the Hyper-Parameter optimization algorithm.
    # NOTE: This is just a copy of the spaces that are auto-generated from the fields of
    # the `BaselineModel.HParams` class. You can change those as you wish though.
    search_space = {}
    best_hparams, best_results = method.hparam_sweep(
        setting, search_space=search_space, experiment_id=None,
    )

    print(f"Best hparams: {best_hparams}, best perf: {best_results}")
    # results = setting.apply(method, config=Config(debug=True))

