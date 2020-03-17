import io
import json
import pprint
from dataclasses import asdict, dataclass, is_dataclass
from os import path
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import torch
import wandb
from simple_parsing import ArgumentParser, subparsers
from simple_parsing.helpers import JsonSerializable
from torch import nn

from experiments.class_incremental import ClassIncremental
from experiments.experiment import Experiment
from experiments.iid import IID
from experiments.task_incremental import TaskIncremental


@dataclass
class RunSettings(JsonSerializable):
    """ Settings for which 'experiment' (experimental setting) to run. 
    
    Each setting has its own set of command-line arguments.
    """
    experiment: Experiment = subparsers({
        "iid": IID,
        "class-incremental": ClassIncremental,
        "task-incremental": TaskIncremental,
    })

    def __post_init__(self):
        if self.experiment.config.verbose:     
            print("Experiment:")
            pprint.pprint(asdict(self.experiment), indent=1)
            print("=" * 40)




if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_arguments(RunSettings, dest="settings")
    args = parser.parse_args()
    settings: RunSettings = args.settings

    config_dict = asdict(settings.experiment)
    if settings.experiment.config.use_wandb:
        wandb_path = settings.experiment.config.log_dir_root.joinpath('wandb')
        wandb_path.mkdir(parents=True, mode=0o777, exist_ok=True)

        # pprint.pprint(config_dict, indent=1)
        # keys_to_remove: List[str] = []

        wandb.init(project='SSCL', name=settings.experiment.config.run_name, config=config_dict, dir=str(wandb_path))
        wandb.run.save()
        settings.experiment.config.run_name = wandb.run.name
        print(f"Using wandb. Experiment name: {settings.experiment.config.run_name}")

    settings.experiment.config.log_dir.mkdir(parents=True, exist_ok=True)
    settings.experiment.save()

    print("-" * 10, f"Starting experiment '{type(settings.experiment).__name__}' ({settings.experiment.config.log_dir})", "-" * 10)
    settings.experiment.run()
    print("-" * 10, f"Experiment '{type(settings.experiment).__name__}' is done.", "-" * 10)
