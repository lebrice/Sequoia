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




# def is_json_serializable(value: str):
#     try:
#         return json.loads(json.dumps(value)) == value 
#     except:
#         return False

# def to_str_dict(d: Dict) -> Dict[str, Union[str, Dict]]:
#     for key, value in list(d.items()):
#         d[key] = to_str(value)
#     return d

# def to_str(value: Any) -> Any:
#     try:
#         return json.dumps(value)
#     except Exception as e:
#         if is_dataclass(value):
#             d = asdict(value)
#             return to_str_dict(d)
#         elif isinstance(value, dict):
#             return to_str_dict(value)
#         elif isinstance(value, Path):
#             return str(value)
#         elif isinstance(value, nn.Module):
#             return 
#         elif isinstance(value, Iterable):
#             return list(map(to_str, value))
#         else:
#             print("Couldn't make the value into a str:", value, e)
#             return str(value)



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
