import io
import json
import pprint
from dataclasses import asdict, dataclass, is_dataclass
from os import path
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union, Optional

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
    notes: Optional[str] = None

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

    config = settings.experiment.config
    config_dict = asdict(settings.experiment)
    # pprint.pprint(config_dict, indent=1)

    if settings.experiment.config.use_wandb:
        wandb_path = settings.experiment.config.log_dir_root.joinpath('wandb')
        wandb_path.mkdir(parents=True, mode=0o777, exist_ok=True)

        config.run_group = config.run_group or type(settings.experiment).__name__
        
        print(f"Using wandb. Experiment name: {config.run_name}")
        if config.run_name is None:
            # TODO: Create a run name using the coefficients of the tasks, etc?
            # At the moment, if no run name is given, ths
            pass
        
        wandb.init(project='SSCL', name=config.run_name, group=config.run_group, config=config_dict, dir=str(wandb_path))
        wandb.run.save()

        if config.run_name is None:
            config.run_name = wandb.run.name
        
        print(f"Using wandb. Group name: {config.run_group} run name: {config.run_name}, log_dir: {config.log_dir}")
    
    log_dir: Path = config.log_dir
    experiment_started = log_dir.is_dir()

    experiment_results_dir = (log_dir / "results")
    experiment_done = experiment_results_dir.is_dir()

    if experiment_done:
        print(f"Experiment is already done. Exiting.")
        exit(0)
    if experiment_started:
        print(f"Experiment is incomplete at directory {log_dir}.")
        # TODO: pick up where we left off ?
        # latest_checkpoint = log_dir / "checkpoints" / "todo"
        # settings.experiment = torch.load(latest_checkpoints)
    
    try:
        print("-" * 10, f"Starting experiment '{type(settings.experiment).__name__}' ({settings.experiment.config.log_dir})", "-" * 10)
        
        settings.experiment.save()
        results: Optional[Dict[Union[str, Path], Any]] = settings.experiment.run()
        # write out the results of an experiment, if any.
        if results is not None:
            for key, value in results.items():
                with open(experiment_results_dir / key) as f:
                    if results is not None:
                        torch.save(results, f)
        
        print("-" * 10, f"Experiment '{type(settings.experiment).__name__}' is done.", "-" * 10)
    
    except Exception as e:
        print(f"Experiment crashed: {e}")
        raise e
