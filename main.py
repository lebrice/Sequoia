"""Entry point used to run experiments.

You can also call the experiments directly.
"""
import argparse
import textwrap
from dataclasses import dataclass
from typing import Any, List, Optional, Type

from config import Config
from experiments import *
from experiments.experiment_base import ExperimentBase
from simple_parsing import ArgumentParser, field
from utils.logging_utils import get_logger
from utils.utils import camel_case

logger = get_logger(__file__)


def main(argv: Optional[List[str]]=None):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(description=textwrap.dedent("""\
        Which Experiment or script to run. Experiments can also be launched by
        executing the corresponding script directly. To get a more detailed view
        of the parameters of each experiment, use the corresponding "--help"
        option, as in "python main.py task_incremental --help"."""))
    
    import inspect
    import experiments
    from experiments import Experiment

    def is_experiment_subclass(c: Any) -> bool:
        return inspect.isclass(c) and issubclass(c, Experiment) and c is not Experiment

    all_experiments: List[Type[Experiment]] = [v for k, v in vars(experiments).items() if is_experiment_subclass(v)]

    # Add a subparser for each Experiment:
    for CustomExperiment in all_experiments:
        logger.debug(f"Adding args for experiment {CustomExperiment}")
        name = camel_case(CustomExperiment.__name__)
        subparser: ArgumentParser = subparsers.add_parser(name, help=CustomExperiment.__doc__)
        subparser.add_arguments(CustomExperiment, "experiment")


    # from experiments.iid import IID
    # subparser = subparsers.add_parser("iid", help=IID.__doc__)
    # subparser.add_arguments(IID, "experiment")
    
    # from experiments.task_incremental import TaskIncremental
    # subparser = subparsers.add_parser("task_incremental", help=TaskIncremental.__doc__)
    # subparser.add_arguments(TaskIncremental, "experiment")

    # from experiments.task_incremental_sem_sup import TaskIncremental_Semi_Supervised
    # subparser = subparsers.add_parser("task_incremental_semi_sup", help=TaskIncremental_Semi_Supervised.__doc__)
    # subparser.add_arguments(TaskIncremental_Semi_Supervised, "experiment")

    # from experiments.active_remembering import ActiveRemembering
    # subparser = subparsers.add_parser("active_remembering", help=ActiveRemembering.__doc__)
    # subparser.add_arguments(ActiveRemembering, "experiment")

    # Scripts to execute:
    from scripts.make_oml_plot import OmlFigureOptions
    subparser = subparsers.add_parser("make_oml_plot", help=OmlFigureOptions.__doc__)
    subparser.add_arguments(OmlFigureOptions, "options")  # Same here.
    
    args = parser.parse_args(argv)

    experiment: Experiment = args.experiment
    experiment.launch()

    
if __name__ == "__main__":
    main()
