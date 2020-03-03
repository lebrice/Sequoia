import pprint
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Any

from simple_parsing import ArgumentParser, subparsers
from experiments.experiment import Experiment
from experiments.iid import IID
from experiments.class_incremental import ClassIncremental

@dataclass
class RunSettings:
    """ Settings for which 'experiment' (experimental setting) to run. 
    
    Each setting has its own set of command-line arguments.
       
    
    """
    experiment: Experiment = subparsers({
        "iid": IID,
        "class_incremental": ClassIncremental,
    })

    def __post_init__(self):
        if self.experiment.config.verbose:     
            print("Experiment:")
            pprint.pprint(asdict(self.experiment), indent=1)
            print("=" * 40)


parser = ArgumentParser()
parser.add_arguments(RunSettings, dest="settings")
args = parser.parse_args()
settings: RunSettings = args.settings

print("-" * 10, f"Starting experiment '{type(settings.experiment).__name__}'", "-" * 10)
settings.experiment.run()
print("-" * 10, f"Experiment '{type(settings.experiment).__name__}' is done.", "-" * 10)