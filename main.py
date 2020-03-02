import pprint
from collections import defaultdict, OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Any

import simple_parsing
import torch
import torch.utils.data
import tqdm
from tqdm import tqdm
from simple_parsing import ArgumentParser, field, subparsers
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.classifier import Classifier
from models.ss_classifier import SelfSupervisedClassifier

from experiments.experiment import Experiment
from experiments.iid import IID
from experiments.self_supervised import SelfSupervised
from datasets.mnist import Mnist

@dataclass
class RunSettings:
    """ Settings for which 'experiment' to run. """

    experiment: Experiment = subparsers({
        "baseline": IID(dataset=Mnist()),
        "baseline_aux": SelfSupervised, # TODO:
    })

    def __post_init__(self):
        if self.experiment.config.verbose:     
            print("Experiment:")
            pprint.pprint(asdict(self.experiment), indent=1)
            print("=" * 40)


parser = ArgumentParser()
parser.add_arguments(RunSettings, dest="settings")
parser.add_subparsers
args = parser.parse_args()
settings: RunSettings = args.settings

print("-" * 10, f"Starting experiment '{type(settings.experiment).__name__}'", "-" * 10)
settings.experiment.run()
print("-" * 10, f"Experiment '{type(settings.experiment).__name__}' is done.", "-" * 10)