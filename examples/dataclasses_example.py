from dataclasses import dataclass


@dataclass
class Point:
    x: float = 1.2
    y: float = 4.5
    
    # This generates the following method (among others):
    # def __init__(self, x: float = 1.2, y: float = 4.5):
    #     self.x = x
    #     self.y = y

p1 = Point(0, 0)
print(p1) # -> Point(x=0, y=0)


# 
# Second example: HyperParameters with simple-parsing:
#

from simple_parsing import ArgumentParser
from simple_parsing.helpers import choice

@dataclass
class HParams:
    """ Hyper-Parameters of my model."""
    # Learning rate.
    learning_rate: float = 3e-4
    # L2 regularization coefficient.
    weight_decay: float = 1e-6
    # Choice of optimizer
    optimizer: str = choice("adam", "sgd", "rmsprop", default="sgd")

parser = ArgumentParser()
parser.add_arguments(HParams, "hparams")
parser.print_help()
"""
usage: dataclasses_example.py [-h] [--learning_rate float]
                              [--weight_decay float]
                              [--optimizer {adam,sgd,rmsprop}]

optional arguments:
  -h, --help            show this help message and exit

HParams ['hparams']:
   Hyper-Parameters of my model.

  --learning_rate float, --hparams.learning_rate float
                        Learning rate. (default: 0.0003)
  --weight_decay float, --hparams.weight_decay float
                        L2 regularization coefficient. (default: 1e-06)
  --optimizer {adam,sgd,rmsprop}, --hparams.optimizer {adam,sgd,rmsprop}
                        Choice of optimizer (default: sgd)
"""

args = parser.parse_args()
hparams: HParams = args.hparams
print(hparams)
"""
HParams(learning_rate=0.0003, weight_decay=1e-06, optimizer='sgd')
"""