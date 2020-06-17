"""Creates the Experiment class dynamically by adding all the addons on top of ExperimentBase.

Also gathers all the `State` and `Config` subclasses (if the addons define them)
and adds them as the base classes of

"""
import inspect
from dataclasses import InitVar, dataclass
from typing import Any, List, Tuple, Type

import addons
from addons.addon import all_addons, config_bases, state_bases
from experiment_base import ExperimentBase
# Load up the addons, each of which adds independent, useful functionality to the Experiment base-class.
# TODO: This might not be the cleanest/most elegant way to do it, but it's better than having files with 1000 lines in my opinion.
from simple_parsing import mutable_field

config_dict = dict()
for base in config_bases:
    config_dict.update(base.__dict__)
Config = type("Config", tuple(config_bases), config_dict)

state_dict = dict()
for base in state_bases:
    state_dict.update(base.__dict__)
State = type(ExperimentBase.State.__name__, tuple(state_bases), {})
experiment_dict = dict()
for base in all_addons:
    experiment_dict.update(base.__dict__)
experiment_dict["State"] = State
experiment_dict["Config"] = Config
Experiment = type("Experiment", tuple(all_addons), experiment_dict)

# a = Experiment()
# print(a)
# exit()

# @dataclass
# class Experiment(*all_addons):
#     @dataclass
#     class Config(*config_bases):
#         pass

#     @dataclass
#     class State(*state_bases):
#         pass


# print(all_addons)
# exit()

# @dataclass  # type: ignore
# class Experiment(ExperimentWithKNN,
#                  ExperimentWithVAE,
#                  TestTimeTrainingAddon,
#                  LabeledPlotRegionsAddon,
#                  ExperimentWithReplay):
#     """ Class used to perform a given 'method' or evaluation/experiment.
#     (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
#     To create a new experiment, subclass this class, and add/change what you
#     need to customize in the Config class.
#     """

#     @dataclass
#     class Config(ExperimentWithKNN.Config,
#                  TestTimeTrainingAddon.Config,
#                  ExperimentWithReplay.Config):
#         """ Describes the parameters of an experiment. """
#         pass
    
#     def __post_init__(self, *args, **kwargs):
#         super().__post_init__(*args, **kwargs)

#     config: Config = mutable_field(Config)

#     # # Experiment Config: non-tunable parameters specific to an experiment.
#     # config: Config = mutable_field(Config)
#     # # Model Hyper-parameters (tunable) settings.
#     # hparams: Classifier.HParams = mutable_field(Classifier.HParams)
#     # # State of the experiment (not parsed form command-line).
#     # state: State = mutable_field(State, init=False)
