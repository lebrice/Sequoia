"""Creates the Experiment class dynamically by adding all the addons on top of ExperimentBase.

Also gathers all the `State` and `Config` subclasses (if the addons define them)
and adds them on top of `

# Load up the addons, each of which adds independent, useful functionality to the Experiment base-class.
# TODO: This might not be the cleanest/most elegant way to do it, but it's better than having files with 1000 lines in my opinion.
"""
import inspect
from dataclasses import InitVar, dataclass
from typing import Any, List, NewType, Tuple, Type, TypeVar

from simple_parsing import mutable_field

from .addons.labeled_plot_regions import LabeledPlotRegionsAddon
from .addons.replay import ReplayAddon
from .addons.representation_knn import KnnAddon
from .addons.test_time_training import TestTimeTrainingAddon
from .addons.vae_addon import SaveVaeSamplesAddon
from .experiment_base import ExperimentBase


@dataclass
class Experiment(
            LabeledPlotRegionsAddon,
            ReplayAddon,
            KnnAddon,
            TestTimeTrainingAddon,
            SaveVaeSamplesAddon,
        ):
    # If the addon has a 'Config' defined, then add it.
    # NOTE: we can't just add all the <Addon>.Config classes, since that would
    # be 
    @dataclass
    class Config(
                # LabeledPlotRegionsAddon.Config,
                ReplayAddon.Config,
                KnnAddon.Config,
                TestTimeTrainingAddon.Config,
                # SaveVaeSamplesAddon.Config,
            ):
        pass
    
    @dataclass
    class State(
                LabeledPlotRegionsAddon.State,
                # ReplayAddon.State,
                # KnnAddon.State,
                # TestTimeTrainingAddon.State,
                # SaveVaeSamplesAddon.State,
            ):
        pass
    
    config: Config = mutable_field(Config)
    state: State = mutable_field(State, init=False)

# def _register_new_experiment_addon(addon: Type[ExperimentAddon]):
#     base_classes: List[Type] = list(Experiment.__bases__)
#     # print(f"Current bases: {base_classes}")
#     n_bases = len(base_classes)
#     if ExperimentBase in base_classes:
#         index_of_exp_base = base_classes.index(ExperimentBase)
#         # print(f"Index of exp_base: {index_of_exp_base}")
#         base_classes.pop(index_of_exp_base)
#     base_classes.insert(0, addon)
#     Experiment.__bases__ = tuple(base_classes)
#     print(Experiment.mro())

# print(f"All addons: {all_addons}")
# for addon in all_addons:
#     _register_new_experiment_addon(addon)

# # Create the Experiment.Config class.
# config_dict = dict()
# for base in config_bases:
#     config_dict.update(base.__dict__)
# Config = type("Config", tuple(config_bases), config_dict)

# # Create the Experiment.State class.
# state_dict = dict()
# for base in state_bases:
#     state_dict.update(base.__dict__)
# State = type(ExperimentBase.State.__name__, tuple(state_bases), {})

# # Create the Experiment class.
# experiment_dict = dict()
# for base in all_addons:
#     experiment_dict.update(base.__dict__)
# experiment_dict["State"] = State
# experiment_dict["Config"] = Config
# Experiment = type("Experiment", tuple(all_addons), experiment_dict)

# ExperimentType = Type[ExperimentBase]
# print("EXPERIMENT")



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
