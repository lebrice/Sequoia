from dataclasses import dataclass
from ..experiment_base import ExperimentBase
from typing import List, Type

# all_addons: List[Type[ExperimentBase]] = []
# config_bases: List[Type[ExperimentBase.Config]] = []
# state_bases: List[Type[ExperimentBase.Config]] = []

@dataclass
class ExperimentAddon(ExperimentBase):
    """ Adds some optional functionality to ExperimentBase.
    
    Should be by disabled by default, and enabled by setting some command-line
    args.
    """

    # def __init_subclass__(cls, *args, **kwargs):
    #     print(f"Registered a new Experiment addon: {cls}")
    #     super().__init_subclass__(*args, **kwargs)
    #     if cls not in all_addons:
    #         all_addons.append(cls)
    #     if cls.Config is not ExperimentBase.Config:
    #         config_bases.append(cls.Config)
    #     if cls.State is not ExperimentBase.State:
    #         state_bases.append(cls.State)

        # Register this addon somewhere?