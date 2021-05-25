from dataclasses import dataclass
from ..continual.setting import ContinualRLSetting
from sequoia.settings.assumptions.context_discreteness import DiscreteContextAssumption
from sequoia.utils.utils import dict_union
from sequoia.settings.rl.envs import MUJOCO_INSTALLED
from typing import ClassVar, Dict


@dataclass
class DiscreteTaskAgnosticRLSetting(DiscreteContextAssumption, ContinualRLSetting):
    """ Continual Reinforcement Learning Setting where there are clear task boundaries,
    but where the task information isn't available.
    """

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[Dict[str, str]] = dict_union(
        ContinualRLSetting.available_datasets,
        {"monsterkong": "MetaMonsterKong-v0"},
        (
            # TODO: Also add the mujoco environments for the changing sizes and masses,
            # which can't be changed on-the-fly atm.
            {}
            if not MUJOCO_INSTALLED
            else {
                # "incremental_half_cheetah": IncrementalHalfCheetahEnv
            }
        ),
    )
