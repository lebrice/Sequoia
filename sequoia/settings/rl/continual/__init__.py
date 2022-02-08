from .environment import GymDataLoader
from .objects import Actions, ActionType, Observations, ObservationType, Rewards, RewardType
from .results import ContinualRLResults
from .setting import ContinualRLSetting
from .tasks import make_continuous_task

ContinualRLEnvironment = GymDataLoader
Results = ContinualRLResults
