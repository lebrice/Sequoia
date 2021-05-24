""" Observations/Actions/Rewards particular to a ClassIncrementalSetting. 

This is just meant as a cleaner way to import the Observations/Actions/Rewards
than particular setting.
""" 
from .setting import IncrementalSLSetting
from .setting import IncrementalSLTestEnvironment
from .setting import IncrementalSLTestEnvironment as ClassIncrementalTestEnvironment
from .setting import IncrementalSLSetting as ClassIncrementalSetting
from ..continual.environment import ContinualSLEnvironment as Environment
Observations = IncrementalSLSetting.Observations
Actions = IncrementalSLSetting.Actions
Rewards = IncrementalSLSetting.Rewards
# Environment = C
Results = IncrementalSLSetting.Results

# ObservationType = TypeVar("ObservationType", bound=Observations)
# ActionType = TypeVar("ActionType", bound=Actions)
# RewardType = TypeVar("RewardType", bound=Rewards)
