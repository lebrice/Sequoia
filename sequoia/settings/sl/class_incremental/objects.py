""" Observations/Actions/Rewards particular to a ClassIncrementalSetting. 

This is just meant as a cleaner way to import the Observations/Actions/Rewards
than particular setting.
""" 
from .setting import ClassIncrementalSetting, PassiveEnvironment
Observations = ClassIncrementalSetting.Observations
Actions = ClassIncrementalSetting.Actions
Rewards = ClassIncrementalSetting.Rewards
Environment = PassiveEnvironment
Results = ClassIncrementalSetting.Results

# ObservationType = TypeVar("ObservationType", bound=Observations)
# ActionType = TypeVar("ActionType", bound=Actions)
# RewardType = TypeVar("RewardType", bound=Rewards)
