""" Observations/Actions/Rewards particular to a ClassIncrementalSetting. 

This is just meant as a cleaner way to import the Observations/Actions/Rewards
than particular setting.
""" 
from .class_incremental_setting import ClassIncrementalSetting, PassiveEnvironment

Observations = ClassIncrementalSetting.Observations
Actions = ClassIncrementalSetting.Actions
Rewards = ClassIncrementalSetting.Rewards
Environment = PassiveEnvironment
Results = ClassIncrementalSetting.Results

