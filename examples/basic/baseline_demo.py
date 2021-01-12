""" Example showing how the BaselineMethod can be applied to get results in both
RL and SL settings.
"""

from sequoia.methods import BaselineMethod
from sequoia.settings import TaskIncrementalSetting, TaskIncrementalRLSetting



if __name__ == "__main__":
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        max_steps=4000,
        nb_tasks=2,
    )
    # setting = TaskIncrementalSetting(
    #     dataset="cifar10",
    # )
    
    method = BaselineMethod()
    
    results = setting.apply(method)
    print(results.summary())