""" Example showing how the BaselineMethod can be applied to get results in both
RL and SL settings.
"""

from sequoia.methods import BaselineMethod
from sequoia.settings import TaskIncrementalSetting, TaskIncrementalRLSetting
from sequoia.common import Config


if __name__ == "__main__":
    # setting = TaskIncrementalRLSetting(
    #     dataset="cartpole",
    #     observe_state_directly=True,
    #     max_steps=4000,
    #     nb_tasks=2,
    # )
    # TODO: The length of each epoch doesn't show up in the progressbar anymore.
    setting = TaskIncrementalSetting(
        dataset="cifar10",
        nb_tasks=2,
    )
    config = Config(render=True, debug=True)
    method = BaselineMethod(config=config, max_epochs=1)
    results = setting.apply(method, config=config)
    print(results.summary())