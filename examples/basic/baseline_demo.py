""" Example showing how the BaselineMethod can be applied to get results in both
RL and SL settings.
"""

from sequoia.methods import BaselineMethod
from sequoia.settings import TaskIncrementalSetting, TaskIncrementalRLSetting, Setting
from sequoia.common import Config
from simple_parsing import ArgumentParser


def baseline_demo_simple():
    config = Config()
    method = BaselineMethod(config=config, max_epochs=1)
    
    ## Create *any* Setting from the tree, for example:
    ## Supervised Learning Setting:
    # setting = TaskIncrementalSetting(
    #     dataset="cifar10",
    #     nb_tasks=2,
    # )
    # Reinforcement Learning Setting:
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        max_steps=4000,
        nb_tasks=2,
    )
    results = setting.apply(method, config=config)
    print(results.summary())
    return results


def baseline_demo_command_line():
    parser = ArgumentParser(__doc__, add_dest_to_option_strings=False)
    
    # Supervised Learning Setting:
    parser.add_arguments(TaskIncrementalSetting, dest="setting")
    # Reinforcement Learning Setting:
    parser.add_arguments(TaskIncrementalRLSetting, dest="setting")

    parser.add_arguments(Config, dest="config")
    BaselineMethod.add_argparse_args(parser, dest="method")

    args = parser.parse_args()

    setting: Setting = args.setting
    config: Config = args.config
    method: BaselineMethod = BaselineMethod.from_argparse_args(args, dest="method")
    
    results = setting.apply(method, config=config)
    print(results.summary())
    return results


if __name__ == "__main__":
    ### Option 1: Create the BaselineMethod and Settings manually.
    baseline_demo_simple()
    
    ### Option 2: Create the BaselineMethod and Settings from the command-line.
    # baseline_demo_command_line()
 