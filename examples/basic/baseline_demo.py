""" Example showing how the BaseMethod can be applied to get results in both
RL and SL settings.
"""

from sequoia.methods import BaseMethod
from sequoia.settings import TaskIncrementalSLSetting, TaskIncrementalRLSetting, Setting
from sequoia.common import Config
from simple_parsing import ArgumentParser


def baseline_demo_simple():
    config = Config()
    method = BaseMethod(config=config, max_epochs=1)
    
    ## Create *any* Setting from the tree, for example:
    ## Supervised Learning Setting:
    # setting = TaskIncrementalSLSetting(
    #     dataset="cifar10",
    #     nb_tasks=2,
    # )
    # Reinforcement Learning Setting:
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        train_max_steps=4000,
        nb_tasks=2,
    )
    results = setting.apply(method, config=config)
    print(results.summary())
    return results


def baseline_demo_command_line():
    parser = ArgumentParser(__doc__, add_dest_to_option_strings=False)
    
    # Supervised Learning Setting:
    parser.add_arguments(TaskIncrementalSLSetting, dest="setting")
    # Reinforcement Learning Setting:
    parser.add_arguments(TaskIncrementalRLSetting, dest="setting")

    parser.add_arguments(Config, dest="config")
    BaseMethod.add_argparse_args(parser, dest="method")

    args = parser.parse_args()

    setting: Setting = args.setting
    config: Config = args.config
    method: BaseMethod = BaseMethod.from_argparse_args(args, dest="method")
    
    results = setting.apply(method, config=config)
    print(results.summary())
    return results


if __name__ == "__main__":
    ### Option 1: Create the BaseMethod and Settings manually.
    baseline_demo_simple()
    
    ### Option 2: Create the BaseMethod and Settings from the command-line.
    # baseline_demo_command_line()
 