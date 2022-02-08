""" Example of how to create an incremental RL Setting with custom environments for each task.

In this example, we create environments using [the `procgen` package](https://github.com/openai/procgen).
"""

import dataclasses
from dataclasses import dataclass, replace
from typing import Dict, List, NamedTuple, Optional, Type, TypeVar

import gym
import numpy as np
from gym3.interop import ToGymEnv

from sequoia.settings.rl import (
    IncrementalRLSetting,
    MultiTaskRLSetting,
    TaskIncrementalRLSetting,
    TraditionalRLSetting,
)


@dataclass
class ProcGenConfig:
    """Options for creating an environment from ProcGen.

    The fields on this dataclass match the arguments that can be passed to `gym.make`, based on the
    README of the procgen repo.
    """

    # Name of environment, or comma-separate list of environment names to instantiate as each env
    # in the VecEnv.
    env_name: str = "coinrun-v0"
    # The number of unique levels that can be generated. Set to 0 to use unlimited levels.
    num_levels: int = 0
    # The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully
    # specify the set of possible levels.
    start_level: int = 0
    # Paint player velocity info in the top left corner. Only supported by certain games.
    paint_vel_info: bool = False
    # Use randomly generated assets in place of human designed assets.
    use_generated_assets: bool = False
    # Set to True to use the debug build if building from source.
    debug: bool = False
    # Useful flag that's passed through to procgen envs. Use however you want during debugging.
    debug_mode: int = 0
    # Determines whether observations are centered on the agent or display the full level.
    # Override at your own risk.
    center_agent: bool = True
    # When you reach the end of a level, the episode is ended and a new level is selected.
    # If use_sequential_levels is set to True, reaching the end of a level does not end the episode,
    # and the seed for the new level is derived from the current level seed.
    # If you combine this with start_level=<some seed> and num_levels=1, you can have a single
    # linear series of levels similar to a gym-retro or ALE game.
    use_sequential_levels: bool = False
    # What variant of the levels to use, the options are "easy", "hard", "extreme", "memory",
    # "exploration". All games support "easy" and "hard", while other options are game-specific.
    # The default is "hard". Switching to "easy" will reduce the number of timesteps required to
    # solve each game and is useful for testing or when working with limited compute resources.
    distribution_mode: str = "hard"
    # Normally games use human designed backgrounds, if this flag is set to False, games will use
    # pure black backgrounds.
    use_backgrounds: bool = True
    # Some games select assets from multiple themes, if this flag is set to True, those games will
    # only use a single theme.
    restrict_themes: bool = False
    # If set to True, games will use monochromatic rectangles instead of human designed assets.
    # Best used with restrict_themes=True.
    use_monochrome_assets: bool = False

    def make_env(self) -> gym.Env:
        """Creates the environment using these options."""
        env_id = f"procgen:procgen-{self.env_name}"
        # Create the env by passing the arguments to gym.make, same as what is done in the README of
        # the procgen repo.
        procgen_env: ToGymEnv = gym.make(
            id=env_id,
            num_levels=self.num_levels,
            start_level=self.start_level,
            paint_vel_info=self.paint_vel_info,
            use_generated_assets=self.use_generated_assets,
            debug=self.debug,
            center_agent=self.center_agent,
            use_sequential_levels=self.use_sequential_levels,
            distribution_mode=self.distribution_mode,
            use_backgrounds=self.use_backgrounds,
            restrict_themes=self.restrict_themes,
            use_monochrome_assets=self.use_monochrome_assets,
        )
        # NOTE: The environments that are created with `gym.make("procgen:procgen-...")` are
        # instances of the `gym3.interop:ToGymEnv` class, which has a slightly different API than
        # the `gym.Env` class:
        # (Taken From gym3/interop.py:)
        # > - The `render()` method does nothing in "human" mode, in "rgb_array" mode the info dict
        #     is checked for a key named "rgb" and info["rgb"][0] is returned if present
        # > - `seed()` and `close() are ignored since gym3 environments do not require these methods
        #
        # Therefore, for now, since in Sequoia we assume that the envs fit the gym.Env API, we have to
        # "patch" these different methods up a bit. Here I suggest we do this using a wrapper
        # (defined below)
        wrapped_env = SequoiaProcGenAdapterWrapper(env=procgen_env)
        return wrapped_env


class SequoiaProcGenAdapterWrapper(gym.Wrapper):
    """A wrapper around an environment from ProcGen to patch up the methods/properties that differ
    from the gym API:

    - The `seed` method doesn't ahve the right number of arguments.
    - The `done` value is of type `np.bool_` instead of a plain bool.
    - `render` returns None.
    """

    def __init__(self, env: ToGymEnv):
        super().__init__(env=env)

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)
        if isinstance(done, np.bool_):
            done = bool(done)
        return obs, rewards, done, info

    def seed(self, seed: Optional[int] = None) -> List[int]:
        # The procgen env apparently doesn't have (or need?) a `seed` method, but they don't
        # implement it corrently, by not accepting a `seed` argument!
        return []

    def render(self, mode: str = "rgb_array"):
        # note: rendering doesn't seem to be working: `self.env.render("rgb_array")` returns None.
        array: Optional[np.ndarray] = self.env.render("rgb_array")
        return array


# Type variable for a type of setting that supports passing envs for each task (all settings below
# `InrementalRLSetting`).
SettingType = TypeVar("SettingType", bound=IncrementalRLSetting)

available_envs = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


def make_procgen_setting(
    env_name: str,
    nb_tasks: int,
    num_levels_per_task: int = 1,
    overlapping_levels_between_tasks: int = 0,
    common_options: ProcGenConfig = None,
    setting_type: Type[SettingType] = TaskIncrementalRLSetting,
) -> SettingType:
    """Creates an RL Setting that uses environments from procgen.

    Parameters
    ----------
    env_name : str
        Name of the environment from procgen to use. Should include the version tag.
        For example: "coinrun-v0".
    nb_tasks : int
        Number of tasks in the setting.
    num_levels_per_task : int, optional
        Number of generated levels per task, by default 1
    overlapping_levels_between_tasks : int, optional
        Number of levels in common between neighbouring tasks. Needs to be less than
        `num_levels_per_task`. Defaults to 0, in which case all tasks distinct levels.
    common_options : ProcGenConfig, optional
        Set of options common to the envs of all the tasks. This can be used to set the starting
        level, for example. Defaults to None, in which case the default options from `ProcGenConfig`
        are used.
    setting_type : Type[SettingType], optional
        The type of setting to create, by default TaskIncrementalRLSetting.

    For example, say `nb_tasks`=5, `num_levels_per_task`=2, `overlapping_levels_between_tasks`=1:

    task #1: levels: [0, 1]
    task #2: levels: [1, 2]
    task #3: levels: [2, 3]
    task #4: levels: [3, 4]
    task #5: levels: [4, 5]

    For example, say `nb_tasks`=5, `num_levels_per_task`=5, `overlapping_levels_between_tasks`=2:
    task #1: levels: [0, 1, 2, 3, 4]
    task #2: levels: [3, 4, 5, 6, 7]
    task #3: levels: [6, 7, 8, 9, 10]
    task #4: levels: [9, 10, 11, 12, 13]
    task #5: levels: [12, 13, 14, 15, 16]

    NOTE: (lebrice): Maybe this (and other benchmark-creating functions) could be classmethods on
    the settings, instead of passing the setting_type as a parameter!

    Returns
    -------
    SettingType
        A Setting of type `setting_type` (`TaskIncrementalRLSetting`) by default, where each task
        uses environments from ProcGen.
    """
    assert overlapping_levels_between_tasks < num_levels_per_task

    # Create the options common to every task.
    if common_options is None:
        common_options = ProcGenConfig(env_name=env_name)
    else:
        common_options = dataclasses.replace(common_options, env_name=env_name)

    # Get the starting levels for each task, as shown in the docstring above.
    offset = num_levels_per_task - overlapping_levels_between_tasks
    first_task_start_level = common_options.start_level
    last_task_start_level = common_options.start_level + offset * nb_tasks
    start_levels: List[int] = list(range(first_task_start_level, last_task_start_level, offset))

    # Create the configurations that will be used to create the train/valid/test environments for
    # each task by starting from the common options, and overwriting the values of `start_level`.
    train_env_configs: List[ProcGenConfig] = [
        replace(common_options, start_level=start_levels[task_id], num_levels=num_levels_per_task)
        for task_id in range(nb_tasks)
    ]
    # NOTE: For now the validation and testing environment are the same as those for training.
    # This could easily be different though!
    # For example:
    # - the test environments could have a background while the train/valid envs don't!
    #   --> This could be super interesting to researchers in Out-of-Distribution RL!
    valid_env_configs: List[ProcGenConfig] = train_env_configs.copy()
    test_env_configs: List[ProcGenConfig] = train_env_configs.copy()

    # Here we pass a list of functions to be called to create each env. This can be a bit better
    # than passing the envs themselves, as it saves some memory, and also because we'll be able to
    # close the envs after each task (since we can always re-create them).
    setting = setting_type(
        dataset=None,
        train_envs=[config.make_env for config in train_env_configs],
        val_envs=[config.make_env for config in valid_env_configs],
        test_envs=[config.make_env for config in test_env_configs],
    )
    return setting


from sequoia.common.config import Config
from sequoia.methods.random_baseline import RandomBaselineMethod


def main_simple():
    # Simple example: Create a Task-Incremental RL setting using procgen envs.
    setting = make_procgen_setting(env_name="coinrun-v0", nb_tasks=5)
    method = RandomBaselineMethod()
    # NOTE: The `render` option isn't yet working (see above)
    results = setting.apply(method, config=Config(debug=True, render=False))
    print(results.summary())


def main_using_other_setting():
    # Example where we change what kind of setting we want to create.
    class Key(NamedTuple):
        stationary_context: bool
        task_labels_at_test_time: bool

    # This is here just to give an idea of the differences between these settings.
    available_settings: Dict[Key, Type[IncrementalRLSetting]] = {
        Key(task_labels_at_test_time=False, stationary_context=False): IncrementalRLSetting,
        Key(task_labels_at_test_time=True, stationary_context=False): TaskIncrementalRLSetting,
        Key(task_labels_at_test_time=False, stationary_context=True): TraditionalRLSetting,
        Key(task_labels_at_test_time=True, stationary_context=True): MultiTaskRLSetting,
    }

    # You can choose whichever setting you want, but for example:
    setting_type = available_settings[Key(task_labels_at_test_time=True, stationary_context=False)]
    # Create the Method.
    method = RandomBaselineMethod()

    setting = make_procgen_setting(env_name="coinrun-v0", nb_tasks=5, setting_type=setting_type)
    results = setting.apply(method, config=Config(debug=True, render=False))
    print(results.summary())


if __name__ == "__main__":
    main_simple()
