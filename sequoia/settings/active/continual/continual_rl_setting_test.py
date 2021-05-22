import random
from pathlib import Path
from typing import List, Optional, Tuple, ClassVar, Type

import gym
import numpy as np
import pytest
from gym import spaces
from gym.vector.utils import batch_space
from sequoia.common.spaces import Image
from sequoia.settings import Setting
from sequoia.common.transforms import Transforms
from sequoia.conftest import (
    DummyEnvironment,
    mujoco_required,
    param_requires_atari_py,
    param_requires_mujoco,
    ATARI_PY_INSTALLED,
    MUJOCO_INSTALLED,
    MONSTERKONG_INSTALLED,
    param_requires_monsterkong,
)
from sequoia.methods import RandomBaselineMethod
from sequoia.settings.assumptions.incremental_test import DummyMethod
from sequoia.utils.utils import take

from .continual_rl_setting import ContinualRLSetting


def test_task_schedule_is_constructed_and_used():
    """
    Test that the tasks are switching over time.
    """
    setting = ContinualRLSetting(dataset="CartPole-v0", max_steps=100, test_steps=100,)
    # assert False, setting.train_task_schedule
    env = setting.train_dataloader(batch_size=None)

    assert len(setting.train_task_schedule) == 2
    assert len(setting.valid_task_schedule) == 2
    assert len(setting.test_task_schedule) == 2

    starting_length = env.length
    assert starting_length == 0.5

    _ = env.reset()
    lengths: List[float] = []
    for i in range(setting.max_steps):
        obs, reward, done, info = env.step(env.action_space.sample())
        if done and i != setting.steps_per_phase - 1:
            # NOTE: Don't reset on the last step
            env.reset()
        # Get the length of the pole from the environment.
        length = env.length
        lengths.append(length)
    assert not all(length == starting_length for length in lengths), lengths


import math

from gym.envs.classic_control.cartpole import CartPoleEnv

theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4
high = np.array(
    [
        x_threshold * 2,
        np.finfo(np.float32).max,
        theta_threshold_radians * 2,
        np.finfo(np.float32).max,
    ],
    dtype=np.float32,
)
expected_cartpole_obs_space = spaces.Box(-high, high, dtype=np.float32)


class TestContinualRLSetting:

    Setting: ClassVar[Type[Setting]] = ContinualRLSetting

    # IDEA: Create a fixture that creates the Setting which can then be tested.
    # TODO: Maybe this is a bit too complicated..
    @pytest.fixture(
        params=[("CartPole-v0", False), ("CartPole-v0", True),]
        + (
            [
                # Since the AtariWrapper gets added by default
                # param_requires_atari_py("Breakout-v0", True, Image(0, 255, (84, 84, 1)),),
                ("Breakout-v0", False),
            ]
            if ATARI_PY_INSTALLED
            else []
        )
        + (
            [("MetaMonsterKong-v0", False), ("MetaMonsterKong-v0", True),]
            if MONSTERKONG_INSTALLED
            else []
        )
        + (
            [("HalfCheetah-v2", False), ("Hopper-v2", False), ("Walker2d-v2", False),]
            if MUJOCO_INSTALLED
            else []
            # TODO: Add support for duckytown envs!!
            # ("duckietown", (120, 160, 3)),
        ),
        scope="session",
    )
    def setting(self, request):
        dataset, force_pixel_observations = request.param
        setting = self.Setting(
            dataset=dataset, force_pixel_observations=force_pixel_observations,
        )

        yield setting
        # assert False, setting

    # TODO: This could be the tests for all the descendants of the RL Settings!
    @pytest.mark.parametrize(
        "dataset, force_pixel_observations, expected_x_space",
        [
            ("CartPole-v0", False, expected_cartpole_obs_space),
            ("CartPole-v0", True, Image(0, 255, (400, 600, 3))),
            # param_requires_atari_py("Breakout-v0", (3, 210, 160)),
            # Since the AtariWrapper gets added by default
            param_requires_atari_py("Breakout-v0", True, Image(0, 255, (84, 84, 1)),),
            # TODO: Add support for duckytown envs!!
            # ("duckietown", (120, 160, 3)),
            param_requires_mujoco(
                "HalfCheetah-v2", False, spaces.Box(-np.inf, np.inf, (17,))
            ),
            param_requires_monsterkong(
                "MetaMonsterKong-v0", True, Image(0, 255, (64, 64, 3))
            ),
        ],
    )
    @pytest.mark.parametrize("batch_size", [None, 1, 3])
    @pytest.mark.timeout(60)
    def test_check_iterate_and_step(
        self,
        dataset: str,
        force_pixel_observations: bool,
        expected_x_space: gym.Space,
        batch_size: Optional[int],
    ):
        """ Test that the observations are of the right type and shape, regardless
        of wether we iterate on the env by calling 'step' or by using it as a
        DataLoader.
        """
        setting = self.Setting(
            dataset=dataset, force_pixel_observations=force_pixel_observations,
        )

        with gym.make(dataset) as temp_env:
            expected_action_space = temp_env.action_space

        if batch_size is not None:
            expected_batched_x_space = batch_space(expected_x_space, batch_size)
            expected_batched_action_space = batch_space(
                setting.action_space, batch_size
            )
        else:
            expected_batched_x_space = expected_x_space
            expected_batched_action_space = expected_action_space

        # BUG: Can't seem to be able to compare
        assert (
            setting.observation_space.x == expected_x_space
        ), setting.observation_space.x.low[1::2]
        assert setting.action_space == expected_action_space

        # TODO: This is changing:
        assert setting.train_transforms == []
        # assert setting.train_transforms == [Transforms.to_tensor, Transforms.three_channels]

        assert setting.nb_tasks == 1

        def check_env_spaces(env: gym.Env) -> None:
            if env.batch_size is not None:
                # TODO: This might not be totally accurate, for example because the
                # TransformObservation wrapper applied to a VectorEnv doesn't change the
                # single_observation_space, AFAIR.
                assert env.single_observation_space.x == expected_x_space
                assert env.single_action_space == expected_action_space

                assert env.observation_space.x == expected_batched_x_space
                assert env.action_space == expected_batched_action_space
            else:
                assert env.observation_space.x == expected_x_space
                assert env.action_space == expected_action_space

        def check_obs(obs: ContinualRLSetting.Observations) -> None:
            assert isinstance(obs, self.Setting.Observations), obs[0].shape
            assert obs.x in expected_batched_x_space
            # In this particular case here, the task labels should be None.
            # FIXME: For InrementalRL, this isn't correct! TestIncrementalRL should
            # therefore have its own version of this function.
            if self.Setting is ContinualRLSetting:
                assert obs.task_labels is None or all(
                    task_label == None for task_label in obs.task_labels
                )

        with setting.train_dataloader(batch_size=batch_size) as env:
            assert env.batch_size == batch_size
            check_env_spaces(env)

            obs = env.reset()
            # BUG: TODO: The observation space that we use should actually check with
            # isinstance and over the fields that fit in the space. Here there is a bug
            # because the env observations also have a `done` field.
            # assert obs in env.observation_space
            assert obs.x in env.observation_space.x
            # BUG: This doesn't currently work: (would need a tuple value rather than an
            # array.
            # assert obs.task_labels in env.observation_space.task_labels
            assert (
                tuple(obs.task_labels) if batch_size else obs.task_labels
            ) in env.observation_space.task_labels

            reset_obs = env.reset()
            check_obs(reset_obs)

            step_obs, *_ = env.step(env.action_space.sample())
            check_obs(step_obs)

            for iter_obs in take(env, 3):
                check_obs(iter_obs)
                _ = env.send(env.action_space.sample())

        with setting.val_dataloader(batch_size=batch_size) as env:
            assert env.batch_size == batch_size
            check_env_spaces(env)

            reset_obs = env.reset()
            check_obs(reset_obs)

            step_obs, *_ = env.step(env.action_space.sample())
            check_obs(step_obs)

            for iter_obs in take(env, 3):
                check_obs(iter_obs)
                _ = env.send(env.action_space.sample())

        # NOTE: Limitting the batch size at test time to None (i.e. a single env)
        # because of how the Monitor class works atm.
        batch_size = None
        expected_batched_x_space = expected_x_space
        expected_batched_action_space = expected_action_space
        with setting.test_dataloader(batch_size=batch_size) as env:
            assert env.batch_size is None
            check_env_spaces(env)

            reset_obs = env.reset()
            check_obs(reset_obs)

            step_obs, *_ = env.step(env.action_space.sample())
            check_obs(step_obs)

            for iter_obs in take(env, 3):
                check_obs(iter_obs)
                _ = env.send(env.action_space.sample())


@pytest.mark.xfail(reason="TODO: DQN model only accepts string environment names...")
def test_dqn_on_env(tmp_path: Path):
    """ TODO: Would be nice if we could have the models work directly on the
    gym envs..
    """
    from pl_bolts.models.rl import DQN
    from pytorch_lightning import Trainer

    setting = ContinualRLSetting()
    env = setting.train_dataloader(batch_size=None)
    model = DQN(env)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmp_path)
    success = trainer.fit(model)
    assert success == 1


def test_passing_task_schedule_sets_other_attributes_correctly():
    # TODO: Figure out a way to test that the tasks are switching over time.
    setting = ContinualRLSetting(
        dataset="CartPole-v0",
        train_task_schedule={
            0: {"gravity": 5.0},
            100: {"gravity": 10.0},
            200: {"gravity": 20.0},
        },
    )
    assert setting.phases == 1
    assert setting.nb_tasks == 2
    assert setting.steps_per_task == 100
    assert setting.test_task_schedule == {
        0: {"gravity": 5.0},
        5_000: {"gravity": 10.0},
        10_000: {"gravity": 20.0},
    }
    assert setting.test_steps == 10_000
    assert setting.test_steps_per_task == 5_000

    setting = ContinualRLSetting(
        dataset="CartPole-v0",
        train_task_schedule={
            0: {"gravity": 5.0},
            100: {"gravity": 10.0},
            200: {"gravity": 20.0},
        },
        test_steps_per_task=100,
    )
    assert setting.phases == 1
    assert setting.nb_tasks == 2
    assert setting.steps_per_task == 100
    assert setting.test_task_schedule == {
        0: {"gravity": 5.0},
        100: {"gravity": 10.0},
        200: {"gravity": 20.0},
    }
    assert setting.test_steps == 200
    assert setting.test_steps_per_task == 100


def test_fit_and_on_task_switch_calls():
    setting = ContinualRLSetting(
        dataset=DummyEnvironment,
        nb_tasks=5,
        steps_per_task=100,
        max_steps=500,
        test_steps_per_task=100,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
    )
    method = DummyMethod()
    _ = setting.apply(method)
    # == 30 task switches in total.
    assert method.n_task_switches == 0
    assert method.n_fit_calls == 1  # TODO: Add something like this.
    assert not method.received_task_ids
    assert not method.received_while_training


from typing import Type

from sequoia.conftest import MUJOCO_INSTALLED

if MUJOCO_INSTALLED:
    from sequoia.settings.active.envs.mujoco import (
        ContinualHalfCheetahEnv,
        ContinualHopperEnv,
        ContinualWalker2dEnv,
    )

    @mujoco_required
    @pytest.mark.parametrize(
        "dataset, expected_env_type",
        [
            ("ContinualHalfCheetah-v2", ContinualHalfCheetahEnv),
            ("HalfCheetah-v2", ContinualHalfCheetahEnv),
            ("half_cheetah", ContinualHalfCheetahEnv),
            ("ContinualHopper-v2", ContinualHopperEnv),
            ("hopper", ContinualHopperEnv),
            ("Hopper-v2", ContinualHopperEnv),
        ],
    )
    def test_mujoco_env_name_maps_to_continual_variant(
        dataset: str, expected_env_type: Type[gym.Env]
    ):
        setting = ContinualRLSetting(
            dataset=dataset, max_steps=10_000, test_steps=10_000
        )
        train_env = setting.train_dataloader()
        assert isinstance(train_env.unwrapped, expected_env_type)


@mujoco_required
@pytest.mark.parametrize("dataset", [])
def test_continual_mujoco(dataset: str):
    """ Trying to get the same-ish setup as the "LPG_FTW" experiments

    See https://github.com/Lifelong-ML/LPG-FTW/tree/master/experiments
    """
    from sequoia.common.metrics.rl_metrics import EpisodeMetrics
    from sequoia.settings import RLResults
    from sequoia.settings.active.envs.mujoco import HalfCheetahGravityEnv

    setting = ContinualRLSetting(dataset=dataset, max_steps=10_000, test_steps=10_000,)
    method = RandomBaselineMethod()

    # TODO: Change what the results look like for Continual envs.
    # (only display the average performance rather than a per-task value).

    # TODO: Using `render=True` causes a silent crash for some reason!
    results: RLResults[EpisodeMetrics] = setting.apply(method)
    assert results.average_final_performance.mean_episode_length == 1000
    assert False, results.average_final_performance
