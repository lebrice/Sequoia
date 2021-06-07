import random
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type

import gym
import numpy as np
import pytest
from gym import spaces
from gym.vector.utils import batch_space
from sequoia.common.config import Config
from sequoia.common.gym_wrappers import IterableWrapper, TransformObservation
from sequoia.common.spaces import Image
from sequoia.common.transforms import Transforms
from sequoia.conftest import (
    ATARI_PY_INSTALLED,
    MONSTERKONG_INSTALLED,
    MUJOCO_INSTALLED,
    DummyEnvironment,
    mujoco_required,
    param_requires_atari_py,
    param_requires_monsterkong,
    param_requires_mujoco,
    xfail_param,
)
from sequoia.methods import Method, RandomBaselineMethod
from sequoia.settings import Environment, Setting
from sequoia.settings.assumptions.incremental_test import DummyMethod as _DummyMethod
from sequoia.utils.utils import take

from .setting import ContinualRLSetting, TaskSchedule, make_continuous_task
from sequoia.settings.rl.setting_test import DummyMethod, CheckAttributesWrapper


@pytest.mark.parametrize(
    "dataset",
    [
        "Acrobot-v0",
        "CartPole-v8",
        "Breakout-v9",
        param_requires_mujoco("Ant-v0"),
        param_requires_mujoco("Hopper-v3"),
        param_requires_monsterkong("MetaMonsterKong-v0"),
    ],
)
def test_passing_unsupported_dataset_raises_error(dataset: Any):
    with pytest.raises((gym.error.Error, NotImplementedError)):
        _ = ContinualRLSetting(dataset=dataset)


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


def make_dataset_fixture(setting_type) -> pytest.fixture:
    """ Create a parametrized fixture that will go through all the available datasets
    for a given setting. """

    def dataset(self, request):
        dataset = request.param
        return dataset

    return pytest.fixture(
        params=list(set(setting_type.available_datasets.values())), scope="module",
    )(dataset)


class TestContinualRLSetting:
    Setting: ClassVar[Type[Setting]] = ContinualRLSetting

    dataset: pytest.fixture = make_dataset_fixture(ContinualRLSetting)
    # @pytest.fixture(
    #     params=list(ContinualRLSetting.available_datasets.keys()),
    #     scope="session",
    # )
    # def dataset(self, request):
    #     dataset = request.param
    #     return dataset

    @pytest.fixture()
    def setting_kwargs(self, dataset: str):
        """ Fixture used to pass keyword arguments when creating a Setting. """
        return {"dataset": dataset}

    def test_passing_supported_dataset(self, setting_kwargs: Dict):
        setting = self.Setting(**setting_kwargs)
        assert setting.train_task_schedule
        assert all(setting.train_task_schedule.values()), "Should have non-empty tasks."
        # assert isinstance(setting._temp_train_env, expected_type)

    def test_task_creation_seeding(
        self, setting_kwargs: Dict[str, Any], config: Config
    ):
        """ Make sure that the tasks are 'reproducible' given a seed. """
        assert config.seed is not None
        setting_1 = self.Setting(**setting_kwargs, config=config)
        assert setting_1.train_task_schedule
        assert all(
            setting_1.train_task_schedule.values()
        ), "Should have non-empty tasks."

        setting_2 = self.Setting(**setting_kwargs, config=config)
        assert setting_2.train_task_schedule
        assert all(
            setting_2.train_task_schedule.values()
        ), "Should have non-empty tasks."

        assert setting_1.train_task_schedule == setting_2.train_task_schedule
        assert setting_1.val_task_schedule == setting_2.val_task_schedule
        assert setting_1.test_task_schedule == setting_2.test_task_schedule

        # Create another setting, with a different seed:
        setting_3 = self.Setting(
            **setting_kwargs, config=replace(config, seed=config.seed + 123)
        )
        assert setting_3.train_task_schedule
        assert all(
            setting_3.train_task_schedule.values()
        ), "Should have non-empty tasks."

        # NOTE: This isn't ideal: in the case where a "nb_tasks" kwarg was passed
        # to the setting constructor (i.e., when this is test is being run while testing
        # a child setting), and when that value is equal to 1, then the task schedules
        # will be identical, since by default we currently don't change the environment
        # for the first task.
        def without_first_task(task_schedule: TaskSchedule) -> TaskSchedule:
            task_schedule = task_schedule.copy()
            task_schedule.pop(0)
            return task_schedule

        for setting_1_schedule, setting_3_schedule in zip(
            map(
                without_first_task,
                [
                    setting_1.train_task_schedule,
                    setting_1.val_task_schedule,
                    setting_1.test_task_schedule,
                ],
            ),
            map(
                without_first_task,
                [
                    setting_3.train_task_schedule,
                    setting_3.val_task_schedule,
                    setting_3.test_task_schedule,
                ],
            ),
        ):
            if setting_1_schedule:
                assert setting_1_schedule != setting_3_schedule
            else:
                assert not setting_3_schedule

    def test_env_attributes_change(
        self, setting_kwargs: Dict[str, Any], config: Config
    ):
        """ Check that the values of the given attributes do change at each step during
        training.
        """
        setting_kwargs.setdefault("train_max_steps", 1000)
        setting_kwargs.setdefault("test_max_steps", 1000)
        setting = self.Setting(**setting_kwargs)
        assert setting.train_task_schedule
        assert all(setting.train_task_schedule.values())

        # task_for_dataset = make_task_for_env(dataset, step=0, change_steps=[0, 1000])
        # attributes = task_for_dataset.keys()
        attributes = set().union(
            *[task.keys() for task in setting.train_task_schedule.values()]
        )

        method = DummyMethod(
            train_wrappers=[partial(CheckAttributesWrapper, attributes=attributes)]
        )

        # method.configure(setting)
        # method.fit(setting.train_dataloader(), setting.val_dataloader())
        results = setting.apply(method, config=config)

        assert results
        self.validate_results(setting, method, results)
        # TODO: Need to limit the episodes per step in MonsterKong.
        # In MonsterKong, we might have 0 reward, since this might not even
        # constitute a full episode.
        # assert results.objective

        for attribute in attributes:
            train_values: List[float] = [
                values[attribute]
                for values_dict in method.all_train_values
                for step, values in values_dict.items()
            ]
            train_steps = setting.train_max_steps

            # Should have one (unique) value for the attribute at each step during training
            if setting.smooth_task_boundaries:
                # This is the truth condition for the ContinualRLSetting.
                # NOTE: There's an offset by 1 here because of when the env is closed.
                # NOTE: This test won't really work with integer values, but that doesn't matter
                # right now because we don't/won't support changing the values of integer
                # parameters in this "continuous" task setting.
                assert (
                    len(set(train_values)) == train_steps - 1
                ), f"{attribute} didn't change enough?"
            else:
                from ..discrete.setting import DiscreteTaskAgnosticRLSetting

                setting: DiscreteTaskAgnosticRLSetting
                train_tasks = setting.nb_tasks
                unique_attribute_values = set(train_values)
                assert len(unique_attribute_values) == train_tasks, (attribute, unique_attribute_values)

    def validate_results(
        self,
        setting: ContinualRLSetting,
        method: Method,
        results: ContinualRLSetting.Results,
    ) -> None:
        assert results
        assert results.objective
        # For now we treat the stream as a single task.
        # assert len(results.task_results) == 1
        # assert sum(results.task_result.metrics) == results.average_metric
        # assert sum(results.task_results) == results.average_metrics
        # self.validate_results(setting, method, results)

    # TODO: This could be the tests for all the descendants of the RL Settings!
    # @pytest.mark.parametrize(
    #     "dataset, force_pixel_observations, expected_x_space",
    #     [
    #         ("CartPole-v0", False, expected_cartpole_obs_space),
    #         ("CartPole-v0", True, Image(0, 255, (400, 600, 3))),
    #         param_requires_mujoco(
    #             "HalfCheetah-v3", False, spaces.Box(-np.inf, np.inf, (17,))
    #         ),
    #         # param_requires_atari_py("Breakout-v0", (3, 210, 160)),
    #         # Since the AtariWrapper gets added by default
    #         # param_requires_atari_py("Breakout-v0", True, Image(0, 255, (84, 84, 1)),),
    #         # param_requires_monsterkong(
    #         #     "MetaMonsterKong-v0", True, Image(0, 255, (64, 64, 3))
    #         # ),
    #     ],
    # )
    @pytest.mark.parametrize(
        "batch_size",
        [
            None,
            1,
            xfail_param(
                3, reason="is_closed and env_dataset cause issues in a batched env.",
            ),
        ],
    )
    @pytest.mark.timeout(60)
    def test_check_iterate_and_step(
        self, setting_kwargs: Dict[str, Any], batch_size: Optional[int],
    ):
        """ Test that the observations are of the right type and shape, regardless
        of wether we iterate on the env by calling 'step' or by using it as a
        DataLoader.
        """
        with gym.make(setting_kwargs["dataset"]) as temp_env:
            expected_x_space = temp_env.observation_space
            expected_action_space = temp_env.action_space

        setting = self.Setting(**setting_kwargs,)

        if batch_size is not None:
            expected_batched_x_space = batch_space(expected_x_space, batch_size)
            expected_batched_action_space = batch_space(
                setting.action_space, batch_size
            )
        else:
            expected_batched_x_space = expected_x_space
            expected_batched_action_space = expected_action_space

        assert setting.observation_space.x == expected_x_space
        assert setting.action_space == expected_action_space

        # TODO: This is changing:
        assert setting.train_transforms == []
        # assert setting.train_transforms == [Transforms.to_tensor, Transforms.three_channels]

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
            # because the env observations also have a `done` field, while the space
            # doesnt.
            # assert obs in env.observation_space
            assert obs.x in env.observation_space.x  # this works though.

            # BUG: This doesn't currently work: (would need a tuple value rather than an
            # array.
            # assert obs.task_labels in env.observation_space.task_labels

            if batch_size:
                # FIXME: This differs between ContinualRL and IncrementalRL:
                if not setting.known_task_boundaries_at_train_time:
                    assert obs.task_labels[0] in setting.task_label_space
                    assert tuple(obs.task_labels) in env.observation_space.task_labels
                else:
                    assert obs.task_labels[0] in setting.task_label_space
                    assert obs.task_labels in env.observation_space.task_labels
                    assert (
                        np.array(obs.task_labels) in env.observation_space.task_labels
                    )
            else:
                assert obs.task_labels in env.observation_space.task_labels

            reset_obs = env.reset()
            check_obs(reset_obs)

            # BUG: Environment is closed? (batch_size = 3, dataset = 'CartPole-v0')
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

    @pytest.mark.no_xvfb
    @pytest.mark.timeout(20)
    @pytest.mark.skipif(
        (not Path("temp").exists()),
        reason="Need temp dir for saving the figure this test creates.",
    )
    @mujoco_required
    def test_show_distributions(self, config: Config):
        setting = self.Setting(
            dataset="half_cheetah",
            max_steps=1_000,
            max_episode_steps=100,
            config=config,
        )
        from functools import partial

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3)
        name_to_env_fn = {
            "train": setting.train_dataloader,
            "valid": setting.val_dataloader,
            "test": setting.test_dataloader,
        }
        for i, (name, env_fn) in enumerate(name_to_env_fn.items()):
            env = env_fn(batch_size=None, num_workers=None)

            gravities: List[float] = []
            task_labels: List[Optional[int]] = []
            total_steps = 0
            while not env.is_closed():
                obs = env.reset()
                done = False
                steps_in_episode = 0

                while not done:
                    t = obs.task_labels
                    obs, reward, done, info = env.step(env.action_space.sample())
                    total_steps += 1
                    steps_in_episode += 1
                    y = reward.y

                    gravities.append(env.gravity)
                    print(total_steps, env.gravity)
                    if total_steps > 100:
                        assert env.gravity != -9.81

                    task_labels.append(t)

            x = np.arange(len(gravities))
            axes[0, i].plot(x, gravities, label="gravities")
            axes[0, i].legend()
            axes[0, i].set_title(f"{name} gravities")
            axes[0, i].set_xlabel("Step index")
            axes[0, i].set_ylabel("Value")

            # for task_id in task_ids:
            #     y = [t_counter.get(task_id) for t_counter in t_counters]
            #     axes[1, i].plot(x, y, label=f"task_id={task_id}")
            # axes[1, i].legend()
            # axes[1, i].set_title(f"{name} task_id")
            # axes[1, i].set_xlabel("Batch index")
            # axes[1, i].set_ylabel("Count in batch")

        plt.legend()

        Path("temp").mkdir(exist_ok=True)
        fig.set_size_inches((6, 4), forward=False)
        plt.savefig(f"temp/{self.Setting.__name__}.png")
        # plt.waitforbuttonpress(10)
        # plt.show()


# @pytest.mark.xfail(reason="TODO: pl_bolts DQN only accepts string environment names..")
# def test_dqn_on_env(tmp_path: Path):
#     """ TODO: Would be nice if we could have the models work directly on the
#     gym envs..
#     """
#     from pl_bolts.models.rl import DQN
#     from pytorch_lightning import Trainer

#     setting = ContinualRLSetting()
#     env = setting.train_dataloader(batch_size=None)
#     model = DQN(env)
#     trainer = Trainer(fast_dev_run=True, default_root_dir=tmp_path)
#     success = trainer.fit(model)
#     assert success == 1


def test_passing_task_schedule_sets_other_attributes_correctly():
    # TODO: Figure out a way to test that the tasks are switching over time.
    setting = ContinualRLSetting(
        dataset="CartPole-v0",
        train_task_schedule={
            0: {"gravity": 5.0},
            100: {"gravity": 10.0},
            200: {"gravity": 20.0},
        },
        # test_max_steps=10_000,
    )
    assert setting.phases == 1
    # assert setting.nb_tasks == 2
    # assert setting.steps_per_task == 100
    assert setting.test_task_schedule == {
        0: {"gravity": 5.0},
        5_000: {"gravity": 10.0},
        10_000: {"gravity": 20.0},
    }
    assert setting.test_max_steps == 10_000
    # assert setting.test_steps_per_task == 5_000

    setting = ContinualRLSetting(
        dataset="CartPole-v0",
        train_task_schedule={
            0: {"gravity": 5.0},
            100: {"gravity": 10.0},
            200: {"gravity": 20.0},
        },
        test_max_steps=2000,
        # test_steps_per_task=100,
    )
    assert setting.phases == 1
    # assert setting.nb_tasks == 2
    # assert setting.steps_per_task == 100
    assert setting.test_task_schedule == {
        0: {"gravity": 5.0},
        1000: {"gravity": 10.0},
        2000: {"gravity": 20.0},
    }
    assert setting.test_max_steps == 2000
    # assert setting.test_steps_per_task == 100


def test_fit_and_on_task_switch_calls():
    setting = ContinualRLSetting(
        dataset="CartPole-v0",
        # nb_tasks=5,
        # steps_per_task=100,
        train_max_steps=500,
        test_max_steps=500,
        # test_steps_per_task=100,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
    )
    method = _DummyMethod()
    _ = setting.apply(method)
    # == 30 task switches in total.
    assert method.n_task_switches == 0
    assert method.n_fit_calls == 1
    assert not method.received_task_ids
    assert not method.received_while_training


if MUJOCO_INSTALLED:
    from sequoia.settings.rl.envs.mujoco import (
        ContinualHalfCheetahV2Env,
        ContinualHalfCheetahV3Env,
        ContinualHopperEnv,
        ContinualWalker2dEnv,
    )

    @mujoco_required
    @pytest.mark.parametrize(
        "dataset, expected_env_type",
        [
            ("ContinualHalfCheetah-v2", ContinualHalfCheetahV2Env),
            ("HalfCheetah-v2", ContinualHalfCheetahV2Env),
            ("halfcheetah", ContinualHalfCheetahV2Env),
            ("ContinualHopper-v2", ContinualHopperEnv),
            ("hopper", ContinualHopperEnv),
            ("Hopper-v2", ContinualHopperEnv),
            ("walker2d", ContinualWalker2dEnv),
            ("Walker2d-v2", ContinualWalker2dEnv),
        ],
    )
    def test_mujoco_env_name_maps_to_continual_variant(
        dataset: str, expected_env_type: Type[gym.Env]
    ):
        setting = ContinualRLSetting(
            dataset=dataset, train_max_steps=10_000, test_max_steps=10_000
        )
        train_env = setting.train_dataloader()
        assert isinstance(train_env.unwrapped, expected_env_type)
