import dataclasses
import enum
import inspect
import functools
import math
import random
from typing import Any, ClassVar, Dict, NamedTuple, Optional, Type

import gym
import numpy as np
import pytest
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from sequoia.common.config import Config
from sequoia.common.gym_wrappers import RenderEnvWrapper
from sequoia.common.spaces import Image, Sparse
from sequoia.conftest import (
    metaworld_required,
    monsterkong_required,
    mtenv_required,
    mujoco_required,
    slow,
    xfail_param,
)
from sequoia.settings.base import Setting
from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.settings.assumptions.incremental_test import OtherDummyMethod
from sequoia.settings.rl import TaskIncrementalRLSetting
from sequoia.settings.rl.continual.setting_test import all_different_from_next
from sequoia.settings.rl.setting_test import DummyMethod

from ..discrete.setting_test import (
    TestDiscreteTaskAgnosticRLSetting as DiscreteTaskAgnosticRLSettingTests,
)
from .setting import IncrementalRLSetting


class TestIncrementalRLSetting(DiscreteTaskAgnosticRLSettingTests):
    Setting: ClassVar[Type[Setting]] = IncrementalRLSetting
    dataset: pytest.fixture

    @pytest.fixture()
    def setting_kwargs(self, dataset: str, nb_tasks: int, config: Config):
        """Fixture used to pass keyword arguments when creating a Setting."""
        kwargs = {"dataset": dataset, "nb_tasks": nb_tasks, "max_episode_steps": 100}
        if dataset.lower().startswith(("walker2d", "hopper", "halfcheetah", "continual")):
            # kwargs["train_max_steps"] = 5_000
            # kwargs["max_episode_steps"] = 100
            pass
        # NOTE: Using 0 workers so I can parallelize the tests without killing my PC.
        config.num_workers = 0
        kwargs["config"] = config
        return kwargs

    def test_passing_supported_dataset(self, setting_kwargs: Dict):
        # Override this test because envs can be passed for each task.
        setting = self.Setting(**setting_kwargs)
        assert setting.train_task_schedule
        if setting.train_envs:
            # Passing the dataset created custom envs for each task (e.g. MT10, CW10, LPG-FTW-(...).
            # The task schedule should have keys for the task boundary steps, but values should be
            # empty dictionaries.
            assert not any(setting.train_task_schedule.values())
        else:
            # Passing the dataset created a task schedule.
            assert all(setting.train_task_schedule.values()), "Should have non-empty tasks."

    def validate_results(
        self,
        setting: IncrementalRLSetting,
        method: DummyMethod,
        results: IncrementalRLSetting.Results,
    ) -> None:
        """Check that the results make sense.
        The Dummy Method used also keeps useful attributes, which we check here.
        """
        assert results
        assert results.objective
        assert len(results.task_sequence_results) == setting.nb_tasks
        assert results.average_final_performance == sum(
            results.task_sequence_results[-1].average_metrics_per_task
        )
        t = setting.nb_tasks
        p = setting.phases
        assert setting.known_task_boundaries_at_train_time
        assert setting.known_task_boundaries_at_test_time
        assert setting.task_labels_at_train_time
        # assert not setting.task_labels_at_test_time
        assert not setting.stationary_context
        if setting.nb_tasks == 1:
            assert not method.received_task_ids
            assert not method.received_while_training
        else:
            assert method.received_task_ids == sum(
                [
                    [t_i] + [t_j if setting.task_labels_at_test_time else None for t_j in range(t)]
                    for t_i in range(t)
                ],
                [],
            )
            assert method.received_while_training == sum(
                [[True] + [False for _ in range(t)] for t_i in range(t)], []
            )

    def test_tasks_are_different(self, setting_kwargs: Dict[str, Any], config: Config):
        """Check that the tasks different from the next.

        NOTE: Overriding this test because task schedules are empty when using custom envs for each
        task.
        """
        config = setting_kwargs.pop("config", config)
        assert config.seed is not None
        setting = self.Setting(**setting_kwargs, config=config)

        # Check that each task is different from the next.
        # NOTE: When custom datasets are used for each task then the task schedules' values are
        # empty, we have to change the test condition a little bit here.
        if setting.train_envs:
            # The dataset being used resulted in creating an env per task, rather than just using
            # one env with a task schedule.
            # Make sure that the fn for creating the env of each task is unique.
            assert all_different_from_next(setting.train_envs)
            assert all_different_from_next(setting.val_envs)
            assert all_different_from_next(setting.test_envs)
        else:
            # Check that each task is different from the next.
            assert all_different_from_next(setting.train_task_schedule.values())
            assert all_different_from_next(setting.val_task_schedule.values())
            assert all_different_from_next(setting.test_task_schedule.values())

    def test_number_of_tasks(self):
        setting = self.Setting(
            dataset="CartPole-v0",
            monitor_training_performance=True,
            nb_tasks=10,
            train_max_steps=10_000,
            test_max_steps=1000,
        )
        assert setting.nb_tasks == 10

    def test_max_number_of_steps_per_task_is_respected(self):
        setting = self.Setting(
            dataset="CartPole-v0",
            monitor_training_performance=True,
            # train_steps_per_task=500,
            nb_tasks=2,
            train_max_steps=1000,
            test_max_steps=1000,
        )
        for task_id in range(setting.phases):
            setting.current_task_id = task_id
            train_env = setting.train_dataloader()
            total_steps = 0
            while total_steps < setting.steps_per_phase:
                print(total_steps)
                obs = train_env.reset()

                done = False
                while not done:
                    if total_steps == setting.current_train_task_length:
                        assert train_env.is_closed()
                        with pytest.raises(gym.error.ClosedEnvironmentError):
                            obs, reward, done, info = train_env.step(
                                train_env.action_space.sample()
                            )
                        return
                    else:
                        obs, reward, done, info = train_env.step(train_env.action_space.sample())
                    total_steps += 1

            assert total_steps == setting.steps_per_phase

            with pytest.raises(gym.error.ClosedEnvironmentError):
                train_env.reset()

    @monsterkong_required
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "state",
        [False, xfail_param(True, reason="TODO: MonsterkongState doesn't work?")],
    )
    def test_monsterkong(self, state: bool):
        """Checks that the MonsterKong env works fine with pixel and state input."""
        setting = self.Setting(
            dataset="StateMetaMonsterKong-v0" if state else "PixelMetaMonsterKong-v0",
            # force_state_observations=state,
            # force_pixel_observations=(not state),
            nb_tasks=5,
            train_max_steps=500,
            test_max_steps=500,
            # train_steps_per_task=100,
            # test_steps_per_task=100,
            train_transforms=[],
            test_transforms=[],
            val_transforms=[],
            max_episode_steps=10,
        )

        if state:
            # State-based monsterkong: We observe a flattened version of the game state
            # (20 x 20 grid + player cell and goal cell, IIRC.)
            assert setting.observation_space.x == spaces.Box(
                0, 292, (402,), np.int16
            ), setting._temp_train_env.observation_space
        else:
            assert setting.observation_space.x == Image(0, 255, (64, 64, 3), np.uint8)

        if setting.task_labels_at_test_time:
            assert setting.observation_space.task_labels == spaces.Discrete(5)
        else:
            assert setting.task_labels_at_train_time
            assert setting.observation_space.task_labels == Sparse(
                spaces.Discrete(5),
                sparsity=0.5,  # 0.5 since we have task labels at train time.
            )

        assert setting.test_max_steps == 500
        with setting.train_dataloader() as env:
            obs = env.reset()
            assert obs in setting.observation_space

        method = DummyMethod()
        results = setting.apply(method)

        self.validate_results(setting, method, results)

    @mujoco_required
    @pytest.mark.parametrize("seed", [None, 123, 456])
    @pytest.mark.parametrize("version", ["v2", "v3"])
    @pytest.mark.parametrize("env_name", ["HalfCheetah", "Hopper", "Walker2d"])
    @pytest.mark.parametrize("modification", ["bodyparts", "gravity"])
    def test_LPG_FTW_datasets(
        self,
        env_name: str,
        modification: str,
        version: str,
        config: Config,
        seed: int,
    ):
        """Test using a dataset from the LPG-FTW paper / repo (continual mujoco variants).

        TODO: Check that:
        - the task sequence is always the same (uses the same seed), regardless of what seed is
          passed;
        - The envs are created correctly;
        - The number of tasks / train steps / test steps / etc is set to the right values.
        """
        # LPG-FTW-{bodysize|gravity}-{HalfCheetah|Hopper|Walker2d}-{v2|v3}
        dataset = f"LPG-FTW-{modification}-{env_name}-{version}"

        # NOTE: Set the seed in the config, preserving the other values:
        config = dataclasses.replace(config, seed=seed)
        nb_tasks: Optional[int] = None
        setting: TaskIncrementalRLSetting = self.Setting(
            dataset=dataset,
            nb_tasks=nb_tasks,
            config=config,
        )

        if nb_tasks is not None:
            assert setting.nb_tasks == nb_tasks
        else:
            assert setting.nb_tasks == 20 if env_name in ["HalfCheetah", "Hopper"] else 50

        assert setting.train_steps_per_task == 100_000
        assert setting.train_max_steps == setting.train_steps_per_task * setting.nb_tasks
        assert setting.test_steps_per_task == 10_000
        assert setting.test_max_steps == setting.test_steps_per_task * setting.nb_tasks
        assert setting.config == config

        expected_values = {
            "bodyparts": {
                "HalfCheetah": np.array(
                    [
                        [1.0667, 1.354, 1.1454, 0.9112],
                        [0.968, 1.3214, 0.8125, 1.2862],
                        [0.9356, 0.7476, 0.9421, 1.397],
                        [1.057, 1.0286, 0.776, 1.3749],
                        [0.7592, 1.3059, 0.6209, 0.9313],
                        [0.8497, 1.016, 0.869, 0.9722],
                        [0.6936, 0.7496, 0.9946, 0.7713],
                        [0.9878, 1.1394, 1.438, 1.3296],
                        [1.1359, 1.1118, 1.4415, 1.3868],
                        [0.5468, 0.9953, 1.3474, 1.3668],
                        [0.7779, 0.5924, 0.8996, 0.8196],
                        [0.9775, 0.7775, 1.3211, 1.1515],
                        [0.6026, 0.833, 0.9688, 1.4437],
                        [0.6035, 1.161, 1.0771, 0.7065],
                        [1.0629, 1.4446, 0.9937, 0.5573],
                        [1.2337, 0.522, 1.0446, 0.86],
                        [0.7313, 1.35, 1.2919, 0.6101],
                        [1.0026, 0.5937, 0.6216, 1.3764],
                        [0.6369, 0.8332, 1.0068, 1.1956],
                        [1.1337, 0.8872, 1.0393, 1.4391],
                    ]
                ),
                "Hopper": np.array(
                    [
                        [0.7135, 0.5054, 1.3158, 1.3817],
                        [1.2478, 1.4622, 0.8828, 0.7484],
                        [0.5758, 1.4022, 1.0022, 1.2518],
                        [1.4175, 0.5328, 0.8692, 0.6997],
                        [0.6962, 1.3126, 1.2338, 1.4018],
                        [1.4837, 1.0798, 0.7868, 0.8489],
                        [1.3545, 0.7424, 1.2719, 1.0976],
                        [0.6088, 0.516, 0.8584, 1.0396],
                        [1.19, 0.6938, 0.5663, 0.8589],
                        [0.8211, 1.3241, 0.9745, 1.345],
                        [0.6572, 1.0763, 1.3601, 0.659],
                        [0.7739, 0.7299, 0.6518, 1.469],
                        [1.0556, 0.7345, 0.532, 1.0279],
                        [1.2296, 0.6701, 1.4398, 1.0611],
                        [0.6225, 1.0743, 0.827, 0.6753],
                        [0.7325, 0.809, 1.2254, 0.9415],
                        [1.4439, 0.9964, 1.4649, 1.333],
                        [0.5189, 0.9123, 1.1166, 1.3882],
                        [1.0468, 1.4162, 1.4152, 1.4333],
                        [1.1143, 1.2726, 1.0209, 1.0729],
                    ]
                ),
                "Walker2d": np.array(
                    [
                        [0.7567, 0.756, 1.4277, 0.9565],
                        [1.4109, 0.5937, 0.7606, 0.6839],
                        [1.0276, 1.2041, 1.4451, 0.8439],
                        [0.9755, 0.8187, 0.591, 0.583],
                        [1.2181, 0.8519, 0.5878, 0.9935],
                        [0.8885, 1.2908, 1.3013, 1.1454],
                        [1.0147, 0.7442, 1.236, 0.5236],
                        [1.1978, 0.5307, 1.4067, 1.1635],
                        [0.9529, 0.8574, 0.6655, 0.5294],
                        [0.8051, 1.1687, 0.8499, 1.3864],
                        [1.2848, 0.8866, 0.5215, 1.0251],
                        [1.2241, 0.7499, 1.1479, 0.5744],
                        [1.2354, 0.5853, 1.1212, 0.5174],
                        [0.7968, 0.7717, 1.2285, 0.8687],
                        [1.0544, 0.5814, 0.8588, 0.687],
                        [1.0695, 0.6469, 0.8567, 0.6682],
                        [1.2904, 0.8367, 1.228, 0.8606],
                        [1.0343, 0.7646, 0.515, 1.3386],
                        [1.1157, 1.2064, 1.0026, 0.9877],
                        [0.6621, 0.809, 1.0466, 0.5361],
                        [0.9291, 0.6168, 0.9013, 1.4358],
                        [1.048, 0.8483, 0.8586, 1.1867],
                        [1.327, 1.0487, 1.4479, 0.9426],
                        [1.2382, 0.8678, 1.0034, 1.2412],
                        [0.5863, 1.4389, 0.934, 1.3923],
                        [1.1379, 1.154, 0.5595, 0.5955],
                        [1.3881, 1.3309, 0.5342, 1.1085],
                        [0.8394, 1.0508, 0.9655, 0.7755],
                        [0.7494, 0.6891, 0.6979, 1.3249],
                        [1.1108, 1.3998, 0.7783, 0.599],
                        [0.8687, 0.5902, 1.212, 0.6375],
                        [0.5668, 0.981, 0.5026, 1.0739],
                        [0.9416, 1.4424, 1.0721, 0.9112],
                        [1.2981, 1.0119, 1.2722, 0.9808],
                        [1.4171, 1.1066, 0.6053, 1.2302],
                        [1.1096, 1.0246, 1.3117, 0.5727],
                        [0.8082, 0.875, 0.9299, 1.2194],
                        [1.0526, 0.961, 1.0492, 1.2552],
                        [1.46, 0.8331, 0.934, 0.5725],
                        [1.3832, 1.4736, 1.2651, 0.7956],
                        [0.68, 1.2663, 1.4183, 0.9284],
                        [1.2713, 0.6865, 0.8331, 1.0081],
                        [1.4115, 0.5781, 0.9823, 0.8094],
                        [1.4614, 0.5998, 1.2237, 1.3794],
                        [1.2385, 1.2489, 0.7521, 0.818],
                        [1.077, 1.2589, 0.748, 1.1483],
                        [0.7855, 1.1619, 0.5537, 1.2367],
                        [1.4765, 1.1728, 0.9052, 1.3113],
                        [1.1144, 0.9986, 1.3052, 0.9948],
                        [1.1542, 1.3616, 0.7465, 0.8679],
                    ]
                ),
            },
            "gravity": {
                "HalfCheetah": np.array(
                    [
                        -10.4648,
                        -13.2825,
                        -11.236,
                        -8.9384,
                        -9.4964,
                        -12.9626,
                        -7.9709,
                        -12.6178,
                        -9.1777,
                        -7.3343,
                        -9.2424,
                        -13.7041,
                        -10.3694,
                        -10.091,
                        -7.6124,
                        -13.4874,
                        -7.4477,
                        -12.8111,
                        -6.0907,
                        -9.1363,
                    ]
                ),
                "Hopper": np.array(
                    [
                        -6.999,
                        -4.9579,
                        -12.9078,
                        -13.5543,
                        -12.2405,
                        -14.3439,
                        -8.6606,
                        -7.3419,
                        -5.6488,
                        -13.7555,
                        -9.8317,
                        -12.2801,
                        -13.9059,
                        -5.2266,
                        -8.5266,
                        -6.8638,
                        -6.83,
                        -12.8763,
                        -12.104,
                        -13.7512,
                    ]
                ),
                "Walker2d": np.array(
                    [
                        -7.4229,
                        -7.4163,
                        -14.006,
                        -9.3835,
                        -13.8414,
                        -5.8243,
                        -7.461,
                        -6.7093,
                        -10.0807,
                        -11.8119,
                        -14.1762,
                        -8.2791,
                        -9.57,
                        -8.031,
                        -5.7979,
                        -5.7189,
                        -11.9495,
                        -8.3575,
                        -5.7666,
                        -9.7467,
                        -8.7165,
                        -12.6623,
                        -12.7656,
                        -11.2362,
                        -9.9544,
                        -7.3011,
                        -12.1249,
                        -5.1366,
                        -11.7508,
                        -5.2058,
                        -13.8,
                        -11.4139,
                        -9.3481,
                        -8.4107,
                        -6.5289,
                        -5.1934,
                        -7.898,
                        -11.4647,
                        -8.3374,
                        -13.6001,
                        -12.6038,
                        -8.6978,
                        -5.1157,
                        -10.0563,
                        -12.0081,
                        -7.3568,
                        -11.2612,
                        -5.6351,
                        -12.1197,
                        -5.7417,
                    ]
                ),
            },
        }

        def _unwrap_partials(env_fn: functools.partial) -> functools.partial:
            from gym.envs.mujoco import MujocoEnv

            # 'unwrap' the env fn:
            while isinstance(env_fn, functools.partial):
                # We want to recover the 'base' env factory (the function that actually creates
                # the modified mujoco env.)
                # NOTE `env_fn` is probably something like:
                # `partial(create_env, base_env_factory,  wrappers=[...])
                # or
                # `partial(foo, env_fn=base_env_factory,  wrappers=[...])
                print(env_fn)
                if inspect.isclass(env_fn.func) and issubclass(env_fn.func, MujocoEnv):
                    # Reached the lowest-level partial, the one we're looking for.
                    break
                if env_fn.args:
                    env_fn = env_fn.args[0]
                else:
                    env_fn = list(env_fn.keywords.values())[0]
            return env_fn

        if modification == "bodyparts":
            expected_factors_for_env = expected_values["bodyparts"][env_name]

            def check_env_fn_matches_expected(task_id: int, env_fn: functools.partial):
                env_fn = _unwrap_partials(env_fn)
                assert isinstance(env_fn, functools.partial)
                kwargs = env_fn.keywords

                for argument_name in ["body_name_to_size_scale", "body_name_to_mass_scale"]:
                    argument_values = np.array(list(kwargs[argument_name].values()))
                    assert (argument_values == expected_factors_for_env[task_id]).all()

            env_fn: functools.partial

            # Inspect the env functions and check that the arguments that would be passed to the
            # constructor make sense.
            # NOTE: Could also create the envs using the setting and inspect these attributes,
            # but I think that inspecting the attributes on the multi-env wrappers used by the
            # Traditional and MultiTask RL settings might not work. This is ok for now.

            for task_id, env_fn in enumerate(setting.train_envs):
                check_env_fn_matches_expected(task_id, env_fn)
            for task_id, env_fn in enumerate(setting.val_envs):
                check_env_fn_matches_expected(task_id, env_fn)
            for task_id, env_fn in enumerate(setting.test_envs):
                check_env_fn_matches_expected(task_id, env_fn)
        elif modification == "gravity":
            expected_gravities_for_env = expected_values["gravity"][env_name]

            def check_env_fn_matches_expected(task_id: int, env_fn: functools.partial):
                env_fn = _unwrap_partials(env_fn)
                kwargs = env_fn.keywords
                gravity_value: float = kwargs["gravity"]
                assert np.isclose(gravity_value, expected_gravities_for_env[task_id])

            for task_id, env_fn in enumerate(setting.train_envs):
                check_env_fn_matches_expected(task_id, env_fn)
            for task_id, env_fn in enumerate(setting.val_envs):
                check_env_fn_matches_expected(task_id, env_fn)
            for task_id, env_fn in enumerate(setting.test_envs):
                check_env_fn_matches_expected(task_id, env_fn)

        # TODO: Not sure if this check will also work with the stationary settings, so skipping it
        # for now.
        if setting.stationary_context:
            return

        # Check that the max episode length is really respected.
        with setting.train_dataloader() as temp_env:
            steps = 0
            obs = temp_env.reset()
            done = False
            while not done:
                action = temp_env.action_space.sample()
                obs, reward, done, info = temp_env.step(action)
                assert obs in temp_env.observation_space
                steps += 1
                assert steps <= 1000
            assert steps <= 1000

        # NOTE: Testing the 'live' envs is much slower, since we have to actually isntantiate the
        # envs. Skipping the rest for now.
        return

        def _check_env_attributes_match(task_id: int, env: gym.Env):
            if modification == "bodyparts":
                size_scales = env.body_name_to_size_scale
                mass_scales = env.body_name_to_mass_scale
                assert size_scales == mass_scales
                assert list(size_scales.values()) == expected_factors_for_env[task_id].tolist()
            elif modification == "gravity":
                gravity = env.gravity
                assert gravity == expected_gravities_for_env[task_id]

        setting.prepare_data()
        for task_id in range(setting.nb_tasks):
            print(f"Testing the 'live' envs for task {task_id}.")
            setting.current_task_id = task_id

            with setting.train_dataloader() as env:
                _check_env_attributes_match(task_id, env)
            with setting.val_dataloader() as env:
                _check_env_attributes_match(task_id, env)
            with setting.test_dataloader() as env:
                _check_env_attributes_match(task_id, env)


@pytest.mark.timeout(120)
def test_action_space_always_matches_obs_batch_size_in_RL(config: Config):
    """ """
    from sequoia.settings import TaskIncrementalRLSetting

    nb_tasks = 2
    batch_size = 1
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        nb_tasks=nb_tasks,
        batch_size=batch_size,
        train_max_steps=200,
        test_max_steps=200,
        num_workers=4,  # Intentionally wrong
        # monitor_training_performance=True, # This is still a TODO in RL.
    )
    total_samples = len(setting.test_dataloader())

    method = OtherDummyMethod()
    _ = setting.apply(method, config=config)

    expected_encountered_batch_sizes = {batch_size or 1}
    last_batch_size = total_samples % (batch_size or 1)
    if last_batch_size != 0:
        expected_encountered_batch_sizes.add(last_batch_size)
    assert set(method.batch_sizes) == expected_encountered_batch_sizes

    # NOTE: Multiply by nb_tasks because the test loop is ran after each training task.
    actual_num_batches = len(method.batch_sizes)
    expected_num_batches = math.ceil(total_samples / (batch_size or 1)) * nb_tasks
    # MINOR BUG: There's an extra batch for each task. Might make sense, or might not,
    # not sure.
    assert actual_num_batches == expected_num_batches + nb_tasks

    expected_total = total_samples * nb_tasks
    actual_total_obs = sum(method.batch_sizes)
    assert actual_total_obs == expected_total + nb_tasks


@mtenv_required
@pytest.mark.xfail(reason="don't know how to get the max path length through mtenv!")
def test_mtenv_meta_world_support():
    from mtenv import MTEnv, make

    env: MTEnv = make("MT-MetaWorld-MT10-v0")
    env.set_task_state(0)
    env.seed(123)
    env.seed_task(123)
    obs = env.reset()
    assert isinstance(obs, dict)
    assert list(obs.keys()) == ["env_obs", "task_obs"]
    print(obs)
    done = False
    # BUG: No idea how to get the max path length, since I'm getting
    # AttributeError: 'MetaWorldMTWrapper' object has no attribute 'max_path_length'
    steps = 0
    while not done and steps < env.max_path_length:
        obs, reward, done, info = env.step(env.action_space.sample())
        # BUG: Can't render when using metaworld through mtenv, since mtenv *contains* a
        # straight-up copy-pasted old version of meta-world, which doesn't support it.
        env.render()
        steps += 1
    env.close()

    env_obs_space = env.observation_space["env_obs"]
    task_obs_space = env.observation_space["task_obs"]
    # TODO: If the task observation space is Discrete(10), then we can't create a
    # setting with more than 10 tasks! We could add a check for this.
    # TODO: Figure out the default number of tasks depending on the chosen dataset.
    setting = IncrementalRLSetting(dataset="MT-MetaWorld-MT10-v0", nb_tasks=3)
    assert setting.observation_space.x == env_obs_space
    assert setting.nb_tasks == 3

    train_env = setting.train_dataloader()
    assert train_env.observation_space.x == env_obs_space
    assert train_env.observation_space.task_labels == spaces.Discrete(3)

    n_episodes = 1
    for episode in range(n_episodes):
        obs = train_env.reset()
        done = False
        steps = 0
        while not done and steps < env.max_path_length:
            obs, reward, done, info = train_env.step(train_env.action_space.sample())
            # BUG: Can't render meta-world env when using mtenv.
            train_env.render()
            steps += 1


# @pytest.mark.no_xvfb
# @pytest.mark.xfail(reason="TODO: Rethink how we want to integrate MetaWorld envs.")
@pytest.mark.skip(reason="BUG: timeout handler seems to be bugged, test lasts forever")
@metaworld_required
@pytest.mark.timeout(60)
def test_metaworld_support(config: Config):
    """Test using metaworld benchmarks as the dataset of an RL Setting.

    NOTE: Uses either a MetaWorldEnv instance as the `dataset`, or the env id.
    TODO: Need to rethink this, we should instead use one env class per task (where each
    task env goes through a subset of the tasks for training)
    """

    # TODO: Add option of passing a benchmark instance?
    setting = IncrementalRLSetting(
        dataset="MT10",
        config=config,
        max_episode_steps=10,
        train_max_steps=500,
        test_max_steps=500,
    )
    assert setting.nb_tasks == len(setting.train_envs)
    assert setting.nb_tasks == 10
    assert setting.train_max_steps == 500
    assert setting.test_max_steps == 500
    assert setting.train_steps_per_task == 50
    assert setting.test_steps_per_task == 50

    method = DummyMethod()
    results = setting.apply(method, config=config)
    assert results.summary()


@slow
@metaworld_required
@pytest.mark.timeout(180)
@pytest.mark.parametrize("dataset", ["CW10", "CW20"])
def test_continual_world_support(dataset: str, config: Config):
    """Test using CW10 and CW20 benchmarks as the dataset of an RL Setting.

    TODO: This test is quite long to run, in part because metaworld takes like 20
    seconds to load, and there being 20 tasks in CW20
    """
    # TODO: Add option of passing a benchmark instance? That might make it quicker to
    # run tests?
    setting = IncrementalRLSetting(
        dataset=dataset,
        config=config,
    )
    assert setting.nb_tasks == 10 if dataset == "CW10" else 20
    assert setting.train_steps_per_task == 1_000_000
    assert setting.train_max_steps == 1_000_000 * setting.nb_tasks
    assert setting.test_steps_per_task == 10_000
    assert setting.test_max_steps == 10_000 * setting.nb_tasks

    setting = IncrementalRLSetting(
        dataset=dataset,
        config=config,
        max_episode_steps=10,
        train_steps_per_task=50,
        test_steps_per_task=50,
    )
    assert setting.nb_tasks == 10 if dataset == "CW10" else 20
    assert setting.train_steps_per_task == 50
    assert setting.test_steps_per_task == 50
    assert setting.train_max_steps == setting.train_steps_per_task * setting.nb_tasks
    assert setting.test_steps_per_task == setting.test_steps_per_task
    assert setting.test_max_steps == setting.test_steps_per_task * setting.nb_tasks

    assert (
        setting.nb_tasks
        == len(setting.train_envs)
        == len(setting.val_envs)
        == len(setting.test_envs)
    )
    method = DummyMethod()
    results = setting.apply(method, config=config)
    assert method.train_episodes_per_task == [5 for _ in range(setting.nb_tasks)]
    assert results.summary()


@pytest.mark.xfail(reason="Metaworld integration isn't done yet")
@metaworld_required
@pytest.mark.timeout(120)
@pytest.mark.parametrize("pass_env_id_instead_of_env_instance", [True, False])
def test_metaworld_auto_task_schedule(pass_env_id_instead_of_env_instance: bool):
    """Test that when passing just an env id from metaworld and a number of tasks,
    the task schedule is created automatically.
    """
    import metaworld
    from metaworld import MetaWorldEnv

    benchmark = metaworld.ML10()  # Construct the benchmark, sampling tasks

    env_name = "reach-v2"
    env_type: Type[MetaWorldEnv] = benchmark.train_classes[env_name]
    env = env_type()

    # TODO: When not passing a nb_tasks, the number of available tasks for that env
    # is used.
    # setting = TaskIncrementalRLSetting(
    #     dataset=env_name if pass_env_id_instead_of_env_instance else env,
    #     train_steps_per_task=1000,
    # )
    # assert setting.nb_tasks == 50
    # assert setting.steps_per_task == 1000
    # assert sorted(setting.train_task_schedule.keys()) == list(range(0, 50_000, 1000))

    # Test passing a number of tasks:

    with pytest.warns(RuntimeWarning):
        setting = TaskIncrementalRLSetting(
            dataset=env_name if pass_env_id_instead_of_env_instance else env,
            train_max_steps=2000,
            nb_tasks=2,
            test_max_steps=2000,
            transforms=[],
        )
    assert setting.nb_tasks == 2
    assert setting.steps_per_task == 1000
    assert sorted(setting.train_task_schedule.keys()) == list(range(0, 2000, 1000))
    from sequoia.common.metrics.rl_metrics import EpisodeMetrics

    method = DummyMethod()
    with pytest.warns(RuntimeWarning):
        results: IncrementalRLSetting.Results[EpisodeMetrics] = setting.apply(method)
    # TODO: Don't know if these values make sense! Rewards are super high, not sure if
    # that's normal in Mujoco/metaworld envs:
    # "Average": {
    #     "Episodes": 66,
    #     "Mean reward per episode": 13622.872306005293,
    #     "Mean reward per step": 90.81914870670195
    # }
    # assert 50 < results.average_final_performance.episodes
    # assert 10_000 < results.average_final_performance.mean_reward_per_episode
    # assert 100 < results.average_final_performance.mean_episode_length <= 150


@pytest.mark.xfail(reason="WIP: Adding dm_control support")
def test_dm_control_support():
    import numpy as np
    from dm_control import suite

    # Load one task:
    env = suite.load(domain_name="cartpole", task_name="swingup")

    # Iterate over a task set:
    for domain_name, task_name in suite.BENCHMARKING:
        task_env = suite.load(domain_name, task_name)

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    time_step = env.reset()
    while not time_step.last():
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        time_step = env.step(action)
        print(time_step.reward, time_step.discount, time_step.observation)


# TODO: Use the task schedule as a way to specify how long each task lasts in a
# given env? For instance:


class PeriodTypeEnum(enum.Enum):
    STEPS = enum.auto()
    EPISODES = enum.auto()


class Period(NamedTuple):
    value: int
    type: PeriodTypeEnum = PeriodTypeEnum.STEPS


steps = lambda v: Period(value=v, type=PeriodTypeEnum.STEPS)
episodes = lambda v: Period(value=v, type=PeriodTypeEnum.EPISODES)

train_task_schedule = {
    steps(10): "CartPole-v0",
    episodes(1000): "Breakout-v0",
}


class TestPassingEnvsForEachTask:
    """Tests that have to do with the feature of passing the list of environments to
    use for each task.
    """

    def test_raises_warning_when_envs_have_different_obs_spaces(self):
        task_envs = ["CartPole-v0", "Pendulum-v0"]
        with pytest.warns(RuntimeWarning, match="doesn't have the same observation space"):
            setting = IncrementalRLSetting(train_envs=task_envs)
            setting.train_dataloader()

    def test_passing_envs_for_each_task(self):
        nb_tasks = 3
        gravities = [random.random() * 10 for _ in range(nb_tasks)]

        def make_random_cartpole_env(task_id):
            def _env_fn() -> CartPoleEnv:
                env = gym.make("CartPole-v0")
                env.gravity = gravities[task_id]
                return env

            return _env_fn

        # task_envs = ["CartPole-v0", "CartPole-v1"]
        task_envs = [make_random_cartpole_env(i) for i in range(nb_tasks)]

        setting = IncrementalRLSetting(train_envs=task_envs)
        assert setting.nb_tasks == nb_tasks

        # TODO: Using 'no-op' task schedules, rather than empty ones.
        # This fixes a bug with the creation of the test environment.
        assert not any(setting.train_task_schedule.values())
        assert not any(setting.val_task_schedule.values())
        assert not any(setting.test_task_schedule.values())
        # assert not setting.train_task_schedule
        # assert not setting.val_task_schedule
        # assert not setting.test_task_schedule

        # assert len(setting.train_task_schedule.keys()) == 2

        setting.current_task_id = 0

        train_env = setting.train_dataloader()
        assert train_env.gravity == gravities[0]

        setting.current_task_id = 1

        train_env = setting.train_dataloader()
        assert train_env.gravity == gravities[1]

        assert isinstance(train_env.unwrapped, CartPoleEnv)

        # Not sure, do we want to add a 'observation_spaces`, `action_spaces` and
        # `reward_spaces` properties?
        assert setting.observation_space.x == train_env.observation_space.x
        if setting.task_labels_at_train_time:
            # TODO: Either add a `__getattr__` proxy on the Sparse space, or create
            # dedicated `SparseDiscrete`, `SparseBox` etc spaces so that we eventually
            # get to use `space.n` on a Sparse space.
            assert train_env.observation_space.task_labels == spaces.Discrete(setting.nb_tasks)
            assert (
                setting.observation_space.task_labels.n == train_env.observation_space.task_labels.n
            )

    def test_command_line(self):
        # TODO: If someone passes the same env ids from the command-line, then shouldn't
        # we somehow vary the tasks by changing the level or something?

        setting = IncrementalRLSetting.from_args(argv="--train_envs CartPole-v0 Pendulum-v0")
        assert setting.train_envs == ["CartPole-v0", "Pendulum-v0"]
        # TODO: Not using this:

    def test_raises_warning_when_envs_have_different_obs_spaces(self):
        task_envs = ["CartPole-v0", "Pendulum-v0"]
        with pytest.warns(RuntimeWarning, match="doesn't have the same observation space"):
            setting = IncrementalRLSetting(train_envs=task_envs)
            setting.train_dataloader()

    def test_random_baseline(self):
        nb_tasks = 3
        gravities = [random.random() * 10 for _ in range(nb_tasks)]
        from gym.wrappers import TimeLimit

        def make_random_cartpole_env(task_id):
            def _env_fn() -> CartPoleEnv:
                env = gym.make("CartPole-v0")
                env = TimeLimit(env, max_episode_steps=50)
                env.gravity = gravities[task_id]
                return env

            return _env_fn

        # task_envs = ["CartPole-v0", "CartPole-v1"]
        task_envs = [make_random_cartpole_env(i) for i in range(nb_tasks)]
        setting = IncrementalRLSetting(
            train_envs=task_envs, train_max_steps=1000, test_max_steps=1000
        )
        assert setting.nb_tasks == nb_tasks
        method = RandomBaselineMethod()

        results = setting.apply(method)
        assert results.objective > 0


@pytest.mark.xfail(reason=f"Don't yet fully changing the size of the body parts.")
@mujoco_required
def test_incremental_mujoco_like_LPG_FTW():
    """Trying to get the same-ish setup as the "LPG_FTW" experiments

    See https://github.com/Lifelong-ML/LPG-FTW/tree/master/experiments
    """
    nb_tasks = 5
    from sequoia.settings.rl.envs.mujoco import ContinualHalfCheetahEnv

    task_gravity_factors = [random.random() + 0.5 for _ in range(nb_tasks)]
    task_size_scale_factors = [random.random() + 0.5 for _ in range(nb_tasks)]

    task_envs = [
        RenderEnvWrapper(
            ContinualHalfCheetahEnv(
                gravity=task_gravity_factors[task_id] * -9.81,
                body_name_to_size_scale={"torso": task_size_scale_factors[task_id]},
            ),
        )
        for task_id in range(nb_tasks)
    ]

    setting = IncrementalRLSetting(
        train_envs=task_envs,
        train_steps_per_task=10_000,
        train_wrappers=RenderEnvWrapper,
        test_max_steps=10_000,
    )
    assert setting.nb_tasks == nb_tasks

    # NOTE: Same as above: we use a `no-op` task schedule, rather than an empty one.
    assert not any(setting.train_task_schedule.values())
    assert not any(setting.val_task_schedule.values())
    assert not any(setting.test_task_schedule.values())
    # assert not setting.train_task_schedule
    # assert not setting.val_task_schedule
    # assert not setting.test_task_schedule

    method = RandomBaselineMethod()

    # TODO: Using `render=True` causes a silent crash for some reason!
    results = setting.apply(method)
    assert results.objective > 0
