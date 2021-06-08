import math
import operator
import random
from typing import ClassVar, Tuple, Type

import gym
import numpy as np
import pytest
from gym import spaces
from sequoia.methods import Method
from sequoia.common.config import Config
from sequoia.common.spaces import Image, Sparse
from sequoia.common.transforms import Transforms
from sequoia.conftest import (
    DummyEnvironment,
    metaworld_required,
    monsterkong_required,
    mtenv_required,
    mujoco_required,
    param_requires_atari_py,
    param_requires_mujoco,
)
from sequoia.settings import Setting
from sequoia.settings.rl import TaskIncrementalRLSetting
from ..discrete.setting_test import (
    TestDiscreteTaskAgnosticRLSetting as DiscreteTaskAgnosticRLSettingTests,
)
from sequoia.settings.assumptions.incremental_test import DummyMethod, OtherDummyMethod
from sequoia.utils.utils import take

from .setting import IncrementalRLSetting
from ..discrete.setting_test import make_dataset_fixture


class TestIncrementalRLSetting(DiscreteTaskAgnosticRLSettingTests):

    Setting: ClassVar[Type[Setting]] = IncrementalRLSetting
    dataset: pytest.fixture = make_dataset_fixture(IncrementalRLSetting)

    @pytest.fixture(params=[1, 2])
    def nb_tasks(self, request):
        n = request.param
        return n

    @pytest.fixture()
    def setting_kwargs(self, dataset: str, nb_tasks: int):
        """ Fixture used to pass keyword arguments when creating a Setting. """
        kwargs = {"dataset": dataset, "nb_tasks": nb_tasks, "max_episode_steps": 100}
        if dataset.lower().startswith(
            ("walker2d", "hopper", "halfcheetah", "continual")
        ):
            # kwargs["train_max_steps"] = 5_000
            # kwargs["max_episode_steps"] = 100
            pass
        return kwargs

    def validate_results(
        self,
        setting: IncrementalRLSetting,
        method: Method,
        results: IncrementalRLSetting.Results,
    ) -> None:
        assert results
        assert results.objective
        assert len(results.task_sequence_results) == setting.nb_tasks
        for task_sequence_result in results.task_sequence_results:
            super().validate_results(setting, method, task_sequence_result)
        assert results.average_final_performance == sum(
            results.task_sequence_results[-1].average_metrics_per_task
        )

    def test_on_task_switch_is_called(self):
        setting = self.Setting(
            dataset="CartPole-v0",
            nb_tasks=5,
            # steps_per_task=100,
            train_max_steps=500,
            test_max_steps=500,
            train_transforms=[],
            test_transforms=[],
            val_transforms=[],
        )
        method = DummyMethod()
        _ = setting.apply(method)
        # 5 after learning task 0
        # 5 after learning task 1
        # 5 after learning task 2
        # 5 after learning task 3
        # 5 after learning task 4
        # == 30 task switches in total.
        assert (
            method.n_task_switches == 30
            if not setting.stationary_context
            else 5
            if setting.known_task_boundaries_at_test_time
            else 0
        )
        if setting.task_labels_at_test_time:
            assert method.received_task_ids == [
                0,
                *list(range(5)),
                1,
                *list(range(5)),
                2,
                *list(range(5)),
                3,
                *list(range(5)),
                4,
                *list(range(5)),
            ]
        elif setting.stationary_context:
            assert method.received_task_ids == [None for _ in range(5)]
        else:
            assert method.received_task_ids == [
                0,
                *[None for _ in range(5)],
                1,
                *[None for _ in range(5)],
                2,
                *[None for _ in range(5)],
                3,
                *[None for _ in range(5)],
                4,
                *[None for _ in range(5)],
            ]
        assert (
            method.received_while_training
            == [
                True,
                *[False for _ in range(5)],
                True,
                *[False for _ in range(5)],
                True,
                *[False for _ in range(5)],
                True,
                *[False for _ in range(5)],
                True,
                *[False for _ in range(5)],
            ]
            if not setting.stationary_context
            else [False for _ in range(5)]
        )

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
            # steps_per_task=500,
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
                        obs, reward, done, info = train_env.step(
                            train_env.action_space.sample()
                        )
                    total_steps += 1

            assert total_steps == setting.steps_per_phase

            with pytest.raises(gym.error.ClosedEnvironmentError):
                train_env.reset()

    @monsterkong_required
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("state", [False, True])
    def test_monsterkong(self, state: bool):
        """ Checks that the MonsterKong env works fine with pixel and state input.
        """
        setting = self.Setting(
            dataset="StateMetaMonsterKong-v0" if state else "PixelMetaMonsterKong-v0",
            # force_state_observations=state,
            # force_pixel_observations=(not state),
            nb_tasks=5,
            train_max_steps=500,
            test_max_steps=500,
            # steps_per_task=100,
            # test_steps_per_task=100,
            train_transforms=[],
            test_transforms=[],
            val_transforms=[],
            max_episode_steps=10,
        )

        if state:
            # State-based monsterkong: We observe a flattened version of the game state
            # (20 x 20 grid + player cell and goal cell, IIRC.)
            assert setting.observation_space.x == spaces.Box(0, 292, (402,), np.int16), setting._temp_train_env.observation_space
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
        _ = setting.apply(method)

        assert (
            method.n_task_switches == 30
            if not setting.stationary_context
            else 5
            if setting.known_task_boundaries_at_test_time
            else 0
        )
        if setting.task_labels_at_test_time:
            assert method.received_task_ids == [
                0,
                *list(range(5)),
                1,
                *list(range(5)),
                2,
                *list(range(5)),
                3,
                *list(range(5)),
                4,
                *list(range(5)),
            ]
        elif setting.stationary_context:
            assert method.received_task_ids == [None for _ in range(setting.nb_tasks)]
        else:
            assert method.received_task_ids == [
                0,
                *[None for _ in range(5)],
                1,
                *[None for _ in range(5)],
                2,
                *[None for _ in range(5)],
                3,
                *[None for _ in range(5)],
                4,
                *[None for _ in range(5)],
            ]
        assert (
            method.received_while_training
            == [
                True,
                *[False for _ in range(5)],
                True,
                *[False for _ in range(5)],
                True,
                *[False for _ in range(5)],
                True,
                *[False for _ in range(5)],
                True,
                *[False for _ in range(5)],
            ]
            if not setting.stationary_context
            else [False for _ in range(5)]
        )


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
@metaworld_required
@pytest.mark.timeout(60)
@pytest.mark.parametrize("pass_env_id_instead_of_env_instance", [True, False])
def test_metaworld_support(pass_env_id_instead_of_env_instance: bool):
    """ Test using metaworld environments as the dataset of a Setting.

    NOTE: Uses either a MetaWorldEnv instance as the `dataset`, or the env id.
    """
    import metaworld
    from metaworld import MetaWorldEnv

    benchmark = metaworld.ML10()  # Construct the benchmark, sampling tasks

    env_name = "reach-v2"
    env_type: Type[MetaWorldEnv] = benchmark.train_classes[env_name]
    env = env_type()

    training_tasks = [
        task for task in benchmark.train_tasks if task.env_name == env_name
    ]
    setting = TaskIncrementalRLSetting(
        dataset=env_name if pass_env_id_instead_of_env_instance else env,
        train_task_schedule={
            i: operator.methodcaller("set_task", task)
            for i, task in enumerate(training_tasks)
        },
        steps_per_task=1000,
        transforms=[],
    )
    assert setting.nb_tasks == 50
    assert setting.steps_per_task == 1000
    assert sorted(setting.train_task_schedule.keys()) == list(range(0, 50_000, 1000))

    # TODO: Clear the transforms by default, and add it back if needed?
    assert setting.train_transforms == []
    assert setting.val_transforms == []
    assert setting.test_transforms == []

    assert setting.observation_space.x == env.observation_space

    # Only test out the first 3 tasks for now.
    # TODO: Also try out the valid and test environments.
    for task_id in range(3):
        setting.current_task_id = task_id

        train_env = setting.train_dataloader()
        assert train_env.observation_space.x == env.observation_space
        assert train_env.observation_space.task_labels == spaces.Discrete(
            setting.nb_tasks
        )

        n_episodes = 1
        for episode in range(n_episodes):
            obs = train_env.reset()
            done = False
            steps = 0
            while not done and steps < env.max_path_length:
                obs, reward, done, info = train_env.step(
                    train_env.action_space.sample()
                )
                # train_env.render()
                steps += 1


@metaworld_required
@pytest.mark.timeout(120)
@pytest.mark.parametrize("pass_env_id_instead_of_env_instance", [True, False])
def test_metaworld_auto_task_schedule(pass_env_id_instead_of_env_instance: bool):
    """ Test that when passing just an env id from metaworld and a number of tasks,
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
    #     steps_per_task=1000,
    # )
    # assert setting.nb_tasks == 50
    # assert setting.steps_per_task == 1000
    # assert sorted(setting.train_task_schedule.keys()) == list(range(0, 50_000, 1000))

    # Test passing a number of tasks:

    with pytest.warns(RuntimeWarning):
        setting = TaskIncrementalRLSetting(
            dataset=env_name if pass_env_id_instead_of_env_instance else env,
            steps_per_task=1000,
            nb_tasks=2,
            test_steps_per_task=1000,
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
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape
        )
        time_step = env.step(action)
        print(time_step.reward, time_step.discount, time_step.observation)


import enum
from functools import partial
from typing import NamedTuple

import gym

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

import random
from functools import partial

import gym
from gym.envs.classic_control import CartPoleEnv, PendulumEnv
from sequoia.common.gym_wrappers import RenderEnvWrapper
from sequoia.methods.random_baseline import RandomBaselineMethod


class TestPassingEnvsForEachTask:
    """ Tests that have to do with the feature of passing the list of environments to
    use for each task.
    """

    # @pytest.mark.xfail(
    #     reason="TODO: Check all env spaces to make sure they match, ideally without "
    #     "having to instiate them."
    # )
    def test_raises_error_when_envs_have_different_obs_spaces(self):
        task_envs = ["CartPole-v0", "Pendulum-v0"]
        with pytest.raises(
            RuntimeError, match="doesn't have the same observation space"
        ):
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
        assert not any(setting.valid_task_schedule.values())
        assert not any(setting.test_task_schedule.values())
        # assert not setting.train_task_schedule
        # assert not setting.valid_task_schedule
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
        assert setting.observation_space == train_env.observation_space

    def test_command_line(self):
        # TODO: If someone passes the same env ids from the command-line, then shouldn't
        # we somehow vary the tasks by changing the level or something?

        setting = IncrementalRLSetting.from_args(
            argv="--train_envs CartPole-v0 Pendulum-v0"
        )
        assert setting.train_envs == ["CartPole-v0", "Pendulum-v0"]

        setting = IncrementalRLSetting.from_args(argv="")
        assert setting == IncrementalRLSetting()
        # TODO: Not using this:
        # assert setting.train_envs == [setting.dataset] * setting.nb_tasks

    def test_raises_error_when_envs_have_different_obs_spaces(self):
        task_envs = ["CartPole-v0", "Pendulum-v0"]
        with pytest.raises(
            RuntimeError, match="doesn't have the same observation space"
        ):
            setting = IncrementalRLSetting(train_envs=task_envs)
            setting.train_dataloader()

    def test_random_baseline(self):
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
        setting = IncrementalRLSetting(
            train_envs=task_envs, steps_per_task=10, test_steps=50
        )
        assert setting.nb_tasks == nb_tasks
        method = RandomBaselineMethod()

        results = setting.apply(method)
        assert results.objective > 0


@pytest.mark.xfail(reason=f"Don't yet fully changing the size of the body parts.")
@mujoco_required
def test_incremental_mujoco_like_LPG_FTW():
    """ Trying to get the same-ish setup as the "LPG_FTW" experiments

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
        steps_per_task=10_000,
        train_wrappers=RenderEnvWrapper,
        test_steps=10_000,
    )
    assert setting.nb_tasks == nb_tasks

    # NOTE: Same as above: we use a `no-op` task schedule, rather than an empty one.
    assert not any(setting.train_task_schedule.values())
    assert not any(setting.valid_task_schedule.values())
    assert not any(setting.test_task_schedule.values())
    # assert not setting.train_task_schedule
    # assert not setting.valid_task_schedule
    # assert not setting.test_task_schedule

    method = RandomBaselineMethod()

    # TODO: Using `render=True` causes a silent crash for some reason!
    results = setting.apply(method)
    assert results.objective > 0

