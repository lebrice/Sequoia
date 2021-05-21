import math
import operator
from typing import Tuple, Type

import gym
import numpy as np
import pytest
from gym import spaces
from sequoia.common.config import Config
from sequoia.common.spaces import Image, Sparse
from sequoia.common.transforms import Transforms
from sequoia.conftest import (
    DummyEnvironment,
    metaworld_required,
    monsterkong_required,
    mtenv_required,
    param_requires_atari_py,
    param_requires_mujoco
)
from sequoia.settings.active import TaskIncrementalRLSetting
from sequoia.settings.assumptions.incremental_test import DummyMethod, OtherDummyMethod
from sequoia.utils.utils import take

from .incremental_rl_setting import IncrementalRLSetting


def test_number_of_tasks():
    setting = IncrementalRLSetting(
        dataset="CartPole-v0",
        observe_state_directly=True,
        monitor_training_performance=True,
        steps_per_task=1000,
        max_steps=10_000,
        test_steps=1000,
    )
    assert setting.nb_tasks == 10


def test_max_number_of_steps_per_task_is_respected():
    setting = IncrementalRLSetting(
        dataset="CartPole-v0",
        observe_state_directly=True,
        monitor_training_performance=True,
        steps_per_task=500,
        max_steps=1000,
        test_steps=1000,
    )
    train_env = setting.train_dataloader()
    total_steps = 0

    while not total_steps > 1000:
        obs = train_env.reset()
        done = False
        while not done:
            if total_steps == setting.steps_per_task:
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


@pytest.mark.timeout(60)
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize(
    "dataset, expected_obs_shape",
    [
        ("CartPole-v0", (3, 400, 600)),
        # param_requires_atari_py("Breakout-v0", (3, 210, 160)),
        param_requires_atari_py(
            "Breakout-v0", (1, 84, 84)
        ),  # Since the Atari Preprocessing is added by default.
        # TODO: Add support for the duckietown env!
        # ("duckietown", (120, 160, 3)),
        param_requires_mujoco(
            "half_cheetah", (17,)
        )
    ],
)
def test_check_iterate_and_step(
    dataset: str, expected_obs_shape: Tuple[int, ...], batch_size: int
):
    # TODO: Fix the default transforms, shouldn't necessarily have `to_tensor` in there.
    setting = IncrementalRLSetting(
        dataset=dataset,
        nb_tasks=5,
        train_transforms=[Transforms.to_tensor],
        val_transforms=[Transforms.to_tensor],
        test_transforms=[Transforms.to_tensor],
    )
    assert setting.train_transforms == [Transforms.to_tensor]
    assert setting.val_transforms == [Transforms.to_tensor]
    assert setting.test_transforms == [Transforms.to_tensor]
    # TODO: Interesting issue: can't pickle only the to_tensor transform, as it modifies
    # the given class in-place?

    assert len(setting.train_task_schedule) == 5
    assert not setting.smooth_task_boundaries
    assert setting.task_labels_at_train_time

    # TODO: Should we have the task label space in this case?
    assert setting.task_labels_at_train_time
    assert not setting.task_labels_at_test_time

    if batch_size is None:
        expected_obs_batch_shape = expected_obs_shape
    else:
        expected_obs_batch_shape = (batch_size, *expected_obs_shape)

    with setting.train_dataloader(batch_size=batch_size) as temp_env:
        obs_space = temp_env.observation_space
        assert obs_space[0] == spaces.Box(
            0.0, 1.0, expected_obs_batch_shape, dtype=np.float32
        )
        assert (
            obs_space[1] == spaces.MultiDiscrete([5] * batch_size)
            if batch_size
            else spaces.Discrete(5)
        )

    with setting.val_dataloader(batch_size=batch_size) as temp_env:
        # No task labels:
        obs_space = temp_env.observation_space

        assert obs_space[0] == spaces.Box(
            0.0, 1.0, expected_obs_batch_shape, dtype=np.float32
        )
        if batch_size:
            assert str(obs_space[1]) == str(spaces.MultiDiscrete([5] * batch_size))
        else:
            # TODO: Should the task labels be given in the valid dataloader if they
            # arent' during testing?
            assert obs_space[1] == spaces.Discrete(5)

    # NOTE: Limitting the batch size at test time to None (i.e. a single env)
    # because of how the Monitor class works atm.

    with setting.test_dataloader(batch_size=None) as temp_env:
        obs_space = temp_env.observation_space
        assert obs_space[1] == Sparse(spaces.Discrete(5), sparsity=1.0)

    def check_obs(obs, task_label: int = None):
        if batch_size is None:
            assert obs[1] == task_label
        else:
            assert isinstance(obs, IncrementalRLSetting.Observations), obs[0].shape
            assert obs.task_labels is task_label or all(
                task_label == task_label for task_label in obs.task_labels
            )

    env = setting.train_dataloader(batch_size=batch_size)
    reset_obs = env.reset()
    check_obs(reset_obs, task_label=0)

    for i in range(5):
        step_obs, *_ = env.step(env.action_space.sample())
        check_obs(step_obs, task_label=0)

    for iter_obs in take(env, 3):
        check_obs(iter_obs, task_label=0)
        _ = env.send(env.action_space.sample())
        env.render("human")

    env.close()


def test_on_task_switch_is_called_incremental_rl():
    setting = IncrementalRLSetting(
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
    # 5 after learning task 0
    # 5 after learning task 1
    # 5 after learning task 2
    # 5 after learning task 3
    # 5 after learning task 4
    # == 30 task switches in total.
    assert method.n_task_switches == 30
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
    assert method.received_while_training == [
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


def test_on_task_switch_is_called_task_incremental_rl():
    setting = IncrementalRLSetting(
        dataset=DummyEnvironment,
        nb_tasks=5,
        steps_per_task=100,
        test_steps_per_task=100,
        max_steps=500,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
        task_labels_at_test_time=True,
    )
    method = DummyMethod()
    _ = setting.apply(method)
    assert method.n_task_switches == 30
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
    assert method.received_while_training == [
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


@pytest.mark.timeout(120)
@monsterkong_required
@pytest.mark.parametrize("task_labels_at_test_time", [False, True])
@pytest.mark.parametrize("state", [False, True])
def test_monsterkong(task_labels_at_test_time: bool, state: bool):
    """ checks that the MonsterKong env works fine with monsterkong and state input. """
    setting = IncrementalRLSetting(
        dataset="monsterkong",
        observe_state_directly=state,
        nb_tasks=5,
        steps_per_task=100,
        test_steps_per_task=100,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
        task_labels_at_test_time=task_labels_at_test_time,
        max_episode_steps=10,
    )

    if state:
        # State-based monsterkong: We observe a flattened version of the game state
        # (20 x 20 grid + player cell and goal cell, IIRC.)
        assert setting.observation_space.x == spaces.Box(0, 292, (402,), np.int16)
    else:
        assert setting.observation_space.x == Image(0, 255, (64, 64, 3), np.uint8)

    if task_labels_at_test_time:
        assert setting.observation_space.task_labels == spaces.Discrete(5)
    else:
        assert setting.observation_space.task_labels == Sparse(
            spaces.Discrete(5), sparsity=0.0
        )

    assert setting.test_steps == 500
    with setting.train_dataloader() as env:
        obs = env.reset()
        assert obs in setting.observation_space

    method = DummyMethod()
    _ = setting.apply(method)

    assert method.n_task_switches == 30
    if task_labels_at_test_time:
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
    assert method.received_while_training == [
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


@pytest.mark.timeout(120)
def test_action_space_always_matches_obs_batch_size_in_RL(config: Config):
    """ """
    from sequoia.settings import TaskIncrementalRLSetting

    nb_tasks = 2
    batch_size = 1
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=nb_tasks,
        batch_size=batch_size,
        steps_per_task=100,
        test_steps_per_task=100,
        num_workers=4,  # Intentionally wrong
        # monitor_training_performance=True, # This is still a TODO in RL.
    )
    # 500 "examples" in the test dataloader, since 5 * 100 steps per task..
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

    env_name = "reach-v1"
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

    env_name = "reach-v1"
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


from functools import partial
import gym

import enum

# TODO: Use the task schedule as a way to specify how long each task lasts in a
# given env? For instance:

from typing import NamedTuple


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

from sequoia.methods.random_baseline import RandomBaselineMethod
from functools import partial
import gym
from gym.envs.classic_control import CartPoleEnv, PendulumEnv
import random

from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.common.gym_wrappers import RenderEnvWrapper


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


@pytest.mark.no_xvfb
def test_incremental_mujoco_like_LPG_FTW():
    """ Trying to get the same-ish setup as the "LPG_FTW" experiments

    See https://github.com/Lifelong-ML/LPG-FTW/tree/master/experiments
    """
    import random

    nb_tasks = 5
    from contextlib import suppress

    with suppress(SystemExit):
        from gym_extensions.continuous.mujoco.modified_half_cheetah import (
            HalfCheetahGravityEnv,
        )
    
    
    
    task_gravity_factors = [random.random() * +0.5 for _ in range(nb_tasks)]
    
    task_envs = [
        RenderEnvWrapper(
            HalfCheetahGravityEnv(
                max_episode_steps=1000,
                reward_threshold=3800.0,
                kwargs=dict(gravity=task_gravity_factor * -9.81),
            )
        )
        for task_id, task_gravity_factor in enumerate(task_gravity_factors)
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
    assert False, results
