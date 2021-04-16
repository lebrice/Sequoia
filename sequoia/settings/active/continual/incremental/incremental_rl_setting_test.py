import math
from typing import Callable, List, Optional, Tuple

import gym
import numpy as np
import pytest
from gym import spaces
from sequoia.common.config import Config
from sequoia.common.spaces import Image, Sparse
from sequoia.common.transforms import ChannelsFirstIfNeeded, ToTensor, Transforms
from sequoia.conftest import (
    xfail_param,
    monsterkong_required,
    param_requires_atari_py,
    metaworld_required,
)
from sequoia.settings import Method
from sequoia.settings.assumptions.incremental import TestEnvironment
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
    ],
)
def test_check_iterate_and_step(
    dataset: str, expected_obs_shape: Tuple[int, ...], batch_size: int
):
    setting = IncrementalRLSetting(
        dataset=dataset, nb_tasks=5, transforms=[Transforms.to_tensor]
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
            # assert str(obs_space[1]) == str(spaces.Tuple([Sparse(spaces.Discrete(5), sparsity=1.) for _ in range(batch_size)]))
        else:
            # TODO: Should the task labels be given in the valid dataloader if they arent' during testing?
            assert obs_space[1] == spaces.Discrete(5)
            # assert obs_space[1] == Sparse(spaces.Discrete(5), sparsity=1.)

    # NOTE: Limitting the batch size at test time to None (i.e. a single env)
    # because of how the Monitor class works atm.

    with setting.test_dataloader(batch_size=None) as temp_env:
        obs_space = temp_env.observation_space
        assert obs_space[1] == Sparse(spaces.Discrete(5), sparsity=1.0)
        # No task labels:
        # if batch_size:
        #     assert str(obs_space[1]) == str(spaces.Tuple([Sparse(spaces.Discrete(5), sparsity=1.) for _ in range(batch_size)]))

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
        reward = env.send(env.action_space.sample())
        env.render("human")

    env.close()


from sequoia.settings.assumptions.incremental_test import DummyMethod
from sequoia.conftest import DummyEnvironment


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
    results = setting.apply(method)
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
    results = setting.apply(method)
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
def test_monsterkong_state(task_labels_at_test_time: bool, state: bool):
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
    from sequoia.common.spaces import NamedTupleSpace, Image

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
    results = setting.apply(method)

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
@monsterkong_required
@pytest.mark.parametrize("task_labels_at_test_time", [False, True])
def test_monsterkong_pixels(task_labels_at_test_time: bool):
    """ checks that the MonsterKong env works fine with monsterkong and state input. """
    setting = IncrementalRLSetting(
        dataset="monsterkong",
        observe_state_directly=False,
        nb_tasks=5,
        steps_per_task=100,
        test_steps_per_task=100,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
        task_labels_at_test_time=task_labels_at_test_time,
        max_episode_steps=10,
    )
    assert setting.test_steps == 500
    assert setting.observation_space.x == Image(0, 255, (64, 64, 3), np.uint8)
    with setting.train_dataloader() as env:
        obs = env.reset()
        assert obs in setting.observation_space

    method = DummyMethod()
    results = setting.apply(method)

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


from gym import Space, spaces
from gym.vector.utils.spaces import batch_space
from sequoia.methods import Method
from sequoia.settings import Actions, Environment, Observations, Setting


class OtherDummyMethod(Method, target_setting=Setting):
    def __init__(self):
        self.batch_sizes: List[int] = []

    def fit(self, train_env: Environment, valid_env: Environment):
        for i, batch in enumerate(train_env):
            if isinstance(batch, Observations):
                observations, rewards = batch, None
            else:
                assert isinstance(batch, tuple) and len(batch) == 2
                observations, rewards = batch

            y_preds = train_env.action_space.sample()
            if rewards is None:
                action_space = train_env.action_space
                if train_env.action_space.shape:
                    obs_batch_size = observations.x.shape[0]
                    # BUG: Fix the `batch_size` attribute on `Batch` so it works
                    # even when task labels are None, by checking wether there is
                    # one or more shapes, and then if there are, then that the first
                    # dimension match between those.
                    action_space_batch_size = action_space.shape[0]
                    if obs_batch_size != action_space_batch_size:
                        action_space = batch_space(
                            train_env.single_action_space, obs_batch_size
                        )

                rewards = train_env.send(Actions(action_space.sample()))

    def get_actions(self, observations: Observations, action_space: Space) -> Actions:
        # This won't work on weirder spaces.
        if action_space.shape:
            assert observations.x.shape[0] == action_space.shape[0]
        if getattr(observations.x, "shape", None):
            batch_size = 1
            if observations.x.ndim > 1:
                batch_size = observations.x.shape[0]
            self.batch_sizes.append(batch_size)
        else:
            self.batch_sizes.append(0)  # X isn't batched.
        return action_space.sample()


def test_action_space_always_matches_obs_batch_size(config: Config):
    """ Make sure that the batch size in the observations always matches the action
    space provided to the `get_actions` method.
    
    ALSO:
    - Make sure that we get asked for actions for all the observations in the test set,
      even when there is a shorter last batch.
    - The total number of observations match the dataset size.
    """
    nb_tasks = 5
    batch_size = 128
    from sequoia.settings import TaskIncrementalSetting

    setting = TaskIncrementalSetting(
        dataset="mnist",
        nb_tasks=nb_tasks,
        batch_size=batch_size,
        num_workers=4,
        monitor_training_performance=True,
    )

    # 10_000 examples in the test dataset of mnist.
    total_samples = len(setting.test_dataloader().dataset)

    method = OtherDummyMethod()
    results = setting.apply(method, config=config)

    # Multiply by nb_tasks because the test loop is ran after each training task.
    assert sum(method.batch_sizes) == total_samples * nb_tasks
    assert len(method.batch_sizes) == math.ceil(total_samples / batch_size) * nb_tasks
    assert set(method.batch_sizes) == {batch_size, total_samples % batch_size}


@pytest.mark.timetout(60)
def test_action_space_always_matches_obs_batch_size_in_RL(config: Config):
    """ Same test as above, but in RL. """
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
    results = setting.apply(method, config=config)

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


from sequoia.conftest import mtenv_required


@mtenv_required
@pytest.mark.xfail(reason="don't know how to get the max path length through mtenv!")
def test_mtenv_meta_world_support():
    from mtenv import make, MTEnv
    from mtenv.envs.metaworld.env import MetaWorldMTWrapper

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
    # 'MetaWorldMTWrapper' object has no attribute 'max_path_length'
    assert False, env.max_path_length

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        # BUG: Can't render when using metaworld through mtenv, since mtenv *contains* a
        # straight-up copy-pasted old version of meta-world, which doesn't support it.
        env.render()
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


@metaworld_required
def test_metaworld_support():
    import metaworld
    import random
    from typing import Type
    from metaworld import MetaWorldEnv
    from sequoia.settings.active import TaskIncrementalRLSetting

    benchmark = metaworld.ML10()  # Construct the benchmark, sampling tasks

    env_name = "reach-v1"
    env_type: Type[MetaWorldEnv] = benchmark.train_classes[env_name]
    env = env_type()

    import operator

    training_tasks = [
        task for task in benchmark.train_tasks if task.env_name == env_name
    ]
    setting = TaskIncrementalRLSetting(
        dataset=env,
        train_task_schedule={
            i: operator.methodcaller("set_task", task)
            for i, task in enumerate(training_tasks)
        },
        steps_per_task=1000,
    )
    assert setting.nb_tasks == 50
    assert setting.steps_per_task == 1000
    assert sorted(setting.train_task_schedule.keys()) == list(range(0, 50_000, 1000))

    # TODO: Clear the transforms by default, and add it back if needed?
    assert setting.train_transforms == []
    assert setting.val_transforms == []
    assert setting.test_transforms == []

    assert setting.observation_space.x == env.observation_space

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
                # BUG: Can't render meta-world env when using mtenv.
                train_env.render()
                steps += 1


@pytest.mark.xfail(reason="WIP: Adding dm_control support")
def test_dm_control_support():
    from dm_control import suite
    import numpy as np

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

