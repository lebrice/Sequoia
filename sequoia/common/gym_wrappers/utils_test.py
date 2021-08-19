from typing import Iterable, Tuple, Callable

import gym
import pytest
import torch
from gym.wrappers import ClipAction
from gym.wrappers.pixel_observation import PixelObservationWrapper
from torch import Tensor
from torch.utils.data import TensorDataset

from sequoia.common.spaces import TensorBox, TensorDiscrete, TypedDictSpace
from sequoia.settings.sl import SLEnvironment
from sequoia.settings.sl.continual import Actions, Observations, Rewards

from .pixel_observation import PixelObservationWrapper
from .utils import has_wrapper, IterableWrapper
from .transform_wrappers import TransformReward


@pytest.mark.parametrize(
    "env,wrapper_type,result",
    [
        (lambda: PixelObservationWrapper(gym.make("CartPole-v0")), ClipAction, False),
        (
            lambda: PixelObservationWrapper(gym.make("CartPole-v0")),
            PixelObservationWrapper,
            True,
        ),
        (
            lambda: PixelObservationWrapper(gym.make("CartPole-v0")),
            PixelObservationWrapper,
            True,
        ),
        # param_requires_atari_py(AtariPreprocessing(gym.make("Breakout-v0")), ClipAction, True),
    ],
)
def test_has_wrapper(env, wrapper_type, result):
    assert has_wrapper(env(), wrapper_type) == result


class TestIterableWrapper:
    def sl_env(
        self,
        task_id: int = 0,
        n_samples: int = 100,
        batch_size: int = 10,
        pretend_to_be_active: bool = False,
    ):
        x_shape = (1,)
        dataset = TensorDataset(
            torch.ones([n_samples, *x_shape], dtype=torch.uint8) * task_id,  # x
            torch.ones((n_samples,), dtype=torch.long) * task_id,  # task labels
            torch.ones((n_samples,), dtype=torch.long) * task_id,  # y
        )

        def split_batch_fn(batch) -> Tuple[Observations, Rewards]:
            x, t, y = batch
            return (
                Observations(x=x, task_labels=t),
                Rewards(y=y),
            )

        env: Iterable[Tuple[Tensor, Tensor]] = SLEnvironment(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            split_batch_fn=split_batch_fn,
            observation_space=TypedDictSpace(
                x=TensorBox(0, 1, x_shape, dtype=torch.uint8),
                task_labels=TensorDiscrete(5),
                dtype=Observations,
            ),
            action_space=TypedDictSpace(y_pred=TensorDiscrete(10), dtype=Actions,),
            reward_space=TypedDictSpace(y=TensorDiscrete(10), dtype=Rewards,),
            pretend_to_be_active=pretend_to_be_active,
        )
        return env

    @pytest.mark.parametrize("active", [True, False])
    def test_reward_isnt_applied_twice_when_iterating_passive_env(self, active: bool):
        from dataclasses import replace
        
        class DoubleRewardsWrapper(TransformReward):
            def __init__(self, env: IterableWrapper, f: Callable = None):
                super().__init__(env, f=self.double_rewards)

            @staticmethod
            def double_rewards(rewards: Rewards) -> Rewards:
                return replace(rewards, y=rewards.y * 2)

        env = self.sl_env(
            task_id=1, n_samples=100, batch_size=10, pretend_to_be_active=active
        )
        env = DoubleRewardsWrapper(env)

        obs = env.reset()
        done = False
        while not done:
            obs, rewards, done, info = env.step(env.action_space.sample())
            assert (rewards.y == 2).all()

        env = self.sl_env(
            task_id=1, n_samples=100, batch_size=10, pretend_to_be_active=active
        )
        env = DoubleRewardsWrapper(env)

        for obs, rewards in env:
            assert (rewards is None) == active
            if rewards is not None:
                assert (rewards.y == 2).all()
            rewards = env.send(env.action_space.sample())
            assert (rewards.y == 2).all()

