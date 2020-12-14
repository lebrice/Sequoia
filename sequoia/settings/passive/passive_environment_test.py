from typing import Iterable, Tuple

import gym
import matplotlib.pyplot as plt
import pytest
import torch
from gym import spaces
from torch import Tensor
from torchvision.datasets import MNIST

from sequoia.common.transforms import Compose, Transforms
from .passive_environment import PassiveEnvironment



def test_passive_environment_as_dataloader():
    batch_size = 1
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    env: Iterable[Tuple[Tensor, Tensor]] = PassiveEnvironment(
        dataset,
        batch_size=batch_size,
        n_classes=10,
    )

    for x, y in env:
        assert x.shape == (batch_size, 3, 28, 28)
        x = x.permute(0, 2, 3, 1)
        assert y.tolist() == [5]
        break

        # reward = env.send(4)
        # assert reward is None, reward
        # plt.imshow(x[0])
        # plt.title(f"y: {y[0]}")
        # plt.waitforbuttonpress(10)


def test_mnist_as_gym_env():
    # from continuum.datasets import MNIST
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    
    batch_size = 4
    env = PassiveEnvironment(dataset,
                             n_classes=10,
                             batch_size=batch_size)
    
    assert env.observation_space.shape == (batch_size, 3, 28, 28)
    assert env.action_space.shape == (batch_size,)
    assert env.reward_space.shape == (batch_size,)
    
    env.seed(123)
    obs = env.reset()
    assert obs.shape == (batch_size, 3, 28, 28)

    for i in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs.shape == (batch_size, 3, 28, 28)
        assert reward.shape == (batch_size,)
        assert not done
    env.close()

import numpy as np
from torch.utils.data import Subset


def test_env_gives_done_on_last_item():
    # from continuum.datasets import MNIST
    max_samples = 100
    batch_size = 1
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    dataset = Subset(dataset, list(range(max_samples)))

    env = PassiveEnvironment(dataset,
                             n_classes=10,
                             batch_size=batch_size)

    assert env.observation_space.shape == (batch_size, 3, 28, 28)
    assert env.action_space.shape == (batch_size,)
    assert env.reward_space.shape == (batch_size,)
    
    env.seed(123)
    obs = env.reset()
    assert obs.shape == (batch_size, 3, 28, 28)
    # Starting at 1 since reset() gives one observation already. 
    for i in range(1, max_samples):
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs.shape == (batch_size, 3, 28, 28)
        assert reward.shape == (batch_size,)
        assert done == (i == max_samples-1), i
        if done:
            break
    else:
        assert False, "Should have reached done=True!"
    assert i == max_samples - 1
    env.close()
    

def test_env_done_works_with_batch_size():
    # from continuum.datasets import MNIST
    max_samples = 100
    batch_size = 5
    max_batches = max_samples // batch_size
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    dataset = Subset(dataset, list(range(max_samples)))

    env = PassiveEnvironment(dataset,
                             n_classes=10,
                             batch_size=batch_size)

    assert env.observation_space.shape == (batch_size, 3, 28, 28)
    assert env.action_space.shape == (batch_size,)
    assert env.reward_space.shape == (batch_size,)
    
    env.seed(123)
    obs = env.reset()
    assert obs.shape == (batch_size, 3, 28, 28)
    # Starting at 1 since reset() gives one observation already. 
    for i in range(1, max_batches):
        
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs.shape == (batch_size, 3, 28, 28)
        assert reward.shape == (batch_size,)
        assert done == (i == max_batches-1), i
        if done:
            break
    else:
        assert False, "Should have reached done=True!"
    assert i == max_batches - 1
    env.close()



def test_multiple_epochs_env():
    max_epochs = 3
    max_samples = 100
    batch_size = 5
    max_batches = max_samples // batch_size
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    dataset = Subset(dataset, list(range(max_samples)))

    env = PassiveEnvironment(dataset,
                             n_classes=10,
                             batch_size=batch_size)

    assert env.observation_space.shape == (batch_size, 3, 28, 28)
    assert env.action_space.shape == (batch_size,)
    assert env.reward_space.shape == (batch_size,)
    
    env.seed(123)
    total_steps = 0
    for epoch in range(max_epochs):
        obs = env.reset()
        total_steps += 1      
        
        assert obs.shape == (batch_size, 3, 28, 28)
        # Starting at 1 since reset() gives one observation already. 
        for i in range(1, max_batches):
            obs, reward, done, info = env.step(env.action_space.sample())
            assert obs.shape == (batch_size, 3, 28, 28)
            assert reward.shape == (batch_size,)
            assert done == (i == max_batches-1), i
            total_steps += 1      
            if done:
                break
        else:
            assert False, "Should have reached done=True!"
        assert i == max_batches - 1
    assert total_steps == max_batches * max_epochs

    env.close()



def test_multiple_epochs_dataloader():
    """ Test that we can iterate on the dataloader more than once. """
    max_epochs = 3
    max_samples = 200
    batch_size = 5
    max_batches = max_samples // batch_size
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    dataset = Subset(dataset, list(range(max_samples)))

    env = PassiveEnvironment(dataset,
                             n_classes=10,
                             batch_size=batch_size)

    assert env.observation_space.shape == (batch_size, 3, 28, 28)
    assert env.action_space.shape == (batch_size,)
    assert env.reward_space.shape == (batch_size,)
    total_steps = 0
    for epoch in range(max_epochs):
        for obs, reward in env:
            assert obs.shape == (batch_size, 3, 28, 28)
            assert reward.shape == (batch_size,)
            total_steps += 1      

    assert total_steps == max_batches * max_epochs



def test_multiple_epochs_dataloader_with_split_batch_fn():
    """ Test that we can iterate on the dataloader more than once. """
    max_epochs = 3
    max_samples = 200
    batch_size = 5
    
    def split_batch_fn(batch):
        x, y, = batch
        # some dummy function.
        return torch.zeros_like(x), y
    
    
    max_batches = max_samples // batch_size
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    dataset = Subset(dataset, list(range(max_samples)))

    env = PassiveEnvironment(dataset,
                             n_classes=10,
                             batch_size=batch_size,
                             split_batch_fn=split_batch_fn)

    assert env.observation_space.shape == (batch_size, 3, 28, 28)
    assert env.action_space.shape == (batch_size,)
    assert env.reward_space.shape == (batch_size,)
    total_steps = 0
    for epoch in range(max_epochs):
        for obs, reward in env:
            assert obs.shape == (batch_size, 3, 28, 28)
            assert torch.all(obs == 0)
            assert reward.shape == (batch_size,)
            total_steps += 1      

    assert total_steps == max_batches * max_epochs



def test_env_requires_reset_before_step():
    # from continuum.datasets import MNIST
    max_samples = 100
    batch_size = 5
    max_batches = max_samples // batch_size
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    dataset = Subset(dataset, list(range(max_samples)))

    env = PassiveEnvironment(dataset,
                             n_classes=10,
                             batch_size=batch_size)

    with pytest.raises(gym.error.ResetNeeded):
        env.step(env.action_space.sample())


def test_split_batch_fn():
    # from continuum.datasets import MNIST
    batch_size = 5
    max_batches = 10
    
    def split_batch_fn(batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        x, y, t = batch
        return (x, t), y
    
    # dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    from continuum import ClassIncremental
    from continuum.datasets import MNIST
    from continuum.tasks import split_train_val

    scenario = ClassIncremental(
        MNIST("data", download=True, train=True),
        increment=2,
        transformations=Compose([Transforms.to_tensor, Transforms.three_channels]),
    )
    
    classes_per_task = scenario.nb_classes // scenario.nb_tasks
    print(f"Number of classes per task {classes_per_task}.")
    for i, task_dataset in enumerate(scenario):
        env = PassiveEnvironment(
            task_dataset,
            n_classes=classes_per_task,
            batch_size=batch_size,
            split_batch_fn=split_batch_fn,
            # Need to pass the observation space, in this case.
            observation_space=spaces.Tuple([
                spaces.Box(low=0, high=1, shape=(3, 28, 28)),
                spaces.Discrete(scenario.nb_tasks) # task label
            ]),
            action_space=spaces.Box(
                low=np.array([i * classes_per_task]),
                high=np.array([(i+1) * classes_per_task]),
                dtype=int,
            )
        )
        assert spaces.Box(
            low=np.array([i * classes_per_task]),
            high=np.array([(i+1) * classes_per_task]),
            dtype=int,
        ).shape == (1,)
        assert isinstance(env.observation_space[0], spaces.Box)
        assert env.observation_space[0].shape == (batch_size, 3, 28, 28)
        assert env.observation_space[1].shape == (batch_size,)
        assert env.action_space.shape == (batch_size, 1)
        assert env.reward_space.shape == (batch_size,)
        
        env.seed(123)
        
        obs = env.reset()
        assert len(obs) == 2
        x, t = obs
        assert x.shape == (batch_size, 3, 28, 28)
        assert t.shape == (batch_size,)

        obs, reward, done, info = env.step(env.action_space.sample())
        assert x.shape == (batch_size, 3, 28, 28)
        assert t.shape == (batch_size,)
        assert reward.shape == (batch_size,)
        assert not done

        env.close()
