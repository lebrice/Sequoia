from typing import Tuple, Iterable

from torch import Tensor
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

from common.transforms import Compose, Transforms

from .passive_environment import PassiveEnvironment


def test_passive_environment_as_dataloader():
    # from continuum.datasets import MNIST
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    env: Iterable[Tuple[Tensor, Tensor]] = PassiveEnvironment(dataset, n_classes=10)

    for x, y in env:
        assert x.shape == (1, 3, 28, 28)
        x = x.permute(0, 2, 3, 1)
        assert y.item() == 5

        # reward = env.send(4)
        # assert reward is None, reward
        # plt.imshow(x[0])
        # plt.title(f"y: {y[0]}")
        # plt.waitforbuttonpress(10)
        break


def test_mnist_as_gym_env():
    # from continuum.datasets import MNIST
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.three_channels]))
    
    batch_size = 4
    env = PassiveEnvironment(dataset,
                             n_classes=10,
                             batch_size=batch_size)
    
    assert env.observation_space[0].shape == (3, 28, 28)
    assert env.action_space[0].shape == ()
    assert env.reward_space[0].shape == ()
    
    env.seed(123)
    obs = env.reset()
    assert obs.shape == (batch_size, 3, 28, 28)

    for i in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs.shape == (batch_size, 3, 28, 28)
        assert reward.shape == (batch_size,)
        assert not done
    env.close()
    