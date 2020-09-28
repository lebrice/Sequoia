from typing import Tuple, Iterable

from torch import Tensor
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

from common.transforms import Compose, Transforms

from .passive_environment import PassiveEnvironment


def test_passive_mnist_environment():
    dataset = MNIST("data", transform=Compose([Transforms.to_tensor, Transforms.fix_channels]))
    env: Iterable[Tuple[Tensor, Tensor]] = PassiveEnvironment(dataset)

    for x, y in env:
        print(x.shape, type(x), y)
        assert x.shape == (1, 3, 28, 28)
        x = x.permute(0, 2, 3, 1)
        assert y.item() == 5

        reward = env.send(4)
        assert reward is None, reward
        # plt.imshow(x[0])
        # plt.title(f"y: {y[0]}")
        # plt.waitforbuttonpress(10)
        break
