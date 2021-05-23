import itertools

from .setting import DomainIncrementalSLSetting
from sequoia.common.spaces import NamedTupleSpace, Image
from gym.spaces import Discrete
import numpy as np


def test_domain_incremental_mnist_setup():
    setting = DomainIncrementalSLSetting(dataset="mnist", increment=2,)
    setting.prepare_data(data_dir="data")
    setting.setup()
    assert setting.observation_space == NamedTupleSpace(
        x=Image(0.0, 1.0, (3, 28, 28), np.float32), task_labels=Discrete(5)
    )

    for i in range(setting.nb_tasks):
        setting.current_task_id = i
        batch_size = 5
        train_loader = setting.train_dataloader(batch_size=batch_size)

        for j, (observations, rewards) in enumerate(
            itertools.islice(train_loader, 100)
        ):
            x = observations.x
            t = observations.task_labels
            y = rewards.y
            print(i, j, y, t)
            assert x.shape == (batch_size, 3, 28, 28)
            assert ((0 <= y) & (y < setting.n_classes_per_task)).all()
            assert all(t == i)
            x = x.permute(0, 2, 3, 1)[0]
            assert x.shape == (28, 28, 3)

            rewards_ = train_loader.send([4 for _ in range(batch_size)])
            assert (rewards.y == rewards_.y).all()

        train_loader.close()

        test_loader = setting.test_dataloader(batch_size=batch_size)
        for j, (observations, rewards) in enumerate(
            itertools.islice(test_loader, 100)
        ):
            assert rewards is None
            
            x = observations.x
            t = observations.task_labels
            assert t is None
            assert x.shape == (batch_size, 3, 28, 28)
            x = x.permute(0, 2, 3, 1)[0]
            assert x.shape == (28, 28, 3)

            rewards = test_loader.send([4 for _ in range(batch_size)])
            assert rewards is not None
            y = rewards.y
            assert ((0 <= y) & (y < setting.n_classes_per_task)).all()

