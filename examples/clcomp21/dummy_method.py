from typing import Optional

import gym
import numpy as np
import tqdm
from torch import Tensor

from sequoia.methods import Method
from sequoia.settings import Actions, Environment, Observations, Setting
from sequoia.settings.sl import SLSetting


class DummyMethod(Method, target_setting=Setting):
    """Dummy method that returns random actions for each observation."""

    def __init__(self):
        self.max_train_episodes: Optional[int] = None

    def configure(self, setting: Setting):
        """Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        if isinstance(setting, SLSetting):
            # Being applied in SL, we will only do one 'epoch" (a.k.a. "episode").
            self.max_train_episodes = 1
        pass

    def fit(self, train_env: Environment, valid_env: Environment):
        """Example train loop.
        You can do whatever you want with train_env and valid_env here.

        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # configure() will have been called by the setting before we get here.
        episodes = 0
        with tqdm.tqdm(desc="training") as train_pbar:

            while not train_env.is_closed():
                for i, batch in enumerate(train_env):
                    if isinstance(batch, Observations):
                        observations, rewards = batch, None
                    else:
                        observations, rewards = batch

                    batch_size = observations.x.shape[0]

                    y_pred = train_env.action_space.sample()

                    # If we're at the last batch, it might have a different size, so w
                    # give only the required number of values.
                    if isinstance(y_pred, (np.ndarray, Tensor)):
                        if y_pred.shape[0] != batch_size:
                            y_pred = y_pred[:batch_size]

                    if rewards is None:
                        rewards = train_env.send(y_pred)

                    train_pbar.set_postfix(
                        {
                            "Episode": episodes,
                            "Step": i,
                        }
                    )
                    # train as you usually would.

                episodes += 1
                if self.max_train_episodes and episodes >= self.max_train_episodes:
                    train_env.close()
                    break

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        """Get a batch of predictions (aka actions) for these observations."""
        y_pred = action_space.sample()
        return self.target_setting.Actions(y_pred)


if __name__ == "__main__":
    from sequoia.common import Config
    from sequoia.settings import ClassIncrementalSetting

    # Create the Method:
    # - Manually:
    method = DummyMethod()

    # NOTE: This Setting is very similar to the one used for the SL track of the
    # competition.
    from sequoia.client import SettingProxy

    setting = SettingProxy(ClassIncrementalSetting, "sl_track")
    # setting = SettingProxy(ClassIncrementalSetting,
    #     dataset="synbols",
    #     nb_tasks=12,
    #     known_task_boundaries_at_test_time=False,
    #     monitor_training_performance=True,
    #     batch_size=32,
    #     num_workers=4,
    # )
    # NOTE: can also use pass a `Config` object to `setting.apply`. This object has some
    # configuration options like device, data_dir, etc.
    results = setting.apply(method, config=Config(data_dir="data"))
    print(results.summary())
