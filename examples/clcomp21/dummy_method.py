import gym
import numpy as np
import tqdm
from sequoia import (
    Actions,
    ClassIncrementalSetting,
    Environment,
    Method,
    Observations,
    PassiveEnvironment,
    Rewards,
)
from torch import Tensor

class DummyMethod(Method, target_setting=ClassIncrementalSetting):
    """ dummy method that does nothing and always returns 0    
    """

    def __init__(self):
        pass

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        pass

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Example train loop.
        You can do whatever you want with train_env and valid_env here.

        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # configure() will have been called by the setting before we get here.
        with tqdm.tqdm(train_env) as train_pbar:
            for i, batch in enumerate(train_pbar):
                if isinstance(batch, Observations):
                    observations, rewards = batch, None
                else:
                    observations, rewards = batch
                    
                batch_size = observations.x.shape[0]

                y_pred = train_env.action_space.sample()

                # If we're at the last batch, it might have a different size, so we give
                # only the required number of values.
                if isinstance(y_pred, (Tensor, np.ndarray)):
                    if y_pred.shape[0] != batch_size:
                        y_pred = y_pred[:batch_size]

                if rewards is None:
                    rewards = train_env.send(y_pred)
                # train as you usually would.

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        y_pred = action_space.sample()
        return y_pred
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
    setting = SettingProxy(ClassIncrementalSetting, "sl_track.yaml")
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
