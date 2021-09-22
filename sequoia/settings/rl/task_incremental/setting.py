from dataclasses import dataclass

from sequoia.utils import constant
from sequoia.settings.base import Rewards, Observations


from ..incremental import IncrementalRLSetting


@dataclass
class TaskIncrementalRLSetting(IncrementalRLSetting):
    """ Continual RL setting with clear task boundaries and task labels.

    The task labels are given at both train and test time.
    """

    task_labels_at_train_time: bool = constant(True)
    task_labels_at_test_time: bool = constant(True)

    # TODO: What is the correct way to do this?
    def reset(self):
        self.num_envs = self.train_env.num_envs
        reset_data = self.train_env.reset()
        return reset_data.x

    # TODO: What is the correct way to do this?
    def step(self, actions):
        new_obs, rewards, dones, infos = self.train_env.step(actions)
        # TODO: Doing rewards and observation transform inline here because some SB3 code that is hit downstream of this expects rewards/obs that are non-wrapped
        # But other parts of code expect it to be wrapped (Sequoia code?)
        rewards = rewards.y if isinstance(rewards, Rewards) else rewards
        new_obs = new_obs.x if isinstance(new_obs, Observations) else new_obs
        return new_obs, rewards, dones, infos
