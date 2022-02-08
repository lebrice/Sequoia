import itertools
import math
from typing import Dict

from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.settings.assumptions.discrete_results import TaskSequenceResults
from sequoia.settings.assumptions.iid_results import TaskResults

from ..continual.test_environment import ContinualRLTestEnvironment


class DiscreteTaskAgnosticRLTestEnvironment(ContinualRLTestEnvironment):
    def __init__(self, *args, task_schedule: Dict, **kwargs):
        super().__init__(*args, task_schedule=task_schedule, **kwargs)
        self.task_schedule = task_schedule
        self.boundary_steps = [step // (self.batch_size or 1) for step in self.task_schedule.keys()]
        # TODO: Removing the last entry since it's the terminal state.
        self.boundary_steps.pop(-1)

    def __len__(self):
        return math.ceil(self.step_limit / (getattr(self.env, "batch_size", 1) or 1))

    def get_results(self) -> TaskSequenceResults[EpisodeMetrics]:
        # TODO: Place the metrics in the right 'bin' at the end of each episode during
        # testing depending on the task at that time, rather than what's happening here,
        # where we're getting all the rewards and episode lengths at the end and then
        # sort it out into the bins based on the task schedule. ALSO: this would make it
        # easier to support monitoring batched RL environments, since these `Monitor`
        # methods (get_episode_rewards, get_episode_lengths, etc) assume the environment
        # isn't batched.
        rewards = self.get_episode_rewards()
        lengths = self.get_episode_lengths()

        task_schedule: Dict[int, Dict] = self.task_schedule
        task_steps = sorted(task_schedule.keys())
        # TODO: Removing the last entry since it's the terminal state.
        task_steps.pop(-1)

        assert 0 in task_steps
        import bisect

        nb_tasks = len(task_steps)
        assert nb_tasks >= 1

        test_results = TaskSequenceResults([TaskResults() for _ in range(nb_tasks)])
        # TODO: Fix this, since the task id might not be related to the steps!
        for step, episode_reward, episode_length in zip(
            itertools.accumulate(lengths), rewards, lengths
        ):
            # Given the step, find the task id.
            task_id = bisect.bisect_right(task_steps, step) - 1

            episode_metric = EpisodeMetrics(
                n_samples=1,
                mean_episode_reward=episode_reward,
                mean_episode_length=episode_length,
            )

            test_results.task_results[task_id].metrics.append(episode_metric)

        return test_results

    def render(self, mode="human", **kwargs):
        # TODO: This might not be setup right. Need to check.
        image_batch = super().render(mode=mode, **kwargs)
        if mode == "rgb_array" and self.batch_size:
            return tile_images(image_batch)
        return image_batch

    def _after_reset(self, observation):
        # Is this going to work fine when the observations are batched though?
        return super()._after_reset(observation)
