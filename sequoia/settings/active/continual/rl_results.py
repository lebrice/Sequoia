import json
from io import StringIO
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

import numpy as np
from simple_parsing import list_field

from sequoia.common import ClassificationMetrics, Metrics, RegressionMetrics
from sequoia.settings.assumptions.incremental import IncrementalSetting
from sequoia.settings.base import Results
from sequoia.utils import mean
from sequoia.utils.plotting import autolabel, plt


@dataclass
class RLResults(IncrementalSetting.Results, Results):
    """ Results for RL settings.

    Similar to ClassIncrementalResults, but here the metrics would be the mean
    reward, something like that.

    TODO: Use the metrics from common/metrics/rl_metrics instead of
    RegressionMetrics.
    TODO: Actually implement this, making sure that the metrics/plots this
    creates make sense.
    """
    objective_name: ClassVar[str] = "Mean Reward"
      
    episode_rewards: List[List[float]] = list_field(repr=False)
    episode_lengths: List[List[int]] = list_field(repr=False)
    test_metrics: List[List[Metrics]] = list_field(repr=False)

    def __post_init__(self):
        if not self.test_metrics:
            self.test_metrics = []
            # TODO: Add some kind of 'episode/RL'-specific Metrics class?
            for task_episode_rewards, task_episode_lengths in zip(self.episode_rewards, self.episode_lengths):
                task_metrics = [
                    RegressionMetrics(mse=episode_total_reward/episode_length)
                    for episode_total_reward, episode_length in zip(task_episode_rewards, task_episode_lengths)
                ]
                self.test_metrics.append(task_metrics)
        self.test_metrics = list(filter(len, self.test_metrics))

    @property
    def objective(self) -> float:
        return self.mean_reward
    
    @property
    def mean_reward(self):
        average_metric = self.average_metrics
        return average_metric.mse

    @property
    def mean_episode_length(self) -> int:
        all_episode_lengths = sum(self.episode_lengths, [])
        return np.mean(all_episode_lengths)

    @property
    def total_steps(self):
        return sum(map(sum, self.episode_lengths))

    @property
    def total_reward(self) -> float:
        return sum(map(sum, self.episode_rewards))
    
    def to_log_dict(self) -> Dict:        
        # TODO: Create a dict of useful things to log.
        log_dict = {}
        for task in range(self.num_tasks):
            steps = sum(self.episode_lengths[task])
            episodes = len(self.episode_lengths[task])
            task_log_dict = {
                "Steps": int(steps),
                "Episodes": int(episodes),
                "Total reward": float(sum(self.episode_rewards[task])),
                "Mean reward / step": float(self.average_metrics_per_task[task].mse),
            }
            mean_episode_length = 0
            if episodes:
                mean_episode_length = float(np.mean(self.episode_lengths[task]))
            task_log_dict["Mean episode length"] = mean_episode_length
            log_dict[str(task)] = task_log_dict
        return log_dict

    def summary(self):
        s = StringIO()
        for task, log_dict in self.to_log_dict().items():
            print(f"Task {task}:", json.dumps(log_dict, indent="\t"), file=s)
        s.seek(0)
        return s.read()

    def make_plots(self):
        results = {
            "mean_reward": self.mean_reward_plot()
        }
        return results

    def mean_reward_plot(self):
        raise NotImplementedError("TODO")
        figure: plt.Figure
        axes: plt.Axes
        figure, axes = plt.subplots()
        x = list(range(self.num_tasks))
        y = [metrics.accuracy for metrics in self.average_metrics_per_task]
        rects = axes.bar(x, y)
        axes.set_title("Task Accuracy")
        axes.set_xlabel("Task")
        axes.set_ylabel("Accuracy")
        axes.set_ylim(0, 1.0)
        autolabel(axes, rects)
        return figure
