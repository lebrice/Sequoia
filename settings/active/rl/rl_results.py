from settings.base import Results
from dataclasses import dataclass, field
from settings.assumptions.incremental import IncrementalSetting
from utils import mean
from common import Metrics, RegressionMetrics, ClassificationMetrics
from typing import List, ClassVar, Dict
from utils.plotting import autolabel, plt
from simple_parsing import list_field
import numpy as np
# @dataclass
# class EpisodeMetrics(Metrics):
#     rewards: List[float]
#     length: int


@dataclass
class RLResults(IncrementalSetting.Results, Results):
    """ Results for RL settings.

    Similar to ClassIncrementalResults, but here the metrics would be the mean
    reward, something like that.
    
    TODO: Actually implement this, making sure that the metrics/plots this creates
    make sense.
    """
    episode_rewards: List[List[float]] = list_field()
    episode_lengths: List[List[int]] = list_field()
    test_metrics: List[List[Metrics]] = list_field()
    objective_name: ClassVar[str] = "Mean Reward"      

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
    def total_steps(self):
        return sum(map(sum, self.episode_lengths))

    def total_reward(self) -> float:
        return sum(map(sum, self.episode_rewards))

    def summary(self):
        lines = [
            f"Total reward: {self.total_reward}"
            f"Mean reward: {self.mean_reward}"
        ]        
        return "\n".join(lines)
               
    
    def make_plots(self):
        results = {
            "mean_reward": self.mean_reward_plot()
        }
        return results

    def mean_reward_plot(self):
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

    def to_log_dict(self) -> Dict[str, float]:
        results = {}
        results[self.objective_name] = self.objective
        return results
        average_metrics = self.average_metrics

        if isinstance(average_metrics, ClassificationMetrics):
            results["accuracy/average"] = average_metrics.accuracy
        elif isinstance(average_metrics, RegressionMetrics):
            results["mse/average"] = average_metrics.mse
        else:
            results["average metrics"] = average_metrics

        for i, average_task_metrics in enumerate(self.average_metrics_per_task):
            if isinstance(average_task_metrics, ClassificationMetrics):
                results[f"accuracy/task_{i}"] = average_task_metrics.accuracy
            elif isinstance(average_task_metrics, RegressionMetrics):
                results[f"mse/task_{i}"] = average_task_metrics.mse
            else:
                results[f"task_{i}"] = average_task_metrics
        return results
