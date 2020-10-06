from settings.base import Results
from dataclasses import dataclass
from settings.assumptions.incremental import IncrementalSetting
from utils import mean
from common import Metrics, RegressionMetrics, ClassificationMetrics
from typing import List


@dataclass
class RLResults(IncrementalSetting.Results, Results):
    """ Results for RL settings.

    Similar to ClassIncrementalResults, but here the metrics would be the mean
    reward, something like that.
    """
    test_metrics: List[List[RegressionMetrics]]

    @property
    def objective(self) -> float:
        return self.mean_reward
    
    @property
    def mean_reward(self):        
        average_metric = self.average_metrics
        assert False, average_metric
    
    @property
    def objective(self) -> float:
        return self.mean_reward
    
    def summary(self):
        return f"Mean reward: {self.mean_reward}"
        return super().summary()
    
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
