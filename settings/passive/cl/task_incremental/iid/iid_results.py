"""Defines the Results of apply a Method to an IID Setting.  
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
from io import StringIO
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

from common import ClassificationMetrics, Loss, Metrics, RegressionMetrics
from settings.base.results import Results
from utils.plotting import PlotSectionLabel, autolabel

from .. import TaskIncrementalResults


@dataclass
class IIDResults(Results):
    """Results of applying a Method on an IID Setting.    
    TODO: This should be customized, as it doesn't really make sense to use the
    same plots as in ClassIncremental (there is only one task).
    """
    test_metric: Metrics
        
    @property
    def objective(self) -> float:
        if isinstance(self.test_metric, ClassificationMetrics):
            return self.test_metric.accuracy
        if isinstance(self.test_metric, RegressionMetrics):
            return self.test_metric.mse
        return self.test_metric

    def save_to_dir(self, save_dir: Union[str, Path]) -> None:
        # TODO: Add wandb logging here somehow.
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        plots: Dict[str, plt.Figure] = self.make_plots()
        # Save the actual 'results' object to a file in the save dir.
        self.save(save_dir / "results.json")
        
        print(f"\nPlots: {plots}\n")
        for fig_name, figure in plots.items():
            print(f"fig_name: {fig_name}")
            figure.show()
            plt.waitforbuttonpress(10)
            path = (save_dir/ fig_name).with_suffix(".jpg")
            path.parent.mkdir(exist_ok=True, parents=True)
            figure.savefig(path)
            print(f"Saved figure at path {path}")

    def make_plots(self):
        results = {
            "class_accuracies": self.class_accuracies_plot()
        }
        return results

    def class_accuracies_plot(self):
        figure: plt.Figure
        axes: plt.Axes
        figure, axes = plt.subplots()
        rects = axes.hist(self.test_metric.class_accuracy)
        axes.set_title("Class Accuracy")
        axes.set_xlabel("Class")
        axes.set_ylabel("Accuracy")
        axes.set_ylim(0, 1.0)
        # autolabel(axes, rects)
        return figure

    def summary(self) -> str:
        s = StringIO()
        with redirect_stdout(s):
            print(f"Average Accuracy: {self.test_metric.accuracy:.2%}")
            for i, class_acc in enumerate(self.test_metric.class_accuracy):
                print(f"Accuracy for class {i}: {class_acc:.3%}")
        s.seek(0)
        return s.read()

    def to_log_dict(self) -> Dict[str, float]:
        results = {}
        results["objective"] = self.objective
        results.update(self.test_metric.to_log_dict(verbose=True))
        return results
