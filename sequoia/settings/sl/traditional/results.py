"""Defines the Results of apply a Method to an IID Setting.  
"""
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt

from sequoia.settings.sl.incremental.results import IncrementalSLResults


class IIDResults(IncrementalSLResults):
    """Results of applying a Method on an IID Setting.

    # TODO: Refactor this to be based on `TaskResults`?
    """

    def save_to_dir(self, save_dir: Union[str, Path]) -> None:
        # TODO: Add wandb logging here somehow.
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        plots: Dict[str, plt.Figure] = self.make_plots()

        # Save the actual 'results' object to a file in the save dir.
        results_json_path = save_dir / "results.json"
        self.save(results_json_path)
        print(f"Saved a copy of the results to {results_json_path}")

        print(f"\nPlots: {plots}\n")
        for fig_name, figure in plots.items():
            print(f"fig_name: {fig_name}")
            # figure.show()
            # plt.waitforbuttonpress(10)
            path = (save_dir / fig_name).with_suffix(".jpg")
            path.parent.mkdir(exist_ok=True, parents=True)
            figure.savefig(path)
            print(f"Saved figure at path {path}")

    def make_plots(self) -> Dict[str, plt.Figure]:
        plots_dict = super().make_plots()
        # TODO: Could add a Confusion Matrix plot?
        plots_dict.update({"class_accuracies": self.class_accuracies_plot()})
        return plots_dict

    def class_accuracies_plot(self):
        figure: plt.Figure
        axes: plt.Axes
        figure, axes = plt.subplots()
        y = self[0][0].average_metrics.class_accuracy
        x = list(range(len(y)))
        rects = axes.bar(x, y)
        axes.set_title("Class Accuracy")
        axes.set_xlabel("Class")
        axes.set_ylabel("Accuracy")
        axes.set_ylim(0, 1.0)
        # autolabel(axes, rects)
        return figure

    # def summary(self) -> str:
    #     s = StringIO()
    #     with redirect_stdout(s):
    #         print(f"Average Accuracy: {self.average_metrics.accuracy:.2%}")
    #         for i, class_acc in enumerate(self.average_metrics.class_accuracy):
    #             print(f"Accuracy for class {i}: {class_acc:.3%}")
    #     s.seek(0)
    #     return s.read()

    def to_log_dict(self, verbose: bool = False) -> Dict[str, float]:
        results = super().to_log_dict(verbose=verbose)
        # Remove the useless 2-levels of nesting from the log_dict
        results.update(results.pop("Task 0").pop("Task 0"))
        # assert False, json.dumps(results, indent="\t")
        return results
