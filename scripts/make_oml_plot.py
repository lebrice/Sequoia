import glob
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from simple_parsing import ArgumentParser, field, list_field

# TODO: fill out a bug for SimpleParsing, when type is List and a custom type is
# given, the custom type should overwrite the List type.

@dataclass
class OmlFigureOptions:
    """ Options for the script making the OML Figure 3 plot. """
    # One or more paths of glob patterns matching the run folders to compare.
    runs: List[str] = list_field(default=["results/TaskIncremental/*"])
    # Output path where the figure should be stored.
    out_path: Path = Path("scripts/oml_plot.png")
    # title to use for the figure.
    title: Optional[str] = None

    def __post_init__(self):
        # The dictionary of result dicts.
        self.results: Dict[Path, Dict] = OrderedDict()

        if len(self.runs) == 1 and isinstance(self.runs[0], list):
            self.runs = self.runs[0]

        paths: List[Path] = []
        for pattern in self.runs:
            paths.extend(map(Path, glob.glob(pattern)))

        def keep_run(path: Path) -> Union[bool, Dict]:
            if not path.is_dir():
                return False
            elif str(path).endswith("wandb"):
                return False
            elif not path.joinpath("results").exists():
                return False
            elif not (path / "results" / "results.json").exists():
                return False
            try:
                # Load the results dict
                with open(path / "results" / "results.json") as f:
                    result_json = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                return False
            else:
                return result_json
        
        for run_path in paths:
            result = keep_run(run_path)
            if result:
                self.results[run_path] = result
        
        if not self.title:
            self.title = str(paths[0].parent)
            if any(str(p.parent) != self.title for p in paths):
                self.title = "Results"
        
        self.make_plot()
        print("DONE, exiting")
        exit()

    def make_plot(self) -> plt.Figure:
        results: Dict[Path, Dict] = self.results
        runs: List[Path] = list(results.keys())

        n_runs = len(runs)
        print(f"Creating the OML plot to compare the {n_runs} different methods:")
        
        fig: plt.Figure = plt.figure()
        fig.suptitle(self.title)

        ax1: plt.Axes = fig.add_subplot(1, 2, 1)
        ax1.set_title("Classification Accuracy on Tasks seen so far")
        ax1.set_xlabel("Number of tasks learned")
        ax1.set_ylabel("Cumulative Validation Accuracy")
        ax1.set_ylim(bottom=0, top=1)
        ax2: plt.Axes = fig.add_subplot(1, 2, 2)
        ax2.set_title(f"Final mean accuracy per Task")
        ax2.set_xlabel("Task ID")

        indicators = ["0", "1.00"]
        bar_height_scale = len(indicators) - 1
        ax2.set_yticks(np.arange(len(indicators)*n_runs))
        ax2.set_yticklabels(indicators*n_runs)
        ax2.set_ylim(top=len(indicators) * n_runs)
        
        # technically, we don't know the amount of tasks yet.
        n_tasks: int = -1

        for i, (run_path, result_json) in enumerate(results.items()):
            # Load up the per-task classification accuracies
            final_task_accuracy = load_array(run_path / "results" / "final_task_accuracy.csv")

            supervised_metrics = result_json["metrics"]["supervised"]
            classification_accuracies = np.array(supervised_metrics["accuracy"])
            accuracy_means = classification_accuracies.mean(axis=0)
            accuracy_stds = classification_accuracies.std(axis=0)
            n_tasks = len(accuracy_means)

            task_accuracy_means = final_task_accuracy.mean(axis=0)
            task_accuracy_std =   final_task_accuracy.std(axis=0)
            ax1.set_xticks(np.arange(n_tasks, dtype=int))
            ax1.set_xticklabels(np.arange(1, n_tasks+1, dtype=int))
            label = str(run_path.parts[-1])

            print(f"Run {run_path}:")
            print("\t Accuracy Means:", accuracy_means)
            print("\t Accuracy STDs:", accuracy_stds)
            print("\t Final Task Accuracy means:", task_accuracy_means)
            print("\t Final Task Accuracy stds:", task_accuracy_std)
            # adding the error plot on the left
            ax1.errorbar(
                x=np.arange(n_tasks),
                y=accuracy_means,
                yerr=accuracy_stds,
                label=label
            )

            # Determining the bottom and height of the bars on the right plot.
            bottom = len(indicators) * ((n_runs - 1) - i)
            height = bar_height_scale * task_accuracy_means
            rects = ax2.bar(
                x=np.arange(n_tasks),
                height=height,
                bottom=bottom,
                yerr=task_accuracy_std,
                label=label
            )
            # adding the percentage labels over the bars on the right plot.
            autolabel(ax2, rects, bar_height_scale)

        ax2.hlines(
            y=np.arange(len(indicators)*n_runs),
            xmin=0-0.5,
            xmax=n_tasks-0.5,
            linestyles="dotted",
            colors="gray",
        )
        ax2.set_xticks(np.arange(n_tasks, dtype=int))
        ax1.legend(loc="upper left")

        fig.savefig(self.out_path)
        fig.show()
        fig.waitforbuttonpress(timeout=30)


def load_array(path: Path) -> np.ndarray:
    """Loads a numpy array from a file.
    
    Args:
        path (Path): A path to load from (the extension is ignored). Will load a
        "csv" file with the given name, if there is one. If not, will look for a
        pytorch pickle file. If there is a pytorch pickle file, will try to
        load it, and then save a csv version for later use.
    
    Returns:
        np.ndarray: The array.
    """
    if path.with_suffix(".pt").exists() and not path.with_suffix(".csv").exists():
        array = torch.load(path.with_suffix(".pt")).detach().numpy()
        np.savetxt(path.with_suffix(".csv"), array, delimiter=",")
    return np.loadtxt(path.with_suffix(".csv"), delimiter=",")


def autolabel(axis, rects: List[plt.Rectangle], bar_height_scale: float=1.):
    """Attach a text label above each bar in *rects*, displaying its height.
    
    Taken from https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        bottom = rect.get_y()
        value = height / bar_height_scale
        if value != 0.0:    
            axis.annotate(
                f"{value:.0%}",
                xy=(rect.get_x() + rect.get_width() / 2, bottom + height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(OmlFigureOptions, "options")
    args = parser.parse_args()
