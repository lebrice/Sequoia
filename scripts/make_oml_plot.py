import torch
from simple_parsing import ArgumentParser, field, list_field
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable
import matplotlib.pyplot as plt
import numpy as np
import glob
import json

# TODO: fill out a bug for SimpleParsing, when type is List and a custom type is
# given, the custom type should overwrite the List type.

@dataclass
class Options:
    """ Options for the script making the OML Figure 3 plot. """
    # One or more paths of glob patterns, each corresponding to a run to compare.
    runs: List[str] = list_field(default=["results/TaskIncremental/*"])
    out_path: Path = Path("scripts/oml_plot.png")
    
    def __post_init__(self):
        if len(self.runs) == 1 and isinstance(self.runs[0], list):
            self.runs = self.runs[0]

        paths: List[Path] = []
        for pattern in self.runs:
            paths.extend(map(Path, glob.glob(pattern)))

        def keep_run(path: Path) -> bool:
            if not path.is_dir():
                return False
            if str(path).endswith("wandb"):
                return False
            if not path.joinpath("results").exists():
                return False
            return True
        
        self.run_paths = list(filter(keep_run, paths))


def make_plot(options: Options) -> plt.Figure:
    runs: List[Path] = options.run_paths
    
    n_runs = len(runs)
    print(f"Creating the OML plot to compare the {n_runs} different methods:")
    
    fig: plt.Figure = plt.figure()
    
    title = str(runs[0].parent)
    for run_path in runs:
        if str(run_path.parent) != title:
            title = "Results"
            break
    fig.suptitle(title)

    ax1: plt.Axes = fig.add_subplot(1, 2, 1)
    ax1.set_title("Cumulative Loss")
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

    for i, run_path in enumerate(runs):
        # Load up the per-task classification accuracies
        final_task_accuracy = load_array(run_path / "results" / "final_task_accuracy.csv")
        try:    
            # Load the results dict
            with open(run_path / "results" / "results.json") as f:
                result_json = json.load(f)
        except FileNotFoundError as e:
            print(f"Skipping run {run_path}: ", e)
            continue

        supervised_metrics = result_json["metrics"]["supervised"]
        classification_accuracies = np.array(supervised_metrics["accuracy"])
        accuracy_means = classification_accuracies.mean(axis=0)
        accuracy_stds = classification_accuracies.std(axis=0)
        n_tasks = len(accuracy_means)

        task_accuracy_means = final_task_accuracy.mean(axis=0)
        task_accuracy_std =   final_task_accuracy.std(axis=0)
        print("n tasks:", n_tasks)
        ax1.set_xticks(np.arange(n_tasks, dtype=int))
        label = str(run_path.parts[-1])

        print(f"Run {run_path}:")
        print("\t Accuracy Means:", accuracy_means)
        print("\t Accuracy STDs:", accuracy_stds)
        print("\t Final Task Accuracy means:", task_accuracy_means)
        print("\t Final Task Accuracy stds:", task_accuracy_std)
        ax1.errorbar(x=np.arange(n_tasks), y=accuracy_means, yerr=accuracy_stds, label=str(run_path.parts[-1]))
        # TODO: figure out how to stack the bars like in OML plot.
        bottom = len(indicators) * i
        print(f"Bottom: {bottom}")
        height = bar_height_scale * task_accuracy_means
        rects = ax2.bar(x=np.arange(n_tasks), height=height, bottom=bottom, yerr=task_accuracy_std, label=label)
        
        autolabel(ax2, rects, bar_height_scale)

    ax2.hlines(y=np.arange(len(indicators)*n_runs), xmin=0-0.5, xmax=n_tasks-1+0.5, linestyles="dotted", colors="gray")
    ax2.set_xticks(np.arange(n_tasks, dtype=int))
    ax1.legend(loc="upper left")

    fig.savefig(options.out_path)
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
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    options: Options = args.options
    print(options)

    make_plot(options)