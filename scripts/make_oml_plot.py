import glob
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from simple_parsing import ArgumentParser, field, list_field

# TODO: fill out a bug for SimpleParsing, when type is List and a custom type is
# given, the custom type should overwrite the List type.


def n_tasks_used(run_path: Path) -> int:
    run_name = run_path.name
    # prefix the baseline run with 0_ so it shows up first in plots.
    if "baseline" in run_name:
        return 0
    # Add a prefix for methods with auxiliary tasks, indicating how many tasks were used.
    tasks = ["rot", "vae", "ae", "simclr", "irm", "mixup", "brightness"]
    count = 0
    for t in tasks:
        if t in run_name:
            run_name = run_name.replace(t, "")
            count += 1
    return count

@dataclass
class OmlFigureOptions:
    """ Options for the script making the OML Figure 3 plot. """
    # One or more paths of glob patterns matching the run folders to compare.
    runs: List[str] = list_field(default=["results/TaskIncremental/*"])
    # Output path where the figure should be stored.
    out_path: Path = Path("scripts/oml_plot.png")
    # title to use for the figure.
    title: Optional[str] = None
    # Also show the figure.
    show: bool = False
    # Exit after creating the figure.
    exit_after: bool = True

    # Add a prefix indicating the number of auxiliary tasks used: ("n_").
    add_ntasks_prefix: bool = False

    def __post_init__(self):
        # The dictionary of result dicts.
        self.results: Dict[Path, Dict] = OrderedDict()

        if len(self.runs) == 1 and isinstance(self.runs[0], list):
            self.runs = self.runs[0]
        print(self.runs)
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
        
        if paths and not self.title:
            self.title = str(paths[0].parent)
            if any(str(p.parent) != self.title for p in paths):
                self.title = "Results"

        fig = self.make_plot()
        maximize_figure()
        self.out_path = Path(self.out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.out_path)
        if self.show:
            plt.show() #close the figure to run the next section

        print(f"Successfully created plot at \"{self.out_path}\"")
        if self.exit_after:
            exit()

    def make_plot(self) -> plt.Figure:
        results: Dict[Path, Dict] = self.results
        runs: List[Path] = list(results.keys())

        n_runs = len(runs)
        print(f"Creating the OML plot to compare the {n_runs} different methods:")
        
        fig: plt.Figure = plt.figure()
        fig.suptitle(self.title)
                
        gs = gridspec.GridSpec(1, 3, width_ratios=[2,1,2])

        ax1: plt.Axes = fig.add_subplot(gs[0])
        ax1.set_title("Cumulative Validation Accuracy During Training")
        ax1.set_xlabel("Number of tasks learned")
        ax1.set_ylabel("Cumulative Validation Accuracy")
        ax1.set_ylim(bottom=0, top=1)

        indicators = ["0", "1.00"]
        bar_height_scale = len(indicators) - 1
        ax2: plt.Axes = fig.add_subplot(gs[2])
        ax2.set_title(f"Per-Task Accuracy At the End of Training")
        ax2.set_xlabel("Task ID")
        ax2.set_yticks(np.arange(len(indicators)*n_runs))
        ax2.set_yticklabels(indicators*n_runs)
        ax2.set_ylim(top=len(indicators) * n_runs)
        
        ax3: plt.Axes = fig.add_subplot(gs[1])
        ax3.set_title(f"Final Cumulative Validation Accuracy")
        ax3.set_yticks(np.arange(len(indicators)*n_runs))
        ax3.set_xticks([])
        ax3.set_yticklabels(indicators*n_runs)
        ax3.set_ylim(top=len(indicators) * n_runs)
        ax3.set_xlim(left=-0.5, right=0.5)
        # technically, we don't know the amount of tasks yet.
        n_tasks: int = -1


        for i, run_path in enumerate(sorted(results, key=n_tasks_used)):
            result_json = results[run_path]
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
            
            label = run_path.name
            n_aux_tasks = n_tasks_used(run_path)
            if self.add_ntasks_prefix and not label.startswith(f"{n_aux_tasks}_"):
                label = f"{n_aux_tasks}_{label}"
            
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

            final_cumul_accuracy = accuracy_means[-1]
            final_cumul_std = accuracy_stds[-1]
            height = bar_height_scale * final_cumul_accuracy
            rects = ax3.bar(
                x=0,
                height=height,
                bottom=bottom,
                yerr=final_cumul_std,
                label=label,
            )
            # adding the percentage labels over the bars on the right plot.
            autolabel(ax3, rects, bar_height_scale, final_cumul_std)

        ax2.hlines(
            y=np.arange(len(indicators)*n_runs),
            xmin=0-0.5,
            xmax=n_tasks-0.5,
            linestyles="dotted",
            colors="gray",
        )
        ax3.hlines(
            y=np.arange(len(indicators)*n_runs),
            xmin=-1,
            xmax=2,
            linestyles="dotted",
            colors="gray",
        )
        ax2.set_xticks(np.arange(n_tasks, dtype=int))
        ax1.legend(loc="upper left")

        return fig

def maximize_figure():
    fig_manager = plt.get_current_fig_manager()
    try:
        fig_manager.window.showMaximized()
    except:
        try:
            fig_manager.window.state('zoomed') #works fine on Windows!
        except:
            try:
                fig_manager.frame.Maximize(True)
            except:
                print("Couldn't maximize the figure.")


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


def autolabel(axis, rects: List[plt.Rectangle], bar_height_scale: float=1., errors: Union[list, np.ndarray, float]=None):
    """Attach a text label above each bar in *rects*, displaying its height.
    
    Taken from https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for i, rect in enumerate(rects):
        height = rect.get_height()
        bottom = rect.get_y()
        value = height / bar_height_scale
        error = None
        if errors is not None:
            if isinstance(errors, (list, np.ndarray)):
                error = errors[i]
            else:
                error = errors
        if value != 0.0:
            value_string = f"{value:.0%}"
            if error is not None:
                value_string = f"{value*100:.1f} ± {error:.1%}"

            axis.annotate(
                value_string,
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
