import glob
import json
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, Callable, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from simple_parsing import ArgumentParser, field, list_field, mutable_field

# TODO: fill out a bug for SimpleParsing, when type is List and a custom type is
# given, the custom type should overwrite the List type.

REQUIRED_FILES: List[str] = [
    "results/state.json",
    # "results/final_task_accuracy.csv"
]


def n_tasks_used(run_path: Path) -> int:
    run_name = run_path.name
    # prefix the baseline run with 0_ so it shows up first in plots.
    if "baseline" in run_name:
        return 0
    # Add a prefix for methods with auxiliary tasks, indicating how many tasks were used.
    tasks = ["rot", "vae", "ae", "simclr", "irm", "mixup", "brightness", "ewc"]
    count = 0
    for t in tasks:
        if t in run_name:
            run_name = run_name.replace(t, "")
            count += 1
    return count


def class_accuracy_v0(result_json: Dict) -> np.ndarray:
    return np.array(result_json["metrics"]["supervised"]["accuracy"])

def class_accuracy_v1(result_json: Dict) -> np.ndarray:
    return np.array(result_json["supervised"]["metrics"]["accuracy"])

def class_accuracy_v2(result_json: Dict) -> np.ndarray:
    return np.array(result_json["Test"]["supervised"]["metrics"]["accuracy"])

def class_accuracy_v3(result_json: Dict) -> np.ndarray:
    # NOTE: We only need to import it so that the TaskIncremental.State class is
    # created and registered as JsonSerializable.
    from task_incremental import TaskIncremental, get_supervised_accuracy
    State = TaskIncremental.State
    results = State.from_dict(result_json, drop_extra_fields=False)
    cumul_losses = results.cumul_losses
    accuracies = np.zeros(len(cumul_losses), dtype=float)
    for i, cumul_loss in enumerate(cumul_losses):
        acc = get_supervised_accuracy(cumul_loss)
        accuracies[i] = acc
    return accuracies


def get_cumul_accuracy(run_dir: Path) -> np.ndarray:
    """Gets the cumulative test accuracies for each task for a single run.

    Unfortunately has to account for a number of ways of getting that value,
    depending on the version of source code that launched it, as the format
    changed a bit over time. Uses the `class_accuracy_v*` functions above,
    trying newer versions first.

    Args:
        run_dir (Path): Directory containing the "results/results.json" file.

    Returns:
        np.ndarray: The array of task accuracies.
    """

    possible_paths: List[Path] = [
        run_dir / "results" / "state.json",
        run_dir / "results" / "results.json",    
    ]

    for path in possible_paths:
        try:
            with open(path) as f:
                results_json = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            continue

        class_accuracy_fns = [class_accuracy_v3, class_accuracy_v2, class_accuracy_v1, class_accuracy_v0]
        for class_accuracy_fn in class_accuracy_fns:
            try:
                accuracy = class_accuracy_fn(results_json)
                # print("Successfully loaded with ", class_accuracy_fn.__name__)
                return accuracy
            except KeyError as e:
                pass
    
    raise RuntimeError(f"Unable to load the cumulative accuracy for run dir {run_dir}")


def get_cumul_accuracies(log_dir: Path) -> np.ndarray:
    """Returns the cumulative test accuracies for each task at the end of training (for all the runs).

    New (upcoming format):
    If there are "run_*" (or "-0") direct subdirectories, stacks the results of
    each run before returning it. See the `get_individual_run_dirs` function.

    Args:
        log_dir (Path): The path to the log directory folder, for example
        "TaskIncremental/cifar10_mh_d_baseline".

    Returns:
        np.ndarray: An array of shape [n_runs, n_tasks] containing the
        cumulative Test classification accuracy during training. 
    """
    task_accuracies_list: List[np.ndarray] = []
    for run_dir in get_nonempty_run_dirs(log_dir):
        accuracy = get_cumul_accuracy(run_dir)
        if accuracy.ndim == 2:
            task_accuracies_list.append(accuracy)
        else:
            task_accuracies_list.append(accuracy[None, :])
    if not task_accuracies_list:
        raise RuntimeError(f"Couldn't load final task accuracies from {log_dir}")
    task_accuracies = np.concatenate(task_accuracies_list)
    return task_accuracies


def get_task_accuracy_v1(run_dir: Path) -> np.ndarray:
    return load_array(run_dir / "results" / "final_task_accuracy.csv")


def get_task_accuracy_v2(run_dir: Path) -> np.ndarray:
    from task_incremental import TaskIncremental, get_supervised_accuracy, get_supervised_metrics
    from common.metrics import ClassificationMetrics, RegressionMetrics
    state = TaskIncremental.State.load_json(run_dir / "results" / "state.json")
    
    tasks = state.tasks
    n_tasks = len(tasks)

    task_losses = state.task_losses
    assert len(task_losses) == n_tasks
    i = len(task_losses) -1
    last_task_losses = task_losses[-1]

    task_accuracies = np.zeros(n_tasks, dtype=float)

    assert len(last_task_losses) == n_tasks
    for j, loss in enumerate(last_task_losses):
        acc = get_supervised_accuracy(loss)
        metric = get_supervised_metrics(loss)
        # print(f"i: {i} j: {j} accuracy: {acc}")
        task_accuracies[j] = acc
    return task_accuracies


def get_final_task_accuracy(run_dir: Path) -> np.ndarray:
    """Loads the final task accuracy for an individual run.

    Args:
        run_dir (Path): The directory of an individual run. 

    Returns:
        np.ndarray: float array of shape [n_tasks]. The array containing the
        mean accuracy for each task at the end of training.
    """
    potential_functions = [get_task_accuracy_v2, get_task_accuracy_v1]
    for fn in potential_functions:
        try:
            return fn(run_dir)
        except Exception as e:
            pass
            # print(f"Exception: {e}")
            # exit()
    raise RuntimeError(f"Unable to load the final task accuracies for run dir {run_dir}")


def get_final_task_accuracies(log_dir: Path) -> np.ndarray:
    """Returns the mean accuracy for each task at the end of training.

    New (upcoming/WIP format):
    If there are "run_*" (or "-0") direct subdirectories, stacks the results of
    each run before returning it. See the `get_individual_run_dirs` function.

    Args:
        log_dir (Path): The "parent" log dir, for example 
        `Path("results/TaskIncremental/cifar10_mh_d_baseline")`.

    Returns:
        np.ndarray: An array of shape [n_runs, n_tasks] containing the mean
        Test accuracy for each task. 
    """
    task_accuracies_list: List[np.ndarray] = [] 
    for run_dir in get_nonempty_run_dirs(log_dir):
        accuracy = get_final_task_accuracy(run_dir)
        if accuracy.ndim == 2:
            task_accuracies_list.append(accuracy)
        else:
            task_accuracies_list.append(accuracy[None, :])
    if not task_accuracies_list:
        raise RuntimeError(f"Couldn't load final task accuracies from {log_dir}")
    task_accuracies = np.concatenate(task_accuracies_list)
    return task_accuracies


def get_nonempty_run_dirs(log_dir: Path) -> Iterable[Path]:
    """Yields the paths to each individual runs of a given (parent) log dir.

    NOTE: May yield the log_dir itself, in the (old setup) case where there are
    no subdirectories for each individual run. Users of this function should not
    use recursive glob calls from the yielded paths (to find "results.json", for
    example) as this might yield duplicates of the other run directories. 

    Args:
        log_dir (Path): The "parent" log dir.

    Returns:
        Iterable[Path]: [description]

    Examples:
    Given the following structure:
    ```console
    baseline/
    ├── run_0
    └── run_1
    ```
    
    >>> print([str(p) for p in get_individual_run_dirs("baseline")])
    ["baseline/run_0", "baseline/run_1"]
    
    
    ```
    baseline/
    ├── run_0
    ├── results
    ├── checkpoints
    ├── plots
    ├── foo
    └── -0
    ```
    >>> print([str(p) for p in get_individual_run_dirs("baseline")])
    ["baseline", "baseline/-0", "baseline/run_0"]
    """
    if is_run_dir(log_dir):
        yield log_dir
    # This weird naming format was due to a bug, and didn't last very long. 
    for run_dir in log_dir.glob("-*"):
        if is_run_dir(run_dir) and run_dir.name.split("-", maxsplit=1)[-1].isdigit():
            yield run_dir
    for run_dir in log_dir.glob("run_*"):
        if is_run_dir(run_dir) and run_dir.name.split("_", maxsplit=1)[-1].isdigit():
            yield run_dir

def is_run_dir(path: Path) -> bool:
    """Returns wether the given Path is a run directory.

    Args:
        path (Path): a given Path.

    Returns:
        bool: True if it is a run dir (contains all the required files).
    """
    required_folders = ""
    if not path.is_dir():
        return False
    for req_fil_path in REQUIRED_FILES:
        if not (path / req_fil_path).exists():
            # print(f"File {path / req_fil_path} doesn't exist!")
            return False
    return True

def is_log_dir(path: Path, recursive: bool=False) -> bool:
    """Returns wether the given path points to a log_dir containing at least one non-empty run.

    Args:
        path (Path): a directory.
        recursive (bool): When True, checks the subdirectories recursively. If
        False, only checks the immediate children directories.

    Returns:
        bool: Wether this directory is or contains non-empty run directories.
    """
    if is_run_dir(path):
        return True
    for child in path.iterdir():
        if recursive and is_log_dir(child, recursive=True):
            return True
        elif is_run_dir(child):
            return True
    return False


def filter_runs(all_log_dirs: List[Path]) -> Tuple[List[Path], List[Path]]:
    kept_runs: List[Path] = []
    lost_runs: List[Path] = []
    for log_dir in all_log_dirs:
        if is_log_dir(log_dir):
            # print(f"dir {log_dir} is a log dir")
            kept_runs.append(log_dir)
        else:
            # print(f"dir {log_dir} isnt a log dir")
            lost_runs.append(log_dir)
    return kept_runs, lost_runs


@dataclass
class OmlFigureOptions:
    """ Options for the script making the OML Figure 3 plot. """
    # One or more paths of glob patterns matching the run folders to compare.
    # NOTE: should ideally be the immediate parent folder of the runs.
    runs: List[str] = list_field(default=["results/TaskIncremental/*"])
    # Output path where the figure should be stored.
    out_path: Path = Path("scripts/oml_plot.png")

    extension: str = ".png"

    # title to use for the figure.
    title: Optional[str] = None
    # Also show the figure.
    show: bool = False
    # Exit after creating the figure.
    exit_after: bool = True

    # Add a prefix indicating the number of auxiliary tasks used: ("n_").
    add_ntasks_prefix: bool = False

    # Wether or not to maximize the figure to fit the current screen size.
    maximize_figure: bool = True
    # A manually specified figure size, in case we don't want to maximize the figure.
    fig_size_inches: Tuple[int, int] = (12, 9)
    
    # An optional function that returns the formatted label, given a run's path
    # and the current auto-formatted label.
    label_formatting_fn: InitVar[Optional[Callable[[Path, str], str]]] = None

    # Where to place the legend.
    legend_position: str = "lower left"

    result_figure: Optional[plt.Figure] = field(init=False, default=None)
    
    classification_accuracies: Dict[Path, np.ndarray] = mutable_field(OrderedDict, init=False)
    final_task_accuracies: Dict[Path, np.ndarray] = mutable_field(OrderedDict, init=False)

    def __post_init__(self, label_formatting_fn: Callable[[Path, str], str]=None):
        self.label_formatting_fn = label_formatting_fn

        if len(self.runs) == 1 and isinstance(self.runs[0], list):
            self.runs = self.runs[0]
        print(self.runs)

        run_paths: List[Path] = []
        for run_pattern in self.runs:
            for p in map(Path, glob.glob(run_pattern)):
                if p.is_dir():
                    run_paths.append(p)

        required_files: List[str] = REQUIRED_FILES

        kept_runs, lost_runs = filter_runs(run_paths)
        
        print("Kept runs:", len(kept_runs))
        for path in kept_runs:
            print("\t", path)

        print("Lost/empty runs:", len(lost_runs))
        for path in lost_runs:
            print("\t", path)
        
        if not kept_runs:
            warnings.warn(
                f"There are NO kept runs for path or pattern(s) {self.runs}. \n"
                "Returning early without creating the figure. \n"
                f"Lost runs: \n"
                +("\n".join(map(str,lost_runs)))
            )
            return
        
        prefix = longest_common_prefix([p.name for p in kept_runs])
        print(f"Common prefix: '{prefix}'")
        if self.title is None and prefix:
            self.title = prefix
            # if any(str(p.parent) != self.title for p in paths):
            #     self.title = "Results"

        fig = self.make_plot(kept_runs)
        
        if self.maximize_figure:
            maximize_figure()
        else:
            fig.set_size_inches(self.fig_size_inches)
        
        if self.out_path:    
            self.out_path = Path(self.out_path)
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.out_path)
            if self.extension:
                fig.savefig(self.out_path.with_suffix(self.extension))

        if self.show:
            plt.show() #close the figure to run the next section
        
        self.result_figure = fig
        print(f"Successfully created plot at \"{self.out_path}\"")
        if self.exit_after:
            exit()

    def make_plot(self, kept_runs: List[Path]) -> plt.Figure:

        runs = kept_runs
        n_runs = len(runs)
        print(f"Creating the OML plot to compare the {n_runs} different methods:")
        
        fig: plt.Figure = plt.figure()
        if self.title:
            fig.suptitle(self.title)
                
        gs = gridspec.GridSpec(1, 3, width_ratios=[2,1,2])

        ax1: plt.Axes = fig.add_subplot(gs[0])
        ax1.set_title("Cumulative Accuracy")
        ax1.set_xlabel("Number of tasks learned")
        ax1.set_ylabel("Cumulative Test Accuracy")
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
        ax3.set_title(f"Final Cumulative Accuracy")
        ax3.set_yticks(np.arange(len(indicators)*n_runs))
        ax3.set_xticks([])
        ax3.set_yticklabels(indicators*n_runs)
        ax3.set_ylim(top=len(indicators) * n_runs)
        ax3.set_xlim(left=-0.5, right=0.5)
        
        # technically, we don't know the amount of tasks yet.
        n_tasks: int = -1
        
        run_names: List[str] = [p.name for p in runs]
        prefix = longest_common_prefix(run_names)

        # Sorted first by number of tasks, then by path.
        runs = sorted(runs, key=lambda p: (n_tasks_used(p), p))
        
        for i, run_path in enumerate(runs):
            print(i, run_path)
            # Get the classification accuracy per task for all runs.
            classification_accuracies = get_cumul_accuracies(run_path)

            # Get the per-task classification accuracy at the end of training
            # for each run.
            final_task_accuracy = get_final_task_accuracies(run_path)
            
            self.final_task_accuracies[run_path] = final_task_accuracy
            self.classification_accuracies[run_path] = classification_accuracies

            accuracy_means = classification_accuracies.mean(axis=0)
            accuracy_stds = classification_accuracies.std(axis=0)
            local_n_tasks = len(accuracy_means)
            n_tasks = max(n_tasks, local_n_tasks)
            
            task_accuracy_means = final_task_accuracy.mean(axis=0)
            task_accuracy_std =   final_task_accuracy.std(axis=0)
            ax1.set_xticks(np.arange(n_tasks, dtype=int))
            ax1.set_xticklabels(np.arange(1, n_tasks+1, dtype=int))
            
            label = run_path.name if not run_path.name.startswith(("run_", "-")) else run_path.parent.name
            label = label.replace(prefix, "")
            if not label:
                label = prefix
            label = label.lstrip("-_")
            
            n_aux_tasks = n_tasks_used(run_path)
            has_ntask_prefix = label[0].isdigit() and label[1] == "_"
            if self.add_ntasks_prefix and not has_ntask_prefix:
                label = f"{n_aux_tasks}_{label}"
            elif has_ntask_prefix:
                # remove the "<n_tasks>_" prefix:
                label = label[2:]

            if self.label_formatting_fn is not None:
                label = self.label_formatting_fn(run_path, label)

            print(f"Run {run_path}:")
            print("\t Accuracy Means:", accuracy_means)
            print("\t Accuracy STDs:", accuracy_stds)
            print("\t Final Task Accuracy means:", task_accuracy_means)
            print("\t Final Task Accuracy stds:", task_accuracy_std)
            
            # adding the error plot on the left
            ax1.errorbar(
                x=np.arange(len(accuracy_means)),
                y=accuracy_means,
                yerr=accuracy_stds,
                label=label
            )

            # Determining the bottom and height of the bars on the right plot.
            bottom = len(indicators) * ((n_runs - 1) - i)
            height = bar_height_scale * task_accuracy_means
            rects = ax2.bar(
                x=np.arange(len(height)),
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

            n_samples = classification_accuracies.shape[0]
            # adding the percentage labels over the bars on the middle plot.
            autolabel(ax3, rects, bar_height_scale, final_cumul_std, n_samples=n_samples)

        ax2.hlines(
            y=np.arange(len(indicators)*n_runs),
            xmin=0-0.5,
            xmax=n_tasks-0.5,
            linestyles="dashdot",
            colors="lightgray",
        )
        ax3.hlines(
            y=np.arange(len(indicators)*n_runs),
            xmin=-1,
            xmax=2,
            linestyles="dashdot",
            colors="lightgray",
        )
        ax2.set_xticks(np.arange(n_tasks, dtype=int))
        ax1.legend(loc=self.legend_position)

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


def longest_common_prefix(values: List[str]) -> str:
    if not values:
        return None

    first = values[0]
    i = 1
    while first[:i] and all(v.startswith(first[:i]) for v in values):
        i += 1
    i -= 1
    return first[:i]


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


def autolabel(axis, rects: List[plt.Rectangle], bar_height_scale: float=1., errors: Union[list, np.ndarray, float]=None, n_samples: int=None):
    """Attach a text label above each bar in *rects*, displaying its height.
    
    Taken from https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    print(f"rectangles: {len(rects)}")
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
            
            if n_samples is not None:
                value_string += f" (n={n_samples})"

            axis.annotate(
                value_string,
                xy=(rect.get_x() + rect.get_width() / 2, bottom + height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )


def format_label(run_path: Path, current_label: str) -> str:
    just_task_names = (current_label
        # Get rid of the prefix that indicates the number of tasks:
        .replace("0_", "_")
        .replace("1_", "_")
        .replace("2_", "_")
        .replace("3_", "_")
        # Get rid of the coefficients:
        # (usually *_1* or *_01* or *_001* and *_nc_*)
        .replace("1", "_")
        .replace("0", "_")
        .replace("nc", "_")
    )
    return " + ".join(just_task_names.replace("_", " ").split())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(OmlFigureOptions, "options")
    args = parser.parse_args()
