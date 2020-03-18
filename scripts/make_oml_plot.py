import torch
from simple_parsing import ArgumentParser, field, list_field
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable
import matplotlib.pyplot as plt
import numpy as np
import glob


# TODO: fill out a bug for SimpleParsing, when type is List and a custom type is
# given, the custom type should overwrite the List type.

@dataclass
class Options:
    """ Options for the script making the OML Figure 3 plot. """
    runs: List[str] = field(type=glob.glob, default_factory=lambda: glob.glob("results/TaskIncremental/*"))
    out_path: Path = Path("scripts/plot.png")

    def __post_init__(self):
        if len(self.runs) == 1 and isinstance(self.runs[0], list):
            self.runs = self.runs[0]


def make_plot(options: Options) -> plt.Figure:
    runs: List[Path] = list(map(Path, options.runs))
    
    kept_runs: List[Path] = []
    for run in runs:
        if not run.is_dir() or str(run).endswith("wandb"):
            continue
        
        valid_losses_path = next(run.glob("*/valid_losses.pt"))
        final_accuracy_path = next(run.glob("*/final_task_accuracy.pt"))
        required_files = [valid_losses_path, final_accuracy_path]
        if all(p.exists() and p.is_file for p in required_files):
            kept_runs.append(run)
    
    n = len(kept_runs)
    print(f"Creating the OML plot to compare the {n} different methods:")

    fig: plt.Figure = plt.figure()
    
    title = str(kept_runs[0].parent)
    for run in kept_runs:
        if str(run.parent) != title:
            title = "Results"
            break
    fig.suptitle(title)

    ax1: plt.Axes = fig.add_subplot(1, 2, 1)
    ax1.set_title("Cumulative Loss")
    ax1.set_xlabel("Number of tasks learned")
    ax1.set_ylabel("Cumulative Validation Loss")
    ax1.legend(loc="upper left")

    ax2: plt.Axes = fig.add_subplot(1, 2, 2)
    ax2.set_title(f"Final mean accuracy per Task")
    ax2.set_xlabel("Task ID")

    indicators = ["0", "1.00"]
    bar_height_scale = len(indicators) - 1
    ax2.set_yticks(np.arange(len(indicators)*n))
    ax2.set_yticklabels(indicators*n)
    ax2.set_ylim(top=len(indicators) * n)
    ax2.hlines(y=np.arange(len(indicators)*n), xmin=0, xmax=5, linestyles="dotted", colors="gray")
    

    for i, run_path in enumerate(kept_runs):
        valid_loss = torch.load(next(run_path.glob("*/valid_losses.pt")))
        final_task_accuracy =torch.load(next(run_path.glob("*/final_task_accuracy.pt")))
        loss_means = valid_loss.mean(dim=0).numpy()
        loss_stds = valid_loss.std(dim=0).numpy()

        task_accuracy_means = final_task_accuracy.mean(dim=0).numpy()
        task_accuracy_std =   final_task_accuracy.std(dim=0).numpy()
        
        n_tasks= len(task_accuracy_means)
        label = str(run_path.parts[-1])

        print(f"Run {run_path}:")
        print("\t Loss Means:", loss_means)
        print("\t Loss STDs:", loss_stds)
        print("\t Final Task Accuracy means:", task_accuracy_means)
        print("\t Final Task Accuracy stds:", task_accuracy_std)
        ax1.errorbar(x=np.arange(n_tasks), y=loss_means, yerr=loss_stds, label=str(run_path.parts[-1]))

        # TODO: figure out how to stack the bars like in OML plot.
        bottom = len(indicators) * i
        print(f"Bottom: {bottom}")
        height = bar_height_scale * task_accuracy_means
        rects = ax2.bar(x=np.arange(n_tasks), height=height, bottom=bottom, yerr=task_accuracy_std, label=label)
        autolabel(ax2, rects, bar_height_scale)

    ax1.legend(loc="upper left")

    fig.savefig(options.out_path)
    fig.show()
    fig.waitforbuttonpress(timeout=30)


def autolabel(axis, rects: List[plt.Rectangle], bar_height_scale: float):
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