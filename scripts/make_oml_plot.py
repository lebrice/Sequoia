import torch
from simple_parsing import ArgumentParser, field
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
    runs_to_compare: List[str] = field(type=glob.glob)
    out_path: Path = Path("scripts/plot.png")

    def __post_init__(self):
        if len(self.runs_to_compare) == 1 and isinstance(self.runs_to_compare[0], list):
            self.runs_to_compare = self.runs_to_compare[0]

def make_plot(options: Options) -> plt.Figure:
    runs: List[Path] = list(map(Path, options.runs_to_compare))
    
    kept_runs: List[Path] = []
    for run in runs:
        if not run.is_dir() or str(run).endswith("wandb"):
            continue
        
        valid_losses_path = (run / "plots" / "valid_losses.pt")
        final_accuracy_path = (run / "plots" / "final_task_accuracy.pt")
        required_files = [valid_losses_path, final_accuracy_path]
        if all(p.exists() and p.is_file for p in required_files):
            kept_runs.append(run)
    

    n = len(kept_runs)
    print(f"Creating the OML plot to compare the {n} different methods:")

    fig: plt.Figure = plt.figure()
    ax1: plt.Axes = fig.add_subplot(1, 2, 1)
    ax1.set_title("Continual Classification Accuracy")
    ax1.set_xlabel("Number of tasks learned")
    ax1.set_ylabel("Classification Loss")
    ax1.legend(loc="upper left")

    ax2: plt.Axes = fig.add_subplot(1, 2, 2)
    ax2.set_title(f"Final mean accuracy per Task")
    ax2.set_xlabel("Task ID")

    ax2.set_yticks(np.arange(0, 2*n, 1, dtype=int))
    ax2.set_yticklabels(['0', '0.25', '0.50' '1.00']*n)

    for i, run_path in enumerate(kept_runs):
        valid_loss = torch.load(run_path / "plots" / "valid_losses.pt")
        final_task_accuracy = torch.load(run_path / "plots" / "final_task_accuracy.pt")

        loss_means = valid_loss.mean(dim=0).numpy()
        loss_stds = valid_loss.std(dim=0).numpy()

        task_accuracy_means = final_task_accuracy.mean(dim=0).numpy()
        task_accuracy_std =   final_task_accuracy.std(dim=0).numpy()
        n_tasks= len(task_accuracy_means)

        print(f"Run {run_path}:")
        print("\t Loss Means:", loss_means)
        print("\t Loss STDs:", loss_stds)
        print("\t Final Task Accuracy means:", task_accuracy_means)
        print("\t Final Task Accuracy stds:", task_accuracy_std)
        ax1.set_xticks(np.arange(n_tasks, dtype=int))
        ax2.set_xticks(np.arange(n_tasks, dtype=int))
        ax1.errorbar(x=np.arange(n_tasks), y=loss_means, yerr=loss_stds, label=str(run_path.parts[-1]))

        # TODO: figure out how to stack the bars like in OML plot.
        bottom = 2*i
        height = task_accuracy_means + bottom
        ax2.bar(x=np.arange(n_tasks), height=height, bottom=bottom, yerr=task_accuracy_std, label=str(run_path.parts[-1]))

    ax1.legend(loc="upper left")

    fig.savefig(options.out_path)
    fig.show()
    fig.waitforbuttonpress(timeout=30)


if __name__ == "__main__":
        
    parser = ArgumentParser()
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    options: Options = args.options
    print(options)

    make_plot(options)