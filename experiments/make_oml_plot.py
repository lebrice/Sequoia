import torch
from simple_parsing import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Options:
    """ Options for the script making the OML Figure 3 plot. """
    runs_to_compare: List[Path]
    out_path: Path = Path("results/plot.png")


def make_plot(options: Options) -> plt.Figure:
    runs: List[Path] = options.runs_to_compare

    fig: plt.Figure = plt.figure()
    ax1: plt.Axes = fig.add_subplot(1, 2, 1)
    ax1.set_title("Continual Classification Accuracy")
    ax1.set_xlabel("Number of tasks learned")
    ax1.set_ylabel("Classification Loss")
    ax1.legend(loc="upper left")

    ax2: plt.Axes = fig.add_subplot(1, 2, 2)
    ax2.set_title(f"Final mean accuracy per Task")
    ax2.set_xlabel("Task ID")

    for run_path in runs:
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
        ax2.bar(x=np.arange(n_tasks), height=task_accuracy_means, yerr=task_accuracy_std, label=str(run_path.parts[-1]))
    
    fig.savefig(options.out_path)
    fig.show()
    fig.waitforbuttonpress(timeout=10)


if __name__ == "__main__":
        
    parser = ArgumentParser()
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    options: Options = args.options
    print(options)

    make_plot(options)