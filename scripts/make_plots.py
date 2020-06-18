import argparse
import contextlib
import os
import shlex
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from distutils.dir_util import copy_tree
from io import StringIO
from pathlib import Path
from typing import Dict, List, Set, Tuple

from simple_parsing import ArgumentParser, choice, field, list_field
from utils.logging_utils import get_logger
from .make_oml_plot import OmlFigureOptions

DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "E:/Google Drive/"))
print(f"Data dir: {DATA_DIR}")

def get_row_name(run_path: Path) -> str:
    name = run_path.name
    return name

def get_column_name(run_path: Path) -> str:
    run_group_path = run_path.parent
    return get_figure_title(run_group_path)

def get_figure_title(run_group_path: Path) -> str:
    title = run_group_path.name
    
    run_name = run_group_path.name
    suffixes: List[str] = ["-sh", "_d", "_ewc"]
    # The description corresponding to each suffix
    suffix_descriptions: List[str] = [
        "single head",
        "detached classifier",
        "EWC",
    ]

    descriptions_to_add: List[str] = []
    while any(run_name.endswith(suffix) for suffix in suffixes):
        for suffix, desc in zip(suffixes, suffix_descriptions):
            if run_name.endswith(suffix):
                run_name = run_name[:run_name.rindex(suffix)]
                descriptions_to_add.append(desc)
                break
    
    descriptions_to_add = sorted(descriptions_to_add)

    title = run_name
    if descriptions_to_add:
        title += " - " + ", ".join(descriptions_to_add)
    # print(f"title for run group path {run_group_path}: {title}")
    return title


def run(args: Dict) -> OmlFigureOptions:
    s = StringIO()
    try:
        with contextlib.redirect_stdout(s):
            obj = OmlFigureOptions(**args)
    except KeyboardInterrupt:
        print(f"Interrupted creation of figure for args {args}")
        return None
    s.seek(0)
    # print(s.read())
    return obj


@dataclass
class Options:
    """ Options for organizing and creating all plots. """
    cleaned_up_results_dir: Path = Path(f"{DATA_DIR}/SSCL/results/merged")
    # Directory where figures should be auto-created.
    figures_dir: Path = Path(f"{DATA_DIR}/SSCL/figures/merged")

    def __call__(self):
        results_dir = self.cleaned_up_results_dir
        figures_dir = self.figures_dir

        print(results_dir)
        print(figures_dir)

        # The arguments for each figure
        args: List[Dict] = []

        for run_group_path in sorted(results_dir.iterdir()):
            run_group_name = run_group_path.name
            if run_group_name == "wandb":
                continue

            print(f"Run group path: {run_group_path}")
            title = get_figure_title(run_group_path)
            legend_pos = ("upper right" if "cifar100" in run_group_name else "lower left") 

            args.append(dict(
                runs=[str(run_group_path / "*")],
                out_path=figures_dir / f"{run_group_name}.pdf",
                exit_after=False,
                add_ntasks_prefix=False,
                title="", #(will be set with the figure captions in LaTeX)
                show=False,
                maximize_figure=False,
                fig_size_inches=(12, 5),
                legend_position=legend_pos,
            ))
        
        import tqdm
        import multiprocessing as mp
        processes = min(len(args), mp.cpu_count())
        print(f"Creating figures using {processes} processes.")
        
        import pandas as pd
        from functools import partial
        
        table_data = defaultdict(partial(defaultdict, dict))

        mp.set_start_method("spawn")
        with mp.Pool(processes) as pool:
            for i, result in enumerate(pool.imap_unordered(run, args)):
                if result.result_figure is not None:
                    print(f"Figure created at path {result.out_path}")
                    
                    for run_path, classification_accs in result.classification_accuracies.items():
                        means = classification_accs.mean(axis=0)
                        stds = classification_accs.std(axis=0)
                        
                        row_name = get_row_name(run_path)
                        column_name = get_column_name(run_path)
                        table_data[row_name][column_name]["means"] = means
                        table_data[row_name][column_name]["stds"] = stds
                else:
                    print(f"Couldn't create figure for path {result.out_path}")
                
                if i == len(args) - 1:
                    print("Reached last figure, closing the pool?")
                    pool.close()

        pool.close()
        print("Done creating all the figures.")
        table_data = pd.DataFrame(table_data)
        
        print(table_data.describe())
        table_data.to_csv("./table_data.csv")
        # self.organized_dir = self.results_dir / (self.server.name + "_organised")


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    options = parser.parse_args().options

    options()
