import argparse
import contextlib
import logging
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

from .make_oml_plot import OmlFigureOptions

logging.basicConfig(level=logging.CRITICAL)

DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "E:/Google Drive/"))
print(f"Data dir: {DATA_DIR}")

def get_figure_title(run_group_path: Path) -> str:
    title = run_group_path.name

    suffixes: List[str] = ["_ewc_d", "_ewc", "_d"]
    # The description corresponding to each suffix
    suffix_descriptions: List[str] = [
        "(+EWC, detached classifier)",
        "(+EWC)",
        "(detached classifier)",
    ]
    
    for suffix, desc in zip(suffixes, suffix_descriptions):
        if title.endswith(suffix):
            title = title.replace(suffix, "")
            title += " " + desc
            break

    print(f"title for run group path {run_group_path}: {title}")
    return title


def run(args: Dict) -> OmlFigureOptions:
    s = StringIO()
    with contextlib.redirect_stdout(s):
        obj = OmlFigureOptions(**args)
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
                title=title,
                show=False,
                maximize_figure=False,
                fig_size_inches=(12, 9),
                legend_position=legend_pos,
            ))

        import tqdm
        import multiprocessing as mp
        processes = min(len(args), mp.cpu_count())
        print(f"Creating figures using {processes} processes.")
        with mp.Pool(processes) as pool:
            for result in pool.imap_unordered(run, args):
                if result.result_figure is not None:
                    print(f"Figure created at path {result.out_path}")
                else:
                    print(f"Couldn't create figure for path {result.out_path}")
        # self.organized_dir = self.results_dir / (self.server.name + "_organised")


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    options = parser.parse_args().options

    options()
